#!/usr/bin/env python3
"""Run the sealed random-weight scalar/BPE MPS runtime preflight."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.compute_conversion import conversion_model_spec
from jamoflow.incremental_blt import (
    IncrementalBltDecoder,
    structural_prefix_boundaries,
)
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_model import build_main_model
from jamoflow.phase2_patching import padded_hf_patch_matrix
from scalar_runtime_core import (
    BPE_PRIMARY_SPEC,
    BPE_SECONDARY_SPEC,
    GLOBAL_POSITION_LIMIT,
    MODEL_SEED,
    W72_SPEC,
    FactorizedUnitBlt,
    IncrementalBpeDecoder,
    IncrementalUnitBltDecoder,
    build_bpe_model,
    encode_units,
    maximum_normalized_error,
    model_parameter_count,
)
from scalar_runtime_protocol import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MPS_ATOL,
    MPS_RTOL,
    OUTPUT_PATH,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    PLAN_PATH,
    PROMPT_BYTES,
    PROTOCOL_ID,
    REPETITIONS,
    REPORT_PATH,
    ROOT,
    RUNTIME_ROLES,
    TIMING_PATH,
    WARMUP_CASES,
    array_sha256,
    canonical_sha256,
    encode_bpe_case,
    hash_file,
    json_bytes,
    load_tokenizers,
    read_json,
    reconstruct_cases,
    role_schedule,
    validate_plan,
)


ACTIVE_PATH = REPORT_PATH.parent / ".active"


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_plan_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("scalar runtime benchmark requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    last_change = _command(
        "git", "log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))
    )
    if len(commit) != 40 or last_change != commit:
        raise ValueError("scalar runtime plan must be committed at current HEAD")
    return commit


def _command_snapshot(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout": result.stdout,
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _command_snapshot(("pmset", "-g", "batt")),
        "settings": _command_snapshot(("pmset", "-g", "custom")),
        "thermal": _command_snapshot(("pmset", "-g", "therm")),
    }


def _synchronize() -> None:
    torch.mps.synchronize()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _assert_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    value = maximum_normalized_error(
        actual,
        expected,
        rtol=MPS_RTOL,
        atol=MPS_ATOL,
    )
    if not math.isfinite(value) or value > 1.0:
        raise AssertionError("scalar runtime cache/full tolerance differs")
    return value


def _byte_full_logits(model: Any, raw: bytes) -> torch.Tensor:
    boundaries = structural_prefix_boundaries(
        raw,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )
    matrix = padded_hf_patch_matrix([boundaries], len(raw))
    values = torch.tensor([list(raw)], dtype=torch.long, device="mps")
    patch_lengths = torch.from_numpy(matrix.astype(np.int64, copy=False)).to("mps")
    return model(
        input_ids=values,
        patch_lengths=patch_lengths,
        use_cache=False,
    ).logits.float()


def _verify_byte(model: Any, prompt: bytes, continuation: bytes) -> dict[str, Any]:
    raw = prompt + continuation
    full = _byte_full_logits(model, raw)
    maximum = 0.0
    comparisons = 0
    sequential = IncrementalBltDecoder(
        model,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )
    for position, value in enumerate(raw[:-1]):
        logits = sequential.consume(value)
        maximum = max(maximum, _assert_error(logits, full[:, position]))
        if not torch.equal(logits.argmax(dim=-1), full[:, position].argmax(dim=-1)):
            raise AssertionError("byte sequential/full argmax differs")
        comparisons += 1
    parallel = IncrementalBltDecoder(
        model,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )
    logits = parallel.prefill_parallel(prompt)
    maximum = max(maximum, _assert_error(logits, full[:, len(prompt) - 1]))
    comparisons += 1
    for offset, value in enumerate(continuation[:-1]):
        logits = parallel.consume(value)
        position = len(prompt) + offset
        maximum = max(maximum, _assert_error(logits, full[:, position]))
        comparisons += 1
    return {
        "comparisons": comparisons,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def _verify_units(
    model: FactorizedUnitBlt,
    representation: str,
    prompt: bytes,
    continuation: bytes,
) -> dict[str, Any]:
    prompt_units = encode_units(prompt, representation)
    continuation_units = encode_units(continuation, representation)
    units = prompt_units + continuation_units
    boundaries = structural_prefix_boundaries(
        prompt + continuation,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )
    unit_starts = []
    offset = 0
    for unit in units:
        unit_starts.append(offset)
        offset += len(unit.raw)
    by_start = {value: index for index, value in enumerate(unit_starts)}
    unit_boundaries = tuple(by_start[value] for value in boundaries)
    full = model.full_hidden(units, unit_boundaries)
    maximum = 0.0
    comparisons = 0
    sampling_comparisons = 0

    sequential = IncrementalUnitBltDecoder(model)
    for position, unit in enumerate(units[:-1]):
        hidden = sequential.consume(unit)
        expected = full[:, position : position + 1]
        maximum = max(maximum, _assert_error(hidden, expected))
        actual_choices = model.sample_fixed_target_route(hidden, units[position + 1])
        expected_choices = model.sample_fixed_target_route(expected, units[position + 1])
        if len(actual_choices) != len(expected_choices) or any(
            not torch.equal(left, right)
            for left, right in zip(actual_choices, expected_choices, strict=True)
        ):
            raise AssertionError("unit sequential/full conditional choice differs")
        sampling_comparisons += len(actual_choices)
        comparisons += 1

    parallel = IncrementalUnitBltDecoder(model)
    hidden = parallel.prefill_parallel(prompt_units)
    position = len(prompt_units) - 1
    maximum = max(maximum, _assert_error(hidden, full[:, position : position + 1]))
    comparisons += 1
    for offset, unit in enumerate(continuation_units[:-1]):
        hidden = parallel.consume(unit)
        position = len(prompt_units) + offset
        expected = full[:, position : position + 1]
        maximum = max(maximum, _assert_error(hidden, expected))
        actual_choices = model.sample_fixed_target_route(
            hidden,
            continuation_units[offset + 1],
        )
        expected_choices = model.sample_fixed_target_route(
            expected,
            continuation_units[offset + 1],
        )
        if any(
            not torch.equal(left, right)
            for left, right in zip(actual_choices, expected_choices, strict=True)
        ):
            raise AssertionError("unit parallel/full conditional choice differs")
        sampling_comparisons += len(actual_choices)
        comparisons += 1
    return {
        "comparisons": comparisons,
        "conditional_sampling_comparisons": sampling_comparisons,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def _verify_bpe(
    model: Any,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
) -> dict[str, Any]:
    ids = tuple(prompt_ids) + tuple(continuation_ids)
    values = torch.tensor([list(ids)], dtype=torch.long, device="mps")
    full = model(input_ids=values, use_cache=False).logits.float()
    maximum = 0.0
    comparisons = 0
    sequential = IncrementalBpeDecoder(model)
    logits = sequential.prefill_parallel(ids[:1])
    maximum = max(maximum, _assert_error(logits, full[:, 0]))
    comparisons += 1
    for position, token_id in enumerate(ids[1:-1], start=1):
        logits = sequential.consume(token_id)
        maximum = max(maximum, _assert_error(logits, full[:, position]))
        if not torch.equal(logits.argmax(dim=-1), full[:, position].argmax(dim=-1)):
            raise AssertionError("BPE sequential/full argmax differs")
        comparisons += 1
    parallel = IncrementalBpeDecoder(model)
    logits = parallel.prefill_parallel(prompt_ids)
    maximum = max(maximum, _assert_error(logits, full[:, len(prompt_ids) - 1]))
    comparisons += 1
    for offset, token_id in enumerate(continuation_ids[:-1]):
        logits = parallel.consume(token_id)
        position = len(prompt_ids) + offset
        maximum = max(maximum, _assert_error(logits, full[:, position]))
        comparisons += 1
    return {
        "comparisons": comparisons,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def _build_models() -> dict[str, Any]:
    models = {
        "byte_w72": build_main_model(
            conversion_model_spec(72),
            seed=MODEL_SEED,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        ),
        "generic_unicode_scalar": FactorizedUnitBlt("generic_unicode_scalar"),
        "hangul_hybrid": FactorizedUnitBlt("hangul_hybrid"),
        "byte_bpe_32000": build_bpe_model(BPE_PRIMARY_SPEC),
        "byte_bpe_16000": build_bpe_model(BPE_SECONDARY_SPEC),
    }
    for role, model in models.items():
        if abs(model_parameter_count(model) / PARAMETER_TARGET - 1.0) > PARAMETER_RELATIVE_TOLERANCE:
            raise ValueError(f"runtime graph parameter count differs: {role}")
        model.eval().to("mps")
    return models


def _prepare_payloads(
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> list[dict[str, Any]]:
    tokenizers = load_tokenizers()
    output = []
    for prompt_row, continuation_row in zip(prompts, continuations, strict=True):
        prompt = bytes(prompt_row)
        continuation = bytes(continuation_row)
        row = {
            "byte_w72": (prompt, continuation),
            "generic_unicode_scalar": (
                encode_units(prompt, "generic_unicode_scalar"),
                encode_units(continuation, "generic_unicode_scalar"),
            ),
            "hangul_hybrid": (
                encode_units(prompt, "hangul_hybrid"),
                encode_units(continuation, "hangul_hybrid"),
            ),
        }
        for vocabulary_size, (tokenizer, table) in tokenizers.items():
            row[f"byte_bpe_{vocabulary_size}"] = encode_bpe_case(
                prompt,
                continuation,
                tokenizer,
                table,
            )
        output.append(row)
    return output


def _timed_trial(role: str, model: Any, payload: tuple[Any, Any]) -> tuple[float, float, float, int]:
    prompt, continuation = payload
    _synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        if role == "byte_w72":
            runtime = IncrementalBltDecoder(
                model,
                "causal_whitespace_grid",
                horizon=512,
                patch_count=72,
                fixed_stride=6,
            )
            logits = runtime.prefill_parallel(prompt)
            _ = logits.argmax(dim=-1)
        elif role in {"generic_unicode_scalar", "hangul_hybrid"}:
            runtime = IncrementalUnitBltDecoder(model)
            hidden = runtime.prefill_parallel(prompt)
            _ = model.sample_fixed_target_route(hidden, continuation[0])
        else:
            runtime = IncrementalBpeDecoder(model)
            logits = runtime.prefill_parallel(prompt)
            _ = logits.argmax(dim=-1)
        _synchronize()
        first = time.perf_counter_ns()
        if role == "byte_w72":
            for value in continuation[:-1]:
                logits = runtime.consume(value)
                _ = logits.argmax(dim=-1)
            observed = len(prompt) + len(continuation) - 1
            diagnostics = runtime.diagnostics
            if (
                diagnostics.observed_bytes != observed
                or diagnostics.local_encoder_cached_bytes != observed
                or diagnostics.local_decoder_cached_bytes != observed
            ):
                raise AssertionError("byte runtime cache length differs")
            steps = len(continuation)
        elif role in {"generic_unicode_scalar", "hangul_hybrid"}:
            for index, unit in enumerate(continuation[:-1]):
                hidden = runtime.consume(unit)
                _ = model.sample_fixed_target_route(hidden, continuation[index + 1])
            diagnostics = runtime.diagnostics()
            observed = len(prompt) + len(continuation) - 1
            if (
                diagnostics["observed_units"] != observed
                or diagnostics["cached_units_encoder"] != observed
                or diagnostics["cached_units_decoder"] != observed
            ):
                raise AssertionError("unit runtime cache length differs")
            steps = len(continuation)
        else:
            for token_id in continuation[:-1]:
                logits = runtime.consume(token_id)
                _ = logits.argmax(dim=-1)
            observed = len(prompt) + len(continuation) - 1
            if runtime.observed_tokens != observed:
                raise AssertionError("BPE runtime cache length differs")
            steps = len(continuation)
        _synchronize()
        finished = time.perf_counter_ns()
    del runtime
    return (
        (first - started) / 1_000_000,
        (finished - first) / 1_000_000,
        (finished - started) / 1_000_000,
        steps,
    )


def _verify_all(
    models: Mapping[str, Any],
    prompts: np.ndarray,
    continuations: np.ndarray,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = {role: [] for role in RUNTIME_ROLES}
    with torch.inference_mode():
        for index in range(WARMUP_CASES):
            prompt = bytes(prompts[index])
            continuation = bytes(continuations[index])
            rows["byte_w72"].append(
                _verify_byte(models["byte_w72"], prompt, continuation)
            )
            for role, representation in (
                ("generic_unicode_scalar", "generic_unicode_scalar"),
                ("hangul_hybrid", "hangul_hybrid"),
            ):
                rows[role].append(
                    _verify_units(models[role], representation, prompt, continuation)
                )
            for role in ("byte_bpe_32000", "byte_bpe_16000"):
                prompt_ids, continuation_ids = payloads[index][role]
                rows[role].append(
                    _verify_bpe(models[role], prompt_ids, continuation_ids)
                )
    return {
        role: {
            "cases": len(values),
            "comparisons": sum(value["comparisons"] for value in values),
            "maximum_normalized_tolerance_ratio": max(
                value["maximum_normalized_tolerance_ratio"] for value in values
            ),
            "pass": all(value["pass"] for value in values),
        }
        for role, values in rows.items()
    }


def main() -> None:
    commit = _require_clean_plan_commit()
    if any(path.exists() for path in (REPORT_PATH, TIMING_PATH, ACTIVE_PATH, OUTPUT_PATH)):
        raise FileExistsError("scalar runtime evidence namespace is not empty")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if hash_file(PLAN_PATH) == plan["dependencies"]["plan_payload_sha256"]:
        raise AssertionError("artifact and canonical plan hash domains unexpectedly alias")
    prompts, continuations, case_metadata = reconstruct_cases()
    if case_metadata != plan["cases"]:
        raise ValueError("scalar runtime cases changed after sealing")
    payloads = _prepare_payloads(prompts, continuations)
    ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _publish(
        ACTIVE_PATH,
        json_bytes({"git_commit": commit, "plan_artifact_sha256": hash_file(PLAN_PATH)}),
    )
    with publication_mps_exclusive():
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise ValueError("scalar runtime environment is ineligible at start")
        models = _build_models()
        correctness = _verify_all(models, prompts, continuations, payloads)
        if not all(value["pass"] for value in correctness.values()):
            raise AssertionError("scalar runtime correctness gate failed")
        # Compile and warm every path in the same deterministic case order.
        for index in range(WARMUP_CASES):
            for role in RUNTIME_ROLES:
                _timed_trial(role, models[role], payloads[index][role])
        arrays = {
            f"{component}__{role}": np.zeros(
                (MEASURED_CASES, REPETITIONS), dtype=np.float64
            )
            for role in RUNTIME_ROLES
            for component in ("ttft_ms", "decode_ms", "end_to_end_ms")
        }
        arrays.update(
            {
                f"continuation_steps__{role}": np.zeros(
                    (MEASURED_CASES, REPETITIONS), dtype=np.int64
                )
                for role in RUNTIME_ROLES
            }
        )
        for measured_index in range(MEASURED_CASES):
            case_index = WARMUP_CASES + measured_index
            for repetition in range(REPETITIONS):
                for role in role_schedule(measured_index, repetition):
                    ttft, decode, end_to_end, steps = _timed_trial(
                        role,
                        models[role],
                        payloads[case_index][role],
                    )
                    arrays[f"ttft_ms__{role}"][measured_index, repetition] = ttft
                    arrays[f"decode_ms__{role}"][measured_index, repetition] = decode
                    arrays[f"end_to_end_ms__{role}"][measured_index, repetition] = end_to_end
                    arrays[f"continuation_steps__{role}"][measured_index, repetition] = steps
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise ValueError("scalar runtime environment is ineligible at end")
        environment = {
            "device": "mps",
            "mps_available": torch.backends.mps.is_available(),
            **current_runtime_environment_contract(),
        }
        parameter_counts = {
            role: model_parameter_count(model) for role, model in models.items()
        }
        for model in models.values():
            model.to("cpu")
        models.clear()
        gc.collect()
        torch.mps.empty_cache()
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during scalar runtime benchmark")
    timing_payload = _npz_bytes(arrays)
    _publish(TIMING_PATH, timing_payload)
    report = {
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "sha256": array_sha256(value),
                "shape": list(value.shape),
            }
            for name, value in arrays.items()
        },
        "case_metadata": case_metadata,
        "claim_boundary": plan["claim_boundary"],
        "complete": True,
        "correctness": correctness,
        "environment": environment,
        "git_commit": commit,
        "kind": "scalar_runtime_preflight_report_v1",
        "parameter_counts": parameter_counts,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "protocol_id": PROTOCOL_ID,
        "schema_version": 1,
        "session_state": {"end": end_state, "start": start_state},
        "timing_artifact_sha256": hashlib.sha256(timing_payload).hexdigest(),
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"wrote {TIMING_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
