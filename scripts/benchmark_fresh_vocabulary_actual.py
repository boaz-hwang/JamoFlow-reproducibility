#!/usr/bin/env python3
"""Measure trained 8K-vs-2K batch-1 end-to-end generation on MPS."""

from __future__ import annotations

import gc
import hashlib
import io
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    WARMUP_CASES,
    balanced_role_order,
)
from fresh_vocabulary_actual_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_BY_ROLE,
    MAXIMUM_FREE_TOKENS,
    MPS_ATOL,
    MPS_RTOL,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    RUNTIME_REPORT_PATH,
    TIMING_PATH,
    VOCABULARY_BY_ROLE,
    array_sha256,
    build_role_model,
    canonical_sha256,
    encode_raw,
    hash_file,
    json_bytes,
    read_json,
    reconstruct_cases,
    validate_plan,
)
from scalar_runtime_core import IncrementalBpeDecoder, maximum_normalized_error
from vocabulary_transfer_probe_core import state_mapping_sha256

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.utf8 import (
    StrictUtf8TokenTransitions,
    compile_strict_utf8_token_transitions,
)


@dataclass(slots=True)
class RoleBundle:
    role: str
    model: Any
    tokenizer: Any
    token_bytes: tuple[bytes, ...]
    transitions: StrictUtf8TokenTransitions
    masks: tuple[torch.Tensor, ...]

    def runtime(self) -> IncrementalBpeDecoder:
        return IncrementalBpeDecoder(self.model)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _snapshot(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout": result.stdout,
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _snapshot(("pmset", "-g", "batt")),
        "settings": _snapshot(("pmset", "-g", "custom")),
        "thermal": _snapshot(("pmset", "-g", "therm")),
    }


def _publish(path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": array_sha256(value),
    }


def _require_plan_commit() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh actual benchmark requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()) != commit:
        raise RuntimeError("fresh actual plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("fresh actual plan parent differs")
    return commit, plan


def _compile_masks(transitions: StrictUtf8TokenTransitions) -> tuple[torch.Tensor, ...]:
    masks = []
    for allowed in transitions.allowed_token_ids:
        mask = torch.zeros(transitions.vocabulary_size, dtype=torch.bool, device="mps")
        mask[torch.tensor(allowed, dtype=torch.long, device="mps")] = True
        masks.append(mask)
    torch.mps.synchronize()
    return tuple(masks)


def load_bundles(plan: Mapping[str, Any]) -> dict[str, RoleBundle]:
    loaded = load_tokenizers()
    bundles: dict[str, RoleBundle] = {}
    for role in ROLES:
        model = build_role_model(role)
        state = torch.load(CHECKPOINT_BY_ROLE[role], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        identity = plan["models"][role]
        if (
            hash_file(CHECKPOINT_BY_ROLE[role]) != identity["checkpoint_artifact_sha256"]
            or state_mapping_sha256(model.state_dict()) != identity["checkpoint_state_sha256"]
            or sum(parameter.numel() for parameter in model.parameters())
            != identity["parameter_count"]
        ):
            raise RuntimeError("fresh actual loaded model identity differs")
        size = VOCABULARY_BY_ROLE[role]
        tokenizer, token_bytes = loaded[size]
        transitions = compile_strict_utf8_token_transitions(token_bytes)
        contract = plan["tokenizer_runtime"][role]["strict_utf8_transitions"]
        if (
            transitions.token_bytes_sha256 != contract["token_bytes_sha256"]
            or transitions.transition_table_sha256 != contract["transition_table_sha256"]
            or transitions.maximum_token_bytes != contract["maximum_token_bytes"]
        ):
            raise RuntimeError("fresh actual token transition identity differs")
        model = model.eval().to("mps")
        bundles[role] = RoleBundle(
            role=role,
            model=model,
            tokenizer=tokenizer,
            token_bytes=token_bytes,
            transitions=transitions,
            masks=_compile_masks(transitions),
        )
    return bundles


def _normalized_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    value = maximum_normalized_error(actual, expected, rtol=MPS_RTOL, atol=MPS_ATOL)
    if not math.isfinite(value) or value > 1.0:
        raise AssertionError("fresh actual cache/full tolerance differs")
    return value


def verify_sequence(
    bundle: RoleBundle,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
) -> dict[str, Any]:
    if not prompt_ids or not output_ids:
        raise ValueError("fresh actual correctness sequence is empty")
    all_ids = tuple(prompt_ids) + tuple(output_ids)
    values = torch.tensor([list(all_ids)], dtype=torch.long, device="mps")
    with torch.inference_mode():
        full = bundle.model(input_ids=values, use_cache=False).logits.float()
        runtime = bundle.runtime()
        logits = runtime.prefill_parallel(prompt_ids)
        maximum = _normalized_error(logits, full[:, len(prompt_ids) - 1])
        comparisons = 1
        argmax_exact = int(
            torch.equal(
                logits.argmax(dim=-1),
                full[:, len(prompt_ids) - 1].argmax(dim=-1),
            )
        )
        for offset, token_id in enumerate(output_ids[:-1]):
            logits = runtime.consume(int(token_id))
            position = len(prompt_ids) + offset
            maximum = max(maximum, _normalized_error(logits, full[:, position]))
            comparisons += 1
            argmax_exact += int(
                torch.equal(logits.argmax(dim=-1), full[:, position].argmax(dim=-1))
            )
    if (
        argmax_exact != comparisons
        or runtime.observed_tokens != len(prompt_ids) + len(output_ids) - 1
    ):
        raise AssertionError("fresh actual cache/full argmax or length differs")
    return {
        "comparisons": comparisons,
        "argmax_comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def generate_free_untimed(
    bundle: RoleBundle,
    prompt_ids: Sequence[int],
) -> tuple[tuple[int, ...], bytes]:
    runtime = bundle.runtime()
    state_index = 0
    token_ids: list[int] = []
    output = bytearray()
    with torch.inference_mode():
        logits = runtime.prefill_parallel(prompt_ids)
        while True:
            token_id = int(
                logits.masked_fill(~bundle.masks[state_index], -torch.inf)
                .argmax(dim=-1)
                .item()
            )
            next_state = bundle.transitions.next_state_indices[state_index][token_id]
            if next_state < 0:
                raise AssertionError("fresh actual UTF-8 mask admitted invalid token")
            token_ids.append(token_id)
            output.extend(bundle.token_bytes[token_id])
            state_index = next_state
            if len(output) >= CONTINUATION_BYTES and state_index == 0:
                break
            if len(token_ids) >= MAXIMUM_FREE_TOKENS:
                raise AssertionError("fresh actual free generation exceeded token bound")
            logits = runtime.consume(token_id)
    raw = bytes(output)
    raw.decode("utf-8", errors="strict")
    maximum = CONTINUATION_BYTES + bundle.transitions.maximum_token_bytes - 1
    if not CONTINUATION_BYTES <= len(raw) <= maximum:
        raise AssertionError("fresh actual free byte bound differs")
    return tuple(token_ids), raw


def correctness_replay(
    bundles: Mapping[str, RoleBundle],
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ROLES:
        bundle = bundles[role]
        rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        with torch.inference_mode():
            for prompt, continuation in zip(
                prompts[:WARMUP_CASES], continuations[:WARMUP_CASES], strict=True
            ):
                prompt_ids = encode_raw(bytes(prompt), bundle.tokenizer, bundle.token_bytes)
                controlled_ids = encode_raw(
                    bytes(continuation), bundle.tokenizer, bundle.token_bytes
                )
                rows["controlled_replay"].append(
                    verify_sequence(bundle, prompt_ids, controlled_ids)
                )
                free_ids, _ = generate_free_untimed(bundle, prompt_ids)
                rows["free_running_utf8_greedy"].append(
                    verify_sequence(bundle, prompt_ids, free_ids)
                )
        output[role] = {}
        for mode in MODES:
            mode_rows = rows[mode]
            comparisons = sum(int(row["comparisons"]) for row in mode_rows)
            argmax = sum(int(row["argmax_exact"]) for row in mode_rows)
            output[role][mode] = {
                "cases": len(mode_rows),
                "comparisons": comparisons,
                "argmax_comparisons": comparisons,
                "argmax_exact": argmax,
                "maximum_normalized_tolerance_ratio": max(
                    float(row["maximum_normalized_tolerance_ratio"])
                    for row in mode_rows
                ),
                "pass": bool(argmax == comparisons and all(row["pass"] for row in mode_rows)),
            }
    return output


def _timed_prompt_ids(bundle: RoleBundle, prompt: bytes) -> tuple[int, ...]:
    text = prompt.decode("utf-8", errors="strict")
    ids = tuple(
        int(value)
        for value in bundle.tokenizer.encode(text, add_special_tokens=False).ids
    )
    if not ids:
        raise AssertionError("fresh actual timed tokenizer returned no IDs")
    return ids


def run_trial(
    bundle: RoleBundle,
    prompt: bytes,
    expected_prompt_ids: Sequence[int],
    controlled_ids: Sequence[int],
    controlled_raw: bytes,
    mode: str,
) -> tuple[dict[str, float], tuple[int, ...], bytes]:
    if mode not in MODES:
        raise ValueError("fresh actual trial mode differs")
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    prompt_ids = _timed_prompt_ids(bundle, prompt)
    encoded = time.perf_counter_ns()
    generated_ids: list[int] = []
    generated = bytearray()
    first = None
    with torch.inference_mode():
        runtime = bundle.runtime()
        logits = runtime.prefill_parallel(prompt_ids)
        if mode == "controlled_replay":
            for index, token_id in enumerate(controlled_ids):
                _ = int(logits.argmax(dim=-1).item())
                generated_ids.append(int(token_id))
                generated.extend(bundle.token_bytes[int(token_id)])
                if first is None:
                    first = time.perf_counter_ns()
                if index + 1 < len(controlled_ids):
                    logits = runtime.consume(int(token_id))
        else:
            state_index = 0
            while True:
                token_id = int(
                    logits.masked_fill(~bundle.masks[state_index], -torch.inf)
                    .argmax(dim=-1)
                    .item()
                )
                next_state = bundle.transitions.next_state_indices[state_index][token_id]
                if next_state < 0:
                    raise AssertionError("fresh actual timed mask admitted invalid token")
                generated_ids.append(token_id)
                generated.extend(bundle.token_bytes[token_id])
                state_index = next_state
                if first is None:
                    first = time.perf_counter_ns()
                if len(generated) >= CONTINUATION_BYTES and state_index == 0:
                    break
                if len(generated_ids) >= MAXIMUM_FREE_TOKENS:
                    raise AssertionError("fresh actual timed free token bound differs")
                logits = runtime.consume(token_id)
        torch.mps.synchronize()
        model_finished = time.perf_counter_ns()
    raw = bytes(generated)
    raw.decode("utf-8", errors="strict")
    finished = time.perf_counter_ns()
    if tuple(prompt_ids) != tuple(expected_prompt_ids):
        raise AssertionError("fresh actual timed prompt tokenization drifted")
    if mode == "controlled_replay" and raw != controlled_raw:
        raise AssertionError("fresh actual controlled output differs")
    maximum = CONTINUATION_BYTES + bundle.transitions.maximum_token_bytes - 1
    if (
        first is None
        or runtime.observed_tokens != len(prompt_ids) + len(generated_ids) - 1
        or not CONTINUATION_BYTES <= len(raw) <= maximum
    ):
        raise AssertionError("fresh actual timed runtime accounting differs")
    metrics = {
        "tokenizer_ms": (encoded - started) / 1_000_000,
        "ttft_ms": (first - started) / 1_000_000,
        "decode_ms": (finished - first) / 1_000_000,
        "model_loop_ms": (model_finished - encoded) / 1_000_000,
        "end_to_end_ms": (finished - started) / 1_000_000,
    }
    if any(not math.isfinite(value) or value <= 0 for value in metrics.values()):
        raise AssertionError("fresh actual timing metric differs")
    return metrics, tuple(generated_ids), raw


def main() -> None:
    commit, plan = _require_plan_commit()
    if any(path.exists() for path in (ACTIVE_PATH, RUNTIME_REPORT_PATH, TIMING_PATH, OUTPUT_PATH)):
        raise FileExistsError("fresh actual runtime namespace is not empty")
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("fresh actual cases changed after sealing")
    _publish(
        ACTIVE_PATH,
        json_bytes(
            {
                "git_commit": commit,
                "plan_artifact_sha256": hash_file(PLAN_PATH),
            }
        ),
    )
    with publication_mps_exclusive():
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise RuntimeError("fresh actual timing environment is ineligible at start")
        bundles = load_bundles(plan)
        correctness = correctness_replay(bundles, prompts, continuations)
        if not all(
            row["pass"]
            for role in correctness.values()
            for row in role.values()
        ):
            raise AssertionError("fresh actual pre-timing correctness failed")
        payloads: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        for role in ROLES:
            bundle = bundles[role]
            payloads[role] = [
                (
                    encode_raw(bytes(prompt), bundle.tokenizer, bundle.token_bytes),
                    encode_raw(bytes(continuation), bundle.tokenizer, bundle.token_bytes),
                )
                for prompt, continuation in zip(prompts, continuations, strict=True)
            ]
        for case_index in range(WARMUP_CASES):
            for mode in MODES:
                for role in ROLES:
                    prompt_ids, continuation_ids = payloads[role][case_index]
                    run_trial(
                        bundles[role],
                        bytes(prompts[case_index]),
                        prompt_ids,
                        continuation_ids,
                        bytes(continuations[case_index]),
                        mode,
                    )

        shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
        arrays: dict[str, np.ndarray] = {
            name: np.empty(shape, dtype=np.float64) for name in TIMING_COMPONENTS
        }
        arrays["output_token_count"] = np.empty(shape, dtype=np.int16)
        arrays["output_raw_byte_count"] = np.empty(shape, dtype=np.int16)
        maximum_output_bytes = max(
            int(plan["tokenizer_runtime"][role]["strict_utf8_transitions"]["maximum_free_output_bytes"])
            for role in ROLES
        )
        arrays["free_token_ids"] = np.full(
            (MEASURED_CASES, REPETITIONS, len(ROLES), MAXIMUM_FREE_TOKENS),
            -1,
            dtype=np.int32,
        )
        arrays["free_output_bytes"] = np.zeros(
            (MEASURED_CASES, REPETITIONS, len(ROLES), maximum_output_bytes),
            dtype=np.uint8,
        )
        arrays["free_output_lengths"] = np.zeros(
            (MEASURED_CASES, REPETITIONS, len(ROLES)), dtype=np.int16
        )
        for case_index in range(MEASURED_CASES):
            source_index = WARMUP_CASES + case_index
            prompt = bytes(prompts[source_index])
            continuation = bytes(continuations[source_index])
            for repetition in range(REPETITIONS):
                for mode_index, mode in enumerate(MODES):
                    for role_index in balanced_role_order(case_index, repetition, mode_index):
                        role = ROLES[role_index]
                        prompt_ids, continuation_ids = payloads[role][source_index]
                        metrics, output_ids, output = run_trial(
                            bundles[role],
                            prompt,
                            prompt_ids,
                            continuation_ids,
                            continuation,
                            mode,
                        )
                        for name, value in metrics.items():
                            arrays[name][mode_index, case_index, repetition, role_index] = value
                        arrays["output_token_count"][
                            mode_index, case_index, repetition, role_index
                        ] = len(output_ids)
                        arrays["output_raw_byte_count"][
                            mode_index, case_index, repetition, role_index
                        ] = len(output)
                        if mode == "free_running_utf8_greedy":
                            arrays["free_token_ids"][
                                case_index, repetition, role_index, : len(output_ids)
                            ] = np.asarray(output_ids, dtype=np.int32)
                            arrays["free_output_bytes"][
                                case_index, repetition, role_index, : len(output)
                            ] = np.frombuffer(output, dtype=np.uint8)
                            arrays["free_output_lengths"][
                                case_index, repetition, role_index
                            ] = len(output)
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError("fresh actual timing environment is ineligible at end")
        for bundle in bundles.values():
            bundle.model.to("cpu")
        del bundles
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    timing_payload = _npz_bytes(arrays)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_actual_one_seed_runtime_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "cases": cases,
        "models": plan["models"],
        "tokenizer_runtime": plan["tokenizer_runtime"],
        "correctness": correctness,
        "arrays": {name: _array_descriptor(value) for name, value in arrays.items()},
        "timing_artifact_sha256": hashlib.sha256(timing_payload).hexdigest(),
        "session_state": {"start": start_state, "end": end_state},
        "timed_scope": plan["experiment"]["timed_scope"],
    }
    report["report_sha256"] = canonical_sha256(report)
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise RuntimeError("repository changed during fresh actual benchmark")
    _publish(TIMING_PATH, timing_payload)
    _publish(RUNTIME_REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise RuntimeError("repository changed after fresh actual benchmark")
    print("status=fresh_vocabulary_actual_timing_complete")


if __name__ == "__main__":
    main()
