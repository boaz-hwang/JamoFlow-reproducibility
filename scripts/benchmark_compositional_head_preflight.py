#!/usr/bin/env python3
"""Benchmark the sealed random-weight compositional-head systems grid on MPS."""

from __future__ import annotations

import gc
import hashlib
import io
import math
import os
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from compositional_head_core import (
    BASE_ROLE,
    MODEL_SEED,
    ROLE_ORDER,
    ROLE_SPECS,
    balanced_role_schedule,
    build_model,
    parse_role,
)
from compositional_head_preflight_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    MEASURED_CASES,
    MPS_ATOL,
    MPS_RTOL,
    PLAN_PATH,
    PROTOCOL_ID,
    REPETITIONS,
    REPORT_PATH,
    RESULT_PATH,
    ROOT,
    TIMING_PATH,
    WARMUP_CASES,
    array_sha256,
    canonical_sha256,
    case_identity,
    current_environment,
    hash_file,
    json_bytes,
    load_tokenizers,
    read_json,
    validate_plan,
)
from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from scalar_runtime_core import (
    IncrementalBpeDecoder,
    maximum_normalized_error,
    model_parameter_count,
)
from token_frontier_protocol import encode_case, reconstruct_cases


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


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


def _publish(path, payload: bytes) -> None:
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


def _maximum_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    value = maximum_normalized_error(actual, expected, rtol=MPS_RTOL, atol=MPS_ATOL)
    if not math.isfinite(value) or value > 1.0:
        raise AssertionError("compositional-head cache/full tolerance differs")
    return value


def _verify_model(
    model: Any,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
) -> dict[str, Any]:
    ids = tuple(prompt_ids) + tuple(continuation_ids)
    values = torch.tensor([list(ids)], dtype=torch.long, device="mps")
    full = model(input_ids=values, use_cache=False).logits.float()
    maximum = 0.0
    comparisons = 0
    argmax_comparisons = 0
    sequential = IncrementalBpeDecoder(model)
    logits = sequential.prefill_parallel(ids[:1])
    maximum = max(maximum, _maximum_error(logits, full[:, 0]))
    comparisons += 1
    argmax_comparisons += int(torch.equal(logits.argmax(-1), full[:, 0].argmax(-1)))
    for position, token_id in enumerate(ids[1:-1], start=1):
        logits = sequential.consume(token_id)
        maximum = max(maximum, _maximum_error(logits, full[:, position]))
        comparisons += 1
        argmax_comparisons += int(
            torch.equal(logits.argmax(-1), full[:, position].argmax(-1))
        )
    parallel = IncrementalBpeDecoder(model)
    logits = parallel.prefill_parallel(prompt_ids)
    position = len(prompt_ids) - 1
    maximum = max(maximum, _maximum_error(logits, full[:, position]))
    comparisons += 1
    argmax_comparisons += int(
        torch.equal(logits.argmax(-1), full[:, position].argmax(-1))
    )
    for offset, token_id in enumerate(continuation_ids[:-1]):
        logits = parallel.consume(token_id)
        position = len(prompt_ids) + offset
        maximum = max(maximum, _maximum_error(logits, full[:, position]))
        comparisons += 1
        argmax_comparisons += int(
            torch.equal(logits.argmax(-1), full[:, position].argmax(-1))
        )
    if argmax_comparisons != comparisons:
        raise AssertionError("compositional-head cached argmax differs")
    return {
        "argmax_comparisons": argmax_comparisons,
        "comparisons": comparisons,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def _payloads(prompts: np.ndarray, continuations: np.ndarray):
    tokenizers = load_tokenizers()
    output = []
    for prompt, continuation in zip(prompts, continuations, strict=True):
        row = {}
        for vocabulary_size, (tokenizer, table) in tokenizers.items():
            row[vocabulary_size] = (
                encode_case(bytes(prompt), tokenizer, table),
                encode_case(bytes(continuation), tokenizer, table),
            )
        output.append(row)
    return output, tokenizers


def _timed_trial(model: Any, payload) -> tuple[float, float, float, int]:
    prompt_ids, continuation_ids = payload
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        runtime = IncrementalBpeDecoder(model)
        logits = runtime.prefill_parallel(prompt_ids)
        _ = logits.argmax(dim=-1)
        torch.mps.synchronize()
        first = time.perf_counter_ns()
        for token_id in continuation_ids[:-1]:
            logits = runtime.consume(token_id)
            _ = logits.argmax(dim=-1)
        observed = len(prompt_ids) + len(continuation_ids) - 1
        if runtime.observed_tokens != observed:
            raise AssertionError("compositional-head cache length differs")
        torch.mps.synchronize()
        finished = time.perf_counter_ns()
    return (
        (first - started) / 1_000_000,
        (finished - first) / 1_000_000,
        (finished - started) / 1_000_000,
        len(continuation_ids),
    )


def _buffer_bytes(model: Any) -> int:
    return sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compositional-head timing requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("compositional-head plan must be current HEAD")
    if any(path.exists() for path in (ACTIVE_PATH, TIMING_PATH, REPORT_PATH, RESULT_PATH)):
        raise RuntimeError("compositional-head timing namespace is not empty")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("compositional-head plan parent differs")
    prompts, continuations, _ = reconstruct_cases()
    if case_identity() != plan["cases"]:
        raise RuntimeError("compositional-head cases changed")
    payloads, tokenizers = _payloads(prompts, continuations)
    _publish(
        ACTIVE_PATH,
        json_bytes(
            {
                "git_commit": commit,
                "kind": "compositional_head_preflight_active_v2",
                "plan_artifact_sha256": hash_file(PLAN_PATH),
            }
        ),
    )
    with publication_mps_exclusive():
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise RuntimeError("compositional-head timing environment is ineligible")
        models = {}
        for role in ROLE_ORDER:
            _, vocabulary_size = parse_role(role)
            table = tokenizers[vocabulary_size][1]
            models[role] = build_model(
                role,
                token_bytes=table if "_code_" in role else None,
                seed=MODEL_SEED,
            ).eval().to("mps")
        correctness = {}
        with torch.inference_mode():
            for role in ROLE_ORDER:
                _, vocabulary_size = parse_role(role)
                rows = [
                    _verify_model(models[role], *payloads[index][vocabulary_size])
                    for index in range(WARMUP_CASES)
                ]
                correctness[role] = {
                    "argmax_comparisons": sum(row["argmax_comparisons"] for row in rows),
                    "cases": len(rows),
                    "comparisons": sum(row["comparisons"] for row in rows),
                    "maximum_normalized_tolerance_ratio": max(
                        row["maximum_normalized_tolerance_ratio"] for row in rows
                    ),
                    "pass": all(row["pass"] for row in rows),
                }
        if not all(row["pass"] for row in correctness.values()):
            raise AssertionError("compositional-head correctness failed")
        for index in range(WARMUP_CASES):
            for role in ROLE_ORDER:
                _, vocabulary_size = parse_role(role)
                _timed_trial(models[role], payloads[index][vocabulary_size])
        arrays = {
            f"{component}__{role}": np.zeros(
                (MEASURED_CASES, REPETITIONS), dtype=np.float64
            )
            for role in ROLE_ORDER
            for component in ("ttft_ms", "decode_ms", "end_to_end_ms")
        }
        arrays.update(
            {
                f"continuation_steps__{role}": np.zeros(
                    (MEASURED_CASES, REPETITIONS), dtype=np.int64
                )
                for role in ROLE_ORDER
            }
        )
        for measured_index in range(MEASURED_CASES):
            case_index = WARMUP_CASES + measured_index
            for repetition in range(REPETITIONS):
                for role in balanced_role_schedule(measured_index, repetition):
                    _, vocabulary_size = parse_role(role)
                    ttft, decode, end_to_end, steps = _timed_trial(
                        models[role], payloads[case_index][vocabulary_size]
                    )
                    arrays[f"ttft_ms__{role}"][measured_index, repetition] = ttft
                    arrays[f"decode_ms__{role}"][measured_index, repetition] = decode
                    arrays[f"end_to_end_ms__{role}"][measured_index, repetition] = end_to_end
                    arrays[f"continuation_steps__{role}"][measured_index, repetition] = steps
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError("compositional-head timing environment changed")
        environment = current_environment()
        if environment != plan["environment"]:
            raise RuntimeError("compositional-head runtime environment drifted")
        parameters = {role: model_parameter_count(models[role]) for role in ROLE_ORDER}
        buffers = {role: _buffer_bytes(models[role]) for role in ROLE_ORDER}
        for role in ROLE_ORDER:
            if parameters[role] != ROLE_SPECS[role].expected_parameters:
                raise AssertionError("compositional-head actual parameters differ")
            models[role].to("cpu")
        models.clear()
        gc.collect()
        torch.mps.empty_cache()
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during compositional-head timing")
    timing_payload = _npz_bytes(arrays)
    _publish(TIMING_PATH, timing_payload)
    report = {
        "schema_version": 2,
        "kind": "compositional_head_systems_preflight_report_v2",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "timing_artifact_sha256": hashlib.sha256(timing_payload).hexdigest(),
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
            for name, value in arrays.items()
        },
        "correctness": correctness,
        "parameter_counts": parameters,
        "runtime_buffer_bytes": buffers,
        "model_contract": plan["model_contract"],
        "assignment_audits": plan["assignment_audits"],
        "cases": plan["cases"],
        "environment": environment,
        "session_state": {"start": start_state, "end": end_state},
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print(f"report={REPORT_PATH.relative_to(ROOT)}")
    print(f"timings={TIMING_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
