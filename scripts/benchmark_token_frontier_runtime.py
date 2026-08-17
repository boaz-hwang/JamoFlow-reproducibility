#!/usr/bin/env python3
"""Benchmark the sealed 18-role random-weight BPE systems frontier on MPS."""

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

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from scalar_runtime_core import (
    IncrementalBpeDecoder,
    maximum_normalized_error,
    model_parameter_count,
)
from token_frontier_core import (
    FRONTIER_SPECS,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    RUNTIME_ROLES,
    balanced_role_schedule,
    build_frontier_model,
    parse_role,
)
from token_frontier_protocol import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODEL_SEED,
    MPS_ATOL,
    MPS_RTOL,
    OPPORTUNITY_REPORT_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROMPT_BYTES,
    PROTOCOL_ID,
    REPETITIONS,
    ROOT,
    RUNTIME_ACTIVE_PATH,
    RUNTIME_REPORT_PATH,
    TIMING_PATH,
    WARMUP_CASES,
    array_sha256,
    canonical_sha256,
    current_frontier_environment,
    encode_case,
    hash_file,
    json_bytes,
    load_tokenizers,
    read_json,
    reconstruct_cases,
    validate_plan,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_plan_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("token frontier timing requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    last_change = _command(
        "git", "log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))
    )
    if len(commit) != 40 or last_change != commit:
        raise ValueError("token frontier plan must be committed at current HEAD")
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


def _assert_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    value = maximum_normalized_error(actual, expected, rtol=MPS_RTOL, atol=MPS_ATOL)
    if not math.isfinite(value) or value > 1.0:
        raise AssertionError("token frontier cache/full tolerance differs")
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
    maximum = max(maximum, _assert_error(logits, full[:, 0]))
    comparisons += 1
    argmax_comparisons += int(
        torch.equal(logits.argmax(dim=-1), full[:, 0].argmax(dim=-1))
    )
    for position, token_id in enumerate(ids[1:-1], start=1):
        logits = sequential.consume(token_id)
        maximum = max(maximum, _assert_error(logits, full[:, position]))
        comparisons += 1
        argmax_comparisons += int(
            torch.equal(logits.argmax(dim=-1), full[:, position].argmax(dim=-1))
        )
    parallel = IncrementalBpeDecoder(model)
    logits = parallel.prefill_parallel(prompt_ids)
    position = len(prompt_ids) - 1
    maximum = max(maximum, _assert_error(logits, full[:, position]))
    comparisons += 1
    argmax_comparisons += int(
        torch.equal(logits.argmax(dim=-1), full[:, position].argmax(dim=-1))
    )
    for offset, token_id in enumerate(continuation_ids[:-1]):
        logits = parallel.consume(token_id)
        position = len(prompt_ids) + offset
        maximum = max(maximum, _assert_error(logits, full[:, position]))
        comparisons += 1
        argmax_comparisons += int(
            torch.equal(logits.argmax(dim=-1), full[:, position].argmax(dim=-1))
        )
    if argmax_comparisons != comparisons:
        raise AssertionError("token frontier cache/full argmax differs")
    return {
        "argmax_comparisons": argmax_comparisons,
        "comparisons": comparisons,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def _prepare_payloads(
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> list[dict[int, tuple[tuple[int, ...], tuple[int, ...]]]]:
    tokenizers = load_tokenizers()
    output = []
    for prompt_row, continuation_row in zip(prompts, continuations, strict=True):
        row = {}
        for size, (tokenizer, table) in tokenizers.items():
            row[size] = (
                encode_case(bytes(prompt_row), tokenizer, table),
                encode_case(bytes(continuation_row), tokenizer, table),
            )
        output.append(row)
    return output


def _timed_trial(
    model: Any,
    payload: tuple[Sequence[int], Sequence[int]],
) -> tuple[float, float, float, int]:
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
            raise AssertionError("token frontier cache length differs")
        torch.mps.synchronize()
        finished = time.perf_counter_ns()
    del runtime
    return (
        (first - started) / 1_000_000,
        (finished - first) / 1_000_000,
        (finished - started) / 1_000_000,
        len(continuation_ids),
    )


def main() -> None:
    commit = _require_clean_plan_commit()
    if not OPPORTUNITY_REPORT_PATH.exists():
        raise FileNotFoundError("token frontier tokenizer audit must run first")
    if any(
        path.exists()
        for path in (RUNTIME_REPORT_PATH, TIMING_PATH, RUNTIME_ACTIVE_PATH, OUTPUT_PATH)
    ):
        raise FileExistsError("token frontier timing namespace is not empty")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _command("git", "rev-parse", "HEAD^") != plan["dependencies"][
        "git_commit_before_plan"
    ]:
        raise ValueError("token frontier plan parent differs")
    tokenizer_report = read_json(OPPORTUNITY_REPORT_PATH)
    if (
        tokenizer_report.get("complete") is not True
        or tokenizer_report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or tokenizer_report.get("git_commit") != commit
    ):
        raise ValueError("token frontier tokenizer report identity differs")
    for size, entry in tokenizer_report["tokenizer_artifacts"].items():
        path = ROOT / entry["path"]
        if hash_file(path) != entry["sha256"] or int(size) not in parse_sizes():
            raise ValueError("token frontier tokenizer artifact differs")
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise ValueError("token frontier cases changed after sealing")
    payloads = _prepare_payloads(prompts, continuations)
    _publish(
        RUNTIME_ACTIVE_PATH,
        json_bytes({"git_commit": commit, "plan_artifact_sha256": hash_file(PLAN_PATH)}),
    )
    with publication_mps_exclusive():
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise ValueError("token frontier environment is ineligible at start")
        models = {}
        for role in RUNTIME_ROLES:
            model = build_frontier_model(role, seed=MODEL_SEED).eval().to("mps")
            count = model_parameter_count(model)
            if count != FRONTIER_SPECS[role].expected_parameters:
                raise ValueError("token frontier actual parameter count differs")
            if abs(count / PARAMETER_TARGET - 1.0) > PARAMETER_RELATIVE_TOLERANCE:
                raise ValueError("token frontier parameter tolerance failed")
            models[role] = model
        correctness = {}
        with torch.inference_mode():
            for role in RUNTIME_ROLES:
                vocabulary_size, _ = parse_role(role)
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
            raise AssertionError("token frontier correctness failed")
        for index in range(WARMUP_CASES):
            for role in RUNTIME_ROLES:
                vocabulary_size, _ = parse_role(role)
                _timed_trial(models[role], payloads[index][vocabulary_size])
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
                for role in balanced_role_schedule(measured_index, repetition):
                    vocabulary_size, _ = parse_role(role)
                    ttft, decode, end_to_end, steps = _timed_trial(
                        models[role], payloads[case_index][vocabulary_size]
                    )
                    arrays[f"ttft_ms__{role}"][measured_index, repetition] = ttft
                    arrays[f"decode_ms__{role}"][measured_index, repetition] = decode
                    arrays[f"end_to_end_ms__{role}"][measured_index, repetition] = end_to_end
                    arrays[f"continuation_steps__{role}"][measured_index, repetition] = steps
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise ValueError("token frontier environment is ineligible at end")
        environment = current_frontier_environment()
        if environment != plan["environment"]:
            raise ValueError("token frontier timing environment changed after plan seal")
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
        raise ValueError("repository changed during token frontier timing")
    timing_payload = _npz_bytes(arrays)
    _publish(TIMING_PATH, timing_payload)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "korean_bpe_systems_frontier_runtime_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "tokenizer_report_artifact_sha256": hash_file(OPPORTUNITY_REPORT_PATH),
        "timing_artifact_sha256": hashlib.sha256(timing_payload).hexdigest(),
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": array_sha256(value)}
            for name, value in arrays.items()
        },
        "cases": cases,
        "correctness": correctness,
        "environment": environment,
        "model_specs": {role: FRONTIER_SPECS[role].to_dict() for role in RUNTIME_ROLES},
        "parameter_counts": parameter_counts,
        "session_state": {"start": start_state, "end": end_state},
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(RUNTIME_REPORT_PATH, json_bytes(report))
    RUNTIME_ACTIVE_PATH.unlink()
    print(f"wrote {RUNTIME_REPORT_PATH.relative_to(ROOT)}")
    print(f"wrote {TIMING_PATH.relative_to(ROOT)}")


def parse_sizes() -> set[int]:
    return {parse_role(role)[0] for role in RUNTIME_ROLES}


if __name__ == "__main__":
    main()
