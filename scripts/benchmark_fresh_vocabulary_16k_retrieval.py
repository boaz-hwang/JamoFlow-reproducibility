#!/usr/bin/env python3
"""Measure actual same-tokenizer retrieval drafting on the trained 16K target."""

from __future__ import annotations

import gc
import hashlib
import io
import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from benchmark_fresh_vocabulary_16k_block import load_target, prepare_payloads
from benchmark_fresh_vocabulary_actual import RoleBundle, verify_sequence
from fresh_vocabulary_16k_retrieval_actual_core import (
    COUNTER_NAMES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    balanced_role_order,
)
from fresh_vocabulary_16k_retrieval_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    MAXIMUM_FREE_TOKENS,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    RUNTIME_REPORT_PATH,
    TIMING_PATH,
    array_sha256,
    canonical_sha256,
    hash_file,
    json_bytes,
    load_table,
    read_json,
    reconstruct_cases,
    validate_plan,
)
from fresh_vocabulary_16k_retrieval_runtime import run_retrieval_trial
from fresh_vocabulary_actual_core import WARMUP_CASES

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


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
        raise RuntimeError("16K retrieval benchmark requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()) != commit:
        raise RuntimeError("16K retrieval plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("16K retrieval plan parent differs")
    return commit, plan


def correctness_replay(
    bundle: RoleBundle,
    table,
    payloads: Sequence[Mapping[str, Any]],
    *,
    continuation_bytes: int,
    maximum_output_bytes: int,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("16K retrieval correctness payloads are empty")
    rows: dict[str, dict[str, dict[str, int | float | bool]]] = {
        role: {
            mode: {
                "cases": 0,
                "target_forward_calls": 0,
                "proposal_attempts": 0,
                "accepted_draft_tokens": 0,
                "outputs_exact": True,
                "cache_lag_exact": True,
            }
            for mode in MODES
        }
        for role in ROLES
    }
    target_comparisons = 0
    target_argmax_exact = 0
    maximum_error = 0.0
    for payload in payloads:
        for mode in MODES:
            ids = payload["controlled_ids" if mode == "controlled_replay" else "free_ids"]
            raw = payload["controlled_raw" if mode == "controlled_replay" else "free_raw"]
            target_row = verify_sequence(bundle, payload["prompt_ids"], ids)
            target_comparisons += int(target_row["comparisons"])
            target_argmax_exact += int(target_row["argmax_exact"])
            maximum_error = max(
                maximum_error,
                float(target_row["maximum_normalized_tolerance_ratio"]),
            )
            for role in ROLES:
                _metrics, trace = run_retrieval_trial(
                    bundle,
                    payload["prompt_raw"],
                    payload["prompt_ids"],
                    ids,
                    raw,
                    table,
                    role=role,
                    mode=mode,
                    continuation_bytes=continuation_bytes,
                    maximum_output_bytes=maximum_output_bytes,
                )
                row = rows[role][mode]
                row["cases"] = int(row["cases"]) + 1
                row["target_forward_calls"] = int(row["target_forward_calls"]) + int(
                    trace.counters["target_forward_calls"]
                )
                row["proposal_attempts"] = int(row["proposal_attempts"]) + int(
                    trace.counters["proposal_attempts"]
                )
                row["accepted_draft_tokens"] = int(row["accepted_draft_tokens"]) + int(
                    trace.counters["accepted_draft_tokens"]
                )
                row["outputs_exact"] = bool(
                    row["outputs_exact"] and trace.token_ids == tuple(ids) and trace.raw == raw
                )
                row["cache_lag_exact"] = bool(
                    row["cache_lag_exact"]
                    and trace.observed_tokens == len(payload["prompt_ids"]) + len(ids) - 1
                )
    overall = bool(
        target_comparisons == target_argmax_exact
        and maximum_error <= 1.0
        and all(
            row["cases"] == len(payloads)
            and row["outputs_exact"] is True
            and row["cache_lag_exact"] is True
            for role in ROLES
            for row in rows[role].values()
        )
    )
    return {
        "target_cache_full": {
            "comparisons": target_comparisons,
            "argmax_exact": target_argmax_exact,
            "maximum_normalized_tolerance_ratio": maximum_error,
            "pass": target_comparisons == target_argmax_exact and maximum_error <= 1.0,
        },
        "by_role_mode": rows,
        "overall_pass": overall,
    }


def _store_free_trace(
    arrays: Mapping[str, np.ndarray],
    *,
    case_index: int,
    repetition: int,
    role_index: int,
    output_ids: Sequence[int],
    output: bytes,
) -> None:
    arrays["free_token_ids"][case_index, repetition, role_index, : len(output_ids)] = np.asarray(
        output_ids, dtype=np.int32
    )
    arrays["free_output_bytes"][case_index, repetition, role_index, : len(output)] = np.frombuffer(
        output, dtype=np.uint8
    )
    arrays["free_output_lengths"][case_index, repetition, role_index] = len(output)


def main() -> None:
    commit, plan = _require_plan_commit()
    if any(path.exists() for path in (ACTIVE_PATH, RUNTIME_REPORT_PATH, TIMING_PATH, OUTPUT_PATH)):
        raise FileExistsError("16K retrieval runtime namespace is not empty")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise FileExistsError("16K retrieval artifact namespace is not empty")
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("16K retrieval cases changed after sealing")
    _publish(
        ACTIVE_PATH,
        json_bytes({"git_commit": commit, "plan_artifact_sha256": hash_file(PLAN_PATH)}),
    )
    maximum_output_bytes = int(
        plan["tokenizer_runtime"]["strict_utf8_transitions"]["maximum_free_output_bytes"]
    )
    with publication_mps_exclusive():
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise RuntimeError("16K retrieval timing environment is ineligible")
        table = load_table()
        bundle = load_target(plan)
        payloads = prepare_payloads(bundle, prompts, continuations)
        warmup_correctness = correctness_replay(
            bundle,
            table,
            payloads[:WARMUP_CASES],
            continuation_bytes=plan["experiment"]["continuation_bytes"],
            maximum_output_bytes=maximum_output_bytes,
        )
        if warmup_correctness["overall_pass"] is not True:
            raise AssertionError("16K retrieval warmup correctness failed")

        shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
        arrays: dict[str, np.ndarray] = {
            name: np.empty(shape, dtype=np.float64) for name in TIMING_COMPONENTS
        }
        arrays.update({name: np.empty(shape, dtype=np.int16) for name in COUNTER_NAMES})
        arrays["output_token_count"] = np.empty(shape, dtype=np.int16)
        arrays["output_raw_byte_count"] = np.empty(shape, dtype=np.int16)
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
            payload = payloads[WARMUP_CASES + case_index]
            for repetition in range(REPETITIONS):
                for mode_index, mode in enumerate(MODES):
                    ids = payload[
                        "controlled_ids" if mode == "controlled_replay" else "free_ids"
                    ]
                    raw = payload[
                        "controlled_raw" if mode == "controlled_replay" else "free_raw"
                    ]
                    for role_index in balanced_role_order(case_index, repetition, mode_index):
                        role = ROLES[role_index]
                        metrics, trace = run_retrieval_trial(
                            bundle,
                            payload["prompt_raw"],
                            payload["prompt_ids"],
                            ids,
                            raw,
                            table,
                            role=role,
                            mode=mode,
                            continuation_bytes=plan["experiment"]["continuation_bytes"],
                            maximum_output_bytes=maximum_output_bytes,
                        )
                        for name in TIMING_COMPONENTS:
                            arrays[name][mode_index, case_index, repetition, role_index] = metrics[name]
                        for name in COUNTER_NAMES:
                            arrays[name][mode_index, case_index, repetition, role_index] = trace.counters[name]
                        arrays["output_token_count"][mode_index, case_index, repetition, role_index] = len(
                            trace.token_ids
                        )
                        arrays["output_raw_byte_count"][mode_index, case_index, repetition, role_index] = len(
                            trace.raw
                        )
                        if mode == "free_running_utf8_greedy":
                            _store_free_trace(
                                arrays,
                                case_index=case_index,
                                repetition=repetition,
                                role_index=role_index,
                                output_ids=trace.token_ids,
                                output=trace.raw,
                            )
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError("16K retrieval timing environment is ineligible at end")
        bundle.model.to("cpu")
        del bundle, payloads, table
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    timing_payload = _npz_bytes(arrays)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_retrieval_runtime_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "cases": cases,
        "target": plan["target"],
        "table": plan["table"],
        "tokenizer_runtime": plan["tokenizer_runtime"],
        "warmup_correctness": warmup_correctness,
        "arrays": {name: _array_descriptor(value) for name, value in arrays.items()},
        "timing_artifact_sha256": hashlib.sha256(timing_payload).hexdigest(),
        "session_state": {"start": start_state, "end": end_state},
        "timed_scope": plan["experiment"]["timed_scope"],
    }
    report["report_sha256"] = canonical_sha256(report)
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("repository changed during 16K retrieval benchmark")
    _publish(TIMING_PATH, timing_payload)
    _publish(RUNTIME_REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print(f"measured_trials={len(MODES) * MEASURED_CASES * REPETITIONS * len(ROLES)}")
    print(f"timing_artifact_sha256={report['timing_artifact_sha256']}")
    print(f"report_sha256={report['report_sha256']}")


if __name__ == "__main__":
    main()
