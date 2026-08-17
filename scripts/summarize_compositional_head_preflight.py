#!/usr/bin/env python3
"""Validate and summarize the sealed compositional-head systems preflight."""

from __future__ import annotations

import math
import os
import subprocess
from typing import Any, Mapping

import numpy as np

from compositional_head_core import (
    BASE_ROLE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    ROLE_ORDER,
    ROLE_SPECS,
    analytical_head_multiply_adds_per_position,
    paired_latency_comparison,
    parameter_fraction_from_baseline,
    parse_role,
    preflight_decision,
)
from compositional_head_preflight_protocol import (
    ACTIVE_PATH,
    MEASURED_CASES,
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
    hash_file,
    json_bytes,
    load_tokenizers,
    read_json,
    validate_plan,
)
from jamoflow.actual_inference_protocol import timing_environment_eligible
from token_frontier_protocol import encode_case, reconstruct_cases


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_clean_root() -> str:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compositional-head summary requires a clean worktree")
    return _git("rev-parse", "HEAD")


def _history(path) -> tuple[str, ...]:
    output = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    return tuple(line for line in output.splitlines() if line)


def _require_single_current_commit(path, commit: str) -> None:
    if not path.is_file() or path.is_symlink() or _history(path) != (commit,):
        raise RuntimeError(
            f"compositional-head evidence is not a one-add current-HEAD blob: {path}"
        )


def _require_binary_head_blob(path) -> None:
    result = subprocess.run(
        ("git", "show", f"HEAD:{path.relative_to(ROOT)}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    if result.stdout != path.read_bytes():
        raise RuntimeError("compositional-head binary evidence differs from HEAD")


def _unsigned_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field)
    return canonical_sha256(unsigned)


def _expected_payloads() -> tuple[dict[str, int], dict[str, np.ndarray]]:
    prompts, continuations, _ = reconstruct_cases()
    tokenizers = load_tokenizers()
    expected_comparisons = {role: 0 for role in ROLE_ORDER}
    expected_steps = {
        role: np.empty((MEASURED_CASES, REPETITIONS), dtype=np.int64)
        for role in ROLE_ORDER
    }
    for role in ROLE_ORDER:
        _, vocabulary_size = parse_role(role)
        tokenizer, table = tokenizers[vocabulary_size]
        for case_index, (prompt, continuation) in enumerate(
            zip(prompts, continuations, strict=True)
        ):
            prompt_ids = encode_case(bytes(prompt), tokenizer, table)
            continuation_ids = encode_case(bytes(continuation), tokenizer, table)
            if case_index < WARMUP_CASES:
                expected_comparisons[role] += (
                    len(prompt_ids) + 2 * len(continuation_ids) - 1
                )
            else:
                expected_steps[role][case_index - WARMUP_CASES, :] = len(
                    continuation_ids
                )
    return expected_comparisons, expected_steps


def _validate_report(
    plan: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    plan_commit: str,
) -> tuple[dict[str, bool], dict[str, np.ndarray]]:
    expected_keys = {
        "arrays",
        "assignment_audits",
        "cases",
        "complete",
        "correctness",
        "environment",
        "git_commit",
        "kind",
        "model_contract",
        "parameter_counts",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "report_sha256",
        "runtime_buffer_bytes",
        "schema_version",
        "session_state",
        "timing_artifact_sha256",
    }
    if set(report) != expected_keys or (
        report["schema_version"] != 2
        or report["kind"] != "compositional_head_systems_preflight_report_v2"
        or report["protocol_id"] != PROTOCOL_ID
        or report["complete"] is not True
        or report["git_commit"] != plan_commit
        or report["plan_artifact_sha256"] != hash_file(PLAN_PATH)
        or report["plan_sha256"] != plan["plan_sha256"]
        or report["timing_artifact_sha256"] != hash_file(TIMING_PATH)
        or _unsigned_hash(report, "report_sha256") != report["report_sha256"]
        or report["model_contract"] != plan["model_contract"]
        or report["assignment_audits"] != plan["assignment_audits"]
        or report["cases"] != case_identity()
        or report["environment"] != plan["environment"]
    ):
        raise ValueError("compositional-head report identity differs")
    if set(report["session_state"]) != {"start", "end"} or not all(
        timing_environment_eligible(report["session_state"][key])
        for key in ("start", "end")
    ):
        raise ValueError("compositional-head timing environment differs")
    expected_parameters = {
        role: ROLE_SPECS[role].expected_parameters for role in ROLE_ORDER
    }
    if report["parameter_counts"] != expected_parameters:
        raise ValueError("compositional-head parameter evidence differs")
    if set(report["runtime_buffer_bytes"]) != set(ROLE_ORDER) or any(
        not isinstance(value, int) or value < 0
        for value in report["runtime_buffer_bytes"].values()
    ):
        raise ValueError("compositional-head runtime buffer evidence differs")

    expected_comparisons, expected_steps = _expected_payloads()
    if set(report["correctness"]) != set(ROLE_ORDER):
        raise ValueError("compositional-head correctness role set differs")
    correctness = {}
    for role, row in report["correctness"].items():
        if set(row) != {
            "argmax_comparisons",
            "cases",
            "comparisons",
            "maximum_normalized_tolerance_ratio",
            "pass",
        } or (
            row["cases"] != WARMUP_CASES
            or row["comparisons"] != expected_comparisons[role]
            or row["argmax_comparisons"] != row["comparisons"]
            or row["pass"] is not True
            or not math.isfinite(row["maximum_normalized_tolerance_ratio"])
            or not 0 <= row["maximum_normalized_tolerance_ratio"] <= 1
        ):
            raise ValueError(f"compositional-head correctness differs: {role}")
        correctness[role] = True

    expected_array_names = {
        f"{component}__{role}"
        for role in ROLE_ORDER
        for component in (
            "ttft_ms",
            "decode_ms",
            "end_to_end_ms",
            "continuation_steps",
        )
    }
    if set(report["arrays"]) != expected_array_names:
        raise ValueError("compositional-head array descriptors differ")
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        if set(archive.files) != expected_array_names:
            raise ValueError("compositional-head timing array set differs")
        arrays = {name: archive[name] for name in archive.files}
    for name, values in arrays.items():
        expected_dtype = "int64" if name.startswith("continuation_steps__") else "float64"
        if (
            str(values.dtype) != expected_dtype
            or values.shape != (MEASURED_CASES, REPETITIONS)
            or report["arrays"][name]
            != {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(f"compositional-head timing array differs: {name}")
    for role in ROLE_ORDER:
        if not np.array_equal(arrays[f"continuation_steps__{role}"], expected_steps[role]):
            raise ValueError(f"compositional-head continuation steps differ: {role}")
    return correctness, arrays


def _metric(role: str, arrays: Mapping[str, np.ndarray], report) -> dict[str, Any]:
    end_to_end = arrays[f"end_to_end_ms__{role}"]
    decode = arrays[f"decode_ms__{role}"]
    steps = arrays[f"continuation_steps__{role}"]
    kind, vocabulary_size = parse_role(role)
    return {
        "head_kind": kind,
        "vocabulary_size": vocabulary_size,
        "parameters": report["parameter_counts"][role],
        "parameter_fraction_from_baseline": parameter_fraction_from_baseline(role),
        "runtime_buffer_bytes": report["runtime_buffer_bytes"][role],
        "analytical_head_multiply_adds_per_position": (
            analytical_head_multiply_adds_per_position(role)
        ),
        "ttft_median_ms": float(
            np.median(np.median(arrays[f"ttft_ms__{role}"], axis=1))
        ),
        "decode_median_ms": float(np.median(np.median(decode, axis=1))),
        "end_to_end_median_ms": float(
            np.median(np.median(end_to_end, axis=1))
        ),
        "decode_ms_per_continuation_step_median": float(
            np.median(np.median(decode / steps, axis=1))
        ),
        "continuation_steps_median": float(
            np.median(np.median(steps, axis=1))
        ),
    }


def main() -> None:
    commit = _require_clean_root()
    if RESULT_PATH.exists() or _history(RESULT_PATH):
        raise RuntimeError("compositional-head summary already exists or has history")
    if ACTIVE_PATH.exists():
        raise RuntimeError("compositional-head benchmark remains active")
    _require_single_current_commit(REPORT_PATH, commit)
    _require_single_current_commit(TIMING_PATH, commit)
    _require_binary_head_blob(REPORT_PATH)
    _require_binary_head_blob(TIMING_PATH)
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    plan_commit = _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT)))
    if _git("rev-parse", "HEAD^") != plan_commit or _git(
        "rev-parse", f"{plan_commit}^"
    ) != plan["git_commit_before_plan"]:
        raise RuntimeError("compositional-head evidence chronology differs")
    report = read_json(REPORT_PATH)
    correctness, arrays = _validate_report(plan, report, plan_commit=plan_commit)
    comparisons = {
        role: paired_latency_comparison(
            arrays[f"end_to_end_ms__{role}"],
            arrays[f"end_to_end_ms__{BASE_ROLE}"],
            arrays[f"continuation_steps__{role}"],
            arrays[f"continuation_steps__{BASE_ROLE}"],
            bootstrap_seed=BOOTSTRAP_SEED + role_index,
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        )
        for role_index, role in enumerate(ROLE_ORDER)
        if role != BASE_ROLE
    }
    decision = preflight_decision(comparisons, correctness)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "kind": "compositional_head_systems_preflight_result_v2",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "report_artifact_sha256": hash_file(REPORT_PATH),
        "timing_artifact_sha256": hash_file(TIMING_PATH),
        "metrics": {
            role: _metric(role, arrays, report) for role in ROLE_ORDER
        },
        "comparisons_to_dense_2k": comparisons,
        "decision": decision,
        "correctness": report["correctness"],
        "assignment_audits": report["assignment_audits"],
        "claim_boundary": {
            **plan["claim_boundary"],
            "controlled_continuation_replay_not_free_generation": True,
            "evidence_committed_before_summary": True,
            "systems_gate_does_not_establish_trained_quality": True,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during compositional-head summary")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(RESULT_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(summary))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote={RESULT_PATH.relative_to(ROOT)}")
    print(f"status={decision['status']}")
    print(f"selected_vocabulary_size={decision['selected_vocabulary_size']}")


if __name__ == "__main__":
    main()
