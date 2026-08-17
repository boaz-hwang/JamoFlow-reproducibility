#!/usr/bin/env python3
"""Validate and summarize the sealed scalar/BPE runtime preflight."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from jamoflow.actual_inference_protocol import timing_environment_eligible
from scalar_runtime_protocol import (
    MEASURED_CASES,
    OUTPUT_PATH,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    PLAN_PATH,
    PROTOCOL_ID,
    REPETITIONS,
    REPORT_PATH,
    ROOT,
    RUNTIME_ROLES,
    TIMING_PATH,
    array_sha256,
    canonical_sha256,
    comparison_summary,
    hash_file,
    json_bytes,
    read_json,
    reconstruct_cases,
    validate_plan,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_root() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("scalar runtime summary requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("scalar runtime summary requires a Git commit")
    return commit


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"scalar runtime summary already exists: {path}")
    history = subprocess.run(
        ["git", "log", "--all", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if history:
        raise FileExistsError("scalar runtime summary has prior Git history")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_arrays(report: dict) -> dict[str, np.ndarray]:
    if hash_file(TIMING_PATH) != report["timing_artifact_sha256"]:
        raise ValueError("scalar runtime timing artifact differs")
    expected = {
        f"{component}__{role}"
        for role in RUNTIME_ROLES
        for component in (
            "ttft_ms",
            "decode_ms",
            "end_to_end_ms",
            "continuation_steps",
        )
    }
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        if set(archive.files) != expected or set(report["arrays"]) != expected:
            raise ValueError("scalar runtime array set differs")
        arrays = {name: archive[name] for name in archive.files}
    for name, value in arrays.items():
        expected_dtype = np.int64 if name.startswith("continuation_steps") else np.float64
        if (
            value.dtype != expected_dtype
            or value.shape != (MEASURED_CASES, REPETITIONS)
            or array_sha256(value) != report["arrays"][name]["sha256"]
            or report["arrays"][name]["dtype"] != str(value.dtype)
            or report["arrays"][name]["shape"] != list(value.shape)
        ):
            raise ValueError(f"scalar runtime array identity differs: {name}")
        if name.startswith("continuation_steps"):
            if np.any(value <= 0):
                raise ValueError("scalar runtime continuation steps are invalid")
        elif not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError("scalar runtime latency is nonpositive or non-finite")
    return arrays


def _validate_report(plan: dict, report: dict) -> None:
    expected = {
        "arrays",
        "case_metadata",
        "claim_boundary",
        "complete",
        "correctness",
        "environment",
        "git_commit",
        "kind",
        "parameter_counts",
        "plan_artifact_sha256",
        "protocol_id",
        "report_sha256",
        "schema_version",
        "session_state",
        "timing_artifact_sha256",
    }
    if set(report) != expected:
        raise ValueError("scalar runtime report schema differs")
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if (
        report["kind"] != "scalar_runtime_preflight_report_v1"
        or report["schema_version"] != 1
        or report["protocol_id"] != PROTOCOL_ID
        or report["complete"] is not True
        or canonical_sha256(unsigned) != report["report_sha256"]
        or report["plan_artifact_sha256"] != hash_file(PLAN_PATH)
        or report["claim_boundary"] != plan["claim_boundary"]
        or report["case_metadata"] != reconstruct_cases()[2]
    ):
        raise ValueError("scalar runtime report identity differs")
    if set(report["correctness"]) != set(RUNTIME_ROLES):
        raise ValueError("scalar runtime correctness role set differs")
    for role, evidence in report["correctness"].items():
        if (
            set(evidence)
            != {
                "cases",
                "comparisons",
                "maximum_normalized_tolerance_ratio",
                "pass",
            }
            or evidence["cases"] != plan["benchmark"]["correctness_cases"]
            or evidence["comparisons"] <= 0
            or evidence["pass"] is not True
            or not math.isfinite(evidence["maximum_normalized_tolerance_ratio"])
            or evidence["maximum_normalized_tolerance_ratio"] < 0
            or evidence["maximum_normalized_tolerance_ratio"] > 1
        ):
            raise ValueError(f"scalar runtime correctness differs: {role}")
    if report["parameter_counts"] != {
        role: plan["graphs"][role]["parameter_count"] for role in RUNTIME_ROLES
    }:
        raise ValueError("scalar runtime parameter counts differ")
    if any(
        abs(value / PARAMETER_TARGET - 1.0) > PARAMETER_RELATIVE_TOLERANCE
        for value in report["parameter_counts"].values()
    ):
        raise ValueError("scalar runtime parameter match failed")
    if report["environment"].get("device") != "mps" or report["environment"].get("mps_available") is not True:
        raise ValueError("scalar runtime did not run on MPS")
    if not all(
        timing_environment_eligible(report["session_state"][key])
        for key in ("start", "end")
    ):
        raise ValueError("scalar runtime thermal/power state was ineligible")


def _candidate_decision(
    role: str,
    comparisons: dict[str, dict],
    correctness: bool,
) -> dict:
    byte = comparisons[f"{role}_vs_byte_w72"]
    bpe32 = comparisons[f"{role}_vs_byte_bpe_32000"]
    bpe16 = comparisons[f"{role}_vs_byte_bpe_16000"]
    checks = {
        "all_correctness": correctness,
        "byte_bootstrap_lower_positive": byte["bootstrap_percentile_95_lower"] > 0,
        "byte_median_reduction_at_least_10_percent": byte["median_latency_reduction"] >= 0.10,
        "byte_positive_prompts_at_least_28": byte["positive_prompt_count"] >= 28,
        "bpe16_lower_bound_not_worse_than_minus_10_percent": bpe16["bootstrap_percentile_95_lower"] >= -0.10,
        "bpe32_lower_bound_not_worse_than_minus_10_percent": bpe32["bootstrap_percentile_95_lower"] >= -0.10,
    }
    if role == "hangul_hybrid":
        hybrid = comparisons["hangul_hybrid_vs_generic_unicode_scalar"]
        checks["hangul_specific_lower_bound_not_worse_than_minus_5_percent"] = (
            hybrid["bootstrap_percentile_95_lower"] >= -0.05
        )
    return {"checks": checks, "pass": all(checks.values())}


def main() -> None:
    commit = _require_clean_root()
    _require_never_published(OUTPUT_PATH)
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    report = read_json(REPORT_PATH)
    _validate_report(plan, report)
    arrays = _load_arrays(report)
    comparisons = {}
    for candidate in ("generic_unicode_scalar", "hangul_hybrid"):
        for reference in ("byte_w72", "byte_bpe_32000", "byte_bpe_16000"):
            comparisons[f"{candidate}_vs_{reference}"] = comparison_summary(
                arrays[f"end_to_end_ms__{candidate}"],
                arrays[f"end_to_end_ms__{reference}"],
            )
    comparisons["hangul_hybrid_vs_generic_unicode_scalar"] = comparison_summary(
        arrays["end_to_end_ms__hangul_hybrid"],
        arrays["end_to_end_ms__generic_unicode_scalar"],
    )
    correctness = all(value["pass"] for value in report["correctness"].values())
    candidates = {
        role: _candidate_decision(role, comparisons, correctness)
        for role in ("generic_unicode_scalar", "hangul_hybrid")
    }
    authorized = [role for role, decision in candidates.items() if decision["pass"]]
    medians = {
        role: {
            component: float(
                np.median(arrays[f"{component}__{role}"])
            )
            for component in ("ttft_ms", "decode_ms", "end_to_end_ms")
        }
        for role in RUNTIME_ROLES
    }
    step_counts = {
        role: {
            "maximum": int(arrays[f"continuation_steps__{role}"].max()),
            "median": float(np.median(arrays[f"continuation_steps__{role}"])),
            "minimum": int(arrays[f"continuation_steps__{role}"].min()),
        }
        for role in RUNTIME_ROLES
    }
    summary = {
        "claim_boundary": plan["claim_boundary"],
        "complete": True,
        "decision": {
            "authorized_one_seed_quality_candidates": authorized,
            "candidate_decisions": candidates,
            "pass": bool(authorized),
            "status": (
                "one_seed_quality_training_authorized"
                if authorized
                else "scalar_runtime_branch_stopped"
            ),
        },
        "git_commit": commit,
        "interpretation": {
            "bpe_sequence_is_materially_shorter": True,
            "controlled_fixed_route_only": True,
            "matched_quality_required_before_speed_claim": True,
            "random_weight_graph_feasibility_only": True,
            "timing_excludes_tokenization_and_unit_encoding": True,
        },
        "kind": "scalar_runtime_preflight_summary_v1",
        "metrics": {
            "comparisons": comparisons,
            "continuation_steps": step_counts,
            "correctness": report["correctness"],
            "latency_medians_ms": medians,
            "parameter_counts": report["parameter_counts"],
        },
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "protocol_id": PROTOCOL_ID,
        "report_artifact_sha256": hash_file(REPORT_PATH),
        "schema_version": 1,
        "timing_artifact_sha256": hash_file(TIMING_PATH),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    _publish(OUTPUT_PATH, json_bytes(summary))
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
