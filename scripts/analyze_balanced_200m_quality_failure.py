#!/usr/bin/env python3
"""Exploratory structural diagnosis of the verified balanced-200M quality failure."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from balanced_200m_failure_analysis_core import (
    ANALYSIS_KIND,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CANDIDATE_PATCH_COUNTS,
    PROTOCOL_ID,
    canonical_bytes,
    canonical_sha256,
    contiguous_block_bootstrap,
    equal_count_quintiles,
    linear_density_heuristic,
    paired_bpb_effects,
    positive_excess_concentration,
    spearman_correlation,
    validate_verification_receipt,
)
from balanced_200m_trained_core import (
    CALIBRATION_BYTES,
    PLAN_PATH,
    ROOT,
    SEQUENCE_LENGTH,
    SOURCE_PATH,
    TRAINING_OUTPUT_PATH,
    calibration_arrays,
    calibration_nll_path,
    training_report_path,
)
from run_balanced_200m_training import _strict_nll
from scale_schedule_extrapolation_core import array_sha256

from jamoflow.hplt3 import hash_file
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_patching import hf_patch_lengths
from jamoflow.phase1 import patch_boundaries_from_lengths, stream_arrays
from jamoflow.phase2_patching import (
    causal_window_grid_trace,
    compact_whitespace_mask,
)

VERIFICATION_PATH = ROOT / "results/balanced-200m-trained-screen-v1/verification.json"
OUTPUT_PATH = ROOT / "results/balanced-200m-trained-screen-v1/quality-failure-analysis.json"


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _head_blob(relative: str) -> bytes:
    return subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _tracked_identity(path: Path) -> dict[str, str]:
    relative = path.relative_to(ROOT).as_posix()
    if _head_blob(relative) != path.read_bytes():
        raise ValueError(f"tracked artifact is not the exact HEAD blob: {relative}")
    commit = _git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise ValueError(f"tracked artifact has no publication commit: {relative}")
    return {"path": relative, "artifact_sha256": hash_file(path), "git_commit": commit}


def _never_published(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    if path.exists() or _git("log", "--all", "--format=%H", "--", relative):
        raise ValueError("balanced-200M failure analysis was already published")


def _count_hangul_syllables(row: np.ndarray) -> int:
    text = bytes(np.asarray(row, dtype=np.uint8)).decode("utf-8", errors="ignore")
    return sum("\uac00" <= character <= "\ud7a3" for character in text)


def _overlap_counts(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_boundaries = patch_boundaries_from_lengths(left)
    right_boundaries = patch_boundaries_from_lengths(right)
    return np.asarray(
        [
            len(set(map(int, a)).intersection(map(int, b)))
            for a, b in zip(left_boundaries, right_boundaries, strict=True)
        ],
        dtype=np.int64,
    )


def _patch_profile(
    boundary_rows: np.ndarray,
    whitespace_rows: np.ndarray,
    patch_count: int,
    c86_matrix: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if patch_count == 86:
        lengths = c86_matrix
        return (
            {
                "policy": "causal_codepoint_grid",
                "patch_count": 86,
                "patch_event_reduction_vs_c86": 0.0,
                "mean_patch_length": float(np.mean(lengths[:, 1:])),
                "median_patch_length": float(np.median(lengths[:, 1:])),
                "p95_patch_length": float(np.quantile(lengths[:, 1:], 0.95)),
                "maximum_patch_length": int(np.max(lengths[:, 1:])),
                "event_trigger_fraction": None,
                "deadline_trigger_fraction": None,
                "mean_absolute_target_displacement": None,
                "mean_boundary_overlap_with_c86": 85.0,
            },
            None,
        )
    rows: list[tuple[int, ...]] = []
    event_counts = np.zeros(len(boundary_rows), dtype=np.int64)
    displacements: list[int] = []
    deadline_total = 0
    event_total = 0
    for index, (boundary, whitespace) in enumerate(
        zip(boundary_rows, whitespace_rows, strict=True)
    ):
        trace = causal_window_grid_trace(boundary, whitespace, patch_count)
        rows.append(hf_patch_lengths(trace.boundaries, SEQUENCE_LENGTH))
        event_counts[index] = trace.trigger_kinds.count("event")
        event_total += trace.trigger_kinds.count("event")
        deadline_total += trace.trigger_kinds.count("deadline")
        displacements.extend(trace.target_displacements)
    lengths = np.ascontiguousarray(np.asarray(rows, dtype=np.uint16))
    overlap = _overlap_counts(lengths, c86_matrix)
    nonfinal = event_total + deadline_total
    return (
        {
            "policy": "causal_whitespace_grid",
            "patch_count": patch_count,
            "patch_event_reduction_vs_c86": 1.0 - patch_count / 86.0,
            "mean_patch_length": float(np.mean(lengths[:, 1:])),
            "median_patch_length": float(np.median(lengths[:, 1:])),
            "p95_patch_length": float(np.quantile(lengths[:, 1:], 0.95)),
            "maximum_patch_length": int(np.max(lengths[:, 1:])),
            "event_trigger_fraction": float(event_total / nonfinal),
            "deadline_trigger_fraction": float(deadline_total / nonfinal),
            "mean_absolute_target_displacement": float(
                np.mean(np.abs(np.asarray(displacements, dtype=np.int64)))
            ),
            "mean_boundary_overlap_with_c86": float(np.mean(overlap)),
        },
        event_counts if patch_count == 72 else None,
    )


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M failure analysis requires a clean worktree")
    _never_published(OUTPUT_PATH)
    base_commit = _git("rev-parse", "HEAD")
    plan = _read(PLAN_PATH)
    summary = _read(TRAINING_OUTPUT_PATH)
    verification = _read(VERIFICATION_PATH)
    validate_verification_receipt(verification)
    dependencies = {
        "plan": _tracked_identity(PLAN_PATH),
        "training_summary": _tracked_identity(TRAINING_OUTPUT_PATH),
        "verification": _tracked_identity(VERIFICATION_PATH),
    }
    if (
        verification["training_summary_sha256"] != summary["summary_sha256"]
        or verification["plan_sha256"] != plan["plan_sha256"]
    ):
        raise ValueError("balanced-200M failure analysis lineage differs")

    inputs, matrices = calibration_arrays()
    if (
        array_sha256(inputs) != plan["data"]["calibration_inputs_array_sha256"]
        or any(
            array_sha256(matrices[role])
            != plan["data"]["calibration_patch_matrix_sha256"][role]
            for role in ("c86", "w72")
        )
    ):
        raise ValueError("balanced-200M calibration reconstruction differs")
    c86_nll = _strict_nll(calibration_nll_path("c86"))
    w72_nll = _strict_nll(calibration_nll_path("w72"))
    effects = paired_bpb_effects(c86_nll, w72_nll)
    if not math.isclose(
        float(np.mean(effects)),
        float(summary["quality"]["w72_minus_c86_bpb"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("balanced-200M paired effect does not reconstruct")

    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    stream_inputs, boundary_rows = stream_arrays(
        stream.data, stream.codepoint_boundaries, SEQUENCE_LENGTH
    )
    whitespace_rows = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    if not np.array_equal(stream_inputs, inputs):
        raise ValueError("balanced-200M analysis stream differs")

    profiles: dict[str, Any] = {}
    w72_event_counts: np.ndarray | None = None
    for patch_count in CANDIDATE_PATCH_COUNTS:
        profile, event_counts = _patch_profile(
            boundary_rows, whitespace_rows, patch_count, matrices["c86"]
        )
        profiles[str(patch_count)] = profile
        if event_counts is not None:
            w72_event_counts = event_counts
    if w72_event_counts is None:
        raise AssertionError("W72 event counts were not generated")

    features = {
        "whitespace_event_count": whitespace_rows.sum(axis=1).astype(np.int64),
        "hangul_syllable_count": np.asarray(
            [_count_hangul_syllables(row) for row in inputs], dtype=np.int64
        ),
        "w72_maximum_patch_length": matrices["w72"][:, 1:].max(axis=1).astype(np.int64),
        "w72_whitespace_trigger_count": w72_event_counts,
        "w72_c86_boundary_overlap_count": _overlap_counts(
            matrices["w72"], matrices["c86"]
        ),
    }
    associations = {
        name: {
            "spearman_correlation_with_effect": spearman_correlation(values, effects),
            "equal_count_quintiles": equal_count_quintiles(values, effects),
        }
        for name, values in features.items()
    }

    c86_report = _read(training_report_path("c86"))
    w72_report = _read(training_report_path("w72"))
    observed_delta = float(summary["quality"]["w72_minus_c86_bpb"])
    payload = {
        "schema_version": 1,
        "kind": ANALYSIS_KIND,
        "protocol_id": PROTOCOL_ID,
        "analysis_base_git_commit": base_commit,
        "dependencies": dependencies,
        "paired_quality": {
            "examples": int(len(effects)),
            "mean_effect_bpb": float(np.mean(effects)),
            "median_effect_bpb": float(np.median(effects)),
            "standard_deviation_bpb": float(np.std(effects, ddof=1)),
            "p05_effect_bpb": float(np.quantile(effects, 0.05)),
            "p95_effect_bpb": float(np.quantile(effects, 0.95)),
            "positive_effect_rate": float(np.mean(effects > 0)),
            "quality_margin_bpb": float(summary["quality"]["maximum_allowed_delta_bpb"]),
            "quality_screen_pass": False,
            "contiguous_block_bootstrap": contiguous_block_bootstrap(effects),
            "positive_excess_concentration": positive_excess_concentration(effects),
        },
        "feature_associations": associations,
        "structural_profiles": profiles,
        "density_design_heuristic": {
            str(count): linear_density_heuristic(observed_delta, count)
            for count in CANDIDATE_PATCH_COUNTS
        },
        "resource_accounting": {
            "raw_bytes_per_parameter": float(
                c86_report["training"]["source_bytes"] / c86_report["parameter_count"]
            ),
            "c86_global_patch_tokens": int(c86_report["training"]["examples"] * 86),
            "w72_global_patch_tokens": int(w72_report["training"]["examples"] * 72),
            "global_patch_token_reduction": 1.0 - 72.0 / 86.0,
            "w72_training_throughput_increase": float(
                w72_report["training"]["source_bytes_per_second"]
                / c86_report["training"]["source_bytes_per_second"]
                - 1.0
            ),
            "w72_calibration_throughput_increase": float(
                w72_report["calibration_evaluation"]["bytes_per_second"]
                / c86_report["calibration_evaluation"]["bytes_per_second"]
                - 1.0
            ),
            "w72_training_elapsed_reduction": float(
                1.0
                - w72_report["training"]["elapsed_seconds"]
                / c86_report["training"]["elapsed_seconds"]
            ),
            "c86_maximum_driver_allocated_bytes": c86_report[
                "maximum_driver_allocated_bytes"
            ],
            "w72_maximum_driver_allocated_bytes": w72_report[
                "maximum_driver_allocated_bytes"
            ],
        },
        "interpretation": {
            "w72_quality_preserved": False,
            "actual_incremental_timing_authorized": False,
            "trained_model_scale_increase_supported": False,
            "random_weight_scaling_generalized_to_trained_quality": False,
            "undertraining_versus_patch_density_identified": False,
            "next_screen_candidate": "w80",
            "next_screen_requires_new_precommitted_plan": True,
        },
        "claim_boundary": {
            "post_outcome_exploratory_analysis": True,
            "per_sequence_patterns_unseen_before_protocol_commit": True,
            "historical_test_or_final_metric_used": False,
            "actual_latency_used": False,
            "one_seed": True,
            "sufficiently_trained_llm_claimed": False,
            "causal_explanation_claimed": False,
        },
    }
    output = {**payload, "analysis_sha256": canonical_sha256(payload)}
    if _git("rev-parse", "HEAD") != base_commit or _git("status", "--porcelain"):
        raise ValueError("repository changed during balanced-200M failure analysis")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_bytes(output))
        handle.flush()
        os.fsync(handle.fileno())
    print("balanced_200m_quality_failure_analysis=published")
    print(f"analysis_sha256={output['analysis_sha256']}")


if __name__ == "__main__":
    main()
