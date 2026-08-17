"""Pure gates for the one-seed trained static-geometry screen."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "jamoflow-static-geometry-one-seed-v1"
SEED = 1729
SEQUENCE_LENGTH = 512
TARGETS_PER_SEQUENCE = SEQUENCE_LENGTH - 1
PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
PROMPT_COUNT = 64
REPETITIONS = 5
WARMUP_PROMPTS = 8
MODES = ("controlled_replay", "free_running_utf8_greedy")
ROLES = ("candidate", "baseline")
QUALITY_MARGIN_BPB = 0.010
QUALITY_BOOTSTRAP_REPETITIONS = 10_000
QUALITY_BOOTSTRAP_SEED = 20_261_001
TIMING_BOOTSTRAP_REPETITIONS = 10_000
TIMING_BOOTSTRAP_SEED = 20_261_002
MINIMUM_TIMING_POINT_REDUCTION = 0.15
MINIMUM_TIMING_LOWER_BOUND = 0.10
MINIMUM_POSITIVE_PROMPTS = 48


def _float32_vector(value: np.ndarray, *, count: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.shape != (count,)
        or not np.isfinite(array).all()
        or np.any(array < 0)
    ):
        raise ValueError("quality loss vector differs")
    return np.ascontiguousarray(array)


def one_seed_document_bootstrap(
    differences_nats: np.ndarray,
    document_indices: np.ndarray,
    *,
    repetitions: int = QUALITY_BOOTSTRAP_REPETITIONS,
    seed: int = QUALITY_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample whole calibration documents for a one-model-seed screen."""

    values = np.asarray(differences_nats, dtype=np.float64)
    indices = np.asarray(document_indices)
    if (
        values.ndim != 1
        or indices.shape != values.shape
        or not np.isfinite(values).all()
        or not np.issubdtype(indices.dtype, np.integer)
        or repetitions <= 0
    ):
        raise ValueError("document bootstrap inputs differ")
    eligible = indices >= 0
    documents = np.unique(indices[eligible])
    if len(documents) < 2:
        raise ValueError("document bootstrap needs two eligible documents")
    dense = np.searchsorted(documents, indices[eligible])
    counts = np.bincount(dense, minlength=len(documents)).astype(np.int64)
    sums = np.bincount(
        dense,
        weights=values[eligible],
        minlength=len(documents),
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    scale = TARGETS_PER_SEQUENCE * math.log(2.0)
    chunk = 256
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        sampled = rng.integers(
            0,
            len(documents),
            size=(size, len(documents)),
        )
        estimates[start : start + size] = (
            sums[sampled].sum(axis=1)
            / (counts[sampled].sum(axis=1) * scale)
        )
    return estimates


def summarize_one_seed_quality(
    *,
    candidate_losses_nats: np.ndarray,
    baseline_losses_nats: np.ndarray,
    document_indices: np.ndarray,
    document_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _float32_vector(
        candidate_losses_nats,
        count=len(document_indices),
    )
    baseline = _float32_vector(
        baseline_losses_nats,
        count=len(document_indices),
    )
    if (
        document_metadata.get("sequence_count") != len(document_indices)
        or document_metadata.get("sequence_length") != SEQUENCE_LENGTH
        or document_metadata.get("eligible_sequence_fraction_pass") is not True
    ):
        raise ValueError("quality document map differs")
    scale = len(candidate) * TARGETS_PER_SEQUENCE * math.log(2.0)
    candidate_bpb = float(candidate.astype(np.float64).sum()) / scale
    baseline_bpb = float(baseline.astype(np.float64).sum()) / scale
    difference = candidate.astype(np.float64) - baseline.astype(np.float64)
    point = float(difference.sum()) / scale
    estimates = one_seed_document_bootstrap(difference, document_indices)
    lower, median, upper = np.quantile(estimates, [0.05, 0.5, 0.95])
    passes = {
        "mean_difference": point <= QUALITY_MARGIN_BPB,
        "one_sided_document_upper": float(upper) <= QUALITY_MARGIN_BPB,
        "document_coverage": True,
    }
    return {
        "difference_direction": "candidate_minus_baseline; lower favors candidate",
        "candidate_bpb": candidate_bpb,
        "baseline_bpb": baseline_bpb,
        "mean_difference_bpb": point,
        "noninferiority_margin_bpb": QUALITY_MARGIN_BPB,
        "document_bootstrap": {
            "repetitions": QUALITY_BOOTSTRAP_REPETITIONS,
            "seed": QUALITY_BOOTSTRAP_SEED,
            "resampling_unit": "whole calibration document; target-byte weighted",
            "central_90_interval": {
                "lower": float(lower),
                "upper": float(upper),
            },
            "median_bpb": float(median),
            "one_sided_95_upper_bpb": float(upper),
        },
        "document_map": dict(document_metadata),
        "passes": passes,
        "overall_pass": bool(all(passes.values())),
        "status": (
            "pass_one_seed_quality_screen"
            if all(passes.values())
            else "fail_one_seed_quality_screen"
        ),
    }


def _timing_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), PROMPT_COUNT, REPETITIONS, len(ROLES))
    if (
        array.dtype != np.float64
        or array.shape != expected
        or not np.isfinite(array).all()
        or np.any(array <= 0)
    ):
        raise ValueError("one-seed timing array differs")
    return array


def _prompt_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(TIMING_BOOTSTRAP_SEED)
    estimates = np.empty(TIMING_BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(TIMING_BOOTSTRAP_REPETITIONS):
        rows = rng.integers(0, PROMPT_COUNT, size=PROMPT_COUNT)
        estimates[repetition] = 1.0 - (
            float(np.median(candidate[rows]))
            / float(np.median(baseline[rows]))
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_one_seed_timing(
    *,
    end_to_end_ms: np.ndarray,
    ttft_ms: np.ndarray,
    decode_ms: np.ndarray,
    emitted_output_bytes: np.ndarray,
    correctness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    end_to_end = _timing_array(end_to_end_ms)
    ttft = _timing_array(ttft_ms)
    decode = _timing_array(decode_ms)
    emitted = np.asarray(emitted_output_bytes)
    expected_emitted = (len(MODES), PROMPT_COUNT, len(ROLES))
    if (
        emitted.dtype != np.int16
        or emitted.shape != expected_emitted
        or np.any(emitted < CONTINUATION_BYTES)
        or np.any(emitted > CONTINUATION_BYTES + 3)
        or np.any(emitted[0] != CONTINUATION_BYTES)
    ):
        raise ValueError("one-seed emitted-byte evidence differs")
    if set(correctness) != set(ROLES):
        raise ValueError("one-seed correctness roles differ")
    expected_correctness_keys = {
        "argmax_comparisons",
        "argmax_exact",
        "boundary_trace_exact",
        "cache_diagnostics_exact",
        "maximum_normalized_logit_error",
        "strict_free_outputs",
    }
    correctness_pass = True
    for role in ROLES:
        row = correctness[role]
        maximum = float(row.get("maximum_normalized_logit_error", math.nan))
        comparisons = int(row.get("argmax_comparisons", -1))
        if (
            set(row) != expected_correctness_keys
            or comparisons != WARMUP_PROMPTS * CONTINUATION_BYTES
            or int(row["argmax_exact"]) != comparisons
            or row["boundary_trace_exact"] is not True
            or row["cache_diagnostics_exact"] is not True
            or int(row["strict_free_outputs"]) != PROMPT_COUNT
            or not np.isfinite(maximum)
            or not 0 <= maximum <= 1
        ):
            correctness_pass = False

    medians = np.median(end_to_end, axis=2)
    rows: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        candidate = medians[mode_index, :, ROLES.index("candidate")]
        baseline = medians[mode_index, :, ROLES.index("baseline")]
        reduction = 1.0 - float(np.median(candidate)) / float(np.median(baseline))
        lower, upper = _prompt_bootstrap(candidate, baseline)
        prompt_effects = 1.0 - candidate / baseline
        positive = int(np.sum(prompt_effects > 0))
        passes = {
            "correctness": correctness_pass,
            "point_reduction": reduction >= MINIMUM_TIMING_POINT_REDUCTION,
            "bootstrap_lower_bound": lower >= MINIMUM_TIMING_LOWER_BOUND,
            "prompt_direction": positive >= MINIMUM_POSITIVE_PROMPTS,
        }
        rows[mode] = {
            "candidate_median_end_to_end_ms": float(np.median(candidate)),
            "baseline_median_end_to_end_ms": float(np.median(baseline)),
            "end_to_end_reduction": reduction,
            "prompt_bootstrap_95_interval": {"lower": lower, "upper": upper},
            "positive_prompt_count": positive,
            "candidate_median_ttft_ms": float(
                np.median(ttft[mode_index, :, :, ROLES.index("candidate")])
            ),
            "baseline_median_ttft_ms": float(
                np.median(ttft[mode_index, :, :, ROLES.index("baseline")])
            ),
            "candidate_median_decode_ms": float(
                np.median(decode[mode_index, :, :, ROLES.index("candidate")])
            ),
            "baseline_median_decode_ms": float(
                np.median(decode[mode_index, :, :, ROLES.index("baseline")])
            ),
            "candidate_mean_emitted_output_bytes": float(
                emitted[mode_index, :, ROLES.index("candidate")].mean()
            ),
            "baseline_mean_emitted_output_bytes": float(
                emitted[mode_index, :, ROLES.index("baseline")].mean()
            ),
            "passes": passes,
            "overall_pass": bool(all(passes.values())),
        }
    overall = correctness_pass and all(row["overall_pass"] for row in rows.values())
    return {
        "thresholds": {
            "minimum_point_reduction": MINIMUM_TIMING_POINT_REDUCTION,
            "minimum_bootstrap_lower_bound": MINIMUM_TIMING_LOWER_BOUND,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        },
        "bootstrap": {
            "repetitions": TIMING_BOOTSTRAP_REPETITIONS,
            "seed": TIMING_BOOTSTRAP_SEED,
            "unit": "calibration prompt after within-prompt repetition median",
        },
        "correctness": {role: dict(correctness[role]) for role in ROLES},
        "modes": rows,
        "overall_pass": overall,
        "status": (
            "pass_one_seed_actual_latency_screen"
            if overall
            else "fail_one_seed_actual_latency_screen"
        ),
    }


def one_seed_decision(
    quality: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    passed = quality.get("overall_pass") is True and timing.get("overall_pass") is True
    return {
        "quality_pass": quality.get("overall_pass") is True,
        "actual_latency_pass": timing.get("overall_pass") is True,
        "multi_seed_static_control_authorized": passed,
        "conditional_local_compute_research_authorized": passed,
        "status": (
            "one_seed_static_control_pass"
            if passed
            else "one_seed_static_control_stopped"
        ),
    }
