"""Pure contracts and aggregation for the perfect-draft block-kernel preflight."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "jamoflow-target-block-kernel-v2"
MICRO_STRATA = ("no_new_patch", "one_new_patch")
MODES = ("sequential_three", "block_three")
MICRO_CASES = 32
MICRO_REPETITIONS = 5
WHOLE_CASES = 16
WHOLE_REPETITIONS = 3
PROMPT_BYTES = 128
MINIMUM_CONTINUATION_BYTES = 255
MAXIMUM_CONTINUATION_BYTES = 258
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260829


def perfect_hangul_groups(data: bytes) -> tuple[bytes, ...]:
    """Group only complete precomposed Hangul scalars; keep all else bytewise."""

    groups: list[bytes] = []
    index = 0
    while index < len(data):
        if 0xEA <= data[index] <= 0xED and index + 3 <= len(data):
            candidate = data[index : index + 3]
            try:
                text = candidate.decode("utf-8")
            except UnicodeDecodeError:
                text = ""
            if len(text) == 1 and 0xAC00 <= ord(text) <= 0xD7A3:
                groups.append(candidate)
                index += 3
                continue
        groups.append(data[index : index + 1])
        index += 1
    if b"".join(groups) != data or any(len(group) not in (1, 3) for group in groups):
        raise AssertionError("perfect Hangul grouping does not reconstruct the input")
    return tuple(groups)


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float64
        or array.shape != shape
        or not np.all(np.isfinite(array))
        or np.any(array <= 0)
    ):
        raise ValueError(f"{name} shape/dtype/value differs")
    return array


def summarize_block_kernel(
    *,
    micro_ms: np.ndarray,
    whole_ms: np.ndarray,
    whole_hangul_blocks: np.ndarray,
    whole_boundary_blocks: np.ndarray,
    correctness: Mapping[str, Any],
    independent_first_acceptance: float,
    independent_pair_acceptance: float,
    independent_head_latency_ms: float,
    minimum_micro_reduction: float,
    minimum_micro_lower_bound: float,
    minimum_perfect_whole_reduction: float,
    minimum_perfect_whole_lower_bound: float,
    minimum_projected_reduction: float,
    minimum_projected_lower_bound: float,
) -> dict[str, Any]:
    micro = _finite(
        "micro_ms",
        micro_ms,
        (len(MICRO_STRATA), MICRO_CASES, MICRO_REPETITIONS, len(MODES)),
    )
    whole = _finite(
        "whole_ms",
        whole_ms,
        (WHOLE_CASES, WHOLE_REPETITIONS, len(MODES)),
    )
    blocks = np.asarray(whole_hangul_blocks)
    boundary = np.asarray(whole_boundary_blocks)
    if (
        blocks.dtype != np.int64
        or boundary.dtype != np.int64
        or blocks.shape != (WHOLE_CASES,)
        or boundary.shape != blocks.shape
        or np.any(blocks <= 0)
        or np.any(boundary < 0)
        or np.any(boundary > blocks)
    ):
        raise ValueError("whole block count arrays differ")
    if (
        set(correctness)
        != {
            "argmax_comparisons",
            "cache_comparisons",
            "maximum_absolute_logit_error",
            "maximum_normalized_tolerance_ratio",
            "pass",
        }
        or correctness["pass"] is not True
        or int(correctness["argmax_comparisons"]) <= 0
        or int(correctness["cache_comparisons"]) <= 0
        or not np.isfinite(float(correctness["maximum_absolute_logit_error"]))
        or not np.isfinite(float(correctness["maximum_normalized_tolerance_ratio"]))
        or float(correctness["maximum_absolute_logit_error"]) < 0
        or float(correctness["maximum_normalized_tolerance_ratio"]) < 0
        or float(correctness["maximum_normalized_tolerance_ratio"]) > 1.0
    ):
        raise ValueError("block correctness evidence differs")
    for value in (
        independent_first_acceptance,
        independent_pair_acceptance,
        independent_head_latency_ms,
        minimum_micro_reduction,
        minimum_micro_lower_bound,
        minimum_perfect_whole_reduction,
        minimum_perfect_whole_lower_bound,
        minimum_projected_reduction,
        minimum_projected_lower_bound,
    ):
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError("block projection input differs")
    if (
        independent_first_acceptance > 1
        or independent_pair_acceptance > independent_first_acceptance
        or independent_head_latency_ms <= 0
        or any(
            value > 1
            for value in (
                minimum_micro_reduction,
                minimum_micro_lower_bound,
                minimum_perfect_whole_reduction,
                minimum_perfect_whole_lower_bound,
                minimum_projected_reduction,
                minimum_projected_lower_bound,
            )
        )
    ):
        raise ValueError("block probability, latency, or threshold differs")
    expected_committed = (
        2.0 + independent_first_acceptance + independent_pair_acceptance
    )
    if not 2.0 <= expected_committed <= 4.0:
        raise ValueError("expected committed-byte opportunity differs")

    stratum_summary: dict[str, Any] = {}
    for stratum_index, stratum in enumerate(MICRO_STRATA):
        sequential = float(np.median(micro[stratum_index, :, :, 0]))
        block = float(np.median(micro[stratum_index, :, :, 1]))
        stratum_summary[stratum] = {
            "sequential_three_median_ms": sequential,
            "block_three_median_ms": block,
            "target_call_reduction": 1.0 - block / sequential,
        }

    boundary_rate = float(boundary.sum() / blocks.sum())
    weighted_sequential = (
        (1.0 - boundary_rate)
        * stratum_summary["no_new_patch"]["sequential_three_median_ms"]
        + boundary_rate
        * stratum_summary["one_new_patch"]["sequential_three_median_ms"]
    )
    weighted_block = (
        (1.0 - boundary_rate)
        * stratum_summary["no_new_patch"]["block_three_median_ms"]
        + boundary_rate
        * stratum_summary["one_new_patch"]["block_three_median_ms"]
    )
    micro_reduction = 1.0 - weighted_block / weighted_sequential
    sequential_per_byte = weighted_sequential / 3.0
    projected_per_byte = (
        weighted_block + independent_head_latency_ms
    ) / expected_committed
    projected_reduction = 1.0 - projected_per_byte / sequential_per_byte

    sequential_whole = float(np.median(whole[:, :, 0]))
    block_whole = float(np.median(whole[:, :, 1]))
    perfect_reduction = 1.0 - block_whole / sequential_whole

    # Repetitions are timing repeats, not independent samples. Collapse them
    # within case first, then bootstrap calibration cases. Micro strata are
    # resampled independently and the whole-path rows provide the empirical
    # patch-boundary mixture for every draw.
    micro_cells = np.median(micro, axis=2)
    whole_cells = np.median(whole, axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    micro_draws = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    whole_draws = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    projected_draws = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for draw in range(BOOTSTRAP_REPETITIONS):
        sampled_micro = [
            rng.integers(0, MICRO_CASES, size=MICRO_CASES)
            for _ in MICRO_STRATA
        ]
        sampled_whole = rng.integers(0, WHOLE_CASES, size=WHOLE_CASES)
        sampled_rates = [
            (
                float(np.median(micro_cells[index, rows, 0])),
                float(np.median(micro_cells[index, rows, 1])),
            )
            for index, rows in enumerate(sampled_micro)
        ]
        sampled_boundary_rate = float(
            boundary[sampled_whole].sum() / blocks[sampled_whole].sum()
        )
        sampled_sequential = (
            (1.0 - sampled_boundary_rate) * sampled_rates[0][0]
            + sampled_boundary_rate * sampled_rates[1][0]
        )
        sampled_block = (
            (1.0 - sampled_boundary_rate) * sampled_rates[0][1]
            + sampled_boundary_rate * sampled_rates[1][1]
        )
        micro_draws[draw] = 1.0 - sampled_block / sampled_sequential
        projected_draws[draw] = 1.0 - (
            (sampled_block + independent_head_latency_ms) / expected_committed
        ) / (sampled_sequential / 3.0)
        whole_draws[draw] = 1.0 - float(
            np.median(whole_cells[sampled_whole, 1])
        ) / float(np.median(whole_cells[sampled_whole, 0]))

    def interval(values: np.ndarray) -> dict[str, float]:
        lower, upper = np.quantile(values, (0.025, 0.975))
        return {"lower": float(lower), "upper": float(upper)}

    micro_interval = interval(micro_draws)
    whole_interval = interval(whole_draws)
    projected_interval = interval(projected_draws)
    passes = {
        "correctness": True,
        "micro_target_block": (
            micro_reduction >= minimum_micro_reduction
            and micro_interval["lower"] >= minimum_micro_lower_bound
        ),
        "perfect_hangul_whole": (
            perfect_reduction >= minimum_perfect_whole_reduction
            and whole_interval["lower"] >= minimum_perfect_whole_lower_bound
        ),
        "independent_projection": (
            projected_reduction >= minimum_projected_reduction
            and projected_interval["lower"] >= minimum_projected_lower_bound
        ),
    }
    return {
        "micro_by_patch_stratum": stratum_summary,
        "whole_empirical_boundary_block_rate": boundary_rate,
        "weighted_micro": {
            "sequential_three_ms": weighted_sequential,
            "block_three_ms": weighted_block,
            "target_block_reduction": micro_reduction,
            "case_bootstrap_95_interval": micro_interval,
        },
        "perfect_hangul_whole_path": {
            "sequential_median_ms": sequential_whole,
            "block_median_ms": block_whole,
            "reduction": perfect_reduction,
            "case_bootstrap_95_interval": whole_interval,
            "total_hangul_blocks": int(blocks.sum()),
            "total_boundary_blocks": int(boundary.sum()),
        },
        "fixed_independent_projection": {
            "first_acceptance": independent_first_acceptance,
            "pair_acceptance": independent_pair_acceptance,
            "expected_committed_bytes_per_verification": expected_committed,
            "head_latency_ms": independent_head_latency_ms,
            "sequential_target_ms_per_byte": sequential_per_byte,
            "projected_target_plus_head_ms_per_committed_byte": projected_per_byte,
            "projected_reduction": projected_reduction,
            "case_bootstrap_95_interval": projected_interval,
            "excludes": [
                "rollback_cache_crop",
                "mismatch_control_flow",
                "output_mask_and_stop_logic",
            ],
        },
        "correctness": dict(correctness),
        "gates": {
            "thresholds": {
                "minimum_micro_target_block_reduction": minimum_micro_reduction,
                "minimum_micro_target_block_lower_bound": minimum_micro_lower_bound,
                "minimum_perfect_hangul_whole_reduction": minimum_perfect_whole_reduction,
                "minimum_perfect_hangul_whole_lower_bound": minimum_perfect_whole_lower_bound,
                "minimum_independent_projected_reduction": minimum_projected_reduction,
                "minimum_independent_projected_lower_bound": minimum_projected_lower_bound,
            },
            "bootstrap": {
                "unit": "calibration case after within-case repetition median",
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": BOOTSTRAP_SEED,
            },
            "passes": passes,
            "full_speculative_runtime_authorized": all(passes.values()),
        },
    }
