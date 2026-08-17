"""Pure contracts for the trained 16K perfect-draft target-block upper bound."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from fresh_vocabulary_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    REPETITIONS,
    TIMING_COMPONENTS,
    paired_prompt_bootstrap,
)

PROTOCOL_ID = "jamoflow-fresh-vocabulary-16k-target-block-upper-bound-v1"
ROLES = (
    "baseline_ar",
    "perfect_block_2",
    "perfect_block_4",
    "perfect_block_8",
)
BLOCK_SIZE_BY_ROLE = {
    "baseline_ar": 1,
    "perfect_block_2": 2,
    "perfect_block_4": 4,
    "perfect_block_8": 8,
}
PRIMARY_ROLE = "perfect_block_4"
PRIMARY_MINIMUM_END_TO_END_REDUCTION = 0.35

# An eight-row Williams-style cycle balances every role in every temporal
# position and reverses the directed carry-over order. 640 measured cells are
# exactly 80 complete cycles.
ROLE_ORDERS = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (3, 2, 1, 0),
    (0, 3, 2, 1),
    (1, 0, 3, 2),
    (2, 1, 0, 3),
)


def balanced_role_order(
    case_index: int,
    repetition_index: int,
    mode_index: int,
) -> tuple[int, ...]:
    if (
        not 0 <= case_index < MEASURED_CASES
        or not 0 <= repetition_index < REPETITIONS
        or not 0 <= mode_index < len(MODES)
    ):
        raise ValueError("16K target-block schedule coordinate differs")
    cell = (case_index * REPETITIONS + repetition_index) * len(MODES) + mode_index
    return ROLE_ORDERS[cell % len(ROLE_ORDERS)]


def correctness_pass(
    correctness: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_cases: int,
) -> bool:
    if set(correctness) != set(ROLES) or expected_cases <= 0:
        return False
    required = {
        "argmax_comparisons",
        "argmax_exact",
        "cases",
        "comparisons",
        "decode_calls",
        "maximum_normalized_tolerance_ratio",
        "pass",
        "trace_contract_exact",
    }
    for role in ROLES:
        if set(correctness[role]) != set(MODES):
            return False
        for mode in MODES:
            row = correctness[role][mode]
            comparisons = int(row.get("comparisons", -1))
            maximum = float(row.get("maximum_normalized_tolerance_ratio", math.nan))
            if (
                set(row) != required
                or row.get("pass") is not True
                or row.get("trace_contract_exact") is not True
                or int(row.get("cases", -1)) != expected_cases
                or comparisons <= 0
                or int(row.get("argmax_comparisons", -1)) != comparisons
                or int(row.get("argmax_exact", -1)) != comparisons
                or int(row.get("decode_calls", -1)) < 0
                or not math.isfinite(maximum)
                or not 0.0 <= maximum <= 1.0
            ):
                return False
    return True


def _timing_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if (
        array.dtype != np.float64
        or array.shape != expected
        or not np.isfinite(array).all()
        or np.any(array <= 0)
    ):
        raise ValueError("16K target-block timing array differs")
    return array


def _count_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if array.dtype != np.int16 or array.shape != expected or np.any(array <= 0):
        raise ValueError("16K target-block count array differs")
    return array


def summarize_upper_bound(
    *,
    timing: Mapping[str, np.ndarray],
    output_token_count: np.ndarray,
    output_raw_byte_count: np.ndarray,
    target_forward_calls: np.ndarray,
    correctness: Mapping[str, Mapping[str, Mapping[str, Any]]],
    maximum_output_bytes: int,
) -> dict[str, Any]:
    if set(timing) != set(TIMING_COMPONENTS):
        raise ValueError("16K target-block timing component set differs")
    arrays = {name: _timing_array(timing[name]) for name in TIMING_COMPONENTS}
    tokens = _count_array(output_token_count)
    raw_bytes = _count_array(output_raw_byte_count)
    calls = _count_array(target_forward_calls)
    if maximum_output_bytes < CONTINUATION_BYTES or np.any(
        raw_bytes > maximum_output_bytes
    ):
        raise ValueError("16K target-block output byte bound differs")
    baseline_index = ROLES.index("baseline_ar")
    for role_index, role in enumerate(ROLES):
        if not np.array_equal(tokens[..., role_index], tokens[..., baseline_index]):
            raise ValueError("16K target-block output token counts differ")
        if not np.array_equal(
            raw_bytes[..., role_index], raw_bytes[..., baseline_index]
        ):
            raise ValueError("16K target-block output byte counts differ")
        expected_calls = (
            tokens[..., role_index]
            if role == "baseline_ar"
            else 1
            + np.ceil((tokens[..., role_index] - 1) / BLOCK_SIZE_BY_ROLE[role]).astype(
                np.int16
            )
        )
        if not np.array_equal(calls[..., role_index], expected_calls):
            raise ValueError("16K target-block target-call accounting differs")

    correctness_ok = correctness_pass(correctness, expected_cases=MEASURED_CASES)
    comparisons: dict[str, Any] = {}
    for role in ROLES[1:]:
        role_index = ROLES.index(role)
        comparisons[role] = {}
        for mode_index, mode in enumerate(MODES):
            candidate = np.median(
                arrays["end_to_end_ms"][mode_index, ..., role_index], axis=1
            )
            baseline = np.median(
                arrays["end_to_end_ms"][mode_index, ..., baseline_index], axis=1
            )
            point = 1.0 - float(np.median(candidate)) / float(np.median(baseline))
            lower, upper = paired_prompt_bootstrap(
                candidate,
                baseline,
                seed_offset=100 * role_index + mode_index,
            )
            positive = int(np.count_nonzero(candidate < baseline))
            components: dict[str, Any] = {}
            for component in TIMING_COMPONENTS:
                candidate_component = np.median(
                    arrays[component][mode_index, ..., role_index], axis=1
                )
                baseline_component = np.median(
                    arrays[component][mode_index, ..., baseline_index], axis=1
                )
                components[component] = {
                    "candidate_median_ms": float(np.median(candidate_component)),
                    "baseline_median_ms": float(np.median(baseline_component)),
                    "median_reduction": 1.0
                    - float(np.median(candidate_component))
                    / float(np.median(baseline_component)),
                }
            comparisons[role][mode] = {
                "end_to_end_reduction": point,
                "paired_prompt_bootstrap_95_interval": {
                    "lower": lower,
                    "upper": upper,
                },
                "positive_prompt_count": positive,
                "component_metrics": components,
                "target_forward_calls_median": float(
                    np.median(calls[mode_index, ..., role_index])
                ),
                "baseline_forward_calls_median": float(
                    np.median(calls[mode_index, ..., baseline_index])
                ),
            }

    primary_passes: dict[str, Any] = {}
    for mode in MODES:
        row = comparisons[PRIMARY_ROLE][mode]
        primary_passes[mode] = {
            "correctness": correctness_ok,
            "point_reduction": row["end_to_end_reduction"]
            >= PRIMARY_MINIMUM_END_TO_END_REDUCTION,
            "bootstrap_lower_positive": row["paired_prompt_bootstrap_95_interval"][
                "lower"
            ]
            > MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "prompt_direction": row["positive_prompt_count"]
            >= MINIMUM_POSITIVE_PROMPTS,
        }
        primary_passes[mode]["overall_pass"] = bool(all(primary_passes[mode].values()))
    overall = correctness_ok and all(
        primary_passes[mode]["overall_pass"] for mode in MODES
    )
    return {
        "comparisons": comparisons,
        "correctness": {
            "independent_measured_case_checkpoint_replay": correctness,
            "overall_pass": correctness_ok,
        },
        "primary_gate": {
            "role": PRIMARY_ROLE,
            "minimum_end_to_end_reduction": PRIMARY_MINIMUM_END_TO_END_REDUCTION,
            "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "requires_both_modes": True,
            "by_mode": primary_passes,
            "overall_pass": overall,
        },
        "status": (
            "pass_16k_target_block_upper_bound"
            if overall
            else "fail_16k_target_block_upper_bound"
        ),
        "learned_draft_prototype_authorized": overall,
        "actual_speculative_efficiency_claimed": False,
    }
