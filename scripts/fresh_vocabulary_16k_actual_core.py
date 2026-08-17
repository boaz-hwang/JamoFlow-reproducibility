"""Pure contracts and statistics for trained fresh-v2 16K actual inference."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from fresh_vocabulary_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_END_TO_END_POINT_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    REPETITIONS,
    TIMING_COMPONENTS,
    paired_prompt_bootstrap,
)

PROTOCOL_ID = "jamoflow-fresh-vocabulary-16k-actual-one-seed-v1"

ROLES = ("candidate_16k", "baseline_2k", "frontier_8k")
PRIMARY_PAIR = ("candidate_16k", "baseline_2k")
SECONDARY_PAIR = ("candidate_16k", "frontier_8k")
PAIR_ORDER = ("candidate_vs_2k", "candidate_vs_8k")
PAIR_ROLES = {
    "candidate_vs_2k": PRIMARY_PAIR,
    "candidate_vs_8k": SECONDARY_PAIR,
}

# Six permutations form a complete Williams-style order balance for three
# roles. The 640 measured cells use them cyclically; permutation counts differ
# by at most one and no result is used to choose an order.
ROLE_PERMUTATIONS = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
    (0, 2, 1),
    (1, 0, 2),
    (2, 1, 0),
)
SECONDARY_MINIMUM_POSITIVE_PROMPTS = MEASURED_CASES // 2 + 1


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
        raise ValueError("fresh-16k actual schedule coordinate differs")
    cell = (case_index * REPETITIONS + repetition_index) * len(MODES) + mode_index
    return ROLE_PERMUTATIONS[cell % len(ROLE_PERMUTATIONS)]


def _timing_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if (
        array.dtype != np.float64
        or array.shape != expected
        or not np.isfinite(array).all()
        or np.any(array <= 0)
    ):
        raise ValueError("fresh-16k actual timing array differs")
    return array


def _count_array(value: np.ndarray, *, minimum: int) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if array.dtype != np.int16 or array.shape != expected or np.any(array < minimum):
        raise ValueError("fresh-16k actual count array differs")
    return array


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
        "trace_contract_exact",
        "maximum_normalized_tolerance_ratio",
        "pass",
    }
    for role in ROLES:
        if set(correctness[role]) != set(MODES):
            return False
        for mode in MODES:
            row = correctness[role][mode]
            maximum = float(row.get("maximum_normalized_tolerance_ratio", math.nan))
            comparisons = int(row.get("comparisons", -1))
            if (
                set(row) != required
                or row.get("pass") is not True
                or row.get("trace_contract_exact") is not True
                or int(row.get("cases", -1)) != expected_cases
                or comparisons <= 0
                or int(row.get("argmax_comparisons", -1)) != comparisons
                or int(row.get("argmax_exact", -1)) != comparisons
                or not math.isfinite(maximum)
                or not 0.0 <= maximum <= 1.0
            ):
                return False
    return True


def _component_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    mode_index: int,
    left_index: int,
    right_index: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for component in TIMING_COMPONENTS:
        values = arrays[component][mode_index]
        left = np.median(values[..., left_index], axis=1)
        right = np.median(values[..., right_index], axis=1)
        output[component] = {
            "candidate_median_ms": float(np.median(left)),
            "reference_median_ms": float(np.median(right)),
            "median_reduction": 1.0 - float(np.median(left)) / float(np.median(right)),
        }
    return output


def _pair_mode_summary(
    arrays: Mapping[str, np.ndarray],
    tokens: np.ndarray,
    raw_bytes: np.ndarray,
    *,
    pair_name: str,
    mode_index: int,
    correctness_ok: bool,
) -> dict[str, Any]:
    left_role, right_role = PAIR_ROLES[pair_name]
    left_index = ROLES.index(left_role)
    right_index = ROLES.index(right_role)
    cells = np.median(arrays["end_to_end_ms"][mode_index], axis=1)
    left = cells[:, left_index]
    right = cells[:, right_index]
    point = 1.0 - float(np.median(left)) / float(np.median(right))
    lower, upper = paired_prompt_bootstrap(
        left,
        right,
        seed_offset=mode_index + len(MODES) * PAIR_ORDER.index(pair_name),
    )
    prompt_effects = 1.0 - left / right
    positive = int(np.count_nonzero(prompt_effects > 0.0))
    if pair_name == "candidate_vs_2k":
        passes = {
            "correctness": correctness_ok,
            "point_reduction": point >= MINIMUM_END_TO_END_POINT_REDUCTION,
            "bootstrap_lower_positive": lower > MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "prompt_direction": positive >= MINIMUM_POSITIVE_PROMPTS,
        }
    else:
        # This comparison is mandatory and result-blind, but does not replace
        # the fixed primary 16K-vs-2K expansion gate. It supports only an
        # incremental 8K->16K frontier statement.
        passes = {
            "correctness": correctness_ok,
            "point_positive": point > 0.0,
            "bootstrap_lower_positive": lower > 0.0,
            "prompt_majority": positive >= SECONDARY_MINIMUM_POSITIVE_PROMPTS,
        }
    return {
        "candidate_role": left_role,
        "reference_role": right_role,
        "end_to_end_reduction": point,
        "paired_prompt_bootstrap_95_interval": {
            "lower": lower,
            "upper": upper,
        },
        "positive_prompt_count": positive,
        "component_metrics": _component_metrics(
            arrays,
            mode_index=mode_index,
            left_index=left_index,
            right_index=right_index,
        ),
        "candidate_output_tokens_median": float(
            np.median(tokens[mode_index, ..., left_index])
        ),
        "reference_output_tokens_median": float(
            np.median(tokens[mode_index, ..., right_index])
        ),
        "candidate_output_bytes_median": float(
            np.median(raw_bytes[mode_index, ..., left_index])
        ),
        "reference_output_bytes_median": float(
            np.median(raw_bytes[mode_index, ..., right_index])
        ),
        "passes": passes,
        "overall_pass": bool(all(passes.values())),
    }


def summarize_actual_preflight(
    *,
    timing: Mapping[str, np.ndarray],
    output_token_count: np.ndarray,
    output_raw_byte_count: np.ndarray,
    correctness: Mapping[str, Mapping[str, Mapping[str, Any]]],
    maximum_output_bytes_by_role: Mapping[str, int],
) -> dict[str, Any]:
    if set(timing) != set(TIMING_COMPONENTS):
        raise ValueError("fresh-16k actual timing component set differs")
    arrays = {name: _timing_array(timing[name]) for name in TIMING_COMPONENTS}
    tokens = _count_array(output_token_count, minimum=1)
    raw_bytes = _count_array(output_raw_byte_count, minimum=CONTINUATION_BYTES)
    if set(maximum_output_bytes_by_role) != set(ROLES):
        raise ValueError("fresh-16k actual maximum output role set differs")
    for role_index, role in enumerate(ROLES):
        maximum = int(maximum_output_bytes_by_role[role])
        if maximum < CONTINUATION_BYTES or np.any(raw_bytes[..., role_index] > maximum):
            raise ValueError("fresh-16k actual output byte bound differs")
        controlled = MODES.index("controlled_replay")
        if np.any(raw_bytes[controlled, ..., role_index] != CONTINUATION_BYTES):
            raise ValueError("fresh-16k actual controlled byte count differs")

    correctness_ok = correctness_pass(
        correctness,
        expected_cases=MEASURED_CASES,
    )
    pairs: dict[str, Any] = {}
    for pair_name in PAIR_ORDER:
        pairs[pair_name] = {
            mode: _pair_mode_summary(
                arrays,
                tokens,
                raw_bytes,
                pair_name=pair_name,
                mode_index=mode_index,
                correctness_ok=correctness_ok,
            )
            for mode_index, mode in enumerate(MODES)
        }
        pairs[pair_name]["overall_pass"] = bool(
            all(pairs[pair_name][mode]["overall_pass"] for mode in MODES)
        )

    primary_pass = correctness_ok and pairs["candidate_vs_2k"]["overall_pass"]
    secondary_pass = correctness_ok and pairs["candidate_vs_8k"]["overall_pass"]
    return {
        "pairs": pairs,
        "correctness": {
            "independent_measured_case_checkpoint_replay": correctness,
            "expected_cases_per_role_mode": MEASURED_CASES,
            "overall_pass": correctness_ok,
        },
        "primary_gate": {
            "pair": "candidate_vs_2k",
            "minimum_end_to_end_point_reduction": MINIMUM_END_TO_END_POINT_REDUCTION,
            "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "requires_both_modes": True,
            "overall_pass": primary_pass,
        },
        "secondary_frontier_diagnostic": {
            "pair": "candidate_vs_8k",
            "requires_positive_point_and_interval_in_both_modes": True,
            "minimum_positive_prompts": SECONDARY_MINIMUM_POSITIVE_PROMPTS,
            "does_not_replace_or_relax_primary_gate": True,
            "overall_pass": secondary_pass,
        },
        "status": (
            "pass_16k_trained_actual_e2e_preflight"
            if primary_pass
            else "fail_16k_trained_actual_e2e_preflight"
        ),
        "multiseed_confirmation_authorized": primary_pass,
        "incremental_16k_frontier_supported": secondary_pass,
    }
