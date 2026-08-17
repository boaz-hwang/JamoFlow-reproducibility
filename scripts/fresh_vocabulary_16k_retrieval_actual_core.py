"""Statistical and accounting contract for actual 16K retrieval drafting."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from fresh_vocabulary_actual_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_END_TO_END_POINT_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    REPETITIONS,
    paired_prompt_bootstrap,
)

PROTOCOL_ID = "jamoflow-fresh-vocabulary-16k-retrieval-actual-v1"
ROLES = (
    "baseline_ar",
    "prompt_lookup_block_4",
    "corpus_ngram_block_4",
    "hybrid_retrieval_block_4",
)
PRIMARY_ROLE = "hybrid_retrieval_block_4"
TIMING_COMPONENTS = (
    "tokenizer_ms",
    "ttft_ms",
    "decode_ms",
    "model_loop_ms",
    "draft_lookup_ms",
    "end_to_end_ms",
)
COUNTER_NAMES = (
    "target_forward_calls",
    "proposal_attempts",
    "proposal_tokens",
    "accepted_draft_tokens",
    "full_accept_cycles",
    "rejection_cycles",
    "no_draft_steps",
    "correction_tokens",
    "bonus_tokens",
    "cropped_input_tokens",
    "corpus_ngram_proposals",
    "prompt_lookup_proposals",
)

# Four-role Williams-style sequence. Every role appears twice in every position
# across an eight-row cycle, with reversed directed carry-over pairs.
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
        raise ValueError("16K retrieval schedule coordinate differs")
    cell = (case_index * REPETITIONS + repetition_index) * len(MODES) + mode_index
    return ROLE_ORDERS[cell % len(ROLE_ORDERS)]


def _timing_array(value: np.ndarray, *, allow_zero: bool = False) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    invalid_sign = np.any(array < 0) if allow_zero else np.any(array <= 0)
    if (
        array.dtype != np.float64
        or array.shape != expected
        or not np.isfinite(array).all()
        or invalid_sign
    ):
        raise ValueError("16K retrieval timing array differs")
    return array


def _positive_count_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if array.dtype != np.int16 or array.shape != expected or np.any(array <= 0):
        raise ValueError("16K retrieval positive count array differs")
    return array


def _nonnegative_count_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if array.dtype != np.int16 or array.shape != expected or np.any(array < 0):
        raise ValueError("16K retrieval counter array differs")
    return array


def _validate_counter_identities(
    counters: Mapping[str, np.ndarray],
    output_tokens: np.ndarray,
) -> None:
    baseline = ROLES.index("baseline_ar")
    if not np.array_equal(counters["target_forward_calls"][..., baseline], output_tokens[..., baseline]):
        raise ValueError("16K retrieval baseline target-call count differs")
    for role_index, role in enumerate(ROLES):
        attempts = counters["proposal_attempts"][..., role_index]
        proposal_tokens = counters["proposal_tokens"][..., role_index]
        accepted = counters["accepted_draft_tokens"][..., role_index]
        corpus = counters["corpus_ngram_proposals"][..., role_index]
        prompt = counters["prompt_lookup_proposals"][..., role_index]
        rejection = counters["rejection_cycles"][..., role_index]
        correction = counters["correction_tokens"][..., role_index]
        if (
            np.any(accepted > proposal_tokens)
            or not np.array_equal(corpus + prompt, attempts)
            or not np.array_equal(rejection, correction)
            or np.any(counters["full_accept_cycles"][..., role_index] + rejection > attempts)
        ):
            raise ValueError(f"16K retrieval counter identity differs: {role}")
        if role == "baseline_ar" and any(
            np.any(counters[name][..., role_index] != 0)
            for name in COUNTER_NAMES
            if name not in {"target_forward_calls", "no_draft_steps"}
        ):
            raise ValueError("16K retrieval baseline contains draft work")
        if role == "prompt_lookup_block_4" and np.any(corpus != 0):
            raise ValueError("16K retrieval prompt-only role used corpus n-grams")
        if role == "corpus_ngram_block_4" and np.any(prompt != 0):
            raise ValueError("16K retrieval corpus-only role used prompt lookup")


def summarize_retrieval_actual(
    *,
    timing: Mapping[str, np.ndarray],
    counters: Mapping[str, np.ndarray],
    output_token_count: np.ndarray,
    output_raw_byte_count: np.ndarray,
    correctness_pass: bool,
    maximum_output_bytes: int,
) -> dict[str, Any]:
    if set(timing) != set(TIMING_COMPONENTS) or set(counters) != set(COUNTER_NAMES):
        raise ValueError("16K retrieval evidence key set differs")
    timing_arrays = {
        name: _timing_array(values, allow_zero=name == "draft_lookup_ms")
        for name, values in timing.items()
    }
    counter_arrays = {
        name: _positive_count_array(values)
        if name == "target_forward_calls"
        else _nonnegative_count_array(values)
        for name, values in counters.items()
    }
    tokens = _positive_count_array(output_token_count)
    raw_bytes = _positive_count_array(output_raw_byte_count)
    if maximum_output_bytes < CONTINUATION_BYTES or np.any(raw_bytes > maximum_output_bytes):
        raise ValueError("16K retrieval output byte bound differs")
    baseline_index = ROLES.index("baseline_ar")
    for role_index in range(len(ROLES)):
        if not np.array_equal(tokens[..., role_index], tokens[..., baseline_index]) or not np.array_equal(
            raw_bytes[..., role_index], raw_bytes[..., baseline_index]
        ):
            raise ValueError("16K retrieval output counts differ across roles")
    _validate_counter_identities(counter_arrays, tokens)

    comparisons: dict[str, Any] = {}
    for role_index, role in enumerate(ROLES[1:], start=1):
        comparisons[role] = {}
        for mode_index, mode in enumerate(MODES):
            candidate = np.median(
                timing_arrays["end_to_end_ms"][mode_index, ..., role_index], axis=1
            )
            baseline = np.median(
                timing_arrays["end_to_end_ms"][mode_index, ..., baseline_index], axis=1
            )
            reduction = 1.0 - float(np.median(candidate)) / float(np.median(baseline))
            lower, upper = paired_prompt_bootstrap(
                candidate,
                baseline,
                seed_offset=100 * role_index + mode_index,
            )
            components: dict[str, Any] = {}
            for component in TIMING_COMPONENTS:
                candidate_component = np.median(
                    timing_arrays[component][mode_index, ..., role_index], axis=1
                )
                baseline_component = np.median(
                    timing_arrays[component][mode_index, ..., baseline_index], axis=1
                )
                baseline_median = float(np.median(baseline_component))
                candidate_median = float(np.median(candidate_component))
                components[component] = {
                    "candidate_median_ms": candidate_median,
                    "baseline_median_ms": baseline_median,
                    "median_reduction": (
                        1.0 - candidate_median / baseline_median
                        if baseline_median > 0
                        else math.nan
                    ),
                }
            proposed = int(np.sum(counter_arrays["proposal_tokens"][mode_index, ..., role_index]))
            accepted = int(np.sum(counter_arrays["accepted_draft_tokens"][mode_index, ..., role_index]))
            attempts = int(np.sum(counter_arrays["proposal_attempts"][mode_index, ..., role_index]))
            comparisons[role][mode] = {
                "end_to_end_reduction": reduction,
                "paired_prompt_bootstrap_95_interval": {"lower": lower, "upper": upper},
                "positive_prompt_count": int(np.count_nonzero(candidate < baseline)),
                "component_metrics": components,
                "target_forward_calls_median": float(
                    np.median(counter_arrays["target_forward_calls"][mode_index, ..., role_index])
                ),
                "baseline_forward_calls_median": float(
                    np.median(counter_arrays["target_forward_calls"][mode_index, ..., baseline_index])
                ),
                "proposal_attempts": attempts,
                "proposal_tokens": proposed,
                "accepted_draft_tokens": accepted,
                "draft_token_acceptance_rate": accepted / proposed if proposed else 0.0,
                "accepted_tokens_per_proposal_cycle": accepted / attempts if attempts else 0.0,
                "corpus_ngram_proposals": int(
                    np.sum(counter_arrays["corpus_ngram_proposals"][mode_index, ..., role_index])
                ),
                "prompt_lookup_proposals": int(
                    np.sum(counter_arrays["prompt_lookup_proposals"][mode_index, ..., role_index])
                ),
            }

    primary_modes: dict[str, Any] = {}
    for mode in MODES:
        row = comparisons[PRIMARY_ROLE][mode]
        primary_modes[mode] = {
            "correctness": bool(correctness_pass),
            "point_reduction": row["end_to_end_reduction"]
            >= MINIMUM_END_TO_END_POINT_REDUCTION,
            "bootstrap_lower_positive": row["paired_prompt_bootstrap_95_interval"]["lower"]
            > MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "prompt_direction": row["positive_prompt_count"] >= MINIMUM_POSITIVE_PROMPTS,
        }
        primary_modes[mode]["overall_pass"] = all(primary_modes[mode].values())
    overall = bool(correctness_pass and all(row["overall_pass"] for row in primary_modes.values()))
    return {
        "comparisons": comparisons,
        "correctness_overall_pass": bool(correctness_pass),
        "primary_gate": {
            "role": PRIMARY_ROLE,
            "minimum_end_to_end_reduction": MINIMUM_END_TO_END_POINT_REDUCTION,
            "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "requires_both_modes": True,
            "by_mode": primary_modes,
            "overall_pass": overall,
        },
        "status": (
            "pass_16k_retrieval_actual_development"
            if overall
            else "fail_16k_retrieval_actual_development"
        ),
        "korean_specific_followup_authorized": overall,
        "generic_retrieval_is_novel_claimed": False,
        "publication_claim": False,
    }
