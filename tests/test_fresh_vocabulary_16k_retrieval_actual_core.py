from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from fresh_vocabulary_16k_retrieval_actual_core import (
    CONTINUATION_BYTES,
    COUNTER_NAMES,
    MEASURED_CASES,
    MODES,
    PRIMARY_ROLE,
    REPETITIONS,
    ROLE_ORDERS,
    ROLES,
    TIMING_COMPONENTS,
    balanced_role_order,
    summarize_retrieval_actual,
)


def _inputs(*, primary_ms: tuple[float, float] = (8.0, 8.0)) -> dict[str, object]:
    shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    end_to_end = np.empty(shape, dtype=np.float64)
    role_ms = {
        "baseline_ar": (10.0, 10.0),
        "prompt_lookup_block_4": (9.0, 9.0),
        "corpus_ngram_block_4": (8.5, 8.5),
        PRIMARY_ROLE: primary_ms,
    }
    for role, by_mode in role_ms.items():
        for mode_index, value in enumerate(by_mode):
            end_to_end[mode_index, ..., ROLES.index(role)] = value
    timing = {
        component: (
            end_to_end.copy() if component == "end_to_end_ms" else end_to_end * 0.5
        )
        for component in TIMING_COMPONENTS
    }
    timing["draft_lookup_ms"] = np.zeros(shape, dtype=np.float64)

    output_tokens = np.full(shape, 32, dtype=np.int16)
    counters = {
        name: np.zeros(shape, dtype=np.int16) for name in COUNTER_NAMES
    }
    counters["target_forward_calls"][..., ROLES.index("baseline_ar")] = 32
    for role in ROLES[1:]:
        role_index = ROLES.index(role)
        counters["target_forward_calls"][..., role_index] = 20
        counters["proposal_attempts"][..., role_index] = 5
        counters["proposal_tokens"][..., role_index] = 15
        counters["accepted_draft_tokens"][..., role_index] = 10
        counters["full_accept_cycles"][..., role_index] = 3
        counters["rejection_cycles"][..., role_index] = 2
        counters["no_draft_steps"][..., role_index] = 10
        counters["correction_tokens"][..., role_index] = 2
        counters["bonus_tokens"][..., role_index] = 3
        counters["cropped_input_tokens"][..., role_index] = 5
    counters["prompt_lookup_proposals"][
        ..., ROLES.index("prompt_lookup_block_4")
    ] = 5
    counters["corpus_ngram_proposals"][
        ..., ROLES.index("corpus_ngram_block_4")
    ] = 5
    counters["corpus_ngram_proposals"][..., ROLES.index(PRIMARY_ROLE)] = 3
    counters["prompt_lookup_proposals"][..., ROLES.index(PRIMARY_ROLE)] = 2
    return {
        "timing": timing,
        "counters": counters,
        "output_token_count": output_tokens,
        "output_raw_byte_count": np.full(
            shape,
            CONTINUATION_BYTES,
            dtype=np.int16,
        ),
        "correctness_pass": True,
        "maximum_output_bytes": CONTINUATION_BYTES + 27,
    }


def test_four_role_schedule_is_exactly_balanced() -> None:
    rows: Counter[tuple[int, ...]] = Counter()
    positions = [[0] * len(ROLES) for _ in ROLES]
    for case_index in range(MEASURED_CASES):
        for repetition_index in range(REPETITIONS):
            for mode_index in range(len(MODES)):
                order = balanced_role_order(case_index, repetition_index, mode_index)
                rows[order] += 1
                for position, role_index in enumerate(order):
                    positions[position][role_index] += 1
    assert set(rows) == set(ROLE_ORDERS)
    assert set(rows.values()) == {80}
    assert all(set(row) == {160} for row in positions)


def test_fixed_hybrid_primary_passes_only_as_development_evidence() -> None:
    result = summarize_retrieval_actual(**_inputs())
    assert result["primary_gate"]["overall_pass"] is True
    assert result["korean_specific_followup_authorized"] is True
    assert result["generic_retrieval_is_novel_claimed"] is False
    assert result["publication_claim"] is False


def test_diagnostic_roles_cannot_rescue_failed_hybrid_primary() -> None:
    result = summarize_retrieval_actual(**_inputs(primary_ms=(9.5, 9.5)))
    assert result["comparisons"]["corpus_ngram_block_4"]["controlled_replay"][
        "end_to_end_reduction"
    ] == pytest.approx(0.15)
    assert result["primary_gate"]["overall_pass"] is False
    assert result["korean_specific_followup_authorized"] is False


def test_counter_source_and_output_identities_are_fail_closed() -> None:
    inputs = _inputs()
    inputs["counters"]["prompt_lookup_proposals"][0, 0, 0, 3] += 1
    with pytest.raises(ValueError, match="counter identity"):
        summarize_retrieval_actual(**inputs)

    inputs = _inputs()
    inputs["output_raw_byte_count"][1, 0, 0, 2] += 1
    with pytest.raises(ValueError, match="output counts"):
        summarize_retrieval_actual(**inputs)


def test_correctness_is_required_in_both_modes() -> None:
    inputs = _inputs()
    inputs["correctness_pass"] = False
    result = summarize_retrieval_actual(**inputs)
    assert result["primary_gate"]["overall_pass"] is False
