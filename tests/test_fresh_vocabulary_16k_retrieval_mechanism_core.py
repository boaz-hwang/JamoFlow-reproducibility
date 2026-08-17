from __future__ import annotations

import json

import numpy as np
from fresh_vocabulary_16k_retrieval_core import (
    CompactBackoffTable,
    OrderTable,
    pack_context,
)
from fresh_vocabulary_16k_retrieval_mechanism_core import (
    ProposalEvent,
    primary_hangul_boundary_contrast,
    replay_proposal_events,
)


def _table() -> CompactBackoffTable:
    rows = {}
    for order, context, token in (
        (1, (70,), 71),
        (2, (80, 81), 82),
        (3, (90, 91, 92), 93),
    ):
        rows[order] = OrderTable(
            order=order,
            contexts=np.asarray([pack_context(context)], dtype=np.uint64),
            next_tokens=np.asarray([token], dtype=np.uint16),
            best_counts=np.asarray([9], dtype=np.uint32),
            total_counts=np.asarray([10], dtype=np.uint32),
        )
    table = CompactBackoffTable(rows)
    table.validate()
    return table


def _transitions(size: int) -> tuple[tuple[int, ...], ...]:
    return (tuple(0 for _ in range(size)),)


def test_prompt_event_replay_reconstructs_the_exact_target_trace() -> None:
    token_bytes = tuple(bytes((value,)) for value in range(128))
    prompt = (10, 20, 10, 20)
    expected = (10, 20, 10, 20, 30)
    events = replay_proposal_events(
        case_index=0,
        mode="free_running_utf8_greedy",
        role="prompt_lookup_block_4",
        prompt_raw=bytes(prompt),
        prompt_ids=prompt,
        expected_raw=bytes(expected),
        expected_ids=expected,
        token_bytes=token_bytes,
        next_state_indices=_transitions(len(token_bytes)),
        table=_table(),
    )
    assert events
    assert sum(event.accepted_tokens for event in events) >= 2
    assert all(event.source in {"prompt_lookup", "none"} for event in events)


def _event(case: int, boundary: str, accepted: int) -> ProposalEvent:
    return ProposalEvent(
        case_index=case,
        mode="free_running_utf8_greedy",
        role="hybrid_retrieval_block_4",
        source="prompt_lookup",
        boundary_class=boundary,
        eojeol_hangul_syllables=2 if boundary == "within_hangul_eojeol" else 0,
        proposal_tokens=3,
        accepted_tokens=accepted,
        outcome="full_accept_bonus" if accepted == 3 else "rejection",
        proposal_contains_whitespace=False,
        prompt_match_tokens=2,
        first_table_order=0,
        minimum_table_confidence=0.0,
    )


def test_primary_hangul_boundary_gate_requires_effect_and_coverage() -> None:
    events = []
    for case in range(20):
        events.extend([_event(case, "within_hangul_eojeol", 2)] * 2)
        events.extend([_event(case, "after_whitespace", 0)] * 2)
    result = primary_hangul_boundary_contrast(events)
    assert result["cycle_counts"] == {
        "within_hangul_eojeol": 40,
        "after_whitespace": 40,
    }
    assert result["paired_case_mean_difference"] == 2.0
    assert result["gate"]["overall_pass"] is True

    weak = [
        _event(case, boundary, 1)
        for case in range(20)
        for boundary in ("within_hangul_eojeol", "after_whitespace")
        for _ in range(2)
    ]
    assert primary_hangul_boundary_contrast(weak)["gate"]["overall_pass"] is False


def test_insufficient_paired_coverage_remains_canonical_json_serializable() -> None:
    events = [_event(case, "within_hangul_eojeol", 2) for case in range(40)]
    result = primary_hangul_boundary_contrast(events)
    assert result["paired_case_count"] == 0
    assert result["paired_case_mean_difference"] is None
    assert result["paired_case_bootstrap_95_interval"] == {
        "lower": None,
        "upper": None,
    }
    assert result["gate"]["minimum_effect"] is False
    assert result["gate"]["bootstrap_lower_positive"] is False
    json.dumps(result, allow_nan=False)
