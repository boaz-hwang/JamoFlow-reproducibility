from __future__ import annotations

import numpy as np

from length_gain import (
    _nonoverlapping_count,
    length_gain_decision,
    rank_length_gain_candidates,
    train_length_gain_vocabulary,
)


def test_candidate_ranking_uses_exact_nonoverlapping_saving() -> None:
    pieces = tuple(bytes((value,)) for value in range(256))
    token_ids = np.frombuffer(b"aaaaa\nabababab", dtype=np.uint8).astype(np.uint16)
    ranked = rank_length_gain_candidates(
        token_ids,
        pieces,
        batch_size=1,
        maximum_token_arity=3,
        maximum_piece_bytes=8,
        score_kind="immediate_saving",
        initial_group_limit=4,
    )
    assert len(ranked) == 1
    assert ranked[0].raw == b"ab"
    assert ranked[0].overlapping_occurrences == 4
    assert ranked[0].nonoverlapping_occurrences == 4
    assert ranked[0].score == 4
    assert _nonoverlapping_count(np.asarray([0, 1, 2, 3]), 2) == 2


def test_candidate_never_crosses_newline() -> None:
    pieces = tuple(bytes((value,)) for value in range(256))
    token_ids = np.frombuffer(b"x\nyx\ny", dtype=np.uint8).astype(np.uint16)
    ranked = rank_length_gain_candidates(
        token_ids,
        pieces,
        batch_size=1,
        maximum_token_arity=3,
        maximum_piece_bytes=8,
        score_kind="immediate_saving",
        initial_group_limit=8,
    )
    assert all(b"\n" not in candidate.raw for candidate in ranked)


def test_tiny_training_is_deterministic_and_monotonic() -> None:
    raw = bytes(range(32, 127)) * 10 + (b"alpha beta gamma delta\n" * 20)
    first = train_length_gain_vocabulary(
        raw,
        vocabulary_size=264,
        batch_size=2,
        maximum_token_arity=4,
        maximum_piece_bytes=16,
    )
    second = train_length_gain_vocabulary(
        raw,
        vocabulary_size=264,
        batch_size=2,
        maximum_token_arity=4,
        maximum_piece_bytes=16,
    )
    assert first == second
    assert len(first.pieces) == 264
    assert first.final_token_count < first.initial_token_count
    assert all(row.realized_token_reduction > 0 for row in first.rounds)
    assert b"".join(first.pieces[value] for value in range(256)) == bytes(range(256))


def test_decision_uses_calibration_and_measured_continuations() -> None:
    metrics = {
        "bpe": {
            "calibration_token_count": 1_000,
            "continuation_token_counts": [10, 10, 100, 100],
        },
        "longest": {
            "calibration_token_count": 901,
            "continuation_token_counts": [999, 999, 89, 90],
        },
        "minimum": {
            "calibration_token_count": 880,
            "continuation_token_counts": [999, 999, 89, 89],
        },
    }
    decision = length_gain_decision(
        metrics,
        baseline_role="bpe",
        primary_order=("longest", "minimum"),
        warmup_cases=2,
        minimum_reduction=0.10,
    )
    assert decision["selected_role"] == "minimum"
    assert not decision["comparisons"]["longest"]["overall_pass"]
    assert decision["comparisons"]["minimum"]["overall_pass"]
