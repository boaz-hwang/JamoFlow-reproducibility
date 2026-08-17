from __future__ import annotations

from pathlib import Path

from length_gain_opportunity_protocol import (
    BATCH_SIZE,
    MAXIMUM_TOKEN_ARITY,
    PRIMARY_ORDER,
    ROLE_ORDER,
    SCORE_KIND,
    load_pieces,
    serialize_pieces,
)


def test_frozen_constructor_coordinates() -> None:
    assert BATCH_SIZE == 8
    assert MAXIMUM_TOKEN_ARITY == 8
    assert SCORE_KIND == "immediate_saving"
    assert ROLE_ORDER[1:] == PRIMARY_ORDER


def test_piece_artifact_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import length_gain_opportunity_protocol as protocol

    pieces = tuple(bytes((value,)) for value in range(256)) + tuple(
        f"piece-{index}".encode("ascii") for index in range(1_792)
    )
    target = tmp_path / "pieces.npz"
    serialize_pieces(target, pieces)
    monkeypatch.setattr(protocol, "VOCABULARY_SIZE", len(pieces))
    assert load_pieces(target) == pieces
