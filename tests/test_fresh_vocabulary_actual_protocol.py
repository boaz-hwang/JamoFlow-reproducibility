from __future__ import annotations

from pathlib import Path

from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_actual_core import MEASURED_CASES, WARMUP_CASES
from fresh_vocabulary_actual_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    VOCABULARY_BY_ROLE,
    build_role_model,
    encode_raw,
    reconstruct_cases,
)
from scalar_runtime_core import model_parameter_count


def test_implementation_manifest_is_unique_and_complete() -> None:
    assert len(IMPLEMENTATION_PATHS) == len(set(IMPLEMENTATION_PATHS))
    assert all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS)


def test_both_tokenizers_roundtrip_same_korean_bytes() -> None:
    raw = "한글 vocabulary actual inference 검증".encode()
    loaded = load_tokenizers()
    for size in VOCABULARY_BY_ROLE.values():
        tokenizer, token_bytes = loaded[size]
        ids = encode_raw(raw, tokenizer, token_bytes)
        assert b"".join(token_bytes[value] for value in ids) == raw


def test_case_reconstruction_is_document_distinct_and_strict_utf8() -> None:
    prompts, continuations, metadata = reconstruct_cases()
    expected = WARMUP_CASES + MEASURED_CASES
    assert prompts.shape == (expected, 128)
    assert continuations.shape == (expected, 128)
    assert metadata["selected_cases"] == expected
    assert metadata["selected_unique_clusters"] == expected
    for row in prompts:
        bytes(row).decode("utf-8", errors="strict")
    for row in continuations:
        bytes(row).decode("utf-8", errors="strict")


def test_role_graphs_match_sealed_parameter_geometry() -> None:
    reference = build_role_model("reference")
    candidate = build_role_model("candidate")
    assert model_parameter_count(reference) == 19_667_328
    assert model_parameter_count(candidate) == 25_172_352
    assert reference.config.tie_word_embeddings is True
    assert candidate.config.tie_word_embeddings is False


def test_no_implementation_path_escapes_repository() -> None:
    root = ROOT.resolve()
    for relative in IMPLEMENTATION_PATHS:
        assert root in (ROOT / Path(relative)).resolve().parents
