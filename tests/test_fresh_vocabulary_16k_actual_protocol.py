from __future__ import annotations

import importlib
from pathlib import Path

from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_actual_core import MEASURED_CASES, ROLES
from fresh_vocabulary_16k_actual_protocol import (
    IMPLEMENTATION_PATHS,
    QUALITY_ROLE_BY_ACTUAL_ROLE,
    ROOT,
    VOCABULARY_BY_ROLE,
    build_role_model,
    encode_raw,
    model_identity,
    quality_result,
    reconstruct_cases,
)
from fresh_vocabulary_actual_core import WARMUP_CASES
from scalar_runtime_core import model_parameter_count


def test_implementation_manifest_is_unique_and_complete() -> None:
    assert len(IMPLEMENTATION_PATHS) == len(set(IMPLEMENTATION_PATHS))
    assert all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS)


def test_all_actual_entrypoints_import() -> None:
    for module in (
        "benchmark_fresh_vocabulary_16k_actual",
        "preflight_fresh_vocabulary_16k_actual",
        "seal_fresh_vocabulary_16k_actual_plan",
        "summarize_fresh_vocabulary_16k_actual",
    ):
        assert importlib.import_module(module) is not None


def test_three_tokenizers_roundtrip_same_korean_bytes() -> None:
    raw = "한글 16K vocabulary actual inference 검증".encode()
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


def test_role_graphs_match_the_sealed_capacity_contract() -> None:
    expected = {
        "baseline_2k": (19_667_328, True),
        "frontier_8k": (25_172_352, False),
        "candidate_16k": (31_168_896, False),
    }
    for role in ROLES:
        model = build_role_model(role)
        parameters, tied = expected[role]
        assert model_parameter_count(model) == parameters
        assert model.config.tie_word_embeddings is tied


def test_quality_result_fixes_the_three_physical_roles_without_latency() -> None:
    result = quality_result()
    identities = model_identity()
    assert result["decision"]["actual_inference_preflight_authorized"] is True
    assert set(identities) == set(ROLES)
    for role in ROLES:
        assert identities[role]["quality_role"] == QUALITY_ROLE_BY_ACTUAL_ROLE[role]
        assert len(identities[role]["checkpoint_artifact_sha256"]) == 64
        assert len(identities[role]["checkpoint_state_sha256"]) == 64
    assert (
        identities["candidate_16k"]["document_bpb"]
        < identities["baseline_2k"]["document_bpb"]
    )
    assert (
        identities["candidate_16k"]["document_bpb"]
        < identities["frontier_8k"]["document_bpb"]
    )


def test_no_implementation_path_escapes_repository() -> None:
    root = ROOT.resolve()
    for relative in IMPLEMENTATION_PATHS:
        assert root in (ROOT / Path(relative)).resolve().parents
