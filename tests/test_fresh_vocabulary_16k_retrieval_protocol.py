from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
from fresh_vocabulary_16k_actual_protocol import canonical_sha256
from fresh_vocabulary_16k_retrieval_actual_core import PRIMARY_ROLE, ROLES
from fresh_vocabulary_16k_retrieval_protocol import (
    IMPLEMENTATION_PATHS,
    MAXIMUM_FREE_TOKENS,
    ROOT,
    build_plan,
    load_table,
    reconstruct_cases,
    validate_plan,
)
from fresh_vocabulary_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    WARMUP_CASES,
)


def test_implementation_manifest_is_unique_and_complete() -> None:
    assert len(IMPLEMENTATION_PATHS) == len(set(IMPLEMENTATION_PATHS))
    assert all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS)


def test_all_retrieval_actual_entrypoints_import() -> None:
    for module in (
        "benchmark_fresh_vocabulary_16k_retrieval",
        "preflight_fresh_vocabulary_16k_retrieval",
        "seal_fresh_vocabulary_16k_retrieval_plan",
        "summarize_fresh_vocabulary_16k_retrieval",
    ):
        assert importlib.import_module(module) is not None


def test_train_only_table_and_cases_are_fixed_before_timing() -> None:
    table = load_table()
    assert table.entry_count == 200_000
    prompts, continuations, metadata = reconstruct_cases()
    expected = WARMUP_CASES + MEASURED_CASES
    assert prompts.shape == continuations.shape == (expected, 128)
    assert metadata["selected_cases"] == expected
    plan = build_plan(git_commit_before_plan="a" * 40)
    validate_plan(plan, verify_derived=True)
    assert plan["experiment"]["roles"] == list(ROLES)
    assert plan["gate"]["primary_role"] == PRIMARY_ROLE
    assert plan["prior_evidence"]["table_uses_train_only"] is True
    assert plan["claim_boundary"]["korean_specific_method_tested"] is False
    assert plan["claim_boundary"]["publication_claim"] is False
    assert MAXIMUM_FREE_TOKENS == CONTINUATION_BYTES + 3


def test_resealed_gate_or_claim_tamper_is_rejected() -> None:
    original = build_plan(git_commit_before_plan="a" * 40)
    for field, nested, replacement in (
        ("gate", "primary_role", "corpus_ngram_block_4"),
        ("claim_boundary", "publication_claim", True),
    ):
        plan = deepcopy(original)
        plan[field][nested] = replacement
        unsigned = dict(plan)
        unsigned.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(unsigned)
        with pytest.raises(ValueError, match="plan identity"):
            validate_plan(plan, verify_derived=False)


def test_no_implementation_path_escapes_repository() -> None:
    root = ROOT.resolve()
    for relative in IMPLEMENTATION_PATHS:
        assert root in (ROOT / Path(relative)).resolve().parents
