from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
from fresh_vocabulary_16k_actual_protocol import canonical_sha256
from fresh_vocabulary_16k_retrieval_mechanism_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    build_plan,
    hypothesis_contract,
    validate_plan,
)


def test_mechanism_implementation_manifest_is_unique_and_complete() -> None:
    assert len(IMPLEMENTATION_PATHS) == len(set(IMPLEMENTATION_PATHS))
    assert all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS)


def test_mechanism_entrypoints_import() -> None:
    for module in (
        "profile_fresh_vocabulary_16k_retrieval_mechanism",
        "seal_fresh_vocabulary_16k_retrieval_mechanism_plan",
    ):
        assert importlib.import_module(module) is not None


def test_primary_contrast_is_fixed_and_claim_scope_is_narrow() -> None:
    plan = build_plan(git_commit_before_plan="a" * 40)
    validate_plan(plan)
    assert plan["hypothesis"] == hypothesis_contract()
    assert plan["hypothesis"]["contrast"] == (
        "within_hangul_eojeol_minus_after_whitespace"
    )
    assert plan["hypothesis"]["no_secondary_feature_fallback"] is True
    assert plan["claim_boundary"]["efficiency_claim"] is False
    assert plan["claim_boundary"]["pass_authorizes_only_disjoint_design"] is True


def test_resealed_hypothesis_or_claim_tamper_is_rejected() -> None:
    original = build_plan(git_commit_before_plan="a" * 40)
    for field, nested, replacement in (
        ("hypothesis", "minimum_point_gap", 0.0),
        ("claim_boundary", "efficiency_claim", True),
    ):
        plan = deepcopy(original)
        plan[field][nested] = replacement
        unsigned = dict(plan)
        unsigned.pop("plan_sha256")
        plan["plan_sha256"] = canonical_sha256(unsigned)
        with pytest.raises(ValueError, match="plan identity"):
            validate_plan(plan)


def test_no_mechanism_implementation_path_escapes_repository() -> None:
    root = ROOT.resolve()
    for relative in IMPLEMENTATION_PATHS:
        assert root in (ROOT / Path(relative)).resolve().parents
