from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
from fresh_vocabulary_16k_actual_protocol import canonical_sha256
from fresh_vocabulary_16k_block_core import (
    MEASURED_CASES,
    PRIMARY_ROLE,
    ROLES,
)
from fresh_vocabulary_16k_block_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    TARGET_VOCABULARY_SIZE,
    build_plan,
    reconstruct_cases,
    target_identity,
    validate_plan,
)
from fresh_vocabulary_actual_core import WARMUP_CASES


def test_implementation_manifest_is_unique_and_complete() -> None:
    assert len(IMPLEMENTATION_PATHS) == len(set(IMPLEMENTATION_PATHS))
    assert all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS)


def test_all_target_block_entrypoints_import() -> None:
    for module in (
        "benchmark_fresh_vocabulary_16k_block",
        "preflight_fresh_vocabulary_16k_block",
        "seal_fresh_vocabulary_16k_block_plan",
        "summarize_fresh_vocabulary_16k_block",
    ):
        assert importlib.import_module(module) is not None


def test_target_identity_is_the_trained_quality_qualified_16k_model() -> None:
    identity = target_identity()
    assert identity["vocabulary_size"] == TARGET_VOCABULARY_SIZE == 16_000
    assert identity["parameter_count"] == 31_168_896
    assert identity["quality_role"] == "dense16k_update_geometry"
    assert len(identity["checkpoint_artifact_sha256"]) == 64
    assert len(identity["checkpoint_state_sha256"]) == 64


def test_cases_and_primary_role_are_fixed_before_timing() -> None:
    prompts, continuations, metadata = reconstruct_cases()
    expected = WARMUP_CASES + MEASURED_CASES
    assert prompts.shape == continuations.shape == (expected, 128)
    assert metadata["selected_cases"] == expected
    plan = build_plan(git_commit_before_plan="a" * 40)
    validate_plan(plan, verify_derived=True)
    assert plan["experiment"]["roles"] == list(ROLES)
    assert plan["gate"]["primary_role"] == PRIMARY_ROLE == "perfect_block_4"
    assert plan["gate"]["diagnostic_fallback"] is False
    assert plan["claim_boundary"]["perfect_draft_upper_bound"] is True
    assert plan["claim_boundary"]["publication_claim"] is False


def test_resealed_gate_or_claim_tamper_is_rejected() -> None:
    original = build_plan(git_commit_before_plan="a" * 40)
    for field, nested, replacement in (
        ("gate", "primary_role", "perfect_block_8"),
        ("claim_boundary", "actual_draft_compute_measured", True),
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
