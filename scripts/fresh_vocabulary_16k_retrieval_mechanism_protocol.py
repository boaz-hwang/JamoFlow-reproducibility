"""Sealed contract for the 16K retrieval mechanism-only development audit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from compositional_head_preflight_protocol import ROOT, current_environment, hash_file
from fresh_vocabulary_16k_actual_protocol import canonical_sha256, json_bytes
from fresh_vocabulary_16k_retrieval_mechanism_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    MINIMUM_ACCEPTED_TOKENS_PER_CYCLE_GAP,
    MINIMUM_PAIRED_CASES,
    MINIMUM_STRATUM_CYCLES,
    MODES,
    PRIMARY_BOUNDARIES,
    PRIMARY_SOURCE,
    PROFILE_ROLES,
)
from fresh_vocabulary_16k_retrieval_protocol import (
    PLAN_PATH as ACTUAL_PLAN_PATH,
)
from fresh_vocabulary_16k_retrieval_protocol import (
    TABLE_PATH,
    TABLE_SEAL_PATH,
    read_json,
)
from fresh_vocabulary_16k_retrieval_protocol import (
    validate_plan as validate_actual_plan,
)

ACTUAL_RESULT_PATH = ROOT / "results/fresh-vocabulary-16k-retrieval-actual-v1/summary.json"
INVALIDATED_V1_PLAN_PATH = (
    ROOT / "data/manifests/fresh-vocabulary-16k-retrieval-mechanism-v1.json"
)
INVALIDATED_V1_OUTPUT_PATH = (
    ROOT / "results/fresh-vocabulary-16k-retrieval-mechanism-v1/summary.json"
)
PLAN_PATH = ROOT / "data/manifests/fresh-vocabulary-16k-retrieval-mechanism-v2.json"
OUTPUT_PATH = ROOT / "results/fresh-vocabulary-16k-retrieval-mechanism-v2/summary.json"
PROTOCOL_ID = "jamoflow-fresh-vocabulary-16k-retrieval-mechanism-v2"

IMPLEMENTATION_PATHS = (
    "data/seals/fresh-vocabulary-16k-retrieval-table-v1.json",
    "data/manifests/fresh-vocabulary-16k-retrieval-mechanism-v1.json",
    "docs/167-fresh-v2-16k-retrieval-actual-result-and-free-path-correction.md",
    "docs/168-fresh-v2-16k-retrieval-mechanism-audit-protocol.md",
    "docs/169-retrieval-mechanism-v1-invalidation-and-v2-correction.md",
    "pyproject.toml",
    "results/fresh-vocabulary-16k-retrieval-actual-v1/summary.json",
    "scripts/benchmark_fresh_vocabulary_16k_block.py",
    "scripts/benchmark_fresh_vocabulary_actual.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/fresh_vocabulary_16k_actual_protocol.py",
    "scripts/fresh_vocabulary_16k_block_protocol.py",
    "scripts/fresh_vocabulary_16k_retrieval_core.py",
    "scripts/fresh_vocabulary_16k_retrieval_mechanism_core.py",
    "scripts/fresh_vocabulary_16k_retrieval_mechanism_protocol.py",
    "scripts/fresh_vocabulary_16k_retrieval_protocol.py",
    "scripts/profile_fresh_vocabulary_16k_retrieval_mechanism.py",
    "scripts/seal_fresh_vocabulary_16k_retrieval_mechanism_plan.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "tests/test_fresh_vocabulary_16k_retrieval_mechanism_core.py",
    "tests/test_fresh_vocabulary_16k_retrieval_mechanism_protocol.py",
)

__all__ = ("json_bytes",)


def _validated_dependencies() -> tuple[dict[str, Any], dict[str, Any]]:
    actual_plan = read_json(ACTUAL_PLAN_PATH)
    validate_actual_plan(actual_plan, verify_derived=False)
    actual_result = read_json(ACTUAL_RESULT_PATH)
    unsigned = dict(actual_result)
    recorded = unsigned.pop("summary_sha256", None)
    if (
        actual_result.get("kind") != "fresh_vocabulary_16k_retrieval_actual_result_v1"
        or actual_result.get("status") != "fail_16k_retrieval_actual_development"
        or canonical_sha256(unsigned) != recorded
        or actual_result.get("independent_correctness", {}).get("overall_pass") is not True
        or actual_result.get("actual_retrieval", {})
        .get("primary_gate", {})
        .get("by_mode", {})
        .get("free_running_utf8_greedy", {})
        .get("overall_pass")
        is not True
        or actual_result.get("actual_retrieval", {})
        .get("primary_gate", {})
        .get("overall_pass")
        is not False
    ):
        raise ValueError("retrieval mechanism dependency result differs")
    return actual_plan, actual_result


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("retrieval mechanism implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "actual_plan": ACTUAL_PLAN_PATH,
        "actual_result": ACTUAL_RESULT_PATH,
        "invalidated_v1_plan": INVALIDATED_V1_PLAN_PATH,
        "retrieval_table_seal": TABLE_SEAL_PATH,
        "retrieval_table_artifact": TABLE_PATH,
    }
    return {
        key: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for key, path in paths.items()
    }


def hypothesis_contract() -> dict[str, Any]:
    return {
        "mode": "free_running_utf8_greedy",
        "role": "hybrid_retrieval_block_4",
        "source": PRIMARY_SOURCE,
        "contrast": "within_hangul_eojeol_minus_after_whitespace",
        "boundary_order": list(PRIMARY_BOUNDARIES),
        "outcome": "accepted_tokens_per_proposal_cycle",
        "minimum_cycles_each": MINIMUM_STRATUM_CYCLES,
        "minimum_paired_cases": MINIMUM_PAIRED_CASES,
        "minimum_point_gap": MINIMUM_ACCEPTED_TOKENS_PER_CYCLE_GAP,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_lower_must_exceed": 0.0,
        "all_conditions_required": True,
        "no_secondary_feature_fallback": True,
    }


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if len(git_commit_before_plan) != 40:
        raise ValueError("retrieval mechanism pre-plan commit differs")
    actual_plan, actual_result = _validated_dependencies()
    if INVALIDATED_V1_OUTPUT_PATH.exists() or hash_file(INVALIDATED_V1_PLAN_PATH) == "":
        raise ValueError("retrieval mechanism v1 invalidation differs")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "fresh_vocabulary_16k_retrieval_mechanism_plan_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_mechanism_replay",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "actual_evidence": {
            "plan_sha256": actual_plan["plan_sha256"],
            "summary_sha256": actual_result["summary_sha256"],
            "joint_gate_pass": False,
            "free_mode_gate_pass": True,
            "independent_correctness_pass": True,
        },
        "correction": {
            "invalidated_v1_plan_artifact_sha256": hash_file(INVALIDATED_V1_PLAN_PATH),
            "v1_result_published": False,
            "failure_stage": "canonical_json_before_result_publish",
            "change": (
                "normalize NumPy comparison booleans to Python bool and unavailable "
                "finite diagnostics to JSON null"
            ),
            "hypothesis_or_gate_changed": False,
            "event_aggregates_observed_before_correction": False,
        },
        "workload": {
            "case_source": "same closed 64-case development set",
            "case_count": 64,
            "modes": list(MODES),
            "roles": list(PROFILE_ROLES),
            "target_outputs_regenerated_from_checkpoint": True,
            "event_metrics_only": True,
            "latency_measured": False,
        },
        "hypothesis": hypothesis_contract(),
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "claim_boundary": {
            "development_mechanism_audit": True,
            "same_closed_cases_reused": True,
            "raw_text_or_token_ids_published": False,
            "efficiency_claim": False,
            "korean_method_implemented": False,
            "pass_authorizes_only_disjoint_design": True,
            "publication_claim": False,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "kind",
        "protocol_id",
        "status",
        "git_commit_before_plan",
        "dependencies",
        "implementation_sha256",
        "environment",
        "actual_evidence",
        "correction",
        "workload",
        "hypothesis",
        "output_path",
        "claim_boundary",
        "plan_sha256",
    }
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    rebuilt = build_plan(git_commit_before_plan=str(plan.get("git_commit_before_plan", "")))
    if (
        set(plan) != expected
        or plan.get("schema_version") != 2
        or plan.get("kind") != "fresh_vocabulary_16k_retrieval_mechanism_plan_v2"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_mechanism_replay"
        or canonical_sha256(unsigned) != recorded
        or dict(plan) != rebuilt
    ):
        raise ValueError("retrieval mechanism plan identity differs")
