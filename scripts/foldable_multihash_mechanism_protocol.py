"""Sealed identities for the foldable multi-hash mechanism-control screen."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from foldable_jamo_residual_protocol import (
    OUTPUT_PATH as PARENT_RESULT_PATH,
)
from foldable_jamo_residual_protocol import (
    PLAN_PATH as PARENT_PLAN_PATH,
)
from foldable_jamo_residual_protocol import (
    ROOT,
    current_environment,
    hash_file,
    read_json,
    training_contract,
)
from foldable_multihash_mechanism_core import (
    INPUT_UPDATE_MULTIPLIER,
    NEW_ROLES,
    OUTPUT_UPDATE_MULTIPLIER,
    PROTOCOL_ID,
)
from seal_foldable_multihash_update_audit_plan import (
    PLAN_PATH as AUDIT_PLAN_PATH,
)
from seal_foldable_multihash_update_audit_plan import (
    RESULT_PATH as AUDIT_RESULT_PATH,
)

PLAN_PATH = ROOT / "data/manifests/foldable-multihash-mechanism-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/foldable-multihash-mechanism-v1"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
DEPLOYED_ROOT = ARTIFACT_ROOT / "deployed"
ACTIVE_ROOT = ARTIFACT_ROOT / "active"
REPORT_PATH = ARTIFACT_ROOT / "campaign.json"
OUTPUT_PATH = ROOT / "results/foldable-multihash-mechanism-v1/summary.json"

IMPLEMENTATION_PATHS = (
    "docs/141-foldable-jamo-residual-result-and-multihash-pivot.md",
    "docs/143-foldable-vocabulary-reparameterization-literature-audit.md",
    "docs/148-foldable-multihash-update-audit-result-and-mechanism-decision.md",
    "docs/149-foldable-multihash-mechanism-control-protocol.md",
    "pyproject.toml",
    "scripts/foldable_jamo_residual_core.py",
    "scripts/foldable_jamo_residual_protocol.py",
    "scripts/foldable_multihash_mechanism_core.py",
    "scripts/foldable_multihash_mechanism_protocol.py",
    "scripts/foldable_multihash_update_audit_core.py",
    "scripts/run_compositional_quality.py",
    "scripts/run_foldable_jamo_residual.py",
    "scripts/run_foldable_multihash_mechanism.py",
    "scripts/seal_foldable_multihash_mechanism_plan.py",
    "scripts/summarize_foldable_multihash_mechanism.py",
    "scripts/vocabulary_transfer_baseline_core.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "tests/test_foldable_multihash_mechanism_core.py",
    "tests/test_foldable_multihash_mechanism_protocol.py",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def role_definition(role: str) -> dict[str, Any]:
    if role == "update_matched_dense":
        return {
            "role": role,
            "training_graph": "ordinary_untied_dense_bpe_8192",
            "assignment_kind": None,
            "post_adamw_new_row_input_multiplier": INPUT_UPDATE_MULTIPLIER,
            "post_adamw_new_row_output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
            "deployed_graph": "ordinary_untied_dense_bpe_8192",
        }
    if role in {"stratified_generic_shuffle", "balanced_random_multihash"}:
        return {
            "role": role,
            "training_graph": "untied_foldable_13x128_residual",
            "assignment_kind": role,
            "post_adamw_new_row_input_multiplier": None,
            "post_adamw_new_row_output_multiplier": None,
            "deployed_graph": "ordinary_untied_dense_bpe_8192",
        }
    raise ValueError("mechanism role differs")


def worker_paths(role: str) -> tuple[Path, dict[int, Path], dict[int, Path], Path]:
    if role not in NEW_ROLES:
        raise ValueError("mechanism worker role differs")
    from foldable_jamo_residual_core import PROBE_STEPS

    return (
        WORKER_ROOT / f"{role}.json",
        {step: CHECKPOINT_ROOT / f"{role}-step-{step:04d}.pt" for step in PROBE_STEPS},
        {step: NLL_ROOT / f"{role}-step-{step:04d}.npz" for step in PROBE_STEPS},
        DEPLOYED_ROOT / f"{role}-step-{PROBE_STEPS[-1]:04d}.pt",
    )


def dependency_identity() -> dict[str, Any]:
    parent_plan = read_json(PARENT_PLAN_PATH)
    parent_result = read_json(PARENT_RESULT_PATH)
    audit_plan = read_json(AUDIT_PLAN_PATH)
    audit_result = read_json(AUDIT_RESULT_PATH)
    return {
        "parent_plan": {
            "path": str(PARENT_PLAN_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(PARENT_PLAN_PATH),
            "payload_sha256": parent_plan["plan_sha256"],
        },
        "parent_result": {
            "path": str(PARENT_RESULT_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(PARENT_RESULT_PATH),
            "payload_sha256": parent_result["summary_sha256"],
        },
        "update_audit_plan": {
            "path": str(AUDIT_PLAN_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(AUDIT_PLAN_PATH),
            "payload_sha256": audit_plan["plan_sha256"],
        },
        "update_audit_result": {
            "path": str(AUDIT_RESULT_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(AUDIT_RESULT_PATH),
            "payload_sha256": audit_result["summary_sha256"],
        },
    }


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected_keys = {
        "assignment_audits",
        "claim_boundary",
        "dependencies",
        "document_common",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "initialization",
        "inventories",
        "kind",
        "output_path",
        "plan_sha256",
        "protocol_id",
        "roles",
        "schema_version",
        "selection_rule",
        "status",
        "training",
        "update_control",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "foldable_multihash_mechanism_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "development_mechanism_controls_sealed_before_new_training"
        or plan.get("output_path") != str(OUTPUT_PATH.relative_to(ROOT))
        or plan.get("training") != training_contract()
        or set(plan.get("roles", {})) != set(NEW_ROLES)
        or plan.get("roles") != {role: role_definition(role) for role in NEW_ROLES}
        or plan.get("update_control")
        != {
            "input_multiplier": INPUT_UPDATE_MULTIPLIER,
            "output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
            "source": "foldable_multihash_update_audit_v4_projection",
            "quality_metric_used": False,
        }
        or set(plan.get("assignment_audits", {}))
        != {"stratified_generic_shuffle", "balanced_random_multihash"}
        or set(plan.get("initialization", {}))
        != {
            "source_role",
            "step_zero_checkpoint_path",
            "step_zero_checkpoint_artifact_sha256",
            "step_zero_checkpoint_state_sha256",
            "update_matched_dense_state_sha256",
            "stratified_generic_shuffle_state_sha256",
            "balanced_random_multihash_state_sha256",
        }
        or plan["initialization"].get("source_role")
        != "untied_generic_surface"
        or any(
            not isinstance(plan["initialization"].get(key), str)
            or len(plan["initialization"][key]) != 64
            for key in (
                "step_zero_checkpoint_artifact_sha256",
                "step_zero_checkpoint_state_sha256",
                "update_matched_dense_state_sha256",
                "stratified_generic_shuffle_state_sha256",
                "balanced_random_multihash_state_sha256",
            )
        )
        or plan.get("selection_rule")
        != {
            "primary_candidate": "untied_generic_surface",
            "primary_control": "update_matched_dense",
            "minimum_advantage_bpb": 0.002,
            "maximum_anchor_gap_bpb": 0.05,
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 20_260_836,
            "random_role_fallback": None,
            "surface_support_requires_both_random_controls": True,
        }
        or plan.get("claim_boundary")
        != {
            "actual_inference_evidence": False,
            "development_data": True,
            "model_seed_count": 1,
            "publication_claim": False,
            "fresh_stage_requires_primary_gate": True,
        }
        or not isinstance(plan.get("implementation_sha256"), Mapping)
        or set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS)
    ):
        raise ValueError("mechanism plan contract differs")
    unsigned = dict(plan)
    receipt = unsigned.pop("plan_sha256", None)
    if canonical_sha256(unsigned) != receipt:
        raise ValueError("mechanism plan hash differs")
    if verify_derived and (
        plan.get("dependencies") != dependency_identity()
        or plan.get("environment") != current_environment()
        or len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS))
        or plan.get("implementation_sha256")
        != {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}
    ):
        raise ValueError("mechanism plan derived identity differs")
