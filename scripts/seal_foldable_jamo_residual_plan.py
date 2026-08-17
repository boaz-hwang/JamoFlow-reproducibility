#!/usr/bin/env python3
"""Seal the six-role foldable-Jamo residual plan before any model loss."""

from __future__ import annotations

import os
import subprocess

from compositional_head_preflight_protocol import tokenizer_identity
from compositional_quality_protocol import inherited_inventories
from foldable_jamo_residual_core import (
    RESIDUAL_ROLES,
    expected_parameter_counts,
    role_definition,
)
from foldable_jamo_residual_protocol import (
    OUTPUT_PATH,
    PARENT_RESOURCE_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    baseline_control_identities,
    canonical_sha256,
    current_environment,
    dependency_identity,
    document_identity,
    hash_file,
    implementation_identity,
    initialization_identities,
    json_bytes,
    parent_anchor,
    read_json,
    selection_rule,
    training_contract,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _never_published(path) -> None:
    history = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if path.exists() or history:
        raise RuntimeError(f"foldable residual artifact already exists or has history: {path}")


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("foldable residual plan requires a clean worktree")
    for path in (PLAN_PATH, REPORT_PATH, OUTPUT_PATH):
        _never_published(path)
    base_commit = _git("rev-parse", "HEAD")
    identities, assignments, exposure = initialization_identities()
    common, documents = document_identity()
    tokenizers = tokenizer_identity()
    inventories = inherited_inventories()
    resource = read_json(PARENT_RESOURCE_PATH)
    plan = {
        "schema_version": 1,
        "kind": "foldable_jamo_residual_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "same_cost_assignments_and_zero_initial_states_sealed_before_loss",
        "git_commit_before_plan": base_commit,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "tokenizers": {key: tokenizers[key] for key in ("2048", "8192")},
        "roles": list(RESIDUAL_ROLES),
        "role_specs": {
            role: {**role_definition(role), **expected_parameter_counts(role)}
            for role in RESIDUAL_ROLES
        },
        "inventories": {"8192": inventories["8192"]},
        "document_common": common,
        "document_inventory": documents,
        "parent_anchor": parent_anchor(),
        "baseline_controls": baseline_control_identities(),
        "exposure_identity": exposure,
        "assignment_audits": assignments,
        "initialization_identities": identities,
        "training": training_contract(),
        "resource_authorization": {
            "artifact_sha256": hash_file(PARENT_RESOURCE_PATH),
            "deployed_activation_geometry_matches_dense_8k": True,
            "deployed_parameter_count_by_role": {
                role: expected_parameter_counts(role)["deployed"]
                for role in RESIDUAL_ROLES
            },
            "parent_projection_pass": resource["projection"]["passes"],
            "training_parameter_count_by_role": {
                role: expected_parameter_counts(role)["training_total"]
                for role in RESIDUAL_ROLES
            },
        },
        "selection_rule": selection_rule(),
        "claim_boundary": {
            "actual_inference_measured": False,
            "calibration_development_only": True,
            "deployed_residual_module_present": False,
            "fresh_equal_history_quality": False,
            "korean_specific_method_screen": True,
            "model_seed_count": 1,
            "publication_quality_claim": False,
            "training_overhead_is_measured_not_free": True,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_plan(plan, verify_derived=False)
    if _git("rev-parse", "HEAD") != base_commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing foldable residual plan")
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"sealed={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
