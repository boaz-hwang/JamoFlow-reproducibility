#!/usr/bin/env python3
"""Seal the nine-role strong vocabulary-transfer baseline closure."""

from __future__ import annotations

import os
import subprocess

from compositional_head_preflight_protocol import tokenizer_identity
from compositional_quality_protocol import inherited_inventories
from vocabulary_transfer_baseline_core import (
    BASELINE_ROLES,
    expected_parameter_count,
    role_definition,
)
from vocabulary_transfer_baseline_protocol import (
    OUTPUT_PATH,
    PARENT_RESOURCE_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    canonical_sha256,
    current_environment,
    dependency_identity,
    hash_file,
    implementation_identity,
    initialization_identities,
    json_bytes,
    parent_anchor,
    previous_probe_evidence,
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
        raise RuntimeError(f"baseline artifact already exists or has history: {path}")


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("baseline plan requires a clean worktree")
    for path in (PLAN_PATH, REPORT_PATH, OUTPUT_PATH):
        _never_published(path)
    base_commit = _git("rev-parse", "HEAD")
    audits, initial_states, metadata = initialization_identities()
    tokenizers = tokenizer_identity()
    inventories = inherited_inventories()
    resource = read_json(PARENT_RESOURCE_PATH)
    plan = {
        "schema_version": 1,
        "kind": "vocabulary_transfer_baseline_closure_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "strong_generic_roles_and_initial_states_sealed_before_loss",
        "git_commit_before_plan": base_commit,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "tokenizers": {key: tokenizers[key] for key in ("2048", "8192")},
        "roles": list(BASELINE_ROLES),
        "role_specs": {
            role: {
                **role_definition(role),
                "expected_parameters": expected_parameter_count(role),
            }
            for role in BASELINE_ROLES
        },
        "inventories": {"8192": inventories["8192"]},
        "parent_anchor": parent_anchor(),
        "previous_probe_evidence": previous_probe_evidence(),
        "source_token_metadata": metadata,
        "initialization_audits": audits,
        "initial_state_sha256": initial_states,
        "training": training_contract(),
        "resource_authorization": {
            "artifact_sha256": hash_file(PARENT_RESOURCE_PATH),
            "activation_geometry_matches_prior_8k_runs": True,
            "parameter_count_by_role": {
                role: expected_parameter_count(role) for role in BASELINE_ROLES
            },
            "parent_projection_pass": resource["projection"]["passes"],
        },
        "selection_rule": selection_rule(),
        "claim_boundary": {
            "actual_inference_measured": False,
            "calibration_development_only": True,
            "eeve_full_seven_stage_reproduced": False,
            "in_place_token_scale_reproduced": False,
            "korean_specific_method_evaluated": False,
            "model_seed_count": 1,
            "publication_quality_claim": False,
            "strong_generic_baseline_closure_only": True,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    if _git("rev-parse", "HEAD") != base_commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing baseline plan")
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"sealed={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()

