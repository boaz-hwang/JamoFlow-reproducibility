#!/usr/bin/env python3
"""Seal the selected 8K one-seed compositional quality experiment."""

from __future__ import annotations

import os
import subprocess

from compositional_quality_core import QUALITY_ROLES
from compositional_quality_protocol import (
    BPE_QUALITY_PLAN_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    RESOURCE_REPORT_PATH,
    ROOT,
    SYSTEMS_RESULT_PATH,
    assignment_audits,
    canonical_sha256,
    current_environment,
    dependency_identity,
    implementation_identity,
    inherited_inventories,
    initial_state_identity,
    json_bytes,
    read_json,
    resource_contract,
    selection_rule,
    tokenizer_identity,
    training_contracts,
    validate_plan,
    QUALITY_SPECS,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _never_published(path) -> None:
    history = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if path.exists() or history:
        raise RuntimeError(f"compositional quality artifact already exists or has history: {path}")


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compositional quality plan requires a clean worktree")
    for path in (PLAN_PATH, RESOURCE_REPORT_PATH, REPORT_PATH, OUTPUT_PATH):
        _never_published(path)
    systems = read_json(SYSTEMS_RESULT_PATH)
    if (
        systems.get("complete") is not True
        or systems.get("decision", {}).get("status")
        != "compositional_head_systems_opportunity_pass"
        or systems["decision"].get("selected_vocabulary_size") != 8_192
        or systems["decision"].get("selected_candidate_role")
        != "hangul_code_v8192"
    ):
        raise RuntimeError("compositional quality systems authorization differs")
    base_commit = _git("rev-parse", "HEAD")
    initial_states, body_state = initial_state_identity()
    bpe_plan = read_json(BPE_QUALITY_PLAN_PATH)
    tokenizers = tokenizer_identity()
    plan = {
        "schema_version": 1,
        "kind": "compositional_head_quality_one_seed_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_resource_probe_training_or_quality",
        "git_commit_before_plan": base_commit,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "tokenizers": {key: tokenizers[key] for key in ("2048", "8192")},
        "roles": list(QUALITY_ROLES),
        "model_specs": {
            role: QUALITY_SPECS[role].to_dict() for role in QUALITY_ROLES
        },
        "initial_state_sha256": initial_states,
        "transformer_body_initial_state_sha256": body_state,
        "assignment_audits": assignment_audits(),
        "inventories": inherited_inventories(),
        "document_common": bpe_plan["document_evaluation"]["common"],
        "training": training_contracts(),
        "resource_contract": resource_contract(),
        "selection_rule": selection_rule(),
        "known_engineering_smoke": {
            "status": "hangul_8k_single_microbatch_forward_backward_only",
            "microbatch": 8,
            "sequence_length": 512,
            "finite": True,
            "elapsed_seconds_observed": True,
            "loss_value_recorded": False,
            "used_to_change_quality_gate_or_roles": False,
        },
        "claim_boundary": {
            "actual_mps_training": True,
            "calibration_development_only": True,
            "model_seed_count": 1,
            "publication_quality_claim": False,
            "random_weight_systems_result_already_known": True,
            "resource_gate_uses_no_recorded_loss_value": True,
            "trained_actual_inference_measured_in_this_stage": False,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    if _git("rev-parse", "HEAD") != base_commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing compositional quality plan")
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"sealed={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
