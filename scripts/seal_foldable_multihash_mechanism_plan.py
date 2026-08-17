#!/usr/bin/env python3
"""Seal the three-role foldable multi-hash mechanism-control screen."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from foldable_jamo_residual_core import RESIDUAL_SLOT_COUNT
from foldable_jamo_residual_protocol import (
    canonical_sha256 as parent_canonical_sha256,
)
from foldable_jamo_residual_protocol import (
    current_environment,
    hash_file,
    read_json,
    target_order,
    training_contract,
)
from foldable_multihash_mechanism_core import (
    INPUT_UPDATE_MULTIPLIER,
    NEW_ROLES,
    OUTPUT_UPDATE_MULTIPLIER,
    PROTOCOL_ID,
    assignment_audit,
    balanced_random_assignment,
    generic_assignment_from_code_indices,
    stratified_generic_shuffle,
)
from foldable_multihash_mechanism_protocol import (
    AUDIT_PLAN_PATH,
    AUDIT_RESULT_PATH,
    IMPLEMENTATION_PATHS,
    OUTPUT_PATH,
    PARENT_PLAN_PATH,
    PARENT_RESULT_PATH,
    PLAN_PATH,
    ROOT,
    canonical_sha256,
    dependency_identity,
    json_bytes,
    role_definition,
    validate_plan,
)
from run_foldable_jamo_residual import (
    _cleanup_data,
    _role_data,
    _scheduled_exposure_counts,
)
from run_foldable_jamo_residual import (
    _paths as parent_paths,
)
from run_foldable_jamo_residual import (
    _validate_worker as validate_parent_worker,
)
from run_foldable_multihash_update_audit import (
    _validate_parent_plan_for_historical_replay,
)
from run_foldable_multihash_update_audit import _validate_plan as validate_audit_plan
from vocabulary_transfer_baseline_core import state_mapping_sha256


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _history(path: Path) -> tuple[str, ...]:
    output = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    return tuple(line for line in output.splitlines() if line)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_result(value: Mapping[str, Any], *, kind: str, receipt_key: str) -> None:
    unsigned = dict(value)
    receipt = unsigned.pop(receipt_key, None)
    if value.get("kind") != kind or canonical_sha256(unsigned) != receipt:
        raise ValueError(f"mechanism parent result differs: {kind}")


def _state_with_assignment_sha256(
    state: Mapping[str, torch.Tensor], assignment: np.ndarray
) -> str:
    offsets = torch.arange(RESIDUAL_SLOT_COUNT, dtype=torch.long) * 128
    codes = torch.from_numpy(assignment.copy()).long() + offsets.unsqueeze(0)
    output = dict(state)
    output["foldable_residual.code_indices"] = codes
    return state_mapping_sha256(output)


def build_plan() -> dict[str, Any]:
    parent_plan = read_json(PARENT_PLAN_PATH)
    _validate_parent_plan_for_historical_replay(parent_plan)
    parent_result = read_json(PARENT_RESULT_PATH)
    _validate_result(
        parent_result,
        kind="foldable_jamo_residual_result_v1",
        receipt_key="summary_sha256",
    )
    parent_commit = parent_result["git_commit"]
    if not validate_parent_worker(
        "untied_generic_surface", parent_commit, parent_plan
    ):
        raise RuntimeError("mechanism generic parent worker is absent")

    audit_plan = read_json(AUDIT_PLAN_PATH)
    validate_audit_plan(audit_plan)
    audit_result = read_json(AUDIT_RESULT_PATH)
    _validate_result(
        audit_result,
        kind="foldable_multihash_update_audit_summary_v4",
        receipt_key="summary_sha256",
    )
    if (
        audit_result.get("plan_artifact_sha256") != hash_file(AUDIT_PLAN_PATH)
        or audit_result.get("plan_payload_sha256") != audit_plan["plan_sha256"]
        or audit_result.get("evidence", {}).get("selected_control")
        != {
            "control_kind": "post_adamw_new_row_update_projection_v1",
            "diagnostics": audit_result["evidence"]["selected_control"]["diagnostics"],
            "input_multiplier": INPUT_UPDATE_MULTIPLIER,
            "output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
            "quality_metric_used": False,
            "source": "fixed_first_training_batch_effective_update_projection",
        }
    ):
        raise RuntimeError("mechanism update-control lineage differs")

    worker = read_json(parent_paths("untied_generic_surface")[0])
    step_zero = worker["checkpoints"]["0"]
    step_zero_path = ROOT / step_zero["checkpoint_path"]
    state = torch.load(step_zero_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, Mapping)
        or hash_file(step_zero_path) != step_zero["checkpoint_artifact_sha256"]
        or state_mapping_sha256(state) != step_zero["checkpoint_state_sha256"]
    ):
        raise RuntimeError("mechanism step-zero state differs")
    generic = generic_assignment_from_code_indices(
        state["foldable_residual.code_indices"]
    )

    data = _role_data(parent_plan)
    try:
        train_count = int(data["train_inventory"].full_sequence_count)
        train_sequences = data["train_memory"][: train_count * 512].reshape(
            train_count, 512
        )
        exposure = _scheduled_exposure_counts(
            train_sequences, target_order(train_count)
        )
        shuffled, shuffled_construction = stratified_generic_shuffle(
            generic, data["token_bytes"], exposure
        )
        balanced, balanced_construction = balanced_random_assignment(generic)
        assignments = {
            "stratified_generic_shuffle": assignment_audit(
                shuffled,
                generic,
                exposure,
                kind="stratified_generic_shuffle",
                construction=shuffled_construction,
            ),
            "balanced_random_multihash": assignment_audit(
                balanced,
                generic,
                exposure,
                kind="balanced_random_multihash",
                construction=balanced_construction,
            ),
        }
    finally:
        _cleanup_data(data)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "foldable_multihash_mechanism_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "development_mechanism_controls_sealed_before_new_training",
        "git_commit_before_plan": _git("rev-parse", "HEAD"),
        "dependencies": dependency_identity(),
        "roles": {role: role_definition(role) for role in NEW_ROLES},
        "assignment_audits": assignments,
        "initialization": {
            "source_role": "untied_generic_surface",
            "step_zero_checkpoint_path": step_zero["checkpoint_path"],
            "step_zero_checkpoint_artifact_sha256": step_zero[
                "checkpoint_artifact_sha256"
            ],
            "step_zero_checkpoint_state_sha256": step_zero[
                "checkpoint_state_sha256"
            ],
            "update_matched_dense_state_sha256": worker["initialization_identity"][
                "folded_dense_state_sha256"
            ],
            "stratified_generic_shuffle_state_sha256": _state_with_assignment_sha256(
                state, shuffled
            ),
            "balanced_random_multihash_state_sha256": _state_with_assignment_sha256(
                state, balanced
            ),
        },
        "update_control": {
            "input_multiplier": INPUT_UPDATE_MULTIPLIER,
            "output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
            "source": "foldable_multihash_update_audit_v4_projection",
            "quality_metric_used": False,
        },
        "training": training_contract(),
        "inventories": parent_plan["inventories"],
        "document_common": parent_plan["document_common"],
        "selection_rule": {
            "primary_candidate": "untied_generic_surface",
            "primary_control": "update_matched_dense",
            "minimum_advantage_bpb": 0.002,
            "maximum_anchor_gap_bpb": 0.05,
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 20_260_836,
            "random_role_fallback": None,
            "surface_support_requires_both_random_controls": True,
        },
        "environment": current_environment(),
        "implementation_sha256": {
            path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS
        },
        "claim_boundary": {
            "actual_inference_evidence": False,
            "development_data": True,
            "model_seed_count": 1,
            "publication_claim": False,
            "fresh_stage_requires_primary_gate": True,
        },
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
    }
    payload["plan_sha256"] = parent_canonical_sha256(payload)
    return payload


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("mechanism plan requires a clean worktree")
    if (
        PLAN_PATH.exists()
        or OUTPUT_PATH.exists()
        or _history(PLAN_PATH)
        or _history(OUTPUT_PATH)
    ):
        raise RuntimeError("mechanism plan or result was already published")
    head = _git("rev-parse", "HEAD")
    plan = build_plan()
    validate_plan(plan, verify_derived=True)
    if _git("rev-parse", "HEAD") != head or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing mechanism plan")
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"plan_path={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
