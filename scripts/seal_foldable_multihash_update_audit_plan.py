#!/usr/bin/env python3
"""Seal the model-quality-free foldable multi-hash first-update audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
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
    target_order,
    training_contract,
)
from foldable_jamo_residual_protocol import (
    WORKER_ROOT as PARENT_WORKER_ROOT,
)
from foldable_multihash_update_audit_core import PROTOCOL_ID, array_sha256

PLAN_PATH = ROOT / "data/manifests/foldable-multihash-update-audit-v4.json"
RESULT_PATH = ROOT / "results/foldable-multihash-update-audit-v4/summary.json"
ROLE = "untied_generic_surface"
WORKER_PATH = PARENT_WORKER_ROOT / f"{ROLE}.json"
IMPLEMENTATION_PATHS = (
    "docs/141-foldable-jamo-residual-result-and-multihash-pivot.md",
    "docs/142-fable5-final-retrospective-and-current-direction.md",
    "docs/143-foldable-vocabulary-reparameterization-literature-audit.md",
    "docs/144-foldable-multihash-update-audit-protocol.md",
    "docs/145-foldable-multihash-update-audit-v1-invalidation-and-v2-correction.md",
    "docs/146-foldable-multihash-update-audit-v2-invalidation-and-v3-correction.md",
    "docs/147-foldable-multihash-update-audit-v3-invalidation-and-v4-correction.md",
    "pyproject.toml",
    "scripts/foldable_jamo_residual_core.py",
    "scripts/foldable_jamo_residual_protocol.py",
    "scripts/foldable_multihash_update_audit_core.py",
    "scripts/run_foldable_jamo_residual.py",
    "scripts/run_foldable_multihash_update_audit.py",
    "scripts/run_vocabulary_transfer_baseline.py",
    "scripts/seal_foldable_multihash_update_audit_plan.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "tests/test_foldable_multihash_update_audit.py",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


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


def build_plan() -> dict[str, Any]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise RuntimeError("update-audit implementation list is duplicated")
    parent_plan = read_json(PARENT_PLAN_PATH)
    parent_result = read_json(PARENT_RESULT_PATH)
    worker = read_json(WORKER_PATH)
    checkpoint = worker["checkpoints"]["0"]
    sequence_count = int(training_contract()["sequence_count"])
    first_batch = target_order(sequence_count)[:32]
    if first_batch.dtype != np.int64 or first_batch.shape != (32,):
        raise RuntimeError("update-audit first-batch coordinates differ")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "foldable_multihash_update_audit_plan_v4",
        "protocol_id": PROTOCOL_ID,
        "status": "first_update_geometry_sealed_before_update_observation",
        "git_commit_before_plan": _git("rev-parse", "HEAD"),
        "role": ROLE,
        "parent": {
            "plan_path": str(PARENT_PLAN_PATH.relative_to(ROOT)),
            "plan_artifact_sha256": hash_file(PARENT_PLAN_PATH),
            "plan_payload_sha256": parent_plan["plan_sha256"],
            "result_path": str(PARENT_RESULT_PATH.relative_to(ROOT)),
            "result_artifact_sha256": hash_file(PARENT_RESULT_PATH),
            "result_payload_sha256": parent_result["summary_sha256"],
            "worker_path": str(WORKER_PATH.relative_to(ROOT)),
            "worker_artifact_sha256": hash_file(WORKER_PATH),
            "worker_payload_sha256": worker["worker_sha256"],
            "step_zero_checkpoint_path": checkpoint["checkpoint_path"],
            "step_zero_checkpoint_artifact_sha256": checkpoint[
                "checkpoint_artifact_sha256"
            ],
            "step_zero_checkpoint_state_sha256": checkpoint["checkpoint_state_sha256"],
        },
        "batch": {
            "sequence_count": sequence_count,
            "effective_batch_size": 32,
            "first_batch_indices_sha256": array_sha256(first_batch),
            "training_order_prefix_sha256": training_contract()[
                "training_order_prefix_sha256"
            ],
        },
        "optimizer": training_contract(),
        "metric": {
            "matrices": ["input", "output"],
            "row_scope": "new_target_rows_2048_through_8191",
            "projection": "dot(multihash,dense)/dot(dense,dense)",
            "control_multiplier_source": "projection_multiplier",
            "control_multiplier_open_interval": [1.0, 16.0],
            "collision_alignment_row_scope": (
                "nonzero_direct_lexical_gradient_rows_per_matrix"
            ),
            "zero_gradient_rows_reported": True,
            "quality_metric_used": False,
        },
        "environment": current_environment(),
        "implementation_sha256": {
            path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS
        },
        "claim_boundary": {
            "actual_inference_evidence": False,
            "development_training_batch": True,
            "model_quality_evidence": False,
            "optimizer_mechanism_audit_only": True,
            "publication_claim": False,
        },
        "output_path": str(RESULT_PATH.relative_to(ROOT)),
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("update-audit plan requires a clean worktree")
    if (
        PLAN_PATH.exists()
        or RESULT_PATH.exists()
        or _history(PLAN_PATH)
        or _history(RESULT_PATH)
    ):
        raise RuntimeError("update-audit plan or result was already published")
    head = _git("rev-parse", "HEAD")
    plan = build_plan()
    if _git("rev-parse", "HEAD") != head or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing update-audit plan")
    _publish(PLAN_PATH, _json_bytes(plan))
    print(f"plan_path={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
