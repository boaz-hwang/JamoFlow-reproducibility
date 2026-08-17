#!/usr/bin/env python3
"""Seal the constant-budget compositional-head systems preflight."""

from __future__ import annotations

import os
import subprocess

from compositional_head_preflight_protocol import (
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    RESULT_PATH,
    ROOT,
    assignment_audits,
    canonical_sha256,
    case_identity,
    current_environment,
    dependency_identity,
    experiment_contract,
    implementation_identity,
    json_bytes,
    model_contract,
    tokenizer_identity,
    TIMING_PATH,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _never_published(path) -> None:
    if path.exists() or _git(
        "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT))
    ):
        raise RuntimeError(f"compositional-head artifact already exists or has history: {path}")


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compositional-head plan requires a clean worktree")
    _never_published(PLAN_PATH)
    _never_published(REPORT_PATH)
    _never_published(TIMING_PATH)
    _never_published(RESULT_PATH)
    base_commit = _git("rev-parse", "HEAD")
    plan = {
        "schema_version": 2,
        "kind": "compositional_head_systems_preflight_plan_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_full_grid_timing",
        "git_commit_before_plan": base_commit,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "tokenizers": tokenizer_identity(),
        "assignment_audits": assignment_audits(),
        "cases": case_identity(),
        "experiment": experiment_contract(),
        "model_contract": model_contract(),
        "known_engineering_smoke": {
            "status": "all_role_construction_and_one_step_forward_smoke_only",
            "roles_exercised": experiment_contract()["role_order"],
            "observed_before_seal": True,
            "used_to_change_role_grid_or_gate": False,
            "metrics_retained_as_evidence": False,
        },
        "claim_boundary": {
            "calibration_development_cases": True,
            "full_grid_random_weight_actual_timing": True,
            "trained_model_quality": False,
            "publication_latency": False,
            "korean_specific_quality_contribution": False,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    validate_plan(plan)
    if _git("rev-parse", "HEAD") != base_commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing compositional-head plan")
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"sealed={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
