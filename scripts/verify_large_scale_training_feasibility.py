#!/usr/bin/env python3
"""Read-only reconstruction of large-scale training feasibility evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from large_scale_training_feasibility_core import (
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    build_summary,
    validate_plan,
    validate_summary,
    validate_worker_report,
    worker_id,
    worker_order,
    worker_report_path,
)

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("large-scale training verifier requires a clean worktree")
    plan = _read_json(PLAN_PATH)
    summary = _read_json(OUTPUT_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_summary(summary)
    commit = summary["summary_base_git_commit"]
    plan_artifact_sha256 = hash_file(PLAN_PATH)
    reports: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    for target, regime, role in worker_order():
        identifier = worker_id(target, regime, role)
        path = worker_report_path(target, regime, role)
        report = _read_json(path)
        validate_worker_report(
            report,
            plan=plan,
            plan_artifact_sha256=plan_artifact_sha256,
            runner_git_commit=commit,
            target=target,
            regime=regime,
            role=role,
        )
        reports[identifier] = report
        evidence[identifier] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
    rebuilt = build_summary(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        reports=reports,
    )
    if rebuilt != summary:
        raise ValueError("large-scale training summary does not reconstruct")
    print("large_scale_training_feasibility_verification=pass")
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
