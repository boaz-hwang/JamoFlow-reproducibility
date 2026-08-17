#!/usr/bin/env python3
"""Read-only reconstruction of the balanced-200M batch-8 preflight."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from balanced_200m_trained_core import (
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    ROLE_ORDER,
    ROOT,
    build_preflight_summary,
    validate_plan,
    validate_preflight_summary,
    worker_report_path,
)
from run_balanced_200m_preflight import _validate_report

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object required")
    return value


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M verifier requires clean worktree")
    plan = _read(PLAN_PATH)
    summary = _read(PREFLIGHT_OUTPUT_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_preflight_summary(summary)
    commit = summary["summary_base_git_commit"]
    reports: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for role in ROLE_ORDER:
        path = worker_report_path(role)
        report = _read(path)
        _validate_report(report, role=role, plan=plan, commit=commit)
        reports[role] = report
        evidence[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
    rebuilt = build_preflight_summary(
        plan=plan,
        plan_artifact_sha256=hash_file(PLAN_PATH),
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        reports=reports,
    )
    if rebuilt != summary:
        raise ValueError("balanced-200M preflight does not reconstruct")
    print("balanced_200m_preflight_verification=pass")
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
