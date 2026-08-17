#!/usr/bin/env python3
"""Verify the W80 resource preflight without running an optimizer step."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from balanced_200m_w80_core import (
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    ROOT,
    build_preflight_summary,
    validate_plan,
    validate_preflight_summary,
    worker_preflight_path,
)
from run_balanced_200m_w80_preflight import validate_worker

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 preflight verifier requires clean worktree")
    plan = _read(PLAN_PATH)
    summary = _read(PREFLIGHT_OUTPUT_PATH)
    report = _read(worker_preflight_path())
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_preflight_summary(summary)
    commit = summary["summary_base_git_commit"]
    validate_worker(report, plan=plan, commit=commit)
    rebuilt = build_preflight_summary(
        plan=plan,
        plan_artifact_sha256=hash_file(PLAN_PATH),
        summary_base_git_commit=commit,
        worker_path=worker_preflight_path().relative_to(ROOT).as_posix(),
        worker_sha256=hash_file(worker_preflight_path()),
        report=report,
    )
    if rebuilt != summary:
        raise ValueError("balanced-200M W80 preflight summary does not reconstruct")
    print("balanced_200m_w80_preflight_verification=pass")


if __name__ == "__main__":
    main()

