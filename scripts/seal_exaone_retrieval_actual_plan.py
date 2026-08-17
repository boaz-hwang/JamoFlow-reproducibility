#!/usr/bin/env python3
"""Seal the EXAONE 7.8B retrieval actual-inference plan."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from exaone_resource_calibration import RESULT_PATH as RESOURCE_RESULT_PATH
from exaone_retrieval_actual import (
    ARTIFACT_ROOT,
    PLAN_PATH,
    SESSION_RECEIPT_ROOT,
    SESSIONS,
    SUMMARY_PATH,
    assert_canonical_workspace_path,
    build_plan,
    session_receipt_path,
    validate_plan,
)
from exaone_retrieval_data import ROOT, canonical_bytes


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_never_published(path: Path) -> None:
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(
            f"artifact was already published: {path.relative_to(ROOT)}"
        )


def _require_exact_head_blob(path: Path) -> None:
    payload = subprocess.check_output(
        ("git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"), cwd=ROOT
    )
    if payload != path.read_bytes():
        raise ValueError(
            f"artifact is not the exact HEAD blob: {path.relative_to(ROOT)}"
        )


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE actual plan requires a clean worktree")
    for path in (ARTIFACT_ROOT, SESSION_RECEIPT_ROOT, PLAN_PATH, SUMMARY_PATH):
        assert_canonical_workspace_path(path)
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.rglob("*")):
        raise FileExistsError("EXAONE actual ignored namespace is not empty")
    if SESSION_RECEIPT_ROOT.exists() and any(SESSION_RECEIPT_ROOT.rglob("*")):
        raise FileExistsError("EXAONE actual receipt namespace is not empty")
    current_paths = [PLAN_PATH, SUMMARY_PATH]
    current_paths.extend(session_receipt_path(index) for index in range(SESSIONS))
    if any(path.exists() for path in current_paths):
        raise FileExistsError("EXAONE actual tracked namespace is not empty")
    for path in current_paths:
        _require_never_published(path)
    _require_exact_head_blob(RESOURCE_RESULT_PATH)

    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan, verify_derived=True)
    if commit != _git("rev-parse", "HEAD") or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during EXAONE actual plan sealing")
    _publish(PLAN_PATH, canonical_bytes(plan))
    print(f"plan_sha256={plan['plan_sha256']}")
    print("commit the actual plan before the first retrieval-table candidate execution")


if __name__ == "__main__":
    main()
