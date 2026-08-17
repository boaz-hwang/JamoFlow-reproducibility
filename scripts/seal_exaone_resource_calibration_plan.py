#!/usr/bin/env python3
"""Seal the baseline-only EXAONE resource-calibration plan."""

from __future__ import annotations

import os
import subprocess

from exaone_resource_calibration import (
    ACTIVE_PATH,
    BASELINE_ARTIFACT_PATH,
    PLAN_PATH,
    RESULT_PATH,
    ROOT,
    build_plan,
    validate_plan,
)
from exaone_retrieval_data import (
    VERIFICATION_PATH as DATA_VERIFICATION_PATH,
)
from exaone_retrieval_data import (
    canonical_bytes,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path) -> None:
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(
            f"artifact was already published: {path.relative_to(ROOT)}"
        )


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE resource plan requires a clean worktree")
    for path in (PLAN_PATH, RESULT_PATH, ACTIVE_PATH, BASELINE_ARTIFACT_PATH):
        if path.exists():
            raise FileExistsError(f"EXAONE resource namespace is not empty: {path}")
    for path in (PLAN_PATH, RESULT_PATH):
        _require_never_published(path)
    verification_blob = subprocess.check_output(
        (
            "git",
            "show",
            f"HEAD:{DATA_VERIFICATION_PATH.relative_to(ROOT).as_posix()}",
        ),
        cwd=ROOT,
    )
    if verification_blob != DATA_VERIFICATION_PATH.read_bytes():
        raise ValueError("EXAONE data verification is not the exact HEAD blob")
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan, verify_derived=True)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during EXAONE resource plan sealing")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PLAN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_bytes(plan))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"plan_sha256={plan['plan_sha256']}")
    print("commit the resource plan before baseline timing")


if __name__ == "__main__":
    main()
