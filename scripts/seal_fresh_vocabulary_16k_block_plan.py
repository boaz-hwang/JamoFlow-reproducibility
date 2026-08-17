#!/usr/bin/env python3
"""Seal trained 16K perfect-draft target-block inputs before timing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fresh_vocabulary_16k_actual_protocol import json_bytes
from fresh_vocabulary_16k_block_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    build_plan,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _git(
        "log",
        "--all",
        "--format=%H",
        "--",
        path.relative_to(ROOT).as_posix(),
    )
    if history:
        raise FileExistsError(f"16K target-block artifact has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("16K target-block plan sealing requires a clean worktree")
    _require_never_published(PLAN_PATH)
    _require_never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists() or (
        ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir())
    ):
        raise FileExistsError("16K target-block artifact namespace is not empty")
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan, verify_derived=True)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while sealing target-block plan")
    _publish(PLAN_PATH, json_bytes(plan))
    expected = f"?? {PLAN_PATH.relative_to(ROOT).as_posix()}"
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all") != expected
    ):
        raise RuntimeError("16K target-block plan changed unexpected paths")
    print("status=sealed_before_16k_target_block_timing")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
