#!/usr/bin/env python3
"""Seal trained vocabulary actual-inference inputs before timing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fresh_vocabulary_actual_protocol import (
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    build_plan,
    json_bytes,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    if history:
        raise FileExistsError(f"fresh actual artifact has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh actual plan sealing requires a clean worktree")
    _require_never_published(PLAN_PATH)
    _require_never_published(OUTPUT_PATH)
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan, verify_derived=True)
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise RuntimeError("repository changed while sealing fresh actual plan")
    _publish(PLAN_PATH, json_bytes(plan))
    expected_status = f"?? {PLAN_PATH.relative_to(ROOT).as_posix()}"
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all") != expected_status
    ):
        raise RuntimeError("fresh actual plan publication changed unexpected paths")
    print("status=sealed_before_trained_actual_timing")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
