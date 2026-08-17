#!/usr/bin/env python3
"""Seal the fixed Hangul-boundary retrieval mechanism hypothesis."""

from __future__ import annotations

import os
import subprocess

from fresh_vocabulary_16k_retrieval_mechanism_protocol import (
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    build_plan,
    json_bytes,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("retrieval mechanism plan sealing requires a clean worktree")
    if PLAN_PATH.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("retrieval mechanism namespace is not empty")
    if _git("log", "--all", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()):
        raise FileExistsError("retrieval mechanism plan was already published")
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during retrieval mechanism plan sealing")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PLAN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(plan))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"plan_sha256={plan['plan_sha256']}")
    print("commit the plan before replaying the retrieval mechanism")


if __name__ == "__main__":
    main()
