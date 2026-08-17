#!/usr/bin/env python3
"""Seal the metric-free EXAONE table/case plan before tokenization."""

from __future__ import annotations

import os
import subprocess

from exaone_retrieval_data import (
    ACTIVE_PATH,
    CASES_PATH,
    PLAN_PATH,
    ROOT,
    SEAL_PATH,
    TABLE_PATH,
    VERIFICATION_ACTIVE_PATH,
    VERIFICATION_PATH,
    build_plan,
    canonical_bytes,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path) -> None:
    history = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    if history:
        raise FileExistsError(f"artifact was already published: {path.relative_to(ROOT)}")


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE retrieval data plan requires a clean worktree")
    for path in (
        PLAN_PATH,
        SEAL_PATH,
        VERIFICATION_PATH,
        ACTIVE_PATH,
        VERIFICATION_ACTIVE_PATH,
        TABLE_PATH,
        CASES_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"EXAONE retrieval data namespace is not empty: {path}")
    for path in (PLAN_PATH, SEAL_PATH, VERIFICATION_PATH):
        _require_never_published(path)
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan, verify_derived=True)
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise RuntimeError("repository changed during EXAONE retrieval data plan sealing")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PLAN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_bytes(plan))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"plan_sha256={plan['plan_sha256']}")
    print("commit the data plan before tokenizer table or case construction")


if __name__ == "__main__":
    main()
