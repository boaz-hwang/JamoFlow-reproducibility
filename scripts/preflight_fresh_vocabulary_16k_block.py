#!/usr/bin/env python3
"""Run a result- and timing-silent target-block MPS feasibility check."""

from __future__ import annotations

import subprocess

from benchmark_fresh_vocabulary_16k_block import preflight_target
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


def _history(path) -> str:
    return _git(
        "log",
        "--all",
        "--format=%H",
        "--",
        path.relative_to(ROOT).as_posix(),
    )


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("16K target-block preflight requires a clean worktree")
    if PLAN_PATH.exists() or OUTPUT_PATH.exists() or ACTIVE_PATH.exists():
        raise FileExistsError("16K target-block evidence namespace is not empty")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise FileExistsError("16K target-block artifact namespace is not empty")
    if _history(PLAN_PATH) or _history(OUTPUT_PATH):
        raise FileExistsError("16K target-block evidence has prior Git history")
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(git_commit_before_plan=commit)
    validate_plan(plan, verify_derived=True)
    preflight_target(plan)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during 16K target-block preflight")
    print("status=finite_loss_and_timing_silent_target_block_mps")


if __name__ == "__main__":
    main()
