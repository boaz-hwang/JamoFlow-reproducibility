#!/usr/bin/env python3
"""Seal the fresh one-seed vocabulary-adaptation plan before model training."""

from __future__ import annotations

import os
import subprocess

from fresh_vocabulary_adaptation_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    REPORT_PATH,
    ROOT,
    build_plan,
    json_bytes,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-adaptation plan sealing requires a clean worktree")
    if any(
        path.exists() for path in (PLAN_PATH, ACTIVE_PATH, REPORT_PATH, OUTPUT_PATH)
    ):
        raise RuntimeError(
            "fresh-adaptation plan or downstream evidence already exists"
        )
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.rglob("*")):
        raise RuntimeError("fresh-adaptation artifact namespace is not empty")
    if _git("log", "--all", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))):
        raise RuntimeError("fresh-adaptation plan was already published")
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(commit)
    validate_plan(plan, verify_derived=True)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed while deriving fresh-adaptation plan")
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"status={plan['status']}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
