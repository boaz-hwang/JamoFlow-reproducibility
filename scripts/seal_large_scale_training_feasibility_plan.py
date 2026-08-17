#!/usr/bin/env python3
"""Seal resource measurement before the first large-scale optimizer step."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from large_scale_training_feasibility_core import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    IMPLEMENTATION_PATHS,
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    SCALE_OUTPUT_PATH,
    SCALE_PLAN_PATH,
    build_plan,
    canonical_bytes,
    validate_plan,
)

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"large-scale training path already exists: {path}")
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(f"large-scale training path has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("large-scale training plan requires a clean worktree")
    _never_published(PLAN_PATH)
    _never_published(OUTPUT_PATH)
    if ARTIFACT_ROOT.exists() or ACTIVE_PATH.exists():
        raise FileExistsError("large-scale training artifact namespace exists")
    scale_plan = _read_json(SCALE_PLAN_PATH)
    scale_summary = _read_json(SCALE_OUTPUT_PATH)
    commit = _git("rev-parse", "HEAD")
    environment = current_runtime_environment_contract()
    plan = build_plan(
        git_commit_before_plan=commit,
        environment=environment,
        implementation_sha256={
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        scale_plan=scale_plan,
        scale_summary=scale_summary,
    )
    validate_plan(plan, current_environment=environment)
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("large-scale training source changed during plan sealing")
    _publish(PLAN_PATH, canonical_bytes(plan))
    expected = f"?? {PLAN_PATH.relative_to(ROOT).as_posix()}"
    if _git("status", "--porcelain") != expected:
        raise ValueError("large-scale training plan is not the only workspace change")
    print(f"plan_path={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
