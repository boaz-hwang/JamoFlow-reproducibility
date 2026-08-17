#!/usr/bin/env python3
"""Seal the balanced-200M trained-screen plan before batch-8 preflight."""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from balanced_200m_trained_core import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    EXPECTED_PARAMETER_COUNT,
    GLOBAL_POSITION_LIMIT,
    IMPLEMENTATION_PATHS,
    MODEL_SEED,
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    RESOURCE_SUMMARY_PATH,
    ROOT,
    SCALE_PLAN_PATH,
    SCALE_SUMMARY_PATH,
    TARGET,
    TRAINING_OUTPUT_PATH,
    build_plan,
    canonical_bytes,
    data_contract,
    validate_plan,
)
from scale_schedule_extrapolation_core import large_scale_model_spec

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import state_sha256
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"balanced-200M path exists: {path}")
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(f"balanced-200M path has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M plan requires a clean worktree")
    for path in (PLAN_PATH, PREFLIGHT_OUTPUT_PATH, TRAINING_OUTPUT_PATH):
        _never_published(path)
    if ARTIFACT_ROOT.exists() or ACTIVE_PATH.exists():
        raise FileExistsError("balanced-200M artifact namespace exists")
    commit = _git("rev-parse", "HEAD")
    model = build_main_model(
        large_scale_model_spec(TARGET, 86),
        seed=MODEL_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    if parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("balanced-200M parameter count differs")
    state = state_sha256(model)
    del model
    gc.collect()
    environment = current_runtime_environment_contract()
    plan = build_plan(
        git_commit_before_plan=commit,
        model_state_sha256=state,
        data=data_contract(),
        environment=environment,
        implementation_sha256={
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        resource_summary=_read_json(RESOURCE_SUMMARY_PATH),
        scale_plan=_read_json(SCALE_PLAN_PATH),
        scale_summary=_read_json(SCALE_SUMMARY_PATH),
    )
    validate_plan(plan, current_environment=environment)
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M source changed during plan sealing")
    _publish(PLAN_PATH, canonical_bytes(plan))
    expected = f"?? {PLAN_PATH.relative_to(ROOT).as_posix()}"
    if _git("status", "--porcelain") != expected:
        raise ValueError("balanced-200M plan is not the only workspace change")
    print(f"plan_path={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
