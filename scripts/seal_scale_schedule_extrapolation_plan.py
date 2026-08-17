#!/usr/bin/env python3
"""Seal the post-100M W72/C86 schedule-extrapolation plan."""

from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

from scale_schedule_extrapolation_core import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    EXPECTED_PARAMETERS,
    GLOBAL_POSITION_LIMIT,
    IMPLEMENTATION_PATHS,
    MODEL_SEED,
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    TARGET_ORDER,
    build_scale_schedule_plan,
    canonical_bytes,
    large_scale_model_spec,
    validate_plan,
)

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import state_sha256
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"scale-schedule path already exists: {path}")
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(f"scale-schedule path has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _model_contract_with_states() -> dict[str, object]:
    rows: dict[str, object] = {}
    for target in TARGET_ORDER:
        spec = large_scale_model_spec(target, 86)
        model = build_main_model(
            spec,
            seed=MODEL_SEED,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        count = parameter_count(model)
        if count != EXPECTED_PARAMETERS[target]:
            raise ValueError(f"scale-schedule parameter count differs: {target}")
        rows[str(target)] = {
            "expected_parameter_count": count,
            "model_state_sha256": state_sha256(model),
            "spec": spec.to_dict(),
        }
        del model
        gc.collect()
    return rows


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("scale-schedule plan sealing requires a clean worktree")
    _never_published(PLAN_PATH)
    _never_published(OUTPUT_PATH)
    if ARTIFACT_ROOT.exists() or ACTIVE_PATH.exists():
        raise FileExistsError("scale-schedule artifact namespace already exists")
    commit = _git("rev-parse", "HEAD")
    environment = current_runtime_environment_contract()
    plan = build_scale_schedule_plan(
        git_commit_before_plan=commit,
        models=_model_contract_with_states(),
        environment=environment,
        implementation_sha256={
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
    )
    validate_plan(plan, current_environment=environment)
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("scale-schedule source changed during plan construction")
    _publish(PLAN_PATH, canonical_bytes(plan))
    expected = f"?? {PLAN_PATH.relative_to(ROOT).as_posix()}"
    if _git("status", "--porcelain") != expected:
        raise ValueError("scale-schedule plan is not the only workspace change")
    print(f"plan_path={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
