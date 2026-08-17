#!/usr/bin/env python3
"""Seal the balanced-200M W80 rescue plan before any W80 model work."""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from balanced_200m_w80_core import (
    ARTIFACT_ROOT,
    BASE_PLAN_PATH,
    BASE_SUMMARY_PATH,
    BASE_VERIFICATION_PATH,
    FAILURE_ANALYSIS_PATH,
    IMPLEMENTATION_PATHS,
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    ROOT,
    SCALE_PLAN_PATH,
    TIMING_SUMMARY_PATH,
    TRAINING_OUTPUT_PATH,
    VERIFICATION_OUTPUT_PATH,
    build_plan,
    canonical_bytes,
    case_contract,
    data_contract,
    validate_plan,
)
import balanced_200m_trained_core as base
from scale_schedule_extrapolation_core import large_scale_model_spec

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import state_sha256
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _head_blob(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)


def _never_published(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    if path.exists() or _git("log", "--all", "--format=%H", "--", relative):
        raise FileExistsError(f"balanced-200M W80 path was published: {relative}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 plan requires a clean worktree")
    for path in (
        PLAN_PATH,
        PREFLIGHT_OUTPUT_PATH,
        TRAINING_OUTPUT_PATH,
        VERIFICATION_OUTPUT_PATH,
        TIMING_SUMMARY_PATH,
    ):
        _never_published(path)
    if ARTIFACT_ROOT.exists() or ARTIFACT_ROOT.is_symlink():
        raise FileExistsError("balanced-200M W80 artifact namespace exists")
    for path in (
        BASE_PLAN_PATH,
        BASE_SUMMARY_PATH,
        BASE_VERIFICATION_PATH,
        FAILURE_ANALYSIS_PATH,
        SCALE_PLAN_PATH,
    ):
        if not path.is_file() or path.is_symlink() or _head_blob(path) != path.read_bytes():
            raise ValueError(f"balanced-200M W80 upstream is not exact HEAD: {path}")
    base_plan = _read(BASE_PLAN_PATH)
    base_summary = _read(BASE_SUMMARY_PATH)
    base_verification = _read(BASE_VERIFICATION_PATH)
    failure_analysis = _read(FAILURE_ANALYSIS_PATH)
    baseline = base_summary["worker_evidence"]["c86"]
    for key in ("report_path", "checkpoint_path", "calibration_nll_path"):
        path = ROOT / baseline[key]
        expected = baseline[key.replace("_path", "_sha256")]
        if not path.is_file() or path.is_symlink() or hash_file(path) != expected:
            raise ValueError(f"balanced-200M W80 baseline artifact differs: {key}")
    commit = _git("rev-parse", "HEAD")
    model = build_main_model(
        large_scale_model_spec(base.TARGET, 86),
        seed=base.MODEL_SEED,
        global_max_position_embeddings=base.GLOBAL_POSITION_LIMIT,
    )
    if parameter_count(model) != base.EXPECTED_PARAMETER_COUNT:
        raise ValueError("balanced-200M W80 model parameter count differs")
    state = state_sha256(model)
    del model
    gc.collect()
    environment = current_runtime_environment_contract()
    plan = build_plan(
        git_commit_before_plan=commit,
        model_state_sha256=state,
        data=data_contract(base_plan),
        cases=case_contract(_read(SCALE_PLAN_PATH)),
        environment=environment,
        implementation_sha256={
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        base_plan=base_plan,
        base_summary=base_summary,
        base_verification=base_verification,
        failure_analysis=failure_analysis,
    )
    validate_plan(plan, current_environment=environment)
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 repository changed while sealing")
    _publish(PLAN_PATH, canonical_bytes(plan))
    expected = f"?? {PLAN_PATH.relative_to(ROOT).as_posix()}"
    if _git("status", "--porcelain") != expected:
        raise ValueError("balanced-200M W80 plan is not the only workspace change")
    print(f"plan_path={PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()

