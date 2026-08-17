#!/usr/bin/env python3
"""Replay W80 calibration and publish the timing-authorization receipt."""

from __future__ import annotations

import gc
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import balanced_200m_trained_core as base
from balanced_200m_w80_core import (
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    ROOT,
    TRAINING_OUTPUT_PATH,
    VERIFICATION_OUTPUT_PATH,
    build_verification_receipt,
    calibration_arrays,
    calibration_nll_path,
    canonical_bytes,
    checkpoint_path,
    summarize_quality,
    validate_plan,
    validate_preflight_summary,
    validate_training_summary,
)
from run_balanced_200m_training import _strict_nll
from run_balanced_200m_w80_training import validate_training_report
from scale_schedule_extrapolation_core import array_sha256, large_scale_model_spec

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive, state_sha256
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import evaluate_main_model


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _history(path: Path) -> tuple[str, ...]:
    raw = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(row for row in raw.splitlines() if row)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 verifier requires a clean worktree")
    if VERIFICATION_OUTPUT_PATH.exists() or _history(VERIFICATION_OUTPUT_PATH):
        raise FileExistsError("balanced-200M W80 verification was published")
    if _git("log", "-1", "--format=%H", "--", TRAINING_OUTPUT_PATH.relative_to(ROOT).as_posix()) != _git("rev-parse", "HEAD"):
        raise ValueError("balanced-200M W80 training summary must be current HEAD")
    commit = _git("rev-parse", "HEAD")
    plan = _read(PLAN_PATH)
    preflight = _read(PREFLIGHT_OUTPUT_PATH)
    summary = _read(TRAINING_OUTPUT_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_preflight_summary(preflight)
    validate_training_summary(summary)
    report = _read(ROOT / summary["candidate_evidence"]["report_path"])
    stored = validate_training_report(
        report,
        plan=plan,
        preflight=preflight,
        commit=summary["summary_base_git_commit"],
    )
    calibration_inputs, matrix = calibration_arrays()
    model = build_main_model(
        large_scale_model_spec(base.TARGET, 86),
        seed=base.MODEL_SEED,
        global_max_position_embeddings=base.GLOBAL_POSITION_LIMIT,
    )
    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if parameter_count(model) != base.EXPECTED_PARAMETER_COUNT or state_sha256(model) != report["trained_state_sha256"]:
        raise ValueError("balanced-200M W80 checkpoint state differs")
    with publication_mps_exclusive():
        evaluation, replay = evaluate_main_model(
            model,
            calibration_inputs,
            matrix,
            "mps",
            batch_size=base.EVALUATION_BATCH_SEQUENCES,
            return_sequence_nll=True,
        )
    if replay is None:
        raise AssertionError("balanced-200M W80 replay NLL was not produced")
    replay = np.ascontiguousarray(replay.astype(np.float32, copy=False))
    if not np.array_equal(stored, replay) or not np.isclose(
        evaluation.bpb,
        report["calibration_evaluation"]["bpb"],
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError("balanced-200M W80 full checkpoint replay differs")
    reference = plan["roles"]["reference"]["immutable_training_evidence"]
    c86_path = ROOT / reference["calibration_nll_path"]
    if hash_file(c86_path) != reference["calibration_nll_sha256"]:
        raise ValueError("balanced-200M W80 reference NLL differs")
    quality = summarize_quality(_strict_nll(c86_path), replay)
    receipt = build_verification_receipt(
        plan=plan,
        training_summary=summary,
        verification_base_git_commit=commit,
        replayed_nll_array_sha256=array_sha256(replay),
        replayed_quality=quality,
    )
    model.to("cpu")
    del model, state
    gc.collect()
    torch.mps.empty_cache()
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 repository changed during verification")
    _publish(VERIFICATION_OUTPUT_PATH, canonical_bytes(receipt))
    print("balanced_200m_w80_checkpoint_replay=pass")
    print(f"status={receipt['status']}")
    print(f"actual_timing_authorized={str(receipt['actual_timing_authorized']).lower()}")
    print(f"receipt_sha256={receipt['receipt_sha256']}")


if __name__ == "__main__":
    main()

