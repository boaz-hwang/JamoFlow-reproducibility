#!/usr/bin/env python3
"""Independently replay balanced-200M calibration from trained checkpoints."""

from __future__ import annotations

import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from balanced_200m_trained_core import (
    EVALUATION_BATCH_SEQUENCES,
    EXPECTED_PARAMETER_COUNT,
    GLOBAL_POSITION_LIMIT,
    MODEL_SEED,
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    ROLE_ORDER,
    ROOT,
    TARGET,
    TRAINING_OUTPUT_PATH,
    build_training_summary,
    calibration_arrays,
    calibration_nll_path,
    checkpoint_path,
    training_report_path,
    validate_plan,
    validate_preflight_summary,
    validate_training_summary,
)
from run_balanced_200m_training import _strict_nll, _validate_training_report
from scale_schedule_extrapolation_core import array_sha256, large_scale_model_spec

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import evaluate_main_model


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M training verifier requires a clean worktree")
    plan = _read(PLAN_PATH)
    preflight = _read(PREFLIGHT_OUTPUT_PATH)
    summary = _read(TRAINING_OUTPUT_PATH)
    environment = current_runtime_environment_contract()
    validate_plan(plan, current_environment=environment)
    validate_preflight_summary(preflight)
    validate_training_summary(summary)
    if (
        summary["plan_sha256"] != plan["plan_sha256"]
        or summary["plan_artifact_sha256"] != hash_file(PLAN_PATH)
        or summary["preflight_summary_sha256"] != preflight["summary_sha256"]
        or summary["preflight_artifact_sha256"] != hash_file(PREFLIGHT_OUTPUT_PATH)
    ):
        raise ValueError("balanced-200M training lineage differs")
    calibration_inputs, matrices = calibration_arrays()
    nll_by_role: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    commit = summary["summary_base_git_commit"]
    with publication_mps_exclusive():
        for role in ROLE_ORDER:
            report = _read(training_report_path(role))
            stored = _validate_training_report(
                report,
                role=role,
                plan=plan,
                preflight=preflight,
                commit=commit,
            )
            model = build_main_model(
                large_scale_model_spec(TARGET, 86),
                seed=MODEL_SEED,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            )
            state = torch.load(
                checkpoint_path(role), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
            if (
                parameter_count(model) != EXPECTED_PARAMETER_COUNT
                or state_sha256(model) != report["trained_state_sha256"]
            ):
                raise ValueError(f"balanced-200M checkpoint state differs: {role}")
            evaluation, replay = evaluate_main_model(
                model,
                calibration_inputs,
                matrices[role],
                "mps",
                batch_size=EVALUATION_BATCH_SEQUENCES,
                return_sequence_nll=True,
            )
            if replay is None:
                raise AssertionError("balanced-200M verifier NLL was not produced")
            replay = np.ascontiguousarray(replay.astype(np.float32, copy=False))
            if not np.array_equal(stored, replay):
                raise ValueError(f"balanced-200M checkpoint NLL replay differs: {role}")
            if not np.isclose(
                float(evaluation.bpb),
                float(report["calibration_evaluation"]["bpb"]),
                rtol=0,
                atol=1e-7,
            ):
                raise ValueError(f"balanced-200M replay BPB differs: {role}")
            nll_by_role[role] = replay
            evidence[role] = {
                "report_path": training_report_path(role).relative_to(ROOT).as_posix(),
                "report_sha256": hash_file(training_report_path(role)),
                "checkpoint_path": checkpoint_path(role).relative_to(ROOT).as_posix(),
                "checkpoint_sha256": hash_file(checkpoint_path(role)),
                "checkpoint_state_sha256": report["trained_state_sha256"],
                "calibration_nll_path": calibration_nll_path(role)
                .relative_to(ROOT)
                .as_posix(),
                "calibration_nll_sha256": hash_file(calibration_nll_path(role)),
                "calibration_nll_array_sha256": array_sha256(replay),
            }
            model.to("cpu")
            del model, state
            gc.collect()
            torch.mps.empty_cache()
            print(f"balanced_200m_checkpoint_replay_complete={role}", flush=True)
    rebuilt = build_training_summary(
        plan=plan,
        plan_artifact_sha256=hash_file(PLAN_PATH),
        preflight=preflight,
        preflight_artifact_sha256=hash_file(PREFLIGHT_OUTPUT_PATH),
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        nll_by_role=nll_by_role,
    )
    if rebuilt != summary:
        raise ValueError("balanced-200M training summary does not reconstruct")
    if any(
        array_sha256(_strict_nll(calibration_nll_path(role)))
        != summary["worker_evidence"][role]["calibration_nll_array_sha256"]
        for role in ROLE_ORDER
    ):
        raise ValueError("balanced-200M stored NLL identity differs")
    print("balanced_200m_training_verification=pass")
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
