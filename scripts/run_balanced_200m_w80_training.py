#!/usr/bin/env python3
"""Train the single sealed balanced-200M W80 rescue candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import balanced_200m_trained_core as base
from balanced_200m_w80_core import (
    ARTIFACT_ROOT,
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    PROTOCOL_ID,
    ROOT,
    TRAINING_ACTIVE_PATH,
    TRAINING_OUTPUT_PATH,
    build_training_summary,
    calibration_arrays,
    calibration_nll_path,
    canonical_bytes,
    checkpoint_path,
    training_arrays,
    training_report_path,
    validate_plan,
    validate_preflight_summary,
)
from run_balanced_200m_training import (
    _npz_bytes,
    _operational,
    _save_checkpoint,
    _snapshot,
    _strict_nll,
    _train,
)
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


def _history(path: Path) -> tuple[str, ...]:
    raw = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(row for row in raw.splitlines() if row)


def _publish(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _context() -> tuple[dict[str, Any], dict[str, Any], str]:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 training requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", PREFLIGHT_OUTPUT_PATH.relative_to(ROOT).as_posix()) != commit:
        raise ValueError("balanced-200M W80 preflight must be current HEAD")
    plan = _read(PLAN_PATH)
    preflight = _read(PREFLIGHT_OUTPUT_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_preflight_summary(preflight)
    if (
        preflight.get("plan_sha256") != plan["plan_sha256"]
        or preflight.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or preflight.get("aggregate", {}).get("training_authorized") is not True
    ):
        raise ValueError("balanced-200M W80 preflight does not authorize training")
    return plan, preflight, commit


def _worker() -> None:
    plan, preflight, commit = _context()
    report_path = training_report_path()
    state_path = checkpoint_path()
    nll_path = calibration_nll_path()
    if any(path.exists() or path.is_symlink() for path in (report_path, state_path, nll_path)):
        raise ValueError("balanced-200M W80 training namespace differs")
    environment = current_runtime_environment_contract()
    _operational()
    inputs, patches, order = training_arrays()
    calibration_inputs, calibration_patches = calibration_arrays()
    if (
        array_sha256(inputs) != plan["data"]["inputs_array_sha256"]
        or array_sha256(order) != plan["data"]["training_order_array_sha256"]
        or array_sha256(patches) != plan["data"]["w80_training_patch_matrix_sha256"]
        or array_sha256(calibration_inputs) != plan["data"]["calibration_inputs_array_sha256"]
        or array_sha256(calibration_patches) != plan["data"]["w80_calibration_patch_matrix_sha256"]
    ):
        raise ValueError("balanced-200M W80 training arrays differ")
    torch.mps.set_per_process_memory_fraction(base.MAXIMUM_RECOMMENDED_MEMORY_FRACTION)
    recommended = int(torch.mps.recommended_max_memory())
    model = build_main_model(
        large_scale_model_spec(base.TARGET, 86),
        seed=base.MODEL_SEED,
        global_max_position_embeddings=base.GLOBAL_POSITION_LIMIT,
    )
    initial_state = state_sha256(model)
    count = parameter_count(model)
    if count != base.EXPECTED_PARAMETER_COUNT or initial_state != plan["model"]["initial_state_sha256"]:
        raise ValueError("balanced-200M W80 training model differs")
    with publication_mps_exclusive():
        model.to("mps").train()
        memory = [_snapshot("model_resident")]
        training, training_memory = _train(model, inputs, patches, order)
        memory.extend(training_memory)
        evaluation, losses = evaluate_main_model(
            model,
            calibration_inputs,
            calibration_patches,
            "mps",
            batch_size=base.EVALUATION_BATCH_SEQUENCES,
            return_sequence_nll=True,
        )
        if losses is None:
            raise AssertionError("balanced-200M W80 calibration NLL was not produced")
        nll = np.ascontiguousarray(losses.astype(np.float32, copy=False))
        memory.append(_snapshot("calibration_complete"))
        model.to("cpu")
        torch.mps.synchronize()
        torch.mps.empty_cache()
    trained_state = state_sha256(model)
    end_environment = current_runtime_environment_contract()
    _operational()
    if end_environment != environment or _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 environment/source changed during training")
    checkpoint_bytes_ready = False
    nll_bytes = _npz_bytes(nll)
    report = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_training_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "runner_git_commit": commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "preflight_summary_sha256": preflight["summary_sha256"],
        "preflight_artifact_sha256": hash_file(PREFLIGHT_OUTPUT_PATH),
        "parameter_count": count,
        "initial_state_sha256": initial_state,
        "trained_state_sha256": trained_state,
        "inputs_array_sha256": array_sha256(inputs),
        "training_order_array_sha256": array_sha256(order),
        "patch_matrix_sha256": array_sha256(patches),
        "calibration_inputs_array_sha256": array_sha256(calibration_inputs),
        "calibration_patch_matrix_sha256": array_sha256(calibration_patches),
        "optimizer": base.optimizer_contract(),
        "training": training,
        "calibration_evaluation": evaluation.to_dict(),
        "checkpoint_artifact": {
            "path": state_path.relative_to(ROOT).as_posix(),
            "sha256": None,
            "state_sha256": trained_state,
        },
        "calibration_nll_artifact": {
            "path": nll_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(nll_bytes).hexdigest(),
            "array_sha256": array_sha256(nll),
        },
        "memory_snapshots": memory,
        "maximum_driver_allocated_bytes": max(row["driver_allocated_bytes"] for row in memory),
        "recommended_max_memory_bytes": recommended,
        "environment_start": environment,
        "environment_end": end_environment,
        "historical_test_or_final_metric_used": False,
        "completed": True,
    }
    # The checkpoint hash is only known after deterministic serialization.  Publish it
    # first, then complete the already fully validated report without changing model work.
    _save_checkpoint(model, state_path)
    checkpoint_bytes_ready = True
    report["checkpoint_artifact"]["sha256"] = hash_file(state_path)
    _publish(nll_path, nll_bytes, mode=0o600)
    _publish(report_path, canonical_bytes(report), mode=0o600)
    if not checkpoint_bytes_ready:
        raise AssertionError("balanced-200M W80 checkpoint publication did not complete")
    del model
    gc.collect()


def validate_training_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    commit: str,
) -> np.ndarray:
    state_path = checkpoint_path()
    nll_path = calibration_nll_path()
    nll = _strict_nll(nll_path)
    checkpoint = report.get("checkpoint_artifact")
    nll_artifact = report.get("calibration_nll_artifact")
    evaluation = report.get("calibration_evaluation")
    training = report.get("training")
    memory = report.get("memory_snapshots")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "balanced_200m_w80_training_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("runner_git_commit") != commit
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("preflight_summary_sha256") != preflight["summary_sha256"]
        or report.get("preflight_artifact_sha256") != hash_file(PREFLIGHT_OUTPUT_PATH)
        or report.get("parameter_count") != base.EXPECTED_PARAMETER_COUNT
        or report.get("initial_state_sha256") != plan["model"]["initial_state_sha256"]
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != state_path.relative_to(ROOT).as_posix()
        or checkpoint.get("sha256") != hash_file(state_path)
        or checkpoint.get("state_sha256") != report.get("trained_state_sha256")
        or not isinstance(nll_artifact, Mapping)
        or nll_artifact.get("path") != nll_path.relative_to(ROOT).as_posix()
        or nll_artifact.get("sha256") != hash_file(nll_path)
        or nll_artifact.get("array_sha256") != array_sha256(nll)
        or report.get("inputs_array_sha256") != plan["data"]["inputs_array_sha256"]
        or report.get("training_order_array_sha256") != plan["data"]["training_order_array_sha256"]
        or report.get("patch_matrix_sha256") != plan["data"]["w80_training_patch_matrix_sha256"]
        or report.get("calibration_inputs_array_sha256") != plan["data"]["calibration_inputs_array_sha256"]
        or report.get("calibration_patch_matrix_sha256") != plan["data"]["w80_calibration_patch_matrix_sha256"]
        or report.get("optimizer") != base.optimizer_contract()
        or not isinstance(training, Mapping)
        or training.get("updates") != base.TOTAL_UPDATES
        or training.get("examples") != base.TRAIN_SEQUENCES
        or training.get("source_bytes") != base.TRAIN_BYTES
        or not isinstance(evaluation, Mapping)
        or evaluation.get("examples") != len(nll)
        or evaluation.get("predicted_bytes") != len(nll) * 511
        or not np.isclose(
            float(evaluation.get("bpb", np.nan)),
            base.bpb_from_sequence_nll(nll),
            rtol=0,
            atol=1e-7,
        )
        or not isinstance(memory, list)
        or not memory
        or report.get("maximum_driver_allocated_bytes") != max(row["driver_allocated_bytes"] for row in memory)
        or report.get("environment_start") != plan["environment"]
        or report.get("environment_end") != plan["environment"]
        or report.get("historical_test_or_final_metric_used") is not False
        or report.get("completed") is not True
    ):
        raise ValueError("balanced-200M W80 training report differs")
    return nll


def _run_all() -> None:
    plan, preflight, commit = _context()
    if TRAINING_OUTPUT_PATH.exists() or _history(TRAINING_OUTPUT_PATH):
        raise FileExistsError("balanced-200M W80 training summary was published")
    active = canonical_bytes({
        "protocol_id": PROTOCOL_ID,
        "plan_sha256": plan["plan_sha256"],
        "preflight_summary_sha256": preflight["summary_sha256"],
        "runner_git_commit": commit,
    })
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if TRAINING_ACTIVE_PATH.exists():
        if TRAINING_ACTIVE_PATH.read_bytes() != active:
            raise ValueError("balanced-200M W80 training marker differs")
    else:
        _publish(TRAINING_ACTIVE_PATH, active, mode=0o600)
    paths = (training_report_path(), checkpoint_path(), calibration_nll_path())
    presence = tuple(path.exists() for path in paths)
    if not any(presence):
        completed = subprocess.run(
            (sys.executable, str(Path(__file__).resolve()), "--worker"),
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("balanced-200M W80 training worker failed")
    elif not all(presence):
        raise ValueError("partial balanced-200M W80 training evidence preserved")
    report = _read(training_report_path())
    w80_nll = validate_training_report(report, plan=plan, preflight=preflight, commit=commit)
    reference = plan["roles"]["reference"]["immutable_training_evidence"]
    c86_path = ROOT / reference["calibration_nll_path"]
    if hash_file(c86_path) != reference["calibration_nll_sha256"]:
        raise ValueError("balanced-200M W80 reference NLL artifact differs")
    c86_nll = _strict_nll(c86_path)
    candidate_evidence = {
        "report_path": training_report_path().relative_to(ROOT).as_posix(),
        "report_sha256": hash_file(training_report_path()),
        "checkpoint_path": checkpoint_path().relative_to(ROOT).as_posix(),
        "checkpoint_sha256": hash_file(checkpoint_path()),
        "checkpoint_state_sha256": report["trained_state_sha256"],
        "calibration_nll_path": calibration_nll_path().relative_to(ROOT).as_posix(),
        "calibration_nll_sha256": hash_file(calibration_nll_path()),
        "calibration_nll_array_sha256": array_sha256(w80_nll),
    }
    summary = build_training_summary(
        plan=plan,
        preflight=preflight,
        summary_base_git_commit=commit,
        candidate_evidence=candidate_evidence,
        c86_nll=c86_nll,
        w80_nll=w80_nll,
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 repository changed during training")
    _publish(TRAINING_OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    TRAINING_ACTIVE_PATH.unlink()
    print(f"status={summary['status']}")
    print(f"w80_minus_c86_bpb={summary['quality']['w80_minus_c86_bpb']:.9f}")
    print(f"bootstrap_upper={summary['quality']['block_bootstrap']['upper']:.9f}")
    print(f"summary_sha256={summary['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        _worker()
    else:
        _run_all()


if __name__ == "__main__":
    main()

