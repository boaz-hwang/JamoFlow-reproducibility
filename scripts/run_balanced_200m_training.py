#!/usr/bin/env python3
"""Train the sealed balanced-200M C86/W72 mechanism-scale screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from balanced_200m_trained_core import (
    ARTIFACT_ROOT,
    BETAS,
    EFFECTIVE_BATCH_SEQUENCES,
    EPSILON,
    EVALUATION_BATCH_SEQUENCES,
    EXPECTED_PARAMETER_COUNT,
    GLOBAL_POSITION_LIMIT,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MICROBATCH_SEQUENCES,
    MINIMUM_LEARNING_RATE,
    MODEL_SEED,
    PLAN_PATH,
    PREFLIGHT_OUTPUT_PATH,
    PROTOCOL_ID,
    ROLE_ORDER,
    ROOT,
    TARGET,
    TOTAL_UPDATES,
    TRAIN_BYTES,
    TRAIN_SEQUENCES,
    TRAINING_ACTIVE_PATH,
    TRAINING_LOG_EVERY_UPDATES,
    TRAINING_OUTPUT_PATH,
    WARMUP_LR_UPDATES,
    WEIGHT_DECAY,
    build_training_summary,
    calibration_arrays,
    calibration_nll_path,
    canonical_bytes,
    checkpoint_path,
    optimizer_contract,
    training_arrays,
    training_report_path,
    validate_plan,
    validate_preflight_summary,
)
from scale_schedule_extrapolation_core import array_sha256, large_scale_model_spec

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import cosine_learning_rate, evaluate_main_model


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _history(path: Path) -> tuple[str, ...]:
    raw = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(line for line in raw.splitlines() if line)


def _publish(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, sequence_nll_nats=np.ascontiguousarray(values))
    return buffer.getvalue()


def _strict_nll(path: Path) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"balanced-200M NLL artifact differs: {path}")
    with np.load(path, allow_pickle=False) as source:
        if source.files != ["sequence_nll_nats"]:
            raise ValueError("balanced-200M NLL schema differs")
        values = np.ascontiguousarray(source["sequence_nll_nats"])
    return values


def _operational() -> None:
    battery = subprocess.run(
        ("pmset", "-g", "batt"), check=False, capture_output=True, text=True
    )
    thermal = subprocess.run(
        ("pmset", "-g", "therm"), check=False, capture_output=True, text=True
    )
    text = thermal.stdout.lower()
    if (
        battery.returncode != 0
        or "drawing from 'ac power'" not in battery.stdout.lower()
        or thermal.returncode != 0
        or "no thermal warning level has been recorded" not in text
        or "no performance warning level has been recorded" not in text
    ):
        raise RuntimeError("balanced-200M training environment is ineligible")


def _snapshot(stage: str) -> dict[str, Any]:
    torch.mps.synchronize()
    return {
        "stage": stage,
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _context() -> tuple[dict[str, Any], dict[str, Any], str]:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M training requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git(
            "log",
            "-1",
            "--format=%H",
            "--",
            PREFLIGHT_OUTPUT_PATH.relative_to(ROOT).as_posix(),
        )
        != commit
    ):
        raise ValueError("balanced-200M preflight summary must be current HEAD")
    plan = _read_json(PLAN_PATH)
    preflight = _read_json(PREFLIGHT_OUTPUT_PATH)
    environment = current_runtime_environment_contract()
    validate_plan(plan, current_environment=environment)
    validate_preflight_summary(preflight)
    if (
        preflight["plan_sha256"] != plan["plan_sha256"]
        or preflight["plan_artifact_sha256"] != hash_file(PLAN_PATH)
        or preflight["aggregate"].get("overall_preflight_pass") is not True
        or preflight["aggregate"].get("training_protocol_may_be_implemented")
        is not True
    ):
        raise ValueError("balanced-200M preflight does not authorize training")
    return plan, preflight, commit


def _patch_tensor(values: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    selected = values[indices]
    used = np.flatnonzero(np.any(selected != 0, axis=0))
    if not used.size:
        raise ValueError("balanced-200M patch batch is empty")
    selected = selected[:, : int(used[-1]) + 1]
    return torch.from_numpy(selected.astype(np.int64, copy=False)).to("mps")


def _train(
    model: Any,
    inputs: np.ndarray,
    patches: np.ndarray,
    order: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, Any]] = []
    snapshots = [_snapshot("optimizer_created")]
    loss_sum = 0.0
    target_count = 0
    final_loss = math.nan
    final_lr = math.nan
    torch.mps.synchronize()
    started = time.perf_counter()
    for step in range(TOTAL_UPDATES):
        learning_rate = cosine_learning_rate(
            step,
            TOTAL_UPDATES,
            WARMUP_LR_UPDATES,
            LEARNING_RATE,
            MINIMUM_LEARNING_RATE,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        start = step * EFFECTIVE_BATCH_SEQUENCES
        update_indices = order[start : start + EFFECTIVE_BATCH_SEQUENCES]
        update_losses: list[torch.Tensor] = []
        for accumulation in range(GRADIENT_ACCUMULATION_STEPS):
            left = accumulation * MICROBATCH_SEQUENCES
            right = left + MICROBATCH_SEQUENCES
            indices = update_indices[left:right]
            batch = torch.from_numpy(inputs[indices].astype(np.int64, copy=False)).to(
                "mps"
            )
            patch = _patch_tensor(patches, indices)
            output = model(
                input_ids=batch,
                patch_lengths=patch,
                labels=batch,
                use_cache=False,
            )
            raw_loss = output.loss
            update_losses.append(raw_loss.detach())
            (raw_loss / GRADIENT_ACCUMULATION_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
        )
        optimizer.step()
        losses = torch.stack(update_losses).float().cpu().numpy()
        if not np.all(np.isfinite(losses)):
            raise RuntimeError("balanced-200M training loss is nonfinite")
        update_loss = float(np.mean(losses, dtype=np.float64))
        local_targets = EFFECTIVE_BATCH_SEQUENCES * (inputs.shape[1] - 1)
        loss_sum += update_loss * local_targets
        target_count += local_targets
        final_loss = update_loss
        final_lr = learning_rate
        if (
            step == 0
            or (step + 1) % TRAINING_LOG_EVERY_UPDATES == 0
            or step + 1 == TOTAL_UPDATES
        ):
            row = {
                "update": step + 1,
                "loss_nats": update_loss,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(
                f"update={step + 1}/{TOTAL_UPDATES} "
                f"loss_nats={update_loss:.6f} lr={learning_rate:.8g}",
                flush=True,
            )
            if (step + 1) % 1_000 == 0:
                _operational()
                snapshots.append(_snapshot(f"update_{step + 1}"))
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    snapshots.append(_snapshot("training_complete"))
    if target_count != TRAIN_SEQUENCES * (inputs.shape[1] - 1):
        raise ValueError("balanced-200M trained target count differs")
    summary = {
        "updates": TOTAL_UPDATES,
        "examples": TRAIN_SEQUENCES,
        "source_bytes": TRAIN_BYTES,
        "predicted_bytes": target_count,
        "mean_loss_nats": loss_sum / target_count,
        "final_loss_nats": final_loss,
        "elapsed_seconds": elapsed,
        "source_bytes_per_second": TRAIN_BYTES / elapsed,
        "final_learning_rate": final_lr,
        "history": history,
    }
    model.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.mps.empty_cache()
    return summary, snapshots


def _save_checkpoint(model: Any, path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"balanced-200M checkpoint exists: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise FileExistsError(f"balanced-200M partial checkpoint exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(model.state_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"balanced-200M checkpoint raced: {path}")
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            raise RuntimeError(
                f"balanced-200M partial checkpoint preserved: {temporary}"
            )
        raise


def _worker(role: str) -> None:
    plan, preflight, commit = _context()
    report_path = training_report_path(role)
    state_path = checkpoint_path(role)
    nll_path = calibration_nll_path(role)
    if role not in ROLE_ORDER or any(
        path.exists() for path in (report_path, state_path, nll_path)
    ):
        raise ValueError("balanced-200M training worker namespace differs")
    _operational()
    environment = current_runtime_environment_contract()
    inputs, matrices, order = training_arrays()
    patches = matrices[role]
    del matrices
    calibration_inputs, calibration_matrices = calibration_arrays()
    calibration_patches = calibration_matrices[role]
    del calibration_matrices
    if (
        array_sha256(inputs) != plan["data"]["inputs_array_sha256"]
        or array_sha256(order) != plan["data"]["training_order_array_sha256"]
        or array_sha256(patches) != plan["data"]["training_patch_matrix_sha256"][role]
        or array_sha256(calibration_inputs)
        != plan["data"]["calibration_inputs_array_sha256"]
        or array_sha256(calibration_patches)
        != plan["data"]["calibration_patch_matrix_sha256"][role]
    ):
        raise ValueError("balanced-200M training data differs")
    torch.mps.set_per_process_memory_fraction(
        plan["preflight"]["maximum_memory_fraction"]
    )
    recommended_memory = int(torch.mps.recommended_max_memory())
    model = build_main_model(
        large_scale_model_spec(TARGET, 86),
        seed=MODEL_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initial_state = state_sha256(model)
    count = parameter_count(model)
    if (
        count != EXPECTED_PARAMETER_COUNT
        or initial_state != plan["model"]["model_state_sha256"]
    ):
        raise ValueError("balanced-200M training model differs")
    with publication_mps_exclusive():
        model.to("mps")
        model.train()
        memory = [_snapshot("model_resident")]
        training, training_memory = _train(model, inputs, patches, order)
        memory.extend(training_memory)
        evaluation, losses = evaluate_main_model(
            model,
            calibration_inputs,
            calibration_patches,
            "mps",
            batch_size=EVALUATION_BATCH_SEQUENCES,
            return_sequence_nll=True,
        )
        if losses is None:
            raise AssertionError("balanced-200M calibration NLL was not produced")
        nll = np.ascontiguousarray(losses.astype(np.float32, copy=False))
        memory.append(_snapshot("calibration_complete"))
        model.to("cpu")
        torch.mps.synchronize()
        torch.mps.empty_cache()
    trained_state = state_sha256(model)
    end_environment = current_runtime_environment_contract()
    _operational()
    if (
        end_environment != environment
        or _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain")
    ):
        raise ValueError("balanced-200M source/environment changed during training")
    _save_checkpoint(model, state_path)
    nll_bytes = _npz_bytes(nll)
    _publish(nll_path, nll_bytes, mode=0o600)
    report = {
        "schema_version": 1,
        "kind": "balanced_200m_training_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "role": role,
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
        "optimizer": optimizer_contract(),
        "training": training,
        "calibration_evaluation": evaluation.to_dict(),
        "checkpoint_artifact": {
            "path": state_path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(state_path),
            "state_sha256": trained_state,
        },
        "calibration_nll_artifact": {
            "path": nll_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(nll_bytes).hexdigest(),
            "array_sha256": array_sha256(nll),
        },
        "memory_snapshots": memory,
        "maximum_driver_allocated_bytes": max(
            row["driver_allocated_bytes"] for row in memory
        ),
        "recommended_max_memory_bytes": recommended_memory,
        "environment_start": environment,
        "environment_end": end_environment,
        "historical_test_or_final_metric_used": False,
        "completed": True,
    }
    _publish(report_path, canonical_bytes(report), mode=0o600)
    del model
    gc.collect()


def _validate_training_report(
    report: Mapping[str, Any],
    *,
    role: str,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    commit: str,
) -> np.ndarray:
    state_path = checkpoint_path(role)
    nll_path = calibration_nll_path(role)
    nll = _strict_nll(nll_path)
    checkpoint = report.get("checkpoint_artifact")
    nll_artifact = report.get("calibration_nll_artifact")
    evaluation = report.get("calibration_evaluation")
    training = report.get("training")
    memory = report.get("memory_snapshots")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "balanced_200m_training_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("role") != role
        or report.get("runner_git_commit") != commit
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("preflight_summary_sha256") != preflight["summary_sha256"]
        or report.get("preflight_artifact_sha256") != hash_file(PREFLIGHT_OUTPUT_PATH)
        or report.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or report.get("initial_state_sha256") != plan["model"]["model_state_sha256"]
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != state_path.relative_to(ROOT).as_posix()
        or checkpoint.get("sha256") != hash_file(state_path)
        or checkpoint.get("state_sha256") != report.get("trained_state_sha256")
        or not isinstance(nll_artifact, Mapping)
        or nll_artifact.get("path") != nll_path.relative_to(ROOT).as_posix()
        or nll_artifact.get("sha256") != hash_file(nll_path)
        or nll_artifact.get("array_sha256") != array_sha256(nll)
        or report.get("inputs_array_sha256") != plan["data"]["inputs_array_sha256"]
        or report.get("training_order_array_sha256")
        != plan["data"]["training_order_array_sha256"]
        or report.get("patch_matrix_sha256")
        != plan["data"]["training_patch_matrix_sha256"][role]
        or report.get("calibration_inputs_array_sha256")
        != plan["data"]["calibration_inputs_array_sha256"]
        or report.get("calibration_patch_matrix_sha256")
        != plan["data"]["calibration_patch_matrix_sha256"][role]
        or report.get("optimizer") != optimizer_contract()
        or not isinstance(training, Mapping)
        or training.get("updates") != TOTAL_UPDATES
        or training.get("examples") != TRAIN_SEQUENCES
        or training.get("source_bytes") != TRAIN_BYTES
        or not isinstance(evaluation, Mapping)
        or evaluation.get("examples") != len(nll)
        or evaluation.get("predicted_bytes") != len(nll) * 511
        or not np.isclose(
            float(evaluation.get("bpb", math.nan)),
            float(nll.astype(np.float64).sum()) / (len(nll) * 511 * math.log(2)),
            rtol=0,
            atol=1e-7,
        )
        or not isinstance(memory, list)
        or not memory
        or report.get("maximum_driver_allocated_bytes")
        != max(row["driver_allocated_bytes"] for row in memory)
        or report.get("environment_start") != plan["environment"]
        or report.get("environment_end") != plan["environment"]
        or report.get("historical_test_or_final_metric_used") is not False
        or report.get("completed") is not True
    ):
        raise ValueError(f"balanced-200M training report differs: {role}")
    return nll


def _run_all() -> None:
    plan, preflight, commit = _context()
    if TRAINING_OUTPUT_PATH.exists() or _history(TRAINING_OUTPUT_PATH):
        raise FileExistsError("balanced-200M training summary was published")
    active = canonical_bytes(
        {
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": plan["plan_sha256"],
            "preflight_summary_sha256": preflight["summary_sha256"],
            "runner_git_commit": commit,
        }
    )
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if TRAINING_ACTIVE_PATH.exists():
        if TRAINING_ACTIVE_PATH.read_bytes() != active:
            raise ValueError("balanced-200M training active marker differs")
    else:
        _publish(TRAINING_ACTIVE_PATH, active, mode=0o600)
    reports: dict[str, Any] = {}
    nll_by_role: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    for role in ROLE_ORDER:
        paths = (
            training_report_path(role),
            checkpoint_path(role),
            calibration_nll_path(role),
        )
        presence = tuple(path.exists() for path in paths)
        if not any(presence):
            completed = subprocess.run(
                (sys.executable, str(Path(__file__).resolve()), "--worker-role", role),
                check=False,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"balanced-200M training worker failed: {role}")
        elif not all(presence):
            raise ValueError(
                f"partial balanced-200M training evidence preserved: {role}"
            )
        report = _read_json(training_report_path(role))
        nll = _validate_training_report(
            report, role=role, plan=plan, preflight=preflight, commit=commit
        )
        reports[role] = report
        nll_by_role[role] = nll
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
            "calibration_nll_array_sha256": array_sha256(nll),
        }
        print(f"balanced_200m_training_worker_complete={role}", flush=True)
    summary = build_training_summary(
        plan=plan,
        plan_artifact_sha256=hash_file(PLAN_PATH),
        preflight=preflight,
        preflight_artifact_sha256=hash_file(PREFLIGHT_OUTPUT_PATH),
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        nll_by_role=nll_by_role,
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M source changed during training campaign")
    _publish(TRAINING_OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    TRAINING_ACTIVE_PATH.unlink()
    print(f"status={summary['status']}")
    print(f"w72_minus_c86_bpb={summary['quality']['w72_minus_c86_bpb']:.9f}")
    print(f"summary_sha256={summary['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-role", choices=ROLE_ORDER)
    args = parser.parse_args()
    if args.worker_role is None:
        _run_all()
    else:
        _worker(args.worker_role)


if __name__ == "__main__":
    main()
