#!/usr/bin/env python3
"""Run the sealed batch-8 training preflight for balanced 200M C86/W72."""

from __future__ import annotations

import argparse
import gc
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
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    BETAS,
    EFFECTIVE_BATCH_SEQUENCES,
    EPSILON,
    EXPECTED_PARAMETER_COUNT,
    GLOBAL_POSITION_LIMIT,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
    MICROBATCH_SEQUENCES,
    MODEL_SEED,
    PLAN_PATH,
    PREFLIGHT_MEASUREMENT_UPDATES,
    PREFLIGHT_OUTPUT_PATH,
    PROTOCOL_ID,
    ROLE_ORDER,
    ROOT,
    TARGET,
    WEIGHT_DECAY,
    build_preflight_summary,
    canonical_bytes,
    preflight_arrays,
    project_preflight,
    validate_plan,
    worker_report_path,
)
from scale_schedule_extrapolation_core import array_sha256, large_scale_model_spec

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _publish(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _history(path: Path) -> tuple[str, ...]:
    raw = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(line for line in raw.splitlines() if line)


def _require_plan_head() -> tuple[dict[str, Any], str, str]:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M preflight requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix())
        != commit
    ):
        raise ValueError("balanced-200M plan must be current HEAD")
    plan = _read_json(PLAN_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    return plan, commit, hash_file(PLAN_PATH)


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
        raise RuntimeError("balanced-200M operational environment is ineligible")


def _snapshot(stage: str) -> dict[str, Any]:
    torch.mps.synchronize()
    return {
        "stage": stage,
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _update(
    model: Any, optimizer: Any, inputs: np.ndarray, patches: np.ndarray, update: int
) -> tuple[float, bool]:
    optimizer.zero_grad(set_to_none=True)
    start_index = update * EFFECTIVE_BATCH_SEQUENCES
    losses: list[torch.Tensor] = []
    torch.mps.synchronize()
    started = time.perf_counter()
    for accumulation in range(GRADIENT_ACCUMULATION_STEPS):
        start = start_index + accumulation * MICROBATCH_SEQUENCES
        end = start + MICROBATCH_SEQUENCES
        batch = torch.from_numpy(inputs[start:end].astype(np.int64, copy=False)).to(
            "mps"
        )
        selected = patches[start:end]
        used_columns = np.flatnonzero(np.any(selected != 0, axis=0))
        if not used_columns.size:
            raise ValueError("balanced-200M preflight patch batch is empty")
        selected = selected[:, : int(used_columns[-1]) + 1]
        patch = torch.from_numpy(selected.astype(np.int64, copy=False)).to("mps")
        output = model(
            input_ids=batch, patch_lengths=patch, labels=batch, use_cache=False
        )
        loss = output.loss / GRADIENT_ACCUMULATION_STEPS
        losses.append(loss.detach())
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    finite = bool(torch.all(torch.isfinite(torch.stack(losses))).item())
    return elapsed, bool(finite and math.isfinite(elapsed) and elapsed > 0)


def _worker(role: str, output: Path) -> None:
    plan, commit, plan_artifact = _require_plan_head()
    if role not in ROLE_ORDER or output.exists():
        raise ValueError("balanced-200M worker identity differs")
    environment = current_runtime_environment_contract()
    _operational()
    inputs, matrices = preflight_arrays()
    inputs = np.ascontiguousarray(inputs)
    patches = np.ascontiguousarray(matrices[role])
    if (
        array_sha256(inputs) != plan["data"]["preflight_inputs_array_sha256"]
        or array_sha256(patches) != plan["data"]["preflight_patch_matrix_sha256"][role]
    ):
        raise ValueError("balanced-200M preflight data differs")
    torch.mps.set_per_process_memory_fraction(MAXIMUM_RECOMMENDED_MEMORY_FRACTION)
    recommended = int(torch.mps.recommended_max_memory())
    model = build_main_model(
        large_scale_model_spec(TARGET, 86),
        seed=MODEL_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    state = state_sha256(model)
    count = parameter_count(model)
    if (
        count != EXPECTED_PARAMETER_COUNT
        or state != plan["model"]["model_state_sha256"]
    ):
        raise ValueError("balanced-200M preflight model differs")
    snapshots: list[dict[str, Any]] = []
    update_seconds: list[float] = []
    with publication_mps_exclusive():
        model = model.to("mps")
        model.train()
        snapshots.append(_snapshot("model_resident"))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            eps=EPSILON,
            weight_decay=WEIGHT_DECAY,
        )
        elapsed, finite = _update(model, optimizer, inputs, patches, 0)
        if not finite or not optimizer.state or elapsed <= 0:
            raise RuntimeError("balanced-200M warmup update failed")
        snapshots.append(_snapshot("optimizer_state_initialized"))
        for measured in range(PREFLIGHT_MEASUREMENT_UPDATES):
            elapsed, finite = _update(model, optimizer, inputs, patches, measured + 1)
            if not finite:
                raise RuntimeError("balanced-200M measured update failed")
            update_seconds.append(elapsed)
            snapshots.append(_snapshot(f"measurement_{measured}"))
    end_environment = current_runtime_environment_contract()
    _operational()
    if (
        end_environment != environment
        or _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain")
    ):
        raise ValueError("balanced-200M worker environment/source changed")
    report = {
        "schema_version": 1,
        "kind": "balanced_200m_batch8_preflight_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "runner_git_commit": commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_sha256": plan_artifact,
        "parameter_count": count,
        "model_state_sha256": state,
        "inputs_array_sha256": array_sha256(inputs),
        "patch_matrix_sha256": array_sha256(patches),
        "optimizer_state_initialized": True,
        "memory_cap_enforced": True,
        "memory_snapshots": snapshots,
        "maximum_driver_allocated_bytes": max(
            row["driver_allocated_bytes"] for row in snapshots
        ),
        "recommended_max_memory_bytes": recommended,
        "measurement": project_preflight(update_seconds),
        "finite": True,
        "completed": True,
        "environment_start": environment,
        "environment_end": end_environment,
    }
    _publish(output, canonical_bytes(report), mode=0o600)
    del optimizer, model
    gc.collect()
    torch.mps.empty_cache()


def _validate_report(
    report: Mapping[str, Any], *, role: str, plan: Mapping[str, Any], commit: str
) -> None:
    snapshots = report.get("memory_snapshots")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "balanced_200m_batch8_preflight_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("role") != role
        or report.get("runner_git_commit") != commit
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or report.get("model_state_sha256") != plan["model"]["model_state_sha256"]
        or report.get("inputs_array_sha256")
        != plan["data"]["preflight_inputs_array_sha256"]
        or report.get("patch_matrix_sha256")
        != plan["data"]["preflight_patch_matrix_sha256"][role]
        or report.get("measurement")
        != project_preflight(
            report.get("measurement", {}).get("measurement_update_seconds", ())
        )
        or report.get("completed") is not True
        or report.get("finite") is not True
        or report.get("optimizer_state_initialized") is not True
        or report.get("memory_cap_enforced") is not True
        or not isinstance(snapshots, list)
        or len(snapshots) != 2 + PREFLIGHT_MEASUREMENT_UPDATES
        or report.get("maximum_driver_allocated_bytes")
        != max(row["driver_allocated_bytes"] for row in snapshots)
        or report.get("environment_start") != plan["environment"]
        or report.get("environment_end") != plan["environment"]
    ):
        raise ValueError("balanced-200M preflight report differs")


def _run_all() -> None:
    plan, commit, plan_artifact = _require_plan_head()
    if PREFLIGHT_OUTPUT_PATH.exists() or _history(PREFLIGHT_OUTPUT_PATH):
        raise FileExistsError("balanced-200M preflight summary was published")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    active = canonical_bytes(
        {
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": plan["plan_sha256"],
            "commit": commit,
        }
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active:
            raise ValueError("balanced-200M active preflight differs")
    else:
        _publish(ACTIVE_PATH, active, mode=0o600)
    reports: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for role in ROLE_ORDER:
        path = worker_report_path(role)
        if not path.exists():
            temporary = ARTIFACT_ROOT / f".{role}.{os.getpid()}.worker"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-role",
                    role,
                    "--worker-output",
                    str(temporary),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"balanced-200M worker failed: {role}: {completed.stderr[-2000:]}"
                )
            temporary.replace(path)
        report = _read_json(path)
        _validate_report(report, role=role, plan=plan, commit=commit)
        reports[role] = report
        evidence[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
        print(f"balanced_200m_preflight_worker_complete={role}", flush=True)
    summary = build_preflight_summary(
        plan=plan,
        plan_artifact_sha256=plan_artifact,
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        reports=reports,
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M source changed during preflight")
    _publish(PREFLIGHT_OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    ACTIVE_PATH.unlink()
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-role")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_role is not None or args.worker_output is not None:
        if args.worker_role is None or args.worker_output is None:
            parser.error("both worker arguments are required")
        _worker(args.worker_role, args.worker_output)
    else:
        _run_all()


if __name__ == "__main__":
    main()
