#!/usr/bin/env python3
"""Run the sealed batch-8 W80 optimizer preflight in a fresh worker."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import balanced_200m_trained_core as base
from balanced_200m_w80_core import (
    ARTIFACT_ROOT,
    PLAN_PATH,
    PREFLIGHT_ACTIVE_PATH,
    PREFLIGHT_OUTPUT_PATH,
    PROTOCOL_ID,
    ROOT,
    build_preflight_summary,
    canonical_bytes,
    preflight_arrays,
    validate_plan,
    worker_preflight_path,
)
from run_balanced_200m_preflight import _operational, _snapshot, _update
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


def _context() -> tuple[dict[str, Any], str]:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 preflight requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()) != commit:
        raise ValueError("balanced-200M W80 plan must be current HEAD")
    plan = _read(PLAN_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    return plan, commit


def _worker(output: Path) -> None:
    plan, commit = _context()
    if output.exists() or output.is_symlink():
        raise FileExistsError("balanced-200M W80 preflight worker output exists")
    environment = current_runtime_environment_contract()
    _operational()
    inputs, patches = preflight_arrays()
    if (
        array_sha256(inputs) != plan["data"]["preflight_inputs_array_sha256"]
        or array_sha256(patches)
        != plan["data"]["w80_preflight_patch_matrix_sha256"]
    ):
        raise ValueError("balanced-200M W80 preflight arrays differ")
    torch.mps.set_per_process_memory_fraction(base.MAXIMUM_RECOMMENDED_MEMORY_FRACTION)
    recommended = int(torch.mps.recommended_max_memory())
    model = build_main_model(
        large_scale_model_spec(base.TARGET, 86),
        seed=base.MODEL_SEED,
        global_max_position_embeddings=base.GLOBAL_POSITION_LIMIT,
    )
    state = state_sha256(model)
    count = parameter_count(model)
    if count != base.EXPECTED_PARAMETER_COUNT or state != plan["model"]["initial_state_sha256"]:
        raise ValueError("balanced-200M W80 preflight model differs")
    snapshots: list[dict[str, Any]] = []
    seconds: list[float] = []
    with publication_mps_exclusive():
        model.to("mps").train()
        snapshots.append(_snapshot("model_resident"))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base.LEARNING_RATE,
            betas=base.BETAS,
            eps=base.EPSILON,
            weight_decay=base.WEIGHT_DECAY,
        )
        elapsed, finite = _update(model, optimizer, inputs, patches, 0)
        if not finite or not optimizer.state:
            raise RuntimeError("balanced-200M W80 warmup update failed")
        snapshots.append(_snapshot("optimizer_state_initialized"))
        for index in range(base.PREFLIGHT_MEASUREMENT_UPDATES):
            elapsed, finite = _update(model, optimizer, inputs, patches, index + 1)
            if not finite:
                raise RuntimeError("balanced-200M W80 measured update failed")
            seconds.append(elapsed)
            snapshots.append(_snapshot(f"measurement_{index}"))
    end_environment = current_runtime_environment_contract()
    _operational()
    if end_environment != environment or _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 preflight environment changed")
    report = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_preflight_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "runner_git_commit": commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "parameter_count": count,
        "model_state_sha256": state,
        "inputs_array_sha256": array_sha256(inputs),
        "patch_matrix_sha256": array_sha256(patches),
        "optimizer_state_initialized": True,
        "memory_cap_enforced": True,
        "memory_snapshots": snapshots,
        "maximum_driver_allocated_bytes": max(row["driver_allocated_bytes"] for row in snapshots),
        "recommended_max_memory_bytes": recommended,
        "measurement": base.project_preflight(seconds),
        "finite": True,
        "completed": True,
        "environment_start": environment,
        "environment_end": end_environment,
    }
    _publish(output, canonical_bytes(report), mode=0o600)
    del optimizer, model
    gc.collect()
    torch.mps.empty_cache()


def validate_worker(report: Mapping[str, Any], *, plan: Mapping[str, Any], commit: str) -> None:
    snapshots = report.get("memory_snapshots")
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "balanced_200m_w80_preflight_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("runner_git_commit") != commit
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("parameter_count") != base.EXPECTED_PARAMETER_COUNT
        or report.get("model_state_sha256") != plan["model"]["initial_state_sha256"]
        or report.get("inputs_array_sha256") != plan["data"]["preflight_inputs_array_sha256"]
        or report.get("patch_matrix_sha256") != plan["data"]["w80_preflight_patch_matrix_sha256"]
        or report.get("measurement")
        != base.project_preflight(report.get("measurement", {}).get("measurement_update_seconds", ()))
        or report.get("completed") is not True
        or report.get("finite") is not True
        or report.get("optimizer_state_initialized") is not True
        or report.get("memory_cap_enforced") is not True
        or not isinstance(snapshots, list)
        or len(snapshots) != 2 + base.PREFLIGHT_MEASUREMENT_UPDATES
        or report.get("maximum_driver_allocated_bytes")
        != max(row["driver_allocated_bytes"] for row in snapshots)
        or report.get("environment_start") != plan["environment"]
        or report.get("environment_end") != plan["environment"]
    ):
        raise ValueError("balanced-200M W80 preflight report differs")


def _run_all() -> None:
    plan, commit = _context()
    if PREFLIGHT_OUTPUT_PATH.exists() or _history(PREFLIGHT_OUTPUT_PATH):
        raise FileExistsError("balanced-200M W80 preflight was published")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    active = canonical_bytes({"protocol_id": PROTOCOL_ID, "plan_sha256": plan["plan_sha256"], "runner_git_commit": commit})
    if PREFLIGHT_ACTIVE_PATH.exists():
        if PREFLIGHT_ACTIVE_PATH.read_bytes() != active:
            raise ValueError("balanced-200M W80 preflight marker differs")
    else:
        _publish(PREFLIGHT_ACTIVE_PATH, active, mode=0o600)
    path = worker_preflight_path()
    if not path.exists():
        temporary = ARTIFACT_ROOT / f".preflight-{os.getpid()}.worker"
        completed = subprocess.run(
            (sys.executable, str(Path(__file__).resolve()), "--worker-output", str(temporary)),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"balanced-200M W80 preflight worker failed: {completed.stderr[-3000:]}")
        temporary.replace(path)
    report = _read(path)
    validate_worker(report, plan=plan, commit=commit)
    summary = build_preflight_summary(
        plan=plan,
        plan_artifact_sha256=hash_file(PLAN_PATH),
        summary_base_git_commit=commit,
        worker_path=path.relative_to(ROOT).as_posix(),
        worker_sha256=hash_file(path),
        report=report,
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 repository changed during preflight")
    _publish(PREFLIGHT_OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    PREFLIGHT_ACTIVE_PATH.unlink()
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker_output is None:
        _run_all()
    else:
        _worker(args.worker_output)


if __name__ == "__main__":
    main()

