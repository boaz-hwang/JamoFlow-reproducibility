#!/usr/bin/env python3
"""Measure real optimizer-step feasibility for the post-100M model family."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from large_scale_training_feasibility_core import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    BETAS,
    CHECKPOINTED_REGIME,
    EPSILON,
    EXPECTED_PARAMETERS,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
    MEASUREMENT_UPDATES,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    STANDARD_REGIME,
    WARMUP_UPDATES,
    WEIGHT_DECAY,
    build_summary,
    canonical_bytes,
    large_scale_model_spec,
    projected_training,
    training_arrays,
    validate_plan,
    validate_worker_report,
    worker_id,
    worker_order,
    worker_report_path,
)
from scale_schedule_extrapolation_core import GLOBAL_POSITION_LIMIT, MODEL_SEED

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
        raise ValueError("large-scale training execution requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix())
        != commit
    ):
        raise ValueError("large-scale training plan must be current HEAD")
    plan = _read_json(PLAN_PATH)
    environment = current_runtime_environment_contract()
    validate_plan(plan, current_environment=environment)
    upstream = plan["upstream"]
    if (
        hash_file(ROOT / upstream["scale_plan_path"])
        != upstream["scale_plan_artifact_sha256"]
        or hash_file(ROOT / upstream["scale_summary_path"])
        != upstream["scale_summary_artifact_sha256"]
    ):
        raise ValueError("large-scale training upstream artifacts changed")
    return plan, commit, hash_file(PLAN_PATH)


def _require_operational_environment() -> None:
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
        raise RuntimeError("large-scale training operational environment is ineligible")


def _snapshot(stage: str) -> dict[str, Any]:
    torch.mps.synchronize()
    return {
        "stage": stage,
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _update(
    model: Any,
    optimizer: Any,
    inputs: np.ndarray,
    patches: np.ndarray,
    *,
    update_index: int,
) -> tuple[float, bool]:
    optimizer.zero_grad(set_to_none=True)
    offset = update_index * GRADIENT_ACCUMULATION_STEPS
    torch.mps.synchronize()
    started = time.perf_counter()
    losses: list[torch.Tensor] = []
    for microstep in range(GRADIENT_ACCUMULATION_STEPS):
        index = (offset + microstep) % len(inputs)
        batch = torch.from_numpy(
            inputs[index : index + 1].astype(np.int64, copy=False)
        ).to("mps")
        patch = torch.from_numpy(
            patches[index : index + 1].astype(np.int64, copy=False)
        ).to("mps")
        output = model(
            input_ids=batch,
            patch_lengths=patch,
            labels=batch,
            use_cache=False,
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


def _base_report(
    *,
    plan: dict[str, Any],
    plan_artifact_sha256: str,
    commit: str,
    target: int,
    regime: str,
    role: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "large_scale_training_feasibility_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "target_millions": target,
        "regime": regime,
        "role": role,
        "runner_git_commit": commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_sha256": plan_artifact_sha256,
        "parameter_count": None,
        "model_state_sha256": None,
        "patch_matrix_sha256": plan["training_data"]["patch_matrix_sha256"][role],
        "training_data_sha256": plan["training_data"]["inputs_array_sha256"],
        "memory_cap_enforced": True,
        "memory_snapshots": [],
        "maximum_driver_allocated_bytes": 0,
        "recommended_max_memory_bytes": None,
        "optimizer_state_initialized": False,
        "measurement": None,
        "finite": False,
        "completed": False,
        "failure": None,
        "environment_start": environment,
        "environment_end": environment,
    }


def _worker(target: int, regime: str, role: str, output: Path) -> None:
    plan, commit, plan_artifact_sha256 = _require_plan_head()
    worker_id(target, regime, role)
    if output.exists():
        raise FileExistsError("large-scale training temporary worker output exists")
    environment = current_runtime_environment_contract()
    report = _base_report(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        commit=commit,
        target=target,
        regime=regime,
        role=role,
        environment=environment,
    )
    stage = "initialization"
    snapshots: list[dict[str, Any]] = []
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is unavailable")
        _require_operational_environment()
        torch.mps.set_per_process_memory_fraction(
            MAXIMUM_RECOMMENDED_MEMORY_FRACTION
        )
        recommended = int(torch.mps.recommended_max_memory())
        if recommended <= 0:
            raise RuntimeError("recommended MPS memory is unavailable")
        report["recommended_max_memory_bytes"] = recommended
        inputs, matrices = training_arrays()
        patches = matrices[role]
        stage = "model_build"
        model = build_main_model(
            large_scale_model_spec(target, 86),
            seed=MODEL_SEED,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        count = parameter_count(model)
        state = state_sha256(model)
        report["parameter_count"] = count
        report["model_state_sha256"] = state
        if (
            count != EXPECTED_PARAMETERS[target]
            or count != plan["models"][str(target)]["expected_parameter_count"]
            or state != plan["models"][str(target)]["model_state_sha256"]
        ):
            raise ValueError("large-scale training model identity differs")
        if regime == CHECKPOINTED_REGIME:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        elif regime != STANDARD_REGIME:
            raise ValueError("large-scale training regime differs")
        stage = "model_to_mps"
        with publication_mps_exclusive():
            model = model.to("mps")
            model.train()
            snapshots.append(_snapshot("model_resident"))
            stage = "optimizer_construct"
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                betas=BETAS,
                eps=EPSILON,
                weight_decay=WEIGHT_DECAY,
            )
            snapshots.append(_snapshot("optimizer_constructed"))
            stage = "warmup_update"
            warmup_seconds, warmup_finite = _update(
                model, optimizer, inputs, patches, update_index=0
            )
            if not warmup_finite or warmup_seconds <= 0 or not optimizer.state:
                raise RuntimeError("large-scale warmup optimizer update is invalid")
            report["optimizer_state_initialized"] = True
            snapshots.append(_snapshot("warmup_optimizer_state_initialized"))
            update_seconds: list[float] = []
            all_finite = True
            for measured in range(MEASUREMENT_UPDATES):
                stage = f"measurement_update_{measured}"
                elapsed, finite = _update(
                    model,
                    optimizer,
                    inputs,
                    patches,
                    update_index=WARMUP_UPDATES + measured,
                )
                update_seconds.append(elapsed)
                all_finite &= finite
                snapshots.append(_snapshot(stage))
            if not all_finite:
                raise RuntimeError("large-scale measured optimizer update is nonfinite")
        report["measurement"] = projected_training(update_seconds)
        report["finite"] = True
        report["completed"] = True
        report["failure"] = None
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
        report["measurement"] = None
        report["finite"] = False
        report["completed"] = False
        report["failure"] = {
            "category": type(exc).__name__,
            "message": str(exc)[-2000:],
            "returncode": 1,
            "stage": stage,
        }
    finally:
        report["memory_snapshots"] = snapshots
        report["maximum_driver_allocated_bytes"] = max(
            (row["driver_allocated_bytes"] for row in snapshots), default=0
        )
        try:
            report["environment_end"] = current_runtime_environment_contract()
        except (OSError, RuntimeError, TypeError, ValueError):
            report["environment_end"] = environment
        if report["environment_end"] != environment:
            report["measurement"] = None
            report["finite"] = False
            report["completed"] = False
            report["failure"] = {
                "category": "EnvironmentChanged",
                "message": "hardware/software environment changed during worker",
                "returncode": 1,
                "stage": "environment_end",
            }
        validate_worker_report(
            report,
            plan=plan,
            plan_artifact_sha256=plan_artifact_sha256,
            runner_git_commit=commit,
            target=target,
            regime=regime,
            role=role,
        )
        _publish(output, canonical_bytes(report), mode=0o600)
        del report
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _parent_failure(
    *,
    plan: dict[str, Any],
    plan_artifact_sha256: str,
    commit: str,
    target: int,
    regime: str,
    role: str,
    returncode: int,
    message: str,
) -> dict[str, Any]:
    report = _base_report(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        commit=commit,
        target=target,
        regime=regime,
        role=role,
        environment=plan["environment"],
    )
    report["failure"] = {
        "category": "WorkerProcessFailure",
        "message": message[-2000:],
        "returncode": returncode,
        "stage": "subprocess",
    }
    return report


def _run_all() -> None:
    plan, commit, plan_artifact_sha256 = _require_plan_head()
    if OUTPUT_PATH.exists() or _history(OUTPUT_PATH):
        raise FileExistsError("large-scale training summary was already published")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    active = canonical_bytes(
        {
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": plan["plan_sha256"],
            "runner_git_commit": commit,
        }
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active:
            raise ValueError("large-scale training active session differs")
    else:
        _publish(ACTIVE_PATH, active, mode=0o600)
    reports: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    for target, regime, role in worker_order():
        identifier = worker_id(target, regime, role)
        path = worker_report_path(target, regime, role)
        if path.exists():
            report = _read_json(path)
        else:
            temporary = ARTIFACT_ROOT / f".{identifier}.{os.getpid()}.worker"
            command = (
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-target",
                str(target),
                "--worker-regime",
                regime,
                "--worker-role",
                role,
                "--worker-output",
                str(temporary),
            )
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            if completed.returncode == 0 and temporary.is_file():
                report = _read_json(temporary)
            else:
                report = _parent_failure(
                    plan=plan,
                    plan_artifact_sha256=plan_artifact_sha256,
                    commit=commit,
                    target=target,
                    regime=regime,
                    role=role,
                    returncode=completed.returncode,
                    message=completed.stderr or completed.stdout or "no worker output",
                )
            validate_worker_report(
                report,
                plan=plan,
                plan_artifact_sha256=plan_artifact_sha256,
                runner_git_commit=commit,
                target=target,
                regime=regime,
                role=role,
            )
            if temporary.exists():
                temporary.replace(path)
            else:
                _publish(path, canonical_bytes(report), mode=0o600)
        validate_worker_report(
            report,
            plan=plan,
            plan_artifact_sha256=plan_artifact_sha256,
            runner_git_commit=commit,
            target=target,
            regime=regime,
            role=role,
        )
        reports[identifier] = report
        evidence[identifier] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
        print(f"resource_worker_complete={identifier}", flush=True)
    summary = build_summary(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        reports=reports,
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("large-scale training source changed during execution")
    _publish(OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    ACTIVE_PATH.unlink()
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-target", type=int)
    parser.add_argument("--worker-regime")
    parser.add_argument("--worker-role")
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    worker_values = (
        args.worker_target,
        args.worker_regime,
        args.worker_role,
        args.worker_output,
    )
    if any(value is not None for value in worker_values):
        if any(value is None for value in worker_values):
            parser.error("all worker arguments are required together")
        _worker(
            args.worker_target,
            args.worker_regime,
            args.worker_role,
            args.worker_output,
        )
    else:
        _run_all()


if __name__ == "__main__":
    main()
