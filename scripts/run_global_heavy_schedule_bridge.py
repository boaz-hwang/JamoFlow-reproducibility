#!/usr/bin/env python3
"""Run the sealed fixed-geometry global-heavy W72/C86 bridge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from global_heavy_schedule_core import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    EXPECTED_GLOBAL_PARAMETER_COUNT,
    EXPECTED_GLOBAL_PARAMETER_SHARE,
    EXPECTED_PARAMETER_COUNT,
    GLOBAL_HEAVY_SPEC,
    GLOBAL_POSITION_LIMIT,
    INNER_REPETITIONS,
    MEASURED_PROMPTS,
    MODEL_SEED,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    SCHEDULE_ORDER,
    SESSION_ORDER,
    WARMUP_PROMPTS,
    build_summary,
    canonical_bytes,
    load_case_arrays,
    mechanism_arrays,
    role_order,
    summarize,
    validate_plan,
    worker_report_path,
    worker_timing_path,
)
from run_scale_schedule_extrapolation import (
    _correctness,
    _offline_boundaries,
    _require_operational_environment,
    _trial,
)
from scale_schedule_extrapolation_core import array_sha256

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _history(path: Path) -> tuple[str, ...]:
    raw = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(line for line in raw.splitlines() if line)


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


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _require_plan_head() -> tuple[dict[str, Any], str, str]:
    if _git("status", "--porcelain"):
        raise ValueError("global-heavy execution requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix())
        != commit
    ):
        raise ValueError("global-heavy plan must be current HEAD")
    plan = _read_json(PLAN_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    upstream = plan["upstream"]
    for path_key, hash_key in (
        ("balanced_summary_path", "balanced_summary_artifact_sha256"),
        ("resource_summary_path", "resource_summary_artifact_sha256"),
    ):
        if hash_file(ROOT / upstream[path_key]) != upstream[hash_key]:
            raise ValueError("global-heavy upstream artifact changed")
    return plan, commit, hash_file(PLAN_PATH)


def _worker(session: str, output_report: Path, output_timing: Path) -> None:
    plan, commit, plan_artifact_sha256 = _require_plan_head()
    if session not in SESSION_ORDER or output_report.exists() or output_timing.exists():
        raise ValueError("global-heavy worker identity or output differs")
    prompts, continuations, _, _, _ = load_case_arrays()
    if plan["cases"]["prompts_array_sha256"] != array_sha256(prompts):
        raise ValueError("global-heavy case arrays differ")
    patch_counts, boundary_hashes = mechanism_arrays(prompts, continuations)
    environment_start = current_runtime_environment_contract()
    _require_operational_environment()
    model = build_main_model(
        GLOBAL_HEAVY_SPEC,
        seed=MODEL_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    count = parameter_count(model)
    global_count = sum(
        parameter.numel()
        for parameter in model.model.global_transformer.parameters()
    )
    state = state_sha256(model)
    if (
        count != EXPECTED_PARAMETER_COUNT
        or global_count != EXPECTED_GLOBAL_PARAMETER_COUNT
        or state != plan["model"]["model_state_sha256"]
    ):
        raise ValueError("global-heavy worker model identity differs")
    timings = np.empty(
        (MEASURED_PROMPTS, INNER_REPETITIONS, len(SCHEDULE_ORDER)),
        dtype=np.float64,
    )
    first_role = np.empty((MEASURED_PROMPTS, INNER_REPETITIONS), dtype=np.uint8)
    memory: list[tuple[int, int]] = []
    session_index = SESSION_ORDER.index(session)
    with publication_mps_exclusive(), torch.inference_mode():
        model = model.to("mps")
        model.eval()
        memory.append(
            (
                int(torch.mps.driver_allocated_memory()),
                int(torch.mps.recommended_max_memory()),
            )
        )
        correctness = {
            role: _correctness(model, prompts, continuations, schedule=role)
            for role in SCHEDULE_ORDER
        }
        for case in range(WARMUP_PROMPTS):
            observed = bytes(prompts[case]) + bytes(continuations[case][:-1])
            for role_index, role in enumerate(SCHEDULE_ORDER):
                elapsed, emitted, driver, recommended = _trial(
                    model,
                    bytes(prompts[case]),
                    bytes(continuations[case]),
                    role,
                    _offline_boundaries(observed, role),
                )
                if elapsed <= 0 or emitted != patch_counts[case, role_index]:
                    raise ValueError("global-heavy warmup mechanism differs")
                memory.append((driver, recommended))
        for prompt_index in range(MEASURED_PROMPTS):
            source_index = WARMUP_PROMPTS + prompt_index
            prompt = bytes(prompts[source_index])
            continuation = bytes(continuations[source_index])
            observed = prompt + continuation[:-1]
            for repetition in range(INNER_REPETITIONS):
                order = role_order(session_index, prompt_index, repetition)
                first_role[prompt_index, repetition] = order[0]
                for role_index in order:
                    role = SCHEDULE_ORDER[role_index]
                    elapsed, emitted, driver, recommended = _trial(
                        model,
                        prompt,
                        continuation,
                        role,
                        _offline_boundaries(observed, role),
                    )
                    if emitted != patch_counts[source_index, role_index]:
                        raise ValueError("global-heavy measured mechanism differs")
                    timings[prompt_index, repetition, role_index] = elapsed
                    memory.append((driver, recommended))
        model = model.to("cpu")
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
    environment_end = current_runtime_environment_contract()
    _require_operational_environment()
    if environment_end != environment_start:
        raise ValueError("global-heavy worker environment changed")
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("global-heavy source changed during worker")
    arrays = {
        "boundary_hashes": boundary_hashes[WARMUP_PROMPTS:],
        "first_role": first_role,
        "patch_counts": patch_counts[WARMUP_PROMPTS:],
        "timings_ms": timings,
    }
    timing_bytes = _npz_bytes(arrays)
    report = {
        "schema_version": 1,
        "kind": "global_heavy_schedule_bridge_worker_v2",
        "protocol_id": PROTOCOL_ID,
        "session_id": session,
        "runner_git_commit": commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_sha256": plan_artifact_sha256,
        "parameter_count": count,
        "global_parameter_count": global_count,
        "global_parameter_share": global_count / count,
        "model_state_sha256": state,
        "same_model_object_for_both_schedules": True,
        "correctness": correctness,
        "maximum_driver_allocated_bytes": max(row[0] for row in memory),
        "recommended_max_memory_bytes": min(row[1] for row in memory),
        "environment_start": environment_start,
        "environment_end": environment_end,
        "timing_artifact": {
            "path": worker_timing_path(session).relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(timing_bytes).hexdigest(),
            "boundary_hashes_array_sha256": array_sha256(arrays["boundary_hashes"]),
            "first_role_array_sha256": array_sha256(first_role),
            "patch_counts_array_sha256": array_sha256(arrays["patch_counts"]),
            "timings_array_sha256": array_sha256(timings),
        },
        "completed": True,
    }
    _publish(output_timing, timing_bytes, mode=0o600)
    _publish(output_report, canonical_bytes(report), mode=0o600)


def _load_worker(
    session: str,
    *,
    plan: Mapping[str, Any],
    runner_commit: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    report_path = worker_report_path(session)
    timing_path = worker_timing_path(session)
    if (
        not report_path.is_file()
        or not timing_path.is_file()
        or report_path.is_symlink()
        or timing_path.is_symlink()
    ):
        raise ValueError("global-heavy worker evidence is incomplete")
    report = _read_json(report_path)
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "global_heavy_schedule_bridge_worker_v2"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("session_id") != session
        or report.get("runner_git_commit") != runner_commit
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or report.get("global_parameter_count") != EXPECTED_GLOBAL_PARAMETER_COUNT
        or report.get("global_parameter_share") != EXPECTED_GLOBAL_PARAMETER_SHARE
        or report.get("model_state_sha256") != plan["model"]["model_state_sha256"]
        or report.get("same_model_object_for_both_schedules") is not True
        or report.get("environment_start") != plan["environment"]
        or report.get("environment_end") != plan["environment"]
        or report.get("completed") is not True
    ):
        raise ValueError("global-heavy worker report identity differs")
    timing_bytes = timing_path.read_bytes()
    artifact = report["timing_artifact"]
    if (
        artifact.get("path") != timing_path.relative_to(ROOT).as_posix()
        or artifact.get("sha256") != hashlib.sha256(timing_bytes).hexdigest()
    ):
        raise ValueError("global-heavy timing artifact identity differs")
    with np.load(io.BytesIO(timing_bytes), allow_pickle=False) as source:
        if set(source.files) != {
            "boundary_hashes",
            "first_role",
            "patch_counts",
            "timings_ms",
        }:
            raise ValueError("global-heavy timing NPZ schema differs")
        arrays = {name: np.ascontiguousarray(source[name]) for name in source.files}
    prompts, continuations, _, _, _ = load_case_arrays()
    expected_counts, expected_hashes = mechanism_arrays(prompts, continuations)
    expected_first = np.empty((MEASURED_PROMPTS, INNER_REPETITIONS), dtype=np.uint8)
    session_index = SESSION_ORDER.index(session)
    for prompt in range(MEASURED_PROMPTS):
        for repetition in range(INNER_REPETITIONS):
            expected_first[prompt, repetition] = role_order(
                session_index, prompt, repetition
            )[0]
    if (
        not np.array_equal(arrays["patch_counts"], expected_counts[WARMUP_PROMPTS:])
        or not np.array_equal(
            arrays["boundary_hashes"], expected_hashes[WARMUP_PROMPTS:]
        )
        or not np.array_equal(arrays["first_role"], expected_first)
        or artifact["boundary_hashes_array_sha256"]
        != array_sha256(arrays["boundary_hashes"])
        or artifact["first_role_array_sha256"] != array_sha256(expected_first)
        or artifact["patch_counts_array_sha256"]
        != array_sha256(arrays["patch_counts"])
        or artifact["timings_array_sha256"] != array_sha256(arrays["timings_ms"])
    ):
        raise ValueError("global-heavy timing arrays differ")
    projection = {
        "session_id": session,
        "parameter_count": report["parameter_count"],
        "global_parameter_count": report["global_parameter_count"],
        "global_parameter_share": report["global_parameter_share"],
        "same_model_object_for_both_schedules": True,
        "correctness": report["correctness"],
        "maximum_driver_allocated_bytes": report["maximum_driver_allocated_bytes"],
        "recommended_max_memory_bytes": report["recommended_max_memory_bytes"],
        "environment_start": report["environment_start"],
        "environment_end": report["environment_end"],
    }
    evidence = {
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "report_sha256": hash_file(report_path),
        "timing_path": timing_path.relative_to(ROOT).as_posix(),
        "timing_sha256": hash_file(timing_path),
    }
    return arrays["timings_ms"], projection, evidence


def _run_all() -> None:
    plan, commit, plan_artifact_sha256 = _require_plan_head()
    if OUTPUT_PATH.exists() or _history(OUTPUT_PATH):
        raise FileExistsError("global-heavy summary was already published")
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
            raise ValueError("global-heavy active session differs")
    else:
        _publish(ACTIVE_PATH, active, mode=0o600)
    timing_rows: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    for session in SESSION_ORDER:
        report_path = worker_report_path(session)
        timing_path = worker_timing_path(session)
        if not report_path.exists() and not timing_path.exists():
            temporary_report = ARTIFACT_ROOT / f".{session}.{os.getpid()}.report"
            temporary_timing = ARTIFACT_ROOT / f".{session}.{os.getpid()}.npz"
            completed = subprocess.run(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-session",
                    session,
                    "--worker-report",
                    str(temporary_report),
                    "--worker-timing",
                    str(temporary_timing),
                ),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"global-heavy worker failed: {session}: {completed.stderr[-2000:]}"
                )
            temporary_report.replace(report_path)
            temporary_timing.replace(timing_path)
        elif not report_path.exists() or not timing_path.exists():
            raise ValueError("global-heavy partial worker evidence exists")
        timing, report, row_evidence = _load_worker(
            session, plan=plan, runner_commit=commit
        )
        timing_rows.append(timing)
        reports.append(report)
        evidence[session] = row_evidence
        print(f"global_heavy_worker_complete={session}", flush=True)
    aggregate = summarize(np.stack(timing_rows), reports)
    summary = build_summary(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        aggregate=aggregate,
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("global-heavy source changed during execution")
    _publish(OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    ACTIVE_PATH.unlink()
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-session")
    parser.add_argument("--worker-report", type=Path)
    parser.add_argument("--worker-timing", type=Path)
    args = parser.parse_args()
    values = (args.worker_session, args.worker_report, args.worker_timing)
    if any(value is not None for value in values):
        if any(value is None for value in values):
            parser.error("all global-heavy worker arguments are required")
        _worker(args.worker_session, args.worker_report, args.worker_timing)
    else:
        _run_all()


if __name__ == "__main__":
    main()
