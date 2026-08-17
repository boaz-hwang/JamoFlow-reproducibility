#!/usr/bin/env python3
"""Run the sealed 200M/400M/800M/1600M W72/C86 extrapolation."""

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
from scale_schedule_extrapolation_core import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    ATOL,
    CONTINUATION_BYTES,
    CORRECTNESS_PROMPTS,
    EXPECTED_PARAMETERS,
    GLOBAL_POSITION_LIMIT,
    INNER_REPETITIONS,
    MEASURED_PROMPTS,
    MODEL_SEED,
    OUTPUT_PATH,
    PLAN_PATH,
    PROMPT_BYTES,
    PROTOCOL_ID,
    ROOT,
    RTOL,
    SCHEDULE_ORDER,
    SESSION_ORDER,
    TARGET_ORDER,
    WARMUP_PROMPTS,
    array_sha256,
    build_scale_schedule_summary,
    canonical_bytes,
    large_scale_model_spec,
    mechanism_arrays,
    role_order,
    summarize_scale_schedule_extrapolation,
    validate_case_arrays,
    validate_plan,
    validate_scale_schedule_summary,
    worker_report_path,
    worker_timing_path,
)

from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import (
    IncrementalBltDecoder,
    structural_prefix_boundaries,
)
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _history(path: Path) -> tuple[str, ...]:
    value = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(line for line in value.splitlines() if line)


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


def _require_plan_commit() -> tuple[dict[str, Any], str]:
    status = _git("status", "--porcelain")
    if status:
        raise ValueError("scale-schedule execution requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix())
        != commit
    ):
        raise ValueError("scale-schedule plan must be the current HEAD commit")
    plan = _read_json(PLAN_PATH)
    validate_plan(
        plan,
        current_environment=current_runtime_environment_contract(),
    )
    return plan, commit


def _require_operational_environment() -> None:
    battery = subprocess.run(
        ("pmset", "-g", "batt"), check=False, capture_output=True, text=True
    )
    thermal = subprocess.run(
        ("pmset", "-g", "therm"), check=False, capture_output=True, text=True
    )
    thermal_text = thermal.stdout.lower()
    if (
        battery.returncode != 0
        or "drawing from 'ac power'" not in battery.stdout.lower()
        or thermal.returncode != 0
        or "no thermal warning level has been recorded" not in thermal_text
        or "no performance warning level has been recorded" not in thermal_text
    ):
        raise RuntimeError("scale-schedule operational environment is ineligible")


def _runtime(model: Any, schedule: str) -> IncrementalBltDecoder:
    if schedule == "c86":
        policy, patches = "causal_codepoint_grid", 86
    elif schedule == "w72":
        policy, patches = "causal_whitespace_grid", 72
    else:
        raise ValueError("scale-schedule runtime role differs")
    return IncrementalBltDecoder(
        model,
        policy,
        horizon=512,
        patch_count=patches,
        fixed_stride=6,
    )


def _offline_boundaries(observed: bytes, schedule: str) -> tuple[int, ...]:
    if schedule == "c86":
        policy, patches = "causal_codepoint_grid", 86
    elif schedule == "w72":
        policy, patches = "causal_whitespace_grid", 72
    else:
        raise ValueError("scale-schedule offline role differs")
    return structural_prefix_boundaries(
        observed,
        policy,
        horizon=512,
        patch_count=patches,
        fixed_stride=6,
    )


def _normalized_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or left.numel() == 0
        or not bool(torch.all(torch.isfinite(left)).item())
        or not bool(torch.all(torch.isfinite(right)).item())
    ):
        raise ValueError("scale-schedule correctness logits differ structurally")
    denominator = ATOL + RTOL * torch.abs(right)
    value = torch.max(torch.abs(left - right) / denominator)
    result = float(value.item())
    if not math.isfinite(result) or result < 0:
        raise ValueError("scale-schedule correctness error is nonfinite")
    return result


def _correctness(
    model: Any,
    prompts: np.ndarray,
    continuations: np.ndarray,
    *,
    schedule: str,
) -> dict[str, Any]:
    comparisons = 0
    argmax_exact = 0
    boundary_prefix_comparisons = 0
    maximum = 0.0
    boundary_exact = True
    offline_boundary_exact = True
    diagnostics_exact = True
    for index in range(CORRECTNESS_PROMPTS):
        prompt = bytes(prompts[WARMUP_PROMPTS + index])
        continuation = bytes(continuations[WARMUP_PROMPTS + index])
        sequential = _runtime(model, schedule)
        parallel = _runtime(model, schedule)
        with torch.inference_mode():
            left = sequential.prefill(prompt)
            right = parallel.prefill_parallel(prompt)
            maximum = max(maximum, _normalized_error(left, right))
            comparisons += 1
            argmax_exact += int(left.argmax().item() == right.argmax().item())
            expected_boundaries = _offline_boundaries(prompt, schedule)
            offline_boundary_exact &= (
                sequential.diagnostics.boundaries == expected_boundaries
                and parallel.diagnostics.boundaries == expected_boundaries
            )
            boundary_prefix_comparisons += 1
            observed = bytearray(prompt)
            for byte in continuation[:-1]:
                left = sequential.consume(byte)
                right = parallel.consume(byte)
                maximum = max(maximum, _normalized_error(left, right))
                comparisons += 1
                argmax_exact += int(left.argmax().item() == right.argmax().item())
                observed.append(byte)
                expected_boundaries = _offline_boundaries(bytes(observed), schedule)
                offline_boundary_exact &= (
                    sequential.diagnostics.boundaries == expected_boundaries
                    and parallel.diagnostics.boundaries == expected_boundaries
                )
                boundary_prefix_comparisons += 1
        left_diagnostics = sequential.diagnostics
        right_diagnostics = parallel.diagnostics
        boundary_exact &= left_diagnostics.boundaries == right_diagnostics.boundaries
        diagnostics_exact &= left_diagnostics == right_diagnostics
    return {
        "argmax_comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "boundary_prefix_comparisons": boundary_prefix_comparisons,
        "boundary_trace_exact": bool(boundary_exact),
        "cache_diagnostics_exact": bool(diagnostics_exact),
        "maximum_normalized_logit_error": maximum,
        "offline_boundary_prefix_exact": bool(offline_boundary_exact),
    }


def _trial(
    model: Any,
    prompt: bytes,
    continuation: bytes,
    schedule: str,
    expected_boundaries: tuple[int, ...],
) -> tuple[float, int, int, int]:
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        runtime = _runtime(model, schedule)
        runtime.prefill_parallel(prompt)
        for byte in continuation[:-1]:
            runtime.consume(byte)
        torch.mps.synchronize()
        driver_allocated = int(torch.mps.driver_allocated_memory())
        recommended_max = int(torch.mps.recommended_max_memory())
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    diagnostics = runtime.diagnostics
    counters = runtime.runtime_counters
    expected = PROMPT_BYTES + CONTINUATION_BYTES - 1
    if (
        diagnostics.observed_bytes != expected
        or diagnostics.local_encoder_cached_bytes != expected
        or diagnostics.local_decoder_cached_bytes != expected
        or diagnostics.global_cached_patches != diagnostics.emitted_data_patches
        or diagnostics.boundaries != expected_boundaries
        or counters.parallel_prefill_calls != 1
        or counters.main_consume_calls != CONTINUATION_BYTES - 1
        or counters.selector_observed_bytes != expected
        or counters.router_forward_calls != 0
        or counters.router_scored_bytes != 0
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise ValueError("scale-schedule timed runtime invariant differs")
    if driver_allocated <= 0 or recommended_max <= 0:
        raise ValueError("scale-schedule synchronized memory snapshot differs")
    return (
        elapsed,
        diagnostics.emitted_data_patches,
        driver_allocated,
        recommended_max,
    )


def _worker(target: int, session: str) -> None:
    plan, commit = _require_plan_commit()
    if target not in TARGET_ORDER or session not in SESSION_ORDER:
        raise ValueError("scale-schedule worker identity differs")
    timing_path = worker_timing_path(target, session)
    report_path = worker_report_path(target, session)
    if timing_path.exists() or report_path.exists():
        raise FileExistsError("scale-schedule worker evidence already exists")
    prompts, continuations = validate_case_arrays(plan)
    all_patch_counts, all_boundary_hashes = mechanism_arrays(prompts, continuations)
    environment_start = current_runtime_environment_contract()
    _require_operational_environment()
    spec = large_scale_model_spec(target, 86)
    model = build_main_model(
        spec,
        seed=MODEL_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    count = parameter_count(model)
    state = state_sha256(model)
    expected_model = plan["models"][str(target)]
    if (
        count != EXPECTED_PARAMETERS[target]
        or count != expected_model["expected_parameter_count"]
        or spec.to_dict() != expected_model["spec"]
        or state != expected_model["model_state_sha256"]
    ):
        raise ValueError("scale-schedule worker model identity differs")

    target_index = TARGET_ORDER.index(target)
    session_index = SESSION_ORDER.index(session)
    timings = np.empty(
        (MEASURED_PROMPTS, INNER_REPETITIONS, len(SCHEDULE_ORDER)),
        dtype=np.float64,
    )
    first_role = np.empty((MEASURED_PROMPTS, INNER_REPETITIONS), dtype=np.uint8)
    memory_snapshots: list[tuple[int, int]] = []
    with publication_mps_exclusive(), torch.inference_mode():
        model = model.to("mps")
        model.eval()
        memory_snapshots.append(
            (
                int(torch.mps.driver_allocated_memory()),
                int(torch.mps.recommended_max_memory()),
            )
        )
        correctness = {
            schedule: _correctness(
                model,
                prompts,
                continuations,
                schedule=schedule,
            )
            for schedule in SCHEDULE_ORDER
        }
        for case in range(WARMUP_PROMPTS):
            for schedule_index, schedule in enumerate(SCHEDULE_ORDER):
                observed = bytes(prompts[case]) + bytes(continuations[case][:-1])
                elapsed, emitted, driver_allocated, recommended_max = _trial(
                    model,
                    bytes(prompts[case]),
                    bytes(continuations[case]),
                    schedule,
                    _offline_boundaries(observed, schedule),
                )
                memory_snapshots.append((driver_allocated, recommended_max))
                if elapsed <= 0 or emitted != all_patch_counts[case, schedule_index]:
                    raise ValueError("scale-schedule warmup mechanism differs")
        for prompt_index in range(MEASURED_PROMPTS):
            source_index = WARMUP_PROMPTS + prompt_index
            prompt = bytes(prompts[source_index])
            continuation = bytes(continuations[source_index])
            observed = prompt + continuation[:-1]
            for repetition in range(INNER_REPETITIONS):
                order = role_order(
                    target_index,
                    session_index,
                    prompt_index,
                    repetition,
                )
                first_role[prompt_index, repetition] = order[0]
                for schedule_index in order:
                    elapsed, emitted, driver_allocated, recommended_max = _trial(
                        model,
                        prompt,
                        continuation,
                        SCHEDULE_ORDER[schedule_index],
                        _offline_boundaries(observed, SCHEDULE_ORDER[schedule_index]),
                    )
                    memory_snapshots.append((driver_allocated, recommended_max))
                    if emitted != all_patch_counts[source_index, schedule_index]:
                        raise ValueError("scale-schedule timed patch count differs")
                    timings[prompt_index, repetition, schedule_index] = elapsed
        model = model.to("cpu")
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    environment_end = current_runtime_environment_contract()
    _require_operational_environment()
    if environment_end != environment_start:
        raise ValueError("scale-schedule worker environment changed")
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("scale-schedule source changed during worker execution")

    arrays = {
        "boundary_hashes": all_boundary_hashes[WARMUP_PROMPTS:],
        "first_role": first_role,
        "patch_counts": all_patch_counts[WARMUP_PROMPTS:],
        "timings_ms": timings,
    }
    timing_bytes = _npz_bytes(arrays)
    report = {
        "schema_version": 1,
        "kind": "scale_schedule_extrapolation_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "target_millions": target,
        "session_id": session,
        "runner_git_commit": commit,
        "plan_sha256": plan["plan_sha256"],
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "parameter_count": count,
        "model_state_sha256": state,
        "same_model_object_for_both_schedules": True,
        "correctness": correctness,
        "maximum_driver_allocated_bytes": max(row[0] for row in memory_snapshots),
        "recommended_max_memory_bytes": min(row[1] for row in memory_snapshots),
        "environment_start": environment_start,
        "environment_end": environment_end,
        "timing_artifact": {
            "path": timing_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(timing_bytes).hexdigest(),
            "boundary_hashes_array_sha256": array_sha256(arrays["boundary_hashes"]),
            "timings_array_sha256": array_sha256(timings),
            "first_role_array_sha256": array_sha256(first_role),
            "patch_counts_array_sha256": array_sha256(arrays["patch_counts"]),
        },
        "completed": True,
    }
    report_bytes = canonical_bytes(report)
    _publish(timing_path, timing_bytes, mode=0o600)
    _publish(report_path, report_bytes, mode=0o600)


def _load_worker(
    target: int,
    session: str,
    *,
    plan: Mapping[str, Any],
    commit: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    timing_path = worker_timing_path(target, session)
    report_path = worker_report_path(target, session)
    if not timing_path.is_file() or not report_path.is_file():
        raise ValueError("scale-schedule worker evidence is incomplete")
    if timing_path.is_symlink() or report_path.is_symlink():
        raise ValueError("scale-schedule worker evidence cannot be a symlink")
    report = _read_json(report_path)
    expected_keys = {
        "completed",
        "correctness",
        "environment_end",
        "environment_start",
        "kind",
        "maximum_driver_allocated_bytes",
        "model_state_sha256",
        "parameter_count",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "recommended_max_memory_bytes",
        "runner_git_commit",
        "same_model_object_for_both_schedules",
        "schema_version",
        "session_id",
        "target_millions",
        "timing_artifact",
    }
    if (
        set(report) != expected_keys
        or report["schema_version"] != 1
        or report["kind"] != "scale_schedule_extrapolation_worker_v1"
        or report["protocol_id"] != PROTOCOL_ID
        or report["target_millions"] != target
        or report["session_id"] != session
        or report["runner_git_commit"] != commit
        or report["plan_sha256"] != plan["plan_sha256"]
        or report["plan_artifact_sha256"] != hash_file(PLAN_PATH)
        or report["parameter_count"]
        != plan["models"][str(target)]["expected_parameter_count"]
        or report["model_state_sha256"]
        != plan["models"][str(target)]["model_state_sha256"]
        or report["same_model_object_for_both_schedules"] is not True
        or report["environment_start"] != plan["environment"]
        or report["environment_end"] != plan["environment"]
        or report["completed"] is not True
    ):
        raise ValueError(f"scale-schedule worker identity differs: {target}")
    timing_bytes = timing_path.read_bytes()
    artifact = report["timing_artifact"]
    if (
        not isinstance(artifact, Mapping)
        or set(artifact)
        != {
            "boundary_hashes_array_sha256",
            "first_role_array_sha256",
            "patch_counts_array_sha256",
            "path",
            "sha256",
            "timings_array_sha256",
        }
        or artifact["path"] != timing_path.relative_to(ROOT).as_posix()
        or artifact["sha256"] != hashlib.sha256(timing_bytes).hexdigest()
    ):
        raise ValueError(f"scale-schedule timing artifact differs: {target}")
    with np.load(io.BytesIO(timing_bytes), allow_pickle=False) as source:
        if set(source.files) != {
            "boundary_hashes",
            "first_role",
            "patch_counts",
            "timings_ms",
        }:
            raise ValueError("scale-schedule timing NPZ schema differs")
        boundary_hashes = np.ascontiguousarray(source["boundary_hashes"])
        timings = np.ascontiguousarray(source["timings_ms"])
        first_role = np.ascontiguousarray(source["first_role"])
        patch_counts = np.ascontiguousarray(source["patch_counts"])
    prompts, continuations = validate_case_arrays(plan)
    expected_counts, expected_hashes = mechanism_arrays(prompts, continuations)
    expected_counts = expected_counts[WARMUP_PROMPTS:]
    expected_hashes = expected_hashes[WARMUP_PROMPTS:]
    if (
        timings.dtype != np.float64
        or timings.shape != (MEASURED_PROMPTS, INNER_REPETITIONS, len(SCHEDULE_ORDER))
        or not np.all(np.isfinite(timings))
        or np.any(timings <= 0)
        or first_role.dtype != np.uint8
        or first_role.shape != (MEASURED_PROMPTS, INNER_REPETITIONS)
        or patch_counts.dtype != np.int64
        or patch_counts.shape != (MEASURED_PROMPTS, len(SCHEDULE_ORDER))
        or boundary_hashes.dtype != np.uint8
        or boundary_hashes.shape != (MEASURED_PROMPTS, len(SCHEDULE_ORDER), 32)
        or not np.array_equal(patch_counts, expected_counts)
        or not np.array_equal(boundary_hashes, expected_hashes)
        or artifact["timings_array_sha256"] != array_sha256(timings)
        or artifact["first_role_array_sha256"] != array_sha256(first_role)
        or artifact["patch_counts_array_sha256"] != array_sha256(patch_counts)
        or artifact["boundary_hashes_array_sha256"] != array_sha256(boundary_hashes)
    ):
        raise ValueError(f"scale-schedule timing arrays differ: {target}")
    target_index = TARGET_ORDER.index(target)
    session_index = SESSION_ORDER.index(session)
    expected_first = np.asarray(
        [
            [
                role_order(
                    target_index,
                    session_index,
                    prompt,
                    repetition,
                )[0]
                for repetition in range(INNER_REPETITIONS)
            ]
            for prompt in range(MEASURED_PROMPTS)
        ],
        dtype=np.uint8,
    )
    if not np.array_equal(first_role, expected_first):
        raise ValueError(f"scale-schedule role order differs: {target}")
    patch_count_summary = {
        schedule: {
            "maximum": int(np.max(patch_counts[:, schedule_index])),
            "median": float(np.median(patch_counts[:, schedule_index])),
            "minimum": int(np.min(patch_counts[:, schedule_index])),
            "sum": int(np.sum(patch_counts[:, schedule_index])),
        }
        for schedule_index, schedule in enumerate(SCHEDULE_ORDER)
    }
    projection = {
        "correctness": report["correctness"],
        "environment_end": report["environment_end"],
        "environment_start": report["environment_start"],
        "maximum_driver_allocated_bytes": report["maximum_driver_allocated_bytes"],
        "model_state_sha256": report["model_state_sha256"],
        "parameter_count": report["parameter_count"],
        "patch_count_summary": patch_count_summary,
        "recommended_max_memory_bytes": report["recommended_max_memory_bytes"],
        "same_model_object_for_both_schedules": report[
            "same_model_object_for_both_schedules"
        ],
        "session_id": session,
        "target_millions": target,
    }
    evidence = {
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "report_sha256": hash_file(report_path),
        "timing_path": timing_path.relative_to(ROOT).as_posix(),
        "timing_sha256": hash_file(timing_path),
    }
    return timings, projection, evidence


def _run_all() -> None:
    plan, commit = _require_plan_commit()
    if OUTPUT_PATH.exists() or _history(OUTPUT_PATH):
        raise FileExistsError("scale-schedule summary exists or has Git history")
    if ACTIVE_PATH.exists() or ARTIFACT_ROOT.exists():
        raise FileExistsError("scale-schedule artifact namespace already exists")
    active = {
        "protocol_id": PROTOCOL_ID,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "runner_git_commit": commit,
        "session_order": list(SESSION_ORDER),
        "target_order": list(TARGET_ORDER),
    }
    _publish(ACTIVE_PATH, canonical_bytes(active), mode=0o600)

    for target in TARGET_ORDER:
        for session in SESSION_ORDER:
            completed = subprocess.run(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-target",
                    str(target),
                    "--worker-session",
                    session,
                ),
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"scale-schedule worker failed for {target}M/{session}:\n"
                    f"{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}"
                )

    timings_by_target: dict[int, np.ndarray] = {}
    reports_by_target: dict[int, tuple[Mapping[str, Any], ...]] = {}
    evidence_by_target: dict[str, Any] = {}
    for target in TARGET_ORDER:
        timing_rows: list[np.ndarray] = []
        projection_rows: list[Mapping[str, Any]] = []
        evidence_rows: dict[str, Any] = {}
        for session in SESSION_ORDER:
            timings, projection, evidence = _load_worker(
                target,
                session,
                plan=plan,
                commit=commit,
            )
            timing_rows.append(timings)
            projection_rows.append(projection)
            evidence_rows[session] = evidence
        timings_by_target[target] = np.stack(timing_rows, axis=0)
        reports_by_target[target] = tuple(projection_rows)
        evidence_by_target[str(target)] = evidence_rows
    aggregate = summarize_scale_schedule_extrapolation(
        timings_by_target=timings_by_target,
        reports_by_target=reports_by_target,
    )
    summary = build_scale_schedule_summary(
        plan_artifact_sha256=hash_file(PLAN_PATH),
        plan_sha256=plan["plan_sha256"],
        summary_base_git_commit=commit,
        worker_evidence=evidence_by_target,
        aggregate=aggregate,
    )
    validate_scale_schedule_summary(summary)
    validate_plan(
        plan,
        current_environment=current_runtime_environment_contract(),
    )
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("scale-schedule source changed before summary publication")
    _publish(OUTPUT_PATH, canonical_bytes(summary), mode=0o644)
    ACTIVE_PATH.unlink()
    expected = f"?? {OUTPUT_PATH.relative_to(ROOT).as_posix()}"
    if _git("status", "--porcelain", "--untracked-files=all") != expected:
        raise ValueError("scale-schedule summary is not the only workspace change")
    primary = aggregate["rows"][str(TARGET_ORDER[-1])]
    print(f"status={aggregate['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")
    print(f"1600m_median_reduction={primary['median_reduction']:.9f}")
    print(f"1600m_bootstrap_95_interval={primary['prompt_bootstrap_95_interval']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-target", type=int, choices=TARGET_ORDER)
    parser.add_argument("--worker-session", choices=SESSION_ORDER)
    arguments = parser.parse_args()
    if arguments.worker_target is None and arguments.worker_session is None:
        _run_all()
    elif arguments.worker_target is None or arguments.worker_session is None:
        parser.error("worker target and session must be provided together")
    else:
        _worker(arguments.worker_target, arguments.worker_session)


if __name__ == "__main__":
    main()
