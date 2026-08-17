#!/usr/bin/env python3
"""Run exactly one fresh-process EXAONE retrieval actual-timing session."""

from __future__ import annotations

import gc
import hashlib
import math
import os
import resource
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from exaone_actual_runtime import load_case_arrays, load_exaone_runtime
from exaone_retrieval_actual import (
    ARTIFACT_ROOT,
    BASELINE_ROLE_INDEX,
    CANDIDATE_ROLE_INDEX,
    COUNTER_NAMES,
    INNER_REPETITIONS,
    MAXIMUM_DRAFT_TOKENS,
    MAXIMUM_MEMORY_FRACTION,
    MEASURED_CASES,
    OUTPUT_TOKENS,
    PLAN_PATH,
    ROLES,
    SESSION_ARRAY_NAMES,
    SESSION_ARTIFACT_ROOT,
    SESSION_RECEIPT_ROOT,
    SESSIONS,
    SUMMARY_PATH,
    TIMING_NAMES,
    WARMUP_CASES,
    actual_mps_exclusive,
    assert_canonical_workspace_path,
    balanced_role_order,
    build_session_receipt,
    measured_case_order,
    read_plan,
    read_session_receipt,
    require_distinct_git_commits,
    session_active_path,
    session_artifact_path,
    session_receipt_path,
    validate_session_arrays,
    validate_session_receipt,
    warmup_case_order,
)
from exaone_retrieval_actual_runtime import (
    ActualGenerationTrial,
    run_actual_baseline_trial,
    run_actual_candidate_trial,
)
from exaone_retrieval_data import (
    ROOT,
    canonical_bytes,
    canonical_sha256,
    hash_file,
    is_sha256,
    npz_bytes,
)

LOCAL_MPS_MARKERS = (
    "benchmark_inference_actual_v5.py",
    "benchmark_phase3_actual_inference.py",
    "measure_inference_memory_v5.py",
    "reconstruct_inference_calibration_v2.py",
    "reconstruct_inference_confirmation_calibration_v2.py",
    "run_inference_final_quality_v2.py",
    "run_phase1.py",
    "run_phase2.py",
    "run_phase2_controls.py",
    "run_phase2_ecological.py",
    "run_phase2_generation.py",
    "run_phase2_normalization.py",
    "run_phase3.py",
    "run_phase3_compute_conversion.py",
    "run_phase3_ecological.py",
    "run_phase3_generation.py",
    "run_phase3_mechanism.py",
    "run_phase3_normalization.py",
    "run_phase3_ood.py",
    "run_exaone_resource_calibration.py",
    "run_exaone_retrieval_actual_session.py",
    "seal_inference_final_quality_lock_v2.py",
    "seal_inference_selection_lock_v2.py",
    "summarize_exaone_retrieval_actual.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _publish(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _history(path: Path) -> tuple[str, ...]:
    value = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(line for line in value.splitlines() if line)


def _require_exact_head_blob(path: Path) -> None:
    payload = subprocess.check_output(
        ("git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"), cwd=ROOT
    )
    if payload != path.read_bytes():
        raise ValueError(
            f"artifact is not the exact HEAD blob: {path.relative_to(ROOT)}"
        )


def _require_ancestor(ancestor: str, descendant: str, *, label: str) -> None:
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=ROOT,
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError(f"EXAONE actual Git chronology differs: {label}")


def _ancestor_pids(rows: list[tuple[int, int, str]]) -> set[int]:
    parents = {pid: parent for pid, parent, _ in rows}
    output = {os.getpid()}
    cursor = os.getpid()
    while cursor in parents and parents[cursor] > 0 and parents[cursor] not in output:
        cursor = parents[cursor]
        output.add(cursor)
    return output


def _operational_environment() -> dict[str, Any]:
    battery = subprocess.run(
        ("pmset", "-g", "batt"), check=False, capture_output=True, text=True
    )
    thermal = subprocess.run(
        ("pmset", "-g", "therm"), check=False, capture_output=True, text=True
    )
    process = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,command="),
        check=False,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, int, str]] = []
    for line in process.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        try:
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
        except ValueError:
            continue
    exempt = _ancestor_pids(rows)
    conflicts = [
        command
        for pid, _, command in rows
        if pid not in exempt and any(marker in command for marker in LOCAL_MPS_MARKERS)
    ]
    thermal_output = thermal.stdout.lower()
    value = {
        "ac_power": bool(
            battery.returncode == 0
            and "drawing from 'ac power'" in battery.stdout.lower()
        ),
        "battery_sha256": hashlib.sha256(battery.stdout.encode("utf-8")).hexdigest(),
        "conflicting_process_count": len(conflicts),
        "process_inventory_pass": bool(
            process.returncode == 0
            and process.stdout.strip()
            and rows
            and os.getpid() in {row[0] for row in rows}
            and not conflicts
        ),
        "process_inventory_sha256": hashlib.sha256(
            process.stdout.encode("utf-8")
        ).hexdigest(),
        "thermal_pass": bool(
            thermal.returncode == 0
            and "no thermal warning level has been recorded" in thermal_output
            and "no performance warning level has been recorded" in thermal_output
        ),
        "thermal_sha256": hashlib.sha256(thermal.stdout.encode("utf-8")).hexdigest(),
    }
    if not (
        value["ac_power"] and value["process_inventory_pass"] and value["thermal_pass"]
    ):
        raise RuntimeError("EXAONE actual operational environment is ineligible")
    return value


def _process_start_token() -> str:
    value = subprocess.check_output(
        ("ps", "-p", str(os.getpid()), "-o", "lstart="), text=True
    ).strip()
    if not value:
        raise ValueError("EXAONE actual process start token is empty")
    return canonical_sha256({"pid": os.getpid(), "process_start": value})


def _completed_prefix(plan: Mapping[str, Any], *, head: str) -> int:
    plan_history = _history(PLAN_PATH)
    if len(plan_history) != 1:
        raise ValueError("EXAONE actual plan publication history differs")
    _require_exact_head_blob(PLAN_PATH)
    if plan["git_commit_before_plan"] == plan_history[0]:
        raise ValueError("EXAONE actual plan was not published after its base commit")
    _require_ancestor(
        plan["git_commit_before_plan"],
        plan_history[0],
        label="implementation base to plan publication",
    )
    _require_ancestor(plan_history[0], head, label="plan to session")
    seen_missing = False
    completed = 0
    preceding_publication_commit = plan_history[0]
    for index in range(SESSIONS):
        receipt_path = session_receipt_path(index)
        artifact_path = session_artifact_path(index)
        active_path = session_active_path(index)
        if active_path.exists():
            raise RuntimeError(
                "unfinished EXAONE actual session requires forensic review"
            )
        receipt_exists = receipt_path.exists()
        artifact_exists = artifact_path.exists()
        if receipt_exists != artifact_exists:
            raise RuntimeError("partial EXAONE actual session requires forensic review")
        if not receipt_exists:
            seen_missing = True
            if _history(receipt_path):
                raise RuntimeError("deleted EXAONE actual receipt cannot be reissued")
            continue
        if seen_missing:
            raise ValueError("EXAONE actual completed sessions are not a prefix")
        history = _history(receipt_path)
        if len(history) != 1:
            raise ValueError("EXAONE actual receipt publication history differs")
        _require_exact_head_blob(receipt_path)
        receipt = read_session_receipt(index, plan=plan, verify_artifact=True)
        _require_ancestor(
            preceding_publication_commit,
            receipt["runner_git_commit"],
            label="preceding evidence publication to next run",
        )
        _require_ancestor(
            plan_history[0], receipt["runner_git_commit"], label="plan to run"
        )
        require_distinct_git_commits(
            receipt["runner_git_commit"], history[0], label="run to own receipt"
        )
        _require_ancestor(
            receipt["runner_git_commit"], history[0], label="run to receipt"
        )
        _require_ancestor(history[0], head, label="receipt to next session")
        preceding_publication_commit = history[0]
        completed += 1
    if SUMMARY_PATH.exists() or _history(SUMMARY_PATH):
        raise RuntimeError("EXAONE actual summary already exists or was deleted")
    expected_artifacts = {session_artifact_path(index) for index in range(completed)}
    actual_artifacts = (
        {
            path
            for path in SESSION_ARTIFACT_ROOT.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if SESSION_ARTIFACT_ROOT.exists()
        else set()
    )
    expected_receipts = {session_receipt_path(index) for index in range(completed)}
    actual_receipts = (
        {
            path
            for path in SESSION_RECEIPT_ROOT.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if SESSION_RECEIPT_ROOT.exists()
        else set()
    )
    if actual_artifacts != expected_artifacts or actual_receipts != expected_receipts:
        raise ValueError("EXAONE actual evidence namespace differs")
    return completed


def _empty_arrays(session_index: int) -> dict[str, np.ndarray]:
    cell_shape = (MEASURED_CASES, INNER_REPETITIONS, len(ROLES))
    arrays: dict[str, np.ndarray] = {
        "case_order": np.asarray(measured_case_order(session_index), dtype=np.uint16),
        "decoded_utf8_sha256": np.zeros(cell_shape + (32,), dtype=np.uint8),
        "first_role": np.zeros((MEASURED_CASES, INNER_REPETITIONS), dtype=np.uint8),
        "output_token_ids": np.zeros(cell_shape + (OUTPUT_TOKENS,), dtype=np.uint32),
        "output_token_sha256": np.zeros(cell_shape + (32,), dtype=np.uint8),
        "peak_active_bytes": np.zeros(cell_shape, dtype=np.uint64),
    }
    arrays.update({name: np.zeros(cell_shape, dtype=np.int64) for name in TIMING_NAMES})
    arrays.update(
        {name: np.zeros(cell_shape, dtype=np.uint16) for name in COUNTER_NAMES}
    )
    if set(arrays) != set(SESSION_ARRAY_NAMES):
        raise AssertionError("EXAONE actual empty array set differs")
    return arrays


def _trial_for_role(bundle, prompt_ids, role_index: int) -> ActualGenerationTrial:
    if role_index == BASELINE_ROLE_INDEX:
        return run_actual_baseline_trial(
            bundle, prompt_ids, output_tokens=OUTPUT_TOKENS
        )
    if role_index == CANDIDATE_ROLE_INDEX:
        return run_actual_candidate_trial(
            bundle,
            prompt_ids,
            output_tokens=OUTPUT_TOKENS,
            maximum_draft_tokens=MAXIMUM_DRAFT_TOKENS,
        )
    raise ValueError("EXAONE actual role index differs")


def _pair_exact(left: ActualGenerationTrial, right: ActualGenerationTrial) -> None:
    if (
        left.output_token_ids != right.output_token_ids
        or left.output_token_sha256 != right.output_token_sha256
        or left.decoded_utf8_sha256 != right.decoded_utf8_sha256
    ):
        raise ValueError("EXAONE actual baseline/candidate output differs")


def _record_trial(
    arrays: dict[str, np.ndarray],
    *,
    case_index: int,
    repetition: int,
    role_index: int,
    trial: ActualGenerationTrial,
    peak_active_bytes: int,
) -> None:
    if (
        len(trial.output_token_ids) != OUTPUT_TOKENS
        or any(not 0 <= int(value) < 2**32 for value in trial.output_token_ids)
        or not is_sha256(trial.output_token_sha256)
        or not is_sha256(trial.decoded_utf8_sha256)
        or peak_active_bytes <= 0
    ):
        raise ValueError("EXAONE actual trial output differs")
    index = (case_index, repetition, role_index)
    arrays["output_token_ids"][index] = np.asarray(
        trial.output_token_ids, dtype=np.uint32
    )
    arrays["output_token_sha256"][index] = np.frombuffer(
        bytes.fromhex(trial.output_token_sha256), dtype=np.uint8
    )
    arrays["decoded_utf8_sha256"][index] = np.frombuffer(
        bytes.fromhex(trial.decoded_utf8_sha256), dtype=np.uint8
    )
    arrays["peak_active_bytes"][index] = np.uint64(peak_active_bytes)
    for name in TIMING_NAMES:
        value = int(getattr(trial, name))
        if value <= 0:
            raise ValueError("EXAONE actual trial timing differs")
        arrays[name][index] = value
    for name in COUNTER_NAMES:
        value = int(getattr(trial, name))
        if not 0 <= value < 2**16:
            raise ValueError("EXAONE actual trial counter exceeds uint16")
        arrays[name][index] = value


def _warmup(bundle, cases: Mapping[str, np.ndarray], session_index: int) -> str:
    rows: list[dict[str, Any]] = []
    for case_index in warmup_case_order(session_index):
        trials: dict[int, ActualGenerationTrial] = {}
        for role_index in balanced_role_order(session_index, case_index, 0):
            trials[role_index] = _trial_for_role(
                bundle, cases["prompt_token_ids"][case_index], role_index
            )
        _pair_exact(trials[BASELINE_ROLE_INDEX], trials[CANDIDATE_ROLE_INDEX])
        rows.append(
            {
                "case_index": case_index,
                "decoded_utf8_sha256": trials[BASELINE_ROLE_INDEX].decoded_utf8_sha256,
                "output_token_sha256": trials[BASELINE_ROLE_INDEX].output_token_sha256,
            }
        )
    return canonical_sha256({"session_index": session_index, "warmup": rows})


def _measure(
    bundle,
    cases: Mapping[str, np.ndarray],
    session_index: int,
    checkpoints: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    arrays = _empty_arrays(session_index)
    for position, case_index in enumerate(measured_case_order(session_index)):
        prompt = cases["prompt_token_ids"][WARMUP_CASES + case_index]
        for repetition in range(INNER_REPETITIONS):
            role_order = balanced_role_order(session_index, case_index, repetition)
            arrays["first_role"][case_index, repetition] = role_order[0]
            trials: dict[int, ActualGenerationTrial] = {}
            for role_index in role_order:
                mx.synchronize()
                mx.reset_peak_memory()
                trial = _trial_for_role(bundle, prompt, role_index)
                peak = int(mx.get_peak_memory())
                _record_trial(
                    arrays,
                    case_index=case_index,
                    repetition=repetition,
                    role_index=role_index,
                    trial=trial,
                    peak_active_bytes=peak,
                )
                trials[role_index] = trial
            _pair_exact(trials[BASELINE_ROLE_INDEX], trials[CANDIDATE_ROLE_INDEX])
        if (position + 1) % 16 == 0:
            checkpoints.append(_operational_environment())
    validate_session_arrays(arrays, session_index=session_index)
    return arrays


def _memory(
    *,
    plan: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    before: tuple[int, int],
) -> dict[str, Any]:
    mx.synchronize()
    active_after = int(mx.get_active_memory())
    cache_after = int(mx.get_cache_memory())
    peak_active = int(np.max(arrays["peak_active_bytes"]))
    process_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    working_set = int(plan["environment"]["mlx"]["max_recommended_working_set_size"])
    maximum_observed = max(
        before[0] + before[1],
        active_after + cache_after,
        peak_active,
        process_peak,
    )
    maximum_allowed = math.floor(MAXIMUM_MEMORY_FRACTION * working_set)
    return {
        "active_after_bytes": active_after,
        "active_before_bytes": before[0],
        "cache_after_bytes": cache_after,
        "cache_before_bytes": before[1],
        "maximum_allowed_bytes": maximum_allowed,
        "maximum_observed_working_set_bytes": maximum_observed,
        "maximum_recommended_working_set_size": working_set,
        "peak_active_bytes": peak_active,
        "process_peak_rss_bytes": process_peak,
        "safety_pass": bool(0 < maximum_observed <= maximum_allowed),
        "working_set_fraction": maximum_observed / working_set,
    }


def _run_locked() -> int:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE actual session requires a clean worktree")
    for path in (ARTIFACT_ROOT, SESSION_RECEIPT_ROOT, PLAN_PATH, SUMMARY_PATH):
        assert_canonical_workspace_path(path)
    head = _git("rev-parse", "HEAD")
    plan = read_plan(verify_derived=True)
    session_index = _completed_prefix(plan, head=head)
    if session_index == SESSIONS:
        print("all five EXAONE actual sessions are complete")
        return 0
    active_path = session_active_path(session_index)
    artifact_path = session_artifact_path(session_index)
    receipt_path = session_receipt_path(session_index)
    for path in (active_path, artifact_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"EXAONE actual session path exists: {path}")
    if _history(receipt_path):
        raise RuntimeError("EXAONE actual receipt cannot be reissued")

    process_token = _process_start_token()
    start = _operational_environment()
    active_payload = canonical_bytes(
        {
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "plan_sha256": plan["plan_sha256"],
            "process_start_token_sha256": process_token,
            "runner_git_commit": head,
            "session_index": session_index,
        }
    )
    _publish(active_path, active_payload, 0o600)
    bundle = None
    try:
        cases = load_case_arrays()
        bundle = load_exaone_runtime(load_table=True)
        mx.synchronize()
        before = (int(mx.get_active_memory()), int(mx.get_cache_memory()))
        checkpoints = [start]
        warmup_root = _warmup(bundle, cases, session_index)
        checkpoints.append(_operational_environment())
        arrays = _measure(bundle, cases, session_index, checkpoints)
        end = _operational_environment()
        checkpoints.append(end)
        if len(checkpoints) != 7:
            raise AssertionError("EXAONE actual checkpoint count differs")
        memory = _memory(plan=plan, arrays=arrays, before=before)
        model_identity = {
            "model_files": bundle.model_files,
            "model_parameter_count": bundle.model_parameter_count,
            "retrieval_table_loaded": True,
            "table_resident_bytes": bundle.table_resident_bytes,
        }
        artifact_bytes = npz_bytes(arrays)
        receipt = build_session_receipt(
            plan=plan,
            session_index=session_index,
            runner_git_commit=head,
            process_start_token_sha256=process_token,
            arrays=arrays,
            artifact_bytes=artifact_bytes,
            model_identity=model_identity,
            operational_start=start,
            operational_end=end,
            operational_checkpoints=checkpoints,
            memory=memory,
            warmup_output_root_sha256=warmup_root,
        )
        if read_plan(verify_derived=True) != plan:
            raise RuntimeError(
                "EXAONE actual plan or environment changed during timing"
            )
        if head != _git("rev-parse", "HEAD") or _git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("repository changed during EXAONE actual timing")
        _publish(artifact_path, artifact_bytes, 0o600)
        validate_session_receipt(
            receipt, plan=plan, session_index=session_index, verify_artifact=True
        )
        _publish(receipt_path, canonical_bytes(receipt), 0o644)
        active_path.unlink()
        print(f"session_index={session_index}")
        print(f"receipt_sha256={receipt['receipt_sha256']}")
        print("correctness_pass=true; performance remains sealed")
        return 0
    finally:
        if bundle is not None:
            del bundle
        gc.collect()
        mx.clear_cache()
        mx.synchronize()


def main() -> None:
    with actual_mps_exclusive():
        raise SystemExit(_run_locked())


if __name__ == "__main__":
    main()
