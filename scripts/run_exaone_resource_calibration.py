#!/usr/bin/env python3
"""Run one sealed baseline-only EXAONE resource-calibration pass."""

from __future__ import annotations

import gc
import math
import os
import resource
import subprocess
from time import perf_counter

import mlx.core as mx
import numpy as np
from exaone_actual_runtime import (
    load_case_arrays,
    load_exaone_runtime,
    run_baseline_trial,
)
from exaone_resource_calibration import (
    ACTIVE_PATH,
    BASELINE_ARTIFACT_PATH,
    OUTPUT_TOKENS,
    PLAN_PATH,
    RESULT_PATH,
    ROOT,
    build_result,
    hash_file,
    read_plan,
    validate_baseline_arrays,
    validate_result,
)
from exaone_retrieval_data import canonical_bytes, npz_bytes

from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _publish(path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_never_published(path) -> None:
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(
            f"artifact was already published: {path.relative_to(ROOT)}"
        )


def _run_locked() -> dict:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE resource calibration requires a clean worktree")
    for path in (ACTIVE_PATH, BASELINE_ARTIFACT_PATH, RESULT_PATH):
        if path.exists():
            raise FileExistsError(f"EXAONE resource namespace is not empty: {path}")
    _require_never_published(RESULT_PATH)
    plan = read_plan(verify_derived=True)
    plan_blob = subprocess.check_output(
        ("git", "show", f"HEAD:{PLAN_PATH.relative_to(ROOT).as_posix()}"), cwd=ROOT
    )
    if plan_blob != PLAN_PATH.read_bytes():
        raise ValueError("EXAONE resource plan is not the exact HEAD blob")
    commit = _git("rev-parse", "HEAD")
    _publish(
        ACTIVE_PATH,
        canonical_bytes(
            {
                "plan_artifact_sha256": hash_file(PLAN_PATH),
                "plan_sha256": plan["plan_sha256"],
                "runner_git_commit": commit,
            }
        ),
        0o600,
    )
    bundle = None
    try:
        cases = load_case_arrays()
        load_started = perf_counter()
        bundle = load_exaone_runtime(load_table=False)
        mx.synchronize()
        model_load_seconds = perf_counter() - load_started
        active_before = int(mx.get_active_memory())
        cache_before = int(mx.get_cache_memory())
        mx.reset_peak_memory()
        trials = [
            run_baseline_trial(
                bundle,
                cases["prompt_token_ids"][index],
                output_tokens=OUTPUT_TOKENS,
            )
            for index in range(len(cases["prompt_token_ids"]))
        ]
        arrays = {
            "decoded_utf8_sha256": np.asarray(
                [list(bytes.fromhex(trial.decoded_utf8_sha256)) for trial in trials],
                dtype=np.uint8,
            ),
            "elapsed_ns": np.asarray(
                [trial.elapsed_ns for trial in trials], dtype=np.int64
            ),
            "output_token_ids": np.asarray(
                [trial.output_token_ids for trial in trials], dtype=np.uint32
            ),
            "output_token_sha256": np.asarray(
                [list(bytes.fromhex(trial.output_token_sha256)) for trial in trials],
                dtype=np.uint8,
            ),
            "prompt_token_count": np.asarray(
                [trial.prompt_token_count for trial in trials], dtype=np.uint16
            ),
            "target_generation_forward_calls": np.asarray(
                [trial.target_generation_forward_calls for trial in trials],
                dtype=np.uint16,
            ),
            "target_prefill_forward_calls": np.asarray(
                [trial.target_prefill_forward_calls for trial in trials],
                dtype=np.uint8,
            ),
        }
        validate_baseline_arrays(arrays)
        mx.synchronize()
        working_set = int(
            plan["environment"]["mlx"]["max_recommended_working_set_size"]
        )
        active_after = int(mx.get_active_memory())
        cache_after = int(mx.get_cache_memory())
        peak_active = int(mx.get_peak_memory())
        process_peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        maximum_observed = max(
            active_before + cache_before,
            active_after + cache_after,
            peak_active,
            process_peak_rss,
        )
        maximum_allowed = math.floor(
            plan["resource_contract"]["maximum_memory_fraction"] * working_set
        )
        memory = {
            "active_after_bytes": active_after,
            "active_before_bytes": active_before,
            "cache_after_bytes": cache_after,
            "cache_before_bytes": cache_before,
            "maximum_allowed_bytes": maximum_allowed,
            "maximum_observed_working_set_bytes": maximum_observed,
            "maximum_recommended_working_set_size": working_set,
            "peak_active_bytes": peak_active,
            "process_peak_rss_bytes": process_peak_rss,
            "safety_pass": bool(0 < maximum_observed <= maximum_allowed),
            "working_set_fraction": maximum_observed / working_set,
        }
        model_identity = {
            "model_files": bundle.model_files,
            "model_parameter_count": bundle.model_parameter_count,
            "retrieval_table_loaded": False,
            "table_resident_bytes": bundle.table_resident_bytes,
        }
        if _git("rev-parse", "HEAD") != commit or _git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("repository changed during EXAONE resource calibration")
        artifact_bytes = npz_bytes(arrays)
        result = build_result(
            plan=plan,
            runner_git_commit=commit,
            arrays=arrays,
            baseline_artifact_bytes=artifact_bytes,
            model_load_seconds=model_load_seconds,
            memory=memory,
            model_identity=model_identity,
        )
        _publish(BASELINE_ARTIFACT_PATH, artifact_bytes, 0o600)
        validate_result(result, plan=plan, verify_artifact=True)
        _publish(RESULT_PATH, canonical_bytes(result), 0o644)
        ACTIVE_PATH.unlink()
        return result
    finally:
        if bundle is not None:
            del bundle
        gc.collect()
        mx.clear_cache()
        mx.synchronize()


def main() -> None:
    with publication_mps_exclusive():
        result = _run_locked()
    decision = result["actual_schedule_decision"]
    print(f"status={result['status']}")
    print(f"summary_sha256={result['summary_sha256']}")
    print(f"selected_schedule={decision['selected']}")
    print("candidate timing and acceptance were not executed")


if __name__ == "__main__":
    main()
