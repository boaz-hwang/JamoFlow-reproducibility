#!/usr/bin/env python3
"""Run the post-v5r3 exploratory 2x2 incremental component profile."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import (
    IncrementalBltDecoder,
    IncrementalStructuralSelector,
)
from jamoflow.inference_actual_runtime_v5 import (
    ACTUAL_INFERENCE_PATCH_HORIZON,
    LoadedActualModel,
    load_actual_model,
    model_spec_for_descriptor,
    release_actual_model,
)
from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_CASE_PATH,
    ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
    ACTUAL_INFERENCE_V5_PLAN_PATH,
    ACTUAL_INFERENCE_V5_PROMPT_BYTES,
    ACTUAL_INFERENCE_V5_SUMMARY_PATH,
    array_sha256,
    current_runtime_environment_contract,
    validate_actual_inference_plan_v5,
)
from inference_component_profile_core import (
    COMPONENTS,
    PROFILE_CHECKPOINT_ROLES,
    PROFILE_COMPONENT_CASES,
    PROFILE_DECODE_BYTES,
    PROFILE_PROTOCOL_ID,
    PROFILE_SCHEDULES,
    PROFILE_SEEDS,
    PROFILE_WARMUP_CASES,
    PROFILE_WHOLE_CASES,
    PROFILE_WHOLE_REPETITIONS,
    WHOLE_METRICS,
    summarize_profile_arrays,
)
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_QUALITY_LOCK_PATH,
    SELECTION_LOCK_PATH,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / ACTUAL_INFERENCE_V5_PLAN_PATH
CASE_PATH = ROOT / ACTUAL_INFERENCE_V5_CASE_PATH
SUMMARY_PATH = ROOT / ACTUAL_INFERENCE_V5_SUMMARY_PATH
AUTHORIZATION_PATH = ROOT / FINAL_AUTHORIZATION_PATH
QUALITY_PATH = ROOT / FINAL_QUALITY_LOCK_PATH
SELECTION_PATH = ROOT / SELECTION_LOCK_PATH
MANIFEST_PATH = ROOT / "data/manifests/exploratory-component-profile-v1.json"
RAW_PATH = ROOT / "artifacts/exploratory-component-profile-v1/raw-profile.npz"
OUTPUT_PATH = ROOT / "results/exploratory-component-profile-v1/summary.json"
MACHINE_LOCK_PATH = Path("/tmp/jamoflow-publication-mps.lock")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _publish_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_commit() -> str:
    if _command("git", "status", "--porcelain"):
        raise ValueError("component profiling requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("component profiling requires a Git commit")
    return commit


def _require_ac_power() -> str:
    state = _command("pmset", "-g", "batt")
    if "Now drawing from 'AC Power'" not in state:
        raise RuntimeError("component profiling requires AC power")
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


@contextmanager
def _machine_lock() -> Iterator[None]:
    with MACHINE_LOCK_PATH.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another JamoFlow MPS process is live") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _synchronize() -> None:
    torch.mps.synchronize()


def _load_context() -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    plan = _read_json(PLAN_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)
    quality = _read_json(QUALITY_PATH)
    selection = _read_json(SELECTION_PATH)
    validate_selection_lock_v2(selection)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection,
    )
    validate_actual_inference_plan_v5(
        plan,
        quality_lock=quality,
        authorization=authorization,
    )
    if hash_file(CASE_PATH) != plan["case_context"]["artifact_sha256"]:
        raise ValueError("profile case artifact differs from v5r3 plan")
    with np.load(CASE_PATH, allow_pickle=False) as archive:
        if set(archive.files) != {"prompts", "replay_continuations"}:
            raise ValueError("profile case schema differs")
        prompts = archive["prompts"]
        continuations = archive["replay_continuations"]
    if (
        prompts.dtype != np.uint8
        or continuations.dtype != np.uint8
        or prompts.shape[1] != ACTUAL_INFERENCE_V5_PROMPT_BYTES
        or continuations.shape[1] != ACTUAL_INFERENCE_V5_CONTINUATION_BYTES
        or array_sha256(prompts) != plan["case_context"]["prompt_array_sha256"]
        or array_sha256(continuations)
        != plan["case_context"]["continuation_array_sha256"]
    ):
        raise ValueError("profile case arrays differ")
    prior = _read_json(SUMMARY_PATH)
    if (
        prior.get("plan_sha256") != plan["plan_sha256"]
        or prior.get("gate", {}).get("status")
        != "fail_matched_quality_actual_efficiency_v5r3"
    ):
        raise ValueError("component profile requires the completed v5r3 result")
    for path, expected in plan["implementation_sha256"].items():
        target = ROOT / path
        if target.exists() and hash_file(target) != expected:
            raise ValueError(f"v5r3 implementation changed before profiling: {path}")
    return plan, authorization, prompts, continuations


def _model_identity(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    identity = plan["timing_pair"]["roles"][role]["model_identity_sha256"]
    matches = [
        item for item in authorization["models"] if item["identity_sha256"] == identity
    ]
    if len(matches) != 1:
        raise ValueError("profile model identity differs")
    return matches[0]


def _runtime(bundle: LoadedActualModel, schedule: str) -> IncrementalBltDecoder:
    spec = PROFILE_SCHEDULES[schedule]
    return IncrementalBltDecoder(
        bundle.model,
        str(spec["policy"]),
        horizon=ACTUAL_INFERENCE_PATCH_HORIZON,
        patch_count=int(spec["patch_count"]),
        fixed_stride=model_spec_for_descriptor(bundle.descriptor).patch_stride,
    )


def _assert_runtime(runtime: IncrementalBltDecoder, observed: int) -> None:
    diagnostics = runtime.diagnostics
    if (
        diagnostics.observed_bytes != observed
        or diagnostics.local_encoder_cached_bytes != observed
        or diagnostics.local_decoder_cached_bytes != observed
        or diagnostics.global_cached_patches != diagnostics.emitted_data_patches
    ):
        raise AssertionError("profile runtime cache invariant differs")


def _whole_trial(
    bundle: LoadedActualModel,
    schedule: str,
    prompt: bytes,
    continuation: bytes,
) -> tuple[np.ndarray, int, int]:
    _synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        runtime = _runtime(bundle, schedule)
        logits = runtime.prefill_parallel(prompt)
        _synchronize()
        prefilled = time.perf_counter_ns()
        prompt_patches = runtime.diagnostics.emitted_data_patches
        for value in continuation[:-1]:
            logits = runtime.consume(value)
        _synchronize()
        finished = time.perf_counter_ns()
    _assert_runtime(runtime, len(prompt) + len(continuation) - 1)
    final_patches = runtime.diagnostics.emitted_data_patches
    result = np.asarray(
        [
            (prefilled - started) / 1_000_000,
            (finished - prefilled) / 1_000_000,
            (finished - started) / 1_000_000,
        ],
        dtype=np.float64,
    )
    del runtime, logits
    return result, int(prompt_patches), int(final_patches)


@contextmanager
def _component_recorder(
    runtime: IncrementalBltDecoder,
) -> Iterator[dict[str, list[float]]]:
    records = {component: [] for component in COMPONENTS}
    originals: dict[str, Any] = {}

    def wrap(original: Any, component: str):
        def measured(*args: Any, **kwargs: Any) -> Any:
            _synchronize()
            started = time.perf_counter_ns()
            value = original(*args, **kwargs)
            _synchronize()
            records[component].append((time.perf_counter_ns() - started) / 1_000_000)
            return value

        return measured

    method_components = {
        "_advance_local_encoder": "local_encoder",
        "_finalize_encoder_patch": "patch_finalize_global",
        "_advance_local_decoder": "local_decoder",
    }
    for name, component in method_components.items():
        originals[name] = getattr(runtime, name)
        setattr(runtime, name, wrap(originals[name], component))
    lm_head = runtime.model.lm_head
    originals["lm_head_forward"] = lm_head.forward
    lm_head.forward = wrap(originals["lm_head_forward"], "lm_head")
    try:
        yield records
    finally:
        for name in method_components:
            setattr(runtime, name, originals[name])
        lm_head.forward = originals["lm_head_forward"]


def _step_trial(
    bundle: LoadedActualModel,
    schedule: str,
    prompt: bytes,
    continuation: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    runtime = _runtime(bundle, schedule)
    with torch.inference_mode():
        runtime.prefill_parallel(prompt)
        _synchronize()
        values: list[float] = []
        boundaries: list[bool] = []
        for value in continuation[:-1]:
            before = runtime.diagnostics.emitted_data_patches
            started = time.perf_counter_ns()
            runtime.consume(value)
            _synchronize()
            values.append((time.perf_counter_ns() - started) / 1_000_000)
            boundaries.append(runtime.diagnostics.emitted_data_patches == before + 1)
    _assert_runtime(runtime, len(prompt) + len(continuation) - 1)
    if len(values) != PROFILE_DECODE_BYTES:
        raise AssertionError("profile step count differs")
    return np.asarray(values, dtype=np.float64), np.asarray(boundaries, dtype=np.bool_)


def _component_trial(
    bundle: LoadedActualModel,
    schedule: str,
    prompt: bytes,
    continuation: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    runtime = _runtime(bundle, schedule)
    with torch.inference_mode():
        runtime.prefill_parallel(prompt)
        _synchronize()
        with _component_recorder(runtime) as records:
            for value in continuation[:-1]:
                runtime.consume(value)
    _assert_runtime(runtime, len(prompt) + len(continuation) - 1)
    totals = np.asarray(
        [sum(records[name]) for name in COMPONENTS],
        dtype=np.float64,
    )
    calls = np.asarray([len(records[name]) for name in COMPONENTS], dtype=np.int64)
    if tuple(calls[[0, 2, 3]]) != (PROFILE_DECODE_BYTES,) * 3 or calls[1] <= 0:
        raise AssertionError("profile component call counts differ")
    return totals, calls


def _selector_benchmark(prompts: np.ndarray, continuations: np.ndarray) -> dict[str, Any]:
    rows = [
        bytes(prompt) + bytes(continuation[:-1])
        for prompt, continuation in zip(prompts, continuations, strict=True)
    ]
    result: dict[str, Any] = {}
    for schedule, spec in PROFILE_SCHEDULES.items():
        samples = []
        for _ in range(100):
            started = time.perf_counter_ns()
            for row in rows:
                selector = IncrementalStructuralSelector(
                    str(spec["policy"]),
                    horizon=ACTUAL_INFERENCE_PATCH_HORIZON,
                    patch_count=int(spec["patch_count"]),
                    fixed_stride=6,
                )
                for value in row:
                    selector.consume(value)
            elapsed = time.perf_counter_ns() - started
            samples.append(elapsed / sum(len(row) for row in rows))
        result[schedule] = {
            "median_ns_per_byte": float(np.median(samples)),
            "repetitions": 100,
        }
    return result


def _sync_overhead() -> dict[str, Any]:
    samples = []
    _synchronize()
    for _ in range(100):
        started = time.perf_counter_ns()
        _synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "idle_synchronize_median_ms": float(np.median(samples)),
        "idle_synchronize_p95_ms": float(np.percentile(samples, 95)),
        "repetitions": 100,
    }


def run() -> None:
    if RAW_PATH.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("component profile output already exists")
    commit = _require_clean_commit()
    power_sha256 = _require_ac_power()
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("component profiling requires Apple MPS")
    plan, authorization, prompts, continuations = _load_context()
    start = PROFILE_WARMUP_CASES
    whole_prompts = prompts[start : start + PROFILE_WHOLE_CASES]
    whole_continuations = continuations[start : start + PROFILE_WHOLE_CASES]
    component_prompts = whole_prompts[:PROFILE_COMPONENT_CASES]
    component_continuations = whole_continuations[:PROFILE_COMPONENT_CASES]
    prefix = (
        len(PROFILE_SEEDS),
        len(PROFILE_CHECKPOINT_ROLES),
        len(PROFILE_SCHEDULES),
    )
    whole = np.zeros(
        (*prefix, PROFILE_WHOLE_CASES, PROFILE_WHOLE_REPETITIONS, len(WHOLE_METRICS)),
        dtype=np.float64,
    )
    patch_shape = (*prefix, PROFILE_WHOLE_CASES, PROFILE_WHOLE_REPETITIONS)
    prompt_patches = np.zeros(patch_shape, dtype=np.int64)
    final_patches = np.zeros(patch_shape, dtype=np.int64)
    step_ms = np.zeros(
        (*prefix, PROFILE_COMPONENT_CASES, PROFILE_DECODE_BYTES),
        dtype=np.float64,
    )
    step_boundary = np.zeros(step_ms.shape, dtype=np.bool_)
    component_total = np.zeros(
        (*prefix, PROFILE_COMPONENT_CASES, len(COMPONENTS)),
        dtype=np.float64,
    )
    component_calls = np.zeros(component_total.shape, dtype=np.int64)
    schedules = tuple(PROFILE_SCHEDULES)

    with _machine_lock():
        for seed_index, seed in enumerate(PROFILE_SEEDS):
            role_order = (
                PROFILE_CHECKPOINT_ROLES
                if seed_index % 2 == 0
                else tuple(reversed(PROFILE_CHECKPOINT_ROLES))
            )
            for role in role_order:
                role_index = PROFILE_CHECKPOINT_ROLES.index(role)
                bundle = load_actual_model(
                    role=role,
                    identity=_model_identity(authorization, plan, role),
                    seed=seed,
                    device="mps",
                )
                try:
                    for warmup_index in range(PROFILE_WARMUP_CASES):
                        for schedule in schedules:
                            _whole_trial(
                                bundle,
                                schedule,
                                bytes(prompts[warmup_index]),
                                bytes(continuations[warmup_index]),
                            )
                    for case_index, (prompt, continuation) in enumerate(
                        zip(whole_prompts, whole_continuations, strict=True)
                    ):
                        for repetition in range(PROFILE_WHOLE_REPETITIONS):
                            order = (
                                schedules
                                if (seed_index + role_index + case_index + repetition) % 2 == 0
                                else tuple(reversed(schedules))
                            )
                            for schedule in order:
                                schedule_index = schedules.index(schedule)
                                values, prompt_count, final_count = _whole_trial(
                                    bundle,
                                    schedule,
                                    bytes(prompt),
                                    bytes(continuation),
                                )
                                index = (
                                    seed_index,
                                    role_index,
                                    schedule_index,
                                    case_index,
                                    repetition,
                                )
                                whole[index] = values
                                prompt_patches[index] = prompt_count
                                final_patches[index] = final_count
                    for case_index, (prompt, continuation) in enumerate(
                        zip(component_prompts, component_continuations, strict=True)
                    ):
                        order = (
                            schedules
                            if (seed_index + role_index + case_index) % 2 == 0
                            else tuple(reversed(schedules))
                        )
                        for schedule in order:
                            schedule_index = schedules.index(schedule)
                            steps, boundaries = _step_trial(
                                bundle,
                                schedule,
                                bytes(prompt),
                                bytes(continuation),
                            )
                            step_ms[
                                seed_index, role_index, schedule_index, case_index
                            ] = steps
                            step_boundary[
                                seed_index, role_index, schedule_index, case_index
                            ] = boundaries
                            totals, calls = _component_trial(
                                bundle,
                                schedule,
                                bytes(prompt),
                                bytes(continuation),
                            )
                            component_total[
                                seed_index, role_index, schedule_index, case_index
                            ] = totals
                            component_calls[
                                seed_index, role_index, schedule_index, case_index
                            ] = calls
                    print(f"profiled seed={seed} checkpoint={role}", flush=True)
                finally:
                    release_actual_model(bundle)
                gc.collect()
        sync_overhead = _sync_overhead()

    arrays = {
        "whole_ms": whole,
        "prompt_patches": prompt_patches,
        "final_patches": final_patches,
        "step_ms": step_ms,
        "step_boundary": step_boundary,
        "component_total_ms": component_total,
        "component_calls": component_calls,
    }
    aggregates = summarize_profile_arrays(**arrays)
    selector = _selector_benchmark(whole_prompts, whole_continuations)
    raw_bytes = _npz_bytes(arrays)
    manifest = _read_json(MANIFEST_PATH)
    expected_manifest = {
        "case_source": "exact v5r3 sealed cases; profiling is post-outcome exploratory",
        "checkpoint_roles": list(PROFILE_CHECKPOINT_ROLES),
        "component_cases": PROFILE_COMPONENT_CASES,
        "decode_observed_bytes": PROFILE_DECODE_BYTES,
        "device": "mps",
        "mps_event_used": False,
        "profile_kind": "two_by_two_checkpoint_x_runtime_schedule",
        "protocol_id": PROFILE_PROTOCOL_ID,
        "repetitions": PROFILE_WHOLE_REPETITIONS,
        "schedules": {
            name: dict(value) for name, value in PROFILE_SCHEDULES.items()
        },
        "seeds": list(PROFILE_SEEDS),
        "synchronization_contract": {
            "component_measurement": "explicit pre/post MPS synchronize; diagnostic only",
            "step_measurement": "one post-consume MPS synchronize per byte; diagnostic only",
            "whole_trial": "same prefill/decode synchronization boundaries as controlled v5r3",
        },
        "warmup_cases": PROFILE_WARMUP_CASES,
        "whole_trial_cases": PROFILE_WHOLE_CASES,
    }
    if manifest != expected_manifest:
        raise ValueError("component profile manifest differs")
    prior = _read_json(SUMMARY_PATH)
    summary: dict[str, Any] = {
        "aggregates": aggregates,
        "claim_scope": {
            "confirmatory": False,
            "final_test_blind": False,
            "purpose": "post-v5r3 exploratory bottleneck diagnosis",
            "synchronized_components_are_production_latency_shares": False,
        },
        "complete": True,
        "environment": current_runtime_environment_contract(),
        "git_commit": commit,
        "kind": "exploratory_incremental_component_profile_v1",
        "manifest_artifact_sha256": hash_file(MANIFEST_PATH),
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "power_snapshot_sha256": power_sha256,
        "protocol_id": PROFILE_PROTOCOL_ID,
        "raw_artifact": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
        "schema_version": 1,
        "selector_cpu_diagnostic": selector,
        "source_v5r3_summary": {
            "artifact_sha256": hash_file(SUMMARY_PATH),
            "gate_status": prior["gate"]["status"],
            "summary_sha256": prior["summary_sha256"],
        },
        "synchronization_overhead": sync_overhead,
    }
    summary["summary_sha256"] = hashlib.sha256(
        _json_bytes(summary)
    ).hexdigest()
    summary_bytes = _json_bytes(summary)
    if _require_clean_commit() != commit or _require_ac_power() != power_sha256:
        raise RuntimeError("repository or power changed during component profile")
    _publish_no_clobber(RAW_PATH, raw_bytes)
    _publish_no_clobber(OUTPUT_PATH, summary_bytes)
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
