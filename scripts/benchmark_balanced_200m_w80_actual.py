#!/usr/bin/env python3
"""Run one fresh-process W80/C86 actual-inference timing session."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import balanced_200m_trained_core as base
from balanced_200m_w80_core import (
    ARTIFACT_ROOT,
    CANDIDATE_ROLE,
    CONTINUATION_BYTES,
    MAXIMUM_FREE_OUTPUT_BYTES,
    PLAN_PATH,
    PROTOCOL_ID,
    PROMPT_BYTES,
    REFERENCE_ROLE,
    ROOT,
    SCALE_PLAN_PATH,
    TIMING_ATOL,
    TIMING_CORRECTNESS_PROMPTS,
    TIMING_MEASURED_PROMPTS,
    TIMING_MODE_ORDER,
    TIMING_REPETITIONS,
    TIMING_ROLE_ORDER,
    TIMING_RTOL,
    TIMING_SESSION_ORDER,
    TIMING_WARMUP_PROMPTS,
    VERIFICATION_OUTPUT_PATH,
    checkpoint_path,
    canonical_bytes,
    timing_array_path,
    timing_report_path,
    timing_role_order,
    validate_plan,
    validate_verification_receipt,
)
from scale_schedule_extrapolation_core import array_sha256, large_scale_model_spec, validate_case_arrays

from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import IncrementalBltDecoder, structural_prefix_boundaries
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive, state_sha256
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    StrictUtf8State,
    advance_strict_utf8,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _head_blob(path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)


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


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _operational() -> None:
    battery = subprocess.run(("pmset", "-g", "batt"), check=False, capture_output=True, text=True)
    thermal = subprocess.run(("pmset", "-g", "therm"), check=False, capture_output=True, text=True)
    text = thermal.stdout.lower()
    if (
        battery.returncode != 0
        or "drawing from 'ac power'" not in battery.stdout.lower()
        or thermal.returncode != 0
        or "no thermal warning level has been recorded" not in text
        or "no performance warning level has been recorded" not in text
    ):
        raise RuntimeError("balanced-200M W80 timing environment is ineligible")


def _context(session: str) -> tuple[dict[str, Any], dict[str, Any], str, int]:
    if session not in TIMING_SESSION_ORDER or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 timing context differs")
    commit = _git("rev-parse", "HEAD")
    plan = _read(PLAN_PATH)
    verification = _read(VERIFICATION_OUTPUT_PATH)
    if _head_blob(PLAN_PATH) != PLAN_PATH.read_bytes() or _head_blob(VERIFICATION_OUTPUT_PATH) != VERIFICATION_OUTPUT_PATH.read_bytes():
        raise ValueError("balanced-200M W80 timing dependencies are not exact HEAD blobs")
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_verification_receipt(verification)
    if (
        verification.get("plan_sha256") != plan["plan_sha256"]
        or verification.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or verification.get("actual_timing_authorized") is not True
    ):
        raise ValueError("balanced-200M W80 quality does not authorize timing")
    index = TIMING_SESSION_ORDER.index(session)
    for prior in TIMING_SESSION_ORDER[:index]:
        path = timing_report_path(prior)
        history = _history(path)
        if not path.is_file() or path.is_symlink() or _head_blob(path) != path.read_bytes() or len(history) != 1:
            raise ValueError("balanced-200M W80 timing receipt prefix differs")
    for later in TIMING_SESSION_ORDER[index:]:
        path = timing_report_path(later)
        if path.exists() or _history(path):
            raise ValueError("balanced-200M W80 timing receipt is not a strict prefix")
    if index == 0:
        expected = _git("log", "-1", "--format=%H", "--", VERIFICATION_OUTPUT_PATH.relative_to(ROOT).as_posix())
    else:
        expected = _history(timing_report_path(TIMING_SESSION_ORDER[index - 1]))[0]
    if commit != expected:
        raise ValueError("balanced-200M W80 current HEAD does not match timing prefix")
    return plan, verification, commit, index


def load_cases(plan: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    scale_plan = _read(SCALE_PLAN_PATH)
    prompts, continuations = validate_case_arrays(scale_plan)
    count = TIMING_WARMUP_PROMPTS + TIMING_MEASURED_PROMPTS
    prompts = np.ascontiguousarray(prompts[:count])
    continuations = np.ascontiguousarray(continuations[:count])
    if (
        prompts.dtype != np.uint8
        or prompts.shape != (count, PROMPT_BYTES)
        or continuations.dtype != np.uint8
        or continuations.shape != (count, CONTINUATION_BYTES)
        or array_sha256(prompts) != plan["cases"]["prompts_array_sha256"]
        or array_sha256(continuations) != plan["cases"]["continuations_array_sha256"]
    ):
        raise ValueError("balanced-200M W80 timing cases differ")
    return prompts, continuations


def _runtime(model: Any, role: str) -> IncrementalBltDecoder:
    if role == CANDIDATE_ROLE:
        policy, patches = "causal_whitespace_grid", 80
    elif role == REFERENCE_ROLE:
        policy, patches = "causal_codepoint_grid", 86
    else:
        raise ValueError("balanced-200M W80 runtime role differs")
    return IncrementalBltDecoder(model, policy, horizon=512, patch_count=patches, fixed_stride=6)


def offline_boundaries(observed: bytes, role: str) -> tuple[int, ...]:
    if role == CANDIDATE_ROLE:
        policy, patches = "causal_whitespace_grid", 80
    elif role == REFERENCE_ROLE:
        policy, patches = "causal_codepoint_grid", 86
    else:
        raise ValueError("balanced-200M W80 offline role differs")
    return structural_prefix_boundaries(observed, policy, horizon=512, patch_count=patches, fixed_stride=6)


def _normalized_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.dtype != right.dtype or not bool(torch.all(torch.isfinite(left)).item()) or not bool(torch.all(torch.isfinite(right)).item()):
        raise ValueError("balanced-200M W80 correctness logits differ")
    denominator = TIMING_ATOL + TIMING_RTOL * torch.abs(right)
    result = float(torch.max(torch.abs(left - right) / denominator).item())
    if not math.isfinite(result) or result < 0:
        raise ValueError("balanced-200M W80 normalized error differs")
    return result


def _masks() -> dict[StrictUtf8State, torch.Tensor]:
    output: dict[StrictUtf8State, torch.Tensor] = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device="mps")
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        output[state] = mask
    torch.mps.synchronize()
    return output


def _controlled_correctness(model: Any, role: str, prompts: np.ndarray, continuations: np.ndarray) -> dict[str, Any]:
    comparisons = 0
    argmax_exact = 0
    maximum = 0.0
    boundary_exact = True
    diagnostics_exact = True
    with torch.inference_mode():
        for prompt_row, continuation_row in zip(prompts[:TIMING_CORRECTNESS_PROMPTS], continuations[:TIMING_CORRECTNESS_PROMPTS], strict=True):
            prompt = bytes(prompt_row)
            sequential = _runtime(model, role)
            parallel = _runtime(model, role)
            left = sequential.prefill(prompt)
            right = parallel.prefill_parallel(prompt)
            maximum = max(maximum, _normalized_error(left, right))
            comparisons += 1
            argmax_exact += int(left.argmax().item() == right.argmax().item())
            observed = bytearray(prompt)
            expected = offline_boundaries(bytes(observed), role)
            boundary_exact &= sequential.diagnostics.boundaries == expected and parallel.diagnostics.boundaries == expected
            for value in bytes(continuation_row[:-1]):
                left = sequential.consume(value)
                right = parallel.consume(value)
                maximum = max(maximum, _normalized_error(left, right))
                comparisons += 1
                argmax_exact += int(left.argmax().item() == right.argmax().item())
                observed.append(value)
                expected = offline_boundaries(bytes(observed), role)
                boundary_exact &= sequential.diagnostics.boundaries == expected and parallel.diagnostics.boundaries == expected
            diagnostics_exact &= sequential.diagnostics == parallel.diagnostics
    return {
        "comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "maximum_normalized_logit_error": maximum,
        "boundary_prefix_exact": bool(boundary_exact),
        "cache_diagnostics_exact": bool(diagnostics_exact),
    }


def _trial(model: Any, role: str, prompt: bytes, continuation: bytes, mode: str, masks: Mapping[StrictUtf8State, torch.Tensor]) -> tuple[float, bytes, int, bytes]:
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    generated = bytearray()
    with torch.inference_mode():
        runtime = _runtime(model, role)
        logits = runtime.prefill_parallel(prompt)
        if mode == "controlled_replay":
            for value in continuation[:-1]:
                logits = runtime.consume(value)
            output = continuation
        elif mode == "free_running_utf8_greedy":
            state = STRICT_UTF8_INITIAL_STATE
            while True:
                value = int(logits.masked_fill(~masks[state], -torch.inf).argmax(dim=-1).item())
                generated.append(value)
                state = advance_strict_utf8(state, value)
                if not state.valid:
                    raise AssertionError("balanced-200M W80 strict mask admitted invalid output")
                if len(generated) >= CONTINUATION_BYTES and state.at_codepoint_boundary:
                    break
                if len(generated) >= MAXIMUM_FREE_OUTPUT_BYTES:
                    raise AssertionError("balanced-200M W80 free output exceeded bound")
                logits = runtime.consume(value)
            output = bytes(generated)
            output.decode("utf-8", errors="strict")
        else:
            raise ValueError("balanced-200M W80 timing mode differs")
        torch.mps.synchronize()
    elapsed = (time.perf_counter_ns() - started) / 1_000_000
    observed = prompt + output[:-1]
    diagnostics = runtime.diagnostics
    counters = runtime.runtime_counters
    expected_boundaries = offline_boundaries(observed, role)
    if (
        not math.isfinite(elapsed)
        or elapsed <= 0
        or diagnostics.observed_bytes != len(observed)
        or diagnostics.local_encoder_cached_bytes != len(observed)
        or diagnostics.local_decoder_cached_bytes != len(observed)
        or diagnostics.global_cached_patches != diagnostics.emitted_data_patches
        or diagnostics.boundaries != expected_boundaries
        or counters.parallel_prefill_calls != 1
        or counters.main_consume_calls != len(output) - 1
        or counters.selector_observed_bytes != len(observed)
        or counters.router_forward_calls != 0
    ):
        raise ValueError("balanced-200M W80 timed runtime invariant differs")
    boundary_hash = hashlib.sha256(np.asarray(expected_boundaries, dtype=np.int64).tobytes()).digest()
    return elapsed, output, diagnostics.emitted_data_patches, boundary_hash


def _free_correctness(model: Any, role: str, prompts: np.ndarray, outputs: np.ndarray, lengths: np.ndarray, masks: Mapping[StrictUtf8State, torch.Tensor]) -> dict[str, Any]:
    comparisons = 0
    argmax_exact = 0
    generated_exact = 0
    maximum = 0.0
    boundary_exact = True
    diagnostics_exact = True
    with torch.inference_mode():
        for index in range(TIMING_MEASURED_PROMPTS):
            prompt = bytes(prompts[TIMING_WARMUP_PROMPTS + index])
            output = bytes(outputs[index, : int(lengths[index])])
            output.decode("utf-8", errors="strict")
            sequential = _runtime(model, role)
            parallel = _runtime(model, role)
            left = sequential.prefill(prompt)
            right = parallel.prefill_parallel(prompt)
            state = STRICT_UTF8_INITIAL_STATE
            observed = bytearray(prompt)
            for position, value in enumerate(output):
                maximum = max(maximum, _normalized_error(left, right))
                comparisons += 1
                argmax_exact += int(left.argmax().item() == right.argmax().item())
                expected_byte = int(right.masked_fill(~masks[state], -torch.inf).argmax(dim=-1).item())
                generated_exact += int(expected_byte == value)
                state = advance_strict_utf8(state, value)
                if not state.valid:
                    raise ValueError("balanced-200M W80 stored free output is invalid")
                if position + 1 < len(output):
                    left = sequential.consume(value)
                    right = parallel.consume(value)
                    observed.append(value)
                    expected = offline_boundaries(bytes(observed), role)
                    boundary_exact &= sequential.diagnostics.boundaries == expected and parallel.diagnostics.boundaries == expected
            if len(output) < CONTINUATION_BYTES or len(output) > MAXIMUM_FREE_OUTPUT_BYTES or not state.at_codepoint_boundary:
                raise ValueError("balanced-200M W80 free output termination differs")
            diagnostics_exact &= sequential.diagnostics == parallel.diagnostics
    return {
        "comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "generated_byte_exact": generated_exact,
        "maximum_normalized_logit_error": maximum,
        "boundary_prefix_exact": bool(boundary_exact),
        "cache_diagnostics_exact": bool(diagnostics_exact),
        "strict_output_count": TIMING_MEASURED_PROMPTS,
    }


def _load_models(plan: Mapping[str, Any], verification: Mapping[str, Any]) -> list[Any]:
    candidate = build_main_model(large_scale_model_spec(base.TARGET, 86), seed=base.MODEL_SEED, global_max_position_embeddings=base.GLOBAL_POSITION_LIMIT)
    reference = build_main_model(large_scale_model_spec(base.TARGET, 86), seed=base.MODEL_SEED, global_max_position_embeddings=base.GLOBAL_POSITION_LIMIT)
    candidate.load_state_dict(torch.load(checkpoint_path(), map_location="cpu", weights_only=True))
    reference_identity = plan["roles"]["reference"]["immutable_training_evidence"]
    reference_path = ROOT / reference_identity["checkpoint_path"]
    reference.load_state_dict(torch.load(reference_path, map_location="cpu", weights_only=True))
    if (
        hash_file(checkpoint_path()) != verification["candidate_checkpoint_sha256"]
        or state_sha256(candidate) != verification["candidate_checkpoint_state_sha256"]
        or hash_file(reference_path) != reference_identity["checkpoint_sha256"]
        or state_sha256(reference) != reference_identity["checkpoint_state_sha256"]
        or parameter_count(candidate) != base.EXPECTED_PARAMETER_COUNT
        or parameter_count(reference) != base.EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("balanced-200M W80 timed checkpoint identity differs")
    return [candidate, reference]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, choices=TIMING_SESSION_ORDER)
    args = parser.parse_args()
    plan, verification, commit, session_index = _context(args.session)
    timing_path = timing_array_path(args.session)
    report_path = timing_report_path(args.session)
    if timing_path.exists() or timing_path.is_symlink() or report_path.exists() or _history(report_path):
        raise FileExistsError("balanced-200M W80 timing session was published")
    _operational()
    environment_start = current_runtime_environment_contract()
    prompts, continuations = load_cases(plan)
    models = _load_models(plan, verification)
    shape = (len(TIMING_MODE_ORDER), TIMING_MEASURED_PROMPTS, TIMING_REPETITIONS, len(TIMING_ROLE_ORDER))
    timings = np.empty(shape, dtype=np.float64)
    first_role = np.empty(shape[:-1], dtype=np.uint8)
    emitted = np.empty((len(TIMING_MODE_ORDER), TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER)), dtype=np.int16)
    patch_counts = np.empty_like(emitted)
    boundary_hashes = np.empty((len(TIMING_MODE_ORDER), TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER), 32), dtype=np.uint8)
    free_outputs = np.zeros((TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER), MAXIMUM_FREE_OUTPUT_BYTES), dtype=np.uint8)
    free_lengths = np.zeros((TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER)), dtype=np.int16)
    with publication_mps_exclusive(), torch.inference_mode():
        models = [model.to("mps").eval() for model in models]
        masks = _masks()
        controlled = {
            role: _controlled_correctness(models[index], role, prompts, continuations)
            for index, role in enumerate(TIMING_ROLE_ORDER)
        }
        for warmup in range(TIMING_WARMUP_PROMPTS):
            for mode in TIMING_MODE_ORDER:
                for role_index, role in enumerate(TIMING_ROLE_ORDER):
                    _trial(models[role_index], role, bytes(prompts[warmup]), bytes(continuations[warmup]), mode, masks)
        for prompt_index in range(TIMING_MEASURED_PROMPTS):
            source = TIMING_WARMUP_PROMPTS + prompt_index
            prompt = bytes(prompts[source])
            continuation = bytes(continuations[source])
            for mode_index, mode in enumerate(TIMING_MODE_ORDER):
                for repetition in range(TIMING_REPETITIONS):
                    order = timing_role_order(session_index, prompt_index, repetition, mode_index)
                    first_role[mode_index, prompt_index, repetition] = order[0]
                    for role_index in order:
                        role = TIMING_ROLE_ORDER[role_index]
                        elapsed, output, patches, boundary_hash = _trial(models[role_index], role, prompt, continuation, mode, masks)
                        timings[mode_index, prompt_index, repetition, role_index] = elapsed
                        if repetition == 0:
                            emitted[mode_index, prompt_index, role_index] = len(output)
                            patch_counts[mode_index, prompt_index, role_index] = patches
                            boundary_hashes[mode_index, prompt_index, role_index] = np.frombuffer(boundary_hash, dtype=np.uint8)
                            if mode == "free_running_utf8_greedy":
                                free_lengths[prompt_index, role_index] = len(output)
                                free_outputs[prompt_index, role_index, : len(output)] = np.frombuffer(output, dtype=np.uint8)
                        elif (
                            emitted[mode_index, prompt_index, role_index] != len(output)
                            or patch_counts[mode_index, prompt_index, role_index] != patches
                            or not np.array_equal(boundary_hashes[mode_index, prompt_index, role_index], np.frombuffer(boundary_hash, dtype=np.uint8))
                        ):
                            raise ValueError("balanced-200M W80 deterministic trial evidence changed")
        free = {
            role: _free_correctness(
                models[index],
                role,
                prompts,
                free_outputs[:, index],
                free_lengths[:, index],
                masks,
            )
            for index, role in enumerate(TIMING_ROLE_ORDER)
        }
        for model in models:
            model.to("cpu")
        torch.mps.synchronize()
    correctness_roles: dict[str, Any] = {}
    for role in TIMING_ROLE_ORDER:
        left = controlled[role]
        right = free[role]
        controlled_pass = bool(
            left["comparisons"] == TIMING_CORRECTNESS_PROMPTS * CONTINUATION_BYTES
            and left["argmax_exact"] == left["comparisons"]
            and left["maximum_normalized_logit_error"] <= 1
            and left["boundary_prefix_exact"]
            and left["cache_diagnostics_exact"]
        )
        free_pass = bool(
            right["argmax_exact"] == right["comparisons"]
            and right["generated_byte_exact"] == right["comparisons"]
            and right["maximum_normalized_logit_error"] <= 1
            and right["boundary_prefix_exact"]
            and right["cache_diagnostics_exact"]
            and right["strict_output_count"] == TIMING_MEASURED_PROMPTS
        )
        correctness_roles[role] = {"controlled": left, "free": right, "pass": controlled_pass and free_pass}
    correctness = {
        "by_role": correctness_roles,
        "overall_pass": bool(all(row["pass"] for row in correctness_roles.values())),
    }
    arrays = {
        "end_to_end_ms": timings,
        "first_role": first_role,
        "emitted_output_bytes": emitted,
        "patch_counts": patch_counts,
        "boundary_hashes": boundary_hashes,
        "free_output_bytes": free_outputs,
        "free_output_lengths": free_lengths,
    }
    artifact_bytes = _npz_bytes(arrays)
    environment_end = current_runtime_environment_contract()
    _operational()
    if environment_end != environment_start or _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 timing environment/source changed")
    report = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_actual_session_v1",
        "protocol_id": PROTOCOL_ID,
        "session_id": args.session,
        "session_index": session_index,
        "runner_git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "verification_artifact_sha256": hash_file(VERIFICATION_OUTPUT_PATH),
        "verification_receipt_sha256": verification["receipt_sha256"],
        "model_identity": {
            CANDIDATE_ROLE: {
                "checkpoint_sha256": verification["candidate_checkpoint_sha256"],
                "checkpoint_state_sha256": verification["candidate_checkpoint_state_sha256"],
            },
            REFERENCE_ROLE: {
                "checkpoint_sha256": plan["roles"]["reference"]["immutable_training_evidence"]["checkpoint_sha256"],
                "checkpoint_state_sha256": plan["roles"]["reference"]["immutable_training_evidence"]["checkpoint_state_sha256"],
            },
        },
        "correctness": correctness,
        "environment_start": environment_start,
        "environment_end": environment_end,
        "timing_artifact": {
            "path": timing_path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "arrays_sha256": {key: array_sha256(value) for key, value in arrays.items()},
        },
        "completed": True,
    }
    _publish(timing_path, artifact_bytes, mode=0o600)
    _publish(report_path, canonical_bytes(report), mode=0o644)
    print(f"session={args.session}")
    print("status=complete_commit_receipt_before_next_session")
    print(f"receipt_sha256={hash_file(report_path)}")
    del models
    gc.collect()
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()

