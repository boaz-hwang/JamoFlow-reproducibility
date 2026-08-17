#!/usr/bin/env python3
"""Validate five committed W80/C86 timing sessions and seal their summary."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from balanced_200m_w80_core import (
    CONTINUATION_BYTES,
    MAXIMUM_FREE_OUTPUT_BYTES,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    TIMING_MEASURED_PROMPTS,
    TIMING_CORRECTNESS_PROMPTS,
    TIMING_MODE_ORDER,
    TIMING_REPETITIONS,
    TIMING_ROLE_ORDER,
    TIMING_SESSION_ORDER,
    TIMING_SUMMARY_PATH,
    VERIFICATION_OUTPUT_PATH,
    canonical_bytes,
    summarize_actual_timing,
    timing_array_path,
    timing_report_path,
    timing_role_order,
    validate_plan,
    validate_verification_receipt,
)
from benchmark_balanced_200m_w80_actual import load_cases, offline_boundaries
from scale_schedule_extrapolation_core import array_sha256

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract


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


def _ancestor(left: str, right: str, *, strict: bool = False) -> None:
    if strict and left == right:
        raise ValueError("balanced-200M W80 strict chronology differs")
    completed = subprocess.run(("git", "merge-base", "--is-ancestor", left, right), cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise ValueError("balanced-200M W80 Git chronology differs")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_session(session: str, plan: Mapping[str, Any], verification: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    report_path = timing_report_path(session)
    timing_path = timing_array_path(session)
    history = _history(report_path)
    if len(history) != 1 or _head_blob(report_path) != report_path.read_bytes() or not timing_path.is_file() or timing_path.is_symlink():
        raise ValueError("balanced-200M W80 session publication differs")
    report = _read(report_path)
    index = TIMING_SESSION_ORDER.index(session)
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "balanced_200m_w80_actual_session_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("session_id") != session
        or report.get("session_index") != index
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("verification_receipt_sha256") != verification["receipt_sha256"]
        or report.get("verification_artifact_sha256") != hash_file(VERIFICATION_OUTPUT_PATH)
        or report.get("environment_start") != plan["environment"]
        or report.get("environment_end") != plan["environment"]
        or report.get("completed") is not True
    ):
        raise ValueError("balanced-200M W80 session identity differs")
    publication_commit = history[0]
    _ancestor(report["runner_git_commit"], publication_commit, strict=True)
    if index:
        prior_commit = _history(timing_report_path(TIMING_SESSION_ORDER[index - 1]))[0]
        _ancestor(prior_commit, report["runner_git_commit"])
    raw = timing_path.read_bytes()
    artifact = report.get("timing_artifact")
    if not isinstance(artifact, Mapping) or artifact.get("path") != timing_path.relative_to(ROOT).as_posix() or artifact.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("balanced-200M W80 timing artifact identity differs")
    with np.load(io.BytesIO(raw), allow_pickle=False) as source:
        expected_keys = {"end_to_end_ms", "first_role", "emitted_output_bytes", "patch_counts", "boundary_hashes", "free_output_bytes", "free_output_lengths"}
        if set(source.files) != expected_keys:
            raise ValueError("balanced-200M W80 timing NPZ schema differs")
        arrays = {key: np.ascontiguousarray(source[key]) for key in expected_keys}
    expected_shape = (len(TIMING_MODE_ORDER), TIMING_MEASURED_PROMPTS, TIMING_REPETITIONS, len(TIMING_ROLE_ORDER))
    timings = arrays["end_to_end_ms"]
    first = arrays["first_role"]
    if (
        timings.dtype != np.float64
        or timings.shape != expected_shape
        or not np.all(np.isfinite(timings))
        or np.any(timings <= 0)
        or first.dtype != np.uint8
        or first.shape != expected_shape[:-1]
        or any(artifact.get("arrays_sha256", {}).get(key) != array_sha256(value) for key, value in arrays.items())
    ):
        raise ValueError("balanced-200M W80 timing arrays differ")
    expected_first = np.asarray(
        [[[timing_role_order(index, prompt, repetition, mode)[0] for repetition in range(TIMING_REPETITIONS)] for prompt in range(TIMING_MEASURED_PROMPTS)] for mode in range(len(TIMING_MODE_ORDER))],
        dtype=np.uint8,
    )
    if not np.array_equal(first, expected_first):
        raise ValueError("balanced-200M W80 role order differs")
    prompts, continuations = load_cases(plan)
    emitted = arrays["emitted_output_bytes"]
    counts = arrays["patch_counts"]
    hashes = arrays["boundary_hashes"]
    free = arrays["free_output_bytes"]
    lengths = arrays["free_output_lengths"]
    if (
        emitted.dtype != np.int16
        or emitted.shape != (len(TIMING_MODE_ORDER), TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER))
        or counts.dtype != np.int16
        or counts.shape != emitted.shape
        or hashes.dtype != np.uint8
        or hashes.shape != (*emitted.shape, 32)
        or free.dtype != np.uint8
        or free.shape != (TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER), MAXIMUM_FREE_OUTPUT_BYTES)
        or lengths.dtype != np.int16
        or lengths.shape != (TIMING_MEASURED_PROMPTS, len(TIMING_ROLE_ORDER))
        or np.any(emitted[0] != CONTINUATION_BYTES)
        or np.any(emitted[1] < CONTINUATION_BYTES)
        or np.any(emitted[1] > MAXIMUM_FREE_OUTPUT_BYTES)
        or not np.array_equal(emitted[1], lengths)
    ):
        raise ValueError("balanced-200M W80 mechanism arrays differ")
    for prompt_index in range(TIMING_MEASURED_PROMPTS):
        prompt = bytes(prompts[4 + prompt_index])
        for mode_index, mode in enumerate(TIMING_MODE_ORDER):
            for role_index, role in enumerate(TIMING_ROLE_ORDER):
                output = bytes(continuations[4 + prompt_index]) if mode == "controlled_replay" else bytes(free[prompt_index, role_index, : int(lengths[prompt_index, role_index])])
                if mode == "free_running_utf8_greedy":
                    output.decode("utf-8", errors="strict")
                    if np.any(free[prompt_index, role_index, len(output) :] != 0):
                        raise ValueError("balanced-200M W80 free-output padding differs")
                if emitted[mode_index, prompt_index, role_index] != len(output):
                    raise ValueError("balanced-200M W80 output length differs")
                boundaries = offline_boundaries(prompt + output[:-1], role)
                digest = hashlib.sha256(np.asarray(boundaries, dtype=np.int64).tobytes()).digest()
                if counts[mode_index, prompt_index, role_index] != len(boundaries) or not np.array_equal(hashes[mode_index, prompt_index, role_index], np.frombuffer(digest, dtype=np.uint8)):
                    raise ValueError("balanced-200M W80 boundary evidence differs")
    correctness = report.get("correctness")
    correctness_pass = isinstance(correctness, Mapping) and correctness.get("overall_pass") is True
    if correctness_pass:
        rows = correctness.get("by_role")
        correctness_pass = isinstance(rows, Mapping) and set(rows) == set(TIMING_ROLE_ORDER)
    if correctness_pass:
        for role_index, role in enumerate(TIMING_ROLE_ORDER):
            row = rows[role]
            controlled = row.get("controlled") if isinstance(row, Mapping) else None
            free_row = row.get("free") if isinstance(row, Mapping) else None
            expected_free = int(np.sum(lengths[:, role_index]))
            correctness_pass &= bool(
                isinstance(controlled, Mapping)
                and set(controlled)
                == {
                    "argmax_exact",
                    "boundary_prefix_exact",
                    "cache_diagnostics_exact",
                    "comparisons",
                    "maximum_normalized_logit_error",
                }
                and controlled.get("comparisons")
                == TIMING_CORRECTNESS_PROMPTS * CONTINUATION_BYTES
                and controlled.get("argmax_exact") == controlled.get("comparisons")
                and controlled.get("boundary_prefix_exact") is True
                and controlled.get("cache_diagnostics_exact") is True
                and 0
                <= float(controlled.get("maximum_normalized_logit_error", np.inf))
                <= 1
                and isinstance(free_row, Mapping)
                and set(free_row)
                == {
                    "argmax_exact",
                    "boundary_prefix_exact",
                    "cache_diagnostics_exact",
                    "comparisons",
                    "generated_byte_exact",
                    "maximum_normalized_logit_error",
                    "strict_output_count",
                }
                and free_row.get("comparisons") == expected_free
                and free_row.get("argmax_exact") == expected_free
                and free_row.get("generated_byte_exact") == expected_free
                and free_row.get("boundary_prefix_exact") is True
                and free_row.get("cache_diagnostics_exact") is True
                and free_row.get("strict_output_count") == TIMING_MEASURED_PROMPTS
                and 0
                <= float(free_row.get("maximum_normalized_logit_error", np.inf))
                <= 1
                and row.get("pass") is True
            )
    if not correctness_pass:
        raise ValueError("balanced-200M W80 correctness did not pass")
    return timings, dict(correctness), {"receipt_path": report_path.relative_to(ROOT).as_posix(), "receipt_sha256": hash_file(report_path), "receipt_git_commit": publication_commit, "timing_path": timing_path.relative_to(ROOT).as_posix(), "timing_sha256": hash_file(timing_path)}


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 summary requires a clean worktree")
    if TIMING_SUMMARY_PATH.exists() or _history(TIMING_SUMMARY_PATH):
        raise FileExistsError("balanced-200M W80 timing summary was published")
    commit = _git("rev-parse", "HEAD")
    plan = _read(PLAN_PATH)
    verification = _read(VERIFICATION_OUTPUT_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_verification_receipt(verification)
    end_to_end: dict[str, np.ndarray] = {}
    correctness: dict[str, Mapping[str, Any]] = {}
    evidence: dict[str, Any] = {}
    for session in TIMING_SESSION_ORDER:
        end_to_end[session], correctness[session], evidence[session] = _load_session(session, plan, verification)
    aggregate = summarize_actual_timing(end_to_end, correctness)
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_actual_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "w80_actual_primary_pass" if aggregate["overall_actual_primary_pass"] else "w80_actual_primary_fail",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "verification_artifact_sha256": hash_file(VERIFICATION_OUTPUT_PATH),
        "verification_receipt_sha256": verification["receipt_sha256"],
        "summary_base_git_commit": commit,
        "session_evidence": evidence,
        "actual_timing": aggregate,
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "quality_matched": True,
            "pure_model_scale_causal_effect_claimed": False,
            "observed_larger_than_compact_requires_primary_pass": True,
            "statistically_larger_requires_strong_scale_amplification_support": True,
        },
    }
    summary = {**payload, "summary_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest()}
    if _git("rev-parse", "HEAD") != commit or _git("status", "--porcelain"):
        raise ValueError("balanced-200M W80 repository changed during summary")
    _publish(TIMING_SUMMARY_PATH, canonical_bytes(summary))
    print(f"status={summary['status']}")
    for mode in TIMING_MODE_ORDER:
        row = aggregate["by_mode"][mode]
        print(f"{mode}_reduction={row['end_to_end_reduction']:.9f}")
        print(f"{mode}_lower={row['crossed_bootstrap_95_interval']['lower']:.9f}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
