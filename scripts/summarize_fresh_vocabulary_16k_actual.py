#!/usr/bin/env python3
"""Independently validate and summarize fresh-v2 trained 16K actual inference."""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_fresh_vocabulary_16k_actual import (
    correctness_replay,
    load_bundles,
)
from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    correctness_pass,
    summarize_actual_preflight,
)
from fresh_vocabulary_16k_actual_protocol import (
    ACTIVE_PATH,
    MAXIMUM_FREE_TOKENS,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    QUALITY_RESULT_PATH,
    ROOT,
    RUNTIME_REPORT_PATH,
    TIMING_PATH,
    VOCABULARY_BY_ROLE,
    array_sha256,
    canonical_sha256,
    encode_raw,
    hash_file,
    json_bytes,
    read_plan_json,
    reconstruct_cases,
    validate_plan,
)
from fresh_vocabulary_actual_core import (
    WARMUP_CASES,
    validate_strict_token_replay,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.utf8 import compile_strict_utf8_token_transitions


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(f"fresh-16k actual result has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_runtime_report(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    cases: Mapping[str, Any],
) -> None:
    unsigned = dict(report)
    recorded = unsigned.pop("report_sha256", None)
    if (
        set(report)
        != {
            "schema_version",
            "kind",
            "protocol_id",
            "complete",
            "git_commit",
            "plan_artifact_sha256",
            "cases",
            "models",
            "tokenizer_runtime",
            "warmup_correctness",
            "arrays",
            "timing_artifact_sha256",
            "session_state",
            "timed_scope",
            "report_sha256",
        }
        or report.get("schema_version") != 1
        or report.get("kind")
        != "fresh_vocabulary_16k_actual_one_seed_runtime_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != _git("rev-parse", "HEAD")
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("cases") != cases
        or report.get("models") != plan["models"]
        or report.get("tokenizer_runtime") != plan["tokenizer_runtime"]
        or report.get("timing_artifact_sha256") != hash_file(TIMING_PATH)
        or report.get("timed_scope") != plan["experiment"]["timed_scope"]
        or canonical_sha256(unsigned) != recorded
        or not correctness_pass(
            report.get("warmup_correctness", {}), expected_cases=WARMUP_CASES
        )
    ):
        raise ValueError("fresh-16k actual runtime report differs")
    if set(report.get("session_state", {})) != {"start", "end"} or not all(
        timing_environment_eligible(report["session_state"][key])
        for key in ("start", "end")
    ):
        raise ValueError("fresh-16k actual timing environment differs")


def _expected_array_names() -> set[str]:
    return set(TIMING_COMPONENTS) | {
        "output_token_count",
        "output_raw_byte_count",
        "free_token_ids",
        "free_output_bytes",
        "free_output_lengths",
    }


def _load_arrays(report: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected = _expected_array_names()
    if set(report.get("arrays", {})) != expected:
        raise ValueError("fresh-16k actual array descriptor set differs")
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("fresh-16k actual timing key set differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    timing_shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    for name, values in arrays.items():
        if report["arrays"][name] != {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": array_sha256(values),
        }:
            raise ValueError(f"fresh-16k actual array descriptor differs: {name}")
        if name in TIMING_COMPONENTS:
            valid = (
                values.dtype == np.float64
                and values.shape == timing_shape
                and np.isfinite(values).all()
                and np.all(values > 0)
            )
        elif name in {"output_token_count", "output_raw_byte_count"}:
            valid = (
                values.dtype == np.int16
                and values.shape == timing_shape
                and np.all(values > 0)
            )
        elif name == "free_token_ids":
            valid = values.dtype == np.int32 and values.shape == (
                MEASURED_CASES,
                REPETITIONS,
                len(ROLES),
                MAXIMUM_FREE_TOKENS,
            )
        elif name == "free_output_lengths":
            valid = (
                values.dtype == np.int16
                and values.shape == (MEASURED_CASES, REPETITIONS, len(ROLES))
                and np.all(values >= CONTINUATION_BYTES)
            )
        else:
            expected_maximum = max(
                int(
                    report["tokenizer_runtime"][role]["strict_utf8_transitions"][
                        "maximum_free_output_bytes"
                    ]
                )
                for role in ROLES
            )
            valid = values.dtype == np.uint8 and values.shape == (
                MEASURED_CASES,
                REPETITIONS,
                len(ROLES),
                expected_maximum,
            )
        if not valid:
            raise ValueError(f"fresh-16k actual timing array differs: {name}")
    return arrays


def _validate_controlled_counts(
    arrays: Mapping[str, np.ndarray],
    continuations: np.ndarray,
) -> None:
    loaded = load_tokenizers()
    mode = MODES.index("controlled_replay")
    measured = continuations[-MEASURED_CASES:]
    for role_index, role in enumerate(ROLES):
        tokenizer, token_bytes = loaded[VOCABULARY_BY_ROLE[role]]
        expected = np.asarray(
            [len(encode_raw(bytes(row), tokenizer, token_bytes)) for row in measured],
            dtype=np.int16,
        )
        actual = arrays["output_token_count"][mode, ..., role_index]
        if not np.array_equal(
            actual, np.repeat(expected[:, None], REPETITIONS, axis=1)
        ) or np.any(
            arrays["output_raw_byte_count"][mode, ..., role_index] != CONTINUATION_BYTES
        ):
            raise ValueError("fresh-16k actual controlled count evidence differs")


def _validate_free_outputs(
    arrays: Mapping[str, np.ndarray],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[tuple[int, ...]]]]:
    loaded = load_tokenizers()
    free_mode = MODES.index("free_running_utf8_greedy")
    summary: dict[str, Any] = {}
    first_ids_by_role: dict[str, list[tuple[int, ...]]] = {}
    for role_index, role in enumerate(ROLES):
        _, token_bytes = loaded[VOCABULARY_BY_ROLE[role]]
        transitions = compile_strict_utf8_token_transitions(token_bytes)
        maximum = int(
            plan["tokenizer_runtime"][role]["strict_utf8_transitions"][
                "maximum_free_output_bytes"
            ]
        )
        token_root = hashlib.sha256()
        output_root = hashlib.sha256()
        first_ids_by_role[role] = []
        for case_index in range(MEASURED_CASES):
            first_ids: tuple[int, ...] | None = None
            first_raw: bytes | None = None
            for repetition in range(REPETITIONS):
                count = int(
                    arrays["output_token_count"][
                        free_mode, case_index, repetition, role_index
                    ]
                )
                raw_count = int(
                    arrays["output_raw_byte_count"][
                        free_mode, case_index, repetition, role_index
                    ]
                )
                if (
                    not 1 <= count <= MAXIMUM_FREE_TOKENS
                    or arrays["free_output_lengths"][case_index, repetition, role_index]
                    != raw_count
                    or not CONTINUATION_BYTES <= raw_count <= maximum
                ):
                    raise ValueError("fresh-16k actual free count evidence differs")
                token_row = arrays["free_token_ids"][case_index, repetition, role_index]
                if np.any(token_row[:count] < 0) or np.any(token_row[count:] != -1):
                    raise ValueError("fresh-16k actual free token padding differs")
                ids = tuple(int(value) for value in token_row[:count])
                raw_row = arrays["free_output_bytes"][
                    case_index, repetition, role_index
                ]
                if np.any(raw_row[raw_count:] != 0):
                    raise ValueError("fresh-16k actual free byte padding differs")
                raw = bytes(raw_row[:raw_count])
                replayed, _ = validate_strict_token_replay(
                    ids,
                    token_bytes=token_bytes,
                    next_state_indices=transitions.next_state_indices,
                )
                if replayed != raw:
                    raise ValueError("fresh-16k actual free token/byte trace differs")
                if repetition == 0:
                    first_ids, first_raw = ids, raw
                    first_ids_by_role[role].append(ids)
                elif ids != first_ids or raw != first_raw:
                    raise ValueError("fresh-16k actual free output is nondeterministic")
                token_root.update(np.asarray(ids, dtype=np.int32).tobytes())
                output_root.update(len(raw).to_bytes(8, "big"))
                output_root.update(raw)
        summary[role] = {
            "strict_valid_output_count": MEASURED_CASES * REPETITIONS,
            "deterministic_across_repetitions": True,
            "token_trace_root_sha256": token_root.hexdigest(),
            "output_root_sha256": output_root.hexdigest(),
        }
    return summary, first_ids_by_role


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-16k actual summary requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    plan_path = PLAN_PATH.relative_to(ROOT).as_posix()
    if _git("log", "-1", "--format=%H", "--", plan_path) != commit:
        raise RuntimeError("fresh-16k actual plan must remain current HEAD")
    _require_never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists():
        raise RuntimeError("fresh-16k actual benchmark remains active")
    plan = read_plan_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("fresh-16k actual cases differ at summary")
    report = read_plan_json(RUNTIME_REPORT_PATH)
    _validate_runtime_report(report, plan, cases)
    arrays = _load_arrays(report)
    _validate_controlled_counts(arrays, continuations)
    free_evidence, expected_free_ids = _validate_free_outputs(arrays, plan)

    with publication_mps_exclusive():
        bundles = load_bundles(plan)
        independent_correctness = correctness_replay(
            bundles,
            prompts[-MEASURED_CASES:],
            continuations[-MEASURED_CASES:],
            expected_free_ids=expected_free_ids,
        )
        for bundle in bundles.values():
            bundle.model.to("cpu")
        del bundles
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
    actual = summarize_actual_preflight(
        timing={name: arrays[name] for name in TIMING_COMPONENTS},
        output_token_count=arrays["output_token_count"],
        output_raw_byte_count=arrays["output_raw_byte_count"],
        correctness=independent_correctness,
        maximum_output_bytes_by_role={
            role: int(
                plan["tokenizer_runtime"][role]["strict_utf8_transitions"][
                    "maximum_free_output_bytes"
                ]
            )
            for role in ROLES
        },
    )
    quality = read_plan_json(QUALITY_RESULT_PATH)
    models = plan["models"]
    candidate = models["candidate_16k"]
    baseline = models["baseline_2k"]
    frontier = models["frontier_8k"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_actual_one_seed_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": actual["status"],
        "git_commit": commit,
        "plan": {
            "artifact_sha256": hash_file(PLAN_PATH),
            "plan_sha256": plan["plan_sha256"],
        },
        "runtime": {
            "report_artifact_sha256": hash_file(RUNTIME_REPORT_PATH),
            "report_sha256": report["report_sha256"],
            "timing_artifact_sha256": hash_file(TIMING_PATH),
            "session_state": report["session_state"],
            "timed_scope": report["timed_scope"],
        },
        "quality_selection": {
            "quality_result_artifact_sha256": hash_file(QUALITY_RESULT_PATH),
            "quality_result_sha256": quality["summary_sha256"],
            "actual_latency_used_for_selection": False,
            "document_bpb_by_role": {
                role: models[role]["document_bpb"] for role in ROLES
            },
        },
        "actual_inference": actual,
        "free_output_evidence": free_evidence,
        "systems_cost": {
            "by_role": models,
            "candidate_vs_2k": {
                "parameter_increase": candidate["parameter_count"]
                / baseline["parameter_count"]
                - 1.0,
                "checkpoint_byte_increase": candidate["checkpoint_bytes"]
                / baseline["checkpoint_bytes"]
                - 1.0,
            },
            "candidate_vs_8k": {
                "parameter_increase": candidate["parameter_count"]
                / frontier["parameter_count"]
                - 1.0,
                "checkpoint_byte_increase": candidate["checkpoint_bytes"]
                / frontier["checkpoint_bytes"]
                - 1.0,
            },
            "memory_improvement_claimed": False,
        },
        "claim_boundary": {
            **plan["claim_boundary"],
            "publication_claim": False,
            "general_hardware_claim": False,
            "one_seed_one_session": True,
            "secondary_pair_cannot_authorize_fallback": True,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during fresh-16k actual summary")
    _publish(OUTPUT_PATH, json_bytes(summary))
    print(f"status={summary['status']}")
    for pair_name in ("candidate_vs_2k", "candidate_vs_8k"):
        for mode in MODES:
            reduction = summary["actual_inference"]["pairs"][pair_name][mode][
                "end_to_end_reduction"
            ]
            print(f"{pair_name}_{mode}_reduction={reduction:.9f}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
