#!/usr/bin/env python3
"""Independently validate and summarize the trained 16K block upper bound."""

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
from benchmark_fresh_vocabulary_16k_block import (
    correctness_replay,
    load_target,
    prepare_payloads,
)
from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_actual_protocol import encode_raw, json_bytes
from fresh_vocabulary_16k_block_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    correctness_pass,
    summarize_upper_bound,
)
from fresh_vocabulary_16k_block_protocol import (
    ACTIVE_PATH,
    MAXIMUM_FREE_TOKENS,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    RUNTIME_REPORT_PATH,
    TARGET_VOCABULARY_SIZE,
    TIMING_PATH,
    array_sha256,
    canonical_sha256,
    hash_file,
    read_json,
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
        "log",
        "--all",
        "--format=%H",
        "--",
        path.relative_to(ROOT).as_posix(),
    )
    if history:
        raise FileExistsError(f"16K target-block result has Git history: {path}")


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
            "target",
            "tokenizer_runtime",
            "warmup_correctness",
            "arrays",
            "timing_artifact_sha256",
            "session_state",
            "timed_scope",
            "report_sha256",
        }
        or report.get("schema_version") != 1
        or report.get("kind") != "fresh_vocabulary_16k_target_block_runtime_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != _git("rev-parse", "HEAD")
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("cases") != cases
        or report.get("target") != plan["target"]
        or report.get("tokenizer_runtime") != plan["tokenizer_runtime"]
        or report.get("timing_artifact_sha256") != hash_file(TIMING_PATH)
        or report.get("timed_scope") != plan["experiment"]["timed_scope"]
        or canonical_sha256(unsigned) != recorded
        or not correctness_pass(
            report.get("warmup_correctness", {}),
            expected_cases=WARMUP_CASES,
        )
    ):
        raise ValueError("16K target-block runtime report differs")
    state = report.get("session_state", {})
    if set(state) != {"start", "end"} or not all(
        timing_environment_eligible(state[key]) for key in ("start", "end")
    ):
        raise ValueError("16K target-block timing environment differs")


def _expected_array_names() -> set[str]:
    return set(TIMING_COMPONENTS) | {
        "output_token_count",
        "output_raw_byte_count",
        "target_forward_calls",
        "free_token_ids",
        "free_output_bytes",
        "free_output_lengths",
    }


def _load_arrays(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    expected = _expected_array_names()
    if set(report.get("arrays", {})) != expected:
        raise ValueError("16K target-block array descriptor set differs")
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("16K target-block timing key set differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    timing_shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    maximum = int(
        plan["tokenizer_runtime"]["strict_utf8_transitions"][
            "maximum_free_output_bytes"
        ]
    )
    for name, values in arrays.items():
        if report["arrays"][name] != {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": array_sha256(values),
        }:
            raise ValueError(f"16K target-block descriptor differs: {name}")
        if name in TIMING_COMPONENTS:
            valid = (
                values.dtype == np.float64
                and values.shape == timing_shape
                and np.isfinite(values).all()
                and np.all(values > 0)
            )
        elif name in {
            "output_token_count",
            "output_raw_byte_count",
            "target_forward_calls",
        }:
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
                and np.all(values <= maximum)
            )
        else:
            valid = values.dtype == np.uint8 and values.shape == (
                MEASURED_CASES,
                REPETITIONS,
                len(ROLES),
                maximum,
            )
        if not valid:
            raise ValueError(f"16K target-block timing array differs: {name}")
    return arrays


def _validate_controlled_counts(
    arrays: Mapping[str, np.ndarray],
    continuations: np.ndarray,
) -> None:
    tokenizer, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    expected = np.asarray(
        [
            len(encode_raw(bytes(row), tokenizer, token_bytes))
            for row in continuations[-MEASURED_CASES:]
        ],
        dtype=np.int16,
    )
    mode_index = MODES.index("controlled_replay")
    repeated = np.repeat(expected[:, None], REPETITIONS, axis=1)
    for role_index in range(len(ROLES)):
        if not np.array_equal(
            arrays["output_token_count"][mode_index, ..., role_index],
            repeated,
        ) or np.any(
            arrays["output_raw_byte_count"][mode_index, ..., role_index]
            != CONTINUATION_BYTES
        ):
            raise ValueError("16K target-block controlled count evidence differs")


def _validate_free_outputs(
    arrays: Mapping[str, np.ndarray],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[int, ...]]]:
    _, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    transitions = compile_strict_utf8_token_transitions(token_bytes)
    maximum = int(
        plan["tokenizer_runtime"]["strict_utf8_transitions"][
            "maximum_free_output_bytes"
        ]
    )
    mode_index = MODES.index("free_running_utf8_greedy")
    token_root = hashlib.sha256()
    output_root = hashlib.sha256()
    first_ids_by_case: list[tuple[int, ...]] = []
    for case_index in range(MEASURED_CASES):
        canonical_ids: tuple[int, ...] | None = None
        canonical_raw: bytes | None = None
        for repetition in range(REPETITIONS):
            for role_index in range(len(ROLES)):
                count = int(
                    arrays["output_token_count"][
                        mode_index,
                        case_index,
                        repetition,
                        role_index,
                    ]
                )
                raw_count = int(
                    arrays["output_raw_byte_count"][
                        mode_index,
                        case_index,
                        repetition,
                        role_index,
                    ]
                )
                if (
                    not 1 <= count <= MAXIMUM_FREE_TOKENS
                    or arrays["free_output_lengths"][
                        case_index,
                        repetition,
                        role_index,
                    ]
                    != raw_count
                    or not CONTINUATION_BYTES <= raw_count <= maximum
                ):
                    raise ValueError("16K target-block free counts differ")
                token_row = arrays["free_token_ids"][
                    case_index,
                    repetition,
                    role_index,
                ]
                if np.any(token_row[:count] < 0) or np.any(token_row[count:] != -1):
                    raise ValueError("16K target-block free token padding differs")
                ids = tuple(int(value) for value in token_row[:count])
                raw_row = arrays["free_output_bytes"][
                    case_index,
                    repetition,
                    role_index,
                ]
                if np.any(raw_row[raw_count:] != 0):
                    raise ValueError("16K target-block free byte padding differs")
                raw = bytes(raw_row[:raw_count])
                replayed, _ = validate_strict_token_replay(
                    ids,
                    token_bytes=token_bytes,
                    next_state_indices=transitions.next_state_indices,
                )
                if replayed != raw:
                    raise ValueError("16K target-block free trace differs")
                if canonical_ids is None:
                    canonical_ids, canonical_raw = ids, raw
                elif ids != canonical_ids or raw != canonical_raw:
                    raise ValueError(
                        "16K target-block output differs across roles or repetitions"
                    )
                token_root.update(len(ids).to_bytes(8, "big"))
                token_root.update(np.asarray(ids, dtype=np.int32).tobytes())
                output_root.update(len(raw).to_bytes(8, "big"))
                output_root.update(raw)
        if canonical_ids is None:
            raise AssertionError("16K target-block free case is empty")
        first_ids_by_case.append(canonical_ids)
    return (
        {
            "strict_valid_trace_count": MEASURED_CASES * REPETITIONS * len(ROLES),
            "exact_across_roles_and_repetitions": True,
            "token_trace_root_sha256": token_root.hexdigest(),
            "output_root_sha256": output_root.hexdigest(),
        },
        first_ids_by_case,
    )


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("16K target-block summary requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    relative = PLAN_PATH.relative_to(ROOT).as_posix()
    if _git("log", "-1", "--format=%H", "--", relative) != commit:
        raise RuntimeError("16K target-block plan must remain current HEAD")
    _require_never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists():
        raise RuntimeError("16K target-block benchmark remains active")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("16K target-block cases differ at summary")
    report = read_json(RUNTIME_REPORT_PATH)
    _validate_runtime_report(report, plan, cases)
    arrays = _load_arrays(report, plan)
    _validate_controlled_counts(arrays, continuations)
    free_evidence, expected_free_ids = _validate_free_outputs(arrays, plan)

    with publication_mps_exclusive():
        bundle = load_target(plan)
        payloads = prepare_payloads(
            bundle,
            prompts[-MEASURED_CASES:],
            continuations[-MEASURED_CASES:],
        )
        for payload, expected in zip(payloads, expected_free_ids, strict=True):
            if payload["free_ids"] != expected:
                raise AssertionError(
                    "16K target-block measured free trace was not regenerated"
                )
        independent_correctness = correctness_replay(bundle, payloads)
        bundle.model.to("cpu")
        del bundle, payloads
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    upper_bound = summarize_upper_bound(
        timing={name: arrays[name] for name in TIMING_COMPONENTS},
        output_token_count=arrays["output_token_count"],
        output_raw_byte_count=arrays["output_raw_byte_count"],
        target_forward_calls=arrays["target_forward_calls"],
        correctness=independent_correctness,
        maximum_output_bytes=int(
            plan["tokenizer_runtime"]["strict_utf8_transitions"][
                "maximum_free_output_bytes"
            ]
        ),
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_target_block_upper_bound_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": upper_bound["status"],
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
        "target": plan["target"],
        "upper_bound": upper_bound,
        "free_output_evidence": free_evidence,
        "decision": {
            "learned_same_tokenizer_draft_fail_fast_authorized": upper_bound[
                "learned_draft_prototype_authorized"
            ],
            "dense_vocabulary_multiseed_authorized": False,
            "actual_speculative_efficiency_claimed": False,
            "diagnostic_block_size_fallback_allowed": False,
        },
        "claim_boundary": {
            **plan["claim_boundary"],
            "one_seed_one_session": True,
            "actual_target_kernel_measured": True,
            "perfect_draft_cost_excluded": True,
            "learned_draft_acceptance_measured": False,
            "general_hardware_claim": False,
            "publication_claim": False,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during 16K target-block summary")
    _publish(OUTPUT_PATH, json_bytes(summary))
    print(f"status={summary['status']}")
    for role in ROLES[1:]:
        for mode in MODES:
            reduction = upper_bound["comparisons"][role][mode]["end_to_end_reduction"]
            print(f"{role}_{mode}_reduction={reduction:.9f}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
