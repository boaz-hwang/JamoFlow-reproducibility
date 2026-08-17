#!/usr/bin/env python3
"""Independently replay and summarize trained 16K actual retrieval drafting."""

from __future__ import annotations

import gc
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_fresh_vocabulary_16k_block import load_target, prepare_payloads
from benchmark_fresh_vocabulary_16k_retrieval import correctness_replay
from fresh_vocabulary_16k_retrieval_actual_core import (
    CONTINUATION_BYTES,
    COUNTER_NAMES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    summarize_retrieval_actual,
)
from fresh_vocabulary_16k_retrieval_protocol import (
    ACTIVE_PATH,
    MAXIMUM_FREE_TOKENS,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    RUNTIME_REPORT_PATH,
    TIMING_PATH,
    array_sha256,
    canonical_sha256,
    hash_file,
    json_bytes,
    load_table,
    read_json,
    reconstruct_cases,
    validate_plan,
)
from summarize_fresh_vocabulary_16k_block import (
    _validate_controlled_counts,
    _validate_free_outputs,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    if _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()):
        raise FileExistsError(f"16K retrieval result has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _correctness_pass(value: Mapping[str, Any], *, expected_cases: int) -> bool:
    if value.get("overall_pass") is not True:
        return False
    target = value.get("target_cache_full", {})
    if (
        target.get("pass") is not True
        or int(target.get("comparisons", -1)) <= 0
        or target.get("comparisons") != target.get("argmax_exact")
        or not 0 <= float(target.get("maximum_normalized_tolerance_ratio", -1)) <= 1
    ):
        return False
    rows = value.get("by_role_mode", {})
    return bool(
        set(rows) == set(ROLES)
        and all(set(rows[role]) == set(MODES) for role in ROLES)
        and all(
            row.get("cases") == expected_cases
            and row.get("outputs_exact") is True
            and row.get("cache_lag_exact") is True
            and int(row.get("target_forward_calls", 0)) > 0
            for role in ROLES
            for row in rows[role].values()
        )
    )


def _validate_runtime_report(
    report: Mapping[str, Any], plan: Mapping[str, Any], cases: Mapping[str, Any]
) -> None:
    unsigned = dict(report)
    recorded = unsigned.pop("report_sha256", None)
    expected = {
        "schema_version",
        "kind",
        "protocol_id",
        "complete",
        "git_commit",
        "plan_artifact_sha256",
        "cases",
        "target",
        "table",
        "tokenizer_runtime",
        "warmup_correctness",
        "arrays",
        "timing_artifact_sha256",
        "session_state",
        "timed_scope",
        "report_sha256",
    }
    if (
        set(report) != expected
        or report.get("kind") != "fresh_vocabulary_16k_retrieval_runtime_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != _git("rev-parse", "HEAD")
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("cases") != cases
        or report.get("target") != plan["target"]
        or report.get("table") != plan["table"]
        or report.get("tokenizer_runtime") != plan["tokenizer_runtime"]
        or report.get("timing_artifact_sha256") != hash_file(TIMING_PATH)
        or report.get("timed_scope") != plan["experiment"]["timed_scope"]
        or canonical_sha256(unsigned) != recorded
        or not _correctness_pass(
            report.get("warmup_correctness", {}),
            expected_cases=plan["experiment"]["warmup_cases"],
        )
    ):
        raise ValueError("16K retrieval runtime report differs")
    state = report.get("session_state", {})
    if set(state) != {"start", "end"} or not all(
        timing_environment_eligible(state[key]) for key in ("start", "end")
    ):
        raise ValueError("16K retrieval timing environment differs")


def _expected_array_names() -> set[str]:
    return set(TIMING_COMPONENTS) | set(COUNTER_NAMES) | {
        "output_token_count",
        "output_raw_byte_count",
        "free_token_ids",
        "free_output_bytes",
        "free_output_lengths",
    }


def _load_arrays(report: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected = _expected_array_names()
    if set(report.get("arrays", {})) != expected:
        raise ValueError("16K retrieval array descriptor set differs")
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("16K retrieval timing key set differs")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    maximum = int(
        plan["tokenizer_runtime"]["strict_utf8_transitions"]["maximum_free_output_bytes"]
    )
    for name, values in arrays.items():
        if report["arrays"][name] != {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": array_sha256(values),
        }:
            raise ValueError(f"16K retrieval array descriptor differs: {name}")
        if name in TIMING_COMPONENTS:
            valid = (
                values.dtype == np.float64
                and values.shape == shape
                and np.isfinite(values).all()
                and np.all(values >= 0)
                and (name == "draft_lookup_ms" or np.all(values > 0))
            )
        elif name in COUNTER_NAMES or name in {"output_token_count", "output_raw_byte_count"}:
            valid = values.dtype == np.int16 and values.shape == shape and np.all(values >= 0)
            if name in {"target_forward_calls", "output_token_count", "output_raw_byte_count"}:
                valid = valid and np.all(values > 0)
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
            raise ValueError(f"16K retrieval timing array differs: {name}")
    return arrays


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("16K retrieval summary requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()) != commit:
        raise RuntimeError("16K retrieval plan must remain current HEAD")
    _require_never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists():
        raise RuntimeError("16K retrieval benchmark remains active")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("16K retrieval cases differ at summary")
    report = read_json(RUNTIME_REPORT_PATH)
    _validate_runtime_report(report, plan, cases)
    arrays = _load_arrays(report, plan)
    _validate_controlled_counts(arrays, continuations)
    free_evidence, expected_free_ids = _validate_free_outputs(arrays, plan)
    maximum = int(
        plan["tokenizer_runtime"]["strict_utf8_transitions"]["maximum_free_output_bytes"]
    )
    with publication_mps_exclusive():
        table = load_table()
        bundle = load_target(plan)
        payloads = prepare_payloads(
            bundle,
            prompts[-MEASURED_CASES:],
            continuations[-MEASURED_CASES:],
        )
        for payload, expected in zip(payloads, expected_free_ids, strict=True):
            if payload["free_ids"] != expected:
                raise AssertionError("16K retrieval measured free trace was not regenerated")
        independent = correctness_replay(
            bundle,
            table,
            payloads,
            continuation_bytes=plan["experiment"]["continuation_bytes"],
            maximum_output_bytes=maximum,
        )
        bundle.model.to("cpu")
        del bundle, table, payloads
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
    if not _correctness_pass(independent, expected_cases=MEASURED_CASES):
        raise AssertionError("16K retrieval independent correctness failed")
    result = summarize_retrieval_actual(
        timing={name: arrays[name] for name in TIMING_COMPONENTS},
        counters={name: arrays[name] for name in COUNTER_NAMES},
        output_token_count=arrays["output_token_count"],
        output_raw_byte_count=arrays["output_raw_byte_count"],
        correctness_pass=True,
        maximum_output_bytes=maximum,
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_retrieval_actual_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": result["status"],
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
        "table": plan["table"],
        "actual_retrieval": result,
        "independent_correctness": independent,
        "free_output_evidence": free_evidence,
        "decision": {
            "korean_specific_disjoint_followup_authorized": result[
                "korean_specific_followup_authorized"
            ],
            "learned_draft_training_authorized": False,
            "generic_retrieval_novelty_claimed": False,
            "diagnostic_role_fallback_allowed": False,
        },
        "claim_boundary": {
            **plan["claim_boundary"],
            "actual_generic_retrieval_measured": True,
            "output_exact": True,
            "one_seed_one_session": True,
            "general_hardware_claim": False,
            "publication_claim": False,
        },
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during 16K retrieval summary")
    _publish(OUTPUT_PATH, json_bytes(summary))
    print(f"status={summary['status']}")
    for role in ROLES[1:]:
        for mode in MODES:
            row = result["comparisons"][role][mode]
            print(f"{role}_{mode}_reduction={row['end_to_end_reduction']:.9f}")
            print(f"{role}_{mode}_acceptance={row['draft_token_acceptance_rate']:.9f}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
