#!/usr/bin/env python3
"""Validate and summarize the sealed Korean BPE systems frontier."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from typing import Any, Mapping

import numpy as np

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_benchmark import paired_prompt_latency
from token_frontier_core import (
    DEPTHS,
    FRONTIER_SPECS,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    RUNTIME_ROLES,
    VOCABULARY_SIZES,
    parse_role,
    role_name,
)
from token_frontier_protocol import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    MEASURED_CASES,
    OPPORTUNITY_REPORT_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPETITIONS,
    ROOT,
    RUNTIME_REPORT_PATH,
    RUNTIME_ACTIVE_PATH,
    TIMING_PATH,
    TOKENIZER_ENCODE_REPETITIONS,
    TOKENIZER_PATHS,
    WARMUP_CASES,
    array_sha256,
    canonical_sha256,
    hash_file,
    json_bytes,
    read_json,
    reconstruct_cases,
    validate_plan,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_root() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("token frontier summary requires a clean root")
    return _command("git", "rev-parse", "HEAD")


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command("git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if history:
        raise FileExistsError(f"token frontier result has Git history: {path}")


def _unsigned_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field)
    return canonical_sha256(unsigned)


def _validate_tokenizers(plan: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "korean_bpe_systems_frontier_tokenizer_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("git_commit") != _command("git", "rev-parse", "HEAD")
        or _unsigned_hash(report, "report_sha256") != report.get("report_sha256")
        or set(report.get("metrics", {})) != {str(value) for value in VOCABULARY_SIZES}
        or set(report.get("tokenizer_artifacts", {})) != {str(value) for value in VOCABULARY_SIZES}
    ):
        raise ValueError("token frontier tokenizer report differs")
    for size in VOCABULARY_SIZES:
        key = str(size)
        artifact = report["tokenizer_artifacts"][key]
        if artifact != {
            "path": str(TOKENIZER_PATHS[size].relative_to(ROOT)),
            "sha256": hash_file(TOKENIZER_PATHS[size]),
        }:
            raise ValueError("token frontier tokenizer artifact identity differs")
        row = report["metrics"][key]
        if (
            row.get("vocabulary_size") != size
            or row.get("roundtrip_identity") is not True
            or row.get("raw_token_bytes_identity") is not True
            or row.get("deterministic_replicate_json_identity") is not True
            or row.get("tokenizer_json_sha256") != artifact["sha256"]
            or not isinstance(row.get("token_count"), int)
            or row["token_count"] <= 0
            or row.get("diagnostic_encode_repetitions")
            != TOKENIZER_ENCODE_REPETITIONS
            or not math.isfinite(row.get("diagnostic_encode_median_ms", math.nan))
            or row["diagnostic_encode_median_ms"] <= 0
            or not math.isfinite(
                row.get("diagnostic_encode_megabytes_per_second", math.nan)
            )
            or row["diagnostic_encode_megabytes_per_second"] <= 0
            or not math.isfinite(row.get("bytes_per_token", math.nan))
            or row["bytes_per_token"] <= 0
        ):
            raise ValueError("token frontier tokenizer metric differs")


def _validate_runtime(plan: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    if (
        report.get("schema_version") != 1
        or report.get("kind") != "korean_bpe_systems_frontier_runtime_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("git_commit") != _command("git", "rev-parse", "HEAD")
        or report.get("tokenizer_report_artifact_sha256") != hash_file(OPPORTUNITY_REPORT_PATH)
        or report.get("timing_artifact_sha256") != hash_file(TIMING_PATH)
        or _unsigned_hash(report, "report_sha256") != report.get("report_sha256")
        or report.get("cases") != reconstruct_cases()[2]
        or report.get("model_specs") != {
            role: FRONTIER_SPECS[role].to_dict() for role in RUNTIME_ROLES
        }
    ):
        raise ValueError("token frontier runtime report differs")
    if set(report.get("correctness", {})) != set(RUNTIME_ROLES):
        raise ValueError("token frontier correctness role set differs")
    for role, row in report["correctness"].items():
        if (
            set(row)
            != {
                "argmax_comparisons",
                "cases",
                "comparisons",
                "maximum_normalized_tolerance_ratio",
                "pass",
            }
            or row["cases"] != WARMUP_CASES
            or row["comparisons"] <= 0
            or row["argmax_comparisons"] != row["comparisons"]
            or row["pass"] is not True
            or not math.isfinite(row["maximum_normalized_tolerance_ratio"])
            or not 0 <= row["maximum_normalized_tolerance_ratio"] <= 1
        ):
            raise ValueError(f"token frontier correctness differs: {role}")
    expected_parameters = {
        role: FRONTIER_SPECS[role].expected_parameters for role in RUNTIME_ROLES
    }
    if report.get("parameter_counts") != expected_parameters or any(
        abs(value / PARAMETER_TARGET - 1) > PARAMETER_RELATIVE_TOLERANCE
        for value in expected_parameters.values()
    ):
        raise ValueError("token frontier parameter evidence differs")
    if report.get("environment") != plan["environment"]:
        raise ValueError("token frontier runtime environment identity differs")
    if not all(
        timing_environment_eligible(report["session_state"][key]) for key in ("start", "end")
    ):
        raise ValueError("token frontier timing environment differs")


def _load_arrays(report: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected = {
        f"{component}__{role}"
        for role in RUNTIME_ROLES
        for component in ("ttft_ms", "decode_ms", "end_to_end_ms", "continuation_steps")
    }
    if set(report.get("arrays", {})) != expected:
        raise ValueError("token frontier array descriptor set differs")
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("token frontier timing key set differs")
        arrays = {name: archive[name] for name in archive.files}
    for name, values in arrays.items():
        descriptor = report["arrays"][name]
        expected_dtype = "int64" if name.startswith("continuation_steps__") else "float64"
        if (
            str(values.dtype) != expected_dtype
            or values.shape != (MEASURED_CASES, REPETITIONS)
            or descriptor
            != {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(f"token frontier timing array differs: {name}")
    return arrays


def _comparison(candidate: np.ndarray, reference: np.ndarray, seed_offset: int) -> dict[str, Any]:
    result = paired_prompt_latency(
        candidate,
        reference,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        bootstrap_seed=BOOTSTRAP_SEED + seed_offset,
    ).to_dict()
    candidate_prompt = np.median(candidate, axis=1)
    reference_prompt = np.median(reference, axis=1)
    result["positive_prompt_count"] = int(np.count_nonzero(candidate_prompt < reference_prompt))
    return result


def main() -> None:
    commit = _require_clean_root()
    _never_published(OUTPUT_PATH)
    if RUNTIME_ACTIVE_PATH.exists():
        raise ValueError("token frontier runtime remains active")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _command("git", "rev-parse", "HEAD^") != plan["dependencies"][
        "git_commit_before_plan"
    ]:
        raise ValueError("token frontier plan parent differs")
    tokenizer_report = read_json(OPPORTUNITY_REPORT_PATH)
    runtime_report = read_json(RUNTIME_REPORT_PATH)
    _validate_tokenizers(plan, tokenizer_report)
    _validate_runtime(plan, runtime_report)
    arrays = _load_arrays(runtime_report)
    prompt_medians = {
        role: np.median(arrays[f"end_to_end_ms__{role}"], axis=1) for role in RUNTIME_ROLES
    }
    aggregate_e2e = {role: float(np.median(values)) for role, values in prompt_medians.items()}
    fastest = min(RUNTIME_ROLES, key=lambda role: (aggregate_e2e[role], RUNTIME_ROLES.index(role)))
    fastest_by_vocabulary = {}
    for size in VOCABULARY_SIZES:
        roles = tuple(role_name(size, depth) for depth in DEPTHS)
        fastest_by_vocabulary[str(size)] = min(
            roles, key=lambda role: (aggregate_e2e[role], RUNTIME_ROLES.index(role))
        )
    metrics = {
        role: {
            "vocabulary_size": parse_role(role)[0],
            "depth": parse_role(role)[1],
            "parameters": runtime_report["parameter_counts"][role],
            "ttft_median_ms": float(np.median(np.median(arrays[f"ttft_ms__{role}"], axis=1))),
            "decode_median_ms": float(np.median(np.median(arrays[f"decode_ms__{role}"], axis=1))),
            "end_to_end_median_ms": aggregate_e2e[role],
            "continuation_steps_median": float(
                np.median(np.median(arrays[f"continuation_steps__{role}"], axis=1))
            ),
            "bytes_per_token": tokenizer_report["metrics"][str(parse_role(role)[0])][
                "bytes_per_token"
            ],
        }
        for role in RUNTIME_ROLES
    }
    comparisons = {
        f"{role}_vs_{fastest}": _comparison(
            arrays[f"end_to_end_ms__{role}"],
            arrays[f"end_to_end_ms__{fastest}"],
            index,
        )
        for index, role in enumerate(RUNTIME_ROLES)
    }
    pareto = []
    for role in RUNTIME_ROLES:
        row = metrics[role]
        dominated = any(
            other != role
            and metrics[other]["end_to_end_median_ms"] <= row["end_to_end_median_ms"]
            and metrics[other]["bytes_per_token"] >= row["bytes_per_token"]
            and (
                metrics[other]["end_to_end_median_ms"] < row["end_to_end_median_ms"]
                or metrics[other]["bytes_per_token"] > row["bytes_per_token"]
            )
            for other in RUNTIME_ROLES
        )
        if not dominated:
            pareto.append(role)
    ranked = sorted(RUNTIME_ROLES, key=lambda role: (aggregate_e2e[role], RUNTIME_ROLES.index(role)))
    top_distinct_vocabulary = []
    seen = set()
    for role in ranked:
        size, _ = parse_role(role)
        if size in seen:
            continue
        seen.add(size)
        top_distinct_vocabulary.append(role)
        if len(top_distinct_vocabulary) == 3:
            break
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "korean_bpe_systems_frontier_result_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "tokenizer_report_artifact_sha256": hash_file(OPPORTUNITY_REPORT_PATH),
        "runtime_report_artifact_sha256": hash_file(RUNTIME_REPORT_PATH),
        "timing_artifact_sha256": hash_file(TIMING_PATH),
        "tokenizer_metrics": tokenizer_report["metrics"],
        "runtime_metrics": metrics,
        "comparisons_to_fastest": comparisons,
        "systems_frontier": {
            "fastest_random_weight_role": fastest,
            "fastest_by_vocabulary": fastest_by_vocabulary,
            "latency_compression_pareto_roles": pareto,
            "top_three_distinct_vocabulary_roles": top_distinct_vocabulary,
        },
        "decision": {
            "all_correctness_pass": all(
                row["pass"] for row in runtime_report["correctness"].values()
            ),
            "status": "bpe_quality_frontier_protocol_authorized",
            "random_weight_result_does_not_select_quality_comparator": True,
            "next_stage": (
                "train a preregistered one-seed quality subset drawn from the systems frontier; "
                "then compare any Korean tokenizer against the fastest quality-qualified BPE"
            ),
        },
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during token frontier summary")
    payload = json_bytes(summary)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
