#!/usr/bin/env python3
"""Validate BPE quality feasibility and choose the largest time-safe byte budget."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Mapping
from typing import Any

import numpy as np
from bpe_quality_feasibility_core import (
    CAMPAIGN_HOUR_LIMIT,
    CANDIDATE_TRAIN_BYTE_BUDGETS,
    DRIVER_MEMORY_FRACTION_LIMIT,
    EVALUATION_BATCH_BY_VOCABULARY,
    MEASURED_EFFECTIVE_STEPS,
    MEASURED_EVALUATION_BATCHES,
    QUALITY_ROLES,
    projected_optimizer_steps,
    quality_role_contract,
)
from bpe_quality_feasibility_protocol import (
    ACTIVE_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    canonical_sha256,
    hash_file,
    json_bytes,
    read_json,
    validate_plan,
)
from token_frontier_core import FRONTIER_SPECS, parse_role

from jamoflow.actual_inference_protocol import timing_environment_eligible


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command("git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if history:
        raise FileExistsError(f"BPE quality feasibility result has Git history: {path}")


def _finite_positive(values: object, expected_length: int) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError("BPE quality feasibility timing length differs")
    output = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in output):
        raise ValueError("BPE quality feasibility timing value differs")
    return output


def _validate_worker(
    plan: Mapping[str, Any],
    role: str,
    entry: Mapping[str, Any],
    commit: str,
) -> dict[str, Any]:
    if set(entry) != {"path", "sha256"}:
        raise ValueError("BPE quality feasibility worker descriptor differs")
    path = ROOT / entry["path"]
    if entry["sha256"] != hash_file(path):
        raise ValueError("BPE quality feasibility worker artifact differs")
    row = read_json(path)
    unsigned = dict(row)
    expected_hash = unsigned.pop("worker_sha256")
    vocabulary, _ = parse_role(role)
    if (
        canonical_sha256(unsigned) != expected_hash
        or row.get("schema_version") != 1
        or row.get("kind") != "bpe_quality_frontier_feasibility_worker_v1"
        or row.get("protocol_id") != PROTOCOL_ID
        or row.get("complete") is not True
        or row.get("git_commit") != commit
        or row.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or row.get("role") != role
        or row.get("contract") != quality_role_contract(role, vocabulary)
        or row.get("parameter_count") != FRONTIER_SPECS[role].expected_parameters
        or row.get("train_inventory") != plan["inventories"][role]["train"]
        or row.get("calibration_inventory")
        != plan["inventories"][role]["calibration"]
        or row.get("environment") != plan["environment"]
        or row.get("loss_values_recorded") is not False
    ):
        raise ValueError(f"BPE quality feasibility worker identity differs: {role}")
    train = _finite_positive(
        row.get("train_effective_step_seconds"), MEASURED_EFFECTIVE_STEPS
    )
    evaluation = _finite_positive(
        row.get("evaluation_batch_seconds"), MEASURED_EVALUATION_BATCHES
    )
    if not all(
        timing_environment_eligible(row["session_state"][key])
        for key in ("start", "end")
    ):
        raise ValueError("BPE quality feasibility worker environment is ineligible")
    memory = row.get("memory")
    if not isinstance(memory, Mapping) or memory.get("resettable_peak_supported") is not False:
        raise ValueError("BPE quality feasibility memory contract differs")
    for section in ("baseline", "maximum_sampled", "released"):
        values = memory.get(section)
        if not isinstance(values, Mapping) or set(values) != {
            "current_allocated_bytes",
            "driver_allocated_bytes",
            "process_max_rss_bytes",
            "recommended_max_bytes",
        }:
            raise ValueError("BPE quality feasibility memory schema differs")
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("BPE quality feasibility memory value differs")
    maximum = memory["maximum_sampled"]
    baseline = memory["baseline"]
    if (
        maximum["recommended_max_bytes"] <= 0
        or maximum["driver_allocated_bytes"] < baseline["driver_allocated_bytes"]
        or maximum["current_allocated_bytes"] < baseline["current_allocated_bytes"]
    ):
        raise ValueError("BPE quality feasibility memory ordering differs")
    return {
        "train_effective_step_median_seconds": float(np.median(train)),
        "evaluation_batch_median_seconds": float(np.median(evaluation)),
        "driver_fraction": maximum["driver_allocated_bytes"]
        / maximum["recommended_max_bytes"],
        "process_rss_fraction": maximum["process_max_rss_bytes"]
        / plan["environment"]["hardware"]["memory_bytes"],
        "worker_artifact_sha256": entry["sha256"],
    }


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("BPE quality feasibility summary requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    _never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists():
        raise ValueError("BPE quality feasibility remains active")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    report = read_json(REPORT_PATH)
    unsigned = dict(report)
    report_hash = unsigned.pop("report_sha256", None)
    if (
        canonical_sha256(unsigned) != report_hash
        or report.get("schema_version") != 1
        or report.get("kind") != "bpe_quality_frontier_feasibility_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or set(report.get("workers", {})) != set(QUALITY_ROLES)
    ):
        raise ValueError("BPE quality feasibility report differs")
    role_metrics = {
        role: _validate_worker(plan, role, report["workers"][role], commit)
        for role in QUALITY_ROLES
    }
    budgets: dict[str, Any] = {}
    selected = None
    all_memory_pass = all(
        row["driver_fraction"] <= DRIVER_MEMORY_FRACTION_LIMIT
        and row["process_rss_fraction"] <= DRIVER_MEMORY_FRACTION_LIMIT
        for row in role_metrics.values()
    )
    for budget in CANDIDATE_TRAIN_BYTE_BUDGETS:
        roles = {}
        total_seconds = 0.0
        for role in QUALITY_ROLES:
            vocabulary, _ = parse_role(role)
            train_sequences = plan["inventories"][role]["train"][
                "full_sequence_count"
            ]
            steps = projected_optimizer_steps(train_sequences, budget)
            eval_sequences = plan["inventories"][role]["calibration"][
                "full_sequence_count"
            ]
            evaluation_batches = math.ceil(
                eval_sequences / EVALUATION_BATCH_BY_VOCABULARY[vocabulary]
            )
            train_seconds = (
                steps * role_metrics[role]["train_effective_step_median_seconds"]
            )
            evaluation_seconds = (
                evaluation_batches
                * role_metrics[role]["evaluation_batch_median_seconds"]
            )
            role_seconds = train_seconds + evaluation_seconds
            total_seconds += role_seconds
            roles[role] = {
                "evaluation_batches": evaluation_batches,
                "evaluation_seconds": evaluation_seconds,
                "optimizer_steps": steps,
                "projected_hours": role_seconds / 3600,
                "train_seconds": train_seconds,
            }
        passes = all_memory_pass and total_seconds / 3600 <= CAMPAIGN_HOUR_LIMIT
        budgets[str(budget)] = {
            "all_memory_pass": all_memory_pass,
            "campaign_hour_limit": CAMPAIGN_HOUR_LIMIT,
            "passes": passes,
            "projected_campaign_hours": total_seconds / 3600,
            "roles": roles,
        }
        if selected is None and passes:
            selected = budget
    status = (
        "quality_frontier_training_authorized"
        if selected is not None
        else "quality_frontier_scale_or_budget_revision_required"
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_feasibility_result_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "report_artifact_sha256": hash_file(REPORT_PATH),
        "role_metrics": role_metrics,
        "budget_projections": budgets,
        "decision": {
            "selected_train_raw_bytes": selected,
            "status": status,
            "quality_or_loss_used": False,
            "roles": list(QUALITY_ROLES),
        },
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during BPE quality feasibility summary")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(summary))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
