#!/usr/bin/env python3
"""Independently replay and summarize fresh vocabulary adaptation."""

from __future__ import annotations

import gc
import os
import subprocess
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from bpe_quality_frontier_core import array_sha256, bpb
from compositional_head_preflight_protocol import current_environment, hash_file
from fresh_vocabulary_adaptation_core import ROLES, adaptation_decision
from fresh_vocabulary_adaptation_protocol import (
    CHECKPOINT_ROOT,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    WORKER_ROOT,
    canonical_sha256,
    dependency_identity,
    implementation_identity,
    json_bytes,
    read_json,
    validate_plan,
)
from run_fresh_vocabulary_adaptation import (
    _build_initial_model,
    _cleanup_role_data,
    _evaluate_contiguous,
    _evaluate_documents,
    _load_nll,
    _load_role_data,
    _validate_worker,
)
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import state_mapping_sha256

from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _context() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-adaptation summary requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT)))
        != commit
    ):
        raise RuntimeError("fresh-adaptation plan must be current HEAD")
    if OUTPUT_PATH.exists() or _git(
        "log", "--all", "--format=%H", "--", str(OUTPUT_PATH.relative_to(ROOT))
    ):
        raise RuntimeError("fresh-adaptation summary cannot be resealed")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if (
        plan["dependencies"] != dependency_identity()
        or plan["implementation_sha256"] != implementation_identity()
        or plan["environment"] != current_environment()
    ):
        raise RuntimeError("fresh-adaptation summary context differs")
    report = read_json(REPORT_PATH)
    unsigned = dict(report)
    report_sha = unsigned.pop("report_sha256", None)
    if (
        canonical_sha256(unsigned) != report_sha
        or report.get("schema_version") != 1
        or report.get("kind") != "fresh_vocabulary_adaptation_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or set(report.get("workers", {})) != set(ROLES)
    ):
        raise RuntimeError("fresh-adaptation campaign report differs")
    for role in ROLES:
        path = WORKER_ROOT / f"{role}.json"
        if report["workers"][role] != {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        } or not _validate_worker(role, commit, plan):
            raise RuntimeError("fresh-adaptation worker lineage differs")
    return commit, plan


def _independent_replay(
    role: str, plan: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = _load_role_data(role, plan)
    checkpoint_path = CHECKPOINT_ROOT / f"{role}.pt"
    stored_path = NLL_ROOT / f"{role}.npz"
    worker = read_json(WORKER_ROOT / f"{role}.json")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(state, Mapping)
            or state_mapping_sha256(state) != worker["checkpoint_state_sha256"]
        ):
            raise RuntimeError("fresh-adaptation replay checkpoint differs")
        model = _build_initial_model(role, plan, data)
        model.load_state_dict(state, strict=True)
        if model_parameter_count(model) != worker["parameter_count"]:
            raise RuntimeError("fresh-adaptation replay parameter count differs")
        started = time.perf_counter()
        with publication_mps_exclusive():
            model = model.to("mps").eval()
            batch_size = plan["training"][role]["evaluation_batch_size"]
            contiguous = _evaluate_contiguous(
                model, data["calibration_sequences"], batch_size
            )
            document = _evaluate_documents(
                model,
                data["document_chunks"],
                data["chunk_documents"],
                len(data["document_raw_bytes"]),
                batch_size,
            )
            model.to("cpu")
            del model
            gc.collect()
            torch.mps.empty_cache()
        arrays = {
            "contiguous_nll_nats": contiguous,
            "contiguous_raw_target_bytes": np.ascontiguousarray(
                data["calibration_raw_target_bytes"], dtype=np.int64
            ),
            "document_nll_nats": document,
            "document_raw_bytes": np.ascontiguousarray(
                data["document_raw_bytes"], dtype=np.int64
            ),
        }
        stored = _load_nll(stored_path)
        if set(arrays) != set(stored) or any(
            not np.array_equal(arrays[name], stored[name]) for name in arrays
        ):
            raise RuntimeError("fresh-adaptation independent replay differs")
        evidence = {
            "checkpoint_artifact_sha256": hash_file(checkpoint_path),
            "checkpoint_state_sha256": state_mapping_sha256(state),
            "nll_artifact_sha256": hash_file(stored_path),
            "array_sha256": {
                name: array_sha256(value) for name, value in arrays.items()
            },
            "bitwise_equal_to_worker": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        return arrays, evidence
    finally:
        _cleanup_role_data(data)


def main() -> None:
    commit, plan = _context()
    arrays_by_role: dict[str, dict[str, np.ndarray]] = {}
    replay: dict[str, Any] = {}
    workers: dict[str, Any] = {}
    for role in ROLES:
        arrays_by_role[role], replay[role] = _independent_replay(role, plan)
        workers[role] = read_json(WORKER_ROOT / f"{role}.json")
    reference_raw = arrays_by_role[ROLES[0]]["document_raw_bytes"]
    if any(
        not np.array_equal(reference_raw, arrays_by_role[role]["document_raw_bytes"])
        for role in ROLES[1:]
    ):
        raise RuntimeError("fresh-adaptation document denominators differ across roles")
    decision = adaptation_decision(
        {role: arrays_by_role[role]["document_nll_nats"] for role in ROLES},
        reference_raw,
    )
    metrics = {
        role: {
            "contiguous_bpb": bpb(
                arrays_by_role[role]["contiguous_nll_nats"],
                arrays_by_role[role]["contiguous_raw_target_bytes"],
            ),
            "document_bpb": bpb(
                arrays_by_role[role]["document_nll_nats"],
                arrays_by_role[role]["document_raw_bytes"],
            ),
            "optimizer_steps": workers[role]["training"]["optimizer_steps"],
            "optimizer_elapsed_seconds": workers[role]["training"][
                "optimizer_elapsed_seconds"
            ],
            "parameter_count": workers[role]["parameter_count"],
            "checkpoint_bytes": (CHECKPOINT_ROOT / f"{role}.pt").stat().st_size,
        }
        for role in ROLES
    }
    baseline_steps = metrics["dense2k_joint"]["optimizer_steps"]
    baseline_parameters = metrics["dense2k_joint"]["parameter_count"]
    systems = {
        role: {
            "optimizer_step_reduction_vs_dense2k": 1.0
            - metrics[role]["optimizer_steps"] / baseline_steps,
            "training_time_reduction_vs_dense2k": 1.0
            - metrics[role]["optimizer_elapsed_seconds"]
            / metrics["dense2k_joint"]["optimizer_elapsed_seconds"],
            "parameter_increase_vs_dense2k": metrics[role]["parameter_count"]
            / baseline_parameters
            - 1.0,
        }
        for role in ROLES[1:]
    }
    selected = decision["selected_dense8k_role_for_actual_preflight"]
    selected_checkpoint = (
        {
            "role": selected,
            "path": str((CHECKPOINT_ROOT / f"{selected}.pt").relative_to(ROOT)),
            "artifact_sha256": hash_file(CHECKPOINT_ROOT / f"{selected}.pt"),
            "state_sha256": workers[selected]["checkpoint_state_sha256"],
        }
        if selected is not None
        else None
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_adaptation_one_seed_result_v1",
        "protocol_id": PROTOCOL_ID,
        "status": decision["status"],
        "git_commit": commit,
        "plan": {
            "path": str(PLAN_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(PLAN_PATH),
            "payload_sha256": plan["plan_sha256"],
        },
        "campaign_report": {
            "path": str(REPORT_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(REPORT_PATH),
        },
        "metrics": metrics,
        "systems_accounting": systems,
        "decision": decision,
        "selected_checkpoint_for_actual_preflight": selected_checkpoint,
        "artifact_lineage": {
            role: {
                "worker_sha256": hash_file(WORKER_ROOT / f"{role}.json"),
                "checkpoint_sha256": hash_file(CHECKPOINT_ROOT / f"{role}.pt"),
                "nll_sha256": hash_file(NLL_ROOT / f"{role}.npz"),
            }
            for role in ROLES
        },
        "independent_nll_recomputation": {
            "pass": True,
            "checkpoint_count": len(ROLES),
            "comparator": "bitwise_float_arrays",
            "by_role": replay,
        },
        "claim_boundary": plan["claim_boundary"],
    }
    result["summary_sha256"] = canonical_sha256(result)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during fresh-adaptation summary")
    _publish(OUTPUT_PATH, json_bytes(result))
    print(f"status={result['status']}")
    print(f"summary_sha256={result['summary_sha256']}")
    print(f"selected={selected}")


if __name__ == "__main__":
    main()
