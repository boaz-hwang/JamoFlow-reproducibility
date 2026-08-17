#!/usr/bin/env python3
"""Independently replay and summarize the vocabulary-transfer short-CPT probe."""

from __future__ import annotations

import gc
import os
import subprocess
import time

import numpy as np
import torch
from bpe_quality_feasibility_core import encode_stream_to_memmap
from bpe_quality_frontier_core import array_sha256, bpb, raw_target_bytes_by_sequence
from compositional_head_preflight_protocol import load_tokenizers
from run_vocabulary_transfer_probe import (
    _evaluate_contiguous,
    _paths,
    _session_state,
    _validate_worker,
)
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import (
    FINAL_PROBE_STEP,
    MODEL_SEED,
    PROBE_STEPS,
    TARGET_VOCABULARY_SIZE,
    TRANSFER_ROLES,
    build_target_graph,
    expected_parameter_count,
    state_mapping_sha256,
    transfer_probe_decision,
)
from vocabulary_transfer_probe_protocol import (
    ACTIVE_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    SOURCE_PATH,
    canonical_sha256,
    current_environment,
    hash_file,
    json_bytes,
    read_json,
    validate_plan,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_data import build_neural_stream


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _never_published(path) -> None:
    history = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if path.exists() or history:
        raise RuntimeError(f"vocabulary-transfer result already exists or has history: {path}")


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("vocabulary-transfer summary requires a clean worktree")
    if ACTIVE_PATH.exists():
        raise RuntimeError("vocabulary-transfer worker campaign is still active")
    _never_published(OUTPUT_PATH)
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("vocabulary-transfer summary requires the plan commit")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    report = read_json(REPORT_PATH)
    unsigned = dict(report)
    receipt = unsigned.pop("report_sha256", None)
    if (
        set(report)
        != {
            "complete",
            "git_commit",
            "kind",
            "plan_artifact_sha256",
            "protocol_id",
            "report_sha256",
            "schema_version",
            "workers",
        }
        or canonical_sha256(unsigned) != receipt
        or report.get("schema_version") != 1
        or report.get("kind") != "vocabulary_transfer_probe_report_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or set(report.get("workers", {})) != set(TRANSFER_ROLES)
    ):
        raise RuntimeError("vocabulary-transfer campaign report differs")
    for role in TRANSFER_ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("vocabulary-transfer worker is incomplete")
        path = _paths(role)[0]
        if report["workers"][role] != {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        }:
            raise RuntimeError("vocabulary-transfer worker report lineage differs")

    tokenizer, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    calibration_stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=8_000_000,
        sequence_length=512,
    )
    inventory, memory, memory_path = encode_stream_to_memmap(
        calibration_stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=16 * 512,
    )
    if inventory.to_dict() != plan["inventories"]["8192"]["calibration"]:
        raise RuntimeError("vocabulary-transfer summary inventory differs")
    sequences = memory[: inventory.full_sequence_count * 512].reshape(
        inventory.full_sequence_count, 512
    )
    raw_target_bytes = raw_target_bytes_by_sequence(sequences, token_bytes)
    if int(raw_target_bytes.sum()) != inventory.predicted_target_raw_bytes:
        raise RuntimeError("vocabulary-transfer summary denominator differs")
    start_state = _session_state()
    if not timing_environment_eligible(start_state):
        raise RuntimeError("vocabulary-transfer summary environment is ineligible")
    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    replay_hashes: dict[str, dict[str, str]] = {}
    started = time.perf_counter()
    with publication_mps_exclusive():
        for role in TRANSFER_ROLES:
            worker = read_json(_paths(role)[0])
            metrics[role] = {}
            replay_hashes[role] = {}
            _, checkpoint_paths, nll_paths = _paths(role)
            for step in PROBE_STEPS:
                evidence = worker["checkpoints"][str(step)]
                state = torch.load(checkpoint_paths[step], map_location="cpu", weights_only=True)
                if (
                    not isinstance(state, dict)
                    or state_mapping_sha256(state) != evidence["checkpoint_state_sha256"]
                ):
                    raise RuntimeError("vocabulary-transfer checkpoint state differs")
                model = build_target_graph(role, seed=MODEL_SEED)
                model.load_state_dict(state, strict=True)
                if model_parameter_count(model) != expected_parameter_count(role):
                    raise RuntimeError("vocabulary-transfer parameter count differs")
                model = model.to("mps").eval()
                replayed = _evaluate_contiguous(model, sequences)
                model.to("cpu")
                del model, state
                with np.load(nll_paths[step], allow_pickle=False) as archive:
                    stored = archive["nll_nats"]
                    stored_bytes = archive["raw_target_bytes"]
                    if not np.array_equal(stored, replayed) or not np.array_equal(
                        stored_bytes, raw_target_bytes
                    ):
                        raise RuntimeError("vocabulary-transfer independent replay differs")
                replay_hashes[role][str(step)] = array_sha256(replayed)
                metrics[role][str(step)] = {
                    "contiguous_bpb": bpb(replayed, raw_target_bytes),
                    "optimizer_steps": step,
                }
                del replayed, stored, stored_bytes
                gc.collect()
                torch.mps.empty_cache()
    replay_elapsed = time.perf_counter() - started
    end_state = _session_state()
    if not timing_environment_eligible(end_state):
        raise RuntimeError("vocabulary-transfer summary environment changed")
    final_values = {
        role: float(metrics[role][str(FINAL_PROBE_STEP)]["contiguous_bpb"])
        for role in TRANSFER_ROLES
    }
    decision = transfer_probe_decision(
        final_values, anchor_bpb=float(plan["parent_anchor"]["contiguous_bpb"])
    )
    worker_summaries = {}
    for role in TRANSFER_ROLES:
        worker = read_json(_paths(role)[0])
        worker_summaries[role] = {
            "parameter_count": worker["parameter_count"],
            "training": worker["training"],
            "initialization_audit": worker["initialization_audit"],
        }
    result = {
        "schema_version": 1,
        "kind": "vocabulary_transfer_probe_result_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "campaign_report_artifact_sha256": hash_file(REPORT_PATH),
        "role_specs": plan["role_specs"],
        "training_contract": plan["training"],
        "selection_rule": plan["selection_rule"],
        "worker_summaries": worker_summaries,
        "metrics": metrics,
        "decision": decision,
        "independent_nll_recomputation": {
            "pass": True,
            "array_comparison": "bitwise_equal",
            "checkpoint_count": len(TRANSFER_ROLES) * len(PROBE_STEPS),
            "replay_elapsed_seconds": replay_elapsed,
            "nll_array_sha256_by_role_step": replay_hashes,
        },
        "artifact_lineage": {
            role: read_json(_paths(role)[0])["checkpoints"] for role in TRANSFER_ROLES
        },
        "environment": current_environment(),
        "session_state": {"start": start_state, "end": end_state},
        "claim_boundary": {
            "actual_inference_measured": False,
            "calibration_development_only": True,
            "korean_specific_method_evaluated": False,
            "model_seed_count": 1,
            "publication_quality_claim": False,
            "short_cpt_initializer_selection_only": True,
        },
    }
    result["summary_sha256"] = canonical_sha256(result)
    del memory
    if os.path.exists(memory_path):
        os.unlink(memory_path)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during vocabulary-transfer summary")
    _publish(OUTPUT_PATH, json_bytes(result))
    print(f"status={decision['status']}")
    print(f"result={OUTPUT_PATH.relative_to(ROOT)}")
    print(f"summary_sha256={result['summary_sha256']}")


if __name__ == "__main__":
    main()
