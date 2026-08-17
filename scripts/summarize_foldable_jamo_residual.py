#!/usr/bin/env python3
"""Independently replay and summarize the foldable-Jamo residual screen."""

from __future__ import annotations

import gc
import os
import subprocess
import time
from collections.abc import Mapping

import numpy as np
import torch

from bpe_quality_frontier_core import array_sha256, bpb, raw_target_bytes_by_sequence
from compositional_head_preflight_protocol import load_tokenizers
from foldable_jamo_residual_core import (
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    RESIDUAL_ROLES,
    build_foldable_model,
    build_folded_dense_model,
    expected_parameter_counts,
    residual_decision,
    role_definition,
    state_mapping_sha256,
)
from foldable_jamo_residual_protocol import (
    ACTIVE_PATH,
    BASELINE_RESULT_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    SOURCE_PATH,
    base_checkpoint_state,
    canonical_sha256,
    current_environment,
    hash_file,
    json_bytes,
    read_json,
    target_order,
    validate_plan,
)
from run_foldable_jamo_residual import (
    _cleanup_data,
    _evaluate_contiguous,
    _load_nll,
    _paths,
    _role_data,
    _scheduled_exposure_counts,
    _session_state,
    _validate_worker,
)
from run_compositional_quality import _evaluate_documents
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_baseline_core import build_target_graph
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    EVALUATION_BATCH_SIZE,
    TARGET_VOCABULARY_SIZE,
    build_canonical_bpe_decomposition_table,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _never_published(path) -> None:
    history = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if path.exists() or history:
        raise RuntimeError(f"foldable residual result already exists or has history: {path}")


def _publish(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_campaign(commit: str, plan: Mapping[str, object]) -> dict[str, object]:
    campaign = read_json(REPORT_PATH)
    unsigned = dict(campaign)
    receipt = unsigned.pop("report_sha256", None)
    if (
        set(campaign)
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
        or campaign.get("schema_version") != 1
        or campaign.get("kind") != "foldable_jamo_residual_report_v1"
        or campaign.get("protocol_id") != PROTOCOL_ID
        or campaign.get("complete") is not True
        or campaign.get("git_commit") != commit
        or campaign.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or set(campaign.get("workers", {})) != set(RESIDUAL_ROLES)
    ):
        raise RuntimeError("foldable residual campaign report differs")
    for role in RESIDUAL_ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("foldable residual worker is incomplete")
        report_path = _paths(role)[0]
        if campaign["workers"][role] != {
            "path": str(report_path.relative_to(ROOT)),
            "sha256": hash_file(report_path),
        }:
            raise RuntimeError("foldable residual worker lineage differs")
    return campaign


def _load_base_control(
    architecture: str,
    plan: Mapping[str, object],
    calibration_sequences: np.ndarray,
    raw_target_bytes: np.ndarray,
    data: Mapping[str, object],
) -> tuple[dict[str, float], np.ndarray, dict[str, str]]:
    control = plan["baseline_controls"][architecture]
    checkpoint_path = ROOT / control["final_checkpoint"]["path"]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, Mapping)
        or hash_file(checkpoint_path) != control["final_checkpoint"]["artifact_sha256"]
        or state_mapping_sha256(state) != control["final_checkpoint"]["state_sha256"]
    ):
        raise RuntimeError("foldable residual base control checkpoint differs")
    model = build_target_graph(control["baseline_role"])
    model.load_state_dict(state, strict=True)
    if model_parameter_count(model) != expected_parameter_counts(f"{architecture}_jamo")[
        "deployed"
    ]:
        raise RuntimeError("foldable residual base deployed graph differs")
    model = model.to("mps").eval()
    contiguous = _evaluate_contiguous(model, calibration_sequences)
    document = _evaluate_documents(
        model,
        data["document_chunks"],
        data["chunk_documents"],
        len(data["document_raw_bytes"]),
        EVALUATION_BATCH_SIZE,
    )
    model.to("cpu")
    del model, state
    nll_path = ROOT / control["final_nll"]["path"]
    with np.load(nll_path, allow_pickle=False) as archive:
        if (
            set(archive.files) != {"nll_nats", "raw_target_bytes"}
            or hash_file(nll_path) != control["final_nll"]["artifact_sha256"]
            or array_sha256(archive["nll_nats"]) != control["final_nll"]["array_sha256"]
            or not np.array_equal(contiguous, archive["nll_nats"])
            or not np.array_equal(raw_target_bytes, archive["raw_target_bytes"])
        ):
            raise RuntimeError("foldable residual base NLL replay differs")
    return (
        {
            "contiguous_bpb": bpb(contiguous, raw_target_bytes),
            "document_bpb": bpb(document, data["document_raw_bytes"]),
        },
        document,
        {
            "checkpoint_state_sha256": control["final_checkpoint"]["state_sha256"],
            "contiguous_nll_array_sha256": array_sha256(contiguous),
            "document_nll_array_sha256": array_sha256(document),
        },
    )


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("foldable residual summary requires a clean worktree")
    if ACTIVE_PATH.exists():
        raise RuntimeError("foldable residual campaign is still active")
    _never_published(OUTPUT_PATH)
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("foldable residual summary requires the plan commit")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    campaign = _load_campaign(commit, plan)
    data = _role_data(plan)
    train_count = data["train_inventory"].full_sequence_count
    calibration_count = data["calibration_inventory"].full_sequence_count
    train_sequences = data["train_memory"][: train_count * 512].reshape(train_count, 512)
    calibration_sequences = data["calibration_memory"][: calibration_count * 512].reshape(
        calibration_count, 512
    )
    order = target_order(train_count)
    exposure_counts = _scheduled_exposure_counts(train_sequences, order)
    raw_target_bytes = raw_target_bytes_by_sequence(
        calibration_sequences, data["token_bytes"]
    )
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, _ = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer,
        target_tokenizer,
        base_pieces,
        data["token_bytes"],
    )
    start_state = _session_state()
    if not timing_environment_eligible(start_state):
        raise RuntimeError("foldable residual summary environment is ineligible")
    metrics: dict[str, dict[str, float]] = {}
    recovery_curves: dict[str, dict[str, dict[str, float | int | None]]] = {}
    document_nll: dict[str, np.ndarray] = {}
    replay_hashes: dict[str, object] = {}
    fold_replay: dict[str, object] = {}
    started = time.perf_counter()
    unfolded_checkpoint_count = 0
    with publication_mps_exclusive():
        for role in RESIDUAL_ROLES:
            worker = read_json(_paths(role)[0])
            _, checkpoints, nlls, folded_path = _paths(role)
            role_metrics: dict[str, dict[str, float | int | None]] = {}
            role_hashes: dict[str, str] = {}
            final_model = None
            for step in PROBE_STEPS:
                state = torch.load(checkpoints[step], map_location="cpu", weights_only=True)
                evidence = worker["checkpoints"][str(step)]
                if (
                    not isinstance(state, Mapping)
                    or state_mapping_sha256(state) != evidence["checkpoint_state_sha256"]
                ):
                    raise RuntimeError("foldable residual replay checkpoint differs")
                model, initializer, assignment = build_foldable_model(
                    role,
                    base_state=base_checkpoint_state(),
                    base_tokenizer=base_tokenizer,
                    base_pieces=base_pieces,
                    target_pieces=data["token_bytes"],
                    decompositions=decompositions,
                    exposure_counts=exposure_counts,
                )
                if (
                    initializer.to_dict()
                    != plan["initialization_identities"][role]["base_initializer_audit"]
                    or assignment.to_dict() != plan["assignment_audits"][role]
                ):
                    raise RuntimeError("foldable residual replay initialization differs")
                model.load_state_dict(state, strict=True)
                if model_parameter_count(model) != expected_parameter_counts(role)["training_total"]:
                    raise RuntimeError("foldable residual replay parameter count differs")
                model = model.to("mps").eval()
                contiguous = _evaluate_contiguous(model, calibration_sequences)
                arrays = _load_nll(nlls[step], final=step == FINAL_PROBE_STEP)
                if (
                    not np.array_equal(contiguous, arrays["contiguous_nll_nats"])
                    or not np.array_equal(raw_target_bytes, arrays["contiguous_raw_target_bytes"])
                ):
                    raise RuntimeError("foldable residual independent contiguous replay differs")
                document_value = None
                if step == FINAL_PROBE_STEP:
                    document = _evaluate_documents(
                        model,
                        data["document_chunks"],
                        data["chunk_documents"],
                        len(data["document_raw_bytes"]),
                        EVALUATION_BATCH_SIZE,
                    )
                    if (
                        not np.array_equal(document, arrays["document_nll_nats"])
                        or not np.array_equal(
                            data["document_raw_bytes"], arrays["document_raw_bytes"]
                        )
                    ):
                        raise RuntimeError("foldable residual independent document replay differs")
                    document_nll[role] = document
                    document_value = bpb(document, data["document_raw_bytes"])
                    final_model = model
                else:
                    model.to("cpu")
                    del model
                role_metrics[str(step)] = {
                    "contiguous_bpb": bpb(contiguous, raw_target_bytes),
                    "document_bpb": document_value,
                    "optimizer_steps": step,
                }
                role_hashes[str(step)] = array_sha256(contiguous)
                unfolded_checkpoint_count += 1
                del state, arrays, contiguous
                gc.collect()
                torch.mps.empty_cache()
            if final_model is None:
                raise AssertionError("foldable residual final model is absent")
            independently_folded = build_folded_dense_model(final_model, role)
            expected_folded = torch.load(folded_path, map_location="cpu", weights_only=True)
            if (
                not isinstance(expected_folded, Mapping)
                or state_mapping_sha256(independently_folded.state_dict())
                != state_mapping_sha256(expected_folded)
                or any(
                    not torch.equal(independently_folded.state_dict()[name], value)
                    for name, value in expected_folded.items()
                )
            ):
                raise RuntimeError("foldable residual independent state materialization differs")
            independently_folded = independently_folded.to("mps").eval()
            folded_contiguous = _evaluate_contiguous(
                independently_folded, calibration_sequences
            )
            folded_document = _evaluate_documents(
                independently_folded,
                data["document_chunks"],
                data["chunk_documents"],
                len(data["document_raw_bytes"]),
                EVALUATION_BATCH_SIZE,
            )
            if (
                array_sha256(folded_contiguous) != role_hashes[str(FINAL_PROBE_STEP)]
                or not np.array_equal(folded_document, document_nll[role])
            ):
                raise RuntimeError("foldable residual independent fold NLL differs")
            final_model.to("cpu")
            independently_folded.to("cpu")
            del final_model, independently_folded, expected_folded
            metrics[role] = role_metrics[str(FINAL_PROBE_STEP)]
            recovery_curves[role] = role_metrics
            replay_hashes[role] = role_hashes
            fold_replay[role] = {
                "contiguous_nll_array_sha256": array_sha256(folded_contiguous),
                "deployed_checkpoint_state_sha256": worker["folded_checkpoint"]["state_sha256"],
                "document_nll_array_sha256": array_sha256(folded_document),
                "pass": True,
            }
            del folded_contiguous, folded_document
            gc.collect()
            torch.mps.empty_cache()
        base_replay: dict[str, object] = {}
        for architecture in ("untied", "tied"):
            metric, document, evidence = _load_base_control(
                architecture,
                plan,
                calibration_sequences,
                raw_target_bytes,
                data,
            )
            role = f"{architecture}_base"
            metrics[role] = metric
            document_nll[role] = document
            base_replay[role] = evidence
            gc.collect()
            torch.mps.empty_cache()
    replay_elapsed = time.perf_counter() - started
    end_state = _session_state()
    if not timing_environment_eligible(end_state):
        raise RuntimeError("foldable residual summary environment changed")
    decision = residual_decision(
        {role: float(values["contiguous_bpb"]) for role, values in metrics.items()},
        document_nll,
        data["document_raw_bytes"],
        anchor_bpb=float(plan["parent_anchor"]["contiguous_bpb"]),
    )
    baseline_result = read_json(BASELINE_RESULT_PATH)
    training_diagnostics = {}
    for role in RESIDUAL_ROLES:
        worker = read_json(_paths(role)[0])
        base_role = role_definition(role)["base_initializer_role"]
        baseline_elapsed = baseline_result["worker_summaries"][base_role]["training"][
            "elapsed_seconds_including_checkpoint_evaluations"
        ]
        training_diagnostics[role] = {
            "baseline_session_elapsed_seconds_including_evaluations": baseline_elapsed,
            "residual_session_elapsed_seconds_including_evaluations": worker["training"][
                "elapsed_seconds_including_evaluations"
            ],
            "residual_optimizer_step_elapsed_seconds": worker["training"][
                "optimizer_step_elapsed_seconds"
            ],
            "training_only_parameter_count": worker["parameter_counts"][
                "training_only_residual"
            ],
            "comparison_is_descriptive_separate_session": True,
            "memory_diagnostics": worker["memory_diagnostics"],
        }
    result = {
        "schema_version": 1,
        "kind": "foldable_jamo_residual_result_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "campaign_report_artifact_sha256": hash_file(REPORT_PATH),
        "assignment_audits": plan["assignment_audits"],
        "role_specs": plan["role_specs"],
        "training_contract": plan["training"],
        "selection_rule": plan["selection_rule"],
        "metrics": metrics,
        "recovery_curves": recovery_curves,
        "decision": decision,
        "training_cost_diagnostics": training_diagnostics,
        "independent_nll_recomputation": {
            "base_control_replay": base_replay,
            "fold_replay": fold_replay,
            "pass": True,
            "replay_elapsed_seconds": replay_elapsed,
            "unfolded_checkpoint_count": unfolded_checkpoint_count,
            "unfolded_nll_array_sha256_by_role_step": replay_hashes,
        },
        "artifact_lineage": {
            role: {
                "checkpoints": read_json(_paths(role)[0])["checkpoints"],
                "folded_checkpoint": read_json(_paths(role)[0])["folded_checkpoint"],
            }
            for role in RESIDUAL_ROLES
        },
        "environment": current_environment(),
        "session_state": {"start": start_state, "end": end_state},
        "claim_boundary": plan["claim_boundary"],
    }
    result["summary_sha256"] = canonical_sha256(result)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during foldable residual summary")
    _publish(OUTPUT_PATH, json_bytes(result))
    _cleanup_data(data)
    print(f"status={decision['status']}")
    print(f"result={OUTPUT_PATH.relative_to(ROOT)}")
    print(f"summary_sha256={result['summary_sha256']}")


if __name__ == "__main__":
    main()
