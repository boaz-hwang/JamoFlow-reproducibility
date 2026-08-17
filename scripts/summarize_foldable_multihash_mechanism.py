#!/usr/bin/env python3
"""Independently replay and summarize the multi-hash mechanism controls."""

from __future__ import annotations

import gc
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bpe_quality_frontier_core import array_sha256, bpb, raw_target_bytes_by_sequence
from compositional_head_preflight_protocol import load_tokenizers
from foldable_jamo_residual_core import (
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    build_foldable_model,
    build_folded_dense_model,
    expected_parameter_counts,
)
from foldable_jamo_residual_protocol import (
    base_checkpoint_state,
    current_environment,
    hash_file,
    read_json,
    target_order,
)
from foldable_multihash_mechanism_core import (
    ALL_ROLES,
    NEW_ROLES,
    mechanism_decision,
)
from foldable_multihash_mechanism_protocol import (
    ACTIVE_ROOT,
    OUTPUT_PATH,
    PARENT_PLAN_PATH,
    PARENT_RESULT_PATH,
    PLAN_PATH,
    REPORT_PATH,
    ROOT,
    canonical_sha256,
    json_bytes,
    validate_plan,
    worker_paths,
)
from run_compositional_quality import _evaluate_documents
from run_foldable_jamo_residual import (
    _cleanup_data,
    _evaluate_contiguous,
    _role_data,
    _scheduled_exposure_counts,
    _session_state,
)
from run_foldable_jamo_residual import (
    _load_nll as load_parent_nll,
)
from run_foldable_jamo_residual import (
    _paths as parent_paths,
)
from run_foldable_jamo_residual import (
    _validate_worker as validate_parent_worker,
)
from run_foldable_multihash_mechanism import (
    _build_initial_model,
    _load_nll,
    _validate_worker,
)
from scalar_runtime_core import model_parameter_count
from summarize_foldable_jamo_residual import _load_base_control
from vocabulary_transfer_baseline_core import state_mapping_sha256
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


def _history(path: Path) -> tuple[str, ...]:
    output = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    return tuple(line for line in output.splitlines() if line)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_campaign(
    commit: str, plan: Mapping[str, Any]
) -> dict[str, Any]:
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
        or campaign.get("kind") != "foldable_multihash_mechanism_campaign_v1"
        or campaign.get("protocol_id") != plan["protocol_id"]
        or campaign.get("complete") is not True
        or campaign.get("git_commit") != commit
        or campaign.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or set(campaign.get("workers", {})) != set(NEW_ROLES)
    ):
        raise RuntimeError("mechanism campaign report differs")
    for role in NEW_ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("mechanism worker is incomplete")
        worker_path = worker_paths(role)[0]
        if campaign["workers"][role] != {
            "path": str(worker_path.relative_to(ROOT)),
            "sha256": hash_file(worker_path),
        }:
            raise RuntimeError("mechanism campaign worker lineage differs")
    return campaign


def _replay_parent_generic(
    parent_plan: Mapping[str, Any],
    parent_result: Mapping[str, Any],
    data: Mapping[str, Any],
    calibration_sequences: np.ndarray,
    exposure_counts: np.ndarray,
    raw_target_bytes: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    role = "untied_generic_surface"
    parent_commit = parent_result["git_commit"]
    if not validate_parent_worker(role, parent_commit, parent_plan):
        raise RuntimeError("mechanism parent generic worker is incomplete")
    worker = read_json(parent_paths(role)[0])
    state_path = parent_paths(role)[1][FINAL_PROBE_STEP]
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    row = worker["checkpoints"][str(FINAL_PROBE_STEP)]
    if (
        not isinstance(state, Mapping)
        or hash_file(state_path) != row["checkpoint_artifact_sha256"]
        or state_mapping_sha256(state) != row["checkpoint_state_sha256"]
    ):
        raise RuntimeError("mechanism parent generic checkpoint differs")
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, _ = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer,
        target_tokenizer,
        base_pieces,
        data["token_bytes"],
    )
    model, _, _ = build_foldable_model(
        role,
        base_state=base_checkpoint_state(),
        base_tokenizer=base_tokenizer,
        base_pieces=base_pieces,
        target_pieces=data["token_bytes"],
        decompositions=decompositions,
        exposure_counts=exposure_counts,
    )
    model.load_state_dict(state, strict=True)
    model = model.to("mps").eval()
    contiguous = _evaluate_contiguous(model, calibration_sequences)
    document = _evaluate_documents(
        model,
        data["document_chunks"],
        data["chunk_documents"],
        len(data["document_raw_bytes"]),
        EVALUATION_BATCH_SIZE,
    )
    arrays = load_parent_nll(
        parent_paths(role)[2][FINAL_PROBE_STEP], final=True
    )
    if (
        not np.array_equal(contiguous, arrays["contiguous_nll_nats"])
        or not np.array_equal(document, arrays["document_nll_nats"])
        or not np.array_equal(raw_target_bytes, arrays["contiguous_raw_target_bytes"])
        or not np.array_equal(data["document_raw_bytes"], arrays["document_raw_bytes"])
    ):
        raise RuntimeError("mechanism parent generic NLL replay differs")
    folded = build_folded_dense_model(model, role)
    deployed_path = parent_paths(role)[3]
    expected_deployed = torch.load(
        deployed_path, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(expected_deployed, Mapping)
        or hash_file(deployed_path)
        != worker["folded_checkpoint"]["artifact_sha256"]
        or state_mapping_sha256(folded.state_dict())
        != state_mapping_sha256(expected_deployed)
    ):
        raise RuntimeError("mechanism parent generic fold replay differs")
    model.to("cpu")
    del model, folded, state, expected_deployed, arrays
    gc.collect()
    torch.mps.empty_cache()
    metric = {
        "contiguous_bpb": bpb(contiguous, raw_target_bytes),
        "document_bpb": bpb(document, data["document_raw_bytes"]),
    }
    expected_metric = parent_result["metrics"][role]
    if (
        metric["contiguous_bpb"] != expected_metric["contiguous_bpb"]
        or metric["document_bpb"] != expected_metric["document_bpb"]
    ):
        raise RuntimeError("mechanism parent generic metric differs")
    return metric, document, {
        "checkpoint_state_sha256": row["checkpoint_state_sha256"],
        "contiguous_nll_array_sha256": array_sha256(contiguous),
        "document_nll_array_sha256": array_sha256(document),
        "deployed_checkpoint_state_sha256": worker["folded_checkpoint"][
            "state_sha256"
        ],
    }


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("mechanism summary requires a clean worktree")
    if any(ACTIVE_ROOT.glob("*")):
        raise RuntimeError("mechanism campaign is still active")
    if OUTPUT_PATH.exists() or _history(OUTPUT_PATH):
        raise RuntimeError("mechanism result already exists or has history")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("mechanism summary requires the plan commit")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=True)
    _load_campaign(commit, plan)
    parent_plan = read_json(PARENT_PLAN_PATH)
    parent_result = read_json(PARENT_RESULT_PATH)
    data = _role_data(parent_plan)
    try:
        train_count = int(data["train_inventory"].full_sequence_count)
        calibration_count = int(data["calibration_inventory"].full_sequence_count)
        train_sequences = data["train_memory"][: train_count * 512].reshape(
            train_count, 512
        )
        calibration_sequences = data["calibration_memory"][: calibration_count * 512].reshape(
            calibration_count, 512
        )
        order = target_order(train_count)
        exposure_counts = _scheduled_exposure_counts(train_sequences, order)
        raw_target_bytes = raw_target_bytes_by_sequence(
            calibration_sequences, data["token_bytes"]
        )
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise RuntimeError("mechanism summary environment is ineligible")
        metrics: dict[str, dict[str, float]] = {}
        document_nll: dict[str, np.ndarray] = {}
        recovery_curves: dict[str, dict[str, Any]] = {}
        replay: dict[str, Any] = {}
        started = time.perf_counter()
        replay_count = 0
        with publication_mps_exclusive():
            for role in NEW_ROLES:
                worker = read_json(worker_paths(role)[0])
                checkpoints = worker_paths(role)[1]
                nlls = worker_paths(role)[2]
                final_model = None
                role_curve: dict[str, Any] = {}
                role_hashes: dict[str, str] = {}
                for step in PROBE_STEPS:
                    state = torch.load(
                        checkpoints[step], map_location="cpu", weights_only=True
                    )
                    evidence = worker["checkpoints"][str(step)]
                    if (
                        not isinstance(state, Mapping)
                        or state_mapping_sha256(state)
                        != evidence["checkpoint_state_sha256"]
                    ):
                        raise RuntimeError("mechanism replay state differs")
                    model, _ = _build_initial_model(
                        role, plan, data, exposure_counts
                    )
                    model.load_state_dict(state, strict=True)
                    model = model.to("mps").eval()
                    contiguous = _evaluate_contiguous(
                        model, calibration_sequences
                    )
                    arrays = _load_nll(
                        nlls[step], final=step == FINAL_PROBE_STEP
                    )
                    if (
                        not np.array_equal(
                            contiguous, arrays["contiguous_nll_nats"]
                        )
                        or not np.array_equal(
                            raw_target_bytes,
                            arrays["contiguous_raw_target_bytes"],
                        )
                    ):
                        raise RuntimeError("mechanism replay contiguous NLL differs")
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
                            not np.array_equal(
                                document, arrays["document_nll_nats"]
                            )
                            or not np.array_equal(
                                data["document_raw_bytes"],
                                arrays["document_raw_bytes"],
                            )
                        ):
                            raise RuntimeError(
                                "mechanism replay document NLL differs"
                            )
                        document_nll[role] = document
                        document_value = bpb(
                            document, data["document_raw_bytes"]
                        )
                        final_model = model
                    else:
                        model.to("cpu")
                        del model
                    role_curve[str(step)] = {
                        "optimizer_steps": step,
                        "contiguous_bpb": bpb(
                            contiguous, raw_target_bytes
                        ),
                        "document_bpb": document_value,
                    }
                    role_hashes[str(step)] = array_sha256(contiguous)
                    replay_count += 1
                    del state, arrays, contiguous
                    gc.collect()
                    torch.mps.empty_cache()
                if final_model is None:
                    raise AssertionError("mechanism final replay model is absent")
                if role == "update_matched_dense":
                    deployed = final_model
                else:
                    deployed = build_folded_dense_model(
                        final_model, "untied_generic_surface"
                    )
                expected_state = torch.load(
                    worker_paths(role)[3], map_location="cpu", weights_only=True
                )
                if (
                    not isinstance(expected_state, Mapping)
                    or state_mapping_sha256(deployed.state_dict())
                    != state_mapping_sha256(expected_state)
                    or model_parameter_count(deployed)
                    != expected_parameter_counts("untied_generic_surface")[
                        "deployed"
                    ]
                ):
                    raise RuntimeError("mechanism deployed state replay differs")
                deployed = deployed.to("mps").eval()
                deployed_contiguous = _evaluate_contiguous(
                    deployed, calibration_sequences
                )
                deployed_document = _evaluate_documents(
                    deployed,
                    data["document_chunks"],
                    data["chunk_documents"],
                    len(data["document_raw_bytes"]),
                    EVALUATION_BATCH_SIZE,
                )
                if (
                    not np.array_equal(
                        deployed_contiguous,
                        _load_nll(nlls[FINAL_PROBE_STEP], final=True)[
                            "contiguous_nll_nats"
                        ],
                    )
                    or not np.array_equal(
                        deployed_document, document_nll[role]
                    )
                ):
                    raise RuntimeError("mechanism deployed NLL replay differs")
                final_model.to("cpu")
                deployed.to("cpu")
                del final_model, deployed, expected_state
                metrics[role] = {
                    "contiguous_bpb": role_curve[str(FINAL_PROBE_STEP)][
                        "contiguous_bpb"
                    ],
                    "document_bpb": role_curve[str(FINAL_PROBE_STEP)][
                        "document_bpb"
                    ],
                }
                recovery_curves[role] = role_curve
                replay[role] = {
                    "checkpoint_nll_sha256_by_step": role_hashes,
                    "deployed_contiguous_nll_array_sha256": array_sha256(
                        deployed_contiguous
                    ),
                    "deployed_document_nll_array_sha256": array_sha256(
                        deployed_document
                    ),
                    "deployed_state_sha256": worker["deployed_checkpoint"][
                        "state_sha256"
                    ],
                }
                del deployed_contiguous, deployed_document
                gc.collect()
                torch.mps.empty_cache()

            generic_metric, generic_document, generic_replay = (
                _replay_parent_generic(
                    parent_plan,
                    parent_result,
                    data,
                    calibration_sequences,
                    exposure_counts,
                    raw_target_bytes,
                )
            )
            metrics["untied_generic_surface"] = generic_metric
            document_nll["untied_generic_surface"] = generic_document
            replay["untied_generic_surface"] = generic_replay
            base_metric, base_document, base_replay = _load_base_control(
                "untied",
                parent_plan,
                calibration_sequences,
                raw_target_bytes,
                data,
            )
            metrics["untied_base"] = base_metric
            document_nll["untied_base"] = base_document
            replay["untied_base"] = base_replay
        if set(metrics) != set(ALL_ROLES) or set(document_nll) != set(ALL_ROLES):
            raise AssertionError("mechanism replay role set differs")
        decision = mechanism_decision(
            {role: row["contiguous_bpb"] for role, row in metrics.items()},
            document_nll,
            data["document_raw_bytes"],
            anchor_bpb=float(parent_plan["parent_anchor"]["contiguous_bpb"]),
        )
        replay_elapsed = time.perf_counter() - started
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError("mechanism summary environment changed")
        result: dict[str, Any] = {
            "schema_version": 1,
            "kind": "foldable_multihash_mechanism_result_v1",
            "protocol_id": plan["protocol_id"],
            "complete": True,
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "campaign_report_artifact_sha256": hash_file(REPORT_PATH),
            "roles": plan["roles"],
            "assignment_audits": plan["assignment_audits"],
            "update_control": plan["update_control"],
            "training_contract": plan["training"],
            "selection_rule": plan["selection_rule"],
            "metrics": metrics,
            "recovery_curves": recovery_curves,
            "decision": decision,
            "training_diagnostics": {
                role: {
                    "training": read_json(worker_paths(role)[0])["training"],
                    "memory_diagnostics": read_json(worker_paths(role)[0])[
                        "memory_diagnostics"
                    ],
                }
                for role in NEW_ROLES
            },
            "independent_recomputation": {
                "pass": True,
                "checkpoint_count": replay_count,
                "elapsed_seconds": replay_elapsed,
                "roles": replay,
            },
            "artifact_lineage": {
                role: {
                    "worker_path": str(worker_paths(role)[0].relative_to(ROOT)),
                    "worker_artifact_sha256": hash_file(worker_paths(role)[0]),
                    "checkpoints": read_json(worker_paths(role)[0])["checkpoints"],
                    "deployed_checkpoint": read_json(worker_paths(role)[0])[
                        "deployed_checkpoint"
                    ],
                }
                for role in NEW_ROLES
            },
            "environment": current_environment(),
            "session_state": {"start": start_state, "end": end_state},
            "claim_boundary": plan["claim_boundary"],
        }
        result["summary_sha256"] = canonical_sha256(result)
        if _git("rev-parse", "HEAD") != commit or _git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("repository changed during mechanism summary")
        _publish(OUTPUT_PATH, json_bytes(result))
        print(f"status={decision['status']}")
        print(f"result={OUTPUT_PATH.relative_to(ROOT)}")
        print(f"summary_sha256={result['summary_sha256']}")
    finally:
        _cleanup_data(data)


if __name__ == "__main__":
    main()
