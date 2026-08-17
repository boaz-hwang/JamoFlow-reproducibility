#!/usr/bin/env python3
"""Run the sealed foldable multi-hash mechanism controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bpe_quality_frontier_core import (
    array_sha256,
    bpb,
    raw_target_bytes_by_sequence,
)
from compositional_head_preflight_protocol import load_tokenizers
from foldable_jamo_residual_core import (
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    build_foldable_model,
    build_folded_dense_model,
    expected_parameter_counts,
    fold_audit,
)
from foldable_jamo_residual_protocol import (
    base_checkpoint_state,
    current_environment,
    hash_file,
    read_json,
    target_order,
    training_contract,
)
from foldable_multihash_mechanism_core import (
    INPUT_UPDATE_MULTIPLIER,
    NEW_ROLES,
    OUTPUT_UPDATE_MULTIPLIER,
    assignment_audit,
    balanced_random_assignment,
    generic_assignment_from_code_indices,
    install_assignment,
    scale_new_row_update_,
    stratified_generic_shuffle,
)
from foldable_multihash_mechanism_protocol import (
    ACTIVE_ROOT,
    OUTPUT_PATH,
    PARENT_PLAN_PATH,
    PLAN_PATH,
    REPORT_PATH,
    ROOT,
    canonical_sha256,
    json_bytes,
    role_definition,
    validate_plan,
    worker_paths,
)
from run_compositional_quality import _evaluate_documents
from run_foldable_jamo_residual import (
    _array_metadata,
    _checkpoint_bytes,
    _cleanup_data,
    _evaluate_contiguous,
    _load_arrays_from_payload,
    _memory_snapshot,
    _npz_bytes,
    _role_data,
    _scheduled_exposure_counts,
    _session_state,
    _set_learning_rates,
)
from run_foldable_jamo_residual import (
    _effective_step as residual_effective_step,
)
from run_foldable_jamo_residual import (
    _optimizer as residual_optimizer,
)
from run_vocabulary_transfer_baseline import _all_parameter_optimizer as dense_optimizer
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_baseline_core import state_mapping_sha256
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_SIZE,
    build_canonical_bpe_decomposition_table,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _publish(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _context() -> tuple[str, dict[str, Any], dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("mechanism campaign requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("mechanism plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=True)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("mechanism plan parent differs")
    parent_plan = read_json(PARENT_PLAN_PATH)
    return commit, plan, parent_plan


def _assignments(
    state: Mapping[str, torch.Tensor],
    data: Mapping[str, Any],
    exposure_counts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    generic = generic_assignment_from_code_indices(
        state["foldable_residual.code_indices"]
    )
    shuffled, shuffled_construction = stratified_generic_shuffle(
        generic, data["token_bytes"], exposure_counts
    )
    balanced, balanced_construction = balanced_random_assignment(generic)
    values = {
        "stratified_generic_shuffle": shuffled,
        "balanced_random_multihash": balanced,
    }
    audits = {
        "stratified_generic_shuffle": assignment_audit(
            shuffled,
            generic,
            exposure_counts,
            kind="stratified_generic_shuffle",
            construction=shuffled_construction,
        ),
        "balanced_random_multihash": assignment_audit(
            balanced,
            generic,
            exposure_counts,
            kind="balanced_random_multihash",
            construction=balanced_construction,
        ),
    }
    return values, audits


def _build_initial_model(
    role: str,
    plan: Mapping[str, Any],
    data: Mapping[str, Any],
    exposure_counts: np.ndarray,
) -> tuple[Any, dict[str, Any] | None]:
    source = plan["initialization"]
    checkpoint_path = ROOT / source["step_zero_checkpoint_path"]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, Mapping)
        or hash_file(checkpoint_path)
        != source["step_zero_checkpoint_artifact_sha256"]
        or state_mapping_sha256(state) != source["step_zero_checkpoint_state_sha256"]
    ):
        raise RuntimeError("mechanism source checkpoint differs")
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, _ = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer,
        target_tokenizer,
        base_pieces,
        data["token_bytes"],
    )
    generic_model, _, _ = build_foldable_model(
        "untied_generic_surface",
        base_state=base_checkpoint_state(),
        base_tokenizer=base_tokenizer,
        base_pieces=base_pieces,
        target_pieces=data["token_bytes"],
        decompositions=decompositions,
        exposure_counts=exposure_counts,
    )
    generic_model.load_state_dict(state, strict=True)
    if role == "update_matched_dense":
        model = build_folded_dense_model(generic_model, "untied_generic_surface")
        audit = None
    else:
        assignments, audits = _assignments(state, data, exposure_counts)
        install_assignment(generic_model, assignments[role])
        model = generic_model
        audit = audits[role]
        if audit != plan["assignment_audits"][role]:
            raise RuntimeError("mechanism assignment audit differs")
    expected_sha = source[f"{role}_state_sha256"]
    if state_mapping_sha256(model.state_dict()) != expected_sha:
        raise RuntimeError("mechanism initial state differs")
    return model, audit


def _dense_effective_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: np.ndarray,
    *,
    step: int,
) -> None:
    model.train()
    _set_learning_rates(optimizer, step)
    optimizer.zero_grad(set_to_none=True)
    finite = torch.ones((), dtype=torch.bool, device="mps")
    for start in range(0, len(batch), TRAIN_MICROBATCH_SIZE):
        values = torch.tensor(
            batch[start : start + TRAIN_MICROBATCH_SIZE],
            dtype=torch.long,
            device="mps",
        )
        output = model(input_ids=values, labels=values, use_cache=False)
        loss = output.loss * (TRAIN_MICROBATCH_SIZE / len(batch))
        finite = finite & torch.isfinite(output.loss.detach())
        loss.backward()
        del values, output, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    input_weight = model.model.embed_tokens.weight
    output_weight = model.lm_head.weight
    input_before = input_weight[BASE_VOCABULARY_SIZE:].detach().clone()
    output_before = output_weight[BASE_VOCABULARY_SIZE:].detach().clone()
    optimizer.step()
    scale_new_row_update_(input_weight, input_before, INPUT_UPDATE_MULTIPLIER)
    scale_new_row_update_(output_weight, output_before, OUTPUT_UPDATE_MULTIPLIER)
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise RuntimeError("mechanism dense training loss became nonfinite")


def _load_nll(path: Path, *, final: bool) -> dict[str, np.ndarray]:
    expected = {"contiguous_nll_nats", "contiguous_raw_target_bytes"}
    if final:
        expected |= {"document_nll_nats", "document_raw_bytes"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RuntimeError("mechanism NLL key set differs")
        return {name: archive[name] for name in archive.files}


def _validate_worker(
    role: str, commit: str, plan: Mapping[str, Any]
) -> bool:
    report_path, checkpoints, nlls, deployed = worker_paths(role)
    paths = [report_path, deployed, *checkpoints.values(), *nlls.values()]
    existing = [path.exists() for path in paths]
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(f"partial mechanism worker requires forensics: {role}")
    report = read_json(report_path)
    unsigned = dict(report)
    receipt = unsigned.pop("worker_sha256", None)
    expected_keys = {
        "assignment_audit",
        "checkpoints",
        "complete",
        "deployed_checkpoint",
        "deployment_audit",
        "environment",
        "git_commit",
        "initial_state_sha256",
        "kind",
        "memory_diagnostics",
        "parameter_counts",
        "plan_artifact_sha256",
        "protocol_id",
        "role",
        "role_definition",
        "schema_version",
        "session_state",
        "training",
        "training_contract",
        "worker_sha256",
    }
    if (
        set(report) != expected_keys
        or canonical_sha256(unsigned) != receipt
        or report.get("schema_version") != 1
        or report.get("kind") != "foldable_multihash_mechanism_worker_v1"
        or report.get("protocol_id") != plan["protocol_id"]
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("role") != role
        or report.get("role_definition") != role_definition(role)
        or report.get("training_contract") != plan["training"]
        or report.get("environment") != current_environment()
        or report.get("initial_state_sha256")
        != plan["initialization"][f"{role}_state_sha256"]
        or report.get("assignment_audit")
        != plan["assignment_audits"].get(role)
        or set(report.get("checkpoints", {})) != {str(step) for step in PROBE_STEPS}
    ):
        raise RuntimeError("mechanism worker receipt differs")
    expected_deployed = expected_parameter_counts("untied_generic_surface")["deployed"]
    expected_parameter_values = {
        "deployed": expected_deployed,
        "training_only_residual": (
            0
            if role == "update_matched_dense"
            else expected_parameter_counts("untied_generic_surface")[
                "training_only_residual"
            ]
        ),
        "training_total": (
            expected_deployed
            if role == "update_matched_dense"
            else expected_parameter_counts("untied_generic_surface")["training_total"]
        ),
    }
    if report.get("parameter_counts") != expected_parameter_values:
        raise RuntimeError("mechanism worker parameter counts differ")
    training = report["training"]
    session = report["session_state"]
    if (
        set(training)
        != {
            "completed",
            "elapsed_seconds_including_evaluations",
            "finite_optimizer_steps",
            "optimizer_step_elapsed_seconds",
        }
        or training["completed"] is not True
        or training["finite_optimizer_steps"] != FINAL_PROBE_STEP
        or not math.isfinite(float(training["elapsed_seconds_including_evaluations"]))
        or float(training["elapsed_seconds_including_evaluations"]) <= 0.0
        or not math.isfinite(float(training["optimizer_step_elapsed_seconds"]))
        or float(training["optimizer_step_elapsed_seconds"]) <= 0.0
        or not isinstance(session, Mapping)
        or set(session) != {"start", "end"}
        or not timing_environment_eligible(session["start"])
        or not timing_environment_eligible(session["end"])
    ):
        raise RuntimeError("mechanism worker training receipt differs")
    calibration = plan["inventories"]["8192"]["calibration"]
    for step in PROBE_STEPS:
        row = report["checkpoints"][str(step)]
        checkpoint_path = checkpoints[step]
        nll_path = nlls[step]
        if (
            set(row)
            != {
                "arrays",
                "contiguous_bpb",
                "checkpoint_artifact_sha256",
                "checkpoint_path",
                "checkpoint_state_sha256",
                "document_bpb",
                "nll_artifact_sha256",
                "nll_path",
            }
            or row["checkpoint_path"] != str(checkpoint_path.relative_to(ROOT))
            or row["checkpoint_artifact_sha256"] != hash_file(checkpoint_path)
            or row["nll_path"] != str(nll_path.relative_to(ROOT))
            or row["nll_artifact_sha256"] != hash_file(nll_path)
        ):
            raise RuntimeError("mechanism worker artifact lineage differs")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        arrays = _load_nll(nll_path, final=step == FINAL_PROBE_STEP)
        if (
            not isinstance(state, Mapping)
            or state_mapping_sha256(state) != row["checkpoint_state_sha256"]
            or row["arrays"]
            != {name: _array_metadata(value) for name, value in arrays.items()}
            or arrays["contiguous_nll_nats"].dtype != np.float32
            or arrays["contiguous_nll_nats"].shape
            != (int(calibration["full_sequence_count"]),)
            or row["contiguous_bpb"]
            != bpb(
                arrays["contiguous_nll_nats"],
                arrays["contiguous_raw_target_bytes"],
            )
        ):
            raise RuntimeError("mechanism worker checkpoint semantics differ")
        if step == FINAL_PROBE_STEP:
            if (
                arrays["document_nll_nats"].dtype != np.float64
                or arrays["document_nll_nats"].shape
                != (int(plan["document_common"]["document_count"]),)
                or row["document_bpb"]
                != bpb(arrays["document_nll_nats"], arrays["document_raw_bytes"])
            ):
                raise RuntimeError("mechanism worker document semantics differ")
        elif row["document_bpb"] is not None:
            raise RuntimeError("mechanism intermediate document metric differs")
    deployed_state = torch.load(deployed, map_location="cpu", weights_only=True)
    deployed_row = report["deployed_checkpoint"]
    if (
        not isinstance(deployed_state, Mapping)
        or deployed_row
        != {
            "path": str(deployed.relative_to(ROOT)),
            "artifact_sha256": hash_file(deployed),
            "state_sha256": state_mapping_sha256(deployed_state),
        }
        or report["deployment_audit"].get("deployed_graph")
        != "ordinary_untied_dense_bpe_8192"
    ):
        raise RuntimeError("mechanism deployed checkpoint differs")
    return True


def _worker(role: str) -> None:
    commit, plan, parent_plan = _context()
    if _validate_worker(role, commit, plan):
        return
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
        if (
            array_sha256(order[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE])
            != plan["training"]["training_order_prefix_sha256"]
        ):
            raise RuntimeError("mechanism training order differs")
        exposure_counts = _scheduled_exposure_counts(train_sequences, order)
        raw_target_bytes = raw_target_bytes_by_sequence(
            calibration_sequences, data["token_bytes"]
        )
        if int(raw_target_bytes.sum()) != data[
            "calibration_inventory"
        ].predicted_target_raw_bytes:
            raise RuntimeError("mechanism calibration denominator differs")
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise RuntimeError("mechanism worker environment is ineligible")
        serialized: dict[int, tuple[bytes, bytes, str, dict[str, Any]]] = {}
        sampled_memory: list[dict[str, int]] = []
        with publication_mps_exclusive():
            model, role_assignment_audit = _build_initial_model(
                role, plan, data, exposure_counts
            )
            initial_state_sha = state_mapping_sha256(model.state_dict())
            if role == "update_matched_dense":
                parameter_counts = {
                    "deployed": model_parameter_count(model),
                    "training_only_residual": 0,
                    "training_total": model_parameter_count(model),
                }
                optimizer = dense_optimizer(model)
            else:
                parameter_counts = expected_parameter_counts(
                    "untied_generic_surface"
                )
                if model_parameter_count(model) != parameter_counts["training_total"]:
                    raise RuntimeError("mechanism residual parameter count differs")
                optimizer = residual_optimizer(model)
            model = model.to("mps")
            finite_steps = 0
            optimizer_step_elapsed = 0.0
            started = time.perf_counter()
            for checkpoint_step in PROBE_STEPS:
                while finite_steps < checkpoint_step:
                    selected = order[
                        finite_steps * EFFECTIVE_BATCH_SIZE : (finite_steps + 1)
                        * EFFECTIVE_BATCH_SIZE
                    ]
                    batch = np.asarray(train_sequences[selected], dtype=np.int64)
                    step_started = time.perf_counter()
                    if role == "update_matched_dense":
                        _dense_effective_step(
                            model, optimizer, batch, step=finite_steps
                        )
                    else:
                        residual_effective_step(
                            model, optimizer, batch, step=finite_steps
                        )
                    optimizer_step_elapsed += time.perf_counter() - step_started
                    finite_steps += 1
                model.eval()
                contiguous = _evaluate_contiguous(model, calibration_sequences)
                arrays: dict[str, np.ndarray] = {
                    "contiguous_nll_nats": contiguous,
                    "contiguous_raw_target_bytes": raw_target_bytes,
                }
                document_bpb = None
                if checkpoint_step == FINAL_PROBE_STEP:
                    document = _evaluate_documents(
                        model,
                        data["document_chunks"],
                        data["chunk_documents"],
                        len(data["document_raw_bytes"]),
                        EVALUATION_BATCH_SIZE,
                    )
                    arrays["document_nll_nats"] = document
                    arrays["document_raw_bytes"] = data["document_raw_bytes"]
                    document_bpb = bpb(document, data["document_raw_bytes"])
                if checkpoint_step == 0:
                    control = parent_plan["baseline_controls"]["untied"][
                        "step_zero_nll"
                    ]
                    control_path = ROOT / control["path"]
                    with np.load(control_path, allow_pickle=False) as archive:
                        if (
                            hash_file(control_path) != control["artifact_sha256"]
                            or set(archive.files) != {"nll_nats", "raw_target_bytes"}
                            or array_sha256(archive["nll_nats"])
                            != control["array_sha256"]
                            or not np.array_equal(
                                contiguous, archive["nll_nats"]
                            )
                            or not np.array_equal(
                                raw_target_bytes, archive["raw_target_bytes"]
                            )
                        ):
                            raise RuntimeError(
                                "mechanism step-zero NLL differs from dense base"
                            )
                checkpoint_payload, checkpoint_sha = _checkpoint_bytes(
                    model.state_dict()
                )
                nll_payload = _npz_bytes(arrays)
                serialized[checkpoint_step] = (
                    checkpoint_payload,
                    nll_payload,
                    checkpoint_sha,
                    {
                        "arrays": {
                            name: _array_metadata(value)
                            for name, value in arrays.items()
                        },
                        "contiguous_bpb": bpb(contiguous, raw_target_bytes),
                        "document_bpb": document_bpb,
                    },
                )
                sampled_memory.append(_memory_snapshot())
                model.train()
            final_arrays = _load_arrays_from_payload(
                serialized[FINAL_PROBE_STEP][1]
            )
            if role == "update_matched_dense":
                deployed_model = model.eval()
                deployment_audit = {
                    "deployed_graph": "ordinary_untied_dense_bpe_8192",
                    "fold_required": False,
                    "contiguous_nll_bitwise_equal": True,
                    "document_nll_bitwise_equal": True,
                }
            else:
                if model.foldable_residual.residuals_are_exact_zero():
                    raise RuntimeError("mechanism residual parameters did not train")
                evidence = fold_audit(model, "untied_generic_surface")
                deployed_model = build_folded_dense_model(
                    model, "untied_generic_surface"
                ).to("mps").eval()
                folded_contiguous = _evaluate_contiguous(
                    deployed_model, calibration_sequences
                )
                folded_document = _evaluate_documents(
                    deployed_model,
                    data["document_chunks"],
                    data["chunk_documents"],
                    len(data["document_raw_bytes"]),
                    EVALUATION_BATCH_SIZE,
                )
                if (
                    not np.array_equal(
                        folded_contiguous, final_arrays["contiguous_nll_nats"]
                    )
                    or not np.array_equal(
                        folded_document, final_arrays["document_nll_nats"]
                    )
                ):
                    raise RuntimeError("mechanism folded NLL differs")
                deployment_audit = {
                    **evidence,
                    "deployed_graph": "ordinary_untied_dense_bpe_8192",
                    "fold_required": True,
                    "contiguous_nll_bitwise_equal": True,
                    "document_nll_bitwise_equal": True,
                }
                del folded_contiguous, folded_document
            deployed_payload, deployed_state_sha = _checkpoint_bytes(
                deployed_model.state_dict()
            )
            if model_parameter_count(deployed_model) != parameter_counts["deployed"]:
                raise RuntimeError("mechanism deployed parameter count differs")
            elapsed = time.perf_counter() - started
            model.to("cpu")
            deployed_model.to("cpu")
            del optimizer, model, deployed_model, final_arrays
            gc.collect()
            torch.mps.empty_cache()
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError("mechanism worker environment changed")

        report_path, checkpoint_paths, nll_paths, deployed_path = worker_paths(role)
        checkpoint_rows: dict[str, Any] = {}
        for step in PROBE_STEPS:
            checkpoint_payload, nll_payload, state_sha, metrics = serialized[step]
            checkpoint_rows[str(step)] = {
                "checkpoint_path": str(checkpoint_paths[step].relative_to(ROOT)),
                "checkpoint_artifact_sha256": hashlib.sha256(
                    checkpoint_payload
                ).hexdigest(),
                "checkpoint_state_sha256": state_sha,
                "nll_path": str(nll_paths[step].relative_to(ROOT)),
                "nll_artifact_sha256": hashlib.sha256(nll_payload).hexdigest(),
                **metrics,
            }
        memory_diagnostics = {
            "interpretation": (
                "sampled post-step allocator diagnostics; not resettable native peaks"
            ),
            "maximum_current_allocated_bytes": max(
                row["current_allocated_bytes"] for row in sampled_memory
            ),
            "maximum_driver_allocated_bytes": max(
                row["driver_allocated_bytes"] for row in sampled_memory
            ),
            "maximum_process_rss_bytes": max(
                row["process_max_rss_bytes"] for row in sampled_memory
            ),
            "recommended_max_bytes": sampled_memory[-1]["recommended_max_bytes"],
            "sample_count": len(sampled_memory),
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "foldable_multihash_mechanism_worker_v1",
            "protocol_id": plan["protocol_id"],
            "complete": True,
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "role": role,
            "role_definition": role_definition(role),
            "assignment_audit": role_assignment_audit,
            "initial_state_sha256": initial_state_sha,
            "parameter_counts": parameter_counts,
            "training_contract": training_contract(),
            "training": {
                "completed": True,
                "elapsed_seconds_including_evaluations": elapsed,
                "finite_optimizer_steps": finite_steps,
                "optimizer_step_elapsed_seconds": optimizer_step_elapsed,
            },
            "memory_diagnostics": memory_diagnostics,
            "deployment_audit": deployment_audit,
            "checkpoints": checkpoint_rows,
            "deployed_checkpoint": {
                "path": str(deployed_path.relative_to(ROOT)),
                "artifact_sha256": hashlib.sha256(deployed_payload).hexdigest(),
                "state_sha256": deployed_state_sha,
            },
            "environment": current_environment(),
            "session_state": {"start": start_state, "end": end_state},
        }
        report["worker_sha256"] = canonical_sha256(report)
        for step in PROBE_STEPS:
            checkpoint_payload, nll_payload, _, _ = serialized[step]
            _publish(checkpoint_paths[step], checkpoint_payload)
            _publish(nll_paths[step], nll_payload)
        _publish(deployed_path, deployed_payload)
        _publish(report_path, json_bytes(report))
    finally:
        _cleanup_data(data)


def _parent() -> None:
    commit, plan, _ = _context()
    if REPORT_PATH.exists() or OUTPUT_PATH.exists():
        raise RuntimeError("mechanism downstream output already exists")
    active = ACTIVE_ROOT / "campaign.json"
    active_payload = json_bytes(
        {
            "git_commit": commit,
            "kind": "foldable_multihash_mechanism_active_v1",
            "plan_artifact_sha256": hash_file(PLAN_PATH),
        }
    )
    if active.exists():
        if active.read_bytes() != active_payload:
            raise RuntimeError("mechanism active campaign differs")
    else:
        _publish(active, active_payload)
    for role in NEW_ROLES:
        if _validate_worker(role, commit, plan):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", role],
            cwd=ROOT,
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}",
            },
        )
    workers: dict[str, Any] = {}
    for role in NEW_ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("mechanism worker did not complete")
        report_path = worker_paths(role)[0]
        workers[role] = {
            "path": str(report_path.relative_to(ROOT)),
            "sha256": hash_file(report_path),
        }
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during mechanism campaign")
    report = {
        "schema_version": 1,
        "kind": "foldable_multihash_mechanism_campaign_v1",
        "protocol_id": plan["protocol_id"],
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "workers": workers,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    active.unlink()
    print("status=foldable_multihash_mechanism_workers_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=NEW_ROLES)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        _parent()


if __name__ == "__main__":
    main()
