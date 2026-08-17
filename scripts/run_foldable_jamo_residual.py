#!/usr/bin/env python3
"""Run the sealed six-role foldable-Jamo residual development screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bpe_quality_feasibility_core import encode_stream_to_memmap
from bpe_quality_frontier_core import (
    array_sha256,
    bpb,
    calibration_document_pieces,
    encode_document_chunks,
    raw_target_bytes_by_sequence,
)
from compositional_head_preflight_protocol import load_tokenizers
from foldable_jamo_residual_core import (
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    RESIDUAL_ROLES,
    build_foldable_model,
    build_folded_dense_model,
    expected_parameter_counts,
    fold_audit,
    folded_dense_state,
    role_definition,
)
from foldable_jamo_residual_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_ROOT,
    FOLDED_CHECKPOINT_ROOT,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    SOURCE_PATH,
    WORKER_ROOT,
    baseline_control_identities,
    base_checkpoint_state,
    canonical_sha256,
    current_environment,
    hash_file,
    json_bytes,
    read_json,
    target_order,
    training_contract,
    validate_plan,
)
from run_compositional_quality import _evaluate_documents
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    BODY_LEARNING_RATE,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    HEAD_PEAK_LEARNING_RATE,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_SIZE,
    WEIGHT_DECAY,
    build_canonical_bpe_decomposition_table,
    probe_learning_rate,
    state_mapping_sha256,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_data import build_neural_stream


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _snapshot(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout": result.stdout,
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _snapshot(("pmset", "-g", "batt")),
        "settings": _snapshot(("pmset", "-g", "custom")),
        "thermal": _snapshot(("pmset", "-g", "therm")),
    }


def _memory_snapshot() -> dict[str, int]:
    torch.mps.synchronize()
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
    }


def _publish(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _checkpoint_bytes(state: Mapping[str, torch.Tensor]) -> tuple[bytes, str]:
    values = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state.items()
    }
    output = io.BytesIO()
    torch.save(values, output)
    return output.getvalue(), state_mapping_sha256(values)


def _context() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("foldable residual requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("foldable residual plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("foldable residual plan parent differs")
    return commit, plan


def _role_data(plan: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    train_stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="train",
        byte_limit=128_000_000,
        sequence_length=512,
    )
    calibration_stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=8_000_000,
        sequence_length=512,
    )
    train_inventory, train_memory, train_path = encode_stream_to_memmap(
        train_stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EFFECTIVE_BATCH_SIZE * 512,
    )
    calibration_inventory, calibration_memory, calibration_path = encode_stream_to_memmap(
        calibration_stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EVALUATION_BATCH_SIZE * 512,
    )
    inherited = plan["inventories"]["8192"]
    if (
        train_inventory.to_dict() != inherited["train"]
        or calibration_inventory.to_dict() != inherited["calibration"]
    ):
        raise RuntimeError("foldable residual token inventory differs")
    pieces, common = calibration_document_pieces(SOURCE_PATH)
    document_inventory, chunks, chunk_documents, document_raw_bytes = encode_document_chunks(
        pieces, tokenizer, token_bytes
    )
    if (
        common != plan["document_common"]
        or document_inventory.to_dict() != plan["document_inventory"]
    ):
        raise RuntimeError("foldable residual document inventory differs")
    return {
        "tokenizer": tokenizer,
        "token_bytes": token_bytes,
        "train_inventory": train_inventory,
        "train_memory": train_memory,
        "train_path": train_path,
        "calibration_inventory": calibration_inventory,
        "calibration_memory": calibration_memory,
        "calibration_path": calibration_path,
        "document_chunks": chunks,
        "chunk_documents": chunk_documents,
        "document_raw_bytes": document_raw_bytes,
    }


def _cleanup_data(data: Mapping[str, Any]) -> None:
    del data["train_memory"], data["calibration_memory"]
    for key in ("train_path", "calibration_path"):
        if os.path.exists(data[key]):
            os.unlink(data[key])


def _scheduled_exposure_counts(
    train_sequences: np.ndarray, order: np.ndarray
) -> np.ndarray:
    selected = order[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE]
    return np.bincount(
        np.asarray(train_sequences[selected], dtype=np.int64).reshape(-1),
        minlength=TARGET_VOCABULARY_SIZE,
    ).astype(np.int64, copy=False)


def _optimizer(model: Any) -> torch.optim.Optimizer:
    residual_ids = {id(parameter) for parameter in model.foldable_residual.parameters()}
    groups: dict[str, list[torch.nn.Parameter]] = {
        "body_decay": [],
        "body_no_decay": [],
        "head": [],
    }
    seen: set[int] = set()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameter.requires_grad_(True)
        if id(parameter) in residual_ids:
            groups["head"].append(parameter)
        elif parameter.ndim >= 2:
            groups["body_decay"].append(parameter)
        else:
            groups["body_no_decay"].append(parameter)
    if (
        len(groups["head"]) != (2 if model.foldable_residual.tied else 4)
        or sum(len(values) for values in groups.values()) != len(seen)
    ):
        raise RuntimeError("foldable residual optimizer grouping differs")
    return torch.optim.AdamW(
        [
            {
                "params": groups["body_decay"],
                "lr": BODY_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "schedule_kind": "body",
            },
            {
                "params": groups["body_no_decay"],
                "lr": BODY_LEARNING_RATE,
                "weight_decay": 0.0,
                "schedule_kind": "body",
            },
            {
                "params": groups["head"],
                "lr": HEAD_PEAK_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "schedule_kind": "head",
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _set_learning_rates(optimizer: torch.optim.Optimizer, step: int) -> None:
    for group in optimizer.param_groups:
        group["lr"] = (
            BODY_LEARNING_RATE
            if group["schedule_kind"] == "body"
            else probe_learning_rate(step, peak=HEAD_PEAK_LEARNING_RATE, minimum=3e-5)
        )


def _effective_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: np.ndarray,
    *,
    step: int,
) -> None:
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
    optimizer.step()
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise RuntimeError("foldable residual training loss became nonfinite")


def _evaluate_contiguous(model: Any, sequences: np.ndarray) -> np.ndarray:
    losses = np.empty(len(sequences), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(sequences), EVALUATION_BATCH_SIZE):
            stop = min(start + EVALUATION_BATCH_SIZE, len(sequences))
            values = torch.tensor(
                np.asarray(sequences[start:stop], dtype=np.int64),
                dtype=torch.long,
                device="mps",
            )
            logits = model(input_ids=values, use_cache=False).logits.float()
            token_nll = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2), values[:, 1:], reduction="none"
            )
            row = token_nll.sum(dim=1)
            if not torch.isfinite(row).all():
                raise RuntimeError("foldable residual contiguous NLL became nonfinite")
            losses[start:stop] = row.cpu().numpy().astype(np.float32, copy=False)
            del values, logits, token_nll, row
    torch.mps.synchronize()
    return losses


def _paths(role: str) -> tuple[Path, dict[int, Path], dict[int, Path], Path]:
    return (
        WORKER_ROOT / f"{role}.json",
        {step: CHECKPOINT_ROOT / f"{role}-step-{step:04d}.pt" for step in PROBE_STEPS},
        {step: NLL_ROOT / f"{role}-step-{step:04d}.npz" for step in PROBE_STEPS},
        FOLDED_CHECKPOINT_ROOT / f"{role}-step-{FINAL_PROBE_STEP:04d}.pt",
    )


def _array_metadata(values: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "sha256": array_sha256(values),
    }


def _load_nll(path: Path, *, final: bool) -> dict[str, np.ndarray]:
    expected = {"contiguous_nll_nats", "contiguous_raw_target_bytes"}
    if final:
        expected |= {"document_nll_nats", "document_raw_bytes"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise RuntimeError("foldable residual NLL key set differs")
        return {name: archive[name] for name in archive.files}


def _validate_worker(role: str, commit: str, plan: Mapping[str, Any]) -> bool:
    report_path, checkpoints, nlls, folded = _paths(role)
    paths = [report_path, folded, *checkpoints.values(), *nlls.values()]
    existing = [path.exists() for path in paths]
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(f"partial foldable residual worker requires forensics: {role}")
    report = read_json(report_path)
    unsigned = dict(report)
    receipt = unsigned.pop("worker_sha256", None)
    if (
        set(report)
        != {
            "assignment_audit",
            "checkpoints",
            "complete",
            "environment",
            "fold_audit",
            "folded_checkpoint",
            "git_commit",
            "initialization_identity",
            "kind",
            "memory_diagnostics",
            "parameter_counts",
            "plan_artifact_sha256",
            "protocol_id",
            "role",
            "schema_version",
            "session_state",
            "training",
            "training_contract",
            "worker_sha256",
        }
        or canonical_sha256(unsigned) != receipt
        or report.get("schema_version") != 1
        or report.get("kind") != "foldable_jamo_residual_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("role") != role
        or report.get("parameter_counts") != expected_parameter_counts(role)
        or report.get("training_contract") != plan["training"]
        or report.get("assignment_audit") != plan["assignment_audits"][role]
        or report.get("initialization_identity")
        != plan["initialization_identities"][role]
        or report.get("environment") != current_environment()
        or set(report.get("checkpoints", {})) != {str(step) for step in PROBE_STEPS}
    ):
        raise RuntimeError("completed foldable residual worker differs")
    training = report.get("training")
    session = report.get("session_state")
    if (
        not isinstance(training, Mapping)
        or set(training)
        != {
            "completed",
            "elapsed_seconds_including_evaluations",
            "finite_optimizer_steps",
            "optimizer_step_elapsed_seconds",
        }
        or training.get("completed") is not True
        or training.get("finite_optimizer_steps") != FINAL_PROBE_STEP
        or not math.isfinite(float(training.get("elapsed_seconds_including_evaluations", math.nan)))
        or float(training["elapsed_seconds_including_evaluations"]) <= 0
        or not math.isfinite(float(training.get("optimizer_step_elapsed_seconds", math.nan)))
        or float(training["optimizer_step_elapsed_seconds"]) <= 0
        or not isinstance(session, Mapping)
        or set(session) != {"start", "end"}
        or not timing_environment_eligible(session["start"])
        or not timing_environment_eligible(session["end"])
    ):
        raise RuntimeError("foldable residual training receipt differs")
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
            raise RuntimeError("foldable residual checkpoint lineage differs")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or state_mapping_sha256(state) != row["checkpoint_state_sha256"]:
            raise RuntimeError("foldable residual checkpoint state differs")
        arrays = _load_nll(nll_path, final=step == FINAL_PROBE_STEP)
        if set(row["arrays"]) != set(arrays):
            raise RuntimeError("foldable residual NLL metadata differs")
        for name, values in arrays.items():
            if row["arrays"][name] != _array_metadata(values):
                raise RuntimeError("foldable residual NLL array identity differs")
        if (
            arrays["contiguous_nll_nats"].dtype != np.float32
            or arrays["contiguous_nll_nats"].shape
            != (int(calibration["full_sequence_count"]),)
            or arrays["contiguous_raw_target_bytes"].shape
            != arrays["contiguous_nll_nats"].shape
            or row["contiguous_bpb"]
            != bpb(arrays["contiguous_nll_nats"], arrays["contiguous_raw_target_bytes"])
        ):
            raise RuntimeError("foldable residual contiguous semantics differ")
        if step == FINAL_PROBE_STEP:
            if (
                arrays["document_nll_nats"].dtype != np.float64
                or arrays["document_nll_nats"].shape
                != (int(plan["document_common"]["document_count"]),)
                or arrays["document_raw_bytes"].shape
                != arrays["document_nll_nats"].shape
                or row["document_bpb"]
                != bpb(arrays["document_nll_nats"], arrays["document_raw_bytes"])
            ):
                raise RuntimeError("foldable residual document semantics differ")
        elif row["document_bpb"] is not None:
            raise RuntimeError("foldable residual intermediate document metric differs")
    folded_row = report["folded_checkpoint"]
    folded_state = torch.load(folded, map_location="cpu", weights_only=True)
    if (
        set(folded_row)
        != {"artifact_sha256", "path", "state_sha256"}
        or folded_row["path"] != str(folded.relative_to(ROOT))
        or folded_row["artifact_sha256"] != hash_file(folded)
        or not isinstance(folded_state, Mapping)
        or folded_row["state_sha256"] != state_mapping_sha256(folded_state)
        or report.get("fold_audit", {}).get("folded_state_sha256")
        != folded_row["state_sha256"]
    ):
        raise RuntimeError("foldable residual deployed checkpoint differs")
    return True


def _worker(role: str) -> None:
    commit, plan = _context()
    if role not in RESIDUAL_ROLES or _validate_worker(role, commit, plan):
        return
    data = _role_data(plan)
    train_count = data["train_inventory"].full_sequence_count
    calibration_count = data["calibration_inventory"].full_sequence_count
    train_sequences = data["train_memory"][: train_count * 512].reshape(train_count, 512)
    calibration_sequences = data["calibration_memory"][: calibration_count * 512].reshape(
        calibration_count, 512
    )
    order = target_order(train_count)
    if array_sha256(order[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE]) != plan["training"][
        "training_order_prefix_sha256"
    ]:
        raise RuntimeError("foldable residual training order differs")
    exposure_counts = _scheduled_exposure_counts(train_sequences, order)
    raw_target_bytes = raw_target_bytes_by_sequence(
        calibration_sequences, data["token_bytes"]
    )
    if int(raw_target_bytes.sum()) != data["calibration_inventory"].predicted_target_raw_bytes:
        raise RuntimeError("foldable residual contiguous denominator differs")
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
        raise RuntimeError("foldable residual worker environment is ineligible")
    serialized: dict[int, tuple[bytes, bytes, str, dict[str, Any]]] = {}
    folded_payload: bytes | None = None
    folded_state_sha: str | None = None
    fold_evidence: dict[str, Any] | None = None
    sampled_memory: list[dict[str, int]] = []
    with publication_mps_exclusive():
        model, initializer_audit, assignment_audit = build_foldable_model(
            role,
            base_state=base_checkpoint_state(),
            base_tokenizer=base_tokenizer,
            base_pieces=base_pieces,
            target_pieces=data["token_bytes"],
            decompositions=decompositions,
            exposure_counts=exposure_counts,
        )
        expected_initial = plan["initialization_identities"][role]
        if (
            initializer_audit.to_dict() != expected_initial["base_initializer_audit"]
            or assignment_audit.to_dict() != plan["assignment_audits"][role]
            or state_mapping_sha256(model.state_dict())
            != expected_initial["training_state_sha256"]
            or state_mapping_sha256(folded_dense_state(model, role))
            != expected_initial["folded_dense_state_sha256"]
        ):
            raise RuntimeError("foldable residual initial identity differs")
        parameter_counts = expected_parameter_counts(role)
        if model_parameter_count(model) != parameter_counts["training_total"]:
            raise RuntimeError("foldable residual training parameter count differs")
        model = model.to("mps")
        optimizer = _optimizer(model)
        finite_steps = 0
        optimizer_step_elapsed = 0.0
        started = time.perf_counter()
        for checkpoint_step in PROBE_STEPS:
            while finite_steps < checkpoint_step:
                selected = order[
                    finite_steps * EFFECTIVE_BATCH_SIZE : (finite_steps + 1)
                    * EFFECTIVE_BATCH_SIZE
                ]
                step_started = time.perf_counter()
                _effective_step(
                    model,
                    optimizer,
                    np.asarray(train_sequences[selected], dtype=np.int64),
                    step=finite_steps,
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
                architecture = role_definition(role)["architecture"]
                control_path = ROOT / plan["baseline_controls"][architecture]["step_zero_nll"]["path"]
                with np.load(control_path, allow_pickle=False) as archive:
                    if (
                        not np.array_equal(contiguous, archive["nll_nats"])
                        or not np.array_equal(raw_target_bytes, archive["raw_target_bytes"])
                    ):
                        raise RuntimeError("foldable residual zero-state NLL differs from generic base")
            checkpoint_payload, checkpoint_sha = _checkpoint_bytes(model.state_dict())
            nll_payload = _npz_bytes(arrays)
            serialized[checkpoint_step] = (
                checkpoint_payload,
                nll_payload,
                checkpoint_sha,
                {
                    "arrays": {name: _array_metadata(value) for name, value in arrays.items()},
                    "contiguous_bpb": bpb(contiguous, raw_target_bytes),
                    "document_bpb": document_bpb,
                },
            )
            sampled_memory.append(_memory_snapshot())
            model.train()
        if model.foldable_residual.residuals_are_exact_zero():
            raise RuntimeError("foldable residual parameters did not train")
        fold_evidence = fold_audit(model, role)
        folded_model = build_folded_dense_model(model, role).to("mps").eval()
        folded_contiguous = _evaluate_contiguous(folded_model, calibration_sequences)
        folded_document = _evaluate_documents(
            folded_model,
            data["document_chunks"],
            data["chunk_documents"],
            len(data["document_raw_bytes"]),
            EVALUATION_BATCH_SIZE,
        )
        final_arrays = _load_arrays_from_payload(serialized[FINAL_PROBE_STEP][1])
        if (
            not np.array_equal(folded_contiguous, final_arrays["contiguous_nll_nats"])
            or not np.array_equal(folded_document, final_arrays["document_nll_nats"])
        ):
            raise RuntimeError("foldable residual materialized NLL differs")
        folded_payload, folded_state_sha = _checkpoint_bytes(folded_model.state_dict())
        if (
            fold_evidence["folded_state_sha256"] != folded_state_sha
            or model_parameter_count(folded_model) != parameter_counts["deployed"]
            or fold_evidence["old_input_rows_unchanged_by_residual"] is not True
            or fold_evidence["old_output_rows_unchanged_by_residual"] is not True
        ):
            raise RuntimeError("foldable residual fold audit differs")
        elapsed = time.perf_counter() - started
        model.to("cpu")
        folded_model.to("cpu")
        del optimizer, model, folded_model, final_arrays, folded_contiguous, folded_document
        gc.collect()
        torch.mps.empty_cache()
    end_state = _session_state()
    if not timing_environment_eligible(end_state):
        raise RuntimeError("foldable residual worker environment changed")
    if folded_payload is None or folded_state_sha is None or fold_evidence is None:
        raise AssertionError("foldable residual final evidence is absent")
    report_path, checkpoint_paths, nll_paths, folded_path = _paths(role)
    checkpoint_rows: dict[str, Any] = {}
    for step in PROBE_STEPS:
        checkpoint_payload, nll_payload, state_sha, metrics = serialized[step]
        checkpoint_rows[str(step)] = {
            "checkpoint_path": str(checkpoint_paths[step].relative_to(ROOT)),
            "checkpoint_artifact_sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "checkpoint_state_sha256": state_sha,
            "nll_path": str(nll_paths[step].relative_to(ROOT)),
            "nll_artifact_sha256": hashlib.sha256(nll_payload).hexdigest(),
            **metrics,
        }
    memory_diagnostics = {
        "interpretation": "sampled post-step allocator diagnostics; not resettable native peaks",
        "maximum_current_allocated_bytes": max(
            row["current_allocated_bytes"] for row in sampled_memory
        ),
        "maximum_driver_allocated_bytes": max(
            row["driver_allocated_bytes"] for row in sampled_memory
        ),
        "maximum_process_rss_bytes": max(row["process_max_rss_bytes"] for row in sampled_memory),
        "recommended_max_bytes": sampled_memory[-1]["recommended_max_bytes"],
        "sample_count": len(sampled_memory),
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "foldable_jamo_residual_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "role": role,
        "parameter_counts": expected_parameter_counts(role),
        "training_contract": training_contract(),
        "assignment_audit": plan["assignment_audits"][role],
        "initialization_identity": plan["initialization_identities"][role],
        "training": {
            "completed": True,
            "elapsed_seconds_including_evaluations": elapsed,
            "finite_optimizer_steps": finite_steps,
            "optimizer_step_elapsed_seconds": optimizer_step_elapsed,
        },
        "memory_diagnostics": memory_diagnostics,
        "fold_audit": {
            **fold_evidence,
            "contiguous_nll_bitwise_equal_before_after_fold": True,
            "document_nll_bitwise_equal_before_after_fold": True,
        },
        "checkpoints": checkpoint_rows,
        "folded_checkpoint": {
            "path": str(folded_path.relative_to(ROOT)),
            "artifact_sha256": hashlib.sha256(folded_payload).hexdigest(),
            "state_sha256": folded_state_sha,
        },
        "environment": current_environment(),
        "session_state": {"start": start_state, "end": end_state},
    }
    report["worker_sha256"] = canonical_sha256(report)
    for step in PROBE_STEPS:
        checkpoint_payload, nll_payload, _, _ = serialized[step]
        _publish(checkpoint_paths[step], checkpoint_payload)
        _publish(nll_paths[step], nll_payload)
    _publish(folded_path, folded_payload)
    _publish(report_path, json_bytes(report))
    _cleanup_data(data)


def _load_arrays_from_payload(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _parent() -> None:
    commit, plan = _context()
    if REPORT_PATH.exists() or OUTPUT_PATH.exists():
        raise RuntimeError("foldable residual downstream output already exists")
    active_payload = json_bytes(
        {
            "git_commit": commit,
            "kind": "foldable_jamo_residual_active_v1",
            "plan_artifact_sha256": hash_file(PLAN_PATH),
        }
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active_payload:
            raise RuntimeError("foldable residual active session differs")
    else:
        _publish(ACTIVE_PATH, active_payload)
    for role in RESIDUAL_ROLES:
        if _validate_worker(role, commit, plan):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", role],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    workers = {}
    for role in RESIDUAL_ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("foldable residual worker did not complete")
        path = _paths(role)[0]
        workers[role] = {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during foldable residual campaign")
    report = {
        "schema_version": 1,
        "kind": "foldable_jamo_residual_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "workers": workers,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print("status=foldable_jamo_residual_workers_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=RESIDUAL_ROLES)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        _parent()


if __name__ == "__main__":
    main()
