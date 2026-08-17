#!/usr/bin/env python3
"""Resource-check, train, and evaluate the sealed one-seed quality grid."""

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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from bpe_quality_feasibility_core import encode_stream_to_memmap, first_sequence_batch
from bpe_quality_frontier_core import (
    array_sha256,
    bpb,
    calibration_document_pieces,
    encode_document_chunks,
    raw_target_bytes_by_sequence,
)
from compositional_quality_core import (
    CALIBRATION_BYTES,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH,
    GRADIENT_CLIP,
    MODEL_SEED,
    QUALITY_ROLES,
    QUALITY_SPECS,
    RESOURCE_CAMPAIGN_HOUR_LIMIT,
    RESOURCE_MEASURED_EVALUATION_BATCHES,
    RESOURCE_MEASURED_STEPS,
    RESOURCE_MEMORY_FRACTION_LIMIT,
    RESOURCE_SAFETY_FACTOR,
    RESOURCE_WARMUP_EVALUATION_BATCHES,
    RESOURCE_WARMUP_STEPS,
    SEQUENCE_LENGTH,
    TRAIN_BYTES,
    TRAIN_MICROBATCH,
    WEIGHT_DECAY,
    build_quality_model,
    cosine_learning_rate,
    deterministic_order,
    state_subset_sha256,
    training_contract,
)
from compositional_quality_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    CHECKPOINT_ROOT,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    RESOURCE_REPORT_PATH,
    RESOURCE_ROOT,
    ROOT,
    SOURCE_PATH,
    WORKER_ROOT,
    canonical_sha256,
    current_environment,
    hash_file,
    json_bytes,
    load_tokenizers,
    read_json,
    validate_plan,
)
from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_data import build_neural_stream
from scalar_runtime_core import model_parameter_count


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


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _checkpoint_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    output = io.BytesIO()
    torch.save(dict(state), output)
    return output.getvalue()


def _context() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compositional quality requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("compositional quality plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("compositional quality plan parent differs")
    return commit, plan


def _optimizer(model: Any) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _mps_memory() -> dict[str, int]:
    torch.mps.synchronize()
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _role_data(role: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    spec = QUALITY_SPECS[role]
    tokenizer, token_bytes = load_tokenizers()[spec.vocabulary_size]
    train_stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="train",
        byte_limit=TRAIN_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    calibration_stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    train_inventory, train_memory, train_path = encode_stream_to_memmap(
        train_stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH,
    )
    calibration_inventory, calibration_memory, calibration_path = encode_stream_to_memmap(
        calibration_stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EVALUATION_BATCH[role] * SEQUENCE_LENGTH,
    )
    inherited = plan["inventories"][str(spec.vocabulary_size)]
    if (
        train_inventory.to_dict() != inherited["train"]
        or calibration_inventory.to_dict() != inherited["calibration"]
    ):
        raise RuntimeError("compositional quality token inventory differs")
    return {
        "tokenizer": tokenizer,
        "token_bytes": token_bytes,
        "train_inventory": train_inventory,
        "train_memory": train_memory,
        "train_path": train_path,
        "calibration_inventory": calibration_inventory,
        "calibration_memory": calibration_memory,
        "calibration_path": calibration_path,
    }


def _cleanup_data(data: Mapping[str, Any]) -> None:
    del data["train_memory"], data["calibration_memory"]
    for key in ("train_path", "calibration_path"):
        if os.path.exists(data[key]):
            os.unlink(data[key])


def _build_checked_model(role: str, token_bytes, plan: Mapping[str, Any]):
    spec = QUALITY_SPECS[role]
    model = build_quality_model(
        role,
        token_bytes=token_bytes if "code" in spec.head_kind else None,
        seed=MODEL_SEED,
    )
    if (
        model_parameter_count(model) != spec.expected_parameters
        or state_subset_sha256(model, transformer_body_only=False)
        != plan["initial_state_sha256"][role]
        or state_subset_sha256(model, transformer_body_only=True)
        != plan["transformer_body_initial_state_sha256"]
    ):
        raise RuntimeError("compositional quality initial model differs")
    return model


def _effective_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: np.ndarray,
    *,
    microbatch: int,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    finite = torch.ones((), dtype=torch.bool, device="mps")
    for start in range(0, len(batch), microbatch):
        selected = batch[start : start + microbatch]
        values = torch.tensor(selected, dtype=torch.long, device="mps")
        output = model(input_ids=values, labels=values, use_cache=False)
        loss = output.loss * (len(selected) / len(batch))
        finite = finite & torch.isfinite(output.loss.detach())
        loss.backward()
        del selected, values, output, loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise RuntimeError("compositional quality training loss became nonfinite")


def _evaluate_loss_only(model: Any, batch: np.ndarray) -> None:
    with torch.inference_mode():
        values = torch.tensor(batch, dtype=torch.long, device="mps")
        output = model(input_ids=values, labels=values, use_cache=False)
        finite = torch.isfinite(output.loss.detach())
        del values, output
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise RuntimeError("compositional quality evaluation loss became nonfinite")


def _resource_worker_path(role: str) -> Path:
    return RESOURCE_ROOT / f"{role}.json"


def _validate_resource_worker(role: str, commit: str, plan: Mapping[str, Any]) -> bool:
    path = _resource_worker_path(role)
    if not path.exists():
        return False
    row = read_json(path)
    unsigned = dict(row)
    receipt = unsigned.pop("worker_sha256", None)
    if (
        canonical_sha256(unsigned) != receipt
        or row.get("schema_version") != 1
        or row.get("kind") != "compositional_quality_resource_worker_v1"
        or row.get("protocol_id") != PROTOCOL_ID
        or row.get("complete") is not True
        or row.get("git_commit") != commit
        or row.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or row.get("role") != role
        or row.get("parameter_count") != QUALITY_SPECS[role].expected_parameters
        or row.get("contract") != plan["training"][role]
        or row.get("loss_values_recorded") is not False
    ):
        raise RuntimeError(f"compositional quality resource worker differs: {role}")
    for key, length in (
        ("train_effective_step_seconds", RESOURCE_MEASURED_STEPS),
        ("evaluation_batch_seconds", RESOURCE_MEASURED_EVALUATION_BATCHES),
    ):
        values = row.get(key)
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(not math.isfinite(float(value)) or float(value) <= 0 for value in values)
        ):
            raise RuntimeError("compositional quality resource timing differs")
    if row.get("environment") != plan["environment"] or not all(
        timing_environment_eligible(row["session_state"][key])
        for key in ("start", "end")
    ):
        raise RuntimeError("compositional quality resource environment differs")
    memory = row.get("memory")
    if not isinstance(memory, Mapping) or memory.get("resettable_peak_supported") is not False:
        raise RuntimeError("compositional quality resource memory schema differs")
    expected_memory = {
        "current_allocated_bytes",
        "driver_allocated_bytes",
        "recommended_max_bytes",
        "process_max_rss_bytes",
    }
    for section in ("baseline", "maximum_sampled", "released"):
        values = memory.get(section)
        if (
            not isinstance(values, Mapping)
            or set(values) != expected_memory
            or any(not isinstance(value, int) or value < 0 for value in values.values())
        ):
            raise RuntimeError("compositional quality resource memory values differ")
    if memory["maximum_sampled"]["recommended_max_bytes"] <= 0:
        raise RuntimeError("compositional quality recommended memory differs")
    return True


def _resource_worker(role: str) -> None:
    commit, plan = _context()
    if role not in QUALITY_ROLES or _resource_worker_path(role).exists():
        raise RuntimeError("compositional quality resource worker namespace differs")
    data = _role_data(role, plan)
    train_batch = first_sequence_batch(data["train_memory"], EFFECTIVE_BATCH_SIZE)
    evaluation_batch = first_sequence_batch(
        data["calibration_memory"], EVALUATION_BATCH[role]
    )
    start_state = _session_state()
    if not timing_environment_eligible(start_state):
        raise RuntimeError("compositional quality resource environment is ineligible")
    with publication_mps_exclusive():
        torch.mps.empty_cache()
        baseline = _mps_memory()
        model = _build_checked_model(role, data["token_bytes"], plan).to("mps").train()
        optimizer = _optimizer(model)
        for _ in range(RESOURCE_WARMUP_STEPS):
            _effective_step(
                model,
                optimizer,
                train_batch,
                microbatch=TRAIN_MICROBATCH[role],
                learning_rate=3e-4,
            )
        samples = [_mps_memory()]
        train_seconds = []
        for _ in range(RESOURCE_MEASURED_STEPS):
            started = time.perf_counter_ns()
            _effective_step(
                model,
                optimizer,
                train_batch,
                microbatch=TRAIN_MICROBATCH[role],
                learning_rate=3e-4,
            )
            train_seconds.append((time.perf_counter_ns() - started) / 1e9)
            samples.append(_mps_memory())
        model.eval()
        for _ in range(RESOURCE_WARMUP_EVALUATION_BATCHES):
            _evaluate_loss_only(model, evaluation_batch)
        evaluation_seconds = []
        for _ in range(RESOURCE_MEASURED_EVALUATION_BATCHES):
            started = time.perf_counter_ns()
            _evaluate_loss_only(model, evaluation_batch)
            evaluation_seconds.append((time.perf_counter_ns() - started) / 1e9)
            samples.append(_mps_memory())
        parameter_count = model_parameter_count(model)
        del optimizer
        model.to("cpu")
        del model
        gc.collect()
        torch.mps.empty_cache()
        released = _mps_memory()
    end_state = _session_state()
    if not timing_environment_eligible(end_state):
        raise RuntimeError("compositional quality resource environment changed")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "compositional_quality_resource_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "role": role,
        "parameter_count": parameter_count,
        "contract": plan["training"][role],
        "train_effective_step_seconds": train_seconds,
        "evaluation_batch_seconds": evaluation_seconds,
        "memory": {
            "baseline": baseline,
            "maximum_sampled": {
                key: max(row[key] for row in samples) for key in samples[0]
            },
            "released": released,
            "resettable_peak_supported": False,
        },
        "environment": current_environment(),
        "session_state": {"start": start_state, "end": end_state},
        "loss_values_recorded": False,
    }
    report["worker_sha256"] = canonical_sha256(report)
    _cleanup_data(data)
    _publish(_resource_worker_path(role), json_bytes(report))


def _resource_projection(commit: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    workers = {}
    total_seconds = 0.0
    memory_pass = True
    for role in QUALITY_ROLES:
        if not _validate_resource_worker(role, commit, plan):
            raise RuntimeError("compositional quality resource worker is incomplete")
        path = _resource_worker_path(role)
        row = read_json(path)
        train_median = float(np.median(row["train_effective_step_seconds"]))
        eval_median = float(np.median(row["evaluation_batch_seconds"]))
        steps = int(plan["training"][role]["total_optimizer_steps"])
        spec = QUALITY_SPECS[role]
        evaluation_sequences = plan["inventories"][str(spec.vocabulary_size)][
            "calibration"
        ]["full_sequence_count"]
        evaluation_batches = math.ceil(evaluation_sequences / EVALUATION_BATCH[role])
        role_seconds = train_median * steps + eval_median * evaluation_batches
        total_seconds += role_seconds
        memory = row["memory"]["maximum_sampled"]
        driver_fraction = memory["driver_allocated_bytes"] / memory["recommended_max_bytes"]
        rss_fraction = memory["process_max_rss_bytes"] / plan["environment"]["hardware"][
            "memory_bytes"
        ]
        memory_pass = memory_pass and max(driver_fraction, rss_fraction) <= RESOURCE_MEMORY_FRACTION_LIMIT
        workers[role] = {
            "worker_artifact_sha256": hash_file(path),
            "train_effective_step_median_seconds": train_median,
            "evaluation_batch_median_seconds": eval_median,
            "optimizer_steps": steps,
            "evaluation_batches": evaluation_batches,
            "projected_core_hours": role_seconds / 3600,
            "driver_memory_fraction": driver_fraction,
            "process_rss_fraction": rss_fraction,
        }
    projected = total_seconds / 3600
    projected_with_safety = projected * RESOURCE_SAFETY_FACTOR
    passes = memory_pass and projected_with_safety <= RESOURCE_CAMPAIGN_HOUR_LIMIT
    return {
        "workers": workers,
        "projected_core_hours": projected,
        "projected_hours_after_safety_factor": projected_with_safety,
        "memory_pass": memory_pass,
        "passes": passes,
        "loss_or_quality_used": False,
    }


def _ensure_resource_report(commit: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    if RESOURCE_REPORT_PATH.exists():
        report = read_json(RESOURCE_REPORT_PATH)
        unsigned = dict(report)
        receipt = unsigned.pop("report_sha256", None)
        if (
            canonical_sha256(unsigned) != receipt
            or report.get("schema_version") != 1
            or report.get("kind") != "compositional_quality_resource_report_v1"
            or report.get("protocol_id") != PROTOCOL_ID
            or report.get("git_commit") != commit
            or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
            or report.get("complete") is not True
        ):
            raise RuntimeError("compositional quality resource report differs")
        expected_projection = _resource_projection(commit, plan)
        if report.get("projection") != expected_projection:
            raise RuntimeError("compositional quality resource projection differs")
        return report
    RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    for role in QUALITY_ROLES:
        if _validate_resource_worker(role, commit, plan):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--resource-worker", role],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    projection = _resource_projection(commit, plan)
    report = {
        "schema_version": 1,
        "kind": "compositional_quality_resource_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "projection": projection,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(RESOURCE_REPORT_PATH, json_bytes(report))
    return report


def _evaluate_contiguous(model: Any, sequences: np.ndarray, batch_size: int) -> np.ndarray:
    losses = np.empty(len(sequences), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            stop = min(start + batch_size, len(sequences))
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
                raise RuntimeError("compositional quality contiguous NLL is nonfinite")
            losses[start:stop] = row.cpu().numpy().astype(np.float32, copy=False)
            del values, logits, token_nll, row
    torch.mps.synchronize()
    return losses


def _evaluate_documents(
    model: Any,
    chunks: tuple[np.ndarray, ...],
    chunk_documents: np.ndarray,
    document_count: int,
    batch_size: int,
) -> np.ndarray:
    output = np.zeros(document_count, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            selected = chunks[start : start + batch_size]
            maximum = max(len(chunk) for chunk in selected)
            inputs = np.zeros((len(selected), maximum), dtype=np.int64)
            attention = np.zeros((len(selected), maximum), dtype=np.int64)
            for row_index, chunk in enumerate(selected):
                inputs[row_index, : len(chunk)] = chunk
                attention[row_index, : len(chunk)] = 1
            values = torch.tensor(inputs, dtype=torch.long, device="mps")
            mask = torch.tensor(attention, dtype=torch.long, device="mps")
            logits = model(input_ids=values, attention_mask=mask, use_cache=False).logits.float()
            token_nll = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2), values[:, 1:], reduction="none"
            )
            row = (token_nll * mask[:, 1:]).sum(dim=1)
            if not torch.isfinite(row).all():
                raise RuntimeError("compositional quality document NLL is nonfinite")
            np.add.at(
                output,
                chunk_documents[start : start + len(selected)],
                row.cpu().numpy().astype(np.float64, copy=False),
            )
            del values, mask, logits, token_nll, row
    torch.mps.synchronize()
    return output


def _worker_paths(role: str) -> tuple[Path, Path, Path]:
    return (
        WORKER_ROOT / f"{role}.json",
        CHECKPOINT_ROOT / f"{role}.pt",
        NLL_ROOT / f"{role}.npz",
    )


def _validate_complete_worker(role: str, commit: str, plan: Mapping[str, Any]) -> bool:
    report_path, checkpoint_path, nll_path = _worker_paths(role)
    existing = tuple(path.exists() for path in (report_path, checkpoint_path, nll_path))
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(f"partial compositional quality worker requires forensics: {role}")
    report = read_json(report_path)
    unsigned = dict(report)
    receipt = unsigned.pop("worker_sha256", None)
    if (
        canonical_sha256(unsigned) != receipt
        or report.get("schema_version") != 1
        or report.get("kind") != "compositional_quality_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("resource_report_artifact_sha256")
        != hash_file(RESOURCE_REPORT_PATH)
        or report.get("role") != role
        or report.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
        or report.get("nll_artifact_sha256") != hash_file(nll_path)
        or report.get("training_contract") != plan["training"][role]
    ):
        raise RuntimeError(f"completed compositional quality worker differs: {role}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = QUALITY_SPECS[role]
    table = load_tokenizers()[spec.vocabulary_size][1]
    model = build_quality_model(
        role,
        token_bytes=table if "code" in spec.head_kind else None,
        seed=MODEL_SEED,
    )
    model.load_state_dict(state, strict=True)
    if state_subset_sha256(model, transformer_body_only=False) != report["trained_state_sha256"]:
        raise RuntimeError("compositional quality checkpoint state differs")
    with np.load(nll_path, allow_pickle=False) as archive:
        if set(archive.files) != set(report["arrays"]):
            raise RuntimeError("compositional quality NLL array set differs")
        for name in archive.files:
            values = archive[name]
            if report["arrays"][name] != {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }:
                raise RuntimeError("compositional quality NLL array differs")
    return True


def _train_worker(role: str) -> None:
    commit, plan = _context()
    if role not in QUALITY_ROLES or _validate_complete_worker(role, commit, plan):
        return
    if not RESOURCE_REPORT_PATH.exists() or _ensure_resource_report(commit, plan)[
        "projection"
    ]["passes"] is not True:
        raise RuntimeError("compositional quality training lacks resource authorization")
    data = _role_data(role, plan)
    train_inventory = data["train_inventory"]
    calibration_inventory = data["calibration_inventory"]
    train_sequences = data["train_memory"][
        : train_inventory.full_sequence_count * SEQUENCE_LENGTH
    ].reshape(train_inventory.full_sequence_count, SEQUENCE_LENGTH)
    calibration_sequences = data["calibration_memory"][
        : calibration_inventory.full_sequence_count * SEQUENCE_LENGTH
    ].reshape(calibration_inventory.full_sequence_count, SEQUENCE_LENGTH)
    order = deterministic_order(train_inventory.full_sequence_count)
    contract = training_contract(role, train_inventory.full_sequence_count)
    contract["training_order_sha256"] = array_sha256(order)
    if contract != plan["training"][role]:
        raise RuntimeError("compositional quality training contract differs")
    raw_target_bytes = raw_target_bytes_by_sequence(
        calibration_sequences, data["token_bytes"]
    )
    if int(raw_target_bytes.sum()) != calibration_inventory.predicted_target_raw_bytes:
        raise RuntimeError("compositional quality raw-byte denominator differs")
    pieces, common = calibration_document_pieces(SOURCE_PATH)
    document_inventory, chunks, chunk_documents, document_raw_bytes = encode_document_chunks(
        pieces, data["tokenizer"], data["token_bytes"]
    )
    if (
        common != {key: plan["document_common"][key] for key in common}
        or document_inventory.to_dict()
        != plan["inventories"][str(QUALITY_SPECS[role].vocabulary_size)]["documents"]
    ):
        raise RuntimeError("compositional quality document inventory differs")
    start_state = _session_state()
    if not timing_environment_eligible(start_state):
        raise RuntimeError("compositional quality worker environment is ineligible")
    with publication_mps_exclusive():
        model = _build_checked_model(role, data["token_bytes"], plan).to("mps").train()
        optimizer = _optimizer(model)
        finite_steps = 0
        torch.mps.synchronize()
        started = time.perf_counter()
        total_steps = int(contract["total_optimizer_steps"])
        warmup = int(contract["warmup_steps"])
        for step in range(total_steps):
            indices = order[step * EFFECTIVE_BATCH_SIZE : (step + 1) * EFFECTIVE_BATCH_SIZE]
            batch = np.asarray(train_sequences[indices], dtype=np.int64)
            _effective_step(
                model,
                optimizer,
                batch,
                microbatch=TRAIN_MICROBATCH[role],
                learning_rate=cosine_learning_rate(step, total_steps, warmup),
            )
            finite_steps += 1
        elapsed = time.perf_counter() - started
        model.eval()
        contiguous_nll = _evaluate_contiguous(
            model, calibration_sequences, EVALUATION_BATCH[role]
        )
        document_nll = _evaluate_documents(
            model,
            chunks,
            chunk_documents,
            len(document_raw_bytes),
            EVALUATION_BATCH[role],
        )
        model.to("cpu")
        trained_state = state_subset_sha256(model, transformer_body_only=False)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        del optimizer, model
        gc.collect()
        torch.mps.empty_cache()
    end_state = _session_state()
    if not timing_environment_eligible(end_state):
        raise RuntimeError("compositional quality worker environment changed")
    checkpoint_payload = _checkpoint_bytes(state)
    arrays = {
        "contiguous_nll_nats": contiguous_nll,
        "contiguous_raw_target_bytes": raw_target_bytes,
        "document_nll_nats": document_nll,
        "document_raw_bytes": document_raw_bytes,
    }
    nll_payload = _npz_bytes(arrays)
    report_path, checkpoint_path, nll_path = _worker_paths(role)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "compositional_quality_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "resource_report_artifact_sha256": hash_file(RESOURCE_REPORT_PATH),
        "role": role,
        "parameter_count": QUALITY_SPECS[role].expected_parameters,
        "training_contract": contract,
        "initial_state_sha256": plan["initial_state_sha256"][role],
        "trained_state_sha256": trained_state,
        "training": {
            "completed": True,
            "elapsed_seconds": elapsed,
            "finite_optimizer_steps": finite_steps,
            "sequence_examples": len(order),
            "total_optimizer_steps": total_steps,
        },
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_artifact_sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "nll_path": str(nll_path.relative_to(ROOT)),
        "nll_artifact_sha256": hashlib.sha256(nll_payload).hexdigest(),
        "arrays": {
            name: {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }
            for name, values in arrays.items()
        },
        "metrics": {
            "contiguous_bpb": bpb(contiguous_nll, raw_target_bytes),
            "document_bpb": bpb(document_nll, document_raw_bytes),
        },
        "environment": current_environment(),
        "session_state": {"start": start_state, "end": end_state},
    }
    report["worker_sha256"] = canonical_sha256(report)
    _publish(checkpoint_path, checkpoint_payload)
    _publish(nll_path, nll_payload)
    _publish(report_path, json_bytes(report))
    _cleanup_data(data)


def _parent() -> None:
    commit, plan = _context()
    if REPORT_PATH.exists() or OUTPUT_PATH.exists():
        raise RuntimeError("compositional quality downstream output already exists")
    active_payload = json_bytes(
        {
            "git_commit": commit,
            "kind": "compositional_quality_active_v1",
            "plan_artifact_sha256": hash_file(PLAN_PATH),
        }
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active_payload:
            raise RuntimeError("compositional quality active session differs")
    else:
        _publish(ACTIVE_PATH, active_payload)
    resource_report = _ensure_resource_report(commit, plan)
    if resource_report["projection"]["passes"] is not True:
        ACTIVE_PATH.unlink()
        print("status=compositional_quality_resource_gate_failed")
        return
    for role in QUALITY_ROLES:
        if _validate_complete_worker(role, commit, plan):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--train-worker", role],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    workers = {}
    for role in QUALITY_ROLES:
        if not _validate_complete_worker(role, commit, plan):
            raise RuntimeError("compositional quality worker did not complete")
        path = _worker_paths(role)[0]
        workers[role] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        }
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during compositional quality campaign")
    report = {
        "schema_version": 1,
        "kind": "compositional_quality_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "resource_report_artifact_sha256": hash_file(RESOURCE_REPORT_PATH),
        "workers": workers,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print(f"wrote={REPORT_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-worker", choices=QUALITY_ROLES)
    parser.add_argument("--train-worker", choices=QUALITY_ROLES)
    args = parser.parse_args()
    if args.resource_worker and args.train_worker:
        raise RuntimeError("compositional quality worker modes are exclusive")
    if args.resource_worker:
        _resource_worker(args.resource_worker)
    elif args.train_worker:
        _train_worker(args.train_worker)
    else:
        _parent()


if __name__ == "__main__":
    main()
