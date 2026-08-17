#!/usr/bin/env python3
"""Train and evaluate the sealed six-role one-seed BPE quality frontier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from bpe_quality_feasibility_core import (
    CALIBRATION_BYTES,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_BY_VOCABULARY,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
    TRAIN_BYTES,
    encode_stream_to_memmap,
)
from bpe_quality_feasibility_protocol import PLAN_PATH as FEASIBILITY_PLAN_PATH
from bpe_quality_frontier_core import (
    GRADIENT_CLIP,
    WEIGHT_DECAY,
    array_sha256,
    bpb,
    calibration_document_pieces,
    cosine_learning_rate,
    deterministic_order,
    encode_document_chunks,
    raw_target_bytes_by_sequence,
    role_training_contract,
)
from bpe_quality_frontier_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_ROOT,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    SOURCE_PATH,
    WORKER_ROOT,
    canonical_sha256,
    current_frontier_environment,
    hash_file,
    json_bytes,
    read_json,
    validate_plan,
)
from scalar_runtime_core import model_parameter_count
from token_frontier_core import FRONTIER_SPECS, build_frontier_model, parse_role
from token_frontier_protocol import load_tokenizers

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_data import build_neural_stream


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _command_snapshot(args: tuple[str, ...]) -> dict[str, Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout": result.stdout,
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _command_snapshot(("pmset", "-g", "batt")),
        "settings": _command_snapshot(("pmset", "-g", "custom")),
        "thermal": _command_snapshot(("pmset", "-g", "therm")),
    }


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _repository_context() -> tuple[str, dict[str, Any], dict[str, Any]]:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("BPE quality frontier requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    last_change = _command(
        "git", "log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))
    )
    if last_change != commit:
        raise ValueError("BPE quality frontier plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if (
        _command("git", "rev-parse", "HEAD^")
        != plan["dependencies"]["git_commit_before_plan"]
    ):
        raise ValueError("BPE quality frontier plan parent differs")
    return commit, plan, read_json(FEASIBILITY_PLAN_PATH)


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


def _train(
    model: Any,
    sequences: np.ndarray,
    order: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    optimizer = _optimizer(model)
    microbatch = int(contract["train_microbatch_size"])
    total_steps = int(contract["total_optimizer_steps"])
    warmup = int(contract["warmup_steps"])
    finite_steps = 0
    torch.mps.synchronize()
    started = time.perf_counter()
    for step in range(total_steps):
        indices = order[step * EFFECTIVE_BATCH_SIZE : (step + 1) * EFFECTIVE_BATCH_SIZE]
        batch = np.asarray(sequences[indices], dtype=np.int64)
        if not 1 <= len(batch) <= EFFECTIVE_BATCH_SIZE:
            raise AssertionError("BPE quality training batch is empty")
        learning_rate = cosine_learning_rate(step, total_steps, warmup)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        finite_losses = torch.ones((), dtype=torch.bool, device="mps")
        for start in range(0, len(batch), microbatch):
            values_np = batch[start : start + microbatch]
            values = torch.tensor(values_np, dtype=torch.long, device="mps")
            output = model(input_ids=values, labels=values, use_cache=False)
            weight = len(values_np) / len(batch)
            loss = output.loss * weight
            finite_losses = finite_losses & torch.isfinite(output.loss.detach())
            loss.backward()
            del output, loss, values
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        torch.mps.synchronize()
        if not bool(finite_losses.item()):
            raise ValueError("BPE quality training loss became nonfinite")
        finite_steps += 1
    elapsed = time.perf_counter() - started
    del optimizer
    if finite_steps != total_steps:
        raise AssertionError("BPE quality training did not finish all steps")
    return {
        "completed": True,
        "elapsed_seconds": elapsed,
        "finite_optimizer_steps": finite_steps,
        "sequence_examples": len(order),
        "total_optimizer_steps": total_steps,
    }


def _evaluate_contiguous(
    model: Any,
    sequences: np.ndarray,
    batch_size: int,
) -> np.ndarray:
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
                logits[:, :-1, :].transpose(1, 2),
                values[:, 1:],
                reduction="none",
            )
            row = token_nll.sum(dim=1)
            if not torch.isfinite(row).all():
                raise ValueError("BPE quality contiguous NLL became nonfinite")
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
            logits = model(
                input_ids=values, attention_mask=mask, use_cache=False
            ).logits.float()
            token_nll = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                values[:, 1:],
                reduction="none",
            )
            row = (token_nll * mask[:, 1:]).sum(dim=1)
            if not torch.isfinite(row).all():
                raise ValueError("BPE quality document NLL became nonfinite")
            np.add.at(
                output,
                chunk_documents[start : start + len(selected)],
                row.cpu().numpy().astype(np.float64, copy=False),
            )
            del values, mask, logits, token_nll, row
    torch.mps.synchronize()
    return output


def _checkpoint_bytes(state: dict[str, torch.Tensor]) -> bytes:
    output = io.BytesIO()
    torch.save(state, output)
    return output.getvalue()


def _validate_complete_worker(role: str, commit: str) -> bool:
    report_path = WORKER_ROOT / f"{role}.json"
    checkpoint_path = CHECKPOINT_ROOT / f"{role}.pt"
    nll_path = NLL_ROOT / f"{role}.npz"
    existing = tuple(path.exists() for path in (report_path, checkpoint_path, nll_path))
    if not any(existing):
        return False
    if not all(existing):
        raise ValueError(
            f"partial BPE quality worker evidence requires forensics: {role}"
        )
    report = read_json(report_path)
    unsigned = dict(report)
    receipt_hash = unsigned.pop("worker_sha256", None)
    if (
        canonical_sha256(unsigned) != receipt_hash
        or report.get("role") != role
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
        or report.get("nll_artifact_sha256") != hash_file(nll_path)
        or report.get("complete") is not True
    ):
        raise ValueError(f"completed BPE quality worker differs: {role}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_frontier_model(role, seed=report["training_contract"]["model_seed"])
    model.load_state_dict(state, strict=True)
    if state_sha256(model) != report["trained_state_sha256"]:
        raise ValueError("completed BPE quality checkpoint state differs")
    del model, state
    with np.load(nll_path, allow_pickle=False) as archive:
        if set(archive.files) != set(report["arrays"]):
            raise ValueError("completed BPE quality NLL keys differ")
        for name in archive.files:
            values = archive[name]
            descriptor = report["arrays"][name]
            if descriptor != {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }:
                raise ValueError("completed BPE quality NLL array differs")
    return True


def _worker(role: str) -> None:
    commit, plan, feasibility_plan = _repository_context()
    if role not in QUALITY_ROLES:
        raise ValueError("BPE quality frontier worker role differs")
    if _validate_complete_worker(role, commit):
        return
    vocabulary, _ = parse_role(role)
    tokenizer, token_bytes = load_tokenizers()[vocabulary]
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
    evaluation_batch = EVALUATION_BATCH_BY_VOCABULARY[vocabulary]
    calibration_inventory, calibration_memory, calibration_path = (
        encode_stream_to_memmap(
            calibration_stream.data,
            tokenizer,
            token_bytes,
            first_batch_token_count=evaluation_batch * SEQUENCE_LENGTH,
        )
    )
    if (
        train_inventory.to_dict() != feasibility_plan["inventories"][role]["train"]
        or calibration_inventory.to_dict()
        != feasibility_plan["inventories"][role]["calibration"]
    ):
        raise ValueError("BPE quality frontier token inventory differs")
    train_sequences = train_memory[
        : train_inventory.full_sequence_count * SEQUENCE_LENGTH
    ].reshape(train_inventory.full_sequence_count, SEQUENCE_LENGTH)
    calibration_sequences = calibration_memory[
        : calibration_inventory.full_sequence_count * SEQUENCE_LENGTH
    ].reshape(calibration_inventory.full_sequence_count, SEQUENCE_LENGTH)
    order = deterministic_order(train_inventory.full_sequence_count)
    contract = role_training_contract(role, train_inventory.full_sequence_count)
    contract["training_order_sha256"] = array_sha256(order)
    if contract != plan["training"][role]:
        raise ValueError("BPE quality frontier worker training contract differs")
    raw_target_bytes = raw_target_bytes_by_sequence(calibration_sequences, token_bytes)
    if int(raw_target_bytes.sum()) != calibration_inventory.predicted_target_raw_bytes:
        raise ValueError("BPE quality contiguous denominator differs")
    pieces, document_common = calibration_document_pieces(SOURCE_PATH)
    document_inventory, chunks, chunk_documents, document_raw_bytes = (
        encode_document_chunks(pieces, tokenizer, token_bytes)
    )
    if (
        document_common
        != {key: plan["document_evaluation"]["common"][key] for key in document_common}
        or document_inventory.to_dict() != plan["document_evaluation"]["by_role"][role]
    ):
        raise ValueError("BPE quality document evaluation identity differs")
    state_start = _session_state()
    if not timing_environment_eligible(state_start):
        raise ValueError("BPE quality frontier worker environment is ineligible")
    with publication_mps_exclusive():
        model = build_frontier_model(role, seed=contract["model_seed"])
        if (
            model_parameter_count(model) != FRONTIER_SPECS[role].expected_parameters
            or state_sha256(model) != plan["initial_state_sha256"][role]
        ):
            raise ValueError("BPE quality frontier initialization differs")
        model.to("mps").train()
        training = _train(model, train_sequences, order, contract)
        model.eval()
        contiguous_nll = _evaluate_contiguous(
            model, calibration_sequences, evaluation_batch
        )
        document_nll = _evaluate_documents(
            model,
            chunks,
            chunk_documents,
            len(document_raw_bytes),
            evaluation_batch,
        )
        model.to("cpu")
        trained_state = state_sha256(model)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        del model
        gc.collect()
        torch.mps.empty_cache()
    state_end = _session_state()
    if not timing_environment_eligible(state_end):
        raise ValueError("BPE quality frontier worker environment became ineligible")
    checkpoint_payload = _checkpoint_bytes(state)
    arrays = {
        "contiguous_nll_nats": contiguous_nll,
        "contiguous_raw_target_bytes": raw_target_bytes,
        "document_nll_nats": document_nll,
        "document_raw_bytes": document_raw_bytes,
    }
    nll_payload = _npz_bytes(arrays)
    checkpoint_path = CHECKPOINT_ROOT / f"{role}.pt"
    nll_path = NLL_ROOT / f"{role}.npz"
    report_path = WORKER_ROOT / f"{role}.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "role": role,
        "parameter_count": FRONTIER_SPECS[role].expected_parameters,
        "training_contract": contract,
        "initial_state_sha256": plan["initial_state_sha256"][role],
        "trained_state_sha256": trained_state,
        "training": training,
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
        "environment": current_frontier_environment(),
        "session_state": {"start": state_start, "end": state_end},
    }
    report["worker_sha256"] = canonical_sha256(report)
    _publish(checkpoint_path, checkpoint_payload)
    _publish(nll_path, nll_payload)
    _publish(report_path, json_bytes(report))
    del train_memory, calibration_memory, state
    os.unlink(train_path)
    os.unlink(calibration_path)


def _parent() -> None:
    commit, _plan, _feasibility = _repository_context()
    if REPORT_PATH.exists() or OUTPUT_PATH.exists():
        raise FileExistsError("BPE quality frontier downstream output already exists")
    active_payload = json_bytes(
        {"git_commit": commit, "plan_artifact_sha256": hash_file(PLAN_PATH)}
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active_payload:
            raise ValueError("BPE quality frontier active session differs")
    else:
        _publish(ACTIVE_PATH, active_payload)
    for role in QUALITY_ROLES:
        if _validate_complete_worker(role, commit):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", role],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    workers = {}
    for role in QUALITY_ROLES:
        if not _validate_complete_worker(role, commit):
            raise ValueError("BPE quality frontier worker did not complete")
        path = WORKER_ROOT / f"{role}.json"
        workers[role] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        }
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during BPE quality frontier")
    report = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "workers": workers,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=QUALITY_ROLES)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        _parent()


if __name__ == "__main__":
    main()
