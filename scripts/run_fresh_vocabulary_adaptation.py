#!/usr/bin/env python3
"""Run the sealed fresh one-seed vocabulary-adaptation workers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import math
import os
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
from compositional_head_core import build_model
from compositional_head_preflight_protocol import (
    current_environment,
    hash_file,
    load_tokenizers,
    tokenizer_identity,
)
from foldable_multihash_mechanism_core import scale_new_row_update_
from fresh_vocabulary_adaptation_core import (
    BASE_VOCABULARY_SIZE,
    BODY_LEARNING_RATE,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_BY_VOCABULARY,
    GRADIENT_CLIP,
    HEAD_PEAK_LEARNING_RATE,
    INPUT_UPDATE_MULTIPLIER,
    OUTPUT_UPDATE_MULTIPLIER,
    ROLES,
    SEQUENCE_LENGTH,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_BY_VOCABULARY,
    WEIGHT_DECAY,
    batch_raw_target_bytes,
    head_learning_rate,
    role_definition,
)
from fresh_vocabulary_adaptation_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_ROOT,
    FRESH_SOURCE_PATH,
    INITIALIZER_ROLE,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    WORKER_ROOT,
    base_checkpoint_state,
    canonical_sha256,
    dependency_identity,
    implementation_identity,
    json_bytes,
    read_json,
    validate_plan,
    verified_fresh_streams,
)
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import (
    build_canonical_bpe_decomposition_table,
    build_transferred_model,
    state_mapping_sha256,
)

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


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


def _checkpoint_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    output = io.BytesIO()
    torch.save(dict(state), output)
    return output.getvalue()


def _context() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-adaptation run requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if (
        _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT)))
        != commit
    ):
        raise RuntimeError("fresh-adaptation plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if (
        _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]
        or plan["dependencies"] != dependency_identity()
        or plan["implementation_sha256"] != implementation_identity()
        or plan["environment"] != current_environment()
        or plan["tokenizers"]
        != {
            key: tokenizer_identity()[key]
            for key in (str(BASE_VOCABULARY_SIZE), str(TARGET_VOCABULARY_SIZE))
        }
    ):
        raise RuntimeError("fresh-adaptation run context differs")
    return commit, plan


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": array_sha256(value),
    }


def _role_paths(role: str) -> tuple[Path, Path, Path]:
    return (
        WORKER_ROOT / f"{role}.json",
        CHECKPOINT_ROOT / f"{role}.pt",
        NLL_ROOT / f"{role}.npz",
    )


def _load_role_data(role: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    definition = role_definition(role)
    vocabulary_size = int(definition["vocabulary_size"])
    tokenizer, token_bytes = load_tokenizers()[vocabulary_size]
    streams = verified_fresh_streams()
    training = plan["training"][role]
    train_inventory, train_memory, train_path = encode_stream_to_memmap(
        streams["train"].data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH,
    )
    calibration_inventory, calibration_memory, calibration_path = (
        encode_stream_to_memmap(
            streams["calibration"].data,
            tokenizer,
            token_bytes,
            first_batch_token_count=(
                EVALUATION_BATCH_BY_VOCABULARY[vocabulary_size] * SEQUENCE_LENGTH
            ),
        )
    )
    expected = plan["inventories"][str(vocabulary_size)]
    if (
        train_inventory.to_dict() != expected["train_tokens"]
        or calibration_inventory.to_dict() != expected["calibration_tokens"]
        or train_inventory.full_sequence_count != training["sequence_count"]
    ):
        raise RuntimeError("fresh-adaptation token inventory differs")
    train_count = int(train_inventory.full_sequence_count)
    calibration_count = int(calibration_inventory.full_sequence_count)
    train_sequences = train_memory[: train_count * SEQUENCE_LENGTH].reshape(
        train_count, SEQUENCE_LENGTH
    )
    calibration_sequences = calibration_memory[
        : calibration_count * SEQUENCE_LENGTH
    ].reshape(calibration_count, SEQUENCE_LENGTH)
    train_raw = raw_target_bytes_by_sequence(train_sequences, token_bytes)
    calibration_raw = raw_target_bytes_by_sequence(calibration_sequences, token_bytes)
    batch_raw = batch_raw_target_bytes(train_raw)
    if (
        array_sha256(train_raw) != expected["train_raw_target_bytes_sha256"]
        or array_sha256(batch_raw)
        != expected["optimizer_batch_raw_target_bytes_sha256"]
        or int(train_raw.sum()) != training["target_raw_bytes"]
        or len(batch_raw) != training["total_optimizer_steps"]
        or int(calibration_raw.sum())
        != calibration_inventory.predicted_target_raw_bytes
    ):
        raise RuntimeError("fresh-adaptation raw-byte inventory differs")
    pieces, common = calibration_document_pieces(FRESH_SOURCE_PATH)
    document_inventory, chunks, chunk_documents, document_raw = encode_document_chunks(
        pieces, tokenizer, token_bytes
    )
    if (
        common != plan["document_common"]
        or document_inventory.to_dict() != expected["document_tokens"]
        or array_sha256(document_raw) != expected["document_raw_bytes_sha256"]
    ):
        raise RuntimeError("fresh-adaptation document inventory differs")
    return {
        "vocabulary_size": vocabulary_size,
        "tokenizer": tokenizer,
        "token_bytes": token_bytes,
        "train_inventory": train_inventory,
        "train_memory": train_memory,
        "train_path": train_path,
        "train_sequences": train_sequences,
        "train_raw_target_bytes": train_raw,
        "optimizer_batch_raw_target_bytes": batch_raw,
        "calibration_inventory": calibration_inventory,
        "calibration_memory": calibration_memory,
        "calibration_path": calibration_path,
        "calibration_sequences": calibration_sequences,
        "calibration_raw_target_bytes": calibration_raw,
        "document_chunks": chunks,
        "chunk_documents": chunk_documents,
        "document_raw_bytes": document_raw,
    }


def _cleanup_role_data(data: Mapping[str, Any]) -> None:
    for key in ("train_memory", "calibration_memory"):
        memory = data[key]
        if hasattr(memory, "_mmap") and memory._mmap is not None:
            memory._mmap.close()
    for key in ("train_path", "calibration_path"):
        if os.path.exists(data[key]):
            os.unlink(data[key])


def _build_initial_model(
    role: str, plan: Mapping[str, Any], data: Mapping[str, Any]
) -> Any:
    initialization = plan["initialization"]
    base_state = base_checkpoint_state()
    if role == "dense2k_joint":
        model = build_model("dense_v2048")
        model.load_state_dict(base_state, strict=True)
        expected_state = initialization["dense2k_initial_state_sha256"]
    else:
        tokenizers = load_tokenizers()
        base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
        target_tokenizer = data["tokenizer"]
        decompositions = build_canonical_bpe_decomposition_table(
            base_tokenizer,
            target_tokenizer,
            base_pieces,
            data["token_bytes"],
        )
        model, audit = build_transferred_model(
            INITIALIZER_ROLE,
            base_state=base_state,
            base_pieces=base_pieces,
            target_pieces=data["token_bytes"],
            decompositions=decompositions,
        )
        if audit.to_dict() != initialization["dense8k_initialization_audit"]:
            raise RuntimeError("fresh-adaptation initialization audit differs")
        expected_state = initialization["dense8k_initial_state_sha256"]
    if (
        model_parameter_count(model) != initialization["parameter_count_by_role"][role]
        or state_mapping_sha256(model.state_dict()) != expected_state
    ):
        raise RuntimeError("fresh-adaptation initial model differs")
    return model


def _unique_parameters(
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    output: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            output.append(parameter)
    return output


def _all_parameter_optimizer(model: Any) -> torch.optim.Optimizer:
    lexical_ids = {
        id(model.model.embed_tokens.weight),
        id(model.lm_head.weight),
    }
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
        if id(parameter) in lexical_ids:
            groups["head"].append(parameter)
        elif parameter.ndim >= 2:
            groups["body_decay"].append(parameter)
        else:
            groups["body_no_decay"].append(parameter)
    if not groups["head"] or sum(map(len, groups.values())) != len(seen):
        raise RuntimeError("fresh-adaptation optimizer grouping differs")
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


def _new_rows_optimizer(model: Any) -> torch.optim.Optimizer:
    lexical = _unique_parameters(
        [model.model.embed_tokens.weight, model.lm_head.weight]
    )
    if len(lexical) != 2:
        raise RuntimeError("fresh-adaptation in-place control requires untied heads")
    lexical_ids = {id(value) for value in lexical}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in lexical_ids)
    return torch.optim.AdamW(
        [
            {
                "params": lexical,
                "lr": HEAD_PEAK_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "schedule_kind": "head",
            }
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _body_state_sha256(model: Any) -> str:
    return state_mapping_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if name not in {"model.embed_tokens.weight", "lm_head.weight"}
        }
    )


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    role: str,
    *,
    cumulative_raw_target_bytes: int,
    total_raw_target_bytes: int,
    stage_one_raw_target_bytes: int | None,
) -> float:
    head_lr = head_learning_rate(
        role,
        cumulative_raw_target_bytes=cumulative_raw_target_bytes,
        total_raw_target_bytes=total_raw_target_bytes,
        stage_one_raw_target_bytes=stage_one_raw_target_bytes,
    )
    for group in optimizer.param_groups:
        group["lr"] = (
            BODY_LEARNING_RATE if group["schedule_kind"] == "body" else head_lr
        )
    return head_lr


def _effective_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: np.ndarray,
    *,
    role: str,
    cumulative_raw_target_bytes: int,
    total_raw_target_bytes: int,
    stage_one_raw_target_bytes: int | None,
    stage_one: bool,
    copied_input_rows: torch.Tensor | None,
    copied_output_rows: torch.Tensor | None,
) -> float:
    head_lr = _set_learning_rates(
        optimizer,
        role,
        cumulative_raw_target_bytes=cumulative_raw_target_bytes,
        total_raw_target_bytes=total_raw_target_bytes,
        stage_one_raw_target_bytes=stage_one_raw_target_bytes,
    )
    optimizer.zero_grad(set_to_none=True)
    finite = torch.ones((), dtype=torch.bool, device="mps")
    microbatch = TRAIN_MICROBATCH_BY_VOCABULARY[
        role_definition(role)["vocabulary_size"]
    ]
    for start in range(0, len(batch), microbatch):
        selected = batch[start : start + microbatch]
        values = torch.tensor(selected, dtype=torch.long, device="mps")
        output = model(input_ids=values, labels=values, use_cache=False)
        loss = output.loss * (len(selected) / len(batch))
        finite = finite & torch.isfinite(output.loss.detach())
        loss.backward()
        del values, output, loss

    input_weight = model.model.embed_tokens.weight
    output_weight = model.lm_head.weight
    if stage_one:
        if (
            copied_input_rows is None
            or copied_output_rows is None
            or input_weight.grad is None
            or output_weight.grad is None
        ):
            raise RuntimeError("fresh-adaptation stage-one state differs")
        input_weight.grad[:BASE_VOCABULARY_SIZE].zero_()
        output_weight.grad[:BASE_VOCABULARY_SIZE].zero_()

    trainable = _unique_parameters(
        [parameter for group in optimizer.param_groups for parameter in group["params"]]
    )
    torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
    input_before = None
    output_before = None
    if role == "dense8k_update_geometry":
        input_before = input_weight[BASE_VOCABULARY_SIZE:].detach().clone()
        output_before = output_weight[BASE_VOCABULARY_SIZE:].detach().clone()
    optimizer.step()
    if stage_one:
        with torch.no_grad():
            input_weight[:BASE_VOCABULARY_SIZE].copy_(copied_input_rows)
            output_weight[:BASE_VOCABULARY_SIZE].copy_(copied_output_rows)
    if role == "dense8k_update_geometry":
        if input_before is None or output_before is None:
            raise AssertionError("fresh-adaptation update snapshot is missing")
        scale_new_row_update_(input_weight, input_before, INPUT_UPDATE_MULTIPLIER)
        scale_new_row_update_(output_weight, output_before, OUTPUT_UPDATE_MULTIPLIER)
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise RuntimeError("fresh-adaptation training loss became nonfinite")
    return head_lr


def _evaluate_contiguous(
    model: Any, sequences: np.ndarray, batch_size: int
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
                logits[:, :-1, :].transpose(1, 2), values[:, 1:], reduction="none"
            )
            row = token_nll.sum(dim=1)
            if not torch.isfinite(row).all():
                raise RuntimeError("fresh-adaptation contiguous NLL became nonfinite")
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
                logits[:, :-1, :].transpose(1, 2), values[:, 1:], reduction="none"
            )
            row = (token_nll * mask[:, 1:]).sum(dim=1)
            if not torch.isfinite(row).all():
                raise RuntimeError("fresh-adaptation document NLL became nonfinite")
            np.add.at(
                output,
                chunk_documents[start : start + len(selected)],
                row.cpu().numpy().astype(np.float64, copy=False),
            )
            del values, mask, logits, token_nll, row
    torch.mps.synchronize()
    return output


def _load_nll(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "contiguous_nll_nats",
            "contiguous_raw_target_bytes",
            "document_nll_nats",
            "document_raw_bytes",
        }:
            raise RuntimeError("fresh-adaptation NLL array set differs")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _validate_worker(role: str, commit: str, plan: Mapping[str, Any]) -> bool:
    report_path, checkpoint_path, nll_path = _role_paths(role)
    exists = tuple(path.exists() for path in (report_path, checkpoint_path, nll_path))
    if not any(exists):
        return False
    if not all(exists):
        raise RuntimeError(
            f"partial fresh-adaptation worker requires forensics: {role}"
        )
    report = read_json(report_path)
    unsigned = dict(report)
    worker_sha = unsigned.pop("worker_sha256", None)
    if (
        canonical_sha256(unsigned) != worker_sha
        or report.get("schema_version") != 1
        or report.get("kind") != "fresh_vocabulary_adaptation_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("role") != role
        or report.get("role_definition") != role_definition(role)
        or report.get("training_contract") != plan["training"][role]
        or report.get("parameter_count")
        != plan["initialization"]["parameter_count_by_role"][role]
        or report.get("checkpoint_path") != str(checkpoint_path.relative_to(ROOT))
        or report.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
        or report.get("nll_path") != str(nll_path.relative_to(ROOT))
        or report.get("nll_artifact_sha256") != hash_file(nll_path)
        or report.get("environment") != current_environment()
    ):
        raise RuntimeError("completed fresh-adaptation worker differs")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    arrays = _load_nll(nll_path)
    if (
        not isinstance(state, Mapping)
        or state_mapping_sha256(state) != report.get("checkpoint_state_sha256")
        or report.get("arrays")
        != {name: _array_descriptor(value) for name, value in arrays.items()}
        or arrays["contiguous_nll_nats"].dtype != np.float32
        or arrays["document_nll_nats"].dtype != np.float64
        or arrays["contiguous_raw_target_bytes"].dtype != np.int64
        or arrays["document_raw_bytes"].dtype != np.int64
        or not all(np.isfinite(value).all() for value in arrays.values())
        or np.any(arrays["contiguous_nll_nats"] < 0)
        or np.any(arrays["document_nll_nats"] < 0)
        or report.get("metrics")
        != {
            "contiguous_bpb": bpb(
                arrays["contiguous_nll_nats"], arrays["contiguous_raw_target_bytes"]
            ),
            "document_bpb": bpb(
                arrays["document_nll_nats"], arrays["document_raw_bytes"]
            ),
        }
    ):
        raise RuntimeError("completed fresh-adaptation evidence differs")
    training = report.get("training")
    if (
        not isinstance(training, Mapping)
        or training.get("completed") is not True
        or training.get("optimizer_steps")
        != plan["training"][role]["total_optimizer_steps"]
        or training.get("seen_sequences") != plan["training"][role]["sequence_count"]
        or training.get("seen_target_raw_bytes")
        != plan["training"][role]["target_raw_bytes"]
        or not math.isfinite(float(training.get("optimizer_elapsed_seconds", math.nan)))
        or float(training["optimizer_elapsed_seconds"]) <= 0
    ):
        raise RuntimeError("completed fresh-adaptation training receipt differs")
    return True


def _worker(role: str) -> None:
    commit, plan = _context()
    if role not in ROLES or _validate_worker(role, commit, plan):
        return
    data = _load_role_data(role, plan)
    training = plan["training"][role]
    start_state = _session_state()
    if not timing_environment_eligible(start_state):
        raise RuntimeError("fresh-adaptation worker environment is ineligible")
    try:
        with publication_mps_exclusive():
            model = _build_initial_model(role, plan, data)
            parameter_count = model_parameter_count(model)
            initial_body_sha = _body_state_sha256(model)
            initial_input_new = (
                model.model.embed_tokens.weight[BASE_VOCABULARY_SIZE:].detach().clone()
                if role == "dense8k_inplace_two_stage"
                else None
            )
            initial_output_new = (
                model.lm_head.weight[BASE_VOCABULARY_SIZE:].detach().clone()
                if role == "dense8k_inplace_two_stage"
                else None
            )
            model = model.to("mps")
            staged = role == "dense8k_inplace_two_stage"
            stage_contract = training.get("inplace_stage")
            stage_one_steps = (
                int(stage_contract["stage_one_optimizer_steps"]) if staged else 0
            )
            stage_one_bytes = (
                int(stage_contract["stage_one_raw_target_bytes"]) if staged else None
            )
            copied_input = (
                model.model.embed_tokens.weight[:BASE_VOCABULARY_SIZE].detach().clone()
                if staged
                else None
            )
            copied_output = (
                model.lm_head.weight[:BASE_VOCABULARY_SIZE].detach().clone()
                if staged
                else None
            )
            optimizer = (
                _new_rows_optimizer(model)
                if staged
                else _all_parameter_optimizer(model)
            )
            batch_raw = data["optimizer_batch_raw_target_bytes"]
            cumulative = np.cumsum(batch_raw, dtype=np.int64)
            optimizer_elapsed = 0.0
            first_head_lr = None
            last_head_lr = None
            stage_one_audit = None
            training_started = time.perf_counter()
            for step in range(len(batch_raw)):
                if staged and step == stage_one_steps:
                    stage_one_audit = {
                        "body_unchanged": _body_state_sha256(model) == initial_body_sha,
                        "copied_input_rows_unchanged": torch.equal(
                            model.model.embed_tokens.weight[
                                :BASE_VOCABULARY_SIZE
                            ].detach(),
                            copied_input,
                        ),
                        "copied_output_rows_unchanged": torch.equal(
                            model.lm_head.weight[:BASE_VOCABULARY_SIZE].detach(),
                            copied_output,
                        ),
                        "new_input_rows_changed": not torch.equal(
                            model.model.embed_tokens.weight[BASE_VOCABULARY_SIZE:]
                            .detach()
                            .cpu(),
                            initial_input_new,
                        ),
                        "new_output_rows_changed": not torch.equal(
                            model.lm_head.weight[BASE_VOCABULARY_SIZE:].detach().cpu(),
                            initial_output_new,
                        ),
                    }
                    if not all(stage_one_audit.values()):
                        raise RuntimeError(
                            "fresh-adaptation in-place stage-one invariant differs"
                        )
                    del optimizer
                    optimizer = _all_parameter_optimizer(model)
                start = step * EFFECTIVE_BATCH_SIZE
                stop = min(start + EFFECTIVE_BATCH_SIZE, len(data["train_sequences"]))
                batch = np.asarray(data["train_sequences"][start:stop], dtype=np.int64)
                model.train()
                step_started = time.perf_counter()
                lr = _effective_step(
                    model,
                    optimizer,
                    batch,
                    role=role,
                    cumulative_raw_target_bytes=int(cumulative[step]),
                    total_raw_target_bytes=int(cumulative[-1]),
                    stage_one_raw_target_bytes=stage_one_bytes,
                    stage_one=staged and step < stage_one_steps,
                    copied_input_rows=copied_input,
                    copied_output_rows=copied_output,
                )
                optimizer_elapsed += time.perf_counter() - step_started
                first_head_lr = lr if first_head_lr is None else first_head_lr
                last_head_lr = lr
            model.eval()
            evaluation_batch = EVALUATION_BATCH_BY_VOCABULARY[data["vocabulary_size"]]
            contiguous = _evaluate_contiguous(
                model, data["calibration_sequences"], evaluation_batch
            )
            document = _evaluate_documents(
                model,
                data["document_chunks"],
                data["chunk_documents"],
                len(data["document_raw_bytes"]),
                evaluation_batch,
            )
            total_elapsed = time.perf_counter() - training_started
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            }
            checkpoint_state_sha = state_mapping_sha256(state)
            model.to("cpu")
            del optimizer, model
            gc.collect()
            torch.mps.empty_cache()
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError("fresh-adaptation worker environment changed")
        if _git("rev-parse", "HEAD") != commit or _git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("fresh-adaptation repository changed during training")
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
        checkpoint_payload = _checkpoint_bytes(state)
        nll_payload = _npz_bytes(arrays)
        report_path, checkpoint_path, nll_path = _role_paths(role)
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "fresh_vocabulary_adaptation_worker_v1",
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "role": role,
            "role_definition": role_definition(role),
            "training_contract": training,
            "parameter_count": parameter_count,
            "initial_state_sha256": (
                plan["initialization"]["dense2k_initial_state_sha256"]
                if role == "dense2k_joint"
                else plan["initialization"]["dense8k_initial_state_sha256"]
            ),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_artifact_sha256": hashlib.sha256(
                checkpoint_payload
            ).hexdigest(),
            "checkpoint_state_sha256": checkpoint_state_sha,
            "nll_path": str(nll_path.relative_to(ROOT)),
            "nll_artifact_sha256": hashlib.sha256(nll_payload).hexdigest(),
            "arrays": {
                name: _array_descriptor(value) for name, value in arrays.items()
            },
            "metrics": {
                "contiguous_bpb": bpb(
                    arrays["contiguous_nll_nats"], arrays["contiguous_raw_target_bytes"]
                ),
                "document_bpb": bpb(
                    arrays["document_nll_nats"], arrays["document_raw_bytes"]
                ),
            },
            "training": {
                "completed": True,
                "optimizer_steps": len(batch_raw),
                "seen_sequences": len(data["train_sequences"]),
                "seen_target_raw_bytes": int(cumulative[-1]),
                "first_head_learning_rate": first_head_lr,
                "last_head_learning_rate": last_head_lr,
                "optimizer_elapsed_seconds": optimizer_elapsed,
                "elapsed_seconds_including_evaluation": total_elapsed,
            },
            "stage_one_audit": stage_one_audit,
            "environment": current_environment(),
            "session_state": {"start": start_state, "end": end_state},
        }
        if (
            report["parameter_count"]
            != plan["initialization"]["parameter_count_by_role"][role]
        ):
            raise RuntimeError("fresh-adaptation trained parameter count differs")
        report["worker_sha256"] = canonical_sha256(report)
        _publish(checkpoint_path, checkpoint_payload)
        _publish(nll_path, nll_payload)
        _publish(report_path, json_bytes(report))
    finally:
        _cleanup_role_data(data)


def _parent() -> None:
    commit, plan = _context()
    if REPORT_PATH.exists() or OUTPUT_PATH.exists():
        raise RuntimeError("fresh-adaptation downstream output already exists")
    active_payload = json_bytes(
        {
            "kind": "fresh_vocabulary_adaptation_active_v1",
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
        }
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active_payload:
            raise RuntimeError("fresh-adaptation active session differs")
    else:
        _publish(ACTIVE_PATH, active_payload)
    for role in ROLES:
        if _validate_worker(role, commit, plan):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", role],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    workers = {}
    for role in ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("fresh-adaptation worker did not complete")
        workers[role] = {
            "path": str((WORKER_ROOT / f"{role}.json").relative_to(ROOT)),
            "sha256": hash_file(WORKER_ROOT / f"{role}.json"),
        }
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("fresh-adaptation repository changed during campaign")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_adaptation_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "workers": workers,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print("status=fresh_vocabulary_adaptation_workers_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=ROLES)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        _parent()


if __name__ == "__main__":
    main()
