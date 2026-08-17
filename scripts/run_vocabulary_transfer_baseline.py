#!/usr/bin/env python3
"""Run the sealed nine-role strong vocabulary-transfer baseline closure."""

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
from bpe_quality_frontier_core import array_sha256, bpb, raw_target_bytes_by_sequence
from compositional_head_preflight_protocol import load_tokenizers
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_baseline_core import (
    BASE_VOCABULARY_SIZE,
    BASELINE_ROLES,
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    TARGET_VOCABULARY_SIZE,
    TWO_STAGE_BOUNDARY,
    build_transferred_model,
    expected_parameter_count,
    mask_old_row_gradient,
    restore_old_rows,
    role_definition,
    state_mapping_sha256,
)
from vocabulary_transfer_baseline_protocol import (
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
from vocabulary_transfer_probe_core import (
    BODY_LEARNING_RATE,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    HEAD_PEAK_LEARNING_RATE,
    TRAIN_MICROBATCH_SIZE,
    WEIGHT_DECAY,
    build_canonical_bpe_decomposition_table,
    probe_learning_rate,
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
        raise RuntimeError("baseline closure requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != commit:
        raise RuntimeError("baseline closure plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("baseline closure plan parent differs")
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
        raise RuntimeError("baseline token inventory differs")
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


def _all_parameter_optimizer(model: Any) -> torch.optim.Optimizer:
    lexical_parameter_ids = {
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
        if id(parameter) in lexical_parameter_ids:
            groups["head"].append(parameter)
        elif parameter.ndim >= 2:
            groups["body_decay"].append(parameter)
        else:
            groups["body_no_decay"].append(parameter)
    if (
        len(groups["head"]) != len(lexical_parameter_ids)
        or sum(len(values) for values in groups.values()) != len(seen)
    ):
        raise RuntimeError("baseline all-parameter optimizer grouping differs")
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


def _stage_one_optimizer(model: Any) -> torch.optim.Optimizer:
    weight = model.model.embed_tokens.weight
    if weight.data_ptr() != model.lm_head.weight.data_ptr():
        raise RuntimeError("baseline stage one requires a tied lexical matrix")
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is weight)
    return torch.optim.AdamW(
        [
            {
                "params": [weight],
                "lr": HEAD_PEAK_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "schedule_kind": "head",
            }
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
    stage_one_copied_rows: torch.Tensor | None,
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
    if stage_one_copied_rows is not None:
        weight = model.model.embed_tokens.weight
        mask_old_row_gradient(weight)
        torch.nn.utils.clip_grad_norm_([weight], GRADIENT_CLIP)
        optimizer.step()
        restore_old_rows(weight, stage_one_copied_rows)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise RuntimeError("baseline training loss became nonfinite")


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
                raise RuntimeError("baseline NLL became nonfinite")
            losses[start:stop] = row.cpu().numpy().astype(np.float32, copy=False)
            del values, logits, token_nll, row
    torch.mps.synchronize()
    return losses


def _body_state_sha256(model: Any) -> str:
    return state_mapping_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if name not in {"model.embed_tokens.weight", "lm_head.weight"}
        }
    )


def _paths(role: str) -> tuple[Path, dict[int, Path], dict[int, Path]]:
    return (
        WORKER_ROOT / f"{role}.json",
        {step: CHECKPOINT_ROOT / f"{role}-step-{step:04d}.pt" for step in PROBE_STEPS},
        {step: NLL_ROOT / f"{role}-step-{step:04d}.npz" for step in PROBE_STEPS},
    )


def _validate_worker(role: str, commit: str, plan: Mapping[str, Any]) -> bool:
    report_path, checkpoint_paths, nll_paths = _paths(role)
    all_paths = [report_path, *checkpoint_paths.values(), *nll_paths.values()]
    existing = [path.exists() for path in all_paths]
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(f"partial baseline worker requires forensics: {role}")
    report = read_json(report_path)
    unsigned = dict(report)
    receipt_sha = unsigned.pop("worker_sha256", None)
    if (
        set(report)
        != {
            "checkpoints",
            "complete",
            "environment",
            "git_commit",
            "initial_state_sha256",
            "initialization_audit",
            "kind",
            "parameter_count",
            "plan_artifact_sha256",
            "protocol_id",
            "role",
            "schema_version",
            "session_state",
            "stage_audit",
            "training",
            "training_contract",
            "worker_sha256",
        }
        or canonical_sha256(unsigned) != receipt_sha
        or report.get("schema_version") != 1
        or report.get("kind") != "vocabulary_transfer_baseline_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("role") != role
        or report.get("parameter_count") != expected_parameter_count(role)
        or report.get("training_contract") != plan["training"]
        or report.get("initialization_audit") != plan["initialization_audits"][role]
        or report.get("initial_state_sha256") != plan["initial_state_sha256"][role]
        or report.get("environment") != current_environment()
        or set(report.get("checkpoints", {})) != {str(step) for step in PROBE_STEPS}
    ):
        raise RuntimeError("completed baseline worker differs")
    training = report.get("training")
    session_state = report.get("session_state")
    definition = role_definition(role)
    expected_stage = {
        "body_and_copied_rows_unchanged_through_stage_one": (
            True if definition["training_schedule"].startswith("new_rows") else None
        ),
        "optimizer_reinitialized_at_step": (
            TWO_STAGE_BOUNDARY
            if definition["training_schedule"].startswith("new_rows")
            else None
        ),
        "stage_one_new_rows_changed": (
            True if definition["training_schedule"].startswith("new_rows") else None
        ),
        "training_schedule": definition["training_schedule"],
    }
    if (
        not isinstance(training, Mapping)
        or set(training)
        != {"completed", "elapsed_seconds_including_checkpoint_evaluations", "finite_optimizer_steps"}
        or training.get("completed") is not True
        or training.get("finite_optimizer_steps") != FINAL_PROBE_STEP
        or not isinstance(training.get("elapsed_seconds_including_checkpoint_evaluations"), (int, float))
        or not math.isfinite(float(training["elapsed_seconds_including_checkpoint_evaluations"]))
        or float(training["elapsed_seconds_including_checkpoint_evaluations"]) <= 0
        or report.get("stage_audit") != expected_stage
        or not isinstance(session_state, Mapping)
        or set(session_state) != {"start", "end"}
        or not timing_environment_eligible(session_state["start"])
        or not timing_environment_eligible(session_state["end"])
    ):
        raise RuntimeError("baseline training or stage receipt differs")
    calibration = plan["inventories"]["8192"]["calibration"]
    expected_count = int(calibration["full_sequence_count"])
    expected_raw_bytes = int(calibration["predicted_target_raw_bytes"])
    for step in PROBE_STEPS:
        row = report["checkpoints"][str(step)]
        checkpoint_path = checkpoint_paths[step]
        nll_path = nll_paths[step]
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "arrays",
                "bpb",
                "checkpoint_artifact_sha256",
                "checkpoint_path",
                "checkpoint_state_sha256",
                "nll_artifact_sha256",
                "nll_path",
            }
            or row.get("checkpoint_path") != str(checkpoint_path.relative_to(ROOT))
            or row.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
            or row.get("nll_path") != str(nll_path.relative_to(ROOT))
            or row.get("nll_artifact_sha256") != hash_file(nll_path)
        ):
            raise RuntimeError("baseline checkpoint lineage differs")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or state_mapping_sha256(state) != row["checkpoint_state_sha256"]:
            raise RuntimeError("baseline checkpoint state differs")
        with np.load(nll_path, allow_pickle=False) as archive:
            if set(archive.files) != {"nll_nats", "raw_target_bytes"}:
                raise RuntimeError("baseline NLL array set differs")
            arrays = row.get("arrays")
            if not isinstance(arrays, Mapping) or set(arrays) != set(archive.files):
                raise RuntimeError("baseline NLL metadata differs")
            for name in archive.files:
                values = archive[name]
                if arrays[name] != {
                    "dtype": str(values.dtype),
                    "shape": list(values.shape),
                    "sha256": array_sha256(values),
                }:
                    raise RuntimeError("baseline NLL array identity differs")
            nll = archive["nll_nats"]
            raw_bytes = archive["raw_target_bytes"]
            if (
                nll.dtype != np.float32
                or nll.shape != (expected_count,)
                or not np.isfinite(nll).all()
                or np.any(nll < 0)
                or raw_bytes.shape != (expected_count,)
                or not np.issubdtype(raw_bytes.dtype, np.integer)
                or np.any(raw_bytes <= 0)
                or int(raw_bytes.sum()) != expected_raw_bytes
                or row.get("bpb") != bpb(nll, raw_bytes)
            ):
                raise RuntimeError("baseline NLL semantics differ")
    return True


def _worker(role: str) -> None:
    commit, plan = _context()
    if role not in BASELINE_ROLES or _validate_worker(role, commit, plan):
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
        raise RuntimeError("baseline order differs")
    raw_target_bytes = raw_target_bytes_by_sequence(calibration_sequences, data["token_bytes"])
    if int(raw_target_bytes.sum()) != data["calibration_inventory"].predicted_target_raw_bytes:
        raise RuntimeError("baseline denominator differs")
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
        raise RuntimeError("baseline worker environment is ineligible")
    serialized: dict[int, tuple[bytes, bytes, str, dict[str, Any]]] = {}
    definition = role_definition(role)
    staged = definition["training_schedule"].startswith("new_rows")
    with publication_mps_exclusive():
        model, audit, _ = build_transferred_model(
            role,
            base_state=base_checkpoint_state(),
            base_tokenizer=base_tokenizer,
            base_pieces=base_pieces,
            target_pieces=data["token_bytes"],
            decompositions=decompositions,
        )
        if audit.to_dict() != plan["initialization_audits"][role]:
            raise RuntimeError("baseline initialization audit differs")
        parameter_count = model_parameter_count(model)
        if parameter_count != expected_parameter_count(role):
            raise RuntimeError("baseline parameter count differs")
        initial_state_sha = state_mapping_sha256(model.state_dict())
        if initial_state_sha != plan["initial_state_sha256"][role]:
            raise RuntimeError("baseline initial state differs")
        initial_body_sha = _body_state_sha256(model)
        initial_new_rows = model.model.embed_tokens.weight[BASE_VOCABULARY_SIZE:].detach().cpu().clone()
        model = model.to("mps")
        copied_rows = (
            model.model.embed_tokens.weight[:BASE_VOCABULARY_SIZE].detach().clone()
            if staged
            else None
        )
        optimizer = _stage_one_optimizer(model) if staged else _all_parameter_optimizer(model)
        optimizer_reinitialized = False
        finite_steps = 0
        started = time.perf_counter()
        for checkpoint_step in PROBE_STEPS:
            while finite_steps < checkpoint_step:
                if staged and finite_steps == TWO_STAGE_BOUNDARY:
                    if _body_state_sha256(model) != initial_body_sha:
                        raise RuntimeError("baseline body drifted during stage one")
                    if copied_rows is None or not torch.equal(
                        model.model.embed_tokens.weight[:BASE_VOCABULARY_SIZE].detach(),
                        copied_rows,
                    ):
                        raise RuntimeError("baseline copied rows drifted during stage one")
                    del optimizer
                    optimizer = _all_parameter_optimizer(model)
                    optimizer_reinitialized = True
                selected = order[
                    finite_steps * EFFECTIVE_BATCH_SIZE : (finite_steps + 1) * EFFECTIVE_BATCH_SIZE
                ]
                batch = np.asarray(train_sequences[selected], dtype=np.int64)
                model.train()
                _effective_step(
                    model,
                    optimizer,
                    batch,
                    step=finite_steps,
                    stage_one_copied_rows=(
                        copied_rows if staged and finite_steps < TWO_STAGE_BOUNDARY else None
                    ),
                )
                finite_steps += 1
            model.eval()
            nll = _evaluate_contiguous(model, calibration_sequences)
            state = {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            }
            checkpoint_payload = _checkpoint_bytes(state)
            arrays = {"nll_nats": nll, "raw_target_bytes": raw_target_bytes}
            nll_payload = _npz_bytes(arrays)
            serialized[checkpoint_step] = (
                checkpoint_payload,
                nll_payload,
                state_mapping_sha256(state),
                {
                    "bpb": bpb(nll, raw_target_bytes),
                    "arrays": {
                        name: {
                            "dtype": str(values.dtype),
                            "shape": list(values.shape),
                            "sha256": array_sha256(values),
                        }
                        for name, values in arrays.items()
                    },
                },
            )
            del state, nll, arrays
        elapsed = time.perf_counter() - started
        if staged and not optimizer_reinitialized:
            raise RuntimeError("baseline stage-two optimizer was not initialized")
        final_body_sha = _body_state_sha256(model)
        final_new_rows = model.model.embed_tokens.weight[BASE_VOCABULARY_SIZE:].detach().cpu()
        stage_one_checkpoint = torch.load(
            io.BytesIO(serialized[TWO_STAGE_BOUNDARY][0]), map_location="cpu", weights_only=True
        )
        stage_one_body_sha = state_mapping_sha256(
            {
                name: value
                for name, value in stage_one_checkpoint.items()
                if name not in {"model.embed_tokens.weight", "lm_head.weight"}
            }
        )
        stage_one_old_rows = stage_one_checkpoint["model.embed_tokens.weight"][:BASE_VOCABULARY_SIZE]
        stage_one_new_rows = stage_one_checkpoint["model.embed_tokens.weight"][BASE_VOCABULARY_SIZE:]
        stage_audit = {
            "body_and_copied_rows_unchanged_through_stage_one": (
                stage_one_body_sha == initial_body_sha
                and torch.equal(stage_one_old_rows, copied_rows.detach().cpu())
                if staged and copied_rows is not None
                else None
            ),
            "optimizer_reinitialized_at_step": TWO_STAGE_BOUNDARY if staged else None,
            "stage_one_new_rows_changed": (
                not torch.equal(stage_one_new_rows, initial_new_rows) if staged else None
            ),
            "training_schedule": definition["training_schedule"],
        }
        if staged and (
            stage_audit["body_and_copied_rows_unchanged_through_stage_one"] is not True
            or stage_audit["stage_one_new_rows_changed"] is not True
            or final_body_sha == initial_body_sha
            or torch.equal(final_new_rows, initial_new_rows)
        ):
            raise RuntimeError("baseline two-stage invariants differ")
        model.to("cpu")
        del optimizer, model, stage_one_checkpoint
        gc.collect()
        torch.mps.empty_cache()
    end_state = _session_state()
    if not timing_environment_eligible(end_state):
        raise RuntimeError("baseline worker environment changed")
    report_path, checkpoint_paths, nll_paths = _paths(role)
    checkpoints = {}
    for step in PROBE_STEPS:
        checkpoint_payload, nll_payload, state_sha, evidence = serialized[step]
        checkpoints[str(step)] = {
            "checkpoint_path": str(checkpoint_paths[step].relative_to(ROOT)),
            "checkpoint_artifact_sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "checkpoint_state_sha256": state_sha,
            "nll_path": str(nll_paths[step].relative_to(ROOT)),
            "nll_artifact_sha256": hashlib.sha256(nll_payload).hexdigest(),
            "arrays": evidence["arrays"],
            "bpb": evidence["bpb"],
        }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vocabulary_transfer_baseline_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "role": role,
        "parameter_count": parameter_count,
        "training_contract": training_contract(),
        "initialization_audit": plan["initialization_audits"][role],
        "initial_state_sha256": plan["initial_state_sha256"][role],
        "training": {
            "completed": True,
            "elapsed_seconds_including_checkpoint_evaluations": elapsed,
            "finite_optimizer_steps": finite_steps,
        },
        "stage_audit": stage_audit,
        "checkpoints": checkpoints,
        "environment": current_environment(),
        "session_state": {"start": start_state, "end": end_state},
    }
    report["worker_sha256"] = canonical_sha256(report)
    for step in PROBE_STEPS:
        checkpoint_payload, nll_payload, _, _ = serialized[step]
        _publish(checkpoint_paths[step], checkpoint_payload)
        _publish(nll_paths[step], nll_payload)
    _publish(report_path, json_bytes(report))
    _cleanup_data(data)


def _parent() -> None:
    commit, plan = _context()
    if REPORT_PATH.exists() or OUTPUT_PATH.exists():
        raise RuntimeError("baseline downstream output already exists")
    active_payload = json_bytes(
        {
            "git_commit": commit,
            "kind": "vocabulary_transfer_baseline_active_v1",
            "plan_artifact_sha256": hash_file(PLAN_PATH),
        }
    )
    if ACTIVE_PATH.exists():
        if ACTIVE_PATH.read_bytes() != active_payload:
            raise RuntimeError("baseline active session differs")
    else:
        _publish(ACTIVE_PATH, active_payload)
    for role in BASELINE_ROLES:
        if _validate_worker(role, commit, plan):
            continue
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", role],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    workers = {}
    for role in BASELINE_ROLES:
        if not _validate_worker(role, commit, plan):
            raise RuntimeError("baseline worker did not complete")
        path = _paths(role)[0]
        workers[role] = {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during baseline closure")
    report = {
        "schema_version": 1,
        "kind": "vocabulary_transfer_baseline_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "workers": workers,
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish(REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    print("status=vocabulary_transfer_baseline_workers_complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=BASELINE_ROLES)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        _parent()


if __name__ == "__main__":
    main()

