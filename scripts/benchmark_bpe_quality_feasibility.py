#!/usr/bin/env python3
"""Measure isolated real MPS train/eval steps for the six BPE frontier roles."""

from __future__ import annotations

import argparse
import gc
import os
import resource
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bpe_quality_feasibility_core import (
    CALIBRATION_BYTES,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_BY_VOCABULARY,
    MEASURED_EFFECTIVE_STEPS,
    MEASURED_EVALUATION_BATCHES,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
    TRAIN_BYTES,
    TRAIN_MICROBATCH_BY_VOCABULARY,
    WARMUP_EFFECTIVE_STEPS,
    WARMUP_EVALUATION_BATCHES,
    encode_stream_to_memmap,
    first_sequence_batch,
    quality_role_contract,
)
from bpe_quality_feasibility_protocol import (
    ACTIVE_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    SOURCE_PATH,
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
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_data import build_neural_stream

WORKER_ROOT = REPORT_PATH.parent / "workers"


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _command_snapshot(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stderr_sha256": __import__("hashlib").sha256(result.stderr.encode()).hexdigest(),
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


def _repository_context() -> tuple[str, dict[str, Any]]:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("BPE quality feasibility requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    last_change = _command(
        "git", "log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))
    )
    if last_change != commit:
        raise ValueError("BPE quality feasibility plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _command("git", "rev-parse", "HEAD^") != plan["dependencies"][
        "git_commit_before_plan"
    ]:
        raise ValueError("BPE quality feasibility plan parent differs")
    return commit, plan


def _mps_memory() -> dict[str, int]:
    torch.mps.synchronize()
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _optimizer(model: Any) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _train_effective_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: np.ndarray,
    microbatch: int,
) -> None:
    accumulation = EFFECTIVE_BATCH_SIZE // microbatch
    optimizer.zero_grad(set_to_none=True)
    last_loss = None
    for index in range(accumulation):
        values = torch.tensor(
            batch[index * microbatch : (index + 1) * microbatch],
            dtype=torch.long,
            device="mps",
        )
        output = model(input_ids=values, labels=values, use_cache=False)
        loss = output.loss / accumulation
        last_loss = loss.detach()
        loss.backward()
        del output, loss, values
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.mps.synchronize()
    if last_loss is None or not bool(torch.isfinite(last_loss).item()):
        raise ValueError("BPE quality feasibility training loss is nonfinite")


def _evaluate_batch(model: Any, batch: np.ndarray) -> None:
    with torch.inference_mode():
        values = torch.tensor(batch, dtype=torch.long, device="mps")
        output = model(input_ids=values, labels=values, use_cache=False)
        finite = torch.isfinite(output.loss)
        del output, values
    torch.mps.synchronize()
    if not bool(finite.item()):
        raise ValueError("BPE quality feasibility evaluation loss is nonfinite")


def _worker(role: str) -> None:
    commit, plan = _repository_context()
    if role not in QUALITY_ROLES:
        raise ValueError("BPE quality feasibility worker role differs")
    vocabulary, _ = parse_role(role)
    output_path = WORKER_ROOT / f"{role}.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    state_start = _session_state()
    if not timing_environment_eligible(state_start):
        raise ValueError("BPE quality feasibility worker environment is ineligible")
    tokenizers = load_tokenizers()
    tokenizer, token_bytes = tokenizers[vocabulary]
    train = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="train",
        byte_limit=TRAIN_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    calibration = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    train_inventory, train_memory, train_path = encode_stream_to_memmap(
        train.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH,
    )
    eval_batch_size = EVALUATION_BATCH_BY_VOCABULARY[vocabulary]
    calibration_inventory, calibration_memory, calibration_path = encode_stream_to_memmap(
        calibration.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=eval_batch_size * SEQUENCE_LENGTH,
    )
    if (
        train_inventory.to_dict() != plan["inventories"][role]["train"]
        or calibration_inventory.to_dict()
        != plan["inventories"][role]["calibration"]
    ):
        raise ValueError("BPE quality feasibility worker inventory differs")
    train_batch = first_sequence_batch(train_memory, EFFECTIVE_BATCH_SIZE)
    eval_batch = first_sequence_batch(calibration_memory, eval_batch_size)
    del train_memory, calibration_memory
    os.unlink(train_path)
    os.unlink(calibration_path)
    with publication_mps_exclusive():
        torch.mps.empty_cache()
        baseline = _mps_memory()
        model = build_frontier_model(role, seed=20_260_816).eval().to("mps")
        if model_parameter_count(model) != FRONTIER_SPECS[role].expected_parameters:
            raise ValueError("BPE quality feasibility model parameters differ")
        optimizer = _optimizer(model)
        microbatch = TRAIN_MICROBATCH_BY_VOCABULARY[vocabulary]
        model.train()
        for _ in range(WARMUP_EFFECTIVE_STEPS):
            _train_effective_step(model, optimizer, train_batch, microbatch)
        train_seconds = []
        memory_samples = [_mps_memory()]
        for _ in range(MEASURED_EFFECTIVE_STEPS):
            started = time.perf_counter_ns()
            _train_effective_step(model, optimizer, train_batch, microbatch)
            finished = time.perf_counter_ns()
            train_seconds.append((finished - started) / 1_000_000_000)
            memory_samples.append(_mps_memory())
        model.eval()
        for _ in range(WARMUP_EVALUATION_BATCHES):
            _evaluate_batch(model, eval_batch)
        evaluation_seconds = []
        for _ in range(MEASURED_EVALUATION_BATCHES):
            started = time.perf_counter_ns()
            _evaluate_batch(model, eval_batch)
            finished = time.perf_counter_ns()
            evaluation_seconds.append((finished - started) / 1_000_000_000)
            memory_samples.append(_mps_memory())
        parameter_count = model_parameter_count(model)
        del optimizer, model
        gc.collect()
        torch.mps.empty_cache()
        released = _mps_memory()
    state_end = _session_state()
    if not timing_environment_eligible(state_end):
        raise ValueError("BPE quality feasibility worker environment became ineligible")
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_feasibility_worker_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "role": role,
        "contract": quality_role_contract(role, vocabulary),
        "parameter_count": parameter_count,
        "train_inventory": train_inventory.to_dict(),
        "calibration_inventory": calibration_inventory.to_dict(),
        "train_effective_step_seconds": train_seconds,
        "evaluation_batch_seconds": evaluation_seconds,
        "memory": {
            "baseline": baseline,
            "maximum_sampled": {
                key: max(row[key] for row in memory_samples)
                for key in memory_samples[0]
            },
            "released": released,
            "resettable_peak_supported": False,
        },
        "environment": current_frontier_environment(),
        "session_state": {"start": state_start, "end": state_end},
        "loss_values_recorded": False,
    }
    report["worker_sha256"] = canonical_sha256(report)
    _publish(output_path, json_bytes(report))


def _parent() -> None:
    commit, _plan = _repository_context()
    if any(path.exists() for path in (ACTIVE_PATH, REPORT_PATH, OUTPUT_PATH)):
        raise FileExistsError("BPE quality feasibility namespace is not empty")
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    if any(WORKER_ROOT.iterdir()):
        raise FileExistsError("BPE quality feasibility worker namespace is not empty")
    _publish(
        ACTIVE_PATH,
        json_bytes(
            {"git_commit": commit, "plan_artifact_sha256": hash_file(PLAN_PATH)}
        ),
    )
    for role in QUALITY_ROLES:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                role,
            ],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        )
    workers = {}
    for role in QUALITY_ROLES:
        path = WORKER_ROOT / f"{role}.json"
        row = read_json(path)
        unsigned = dict(row)
        expected = unsigned.pop("worker_sha256", None)
        if (
            canonical_sha256(unsigned) != expected
            or row.get("role") != role
            or row.get("git_commit") != commit
            or row.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
            or row.get("complete") is not True
        ):
            raise ValueError("BPE quality feasibility worker evidence differs")
        workers[role] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        }
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during BPE quality feasibility")
    report = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_feasibility_report_v1",
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
