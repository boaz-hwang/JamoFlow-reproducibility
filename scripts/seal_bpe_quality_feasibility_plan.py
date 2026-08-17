#!/usr/bin/env python3
"""Seal BPE quality-frontier feasibility before observing train-step timing."""

from __future__ import annotations

import os
import subprocess

from bpe_quality_feasibility_core import (
    CALIBRATION_BYTES,
    CAMPAIGN_HOUR_LIMIT,
    CANDIDATE_TRAIN_BYTE_BUDGETS,
    DRIVER_MEMORY_FRACTION_LIMIT,
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
)
from bpe_quality_feasibility_protocol import (
    IMPLEMENTATION_PATHS,
    INTEGRITY_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    SOURCE_PATH,
    SYSTEMS_RESULT_PATH,
    TOKENIZER_PATHS,
    canonical_sha256,
    current_frontier_environment,
    hash_file,
    json_bytes,
)
from token_frontier_core import FRONTIER_SPECS, parse_role
from token_frontier_protocol import load_tokenizers

from jamoflow.neural_data import build_neural_stream


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command("git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if history:
        raise FileExistsError(f"BPE quality feasibility path has Git history: {path}")


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("BPE quality feasibility plan requires a clean root")
    _never_published(PLAN_PATH)
    _never_published(OUTPUT_PATH)
    commit = _command("git", "rev-parse", "HEAD")
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
    if len(train.data) != TRAIN_BYTES or len(calibration.data) != CALIBRATION_BYTES:
        raise ValueError("BPE quality feasibility source stream is incomplete")
    tokenizers = load_tokenizers()
    inventories = {}
    for role in QUALITY_ROLES:
        vocabulary, _ = parse_role(role)
        tokenizer, token_bytes = tokenizers[vocabulary]
        split_rows = {}
        for split, raw, first_count in (
            (
                "train",
                train.data,
                EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH,
            ),
            (
                "calibration",
                calibration.data,
                EVALUATION_BATCH_BY_VOCABULARY[vocabulary] * SEQUENCE_LENGTH,
            ),
        ):
            inventory, memory, temporary = encode_stream_to_memmap(
                raw,
                tokenizer,
                token_bytes,
                first_batch_token_count=first_count,
            )
            del memory
            os.unlink(temporary)
            split_rows[split] = inventory.to_dict()
        inventories[role] = split_rows
    dependencies = {
        "git_commit_before_plan": commit,
        "integrity": {
            "path": str(INTEGRITY_PATH.relative_to(ROOT)),
            "sha256": hash_file(INTEGRITY_PATH),
        },
        "source": {
            "path": str(SOURCE_PATH.relative_to(ROOT)),
            "sha256": hash_file(SOURCE_PATH),
        },
        "systems_result": {
            "path": str(SYSTEMS_RESULT_PATH.relative_to(ROOT)),
            "sha256": hash_file(SYSTEMS_RESULT_PATH),
        },
        "tokenizers": {
            str(size): {
                "path": str(TOKENIZER_PATHS[size].relative_to(ROOT)),
                "sha256": hash_file(TOKENIZER_PATHS[size]),
            }
            for size, _ in map(parse_role, QUALITY_ROLES)
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_feasibility_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_training_step_timing",
        "dependencies": dependencies,
        "environment": current_frontier_environment(),
        "roles": list(QUALITY_ROLES),
        "model_specs": {
            role: FRONTIER_SPECS[role].to_dict() for role in QUALITY_ROLES
        },
        "inventories": inventories,
        "feasibility": {
            "calibration_bytes": CALIBRATION_BYTES,
            "campaign_hour_limit": CAMPAIGN_HOUR_LIMIT,
            "candidate_train_byte_budgets": list(CANDIDATE_TRAIN_BYTE_BUDGETS),
            "driver_memory_fraction_limit": DRIVER_MEMORY_FRACTION_LIMIT,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "evaluation_batch_by_vocabulary": {
                str(key): value
                for key, value in EVALUATION_BATCH_BY_VOCABULARY.items()
            },
            "measured_effective_steps": MEASURED_EFFECTIVE_STEPS,
            "measured_evaluation_batches": MEASURED_EVALUATION_BATCHES,
            "sequence_length": SEQUENCE_LENGTH,
            "train_bytes": TRAIN_BYTES,
            "train_microbatch_by_vocabulary": {
                str(key): value
                for key, value in TRAIN_MICROBATCH_BY_VOCABULARY.items()
            },
            "warmup_effective_steps": WARMUP_EFFECTIVE_STEPS,
            "warmup_evaluation_batches": WARMUP_EVALUATION_BATCHES,
        },
        "implementation_sha256": {
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        "claim_boundary": {
            "actual_mps_training_and_evaluation_steps": True,
            "model_quality_measured": False,
            "projection_not_full_training": True,
            "random_weights": True,
            "selected_budget_uses_only_time_and_memory": True,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PLAN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
