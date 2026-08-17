#!/usr/bin/env python3
"""Independently reconstruct and evaluate the sealed Length-Gain vocabulary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time

from tokenizers import Tokenizer

from fixed_byte_tokenizer import build_fixed_byte_tokenizer
from length_gain import (
    evaluate_length_gain_opportunity,
    length_gain_decision,
    train_length_gain_vocabulary,
    training_public_metadata,
)
from length_gain_opportunity_protocol import (
    BATCH_SIZE,
    BPE_ROLE,
    BPE_TOKENIZER_PATH,
    ENCODE_REPETITIONS,
    LONGEST_ROLE,
    MAXIMUM_PIECE_BYTES,
    MAXIMUM_TOKEN_ARITY,
    MINIMUM_REDUCTION,
    MINIMUM_ROLE,
    PIECES_PATH,
    PLAN_PATH,
    PRIMARY_ORDER,
    PROTOCOL_ID,
    RESULT_PATH,
    ROLE_ORDER,
    ROOT,
    SCORE_KIND,
    SUMMARY_ACTIVE_PATH,
    VOCABULARY_SIZE,
    WARMUP_CASES,
    WORKER_PATH,
    canonical_sha256,
    current_environment,
    hash_file,
    implementation_identity,
    json_bytes,
    load_pieces,
    read_json,
    reconstruct_all_inputs,
    validate_plan,
)


def _git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_stage = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    stage = Path(raw_stage)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _validate_worker(worker: dict, plan: dict) -> None:
    expected = {
        "complete",
        "environment",
        "git_commit",
        "implementation_sha256",
        "kind",
        "pieces_artifact_sha256",
        "plan_artifact_sha256",
        "plan_sha256",
        "schema_version",
        "train_stream",
        "training",
        "worker_seconds",
        "worker_sha256",
    }
    if set(worker) != expected:
        raise ValueError("length-gain worker schema differs")
    unsigned = dict(worker)
    unsigned.pop("worker_sha256")
    if (
        worker["schema_version"] != 1
        or worker["kind"] != "length_gain_opportunity_worker_v1"
        or worker["complete"] is not True
        or canonical_sha256(unsigned) != worker["worker_sha256"]
        or worker["plan_sha256"] != plan["plan_sha256"]
        or worker["plan_artifact_sha256"] != hash_file(PLAN_PATH)
        or worker["environment"] != current_environment()
        or worker["implementation_sha256"] != implementation_identity()
        or worker["pieces_artifact_sha256"] != hash_file(PIECES_PATH)
    ):
        raise ValueError("length-gain worker identity differs")


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("length-gain summary requires a clean worktree")
    if RESULT_PATH.exists() or _git("log", "--all", "--format=%H", "--", str(RESULT_PATH.relative_to(ROOT))):
        raise RuntimeError("length-gain result already exists or has history")
    if SUMMARY_ACTIVE_PATH.exists():
        raise RuntimeError("length-gain summary active marker already exists")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    worker = read_json(WORKER_PATH)
    _validate_worker(worker, plan)
    base_commit = _git("rev-parse", "HEAD")
    identities_before = {
        "pieces": hash_file(PIECES_PATH),
        "plan": hash_file(PLAN_PATH),
        "worker": hash_file(WORKER_PATH),
    }
    _publish(
        SUMMARY_ACTIVE_PATH,
        json_bytes(
            {
                "kind": "length_gain_summary_active_v1",
                "git_commit": base_commit,
                "plan_sha256": plan["plan_sha256"],
                "worker_sha256": worker["worker_sha256"],
            }
        ),
    )

    train_raw, calibration_raw, prompts, continuations, shared_inputs = reconstruct_all_inputs()
    started = time.perf_counter()
    replay = train_length_gain_vocabulary(
        train_raw,
        vocabulary_size=VOCABULARY_SIZE,
        batch_size=BATCH_SIZE,
        maximum_token_arity=MAXIMUM_TOKEN_ARITY,
        maximum_piece_bytes=MAXIMUM_PIECE_BYTES,
        score_kind=SCORE_KIND,
    )
    replay_metadata = training_public_metadata(replay)
    stored_pieces = load_pieces(PIECES_PATH)
    if stored_pieces != replay.pieces or worker["training"] != replay_metadata:
        raise RuntimeError("length-gain independent construction differs")

    tokenizers = {
        BPE_ROLE: Tokenizer.from_file(str(BPE_TOKENIZER_PATH)),
        LONGEST_ROLE: build_fixed_byte_tokenizer(
            replay.pieces,
            segmentation="leftmost_longest",
            maximum_piece_bytes=MAXIMUM_PIECE_BYTES,
        ),
        MINIMUM_ROLE: build_fixed_byte_tokenizer(
            replay.pieces,
            segmentation="minimum_token_dp",
            maximum_piece_bytes=MAXIMUM_PIECE_BYTES,
        ),
    }
    metrics = {
        role: evaluate_length_gain_opportunity(
            role=role,
            tokenizer=tokenizers[role],
            calibration_raw=calibration_raw,
            prompts=prompts,
            continuations=continuations,
            encode_repetitions=ENCODE_REPETITIONS,
        ).to_dict()
        for role in ROLE_ORDER
    }
    decision = length_gain_decision(
        metrics,
        baseline_role=BPE_ROLE,
        primary_order=PRIMARY_ORDER,
        warmup_cases=WARMUP_CASES,
        minimum_reduction=MINIMUM_REDUCTION,
    )
    result = {
        "schema_version": 1,
        "kind": "length_gain_opportunity_result_v1",
        "protocol_id": PROTOCOL_ID,
        "plan_artifact_sha256": identities_before["plan"],
        "plan_sha256": plan["plan_sha256"],
        "worker_artifact_sha256": identities_before["worker"],
        "worker_sha256": worker["worker_sha256"],
        "pieces_artifact_sha256": identities_before["pieces"],
        "shared_inputs": shared_inputs,
        "training": replay_metadata,
        "metrics_by_role": metrics,
        "decision": decision,
        "independent_full_construction_replay": True,
        "replay_and_evaluation_seconds": time.perf_counter() - started,
        "claim_boundary": plan["claim_boundary"],
        "complete": True,
    }
    result["summary_sha256"] = canonical_sha256(result)
    if (
        _git("rev-parse", "HEAD") != base_commit
        or _git("status", "--porcelain")
        or identities_before
        != {
            "pieces": hash_file(PIECES_PATH),
            "plan": hash_file(PLAN_PATH),
            "worker": hash_file(WORKER_PATH),
        }
    ):
        raise RuntimeError("repository or evidence changed during length-gain summary")
    _publish(RESULT_PATH, json_bytes(result))
    SUMMARY_ACTIVE_PATH.unlink()
    print(f"result={RESULT_PATH.relative_to(ROOT)}")
    print(f"status={decision['status']}")
    print(f"selected_role={decision['selected_role']}")
    for role in PRIMARY_ORDER:
        comparison = decision["comparisons"][role]
        print(
            role,
            f"calibration_reduction={comparison['calibration_token_reduction']:.6%}",
            f"continuation_reduction={comparison['continuation_token_reduction']:.6%}",
        )


if __name__ == "__main__":
    main()
