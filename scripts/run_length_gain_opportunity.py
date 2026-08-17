#!/usr/bin/env python3
"""Construct the sealed Length-Gain vocabulary from train bytes only."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import time

from length_gain import train_length_gain_vocabulary, training_public_metadata
from length_gain_opportunity_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    BATCH_SIZE,
    MAXIMUM_PIECE_BYTES,
    MAXIMUM_TOKEN_ARITY,
    PIECES_PATH,
    PLAN_PATH,
    RESULT_PATH,
    ROOT,
    SCORE_KIND,
    VOCABULARY_SIZE,
    WORKER_PATH,
    canonical_sha256,
    current_environment,
    hash_file,
    implementation_identity,
    json_bytes,
    load_pieces,
    read_json,
    reconstruct_train_stream,
    serialize_pieces,
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


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("length-gain worker requires a clean worktree")
    if RESULT_PATH.exists():
        raise RuntimeError("length-gain result already exists")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    plan_artifact_sha256 = hash_file(PLAN_PATH)
    base_commit = _git("rev-parse", "HEAD")
    if WORKER_PATH.exists() or PIECES_PATH.exists() or ACTIVE_PATH.exists():
        raise RuntimeError("length-gain worker namespace is not empty")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    active = {
        "kind": "length_gain_worker_active_v1",
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "git_commit": base_commit,
    }
    _publish(ACTIVE_PATH, json_bytes(active))
    started = time.perf_counter()
    train_raw, train_metadata = reconstruct_train_stream()
    result = train_length_gain_vocabulary(
        train_raw,
        vocabulary_size=VOCABULARY_SIZE,
        batch_size=BATCH_SIZE,
        maximum_token_arity=MAXIMUM_TOKEN_ARITY,
        maximum_piece_bytes=MAXIMUM_PIECE_BYTES,
        score_kind=SCORE_KIND,
    )
    # Serialize to a private stage and completely reload it before publication.
    fd, raw_stage = tempfile.mkstemp(prefix=".pieces.", suffix=".npz", dir=ARTIFACT_ROOT)
    os.close(fd)
    stage = Path(raw_stage)
    try:
        serialize_pieces(stage, result.pieces)
        if load_pieces(stage) != result.pieces:
            raise RuntimeError("length-gain staged pieces differ")
        pieces_bytes = stage.read_bytes()
    finally:
        stage.unlink(missing_ok=True)
    worker = {
        "schema_version": 1,
        "kind": "length_gain_opportunity_worker_v1",
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "git_commit": base_commit,
        "environment": current_environment(),
        "implementation_sha256": implementation_identity(),
        "train_stream": train_metadata,
        "training": training_public_metadata(result),
        "pieces_artifact_sha256": __import__("hashlib").sha256(pieces_bytes).hexdigest(),
        "worker_seconds": time.perf_counter() - started,
        "complete": True,
    }
    worker["worker_sha256"] = canonical_sha256(worker)
    if _git("rev-parse", "HEAD") != base_commit or _git("status", "--porcelain"):
        raise RuntimeError("repository changed during length-gain construction")
    _publish(PIECES_PATH, pieces_bytes)
    _publish(WORKER_PATH, json_bytes(worker))
    ACTIVE_PATH.unlink()
    print(f"worker={WORKER_PATH.relative_to(ROOT)}")
    print(f"seconds={worker['worker_seconds']:.3f}")
    print(f"train_tokens={result.final_token_count}")


if __name__ == "__main__":
    main()
