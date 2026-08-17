#!/usr/bin/env python3
"""Train deterministic Byte-Unigram and measure the sealed same-2K roles."""

from __future__ import annotations

import io
import json
import os
import subprocess

import numpy as np
from tokenizers import Tokenizer

from same2k_opportunity_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    PIECES_PATH,
    PLAN_PATH,
    ROOT,
    SENTENCEPIECE_MODEL_PATH,
    SOURCE_PATH,
    TRAINED_TOKENIZER_PATH,
    WORKER_PATH,
    hash_file,
    json_bytes,
    load_bpe_tokenizer,
    read_json,
    reconstruct_shared_inputs,
    validate_plan,
)
from byte_unigram import train_deterministic_byte_unigram
from jamoflow.corpus import load_records, partition_records
from same2k_opportunity import (
    BPE_ROLE,
    LONGEST_MATCH_ROLE,
    MINIMUM_TOKEN_ROLE,
    SCORED_UNIGRAM_ROLE,
    evaluate_tokenizer_opportunity,
)
from fixed_byte_tokenizer import (
    build_fixed_byte_tokenizer,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _save_piece_artifact(pieces: tuple[bytes, ...], scores: tuple[float, ...]) -> bytes:
    offsets = np.empty(len(pieces), dtype=np.int64)
    lengths = np.asarray([len(piece) for piece in pieces], dtype=np.int64)
    raw = b"".join(pieces)
    cursor = 0
    for index, piece in enumerate(pieces):
        offsets[index] = cursor
        cursor += len(piece)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        lengths=lengths,
        offsets=offsets,
        raw=np.frombuffer(raw, dtype=np.uint8),
        scores=np.asarray(scores, dtype=np.float64),
    )
    return buffer.getvalue()


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("same-2K worker requires a clean root")
    if any(path.exists() for path in (WORKER_PATH, TRAINED_TOKENIZER_PATH, SENTENCEPIECE_MODEL_PATH, PIECES_PATH)):
        raise FileExistsError("same-2K worker evidence already exists")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    run_commit = _command("git", "rev-parse", "HEAD")
    if (
        _command("git", "show", f"HEAD:{PLAN_PATH.relative_to(ROOT)}")
        != PLAN_PATH.read_text(encoding="utf-8").rstrip("\n")
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", plan["git_commit_before_plan"], run_commit],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("same-2K plan is not an exact tracked descendant")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    active = json_bytes({"plan_sha256": plan["plan_sha256"], "run_git_commit": run_commit})
    with ACTIVE_PATH.open("xb") as output:
        output.write(active)

    records = load_records([SOURCE_PATH], corpus_format="jsonl", text_field="text", deduplicate=True)
    train = tuple(record.text for record in partition_records(records)["train"] if record.text is not None)
    tokenizer, pieces, scores, model_proto, metadata = train_deterministic_byte_unigram(train)
    token_json = tokenizer.to_str(pretty=False).encode("utf-8")
    deployable_scored_tokenizer = Tokenizer.from_str(token_json.decode("utf-8"))
    piece_artifact = _save_piece_artifact(pieces, scores)
    # Construct all runtimes before publishing anything, so failures do not
    # leave an unrecoverable partial evidence tuple.
    tokenizers = {
        BPE_ROLE: load_bpe_tokenizer(),
        SCORED_UNIGRAM_ROLE: deployable_scored_tokenizer,
        LONGEST_MATCH_ROLE: build_fixed_byte_tokenizer(pieces, segmentation="leftmost_longest"),
        MINIMUM_TOKEN_ROLE: build_fixed_byte_tokenizer(pieces, segmentation="minimum_token_dp"),
    }
    calibration, prompts, continuations, _ = reconstruct_shared_inputs()
    metrics = {
        role: evaluate_tokenizer_opportunity(
            role=role,
            tokenizer=value,
            calibration_raw=calibration,
            prompts=prompts,
            continuations=continuations,
        ).to_dict()
        for role, value in tokenizers.items()
    }
    worker = {
        "schema_version": 1,
        "kind": "same2k_generic_opportunity_worker_v6",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "run_git_commit": run_commit,
        "training_metadata": metadata.to_dict(),
        "artifacts": {},
        "metrics_by_role": metrics,
        "complete": True,
    }
    for path, content in (
        (TRAINED_TOKENIZER_PATH, token_json),
        (SENTENCEPIECE_MODEL_PATH, model_proto),
        (PIECES_PATH, piece_artifact),
    ):
        with path.open("xb") as output:
            output.write(content)
        worker["artifacts"][str(path.relative_to(ROOT))] = hash_file(path)
    with WORKER_PATH.open("xb") as output:
        output.write(json_bytes(worker))
    if _command("git", "rev-parse", "HEAD") != run_commit or _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("same-2K repository changed during worker execution")
    ACTIVE_PATH.unlink()
    print(f"completed {WORKER_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
