#!/usr/bin/env python3
"""Build the fixed train-only compact 16K token n-gram draft table."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tokenizers
from bpe_quality_feasibility_core import encode_stream_to_memmap
from compositional_head_preflight_protocol import ROOT, hash_file, load_tokenizers
from fresh_vocabulary_16k_protocol import (
    FRESH_SEAL_PATH,
    FRESH_SOURCE_PATH,
    TRAIN_BYTES,
    canonical_sha256,
    json_bytes,
    read_json,
    verified_fresh_streams,
)
from fresh_vocabulary_16k_protocol import PLAN_PATH as QUALITY_PLAN_PATH
from fresh_vocabulary_16k_retrieval_core import (
    MAXIMUM_TABLE_ENTRIES,
    TABLE_ARRAY_NAMES,
    VOCABULARY_SIZE,
    array_sha256,
    build_compact_backoff_table,
    table_report,
)
from token_frontier_protocol import TOKENIZER_PATHS

PROTOCOL_ID = "jamoflow-fresh-vocabulary-16k-retrieval-table-v1"
ARTIFACT_ROOT = ROOT / "artifacts/fresh-vocabulary-16k-retrieval-table-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
TABLE_PATH = ARTIFACT_ROOT / "compact-token-ngram.npz"
SEAL_PATH = ROOT / "data/seals/fresh-vocabulary-16k-retrieval-table-v1.json"

IMPLEMENTATION_PATHS = (
    "docs/165-retrieval-draft-literature-audit-and-fail-fast-direction.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/build_fresh_vocabulary_16k_retrieval_table.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/fresh_vocabulary_16k_protocol.py",
    "scripts/fresh_vocabulary_16k_retrieval_core.py",
    "scripts/token_frontier_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_fresh_vocabulary_16k_retrieval_core.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


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


def _require_clean_unused_namespace() -> str:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("retrieval-table build requires a clean worktree")
    if any(path.exists() for path in (ACTIVE_PATH, TABLE_PATH, SEAL_PATH)):
        raise FileExistsError("retrieval-table namespace is not empty")
    history = _git("log", "--all", "--format=%H", "--", str(SEAL_PATH.relative_to(ROOT)))
    if history:
        raise RuntimeError("retrieval-table seal was already published")
    return _git("rev-parse", "HEAD")


def _validate_training_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    quality = read_json(QUALITY_PLAN_PATH)
    expected = quality["inventories"][str(VOCABULARY_SIZE)]["train_tokens"]
    actual = inventory.to_dict()
    if actual != expected:
        raise ValueError("retrieval-table train token inventory differs")
    return actual


def main() -> None:
    commit = _require_clean_unused_namespace()
    active = json_bytes(
        {
            "protocol_id": PROTOCOL_ID,
            "git_commit": commit,
            "source_sha256": hash_file(FRESH_SOURCE_PATH),
            "tokenizer_sha256": hash_file(TOKENIZER_PATHS[VOCABULARY_SIZE]),
        }
    )
    _publish(ACTIVE_PATH, active)
    stream = verified_fresh_streams()["train"]
    tokenizer, token_bytes = load_tokenizers()[VOCABULARY_SIZE]
    inventory, memory, temporary_path = encode_stream_to_memmap(
        stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=16_384,
    )
    try:
        inventory_payload = _validate_training_inventory(inventory)
        table = build_compact_backoff_table(
            memory,
            vocabulary_size=VOCABULARY_SIZE,
            maximum_entries=MAXIMUM_TABLE_ENTRIES,
        )
        arrays = table.to_arrays()
        if set(arrays) != set(TABLE_ARRAY_NAMES):
            raise AssertionError("retrieval-table output array set differs")
        table_payload = _npz_bytes(arrays)
        array_descriptors = {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": array_sha256(value),
            }
            for name, value in arrays.items()
        }
        implementation = {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}
        if len(implementation) != len(IMPLEMENTATION_PATHS):
            raise AssertionError("retrieval-table implementation paths duplicate")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "fresh_vocabulary_16k_retrieval_table_seal_v1",
            "protocol_id": PROTOCOL_ID,
            "complete": True,
            "git_commit": commit,
            "source": {
                "path": str(FRESH_SOURCE_PATH.relative_to(ROOT)),
                "sha256": hash_file(FRESH_SOURCE_PATH),
                "seal_path": str(FRESH_SEAL_PATH.relative_to(ROOT)),
                "seal_sha256": hash_file(FRESH_SEAL_PATH),
                "split": "train",
                "raw_stream_bytes": TRAIN_BYTES,
                "stream_sha256": hashlib.sha256(stream.data).hexdigest(),
            },
            "tokenizer": {
                "path": str(TOKENIZER_PATHS[VOCABULARY_SIZE].relative_to(ROOT)),
                "sha256": hash_file(TOKENIZER_PATHS[VOCABULARY_SIZE]),
                "vocabulary_size": VOCABULARY_SIZE,
            },
            "training_token_inventory": inventory_payload,
            "table_contract": table_report(table),
            "table_artifact": {
                "path": str(TABLE_PATH.relative_to(ROOT)),
                "sha256": hashlib.sha256(table_payload).hexdigest(),
                "bytes": len(table_payload),
                "arrays": array_descriptors,
            },
            "implementation_sha256": implementation,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "tokenizers": tokenizers.__version__,
            },
            "result_inputs": {
                "train_split_tokens": True,
                "calibration_tokens": False,
                "historical_test_metrics": False,
                "sealed_final_test": False,
                "model_checkpoint_or_logits": False,
                "latency": False,
            },
        }
        payload["seal_sha256"] = canonical_sha256(payload)
        if _git("rev-parse", "HEAD") != commit or _git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("repository changed during retrieval-table build")
        _publish(TABLE_PATH, table_payload)
        _publish(SEAL_PATH, json_bytes(payload), mode=0o644)
        ACTIVE_PATH.unlink()
        print(f"retrieval table entries: {table.entry_count}")
        print(f"retrieval table artifact sha256: {payload['table_artifact']['sha256']}")
        print(f"retrieval table seal sha256: {payload['seal_sha256']}")
        print("commit the tracked table seal before building the timing protocol")
    finally:
        del memory
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


if __name__ == "__main__":
    main()
