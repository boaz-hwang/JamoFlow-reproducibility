#!/usr/bin/env python3
"""Build the train-only EXAONE n-gram table and metric-free timing cases."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from exaone_retrieval_data import (
    ACTIVE_PATH,
    CASES_PATH,
    EVALUATION_EXACT_SET_DOMAIN,
    EVALUATION_NORMALIZED_SET_DOMAIN,
    EVALUATION_SEAL_PATH,
    EVALUATION_SOURCE_PATH,
    PLAN_PATH,
    PRIMARY_MODEL,
    ROOT,
    SEAL_PATH,
    TABLE_PATH,
    TOKENIZER_RUNTIME_FILES,
    TRAIN_DOCUMENT_BYTE_BUDGET,
    TRAIN_EXACT_SET_DOMAIN,
    TRAIN_NORMALIZED_SET_DOMAIN,
    TRAIN_SEAL_PATH,
    TRAIN_SOURCE_PATH,
    VERIFICATION_PATH,
    VOCABULARY_SIZE,
    array_sha256,
    build_case_arrays,
    build_compact_backoff_table,
    build_seal,
    canonical_bytes,
    compatibility_projection,
    digest_set_commitment,
    hash_file,
    npz_bytes,
    read_plan,
    table_from_arrays,
    token_ids_uint32,
    validate_case_arrays,
    validate_seal,
    validate_tokenizer_runtime_identity,
)
from hplt3_fresh_adaptation_v2_protocol import (
    validate_seal_envelope as validate_train_seal_envelope,
)
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_tokenizer

from jamoflow.corpus import load_records, partition_records
from jamoflow.hplt3_final_test import (
    normalized_record_digest,
)
from jamoflow.hplt3_final_test import (
    validate_seal_envelope as validate_evaluation_seal_envelope,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _publish(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_seal_envelope(path: Path, *, role: str) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if role == "train":
        validate_train_seal_envelope(envelope)
    elif role == "evaluation":
        validate_evaluation_seal_envelope(envelope)
    else:
        raise ValueError("EXAONE source seal role differs")
    return envelope["payload"]


def _verified_records() -> tuple[list, list]:
    train_seal = _read_seal_envelope(TRAIN_SEAL_PATH, role="train")
    evaluation_seal = _read_seal_envelope(
        EVALUATION_SEAL_PATH, role="evaluation"
    )
    if (
        hash_file(TRAIN_SOURCE_PATH) != train_seal["output"]["sha256"]
        or TRAIN_SOURCE_PATH.stat().st_size != train_seal["output"]["bytes"]
        or train_seal["splits"]["train"]["stream_bytes"]
        != TRAIN_DOCUMENT_BYTE_BUDGET
        or train_seal["exclusions"]["final_exact_count"] != 1_482
        or hash_file(EVALUATION_SOURCE_PATH)
        != evaluation_seal["output"]["full_jsonl_sha256"]
        or EVALUATION_SOURCE_PATH.stat().st_size
        != evaluation_seal["output"]["full_jsonl_bytes"]
        or evaluation_seal["selection"]["selected_document_count"] != 1_482
        or evaluation_seal["selection"]["intersection_count"] != 0
        or evaluation_seal["selection"]["normalized_intersection_count"] != 0
    ):
        raise ValueError("EXAONE retrieval source/seal identity differs")
    train_all = load_records(
        [TRAIN_SOURCE_PATH], corpus_format="jsonl", text_field="text", deduplicate=True
    )
    train_records = partition_records(train_all)["train"]
    evaluation_records = load_records(
        [EVALUATION_SOURCE_PATH],
        corpus_format="jsonl",
        text_field="text",
        deduplicate=True,
    )
    if (
        len(train_records) != train_seal["splits"]["train"]["selected_document_count"]
        or len(evaluation_records)
        != evaluation_seal["selection"]["selected_document_count"]
        or {record.record_id for record in train_records}
        & {record.record_id for record in evaluation_records}
    ):
        raise ValueError("EXAONE retrieval train/evaluation records differ")
    return train_records, evaluation_records


def _disjointness_inventory(train_records: list, evaluation_records: list) -> dict:
    train_exact = {hashlib.sha256(record.raw).digest() for record in train_records}
    evaluation_exact = {
        hashlib.sha256(record.raw).digest() for record in evaluation_records
    }
    train_normalized = {
        normalized_record_digest(record.text)
        for record in train_records
        if record.text is not None
    }
    evaluation_normalized = {
        normalized_record_digest(record.text)
        for record in evaluation_records
        if record.text is not None
    }
    if (
        len(train_exact) != len(train_records)
        or len(evaluation_exact) != len(evaluation_records)
        or len(train_normalized) != len(train_records)
        or len(evaluation_normalized) != len(evaluation_records)
        or train_exact & evaluation_exact
        or train_normalized & evaluation_normalized
    ):
        raise ValueError("EXAONE retrieval normalized train/evaluation overlap")
    return {
        "evaluation_exact_commitment_sha256": digest_set_commitment(
            tuple(evaluation_exact), domain=EVALUATION_EXACT_SET_DOMAIN
        ),
        "evaluation_normalized_commitment_sha256": digest_set_commitment(
            tuple(evaluation_normalized), domain=EVALUATION_NORMALIZED_SET_DOMAIN
        ),
        "normalized_document_algorithm": (
            "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
        ),
        "selected_train_exact_commitment_sha256": digest_set_commitment(
            tuple(train_exact), domain=TRAIN_EXACT_SET_DOMAIN
        ),
        "selected_train_normalized_commitment_sha256": digest_set_commitment(
            tuple(train_normalized), domain=TRAIN_NORMALIZED_SET_DOMAIN
        ),
        "train_evaluation_exact_intersection_count": 0,
        "train_evaluation_normalized_intersection_count": 0,
    }


def _full_document_prefix(records: list) -> tuple[list, int]:
    selected = []
    serialized_bytes = 0
    for record in records:
        if record.text is None:
            continue
        increment = len(record.raw) + (1 if selected else 0)
        if serialized_bytes + increment > TRAIN_DOCUMENT_BYTE_BUDGET:
            break
        selected.append(record)
        serialized_bytes += increment
    if not selected or serialized_bytes <= 0:
        raise ValueError("EXAONE retrieval train document prefix is empty")
    return selected, serialized_bytes


def _encode_train_documents(records: list, tokenizer) -> tuple[np.memmap, str, dict]:
    ARTIFACT_ROOT = TABLE_PATH.parent
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix="exaone-train-tokens-", suffix=".uint32", dir=ARTIFACT_ROOT
    )
    os.close(descriptor)
    token_count = 0
    newline = token_ids_uint32(
        tokenizer.encode("\n", add_special_tokens=False), label="newline"
    )
    try:
        with open(temporary_path, "wb") as handle:
            for index, record in enumerate(records):
                if index:
                    newline.tofile(handle)
                    token_count += len(newline)
                values = token_ids_uint32(
                    tokenizer.encode(record.text, add_special_tokens=False),
                    label="train document",
                )
                values.tofile(handle)
                token_count += len(values)
                if token_count >= 2**32:
                    raise OverflowError("EXAONE train token count exceeds uint32")
            handle.flush()
            os.fsync(handle.fileno())
        memory = np.memmap(temporary_path, dtype=np.uint32, mode="r", shape=(token_count,))
        inventory = {
            "document_count": len(records),
            "newline_token_count": len(newline),
            "token_count": token_count,
            "token_ids_sha256": array_sha256(memory),
        }
        return memory, temporary_path, inventory
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _load_tokenizer(plan_compatibility: dict):
    current_compatibility = compatibility_projection()
    if current_compatibility != plan_compatibility:
        raise ValueError("EXAONE compatibility result changed before tokenization")
    snapshot = Path(
        snapshot_download(
            repo_id=PRIMARY_MODEL["repo_id"],
            revision=PRIMARY_MODEL["revision"],
            allow_patterns=["*.json", "*.py", "*.txt"],
            local_files_only=True,
        )
    )
    observed_files = {
        name: {
            "bytes": (snapshot / name).stat().st_size,
            "sha256": hash_file(snapshot / name),
        }
        for name in TOKENIZER_RUNTIME_FILES
    }
    if observed_files != plan_compatibility["model_files"]:
        raise ValueError("EXAONE tokenizer snapshot differs from V4 compatibility")
    tokenizer = load_tokenizer(
        snapshot, tokenizer_config_extra={"trust_remote_code": True}
    )
    if int(tokenizer.vocab_size) != VOCABULARY_SIZE:
        raise ValueError("EXAONE retrieval tokenizer vocabulary differs")
    identity = {
        "compatibility_result_summary_sha256": plan_compatibility[
            "result_summary_sha256"
        ],
        "files": observed_files,
        "repo_id": PRIMARY_MODEL["repo_id"],
        "revision": PRIMARY_MODEL["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "vocabulary_size": int(tokenizer.vocab_size),
    }
    validate_tokenizer_runtime_identity(
        identity, compatibility=plan_compatibility
    )
    return tokenizer, identity


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE retrieval data build requires a clean worktree")
    for path in (ACTIVE_PATH, TABLE_PATH, CASES_PATH, SEAL_PATH, VERIFICATION_PATH):
        if path.exists():
            raise FileExistsError(f"EXAONE retrieval data namespace is not empty: {path}")
    history = _git("log", "--all", "--format=%H", "--", SEAL_PATH.relative_to(ROOT).as_posix())
    if history:
        raise FileExistsError("EXAONE retrieval data seal was already published")
    plan = read_plan(verify_derived=True)
    plan_blob = subprocess.check_output(
        ("git", "show", f"HEAD:{PLAN_PATH.relative_to(ROOT).as_posix()}"), cwd=ROOT
    )
    if plan_blob != PLAN_PATH.read_bytes():
        raise ValueError("EXAONE retrieval data plan is not the exact HEAD blob")
    commit = _git("rev-parse", "HEAD")
    _publish(
        ACTIVE_PATH,
        canonical_bytes(
            {
                "builder_git_commit": commit,
                "plan_artifact_sha256": hash_file(PLAN_PATH),
                "plan_sha256": plan["plan_sha256"],
            }
        ),
        0o600,
    )
    token_memory = None
    temporary_path = None
    try:
        train_records, evaluation_records = _verified_records()
        selected_train, serialized_bytes = _full_document_prefix(train_records)
        tokenizer, tokenizer_runtime = _load_tokenizer(plan["compatibility"])
        token_memory, temporary_path, inventory = _encode_train_documents(
            selected_train, tokenizer
        )
        inventory.update(
            {
                "available_train_document_count": len(train_records),
                "evaluation_document_count": len(evaluation_records),
                "full_document_serialized_bytes": serialized_bytes,
                "train_document_byte_budget": TRAIN_DOCUMENT_BYTE_BUDGET,
                **_disjointness_inventory(selected_train, evaluation_records),
            }
        )
        table = build_compact_backoff_table(token_memory)
        table_arrays = table.to_arrays()
        table_from_arrays(table_arrays)
        rank_key = bytes.fromhex(
            plan["data_contract"]["case_selection"]["case_rank_key_hex"]
        )
        case_arrays, case_report = build_case_arrays(
            evaluation_records, tokenizer, rank_key=rank_key
        )
        validate_case_arrays(case_arrays, rank_key=rank_key)
        table_payload = npz_bytes(table_arrays)
        case_payload = npz_bytes(case_arrays)
        if (
            _git("rev-parse", "HEAD") != commit
            or _git("status", "--porcelain", "--untracked-files=all")
        ):
            raise RuntimeError("repository changed during EXAONE retrieval data build")
        _publish(TABLE_PATH, table_payload, 0o600)
        _publish(CASES_PATH, case_payload, 0o600)
        seal = build_seal(
            plan=plan,
            builder_git_commit=commit,
            table=table,
            table_arrays=table_arrays,
            case_arrays=case_arrays,
            case_report=case_report,
            tokenizer_runtime=tokenizer_runtime,
            training_inventory=inventory,
        )
        validate_seal(seal, plan=plan, verify_artifacts=True)
        _publish(SEAL_PATH, canonical_bytes(seal), 0o644)
        ACTIVE_PATH.unlink()
        print(f"table_entries={table.entry_count}")
        print(f"train_tokens={inventory['token_count']}")
        print(f"eligible_cases={case_report['eligible_document_count']}")
        print(f"seal_sha256={seal['seal_sha256']}")
        print("commit the tracked data seal before resource calibration")
    finally:
        if token_memory is not None:
            del token_memory
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


if __name__ == "__main__":
    main()
