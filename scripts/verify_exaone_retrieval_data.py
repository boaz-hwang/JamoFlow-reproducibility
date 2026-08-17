#!/usr/bin/env python3
"""Independently rebuild and verify the sealed EXAONE retrieval data."""

from __future__ import annotations

import os
import subprocess

import numpy as np
from exaone_retrieval_data import (
    CASE_ARRAY_NAMES,
    CASES_PATH,
    PLAN_PATH,
    ROOT,
    SEAL_PATH,
    TABLE_ARRAY_NAMES,
    TABLE_PATH,
    TRAIN_DOCUMENT_BYTE_BUDGET,
    VERIFICATION_ACTIVE_PATH,
    VERIFICATION_PATH,
    _load_npz,
    build_case_arrays,
    build_compact_backoff_table,
    build_verification,
    canonical_bytes,
    hash_file,
    read_plan,
    read_seal,
    validate_case_arrays,
    validate_verification,
)
from prepare_exaone_retrieval_data import (
    _disjointness_inventory,
    _encode_train_documents,
    _full_document_prefix,
    _load_tokenizer,
    _publish,
    _verified_records,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _exact_head_blob(path) -> None:
    blob = subprocess.check_output(
        ("git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"), cwd=ROOT
    )
    if blob != path.read_bytes():
        raise ValueError(f"EXAONE verification dependency is not HEAD: {path.name}")


def _arrays_equal(
    observed: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> bool:
    return set(observed) == set(expected) and all(
        np.array_equal(observed[name], expected[name]) for name in observed
    )


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE data verification requires a clean worktree")
    for path in (VERIFICATION_ACTIVE_PATH, VERIFICATION_PATH):
        if path.exists():
            raise FileExistsError(f"EXAONE verification namespace is not empty: {path}")
    history = _git(
        "log",
        "--all",
        "--format=%H",
        "--",
        VERIFICATION_PATH.relative_to(ROOT).as_posix(),
    )
    if history:
        raise FileExistsError("EXAONE data verification was already published")
    plan = read_plan(verify_derived=True)
    seal = read_seal(verify_artifacts=True)
    _exact_head_blob(PLAN_PATH)
    _exact_head_blob(SEAL_PATH)
    commit = _git("rev-parse", "HEAD")
    _publish(
        VERIFICATION_ACTIVE_PATH,
        canonical_bytes(
            {
                "plan_sha256": plan["plan_sha256"],
                "seal_artifact_sha256": hash_file(SEAL_PATH),
                "seal_sha256": seal["seal_sha256"],
                "verifier_git_commit": commit,
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
        reconstructed_table = build_compact_backoff_table(token_memory)
        reconstructed_table_arrays = reconstructed_table.to_arrays()
        rank_key = bytes.fromhex(
            plan["data_contract"]["case_selection"]["case_rank_key_hex"]
        )
        reconstructed_case_arrays, reconstructed_case_report = build_case_arrays(
            evaluation_records, tokenizer, rank_key=rank_key
        )
        validate_case_arrays(reconstructed_case_arrays, rank_key=rank_key)
        stored_table_arrays = _load_npz(TABLE_PATH, TABLE_ARRAY_NAMES)
        stored_case_arrays = _load_npz(CASES_PATH, CASE_ARRAY_NAMES)
        if (
            not _arrays_equal(reconstructed_table_arrays, stored_table_arrays)
            or not _arrays_equal(reconstructed_case_arrays, stored_case_arrays)
            or inventory != seal["training_inventory"]
            or reconstructed_case_report != seal["case_contract"]
            or tokenizer_runtime != seal["tokenizer_runtime"]
        ):
            raise ValueError("EXAONE independent data reconstruction differs")
        verification = build_verification(
            plan=plan,
            seal=seal,
            verifier_git_commit=commit,
            reconstructed_table_arrays=reconstructed_table_arrays,
            reconstructed_case_arrays=reconstructed_case_arrays,
            reconstructed_training_inventory=inventory,
            tokenizer_runtime=tokenizer_runtime,
        )
        validate_verification(verification, plan=plan, seal=seal)
        if (
            _git("rev-parse", "HEAD") != commit
            or _git("status", "--porcelain", "--untracked-files=all")
        ):
            raise RuntimeError("repository changed during EXAONE data verification")
        _publish(VERIFICATION_PATH, canonical_bytes(verification), 0o644)
        VERIFICATION_ACTIVE_PATH.unlink()
        print(f"verification_sha256={verification['verification_sha256']}")
        print("commit the tracked verification before resource calibration")
    finally:
        if token_memory is not None:
            del token_memory
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


if __name__ == "__main__":
    main()
