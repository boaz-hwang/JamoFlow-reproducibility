#!/usr/bin/env python3
"""Seal the Korean BPE systems-frontier plan before new counts or timings."""

from __future__ import annotations

import os
import subprocess

from jamoflow.publication_bpe import PINNED_TOKENIZERS_VERSION
from token_frontier_core import FRONTIER_SPECS, RUNTIME_ROLES
from token_frontier_protocol import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONTINUATION_BYTES,
    DEPTHS,
    IMPLEMENTATION_PATHS,
    INTEGRITY_PATH,
    KNOWN_PRESEAL_ENGINEERING_ANCHORS,
    MEASURED_CASES,
    MODEL_SEED,
    MPS_ATOL,
    MPS_RTOL,
    OUTPUT_PATH,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    PLAN_PATH,
    PRIOR_RESULT_PATH,
    PRIOR_OPPORTUNITY_PATH,
    PROMPT_BYTES,
    PROTOCOL_ID,
    REPETITIONS,
    ROOT,
    SOURCE_PATH,
    TOKENIZER_ENCODE_REPETITIONS,
    VOCABULARY_SIZES,
    WARMUP_CASES,
    canonical_sha256,
    current_frontier_environment,
    hash_file,
    json_bytes,
    reconstruct_cases,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command("git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if history:
        raise FileExistsError(f"token frontier path has Git history: {path}")


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("token frontier plan sealing requires a clean root")
    _never_published(PLAN_PATH)
    _never_published(OUTPUT_PATH)
    commit = _command("git", "rev-parse", "HEAD")
    prompts, continuations, cases = reconstruct_cases()
    if prompts.shape != (WARMUP_CASES + MEASURED_CASES, PROMPT_BYTES):
        raise AssertionError("token frontier prompt shape differs")
    if continuations.shape != (WARMUP_CASES + MEASURED_CASES, CONTINUATION_BYTES):
        raise AssertionError("token frontier continuation shape differs")
    payload = {
        "schema_version": 1,
        "kind": "korean_bpe_systems_frontier_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_after_known_16k_32k_anchors_before_new_grid_counts_and_runtime",
        "dependencies": {
            "git_commit_before_plan": commit,
            "integrity_path": str(INTEGRITY_PATH.relative_to(ROOT)),
            "integrity_sha256": hash_file(INTEGRITY_PATH),
            "prior_scalar_opportunity_path": str(PRIOR_OPPORTUNITY_PATH.relative_to(ROOT)),
            "prior_scalar_opportunity_sha256": hash_file(PRIOR_OPPORTUNITY_PATH),
            "prior_scalar_runtime_path": str(PRIOR_RESULT_PATH.relative_to(ROOT)),
            "prior_scalar_runtime_sha256": hash_file(PRIOR_RESULT_PATH),
            "source_path": str(SOURCE_PATH.relative_to(ROOT)),
            "source_sha256": hash_file(SOURCE_PATH),
        },
        "environment": current_frontier_environment(),
        "known_preseal_engineering_anchors": KNOWN_PRESEAL_ENGINEERING_ANCHORS,
        "tokenizer": {
            "add_prefix_space": False,
            "byte_fallback": False,
            "dropout": None,
            "initial_alphabet": "complete ByteLevel alphabet",
            "minimum_frequency": 2,
            "normalizer": None,
            "replicate_training_for_exact_json_determinism": True,
            "diagnostic_calibration_encode_repetitions": TOKENIZER_ENCODE_REPETITIONS,
            "tokenizers_version": PINNED_TOKENIZERS_VERSION,
            "train_split_only": True,
            "use_regex": True,
            "vocabulary_sizes": list(VOCABULARY_SIZES),
        },
        "model_specs": {role: FRONTIER_SPECS[role].to_dict() for role in RUNTIME_ROLES},
        "experiment": {
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "depths": list(DEPTHS),
            "measured_cases": MEASURED_CASES,
            "model_seed": MODEL_SEED,
            "mps_atol": MPS_ATOL,
            "mps_rtol": MPS_RTOL,
            "parameter_relative_tolerance": PARAMETER_RELATIVE_TOLERANCE,
            "parameter_target": PARAMETER_TARGET,
            "prompt_bytes": PROMPT_BYTES,
            "continuation_bytes": CONTINUATION_BYTES,
            "repetitions": REPETITIONS,
            "roles": list(RUNTIME_ROLES),
            "vocabulary_sizes": list(VOCABULARY_SIZES),
            "warmup_cases": WARMUP_CASES,
        },
        "cases": cases,
        "implementation_sha256": {
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        "claim_boundary": {
            "actual_model_graph_timing": True,
            "calibration_development_only": True,
            "free_running_generation": False,
            "matched_quality": False,
            "new_tokenizer_method": False,
            "random_weights_only": True,
            "tokenization_inside_model_timer": False,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    encoded = json_bytes(payload)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PLAN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
