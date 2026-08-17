#!/usr/bin/env python3
"""Seal the deterministic generic same-2K opportunity gate."""

from __future__ import annotations

import subprocess

from same2k_opportunity_protocol import (
    IMPLEMENTATION_PATHS,
    PLAN_PATH,
    RESULT_PATH,
    ROOT,
    canonical_sha256,
    current_environment,
    dependency_identity,
    hash_file,
    json_bytes,
    reconstruct_shared_inputs,
    validate_plan,
)
from same2k_opportunity import (
    ENCODE_REPETITIONS,
    MEASURED_CASES,
    MINIMUM_STEP_REDUCTION,
    ROLES,
    VOCABULARY_SIZE,
    WARMUP_CASES,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command("git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if history:
        raise FileExistsError(f"same-2K path has Git history: {path}")


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("same-2K plan requires a clean root")
    _never_published(PLAN_PATH)
    _never_published(RESULT_PATH)
    _, _, _, shared = reconstruct_shared_inputs()
    payload = {
        "schema_version": 1,
        "kind": "same2k_generic_opportunity_plan_v6",
        "protocol_id": "jamoflow-same2k-generic-opportunity-v6",
        "status": "sealed_before_deterministic_full_corpus_training",
        "git_commit_before_plan": _command("git", "rev-parse", "HEAD"),
        "dependencies": dependency_identity(),
        "environment": current_environment(),
        "implementation_sha256": {
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        "shared_inputs": shared,
        "experiment": {
            "calibration_bytes": 8_000_000,
            "encode_repetitions": ENCODE_REPETITIONS,
            "measured_cases": MEASURED_CASES,
            "minimum_step_reduction": MINIMUM_STEP_REDUCTION,
            "roles": list(ROLES),
            "tokenizer_training_maximum_piece_bytes": 48,
            "vocabulary_size": VOCABULARY_SIZE,
            "warmup_cases": WARMUP_CASES,
        },
        "known_exploratory_anchors": {
            "hf_unigram_no_regex_calibration_token_count": 2_173_590,
            "hf_unigram_no_regex_reduction_vs_bpe": 0.03971148799457114,
            "hf_unigram_regex_calibration_token_count": 2_328_984,
            "hf_unigram_regex_reduction_vs_bpe": -0.02894132740970079,
            "interpretation": "unsealed nondeterministic API exploration; disclosed, not selection evidence",
        },
        "claim_boundary": {
            "actual_model_latency": False,
            "calibration_development_only": True,
            "korean_aware_method_evaluated": False,
            "model_quality_used": False,
            "publication_evidence": False,
            "token_only_generic_upper_bound": True,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    validate_plan(payload)
    if _command("git", "rev-parse", "HEAD") != payload["git_commit_before_plan"]:
        raise ValueError("same-2K HEAD changed during plan sealing")
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLAN_PATH.open("xb") as output:
        output.write(json_bytes(payload))
    print(f"sealed {PLAN_PATH.relative_to(ROOT)} plan={payload['plan_sha256']}")


if __name__ == "__main__":
    main()
