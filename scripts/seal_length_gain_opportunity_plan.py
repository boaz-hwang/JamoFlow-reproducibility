#!/usr/bin/env python3
"""Seal the Length-Gain opportunity plan before any candidate calibration count."""

from __future__ import annotations

import subprocess

from length_gain_opportunity_protocol import (
    BATCH_SIZE,
    ENCODE_REPETITIONS,
    IMPLEMENTATION_PATHS,
    MAXIMUM_PIECE_BYTES,
    MAXIMUM_TOKEN_ARITY,
    MEASURED_CASES,
    MINIMUM_REDUCTION,
    PLAN_PATH,
    PRIMARY_ORDER,
    PROTOCOL_ID,
    RESULT_PATH,
    ROLE_ORDER,
    ROOT,
    SCORE_KIND,
    TRAIN_BYTES,
    VOCABULARY_SIZE,
    WARMUP_CASES,
    canonical_sha256,
    current_environment,
    dependency_identity,
    implementation_identity,
    json_bytes,
    reconstruct_all_inputs,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("length-gain plan requires a clean worktree")
    if PLAN_PATH.exists() or RESULT_PATH.exists() or _git("log", "--all", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))):
        raise RuntimeError("length-gain plan/result already exists or has history")
    _, _, _, _, shared_inputs = reconstruct_all_inputs()
    payload = {
        "schema_version": 1,
        "kind": "length_gain_opportunity_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_first_length_gain_calibration_evaluation",
        "git_commit_before_plan": _git("rev-parse", "HEAD"),
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "shared_inputs": shared_inputs,
        "experiment": {
            "batch_size": BATCH_SIZE,
            "encode_repetitions": ENCODE_REPETITIONS,
            "maximum_piece_bytes": MAXIMUM_PIECE_BYTES,
            "maximum_token_arity": MAXIMUM_TOKEN_ARITY,
            "measured_cases": MEASURED_CASES,
            "minimum_reduction": MINIMUM_REDUCTION,
            "primary_order": list(PRIMARY_ORDER),
            "role_order": list(ROLE_ORDER),
            "score_kind": SCORE_KIND,
            "train_bytes": TRAIN_BYTES,
            "vocabulary_size": VOCABULARY_SIZE,
            "warmup_cases": WARMUP_CASES,
        },
        "known_train_only_preflight": {
            "calibration_metrics_seen": False,
            "sample_bytes": 1_999_872,
            "overlap_score_batch_256_reduction_vs_bpe": -0.0427003051823347,
            "overlap_score_batch_32_immediate_reduction_vs_bpe": 0.0501749303701462,
            "overlap_score_batch_8_immediate_reduction_vs_bpe": 0.06415855069017329,
            "overlap_score_batch_32_current_length_reduction_vs_bpe": 0.04406948212683626,
            "exact_nonoverlap_first_round_seconds": 6.556800791993737,
            "interpretation": (
                "Train-only approximations selected batch=8 and immediate saving. "
                "They are disclosed engineering anchors, not gate evidence; the sealed "
                "constructor uses exact non-overlapping occurrence counts and 8M train bytes."
            ),
        },
        "claim_boundary": {
            "actual_model_latency": False,
            "calibration_development_only": True,
            "korean_complete_variant_evaluated": False,
            "matched_quality": False,
            "publication_evidence": False,
            "train_only_configuration_selection": True,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_bytes(json_bytes(payload))
    if _git("rev-parse", "HEAD") != payload["git_commit_before_plan"]:
        raise RuntimeError("repository changed while sealing length-gain plan")
    print(f"sealed {PLAN_PATH.relative_to(ROOT)}")
    print(f"plan_sha256={payload['plan_sha256']}")
    print(f"implementation_files={len(IMPLEMENTATION_PATHS)}")


if __name__ == "__main__":
    main()
