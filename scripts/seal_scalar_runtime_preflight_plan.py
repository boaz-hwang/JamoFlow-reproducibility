#!/usr/bin/env python3
"""Seal the model-free scalar runtime plan before any timing."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from scalar_runtime_core import (
    BPE_PRIMARY_SPEC,
    BPE_SECONDARY_SPEC,
    FactorizedUnitBlt,
    build_bpe_model,
    model_parameter_count,
)
from scalar_runtime_protocol import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONTINUATION_BYTES,
    IMPLEMENTATION_PATHS,
    INTEGRITY_PATH,
    MEASURED_CASES,
    MPS_ATOL,
    MPS_RTOL,
    OPPORTUNITY_PATH,
    OUTPUT_PATH,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    PLAN_PATH,
    PROMPT_BYTES,
    PROTOCOL_ID,
    REPETITIONS,
    REPORT_PATH,
    ROOT,
    RUNTIME_ROLES,
    SOURCE_PATH,
    TIMING_PATH,
    TOKENIZER_PATHS,
    WARMUP_CASES,
    canonical_sha256,
    hash_file,
    json_bytes,
    reconstruct_cases,
    schedule_sha256,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_root() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("scalar runtime plan requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("scalar runtime plan requires a Git commit")
    return commit


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"scalar runtime artifact already exists: {path}")
    history = subprocess.run(
        ["git", "log", "--all", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if history:
        raise FileExistsError(f"scalar runtime artifact has Git history: {path}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    commit = _require_clean_root()
    for path in (PLAN_PATH, REPORT_PATH, TIMING_PATH, OUTPUT_PATH):
        _require_never_published(path)
    opportunity = __import__("json").loads(OPPORTUNITY_PATH.read_text(encoding="utf-8"))
    if opportunity["decision"]["pass"] is not True:
        raise ValueError("scalar opportunity did not authorize runtime construction")
    expected_tokenizers = {
        int(size): value["tokenizer_json_sha256"]
        for size, value in opportunity["metrics"]["bpe"].items()
    }
    for vocabulary_size, path in TOKENIZER_PATHS.items():
        if hash_file(path) != expected_tokenizers[vocabulary_size]:
            raise ValueError("scalar runtime tokenizer differs from opportunity audit")
    prompts, continuations, cases = reconstruct_cases()
    if prompts.shape != (WARMUP_CASES + MEASURED_CASES, PROMPT_BYTES):
        raise ValueError("scalar runtime prompt shape differs")
    if continuations.shape != (
        WARMUP_CASES + MEASURED_CASES,
        CONTINUATION_BYTES,
    ):
        raise ValueError("scalar runtime continuation shape differs")
    graphs = {
        "byte_w72": {
            "parameter_count": PARAMETER_TARGET,
            "specification": "sealed W72 BLT, 72 causal-whitespace-grid patches",
        },
        "generic_unicode_scalar": {
            "parameter_count": model_parameter_count(
                FactorizedUnitBlt("generic_unicode_scalar")
            ),
            "specification": "W72 BLT backbone plus 3x64 conditional UTF-8 heads",
        },
        "hangul_hybrid": {
            "parameter_count": model_parameter_count(
                FactorizedUnitBlt("hangul_hybrid")
            ),
            "specification": "W72 BLT backbone plus 19/21/28 conditional LVT heads",
        },
        "byte_bpe_32000": {
            "parameter_count": model_parameter_count(build_bpe_model(BPE_PRIMARY_SPEC)),
            "specification": dict(BPE_PRIMARY_SPEC),
        },
        "byte_bpe_16000": {
            "parameter_count": model_parameter_count(build_bpe_model(BPE_SECONDARY_SPEC)),
            "specification": dict(BPE_SECONDARY_SPEC),
        },
    }
    for graph in graphs.values():
        graph["relative_parameter_difference"] = (
            graph["parameter_count"] / PARAMETER_TARGET - 1.0
        )
        if abs(graph["relative_parameter_difference"]) > PARAMETER_RELATIVE_TOLERANCE:
            raise ValueError("scalar runtime graph is not parameter matched")
    plan = {
        "benchmark": {
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "continuation_raw_bytes": CONTINUATION_BYTES,
            "correctness_cases": WARMUP_CASES,
            "correctness_tolerance": {
                "atol": MPS_ATOL,
                "normalized_worst_error_maximum": 1.0,
                "rtol": MPS_RTOL,
            },
            "device": "mps",
            "measured_cases": MEASURED_CASES,
            "mode": "controlled_fixed_route_sampling",
            "model_seed": 20_260_814,
            "prompt_raw_bytes": PROMPT_BYTES,
            "repetitions": REPETITIONS,
            "roles": list(RUNTIME_ROLES),
            "schedule_sha256": schedule_sha256(),
            "warmup_cases": WARMUP_CASES,
        },
        "cases": cases,
        "claim_boundary": {
            "actual_mps_wall_time": True,
            "calibration_development_cases": True,
            "controlled_target_route_not_free_generation": True,
            "matched_quality_evidence": False,
            "random_weights_only": True,
            "training_or_model_loss_read": False,
            "publication_speed_claim": False,
            "tokenization_and_unit_encoding_outside_timing": True,
        },
        "decision_rule": {
            "bpe_competitive_minimum_bootstrap_lower_reduction": -0.10,
            "byte_minimum_bootstrap_lower_reduction": 0.0,
            "byte_minimum_median_reduction": 0.10,
            "byte_minimum_positive_prompts": 28,
            "hangul_specific_maximum_lower_bound_slowdown_vs_generic": 0.05,
            "maximum_relative_parameter_difference": PARAMETER_RELATIVE_TOLERANCE,
            "requires_all_correctness_checks": True,
            "training_authorized_if_any_scalar_candidate_passes": True,
        },
        "dependencies": {
            "integrity_artifact_sha256": hash_file(INTEGRITY_PATH),
            "opportunity_artifact_sha256": hash_file(OPPORTUNITY_PATH),
            "opportunity_summary_sha256": opportunity["summary_sha256"],
            "source_artifact_sha256": hash_file(SOURCE_PATH),
            "tokenizer_artifact_sha256": {
                str(size): hash_file(path) for size, path in TOKENIZER_PATHS.items()
            },
        },
        "graphs": graphs,
        "implementation_sha256": {
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        "kind": "scalar_runtime_preflight_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "schema_version": 1,
        "status": "sealed_before_runtime_measurement",
    }
    plan["dependencies"]["plan_base_git_commit"] = commit
    plan["dependencies"]["plan_payload_sha256"] = canonical_sha256(plan)
    _publish(PLAN_PATH, json_bytes(plan))
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
