"""Sealed inputs for the trained 16K target-block upper-bound preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from compositional_head_preflight_protocol import (
    ROOT,
    current_environment,
    hash_file,
)
from fresh_vocabulary_16k_actual_protocol import (
    CHECKPOINT_BY_ROLE,
    QUALITY_RESULT_PATH,
    VOCABULARY_BY_ROLE,
    build_role_model,
    canonical_sha256,
    read_plan_json,
    reconstruct_cases,
)
from fresh_vocabulary_16k_actual_protocol import (
    IMPLEMENTATION_PATHS as ACTUAL_IMPLEMENTATION_PATHS,
)
from fresh_vocabulary_16k_actual_protocol import (
    OUTPUT_PATH as ACTUAL_RESULT_PATH,
)
from fresh_vocabulary_16k_actual_protocol import (
    PLAN_PATH as ACTUAL_PLAN_PATH,
)
from fresh_vocabulary_16k_block_core import (
    BLOCK_SIZE_BY_ROLE,
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    PRIMARY_MINIMUM_END_TO_END_REDUCTION,
    PRIMARY_ROLE,
    PROTOCOL_ID,
    REPETITIONS,
    ROLES,
)
from fresh_vocabulary_actual_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    PROMPT_BYTES,
    WARMUP_CASES,
)
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import state_mapping_sha256

PLAN_PATH = ROOT / "data/manifests/fresh-vocabulary-16k-target-block-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/fresh-vocabulary-16k-target-block-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
TIMING_PATH = ARTIFACT_ROOT / "timing.npz"
RUNTIME_REPORT_PATH = ARTIFACT_ROOT / "runtime-report.json"
OUTPUT_PATH = ROOT / "results/fresh-vocabulary-16k-target-block-v1/summary.json"
MECHANISM_PATH = ACTUAL_RESULT_PATH.parent / "mechanism-diagnostic.json"
TARGET_CHECKPOINT_PATH = CHECKPOINT_BY_ROLE["candidate_16k"]
TARGET_VOCABULARY_SIZE = VOCABULARY_BY_ROLE["candidate_16k"]
MPS_ATOL = 1e-4
MPS_RTOL = 2e-5
MAXIMUM_FREE_TOKENS = CONTINUATION_BYTES + 3
TIMED_SCOPE = (
    "raw-prompt UTF-8 decode and 16K tokenizer encode; fresh target KV-cache; "
    "parallel prefill; target block forwards; every target verifier argmax and "
    "device-host readback; token-byte reconstruction; strict UTF-8 transitions, "
    "stop, and decode; final MPS synchronization. Known-correct perfect-draft "
    "token construction is excluded."
)

_BLOCK_IMPLEMENTATION_PATHS = (
    "docs/103-exact-speculative-w72-result-and-pivot.md",
    "docs/162-fresh-v2-16k-trained-actual-result-and-block-pivot.md",
    "docs/163-fresh-v2-16k-target-block-upper-bound-protocol.md",
    "scripts/benchmark_fresh_vocabulary_16k_block.py",
    "scripts/fresh_vocabulary_16k_block_core.py",
    "scripts/fresh_vocabulary_16k_block_protocol.py",
    "scripts/fresh_vocabulary_16k_block_runtime.py",
    "scripts/preflight_fresh_vocabulary_16k_block.py",
    "scripts/seal_fresh_vocabulary_16k_block_plan.py",
    "scripts/summarize_fresh_vocabulary_16k_block.py",
    "tests/test_fresh_vocabulary_16k_block_core.py",
    "tests/test_fresh_vocabulary_16k_block_protocol.py",
    "tests/test_fresh_vocabulary_16k_block_runtime.py",
)
IMPLEMENTATION_PATHS = ACTUAL_IMPLEMENTATION_PATHS + tuple(
    path
    for path in _BLOCK_IMPLEMENTATION_PATHS
    if path not in ACTUAL_IMPLEMENTATION_PATHS
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"16K target-block JSON root differs: {path}")
    return value


def array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("16K target-block implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def _validated_source_results() -> tuple[dict[str, Any], dict[str, Any]]:
    actual = read_json(ACTUAL_RESULT_PATH)
    unsigned = dict(actual)
    recorded = unsigned.pop("summary_sha256", None)
    diagnostic = read_json(MECHANISM_PATH)
    diagnostic_unsigned = dict(diagnostic)
    diagnostic_recorded = diagnostic_unsigned.pop("diagnostic_sha256", None)
    if (
        actual.get("kind") != "fresh_vocabulary_16k_actual_one_seed_result_v1"
        or actual.get("status") != "fail_16k_trained_actual_e2e_preflight"
        or actual.get("actual_inference", {})
        .get("primary_gate", {})
        .get("overall_pass")
        is not False
        or canonical_sha256(unsigned) != recorded
        or diagnostic.get("kind")
        != "fresh_vocabulary_16k_actual_mechanism_diagnostic_v1"
        or diagnostic.get("status") != "post_hoc_descriptive_non_authorizing"
        or diagnostic.get("claim_boundary", {}).get("authorizes_multiseed") is not False
        or canonical_sha256(diagnostic_unsigned) != diagnostic_recorded
    ):
        raise ValueError("16K target-block source result differs")
    return actual, diagnostic


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "actual_plan": ACTUAL_PLAN_PATH,
        "actual_result": ACTUAL_RESULT_PATH,
        "mechanism_diagnostic": MECHANISM_PATH,
        "quality_result": QUALITY_RESULT_PATH,
        "target_checkpoint": TARGET_CHECKPOINT_PATH,
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in paths.items()
    }


def target_identity() -> dict[str, Any]:
    actual, _ = _validated_source_results()
    expected = actual["systems_cost"]["by_role"]["candidate_16k"]
    model = build_role_model("candidate_16k")
    state = torch.load(TARGET_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    output = {
        "quality_role": expected["quality_role"],
        "checkpoint_path": str(TARGET_CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_artifact_sha256": hash_file(TARGET_CHECKPOINT_PATH),
        "checkpoint_state_sha256": state_mapping_sha256(model.state_dict()),
        "checkpoint_bytes": TARGET_CHECKPOINT_PATH.stat().st_size,
        "parameter_count": model_parameter_count(model),
        "vocabulary_size": TARGET_VOCABULARY_SIZE,
        "document_bpb": expected["document_bpb"],
    }
    if output != expected:
        raise ValueError("16K target-block target identity differs")
    return output


def experiment_contract() -> dict[str, Any]:
    return {
        "roles": list(ROLES),
        "modes": list(MODES),
        "block_size_by_role": dict(BLOCK_SIZE_BY_ROLE),
        "primary_role": PRIMARY_ROLE,
        "prompt_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "warmup_cases": WARMUP_CASES,
        "measured_cases": MEASURED_CASES,
        "repetitions": REPETITIONS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "minimum_end_to_end_reduction": PRIMARY_MINIMUM_END_TO_END_REDUCTION,
        "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "mps_atol": MPS_ATOL,
        "mps_rtol": MPS_RTOL,
        "maximum_free_tokens": MAXIMUM_FREE_TOKENS,
        "timed_scope": TIMED_SCOPE,
        "perfect_draft_compute_inside_timing": False,
        "target_verifier_and_strict_utf8_inside_timing": True,
        "checkpoint_loading_inside_timing": False,
        "all_roles_share_one_resident_target": True,
        "free_perfect_draft_generated_before_timing": True,
        "primary_block_size_fixed_before_timing": 4,
        "diagnostic_block_sizes_cannot_replace_primary": [2, 8],
    }


def related_work_boundary_contract() -> dict[str, bool]:
    return {
        "standard_speculative_decoding_is_novel": False,
        "block_verification_is_novel": False,
        "speculative_vocabulary_is_prior_work": True,
        "multilingual_draft_weakness_is_prior_work": True,
        "this_stage_is_kernel_opportunity_only": True,
    }


def gate_contract() -> dict[str, Any]:
    return {
        "primary_role": PRIMARY_ROLE,
        "requires_both_modes": True,
        "minimum_end_to_end_reduction": PRIMARY_MINIMUM_END_TO_END_REDUCTION,
        "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "all_correctness_required": True,
        "pass_authorizes": "same-tokenizer learned-draft fail-fast only",
        "diagnostic_fallback": False,
    }


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "trained_target": True,
        "perfect_draft_upper_bound": True,
        "actual_draft_compute_measured": False,
        "actual_speculative_runtime_implemented": False,
        "quality_or_output_changed": False,
        "publication_claim": False,
    }


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if (
        not isinstance(git_commit_before_plan, str)
        or len(git_commit_before_plan) != 40
        or any(
            character not in "0123456789abcdef" for character in git_commit_before_plan
        )
    ):
        raise ValueError("16K target-block pre-plan commit differs")
    actual, diagnostic = _validated_source_results()
    _, _, cases = reconstruct_cases()
    actual_plan = read_plan_json(ACTUAL_PLAN_PATH)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_target_block_upper_bound_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_target_block_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "prior_evidence": {
            "actual_result_sha256": actual["summary_sha256"],
            "actual_primary_pass": False,
            "mechanism_diagnostic_sha256": diagnostic["diagnostic_sha256"],
            "prior_result_used_to_motivate_new_architecture": True,
            "prior_result_does_not_authorize_efficiency_claim": True,
        },
        "target": target_identity(),
        "tokenizer_runtime": actual_plan["tokenizer_runtime"]["candidate_16k"],
        "cases": cases,
        "experiment": experiment_contract(),
        "related_work_boundary": related_work_boundary_contract(),
        "gate": gate_contract(),
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "claim_boundary": claim_boundary_contract(),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected = {
        "schema_version",
        "kind",
        "protocol_id",
        "status",
        "git_commit_before_plan",
        "dependencies",
        "implementation_sha256",
        "environment",
        "prior_evidence",
        "target",
        "tokenizer_runtime",
        "cases",
        "experiment",
        "related_work_boundary",
        "gate",
        "output_path",
        "claim_boundary",
        "plan_sha256",
    }
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("kind") != "fresh_vocabulary_16k_target_block_upper_bound_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_target_block_timing"
        or not isinstance(plan.get("git_commit_before_plan"), str)
        or len(plan["git_commit_before_plan"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in plan["git_commit_before_plan"]
        )
        or canonical_sha256(unsigned) != recorded
        or plan.get("dependencies") != dependency_identity()
        or plan.get("implementation_sha256") != implementation_identity()
        or plan.get("environment") != current_environment()
        or plan.get("experiment") != experiment_contract()
        or plan.get("related_work_boundary") != related_work_boundary_contract()
        or plan.get("gate") != gate_contract()
        or plan.get("claim_boundary") != claim_boundary_contract()
        or plan.get("output_path") != str(OUTPUT_PATH.relative_to(ROOT))
    ):
        raise ValueError("16K target-block plan identity differs")
    actual, diagnostic = _validated_source_results()
    if plan.get("prior_evidence") != {
        "actual_result_sha256": actual["summary_sha256"],
        "actual_primary_pass": False,
        "mechanism_diagnostic_sha256": diagnostic["diagnostic_sha256"],
        "prior_result_used_to_motivate_new_architecture": True,
        "prior_result_does_not_authorize_efficiency_claim": True,
    }:
        raise ValueError("16K target-block prior evidence differs")
    if verify_derived:
        _, _, cases = reconstruct_cases()
        actual_plan = read_plan_json(ACTUAL_PLAN_PATH)
        if (
            plan.get("target") != target_identity()
            or plan.get("cases") != cases
            or plan.get("tokenizer_runtime")
            != actual_plan["tokenizer_runtime"]["candidate_16k"]
        ):
            raise ValueError("16K target-block derived fields differ")
