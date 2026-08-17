"""Sealed inputs for the trained 16K actual retrieval-draft development preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from compositional_head_preflight_protocol import ROOT, current_environment, hash_file
from fresh_vocabulary_16k_actual_protocol import (
    canonical_sha256,
    json_bytes,
    read_plan_json,
    reconstruct_cases,
)
from fresh_vocabulary_16k_block_protocol import (
    OUTPUT_PATH as UPPER_BOUND_RESULT_PATH,
)
from fresh_vocabulary_16k_block_protocol import (
    PLAN_PATH as UPPER_BOUND_PLAN_PATH,
)
from fresh_vocabulary_16k_block_protocol import (
    TARGET_CHECKPOINT_PATH,
    read_json,
    target_identity,
)
from fresh_vocabulary_16k_retrieval_actual_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONTINUATION_BYTES,
    COUNTER_NAMES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_END_TO_END_POINT_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    PRIMARY_ROLE,
    PROTOCOL_ID,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
)
from fresh_vocabulary_16k_retrieval_core import (
    MAXIMUM_CONTEXT_ORDER,
    MAXIMUM_DRAFT_TOKENS,
    MAXIMUM_PROMPT_MATCH,
    MAXIMUM_TABLE_ENTRIES,
    MINIMUM_CONTEXT_COUNT,
    MINIMUM_NEXT_TOKEN_PROBABILITY,
    TABLE_ARRAY_NAMES,
    table_from_arrays,
    table_report,
)
from fresh_vocabulary_actual_core import PROMPT_BYTES, WARMUP_CASES

__all__ = ("json_bytes",)

PLAN_PATH = ROOT / "data/manifests/fresh-vocabulary-16k-retrieval-actual-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/fresh-vocabulary-16k-retrieval-actual-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
TIMING_PATH = ARTIFACT_ROOT / "timing.npz"
RUNTIME_REPORT_PATH = ARTIFACT_ROOT / "runtime-report.json"
OUTPUT_PATH = ROOT / "results/fresh-vocabulary-16k-retrieval-actual-v1/summary.json"
TABLE_SEAL_PATH = ROOT / "data/seals/fresh-vocabulary-16k-retrieval-table-v1.json"
TABLE_PATH = ROOT / "artifacts/fresh-vocabulary-16k-retrieval-table-v1/compact-token-ngram.npz"
MAXIMUM_FREE_TOKENS = CONTINUATION_BYTES + 3
MPS_ATOL = 1e-4
MPS_RTOL = 2e-5

IMPLEMENTATION_PATHS = (
    "data/seals/fresh-vocabulary-16k-retrieval-table-v1.json",
    "docs/164-fresh-v2-16k-target-block-upper-bound-result.md",
    "docs/165-retrieval-draft-literature-audit-and-fail-fast-direction.md",
    "docs/166-fresh-v2-16k-retrieval-draft-actual-protocol.md",
    "pyproject.toml",
    "scripts/benchmark_fresh_vocabulary_16k_block.py",
    "scripts/benchmark_fresh_vocabulary_16k_retrieval.py",
    "scripts/benchmark_fresh_vocabulary_actual.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/fresh_vocabulary_16k_actual_protocol.py",
    "scripts/fresh_vocabulary_16k_block_protocol.py",
    "scripts/fresh_vocabulary_16k_block_runtime.py",
    "scripts/fresh_vocabulary_16k_retrieval_actual_core.py",
    "scripts/fresh_vocabulary_16k_retrieval_core.py",
    "scripts/fresh_vocabulary_16k_retrieval_protocol.py",
    "scripts/fresh_vocabulary_16k_retrieval_runtime.py",
    "scripts/preflight_fresh_vocabulary_16k_retrieval.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_fresh_vocabulary_16k_retrieval_plan.py",
    "scripts/summarize_fresh_vocabulary_16k_retrieval.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/utf8.py",
    "tests/test_fresh_vocabulary_16k_retrieval_actual_core.py",
    "tests/test_fresh_vocabulary_16k_retrieval_core.py",
    "tests/test_fresh_vocabulary_16k_retrieval_protocol.py",
    "tests/test_fresh_vocabulary_16k_retrieval_runtime.py",
)

TIMED_SCOPE = (
    "raw-prompt UTF-8 decode and 16K tokenizer encode; fresh target KV-cache; "
    "parallel prefill; train-only compact token n-gram lookup; prompt+self-output "
    "suffix lookup; target block verification; every verifier argmax and device-host "
    "readback; rejection and DynamicCache crop; correction/bonus; token-byte "
    "reconstruction; strict UTF-8 transition/stop/decode; final MPS synchronization"
)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("16K retrieval implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def _validated_upper_bound() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = read_plan_json(UPPER_BOUND_PLAN_PATH)
    result = read_json(UPPER_BOUND_RESULT_PATH)
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256", None)
    if (
        result.get("kind") != "fresh_vocabulary_16k_target_block_upper_bound_result_v1"
        or result.get("status") != "pass_16k_target_block_upper_bound"
        or result.get("upper_bound", {}).get("primary_gate", {}).get("overall_pass") is not True
        or result.get("decision", {}).get("learned_same_tokenizer_draft_fail_fast_authorized")
        is not True
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("16K retrieval upper-bound authorization differs")
    return plan, result


def _validate_table_seal() -> dict[str, Any]:
    seal = json.loads(TABLE_SEAL_PATH.read_text(encoding="utf-8"))
    unsigned = dict(seal)
    recorded = unsigned.pop("seal_sha256", None)
    artifact = seal.get("table_artifact", {})
    if (
        seal.get("kind") != "fresh_vocabulary_16k_retrieval_table_seal_v1"
        or seal.get("complete") is not True
        or canonical_sha256(unsigned) != recorded
        or artifact.get("path") != str(TABLE_PATH.relative_to(ROOT))
        or artifact.get("sha256") != hash_file(TABLE_PATH)
        or artifact.get("bytes") != TABLE_PATH.stat().st_size
        or seal.get("result_inputs")
        != {
            "train_split_tokens": True,
            "calibration_tokens": False,
            "historical_test_metrics": False,
            "sealed_final_test": False,
            "model_checkpoint_or_logits": False,
            "latency": False,
        }
    ):
        raise ValueError("16K retrieval table seal differs")
    with np.load(TABLE_PATH, allow_pickle=False) as archive:
        if set(archive.files) != set(TABLE_ARRAY_NAMES):
            raise ValueError("16K retrieval table artifact keys differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    table = table_from_arrays(arrays)
    if table_report(table) != seal.get("table_contract"):
        raise ValueError("16K retrieval table content differs")
    for name, values in arrays.items():
        if artifact["arrays"].get(name) != {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": array_sha256(values),
        }:
            raise ValueError(f"16K retrieval table array differs: {name}")
    return seal


def load_table():
    _validate_table_seal()
    with np.load(TABLE_PATH, allow_pickle=False) as archive:
        return table_from_arrays(
            {name: np.ascontiguousarray(archive[name]) for name in archive.files}
        )


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "upper_bound_plan": UPPER_BOUND_PLAN_PATH,
        "upper_bound_result": UPPER_BOUND_RESULT_PATH,
        "retrieval_table_seal": TABLE_SEAL_PATH,
        "retrieval_table_artifact": TABLE_PATH,
        "target_checkpoint": TARGET_CHECKPOINT_PATH,
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in paths.items()
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "roles": list(ROLES),
        "modes": list(MODES),
        "primary_role": PRIMARY_ROLE,
        "prompt_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "warmup_cases": WARMUP_CASES,
        "measured_cases": MEASURED_CASES,
        "repetitions": REPETITIONS,
        "maximum_draft_tokens": MAXIMUM_DRAFT_TOKENS,
        "target_block_input_tokens": MAXIMUM_DRAFT_TOKENS + 1,
        "maximum_prompt_match": MAXIMUM_PROMPT_MATCH,
        "maximum_context_order": MAXIMUM_CONTEXT_ORDER,
        "maximum_table_entries": MAXIMUM_TABLE_ENTRIES,
        "minimum_context_count": MINIMUM_CONTEXT_COUNT,
        "minimum_next_token_probability": MINIMUM_NEXT_TOKEN_PROBABILITY,
        "maximum_free_tokens": MAXIMUM_FREE_TOKENS,
        "mps_atol": MPS_ATOL,
        "mps_rtol": MPS_RTOL,
        "timing_components": list(TIMING_COMPONENTS),
        "counter_names": list(COUNTER_NAMES),
        "timed_scope": TIMED_SCOPE,
        "dictionary_precedence": "train_corpus_first",
        "prompt_lookup_fallback": True,
        "proposal_absence_falls_back_to_ar": True,
        "all_roles_share_one_resident_target": True,
        "checkpoint_and_table_loading_inside_timing": False,
    }


def gate_contract() -> dict[str, Any]:
    return {
        "primary_role": PRIMARY_ROLE,
        "requires_both_modes": True,
        "minimum_end_to_end_reduction": MINIMUM_END_TO_END_POINT_REDUCTION,
        "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "all_output_and_cache_correctness_required": True,
        "diagnostic_roles_cannot_replace_primary": list(ROLES[1:3]),
        "pass_authorizes": "disjoint Korean-aware versus generic-hybrid design only",
    }


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "development_one_seed_one_session": True,
        "existing_development_cases_reused": True,
        "trained_target": True,
        "actual_draft_compute_inside_timing": True,
        "actual_rejection_and_rollback_inside_timing": True,
        "generic_retrieval_is_prior_work": True,
        "korean_specific_method_tested": False,
        "confirmatory_or_final_blind": False,
        "publication_claim": False,
    }


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if len(git_commit_before_plan) != 40:
        raise ValueError("16K retrieval pre-plan commit differs")
    upper_plan, upper_result = _validated_upper_bound()
    table_seal = _validate_table_seal()
    _, _, cases = reconstruct_cases()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_retrieval_actual_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_retrieval_actual_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "prior_evidence": {
            "upper_bound_plan_sha256": upper_plan["plan_sha256"],
            "upper_bound_summary_sha256": upper_result["summary_sha256"],
            "upper_bound_primary_pass": True,
            "table_seal_sha256": table_seal["seal_sha256"],
            "table_uses_train_only": True,
        },
        "target": target_identity(),
        "tokenizer_runtime": upper_plan["tokenizer_runtime"],
        "table": {
            "seal_artifact_sha256": hash_file(TABLE_SEAL_PATH),
            "seal_sha256": table_seal["seal_sha256"],
            "artifact_sha256": hash_file(TABLE_PATH),
            "contract": table_seal["table_contract"],
        },
        "cases": cases,
        "experiment": experiment_contract(),
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
        "table",
        "cases",
        "experiment",
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
        or plan.get("kind") != "fresh_vocabulary_16k_retrieval_actual_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_retrieval_actual_timing"
        or canonical_sha256(unsigned) != recorded
        or plan.get("dependencies") != dependency_identity()
        or plan.get("implementation_sha256") != implementation_identity()
        or plan.get("environment") != current_environment()
        or plan.get("experiment") != experiment_contract()
        or plan.get("gate") != gate_contract()
        or plan.get("claim_boundary") != claim_boundary_contract()
        or plan.get("output_path") != str(OUTPUT_PATH.relative_to(ROOT))
    ):
        raise ValueError("16K retrieval plan identity differs")
    upper_plan, upper_result = _validated_upper_bound()
    table_seal = _validate_table_seal()
    if plan.get("prior_evidence") != {
        "upper_bound_plan_sha256": upper_plan["plan_sha256"],
        "upper_bound_summary_sha256": upper_result["summary_sha256"],
        "upper_bound_primary_pass": True,
        "table_seal_sha256": table_seal["seal_sha256"],
        "table_uses_train_only": True,
    } or plan.get("table") != {
        "seal_artifact_sha256": hash_file(TABLE_SEAL_PATH),
        "seal_sha256": table_seal["seal_sha256"],
        "artifact_sha256": hash_file(TABLE_PATH),
        "contract": table_seal["table_contract"],
    }:
        raise ValueError("16K retrieval prior evidence differs")
    if verify_derived:
        _, _, cases = reconstruct_cases()
        if (
            plan.get("target") != target_identity()
            or plan.get("tokenizer_runtime") != upper_plan["tokenizer_runtime"]
            or plan.get("cases") != cases
        ):
            raise ValueError("16K retrieval derived identity differs")
