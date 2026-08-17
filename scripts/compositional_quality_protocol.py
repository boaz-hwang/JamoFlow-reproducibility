"""Sealed one-seed quality protocol for the selected 8K compositional head."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from compositional_head_preflight_protocol import (
    INTEGRITY_PATH,
    ROOT,
    SOURCE_PATH,
    current_environment,
    hash_file,
    load_tokenizers,
    tokenizer_identity,
)
from compositional_quality_core import (
    BASELINE_NONINFERIORITY_BPB,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CALIBRATION_BYTES,
    CONTROL_NONINFERIORITY_BPB,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH,
    MINIMUM_KOREAN_ADVANTAGE_BPB,
    MODEL_SEED,
    ORDER_SEED,
    QUALITY_ROLES,
    QUALITY_SPECS,
    RESOURCE_CAMPAIGN_HOUR_LIMIT,
    RESOURCE_MEASURED_EVALUATION_BATCHES,
    RESOURCE_MEASURED_STEPS,
    RESOURCE_MEMORY_FRACTION_LIMIT,
    RESOURCE_SAFETY_FACTOR,
    RESOURCE_WARMUP_EVALUATION_BATCHES,
    RESOURCE_WARMUP_STEPS,
    SEQUENCE_LENGTH,
    TRAIN_BYTES,
    TRAIN_MICROBATCH,
    assignment_audit,
    build_quality_model,
    deterministic_order,
    state_subset_sha256,
    training_contract,
)
from bpe_quality_feasibility_core import validate_inventory
from bpe_quality_frontier_core import array_sha256


PROTOCOL_ID = "jamoflow-compositional-head-quality-one-seed-v1"
PLAN_PATH = ROOT / "data/manifests/compositional-head-quality-one-seed-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/compositional-head-quality-one-seed-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
RESOURCE_ROOT = ARTIFACT_ROOT / "resource-workers"
RESOURCE_REPORT_PATH = ARTIFACT_ROOT / "resource-report.json"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/compositional-head-quality-one-seed-v1/summary.json"

SYSTEMS_RESULT_PATH = ROOT / "results/compositional-head-systems-preflight-v2/summary.json"
BPE_FEASIBILITY_PLAN_PATH = ROOT / "data/manifests/bpe-quality-frontier-feasibility-v1.json"
BPE_QUALITY_PLAN_PATH = ROOT / "data/manifests/bpe-quality-frontier-one-seed-v1.json"
BPE_QUALITY_RESULT_PATH = ROOT / "results/bpe-quality-frontier-one-seed-v1/summary.json"

IMPLEMENTATION_PATHS = (
    "docs/133-compositional-head-systems-preflight-result.md",
    "docs/134-compositional-head-quality-one-seed-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/compositional_quality_core.py",
    "scripts/compositional_quality_protocol.py",
    "scripts/compositional_token_head.py",
    "scripts/run_compositional_quality.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_compositional_quality_plan.py",
    "scripts/summarize_compositional_quality.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_compositional_quality.py",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("compositional quality JSON root differs")
    return value


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "bpe_feasibility_plan": BPE_FEASIBILITY_PLAN_PATH,
        "bpe_quality_plan": BPE_QUALITY_PLAN_PATH,
        "bpe_quality_result": BPE_QUALITY_RESULT_PATH,
        "integrity": INTEGRITY_PATH,
        "source": SOURCE_PATH,
        "systems_result": SYSTEMS_RESULT_PATH,
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in paths.items()
    }


def implementation_identity() -> dict[str, str]:
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def inherited_inventories() -> dict[str, Any]:
    feasibility = read_json(BPE_FEASIBILITY_PLAN_PATH)
    quality = read_json(BPE_QUALITY_PLAN_PATH)
    role_by_size = {"2048": "byte_bpe_v2048_d8", "8192": "byte_bpe_v8192_d8"}
    return {
        size: {
            "train": feasibility["inventories"][role]["train"],
            "calibration": feasibility["inventories"][role]["calibration"],
            "documents": quality["document_evaluation"]["by_role"][role],
        }
        for size, role in role_by_size.items()
    }


def training_contracts() -> dict[str, Any]:
    inventories = inherited_inventories()
    output = {}
    for role, spec in QUALITY_SPECS.items():
        count = inventories[str(spec.vocabulary_size)]["train"]["full_sequence_count"]
        row = training_contract(role, count)
        row["training_order_sha256"] = array_sha256(deterministic_order(count))
        output[role] = row
    return output


def assignment_audits() -> dict[str, Any]:
    table = load_tokenizers()[8_192][1]
    return {
        role: row
        for role in QUALITY_ROLES
        if (row := assignment_audit(role, table)) is not None
    }


def initial_state_identity() -> tuple[dict[str, str], str]:
    tokenizers = load_tokenizers()
    states = {}
    body = None
    for role, spec in QUALITY_SPECS.items():
        table = tokenizers[spec.vocabulary_size][1]
        model = build_quality_model(
            role,
            token_bytes=table if "code" in spec.head_kind else None,
            seed=MODEL_SEED,
        )
        states[role] = state_subset_sha256(model, transformer_body_only=False)
        body_hash = state_subset_sha256(model, transformer_body_only=True)
        if body is None:
            body = body_hash
        elif body_hash != body:
            raise ValueError("compositional quality Transformer bodies differ")
    if body is None:
        raise AssertionError("compositional quality body identity is missing")
    return states, body


def resource_contract() -> dict[str, Any]:
    return {
        "campaign_hour_limit_after_safety_factor": RESOURCE_CAMPAIGN_HOUR_LIMIT,
        "evaluation_batch_by_role": EVALUATION_BATCH,
        "measured_effective_steps": RESOURCE_MEASURED_STEPS,
        "measured_evaluation_batches": RESOURCE_MEASURED_EVALUATION_BATCHES,
        "memory_fraction_limit": RESOURCE_MEMORY_FRACTION_LIMIT,
        "quality_or_loss_values_recorded": False,
        "safety_factor": RESOURCE_SAFETY_FACTOR,
        "train_microbatch_by_role": TRAIN_MICROBATCH,
        "warmup_effective_steps": RESOURCE_WARMUP_STEPS,
        "warmup_evaluation_batches": RESOURCE_WARMUP_EVALUATION_BATCHES,
    }


def selection_rule() -> dict[str, Any]:
    return {
        "baseline_noninferiority_bpb": BASELINE_NONINFERIORITY_BPB,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "control_noninferiority_bpb": CONTROL_NONINFERIORITY_BPB,
        "minimum_korean_advantage_bpb": MINIMUM_KOREAN_ADVANTAGE_BPB,
        "primary_candidate": "hangul_code_v8192",
        "required_contrasts": {
            "hangul_vs_dense_2k": "contiguous/document/upper <= +0.010",
            "hangul_vs_generic": "contiguous/document <= -0.002 and upper <= 0",
            "hangul_vs_low_rank": "contiguous/document/upper <= +0.002",
            "hangul_vs_shuffled": "contiguous/document <= -0.002 and upper <= 0",
        },
        "trained_actual_inference_requires_all_primary_contrasts": True,
        "candidate_fallback": None,
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "assignment_audits",
        "claim_boundary",
        "dependencies",
        "document_common",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "initial_state_sha256",
        "inventories",
        "kind",
        "known_engineering_smoke",
        "model_specs",
        "plan_sha256",
        "protocol_id",
        "resource_contract",
        "roles",
        "schema_version",
        "selection_rule",
        "status",
        "tokenizers",
        "training",
        "transformer_body_initial_state_sha256",
    }
    if set(plan) != expected or (
        plan["schema_version"] != 1
        or plan["kind"] != "compositional_head_quality_one_seed_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_resource_probe_training_or_quality"
    ):
        raise ValueError("compositional quality plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("compositional quality plan hash differs")
    if (
        not isinstance(plan["git_commit_before_plan"], str)
        or len(plan["git_commit_before_plan"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in plan["git_commit_before_plan"]
        )
    ):
        raise ValueError("compositional quality base commit differs")
    if plan["dependencies"] != dependency_identity():
        raise ValueError("compositional quality dependencies differ")
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("compositional quality implementation differs")
    if plan["environment"] != current_environment():
        raise ValueError("compositional quality environment differs")
    tokenizers = tokenizer_identity()
    if plan["tokenizers"] != {key: tokenizers[key] for key in ("2048", "8192")}:
        raise ValueError("compositional quality tokenizers differ")
    if plan["roles"] != list(QUALITY_ROLES) or plan["model_specs"] != {
        role: QUALITY_SPECS[role].to_dict() for role in QUALITY_ROLES
    }:
        raise ValueError("compositional quality role contract differs")
    if plan["inventories"] != inherited_inventories():
        raise ValueError("compositional quality inherited inventories differ")
    for row in plan["inventories"].values():
        validate_inventory(row["train"])
        validate_inventory(row["calibration"])
    if plan["training"] != training_contracts():
        raise ValueError("compositional quality training contracts differ")
    if plan["assignment_audits"] != assignment_audits():
        raise ValueError("compositional quality assignments differ")
    if plan["resource_contract"] != resource_contract():
        raise ValueError("compositional quality resource contract differs")
    if plan["selection_rule"] != selection_rule():
        raise ValueError("compositional quality selection rule differs")
    if plan["known_engineering_smoke"] != {
        "status": "hangul_8k_single_microbatch_forward_backward_only",
        "microbatch": 8,
        "sequence_length": 512,
        "finite": True,
        "elapsed_seconds_observed": True,
        "loss_value_recorded": False,
        "used_to_change_quality_gate_or_roles": False,
    }:
        raise ValueError("compositional quality disclosed smoke differs")
    if set(plan["initial_state_sha256"]) != set(QUALITY_ROLES) or any(
        not isinstance(value, str) or len(value) != 64
        for value in plan["initial_state_sha256"].values()
    ) or not isinstance(plan["transformer_body_initial_state_sha256"], str) or len(
        plan["transformer_body_initial_state_sha256"]
    ) != 64:
        raise ValueError("compositional quality initial state identity differs")
    bpe_plan = read_json(BPE_QUALITY_PLAN_PATH)
    if plan["document_common"] != bpe_plan["document_evaluation"]["common"]:
        raise ValueError("compositional quality document set differs")
    if plan["claim_boundary"] != {
        "actual_mps_training": True,
        "calibration_development_only": True,
        "model_seed_count": 1,
        "publication_quality_claim": False,
        "random_weight_systems_result_already_known": True,
        "resource_gate_uses_no_recorded_loss_value": True,
        "trained_actual_inference_measured_in_this_stage": False,
    }:
        raise ValueError("compositional quality claim boundary differs")
