"""Sealed protocol helpers for the vocabulary-transfer development probe."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bpe_quality_frontier_core import array_sha256
from compositional_head_preflight_protocol import (
    INTEGRITY_PATH,
    ROOT,
    SOURCE_PATH,
    current_environment,
    hash_file,
    load_tokenizers,
    tokenizer_identity,
)
from compositional_quality_core import state_subset_sha256
from compositional_quality_protocol import inherited_inventories
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    BODY_LEARNING_RATE,
    COMPOSED_ROLES,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    FINAL_PROBE_STEP,
    GRADIENT_CLIP,
    HEAD_MINIMUM_LEARNING_RATE,
    HEAD_PEAK_LEARNING_RATE,
    MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
    MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
    MODEL_SEED,
    ORDER_SEED,
    PROBE_STEPS,
    RANDOM_CONTROL_BY_ROLE,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_SIZE,
    TRANSFER_ROLES,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    build_canonical_bpe_decomposition_table,
    build_transferred_model,
    expected_parameter_count,
    role_definition,
    state_mapping_sha256,
)

PROTOCOL_ID = "jamoflow-vocabulary-transfer-probe-v1"
PLAN_PATH = ROOT / "data/manifests/vocabulary-transfer-probe-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/vocabulary-transfer-probe-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/vocabulary-transfer-probe-v1/summary.json"

PARENT_PLAN_PATH = ROOT / "data/manifests/compositional-head-quality-one-seed-v1.json"
PARENT_RESULT_PATH = ROOT / "results/compositional-head-quality-one-seed-v1/summary.json"
PARENT_RESOURCE_PATH = ROOT / "artifacts/compositional-head-quality-one-seed-v1/resource-report.json"
BASE_CHECKPOINT_PATH = ROOT / "artifacts/compositional-head-quality-one-seed-v1/checkpoints/dense_v2048.pt"

IMPLEMENTATION_PATHS = (
    "docs/135-compositional-head-quality-result-and-transfer-pivot.md",
    "docs/136-vocabulary-transfer-probe-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/compositional_quality_core.py",
    "scripts/compositional_quality_protocol.py",
    "scripts/compositional_token_head.py",
    "scripts/run_compositional_quality.py",
    "scripts/run_vocabulary_transfer_probe.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_vocabulary_transfer_probe_plan.py",
    "scripts/summarize_vocabulary_transfer_probe.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "scripts/vocabulary_transfer_probe_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_vocabulary_transfer_probe.py",
    "tests/test_vocabulary_transfer_probe_core.py",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("vocabulary-transfer JSON root differs")
    return value


def implementation_identity() -> dict[str, str]:
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "base_checkpoint": BASE_CHECKPOINT_PATH,
        "integrity": INTEGRITY_PATH,
        "parent_plan": PARENT_PLAN_PATH,
        "parent_resource": PARENT_RESOURCE_PATH,
        "parent_result": PARENT_RESULT_PATH,
        "source": SOURCE_PATH,
    }
    return {
        key: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for key, path in paths.items()
    }


def base_checkpoint_state() -> dict[str, torch.Tensor]:
    result = read_json(PARENT_RESULT_PATH)
    lineage = result["artifact_lineage"]["dense_v2048"]
    if lineage["checkpoint_sha256"] != hash_file(BASE_CHECKPOINT_PATH):
        raise ValueError("vocabulary-transfer base checkpoint artifact differs")
    state = torch.load(BASE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("vocabulary-transfer base checkpoint payload differs")
    output = {name: value.detach().cpu().contiguous() for name, value in state.items()}
    if state_mapping_sha256(output) != lineage["checkpoint_state_sha256"]:
        raise ValueError("vocabulary-transfer base checkpoint state differs")
    return output


def target_order(sequence_count: int) -> np.ndarray:
    if sequence_count < FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE:
        raise ValueError("vocabulary-transfer training inventory is too small")
    return np.random.default_rng(ORDER_SEED).permutation(sequence_count).astype(np.int64, copy=False)


def training_contract() -> dict[str, Any]:
    inventory = inherited_inventories()[str(TARGET_VOCABULARY_SIZE)]["train"]
    count = int(inventory["full_sequence_count"])
    order = target_order(count)
    return {
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.95,
        "adamw_epsilon": 1e-8,
        "body_learning_rate": BODY_LEARNING_RATE,
        "checkpoint_steps": list(PROBE_STEPS),
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "final_probe_step": FINAL_PROBE_STEP,
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // TRAIN_MICROBATCH_SIZE,
        "gradient_clip": GRADIENT_CLIP,
        "head_minimum_learning_rate": HEAD_MINIMUM_LEARNING_RATE,
        "head_peak_learning_rate": HEAD_PEAK_LEARNING_RATE,
        "model_seed": MODEL_SEED,
        "optimizer": "AdamW",
        "order_seed": ORDER_SEED,
        "sequence_count": count,
        "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
        "training_order_prefix_sha256": array_sha256(
            order[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE]
        ),
        "warmup_steps": WARMUP_STEPS,
        "weight_decay_for_matrix_parameters": WEIGHT_DECAY,
        "weight_decay_for_vector_parameters": 0.0,
    }


def parent_anchor() -> dict[str, Any]:
    result = read_json(PARENT_RESULT_PATH)
    metric = result["metrics"]["dense_v2048"]
    replay = result["independent_nll_recomputation"]
    if result["decision"]["overall_pass"] is not False or replay["pass"] is not True:
        raise ValueError("vocabulary-transfer parent rejection differs")
    return {
        "contiguous_bpb": metric["contiguous_bpb"],
        "document_bpb": metric["document_bpb"],
        "checkpoint_artifact_sha256": result["artifact_lineage"]["dense_v2048"][
            "checkpoint_sha256"
        ],
        "checkpoint_state_sha256": result["artifact_lineage"]["dense_v2048"][
            "checkpoint_state_sha256"
        ],
        "parent_summary_sha256": result["summary_sha256"],
    }


def initialization_identities() -> tuple[dict[str, Any], dict[str, str]]:
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, target_pieces = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer,
        target_tokenizer,
        base_pieces,
        target_pieces,
    )
    state = base_checkpoint_state()
    audits: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for role in TRANSFER_ROLES:
        model, audit = build_transferred_model(
            role,
            base_state=state,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=decompositions,
        )
        audits[role] = audit.to_dict()
        hashes[role] = state_subset_sha256(model, transformer_body_only=False)
    return audits, hashes


def selection_rule() -> dict[str, Any]:
    return {
        "candidate_pool": list(COMPOSED_ROLES),
        "exact_tie_order": list(COMPOSED_ROLES),
        "random_control_by_candidate": dict(RANDOM_CONTROL_BY_ROLE),
        "selection_checkpoint_step": FINAL_PROBE_STEP,
        "minimum_initialization_advantage_bpb": MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
        "maximum_anchor_gap_for_full_cpt_bpb": MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
        "requires_both_gates": True,
        "korean_specific_fallback": None,
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "claim_boundary",
        "dependencies",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "initial_state_sha256",
        "initialization_audits",
        "inventories",
        "kind",
        "parent_anchor",
        "plan_sha256",
        "protocol_id",
        "resource_authorization",
        "roles",
        "role_specs",
        "schema_version",
        "selection_rule",
        "status",
        "tokenizers",
        "training",
    }
    if set(plan) != expected or (
        plan.get("schema_version") != 1
        or plan.get("kind") != "vocabulary_transfer_probe_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status")
        != "deterministic_initialization_identity_sealed_before_probe_loss"
    ):
        raise ValueError("vocabulary-transfer plan identity differs")
    base_commit = plan.get("git_commit_before_plan")
    if (
        not isinstance(base_commit, str)
        or len(base_commit) != 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise ValueError("vocabulary-transfer plan base commit differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("vocabulary-transfer plan hash differs")
    if plan["dependencies"] != dependency_identity() or plan["implementation_sha256"] != implementation_identity():
        raise ValueError("vocabulary-transfer plan lineage differs")
    if plan["environment"] != current_environment():
        raise ValueError("vocabulary-transfer environment differs")
    tokenizers = tokenizer_identity()
    if plan["tokenizers"] != {key: tokenizers[key] for key in ("2048", "8192")}:
        raise ValueError("vocabulary-transfer tokenizer identity differs")
    if plan["roles"] != list(TRANSFER_ROLES):
        raise ValueError("vocabulary-transfer roles differ")
    expected_role_specs = {
        role: {
            **role_definition(role),
            "expected_parameters": expected_parameter_count(role),
        }
        for role in TRANSFER_ROLES
    }
    if plan["role_specs"] != expected_role_specs:
        raise ValueError("vocabulary-transfer role specifications differ")
    inventories = inherited_inventories()
    if plan["inventories"] != {"8192": inventories["8192"]}:
        raise ValueError("vocabulary-transfer inventories differ")
    if plan["training"] != training_contract() or plan["parent_anchor"] != parent_anchor():
        raise ValueError("vocabulary-transfer training or anchor differs")
    if plan["selection_rule"] != selection_rule():
        raise ValueError("vocabulary-transfer selection rule differs")
    audits, states = initialization_identities()
    if plan["initialization_audits"] != audits or plan["initial_state_sha256"] != states:
        raise ValueError("vocabulary-transfer initialization differs")
    resource = read_json(PARENT_RESOURCE_PATH)
    expected_resource = {
        "artifact_sha256": hash_file(PARENT_RESOURCE_PATH),
        "tied_dense_8k_graph_and_microbatch_match": True,
        "untied_activation_geometry_matches_tied": True,
        "parameter_count_by_role": {
            role: expected_parameter_count(role) for role in TRANSFER_ROLES
        },
        "parent_projection_pass": resource["projection"]["passes"],
        "parent_dense_8k_worker_sha256": resource["projection"]["workers"]["dense_v8192"][
            "worker_artifact_sha256"
        ],
    }
    if plan["resource_authorization"] != expected_resource or expected_resource["parent_projection_pass"] is not True:
        raise ValueError("vocabulary-transfer resource authorization differs")
    if plan["claim_boundary"] != {
        "actual_inference_measured": False,
        "calibration_development_only": True,
        "korean_specific_method_evaluated": False,
        "model_seed_count": 1,
        "publication_quality_claim": False,
        "short_cpt_initializer_selection_only": True,
    }:
        raise ValueError("vocabulary-transfer claim boundary differs")
