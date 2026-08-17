"""Protocol helpers for the strong vocabulary-transfer baseline closure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
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
from compositional_quality_protocol import inherited_inventories
from vocabulary_transfer_baseline_core import (
    BASELINE_ROLES,
    COMPOSED_BASELINE_ROLES,
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    RANDOM_CONTROL_BY_ROLE,
    TWO_STAGE_BOUNDARY,
    TWO_STAGE_FULL_STEPS,
    build_transferred_model,
    expected_parameter_count,
    role_definition,
    source_token_metadata,
    state_mapping_sha256,
)
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    BODY_LEARNING_RATE,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    HEAD_MINIMUM_LEARNING_RATE,
    HEAD_PEAK_LEARNING_RATE,
    MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
    MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
    MODEL_SEED,
    ORDER_SEED,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    build_canonical_bpe_decomposition_table,
)
from vocabulary_transfer_probe_protocol import (
    BASE_CHECKPOINT_PATH,
    PARENT_PLAN_PATH,
    PARENT_RESOURCE_PATH,
    PARENT_RESULT_PATH,
    base_checkpoint_state,
    parent_anchor,
)

PROTOCOL_ID = "jamoflow-vocabulary-transfer-baseline-closure-v1"
PLAN_PATH = ROOT / "data/manifests/vocabulary-transfer-baseline-closure-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/vocabulary-transfer-baseline-closure-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/vocabulary-transfer-baseline-closure-v1/summary.json"

PREVIOUS_PROBE_PLAN_PATH = ROOT / "data/manifests/vocabulary-transfer-probe-v1.json"
PREVIOUS_PROBE_RESULT_PATH = ROOT / "results/vocabulary-transfer-probe-v1/summary.json"

IMPLEMENTATION_PATHS = (
    "docs/137-vocabulary-transfer-probe-result-and-baseline-closure.md",
    "docs/138-strong-vocabulary-transfer-baseline-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/compositional_quality_core.py",
    "scripts/compositional_quality_protocol.py",
    "scripts/run_vocabulary_transfer_baseline.py",
    "scripts/run_vocabulary_transfer_probe.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_vocabulary_transfer_baseline_plan.py",
    "scripts/summarize_vocabulary_transfer_baseline.py",
    "scripts/vocabulary_transfer_baseline_core.py",
    "scripts/vocabulary_transfer_baseline_protocol.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "scripts/vocabulary_transfer_probe_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "tests/test_vocabulary_transfer_baseline.py",
    "tests/test_vocabulary_transfer_baseline_core.py",
)


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("baseline JSON root differs")
    return value


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise ValueError("baseline implementation path is duplicated")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "base_checkpoint": BASE_CHECKPOINT_PATH,
        "integrity": INTEGRITY_PATH,
        "parent_plan": PARENT_PLAN_PATH,
        "parent_resource": PARENT_RESOURCE_PATH,
        "parent_result": PARENT_RESULT_PATH,
        "previous_probe_plan": PREVIOUS_PROBE_PLAN_PATH,
        "previous_probe_result": PREVIOUS_PROBE_RESULT_PATH,
        "source": SOURCE_PATH,
    }
    return {
        key: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for key, path in paths.items()
    }


def target_order(sequence_count: int) -> np.ndarray:
    if sequence_count < FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE:
        raise ValueError("baseline training inventory is too small")
    return np.random.default_rng(ORDER_SEED).permutation(sequence_count).astype(
        np.int64, copy=False
    )


def training_contract() -> dict[str, Any]:
    inventory = inherited_inventories()[str(TARGET_VOCABULARY_SIZE)]["train"]
    count = int(inventory["full_sequence_count"])
    order = target_order(count)
    return {
        "adamw_beta1": 0.9,
        "adamw_beta2": 0.95,
        "adamw_epsilon": 1e-8,
        "all_parameter_schedule": "same global 512-step LR schedule as prior probe",
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
        "optimizer_reinitialized_at_stage_two": True,
        "order_seed": ORDER_SEED,
        "sequence_count": count,
        "stage_one_copied_rows_restored_and_checked_each_step": True,
        "stage_one_new_rows_only_steps": TWO_STAGE_BOUNDARY,
        "stage_two_all_parameter_steps": TWO_STAGE_FULL_STEPS,
        "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
        "training_order_prefix_sha256": array_sha256(
            order[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE]
        ),
        "two_stage_interpretation": (
            "compact 307:205 ratio analogue of 600B:400B; not an exact scale reproduction"
        ),
        "warmup_steps": WARMUP_STEPS,
        "weight_decay_for_matrix_parameters": WEIGHT_DECAY,
        "weight_decay_for_vector_parameters": 0.0,
    }


def previous_probe_evidence() -> dict[str, Any]:
    result = read_json(PREVIOUS_PROBE_RESULT_PATH)
    if (
        result.get("kind") != "vocabulary_transfer_probe_result_v1"
        or result.get("complete") is not True
        or result.get("decision", {}).get("status") != "vocabulary_transfer_probe_pass"
        or result.get("independent_nll_recomputation", {}).get("pass") is not True
    ):
        raise ValueError("baseline parent vocabulary-transfer result differs")
    return {
        "artifact_sha256": hash_file(PREVIOUS_PROBE_RESULT_PATH),
        "summary_sha256": result["summary_sha256"],
        "selected_role": result["decision"]["selected_composed_initializer"],
        "selected_final_bpb": result["decision"]["selected_final_bpb"],
        "selected_anchor_gap_bpb": result["decision"]["selected_anchor_gap_bpb"],
        "selected_composed_advantage_bpb": result["decision"][
            "selected_composed_advantage_bpb"
        ],
        "checkpoint_replay_count": result["independent_nll_recomputation"][
            "checkpoint_count"
        ],
    }


def initialization_identities() -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, target_pieces = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer, target_tokenizer, base_pieces, target_pieces
    )
    state = base_checkpoint_state()
    metadata = source_token_metadata(base_tokenizer)
    audits: dict[str, Any] = {}
    states: dict[str, str] = {}
    for role in BASELINE_ROLES:
        model, audit, actual_metadata = build_transferred_model(
            role,
            base_state=state,
            base_tokenizer=base_tokenizer,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=decompositions,
        )
        if actual_metadata != metadata:
            raise AssertionError("baseline source metadata differs across roles")
        audits[role] = audit.to_dict()
        states[role] = state_mapping_sha256(model.state_dict())
    return audits, states, metadata.to_dict()


def selection_rule() -> dict[str, Any]:
    return {
        "candidate_pool": list(COMPOSED_BASELINE_ROLES),
        "exact_tie_order": list(BASELINE_ROLES),
        "random_control_by_candidate": dict(RANDOM_CONTROL_BY_ROLE),
        "selection_checkpoint_step": FINAL_PROBE_STEP,
        "step_zero_or_step_fifty_can_select": False,
        "minimum_initialization_advantage_bpb": MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
        "maximum_anchor_gap_for_full_cpt_bpb": MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
        "preserve_best_qualified_tied_and_untied_pareto_roles": True,
        "requires_both_gates": True,
        "result_dependent_role_addition": False,
        "korean_specific_fallback": None,
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected_keys = {
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
        "previous_probe_evidence",
        "protocol_id",
        "resource_authorization",
        "roles",
        "role_specs",
        "schema_version",
        "selection_rule",
        "source_token_metadata",
        "status",
        "tokenizers",
        "training",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "vocabulary_transfer_baseline_closure_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status")
        != "strong_generic_roles_and_initial_states_sealed_before_loss"
    ):
        raise ValueError("baseline plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("baseline plan hash differs")
    base_commit = plan.get("git_commit_before_plan")
    if (
        not isinstance(base_commit, str)
        or len(base_commit) != 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise ValueError("baseline plan base commit differs")
    if plan["dependencies"] != dependency_identity():
        raise ValueError("baseline dependency identity differs")
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("baseline implementation identity differs")
    if plan["environment"] != current_environment():
        raise ValueError("baseline environment differs")
    tokenizers = tokenizer_identity()
    if plan["tokenizers"] != {key: tokenizers[key] for key in ("2048", "8192")}:
        raise ValueError("baseline tokenizer identity differs")
    inventories = inherited_inventories()
    if plan["inventories"] != {"8192": inventories["8192"]}:
        raise ValueError("baseline inventories differ")
    if plan["roles"] != list(BASELINE_ROLES):
        raise ValueError("baseline roles differ")
    expected_specs = {
        role: {**role_definition(role), "expected_parameters": expected_parameter_count(role)}
        for role in BASELINE_ROLES
    }
    if plan["role_specs"] != expected_specs:
        raise ValueError("baseline role specification differs")
    if (
        plan["training"] != training_contract()
        or plan["parent_anchor"] != parent_anchor()
        or plan["previous_probe_evidence"] != previous_probe_evidence()
        or plan["selection_rule"] != selection_rule()
    ):
        raise ValueError("baseline training, anchor, prior evidence, or selection differs")
    audits, states, metadata = initialization_identities()
    if (
        plan["initialization_audits"] != audits
        or plan["initial_state_sha256"] != states
        or plan["source_token_metadata"] != metadata
    ):
        raise ValueError("baseline initialization identity differs")
    resource = read_json(PARENT_RESOURCE_PATH)
    expected_resource = {
        "artifact_sha256": hash_file(PARENT_RESOURCE_PATH),
        "activation_geometry_matches_prior_8k_runs": True,
        "parameter_count_by_role": {
            role: expected_parameter_count(role) for role in BASELINE_ROLES
        },
        "parent_projection_pass": resource["projection"]["passes"],
    }
    if (
        plan["resource_authorization"] != expected_resource
        or expected_resource["parent_projection_pass"] is not True
    ):
        raise ValueError("baseline resource authorization differs")
    if plan["claim_boundary"] != {
        "actual_inference_measured": False,
        "calibration_development_only": True,
        "eeve_full_seven_stage_reproduced": False,
        "in_place_token_scale_reproduced": False,
        "korean_specific_method_evaluated": False,
        "model_seed_count": 1,
        "publication_quality_claim": False,
        "strong_generic_baseline_closure_only": True,
    }:
        raise ValueError("baseline claim boundary differs")

