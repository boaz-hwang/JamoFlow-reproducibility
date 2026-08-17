"""Sealed protocol helpers for the training-only foldable-Jamo residual screen."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bpe_quality_feasibility_core import encode_stream_to_memmap
from bpe_quality_frontier_core import (
    array_sha256,
    calibration_document_pieces,
    encode_document_chunks,
)
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
from foldable_jamo_residual_core import (
    BASELINE_ROLE_BY_ARCHITECTURE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    FINAL_PROBE_STEP,
    MAXIMUM_ANCHOR_GAP_BPB,
    MINIMUM_JAMO_ADVANTAGE_BPB,
    PROBE_STEPS,
    RESIDUAL_ROLES,
    audit_residual_assignment,
    build_foldable_model,
    expected_parameter_counts,
    exposure_counts_sha256,
    folded_dense_state,
    role_definition,
)
from vocabulary_transfer_baseline_core import state_mapping_sha256
from vocabulary_transfer_baseline_protocol import (
    OUTPUT_PATH as BASELINE_RESULT_PATH,
    PLAN_PATH as BASELINE_PLAN_PATH,
    CHECKPOINT_ROOT as BASELINE_CHECKPOINT_ROOT,
    NLL_ROOT as BASELINE_NLL_ROOT,
    base_checkpoint_state,
    parent_anchor,
    target_order,
)
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    BODY_LEARNING_RATE,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP,
    HEAD_MINIMUM_LEARNING_RATE,
    HEAD_PEAK_LEARNING_RATE,
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
    PARENT_RESOURCE_PATH,
)
from jamoflow.neural_data import build_neural_stream


PROTOCOL_ID = "jamoflow-foldable-jamo-residual-v1"
PLAN_PATH = ROOT / "data/manifests/foldable-jamo-residual-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/foldable-jamo-residual-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
FOLDED_CHECKPOINT_ROOT = ARTIFACT_ROOT / "folded-checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/foldable-jamo-residual-v1/summary.json"

IMPLEMENTATION_PATHS = (
    "docs/139-strong-vocabulary-transfer-baseline-result-and-foldable-jamo-decision.md",
    "docs/140-foldable-jamo-residual-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/compositional_quality_protocol.py",
    "scripts/compositional_token_head.py",
    "scripts/foldable_jamo_residual_core.py",
    "scripts/foldable_jamo_residual_protocol.py",
    "scripts/run_foldable_jamo_residual.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_foldable_jamo_residual_plan.py",
    "scripts/summarize_foldable_jamo_residual.py",
    "scripts/vocabulary_transfer_baseline_core.py",
    "scripts/vocabulary_transfer_baseline_protocol.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "scripts/vocabulary_transfer_probe_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "tests/test_foldable_jamo_residual.py",
    "tests/test_foldable_jamo_residual_core.py",
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
        raise TypeError("foldable residual JSON root differs")
    return value


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise ValueError("foldable residual implementation path is duplicated")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "base_checkpoint": BASE_CHECKPOINT_PATH,
        "baseline_plan": BASELINE_PLAN_PATH,
        "baseline_result": BASELINE_RESULT_PATH,
        "integrity": INTEGRITY_PATH,
        "parent_resource": PARENT_RESOURCE_PATH,
        "source": SOURCE_PATH,
    }
    return {
        key: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for key, path in paths.items()
    }


def training_contract() -> dict[str, Any]:
    inventory = inherited_inventories()[str(TARGET_VOCABULARY_SIZE)]["train"]
    sequence_count = int(inventory["full_sequence_count"])
    order = target_order(sequence_count)
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
        "head_and_residual_minimum_learning_rate": HEAD_MINIMUM_LEARNING_RATE,
        "head_and_residual_peak_learning_rate": HEAD_PEAK_LEARNING_RATE,
        "model_seed": MODEL_SEED,
        "optimizer": "AdamW",
        "order_seed": ORDER_SEED,
        "residual_schedule": "all dense and residual parameters open from step zero",
        "sequence_count": sequence_count,
        "train_microbatch_size": TRAIN_MICROBATCH_SIZE,
        "training_order_prefix_sha256": array_sha256(
            order[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE]
        ),
        "training_schedule_token_count": FINAL_PROBE_STEP
        * EFFECTIVE_BATCH_SIZE
        * 512,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay_for_matrix_parameters": WEIGHT_DECAY,
        "weight_decay_for_vector_parameters": 0.0,
    }


def scheduled_exposure_counts() -> tuple[np.ndarray, dict[str, Any]]:
    tokenizer, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="train",
        byte_limit=128_000_000,
        sequence_length=512,
    )
    inventory, memory, memory_path = encode_stream_to_memmap(
        stream.data,
        tokenizer,
        token_bytes,
        first_batch_token_count=EFFECTIVE_BATCH_SIZE * 512,
    )
    try:
        inherited = inherited_inventories()[str(TARGET_VOCABULARY_SIZE)]["train"]
        if inventory.to_dict() != inherited:
            raise ValueError("foldable residual train inventory differs")
        sequence_count = inventory.full_sequence_count
        sequences = memory[: sequence_count * 512].reshape(sequence_count, 512)
        selected = target_order(sequence_count)[: FINAL_PROBE_STEP * EFFECTIVE_BATCH_SIZE]
        counts = np.bincount(
            np.asarray(sequences[selected], dtype=np.int64).reshape(-1),
            minlength=TARGET_VOCABULARY_SIZE,
        ).astype(np.int64, copy=False)
        new_counts = counts[BASE_VOCABULARY_SIZE:]
        identity = {
            "exposure_counts_sha256": exposure_counts_sha256(counts),
            "maximum_new_row_exposure": int(new_counts.max()),
            "minimum_new_row_exposure": int(new_counts.min()),
            "scheduled_token_count": int(counts.sum()),
            "seen_new_row_count": int(np.count_nonzero(new_counts)),
            "unseen_new_row_count": int(np.count_nonzero(new_counts == 0)),
        }
        if identity["scheduled_token_count"] != training_contract()[
            "training_schedule_token_count"
        ]:
            raise AssertionError("foldable residual scheduled exposure total differs")
        return counts, identity
    finally:
        del memory
        if os.path.exists(memory_path):
            os.unlink(memory_path)


def document_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    tokenizer, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    pieces, common = calibration_document_pieces(SOURCE_PATH)
    inventory, _chunks, _documents, raw_bytes = encode_document_chunks(
        pieces, tokenizer, token_bytes
    )
    if int(raw_bytes.sum()) != common["raw_bytes"]:
        raise AssertionError("foldable residual document bytes differ")
    return common, inventory.to_dict()


def baseline_control_identities() -> dict[str, Any]:
    result = read_json(BASELINE_RESULT_PATH)
    plan = read_json(BASELINE_PLAN_PATH)
    if (
        result.get("kind") != "vocabulary_transfer_baseline_result_v1"
        or result.get("complete") is not True
        or result.get("decision", {}).get("status") != "strong_generic_baseline_pass"
        or result.get("decision", {}).get("best_untied_pareto_role")
        != BASELINE_ROLE_BY_ARCHITECTURE["untied"]
        or result.get("decision", {}).get("best_tied_pareto_role") is not None
        or result.get("independent_nll_recomputation", {}).get("pass") is not True
    ):
        raise ValueError("foldable residual baseline result differs")
    output: dict[str, Any] = {}
    for architecture, role in BASELINE_ROLE_BY_ARCHITECTURE.items():
        checkpoint_path = BASELINE_CHECKPOINT_ROOT / f"{role}-step-{FINAL_PROBE_STEP:04d}.pt"
        step_zero_nll = BASELINE_NLL_ROOT / f"{role}-step-0000.npz"
        final_nll = BASELINE_NLL_ROOT / f"{role}-step-{FINAL_PROBE_STEP:04d}.npz"
        lineage = result["artifact_lineage"][role][str(FINAL_PROBE_STEP)]
        zero_lineage = result["artifact_lineage"][role]["0"]
        if (
            lineage["checkpoint_artifact_sha256"] != hash_file(checkpoint_path)
            or lineage["nll_artifact_sha256"] != hash_file(final_nll)
            or zero_lineage["nll_artifact_sha256"] != hash_file(step_zero_nll)
            or plan["initial_state_sha256"][role]
            != result["artifact_lineage"][role]["0"]["checkpoint_state_sha256"]
        ):
            raise ValueError("foldable residual baseline artifact differs")
        output[architecture] = {
            "baseline_role": role,
            "final_checkpoint": {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "artifact_sha256": hash_file(checkpoint_path),
                "state_sha256": lineage["checkpoint_state_sha256"],
            },
            "final_contiguous_bpb": result["metrics"][role][str(FINAL_PROBE_STEP)][
                "contiguous_bpb"
            ],
            "final_nll": {
                "path": str(final_nll.relative_to(ROOT)),
                "artifact_sha256": hash_file(final_nll),
                "array_sha256": lineage["arrays"]["nll_nats"]["sha256"],
            },
            "initial_dense_state_sha256": plan["initial_state_sha256"][role],
            "step_zero_nll": {
                "path": str(step_zero_nll.relative_to(ROOT)),
                "artifact_sha256": hash_file(step_zero_nll),
                "array_sha256": zero_lineage["arrays"]["nll_nats"]["sha256"],
            },
        }
    return output


def initialization_identities() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, target_pieces = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer, target_tokenizer, base_pieces, target_pieces
    )
    counts, exposure_identity = scheduled_exposure_counts()
    base_state = base_checkpoint_state()
    baseline_plan = read_json(BASELINE_PLAN_PATH)
    identities: dict[str, Any] = {}
    assignments: dict[str, Any] = {}
    for role in RESIDUAL_ROLES:
        model, initializer_audit, assignment_audit = build_foldable_model(
            role,
            base_state=base_state,
            base_tokenizer=base_tokenizer,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=decompositions,
            exposure_counts=counts,
        )
        definition = role_definition(role)
        folded = folded_dense_state(model, role)
        folded_sha = state_mapping_sha256(folded)
        expected_folded_sha = baseline_plan["initial_state_sha256"][
            definition["base_initializer_role"]
        ]
        if (
            not model.foldable_residual.residuals_are_exact_zero()
            or folded_sha != expected_folded_sha
        ):
            raise AssertionError("foldable residual zero initialization differs")
        counts_by_role = expected_parameter_counts(role)
        actual_training = sum(parameter.numel() for parameter in model.parameters())
        if actual_training != counts_by_role["training_total"]:
            raise AssertionError("foldable residual training parameter count differs")
        identities[role] = {
            "base_initializer_audit": initializer_audit.to_dict(),
            "folded_dense_state_sha256": folded_sha,
            "training_state_sha256": state_mapping_sha256(model.state_dict()),
            "zero_initialized_residual": True,
            **counts_by_role,
        }
        assignments[role] = assignment_audit.to_dict()
        del model, folded
        gc.collect()
    return identities, assignments, exposure_identity


def selection_rule() -> dict[str, Any]:
    return {
        "architecture_order": ["untied", "tied"],
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidate_by_architecture": {
            architecture: f"{architecture}_jamo"
            for architecture in ("untied", "tied")
        },
        "controls_by_architecture": {
            architecture: [
                f"{architecture}_base",
                f"{architecture}_generic_surface",
                f"{architecture}_shuffled_jamo",
            ]
            for architecture in ("untied", "tied")
        },
        "document_bootstrap_upper_must_be_nonpositive": True,
        "maximum_anchor_gap_bpb": MAXIMUM_ANCHOR_GAP_BPB,
        "minimum_jamo_advantage_over_generic_and_shuffle_bpb": MINIMUM_JAMO_ADVANTAGE_BPB,
        "minimum_jamo_advantage_over_no_residual_base_bpb": 0.0,
        "requires_contiguous_and_document_point_advantage": True,
        "selection_checkpoint_step": FINAL_PROBE_STEP,
        "threshold_or_role_fallback": None,
    }


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool = False) -> None:
    expected_keys = {
        "assignment_audits",
        "baseline_controls",
        "claim_boundary",
        "dependencies",
        "document_common",
        "document_inventory",
        "environment",
        "exposure_identity",
        "git_commit_before_plan",
        "implementation_sha256",
        "initialization_identities",
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
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "foldable_jamo_residual_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status")
        != "same_cost_assignments_and_zero_initial_states_sealed_before_loss"
    ):
        raise ValueError("foldable residual plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("foldable residual plan hash differs")
    base_commit = plan.get("git_commit_before_plan")
    if (
        not isinstance(base_commit, str)
        or len(base_commit) != 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise ValueError("foldable residual plan base commit differs")
    if (
        plan["dependencies"] != dependency_identity()
        or plan["implementation_sha256"] != implementation_identity()
        or plan["environment"] != current_environment()
        or plan["training"] != training_contract()
        or plan["selection_rule"] != selection_rule()
    ):
        raise ValueError("foldable residual implementation or protocol differs")
    tokenizers = tokenizer_identity()
    if plan["tokenizers"] != {key: tokenizers[key] for key in ("2048", "8192")}:
        raise ValueError("foldable residual tokenizer identity differs")
    inventories = inherited_inventories()
    if plan["inventories"] != {"8192": inventories["8192"]}:
        raise ValueError("foldable residual inventory differs")
    if plan["roles"] != list(RESIDUAL_ROLES):
        raise ValueError("foldable residual role set differs")
    expected_specs = {
        role: {**role_definition(role), **expected_parameter_counts(role)}
        for role in RESIDUAL_ROLES
    }
    if plan["role_specs"] != expected_specs:
        raise ValueError("foldable residual role specification differs")
    if plan["baseline_controls"] != baseline_control_identities():
        raise ValueError("foldable residual baseline controls differ")
    if plan["parent_anchor"] != parent_anchor():
        raise ValueError("foldable residual parent anchor differs")
    common, document_inventory = document_identity()
    if (
        plan["document_common"] != common
        or plan["document_inventory"] != document_inventory
    ):
        raise ValueError("foldable residual document identity differs")
    resource = read_json(PARENT_RESOURCE_PATH)
    expected_resource = {
        "artifact_sha256": hash_file(PARENT_RESOURCE_PATH),
        "deployed_activation_geometry_matches_dense_8k": True,
        "deployed_parameter_count_by_role": {
            role: expected_parameter_counts(role)["deployed"] for role in RESIDUAL_ROLES
        },
        "parent_projection_pass": resource["projection"]["passes"],
        "training_parameter_count_by_role": {
            role: expected_parameter_counts(role)["training_total"] for role in RESIDUAL_ROLES
        },
    }
    if plan["resource_authorization"] != expected_resource:
        raise ValueError("foldable residual resource authorization differs")
    if plan["claim_boundary"] != {
        "actual_inference_measured": False,
        "calibration_development_only": True,
        "deployed_residual_module_present": False,
        "fresh_equal_history_quality": False,
        "korean_specific_method_screen": True,
        "model_seed_count": 1,
        "publication_quality_claim": False,
        "training_overhead_is_measured_not_free": True,
    }:
        raise ValueError("foldable residual claim boundary differs")
    if verify_derived:
        identities, assignments, exposure_identity = initialization_identities()
        if (
            plan["initialization_identities"] != identities
            or plan["assignment_audits"] != assignments
            or plan["exposure_identity"] != exposure_identity
        ):
            raise ValueError("foldable residual derived identity differs")
    else:
        if (
            set(plan["initialization_identities"]) != set(RESIDUAL_ROLES)
            or set(plan["assignment_audits"]) != set(RESIDUAL_ROLES)
            or set(plan["exposure_identity"])
            != {
                "exposure_counts_sha256",
                "maximum_new_row_exposure",
                "minimum_new_row_exposure",
                "scheduled_token_count",
                "seen_new_row_count",
                "unseen_new_row_count",
            }
        ):
            raise ValueError("foldable residual stored derived identity differs")
