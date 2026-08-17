"""Sealed protocol helpers for fresh Korean vocabulary adaptation."""

from __future__ import annotations

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
    raw_target_bytes_by_sequence,
)
from compositional_head_core import ROLE_SPECS, build_model
from compositional_head_preflight_protocol import (
    ROOT,
    current_environment,
    hash_file,
    load_tokenizers,
    tokenizer_identity,
)
from fresh_vocabulary_adaptation_core import (
    BASE_VOCABULARY_SIZE,
    BODY_LEARNING_RATE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    DEPLOYMENT_TIE_ORDER,
    EFFECTIVE_BATCH_SIZE,
    EIGHT_K_ROLES,
    EVALUATION_BATCH_BY_VOCABULARY,
    GRADIENT_CLIP,
    HEAD_MINIMUM_LEARNING_RATE,
    HEAD_PEAK_LEARNING_RATE,
    METHOD_MINIMUM_ADVANTAGE_BPB,
    QUALITY_NONINFERIORITY_MARGIN_BPB,
    ROLES,
    SEQUENCE_LENGTH,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_BY_VOCABULARY,
    WARMUP_RAW_FRACTION,
    WEIGHT_DECAY,
    batch_raw_target_bytes,
    inplace_stage_contract,
    role_definition,
)
from hplt3_fresh_adaptation_protocol import validate_seal_envelope
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import (
    build_canonical_bpe_decomposition_table,
    build_transferred_model,
    expected_parameter_count,
    state_mapping_sha256,
)
from vocabulary_transfer_probe_protocol import (
    BASE_CHECKPOINT_PATH,
    PARENT_PLAN_PATH,
    PARENT_RESULT_PATH,
    base_checkpoint_state,
)

from jamoflow.neural_data import NeuralStream, build_neural_stream

PROTOCOL_ID = "jamoflow-fresh-vocabulary-adaptation-one-seed-v1"
PLAN_PATH = ROOT / "data/manifests/fresh-vocabulary-adaptation-one-seed-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/fresh-vocabulary-adaptation-one-seed-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/fresh-vocabulary-adaptation-one-seed-v1/summary.json"

FRESH_MANIFEST_PATH = ROOT / "data/manifests/hplt3-korean-vocab-adaptation-v1.json"
FRESH_SEAL_PATH = ROOT / "data/seals/hplt3-korean-vocab-adaptation-v1.json"
FRESH_SOURCE_PATH = ROOT / "data/processed/hplt3-korean-vocab-adaptation-v1/ko.jsonl"
UPDATE_AUDIT_RESULT_PATH = (
    ROOT / "results/foldable-multihash-update-audit-v4/summary.json"
)
MECHANISM_RESULT_PATH = ROOT / "results/foldable-multihash-mechanism-v1/summary.json"

INITIALIZER_ROLE = "untied_uniform_in_byte_weighted_out"
TRAIN_BYTES = 128_000_000
CALIBRATION_BYTES = 8_000_000

IMPLEMENTATION_PATHS = (
    "docs/142-fable5-final-retrospective-and-current-direction.md",
    "docs/150-foldable-multihash-mechanism-result-and-optimizer-pivot.md",
    "docs/151-fresh-vocabulary-adaptation-data-protocol.md",
    "docs/152-fresh-vocabulary-adaptation-data-result.md",
    "docs/153-fresh-vocabulary-adaptation-one-seed-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/foldable_multihash_mechanism_core.py",
    "scripts/fresh_vocabulary_adaptation_core.py",
    "scripts/fresh_vocabulary_adaptation_protocol.py",
    "scripts/hplt3_fresh_adaptation_protocol.py",
    "scripts/run_fresh_vocabulary_adaptation.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_fresh_vocabulary_adaptation_plan.py",
    "scripts/summarize_fresh_vocabulary_adaptation.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "scripts/vocabulary_transfer_probe_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_fresh_vocabulary_adaptation_core.py",
    "tests/test_fresh_vocabulary_adaptation_protocol.py",
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
        raise TypeError(f"fresh-adaptation JSON root differs: {path}")
    return value


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("fresh-adaptation implementation list is duplicated")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "base_checkpoint": BASE_CHECKPOINT_PATH,
        "fresh_manifest": FRESH_MANIFEST_PATH,
        "fresh_output": FRESH_SOURCE_PATH,
        "fresh_seal": FRESH_SEAL_PATH,
        "mechanism_result": MECHANISM_RESULT_PATH,
        "parent_model_plan": PARENT_PLAN_PATH,
        "parent_model_result": PARENT_RESULT_PATH,
        "update_audit_result": UPDATE_AUDIT_RESULT_PATH,
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in paths.items()
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verified_fresh_streams() -> dict[str, NeuralStream]:
    seal = read_json(FRESH_SEAL_PATH)
    validate_seal_envelope(seal)
    payload = seal["payload"]
    if (
        hash_file(FRESH_SOURCE_PATH) != payload["output"]["sha256"]
        or FRESH_SOURCE_PATH.stat().st_size != payload["output"]["bytes"]
    ):
        raise ValueError("fresh-adaptation source differs from its seal")
    output: dict[str, NeuralStream] = {}
    for split, byte_limit in (
        ("train", TRAIN_BYTES),
        ("calibration", CALIBRATION_BYTES),
    ):
        stream = build_neural_stream(
            FRESH_SOURCE_PATH,
            language="ko",
            split=split,
            byte_limit=byte_limit,
            sequence_length=SEQUENCE_LENGTH,
        )
        row = payload["splits"][split]
        if (
            len(stream.data) != row["stream_bytes"]
            or stream.sequence_count != row["sequence_count"]
            or stream.available_bytes != row["available_stream_bytes"]
            or stream.selected_records != row["selected_document_count"]
            or _sha256_bytes(stream.data) != row["stream_sha256"]
        ):
            raise ValueError(f"fresh-adaptation {split} stream differs")
        output[split] = stream
    return output


def _cleanup_memmap(memory: np.memmap, path: str) -> None:
    del memory
    if os.path.exists(path):
        os.unlink(path)


def inventory_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    streams = verified_fresh_streams()
    tokenizers = load_tokenizers()
    pieces, document_common = calibration_document_pieces(FRESH_SOURCE_PATH)
    inventories: dict[str, Any] = {}
    for vocabulary_size in (BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE):
        tokenizer, token_bytes = tokenizers[vocabulary_size]
        train_inventory, train_memory, train_path = encode_stream_to_memmap(
            streams["train"].data,
            tokenizer,
            token_bytes,
            first_batch_token_count=EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH,
        )
        calibration_inventory, calibration_memory, calibration_path = (
            encode_stream_to_memmap(
                streams["calibration"].data,
                tokenizer,
                token_bytes,
                first_batch_token_count=(
                    EVALUATION_BATCH_BY_VOCABULARY[vocabulary_size] * SEQUENCE_LENGTH
                ),
            )
        )
        try:
            train_count = int(train_inventory.full_sequence_count)
            train_sequences = train_memory[: train_count * SEQUENCE_LENGTH].reshape(
                train_count, SEQUENCE_LENGTH
            )
            train_raw = raw_target_bytes_by_sequence(train_sequences, token_bytes)
            batches = batch_raw_target_bytes(train_raw)
            document_inventory, _chunks, _documents, document_raw = (
                encode_document_chunks(pieces, tokenizer, token_bytes)
            )
            inventories[str(vocabulary_size)] = {
                "train_stream": {
                    **streams["train"].metadata(),
                    "sha256": _sha256_bytes(streams["train"].data),
                },
                "calibration_stream": {
                    **streams["calibration"].metadata(),
                    "sha256": _sha256_bytes(streams["calibration"].data),
                },
                "train_tokens": train_inventory.to_dict(),
                "calibration_tokens": calibration_inventory.to_dict(),
                "train_raw_target_bytes_sha256": array_sha256(train_raw),
                "optimizer_batch_raw_target_bytes_sha256": array_sha256(batches),
                "total_optimizer_steps": len(batches),
                "document_tokens": document_inventory.to_dict(),
                "document_raw_bytes_sha256": array_sha256(document_raw),
            }
            if vocabulary_size == TARGET_VOCABULARY_SIZE:
                inventories[str(vocabulary_size)]["inplace_stage"] = (
                    inplace_stage_contract(train_raw)
                )
        finally:
            _cleanup_memmap(train_memory, train_path)
            _cleanup_memmap(calibration_memory, calibration_path)
    if (
        inventories[str(BASE_VOCABULARY_SIZE)]["document_raw_bytes_sha256"]
        != inventories[str(TARGET_VOCABULARY_SIZE)]["document_raw_bytes_sha256"]
    ):
        raise AssertionError("fresh-adaptation document denominators differ")
    return inventories, document_common


def initialization_contract() -> dict[str, Any]:
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, target_pieces = tokenizers[TARGET_VOCABULARY_SIZE]
    decompositions = build_canonical_bpe_decomposition_table(
        base_tokenizer,
        target_tokenizer,
        base_pieces,
        target_pieces,
    )
    base_state = base_checkpoint_state()
    base_model = build_model("dense_v2048")
    base_model.load_state_dict(base_state, strict=True)
    transferred, audit = build_transferred_model(
        INITIALIZER_ROLE,
        base_state=base_state,
        base_pieces=base_pieces,
        target_pieces=target_pieces,
        decompositions=decompositions,
    )
    base_count = model_parameter_count(base_model)
    target_count = model_parameter_count(transferred)
    if base_count != ROLE_SPECS[
        "dense_v2048"
    ].expected_parameters or target_count != expected_parameter_count(INITIALIZER_ROLE):
        raise AssertionError("fresh-adaptation parameter contract differs")
    return {
        "base_checkpoint": {
            "path": str(BASE_CHECKPOINT_PATH.relative_to(ROOT)),
            "artifact_sha256": hash_file(BASE_CHECKPOINT_PATH),
            "state_sha256": state_mapping_sha256(base_state),
        },
        "dense2k_initial_state_sha256": state_mapping_sha256(base_model.state_dict()),
        "dense8k_initializer_role": INITIALIZER_ROLE,
        "dense8k_initialization_audit": audit.to_dict(),
        "dense8k_initial_state_sha256": state_mapping_sha256(transferred.state_dict()),
        "parameter_count_by_role": {
            role: base_count if role == "dense2k_joint" else target_count
            for role in ROLES
        },
    }


def training_contract(inventories: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role in ROLES:
        definition = role_definition(role)
        vocabulary_size = int(definition["vocabulary_size"])
        inventory = inventories[str(vocabulary_size)]
        row = {
            "adamw_beta1": 0.9,
            "adamw_beta2": 0.95,
            "adamw_epsilon": 1e-8,
            "body_learning_rate": BODY_LEARNING_RATE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_BY_VOCABULARY[vocabulary_size],
            "gradient_clip": GRADIENT_CLIP,
            "head_minimum_learning_rate": HEAD_MINIMUM_LEARNING_RATE,
            "head_peak_learning_rate": HEAD_PEAK_LEARNING_RATE,
            "optimizer": "AdamW",
            "ordering": "sealed_rank_order_contiguous_no_permutation",
            "raw_stream_bytes": TRAIN_BYTES,
            "sequence_count": inventory["train_tokens"]["full_sequence_count"],
            "target_raw_bytes": inventory["train_tokens"]["predicted_target_raw_bytes"],
            "token_ids_sha256": inventory["train_tokens"]["token_ids_sha256"],
            "total_optimizer_steps": inventory["total_optimizer_steps"],
            "train_microbatch_size": TRAIN_MICROBATCH_BY_VOCABULARY[vocabulary_size],
            "vocabulary_size": vocabulary_size,
            "warmup_raw_fraction": WARMUP_RAW_FRACTION,
            "weight_decay_for_matrix_parameters": WEIGHT_DECAY,
            "weight_decay_for_vector_parameters": 0.0,
        }
        if role == "dense8k_inplace_two_stage":
            row["inplace_stage"] = inventory["inplace_stage"]
            row["stage_one_learning_rate"] = "raw_progress_warmup_then_constant_peak"
            row["stage_two_learning_rate"] = "raw_progress_rewarmup_then_cosine"
            row["optimizer_reinitialized_at_stage_two"] = True
            row["copied_input_and_output_rows_restored_after_every_stage_one_step"] = (
                True
            )
        output[role] = row
    return output


def decision_contract() -> dict[str, Any]:
    return {
        "actual_preflight_requires_any_quality_qualified_dense8k": True,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "deployment_candidate_pool": list(EIGHT_K_ROLES),
        "deployment_exact_tie_order": list(DEPLOYMENT_TIE_ORDER),
        "deployment_selection_metric": "lowest_fresh_calibration_document_bpb",
        "method_candidate": "dense8k_update_geometry",
        "method_controls": ["dense8k_standard_joint", "dense8k_inplace_two_stage"],
        "method_minimum_advantage_bpb": METHOD_MINIMUM_ADVANTAGE_BPB,
        "method_requires_both_control_comparisons": True,
        "quality_reference": "dense2k_joint",
        "quality_noninferiority_margin_bpb": QUALITY_NONINFERIORITY_MARGIN_BPB,
        "selection_uses_fresh_calibration_only": True,
        "sealed_final_test_used": False,
        "actual_latency_used": False,
    }


def build_plan(git_commit_before_plan: str) -> dict[str, Any]:
    inventories, document_common = inventory_contract()
    plan: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_adaptation_one_seed_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_fresh_training",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "environment": current_environment(),
        "implementation_sha256": implementation_identity(),
        "tokenizers": {
            key: tokenizer_identity()[key]
            for key in (str(BASE_VOCABULARY_SIZE), str(TARGET_VOCABULARY_SIZE))
        },
        "roles": {role: role_definition(role) for role in ROLES},
        "initialization": initialization_contract(),
        "inventories": inventories,
        "document_common": document_common,
        "training": training_contract(inventories),
        "decision": decision_contract(),
        "claim_boundary": {
            "development_one_seed": True,
            "publication_claim": False,
            "method_claim_requires_fresh_multiseed_confirmation": True,
            "efficiency_success_requires_trained_controlled_and_free_actual_e2e_ge_10pct": True,
            "analytical_or_random_weight_latency_is_not_success": True,
            "parameter_and_memory_increase_must_be_reported": True,
        },
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected = {
        "claim_boundary",
        "decision",
        "dependencies",
        "document_common",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "initialization",
        "inventories",
        "kind",
        "output_path",
        "plan_sha256",
        "protocol_id",
        "roles",
        "schema_version",
        "status",
        "tokenizers",
        "training",
    }
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("kind") != "fresh_vocabulary_adaptation_one_seed_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_fresh_training"
        or plan.get("output_path") != str(OUTPUT_PATH.relative_to(ROOT))
        or plan.get("roles") != {role: role_definition(role) for role in ROLES}
        or plan.get("decision") != decision_contract()
    ):
        raise ValueError("fresh-adaptation plan contract differs")
    unsigned = dict(plan)
    receipt = unsigned.pop("plan_sha256", None)
    if canonical_sha256(unsigned) != receipt:
        raise ValueError("fresh-adaptation plan hash differs")
    if plan.get("training") != training_contract(plan["inventories"]):
        raise ValueError("fresh-adaptation training contract differs")
    if verify_derived:
        inventories, document_common = inventory_contract()
        expected_tokenizers = {
            key: tokenizer_identity()[key]
            for key in (str(BASE_VOCABULARY_SIZE), str(TARGET_VOCABULARY_SIZE))
        }
        if (
            plan.get("dependencies") != dependency_identity()
            or plan.get("environment") != current_environment()
            or plan.get("implementation_sha256") != implementation_identity()
            or plan.get("tokenizers") != expected_tokenizers
            or plan.get("initialization") != initialization_contract()
            or plan.get("inventories") != inventories
            or plan.get("document_common") != document_common
        ):
            raise ValueError("fresh-adaptation derived plan identity differs")
