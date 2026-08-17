"""Contracts for the fixed balanced-200M trained scale screen."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from large_scale_training_feasibility_core import _patch_matrices
from large_scale_training_feasibility_core import (
    validate_summary as validate_resource_summary,
)
from scale_schedule_extrapolation_core import (
    CONTINUATION_BYTES,
    EXPECTED_PARAMETERS,
    GLOBAL_POSITION_LIMIT,
    INNER_REPETITIONS,
    MEASURED_PROMPTS,
    MODEL_SEED,
    PROMPT_BYTES,
    ROOT,
    SCHEDULE_ORDER,
    SESSION_ORDER,
    WARMUP_PROMPTS,
    array_sha256,
    canonical_sha256,
    is_git_commit,
    is_sha256,
    large_scale_model_spec,
    validate_scale_schedule_summary,
)
from scale_schedule_extrapolation_core import (
    PLAN_PATH as SCALE_PLAN_PATH,
)
from scale_schedule_extrapolation_core import (
    validate_plan as validate_scale_plan,
)

from jamoflow.hplt3 import hash_file
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import compact_whitespace_mask

PROTOCOL_ID = "jamoflow-balanced-200m-trained-screen-v1"
PLAN_PATH = ROOT / "data/manifests/balanced-200m-trained-screen-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/balanced-200m-trained-screen-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".preflight-active"
TRAINING_ACTIVE_PATH = ARTIFACT_ROOT / ".training-active"
PREFLIGHT_OUTPUT_PATH = ROOT / "results/balanced-200m-trained-screen-v1/preflight.json"
TRAINING_OUTPUT_PATH = (
    ROOT / "results/balanced-200m-trained-screen-v1/training-summary.json"
)
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
INTEGRITY_PATH = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
RESOURCE_SUMMARY_PATH = (
    ROOT / "results/large-scale-training-feasibility-v1/summary.json"
)
SCALE_SUMMARY_PATH = ROOT / "results/scale-schedule-extrapolation-v1/summary.json"
TARGET = 200
EXPECTED_PARAMETER_COUNT = EXPECTED_PARAMETERS[TARGET]
ROLE_ORDER = SCHEDULE_ORDER
SEQUENCE_LENGTH = 512
NOMINAL_TRAIN_BYTES = 128_000_000
AVAILABLE_TRAIN_SEQUENCES = NOMINAL_TRAIN_BYTES // SEQUENCE_LENGTH
MICROBATCH_SEQUENCES = 8
GRADIENT_ACCUMULATION_STEPS = 4
EFFECTIVE_BATCH_SEQUENCES = MICROBATCH_SEQUENCES * GRADIENT_ACCUMULATION_STEPS
TRAIN_SEQUENCES = (
    AVAILABLE_TRAIN_SEQUENCES - AVAILABLE_TRAIN_SEQUENCES % EFFECTIVE_BATCH_SEQUENCES
)
TRAIN_BYTES = TRAIN_SEQUENCES * SEQUENCE_LENGTH
TOTAL_UPDATES = TRAIN_SEQUENCES // EFFECTIVE_BATCH_SEQUENCES
CALIBRATION_BYTES = 8_000_000
TRAINING_ORDER_SEED = 20260901
LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE = 3e-5
WARMUP_LR_UPDATES = 100
BETAS = (0.9, 0.95)
EPSILON = 1e-8
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
PREFLIGHT_WARMUP_UPDATES = 1
PREFLIGHT_MEASUREMENT_UPDATES = 2
PREFLIGHT_EXAMPLES = (
    PREFLIGHT_WARMUP_UPDATES + PREFLIGHT_MEASUREMENT_UPDATES
) * EFFECTIVE_BATCH_SEQUENCES
MAXIMUM_RECOMMENDED_MEMORY_FRACTION = 0.75
MAXIMUM_HOURS_PER_ROLE = 12.0
MAXIMUM_HOURS_PER_PAIR = 24.0
QUALITY_MARGIN_BPB = 0.010
EVALUATION_BATCH_SEQUENCES = 8
TRAINING_LOG_EVERY_UPDATES = 100
TRAINED_TIMING_BOOTSTRAP_REPETITIONS = 10_000
TRAINED_TIMING_BOOTSTRAP_SEED = 20260902
TRAINED_TIMING_MINIMUM_POINT_REDUCTION = 0.0
TRAINED_TIMING_MINIMUM_LOWER_BOUND = 0.0
TRAINED_TIMING_MINIMUM_POSITIVE_PROMPTS = 15

IMPLEMENTATION_PATHS = (
    "docs/194-global-heavy-result-and-trained-scale-pivot.md",
    "docs/195-balanced-200m-trained-screen-protocol.md",
    "pyproject.toml",
    "scripts/balanced_200m_trained_core.py",
    "scripts/large_scale_training_feasibility_core.py",
    "scripts/run_balanced_200m_preflight.py",
    "scripts/run_balanced_200m_training.py",
    "scripts/seal_balanced_200m_trained_plan.py",
    "scripts/verify_balanced_200m_preflight.py",
    "scripts/verify_balanced_200m_training.py",
    "scripts/scale_schedule_extrapolation_core.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "tests/test_balanced_200m_trained_screen.py",
)


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def worker_report_path(role: str) -> Path:
    if role not in ROLE_ORDER:
        raise ValueError("balanced-200M role differs")
    return ARTIFACT_ROOT / f"preflight-{role}.json"


def training_report_path(role: str) -> Path:
    if role not in ROLE_ORDER:
        raise ValueError("balanced-200M training role differs")
    return ARTIFACT_ROOT / f"training-{role}.json"


def checkpoint_path(role: str) -> Path:
    if role not in ROLE_ORDER:
        raise ValueError("balanced-200M checkpoint role differs")
    return ARTIFACT_ROOT / f"trained-{role}.pt"


def calibration_nll_path(role: str) -> Path:
    if role not in ROLE_ORDER:
        raise ValueError("balanced-200M NLL role differs")
    return ARTIFACT_ROOT / f"calibration-{role}-nll.npz"


def _stream_arrays(
    split: str, byte_limit: int
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split=split,
        byte_limit=byte_limit,
        sequence_length=SEQUENCE_LENGTH,
    )
    inputs, boundaries = stream_arrays(
        stream.data, stream.codepoint_boundaries, stream.sequence_length
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    matrices = _patch_matrices(boundaries, whitespace)
    return (
        np.ascontiguousarray(inputs),
        {role: np.ascontiguousarray(matrices[role]) for role in ROLE_ORDER},
        stream.metadata(),
    )


def training_arrays() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    inputs, matrices, _ = _stream_arrays("train", NOMINAL_TRAIN_BYTES)
    if len(inputs) != AVAILABLE_TRAIN_SEQUENCES:
        raise ValueError("balanced-200M nominal train stream differs")
    order = np.random.default_rng(TRAINING_ORDER_SEED).permutation(len(inputs))
    order = np.ascontiguousarray(order[:TRAIN_SEQUENCES], dtype=np.int64)
    if len(np.unique(order)) != TRAIN_SEQUENCES:
        raise ValueError("balanced-200M training order is not unique")
    return inputs, matrices, order


def preflight_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    inputs, matrices, _ = _stream_arrays("train", PREFLIGHT_EXAMPLES * SEQUENCE_LENGTH)
    if inputs.shape != (PREFLIGHT_EXAMPLES, SEQUENCE_LENGTH):
        raise ValueError("balanced-200M preflight stream differs")
    return inputs, matrices


def calibration_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    inputs, matrices, _ = _stream_arrays("calibration", CALIBRATION_BYTES)
    return inputs, matrices


def data_contract() -> dict[str, Any]:
    inputs, matrices, order = training_arrays()
    calibration_inputs, calibration_matrices = calibration_arrays()
    return {
        "source_path": SOURCE_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": hash_file(SOURCE_PATH),
        "integrity_path": INTEGRITY_PATH.relative_to(ROOT).as_posix(),
        "integrity_sha256": hash_file(INTEGRITY_PATH),
        "sequence_length": SEQUENCE_LENGTH,
        "nominal_train_bytes": NOMINAL_TRAIN_BYTES,
        "available_train_sequences": AVAILABLE_TRAIN_SEQUENCES,
        "used_train_sequences": TRAIN_SEQUENCES,
        "used_train_bytes": TRAIN_BYTES,
        "dropped_train_sequences": AVAILABLE_TRAIN_SEQUENCES - TRAIN_SEQUENCES,
        "inputs_array_sha256": array_sha256(inputs),
        "training_order_seed": TRAINING_ORDER_SEED,
        "training_order_array_sha256": array_sha256(order),
        "training_patch_matrix_sha256": {
            role: array_sha256(matrices[role]) for role in ROLE_ORDER
        },
        "preflight_examples": PREFLIGHT_EXAMPLES,
        "preflight_selection": (
            "first 96 complete 512-byte sequences in canonical train stream"
        ),
        "preflight_inputs_array_sha256": array_sha256(inputs[:PREFLIGHT_EXAMPLES]),
        "preflight_patch_matrix_sha256": {
            role: array_sha256(matrices[role][:PREFLIGHT_EXAMPLES])
            for role in ROLE_ORDER
        },
        "calibration_bytes": CALIBRATION_BYTES,
        "calibration_examples": len(calibration_inputs),
        "calibration_inputs_array_sha256": array_sha256(calibration_inputs),
        "calibration_patch_matrix_sha256": {
            role: array_sha256(calibration_matrices[role]) for role in ROLE_ORDER
        },
        "historical_test_or_final_metric_used": False,
    }


def optimizer_contract() -> dict[str, Any]:
    return {
        "name": "torch.optim.AdamW",
        "parameter_dtype": "float32",
        "microbatch_sequences": MICROBATCH_SEQUENCES,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_sequences": EFFECTIVE_BATCH_SEQUENCES,
        "source_bytes_per_update": EFFECTIVE_BATCH_SEQUENCES * SEQUENCE_LENGTH,
        "total_updates": TOTAL_UPDATES,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "warmup_lr_updates": WARMUP_LR_UPDATES,
        "betas": list(BETAS),
        "epsilon": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "evaluation_batch_sequences": EVALUATION_BATCH_SEQUENCES,
        "training_log_every_updates": TRAINING_LOG_EVERY_UPDATES,
    }


def trained_timing_contract() -> dict[str, Any]:
    return {
        "session_order": list(SESSION_ORDER),
        "warmup_prompts": WARMUP_PROMPTS,
        "measured_prompts": MEASURED_PROMPTS,
        "inner_repetitions": INNER_REPETITIONS,
        "prompt_bytes": PROMPT_BYTES,
        "controlled_continuation_bytes": CONTINUATION_BYTES,
        "bootstrap_repetitions": TRAINED_TIMING_BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": TRAINED_TIMING_BOOTSTRAP_SEED,
        "minimum_point_reduction_exclusive": TRAINED_TIMING_MINIMUM_POINT_REDUCTION,
        "minimum_bootstrap_lower_exclusive": TRAINED_TIMING_MINIMUM_LOWER_BOUND,
        "minimum_positive_prompts": TRAINED_TIMING_MINIMUM_POSITIVE_PROMPTS,
        "both_roles_use_distinct_trained_checkpoints": True,
        "timing_starts_only_after_quality_and_independent_replay_pass": True,
        "ten_percent_gate_required": False,
    }


def project_preflight(update_seconds: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(tuple(update_seconds), dtype=np.float64)
    if (
        values.shape != (PREFLIGHT_MEASUREMENT_UPDATES,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("balanced-200M preflight timings differ")
    median = float(np.median(values))
    return {
        "measurement_update_seconds": values.tolist(),
        "median_update_seconds": median,
        "projected_updates": TOTAL_UPDATES,
        "projected_train_bytes": TRAIN_BYTES,
        "projected_hours": median * TOTAL_UPDATES / 3600,
    }


def build_plan(
    *,
    git_commit_before_plan: str,
    model_state_sha256: str,
    data: Mapping[str, Any],
    environment: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    resource_summary: Mapping[str, Any],
    scale_plan: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
) -> dict[str, Any]:
    validate_resource_summary(resource_summary)
    validate_scale_plan(scale_plan, verify_implementation=False)
    validate_scale_schedule_summary(scale_summary)
    if (
        resource_summary.get("aggregate", {})
        .get("by_target", {})
        .get("200", {})
        .get("standard", {})
        .get("pair_resource_pass")
        is not True
        or scale_summary.get("aggregate", {})
        .get("rows", {})
        .get("200", {})
        .get("median_reduction")
        != 0.07217533845984225
        or scale_summary.get("plan_sha256") != scale_plan.get("plan_sha256")
        or scale_summary.get("plan_artifact_sha256") != hash_file(SCALE_PLAN_PATH)
        or not is_sha256(model_state_sha256)
    ):
        raise ValueError("balanced-200M upstream evidence differs")
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_trained_screen_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_batch8_preflight_and_training",
        "git_commit_before_plan": git_commit_before_plan,
        "model": {
            "target_millions": TARGET,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "spec": large_scale_model_spec(TARGET, 86).to_dict(),
            "model_seed": MODEL_SEED,
            "global_position_limit": GLOBAL_POSITION_LIMIT,
            "model_state_sha256": model_state_sha256,
        },
        "roles": {
            "order": list(ROLE_ORDER),
            "c86": {"policy": "causal_codepoint_grid", "patch_count": 86},
            "w72": {"policy": "causal_whitespace_grid", "patch_count": 72},
        },
        "data": dict(data),
        "optimizer": optimizer_contract(),
        "preflight": {
            "warmup_updates": PREFLIGHT_WARMUP_UPDATES,
            "measurement_updates": PREFLIGHT_MEASUREMENT_UPDATES,
            "maximum_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
            "maximum_hours_per_role": MAXIMUM_HOURS_PER_ROLE,
            "maximum_hours_per_pair": MAXIMUM_HOURS_PER_PAIR,
        },
        "quality_gate": {
            "calibration_bpb_margin": QUALITY_MARGIN_BPB,
            "seed_count": 1,
            "historical_test_used_for_gate": False,
            "actual_timing_requires_quality_pass": True,
        },
        "trained_timing_gate": trained_timing_contract(),
        "upstream": {
            "resource_summary_path": RESOURCE_SUMMARY_PATH.relative_to(ROOT).as_posix(),
            "resource_summary_artifact_sha256": hash_file(RESOURCE_SUMMARY_PATH),
            "resource_summary_sha256": resource_summary["summary_sha256"],
            "scale_plan_path": SCALE_PLAN_PATH.relative_to(ROOT).as_posix(),
            "scale_plan_artifact_sha256": hash_file(SCALE_PLAN_PATH),
            "scale_plan_sha256": scale_plan["plan_sha256"],
            "scale_summary_path": SCALE_SUMMARY_PATH.relative_to(ROOT).as_posix(),
            "scale_summary_artifact_sha256": hash_file(SCALE_SUMMARY_PATH),
            "scale_summary_sha256": scale_summary["summary_sha256"],
        },
        "environment": dict(environment),
        "implementation_sha256": dict(implementation_sha256),
        "outputs": {
            "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "training_active_path": TRAINING_ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "preflight_summary_path": PREFLIGHT_OUTPUT_PATH.relative_to(
                ROOT
            ).as_posix(),
            "training_summary_path": TRAINING_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "training_reports": {
                role: training_report_path(role).relative_to(ROOT).as_posix()
                for role in ROLE_ORDER
            },
            "checkpoints": {
                role: checkpoint_path(role).relative_to(ROOT).as_posix()
                for role in ROLE_ORDER
            },
            "calibration_nll": {
                role: calibration_nll_path(role).relative_to(ROOT).as_posix()
                for role in ROLE_ORDER
            },
        },
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "training_starts_only_after_committed_preflight_pass": True,
            "quality_claimed_before_training": False,
        },
    }
    value = {**payload, "plan_sha256": canonical_sha256(payload)}
    validate_plan(value, current_environment=environment, verify_implementation=False)
    return value


def validate_plan(
    value: Mapping[str, Any],
    *,
    current_environment: Mapping[str, Any] | None = None,
    verify_implementation: bool = True,
) -> None:
    expected = {
        "claim_boundary",
        "data",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "model",
        "optimizer",
        "outputs",
        "plan_sha256",
        "preflight",
        "protocol_id",
        "quality_gate",
        "roles",
        "schema_version",
        "status",
        "trained_timing_gate",
        "upstream",
    }
    if set(value) != expected:
        raise ValueError("balanced-200M plan schema differs")
    payload = dict(value)
    claimed = payload.pop("plan_sha256")
    if (
        value["schema_version"] != 1
        or value["kind"] != "balanced_200m_trained_screen_plan_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or value["status"] != "sealed_before_batch8_preflight_and_training"
        or not is_git_commit(value["git_commit_before_plan"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value["optimizer"] != optimizer_contract()
        or set(value["model"])
        != {
            "expected_parameter_count",
            "global_position_limit",
            "model_seed",
            "model_state_sha256",
            "spec",
            "target_millions",
        }
        or value["model"].get("target_millions") != TARGET
        or value["model"].get("expected_parameter_count") != EXPECTED_PARAMETER_COUNT
        or value["model"].get("spec") != large_scale_model_spec(TARGET, 86).to_dict()
        or value["model"].get("model_seed") != MODEL_SEED
        or value["model"].get("global_position_limit") != GLOBAL_POSITION_LIMIT
        or not is_sha256(value["model"].get("model_state_sha256"))
        or value["roles"]
        != {
            "order": list(ROLE_ORDER),
            "c86": {"policy": "causal_codepoint_grid", "patch_count": 86},
            "w72": {"policy": "causal_whitespace_grid", "patch_count": 72},
        }
        or value["data"].get("used_train_sequences") != TRAIN_SEQUENCES
        or value["data"].get("used_train_bytes") != TRAIN_BYTES
        or value["data"].get("historical_test_or_final_metric_used") is not False
        or value["preflight"]
        != {
            "warmup_updates": PREFLIGHT_WARMUP_UPDATES,
            "measurement_updates": PREFLIGHT_MEASUREMENT_UPDATES,
            "maximum_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
            "maximum_hours_per_role": MAXIMUM_HOURS_PER_ROLE,
            "maximum_hours_per_pair": MAXIMUM_HOURS_PER_PAIR,
        }
        or value["quality_gate"]
        != {
            "calibration_bpb_margin": QUALITY_MARGIN_BPB,
            "seed_count": 1,
            "historical_test_used_for_gate": False,
            "actual_timing_requires_quality_pass": True,
        }
        or value["trained_timing_gate"] != trained_timing_contract()
        or value["outputs"]
        != {
            "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "training_active_path": TRAINING_ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "preflight_summary_path": PREFLIGHT_OUTPUT_PATH.relative_to(
                ROOT
            ).as_posix(),
            "training_summary_path": TRAINING_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "training_reports": {
                role: training_report_path(role).relative_to(ROOT).as_posix()
                for role in ROLE_ORDER
            },
            "checkpoints": {
                role: checkpoint_path(role).relative_to(ROOT).as_posix()
                for role in ROLE_ORDER
            },
            "calibration_nll": {
                role: calibration_nll_path(role).relative_to(ROOT).as_posix()
                for role in ROLE_ORDER
            },
        }
        or value["claim_boundary"]
        != {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "training_starts_only_after_committed_preflight_pass": True,
            "quality_claimed_before_training": False,
        }
    ):
        raise ValueError("balanced-200M plan identity differs")
    data = value["data"]
    expected_data_keys = {
        "available_train_sequences",
        "calibration_bytes",
        "calibration_examples",
        "calibration_inputs_array_sha256",
        "calibration_patch_matrix_sha256",
        "dropped_train_sequences",
        "historical_test_or_final_metric_used",
        "inputs_array_sha256",
        "integrity_path",
        "integrity_sha256",
        "nominal_train_bytes",
        "preflight_examples",
        "preflight_inputs_array_sha256",
        "preflight_patch_matrix_sha256",
        "preflight_selection",
        "sequence_length",
        "source_path",
        "source_sha256",
        "training_order_array_sha256",
        "training_order_seed",
        "training_patch_matrix_sha256",
        "used_train_bytes",
        "used_train_sequences",
    }
    required_data_hashes = (
        "source_sha256",
        "integrity_sha256",
        "inputs_array_sha256",
        "training_order_array_sha256",
        "preflight_inputs_array_sha256",
        "calibration_inputs_array_sha256",
    )
    if (
        not isinstance(data, Mapping)
        or set(data) != expected_data_keys
        or any(not is_sha256(data.get(key)) for key in required_data_hashes)
        or data.get("source_path") != SOURCE_PATH.relative_to(ROOT).as_posix()
        or data.get("integrity_path") != INTEGRITY_PATH.relative_to(ROOT).as_posix()
        or data.get("sequence_length") != SEQUENCE_LENGTH
        or data.get("nominal_train_bytes") != NOMINAL_TRAIN_BYTES
        or data.get("training_order_seed") != TRAINING_ORDER_SEED
        or data.get("preflight_selection")
        != "first 96 complete 512-byte sequences in canonical train stream"
        or set(data.get("training_patch_matrix_sha256", {})) != set(ROLE_ORDER)
        or set(data.get("preflight_patch_matrix_sha256", {})) != set(ROLE_ORDER)
        or set(data.get("calibration_patch_matrix_sha256", {})) != set(ROLE_ORDER)
        or any(
            not is_sha256(data[group][role])
            for group in (
                "training_patch_matrix_sha256",
                "preflight_patch_matrix_sha256",
                "calibration_patch_matrix_sha256",
            )
            for role in ROLE_ORDER
        )
        or data.get("available_train_sequences") != AVAILABLE_TRAIN_SEQUENCES
        or data.get("dropped_train_sequences")
        != AVAILABLE_TRAIN_SEQUENCES - TRAIN_SEQUENCES
        or data.get("preflight_examples") != PREFLIGHT_EXAMPLES
        or data.get("calibration_bytes") != CALIBRATION_BYTES
        or data.get("calibration_examples") != CALIBRATION_BYTES // SEQUENCE_LENGTH
    ):
        raise ValueError("balanced-200M data contract differs")
    upstream = value["upstream"]
    if (
        not isinstance(upstream, Mapping)
        or set(upstream)
        != {
            "resource_summary_artifact_sha256",
            "resource_summary_path",
            "resource_summary_sha256",
            "scale_plan_artifact_sha256",
            "scale_plan_path",
            "scale_plan_sha256",
            "scale_summary_artifact_sha256",
            "scale_summary_path",
            "scale_summary_sha256",
        }
        or upstream.get("resource_summary_path")
        != RESOURCE_SUMMARY_PATH.relative_to(ROOT).as_posix()
        or upstream.get("scale_summary_path")
        != SCALE_SUMMARY_PATH.relative_to(ROOT).as_posix()
        or upstream.get("scale_plan_path")
        != SCALE_PLAN_PATH.relative_to(ROOT).as_posix()
        or any(
            not is_sha256(upstream.get(key))
            for key in (
                "resource_summary_artifact_sha256",
                "resource_summary_sha256",
                "scale_plan_artifact_sha256",
                "scale_plan_sha256",
                "scale_summary_artifact_sha256",
                "scale_summary_sha256",
            )
        )
    ):
        raise ValueError("balanced-200M upstream identity differs")
    implementation = value["implementation_sha256"]
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or any(not is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
    ):
        raise ValueError("balanced-200M implementation set differs")
    if current_environment is not None and value["environment"] != current_environment:
        raise ValueError("balanced-200M environment differs")
    if verify_implementation:
        for relative in IMPLEMENTATION_PATHS:
            path = ROOT / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or hash_file(path) != implementation[relative]
            ):
                raise ValueError(f"balanced-200M implementation differs: {relative}")


def preflight_pass(report: Mapping[str, Any]) -> bool:
    measurement = report.get("measurement")
    recommended = report.get("recommended_max_memory_bytes")
    maximum = report.get("maximum_driver_allocated_bytes")
    return bool(
        report.get("completed") is True
        and report.get("finite") is True
        and report.get("optimizer_state_initialized") is True
        and isinstance(measurement, Mapping)
        and math.isfinite(float(measurement.get("projected_hours", math.inf)))
        and 0 < float(measurement["projected_hours"]) <= MAXIMUM_HOURS_PER_ROLE
        and type(maximum) is int
        and type(recommended) is int
        and 0 < maximum <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION * recommended
    )


def summarize_preflight(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(reports) != set(ROLE_ORDER):
        raise ValueError("balanced-200M preflight role set differs")
    role_pass = {role: preflight_pass(reports[role]) for role in ROLE_ORDER}
    pair_hours = (
        sum(
            float(reports[role]["measurement"]["projected_hours"])
            for role in ROLE_ORDER
        )
        if all(role_pass.values())
        else None
    )
    passed = bool(
        all(role_pass.values())
        and pair_hours is not None
        and pair_hours <= MAXIMUM_HOURS_PER_PAIR
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "role_pass": role_pass,
        "pair_projected_hours": pair_hours,
        "overall_preflight_pass": passed,
        "training_protocol_may_be_implemented": passed,
        "training_directly_started": False,
        "status": "balanced_200m_batch8_preflight_pass"
        if passed
        else "balanced_200m_batch8_preflight_fail",
    }


def build_preflight_summary(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    summary_base_git_commit: str,
    worker_evidence: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = summarize_preflight(reports)
    if (
        not is_sha256(plan_artifact_sha256)
        or not is_git_commit(summary_base_git_commit)
        or set(worker_evidence) != set(ROLE_ORDER)
    ):
        raise ValueError("balanced-200M preflight summary dependency differs")
    for role in ROLE_ORDER:
        row = worker_evidence[role]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or row["path"] != worker_report_path(role).relative_to(ROOT).as_posix()
            or not is_sha256(row["sha256"])
        ):
            raise ValueError("balanced-200M preflight evidence differs")
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_batch8_preflight_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": aggregate["status"],
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "worker_evidence": dict(worker_evidence),
        "aggregate": aggregate,
        "claim_boundary": {
            "resource_preflight_only": True,
            "quality_claimed": False,
            "training_started": False,
        },
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def validate_preflight_summary(value: Mapping[str, Any]) -> None:
    expected = {
        "aggregate",
        "claim_boundary",
        "kind",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "status",
        "summary_base_git_commit",
        "summary_sha256",
        "worker_evidence",
    }
    if set(value) != expected:
        raise ValueError("balanced-200M preflight summary schema differs")
    payload = dict(value)
    claimed = payload.pop("summary_sha256")
    if (
        value["schema_version"] != 1
        or value["kind"] != "balanced_200m_batch8_preflight_summary_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or not is_sha256(value["plan_artifact_sha256"])
        or not is_sha256(value["plan_sha256"])
        or not is_git_commit(value["summary_base_git_commit"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value["status"] != value["aggregate"].get("status")
        or value["claim_boundary"]
        != {
            "resource_preflight_only": True,
            "quality_claimed": False,
            "training_started": False,
        }
    ):
        raise ValueError("balanced-200M preflight summary identity differs")


def bpb_from_sequence_nll(values: np.ndarray) -> float:
    losses = np.asarray(values)
    expected = CALIBRATION_BYTES // SEQUENCE_LENGTH
    if (
        losses.dtype != np.float32
        or losses.shape != (expected,)
        or not np.all(np.isfinite(losses))
        or np.any(losses < 0)
    ):
        raise ValueError("balanced-200M calibration NLL differs")
    predicted_bytes = expected * (SEQUENCE_LENGTH - 1)
    return float(losses.astype(np.float64).sum()) / predicted_bytes / math.log(2)


def summarize_training_quality(
    nll_by_role: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if set(nll_by_role) != set(ROLE_ORDER):
        raise ValueError("balanced-200M quality role set differs")
    bpb = {role: bpb_from_sequence_nll(nll_by_role[role]) for role in ROLE_ORDER}
    delta = bpb["w72"] - bpb["c86"]
    passed = bool(math.isfinite(delta) and delta <= QUALITY_MARGIN_BPB)
    return {
        "calibration_examples": CALIBRATION_BYTES // SEQUENCE_LENGTH,
        "predicted_bytes_per_role": (CALIBRATION_BYTES // SEQUENCE_LENGTH)
        * (SEQUENCE_LENGTH - 1),
        "bpb_by_role": bpb,
        "w72_minus_c86_bpb": delta,
        "maximum_allowed_delta_bpb": QUALITY_MARGIN_BPB,
        "quality_screen_pass": passed,
        "actual_timing_authorized": passed,
        "multiseed_quality_claimed": False,
    }


def build_training_summary(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    preflight: Mapping[str, Any],
    preflight_artifact_sha256: str,
    summary_base_git_commit: str,
    worker_evidence: Mapping[str, Mapping[str, Any]],
    nll_by_role: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    validate_preflight_summary(preflight)
    if (
        preflight["aggregate"].get("overall_preflight_pass") is not True
        or not is_sha256(plan_artifact_sha256)
        or not is_sha256(preflight_artifact_sha256)
        or not is_git_commit(summary_base_git_commit)
        or set(worker_evidence) != set(ROLE_ORDER)
    ):
        raise ValueError("balanced-200M training summary dependency differs")
    expected_keys = {
        "report_path",
        "report_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_state_sha256",
        "calibration_nll_path",
        "calibration_nll_sha256",
        "calibration_nll_array_sha256",
    }
    for role in ROLE_ORDER:
        row = worker_evidence[role]
        if (
            set(row) != expected_keys
            or row["report_path"]
            != training_report_path(role).relative_to(ROOT).as_posix()
            or row["checkpoint_path"]
            != checkpoint_path(role).relative_to(ROOT).as_posix()
            or row["calibration_nll_path"]
            != calibration_nll_path(role).relative_to(ROOT).as_posix()
            or any(
                not is_sha256(row[key])
                for key in expected_keys
                if key.endswith("sha256")
            )
        ):
            raise ValueError("balanced-200M training evidence differs")
    quality = summarize_training_quality(nll_by_role)
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_trained_screen_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "balanced_200m_quality_pass"
            if quality["quality_screen_pass"]
            else "balanced_200m_quality_fail"
        ),
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "preflight_artifact_sha256": preflight_artifact_sha256,
        "preflight_summary_sha256": preflight["summary_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "worker_evidence": {role: dict(worker_evidence[role]) for role in ROLE_ORDER},
        "quality": quality,
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "historical_test_or_final_metric_used": False,
            "actual_timing_requires_quality_pass": True,
            "quality_independent_checkpoint_replay_pending": True,
        },
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def validate_training_summary(value: Mapping[str, Any]) -> None:
    expected = {
        "claim_boundary",
        "kind",
        "plan_artifact_sha256",
        "plan_sha256",
        "preflight_artifact_sha256",
        "preflight_summary_sha256",
        "protocol_id",
        "quality",
        "schema_version",
        "status",
        "summary_base_git_commit",
        "summary_sha256",
        "worker_evidence",
    }
    if set(value) != expected:
        raise ValueError("balanced-200M training summary schema differs")
    payload = dict(value)
    claimed = payload.pop("summary_sha256")
    quality = value.get("quality")
    if (
        value["schema_version"] != 1
        or value["kind"] != "balanced_200m_trained_screen_summary_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or not is_sha256(value["plan_artifact_sha256"])
        or not is_sha256(value["plan_sha256"])
        or not is_sha256(value["preflight_artifact_sha256"])
        or not is_sha256(value["preflight_summary_sha256"])
        or not is_git_commit(value["summary_base_git_commit"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or not isinstance(quality, Mapping)
        or value["status"]
        != (
            "balanced_200m_quality_pass"
            if quality.get("quality_screen_pass") is True
            else "balanced_200m_quality_fail"
        )
        or value["claim_boundary"]
        != {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "historical_test_or_final_metric_used": False,
            "actual_timing_requires_quality_pass": True,
            "quality_independent_checkpoint_replay_pending": True,
        }
    ):
        raise ValueError("balanced-200M training summary identity differs")
