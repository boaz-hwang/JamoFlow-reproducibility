"""Contracts for post-1.6B training-resource feasibility measurements."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scale_schedule_extrapolation_core import (
    EXPECTED_PARAMETERS,
    ROOT,
    SCHEDULE_ORDER,
    TARGET_ORDER,
    array_sha256,
    canonical_sha256,
    is_git_commit,
    is_sha256,
    large_scale_model_spec,
    validate_scale_schedule_summary,
)
from scale_schedule_extrapolation_core import (
    OUTPUT_PATH as SCALE_OUTPUT_PATH,
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
from jamoflow.phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_window_grid_trace,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
)

PROTOCOL_ID = "jamoflow-large-scale-training-feasibility-v1"
PLAN_PATH = ROOT / "data/manifests/large-scale-training-feasibility-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/large-scale-training-feasibility-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
OUTPUT_PATH = ROOT / "results/large-scale-training-feasibility-v1/summary.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"

ROLE_ORDER = SCHEDULE_ORDER
STANDARD_REGIME = "standard"
CHECKPOINTED_REGIME = "gradient_checkpointed"
REGIME_ORDER = (STANDARD_REGIME, CHECKPOINTED_REGIME)
CHECKPOINTED_TARGETS = (1600,)
SEQUENCE_LENGTH = 512
TRAINING_EXAMPLES = 32
MICROBATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_UPDATES = 1
MEASUREMENT_UPDATES = 2
SOURCE_BYTE_BUDGETS = (64_000_000, 256_000_000)
MAXIMUM_RECOMMENDED_MEMORY_FRACTION = 0.75
MAXIMUM_64M_HOURS_PER_MODEL = 120.0
MAXIMUM_64M_HOURS_PER_PAIR = 240.0
LEARNING_RATE = 1.5e-4
BETAS = (0.9, 0.95)
EPSILON = 1e-8
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0

IMPLEMENTATION_PATHS = (
    "docs/190-scale-schedule-extrapolation-result-and-research-pivot.md",
    "docs/191-large-scale-training-feasibility-protocol.md",
    "pyproject.toml",
    "scripts/large_scale_training_feasibility_core.py",
    "scripts/run_large_scale_training_feasibility.py",
    "scripts/scale_schedule_extrapolation_core.py",
    "scripts/seal_large_scale_training_feasibility_plan.py",
    "scripts/verify_large_scale_training_feasibility.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "tests/test_large_scale_training_feasibility.py",
)


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def worker_order() -> tuple[tuple[int, str, str], ...]:
    standard = tuple(
        (target, STANDARD_REGIME, role)
        for target in TARGET_ORDER
        for role in ROLE_ORDER
    )
    checkpointed = tuple(
        (target, CHECKPOINTED_REGIME, role)
        for target in CHECKPOINTED_TARGETS
        for role in ROLE_ORDER
    )
    return (*standard, *checkpointed)


def worker_id(target: int, regime: str, role: str) -> str:
    if (target, regime, role) not in worker_order():
        raise ValueError("large-scale training worker identity differs")
    return f"target-{target}-{regime}-{role}"


def worker_report_path(target: int, regime: str, role: str) -> Path:
    return ARTIFACT_ROOT / f"{worker_id(target, regime, role)}.json"


def _patch_matrices(
    boundaries: np.ndarray,
    whitespace: np.ndarray,
) -> dict[str, np.ndarray]:
    if boundaries.shape != whitespace.shape or boundaries.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("large-scale training boundary arrays differ")
    codepoint_rows: list[tuple[int, ...]] = []
    whitespace_rows: list[tuple[int, ...]] = []
    for boundary, spaces in zip(boundaries, whitespace, strict=True):
        codepoint_rows.append(causal_codepoint_grid_boundaries(boundary, 86))
        whitespace_rows.append(
            causal_window_grid_trace(boundary, spaces, 72).boundaries
        )
    return {
        "c86": padded_hf_patch_matrix(codepoint_rows, SEQUENCE_LENGTH),
        "w72": padded_hf_patch_matrix(whitespace_rows, SEQUENCE_LENGTH),
    }


def training_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="train",
        byte_limit=TRAINING_EXAMPLES * SEQUENCE_LENGTH,
        sequence_length=SEQUENCE_LENGTH,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    matrices = _patch_matrices(boundaries, whitespace)
    if inputs.dtype != np.uint8 or inputs.shape != (
        TRAINING_EXAMPLES,
        SEQUENCE_LENGTH,
    ):
        raise ValueError("large-scale training input batch differs")
    if matrices["c86"].shape != (TRAINING_EXAMPLES, 87):
        raise ValueError("large-scale C86 matrix differs")
    if matrices["w72"].shape != (TRAINING_EXAMPLES, 73):
        raise ValueError("large-scale W72 matrix differs")
    return np.ascontiguousarray(inputs), {
        role: np.ascontiguousarray(matrices[role]) for role in ROLE_ORDER
    }


def training_data_contract() -> dict[str, Any]:
    inputs, matrices = training_arrays()
    stream_bytes = np.ascontiguousarray(inputs).tobytes()
    return {
        "source_path": SOURCE_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": hash_file(SOURCE_PATH),
        "split": "train",
        "selection": "first 32 complete 512-byte sequences in canonical train stream",
        "sequence_length": SEQUENCE_LENGTH,
        "examples": TRAINING_EXAMPLES,
        "stream_sha256": hashlib.sha256(stream_bytes).hexdigest(),
        "inputs_array_sha256": array_sha256(inputs),
        "patch_matrix_sha256": {
            role: array_sha256(matrices[role]) for role in ROLE_ORDER
        },
        "patch_matrix_shape": {
            role: list(matrices[role].shape) for role in ROLE_ORDER
        },
    }


def optimizer_contract() -> dict[str, Any]:
    return {
        "name": "torch.optim.AdamW",
        "parameter_dtype": "float32",
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "epsilon": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "microbatch_size": MICROBATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_sequences": (
            MICROBATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        ),
        "source_bytes_per_update": (
            SEQUENCE_LENGTH
            * MICROBATCH_SIZE
            * GRADIENT_ACCUMULATION_STEPS
        ),
        "warmup_updates": WARMUP_UPDATES,
        "measurement_updates": MEASUREMENT_UPDATES,
        "loss_divisor": GRADIENT_ACCUMULATION_STEPS,
    }


def projection_contract() -> dict[str, Any]:
    return {
        "source_byte_budgets": list(SOURCE_BYTE_BUDGETS),
        "maximum_64m_hours_per_model": MAXIMUM_64M_HOURS_PER_MODEL,
        "maximum_64m_hours_per_pair": MAXIMUM_64M_HOURS_PER_PAIR,
        "maximum_recommended_memory_fraction": (
            MAXIMUM_RECOMMENDED_MEMORY_FRACTION
        ),
        "selection_rule": (
            "standard regime preferred; 1600M checkpointed may rescue only the "
            "1600M resource endpoint; lower targets are diagnostic and cannot "
            "replace the fixed 1600M systems endpoint"
        ),
    }


def build_plan(
    *,
    git_commit_before_plan: str,
    environment: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    scale_plan: Mapping[str, Any],
    scale_summary: Mapping[str, Any],
) -> dict[str, Any]:
    validate_scale_plan(scale_plan, verify_implementation=False)
    validate_scale_schedule_summary(scale_summary)
    if (
        scale_summary.get("status") != "large_scale_10_percent_headroom_detected"
        or scale_summary.get("claim_boundary", {}).get(
            "ten_percent_large_scale_headroom_detected"
        )
        is not True
        or scale_summary.get("plan_artifact_sha256") != hash_file(SCALE_PLAN_PATH)
        or scale_summary.get("plan_sha256") != scale_plan.get("plan_sha256")
    ):
        raise ValueError("large-scale systems headroom dependency differs")
    models = {
        str(target): dict(scale_plan["models"][str(target)])
        for target in TARGET_ORDER
    }
    payload = {
        "schema_version": 1,
        "kind": "large_scale_training_feasibility_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_first_large_scale_optimizer_step",
        "git_commit_before_plan": git_commit_before_plan,
        "upstream": {
            "scale_plan_path": SCALE_PLAN_PATH.relative_to(ROOT).as_posix(),
            "scale_plan_artifact_sha256": hash_file(SCALE_PLAN_PATH),
            "scale_plan_sha256": scale_plan["plan_sha256"],
            "scale_summary_path": SCALE_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "scale_summary_artifact_sha256": hash_file(SCALE_OUTPUT_PATH),
            "scale_summary_sha256": scale_summary["summary_sha256"],
        },
        "models": models,
        "roles": {
            "order": list(ROLE_ORDER),
            "c86": {"policy": "causal_codepoint_grid", "patch_count": 86},
            "w72": {"policy": "causal_whitespace_grid", "patch_count": 72},
        },
        "regimes": {
            "order": list(REGIME_ORDER),
            "standard_targets": list(TARGET_ORDER),
            "gradient_checkpointed_targets": list(CHECKPOINTED_TARGETS),
        },
        "worker_order": [
            worker_id(target, regime, role)
            for target, regime, role in worker_order()
        ],
        "training_data": training_data_contract(),
        "optimizer": optimizer_contract(),
        "projection": projection_contract(),
        "environment": dict(environment),
        "implementation_sha256": dict(implementation_sha256),
        "outputs": {
            "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        },
        "claim_boundary": {
            "resource_measurement_only": True,
            "quality_claimed": False,
            "training_directly_authorized": False,
            "lower_target_fallback_authorized": False,
            "gradient_checkpointing_is_resource_fallback_only": True,
            "actual_optimizer_state_required": True,
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
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "models",
        "optimizer",
        "outputs",
        "plan_sha256",
        "projection",
        "protocol_id",
        "regimes",
        "roles",
        "schema_version",
        "status",
        "training_data",
        "upstream",
        "worker_order",
    }
    if set(value) != expected:
        raise ValueError("large-scale training plan schema differs")
    payload = dict(value)
    claimed = payload.pop("plan_sha256")
    if (
        value["schema_version"] != 1
        or value["kind"] != "large_scale_training_feasibility_plan_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or value["status"] != "sealed_before_first_large_scale_optimizer_step"
        or not is_git_commit(value["git_commit_before_plan"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value["roles"]
        != {
            "order": list(ROLE_ORDER),
            "c86": {"policy": "causal_codepoint_grid", "patch_count": 86},
            "w72": {"policy": "causal_whitespace_grid", "patch_count": 72},
        }
        or value["regimes"]
        != {
            "order": list(REGIME_ORDER),
            "standard_targets": list(TARGET_ORDER),
            "gradient_checkpointed_targets": list(CHECKPOINTED_TARGETS),
        }
        or value["worker_order"]
        != [worker_id(*row) for row in worker_order()]
        or value["optimizer"] != optimizer_contract()
        or value["projection"] != projection_contract()
        or value["training_data"] != training_data_contract()
        or value["claim_boundary"]
        != {
            "resource_measurement_only": True,
            "quality_claimed": False,
            "training_directly_authorized": False,
            "lower_target_fallback_authorized": False,
            "gradient_checkpointing_is_resource_fallback_only": True,
            "actual_optimizer_state_required": True,
        }
    ):
        raise ValueError("large-scale training plan identity differs")
    if set(value["models"]) != {str(target) for target in TARGET_ORDER}:
        raise ValueError("large-scale training model set differs")
    for target in TARGET_ORDER:
        row = value["models"][str(target)]
        if (
            not isinstance(row, Mapping)
            or row.get("expected_parameter_count") != EXPECTED_PARAMETERS[target]
            or row.get("spec") != large_scale_model_spec(target, 86).to_dict()
            or not is_sha256(row.get("model_state_sha256"))
        ):
            raise ValueError("large-scale training model identity differs")
    upstream = value["upstream"]
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("scale_plan_path")
        != SCALE_PLAN_PATH.relative_to(ROOT).as_posix()
        or upstream.get("scale_summary_path")
        != SCALE_OUTPUT_PATH.relative_to(ROOT).as_posix()
        or any(
            not is_sha256(upstream.get(key))
            for key in (
                "scale_plan_artifact_sha256",
                "scale_plan_sha256",
                "scale_summary_artifact_sha256",
                "scale_summary_sha256",
            )
        )
    ):
        raise ValueError("large-scale training upstream identity differs")
    implementation = value["implementation_sha256"]
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or any(not is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
    ):
        raise ValueError("large-scale training implementation set differs")
    if current_environment is not None and value["environment"] != current_environment:
        raise ValueError("large-scale training environment differs")
    if verify_implementation:
        for relative in IMPLEMENTATION_PATHS:
            path = ROOT / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or hash_file(path) != implementation[relative]
            ):
                raise ValueError(f"large-scale training implementation differs: {relative}")


def projected_training(
    update_seconds: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(tuple(update_seconds), dtype=np.float64)
    if (
        values.shape != (MEASUREMENT_UPDATES,)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ValueError("large-scale training update timings differ")
    median = float(np.median(values))
    bytes_per_update = optimizer_contract()["source_bytes_per_update"]
    by_budget: dict[str, Any] = {}
    for budget in SOURCE_BYTE_BUDGETS:
        updates = math.ceil(budget / bytes_per_update)
        by_budget[str(budget)] = {
            "optimizer_updates": updates,
            "projected_source_bytes": updates * bytes_per_update,
            "projected_hours_per_model": median * updates / 3600,
        }
    return {
        "measurement_update_seconds": values.tolist(),
        "median_update_seconds": median,
        "by_source_byte_budget": by_budget,
    }


def validate_worker_report(
    report: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    runner_git_commit: str,
    target: int,
    regime: str,
    role: str,
) -> None:
    common = {
        "completed",
        "environment_end",
        "environment_start",
        "failure",
        "finite",
        "kind",
        "maximum_driver_allocated_bytes",
        "measurement",
        "memory_cap_enforced",
        "memory_snapshots",
        "model_state_sha256",
        "optimizer_state_initialized",
        "parameter_count",
        "patch_matrix_sha256",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "recommended_max_memory_bytes",
        "regime",
        "role",
        "runner_git_commit",
        "schema_version",
        "target_millions",
        "training_data_sha256",
    }
    if set(report) != common:
        raise ValueError("large-scale training worker schema differs")
    if (
        report["schema_version"] != 1
        or report["kind"] != "large_scale_training_feasibility_worker_v1"
        or report["protocol_id"] != PROTOCOL_ID
        or report["target_millions"] != target
        or report["regime"] != regime
        or report["role"] != role
        or report["runner_git_commit"] != runner_git_commit
        or report["plan_sha256"] != plan["plan_sha256"]
        or report["plan_artifact_sha256"] != plan_artifact_sha256
        or report["parameter_count"] not in (None, EXPECTED_PARAMETERS[target])
        or report["model_state_sha256"]
        not in (None, plan["models"][str(target)]["model_state_sha256"])
        or report["patch_matrix_sha256"]
        != plan["training_data"]["patch_matrix_sha256"][role]
        or report["training_data_sha256"]
        != plan["training_data"]["inputs_array_sha256"]
        or report["memory_cap_enforced"] is not True
        or report["environment_start"] != plan["environment"]
        or report["environment_end"] != plan["environment"]
    ):
        raise ValueError("large-scale training worker identity differs")
    snapshots = report["memory_snapshots"]
    if not isinstance(snapshots, list):
        raise TypeError("large-scale training memory snapshots differ")
    maximum = 0
    for snapshot in snapshots:
        if (
            not isinstance(snapshot, Mapping)
            or set(snapshot)
            != {"current_allocated_bytes", "driver_allocated_bytes", "stage"}
            or not isinstance(snapshot["stage"], str)
            or type(snapshot["current_allocated_bytes"]) is not int
            or type(snapshot["driver_allocated_bytes"]) is not int
            or snapshot["current_allocated_bytes"] < 0
            or snapshot["driver_allocated_bytes"] < 0
        ):
            raise ValueError("large-scale training memory row differs")
        maximum = max(maximum, snapshot["driver_allocated_bytes"])
    if report["maximum_driver_allocated_bytes"] != maximum:
        raise ValueError("large-scale training maximum memory differs")
    if report["completed"] is False:
        recommended = report["recommended_max_memory_bytes"]
        if (
            report["finite"] is not False
            or report["optimizer_state_initialized"] not in (False, True)
            or report["measurement"] is not None
            or report["maximum_driver_allocated_bytes"] != maximum
            or recommended is not None
            and (type(recommended) is not int or recommended <= 0)
            or not isinstance(report["failure"], Mapping)
            or set(report["failure"])
            != {"category", "message", "returncode", "stage"}
        ):
            raise ValueError("large-scale training failure record differs")
        return
    recommended = report["recommended_max_memory_bytes"]
    if type(recommended) is not int or recommended <= 0:
        raise ValueError("large-scale training recommended memory differs")
    if (
        report["completed"] is not True
        or report["finite"] is not True
        or report["optimizer_state_initialized"] is not True
        or report["failure"] is not None
        or report["parameter_count"] != EXPECTED_PARAMETERS[target]
        or not snapshots
        or maximum <= 0
        or maximum
        > MAXIMUM_RECOMMENDED_MEMORY_FRACTION * recommended
        or report["measurement"]
        != projected_training(report["measurement"]["measurement_update_seconds"])
    ):
        raise ValueError("large-scale training successful measurement differs")


def resource_pass(report: Mapping[str, Any]) -> bool:
    if report.get("completed") is not True or report.get("finite") is not True:
        return False
    recommended = report.get("recommended_max_memory_bytes")
    maximum = report.get("maximum_driver_allocated_bytes")
    measurement = report.get("measurement")
    try:
        hours = float(
            measurement["by_source_byte_budget"]["64000000"][
                "projected_hours_per_model"
            ]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        type(recommended) is int
        and recommended > 0
        and type(maximum) is int
        and 0 < maximum <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION * recommended
        and math.isfinite(hours)
        and 0 < hours <= MAXIMUM_64M_HOURS_PER_MODEL
    )


def summarize_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {worker_id(*row) for row in worker_order()}
    if set(reports) != expected:
        raise ValueError("large-scale training report set differs")
    by_target: dict[str, Any] = {}
    standard_feasible: list[int] = []
    for target in TARGET_ORDER:
        roles = {
            role: reports[worker_id(target, STANDARD_REGIME, role)]
            for role in ROLE_ORDER
        }
        role_pass = {role: resource_pass(roles[role]) for role in ROLE_ORDER}
        pair_hours = (
            sum(
                float(
                    roles[role]["measurement"]["by_source_byte_budget"][
                        "64000000"
                    ]["projected_hours_per_model"]
                )
                for role in ROLE_ORDER
            )
            if all(role_pass.values())
            else None
        )
        pair_pass = bool(
            all(role_pass.values())
            and pair_hours is not None
            and pair_hours <= MAXIMUM_64M_HOURS_PER_PAIR
        )
        if pair_pass:
            standard_feasible.append(target)
        by_target[str(target)] = {
            "standard": {
                "role_pass": role_pass,
                "pair_projected_hours_64m": pair_hours,
                "pair_resource_pass": pair_pass,
            }
        }
    checkpoint_roles = {
        role: reports[worker_id(1600, CHECKPOINTED_REGIME, role)]
        for role in ROLE_ORDER
    }
    checkpoint_role_pass = {
        role: resource_pass(checkpoint_roles[role]) for role in ROLE_ORDER
    }
    checkpoint_pair_hours = (
        sum(
            float(
                checkpoint_roles[role]["measurement"]["by_source_byte_budget"][
                    "64000000"
                ]["projected_hours_per_model"]
            )
            for role in ROLE_ORDER
        )
        if all(checkpoint_role_pass.values())
        else None
    )
    checkpoint_pair_pass = bool(
        all(checkpoint_role_pass.values())
        and checkpoint_pair_hours is not None
        and checkpoint_pair_hours <= MAXIMUM_64M_HOURS_PER_PAIR
    )
    by_target["1600"][CHECKPOINTED_REGIME] = {
        "role_pass": checkpoint_role_pass,
        "pair_projected_hours_64m": checkpoint_pair_hours,
        "pair_resource_pass": checkpoint_pair_pass,
    }
    if by_target["1600"][STANDARD_REGIME]["pair_resource_pass"]:
        selected_regime: str | None = STANDARD_REGIME
    elif checkpoint_pair_pass:
        selected_regime = CHECKPOINTED_REGIME
    else:
        selected_regime = None
    direct = selected_regime is not None
    return {
        "protocol_id": PROTOCOL_ID,
        "by_target": by_target,
        "largest_standard_resource_feasible_target_millions": (
            max(standard_feasible) if standard_feasible else None
        ),
        "primary_1600_resource_feasible": direct,
        "primary_1600_selected_regime": selected_regime,
        "balanced_trained_bridge_protocol_may_be_designed": direct,
        "lower_target_fallback_authorized": False,
        "global_heavy_architecture_pivot_required": not direct,
        "status": (
            "balanced_1600_training_pilot_resource_feasible"
            if direct
            else "balanced_1600_training_infeasible_architecture_pivot_required"
        ),
    }


def build_summary(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    summary_base_git_commit: str,
    worker_evidence: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    aggregate = summarize_reports(reports)
    if (
        not is_sha256(plan_artifact_sha256)
        or not is_git_commit(summary_base_git_commit)
        or set(worker_evidence) != {worker_id(*row) for row in worker_order()}
    ):
        raise ValueError("large-scale training summary dependency differs")
    for identifier, evidence in worker_evidence.items():
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != {"path", "sha256"}
            or evidence["path"]
            != worker_report_path(*_parse_worker_id(identifier)).relative_to(ROOT).as_posix()
            or not is_sha256(evidence["sha256"])
        ):
            raise ValueError("large-scale training evidence row differs")
    payload = {
        "schema_version": 1,
        "kind": "large_scale_training_feasibility_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": aggregate["status"],
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "worker_evidence": dict(worker_evidence),
        "aggregate": aggregate,
        "claim_boundary": {
            "resource_measurement_only": True,
            "quality_claimed": False,
            "training_directly_authorized": False,
            "lower_target_fallback_authorized": False,
        },
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def _parse_worker_id(identifier: str) -> tuple[int, str, str]:
    for row in worker_order():
        if identifier == worker_id(*row):
            return row
    raise ValueError("large-scale training worker id differs")


def validate_summary(value: Mapping[str, Any]) -> None:
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
        raise ValueError("large-scale training summary schema differs")
    payload = dict(value)
    claimed = payload.pop("summary_sha256")
    if (
        value["schema_version"] != 1
        or value["kind"] != "large_scale_training_feasibility_summary_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or not is_sha256(value["plan_artifact_sha256"])
        or not is_sha256(value["plan_sha256"])
        or not is_git_commit(value["summary_base_git_commit"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value["status"] != value["aggregate"].get("status")
        or value["claim_boundary"]
        != {
            "resource_measurement_only": True,
            "quality_claimed": False,
            "training_directly_authorized": False,
            "lower_target_fallback_authorized": False,
        }
    ):
        raise ValueError("large-scale training summary identity differs")
