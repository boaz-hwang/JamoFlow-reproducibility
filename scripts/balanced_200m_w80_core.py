"""Pure contracts and statistics for the balanced-200M W80 rescue screen."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import balanced_200m_trained_core as base
from balanced_200m_failure_analysis_core import (
    contiguous_block_bootstrap,
    paired_bpb_effects,
    validate_verification_receipt as validate_base_verification,
)
from scale_schedule_extrapolation_core import (
    CONTINUATION_BYTES,
    GLOBAL_POSITION_LIMIT,
    MODEL_SEED,
    PROMPT_BYTES,
    ROOT,
    array_sha256,
    canonical_sha256,
    is_git_commit,
    is_sha256,
    large_scale_model_spec,
    validate_case_arrays,
    validate_plan as validate_scale_plan,
)
from scale_schedule_extrapolation_core import PLAN_PATH as SCALE_PLAN_PATH

from jamoflow.hplt3 import hash_file
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    causal_window_grid_trace,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
)

PROTOCOL_ID = "jamoflow-balanced-200m-w80-rescue-v1"
PLAN_PATH = ROOT / "data/manifests/balanced-200m-w80-rescue-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/balanced-200m-w80-rescue-v1"
PREFLIGHT_ACTIVE_PATH = ARTIFACT_ROOT / ".preflight-active"
TRAINING_ACTIVE_PATH = ARTIFACT_ROOT / ".training-active"
PREFLIGHT_OUTPUT_PATH = ROOT / "results/balanced-200m-w80-rescue-v1/preflight.json"
TRAINING_OUTPUT_PATH = (
    ROOT / "results/balanced-200m-w80-rescue-v1/training-summary.json"
)
VERIFICATION_OUTPUT_PATH = (
    ROOT / "results/balanced-200m-w80-rescue-v1/verification.json"
)
TIMING_RECEIPT_ROOT = ROOT / "results/balanced-200m-w80-rescue-v1/sessions"
TIMING_SUMMARY_PATH = ROOT / "results/balanced-200m-w80-rescue-v1/actual-summary.json"

BASE_PLAN_PATH = base.PLAN_PATH
BASE_SUMMARY_PATH = base.TRAINING_OUTPUT_PATH
BASE_VERIFICATION_PATH = (
    ROOT / "results/balanced-200m-trained-screen-v1/verification.json"
)
FAILURE_ANALYSIS_PATH = (
    ROOT / "results/balanced-200m-trained-screen-v1/quality-failure-analysis.json"
)

CANDIDATE_ROLE = "w80"
REFERENCE_ROLE = "c86"
TIMING_ROLE_ORDER = (CANDIDATE_ROLE, REFERENCE_ROLE)
CANDIDATE_PATCH_COUNT = 80
REFERENCE_PATCH_COUNT = 86
QUALITY_MARGIN_BPB = 0.010
QUALITY_BLOCK_SIZE = 64
QUALITY_BOOTSTRAP_REPETITIONS = 10_000
QUALITY_BOOTSTRAP_SEED = 20260904

TIMING_MODE_ORDER = ("controlled_replay", "free_running_utf8_greedy")
TIMING_SESSION_ORDER = tuple(f"session-{index}" for index in range(5))
TIMING_WARMUP_PROMPTS = 4
TIMING_MEASURED_PROMPTS = 16
TIMING_REPETITIONS = 3
TIMING_CORRECTNESS_PROMPTS = 4
TIMING_BOOTSTRAP_REPETITIONS = 10_000
TIMING_BOOTSTRAP_SEED = 20260905
TIMING_MINIMUM_POSITIVE_PROMPTS = 15
TIMING_MINIMUM_POSITIVE_SESSIONS = len(TIMING_SESSION_ORDER)
COMPACT_REFERENCE_REDUCTION = {
    "controlled_replay": 0.026283464474602614,
    "free_running_utf8_greedy": 0.025305234146383637,
}
TIMING_RTOL = 1e-4
TIMING_ATOL = 2e-5
MAXIMUM_FREE_OUTPUT_BYTES = CONTINUATION_BYTES + 3

IMPLEMENTATION_PATHS = (
    "docs/197-balanced-200m-quality-failure-result-and-w80-pivot.md",
    "docs/198-balanced-200m-w80-rescue-protocol.md",
    "pyproject.toml",
    "scripts/balanced_200m_failure_analysis_core.py",
    "scripts/balanced_200m_trained_core.py",
    "scripts/balanced_200m_w80_core.py",
    "scripts/benchmark_balanced_200m_w80_actual.py",
    "scripts/run_balanced_200m_preflight.py",
    "scripts/run_balanced_200m_training.py",
    "scripts/run_balanced_200m_w80_preflight.py",
    "scripts/run_balanced_200m_w80_training.py",
    "scripts/seal_balanced_200m_w80_plan.py",
    "scripts/summarize_balanced_200m_w80_actual.py",
    "scripts/verify_balanced_200m_w80_preflight.py",
    "scripts/verify_balanced_200m_w80_training.py",
    "scripts/scale_schedule_extrapolation_core.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/utf8.py",
    "tests/test_balanced_200m_w80_rescue.py",
)


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def worker_preflight_path() -> Path:
    return ARTIFACT_ROOT / "preflight-w80.json"


def training_report_path() -> Path:
    return ARTIFACT_ROOT / "training-w80.json"


def checkpoint_path() -> Path:
    return ARTIFACT_ROOT / "trained-w80.pt"


def calibration_nll_path() -> Path:
    return ARTIFACT_ROOT / "calibration-w80-nll.npz"


def timing_report_path(session: str) -> Path:
    if session not in TIMING_SESSION_ORDER:
        raise ValueError("balanced-200M W80 timing session differs")
    return TIMING_RECEIPT_ROOT / f"{session}.json"


def timing_array_path(session: str) -> Path:
    if session not in TIMING_SESSION_ORDER:
        raise ValueError("balanced-200M W80 timing session differs")
    return ARTIFACT_ROOT / "timing" / f"{session}.npz"


def _w80_matrix(boundaries: np.ndarray, whitespace: np.ndarray) -> np.ndarray:
    if (
        boundaries.dtype != np.uint8
        or whitespace.dtype != np.uint8
        or boundaries.shape != whitespace.shape
        or boundaries.ndim != 2
        or boundaries.shape[1] != base.SEQUENCE_LENGTH
    ):
        raise ValueError("balanced-200M W80 boundary arrays differ")
    rows = [
        causal_window_grid_trace(boundary, spaces, CANDIDATE_PATCH_COUNT).boundaries
        for boundary, spaces in zip(boundaries, whitespace, strict=True)
    ]
    return np.ascontiguousarray(
        padded_hf_patch_matrix(rows, base.SEQUENCE_LENGTH)
    )


def _stream(
    split: str, byte_limit: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stream = build_neural_stream(
        base.SOURCE_PATH,
        language="ko",
        split=split,
        byte_limit=byte_limit,
        sequence_length=base.SEQUENCE_LENGTH,
    )
    inputs, boundaries = stream_arrays(
        stream.data, stream.codepoint_boundaries, stream.sequence_length
    )
    inputs = np.ascontiguousarray(inputs)
    boundaries = np.ascontiguousarray(boundaries.astype(np.uint8, copy=False))
    whitespace = np.ascontiguousarray(
        compact_whitespace_mask(stream.data)
        .reshape(inputs.shape)
        .astype(np.uint8, copy=False)
    )
    return inputs, _w80_matrix(boundaries, whitespace), stream.metadata()


def training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs, matrix, _ = _stream("train", base.NOMINAL_TRAIN_BYTES)
    if len(inputs) != base.AVAILABLE_TRAIN_SEQUENCES:
        raise ValueError("balanced-200M W80 nominal train stream differs")
    order = np.random.default_rng(base.TRAINING_ORDER_SEED).permutation(len(inputs))
    order = np.ascontiguousarray(order[: base.TRAIN_SEQUENCES], dtype=np.int64)
    if len(np.unique(order)) != base.TRAIN_SEQUENCES:
        raise ValueError("balanced-200M W80 training order is not unique")
    return inputs, matrix, order


def preflight_arrays() -> tuple[np.ndarray, np.ndarray]:
    inputs, matrix, _ = _stream(
        "train", base.PREFLIGHT_EXAMPLES * base.SEQUENCE_LENGTH
    )
    if inputs.shape != (base.PREFLIGHT_EXAMPLES, base.SEQUENCE_LENGTH):
        raise ValueError("balanced-200M W80 preflight stream differs")
    return inputs, matrix


def calibration_arrays() -> tuple[np.ndarray, np.ndarray]:
    inputs, matrix, _ = _stream("calibration", base.CALIBRATION_BYTES)
    return inputs, matrix


def timing_contract() -> dict[str, Any]:
    return {
        "mode_order": list(TIMING_MODE_ORDER),
        "session_order": list(TIMING_SESSION_ORDER),
        "warmup_prompts": TIMING_WARMUP_PROMPTS,
        "measured_prompts": TIMING_MEASURED_PROMPTS,
        "repetitions": TIMING_REPETITIONS,
        "correctness_prompts": TIMING_CORRECTNESS_PROMPTS,
        "prompt_bytes": PROMPT_BYTES,
        "controlled_continuation_bytes": CONTINUATION_BYTES,
        "maximum_free_output_bytes": MAXIMUM_FREE_OUTPUT_BYTES,
        "bootstrap_repetitions": TIMING_BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": TIMING_BOOTSTRAP_SEED,
        "minimum_positive_prompts": TIMING_MINIMUM_POSITIVE_PROMPTS,
        "minimum_positive_sessions": TIMING_MINIMUM_POSITIVE_SESSIONS,
        "compact_reference_reduction_exclusive": dict(COMPACT_REFERENCE_REDUCTION),
        "rtol": TIMING_RTOL,
        "atol": TIMING_ATOL,
        "timing_scope": (
            "runtime construction + structural selector + parallel prefill + "
            "cached incremental decode + argmax/UTF8 DFA/stop + final MPS sync"
        ),
        "repetitions_are_independent_samples": False,
        "strong_amplification_requires_lower_over_compact_point": True,
    }


def quality_contract() -> dict[str, Any]:
    return {
        "margin_bpb": QUALITY_MARGIN_BPB,
        "block_size_sequences": QUALITY_BLOCK_SIZE,
        "bootstrap_repetitions": QUALITY_BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": QUALITY_BOOTSTRAP_SEED,
        "mean_delta_must_be_at_most_margin": True,
        "bootstrap_97_5_percent_upper_must_be_at_most_margin": True,
        "independent_full_checkpoint_replay_required": True,
        "historical_test_or_final_metric_used": False,
    }


def data_contract(base_plan: Mapping[str, Any]) -> dict[str, Any]:
    inputs, matrix, order = training_arrays()
    preflight_inputs, preflight_matrix = preflight_arrays()
    calibration_inputs, calibration_matrix = calibration_arrays()
    data = base_plan["data"]
    if (
        array_sha256(inputs) != data["inputs_array_sha256"]
        or array_sha256(order) != data["training_order_array_sha256"]
        or array_sha256(preflight_inputs) != data["preflight_inputs_array_sha256"]
        or array_sha256(calibration_inputs)
        != data["calibration_inputs_array_sha256"]
    ):
        raise ValueError("balanced-200M W80 common data differs from C86/W72")
    return {
        "source_path": data["source_path"],
        "source_sha256": data["source_sha256"],
        "integrity_path": data["integrity_path"],
        "integrity_sha256": data["integrity_sha256"],
        "sequence_length": base.SEQUENCE_LENGTH,
        "used_train_sequences": base.TRAIN_SEQUENCES,
        "used_train_bytes": base.TRAIN_BYTES,
        "inputs_array_sha256": array_sha256(inputs),
        "training_order_seed": base.TRAINING_ORDER_SEED,
        "training_order_array_sha256": array_sha256(order),
        "w80_training_patch_matrix_sha256": array_sha256(matrix),
        "preflight_examples": base.PREFLIGHT_EXAMPLES,
        "preflight_inputs_array_sha256": array_sha256(preflight_inputs),
        "w80_preflight_patch_matrix_sha256": array_sha256(preflight_matrix),
        "calibration_bytes": base.CALIBRATION_BYTES,
        "calibration_examples": len(calibration_inputs),
        "calibration_inputs_array_sha256": array_sha256(calibration_inputs),
        "w80_calibration_patch_matrix_sha256": array_sha256(calibration_matrix),
        "c86_training_patch_matrix_sha256": data["training_patch_matrix_sha256"][
            REFERENCE_ROLE
        ],
        "c86_calibration_patch_matrix_sha256": data[
            "calibration_patch_matrix_sha256"
        ][REFERENCE_ROLE],
        "historical_test_or_final_metric_used": False,
    }


def case_contract(scale_plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_scale_plan(scale_plan, verify_implementation=False)
    prompts, continuations = validate_case_arrays(scale_plan)
    required = TIMING_WARMUP_PROMPTS + TIMING_MEASURED_PROMPTS
    prompts = np.ascontiguousarray(prompts[:required])
    continuations = np.ascontiguousarray(continuations[:required])
    return {
        "upstream_scale_plan_path": SCALE_PLAN_PATH.relative_to(ROOT).as_posix(),
        "upstream_scale_plan_artifact_sha256": hash_file(SCALE_PLAN_PATH),
        "upstream_scale_plan_sha256": scale_plan["plan_sha256"],
        "case_count": required,
        "prompts_array_sha256": array_sha256(prompts),
        "continuations_array_sha256": array_sha256(continuations),
        "selection_uses_w80_output_quality_or_latency": False,
    }


def _tracked_dependency(path: Path, value: Mapping[str, Any], identity_key: str) -> dict[str, Any]:
    identity = value.get(identity_key)
    if not is_sha256(identity):
        raise ValueError(f"balanced-200M W80 upstream identity differs: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "artifact_sha256": hash_file(path),
        identity_key: identity,
    }


def build_plan(
    *,
    git_commit_before_plan: str,
    model_state_sha256: str,
    data: Mapping[str, Any],
    cases: Mapping[str, Any],
    environment: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    base_plan: Mapping[str, Any],
    base_summary: Mapping[str, Any],
    base_verification: Mapping[str, Any],
    failure_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    base.validate_plan(base_plan, verify_implementation=False)
    base.validate_training_summary(base_summary)
    validate_base_verification(base_verification)
    if (
        base_summary["status"] != "balanced_200m_quality_fail"
        or base_summary["quality"]["w72_minus_c86_bpb"]
        != 0.02420047794544211
        or base_verification["training_summary_sha256"]
        != base_summary["summary_sha256"]
        or base_verification["independent_checkpoint_replay_pass"] is not True
        or failure_analysis.get("analysis_sha256")
        != "ef8be87e1d9a300b7c64a90bda094d33f67f4bb520c152eb9b0ceca2e8196f7f"
        or failure_analysis.get("interpretation", {}).get("next_screen_candidate")
        != CANDIDATE_ROLE
        or model_state_sha256 != base_plan["model"]["model_state_sha256"]
    ):
        raise ValueError("balanced-200M W80 prerequisite result differs")
    baseline = base_summary["worker_evidence"][REFERENCE_ROLE]
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_rescue_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_w80_preflight_training_quality_and_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "model": {
            "target_millions": base.TARGET,
            "expected_parameter_count": base.EXPECTED_PARAMETER_COUNT,
            "spec": large_scale_model_spec(base.TARGET, REFERENCE_PATCH_COUNT).to_dict(),
            "model_seed": MODEL_SEED,
            "global_position_limit": GLOBAL_POSITION_LIMIT,
            "initial_state_sha256": model_state_sha256,
        },
        "roles": {
            "candidate": {
                "role": CANDIDATE_ROLE,
                "policy": "causal_whitespace_grid",
                "patch_count": CANDIDATE_PATCH_COUNT,
            },
            "reference": {
                "role": REFERENCE_ROLE,
                "policy": "causal_codepoint_grid",
                "patch_count": REFERENCE_PATCH_COUNT,
                "immutable_training_evidence": dict(baseline),
            },
        },
        "data": dict(data),
        "optimizer": base.optimizer_contract(),
        "preflight": {
            "warmup_updates": base.PREFLIGHT_WARMUP_UPDATES,
            "measurement_updates": base.PREFLIGHT_MEASUREMENT_UPDATES,
            "maximum_memory_fraction": base.MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
            "maximum_hours_candidate": base.MAXIMUM_HOURS_PER_ROLE,
        },
        "quality_gate": quality_contract(),
        "actual_timing_gate": timing_contract(),
        "cases": dict(cases),
        "upstream": {
            "base_plan": _tracked_dependency(BASE_PLAN_PATH, base_plan, "plan_sha256"),
            "base_training_summary": _tracked_dependency(
                BASE_SUMMARY_PATH, base_summary, "summary_sha256"
            ),
            "base_verification": _tracked_dependency(
                BASE_VERIFICATION_PATH, base_verification, "receipt_sha256"
            ),
            "failure_analysis": _tracked_dependency(
                FAILURE_ANALYSIS_PATH, failure_analysis, "analysis_sha256"
            ),
        },
        "environment": dict(environment),
        "implementation_sha256": dict(implementation_sha256),
        "outputs": {
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "preflight_summary_path": PREFLIGHT_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "training_summary_path": TRAINING_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "verification_path": VERIFICATION_OUTPUT_PATH.relative_to(ROOT).as_posix(),
            "timing_receipt_root": TIMING_RECEIPT_ROOT.relative_to(ROOT).as_posix(),
            "timing_summary_path": TIMING_SUMMARY_PATH.relative_to(ROOT).as_posix(),
            "candidate_report_path": training_report_path().relative_to(ROOT).as_posix(),
            "candidate_checkpoint_path": checkpoint_path().relative_to(ROOT).as_posix(),
            "candidate_nll_path": calibration_nll_path().relative_to(ROOT).as_posix(),
        },
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "pure_model_scale_causal_effect_claimed": False,
            "single_candidate_no_fallback": True,
            "w82_or_w84_automatic_followup_authorized": False,
            "actual_timing_requires_quality_and_independent_replay_pass": True,
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
        "actual_timing_gate",
        "cases",
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
        "upstream",
    }
    payload = dict(value)
    claimed = payload.pop("plan_sha256", None)
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("kind") != "balanced_200m_w80_rescue_plan_v1"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("status")
        != "sealed_before_w80_preflight_training_quality_and_timing"
        or not is_git_commit(value.get("git_commit_before_plan"))
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value.get("optimizer") != base.optimizer_contract()
        or value.get("quality_gate") != quality_contract()
        or value.get("actual_timing_gate") != timing_contract()
        or value.get("model")
        != {
            "target_millions": base.TARGET,
            "expected_parameter_count": base.EXPECTED_PARAMETER_COUNT,
            "spec": large_scale_model_spec(base.TARGET, REFERENCE_PATCH_COUNT).to_dict(),
            "model_seed": MODEL_SEED,
            "global_position_limit": GLOBAL_POSITION_LIMIT,
            "initial_state_sha256": value.get("model", {}).get("initial_state_sha256"),
        }
        or not is_sha256(value.get("model", {}).get("initial_state_sha256"))
        or value.get("roles", {}).get("candidate")
        != {
            "role": CANDIDATE_ROLE,
            "policy": "causal_whitespace_grid",
            "patch_count": CANDIDATE_PATCH_COUNT,
        }
        or value.get("roles", {}).get("reference", {}).get("role") != REFERENCE_ROLE
        or value.get("roles", {}).get("reference", {}).get("policy")
        != "causal_codepoint_grid"
        or value.get("roles", {}).get("reference", {}).get("patch_count")
        != REFERENCE_PATCH_COUNT
        or value.get("preflight")
        != {
            "warmup_updates": base.PREFLIGHT_WARMUP_UPDATES,
            "measurement_updates": base.PREFLIGHT_MEASUREMENT_UPDATES,
            "maximum_memory_fraction": base.MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
            "maximum_hours_candidate": base.MAXIMUM_HOURS_PER_ROLE,
        }
        or value.get("claim_boundary")
        != {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "pure_model_scale_causal_effect_claimed": False,
            "single_candidate_no_fallback": True,
            "w82_or_w84_automatic_followup_authorized": False,
            "actual_timing_requires_quality_and_independent_replay_pass": True,
        }
    ):
        raise ValueError("balanced-200M W80 plan identity differs")
    data = value.get("data")
    data_hashes = (
        "source_sha256",
        "integrity_sha256",
        "inputs_array_sha256",
        "training_order_array_sha256",
        "w80_training_patch_matrix_sha256",
        "preflight_inputs_array_sha256",
        "w80_preflight_patch_matrix_sha256",
        "calibration_inputs_array_sha256",
        "w80_calibration_patch_matrix_sha256",
        "c86_training_patch_matrix_sha256",
        "c86_calibration_patch_matrix_sha256",
    )
    if (
        not isinstance(data, Mapping)
        or any(not is_sha256(data.get(key)) for key in data_hashes)
        or data.get("used_train_sequences") != base.TRAIN_SEQUENCES
        or data.get("used_train_bytes") != base.TRAIN_BYTES
        or data.get("calibration_examples")
        != base.CALIBRATION_BYTES // base.SEQUENCE_LENGTH
        or data.get("historical_test_or_final_metric_used") is not False
    ):
        raise ValueError("balanced-200M W80 data identity differs")
    cases = value.get("cases")
    if (
        not isinstance(cases, Mapping)
        or cases.get("case_count")
        != TIMING_WARMUP_PROMPTS + TIMING_MEASURED_PROMPTS
        or not is_sha256(cases.get("prompts_array_sha256"))
        or not is_sha256(cases.get("continuations_array_sha256"))
        or cases.get("selection_uses_w80_output_quality_or_latency") is not False
    ):
        raise ValueError("balanced-200M W80 case identity differs")
    implementation = value.get("implementation_sha256")
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or any(not is_sha256(implementation.get(path)) for path in IMPLEMENTATION_PATHS)
    ):
        raise ValueError("balanced-200M W80 implementation identity differs")
    if current_environment is not None and value.get("environment") != current_environment:
        raise ValueError("balanced-200M W80 environment differs")
    if verify_implementation:
        for relative in IMPLEMENTATION_PATHS:
            path = ROOT / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or hash_file(path) != implementation[relative]
            ):
                raise ValueError(f"balanced-200M W80 implementation differs: {relative}")


def preflight_pass(report: Mapping[str, Any]) -> bool:
    measurement = report.get("measurement")
    maximum = report.get("maximum_driver_allocated_bytes")
    recommended = report.get("recommended_max_memory_bytes")
    return bool(
        report.get("completed") is True
        and report.get("finite") is True
        and report.get("optimizer_state_initialized") is True
        and isinstance(measurement, Mapping)
        and 0 < float(measurement.get("projected_hours", math.inf))
        <= base.MAXIMUM_HOURS_PER_ROLE
        and type(maximum) is int
        and type(recommended) is int
        and 0 < maximum <= base.MAXIMUM_RECOMMENDED_MEMORY_FRACTION * recommended
    )


def build_preflight_summary(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    summary_base_git_commit: str,
    worker_path: str,
    worker_sha256: str,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    passed = preflight_pass(report)
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_preflight_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "w80_preflight_pass" if passed else "w80_preflight_fail",
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "worker_evidence": {"path": worker_path, "sha256": worker_sha256},
        "aggregate": {
            "candidate_preflight_pass": passed,
            "projected_hours": report.get("measurement", {}).get("projected_hours"),
            "training_authorized": passed,
        },
        "claim_boundary": {"resource_only": True, "quality_claimed": False},
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def validate_preflight_summary(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("summary_sha256", None)
    aggregate = value.get("aggregate")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "balanced_200m_w80_preflight_summary_v1"
        or value.get("protocol_id") != PROTOCOL_ID
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or not isinstance(aggregate, Mapping)
        or value.get("status")
        != ("w80_preflight_pass" if aggregate.get("candidate_preflight_pass") else "w80_preflight_fail")
        or aggregate.get("training_authorized")
        is not aggregate.get("candidate_preflight_pass")
    ):
        raise ValueError("balanced-200M W80 preflight summary differs")


def summarize_quality(
    c86_nll: np.ndarray, w80_nll: np.ndarray
) -> dict[str, Any]:
    effects = paired_bpb_effects(c86_nll, w80_nll)
    candidate_bpb = base.bpb_from_sequence_nll(w80_nll)
    reference_bpb = base.bpb_from_sequence_nll(c86_nll)
    delta = candidate_bpb - reference_bpb
    bootstrap = contiguous_block_bootstrap(
        effects,
        block_size=QUALITY_BLOCK_SIZE,
        repetitions=QUALITY_BOOTSTRAP_REPETITIONS,
        seed=QUALITY_BOOTSTRAP_SEED,
    )
    clauses = {
        "mean_delta_at_most_margin": bool(delta <= QUALITY_MARGIN_BPB),
        "bootstrap_upper_at_most_margin": bool(
            float(bootstrap["upper"]) <= QUALITY_MARGIN_BPB
        ),
    }
    return {
        "candidate_bpb": candidate_bpb,
        "reference_bpb": reference_bpb,
        "w80_minus_c86_bpb": delta,
        "maximum_allowed_delta_bpb": QUALITY_MARGIN_BPB,
        "block_bootstrap": bootstrap,
        "clauses": clauses,
        "quality_gate_pass_pending_independent_replay": bool(all(clauses.values())),
        "actual_timing_authorized": False,
    }


def build_training_summary(
    *,
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    summary_base_git_commit: str,
    candidate_evidence: Mapping[str, Any],
    c86_nll: np.ndarray,
    w80_nll: np.ndarray,
) -> dict[str, Any]:
    validate_preflight_summary(preflight)
    quality = summarize_quality(c86_nll, w80_nll)
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_training_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "w80_quality_pass_pending_replay"
            if quality["quality_gate_pass_pending_independent_replay"]
            else "w80_quality_fail"
        ),
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "preflight_artifact_sha256": hash_file(PREFLIGHT_OUTPUT_PATH),
        "preflight_summary_sha256": preflight["summary_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "candidate_evidence": dict(candidate_evidence),
        "reference_evidence": dict(
            plan["roles"]["reference"]["immutable_training_evidence"]
        ),
        "quality": quality,
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "actual_timing_requires_independent_replay": True,
            "pure_scale_effect_claimed": False,
        },
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def validate_training_summary(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("summary_sha256", None)
    quality = value.get("quality")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "balanced_200m_w80_training_summary_v1"
        or value.get("protocol_id") != PROTOCOL_ID
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or not isinstance(quality, Mapping)
        or quality.get("actual_timing_authorized") is not False
        or value.get("status")
        != (
            "w80_quality_pass_pending_replay"
            if quality.get("quality_gate_pass_pending_independent_replay") is True
            else "w80_quality_fail"
        )
    ):
        raise ValueError("balanced-200M W80 training summary differs")


def build_verification_receipt(
    *,
    plan: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    verification_base_git_commit: str,
    replayed_nll_array_sha256: str,
    replayed_quality: Mapping[str, Any],
) -> dict[str, Any]:
    validate_training_summary(training_summary)
    if replayed_quality != training_summary["quality"]:
        raise ValueError("balanced-200M W80 replayed quality differs")
    quality_pass = bool(
        replayed_quality["quality_gate_pass_pending_independent_replay"]
    )
    payload = {
        "schema_version": 1,
        "kind": "balanced_200m_w80_checkpoint_replay_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "w80_quality_and_replay_pass" if quality_pass else "w80_quality_fail",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "training_summary_artifact_sha256": hash_file(TRAINING_OUTPUT_PATH),
        "training_summary_sha256": training_summary["summary_sha256"],
        "verification_base_git_commit": verification_base_git_commit,
        "candidate_checkpoint_sha256": training_summary["candidate_evidence"][
            "checkpoint_sha256"
        ],
        "candidate_checkpoint_state_sha256": training_summary["candidate_evidence"][
            "checkpoint_state_sha256"
        ],
        "replayed_nll_array_sha256": replayed_nll_array_sha256,
        "quality": dict(replayed_quality),
        "independent_full_checkpoint_replay_pass": True,
        "actual_timing_authorized": quality_pass,
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "full_calibration_forward_replayed": True,
            "actual_timing_executed": False,
            "pure_scale_effect_claimed": False,
        },
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_verification_receipt(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    quality = value.get("quality")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "balanced_200m_w80_checkpoint_replay_receipt_v1"
        or value.get("protocol_id") != PROTOCOL_ID
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or not isinstance(quality, Mapping)
        or value.get("independent_full_checkpoint_replay_pass") is not True
        or value.get("actual_timing_authorized")
        is not quality.get("quality_gate_pass_pending_independent_replay")
    ):
        raise ValueError("balanced-200M W80 verification receipt differs")


def timing_role_order(
    session_index: int, prompt_index: int, repetition: int, mode_index: int
) -> tuple[int, int]:
    first = (session_index + prompt_index + repetition + mode_index) % 2
    return (first, 1 - first)


def _crossed_bootstrap(
    candidate: np.ndarray, reference: np.ndarray
) -> tuple[float, float]:
    if (
        candidate.shape != (len(TIMING_SESSION_ORDER), TIMING_MEASURED_PROMPTS)
        or reference.shape != candidate.shape
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(reference))
        or np.any(candidate <= 0)
        or np.any(reference <= 0)
    ):
        raise ValueError("balanced-200M W80 timing cells differ")
    rng = np.random.default_rng(TIMING_BOOTSTRAP_SEED)
    estimates = np.empty(TIMING_BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(TIMING_BOOTSTRAP_REPETITIONS):
        sessions = rng.integers(0, len(TIMING_SESSION_ORDER), len(TIMING_SESSION_ORDER))
        prompts = rng.integers(0, TIMING_MEASURED_PROMPTS, TIMING_MEASURED_PROMPTS)
        left = candidate[np.ix_(sessions, prompts)]
        right = reference[np.ix_(sessions, prompts)]
        estimates[index] = 1.0 - float(np.median(left)) / float(np.median(right))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_actual_timing(
    end_to_end_by_session: Mapping[str, np.ndarray],
    correctness_by_session: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        tuple(end_to_end_by_session) != TIMING_SESSION_ORDER
        or tuple(correctness_by_session) != TIMING_SESSION_ORDER
    ):
        raise ValueError("balanced-200M W80 timing session set differs")
    expected_shape = (
        len(TIMING_MODE_ORDER),
        TIMING_MEASURED_PROMPTS,
        TIMING_REPETITIONS,
        len(TIMING_ROLE_ORDER),
    )
    arrays = []
    correctness_pass = True
    for session in TIMING_SESSION_ORDER:
        array = np.asarray(end_to_end_by_session[session])
        if (
            array.dtype != np.float64
            or array.shape != expected_shape
            or not np.all(np.isfinite(array))
            or np.any(array <= 0)
        ):
            raise ValueError("balanced-200M W80 timing array differs")
        arrays.append(array)
        correctness_pass &= correctness_by_session[session].get("overall_pass") is True
    values = np.stack(arrays)
    medians = np.median(values, axis=3)
    output: dict[str, Any] = {}
    for mode_index, mode in enumerate(TIMING_MODE_ORDER):
        candidate = medians[:, mode_index, :, TIMING_ROLE_ORDER.index(CANDIDATE_ROLE)]
        reference = medians[:, mode_index, :, TIMING_ROLE_ORDER.index(REFERENCE_ROLE)]
        point = 1.0 - float(np.median(candidate)) / float(np.median(reference))
        lower, upper = _crossed_bootstrap(candidate, reference)
        prompt_effects = 1.0 - np.median(candidate, axis=0) / np.median(reference, axis=0)
        session_effects = 1.0 - np.median(candidate, axis=1) / np.median(reference, axis=1)
        compact = COMPACT_REFERENCE_REDUCTION[mode]
        clauses = {
            "correctness": bool(correctness_pass),
            "point_exceeds_compact_reference": bool(point > compact),
            "bootstrap_lower_positive": bool(lower > 0),
            "positive_prompts": bool(
                int(np.sum(prompt_effects > 0)) >= TIMING_MINIMUM_POSITIVE_PROMPTS
            ),
            "all_sessions_positive": bool(
                int(np.sum(session_effects > 0)) >= TIMING_MINIMUM_POSITIVE_SESSIONS
            ),
        }
        output[mode] = {
            "candidate_median_end_to_end_ms": float(np.median(candidate)),
            "reference_median_end_to_end_ms": float(np.median(reference)),
            "end_to_end_reduction": point,
            "crossed_bootstrap_95_interval": {"lower": lower, "upper": upper},
            "compact_reference_reduction": compact,
            "positive_prompt_count": int(np.sum(prompt_effects > 0)),
            "positive_session_count": int(np.sum(session_effects > 0)),
            "clauses": clauses,
            "actual_primary_pass": bool(all(clauses.values())),
            "strong_scale_amplification_support": bool(
                all(clauses.values()) and lower > compact
            ),
        }
    return {
        "by_mode": output,
        "overall_actual_primary_pass": bool(
            all(output[mode]["actual_primary_pass"] for mode in TIMING_MODE_ORDER)
        ),
        "strong_scale_amplification_support": bool(
            all(
                output[mode]["strong_scale_amplification_support"]
                for mode in TIMING_MODE_ORDER
            )
        ),
    }
