"""Pure contracts and statistics for the fixed 46.6M global-heavy bridge."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scale_schedule_extrapolation_core import (
    ATOL,
    CONTINUATION_BYTES,
    CORRECTNESS_PROMPTS,
    GLOBAL_POSITION_LIMIT,
    INNER_REPETITIONS,
    MEASURED_PROMPTS,
    MODEL_SEED,
    PROMPT_BYTES,
    ROOT,
    RTOL,
    SCHEDULE_ORDER,
    SESSION_ORDER,
    WARMUP_PROMPTS,
    canonical_sha256,
    case_contract,
    is_git_commit,
    is_sha256,
    load_case_arrays,
    mechanism_arrays,
    schedule_contract,
)

from jamoflow.hplt3 import hash_file
from jamoflow.neural_model import Phase1ModelSpec

PROTOCOL_ID = "jamoflow-global-heavy-schedule-bridge-v2"
PLAN_PATH = ROOT / "data/manifests/global-heavy-schedule-bridge-v2.json"
ARTIFACT_ROOT = ROOT / "artifacts/global-heavy-schedule-bridge-v2"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
OUTPUT_PATH = ROOT / "results/global-heavy-schedule-bridge-v2/summary.json"
BALANCED_SUMMARY_PATH = ROOT / "results/scale-schedule-preflight-v1/summary.json"
RESOURCE_SUMMARY_PATH = ROOT / "results/large-scale-training-feasibility-v1/summary.json"

EXPECTED_PARAMETER_COUNT = 46_644_640
EXPECTED_GLOBAL_PARAMETER_COUNT = 42_813_440
EXPECTED_GLOBAL_PARAMETER_SHARE = (
    EXPECTED_GLOBAL_PARAMETER_COUNT / EXPECTED_PARAMETER_COUNT
)
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260831
MINIMUM_POINT_REDUCTION = 0.10
MINIMUM_BOOTSTRAP_LOWER_BOUND = 0.08
MINIMUM_POSITIVE_PROMPTS = 15
MINIMUM_POSITIVE_SESSIONS = 3
MINIMUM_SESSIONS_AT_POINT_TARGET = 2
MAXIMUM_RECOMMENDED_MEMORY_FRACTION = 0.75

GLOBAL_HEAVY_SPEC = Phase1ModelSpec(
    sequence_length=512,
    patch_count=86,
    patch_stride=6,
    local_width=160,
    global_width=640,
    local_heads=5,
    global_heads=10,
    encoder_layers=1,
    global_layers=8,
    decoder_layers=1,
    local_ffn=480,
    global_ffn=1920,
    cross_attention_k=2,
    hash_group_size=3,
    hash_vocabulary=16384,
    router_width=160,
    router_heads=5,
    router_layers=2,
    router_ffn=480,
)

IMPLEMENTATION_PATHS = (
    "docs/187-scale-schedule-preflight-result-and-terminal-research-decision.md",
    "docs/192-large-scale-training-feasibility-result-and-architecture-pivot.md",
    "docs/193-global-heavy-schedule-bridge-protocol.md",
    "pyproject.toml",
    "scripts/global_heavy_schedule_core.py",
    "scripts/run_global_heavy_schedule_bridge.py",
    "scripts/run_scale_schedule_extrapolation.py",
    "scripts/scale_schedule_extrapolation_core.py",
    "scripts/seal_global_heavy_schedule_plan.py",
    "scripts/verify_global_heavy_schedule_bridge.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/patching.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/utf8.py",
    "tests/test_global_heavy_schedule_bridge.py",
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


def worker_report_path(session: str) -> Path:
    if session not in SESSION_ORDER:
        raise ValueError("global-heavy session differs")
    return ARTIFACT_ROOT / f"{session}-report.json"


def worker_timing_path(session: str) -> Path:
    if session not in SESSION_ORDER:
        raise ValueError("global-heavy session differs")
    return ARTIFACT_ROOT / f"{session}-timings.npz"


def global_heavy_model_contract() -> dict[str, Any]:
    return {
        "spec": GLOBAL_HEAVY_SPEC.to_dict(),
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
        "expected_global_parameter_count": EXPECTED_GLOBAL_PARAMETER_COUNT,
        "expected_global_parameter_share": EXPECTED_GLOBAL_PARAMETER_SHARE,
        "comparison_balanced_parameter_count": 49_823_488,
        "comparison_balanced_median_reduction": 0.035717306936173254,
        "architecture_selection": (
            "single analytic geometry chosen before timing; local:global width "
            "1:4 and 91.786% of parameters in the global transformer"
        ),
    }


def build_plan(
    *,
    git_commit_before_plan: str,
    model_state_sha256: str,
    environment: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
    balanced_summary: Mapping[str, Any],
    resource_summary: Mapping[str, Any],
) -> dict[str, Any]:
    from large_scale_training_feasibility_core import (
        validate_summary as validate_resource,
    )
    from scale_schedule_preflight_core import validate_scale_schedule_summary

    validate_scale_schedule_summary(balanced_summary)
    validate_resource(resource_summary)
    balanced = balanced_summary["aggregate"]["rows"]["50"]
    if (
        balanced["median_reduction"]
        != global_heavy_model_contract()["comparison_balanced_median_reduction"]
        or resource_summary["aggregate"]["primary_1600_resource_feasible"] is not True
        or not is_sha256(model_state_sha256)
    ):
        raise ValueError("global-heavy upstream evidence differs")
    payload = {
        "schema_version": 1,
        "kind": "global_heavy_schedule_bridge_plan_v2",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_first_global_heavy_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "model": {
            **global_heavy_model_contract(),
            "model_seed": MODEL_SEED,
            "global_position_limit": GLOBAL_POSITION_LIMIT,
            "model_state_sha256": model_state_sha256,
        },
        "cases": case_contract(),
        "schedules": schedule_contract(),
        "timing": {
            "session_order": list(SESSION_ORDER),
            "warmup_prompts": WARMUP_PROMPTS,
            "measured_prompts": MEASURED_PROMPTS,
            "inner_repetitions": INNER_REPETITIONS,
            "correctness_prompts": CORRECTNESS_PROMPTS,
            "prompt_bytes": PROMPT_BYTES,
            "continuation_bytes": CONTINUATION_BYTES,
            "role_order": "(session_index + prompt_index + repetition) mod 2",
            "rtol": RTOL,
            "atol": ATOL,
            "scope": (
                "fresh runtime, parallel prefill, 127 controlled consumes, "
                "final MPS synchronize"
            ),
        },
        "gate": {
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
            "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "minimum_positive_sessions": MINIMUM_POSITIVE_SESSIONS,
            "minimum_sessions_at_point_target": MINIMUM_SESSIONS_AT_POINT_TARGET,
            "maximum_recommended_memory_fraction": (
                MAXIMUM_RECOMMENDED_MEMORY_FRACTION
            ),
            "stop_rule": (
                "failure ends this fixed geometry; no larger or modified geometry fallback"
            ),
        },
        "upstream": {
            "balanced_summary_path": BALANCED_SUMMARY_PATH.relative_to(ROOT).as_posix(),
            "balanced_summary_artifact_sha256": hash_file(BALANCED_SUMMARY_PATH),
            "balanced_summary_sha256": balanced_summary["summary_sha256"],
            "resource_summary_path": RESOURCE_SUMMARY_PATH.relative_to(ROOT).as_posix(),
            "resource_summary_artifact_sha256": hash_file(RESOURCE_SUMMARY_PATH),
            "resource_summary_sha256": resource_summary["summary_sha256"],
        },
        "environment": dict(environment),
        "implementation_sha256": dict(implementation_sha256),
        "outputs": {
            "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        },
        "claim_boundary": {
            "single_fixed_architecture": True,
            "random_weight_systems_test": True,
            "trained_quality_claimed": False,
            "free_running_claimed": False,
            "training_directly_authorized": False,
            "confirmatory_or_final_claimed": False,
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
        "cases",
        "claim_boundary",
        "environment",
        "gate",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "model",
        "outputs",
        "plan_sha256",
        "protocol_id",
        "schedules",
        "schema_version",
        "status",
        "timing",
        "upstream",
    }
    if set(value) != expected:
        raise ValueError("global-heavy plan schema differs")
    payload = dict(value)
    claimed = payload.pop("plan_sha256")
    expected_model = {
        **global_heavy_model_contract(),
        "model_seed": MODEL_SEED,
        "global_position_limit": GLOBAL_POSITION_LIMIT,
        "model_state_sha256": value["model"].get("model_state_sha256"),
    }
    expected_timing = {
        "session_order": list(SESSION_ORDER),
        "warmup_prompts": WARMUP_PROMPTS,
        "measured_prompts": MEASURED_PROMPTS,
        "inner_repetitions": INNER_REPETITIONS,
        "correctness_prompts": CORRECTNESS_PROMPTS,
        "prompt_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "role_order": "(session_index + prompt_index + repetition) mod 2",
        "rtol": RTOL,
        "atol": ATOL,
        "scope": (
            "fresh runtime, parallel prefill, 127 controlled consumes, "
            "final MPS synchronize"
        ),
    }
    expected_gate = {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
        "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "minimum_positive_sessions": MINIMUM_POSITIVE_SESSIONS,
        "minimum_sessions_at_point_target": MINIMUM_SESSIONS_AT_POINT_TARGET,
        "maximum_recommended_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
        "stop_rule": (
            "failure ends this fixed geometry; no larger or modified geometry fallback"
        ),
    }
    if (
        value["schema_version"] != 1
        or value["kind"] != "global_heavy_schedule_bridge_plan_v2"
        or value["protocol_id"] != PROTOCOL_ID
        or value["status"] != "sealed_before_first_global_heavy_timing"
        or not is_git_commit(value["git_commit_before_plan"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value["model"] != expected_model
        or not is_sha256(value["model"].get("model_state_sha256"))
        or value["cases"] != case_contract()
        or value["schedules"] != schedule_contract()
        or value["timing"] != expected_timing
        or value["gate"] != expected_gate
        or value["outputs"]
        != {
            "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        }
        or value["claim_boundary"]
        != {
            "single_fixed_architecture": True,
            "random_weight_systems_test": True,
            "trained_quality_claimed": False,
            "free_running_claimed": False,
            "training_directly_authorized": False,
            "confirmatory_or_final_claimed": False,
        }
    ):
        raise ValueError("global-heavy plan identity differs")
    upstream = value["upstream"]
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("balanced_summary_path")
        != BALANCED_SUMMARY_PATH.relative_to(ROOT).as_posix()
        or upstream.get("resource_summary_path")
        != RESOURCE_SUMMARY_PATH.relative_to(ROOT).as_posix()
        or any(
            not is_sha256(upstream.get(key))
            for key in (
                "balanced_summary_artifact_sha256",
                "balanced_summary_sha256",
                "resource_summary_artifact_sha256",
                "resource_summary_sha256",
            )
        )
    ):
        raise ValueError("global-heavy upstream identity differs")
    implementation = value["implementation_sha256"]
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or any(not is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
    ):
        raise ValueError("global-heavy implementation set differs")
    if current_environment is not None and value["environment"] != current_environment:
        raise ValueError("global-heavy environment differs")
    if verify_implementation:
        for relative in IMPLEMENTATION_PATHS:
            path = ROOT / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or hash_file(path) != implementation[relative]
            ):
                raise ValueError(f"global-heavy implementation differs: {relative}")


def role_order(
    session_index: int, prompt_index: int, repetition: int
) -> tuple[int, int]:
    if (
        not 0 <= session_index < len(SESSION_ORDER)
        or not 0 <= prompt_index < MEASURED_PROMPTS
        or not 0 <= repetition < INNER_REPETITIONS
    ):
        raise ValueError("global-heavy role-order coordinate differs")
    first = (session_index + prompt_index + repetition) % 2
    return first, 1 - first


def _correctness_pass(value: Mapping[str, Any]) -> bool:
    expected = {
        "argmax_comparisons",
        "argmax_exact",
        "boundary_prefix_comparisons",
        "boundary_trace_exact",
        "cache_diagnostics_exact",
        "maximum_normalized_logit_error",
        "offline_boundary_prefix_exact",
    }
    comparisons = CORRECTNESS_PROMPTS * CONTINUATION_BYTES
    maximum = value.get("maximum_normalized_logit_error")
    return bool(
        set(value) == expected
        and value["argmax_comparisons"] == comparisons
        and value["argmax_exact"] == comparisons
        and value["boundary_prefix_comparisons"] == comparisons
        and value["boundary_trace_exact"] is True
        and value["cache_diagnostics_exact"] is True
        and value["offline_boundary_prefix_exact"] is True
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(float(maximum))
        and 0 <= float(maximum) <= 1
    )


def _reduction(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.median(reference))
    if denominator <= 0:
        raise ValueError("global-heavy reference timing is nonpositive")
    return 1.0 - float(np.median(candidate)) / denominator


def _bootstrap(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    if candidate.shape != (len(SESSION_ORDER), MEASURED_PROMPTS):
        raise ValueError("global-heavy bootstrap candidate shape differs")
    if reference.shape != candidate.shape:
        raise ValueError("global-heavy bootstrap reference shape differs")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sessions = rng.integers(0, len(SESSION_ORDER), size=len(SESSION_ORDER))
        prompts = rng.integers(0, MEASURED_PROMPTS, size=MEASURED_PROMPTS)
        cells = np.ix_(sessions, prompts)
        values[index] = _reduction(candidate[cells], reference[cells])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def summarize(
    timings: np.ndarray,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    array = np.asarray(timings)
    if (
        array.dtype != np.float64
        or array.shape
        != (
            len(SESSION_ORDER),
            MEASURED_PROMPTS,
            INNER_REPETITIONS,
            len(SCHEDULE_ORDER),
        )
        or not np.all(np.isfinite(array))
        or np.any(array <= 0)
    ):
        raise ValueError("global-heavy timing array differs")
    rows = tuple(reports)
    if len(rows) != len(SESSION_ORDER):
        raise ValueError("global-heavy report count differs")
    evidence_by_session: dict[str, bool] = {}
    memory_fractions: list[float] = []
    for session, report in zip(SESSION_ORDER, rows, strict=True):
        correctness = report.get("correctness")
        memory = report.get("maximum_driver_allocated_bytes")
        recommended = report.get("recommended_max_memory_bytes")
        valid = bool(
            report.get("session_id") == session
            and report.get("parameter_count") == EXPECTED_PARAMETER_COUNT
            and report.get("global_parameter_count")
            == EXPECTED_GLOBAL_PARAMETER_COUNT
            and report.get("global_parameter_share")
            == EXPECTED_GLOBAL_PARAMETER_SHARE
            and report.get("same_model_object_for_both_schedules") is True
            and isinstance(correctness, Mapping)
            and set(correctness) == set(SCHEDULE_ORDER)
            and all(_correctness_pass(correctness[role]) for role in SCHEDULE_ORDER)
            and type(memory) is int
            and type(recommended) is int
            and 0 < memory <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION * recommended
            and report.get("environment_start") == report.get("environment_end")
        )
        evidence_by_session[session] = valid
        memory_fractions.append(
            float(memory / recommended)
            if type(memory) is int and type(recommended) is int and recommended > 0
            else math.inf
        )
    cells = np.median(array, axis=2)
    reference = cells[:, :, SCHEDULE_ORDER.index("c86")]
    candidate = cells[:, :, SCHEDULE_ORDER.index("w72")]
    point = _reduction(candidate, reference)
    lower, upper = _bootstrap(candidate, reference)
    session_reductions = {
        session: _reduction(candidate[index], reference[index])
        for index, session in enumerate(SESSION_ORDER)
    }
    prompt_reductions = np.asarray(
        [
            _reduction(candidate[:, index], reference[:, index])
            for index in range(MEASURED_PROMPTS)
        ]
    )
    positive_prompts = int(np.sum(prompt_reductions > 0))
    positive_sessions = sum(value > 0 for value in session_reductions.values())
    sessions_at_target = sum(
        value >= MINIMUM_POINT_REDUCTION for value in session_reductions.values()
    )
    gates = {
        "evidence_valid": all(evidence_by_session.values()),
        "point_reduction_at_least_10_percent": point >= MINIMUM_POINT_REDUCTION,
        "bootstrap_lower_at_least_8_percent": lower
        >= MINIMUM_BOOTSTRAP_LOWER_BOUND,
        "positive_prompts_at_least_15": positive_prompts
        >= MINIMUM_POSITIVE_PROMPTS,
        "all_three_sessions_positive": positive_sessions
        >= MINIMUM_POSITIVE_SESSIONS,
        "at_least_two_sessions_reach_10_percent": sessions_at_target
        >= MINIMUM_SESSIONS_AT_POINT_TARGET,
    }
    passed = all(gates.values())
    patch_counts, _ = mechanism_arrays(*load_case_arrays()[:2])
    measured_counts = patch_counts[WARMUP_PROMPTS:]
    patch_reduction = 1.0 - float(
        measured_counts[:, SCHEDULE_ORDER.index("w72")].sum()
    ) / float(measured_counts[:, SCHEDULE_ORDER.index("c86")].sum())
    return {
        "protocol_id": PROTOCOL_ID,
        "c86_median_ms": float(np.median(reference)),
        "w72_median_ms": float(np.median(candidate)),
        "median_reduction": point,
        "crossed_bootstrap_95_interval": {"lower": lower, "upper": upper},
        "session_reductions": session_reductions,
        "positive_prompt_count": positive_prompts,
        "positive_session_count": positive_sessions,
        "sessions_at_least_10_percent": sessions_at_target,
        "session_evidence_validity": evidence_by_session,
        "maximum_memory_fraction": max(memory_fractions),
        "patch_event_reduction": patch_reduction,
        "affected_time_share_proxy": point / patch_reduction,
        "balanced_49m_median_reduction": global_heavy_model_contract()[
            "comparison_balanced_median_reduction"
        ],
        "increment_over_balanced_percentage_points": 100
        * (
            point
            - global_heavy_model_contract()["comparison_balanced_median_reduction"]
        ),
        "gates": gates,
        "overall_threshold_pass": passed,
        "status": (
            "global_heavy_10_percent_headroom_detected"
            if passed
            else "global_heavy_fixed_geometry_failed_10_percent_gate"
        ),
        "trained_quality_protocol_may_be_designed": passed,
    }


def build_summary(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    summary_base_git_commit: str,
    worker_evidence: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not is_sha256(plan_artifact_sha256)
        or not is_git_commit(summary_base_git_commit)
        or set(worker_evidence) != set(SESSION_ORDER)
        or aggregate.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("global-heavy summary dependency differs")
    for session in SESSION_ORDER:
        row = worker_evidence[session]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"report_path", "report_sha256", "timing_path", "timing_sha256"}
            or row["report_path"]
            != worker_report_path(session).relative_to(ROOT).as_posix()
            or row["timing_path"]
            != worker_timing_path(session).relative_to(ROOT).as_posix()
            or not is_sha256(row["report_sha256"])
            or not is_sha256(row["timing_sha256"])
        ):
            raise ValueError("global-heavy worker evidence differs")
    payload = {
        "schema_version": 1,
        "kind": "global_heavy_schedule_bridge_summary_v2",
        "protocol_id": PROTOCOL_ID,
        "status": aggregate["status"],
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "worker_evidence": dict(worker_evidence),
        "aggregate": dict(aggregate),
        "claim_boundary": {
            "random_weight_systems_result": True,
            "trained_quality_claimed": False,
            "free_running_claimed": False,
            "training_directly_authorized": False,
            "single_fixed_architecture": True,
        },
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


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
        raise ValueError("global-heavy summary schema differs")
    payload = dict(value)
    claimed = payload.pop("summary_sha256")
    if (
        value["schema_version"] != 1
        or value["kind"] != "global_heavy_schedule_bridge_summary_v2"
        or value["protocol_id"] != PROTOCOL_ID
        or not is_sha256(value["plan_artifact_sha256"])
        or not is_sha256(value["plan_sha256"])
        or not is_git_commit(value["summary_base_git_commit"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or value["status"] != value["aggregate"].get("status")
        or value["claim_boundary"]
        != {
            "random_weight_systems_result": True,
            "trained_quality_claimed": False,
            "free_running_claimed": False,
            "training_directly_authorized": False,
            "single_fixed_architecture": True,
        }
    ):
        raise ValueError("global-heavy summary identity differs")
