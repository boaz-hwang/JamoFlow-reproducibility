"""Contracts for baseline-only EXAONE resource calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from exaone_retrieval_data import (
    CASES_PATH,
    MEASURED_CASES,
    PRIMARY_MODEL,
    ROOT,
    TOTAL_CASES,
    WARMUP_CASES,
    array_sha256,
    canonical_bytes,
    canonical_sha256,
    hash_file,
    is_sha256,
    read_validated_compatibility_result,
    read_verification,
)
from exaone_retrieval_data import (
    PLAN_PATH as DATA_PLAN_PATH,
)
from exaone_retrieval_data import (
    SEAL_PATH as DATA_SEAL_PATH,
)
from exaone_retrieval_data import (
    VERIFICATION_PATH as DATA_VERIFICATION_PATH,
)
from large_model_retrieval_preflight import (
    environment_identity,
    token_sequence_sha256,
    validate_environment,
)

INVALIDATED_V1_PLAN_PATH = ROOT / "data/manifests/exaone-resource-calibration-v1.json"
INVALIDATED_V1_ACTIVE_PATH = ROOT / "artifacts/exaone-resource-calibration-v1/.active"
INVALIDATED_V1_RESULT_PATH = (
    ROOT / "results/exaone-resource-calibration-v1/invalidation.json"
)
INVALIDATED_V2_PLAN_PATH = ROOT / "data/manifests/exaone-resource-calibration-v2.json"
INVALIDATED_V2_RESULT_PATH = (
    ROOT / "results/exaone-resource-calibration-v2/invalidation.json"
)

PLAN_PATH = ROOT / "data/manifests/exaone-resource-calibration-v3.json"
ARTIFACT_ROOT = ROOT / "artifacts/exaone-resource-calibration-v3"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
BASELINE_ARTIFACT_PATH = ARTIFACT_ROOT / "baseline.npz"
RESULT_PATH = ROOT / "results/exaone-resource-calibration-v3/summary.json"

PROTOCOL_ID = "jamoflow-exaone-resource-calibration-v3"
PLAN_KIND = "exaone_resource_calibration_plan_v3"
RESULT_KIND = "exaone_resource_calibration_result_v3"

OUTPUT_TOKENS = 128
CALIBRATION_REPETITIONS = 1
MAXIMUM_MEMORY_FRACTION = 0.75
MAXIMUM_ACTUAL_CAMPAIGN_HOURS = 8.0
CANDIDATE_UPPER_TIME_MULTIPLIER = 2.0
SESSION_FIXED_OVERHEAD_SECONDS = 120.0
ACTUAL_SCHEDULE_CANDIDATES = ((5, 3), (5, 2), (5, 1), (3, 1))

BASELINE_ARRAY_NAMES = (
    "decoded_utf8_sha256",
    "elapsed_ns",
    "output_token_ids",
    "output_token_sha256",
    "prompt_token_count",
    "target_generation_forward_calls",
    "target_prefill_forward_calls",
)

IMPLEMENTATION_PATHS = (
    "docs/178-exaone-retrieval-data-result-and-resource-calibration-decision.md",
    "docs/179-exaone-baseline-resource-calibration-protocol.md",
    "docs/180-exaone-resource-calibration-v1-invalidation-and-v2-correction.md",
    "docs/181-exaone-resource-calibration-v2-invalidation-and-v3-correction.md",
    "requirements/apple-retrieval-v1.txt",
    "scripts/exaone_actual_runtime.py",
    "scripts/exaone_resource_calibration.py",
    "scripts/exaone_retrieval_data.py",
    "scripts/mlx_retrieval_runtime.py",
    "scripts/run_exaone_resource_calibration.py",
    "scripts/seal_exaone_resource_calibration_plan.py",
    "tests/test_exaone_resource_calibration.py",
)

INVALIDATED_V1_IDENTITY = {
    "active_marker_sha256": (
        "237305827eac34e68786335988862228b8bc613dca7d87ba1da08ae60bd0f8bb"
    ),
    "invalidation_sha256": (
        "8dec5399fe144bf670cfa90c40b0d4aab1afa22ca88ed80bdb385cccfac48a74"
    ),
    "plan_artifact_sha256": (
        "0423924aa60206d88f85f37d256d99529cac8b879aecd1851020a1fdd7ea5f40"
    ),
    "plan_sha256": ("3efc5694169e87c7c30058e120456e0d93eb6b01cc9eab0c82a383cb7f8fcd0b"),
    "runner_git_commit": "906088292885e4219a320b97893660daba40c326",
}

INVALIDATED_V2_IDENTITY = {
    "active_marker_sha256": (
        "36ba9cee76855ede80e488e92fab8c6213cf84bc5f60973bfa0baf1dba807cf7"
    ),
    "invalidation_sha256": (
        "6cf34895d6c413a31a1de667d098abda91bd9283d7385a1eb5e2e5935ce1616c"
    ),
    "plan_artifact_sha256": (
        "c83fd6751fe55996998784819f32f09fe3aa364fbc684aaea789e5f95bbf6669"
    ),
    "plan_sha256": ("34f22dfdab4abd5ec81f1a4dff6dd99377d3ddb2dbb37d116af0175d1194a8e5"),
    "runner_git_commit": "cd97adb513ec73fa96ca356f9772f5247fbf0828",
}


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("EXAONE resource implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def read_invalidated_v1() -> dict[str, Any]:
    value = json.loads(INVALIDATED_V1_RESULT_PATH.read_text(encoding="utf-8"))
    expected = {
        "active_marker_payload",
        "active_marker_sha256",
        "baseline_latency_observed",
        "baseline_model_output_observed",
        "baseline_trial_count",
        "candidate_acceptance_observed",
        "candidate_executed",
        "candidate_latency_observed",
        "failed_stage",
        "failure_class",
        "invalidation_sha256",
        "kind",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "runner_git_commit",
        "schema_version",
        "status",
        "successor_protocol_id",
    }
    unsigned = dict(value)
    recorded = unsigned.pop("invalidation_sha256", None)
    expected_active = {
        "plan_artifact_sha256": INVALIDATED_V1_IDENTITY["plan_artifact_sha256"],
        "plan_sha256": INVALIDATED_V1_IDENTITY["plan_sha256"],
        "runner_git_commit": INVALIDATED_V1_IDENTITY["runner_git_commit"],
    }
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("kind") != "exaone_resource_calibration_v1_invalidation"
        or value.get("protocol_id") != "jamoflow-exaone-resource-calibration-v1"
        or value.get("status") != "invalidated_before_first_baseline_trial"
        or value.get("failed_stage")
        != "model_load_validation_before_trial_construction"
        or value.get("failure_class") != "mlx_loaded_config_projection_key_mismatch"
        or value.get("baseline_trial_count") != 0
        or value.get("baseline_latency_observed") is not False
        or value.get("baseline_model_output_observed") is not False
        or value.get("candidate_executed") is not False
        or value.get("candidate_latency_observed") is not False
        or value.get("candidate_acceptance_observed") is not False
        or value.get("successor_protocol_id")
        != "jamoflow-exaone-resource-calibration-v2"
        or value.get("plan_artifact_sha256")
        != INVALIDATED_V1_IDENTITY["plan_artifact_sha256"]
        or value.get("plan_sha256") != INVALIDATED_V1_IDENTITY["plan_sha256"]
        or value.get("runner_git_commit")
        != INVALIDATED_V1_IDENTITY["runner_git_commit"]
        or value.get("active_marker_sha256")
        != INVALIDATED_V1_IDENTITY["active_marker_sha256"]
        or value.get("active_marker_payload") != expected_active
        or hashlib.sha256(canonical_bytes(expected_active)).hexdigest()
        != INVALIDATED_V1_IDENTITY["active_marker_sha256"]
        or recorded != INVALIDATED_V1_IDENTITY["invalidation_sha256"]
        or canonical_sha256(unsigned) != recorded
        or hash_file(INVALIDATED_V1_PLAN_PATH)
        != INVALIDATED_V1_IDENTITY["plan_artifact_sha256"]
    ):
        raise ValueError("EXAONE resource V1 invalidation differs")
    return value


def read_invalidated_v2() -> dict[str, Any]:
    value = json.loads(INVALIDATED_V2_RESULT_PATH.read_text(encoding="utf-8"))
    expected = {
        "active_marker_payload",
        "active_marker_sha256",
        "baseline_artifact_persisted",
        "baseline_generation_entered",
        "baseline_numeric_latency_exposed",
        "baseline_output_exposed",
        "baseline_summary_persisted",
        "candidate_acceptance_observed",
        "candidate_executed",
        "candidate_latency_observed",
        "completed_baseline_trial_count_recorded",
        "failed_stage",
        "failure_class",
        "invalidation_sha256",
        "kind",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "runner_git_commit",
        "schema_version",
        "status",
        "successor_protocol_id",
    }
    unsigned = dict(value)
    recorded = unsigned.pop("invalidation_sha256", None)
    expected_active = {
        "plan_artifact_sha256": INVALIDATED_V2_IDENTITY["plan_artifact_sha256"],
        "plan_sha256": INVALIDATED_V2_IDENTITY["plan_sha256"],
        "runner_git_commit": INVALIDATED_V2_IDENTITY["runner_git_commit"],
    }
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or value.get("kind") != "exaone_resource_calibration_v2_invalidation"
        or value.get("protocol_id") != "jamoflow-exaone-resource-calibration-v2"
        or value.get("status")
        != "invalidated_during_baseline_trial_before_artifact_publication"
        or value.get("failed_stage") != "post_timer_generated_decode_correctness_check"
        or value.get("failure_class") != "generated_token_decode_reencode_not_identity"
        or value.get("baseline_generation_entered") is not True
        or value.get("baseline_artifact_persisted") is not False
        or value.get("baseline_summary_persisted") is not False
        or value.get("baseline_numeric_latency_exposed") is not False
        or value.get("baseline_output_exposed") is not False
        or value.get("completed_baseline_trial_count_recorded") is not False
        or value.get("candidate_executed") is not False
        or value.get("candidate_latency_observed") is not False
        or value.get("candidate_acceptance_observed") is not False
        or value.get("successor_protocol_id") != PROTOCOL_ID
        or value.get("plan_artifact_sha256")
        != INVALIDATED_V2_IDENTITY["plan_artifact_sha256"]
        or value.get("plan_sha256") != INVALIDATED_V2_IDENTITY["plan_sha256"]
        or value.get("runner_git_commit")
        != INVALIDATED_V2_IDENTITY["runner_git_commit"]
        or value.get("active_marker_sha256")
        != INVALIDATED_V2_IDENTITY["active_marker_sha256"]
        or value.get("active_marker_payload") != expected_active
        or hashlib.sha256(canonical_bytes(expected_active)).hexdigest()
        != INVALIDATED_V2_IDENTITY["active_marker_sha256"]
        or recorded != INVALIDATED_V2_IDENTITY["invalidation_sha256"]
        or canonical_sha256(unsigned) != recorded
        or hash_file(INVALIDATED_V2_PLAN_PATH)
        != INVALIDATED_V2_IDENTITY["plan_artifact_sha256"]
    ):
        raise ValueError("EXAONE resource V2 invalidation differs")
    return value


def dependency_identity() -> dict[str, dict[str, Any]]:
    read_verification()
    read_invalidated_v1()
    read_invalidated_v2()
    paths = {
        "case_artifact": CASES_PATH,
        "data_plan": DATA_PLAN_PATH,
        "data_seal": DATA_SEAL_PATH,
        "data_verification": DATA_VERIFICATION_PATH,
        "invalidated_v1_plan": INVALIDATED_V1_PLAN_PATH,
        "invalidated_v1_result": INVALIDATED_V1_RESULT_PATH,
        "invalidated_v2_plan": INVALIDATED_V2_PLAN_PATH,
        "invalidated_v2_result": INVALIDATED_V2_RESULT_PATH,
    }
    return {
        name: {
            "bytes": path.stat().st_size,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
        for name, path in paths.items()
    }


def resource_contract() -> dict[str, Any]:
    return {
        "actual_schedule_candidates": [
            {"inner_repetitions": repetitions, "sessions": sessions}
            for sessions, repetitions in ACTUAL_SCHEDULE_CANDIDATES
        ],
        "calibration_repetitions": CALIBRATION_REPETITIONS,
        "candidate_executed": False,
        "candidate_upper_time_multiplier": CANDIDATE_UPPER_TIME_MULTIPLIER,
        "case_order": "sealed_rank_order_warmup_then_measured",
        "maximum_actual_campaign_hours": MAXIMUM_ACTUAL_CAMPAIGN_HOURS,
        "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
        "measured_cases": MEASURED_CASES,
        "model_and_case_load_inside_timer": False,
        "output_tokens": OUTPUT_TOKENS,
        "prompt_tokenization_inside_timer": True,
        "result_detokenization_inside_timer": True,
        "session_fixed_overhead_seconds": SESSION_FIXED_OVERHEAD_SECONDS,
        "target_path": "ordinary_greedy_only",
        "warmup_cases": WARMUP_CASES,
    }


def result_input_contract() -> dict[str, bool]:
    return {
        "baseline_latency": True,
        "baseline_model_output": True,
        "candidate_acceptance": False,
        "candidate_latency": False,
        "candidate_model_output": False,
        "retrieval_table_loaded": False,
    }


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "actual_candidate_efficiency_tested": False,
        "baseline_resource_feasibility_only": True,
        "generic_retrieval_scale_transfer_tested": False,
        "publication_efficiency_claim": False,
        "session_and_repetition_selection_is_baseline_only": True,
    }


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if not _is_git_commit(git_commit_before_plan):
        raise ValueError("EXAONE resource pre-plan commit differs")
    dependencies = dependency_identity()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_baseline_resource_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependencies,
        "implementation_sha256": implementation_identity(),
        "environment": environment_identity(),
        "resource_contract": resource_contract(),
        "result_inputs": result_input_contract(),
        "outputs": {
            "baseline_artifact_path": BASELINE_ARTIFACT_PATH.relative_to(
                ROOT
            ).as_posix(),
            "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        },
        "claim_boundary": claim_boundary_contract(),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    validate_plan(payload, verify_derived=True)
    return payload


def _validate_dependency_identity(value: object) -> None:
    expected_paths = {
        "case_artifact": CASES_PATH,
        "data_plan": DATA_PLAN_PATH,
        "data_seal": DATA_SEAL_PATH,
        "data_verification": DATA_VERIFICATION_PATH,
        "invalidated_v1_plan": INVALIDATED_V1_PLAN_PATH,
        "invalidated_v1_result": INVALIDATED_V1_RESULT_PATH,
        "invalidated_v2_plan": INVALIDATED_V2_PLAN_PATH,
        "invalidated_v2_result": INVALIDATED_V2_RESULT_PATH,
    }
    if not isinstance(value, Mapping) or set(value) != set(expected_paths):
        raise ValueError("EXAONE resource dependency set differs")
    for name, row in value.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "path", "sha256"}
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or row["path"] != expected_paths[name].relative_to(ROOT).as_posix()
            or not is_sha256(row["sha256"])
        ):
            raise ValueError(f"EXAONE resource dependency differs: {name}")


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected = {
        "claim_boundary",
        "dependencies",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "outputs",
        "plan_sha256",
        "protocol_id",
        "resource_contract",
        "result_inputs",
        "schema_version",
        "status",
    }
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    _validate_dependency_identity(plan.get("dependencies"))
    implementation = plan.get("implementation_sha256")
    environment = plan.get("environment")
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("kind") != PLAN_KIND
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_baseline_resource_timing"
        or not _is_git_commit(plan.get("git_commit_before_plan"))
        or not isinstance(implementation, Mapping)
        or len(implementation) != len(IMPLEMENTATION_PATHS)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or not all(is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
        or not isinstance(environment, Mapping)
        or plan.get("resource_contract") != resource_contract()
        or plan.get("result_inputs") != result_input_contract()
        or plan.get("claim_boundary") != claim_boundary_contract()
        or plan.get("outputs")
        != {
            "baseline_artifact_path": BASELINE_ARTIFACT_PATH.relative_to(
                ROOT
            ).as_posix(),
            "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        }
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE resource plan identity differs")
    validate_environment(environment)
    if verify_derived:
        if plan["dependencies"] != dependency_identity():
            raise ValueError("EXAONE resource dependencies changed")
        if plan["implementation_sha256"] != implementation_identity():
            raise ValueError("EXAONE resource implementation changed")
        if plan["environment"] != environment_identity():
            raise ValueError("EXAONE resource environment changed")


def read_plan(*, verify_derived: bool) -> dict[str, Any]:
    value = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    validate_plan(value, verify_derived=verify_derived)
    return value


def select_actual_schedule(
    *,
    model_load_seconds: float,
    warmup_total_seconds: float,
    measured_total_seconds: float,
) -> dict[str, Any]:
    values = (model_load_seconds, warmup_total_seconds, measured_total_seconds)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("EXAONE baseline resource durations differ")
    projections = []
    selected = None
    for sessions, repetitions in ACTUAL_SCHEDULE_CANDIDATES:
        projected_seconds = sessions * (
            model_load_seconds
            + SESSION_FIXED_OVERHEAD_SECONDS
            + (warmup_total_seconds + repetitions * measured_total_seconds)
            * (1.0 + CANDIDATE_UPPER_TIME_MULTIPLIER)
        )
        row = {
            "inner_repetitions": repetitions,
            "projected_campaign_hours": projected_seconds / 3_600.0,
            "sessions": sessions,
        }
        projections.append(row)
        if (
            selected is None
            and row["projected_campaign_hours"] <= MAXIMUM_ACTUAL_CAMPAIGN_HOURS
        ):
            selected = row
    return {
        "candidate_upper_time_multiplier": CANDIDATE_UPPER_TIME_MULTIPLIER,
        "maximum_campaign_hours": MAXIMUM_ACTUAL_CAMPAIGN_HOURS,
        "projections": projections,
        "selected": selected,
        "status": "feasible" if selected is not None else "infeasible",
    }


def artifact_descriptor(
    payload: bytes, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "arrays": {
            name: {
                "dtype": str(np.asarray(value).dtype),
                "sha256": array_sha256(np.asarray(value)),
                "shape": list(np.asarray(value).shape),
            }
            for name, value in sorted(arrays.items())
        },
        "bytes": len(payload),
        "path": BASELINE_ARTIFACT_PATH.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_artifact() -> dict[str, np.ndarray]:
    with np.load(BASELINE_ARTIFACT_PATH, allow_pickle=False) as archive:
        if set(archive.files) != set(BASELINE_ARRAY_NAMES):
            raise ValueError("EXAONE resource artifact keys differ")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def validate_baseline_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "decoded_utf8_sha256": (np.dtype("uint8"), (TOTAL_CASES, 32)),
        "elapsed_ns": (np.dtype("int64"), (TOTAL_CASES,)),
        "output_token_ids": (np.dtype("uint32"), (TOTAL_CASES, OUTPUT_TOKENS)),
        "output_token_sha256": (np.dtype("uint8"), (TOTAL_CASES, 32)),
        "prompt_token_count": (np.dtype("uint16"), (TOTAL_CASES,)),
        "target_generation_forward_calls": (
            np.dtype("uint16"),
            (TOTAL_CASES,),
        ),
        "target_prefill_forward_calls": (np.dtype("uint8"), (TOTAL_CASES,)),
    }
    if set(arrays) != set(expected):
        raise ValueError("EXAONE resource array set differs")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"EXAONE resource array differs: {name}")
    if (
        np.any(arrays["elapsed_ns"] <= 0)
        or np.any(arrays["prompt_token_count"] != 128)
        or np.any(
            arrays["output_token_ids"]
            >= PRIMARY_MODEL["config_projection"]["vocab_size"]
        )
        or np.any(arrays["target_generation_forward_calls"] != OUTPUT_TOKENS)
        or np.any(arrays["target_prefill_forward_calls"] != 1)
    ):
        raise ValueError("EXAONE resource counters differ")
    expected_hashes = np.asarray(
        [
            list(
                bytes.fromhex(token_sequence_sha256(tuple(int(token) for token in row)))
            )
            for row in arrays["output_token_ids"]
        ],
        dtype=np.uint8,
    )
    if not np.array_equal(arrays["output_token_sha256"], expected_hashes):
        raise ValueError("EXAONE resource output-token hashes differ")


def timing_summary(
    arrays: Mapping[str, np.ndarray], *, model_load_seconds: float
) -> dict[str, float]:
    validate_baseline_arrays(arrays)
    if not math.isfinite(model_load_seconds) or model_load_seconds <= 0:
        raise ValueError("EXAONE model-load duration differs")
    measured = arrays["elapsed_ns"][WARMUP_CASES:].astype(np.float64) / 1e9
    warmup = arrays["elapsed_ns"][:WARMUP_CASES].astype(np.float64) / 1e9
    measured_total = float(np.sum(measured))
    return {
        "measured_case_median_seconds": float(np.median(measured)),
        "measured_case_p95_seconds": float(np.quantile(measured, 0.95)),
        "measured_total_seconds": measured_total,
        "model_load_seconds": float(model_load_seconds),
        "output_tokens_per_second": float(
            MEASURED_CASES * OUTPUT_TOKENS / measured_total
        ),
        "warmup_total_seconds": float(np.sum(warmup)),
    }


def build_result(
    *,
    plan: Mapping[str, Any],
    runner_git_commit: str,
    arrays: Mapping[str, np.ndarray],
    baseline_artifact_bytes: bytes,
    model_load_seconds: float,
    memory: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    timing = timing_summary(arrays, model_load_seconds=model_load_seconds)
    schedule = select_actual_schedule(
        model_load_seconds=model_load_seconds,
        warmup_total_seconds=timing["warmup_total_seconds"],
        measured_total_seconds=timing["measured_total_seconds"],
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": (
            "pass_baseline_resource_feasibility"
            if schedule["status"] == "feasible" and memory.get("safety_pass") is True
            else "stop_resource_infeasible"
        ),
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "runner_git_commit": runner_git_commit,
        "dependencies": plan["dependencies"],
        "environment": plan["environment"],
        "model_identity": dict(model_identity),
        "baseline_artifact": artifact_descriptor(baseline_artifact_bytes, arrays),
        "timing": timing,
        "memory": dict(memory),
        "actual_schedule_decision": schedule,
        "result_inputs": result_input_contract(),
        "claim_boundary": claim_boundary_contract(),
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    validate_result(payload, plan=plan, verify_artifact=False)
    return payload


def validate_result(
    result: Mapping[str, Any], *, plan: Mapping[str, Any], verify_artifact: bool
) -> None:
    expected = {
        "actual_schedule_decision",
        "baseline_artifact",
        "claim_boundary",
        "dependencies",
        "environment",
        "kind",
        "memory",
        "model_identity",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "result_inputs",
        "runner_git_commit",
        "schema_version",
        "status",
        "summary_sha256",
        "timing",
    }
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256", None)
    timing = result.get("timing")
    memory = result.get("memory")
    schedule = result.get("actual_schedule_decision")
    model_identity = result.get("model_identity")
    if (
        set(result) != expected
        or result.get("schema_version") != 1
        or result.get("kind") != RESULT_KIND
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("status")
        not in {"pass_baseline_resource_feasibility", "stop_resource_infeasible"}
        or result.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or result.get("plan_sha256") != plan["plan_sha256"]
        or not _is_git_commit(result.get("runner_git_commit"))
        or result.get("dependencies") != plan["dependencies"]
        or result.get("environment") != plan["environment"]
        or result.get("result_inputs") != result_input_contract()
        or result.get("claim_boundary") != claim_boundary_contract()
        or not isinstance(timing, Mapping)
        or set(timing)
        != {
            "measured_case_median_seconds",
            "measured_case_p95_seconds",
            "measured_total_seconds",
            "model_load_seconds",
            "output_tokens_per_second",
            "warmup_total_seconds",
        }
        or not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0
            for value in timing.values()
        )
        or not isinstance(memory, Mapping)
        or set(memory)
        != {
            "active_after_bytes",
            "active_before_bytes",
            "cache_after_bytes",
            "cache_before_bytes",
            "maximum_allowed_bytes",
            "maximum_observed_working_set_bytes",
            "maximum_recommended_working_set_size",
            "peak_active_bytes",
            "process_peak_rss_bytes",
            "safety_pass",
            "working_set_fraction",
        }
        or not all(
            type(memory.get(name)) is int and int(memory[name]) >= 0
            for name in (
                "active_after_bytes",
                "active_before_bytes",
                "cache_after_bytes",
                "cache_before_bytes",
                "maximum_allowed_bytes",
                "maximum_observed_working_set_bytes",
                "maximum_recommended_working_set_size",
                "peak_active_bytes",
                "process_peak_rss_bytes",
            )
        )
        or isinstance(memory.get("working_set_fraction"), bool)
        or not isinstance(memory.get("working_set_fraction"), (int, float))
        or not math.isfinite(float(memory["working_set_fraction"]))
        or float(memory["working_set_fraction"]) <= 0
        or memory["active_before_bytes"] <= 0
        or memory["active_after_bytes"] <= 0
        or memory["peak_active_bytes"] <= 0
        or memory["process_peak_rss_bytes"] <= 0
        or memory["maximum_observed_working_set_bytes"]
        != max(
            memory["active_before_bytes"] + memory["cache_before_bytes"],
            memory["active_after_bytes"] + memory["cache_after_bytes"],
            memory["peak_active_bytes"],
            memory["process_peak_rss_bytes"],
        )
        or memory["maximum_allowed_bytes"]
        != math.floor(
            MAXIMUM_MEMORY_FRACTION * memory["maximum_recommended_working_set_size"]
        )
        or memory["maximum_recommended_working_set_size"] <= 0
        or not math.isclose(
            float(memory["working_set_fraction"]),
            memory["maximum_observed_working_set_bytes"]
            / memory["maximum_recommended_working_set_size"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or memory.get("safety_pass")
        is not bool(
            0
            < memory["maximum_observed_working_set_bytes"]
            <= memory["maximum_allowed_bytes"]
        )
        or not isinstance(model_identity, Mapping)
        or set(model_identity)
        != {
            "model_files",
            "model_parameter_count",
            "retrieval_table_loaded",
            "table_resident_bytes",
        }
        or not isinstance(model_identity["model_files"], Mapping)
        or type(model_identity["model_parameter_count"]) is not int
        or model_identity["model_parameter_count"] <= 0
        or model_identity["retrieval_table_loaded"] is not False
        or model_identity["table_resident_bytes"] != 0
        or not isinstance(schedule, Mapping)
        or schedule
        != select_actual_schedule(
            model_load_seconds=float(timing["model_load_seconds"]),
            warmup_total_seconds=float(timing["warmup_total_seconds"]),
            measured_total_seconds=float(timing["measured_total_seconds"]),
        )
        or result.get("status")
        != (
            "pass_baseline_resource_feasibility"
            if schedule["status"] == "feasible" and memory.get("safety_pass") is True
            else "stop_resource_infeasible"
        )
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE resource result identity differs")
    model_files = model_identity["model_files"]
    if set(model_files) != set(PRIMARY_MODEL["expected_files"]):
        raise ValueError("EXAONE resource model-file set differs")
    for name, row in model_files.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "sha256"}
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
            or not is_sha256(row["sha256"])
        ):
            raise ValueError(f"EXAONE resource model file differs: {name}")
    compatibility = read_validated_compatibility_result()
    if (
        model_files != compatibility["model_files"]
        or model_identity["model_parameter_count"]
        != compatibility["memory"]["model_parameters"]
    ):
        raise ValueError("EXAONE resource model identity differs")
    artifact = result.get("baseline_artifact")
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != {"arrays", "bytes", "path", "sha256"}
        or artifact.get("path") != BASELINE_ARTIFACT_PATH.relative_to(ROOT).as_posix()
        or type(artifact.get("bytes")) is not int
        or int(artifact["bytes"]) <= 0
        or not is_sha256(artifact.get("sha256"))
        or not isinstance(artifact.get("arrays"), Mapping)
        or set(artifact["arrays"]) != set(BASELINE_ARRAY_NAMES)
    ):
        raise ValueError("EXAONE resource artifact descriptor differs")
    expected_arrays = {
        "decoded_utf8_sha256": ("uint8", [TOTAL_CASES, 32]),
        "elapsed_ns": ("int64", [TOTAL_CASES]),
        "output_token_ids": ("uint32", [TOTAL_CASES, OUTPUT_TOKENS]),
        "output_token_sha256": ("uint8", [TOTAL_CASES, 32]),
        "prompt_token_count": ("uint16", [TOTAL_CASES]),
        "target_generation_forward_calls": ("uint16", [TOTAL_CASES]),
        "target_prefill_forward_calls": ("uint8", [TOTAL_CASES]),
    }
    for name, (dtype, shape) in expected_arrays.items():
        row = artifact["arrays"][name]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"dtype", "sha256", "shape"}
            or row.get("dtype") != dtype
            or row.get("shape") != shape
            or not is_sha256(row.get("sha256"))
        ):
            raise ValueError(f"EXAONE resource array descriptor differs: {name}")
    if verify_artifact:
        arrays = _load_artifact()
        validate_baseline_arrays(arrays)
        payload = BASELINE_ARTIFACT_PATH.read_bytes()
        if result["baseline_artifact"] != artifact_descriptor(payload, arrays):
            raise ValueError("EXAONE resource artifact differs")
        expected_timing = timing_summary(
            arrays, model_load_seconds=float(timing["model_load_seconds"])
        )
        if dict(timing) != expected_timing:
            raise ValueError("EXAONE resource timing reconstruction differs")


def read_result(*, verify_artifact: bool) -> dict[str, Any]:
    plan = read_plan(verify_derived=True)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    validate_result(result, plan=plan, verify_artifact=verify_artifact)
    return result
