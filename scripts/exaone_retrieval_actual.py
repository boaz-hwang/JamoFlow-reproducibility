"""Contracts and statistics for the EXAONE retrieval actual-inference study."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from exaone_resource_calibration import (
    PLAN_PATH as RESOURCE_PLAN_PATH,
)
from exaone_resource_calibration import (
    RESULT_PATH as RESOURCE_RESULT_PATH,
)
from exaone_resource_calibration import read_result as read_resource_result
from exaone_retrieval_data import (
    CASES_PATH,
    MAXIMUM_DRAFT_TOKENS,
    MEASURED_CASES,
    PRIMARY_MODEL,
    ROOT,
    TABLE_PATH,
    WARMUP_CASES,
    array_sha256,
    canonical_sha256,
    hash_file,
    is_sha256,
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
from exaone_retrieval_data import read_seal as read_data_seal
from large_model_retrieval_preflight import (
    environment_identity,
    token_sequence_sha256,
    validate_environment,
)

PLAN_PATH = ROOT / "data/manifests/exaone-retrieval-actual-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/exaone-retrieval-actual-v1"
SESSION_ARTIFACT_ROOT = ARTIFACT_ROOT / "sessions"
SESSION_RECEIPT_ROOT = ROOT / "results/exaone-retrieval-actual-v1/sessions"
SUMMARY_PATH = ROOT / "results/exaone-retrieval-actual-v1/summary.json"

PROTOCOL_ID = "jamoflow-exaone-retrieval-actual-v1"
PLAN_KIND = "exaone_retrieval_actual_plan_v1"
SESSION_KIND = "exaone_retrieval_actual_session_v1"
SUMMARY_KIND = "exaone_retrieval_actual_summary_v1"

ROLES = ("baseline_ar", "hybrid_retrieval_block_3")
BASELINE_ROLE_INDEX = 0
CANDIDATE_ROLE_INDEX = 1
SESSIONS = 5
INNER_REPETITIONS = 3
OUTPUT_TOKENS = 128
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_815
MINIMUM_POINT_REDUCTION = 0.10
MINIMUM_BOOTSTRAP_LOWER_REDUCTION = 0.0
MINIMUM_POSITIVE_PROMPTS = 48
MAXIMUM_MEMORY_FRACTION = 0.75
MEASURED_ORDER_STRIDE = 13
WARMUP_ORDER_STRIDE = 3

TIMING_NAMES = (
    "detokenization_ns",
    "elapsed_ns",
    "generation_ns",
    "tokenization_ns",
)
COUNTER_NAMES = (
    "accepted_draft_tokens",
    "bonus_tokens",
    "corpus_accepted_draft_tokens",
    "corpus_proposal_calls",
    "corpus_proposed_tokens",
    "correction_tokens",
    "final_cache_offset",
    "full_accept_cycles",
    "immediate_reject_cycles",
    "no_proposal_calls",
    "partial_accept_cycles",
    "prompt_accepted_draft_tokens",
    "prompt_proposal_calls",
    "prompt_proposed_tokens",
    "prompt_token_count",
    "proposal_attempts",
    "proposed_tokens",
    "target_generation_forward_calls",
    "target_prefill_forward_calls",
)
SESSION_ARRAY_NAMES = (
    "case_order",
    "decoded_utf8_sha256",
    *TIMING_NAMES,
    "first_role",
    "output_token_ids",
    "output_token_sha256",
    "peak_active_bytes",
    *COUNTER_NAMES,
)

IMPLEMENTATION_PATHS = (
    "docs/171-retrieval-novelty-closure-and-large-model-replication-direction.md",
    "docs/177-exaone-retrieval-data-and-case-protocol.md",
    "docs/182-exaone-resource-calibration-result-and-actual-decision.md",
    "docs/183-exaone-retrieval-actual-protocol.md",
    "docs/184-exaone-case-selection-provenance-correction.md",
    "pyproject.toml",
    "requirements/apple-retrieval-v1.txt",
    "scripts/exaone_actual_runtime.py",
    "scripts/exaone_retrieval_actual.py",
    "scripts/exaone_retrieval_actual_runtime.py",
    "scripts/exaone_retrieval_data.py",
    "scripts/exaone_resource_calibration.py",
    "scripts/large_model_retrieval_preflight.py",
    "scripts/mlx_retrieval_runtime.py",
    "scripts/run_exaone_retrieval_actual_session.py",
    "scripts/seal_exaone_retrieval_actual_plan.py",
    "scripts/summarize_exaone_retrieval_actual.py",
    "scripts/verify_exaone_retrieval_actual_summary.py",
    "tests/test_exaone_retrieval_actual.py",
)

PUBLICATION_MPS_LOCK_PATH = Path("/tmp/jamoflow-publication-mps.lock")


@contextmanager
def actual_mps_exclusive():
    """Share the machine-global publication lock with other neural runners."""

    descriptor = os.open(
        PUBLICATION_MPS_LOCK_PATH,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another publication MPS process holds the lock"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def assert_canonical_workspace_path(path: Path) -> None:
    workspace = ROOT.resolve()
    target = path.absolute()
    try:
        relative = target.relative_to(ROOT.absolute())
    except ValueError as error:
        raise ValueError("EXAONE actual namespace is outside the repository") from error
    cursor = ROOT.absolute()
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"EXAONE actual namespace contains a symlink: {cursor}")
    resolved = target.resolve()
    if workspace not in (resolved, *resolved.parents):
        raise ValueError("EXAONE actual namespace resolves outside the repository")


def session_artifact_path(session_index: int) -> Path:
    return SESSION_ARTIFACT_ROOT / f"session-{session_index}.npz"


def session_active_path(session_index: int) -> Path:
    return SESSION_ARTIFACT_ROOT / f"session-{session_index}.active"


def session_receipt_path(session_index: int) -> Path:
    return SESSION_RECEIPT_ROOT / f"session-{session_index}.json"


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def require_distinct_git_commits(
    ancestor: object, descendant: object, *, label: str
) -> None:
    if (
        not _is_git_commit(ancestor)
        or not _is_git_commit(descendant)
        or ancestor == descendant
    ):
        raise ValueError(f"EXAONE actual strict Git chronology differs: {label}")


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("EXAONE actual implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def dependency_identity() -> dict[str, dict[str, Any]]:
    read_verification()
    read_resource_result(verify_artifact=True)
    paths = {
        "case_artifact": CASES_PATH,
        "data_plan": DATA_PLAN_PATH,
        "data_seal": DATA_SEAL_PATH,
        "data_verification": DATA_VERIFICATION_PATH,
        "resource_plan": RESOURCE_PLAN_PATH,
        "resource_result": RESOURCE_RESULT_PATH,
        "table_artifact": TABLE_PATH,
    }
    return {
        name: {
            "bytes": path.stat().st_size,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
        for name, path in paths.items()
    }


def measured_case_order(session_index: int) -> tuple[int, ...]:
    if not 0 <= session_index < SESSIONS:
        raise ValueError("EXAONE actual session index differs")
    offset = session_index * MEASURED_ORDER_STRIDE
    return tuple((offset + index) % MEASURED_CASES for index in range(MEASURED_CASES))


def warmup_case_order(session_index: int) -> tuple[int, ...]:
    if not 0 <= session_index < SESSIONS:
        raise ValueError("EXAONE actual session index differs")
    offset = session_index * WARMUP_ORDER_STRIDE
    return tuple((offset + index) % WARMUP_CASES for index in range(WARMUP_CASES))


def balanced_role_order(
    session_index: int, case_index: int, repetition_index: int
) -> tuple[int, int]:
    if (
        not 0 <= session_index < SESSIONS
        or not 0 <= case_index < MEASURED_CASES
        or not 0 <= repetition_index < INNER_REPETITIONS
    ):
        raise ValueError("EXAONE actual schedule coordinate differs")
    return (
        (BASELINE_ROLE_INDEX, CANDIDATE_ROLE_INDEX)
        if (session_index + case_index // 2 + repetition_index) % 2 == 0
        else (CANDIDATE_ROLE_INDEX, BASELINE_ROLE_INDEX)
    )


def actual_contract() -> dict[str, Any]:
    return {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_scheme": {
            "paired_roles_share_indices": True,
            "quantile_method": "linear",
            "resample_axes": ["fresh_process_session", "prompt_document"],
            "repetitions_are_collapsed_before_resampling": True,
        },
        "case_order": {
            "measured_cyclic_stride": MEASURED_ORDER_STRIDE,
            "warmup_cyclic_stride": WARMUP_ORDER_STRIDE,
        },
        "candidate": {
            "maximum_draft_tokens": MAXIMUM_DRAFT_TOKENS,
            "proposal_precedence": "corpus_ngram_then_prompt_lookup_then_none",
            "table_loaded_outside_timer": True,
        },
        "inner_repetitions": INNER_REPETITIONS,
        "measured_cases": MEASURED_CASES,
        "memory_safety": {
            "failure_blocks_primary": True,
            "maximum_recommended_working_set_fraction": MAXIMUM_MEMORY_FRACTION,
        },
        "operational_eligibility_checkpoints": {
            "after_warmup": True,
            "end": True,
            "every_measured_case_positions": 16,
            "known_jamoflow_neural_process_inventory": True,
            "start": True,
        },
        "output_tokens": OUTPUT_TOKENS,
        "primary_estimand": "free_running_fixed_token_end_to_end_latency",
        "role_order": (
            "baseline_first_when_session_plus_floor_canonical_case_over_2_plus_"
            "repetition_is_even"
        ),
        "roles": list(ROLES),
        "sessions": SESSIONS,
        "stop_semantics": {
            "eos_stops_generation": False,
            "fixed_output_tokens": OUTPUT_TOKENS,
        },
        "timer": {
            "final_mlx_synchronize": True,
            "fresh_kv_cache": True,
            "full_detokenization": True,
            "model_and_table_load": False,
            "prompt_tokenization": True,
            "proposal_lookup_and_verification": True,
        },
        "warmup_cases": WARMUP_CASES,
    }


def gate_contract() -> dict[str, Any]:
    return {
        "all_sessions_positive": True,
        "bootstrap_lower_reduction_strictly_greater_than": (
            MINIMUM_BOOTSTRAP_LOWER_REDUCTION
        ),
        "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "output_token_and_decoded_hash_exact": True,
    }


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "actual_retrieval_acceptance_observed_before_plan": False,
        "actual_retrieval_latency_observed_before_plan": False,
        "baseline_resource_latency_observed_before_plan": True,
        "case_rank_seed_includes_compatibility_model_output_hash": True,
        "case_selection_model_output_blind": False,
        "compatibility_model_outputs_observed_before_plan": True,
        "confirmatory_or_final_blind": False,
        "evaluation_pool_previously_used": True,
        "evaluation_case_candidate_output_observed_before_plan": False,
        "final_quality_claim": False,
        "generic_retrieval_is_novel": False,
        "korean_specific_method_tested": False,
        "memory_improvement_primary": False,
        "prospectively_sealed_actual_timing": True,
        "raw_completion_not_chat_template": True,
        "korean_centric_scale_transfer_evidence": True,
        "single_apple_hardware_only": True,
    }


def _validated_resource_schedule(resource_result: object) -> Mapping[str, Any]:
    if not isinstance(resource_result, Mapping):
        raise TypeError("EXAONE actual resource result differs")
    memory = resource_result.get("memory")
    decision = resource_result.get("actual_schedule_decision")
    selected = decision.get("selected") if isinstance(decision, Mapping) else None
    if (
        resource_result.get("status") != "pass_baseline_resource_feasibility"
        or not isinstance(memory, Mapping)
        or memory.get("safety_pass") is not True
        or not isinstance(decision, Mapping)
        or decision.get("status") != "feasible"
        or not isinstance(selected, Mapping)
        or set(selected)
        != {"inner_repetitions", "projected_campaign_hours", "sessions"}
        or selected.get("inner_repetitions") != INNER_REPETITIONS
        or selected.get("sessions") != SESSIONS
        or not isinstance(selected.get("projected_campaign_hours"), float)
        or not math.isfinite(selected["projected_campaign_hours"])
        or selected["projected_campaign_hours"] <= 0
    ):
        raise ValueError("EXAONE actual resource schedule differs")
    return selected


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if not _is_git_commit(git_commit_before_plan):
        raise ValueError("EXAONE actual pre-plan commit differs")
    resource_result = read_resource_result(verify_artifact=True)
    data_seal = read_data_seal(verify_artifacts=True)
    selected = _validated_resource_schedule(resource_result)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_first_retrieval_table_candidate_execution",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": resource_result["environment"],
        "model_identity": {
            "model_files": resource_result["model_identity"]["model_files"],
            "model_parameter_count": resource_result["model_identity"][
                "model_parameter_count"
            ],
        },
        "table_identity": data_seal["table_artifact"],
        "case_identity": data_seal["case_artifact"],
        "resource_schedule": selected,
        "actual_contract": actual_contract(),
        "gate_contract": gate_contract(),
        "claim_boundary": claim_boundary_contract(),
        "outputs": {
            "session_artifact_root": SESSION_ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "session_receipt_root": SESSION_RECEIPT_ROOT.relative_to(ROOT).as_posix(),
            "summary_path": SUMMARY_PATH.relative_to(ROOT).as_posix(),
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    validate_plan(payload, verify_derived=True)
    return payload


def _validate_file_identity(value: object, expected: Mapping[str, Path]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("EXAONE actual dependency set differs")
    for name, row in value.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "path", "sha256"}
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or row.get("path") != expected[name].relative_to(ROOT).as_posix()
            or not is_sha256(row.get("sha256"))
        ):
            raise ValueError(f"EXAONE actual dependency differs: {name}")


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected = {
        "actual_contract",
        "case_identity",
        "claim_boundary",
        "dependencies",
        "environment",
        "gate_contract",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "model_identity",
        "outputs",
        "plan_sha256",
        "protocol_id",
        "resource_schedule",
        "schema_version",
        "status",
        "table_identity",
    }
    dependency_paths = {
        "case_artifact": CASES_PATH,
        "data_plan": DATA_PLAN_PATH,
        "data_seal": DATA_SEAL_PATH,
        "data_verification": DATA_VERIFICATION_PATH,
        "resource_plan": RESOURCE_PLAN_PATH,
        "resource_result": RESOURCE_RESULT_PATH,
        "table_artifact": TABLE_PATH,
    }
    _validate_file_identity(plan.get("dependencies"), dependency_paths)
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    implementation = plan.get("implementation_sha256")
    resource = json.loads(RESOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    data_seal = json.loads(DATA_SEAL_PATH.read_text(encoding="utf-8"))
    selected = _validated_resource_schedule(resource)
    resource_environment = resource.get("environment")
    if not isinstance(resource_environment, Mapping):
        raise TypeError("EXAONE actual resource environment differs")
    validate_environment(resource_environment)
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("kind") != PLAN_KIND
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status")
        != "sealed_before_first_retrieval_table_candidate_execution"
        or not _is_git_commit(plan.get("git_commit_before_plan"))
        or not isinstance(implementation, Mapping)
        or len(implementation) != len(IMPLEMENTATION_PATHS)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or not all(is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
        or plan.get("actual_contract") != actual_contract()
        or plan.get("gate_contract") != gate_contract()
        or plan.get("claim_boundary") != claim_boundary_contract()
        or plan.get("resource_schedule") != selected
        or plan.get("model_identity")
        != {
            "model_files": resource["model_identity"]["model_files"],
            "model_parameter_count": resource["model_identity"][
                "model_parameter_count"
            ],
        }
        or plan.get("environment") != resource["environment"]
        or plan.get("table_identity") != data_seal["table_artifact"]
        or plan.get("case_identity") != data_seal["case_artifact"]
        or plan.get("outputs")
        != {
            "session_artifact_root": SESSION_ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "session_receipt_root": SESSION_RECEIPT_ROOT.relative_to(ROOT).as_posix(),
            "summary_path": SUMMARY_PATH.relative_to(ROOT).as_posix(),
        }
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE actual plan identity differs")
    if verify_derived:
        if plan["dependencies"] != dependency_identity():
            raise ValueError("EXAONE actual dependencies changed")
        if plan["implementation_sha256"] != implementation_identity():
            raise ValueError("EXAONE actual implementation changed")
        if plan["environment"] != environment_identity():
            raise ValueError("EXAONE actual execution environment changed")


def read_plan(*, verify_derived: bool) -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    validate_plan(plan, verify_derived=verify_derived)
    return plan


def artifact_descriptor(
    path: Path, payload: bytes, arrays: Mapping[str, np.ndarray]
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
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def expected_table_resident_bytes(plan: Mapping[str, Any]) -> int:
    table = plan.get("table_identity")
    arrays = table.get("arrays") if isinstance(table, Mapping) else None
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("EXAONE actual table-array identity differs")
    total = 0
    for row in arrays.values():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"dtype", "sha256", "shape"}
            or not isinstance(row.get("dtype"), str)
            or not isinstance(row.get("shape"), list)
            or not row["shape"]
            or any(type(value) is not int or value <= 0 for value in row["shape"])
            or not is_sha256(row.get("sha256"))
        ):
            raise ValueError("EXAONE actual table-array descriptor differs")
        try:
            dtype = np.dtype(row["dtype"])
        except (TypeError, ValueError) as error:
            raise ValueError("EXAONE actual table dtype differs") from error
        total += int(dtype.itemsize) * math.prod(row["shape"])
    if total <= 0:
        raise ValueError("EXAONE actual table resident bytes differ")
    return total


def load_session_arrays(session_index: int) -> dict[str, np.ndarray]:
    path = session_artifact_path(session_index)
    if path.is_symlink() or not path.is_file():
        raise ValueError("EXAONE actual session artifact is missing or a symlink")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(SESSION_ARRAY_NAMES):
            raise ValueError("EXAONE actual session artifact keys differ")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    validate_session_arrays(arrays, session_index=session_index)
    return arrays


def _validate_operational_environment(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "ac_power",
            "battery_sha256",
            "conflicting_process_count",
            "process_inventory_pass",
            "process_inventory_sha256",
            "thermal_pass",
            "thermal_sha256",
        }
        or value.get("ac_power") is not True
        or value.get("thermal_pass") is not True
        or value.get("process_inventory_pass") is not True
        or value.get("conflicting_process_count") != 0
        or not is_sha256(value.get("battery_sha256"))
        or not is_sha256(value.get("process_inventory_sha256"))
        or not is_sha256(value.get("thermal_sha256"))
    ):
        raise ValueError("EXAONE actual operational environment differs")


def _validate_memory(value: object, *, plan: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
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
    }:
        raise ValueError("EXAONE actual session memory schema differs")
    integer_names = set(value) - {"safety_pass", "working_set_fraction"}
    if any(
        type(value.get(name)) is not int or value[name] < 0 for name in integer_names
    ):
        raise ValueError("EXAONE actual session memory value differs")
    working_set = plan["environment"]["mlx"]["max_recommended_working_set_size"]
    observed = max(
        value["active_before_bytes"] + value["cache_before_bytes"],
        value["active_after_bytes"] + value["cache_after_bytes"],
        value["peak_active_bytes"],
        value["process_peak_rss_bytes"],
    )
    allowed = math.floor(MAXIMUM_MEMORY_FRACTION * working_set)
    if (
        value["active_before_bytes"] <= 0
        or value["active_after_bytes"] <= 0
        or value["peak_active_bytes"] <= 0
        or value["process_peak_rss_bytes"] <= 0
        or value["maximum_recommended_working_set_size"] != working_set
        or value["maximum_observed_working_set_bytes"] != observed
        or value["maximum_allowed_bytes"] != allowed
        or value["safety_pass"] is not bool(0 < observed <= allowed)
        or value["safety_pass"] is not True
        or not isinstance(value["working_set_fraction"], float)
        or not math.isclose(
            value["working_set_fraction"],
            observed / working_set,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("EXAONE actual session memory identity differs")


def build_session_receipt(
    *,
    plan: Mapping[str, Any],
    session_index: int,
    runner_git_commit: str,
    process_start_token_sha256: str,
    arrays: Mapping[str, np.ndarray],
    artifact_bytes: bytes,
    model_identity: Mapping[str, Any],
    operational_start: Mapping[str, Any],
    operational_end: Mapping[str, Any],
    operational_checkpoints: Sequence[Mapping[str, Any]],
    memory: Mapping[str, Any],
    warmup_output_root_sha256: str,
) -> dict[str, Any]:
    validate_session_arrays(arrays, session_index=session_index)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": SESSION_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "complete_correctness_pass_no_performance_summary",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "session_index": session_index,
        "runner_git_commit": runner_git_commit,
        "process_start_token_sha256": process_start_token_sha256,
        "environment": plan["environment"],
        "operational_start": dict(operational_start),
        "operational_end": dict(operational_end),
        "operational_checkpoint_count": len(operational_checkpoints),
        "operational_checkpoint_root_sha256": canonical_sha256(
            {"checkpoints": list(operational_checkpoints)}
        ),
        "operational_checkpoints": [dict(value) for value in operational_checkpoints],
        "model_identity": dict(model_identity),
        "table_identity": plan["table_identity"],
        "artifact": artifact_descriptor(
            session_artifact_path(session_index), artifact_bytes, arrays
        ),
        "correctness": {
            "measured_pair_comparisons": MEASURED_CASES * INNER_REPETITIONS,
            "measured_token_id_exact": True,
            "measured_decoded_hash_exact": True,
            "session_array_contract_pass": True,
            "warmup_pair_comparisons": WARMUP_CASES,
            "warmup_token_id_and_decoded_hash_exact": True,
        },
        "warmup_output_root_sha256": warmup_output_root_sha256,
        "memory": dict(memory),
        "performance_summary_in_receipt": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    validate_session_receipt(
        payload, plan=plan, session_index=session_index, verify_artifact=False
    )
    return payload


def validate_session_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    session_index: int,
    verify_artifact: bool,
) -> None:
    expected = {
        "artifact",
        "correctness",
        "environment",
        "kind",
        "memory",
        "model_identity",
        "operational_end",
        "operational_checkpoint_count",
        "operational_checkpoint_root_sha256",
        "operational_checkpoints",
        "operational_start",
        "performance_summary_in_receipt",
        "plan_artifact_sha256",
        "plan_sha256",
        "process_start_token_sha256",
        "protocol_id",
        "receipt_sha256",
        "runner_git_commit",
        "schema_version",
        "session_index",
        "status",
        "table_identity",
        "warmup_output_root_sha256",
    }
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256", None)
    artifact = receipt.get("artifact")
    correctness = receipt.get("correctness")
    model = receipt.get("model_identity")
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != 1
        or receipt.get("kind") != SESSION_KIND
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "complete_correctness_pass_no_performance_summary"
        or receipt.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("session_index") != session_index
        or not _is_git_commit(receipt.get("runner_git_commit"))
        or not is_sha256(receipt.get("process_start_token_sha256"))
        or receipt.get("environment") != plan["environment"]
        or receipt.get("table_identity") != plan["table_identity"]
        or receipt.get("performance_summary_in_receipt") is not False
        or not isinstance(receipt.get("operational_checkpoints"), list)
        or len(receipt["operational_checkpoints"]) != 7
        or receipt.get("operational_checkpoint_count")
        != len(receipt["operational_checkpoints"])
        or not is_sha256(receipt.get("operational_checkpoint_root_sha256"))
        or receipt.get("operational_checkpoint_root_sha256")
        != canonical_sha256({"checkpoints": receipt["operational_checkpoints"]})
        or receipt["operational_checkpoints"][0] != receipt.get("operational_start")
        or receipt["operational_checkpoints"][-1] != receipt.get("operational_end")
        or not is_sha256(receipt.get("warmup_output_root_sha256"))
        or correctness
        != {
            "measured_pair_comparisons": MEASURED_CASES * INNER_REPETITIONS,
            "measured_token_id_exact": True,
            "measured_decoded_hash_exact": True,
            "session_array_contract_pass": True,
            "warmup_pair_comparisons": WARMUP_CASES,
            "warmup_token_id_and_decoded_hash_exact": True,
        }
        or not isinstance(model, Mapping)
        or model
        != {
            **plan["model_identity"],
            "retrieval_table_loaded": True,
            "table_resident_bytes": expected_table_resident_bytes(plan),
        }
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE actual session receipt differs")
    _validate_operational_environment(receipt.get("operational_start"))
    _validate_operational_environment(receipt.get("operational_end"))
    for checkpoint in receipt["operational_checkpoints"]:
        _validate_operational_environment(checkpoint)
    _validate_memory(receipt.get("memory"), plan=plan)
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != {"arrays", "bytes", "path", "sha256"}
        or artifact.get("path")
        != session_artifact_path(session_index).relative_to(ROOT).as_posix()
        or type(artifact.get("bytes")) is not int
        or artifact["bytes"] <= 0
        or not is_sha256(artifact.get("sha256"))
        or not isinstance(artifact.get("arrays"), Mapping)
        or set(artifact["arrays"]) != set(SESSION_ARRAY_NAMES)
    ):
        raise ValueError("EXAONE actual session artifact descriptor differs")
    for name, row in artifact["arrays"].items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"dtype", "sha256", "shape"}
            or not isinstance(row.get("dtype"), str)
            or not isinstance(row.get("shape"), list)
            or not is_sha256(row.get("sha256"))
        ):
            raise ValueError(f"EXAONE actual session array descriptor differs: {name}")
    if verify_artifact:
        arrays = load_session_arrays(session_index)
        payload = session_artifact_path(session_index).read_bytes()
        if artifact != artifact_descriptor(
            session_artifact_path(session_index), payload, arrays
        ):
            raise ValueError("EXAONE actual session artifact differs")


def read_session_receipt(
    session_index: int, *, plan: Mapping[str, Any], verify_artifact: bool
) -> dict[str, Any]:
    receipt = json.loads(
        session_receipt_path(session_index).read_text(encoding="utf-8")
    )
    validate_session_receipt(
        receipt,
        plan=plan,
        session_index=session_index,
        verify_artifact=verify_artifact,
    )
    return receipt


def _token_hash_rows(output: np.ndarray) -> np.ndarray:
    flat = output.reshape(-1, OUTPUT_TOKENS)
    hashes = np.asarray(
        [
            list(
                bytes.fromhex(token_sequence_sha256(tuple(int(token) for token in row)))
            )
            for row in flat
        ],
        dtype=np.uint8,
    )
    return hashes.reshape(output.shape[:-1] + (32,))


def validate_session_arrays(
    arrays: Mapping[str, np.ndarray], *, session_index: int
) -> None:
    cell_shape = (MEASURED_CASES, INNER_REPETITIONS, len(ROLES))
    expected: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
        "case_order": (np.dtype("uint16"), (MEASURED_CASES,)),
        "decoded_utf8_sha256": (np.dtype("uint8"), cell_shape + (32,)),
        "first_role": (
            np.dtype("uint8"),
            (MEASURED_CASES, INNER_REPETITIONS),
        ),
        "output_token_ids": (
            np.dtype("uint32"),
            cell_shape + (OUTPUT_TOKENS,),
        ),
        "output_token_sha256": (np.dtype("uint8"), cell_shape + (32,)),
        "peak_active_bytes": (np.dtype("uint64"), cell_shape),
    }
    expected.update({name: (np.dtype("int64"), cell_shape) for name in TIMING_NAMES})
    expected.update({name: (np.dtype("uint16"), cell_shape) for name in COUNTER_NAMES})
    if set(arrays) != set(SESSION_ARRAY_NAMES) or set(arrays) != set(expected):
        raise ValueError("EXAONE actual session array set differs")
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"EXAONE actual session array differs: {name}")
    if tuple(int(value) for value in arrays["case_order"]) != measured_case_order(
        session_index
    ):
        raise ValueError("EXAONE actual measured case order differs")
    for case_index in measured_case_order(session_index):
        for repetition in range(INNER_REPETITIONS):
            expected_first = balanced_role_order(session_index, case_index, repetition)[
                0
            ]
            if int(arrays["first_role"][case_index, repetition]) != expected_first:
                raise ValueError("EXAONE actual first-role schedule differs")
    if (
        any(np.any(arrays[name] <= 0) for name in TIMING_NAMES)
        or np.any(arrays["peak_active_bytes"] <= 0)
        or not np.array_equal(
            arrays["elapsed_ns"],
            arrays["tokenization_ns"]
            + arrays["generation_ns"]
            + arrays["detokenization_ns"],
        )
        or np.any(arrays["prompt_token_count"] != 128)
        or np.any(arrays["target_prefill_forward_calls"] != 1)
        or np.any(
            arrays["output_token_ids"]
            >= PRIMARY_MODEL["config_projection"]["vocab_size"]
        )
        or not np.array_equal(
            arrays["output_token_sha256"],
            _token_hash_rows(arrays["output_token_ids"]),
        )
    ):
        raise ValueError("EXAONE actual session timing or token identity differs")
    anchor_tokens = arrays["output_token_ids"][
        :, :1, BASELINE_ROLE_INDEX : BASELINE_ROLE_INDEX + 1
    ]
    anchor_hashes = arrays["decoded_utf8_sha256"][
        :, :1, BASELINE_ROLE_INDEX : BASELINE_ROLE_INDEX + 1
    ]
    if not np.array_equal(
        arrays["output_token_ids"],
        np.broadcast_to(anchor_tokens, arrays["output_token_ids"].shape),
    ) or not np.array_equal(
        arrays["decoded_utf8_sha256"],
        np.broadcast_to(anchor_hashes, arrays["decoded_utf8_sha256"].shape),
    ):
        raise ValueError("EXAONE actual output differs across roles or repetitions")

    baseline = (..., BASELINE_ROLE_INDEX)
    candidate = (..., CANDIDATE_ROLE_INDEX)
    if (
        np.any(arrays["target_generation_forward_calls"][baseline] != OUTPUT_TOKENS)
        or np.any(arrays["no_proposal_calls"][baseline] != OUTPUT_TOKENS)
        or np.any(arrays["final_cache_offset"][baseline] != 255)
        or any(
            np.any(arrays[name][baseline] != 0)
            for name in (
                "accepted_draft_tokens",
                "bonus_tokens",
                "corpus_accepted_draft_tokens",
                "corpus_proposal_calls",
                "corpus_proposed_tokens",
                "correction_tokens",
                "full_accept_cycles",
                "immediate_reject_cycles",
                "partial_accept_cycles",
                "prompt_accepted_draft_tokens",
                "prompt_proposal_calls",
                "prompt_proposed_tokens",
                "proposal_attempts",
                "proposed_tokens",
            )
        )
    ):
        raise ValueError("EXAONE actual baseline counter identity differs")
    target = arrays["target_generation_forward_calls"][candidate]
    proposals = arrays["proposal_attempts"][candidate]
    no_proposal = arrays["no_proposal_calls"][candidate]
    proposed = arrays["proposed_tokens"][candidate]
    accepted = arrays["accepted_draft_tokens"][candidate]
    corpus_proposed = arrays["corpus_proposed_tokens"][candidate]
    prompt_proposed = arrays["prompt_proposed_tokens"][candidate]
    corpus_accepted = arrays["corpus_accepted_draft_tokens"][candidate]
    prompt_accepted = arrays["prompt_accepted_draft_tokens"][candidate]
    full = arrays["full_accept_cycles"][candidate]
    immediate = arrays["immediate_reject_cycles"][candidate]
    partial = arrays["partial_accept_cycles"][candidate]
    correction = arrays["correction_tokens"][candidate]
    bonus = arrays["bonus_tokens"][candidate]
    if (
        np.any(target == 0)
        or np.any(target > OUTPUT_TOKENS)
        or not np.array_equal(
            proposals,
            arrays["corpus_proposal_calls"][candidate]
            + arrays["prompt_proposal_calls"][candidate],
        )
        or not np.array_equal(target, proposals + no_proposal)
        or not np.array_equal(proposals, full + immediate + partial)
        or not np.array_equal(correction, immediate + partial)
        or not np.array_equal(proposed, corpus_proposed + prompt_proposed)
        or not np.array_equal(accepted, corpus_accepted + prompt_accepted)
        or np.any(accepted > proposed)
        or np.any(corpus_accepted > corpus_proposed)
        or np.any(prompt_accepted > prompt_proposed)
        or np.any(bonus > full)
        or np.any(proposed < proposals)
        or np.any(corpus_proposed < arrays["corpus_proposal_calls"][candidate])
        or np.any(prompt_proposed < arrays["prompt_proposal_calls"][candidate])
        or np.any(proposed > MAXIMUM_DRAFT_TOKENS * proposals)
        or np.any(
            corpus_proposed
            > MAXIMUM_DRAFT_TOKENS * arrays["corpus_proposal_calls"][candidate]
        )
        or np.any(
            prompt_proposed
            > MAXIMUM_DRAFT_TOKENS * arrays["prompt_proposal_calls"][candidate]
        )
        or not np.array_equal(
            np.full_like(target, OUTPUT_TOKENS),
            no_proposal + accepted + correction + bonus,
        )
        or np.any(arrays["final_cache_offset"][candidate] != 255)
    ):
        raise ValueError("EXAONE actual candidate counter identity differs")


def _crossed_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray
) -> tuple[float, float]:
    if (
        candidate.shape != (SESSIONS, MEASURED_CASES)
        or baseline.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
        or np.any(candidate <= 0)
        or np.any(baseline <= 0)
    ):
        raise ValueError("EXAONE actual bootstrap cells differ")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sessions = rng.integers(0, SESSIONS, size=SESSIONS)
        prompts = rng.integers(0, MEASURED_CASES, size=MEASURED_CASES)
        candidate_sample = candidate[np.ix_(sessions, prompts)]
        baseline_sample = baseline[np.ix_(sessions, prompts)]
        estimates[index] = 1.0 - float(np.median(candidate_sample)) / float(
            np.median(baseline_sample)
        )
    return float(np.quantile(estimates, 0.025, method="linear")), float(
        np.quantile(estimates, 0.975, method="linear")
    )


def summarize_actual_arrays(
    sessions: list[Mapping[str, np.ndarray]], *, correctness_pass: bool
) -> dict[str, Any]:
    if len(sessions) != SESSIONS or type(correctness_pass) is not bool:
        raise ValueError("EXAONE actual session count differs")
    for index, arrays in enumerate(sessions):
        validate_session_arrays(arrays, session_index=index)
    timing = (
        np.stack([arrays["elapsed_ns"] for arrays in sessions]).astype(np.float64) / 1e9
    )
    cells = np.median(timing, axis=2)
    baseline = cells[..., BASELINE_ROLE_INDEX]
    candidate = cells[..., CANDIDATE_ROLE_INDEX]
    point = 1.0 - float(np.median(candidate)) / float(np.median(baseline))
    lower, upper = _crossed_bootstrap(candidate, baseline)
    session_reductions = [
        1.0 - float(np.median(candidate[index])) / float(np.median(baseline[index]))
        for index in range(SESSIONS)
    ]
    prompt_candidate = np.median(candidate, axis=0)
    prompt_baseline = np.median(baseline, axis=0)
    positive_prompts = int(np.count_nonzero(prompt_candidate < prompt_baseline))
    first_role = np.stack([arrays["first_role"] for arrays in sessions])
    order_diagnostic = {}
    for label, role_index in (
        ("baseline_first", BASELINE_ROLE_INDEX),
        ("candidate_first", CANDIDATE_ROLE_INDEX),
    ):
        mask = first_role == role_index
        order_baseline = timing[..., BASELINE_ROLE_INDEX][mask]
        order_candidate = timing[..., CANDIDATE_ROLE_INDEX][mask]
        if len(order_baseline) != SESSIONS * MEASURED_CASES * INNER_REPETITIONS // 2:
            raise ValueError("EXAONE actual role-order balance differs")
        order_diagnostic[label] = {
            "paired_trial_count": len(order_baseline),
            "baseline_median_seconds": float(np.median(order_baseline)),
            "candidate_median_seconds": float(np.median(order_candidate)),
            "median_reduction": 1.0
            - float(np.median(order_candidate)) / float(np.median(order_baseline)),
        }

    components: dict[str, Any] = {}
    for name in TIMING_NAMES:
        values = np.stack([arrays[name] for arrays in sessions]).astype(np.float64)
        component_cells = np.median(values, axis=2) / 1e6
        baseline_median = float(np.median(component_cells[..., BASELINE_ROLE_INDEX]))
        candidate_median = float(np.median(component_cells[..., CANDIDATE_ROLE_INDEX]))
        components[name.removesuffix("_ns") + "_ms"] = {
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "median_reduction": 1.0 - candidate_median / baseline_median,
        }

    counters = {
        name: np.stack([arrays[name] for arrays in sessions]) for name in COUNTER_NAMES
    }
    proposed = int(np.sum(counters["proposed_tokens"][..., CANDIDATE_ROLE_INDEX]))
    accepted = int(np.sum(counters["accepted_draft_tokens"][..., CANDIDATE_ROLE_INDEX]))
    attempts = int(np.sum(counters["proposal_attempts"][..., CANDIDATE_ROLE_INDEX]))
    candidate_calls = counters["target_generation_forward_calls"][
        ..., CANDIDATE_ROLE_INDEX
    ]
    baseline_calls = counters["target_generation_forward_calls"][
        ..., BASELINE_ROLE_INDEX
    ]
    gate = {
        "correctness": bool(correctness_pass),
        "point_reduction_at_least_10_percent": point >= MINIMUM_POINT_REDUCTION,
        "bootstrap_lower_strictly_positive": (
            lower > MINIMUM_BOOTSTRAP_LOWER_REDUCTION
        ),
        "positive_prompts_at_least_48": positive_prompts >= MINIMUM_POSITIVE_PROMPTS,
        "all_five_sessions_positive": all(value > 0 for value in session_reductions),
    }
    gate["overall_pass"] = all(gate.values())
    source_mechanism = {}
    for source in ("corpus", "prompt"):
        calls = int(
            np.sum(counters[f"{source}_proposal_calls"][..., CANDIDATE_ROLE_INDEX])
        )
        source_proposed = int(
            np.sum(counters[f"{source}_proposed_tokens"][..., CANDIDATE_ROLE_INDEX])
        )
        source_accepted = int(
            np.sum(
                counters[f"{source}_accepted_draft_tokens"][..., CANDIDATE_ROLE_INDEX]
            )
        )
        source_mechanism[source] = {
            "accepted_draft_tokens": source_accepted,
            "draft_token_acceptance_rate": (
                source_accepted / source_proposed if source_proposed else 0.0
            ),
            "proposal_calls": calls,
            "proposed_tokens": source_proposed,
        }

    return {
        "primary_end_to_end": {
            "baseline_cell_median_seconds": float(np.median(baseline)),
            "candidate_cell_median_seconds": float(np.median(candidate)),
            "median_reduction": point,
            "crossed_session_prompt_bootstrap_95_interval": {
                "lower": lower,
                "upper": upper,
            },
            "positive_prompt_count": positive_prompts,
            "session_reductions": session_reductions,
        },
        "component_timing": components,
        "role_order_diagnostic": order_diagnostic,
        "mechanism": {
            "accepted_draft_tokens": accepted,
            "accepted_tokens_per_proposal_cycle": (
                accepted / attempts if attempts else 0.0
            ),
            "baseline_target_forward_calls_median": float(np.median(baseline_calls)),
            "candidate_target_forward_calls_median": float(np.median(candidate_calls)),
            "corpus_proposal_calls": int(
                np.sum(counters["corpus_proposal_calls"][..., CANDIDATE_ROLE_INDEX])
            ),
            "draft_token_acceptance_rate": accepted / proposed if proposed else 0.0,
            "no_proposal_calls": int(
                np.sum(counters["no_proposal_calls"][..., CANDIDATE_ROLE_INDEX])
            ),
            "prompt_proposal_calls": int(
                np.sum(counters["prompt_proposal_calls"][..., CANDIDATE_ROLE_INDEX])
            ),
            "proposal_attempts": attempts,
            "proposed_tokens": proposed,
            "source_breakdown": source_mechanism,
            "target_forward_call_reduction": 1.0
            - float(np.median(candidate_calls)) / float(np.median(baseline_calls)),
        },
        "primary_gate": gate,
        "status": (
            "pass_generic_retrieval_scale_transfer_actual"
            if gate["overall_pass"]
            else "fail_generic_retrieval_scale_transfer_actual"
        ),
        "generic_retrieval_is_novel_claimed": False,
        "korean_specific_followup_authorized": bool(gate["overall_pass"]),
    }


def _summary_interpretation(overall_pass: bool) -> dict[str, Any]:
    return {
        "case_selection_model_output_blind": False,
        "evidence_scope": "exploratory_candidate_actual_timing_scale_transfer",
        "generic_retrieval_novelty_claimed": False,
        "historically_used_evaluation_pool": True,
        "primary_claim": (
            "Korean-centric 7.8B raw-completion scale-transfer evidence "
            "for exact retrieval speculative decoding"
            if overall_pass
            else "no qualifying Korean-centric 7.8B scale-transfer evidence"
        ),
        "raw_completion_only": True,
        "single_apple_m4_pro_environment": True,
    }


def validate_actual_summary(
    summary: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_lineage: Sequence[Mapping[str, Any]] | None = None,
    expected_replay: Mapping[str, Any] | None = None,
    expected_statistics: Mapping[str, Any] | None = None,
    expected_memory: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "claim_boundary",
        "independent_replay",
        "interpretation",
        "kind",
        "memory",
        "plan_artifact_sha256",
        "plan_publication_git_commit",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "session_lineage",
        "statistics",
        "status",
        "summary_base_git_commit",
        "summary_sha256",
    }
    unsigned = dict(summary)
    recorded = unsigned.pop("summary_sha256", None)
    statistics = summary.get("statistics")
    gate = statistics.get("primary_gate") if isinstance(statistics, Mapping) else None
    lineage = summary.get("session_lineage")
    replay = summary.get("independent_replay")
    memory = summary.get("memory")
    gate_keys = {
        "all_five_sessions_positive",
        "bootstrap_lower_strictly_positive",
        "correctness",
        "overall_pass",
        "point_reduction_at_least_10_percent",
        "positive_prompts_at_least_48",
    }
    if (
        set(summary) != expected
        or summary.get("schema_version") != 1
        or summary.get("kind") != SUMMARY_KIND
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or not _is_git_commit(summary.get("plan_publication_git_commit"))
        or summary.get("plan_sha256") != plan["plan_sha256"]
        or not _is_git_commit(summary.get("summary_base_git_commit"))
        or summary.get("claim_boundary") != plan["claim_boundary"]
        or not isinstance(statistics, Mapping)
        or set(statistics)
        != {
            "component_timing",
            "generic_retrieval_is_novel_claimed",
            "korean_specific_followup_authorized",
            "mechanism",
            "primary_end_to_end",
            "primary_gate",
            "role_order_diagnostic",
            "status",
        }
        or not isinstance(gate, Mapping)
        or set(gate) != gate_keys
        or any(type(gate[name]) is not bool for name in gate_keys)
        or gate["correctness"] is not True
        or gate["overall_pass"]
        is not all(gate[name] for name in gate_keys - {"overall_pass"})
        or statistics.get("generic_retrieval_is_novel_claimed") is not False
        or statistics.get("korean_specific_followup_authorized")
        is not gate["overall_pass"]
        or statistics.get("status")
        != (
            "pass_generic_retrieval_scale_transfer_actual"
            if gate["overall_pass"]
            else "fail_generic_retrieval_scale_transfer_actual"
        )
        or summary.get("status") != statistics.get("status")
        or summary.get("interpretation")
        != _summary_interpretation(gate["overall_pass"])
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE actual summary identity differs")
    if not isinstance(lineage, list) or len(lineage) != SESSIONS:
        raise ValueError("EXAONE actual summary lineage differs")
    for index, row in enumerate(lineage):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "artifact_sha256",
                "receipt_artifact_sha256",
                "receipt_publication_git_commit",
                "receipt_sha256",
                "runner_git_commit",
                "session_index",
            }
            or row.get("session_index") != index
            or not is_sha256(row.get("artifact_sha256"))
            or not is_sha256(row.get("receipt_artifact_sha256"))
            or not is_sha256(row.get("receipt_sha256"))
            or not _is_git_commit(row.get("runner_git_commit"))
            or not _is_git_commit(row.get("receipt_publication_git_commit"))
        ):
            raise ValueError("EXAONE actual summary lineage differs")
    if (
        not isinstance(replay, Mapping)
        or set(replay)
        != {
            "independent_checkpoint_forward_replay",
            "measured_case_count",
            "replay_root_sha256",
            "stored_trial_comparisons",
            "warmup_session_root_comparisons",
        }
        or replay.get("independent_checkpoint_forward_replay") is not True
        or replay.get("measured_case_count") != MEASURED_CASES
        or replay.get("stored_trial_comparisons")
        != SESSIONS * MEASURED_CASES * INNER_REPETITIONS * len(ROLES)
        or replay.get("warmup_session_root_comparisons") != SESSIONS
        or not is_sha256(replay.get("replay_root_sha256"))
    ):
        raise ValueError("EXAONE actual summary replay differs")
    if (
        not isinstance(memory, Mapping)
        or set(memory)
        != {
            "all_session_memory_safety_pass",
            "baseline_trial_peak_active_bytes_maximum",
            "baseline_trial_peak_active_bytes_median",
            "candidate_trial_peak_active_bytes_maximum",
            "candidate_trial_peak_active_bytes_median",
            "claim_scope",
            "session_working_set_fraction_maximum",
        }
        or memory.get("claim_scope") != "descriptive_only_not_a_memory_improvement_gate"
        or memory.get("all_session_memory_safety_pass") is not True
        or any(
            type(memory.get(name)) not in (int, float)
            or not math.isfinite(memory[name])
            or memory[name] <= 0
            for name in (
                "baseline_trial_peak_active_bytes_maximum",
                "baseline_trial_peak_active_bytes_median",
                "candidate_trial_peak_active_bytes_maximum",
                "candidate_trial_peak_active_bytes_median",
            )
        )
        or type(memory.get("session_working_set_fraction_maximum")) is not float
        or not math.isfinite(memory["session_working_set_fraction_maximum"])
        or not 0
        < memory["session_working_set_fraction_maximum"]
        <= MAXIMUM_MEMORY_FRACTION
    ):
        raise ValueError("EXAONE actual summary memory differs")
    for expected_value, actual_value, label in (
        (expected_lineage, lineage, "lineage"),
        (expected_replay, replay, "replay"),
        (expected_statistics, statistics, "statistics"),
        (expected_memory, memory, "memory"),
    ):
        if expected_value is not None and actual_value != expected_value:
            raise ValueError(f"EXAONE actual summary {label} reconstruction differs")


def build_actual_summary(
    *,
    plan: Mapping[str, Any],
    summary_base_git_commit: str,
    plan_publication_git_commit: str,
    session_lineage: Sequence[Mapping[str, Any]],
    independent_replay: Mapping[str, Any],
    statistics: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    gate = statistics.get("primary_gate")
    if not isinstance(gate, Mapping) or type(gate.get("overall_pass")) is not bool:
        raise ValueError("EXAONE actual summary gate differs")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": SUMMARY_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": statistics["status"],
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_publication_git_commit": plan_publication_git_commit,
        "plan_sha256": plan["plan_sha256"],
        "summary_base_git_commit": summary_base_git_commit,
        "session_lineage": [dict(row) for row in session_lineage],
        "independent_replay": dict(independent_replay),
        "statistics": dict(statistics),
        "memory": dict(memory),
        "claim_boundary": plan["claim_boundary"],
        "interpretation": _summary_interpretation(gate["overall_pass"]),
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    validate_actual_summary(
        payload,
        plan=plan,
        expected_lineage=session_lineage,
        expected_replay=independent_replay,
        expected_statistics=statistics,
        expected_memory=memory,
    )
    return payload


def read_actual_summary(
    *,
    plan: Mapping[str, Any],
    verify_derived: bool,
    expected_lineage: Sequence[Mapping[str, Any]] | None = None,
    expected_replay: Mapping[str, Any] | None = None,
    expected_statistics: Mapping[str, Any] | None = None,
    expected_memory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if type(verify_derived) is not bool:
        raise TypeError("EXAONE actual summary verification mode differs")
    if verify_derived and any(
        value is None
        for value in (
            expected_lineage,
            expected_replay,
            expected_statistics,
            expected_memory,
        )
    ):
        raise ValueError("EXAONE actual derived summary evidence is incomplete")
    value = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    validate_actual_summary(
        value,
        plan=plan,
        expected_lineage=expected_lineage if verify_derived else None,
        expected_replay=expected_replay if verify_derived else None,
        expected_statistics=expected_statistics if verify_derived else None,
        expected_memory=expected_memory if verify_derived else None,
    )
    return value
