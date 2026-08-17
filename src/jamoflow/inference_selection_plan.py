"""Result-blind plan for canonical Phase 3 inference selection v2."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .compute_conversion import CONVERSION_POLICIES, CONVERSION_RATES
from .inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    CONFIRMATION_SEEDS,
    INITIAL_SEEDS,
)
from .phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
)


PLAN_KIND = "phase3_inference_selection_plan_v2"
PLAN_PROTOCOL_VERSION = 2
FINAL_TEST_MANIFEST_PATH = "data/manifests/hplt3-korean-final-test-v1.json"
FINAL_TEST_SEAL_PATH = "data/seals/hplt3-korean-final-test-v1.json"
PHASE3_ALL_INITIAL_SUMMARY_PATH = "results/phase3-all-initial/summary.json"
PHASE3_PRIMARY_SUMMARY_PATH = "results/phase3-primary-five-seed/summary.json"
SELECTION_PLAN_PATH = "data/manifests/phase3-inference-selection-plan-v2.json"
CALIBRATION_EVIDENCE_PATH = (
    "results/phase3-inference-selection-v2/calibration-evidence.json"
)
SELECTION_LOCK_PATH = "results/phase3-inference-selection-v2/selection-lock.json"

_TOP_KEYS = {
    "calibration_evaluator",
    "execution_paths",
    "final_test",
    "historical_screening",
    "initial_design",
    "kind",
    "plan_git_commit",
    "plan_sha256",
    "protocol_version",
    "schema_version",
    "selection_rules",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def build_selection_plan_v2(
    *,
    plan_git_commit: str,
    final_test_manifest_sha256: str,
    final_test_seal_sha256: str,
    final_test_payload_sha256: str,
    phase3_all_initial_summary_sha256: str,
    phase3_primary_summary_sha256: str,
    source_artifact_sha256: str,
    source_integrity_artifact_sha256: str,
    calibration_stream_sha256: str,
    calibration_sequence_count: int,
) -> dict[str, Any]:
    """Build a plan that has no final-test metric or latency input."""

    hashes = (
        final_test_manifest_sha256,
        final_test_seal_sha256,
        final_test_payload_sha256,
        phase3_all_initial_summary_sha256,
        phase3_primary_summary_sha256,
        source_artifact_sha256,
        source_integrity_artifact_sha256,
        calibration_stream_sha256,
    )
    if not _is_git_commit(plan_git_commit) or not all(
        _is_sha256(value) for value in hashes
    ):
        raise ValueError("selection-v2 plan identities are malformed")
    if (
        not isinstance(calibration_sequence_count, int)
        or isinstance(calibration_sequence_count, bool)
        or calibration_sequence_count <= 0
    ):
        raise ValueError("selection-v2 calibration count must be positive")
    unsigned: dict[str, Any] = {
        "calibration_evaluator": {
            "aggregation": "float64-fsum-nats/(count*511*ln2)",
            "batch_size": PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
            "device": "mps",
            "evaluator_protocol": "jamoflow-calibration-evaluator-v2",
            "input_stream_sha256": calibration_stream_sha256,
            "output_dtype": "float32",
            "predicted_bytes_per_sequence": PHASE3_MODEL_SPEC.sequence_length - 1,
            "requires_checkpoint_state_reconstruction": True,
            "sequence_count": calibration_sequence_count,
            "sequence_length": PHASE3_MODEL_SPEC.sequence_length,
            "split": "calibration",
        },
        "execution_paths": {
            "calibration_evidence": CALIBRATION_EVIDENCE_PATH,
            "conversion_artifact_root": "artifacts/phase3-compute-conversion",
            "conversion_run_root": "runs/phase3-compute-conversion",
            "phase3_artifact_root": "artifacts/phase3",
            "phase3_run_root": "runs/phase3",
            "selection_lock": SELECTION_LOCK_PATH,
            "selection_plan": SELECTION_PLAN_PATH,
        },
        "final_test": {
            "dataset_id": "hplt3-korean-final-test-v1",
            "evaluated_at_plan": False,
            "manifest_path": FINAL_TEST_MANIFEST_PATH,
            "manifest_sha256": final_test_manifest_sha256,
            "seal_path": FINAL_TEST_SEAL_PATH,
            "seal_payload_sha256": final_test_payload_sha256,
            "seal_sha256": final_test_seal_sha256,
        },
        "historical_screening": {
            "all_initial_summary": {
                "authorizes_final_claim": False,
                "authorizes_selection": False,
                "authorizes_timing": False,
                "path": PHASE3_ALL_INITIAL_SUMMARY_PATH,
                "role": "development_screening_only",
                "sha256": phase3_all_initial_summary_sha256,
            },
            "primary_summary": {
                "authorizes_final_claim": False,
                "authorizes_selection": False,
                "authorizes_timing": False,
                "path": PHASE3_PRIMARY_SUMMARY_PATH,
                "role": "development_screening_only",
                "sha256": phase3_primary_summary_sha256,
            },
        },
        "initial_design": {
            "calibration_policy_order": list(CALIBRATION_POLICY_ORDER),
            "confirmation_seed_order": list(CONFIRMATION_SEEDS),
            "conversion_policy_order": list(CONVERSION_POLICIES),
            "initial_seed_order": list(INITIAL_SEEDS),
            "phase3_policy_order": list(PHASE3_POLICIES),
            "rate_order": list(CONVERSION_RATES),
            "source_artifact_sha256": source_artifact_sha256,
            "source_integrity_artifact_sha256": (
                source_integrity_artifact_sha256
            ),
        },
        "kind": PLAN_KIND,
        "plan_git_commit": plan_git_commit,
        "protocol_version": PLAN_PROTOCOL_VERSION,
        "schema_version": 2,
        "selection_rules": {
            "candidate": "whitespace policy at selected rate",
            "final_test_input": False,
            "historical_screening_test_input": False,
            "latency_input": False,
            "rate_margin_bpb": 0.010,
            "rate_minimum_seed_count_within_margin": 2,
            "rate_rule": "first passing rate in fixed 64,72 order",
            "reference_order": [*PHASE3_POLICIES, "selected_rate_codepoint"],
            "broad_reference_futility_margin_bpb": 0.010,
            "broad_reference_futility_minimum_seed_count_within_margin": 2,
            "reference_rule": "lowest initial-three-seed mean calibration BPB",
            "reference_tie_break": "first in fixed order on exact tie",
            "terminal_no_rate": True,
        },
    }
    plan = {
        **unsigned,
        "plan_sha256": sha256(_canonical_bytes(unsigned)).hexdigest(),
    }
    validate_selection_plan_v2(plan)
    return plan


def validate_selection_plan_v2(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, Mapping) or set(plan) != _TOP_KEYS:
        raise ValueError("selection-v2 plan is not the sealed schema")
    calibration = plan.get("calibration_evaluator")
    paths = plan.get("execution_paths")
    final_test = plan.get("final_test")
    screening = plan.get("historical_screening")
    design = plan.get("initial_design")
    rules = plan.get("selection_rules")
    if not all(
        isinstance(value, Mapping)
        for value in (calibration, paths, final_test, screening, design, rules)
    ):
        raise ValueError("selection-v2 plan sections must be objects")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("kind") != PLAN_KIND
        or plan.get("schema_version") != 2
        or plan.get("protocol_version") != PLAN_PROTOCOL_VERSION
        or not _is_git_commit(plan.get("plan_git_commit"))
        or not _is_sha256(plan.get("plan_sha256"))
        or plan["plan_sha256"] != sha256(_canonical_bytes(unsigned)).hexdigest()
    ):
        raise ValueError("selection-v2 plan identity is invalid")
    if (
        set(final_test)
        != {
            "dataset_id",
            "evaluated_at_plan",
            "manifest_path",
            "manifest_sha256",
            "seal_path",
            "seal_payload_sha256",
            "seal_sha256",
        }
        or final_test.get("dataset_id") != "hplt3-korean-final-test-v1"
        or final_test.get("evaluated_at_plan") is not False
        or final_test.get("manifest_path") != FINAL_TEST_MANIFEST_PATH
        or final_test.get("seal_path") != FINAL_TEST_SEAL_PATH
        or not all(
            _is_sha256(final_test.get(key))
            for key in ("manifest_sha256", "seal_payload_sha256", "seal_sha256")
        )
    ):
        raise ValueError("selection-v2 final-test identity is invalid")
    expected_paths = {
        "calibration_evidence": CALIBRATION_EVIDENCE_PATH,
        "conversion_artifact_root": "artifacts/phase3-compute-conversion",
        "conversion_run_root": "runs/phase3-compute-conversion",
        "phase3_artifact_root": "artifacts/phase3",
        "phase3_run_root": "runs/phase3",
        "selection_lock": SELECTION_LOCK_PATH,
        "selection_plan": SELECTION_PLAN_PATH,
    }
    if dict(paths) != expected_paths:
        raise ValueError("selection-v2 execution paths are not canonical")
    if (
        set(design)
        != {
            "calibration_policy_order",
            "confirmation_seed_order",
            "conversion_policy_order",
            "initial_seed_order",
            "phase3_policy_order",
            "rate_order",
            "source_artifact_sha256",
            "source_integrity_artifact_sha256",
        }
        or tuple(design.get("initial_seed_order", ())) != INITIAL_SEEDS
        or tuple(design.get("confirmation_seed_order", ()))
        != CONFIRMATION_SEEDS
        or tuple(design.get("phase3_policy_order", ())) != PHASE3_POLICIES
        or tuple(design.get("conversion_policy_order", ()))
        != CONVERSION_POLICIES
        or tuple(design.get("calibration_policy_order", ()))
        != CALIBRATION_POLICY_ORDER
        or tuple(design.get("rate_order", ())) != CONVERSION_RATES
        or not _is_sha256(design.get("source_artifact_sha256"))
        or not _is_sha256(design.get("source_integrity_artifact_sha256"))
    ):
        raise ValueError("selection-v2 initial design is invalid")
    if (
        set(calibration)
        != {
            "aggregation",
            "batch_size",
            "device",
            "evaluator_protocol",
            "input_stream_sha256",
            "output_dtype",
            "predicted_bytes_per_sequence",
            "requires_checkpoint_state_reconstruction",
            "sequence_count",
            "sequence_length",
            "split",
        }
        or calibration.get("aggregation")
        != "float64-fsum-nats/(count*511*ln2)"
        or calibration.get("batch_size")
        != PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size
        or calibration.get("device") != "mps"
        or calibration.get("evaluator_protocol")
        != "jamoflow-calibration-evaluator-v2"
        or calibration.get("output_dtype") != "float32"
        or calibration.get("predicted_bytes_per_sequence") != 511
        or calibration.get("requires_checkpoint_state_reconstruction") is not True
        or not isinstance(calibration.get("sequence_count"), int)
        or isinstance(calibration.get("sequence_count"), bool)
        or calibration.get("sequence_count") <= 0
        or calibration.get("sequence_length") != 512
        or calibration.get("split") != "calibration"
        or not _is_sha256(calibration.get("input_stream_sha256"))
    ):
        raise ValueError("selection-v2 calibration evaluator is invalid")
    expected_rules = {
        "candidate": "whitespace policy at selected rate",
        "final_test_input": False,
        "historical_screening_test_input": False,
        "latency_input": False,
        "rate_margin_bpb": 0.010,
        "rate_minimum_seed_count_within_margin": 2,
        "rate_rule": "first passing rate in fixed 64,72 order",
        "reference_order": [*PHASE3_POLICIES, "selected_rate_codepoint"],
        "broad_reference_futility_margin_bpb": 0.010,
        "broad_reference_futility_minimum_seed_count_within_margin": 2,
        "reference_rule": "lowest initial-three-seed mean calibration BPB",
        "reference_tie_break": "first in fixed order on exact tie",
        "terminal_no_rate": True,
    }
    if dict(rules) != expected_rules or not math.isclose(
        float(rules["rate_margin_bpb"]), 0.010, rel_tol=0, abs_tol=0
    ):
        raise ValueError("selection-v2 selection rules are invalid")
    if set(screening) != {"all_initial_summary", "primary_summary"}:
        raise ValueError("selection-v2 screening identities are incomplete")
    expected_screening_paths = {
        "all_initial_summary": PHASE3_ALL_INITIAL_SUMMARY_PATH,
        "primary_summary": PHASE3_PRIMARY_SUMMARY_PATH,
    }
    for key, path in expected_screening_paths.items():
        value = screening.get(key)
        if not isinstance(value, Mapping) or dict(value) != {
            "authorizes_final_claim": False,
            "authorizes_selection": False,
            "authorizes_timing": False,
            "path": path,
            "role": "development_screening_only",
            "sha256": value.get("sha256"),
        } or not _is_sha256(value.get("sha256")):
            raise ValueError("selection-v2 screening role is invalid")
