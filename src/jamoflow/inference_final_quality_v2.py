"""Sealed-final role resolution and five-seed quality gates for inference v2."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .compute_conversion import CONVERSION_RATES, conversion_policy
from .document_inference import (
    DocumentWindowMap,
    document_cluster_contrast_summary,
)
from .inference_quality import inference_quality_noninferiority
from .inference_selection_v2 import validate_selection_lock_v2
from .phase1_analysis import paired_t_interval


FINAL_SEEDS = (1729, 2718, 31415, 57721, 65537)
PRIMARY_LOGICAL_ROLES = (
    "candidate",
    "matched_efficiency_baseline",
    "same_rate_codepoint_control",
)
BROAD_REFERENCE_ROLE = "broad_reference"
FINAL_LOGICAL_ROLES = (*PRIMARY_LOGICAL_ROLES, BROAD_REFERENCE_ROLE)
FINAL_BOOTSTRAP_REPETITIONS = 10_000
FINAL_BOOTSTRAP_SEED = 20_260_814
REFERENCE_NONINFERIORITY_MARGIN_BPB = 0.010
MECHANISM_MAXIMUM_MEAN_EFFECT_BPB = -0.002
MECHANISM_REQUIRED_NEGATIVE_SEEDS = 4


def _copy_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "model_family",
        "patch_count",
        "policy",
        "requires_entropy_router",
        "runtime_policy",
    }
    if not keys <= set(value):
        raise ValueError("final-quality model descriptor is incomplete")
    descriptor = {key: value[key] for key in sorted(keys)}
    if (
        descriptor["model_family"] not in {"phase3", "compute_conversion"}
        or not isinstance(descriptor["patch_count"], int)
        or isinstance(descriptor["patch_count"], bool)
        or descriptor["patch_count"] <= 0
        or not isinstance(descriptor["policy"], str)
        or not descriptor["policy"]
        or not isinstance(descriptor["runtime_policy"], str)
        or not descriptor["runtime_policy"]
        or not isinstance(descriptor["requires_entropy_router"], bool)
    ):
        raise ValueError("final-quality model descriptor is malformed")
    return descriptor


def resolve_final_evaluation_roles(
    selection_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve all logical roles from the one canonical calibration-only lock."""

    validate_selection_lock_v2(selection_lock)
    decision = selection_lock["decision"]
    if decision.get("status") != "locked_pending_confirmation_and_new_final_test":
        raise ValueError("final quality requires a successful selection-v2 lock")
    rate = decision.get("rate_selection", {}).get("selected_rate")
    if rate not in CONVERSION_RATES:
        raise ValueError("final quality has no locked conversion rate")
    candidate = _copy_descriptor(decision["candidate"])
    matched_baseline = _copy_descriptor(decision["matched_efficiency_baseline"])
    broad_reference = _copy_descriptor(decision["reference"])
    control = {
        "model_family": "compute_conversion",
        "patch_count": rate,
        "policy": conversion_policy("codepoint", rate),
        "requires_entropy_router": False,
        "runtime_policy": "causal_codepoint_grid",
    }
    if (
        candidate["model_family"] != "compute_conversion"
        or candidate["patch_count"] != rate
        or candidate["policy"] != conversion_policy("whitespace", rate)
        or candidate["runtime_policy"] != "causal_whitespace_grid"
        or candidate["requires_entropy_router"] is not False
    ):
        raise ValueError("final-quality candidate differs from the locked rate")
    if (
        matched_baseline["model_family"] != "phase3"
        or matched_baseline["patch_count"] != 86
        or matched_baseline["policy"] != "causal_codepoint_grid"
        or matched_baseline["runtime_policy"] != "causal_codepoint_grid"
        or matched_baseline["requires_entropy_router"] is not False
    ):
        raise ValueError("final-quality matched baseline is not locked C86")

    logical = {
        "candidate": candidate,
        "matched_efficiency_baseline": matched_baseline,
        "same_rate_codepoint_control": control,
    }
    broad_status = decision.get("broad_reference_evaluation_status")
    if broad_status not in {
        "eligible_pending_confirmation",
        "not_authorized_calibration_futility",
    }:
        raise ValueError("final-quality broad-reference status is invalid")
    evaluation_role_order = list(PRIMARY_LOGICAL_ROLES)
    if broad_status == "eligible_pending_confirmation":
        logical[BROAD_REFERENCE_ROLE] = broad_reference
        evaluation_role_order.append(BROAD_REFERENCE_ROLE)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    aliases: dict[str, str] = {}
    for role in evaluation_role_order:
        descriptor = logical[role]
        identity = (
            descriptor["model_family"],
            descriptor["policy"],
            descriptor["patch_count"],
        )
        artifact_role = next(
            (
                existing["artifact_role"]
                for existing in unique
                if (
                    existing["model_family"],
                    existing["policy"],
                    existing["patch_count"],
                )
                == identity
            ),
            role,
        )
        aliases[role] = artifact_role
        if identity not in seen:
            unique.append({"artifact_role": role, **descriptor})
            seen.add(identity)
    return {
        "broad_reference": {
            "descriptor": broad_reference,
            "evaluation_status": broad_status,
        },
        "evaluation_role_order": evaluation_role_order,
        "logical_roles": logical,
        "role_to_artifact_role": aliases,
        "seed_order": list(FINAL_SEEDS),
        "unique_models": unique,
    }


def _validate_losses(
    losses_by_role: Mapping[str, Mapping[int, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    role_order = tuple(losses_by_role)
    if role_order not in (
        PRIMARY_LOGICAL_ROLES,
        (*PRIMARY_LOGICAL_ROLES, BROAD_REFERENCE_ROLE),
    ):
        raise ValueError("final quality requires the exact ordered logical roles")
    validated: dict[str, dict[int, np.ndarray]] = {}
    expected_shape: tuple[int, ...] | None = None
    for role in role_order:
        row = losses_by_role[role]
        if set(row) != set(FINAL_SEEDS):
            raise ValueError(f"final quality has an incomplete seed set: {role}")
        validated[role] = {}
        for seed in FINAL_SEEDS:
            values = np.asarray(row[seed])
            if (
                values.dtype != np.float32
                or values.ndim != 1
                or not len(values)
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError("final quality losses must be finite float32 vectors")
            if expected_shape is None:
                expected_shape = values.shape
            elif values.shape != expected_shape:
                raise ValueError("final quality roles do not share one sequence set")
            validated[role][seed] = values
    return validated


def final_quality_gate_v2(
    losses_by_role: Mapping[str, Mapping[int, np.ndarray]],
    *,
    role_descriptors: Mapping[str, Mapping[str, Any]],
    document_window_map: DocumentWindowMap,
    targets_per_sequence: int = 511,
    bootstrap_repetitions: int = FINAL_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = FINAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate reference noninferiority and the W-vs-C mechanism contrast."""

    if tuple(role_descriptors) != tuple(losses_by_role):
        raise ValueError("final quality descriptors must match ordered evaluated roles")
    if (
        targets_per_sequence != 511
        or bootstrap_repetitions <= 0
        or document_window_map.sequence_length != 512
    ):
        raise ValueError("final quality protocol constants differ")
    losses = _validate_losses(losses_by_role)
    sequence_count = len(losses["candidate"][FINAL_SEEDS[0]])
    if document_window_map.sequence_count != sequence_count:
        raise ValueError("final quality document map differs from loss vectors")

    candidate_policy = str(role_descriptors["candidate"]["policy"])
    matched_policy = str(
        role_descriptors["matched_efficiency_baseline"]["policy"]
    )
    control_policy = str(
        role_descriptors["same_rate_codepoint_control"]["policy"]
    )
    matched_efficiency_gate = inference_quality_noninferiority(
        losses["candidate"],
        losses["matched_efficiency_baseline"],
        seed_order=FINAL_SEEDS,
        candidate_policy=candidate_policy,
        reference_policy=matched_policy,
        targets_per_sequence=targets_per_sequence,
        document_window_map=document_window_map,
        margin_bpb=REFERENCE_NONINFERIORITY_MARGIN_BPB,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    ).to_dict()

    scale = targets_per_sequence * math.log(2.0)
    differences_nats = [
        losses["candidate"][seed].astype(np.float64)
        - losses["same_rate_codepoint_control"][seed].astype(np.float64)
        for seed in FINAL_SEEDS
    ]
    effects = [float(values.mean()) / scale for values in differences_nats]
    interval = paired_t_interval(effects)
    document_cluster = document_cluster_contrast_summary(
        differences_nats,
        document_window_map,
        targets_per_sequence=targets_per_sequence,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 200,
    )
    negative_seed_count = sum(effect < 0 for effect in effects)
    mechanism_pass = bool(
        interval.mean <= MECHANISM_MAXIMUM_MEAN_EFFECT_BPB
        and interval.upper < 0
        and float(document_cluster["upper"]) < 0
        and document_window_map.coverage_pass
        and negative_seed_count >= MECHANISM_REQUIRED_NEGATIVE_SEEDS
    )
    mechanism = {
        "candidate_policy": candidate_policy,
        "control_policy": control_policy,
        "difference_direction": "candidate_minus_control; lower favors candidate",
        "document_cluster": document_cluster,
        "maximum_mean_effect_bpb": MECHANISM_MAXIMUM_MEAN_EFFECT_BPB,
        "negative_seed_count": negative_seed_count,
        "overall_pass": mechanism_pass,
        "paired_differences_bpb": effects,
        "paired_seed_t_95": interval.to_dict(),
        "required_negative_seed_count": MECHANISM_REQUIRED_NEGATIVE_SEEDS,
        "status": "pass" if mechanism_pass else "fail_mechanism_replication",
    }
    broad_gate = None
    if BROAD_REFERENCE_ROLE in losses:
        broad_gate = inference_quality_noninferiority(
            losses["candidate"],
            losses[BROAD_REFERENCE_ROLE],
            seed_order=FINAL_SEEDS,
            candidate_policy=candidate_policy,
            reference_policy=str(
                role_descriptors[BROAD_REFERENCE_ROLE]["policy"]
            ),
            targets_per_sequence=targets_per_sequence,
            document_window_map=document_window_map,
            margin_bpb=REFERENCE_NONINFERIORITY_MARGIN_BPB,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed + 400,
        ).to_dict()
    overall_pass = bool(
        matched_efficiency_gate["overall_pass"] and mechanism_pass
    )
    return {
        "actual_timing_authorized": bool(
            matched_efficiency_gate["overall_pass"]
        ),
        "broad_actual_timing_authorized": (
            bool(broad_gate["overall_pass"])
            if isinstance(broad_gate, Mapping)
            else False
        ),
        "broad_candidate_vs_strongest_reference": broad_gate,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": bootstrap_seed,
        "candidate_vs_matched_efficiency_baseline": matched_efficiency_gate,
        "evaluated_role_order": list(losses),
        "mechanism_candidate_vs_same_rate_codepoint": mechanism,
        "mechanism_timing_authorized": mechanism_pass,
        "matched_quality_timing_authorized": bool(
            matched_efficiency_gate["overall_pass"]
        ),
        "overall_pass": overall_pass,
        "seed_order": list(FINAL_SEEDS),
        "sequence_count": sequence_count,
        "status": "pass" if overall_pass else "fail_final_quality_v2",
        "targets_per_sequence": targets_per_sequence,
    }
