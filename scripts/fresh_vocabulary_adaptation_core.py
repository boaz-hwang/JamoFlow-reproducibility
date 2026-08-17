"""Core contracts for the fresh one-seed vocabulary-adaptation screen."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from bpe_quality_frontier_core import bpb, document_bootstrap_upper

ROLES = (
    "dense2k_joint",
    "dense8k_standard_joint",
    "dense8k_inplace_two_stage",
    "dense8k_update_geometry",
)
EIGHT_K_ROLES = ROLES[1:]
DEPLOYMENT_TIE_ORDER = (
    "dense8k_standard_joint",
    "dense8k_inplace_two_stage",
    "dense8k_update_geometry",
)

BASE_VOCABULARY_SIZE = 2_048
TARGET_VOCABULARY_SIZE = 8_192
SEQUENCE_LENGTH = 512
EFFECTIVE_BATCH_SIZE = 32
TRAIN_MICROBATCH_BY_VOCABULARY = {2_048: 32, 8_192: 8}
EVALUATION_BATCH_BY_VOCABULARY = {2_048: 64, 8_192: 16}

BODY_LEARNING_RATE = 3e-5
HEAD_PEAK_LEARNING_RATE = 3e-4
HEAD_MINIMUM_LEARNING_RATE = 3e-5
WARMUP_RAW_FRACTION = 0.05
INPLACE_STAGE_ONE_RAW_FRACTION = 0.60
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0

# These values were fixed by the model-loss-free first-AdamW-update audit.
INPUT_UPDATE_MULTIPLIER = 1.485414522979104
OUTPUT_UPDATE_MULTIPLIER = 2.170601418278963

QUALITY_NONINFERIORITY_MARGIN_BPB = 0.010
METHOD_MINIMUM_ADVANTAGE_BPB = 0.002
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_840


def role_definition(role: str) -> dict[str, Any]:
    definitions = {
        "dense2k_joint": {
            "vocabulary_size": BASE_VOCABULARY_SIZE,
            "initialization": "exact_source_checkpoint",
            "schedule": "all_parameter_joint_raw_progress_cosine",
            "post_adamw_new_row_scaling": None,
        },
        "dense8k_standard_joint": {
            "vocabulary_size": TARGET_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "all_parameter_joint_raw_progress_cosine",
            "post_adamw_new_row_scaling": None,
        },
        "dense8k_inplace_two_stage": {
            "vocabulary_size": TARGET_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "new_rows_only_60pct_then_all_40pct",
            "post_adamw_new_row_scaling": None,
        },
        "dense8k_update_geometry": {
            "vocabulary_size": TARGET_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "all_parameter_joint_raw_progress_cosine",
            "post_adamw_new_row_scaling": {
                "input_multiplier": INPUT_UPDATE_MULTIPLIER,
                "output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
                "source": "foldable_multihash_update_audit_v4_first_adamw_projection",
                "validation_metric_used": False,
            },
        },
    }
    if role not in definitions:
        raise ValueError("fresh-adaptation role differs")
    return dict(definitions[role])


def total_optimizer_steps(sequence_count: int) -> int:
    if sequence_count <= 0:
        raise ValueError("fresh-adaptation sequence count differs")
    return math.ceil(sequence_count / EFFECTIVE_BATCH_SIZE)


def batch_raw_target_bytes(raw_target_bytes: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_target_bytes)
    if (
        values.ndim != 1
        or len(values) <= 0
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values <= 0)
    ):
        raise ValueError("fresh-adaptation raw-target array differs")
    return np.asarray(
        [
            int(values[start : start + EFFECTIVE_BATCH_SIZE].sum())
            for start in range(0, len(values), EFFECTIVE_BATCH_SIZE)
        ],
        dtype=np.int64,
    )


def inplace_stage_contract(raw_target_bytes: np.ndarray) -> dict[str, Any]:
    batches = batch_raw_target_bytes(raw_target_bytes)
    cumulative = np.cumsum(batches, dtype=np.int64)
    total = int(cumulative[-1])
    target = total * INPLACE_STAGE_ONE_RAW_FRACTION
    stage_one_steps = int(np.searchsorted(cumulative, target, side="left")) + 1
    stage_one_bytes = int(cumulative[stage_one_steps - 1])
    if not 0 < stage_one_steps < len(batches) or not 0 < stage_one_bytes < total:
        raise ValueError("fresh-adaptation two-stage boundary differs")
    return {
        "boundary_rule": "first_complete_effective_batch_reaching_60pct_raw_target_bytes",
        "requested_stage_one_raw_fraction": INPLACE_STAGE_ONE_RAW_FRACTION,
        "stage_one_optimizer_steps": stage_one_steps,
        "stage_one_raw_target_bytes": stage_one_bytes,
        "stage_one_realized_raw_fraction": stage_one_bytes / total,
        "stage_two_optimizer_steps": len(batches) - stage_one_steps,
        "stage_two_raw_target_bytes": total - stage_one_bytes,
        "total_optimizer_steps": len(batches),
        "total_raw_target_bytes": total,
    }


def _cosine_from_progress(progress: float) -> float:
    if not 0.0 < progress <= 1.0:
        raise ValueError("fresh-adaptation learning-rate progress differs")
    if progress <= WARMUP_RAW_FRACTION:
        return HEAD_PEAK_LEARNING_RATE * progress / WARMUP_RAW_FRACTION
    decay = (progress - WARMUP_RAW_FRACTION) / (1.0 - WARMUP_RAW_FRACTION)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay))
    return (
        HEAD_MINIMUM_LEARNING_RATE
        + (HEAD_PEAK_LEARNING_RATE - HEAD_MINIMUM_LEARNING_RATE) * cosine
    )


def head_learning_rate(
    role: str,
    *,
    cumulative_raw_target_bytes: int,
    total_raw_target_bytes: int,
    stage_one_raw_target_bytes: int | None,
) -> float:
    """Return the predeclared LR at the end of the current raw-byte batch."""

    if (
        role not in ROLES
        or total_raw_target_bytes <= 0
        or not 0 < cumulative_raw_target_bytes <= total_raw_target_bytes
    ):
        raise ValueError("fresh-adaptation learning-rate coordinates differ")
    if role != "dense8k_inplace_two_stage":
        if stage_one_raw_target_bytes is not None:
            raise ValueError("fresh-adaptation non-staged role has a stage boundary")
        return _cosine_from_progress(
            cumulative_raw_target_bytes / total_raw_target_bytes
        )
    if (
        stage_one_raw_target_bytes is None
        or not 0 < stage_one_raw_target_bytes < total_raw_target_bytes
    ):
        raise ValueError("fresh-adaptation staged learning-rate boundary differs")
    if cumulative_raw_target_bytes <= stage_one_raw_target_bytes:
        progress = cumulative_raw_target_bytes / stage_one_raw_target_bytes
        return HEAD_PEAK_LEARNING_RATE * min(1.0, progress / WARMUP_RAW_FRACTION)
    progress = (cumulative_raw_target_bytes - stage_one_raw_target_bytes) / (
        total_raw_target_bytes - stage_one_raw_target_bytes
    )
    return _cosine_from_progress(progress)


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    raw_bytes: np.ndarray,
) -> dict[str, float]:
    point, lower, upper = document_bootstrap_upper(
        candidate,
        reference,
        raw_bytes,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    return {"point_bpb": point, "lower_95_bpb": lower, "upper_95_bpb": upper}


def adaptation_decision(
    document_nll_by_role: Mapping[str, np.ndarray],
    document_raw_bytes: np.ndarray,
) -> dict[str, Any]:
    if set(document_nll_by_role) != set(ROLES):
        raise ValueError("fresh-adaptation decision role set differs")
    raw = np.asarray(document_raw_bytes)
    arrays = {role: np.asarray(document_nll_by_role[role]) for role in ROLES}
    if (
        raw.ndim != 1
        or len(raw) < 2
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw <= 0)
        or any(
            values.shape != raw.shape
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            for values in arrays.values()
        )
    ):
        raise ValueError("fresh-adaptation decision arrays differ")

    document_bpb = {role: bpb(arrays[role], raw) for role in ROLES}
    noninferiority: dict[str, dict[str, Any]] = {}
    qualified: list[str] = []
    for role in EIGHT_K_ROLES:
        row: dict[str, Any] = _comparison(arrays[role], arrays["dense2k_joint"], raw)
        row["margin_bpb"] = QUALITY_NONINFERIORITY_MARGIN_BPB
        row["pass"] = (
            row["point_bpb"] <= QUALITY_NONINFERIORITY_MARGIN_BPB
            and row["upper_95_bpb"] <= QUALITY_NONINFERIORITY_MARGIN_BPB
        )
        noninferiority[role] = row
        if row["pass"]:
            qualified.append(role)

    selected = (
        min(
            qualified,
            key=lambda role: (
                document_bpb[role],
                DEPLOYMENT_TIE_ORDER.index(role),
            ),
        )
        if qualified
        else None
    )

    method_comparisons: dict[str, dict[str, Any]] = {}
    for control in (
        "dense8k_standard_joint",
        "dense8k_inplace_two_stage",
    ):
        row = _comparison(arrays["dense8k_update_geometry"], arrays[control], raw)
        row["minimum_advantage_bpb"] = METHOD_MINIMUM_ADVANTAGE_BPB
        row["pass"] = (
            row["point_bpb"] <= -METHOD_MINIMUM_ADVANTAGE_BPB
            and row["upper_95_bpb"] <= 0.0
        )
        method_comparisons[control] = row
    method_pass = all(row["pass"] for row in method_comparisons.values())

    if selected is None:
        status = "no_quality_qualified_dense8k"
    elif method_pass:
        status = "optimizer_geometry_and_deployment_opportunity"
    else:
        status = "deployment_opportunity_without_optimizer_novelty"
    return {
        "status": status,
        "document_bpb_by_role": document_bpb,
        "dense8k_noninferiority_vs_dense2k": noninferiority,
        "quality_qualified_dense8k_roles": qualified,
        "selected_dense8k_role_for_actual_preflight": selected,
        "deployment_tie_order": list(DEPLOYMENT_TIE_ORDER),
        "actual_inference_preflight_authorized": selected is not None,
        "optimizer_geometry_method_comparisons": method_comparisons,
        "optimizer_geometry_method_supported": method_pass,
        "fresh_multiseed_method_confirmation_authorized": method_pass,
        "publication_claim_authorized": False,
    }
