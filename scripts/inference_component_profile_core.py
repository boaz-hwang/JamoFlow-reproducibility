"""Pure contracts and aggregation for exploratory incremental profiling."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


PROFILE_PROTOCOL_ID = "jamoflow-exploratory-component-profile-v1"
PROFILE_SEEDS = (1729, 2718, 31415, 57721, 65537)
PROFILE_CHECKPOINT_ROLES = ("candidate", "reference")
PROFILE_SCHEDULES: Mapping[str, Mapping[str, Any]] = {
    "W72": {
        "patch_count": 72,
        "policy": "causal_whitespace_grid",
    },
    "C86": {
        "patch_count": 86,
        "policy": "causal_codepoint_grid",
    },
}
PROFILE_WHOLE_CASES = 16
PROFILE_WHOLE_REPETITIONS = 3
PROFILE_COMPONENT_CASES = 4
PROFILE_WARMUP_CASES = 4
PROFILE_DECODE_BYTES = 127
WHOLE_METRICS = ("ttft_ms", "decode_ms", "end_to_end_ms")
COMPONENTS = (
    "local_encoder",
    "patch_finalize_global",
    "local_decoder",
    "lm_head",
)


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != shape:
        raise ValueError(f"{name} shape/dtype differs")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} contains invalid values")
    return array


def _cell_points(values: np.ndarray) -> tuple[float, list[float]]:
    """Collapse repetitions, then cases, without treating reps as samples."""

    case_points = np.median(values, axis=-1)
    seed_points = np.median(case_points, axis=-1)
    return float(np.median(case_points)), [float(x) for x in seed_points]


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_point, left_seeds = _cell_points(left)
    right_point, right_seeds = _cell_points(right)
    if right_point <= 0 or any(value <= 0 for value in right_seeds):
        raise ValueError("profile comparison denominator is nonpositive")
    per_seed = [
        1.0 - left_value / right_value
        for left_value, right_value in zip(left_seeds, right_seeds, strict=True)
    ]
    return {
        "left_median_ms": left_point,
        "right_median_ms": right_point,
        "median_reduction": 1.0 - left_point / right_point,
        "per_seed_reduction": {
            str(seed): float(value)
            for seed, value in zip(PROFILE_SEEDS, per_seed, strict=True)
        },
        "positive_seed_count": int(sum(value > 0 for value in per_seed)),
    }


def summarize_profile_arrays(
    *,
    whole_ms: np.ndarray,
    step_ms: np.ndarray,
    step_boundary: np.ndarray,
    component_total_ms: np.ndarray,
    component_calls: np.ndarray,
    prompt_patches: np.ndarray,
    final_patches: np.ndarray,
) -> dict[str, Any]:
    """Build deterministic descriptive aggregates from fixed profile arrays."""

    prefix = (
        len(PROFILE_SEEDS),
        len(PROFILE_CHECKPOINT_ROLES),
        len(PROFILE_SCHEDULES),
    )
    whole = _finite(
        "whole_ms",
        whole_ms,
        (*prefix, PROFILE_WHOLE_CASES, PROFILE_WHOLE_REPETITIONS, len(WHOLE_METRICS)),
    )
    steps = _finite(
        "step_ms",
        step_ms,
        (*prefix, PROFILE_COMPONENT_CASES, PROFILE_DECODE_BYTES),
    )
    boundaries = np.asarray(step_boundary)
    if boundaries.dtype != np.bool_ or boundaries.shape != steps.shape:
        raise ValueError("step boundary array differs")
    totals = _finite(
        "component_total_ms",
        component_total_ms,
        (*prefix, PROFILE_COMPONENT_CASES, len(COMPONENTS)),
    )
    calls = np.asarray(component_calls)
    if (
        calls.dtype != np.int64
        or calls.shape != totals.shape
        or np.any(calls <= 0)
    ):
        raise ValueError("component calls differ")
    prompts = np.asarray(prompt_patches)
    finals = np.asarray(final_patches)
    patch_shape = (*prefix, PROFILE_WHOLE_CASES, PROFILE_WHOLE_REPETITIONS)
    if (
        prompts.dtype != np.int64
        or finals.dtype != np.int64
        or prompts.shape != patch_shape
        or finals.shape != patch_shape
        or np.any(prompts <= 0)
        or np.any(finals < prompts)
    ):
        raise ValueError("profile patch counts differ")

    whole_summary: dict[str, Any] = {}
    for metric_index, metric in enumerate(WHOLE_METRICS):
        cells: dict[str, Any] = {}
        for role_index, role in enumerate(PROFILE_CHECKPOINT_ROLES):
            cells[role] = {}
            for schedule_index, schedule in enumerate(PROFILE_SCHEDULES):
                point, per_seed = _cell_points(
                    whole[:, role_index, schedule_index, :, :, metric_index]
                )
                cells[role][schedule] = {
                    "median_ms": point,
                    "per_seed_median_ms": {
                        str(seed): value
                        for seed, value in zip(PROFILE_SEEDS, per_seed, strict=True)
                    },
                }
        schedule_effect = {
            role: _comparison(
                whole[:, role_index, 0, :, :, metric_index],
                whole[:, role_index, 1, :, :, metric_index],
            )
            for role_index, role in enumerate(PROFILE_CHECKPOINT_ROLES)
        }
        native = _comparison(
            whole[:, 0, 0, :, :, metric_index],
            whole[:, 1, 1, :, :, metric_index],
        )
        whole_summary[metric] = {
            "cells": cells,
            "same_checkpoint_W72_vs_C86_schedule": schedule_effect,
            "native_candidate_W72_vs_reference_C86": native,
        }

    step_summary: dict[str, Any] = {}
    component_summary: dict[str, Any] = {}
    patch_summary: dict[str, Any] = {}
    for role_index, role in enumerate(PROFILE_CHECKPOINT_ROLES):
        step_summary[role] = {}
        component_summary[role] = {}
        patch_summary[role] = {}
        for schedule_index, schedule in enumerate(PROFILE_SCHEDULES):
            values = steps[:, role_index, schedule_index]
            mask = boundaries[:, role_index, schedule_index]
            if not np.any(mask) or not np.any(~mask):
                raise ValueError("profile lacks boundary or non-boundary steps")
            boundary_values = values[mask]
            nonboundary_values = values[~mask]
            step_summary[role][schedule] = {
                "boundary_step_count": int(mask.sum()),
                "nonboundary_step_count": int((~mask).sum()),
                "boundary_step_median_ms": float(np.median(boundary_values)),
                "nonboundary_step_median_ms": float(np.median(nonboundary_values)),
                "median_boundary_increment_ms": float(
                    np.median(boundary_values) - np.median(nonboundary_values)
                ),
            }
            component_summary[role][schedule] = {}
            for component_index, component in enumerate(COMPONENTS):
                per_call = (
                    totals[:, role_index, schedule_index, :, component_index]
                    / calls[:, role_index, schedule_index, :, component_index]
                )
                component_summary[role][schedule][component] = {
                    "synchronized_per_call_median_ms": float(np.median(per_call)),
                    "total_call_count": int(
                        calls[:, role_index, schedule_index, :, component_index].sum()
                    ),
                }
            prompt_values = prompts[:, role_index, schedule_index]
            final_values = finals[:, role_index, schedule_index]
            patch_summary[role][schedule] = {
                "prompt_patch_median": float(np.median(prompt_values)),
                "final_patch_median": float(np.median(final_values)),
                "decode_new_patch_median": float(
                    np.median(final_values - prompt_values)
                ),
            }

    return {
        "component_synchronized_diagnostic": component_summary,
        "patch_counts": patch_summary,
        "step_synchronized_diagnostic": step_summary,
        "whole_trial": whole_summary,
    }
