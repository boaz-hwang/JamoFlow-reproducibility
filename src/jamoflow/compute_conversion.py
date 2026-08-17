"""Preregistered lower-global-rate C/W policies and blinded selection gates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .neural_model import Phase1ModelSpec
from .phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_window_grid_trace,
    padded_hf_patch_matrix,
)
from .phase3 import PHASE3_MODEL_SPEC


CONVERSION_RATES = (64, 72)
CONVERSION_KINDS = ("codepoint", "whitespace")


def conversion_policy(kind: str, rate: int) -> str:
    if kind not in CONVERSION_KINDS or rate not in CONVERSION_RATES:
        raise ValueError("unknown compute-conversion policy")
    return f"causal_{kind}_grid_{rate}"


CONVERSION_POLICIES = tuple(
    conversion_policy(kind, rate)
    for rate in CONVERSION_RATES
    for kind in CONVERSION_KINDS
)


def conversion_model_spec(rate: int) -> Phase1ModelSpec:
    if rate not in CONVERSION_RATES:
        raise ValueError("unknown compute-conversion rate")
    return replace(PHASE3_MODEL_SPEC, patch_count=rate)


def conversion_patch_matrices(
    boundary_masks: np.ndarray,
    whitespace_masks: np.ndarray,
    *,
    rate: int,
) -> dict[str, np.ndarray]:
    """Construct exact-rate C/W matrices for one preregistered rate."""

    spec = conversion_model_spec(rate)
    expected = (len(boundary_masks), spec.sequence_length)
    if boundary_masks.ndim != 2 or boundary_masks.shape != expected:
        raise ValueError("conversion boundary masks have an unexpected shape")
    if whitespace_masks.shape != boundary_masks.shape:
        raise ValueError("conversion whitespace masks must match boundaries")
    codepoint_rows = []
    whitespace_rows = []
    for boundaries, whitespace in zip(
        boundary_masks,
        whitespace_masks,
        strict=True,
    ):
        codepoint_rows.append(
            causal_codepoint_grid_boundaries(boundaries, rate)
        )
        whitespace_rows.append(
            causal_window_grid_trace(boundaries, whitespace, rate).boundaries
        )
    return {
        conversion_policy("codepoint", rate): padded_hf_patch_matrix(
            codepoint_rows,
            spec.sequence_length,
        ),
        conversion_policy("whitespace", rate): padded_hf_patch_matrix(
            whitespace_rows,
            spec.sequence_length,
        ),
    }


@dataclass(frozen=True, slots=True)
class RateSelection:
    selected_rate: int | None
    selected_policy: str | None
    status: str
    by_rate: dict[str, dict[str, float | int | bool]]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_rate": self.selected_rate,
            "selected_policy": self.selected_policy,
            "status": self.status,
            "by_rate": self.by_rate,
        }


def select_rate_from_calibration(
    calibration_bpb: Mapping[int, Mapping[str, float]],
    primary_codepoint_bpb: Mapping[int, float],
    *,
    margin_bpb: float = 0.010,
) -> RateSelection:
    """Choose 64 before 72 using calibration only and a fixed margin."""

    seeds = tuple(sorted(primary_codepoint_bpb))
    if len(seeds) != 3 or set(calibration_bpb) != set(seeds):
        raise ValueError("rate selection requires exactly three paired seeds")
    by_rate: dict[str, dict[str, float | int | bool]] = {}
    selected: int | None = None
    for rate in CONVERSION_RATES:
        policy = conversion_policy("whitespace", rate)
        if any(policy not in calibration_bpb[seed] for seed in seeds):
            raise ValueError(f"missing calibration policy: {policy}")
        effects = [
            float(calibration_bpb[seed][policy])
            - float(primary_codepoint_bpb[seed])
            for seed in seeds
        ]
        mean_effect = float(np.mean(effects))
        within = sum(effect <= margin_bpb for effect in effects)
        passed = mean_effect <= margin_bpb and within >= 2
        by_rate[str(rate)] = {
            "mean_whitespace_minus_primary_codepoint_bpb": mean_effect,
            "seed_count_within_margin": within,
            "required_seed_count_within_margin": 2,
            "margin_bpb": margin_bpb,
            "pass": passed,
        }
        if selected is None and passed:
            selected = rate
    return RateSelection(
        selected_rate=selected,
        selected_policy=(
            conversion_policy("whitespace", selected)
            if selected is not None
            else None
        ),
        status="selected" if selected is not None else "fail_no_rate",
        by_rate=by_rate,
    )


def initial_conversion_gate(
    test_bpb: Mapping[int, Mapping[str, float]],
    primary_codepoint_bpb: Mapping[int, float],
    *,
    selected_rate: int,
    noninferiority_margin_bpb: float = 0.010,
    same_rate_mean_effect_bpb: float = -0.002,
) -> dict[str, object]:
    """Evaluate held-out quality only after calibration has selected a rate."""

    seeds = tuple(sorted(primary_codepoint_bpb))
    if len(seeds) != 3 or set(test_bpb) != set(seeds):
        raise ValueError("initial conversion gate requires three paired seeds")
    whitespace = conversion_policy("whitespace", selected_rate)
    codepoint = conversion_policy("codepoint", selected_rate)
    if any(
        whitespace not in test_bpb[seed] or codepoint not in test_bpb[seed]
        for seed in seeds
    ):
        raise ValueError("initial conversion gate lacks selected rate policies")
    versus_primary = [
        float(test_bpb[seed][whitespace])
        - float(primary_codepoint_bpb[seed])
        for seed in seeds
    ]
    versus_same_rate = [
        float(test_bpb[seed][whitespace]) - float(test_bpb[seed][codepoint])
        for seed in seeds
    ]
    primary_mean = float(np.mean(versus_primary))
    same_rate_mean = float(np.mean(versus_same_rate))
    primary_count = sum(
        effect <= noninferiority_margin_bpb for effect in versus_primary
    )
    same_rate_negative_count = sum(effect < 0 for effect in versus_same_rate)
    passed = bool(
        primary_mean <= noninferiority_margin_bpb
        and primary_count >= 2
        and same_rate_mean <= same_rate_mean_effect_bpb
        and same_rate_negative_count >= 2
    )
    return {
        "status": "pass" if passed else "fail_quality_conversion",
        "overall_pass": passed,
        "selected_rate": selected_rate,
        "whitespace_policy": whitespace,
        "same_rate_codepoint_policy": codepoint,
        "seed_order": list(seeds),
        "whitespace_minus_primary_codepoint_effects_bpb": versus_primary,
        "mean_whitespace_minus_primary_codepoint_bpb": primary_mean,
        "primary_noninferiority_margin_bpb": noninferiority_margin_bpb,
        "primary_seed_count_within_margin": primary_count,
        "required_primary_seed_count_within_margin": 2,
        "whitespace_minus_same_rate_codepoint_effects_bpb": versus_same_rate,
        "mean_whitespace_minus_same_rate_codepoint_bpb": same_rate_mean,
        "maximum_same_rate_mean_effect_bpb": same_rate_mean_effect_bpb,
        "same_rate_negative_seed_count": same_rate_negative_count,
        "required_same_rate_negative_seed_count": 2,
    }
