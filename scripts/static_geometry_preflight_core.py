"""Pure contracts and statistics for the static BLT geometry preflight."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from jamoflow.neural_model import Phase1ModelSpec
from jamoflow.phase3 import PHASE3_MODEL_SPEC


PROTOCOL_ID = "jamoflow-static-geometry-preflight-v1"
MODEL_SEED = 20260813
PROMPT_COUNT = 32
PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
REPETITIONS = 3
WARMUP_PROMPTS = 4
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260901
MINIMUM_POINT_REDUCTION = 0.20
MINIMUM_BOOTSTRAP_LOWER_BOUND = 0.15
MINIMUM_POSITIVE_PROMPTS = 24
MAXIMUM_PARAMETER_RELATIVE_DIFFERENCE = 0.0025
MINIMUM_ANALYTICAL_FLOP_REDUCTION = 0.20

BASELINE = "baseline_w72"
CANDIDATE_ORDER = (
    "thin128_e1_d2_g384x9",
    "thin160_e1_d1_g384x9",
    "thin128_e1_d1_g384x9",
)
GEOMETRY_ORDER = (BASELINE, *CANDIDATE_ORDER)

_OVERRIDES: Mapping[str, Mapping[str, int]] = {
    BASELINE: {},
    # Preserve both decoder layers while reducing local width and one encoder
    # layer.  The global FFN absorbs the small parameter-count residual.
    "thin128_e1_d2_g384x9": {
        "local_width": 128,
        "local_heads": 8,
        "local_ffn": 384,
        "encoder_layers": 1,
        "decoder_layers": 2,
        "global_layers": 9,
        "global_ffn": 1168,
    },
    # Preserve more local width while reducing both local depths.
    "thin160_e1_d1_g384x9": {
        "local_width": 160,
        "local_heads": 8,
        "local_ffn": 480,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "global_layers": 9,
        "global_ffn": 1128,
    },
    # Most aggressive predeclared local reduction.
    "thin128_e1_d1_g384x9": {
        "local_width": 128,
        "local_heads": 8,
        "local_ffn": 384,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "global_layers": 9,
        "global_ffn": 1192,
    },
}


def geometry_spec(name: str) -> Phase1ModelSpec:
    """Return one fixed W72 geometry without mutating the Phase-3 spec."""

    if name not in _OVERRIDES:
        raise ValueError(f"unknown static geometry: {name}")
    return replace(PHASE3_MODEL_SPEC, patch_count=72, **_OVERRIDES[name])


def geometry_contract() -> dict[str, Any]:
    """Return the complete ordered geometry definition for a sealed plan."""

    return {
        "baseline": BASELINE,
        "candidate_order": list(CANDIDATE_ORDER),
        "geometries": {
            name: geometry_spec(name).to_dict() for name in GEOMETRY_ORDER
        },
    }


def validate_geometry_contract(value: Mapping[str, Any]) -> None:
    if value != geometry_contract():
        raise ValueError("static geometry contract differs")


def _finite_timings(value: np.ndarray) -> np.ndarray:
    timings = np.asarray(value)
    expected = (PROMPT_COUNT, REPETITIONS, len(GEOMETRY_ORDER))
    if timings.dtype != np.float64 or timings.shape != expected:
        raise ValueError("geometry timing array shape/dtype differs")
    if not np.all(np.isfinite(timings)) or np.any(timings <= 0):
        raise ValueError("geometry timings must be finite and positive")
    return timings


def _prompt_points(timings: np.ndarray) -> np.ndarray:
    return np.median(timings, axis=1)


def _reduction(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.median(right))
    if denominator <= 0:
        raise ValueError("geometry baseline timing is nonpositive")
    return 1.0 - float(np.median(left)) / denominator


def _bootstrap_interval(
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        rows = rng.integers(0, PROMPT_COUNT, size=PROMPT_COUNT)
        values[repetition] = _reduction(candidate[rows], baseline[rows])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_geometry_preflight(
    *,
    timings_ms: np.ndarray,
    parameter_counts: Mapping[str, int],
    analytical_flops: Mapping[str, int],
    correctness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize a fixed calibration-only static-geometry screen."""

    timings = _finite_timings(timings_ms)
    if set(parameter_counts) != set(GEOMETRY_ORDER):
        raise ValueError("geometry parameter-count keys differ")
    if set(analytical_flops) != set(GEOMETRY_ORDER):
        raise ValueError("geometry analytical-FLOP keys differ")
    if set(correctness) != set(GEOMETRY_ORDER):
        raise ValueError("geometry correctness keys differ")
    if any(not isinstance(parameter_counts[name], int) or parameter_counts[name] <= 0 for name in GEOMETRY_ORDER):
        raise ValueError("geometry parameter count differs")
    if any(not isinstance(analytical_flops[name], int) or analytical_flops[name] <= 0 for name in GEOMETRY_ORDER):
        raise ValueError("geometry analytical FLOPs differ")

    points = _prompt_points(timings)
    baseline_index = GEOMETRY_ORDER.index(BASELINE)
    baseline = points[:, baseline_index]
    baseline_parameters = parameter_counts[BASELINE]
    baseline_flops = analytical_flops[BASELINE]
    rows: dict[str, Any] = {}
    first_passing: str | None = None

    for index, name in enumerate(GEOMETRY_ORDER):
        values = points[:, index]
        relative_parameters = abs(parameter_counts[name] / baseline_parameters - 1.0)
        flop_reduction = 1.0 - analytical_flops[name] / baseline_flops
        evidence = correctness[name]
        expected_correctness_keys = {
            "argmax_comparisons",
            "argmax_exact",
            "boundary_trace_exact",
            "cache_diagnostics_exact",
            "maximum_normalized_logit_error",
        }
        if set(evidence) != expected_correctness_keys:
            raise ValueError(f"geometry correctness schema differs: {name}")
        comparisons = int(evidence["argmax_comparisons"])
        maximum_error = float(evidence["maximum_normalized_logit_error"])
        correctness_pass = (
            comparisons == CONTINUATION_BYTES
            and int(evidence["argmax_exact"]) == comparisons
            and evidence["boundary_trace_exact"] is True
            and evidence["cache_diagnostics_exact"] is True
            and np.isfinite(maximum_error)
            and 0 <= maximum_error <= 1
        )
        if name == BASELINE:
            reduction = 0.0
            lower = 0.0
            upper = 0.0
            positive = 0
            passes = {
                "correctness": correctness_pass,
                "parameter_match": True,
                "analytical_flops": True,
                "point_reduction": True,
                "bootstrap_lower_bound": True,
                "prompt_direction": True,
            }
        else:
            reduction = _reduction(values, baseline)
            lower, upper = _bootstrap_interval(values, baseline)
            prompt_effects = 1.0 - values / baseline
            positive = int(np.sum(prompt_effects > 0))
            passes = {
                "correctness": correctness_pass,
                "parameter_match": relative_parameters
                <= MAXIMUM_PARAMETER_RELATIVE_DIFFERENCE,
                "analytical_flops": flop_reduction
                >= MINIMUM_ANALYTICAL_FLOP_REDUCTION,
                "point_reduction": reduction >= MINIMUM_POINT_REDUCTION,
                "bootstrap_lower_bound": lower
                >= MINIMUM_BOOTSTRAP_LOWER_BOUND,
                "prompt_direction": positive >= MINIMUM_POSITIVE_PROMPTS,
            }
            if first_passing is None and all(passes.values()):
                first_passing = name
        rows[name] = {
            "median_ms": float(np.median(values)),
            "parameter_count": int(parameter_counts[name]),
            "parameter_relative_difference": float(relative_parameters),
            "analytical_dense_matmul_flops": int(analytical_flops[name]),
            "analytical_flop_reduction": float(flop_reduction),
            "end_to_end_reduction": float(reduction),
            "prompt_bootstrap_95_interval": {
                "lower": float(lower),
                "upper": float(upper),
            },
            "positive_prompt_count": positive,
            "correctness": dict(evidence),
            "passes": passes,
            "overall_pass": bool(all(passes.values())),
        }

    return {
        "protocol_id": PROTOCOL_ID,
        "bootstrap": {
            "unit": "calibration prompt after within-prompt repetition median",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "thresholds": {
            "maximum_parameter_relative_difference": MAXIMUM_PARAMETER_RELATIVE_DIFFERENCE,
            "minimum_analytical_flop_reduction": MINIMUM_ANALYTICAL_FLOP_REDUCTION,
            "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
            "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        },
        "rows": rows,
        "selection": {
            "rule": "first passing candidate in the fixed quality-conservative order",
            "candidate_order": list(CANDIDATE_ORDER),
            "selected_candidate": first_passing,
            "one_seed_training_authorized": first_passing is not None,
        },
        "status": (
            "one_seed_static_control_authorized"
            if first_passing is not None
            else "static_geometry_branch_stopped"
        ),
    }
