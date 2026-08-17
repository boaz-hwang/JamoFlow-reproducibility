"""Pure statistics for the frozen-W72 conditional-local sensitivity screen."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from static_geometry_one_seed_core import one_seed_document_bootstrap


PROTOCOL_ID = "jamoflow-conditional-local-frozen-sensitivity-v1"
ROUTE_ORDER = ("utf8_incomplete", "hangul_prefix")
PAIR_ORDER = (
    "encoder_decoder__second_layer_kv",
    "decoder__second_layer_kv",
    "encoder_decoder__second_mlp",
    "decoder__second_mlp",
)
CANDIDATE_ORDER = tuple(
    f"{route}__{pair}" for pair in PAIR_ORDER for route in ROUTE_ORDER
)
RISK_MARGIN_BPB = 0.020
MINIMUM_EASY_RATE = 0.30
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_261_101
TARGETS_PER_SEQUENCE = 511
PREOUTCOME_ROUTE_GEOMETRY = {
    "total_positions": 8_000_000,
    "utf8_incomplete_easy_positions": 4_664_439,
    "hangul_prefix_easy_positions": 4_602_889,
    "hangul_is_subset_of_utf8_incomplete": True,
}


def candidate_definition(name: str) -> dict[str, str]:
    parts = name.split("__")
    if len(parts) != 3 or name not in CANDIDATE_ORDER:
        raise ValueError(f"unknown conditional sensitivity candidate: {name}")
    route, components, operator = parts
    return {
        "route_policy": route,
        "components": components,
        "operator": operator,
    }


def _loss_vector(value: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.float32
        or array.shape != (count,)
        or not np.isfinite(array).all()
        or np.any(array < 0)
    ):
        raise ValueError("conditional sensitivity NLL vector differs")
    return np.ascontiguousarray(array)


def summarize_frozen_sensitivity(
    *,
    candidate_losses_nats: Mapping[str, np.ndarray],
    baseline_losses_nats: np.ndarray,
    document_indices: np.ndarray,
    route_rates: Mapping[str, float],
    eligible_sequence_fraction: float,
) -> dict[str, Any]:
    """Apply the locked paired quality-risk gate and pairwise selection rule."""

    if set(candidate_losses_nats) != set(CANDIDATE_ORDER):
        raise ValueError("conditional sensitivity candidate NLL keys differ")
    if set(route_rates) != set(ROUTE_ORDER):
        raise ValueError("conditional sensitivity route-rate keys differ")
    baseline = np.asarray(baseline_losses_nats)
    count = len(baseline)
    baseline = _loss_vector(baseline, count)
    documents = np.asarray(document_indices)
    if (
        documents.shape != (count,)
        or not np.issubdtype(documents.dtype, np.integer)
        or not 0 <= float(eligible_sequence_fraction) <= 1
    ):
        raise ValueError("conditional sensitivity document evidence differs")
    scale = TARGETS_PER_SEQUENCE * math.log(2.0)
    rows: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        values = _loss_vector(candidate_losses_nats[name], count)
        differences = values.astype(np.float64) - baseline.astype(np.float64)
        mean = float(differences.sum() / (count * scale))
        samples = one_seed_document_bootstrap(
            differences,
            documents,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
        )
        lower, median, upper = np.quantile(samples, [0.05, 0.5, 0.95])
        route = candidate_definition(name)["route_policy"]
        rate = float(route_rates[route])
        if not np.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("conditional sensitivity route rate differs")
        passes = {
            "mean_risk_margin": mean <= RISK_MARGIN_BPB,
            "document_upper_risk_margin": float(upper) <= RISK_MARGIN_BPB,
            "minimum_easy_rate": rate >= MINIMUM_EASY_RATE,
            "document_coverage": float(eligible_sequence_fraction) >= 0.95,
        }
        rows[name] = {
            **candidate_definition(name),
            "candidate_bpb": float(values.sum() / (count * scale)),
            "baseline_bpb": float(baseline.sum() / (count * scale)),
            "mean_difference_bpb": mean,
            "easy_position_rate": rate,
            "document_bootstrap": {
                "central_90_interval": {
                    "lower": float(lower),
                    "upper": float(upper),
                },
                "median_bpb": float(median),
                "one_sided_95_upper_bpb": float(upper),
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": BOOTSTRAP_SEED,
            },
            "passes": passes,
            "overall_pass": all(passes.values()),
        }

    selected_pair = None
    pair_rows: dict[str, Any] = {}
    for pair in PAIR_ORDER:
        candidates = [f"{route}__{pair}" for route in ROUTE_ORDER]
        pair_pass = all(rows[name]["overall_pass"] for name in candidates)
        pair_rows[pair] = {
            "candidate_order": candidates,
            "both_routes_pass": pair_pass,
        }
        if selected_pair is None and pair_pass:
            selected_pair = pair
    return {
        "rows": rows,
        "pairs": pair_rows,
        "selection": {
            "selected_pair": selected_pair,
            "actual_runtime_prototype_authorized": selected_pair is not None,
            "status": (
                "conditional_runtime_prototype_authorized"
                if selected_pair is not None
                else "conditional_branch_not_advanced_by_frozen_screen"
            ),
        },
        "interpretation": {
            "operator_component_feasibility_only": True,
            "hangul_specific_effect_identified": False,
            "trained_conditional_model_falsified_on_failure": False,
        },
        "thresholds": {
            "risk_margin_bpb": RISK_MARGIN_BPB,
            "minimum_easy_rate": MINIMUM_EASY_RATE,
            "minimum_document_coverage": 0.95,
        },
    }
