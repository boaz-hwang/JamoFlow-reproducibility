"""Pure aggregation contract for the exact W72 speculative runtime preflight."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


PROTOCOL_ID = "jamoflow-speculative-w72-preflight-v1"
MODES = ("baseline_ar", "speculative_hangul")
PROMPT_COUNT = 128
REPETITIONS = 3
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260831
COUNTER_KEYS = (
    "emitted_bytes",
    "sequential_target_calls",
    "target_block_calls",
    "draft_head_calls",
    "first_draft_accepts",
    "complete_pair_accepts",
    "first_mismatches",
    "second_mismatches",
    "cropped_speculative_bytes",
    "correction_bytes",
    "bonus_bytes",
    "retry_block_calls",
    "retry_third_accepts",
    "retry_third_mismatches",
)


def summarize_speculative_preflight(
    *,
    timings_ms: np.ndarray,
    output_lengths: np.ndarray,
    speculative_counters: np.ndarray,
    correctness: Mapping[str, Any],
    minimum_point_reduction: float,
    minimum_lower_bound: float,
    minimum_positive_prompts: int,
) -> dict[str, Any]:
    timings = np.asarray(timings_ms)
    lengths = np.asarray(output_lengths)
    counters = np.asarray(speculative_counters)
    if (
        timings.dtype != np.float64
        or timings.shape != (PROMPT_COUNT, REPETITIONS, len(MODES))
        or not np.all(np.isfinite(timings))
        or np.any(timings <= 0)
    ):
        raise ValueError("speculative timing array differs")
    if (
        lengths.dtype != np.int64
        or lengths.shape != (PROMPT_COUNT,)
        or np.any((lengths < 128) | (lengths > 131))
    ):
        raise ValueError("speculative output lengths differ")
    if (
        counters.dtype != np.int64
        or counters.shape != (PROMPT_COUNT, len(COUNTER_KEYS))
        or np.any(counters < 0)
    ):
        raise ValueError("speculative counter array differs")
    expected_correctness_keys = {
        "all_outputs_exact",
        "cache_comparisons",
        "output_comparisons",
        "output_hash_root_sha256",
    }
    if (
        set(correctness) != expected_correctness_keys
        or correctness["all_outputs_exact"] is not True
        or int(correctness["output_comparisons"]) != PROMPT_COUNT
        or int(correctness["cache_comparisons"]) != PROMPT_COUNT
        or not isinstance(correctness["output_hash_root_sha256"], str)
        or len(correctness["output_hash_root_sha256"]) != 64
    ):
        raise ValueError("speculative correctness evidence differs")
    if (
        not 0 <= minimum_lower_bound <= minimum_point_reduction <= 1
        or not 1 <= minimum_positive_prompts <= PROMPT_COUNT
    ):
        raise ValueError("speculative gate threshold differs")

    index = {key: COUNTER_KEYS.index(key) for key in COUNTER_KEYS}
    if not np.array_equal(counters[:, index["emitted_bytes"]], lengths):
        raise ValueError("speculative emitted-byte counters differ")
    if not np.array_equal(
        counters[:, index["first_draft_accepts"]]
        + counters[:, index["first_mismatches"]],
        counters[:, index["draft_head_calls"]],
    ):
        raise ValueError("speculative first-accept counters differ")
    if not np.array_equal(
        counters[:, index["complete_pair_accepts"]]
        + counters[:, index["second_mismatches"]],
        counters[:, index["first_draft_accepts"]],
    ):
        raise ValueError("speculative pair-accept counters differ")
    if np.any(
        counters[:, index["retry_third_accepts"]]
        + counters[:, index["retry_third_mismatches"]]
        > counters[:, index["retry_block_calls"]]
    ):
        raise ValueError("speculative retry counters differ")

    cells = np.median(timings, axis=1)
    baseline = float(np.median(cells[:, 0]))
    speculative = float(np.median(cells[:, 1]))
    point = 1.0 - speculative / baseline
    prompt_reductions = 1.0 - cells[:, 1] / cells[:, 0]
    positive = int(np.count_nonzero(prompt_reductions > 0))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for draw in range(BOOTSTRAP_REPETITIONS):
        rows = rng.integers(0, PROMPT_COUNT, size=PROMPT_COUNT)
        draws[draw] = 1.0 - float(np.median(cells[rows, 1])) / float(
            np.median(cells[rows, 0])
        )
    lower, upper = np.quantile(draws, (0.025, 0.975))

    totals = {
        key: int(counters[:, column].sum())
        for column, key in enumerate(COUNTER_KEYS)
    }
    baseline_target_calls = int(np.sum(lengths - 1))
    realized_target_invocations = (
        totals["sequential_target_calls"] + totals["target_block_calls"]
    )
    gates = {
        "exact_output_and_cache": True,
        "point_reduction": point >= minimum_point_reduction,
        "bootstrap_lower_bound": float(lower) >= minimum_lower_bound,
        "prompt_direction": positive >= minimum_positive_prompts,
    }
    return {
        "end_to_end": {
            "baseline_ar_median_ms": baseline,
            "speculative_hangul_median_ms": speculative,
            "reduction": point,
            "prompt_bootstrap_95_interval": {
                "lower": float(lower),
                "upper": float(upper),
            },
            "positive_prompt_count": positive,
            "prompt_count": PROMPT_COUNT,
        },
        "mechanism": {
            "counter_totals": totals,
            "baseline_target_calls": baseline_target_calls,
            "speculative_target_invocations": realized_target_invocations,
            "target_invocation_reduction": 1.0
            - realized_target_invocations / baseline_target_calls,
            "first_acceptance": totals["first_draft_accepts"]
            / totals["draft_head_calls"],
            "complete_pair_acceptance": totals["complete_pair_accepts"]
            / totals["draft_head_calls"],
            "retry_third_acceptance": totals["retry_third_accepts"]
            / max(1, totals["retry_block_calls"]),
        },
        "correctness": dict(correctness),
        "bootstrap": {
            "unit": "calibration prompt after within-prompt repetition median",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
        },
        "gates": {
            "thresholds": {
                "minimum_point_reduction": minimum_point_reduction,
                "minimum_bootstrap_lower_bound": minimum_lower_bound,
                "minimum_positive_prompts": minimum_positive_prompts,
            },
            "passes": gates,
            "multi_seed_generic_comparator_authorized": all(gates.values()),
        },
    }
