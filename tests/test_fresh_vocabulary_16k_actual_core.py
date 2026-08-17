from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from fresh_vocabulary_16k_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLE_PERMUTATIONS,
    ROLES,
    TIMING_COMPONENTS,
    balanced_role_order,
    summarize_actual_preflight,
)


def _correctness() -> dict[str, dict[str, dict[str, object]]]:
    return {
        role: {
            mode: {
                "cases": MEASURED_CASES,
                "comparisons": 1_000,
                "argmax_comparisons": 1_000,
                "argmax_exact": 1_000,
                "trace_contract_exact": True,
                "maximum_normalized_tolerance_ratio": 0.25,
                "pass": True,
            }
            for mode in MODES
        }
        for role in ROLES
    }


def _inputs(
    *,
    candidate_ms: tuple[float, float] = (8.0, 8.0),
    baseline_ms: tuple[float, float] = (10.0, 10.0),
    frontier_ms: tuple[float, float] = (9.0, 9.0),
) -> dict[str, object]:
    shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    end_to_end = np.empty(shape, dtype=np.float64)
    for mode_index in range(len(MODES)):
        end_to_end[mode_index, ..., ROLES.index("candidate_16k")] = candidate_ms[
            mode_index
        ]
        end_to_end[mode_index, ..., ROLES.index("baseline_2k")] = baseline_ms[
            mode_index
        ]
        end_to_end[mode_index, ..., ROLES.index("frontier_8k")] = frontier_ms[
            mode_index
        ]
    timing = {
        name: end_to_end.copy() if name == "end_to_end_ms" else end_to_end * 0.5
        for name in TIMING_COMPONENTS
    }
    return {
        "timing": timing,
        "output_token_count": np.full(shape, 32, dtype=np.int16),
        "output_raw_byte_count": np.full(shape, CONTINUATION_BYTES, dtype=np.int16),
        "correctness": _correctness(),
        "maximum_output_bytes_by_role": {
            role: CONTINUATION_BYTES + 20 for role in ROLES
        },
    }


def test_three_role_schedule_cycles_all_permutations_without_material_imbalance() -> (
    None
):
    counts: Counter[tuple[int, ...]] = Counter()
    positions = [[0] * len(ROLES) for _ in ROLES]
    for case_index in range(MEASURED_CASES):
        for repetition in range(REPETITIONS):
            for mode_index in range(len(MODES)):
                order = balanced_role_order(case_index, repetition, mode_index)
                assert order in ROLE_PERMUTATIONS
                counts[order] += 1
                for position, role_index in enumerate(order):
                    positions[position][role_index] += 1
    assert set(counts) == set(ROLE_PERMUTATIONS)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(max(row) - min(row) <= 1 for row in positions)


def test_primary_gate_alone_authorizes_multiseed() -> None:
    result = summarize_actual_preflight(**_inputs(frontier_ms=(7.0, 7.0)))
    assert result["primary_gate"]["overall_pass"] is True
    assert result["secondary_frontier_diagnostic"]["overall_pass"] is False
    assert result["multiseed_confirmation_authorized"] is True
    assert result["incremental_16k_frontier_supported"] is False


def test_secondary_pair_cannot_rescue_failed_primary_gate() -> None:
    result = summarize_actual_preflight(
        **_inputs(candidate_ms=(8.0, 9.5), frontier_ms=(9.0, 10.0))
    )
    assert result["pairs"]["candidate_vs_8k"]["overall_pass"] is True
    assert result["pairs"]["candidate_vs_2k"]["overall_pass"] is False
    assert result["multiseed_confirmation_authorized"] is False
    assert result["status"] == "fail_16k_trained_actual_e2e_preflight"


def test_gate_rejects_bad_correctness_or_output_counts() -> None:
    inputs = _inputs()
    inputs["correctness"]["candidate_16k"]["free_running_utf8_greedy"][
        "trace_contract_exact"
    ] = False
    result = summarize_actual_preflight(**inputs)
    assert result["multiseed_confirmation_authorized"] is False

    inputs = _inputs()
    inputs["output_raw_byte_count"][0, 0, 0, 0] = CONTINUATION_BYTES - 1
    with pytest.raises(ValueError, match="count array"):
        summarize_actual_preflight(**inputs)


def test_secondary_requires_positive_interval_and_prompt_majority() -> None:
    inputs = _inputs(frontier_ms=(8.0, 8.0))
    result = summarize_actual_preflight(**inputs)
    assert result["secondary_frontier_diagnostic"]["overall_pass"] is False
    for mode in MODES:
        assert (
            result["pairs"]["candidate_vs_8k"][mode]["passes"]["point_positive"]
            is False
        )
