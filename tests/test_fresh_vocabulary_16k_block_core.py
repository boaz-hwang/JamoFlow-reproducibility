from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from fresh_vocabulary_16k_block_core import (
    BLOCK_SIZE_BY_ROLE,
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLE_ORDERS,
    ROLES,
    TIMING_COMPONENTS,
    balanced_role_order,
    summarize_upper_bound,
)


def _correctness() -> dict[str, dict[str, dict[str, object]]]:
    return {
        role: {
            mode: {
                "cases": MEASURED_CASES,
                "comparisons": 1_000,
                "argmax_comparisons": 1_000,
                "argmax_exact": 1_000,
                "decode_calls": 250,
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
    baseline_ms: tuple[float, float] = (10.0, 10.0),
    block4_ms: tuple[float, float] = (6.0, 6.0),
) -> dict[str, object]:
    shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    values = np.empty(shape, dtype=np.float64)
    by_role = {
        "baseline_ar": baseline_ms,
        "perfect_block_2": (7.0, 7.0),
        "perfect_block_4": block4_ms,
        "perfect_block_8": (5.0, 5.0),
    }
    for role, mode_values in by_role.items():
        for mode_index, value in enumerate(mode_values):
            values[mode_index, ..., ROLES.index(role)] = value
    tokens = np.full(shape, 25, dtype=np.int16)
    calls = np.empty(shape, dtype=np.int16)
    for role_index, role in enumerate(ROLES):
        calls[..., role_index] = (
            tokens[..., role_index]
            if role == "baseline_ar"
            else 1
            + np.ceil((tokens[..., role_index] - 1) / BLOCK_SIZE_BY_ROLE[role]).astype(
                np.int16
            )
        )
    return {
        "timing": {
            name: values.copy() if name == "end_to_end_ms" else values * 0.5
            for name in TIMING_COMPONENTS
        },
        "output_token_count": tokens,
        "output_raw_byte_count": np.full(
            shape,
            CONTINUATION_BYTES,
            dtype=np.int16,
        ),
        "target_forward_calls": calls,
        "correctness": _correctness(),
        "maximum_output_bytes": CONTINUATION_BYTES + 20,
    }


def test_four_role_schedule_is_exactly_balanced() -> None:
    counts: Counter[tuple[int, ...]] = Counter()
    positions = [[0] * len(ROLES) for _ in ROLES]
    for case_index in range(MEASURED_CASES):
        for repetition in range(REPETITIONS):
            for mode_index in range(len(MODES)):
                order = balanced_role_order(case_index, repetition, mode_index)
                counts[order] += 1
                for position, role_index in enumerate(order):
                    positions[position][role_index] += 1
    assert set(counts) == set(ROLE_ORDERS)
    assert set(counts.values()) == {80}
    assert all(set(row) == {160} for row in positions)


def test_fixed_block4_primary_passes_and_only_authorizes_draft_fail_fast() -> None:
    result = summarize_upper_bound(**_inputs())
    assert result["primary_gate"]["overall_pass"] is True
    assert result["learned_draft_prototype_authorized"] is True
    assert result["actual_speculative_efficiency_claimed"] is False


def test_diagnostic_block8_cannot_rescue_failed_block4() -> None:
    result = summarize_upper_bound(**_inputs(block4_ms=(7.5, 7.5)))
    assert result["comparisons"]["perfect_block_8"]["controlled_replay"][
        "end_to_end_reduction"
    ] == pytest.approx(0.5)
    assert result["primary_gate"]["overall_pass"] is False
    assert result["learned_draft_prototype_authorized"] is False


def test_target_call_accounting_and_correctness_are_fail_closed() -> None:
    inputs = _inputs()
    inputs["target_forward_calls"][0, 0, 0, 2] += 1
    with pytest.raises(ValueError, match="call accounting"):
        summarize_upper_bound(**inputs)

    inputs = _inputs()
    inputs["correctness"]["perfect_block_4"]["free_running_utf8_greedy"][
        "trace_contract_exact"
    ] = False
    result = summarize_upper_bound(**inputs)
    assert result["primary_gate"]["overall_pass"] is False


def test_output_counts_must_match_baseline_exactly() -> None:
    inputs = _inputs()
    inputs["output_token_count"][1, 0, 0, 3] += 1
    with pytest.raises(ValueError, match="token counts"):
        summarize_upper_bound(**inputs)
