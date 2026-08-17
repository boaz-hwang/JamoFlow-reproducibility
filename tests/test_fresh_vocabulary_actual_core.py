from __future__ import annotations

import numpy as np
import pytest
from fresh_vocabulary_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    WARMUP_CASES,
    balanced_role_order,
    summarize_actual_preflight,
    validate_strict_token_replay,
)

from jamoflow.utf8 import compile_strict_utf8_token_transitions


def _correctness() -> dict[str, dict[str, dict[str, object]]]:
    return {
        role: {
            mode: {
                "cases": WARMUP_CASES,
                "comparisons": 20,
                "argmax_comparisons": 20,
                "argmax_exact": 20,
                "maximum_normalized_tolerance_ratio": 0.25,
                "pass": True,
            }
            for mode in MODES
        }
        for role in ROLES
    }


def _inputs(candidate_ms: tuple[float, float]) -> dict[str, object]:
    shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    end_to_end = np.empty(shape, dtype=np.float64)
    for mode_index, value in enumerate(candidate_ms):
        end_to_end[mode_index, ..., ROLES.index("candidate")] = value
        end_to_end[mode_index, ..., ROLES.index("reference")] = 10.0
    timing = {
        name: end_to_end.copy() if name == "end_to_end_ms" else end_to_end * 0.5
        for name in TIMING_COMPONENTS
    }
    tokens = np.full(shape, 32, dtype=np.int16)
    raw = np.full(shape, CONTINUATION_BYTES, dtype=np.int16)
    return {
        "timing": timing,
        "output_token_count": tokens,
        "output_raw_byte_count": raw,
        "correctness": _correctness(),
        "maximum_output_bytes_by_role": {
            "candidate": CONTINUATION_BYTES + 20,
            "reference": CONTINUATION_BYTES + 20,
        },
    }


def test_balanced_schedule_is_exact() -> None:
    first = [0, 0]
    for case_index in range(MEASURED_CASES):
        for repetition in range(REPETITIONS):
            for mode_index in range(len(MODES)):
                order = balanced_role_order(case_index, repetition, mode_index)
                assert sorted(order) == [0, 1]
                first[order[0]] += 1
    assert first[0] == first[1]


def test_actual_gate_requires_both_modes() -> None:
    passed = summarize_actual_preflight(**_inputs((8.0, 8.5)))
    assert passed["gate"]["overall_pass"] is True
    assert passed["multiseed_confirmation_authorized"] is True

    failed = summarize_actual_preflight(**_inputs((8.0, 9.5)))
    assert failed["gate"]["overall_pass"] is False
    assert failed["modes"]["controlled_replay"]["overall_pass"] is True
    assert failed["modes"]["free_running_utf8_greedy"]["overall_pass"] is False


def test_actual_gate_rejects_bad_correctness_and_counts() -> None:
    inputs = _inputs((8.0, 8.0))
    inputs["correctness"]["candidate"]["controlled_replay"]["argmax_exact"] = 19
    assert summarize_actual_preflight(**inputs)["gate"]["overall_pass"] is False

    inputs = _inputs((8.0, 8.0))
    inputs["output_raw_byte_count"][0, 0, 0, 0] = CONTINUATION_BYTES - 1
    with pytest.raises(ValueError, match="count array"):
        summarize_actual_preflight(**inputs)


def test_strict_token_trace_is_independently_replayed() -> None:
    pieces = tuple(bytes([value]) for value in range(256)) + ("한".encode(),)
    transitions = compile_strict_utf8_token_transitions(pieces)
    raw, state = validate_strict_token_replay(
        [256] * 43,
        token_bytes=pieces,
        next_state_indices=transitions.next_state_indices,
    )
    assert state == 0
    assert raw.decode("utf-8") == "한" * 43
    with pytest.raises(ValueError, match="violates strict UTF-8"):
        validate_strict_token_replay(
            [255] * 128,
            token_bytes=pieces,
            next_state_indices=transitions.next_state_indices,
        )
