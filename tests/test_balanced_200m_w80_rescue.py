from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import balanced_200m_w80_core as module


def _nll_pair(delta_bpb: float) -> tuple[np.ndarray, np.ndarray]:
    reference = np.full(15_625, 100.0, dtype=np.float32)
    difference = np.float32(delta_bpb * 511 * math.log(2.0))
    candidate = np.ascontiguousarray(reference + difference)
    return reference, candidate


def _timing_pair(reduction: float) -> dict[str, np.ndarray]:
    shape = (
        len(module.TIMING_MODE_ORDER),
        module.TIMING_MEASURED_PROMPTS,
        module.TIMING_REPETITIONS,
        len(module.TIMING_ROLE_ORDER),
    )
    output: dict[str, np.ndarray] = {}
    for session_index, session in enumerate(module.TIMING_SESSION_ORDER):
        values = np.empty(shape, dtype=np.float64)
        for mode in range(len(module.TIMING_MODE_ORDER)):
            for prompt in range(module.TIMING_MEASURED_PROMPTS):
                reference = 100.0 + session_index + prompt / 100.0 + mode / 10.0
                values[mode, prompt, :, module.TIMING_ROLE_ORDER.index("c86")] = reference
                values[mode, prompt, :, module.TIMING_ROLE_ORDER.index("w80")] = reference * (1.0 - reduction)
        output[session] = values
    return output


def test_w80_matrix_has_exact_candidate_patch_width() -> None:
    boundaries = np.ones((3, 512), dtype=np.uint8)
    whitespace = np.zeros_like(boundaries)
    whitespace[:, 10::17] = 1
    matrix = module._w80_matrix(boundaries, whitespace)
    assert matrix.dtype == np.uint16
    assert matrix.shape == (3, 81)
    assert np.all(np.count_nonzero(matrix, axis=1) == 81)
    assert np.all(matrix.sum(axis=1) == 513)


def test_quality_requires_mean_and_block_upper() -> None:
    reference, candidate = _nll_pair(0.005)
    passed = module.summarize_quality(reference, candidate)
    assert passed["clauses"] == {
        "mean_delta_at_most_margin": True,
        "bootstrap_upper_at_most_margin": True,
    }
    assert passed["quality_gate_pass_pending_independent_replay"] is True
    assert passed["actual_timing_authorized"] is False

    reference, candidate = _nll_pair(0.012)
    failed = module.summarize_quality(reference, candidate)
    assert failed["clauses"]["mean_delta_at_most_margin"] is False
    assert failed["clauses"]["bootstrap_upper_at_most_margin"] is False
    assert failed["quality_gate_pass_pending_independent_replay"] is False


def test_quality_rejects_local_instability_even_when_mean_passes() -> None:
    reference = np.full(15_625, 100.0, dtype=np.float32)
    effects = np.full(15_625, -0.017365853658536587, dtype=np.float64)
    effects[: 64 * 80] = 0.06
    candidate = np.ascontiguousarray(
        reference
        + (effects * (511 * math.log(2.0))).astype(np.float32)
    )
    result = module.summarize_quality(reference, candidate)
    assert result["w80_minus_c86_bpb"] <= module.QUALITY_MARGIN_BPB
    assert result["block_bootstrap"]["upper"] > module.QUALITY_MARGIN_BPB
    assert result["quality_gate_pass_pending_independent_replay"] is False


def test_role_order_balances_every_fixed_cell_across_five_sessions() -> None:
    for prompt in range(module.TIMING_MEASURED_PROMPTS):
        for repetition in range(module.TIMING_REPETITIONS):
            for mode in range(len(module.TIMING_MODE_ORDER)):
                first = [
                    module.timing_role_order(session, prompt, repetition, mode)[0]
                    for session in range(len(module.TIMING_SESSION_ORDER))
                ]
                assert sum(first) in {2, 3}


def test_actual_gate_requires_both_modes_to_exceed_compact_reference() -> None:
    correctness = {
        session: {"overall_pass": True}
        for session in module.TIMING_SESSION_ORDER
    }
    passed = module.summarize_actual_timing(_timing_pair(0.04), correctness)
    assert passed["overall_actual_primary_pass"] is True
    assert passed["strong_scale_amplification_support"] is True
    assert all(
        passed["by_mode"][mode]["actual_primary_pass"]
        for mode in module.TIMING_MODE_ORDER
    )

    failed = module.summarize_actual_timing(_timing_pair(0.02), correctness)
    assert failed["overall_actual_primary_pass"] is False
    assert all(
        failed["by_mode"][mode]["clauses"][
            "point_exceeds_compact_reference"
        ]
        is False
        for mode in module.TIMING_MODE_ORDER
    )


def test_one_bad_session_or_correctness_fails_actual_gate() -> None:
    arrays = _timing_pair(0.04)
    bad = arrays["session-4"].copy()
    candidate = module.TIMING_ROLE_ORDER.index("w80")
    reference = module.TIMING_ROLE_ORDER.index("c86")
    bad[..., candidate] = bad[..., reference] * 1.03
    arrays["session-4"] = bad
    correctness = {
        session: {"overall_pass": True}
        for session in module.TIMING_SESSION_ORDER
    }
    result = module.summarize_actual_timing(arrays, correctness)
    assert result["overall_actual_primary_pass"] is False
    assert all(
        result["by_mode"][mode]["clauses"]["all_sessions_positive"] is False
        for mode in module.TIMING_MODE_ORDER
    )

    arrays = _timing_pair(0.04)
    correctness["session-2"] = {"overall_pass": False}
    result = module.summarize_actual_timing(arrays, correctness)
    assert result["overall_actual_primary_pass"] is False
    assert all(
        result["by_mode"][mode]["clauses"]["correctness"] is False
        for mode in module.TIMING_MODE_ORDER
    )


def test_protocol_is_single_candidate_and_timing_is_quality_conditional() -> None:
    contract = module.quality_contract()
    timing = module.timing_contract()
    assert module.CANDIDATE_PATCH_COUNT == 80
    assert contract["independent_full_checkpoint_replay_required"] is True
    assert contract["historical_test_or_final_metric_used"] is False
    assert timing["strong_amplification_requires_lower_over_compact_point"] is True
    assert set(module.COMPACT_REFERENCE_REDUCTION) == set(module.TIMING_MODE_ORDER)
    text = (ROOT / "scripts/benchmark_balanced_200m_w80_actual.py").read_text(
        encoding="utf-8"
    )
    assert 'verification.get("actual_timing_authorized") is not True' in text
    assert "w82" not in text.lower()
    assert "w84" not in text.lower()


def test_timing_input_shape_and_session_set_are_fail_closed() -> None:
    correctness = {
        session: {"overall_pass": True}
        for session in module.TIMING_SESSION_ORDER
    }
    arrays = _timing_pair(0.04)
    arrays.pop("session-4")
    with pytest.raises(ValueError, match="session set"):
        module.summarize_actual_timing(arrays, correctness)
