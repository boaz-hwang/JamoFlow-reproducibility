from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/fresh-vocabulary-16k-target-block-v1/summary.json"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_target_block_result_is_self_consistent_and_narrowly_scoped() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256")
    assert _canonical_sha256(unsigned) == recorded
    assert recorded == "ad7db5d5699735eb27d798c1b716060cb5670a5b721eec1e5f82154d8f81185e"
    assert result["status"] == "pass_16k_target_block_upper_bound"
    assert result["upper_bound"]["primary_gate"]["role"] == "perfect_block_4"
    assert result["upper_bound"]["primary_gate"]["overall_pass"] is True
    assert result["upper_bound"]["correctness"]["overall_pass"] is True
    assert result["decision"][
        "learned_same_tokenizer_draft_fail_fast_authorized"
    ] is True
    assert result["decision"]["actual_speculative_efficiency_claimed"] is False
    assert result["claim_boundary"]["perfect_draft_cost_excluded"] is True
    assert result["claim_boundary"]["publication_claim"] is False


def test_primary_block4_passes_both_fixed_mode_gates_without_fallback() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    upper = result["upper_bound"]
    assert result["decision"]["diagnostic_block_size_fallback_allowed"] is False
    for mode, expected in (
        ("controlled_replay", 0.6392713447427524),
        ("free_running_utf8_greedy", 0.6568336146866292),
    ):
        row = upper["comparisons"]["perfect_block_4"][mode]
        assert row["end_to_end_reduction"] == pytest.approx(expected)
        assert row["paired_prompt_bootstrap_95_interval"]["lower"] > 0
        assert row["positive_prompt_count"] == 64
        assert upper["primary_gate"]["by_mode"][mode]["overall_pass"] is True


def test_all_free_outputs_and_independent_logits_are_exact() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert result["free_output_evidence"]["strict_valid_trace_count"] == 1_280
    assert result["free_output_evidence"][
        "exact_across_roles_and_repetitions"
    ] is True
    correctness = result["upper_bound"]["correctness"][
        "independent_measured_case_checkpoint_replay"
    ]
    for role in correctness.values():
        for mode in role.values():
            assert mode["pass"] is True
            assert mode["argmax_exact"] == mode["argmax_comparisons"]
            assert mode["maximum_normalized_tolerance_ratio"] <= 1
