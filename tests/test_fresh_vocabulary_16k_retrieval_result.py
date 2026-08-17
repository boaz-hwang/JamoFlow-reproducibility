from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/fresh-vocabulary-16k-retrieval-actual-v1/summary.json"


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


def _result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def test_retrieval_result_is_self_consistent_and_primary_negative() -> None:
    result = _result()
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256")
    assert _canonical_sha256(unsigned) == recorded
    assert recorded == "3083986fc9fe3d2da8800143bd6d2aee89817206d24dcd51d31cd97e8967281b"
    assert result["status"] == "fail_16k_retrieval_actual_development"
    assert result["actual_retrieval"]["primary_gate"]["role"] == (
        "hybrid_retrieval_block_4"
    )
    assert result["actual_retrieval"]["primary_gate"]["overall_pass"] is False
    assert result["decision"]["korean_specific_disjoint_followup_authorized"] is False
    assert result["decision"]["diagnostic_role_fallback_allowed"] is False
    assert result["claim_boundary"]["publication_claim"] is False


def test_hybrid_free_pass_does_not_erase_controlled_failure() -> None:
    result = _result()
    actual = result["actual_retrieval"]
    hybrid = actual["comparisons"]["hybrid_retrieval_block_4"]
    controlled = hybrid["controlled_replay"]
    free = hybrid["free_running_utf8_greedy"]
    assert controlled["end_to_end_reduction"] == pytest.approx(0.05310067485592962)
    assert controlled["positive_prompt_count"] == 45
    assert controlled["paired_prompt_bootstrap_95_interval"]["lower"] > 0
    assert actual["primary_gate"]["by_mode"]["controlled_replay"][
        "overall_pass"
    ] is False
    assert free["end_to_end_reduction"] == pytest.approx(0.2624360614883763)
    assert free["positive_prompt_count"] == 61
    assert free["paired_prompt_bootstrap_95_interval"]["lower"] > 0.1
    assert actual["primary_gate"]["by_mode"]["free_running_utf8_greedy"][
        "overall_pass"
    ] is True


def test_outputs_and_independent_checkpoint_replay_are_exact() -> None:
    result = _result()
    assert result["independent_correctness"]["overall_pass"] is True
    assert result["free_output_evidence"]["strict_valid_trace_count"] == 1_280
    assert result["free_output_evidence"]["exact_across_roles_and_repetitions"] is True
    target = result["independent_correctness"]["target_cache_full"]
    assert target["pass"] is True
    assert target["comparisons"] == target["argmax_exact"]
    assert target["maximum_normalized_tolerance_ratio"] <= 1
