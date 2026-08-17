from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/fresh-vocabulary-16k-retrieval-mechanism-v2/summary.json"


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


def test_mechanism_result_is_self_consistent_and_negative() -> None:
    result = _result()
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256")
    assert _canonical_sha256(unsigned) == recorded
    assert recorded == "73044422788ee992bc76c31e2ac25965d903d8846137cc3fa0b00e2f66fd601c"
    assert result["status"] == "fail_hangul_boundary_mechanism_screen"
    assert result["event_stream"]["aggregate_replay_matches_timing_counters"] is True
    assert result["event_stream"]["event_count"] == 9_335
    assert result["mechanism"]["decision"] == {
        "disjoint_actual_design_authorized": False,
        "efficiency_claim": False,
        "hangul_boundary_router_hypothesis_supported": False,
    }


def test_primary_hangul_boundary_contrast_fails_in_the_opposite_direction() -> None:
    result = _result()
    row = result["mechanism"]["primary_hangul_boundary_contrast"]
    assert row["cycle_counts"] == {
        "after_whitespace": 101,
        "within_hangul_eojeol": 244,
    }
    assert row["paired_case_count"] == 13
    assert row["paired_case_mean_difference"] == pytest.approx(-0.2461538461538461)
    assert row["accepted_tokens_per_cycle"]["within_hangul_eojeol"] == pytest.approx(
        1.4057377049180328
    )
    assert row["accepted_tokens_per_cycle"]["after_whitespace"] == pytest.approx(
        1.7524752475247525
    )
    assert row["gate"]["overall_pass"] is False


def test_secondary_rows_are_descriptive_and_do_not_authorize_fallback() -> None:
    result = _result()
    rows = result["mechanism"]["aggregate_rows"]
    missing = rows[
        "free_running_utf8_greedy/hybrid_retrieval_block_4/none/within_hangul_eojeol"
    ]
    assert missing["cycles"] == 508
    assert missing["proposal_cycles"] == 0
    assert result["claim_boundary"]["pass_authorizes_only_disjoint_design"] is True
    assert result["mechanism"]["decision"]["disjoint_actual_design_authorized"] is False
