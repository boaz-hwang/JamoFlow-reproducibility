from __future__ import annotations

import unittest

import numpy as np

from same2k_opportunity import (
    BPE_ROLE,
    LONGEST_MATCH_ROLE,
    MEASURED_CASES,
    MINIMUM_TOKEN_ROLE,
    ROLES,
    SCORED_UNIGRAM_ROLE,
    WARMUP_CASES,
    canonical_text_pieces,
    metrics_identity_without_timing,
    opportunity_decision,
)


def _row(token_count: int, continuation_count: int) -> dict[str, object]:
    return {
        "calibration_token_count": token_count,
        "continuation_token_counts": [continuation_count]
        * (WARMUP_CASES + MEASURED_CASES),
    }


class Same2kOpportunityTest(unittest.TestCase):
    def test_canonical_pieces_preserve_newlines_and_utf8_tail(self) -> None:
        raw = "첫 줄\n둘째 줄\n끝".encode("utf-8") + b"\xed\x95"
        pieces = canonical_text_pieces(raw)
        self.assertEqual(b"".join(piece for _, piece in pieces), raw[:-2])
        self.assertTrue(all(text.encode("utf-8") == piece for text, piece in pieces))

    def test_gate_requires_both_calibration_and_continuation_reduction(self) -> None:
        values = {
            BPE_ROLE: _row(1000, 100),
            SCORED_UNIGRAM_ROLE: _row(890, 95),
            LONGEST_MATCH_ROLE: _row(950, 89),
            MINIMUM_TOKEN_ROLE: _row(890, 89),
        }
        result = opportunity_decision(values)
        self.assertEqual(result["eligible_generic_roles"], [MINIMUM_TOKEN_ROLE])
        self.assertFalse(
            result["comparisons"][SCORED_UNIGRAM_ROLE][
                "passes_token_only_opportunity"
            ]
        )
        self.assertFalse(
            result["comparisons"][LONGEST_MATCH_ROLE][
                "passes_token_only_opportunity"
            ]
        )
        self.assertEqual(
            result["next_action"], "one_seed_quality_training_for_eligible_generic_roles"
        )

    def test_no_passing_role_routes_to_length_gain_vocabulary(self) -> None:
        values = {role: _row(1000 - index * 10, 100 - index) for index, role in enumerate(ROLES)}
        result = opportunity_decision(values)
        self.assertEqual(result["eligible_generic_roles"], [])
        self.assertEqual(
            result["next_action"],
            "construct_length_gain_vocabulary_before_any_model_training",
        )

    def test_replay_identity_excludes_only_wall_clock_diagnostics(self) -> None:
        value = {
            "calibration_token_count": 100,
            "encode_seconds": [1.0, 2.0],
            "encode_median_megabytes_per_second": 4.0,
        }
        second = {**value, "encode_seconds": [3.0, 4.0], "encode_median_megabytes_per_second": 2.0}
        self.assertEqual(
            metrics_identity_without_timing(value),
            metrics_identity_without_timing(second),
        )
        second["calibration_token_count"] = 99
        self.assertNotEqual(
            metrics_identity_without_timing(value),
            metrics_identity_without_timing(second),
        )

    def test_replay_identity_normalizes_json_tuple_transport(self) -> None:
        first = {
            "counts": (1, 2),
            "encode_seconds": (1.0,),
            "encode_median_megabytes_per_second": 2.0,
        }
        second = {
            "counts": [1, 2],
            "encode_seconds": [3.0],
            "encode_median_megabytes_per_second": 4.0,
        }
        self.assertEqual(
            metrics_identity_without_timing(first),
            metrics_identity_without_timing(second),
        )


if __name__ == "__main__":
    unittest.main()
