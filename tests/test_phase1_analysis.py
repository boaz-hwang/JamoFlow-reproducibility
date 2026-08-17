import math
import unittest

import numpy as np

from jamoflow.phase1_analysis import (
    aggregate_numeric_mappings,
    boundary_unicode_diagnostics,
    hierarchical_paired_bootstrap,
    nearest_boundary_displacement,
    paired_t_interval,
)


class Phase1AnalysisTests(unittest.TestCase):
    def test_paired_t_interval_uses_seed_level_variance(self) -> None:
        interval = paired_t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_half_width = 2.776445 * math.sqrt(2.5) / math.sqrt(5)

        self.assertEqual(interval.count, 5)
        self.assertAlmostEqual(interval.mean, 3.0)
        self.assertAlmostEqual(interval.lower, 3.0 - expected_half_width)
        self.assertAlmostEqual(interval.upper, 3.0 + expected_half_width)

    def test_hierarchical_bootstrap_preserves_constant_paired_effect(self) -> None:
        # One nat over ten predicted bytes has the same BPB effect for every
        # sequence and seed, so every bootstrap replicate must be identical.
        values = [np.ones(7), np.ones(7), np.ones(7)]
        interval = hierarchical_paired_bootstrap(
            values,
            targets_per_sequence=10,
            repetitions=200,
            seed=7,
            chunk_size=31,
        )
        expected = 1 / (10 * math.log(2))

        self.assertAlmostEqual(interval.mean, expected)
        self.assertAlmostEqual(interval.lower, expected)
        self.assertAlmostEqual(interval.upper, expected)
        self.assertEqual(
            interval.resampling_design,
            "crossed seeds x shared test sequences",
        )

    def test_hierarchical_bootstrap_preserves_shared_sequence_draws(self) -> None:
        values = np.asarray([-math.log(2), math.log(2)])
        interval = hierarchical_paired_bootstrap(
            [values, values.copy()],
            targets_per_sequence=1,
            repetitions=200,
            seed=19,
            chunk_size=17,
        )
        self.assertGreaterEqual(interval.lower, -1.0)
        self.assertLessEqual(interval.upper, 1.0)

    def test_hierarchical_bootstrap_rejects_different_test_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "same sequences"):
            hierarchical_paired_bootstrap(
                [np.zeros(4), np.zeros(5)],
                targets_per_sequence=1,
                repetitions=10,
            )

    def test_numeric_mapping_aggregation_uses_common_numeric_fields(self) -> None:
        summary = aggregate_numeric_mappings(
            [
                {"count": 2, "rate": 0.25, "label": "a"},
                {"count": 4, "rate": 0.75, "extra": 1},
            ]
        )

        self.assertEqual(summary["count"]["mean"], 3.0)
        self.assertEqual(summary["rate"]["values"], [0.25, 0.75])
        self.assertNotIn("label", summary)
        self.assertNotIn("extra", summary)

    def test_unicode_diagnostics_distinguish_hangul_and_cjk_interiors(self) -> None:
        # Data patches begin once inside each three-byte codepoint and once at A.
        data = "한中A".encode("utf-8")
        lengths = np.asarray([[1, 1, 3, 3]], dtype=np.uint16)
        diagnostics = boundary_unicode_diagnostics(
            lengths,
            data,
            sequence_length=len(data),
        )

        self.assertEqual(
            diagnostics["inside_precomposed_hangul_syllable_count"], 1
        )
        self.assertEqual(diagnostics["inside_cjk_ideograph_count"], 1)
        self.assertEqual(diagnostics["inside_other_codepoint_count"], 0)

    def test_nearest_boundary_displacement_is_symmetric(self) -> None:
        first = np.asarray([[1, 2, 3, 3]], dtype=np.uint16)
        second = np.asarray([[1, 3, 2, 3]], dtype=np.uint16)
        result = nearest_boundary_displacement(first, second)

        self.assertEqual(result["symmetric_boundary_observations"], 4)
        self.assertAlmostEqual(result["exact_match_rate"], 0.5)
        self.assertAlmostEqual(result["mean_nearest_displacement_bytes"], 0.5)


if __name__ == "__main__":
    unittest.main()
