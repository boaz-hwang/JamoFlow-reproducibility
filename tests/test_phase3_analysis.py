from __future__ import annotations

import unittest

import numpy as np

from jamoflow.phase3_analysis import (
    empirical_nonnegative_bootstrap_tail,
    hierarchical_paired_bootstrap_estimates,
    holm_step_down_adjusted_values,
    paired_seed_lower_t_pvalue,
    phase3_test_strata,
    student_t_cdf,
)
from jamoflow.utf8 import prefix_boundary_mask


class Phase3AnalysisTests(unittest.TestCase):
    def test_strata_partition_preregistered_axes(self) -> None:
        chunks = [
            ("가" * 160 + "\n").encode("utf-8"),
            ("한글 English " * 30).encode("utf-8"),
            ("ASCII only words " * 30).encode("utf-8"),
        ]
        data = b"".join(chunk[:512].ljust(512, b" ") for chunk in chunks)
        boundaries = np.frombuffer(
            bytes(prefix_boundary_mask(data)[:-1]), dtype=np.uint8
        ).reshape(-1, 512)
        strata, metadata = phase3_test_strata(data, boundaries)
        hangul_partition = sum(
            strata[name].selected.astype(np.int8)
            for name in (
                "hangul_byte_fraction_lt_25",
                "hangul_byte_fraction_25_to_75",
                "hangul_byte_fraction_ge_75",
            )
        )
        whitespace_partition = sum(
            strata[f"whitespace_rate_t{index}"].selected.astype(np.int8)
            for index in range(1, 4)
        )
        self.assertTrue(np.all(hangul_partition == 1))
        self.assertTrue(np.all(whitespace_partition == 1))
        self.assertEqual(metadata["sequence_count"], 3)
        self.assertTrue(strata["ascii_latin_present"].selected[1])
        self.assertFalse(strata["ascii_latin_present"].selected[0])
        self.assertTrue(strata["newline_present"].selected[0])

    def test_holm_adjustment_is_monotonic_in_sorted_order(self) -> None:
        adjusted = holm_step_down_adjusted_values(
            {"a": 0.01, "b": 0.04, "c": 0.03}
        )
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_student_t_cdf_matches_tabulated_critical_values(self) -> None:
        self.assertAlmostEqual(student_t_cdf(0.0, 4), 0.5, places=12)
        self.assertAlmostEqual(student_t_cdf(-2.131847, 4), 0.05, places=6)
        self.assertAlmostEqual(student_t_cdf(-2.776445, 4), 0.025, places=6)
        self.assertAlmostEqual(
            student_t_cdf(2.776445, 4),
            0.975,
            places=6,
        )

    def test_paired_seed_lower_t_pvalue_handles_constant_effects(self) -> None:
        self.assertEqual(paired_seed_lower_t_pvalue([-1.0] * 5), 0.0)
        self.assertEqual(paired_seed_lower_t_pvalue([1.0] * 5), 1.0)
        self.assertEqual(paired_seed_lower_t_pvalue([0.0] * 5), 0.5)

    def test_empirical_nonnegative_tail_uses_add_one(self) -> None:
        self.assertEqual(
            empirical_nonnegative_bootstrap_tail([-2.0, -1.0, -0.5]),
            0.25,
        )
        self.assertEqual(
            empirical_nonnegative_bootstrap_tail([-1.0, 0.1, 0.2]),
            0.75,
        )

    def test_hierarchical_bootstrap_preserves_constant_effect(self) -> None:
        targets = 511
        scale = targets * np.log(2.0)
        arrays = [np.full(32, -0.01 * scale), np.full(32, -0.01 * scale)]
        estimates = hierarchical_paired_bootstrap_estimates(
            arrays,
            targets_per_sequence=targets,
            repetitions=100,
            seed=17,
            chunk_size=25,
        )
        np.testing.assert_allclose(estimates, -0.01, atol=1e-12)

    def test_hierarchical_bootstrap_uses_shared_sequence_resamples(self) -> None:
        scale = np.log(2.0)
        values = np.asarray([-scale, scale], dtype=np.float64)
        estimates = hierarchical_paired_bootstrap_estimates(
            [values, values.copy()],
            targets_per_sequence=1,
            repetitions=500,
            seed=23,
            chunk_size=31,
        )
        observed = set(np.round(estimates, 12))
        self.assertEqual(observed, {-1.0, 0.0, 1.0})

    def test_hierarchical_bootstrap_rejects_non_crossed_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same sequences"):
            hierarchical_paired_bootstrap_estimates(
                [np.zeros(4), np.zeros(5)],
                targets_per_sequence=1,
                repetitions=10,
            )


if __name__ == "__main__":
    unittest.main()
