import math
import unittest

import numpy as np

from jamoflow.document_inference import (
    crossed_document_cluster_bootstrap_estimates,
    document_cluster_contrast_summary,
    document_window_map_from_spans,
)


class DocumentInferenceTests(unittest.TestCase):
    def test_windows_crossing_document_or_separator_are_excluded(self) -> None:
        mapping = document_window_map_from_spans(
            2048,
            512,
            ((0, 1024), (1025, 2048)),
        )
        np.testing.assert_array_equal(
            mapping.document_indices,
            np.asarray([0, 0, -1, 1], dtype=np.int32),
        )
        self.assertEqual(mapping.eligible_sequence_count, 3)
        self.assertEqual(mapping.eligible_document_count, 2)
        self.assertAlmostEqual(mapping.eligible_sequence_fraction, 0.75)
        self.assertFalse(mapping.coverage_pass)

    def test_document_bootstrap_preserves_constant_effect(self) -> None:
        targets = 511
        scale = targets * math.log(2.0)
        indices = np.asarray([0, 0, -1, 1, 2, 2], dtype=np.int32)
        values = np.full(len(indices), -0.01 * scale)
        estimates = crossed_document_cluster_bootstrap_estimates(
            [values, values.copy(), values.copy()],
            indices,
            targets_per_sequence=targets,
            repetitions=100,
            seed=7,
            chunk_size=17,
        )
        np.testing.assert_allclose(estimates, -0.01, atol=1e-12)

    def test_document_bootstrap_resamples_whole_documents(self) -> None:
        targets = 511
        scale = targets * math.log(2.0)
        indices = np.asarray([0, 0, 1, 1], dtype=np.int32)
        values = np.asarray([-0.1, -0.1, 0.1, 0.1]) * scale
        estimates = crossed_document_cluster_bootstrap_estimates(
            [values, values.copy()],
            indices,
            targets_per_sequence=targets,
            repetitions=500,
            seed=13,
            chunk_size=31,
        )
        self.assertTrue(np.any(estimates < -0.09))
        self.assertTrue(np.any(estimates > 0.09))

    def test_document_summary_uses_only_eligible_windows(self) -> None:
        mapping = document_window_map_from_spans(
            2560,
            512,
            ((0, 1024), (1025, 2048), (2048, 2560)),
        )
        scale = 511 * math.log(2.0)
        values = np.asarray(
            [-0.01 * scale, -0.01 * scale, 10 * scale, -0.01 * scale, -0.01 * scale]
        )
        summary = document_cluster_contrast_summary(
            [values, values.copy()],
            mapping,
            targets_per_sequence=511,
            repetitions=100,
            seed=11,
        )
        self.assertAlmostEqual(
            summary["mean_effect_on_eligible_windows_bpb"],
            -0.01,
        )
        self.assertLess(summary["upper"], 0)

    def test_bootstrap_rejects_mismatched_sequence_maps(self) -> None:
        with self.assertRaisesRegex(ValueError, "shared"):
            crossed_document_cluster_bootstrap_estimates(
                [np.zeros(4), np.zeros(5)],
                np.asarray([0, 0, 1, 1], dtype=np.int32),
                targets_per_sequence=1,
                repetitions=10,
            )


if __name__ == "__main__":
    unittest.main()
