import math
import unittest

import numpy as np

from jamoflow.document_inference import document_window_map_from_spans
from jamoflow.inference_quality import inference_quality_noninferiority


class InferenceQualityTests(unittest.TestCase):
    seeds = (1729, 2718, 31415, 57721, 65537)

    def _losses(self, effects_bpb: list[float]) -> tuple[dict, dict]:
        scale = 511 * math.log(2)
        reference = {
            seed: np.full(32, 10.0)
            for seed in self.seeds
        }
        candidate = {
            seed: np.full(32, 10.0 + effect * scale)
            for seed, effect in zip(self.seeds, effects_bpb, strict=True)
        }
        return candidate, reference

    def _window_map(self):
        return document_window_map_from_spans(
            32 * 512,
            512,
            tuple(
                (start, start + 8 * 512)
                for start in range(0, 32 * 512, 8 * 512)
            ),
        )

    def test_quality_gate_passes_when_paired_upper_is_inside_margin(self) -> None:
        candidate, reference = self._losses([0.004] * 5)
        result = inference_quality_noninferiority(
            candidate,
            reference,
            seed_order=self.seeds,
            candidate_policy="W64",
            reference_policy="C86",
            targets_per_sequence=511,
            document_window_map=self._window_map(),
            bootstrap_repetitions=100,
        )
        self.assertTrue(result.overall_pass)
        self.assertAlmostEqual(result.paired_seed_t_95_upper_bpb, 0.004)
        self.assertEqual(result.required_seed_count_within_margin, 4)
        self.assertEqual(result.seed_count_within_margin, 5)
        self.assertTrue(result.document_cluster_coverage_pass)

    def test_quality_gate_rejects_uncertain_seed_effect(self) -> None:
        candidate, reference = self._losses(
            [-0.005, -0.005, 0.009, 0.009, 0.020]
        )
        result = inference_quality_noninferiority(
            candidate,
            reference,
            seed_order=self.seeds,
            candidate_policy="W64",
            reference_policy="C86",
            targets_per_sequence=511,
            document_window_map=self._window_map(),
            bootstrap_repetitions=100,
        )
        self.assertFalse(result.overall_pass)
        self.assertGreaterEqual(result.paired_seed_t_95_upper_bpb, 0.010)

    def test_quality_gate_requires_crossed_loss_shapes(self) -> None:
        candidate, reference = self._losses([0.0] * 5)
        candidate[1729] = np.ones(31)
        with self.assertRaisesRegex(ValueError, "equal"):
            inference_quality_noninferiority(
                candidate,
                reference,
                seed_order=self.seeds,
                candidate_policy="W64",
                reference_policy="C86",
                targets_per_sequence=511,
                document_window_map=self._window_map(),
                bootstrap_repetitions=10,
            )


if __name__ == "__main__":
    unittest.main()
