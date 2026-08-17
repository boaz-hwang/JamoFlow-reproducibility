import unittest

import numpy as np

from jamoflow.inference_benchmark import (
    latency_component_pass,
    multiseed_latency_component_pass,
    multiseed_paired_latency,
    paired_prompt_latency,
    select_inference_cases,
    timing_order_schedule,
    verification_prefix_lengths,
)
from jamoflow.utf8 import prefix_boundary_mask


class InferenceBenchmarkTests(unittest.TestCase):
    def test_case_selection_is_deterministic_deduplicated_and_closed(self) -> None:
        rows = []
        for index in range(20):
            raw = (
                f"한국어 문장 번호 {index:02d} 입니다. " * 20
            ).encode("utf-8")
            rows.append(np.frombuffer(raw[:256], dtype=np.uint8))
        inputs = np.stack(rows)
        boundaries = np.stack(
            [
                np.frombuffer(
                    prefix_boundary_mask(bytes(row))[:-1],
                    dtype=np.uint8,
                )
                for row in inputs
            ]
        )
        first = select_inference_cases(
            inputs,
            boundaries,
            case_count=8,
            prompt_length=65,
            continuation_length=65,
        )
        second = select_inference_cases(
            inputs[::-1],
            boundaries[::-1],
            case_count=8,
            prompt_length=65,
            continuation_length=65,
        )
        np.testing.assert_array_equal(first.prompts, second.prompts)
        np.testing.assert_array_equal(
            first.replay_continuations,
            second.replay_continuations,
        )
        self.assertEqual(len({bytes(row) for row in first.prompts}), 8)
        self.assertNotIn("hash", first.public_metadata())

    def test_case_selection_uses_at_most_one_prompt_per_cluster(self) -> None:
        rows = []
        for index in range(20):
            raw = (f"한국어 군집 문장 {index:02d} 입니다. " * 20).encode("utf-8")
            rows.append(np.frombuffer(raw[:256], dtype=np.uint8))
        inputs = np.stack(rows)
        boundaries = np.stack(
            [
                np.frombuffer(
                    prefix_boundary_mask(bytes(row))[:-1],
                    dtype=np.uint8,
                )
                for row in inputs
            ]
        )
        cluster_ids = np.arange(20, dtype=np.int32) // 2
        first = select_inference_cases(
            inputs,
            boundaries,
            cluster_ids=cluster_ids,
            case_count=8,
            prompt_length=65,
            continuation_length=65,
        )
        second = select_inference_cases(
            inputs[::-1],
            boundaries[::-1],
            cluster_ids=cluster_ids[::-1],
            case_count=8,
            prompt_length=65,
            continuation_length=65,
        )
        np.testing.assert_array_equal(first.prompts, second.prompts)
        np.testing.assert_array_equal(
            first.replay_continuations,
            second.replay_continuations,
        )
        self.assertEqual(first.selected_unique_clusters, 8)
        self.assertEqual(first.public_metadata()["selected_unique_clusters"], 8)

    def test_cluster_selection_rejects_malformed_ids(self) -> None:
        inputs = np.zeros((2, 256), dtype=np.uint8)
        boundaries = np.ones_like(inputs)
        with self.assertRaisesRegex(ValueError, "cluster IDs"):
            select_inference_cases(
                inputs,
                boundaries,
                cluster_ids=np.asarray([0], dtype=np.int32),
            )

    def test_paired_prompt_bootstrap_detects_stable_speedup(self) -> None:
        reference = np.full((64, 3), 10.0)
        candidate = np.full((64, 3), 8.0)
        summary = paired_prompt_latency(candidate, reference)
        self.assertAlmostEqual(summary.median_latency_reduction, 0.2)
        self.assertGreater(summary.bootstrap_percentile_95_lower, 0)
        self.assertTrue(latency_component_pass(summary))

    def test_paired_prompt_gate_rejects_input_unstable_median(self) -> None:
        reference = np.full((64, 3), 10.0)
        candidate = np.full((64, 3), 8.0)
        candidate[32:] = 20.0
        summary = paired_prompt_latency(candidate, reference)
        self.assertFalse(latency_component_pass(summary))

    def test_paired_prompt_timings_reject_malformed_values(self) -> None:
        with self.assertRaises(ValueError):
            paired_prompt_latency(np.ones((2, 2)), np.ones((2, 3)))
        with self.assertRaises(ValueError):
            paired_prompt_latency(np.zeros((2, 2)), np.ones((2, 2)))

    def test_crossed_multiseed_latency_detects_replicated_speedup(self) -> None:
        reference = np.full((5, 16, 3), 10.0)
        candidate = np.full((5, 16, 3), 8.0)
        summary = multiseed_paired_latency(
            candidate,
            reference,
            (1729, 2718, 31415, 57721, 65537),
            bootstrap_repetitions=200,
        )
        self.assertAlmostEqual(summary.crossed_median_latency_reduction, 0.2)
        self.assertAlmostEqual(summary.median_seed_point_reduction, 0.2)
        self.assertEqual(summary.positive_seed_count, 5)
        self.assertTrue(multiseed_latency_component_pass(summary))

    def test_crossed_multiseed_gate_requires_direction_in_four_seeds(self) -> None:
        reference = np.full((5, 16, 3), 10.0)
        candidate = np.full((5, 16, 3), 8.0)
        candidate[3:] = 11.0
        summary = multiseed_paired_latency(
            candidate,
            reference,
            (1729, 2718, 31415, 57721, 65537),
            bootstrap_repetitions=200,
        )
        self.assertEqual(summary.positive_seed_count, 3)
        self.assertFalse(multiseed_latency_component_pass(summary))

    def test_timing_order_is_seeded_and_balanced(self) -> None:
        first = timing_order_schedule(
            (1729, 2718),
            mode_count=2,
            prompt_count=5,
            repetitions=3,
        )
        second = timing_order_schedule(
            (1729, 2718),
            mode_count=2,
            prompt_count=5,
            repetitions=3,
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 2, 5, 3))
        for values in first.reshape(4, -1):
            self.assertLessEqual(abs(int(values.sum()) - (len(values) // 2)), 1)

    def test_verification_positions_cover_every_boundary_transition(self) -> None:
        boundaries = (0, 6, 12, 18)
        selected = verification_prefix_lengths(
            boundaries,
            24,
            minimum_positions=16,
        )
        for boundary in boundaries:
            for length in (boundary, boundary + 1, boundary + 2):
                if 1 <= length <= 24:
                    self.assertIn(length, selected)
        self.assertGreaterEqual(len(selected), 16)
        self.assertEqual(selected, tuple(sorted(selected)))


if __name__ == "__main__":
    unittest.main()
