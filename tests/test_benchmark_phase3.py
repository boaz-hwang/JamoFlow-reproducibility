import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.utf8 import prefix_boundary_mask


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_phase3.py"
SPEC = importlib.util.spec_from_file_location("benchmark_phase3", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase3BenchmarkTests(unittest.TestCase):
    def _masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        repeated = ("한국어 test 문장과 공백. " * 80).encode("utf-8")
        data = repeated[: PHASE3_MODEL_SPEC.sequence_length]
        boundaries = np.frombuffer(
            prefix_boundary_mask(data)[:-1], dtype=np.uint8
        ).reshape(1, -1)
        whitespace = compact_whitespace_mask(data).reshape(1, -1)
        spacelike = spacebyte_causal_prefix_mask(data).reshape(1, -1)
        return boundaries, whitespace, spacelike

    def test_structural_selectors_reconstruct_training_matrices(self) -> None:
        boundaries, whitespace, spacelike = self._masks()
        expected = structural_patch_matrices(
            boundaries, whitespace, spacelike
        )
        for policy, matrix in expected.items():
            selected = MODULE._structural_selector(
                policy, boundaries, whitespace, spacelike
            )
            np.testing.assert_array_equal(selected, matrix)

    def test_aligned_entropy_and_threshold_selector_cover_window(self) -> None:
        logits = torch.zeros((2, PHASE3_MODEL_SPEC.sequence_length, 256))
        scores = MODULE._aligned_entropy_scores(logits)
        self.assertEqual(scores.shape, (2, PHASE3_MODEL_SPEC.sequence_length))
        self.assertEqual(float(scores[0, 0]), 0.0)
        matrix = MODULE._entropy_selector(scores, 1.0, None)
        self.assertTrue(np.all(matrix[:, 0] == 1))
        self.assertTrue(
            np.all(matrix[:, 1:].astype(np.int64).sum(axis=1) == 512)
        )

    def test_timing_summary_reports_preregistered_tail(self) -> None:
        summary = MODULE._timing_summary([float(value) for value in range(1, 101)])
        self.assertEqual(summary["repetitions"], 100)
        self.assertEqual(summary["median_ms"], 50.5)
        self.assertIn("p95_ms", summary)

    def test_benchmark_index_matrix_is_seeded_and_disjoint(self) -> None:
        first = MODULE._benchmark_index_matrix(100, 8, 4, 1729)
        second = MODULE._benchmark_index_matrix(100, 8, 4, 1729)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (4, 8))
        self.assertEqual(len(np.unique(first)), 32)
        np.testing.assert_array_equal(first[:, :1], second[:, :1])

    def test_interleaving_uses_shared_balanced_input_batches(self) -> None:
        calls: list[tuple[str, int]] = []
        functions = {
            name: [
                (
                    lambda name=name, batch_id=batch_id: calls.append(
                        (name, batch_id)
                    )
                )
                for batch_id in range(4)
            ]
            for name in ("a", "b")
        }
        output = MODULE.benchmark_interleaved(
            functions,
            warmup_rounds=4,
            repetitions=8,
            seed=1729,
            device=None,
        )
        self.assertEqual(len(calls), 24)
        for offset in range(0, len(calls), 2):
            self.assertEqual(calls[offset][1], calls[offset + 1][1])
        for summary in output.values():
            self.assertEqual(summary["input_batches"], 4)
            self.assertEqual(
                summary["input_batch_measurement_counts"],
                {"0": 2, "1": 2, "2": 2, "3": 2},
            )
            self.assertEqual(len(summary["measurement_input_batch_ids"]), 8)
            self.assertEqual(
                {
                    batch_id: summary["measurement_input_batch_ids"].count(
                        batch_id
                    )
                    for batch_id in range(4)
                },
                {0: 2, 1: 2, 2: 2, 3: 2},
            )


if __name__ == "__main__":
    unittest.main()
