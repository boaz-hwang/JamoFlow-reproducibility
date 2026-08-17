import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from jamoflow.neural_training import (
    _patch_batch,
    cosine_learning_rate,
    evaluate_main_model_masked,
    shuffled_indices,
)


class _UniformModel:
    def to(self, _device: str):
        return self

    def eval(self):
        return self

    def __call__(self, *, input_ids, patch_lengths, use_cache):
        del patch_lengths, use_cache
        return SimpleNamespace(
            logits=torch.zeros(
                (*input_ids.shape, 256),
                dtype=torch.float32,
                device=input_ids.device,
            )
        )


class NeuralTrainingTests(unittest.TestCase):
    def test_cosine_schedule_has_fixed_endpoints(self) -> None:
        values = [
            cosine_learning_rate(step, 10, 2, 3e-4, 3e-5)
            for step in range(10)
        ]
        self.assertAlmostEqual(values[0], 1.5e-4)
        self.assertAlmostEqual(values[1], 3e-4)
        self.assertAlmostEqual(values[-1], 3e-5)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_shuffle_is_seeded(self) -> None:
        first = shuffled_indices(100, 1729)
        second = shuffled_indices(100, 1729)
        third = shuffled_indices(100, 2718)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))

    def test_patch_batch_trims_only_globally_unused_trailing_columns(self) -> None:
        values = np.asarray(
            [
                [1, 4, 4, 0, 0],
                [1, 2, 2, 2, 2],
                [1, 8, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        selected = _patch_batch(values, np.asarray([0, 2]), "cpu")
        self.assertEqual(tuple(selected.shape), (2, 3))
        np.testing.assert_array_equal(
            selected.numpy(),
            np.asarray([[1, 4, 4], [1, 8, 0]], dtype=np.int64),
        )

    def test_masked_evaluation_excludes_unselected_targets(self) -> None:
        inputs = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
        patches = np.asarray([[1, 2, 2], [1, 2, 2]], dtype=np.uint16)
        mask = np.asarray(
            [[True, True, False], [False, True, False]],
            dtype=np.bool_,
        )
        summary, sequence_nll = evaluate_main_model_masked(
            _UniformModel(),
            inputs,
            patches,
            mask,
            "cpu",
            batch_size=1,
            return_sequence_nll=True,
        )
        self.assertEqual(summary.predicted_bytes, 3)
        self.assertAlmostEqual(summary.bpb, 8.0, places=5)
        assert sequence_nll is not None
        np.testing.assert_allclose(
            sequence_nll,
            np.asarray([2 * math.log(256), math.log(256)]),
            rtol=1e-6,
        )

    def test_masked_evaluation_rejects_wrong_mask_shape(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_main_model_masked(
                _UniformModel(),
                np.zeros((1, 4), dtype=np.uint8),
                np.asarray([[1, 4]], dtype=np.uint16),
                np.ones((1, 4), dtype=np.bool_),
                "cpu",
            )


if __name__ == "__main__":
    unittest.main()
