import unittest

import numpy as np

from jamoflow.neural_model import DEFAULT_MODEL_SPEC
from jamoflow.phase1 import (
    POLICIES,
    boundary_overlap,
    entropy_patch_matrices,
    fixed_patch_matrices,
    patch_diagnostics,
    selected_boundary_entropy,
)
from jamoflow.utf8 import prefix_boundary_mask


class Phase1Tests(unittest.TestCase):
    def test_all_primary_policies_have_identical_patch_count(self) -> None:
        text = ("한글 연구 English 中文 123. " * 30).encode("utf-8")[:256]
        inputs = np.frombuffer(text, dtype=np.uint8).reshape(1, 256)
        masks = np.frombuffer(
            bytes(prefix_boundary_mask(text)[:-1]), dtype=np.uint8
        ).reshape(1, 256)
        scores = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, 256)
        matrices = {
            **fixed_patch_matrices(masks),
            **entropy_patch_matrices(scores, masks),
        }

        self.assertEqual(set(matrices), set(POLICIES))
        for policy, matrix in matrices.items():
            with self.subTest(policy=policy):
                self.assertEqual(matrix.shape, (1, 44))
                self.assertEqual(int(matrix[:, 1:].sum()), 256)

        fixed_full = patch_diagnostics(matrices["fixed_byte"], masks)
        fixed_aligned = patch_diagnostics(matrices["fixed_codepoint"], masks)
        self.assertGreater(fixed_full.internal_codepoint_boundary_rate, 0)
        self.assertEqual(fixed_aligned.internal_codepoint_boundary_rate, 0)
        self.assertEqual(inputs.shape, (1, DEFAULT_MODEL_SPEC.sequence_length))
        self.assertGreaterEqual(
            boundary_overlap(
                matrices["entropy_full"], matrices["entropy_codepoint"]
            ),
            0,
        )
        self.assertTrue(
            np.isfinite(
                selected_boundary_entropy(matrices["entropy_full"], scores)
            )
        )


if __name__ == "__main__":
    unittest.main()
