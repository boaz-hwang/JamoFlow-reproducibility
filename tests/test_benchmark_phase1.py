import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_phase1.py"
SPEC = importlib.util.spec_from_file_location("benchmark_phase1", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase1BenchmarkTests(unittest.TestCase):
    def test_boundary_helpers_produce_exact_rate_matrices(self) -> None:
        from jamoflow.neural_model import DEFAULT_MODEL_SPEC
        from jamoflow.utf8 import prefix_boundary_mask

        row = ("한국어 test " * 40).encode("utf-8")[:256]
        self.assertEqual(len(row), 256)
        masks = np.asarray(prefix_boundary_mask(row)[:-1], dtype=np.uint8)[None, :]
        scores = np.linspace(0, 1, 256, dtype=np.float32)[None, :]

        fixed = MODULE._fixed_codepoint_lengths(masks)
        entropy = MODULE._entropy_lengths(scores, masks, True)
        for matrix in (fixed, entropy):
            self.assertEqual(matrix.shape, (1, DEFAULT_MODEL_SPEC.patch_count + 1))
            self.assertEqual(matrix[0, 0], 1)
            self.assertEqual(matrix[0, 1:].sum(), 256)


if __name__ == "__main__":
    unittest.main()
