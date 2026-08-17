import importlib.util
from pathlib import Path
import unittest

import numpy as np

from jamoflow.utf8 import prefix_boundary_mask


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_phase2.py"
SPEC = importlib.util.spec_from_file_location("benchmark_phase2", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase2BenchmarkTests(unittest.TestCase):
    def test_pipeline_matrix_helpers_cover_every_byte(self) -> None:
        row = ("한국어 test 문장 " * 30).encode("utf-8")[:256]
        masks = np.frombuffer(prefix_boundary_mask(row)[:-1], dtype=np.uint8)[None, :]
        events = np.zeros_like(masks)
        events[:, 10::17] = 1

        matrices = (
            MODULE._fixed_lengths(1),
            MODULE._codepoint_lengths(masks),
            MODULE._event_lengths(masks, events),
        )
        for matrix in matrices:
            self.assertEqual(matrix[0, 0], 1)
            self.assertEqual(matrix[0, 1:].sum(), 256)
            self.assertEqual(MODULE._patch_counts(matrix)[0], 43)

    def test_trim_matrix_preserves_row_padding(self) -> None:
        matrix = np.asarray([[1, 4, 4, 0], [1, 8, 0, 0]], dtype=np.uint16)
        trimmed = MODULE._trim_matrix(matrix)
        self.assertEqual(trimmed.shape, (2, 3))
        self.assertEqual(trimmed[1, 2], 0)


if __name__ == "__main__":
    unittest.main()
