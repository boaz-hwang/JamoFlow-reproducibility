import tempfile
import unittest
from pathlib import Path

import numpy as np

from jamoflow.corpus import Record, split_for_record, stable_record_id
from jamoflow.ecological import (
    build_private_markdown_test_stream,
    stratum_bpb,
    whitespace_grid_patch_matrix,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import compact_whitespace_mask, variable_patch_diagnostics
from jamoflow.utf8 import prefix_boundary_mask


def _test_partition_text(prefix: str) -> str:
    for index in range(10_000):
        text = f"{prefix}-{index}-한국어 문서 " * 40
        raw = text.encode("utf-8")
        record = Record(stable_record_id(raw), "unused", 1, raw, text)
        if split_for_record(record) == "test":
            return text
    raise AssertionError("failed to construct a test-partition fixture")


class EcologicalTests(unittest.TestCase):
    def test_private_stream_exposes_only_aggregate_metadata(self) -> None:
        text = _test_partition_text("private")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text(text, encoding="utf-8")
            (root / "duplicate.md").write_text(text, encoding="utf-8")
            (root / "ignored.txt").write_text(text, encoding="utf-8")
            stream = build_private_markdown_test_stream(root, sequence_length=32)

        metadata = stream.public_metadata()
        self.assertEqual(metadata["discovered_markdown_files"], 2)
        self.assertEqual(metadata["duplicate_nonempty_records"], 1)
        self.assertEqual(metadata["test_partition_records"], 1)
        self.assertEqual(len(stream.data) % 32, 0)
        self.assertNotIn("data", metadata)
        self.assertNotIn("codepoint_boundaries", metadata)
        self.assertFalse(any(str(root) in str(value) for value in metadata.values()))

    def test_whitespace_matrix_is_exact_rate_and_unicode_safe(self) -> None:
        data = ("한국어 문서 테스트 " * 20).encode("utf-8")
        data = data[: len(data) - len(data) % 64]
        boundaries = bytes(prefix_boundary_mask(data)[:-1])
        _, masks = stream_arrays(data, boundaries, 64)
        whitespace = compact_whitespace_mask(data).reshape(masks.shape)
        matrix = whitespace_grid_patch_matrix(
            masks,
            whitespace,
            patch_count=8,
        )
        diagnostics = variable_patch_diagnostics(matrix, masks)
        self.assertEqual(diagnostics.minimum_data_patches, 8)
        self.assertEqual(diagnostics.maximum_data_patches, 8)
        self.assertEqual(diagnostics.internal_codepoint_boundary_rate, 0.0)

    def test_stratum_bpb_uses_only_selected_sequences(self) -> None:
        losses = np.array([2.0, 4.0, 100.0])
        selected = np.array([True, True, False])
        observed = stratum_bpb(losses, selected, targets_per_sequence=2)
        self.assertAlmostEqual(observed, 3.0 / (2 * np.log(2)))


if __name__ == "__main__":
    unittest.main()
