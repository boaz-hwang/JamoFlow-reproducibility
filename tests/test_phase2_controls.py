import unittest

import numpy as np

from jamoflow.phase2_controls import aligned_pack_stream, trace_diagnostics
from jamoflow.phase2_patching import causal_window_grid_trace


class Phase2ControlTests(unittest.TestCase):
    def test_aligned_pack_preserves_complete_rows_with_small_padding(self) -> None:
        data = ("한글 A🙂 문장 " * 100).encode("utf-8")[:2048]
        packed = aligned_pack_stream(data, sequence_length=64)
        self.assertGreater(packed.sequence_count, 1)
        self.assertEqual(len(packed.data) % 64, 0)
        self.assertEqual(
            packed.raw_bytes_used + packed.dropped_tail_bytes,
            len(data),
        )
        self.assertLessEqual(
            packed.inserted_newline_bytes,
            packed.sequence_count * 3,
        )
        masks = np.frombuffer(packed.codepoint_boundaries, dtype=np.uint8).reshape(
            -1, 64
        )
        self.assertTrue(np.all(masks[:, 0]))
        for row in packed.data[::64]:
            self.assertIsInstance(row, int)

    def test_trace_diagnostics_classifies_selected_events(self) -> None:
        mask = [True] * 18
        event = [False] * 18
        event[4] = True
        trace = causal_window_grid_trace(mask, event, patch_count=3)
        whitespace = np.zeros((1, 18), dtype=np.uint8)
        punctuation = np.zeros((1, 18), dtype=np.uint8)
        whitespace[0, 4] = 1
        result = trace_diagnostics(
            [trace],
            whitespace_masks=whitespace,
            punctuation_masks=punctuation,
        )
        self.assertEqual(result["events"], 1)
        self.assertEqual(result["deadlines"], 0)
        self.assertEqual(result["selected_event_whitespace_rate"], 1.0)
        self.assertEqual(result["selected_event_punctuation_rate"], 0.0)

    def test_custom_end_boundaries_can_require_larger_padding(self) -> None:
        data = b"A" * 200
        candidates = np.zeros(len(data) + 1, dtype=np.uint8)
        candidates[0] = 1
        candidates[::7] = 1
        candidates[-1] = 1
        packed = aligned_pack_stream(
            data,
            sequence_length=64,
            end_boundary_mask=candidates,
            maximum_padding=6,
        )
        self.assertEqual(len(packed.data) % 64, 0)
        self.assertLessEqual(
            packed.inserted_newline_bytes,
            packed.sequence_count * 6,
        )


if __name__ == "__main__":
    unittest.main()
