import math
import unittest

import numpy as np

from jamoflow.phase2_patching import (
    calibrate_threshold,
    calibrate_placebo_threshold,
    causal_codepoint_grid_boundaries,
    causal_eojeol_grid_boundaries,
    causal_offset_grid_boundaries,
    causal_window_grid_trace,
    compact_delimiter_mask,
    entropy_threshold_boundaries,
    event_trigger_fraction,
    padded_hf_patch_matrix,
    scheduled_targets,
    structural_patch_matrices,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
    rolling_hash_event_mask,
)
from jamoflow.utf8 import prefix_boundary_mask


class Phase2PatchingTests(unittest.TestCase):
    @staticmethod
    def _korean_row() -> bytes:
        source = ("한국어 연구는 경계를 본다. 다음 어절 2026! " * 20).encode(
            "utf-8"
        )
        return source[:256]

    def test_preregistered_grid_targets(self) -> None:
        targets = scheduled_targets(256, 43)
        self.assertEqual(len(targets), 42)
        self.assertEqual(targets[0], 6)
        self.assertEqual(targets[-1], 251)
        self.assertTrue(all(left < right for left, right in zip(targets, targets[1:])))

    def test_codepoint_grid_is_exact_and_prefix_invariant(self) -> None:
        data = self._korean_row()
        mask = prefix_boundary_mask(data)[:-1]
        complete = causal_codepoint_grid_boundaries(mask, 43)

        self.assertEqual(len(complete), 43)
        self.assertTrue(all(mask[position] for position in complete[1:]))
        for observed in range(1, len(mask) + 1):
            partial = causal_codepoint_grid_boundaries(
                mask[:observed],
                43,
                sequence_length=256,
                require_complete=False,
            )
            self.assertEqual(
                partial,
                tuple(position for position in complete if position < observed),
            )

    def test_eojeol_grid_is_exact_prefix_invariant_and_causal(self) -> None:
        data = self._korean_row()
        mask = prefix_boundary_mask(data)[:-1]
        delimiters = compact_delimiter_mask(data)
        complete = causal_eojeol_grid_boundaries(mask, delimiters, 43)

        self.assertEqual(len(complete), 43)
        self.assertTrue(all(mask[position] for position in complete[1:]))
        for observed in range(1, len(mask) + 1):
            partial = causal_eojeol_grid_boundaries(
                mask[:observed],
                delimiters[:observed],
                43,
                sequence_length=256,
                require_complete=False,
            )
            self.assertEqual(
                partial,
                tuple(position for position in complete if position < observed),
            )

    def test_last_eojeol_target_does_not_look_early(self) -> None:
        mask = [True] * 12
        delimiters = [True] * 12
        boundaries = causal_eojeol_grid_boundaries(mask, delimiters, 3)
        self.assertEqual(boundaries, (0, 2, 8))

    def test_window_trace_records_event_deadline_and_final(self) -> None:
        mask = [True] * 18
        events = [False] * 18
        events[4] = True
        trace = causal_window_grid_trace(mask, events, patch_count=3)
        self.assertEqual(trace.boundaries, (0, 4, 12))
        self.assertEqual(trace.trigger_kinds, ("event", "final"))
        self.assertEqual(trace.target_displacements, (-2, 0))
        self.assertAlmostEqual(event_trigger_fraction([trace]), 1.0)

    def test_offset_grids_are_exact_and_prefix_invariant(self) -> None:
        data = self._korean_row()
        mask = prefix_boundary_mask(data)[:-1]
        for offset in (-2, 2):
            with self.subTest(offset=offset):
                complete = causal_offset_grid_boundaries(mask, 43, offset=offset)
                self.assertEqual(len(complete), 43)
                for observed in range(1, 257):
                    partial = causal_offset_grid_boundaries(
                        mask[:observed],
                        43,
                        offset=offset,
                        sequence_length=256,
                        require_complete=False,
                    )
                    self.assertEqual(
                        partial,
                        tuple(position for position in complete if position < observed),
                    )

    def test_rolling_hash_events_are_prefix_causal_and_calibratable(self) -> None:
        data = self._korean_row()
        complete = rolling_hash_event_mask(data, 20_000)
        for observed in (1, 17, 128, 256):
            partial = rolling_hash_event_mask(data[:observed], 20_000)
            np.testing.assert_array_equal(partial, complete[:observed])

        inputs = np.stack(
            [np.frombuffer(data, dtype=np.uint8), np.frombuffer(data, dtype=np.uint8)]
        )
        masks = np.stack(
            [np.frombuffer(prefix_boundary_mask(data)[:-1], dtype=np.uint8)] * 2
        )
        calibration = calibrate_placebo_threshold(
            inputs,
            masks,
            target_event_trigger_fraction=0.5,
            patch_count=43,
            hash_bits=8,
        )
        self.assertGreaterEqual(calibration.low_bit_threshold, 0)
        self.assertLessEqual(calibration.low_bit_threshold, 256)
        self.assertLess(calibration.absolute_error, 0.1)

    def test_unicode_delimiter_mask_marks_observed_prefix(self) -> None:
        data = "가 나,다".encode("utf-8")
        delimiters = compact_delimiter_mask(data)
        space_end = len("가 ".encode("utf-8"))
        comma_end = len("가 나,".encode("utf-8"))
        self.assertEqual(delimiters[space_end], 1)
        self.assertEqual(delimiters[comma_end], 1)
        self.assertEqual(delimiters[len("가".encode("utf-8"))], 0)

    def test_entropy_threshold_and_codepoint_candidate_cap(self) -> None:
        scores = [0.0, 0.1, 0.8, 0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1]
        self.assertEqual(
            entropy_threshold_boundaries(scores, 0.75, maximum_patch_length=4),
            (0, 2, 6),
        )
        candidates = [True, False, False, True, False, False, True, False, False, True]
        self.assertEqual(
            entropy_threshold_boundaries(
                scores,
                math.inf,
                candidate_mask=candidates,
                maximum_patch_length=4,
            ),
            (0, 6),
        )

    def test_threshold_calibration_hits_mean_rate(self) -> None:
        scores = np.random.default_rng(1729).uniform(size=(64, 64)).astype(
            np.float32
        )
        calibration = calibrate_threshold(
            scores,
            target_mean_patches=16.0,
            maximum_patch_length=12,
            tolerance=0.1,
        )
        self.assertLessEqual(calibration.absolute_error, 0.1)
        self.assertAlmostEqual(
            calibration.mean_data_patches,
            calibration.target_mean_patches,
            delta=0.1,
        )

    def test_padded_matrix_invariants_and_diagnostics(self) -> None:
        matrix = padded_hf_patch_matrix(
            [(0, 4), (0, 2, 4, 6)],
            sequence_length=8,
        )
        np.testing.assert_array_equal(
            matrix,
            np.asarray(
                [[1, 4, 4, 0, 0], [1, 2, 2, 2, 2]],
                dtype=np.uint16,
            ),
        )
        validate_padded_patch_matrix(matrix, 8)
        diagnostics = variable_patch_diagnostics(
            matrix,
            np.ones((2, 8), dtype=np.uint8),
        )
        self.assertEqual(diagnostics.data_patches, 6)
        self.assertEqual(diagnostics.padding_slots, 2)
        self.assertEqual(diagnostics.internal_codepoint_boundaries, 0)
        self.assertAlmostEqual(diagnostics.mean_data_patches, 3.0)

    def test_structural_matrices_are_exact_rate(self) -> None:
        rows = [self._korean_row(), self._korean_row()[::-1][::-1]]
        boundaries = np.stack(
            [np.frombuffer(prefix_boundary_mask(row)[:-1], dtype=np.uint8) for row in rows]
        )
        delimiters = np.stack([compact_delimiter_mask(row) for row in rows])
        matrices = structural_patch_matrices(boundaries, delimiters)
        self.assertEqual(
            set(matrices),
            {"fixed_byte_6", "causal_codepoint_grid", "causal_eojeol_grid"},
        )
        for matrix in matrices.values():
            self.assertEqual(matrix.shape, (2, 44))
            np.testing.assert_array_equal(matrix[:, 1:].sum(axis=1), [256, 256])

    def test_threshold_matrix_has_valid_variable_width(self) -> None:
        scores = np.stack(
            [
                np.linspace(0.0, 1.0, 32),
                np.linspace(1.0, 0.0, 32),
            ]
        )
        matrix = threshold_patch_matrix(
            scores,
            0.7,
            maximum_patch_length=8,
        )
        validate_padded_patch_matrix(matrix, 32)
        self.assertTrue(np.any(matrix == 0))

    def test_nontrailing_zero_padding_is_rejected(self) -> None:
        bad = np.asarray([[1, 4, 0, 4]], dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "trail"):
            validate_padded_patch_matrix(bad, 8)


if __name__ == "__main__":
    unittest.main()
