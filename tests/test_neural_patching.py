import unittest

from jamoflow.neural_patching import (
    boundaries_to_lengths,
    entropy_boundaries,
    fixed_byte_boundaries,
    fixed_codepoint_boundaries,
    hf_patch_lengths,
    validate_exact_rate,
)
from jamoflow.utf8 import prefix_boundary_mask


class NeuralPatchingTests(unittest.TestCase):
    def test_fixed_byte_has_exact_primary_rate(self) -> None:
        boundaries = fixed_byte_boundaries(256, 6)
        lengths = hf_patch_lengths(boundaries, 256)

        self.assertEqual(len(boundaries), 43)
        self.assertEqual(lengths[-1], 4)
        validate_exact_rate(lengths, sequence_length=256, patch_count=43)

    def test_codepoint_control_never_splits_hangul(self) -> None:
        data = ("한글 연구 A " * 20).encode("utf-8")[:256]
        mask = prefix_boundary_mask(data)[:-1]
        boundaries = fixed_codepoint_boundaries(mask, patch_count=43)

        self.assertEqual(len(boundaries), 43)
        self.assertTrue(all(mask[index] for index in boundaries[1:]))
        self.assertEqual(sum(boundaries_to_lengths(boundaries, len(data))), len(data))

    def test_entropy_selection_uses_score_then_earlier_tie_break(self) -> None:
        scores = [0.0, 2.0, 3.0, 3.0, 1.0, 4.0]
        self.assertEqual(entropy_boundaries(scores, patch_count=3), (0, 2, 5))

        mask = [True, True, False, True, True, True]
        self.assertEqual(
            entropy_boundaries(scores, patch_count=3, candidate_mask=mask),
            (0, 3, 5),
        )

    def test_candidate_shortage_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "found 1"):
            fixed_codepoint_boundaries([True, False, True], patch_count=3)


if __name__ == "__main__":
    unittest.main()
