import unittest

import numpy as np

from jamoflow.normalization import (
    compatibility_jamo_text,
    oracle_hangul_unit_boundary_mask,
    padded_normalization_stream,
    represented_source_prefix_length,
    transform_text,
)
from jamoflow.utf8 import prefix_boundary_mask


class NormalizationTests(unittest.TestCase):
    def test_compatibility_decomposition_is_algorithmic(self) -> None:
        self.assertEqual(compatibility_jamo_text("가각힣 A"), "ㄱㅏㄱㅏㄱㅎㅣㅎ A")

    def test_oracle_groups_nfd_l_v_optional_t(self) -> None:
        nfd = transform_text("가각A", "nfd").encode("utf-8")
        codepoint = np.frombuffer(prefix_boundary_mask(nfd)[:-1], dtype=np.uint8)
        oracle = oracle_hangul_unit_boundary_mask(nfd)
        codepoint_positions = np.flatnonzero(codepoint).tolist()
        oracle_positions = np.flatnonzero(oracle).tolist()
        self.assertGreater(len(codepoint_positions), len(oracle_positions))
        self.assertEqual(
            oracle_positions,
            [
                0,
                len(transform_text("가", "nfd").encode("utf-8")),
                len(transform_text("가각", "nfd").encode("utf-8")),
            ],
        )

    def test_oracle_equals_codepoint_mask_for_precomposed_nfc(self) -> None:
        data = "가각A".encode("utf-8")
        codepoint = np.frombuffer(prefix_boundary_mask(data)[:-1], dtype=np.uint8)
        oracle = oracle_hangul_unit_boundary_mask(data)
        np.testing.assert_array_equal(oracle, codepoint)

    def test_represented_prefix_accounts_for_transform_expansion(self) -> None:
        source = "가각ABC"
        transformed = transform_text(source, "nfd").encode("utf-8")
        first_two = len(transform_text(source[:2], "nfd").encode("utf-8"))
        self.assertEqual(
            represented_source_prefix_length(source, "nfd", first_two),
            2,
        )
        self.assertEqual(
            represented_source_prefix_length(source, "nfd", len(transformed)),
            len(source),
        )

    def test_terminal_padding_is_not_scored(self) -> None:
        stream = padded_normalization_stream("가A", "nfc", 8)
        self.assertEqual(stream.data, "가A".encode("utf-8") + b"\n" * 4)
        self.assertEqual(stream.target_mask.shape, (1, 7))
        self.assertEqual(stream.target_mask.tolist(), [[True, True, True, False, False, False, False]])
        self.assertEqual(stream.scored_actual_target_bytes, 3)
        self.assertEqual(stream.metadata()["row_leading_unscored_actual_bytes"], 1)

    def test_each_row_leading_byte_is_context_only(self) -> None:
        stream = padded_normalization_stream("abcdefghij", "nfc", 4)
        self.assertEqual(stream.sequence_count, 3)
        self.assertEqual(stream.terminal_padding_bytes, 2)
        self.assertEqual(stream.target_mask.tolist(), [
            [True, True, True],
            [True, True, True],
            [True, False, False],
        ])
        self.assertEqual(stream.scored_actual_target_bytes, 7)

    def test_nfd_uses_the_same_source_but_more_bytes(self) -> None:
        nfc = padded_normalization_stream("가각", "nfc", 8)
        nfd = padded_normalization_stream("가각", "nfd", 8)
        self.assertEqual(nfc.actual_transformed_bytes, 6)
        self.assertEqual(nfd.actual_transformed_bytes, 15)
        self.assertGreater(nfd.sequence_count, nfc.sequence_count)

    def test_empty_normalization_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            padded_normalization_stream("", "nfc", 8)


if __name__ == "__main__":
    unittest.main()
