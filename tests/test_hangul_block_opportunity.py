import math
import unittest

from scripts.analyze_hangul_block_opportunity import (
    _hangul_indices,
    analyze_stream,
    iter_utf8_scalars,
)


class HangulBlockOpportunityTest(unittest.TestCase):
    def test_strict_scanner_and_hangul_decomposition(self):
        text = "가Aé😀각"
        rows = list(iter_utf8_scalars(text.encode("utf-8")))
        self.assertEqual([row[0] for row in rows], [ord(char) for char in text])
        self.assertEqual([row[1] for row in rows], [3, 1, 2, 4, 3])
        self.assertEqual(_hangul_indices(ord("가")), (0, 0, 0))
        self.assertEqual(_hangul_indices(ord("각")), (0, 0, 1))
        self.assertEqual(_hangul_indices(0xD7A3), (18, 20, 27))

    def test_valid_truncated_suffix_is_excluded(self):
        data = "가A".encode("utf-8") + "힣".encode("utf-8")[:2]
        rows = list(iter_utf8_scalars(data))
        self.assertEqual(len(rows), 2)
        summary = analyze_stream(data, [2, 3, 4, 8])
        self.assertEqual(summary["stream"]["complete_scalar_bytes"], 4)
        self.assertEqual(summary["stream"]["trailing_incomplete_bytes"], 2)

    def test_oracle_accounting_closes(self):
        data = "가나다Aé".encode("utf-8")
        summary = analyze_stream(data, [2, 3, 4, 8])
        oracle = summary["target_call_oracles"]
        self.assertEqual(oracle["byte_autoregressive_calls"], 12)
        self.assertEqual(oracle["one_call_per_scalar"]["calls"], 5)
        self.assertEqual(oracle["hangul_only_adaptive"]["calls"], 6)
        self.assertEqual(oracle["hangul_only_adaptive"]["saved_calls"], 6)
        self.assertEqual(oracle["fixed_byte_blocks"]["3"]["calls"], 4)
        self.assertTrue(math.isclose(oracle["hangul_share_of_scalar_savings"], 6 / 7))

    def test_invalid_utf8_is_rejected(self):
        with self.assertRaises(ValueError):
            list(iter_utf8_scalars(b"\xed\xa0\x80"))
        with self.assertRaises(ValueError):
            list(iter_utf8_scalars(b"\xc0\x80"))


if __name__ == "__main__":
    unittest.main()
