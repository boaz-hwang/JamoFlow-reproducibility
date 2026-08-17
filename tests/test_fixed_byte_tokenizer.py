from __future__ import annotations

import random
import unittest

from fixed_byte_tokenizer import (
    audit_fixed_byte_tokenizer,
    build_fixed_byte_tokenizer,
    build_scored_byte_unigram_tokenizer,
    canonical_fixed_tokenizer_descriptor,
    decode_ids_to_bytes,
    encode_raw_bytes,
    ordered_pieces_sha256,
    validate_ordered_byte_pieces,
)


def _pieces(*extra: bytes) -> tuple[bytes, ...]:
    return tuple(bytes((value,)) for value in range(256)) + extra


class FixedByteTokenizerTest(unittest.TestCase):
    def test_leftmost_longest_and_minimum_token_are_distinct(self) -> None:
        # Greedy takes ``abc`` + ``d`` + ``e``.  The global DP instead takes
        # ``ab`` + ``cde``.  This guards the exact Length-MAX paper/package
        # distinction that motivates the ablation.
        pieces = _pieces(b"abc", b"ab", b"cde")
        greedy = build_fixed_byte_tokenizer(pieces, segmentation="leftmost_longest")
        dynamic = build_fixed_byte_tokenizer(pieces, segmentation="minimum_token_dp")
        greedy_ids = encode_raw_bytes(greedy, b"abcde")
        dynamic_ids = encode_raw_bytes(dynamic, b"abcde")
        self.assertEqual(len(greedy_ids), 3)
        self.assertEqual(len(dynamic_ids), 2)
        self.assertEqual(decode_ids_to_bytes(greedy, greedy_ids), b"abcde")
        self.assertEqual(decode_ids_to_bytes(dynamic, dynamic_ids), b"abcde")

    def test_leftmost_longest_is_exact_on_long_unsplit_input(self) -> None:
        pieces = _pieces(b"a" * 48, b"a" * 31, b"ab", b"ba")
        tokenizer = build_fixed_byte_tokenizer(
            pieces,
            segmentation="leftmost_longest",
        )
        raw = b"a" * 100_000 + b"ba"
        ids = encode_raw_bytes(tokenizer, raw)
        self.assertEqual(decode_ids_to_bytes(tokenizer, ids), raw)
        self.assertEqual(ids[0], 256)
        self.assertEqual(len(ids), 2_100)

    def test_bounded_trie_matches_wordpiece_reference_on_short_inputs(self) -> None:
        pieces = _pieces(b"abc", b"ab", b"bc", b"cab", b"  ", b"\nnext")
        tokenizer = build_fixed_byte_tokenizer(
            pieces,
            segmentation="leftmost_longest",
        )
        reference = tokenizer._base_tokenizer
        generator = random.Random(1729)
        alphabet = "abc \n한글"
        for _ in range(100):
            text = "".join(generator.choice(alphabet) for _ in range(80))
            self.assertEqual(
                tokenizer.encode(text, add_special_tokens=False).ids,
                tuple(reference.encode(text, add_special_tokens=False).ids),
            )

    def test_full_byte_fallback_handles_invalid_utf8(self) -> None:
        pieces = _pieces(" 한글 ".encode("utf-8"), b"\x00\xff")
        for segmentation in ("leftmost_longest", "minimum_token_dp"):
            tokenizer = build_fixed_byte_tokenizer(pieces, segmentation=segmentation)
            raw = bytes(range(256)) + b"\xff\xfe\xc0\xaf"
            ids = encode_raw_bytes(tokenizer, raw)
            self.assertEqual(decode_ids_to_bytes(tokenizer, ids), raw)

    def test_utf8_and_whitespace_roundtrip_is_exact(self) -> None:
        pieces = _pieces(" 한글 ".encode("utf-8"), b"two  spaces", b"\nnext")
        samples = ("한글", " 한글  문장\n다음\t줄 ", "ASCII and 한글")
        for segmentation in ("leftmost_longest", "minimum_token_dp"):
            tokenizer = build_fixed_byte_tokenizer(pieces, segmentation=segmentation)
            audit = audit_fixed_byte_tokenizer(
                tokenizer,
                pieces=pieces,
                segmentation=segmentation,
                utf8_samples=samples,
            )
            self.assertTrue(audit.overall_pass)
            self.assertTrue(audit.exact_utf8_roundtrip)
            self.assertTrue(audit.byte_alphabet_ids_are_identity)

    def test_piece_contract_rejects_missing_or_duplicate_fallback(self) -> None:
        with self.assertRaises(ValueError):
            validate_ordered_byte_pieces(_pieces()[1:])
        with self.assertRaises(ValueError):
            validate_ordered_byte_pieces(_pieces(b"a"))
        with self.assertRaises(ValueError):
            validate_ordered_byte_pieces(_pieces(b"x" * 49))

    def test_hash_and_descriptor_are_order_sensitive(self) -> None:
        first = _pieces(b"ab", b"bc")
        second = _pieces(b"bc", b"ab")
        self.assertNotEqual(ordered_pieces_sha256(first), ordered_pieces_sha256(second))
        descriptor = canonical_fixed_tokenizer_descriptor(
            pieces=first,
            segmentation="leftmost_longest",
        )
        self.assertEqual(descriptor["vocabulary_size"], len(first))
        self.assertEqual(len(descriptor["descriptor_sha256"]), 64)

    def test_scored_unigram_uses_scores_and_preserves_bytes(self) -> None:
        pieces = _pieces(b"ab", b"bc")
        scores = [-10.0] * 256 + [-0.1, -4.0]
        tokenizer = build_scored_byte_unigram_tokenizer(pieces, scores)
        ids = encode_raw_bytes(tokenizer, b"abc")
        self.assertEqual(decode_ids_to_bytes(tokenizer, ids), b"abc")
        self.assertEqual(ids[0], 256)
        with self.assertRaises(ValueError):
            build_scored_byte_unigram_tokenizer(pieces, scores[:-1])


if __name__ == "__main__":
    unittest.main()
