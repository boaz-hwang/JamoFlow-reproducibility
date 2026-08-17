from __future__ import annotations

import hashlib
import unittest

from byte_unigram import (
    byte_level_alphabet,
    bytes_to_level_string,
    level_string_to_bytes,
    project_mandatory_byte_fallback,
    train_deterministic_byte_unigram,
)
from fixed_byte_tokenizer import decode_ids_to_bytes, encode_raw_bytes


def _training_rows() -> tuple[str, ...]:
    stems = ("연구", "언어", "모델", "한국", "효율", "토큰", "문장", "실험")
    endings = ("입니다", "이었다", "에서는", "으로부터", "하도록", "한다")
    return tuple(
        f"{left}{right} {stems[(index + 3) % len(stems)]}{endings[index % len(endings)]} {index}"
        for index, (left, right) in enumerate(
            (stems[index % len(stems)], endings[index % len(endings)])
            for index in range(800)
        )
    )


class ByteUnigramTest(unittest.TestCase):
    def test_level_alphabet_is_bijective_and_whitespace_free(self) -> None:
        alphabet = byte_level_alphabet()
        self.assertEqual(len(alphabet), 256)
        self.assertEqual(len(set(alphabet)), 256)
        self.assertFalse(any(value.isspace() for value in alphabet))
        raw = bytes(range(256))
        self.assertEqual(level_string_to_bytes(bytes_to_level_string(raw)), raw)

    def test_training_is_deterministic_and_byte_exact(self) -> None:
        rows = _training_rows()
        results = [
            train_deterministic_byte_unigram(
                rows,
                vocabulary_size=320,
                maximum_piece_bytes=24,
            )
            for _ in range(2)
        ]
        first, second = results
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(
            hashlib.sha256(first[3]).hexdigest(),
            hashlib.sha256(second[3]).hexdigest(),
        )
        self.assertEqual(first[4], second[4])
        raw = b"\x00\xff" + "한국어  문장\n".encode("utf-8")
        ids = encode_raw_bytes(first[0], raw)
        self.assertEqual(decode_ids_to_bytes(first[0], ids), raw)
        self.assertEqual(first[4].vocabulary_size, 320)
        self.assertEqual(first[4].source_document_count, len(rows))
        self.assertTrue(first[4].overall_pass)

    def test_invalid_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            train_deterministic_byte_unigram([], vocabulary_size=300)
        with self.assertRaises(ValueError):
            train_deterministic_byte_unigram([""], vocabulary_size=300)

    def test_non_nfc_text_is_preserved_without_normalization(self) -> None:
        rows = tuple(row + " A\u030a" for row in _training_rows())
        tokenizer, _, _, _, metadata = train_deterministic_byte_unigram(
            rows,
            vocabulary_size=300,
            maximum_piece_bytes=16,
        )
        text = "NFD-like A\u030a와 한글"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)
        self.assertTrue(metadata.overall_pass)

    def test_projection_inserts_missing_bytes_and_prunes_low_score_pieces(self) -> None:
        rows = [
            (bytes((value,)), -float(value + 1))
            for value in range(255)
        ]
        rows.extend([(b"ab", -0.1), (b"bc", -0.2), (b"cd", -9.0)])
        pieces, scores, metadata = project_mandatory_byte_fallback(
            rows,
            vocabulary_size=258,
        )
        self.assertEqual(pieces[:256], tuple(bytes((value,)) for value in range(256)))
        self.assertEqual(pieces[256:], (b"ab", b"bc"))
        self.assertEqual(len(scores), 258)
        self.assertEqual(metadata["missing_single_bytes_inserted"], 1)
        self.assertEqual(metadata["learned_pieces_dropped_for_fallback"], 1)
        self.assertLess(scores[255], min(score for _, score in rows))


if __name__ == "__main__":
    unittest.main()
