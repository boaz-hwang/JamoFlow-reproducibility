from __future__ import annotations

import json
import unittest

from tokenizers import Tokenizer

from fixed_byte_tokenizer import build_scored_byte_unigram_tokenizer
from same2k_opportunity_protocol import (
    EXPECTED_DEPENDENCY_SHA256,
    dependency_identity,
    tokenizer_payload_semantic_sha256,
    tokenizer_semantic_sha256,
)


class Same2kOpportunityProtocolTest(unittest.TestCase):
    def test_dependency_hashes_are_pinned_and_current(self) -> None:
        identity = dependency_identity()
        self.assertEqual(set(identity), set(EXPECTED_DEPENDENCY_SHA256))
        self.assertEqual(
            {name: value["sha256"] for name, value in identity.items()},
            EXPECTED_DEPENDENCY_SHA256,
        )

    def test_tokenizer_identity_is_json_key_order_independent(self) -> None:
        pieces = tuple(bytes((value,)) for value in range(256)) + (b"ab", b"bc")
        scores = (-10.0,) * 256 + (-1.0, -2.0)
        first = build_scored_byte_unigram_tokenizer(pieces, scores)
        reloaded = Tokenizer.from_str(first.to_str(pretty=False))
        self.assertEqual(
            tokenizer_payload_semantic_sha256(
                json.loads(first.to_str(pretty=False))
            ),
            tokenizer_semantic_sha256(first),
        )
        self.assertEqual(first.get_vocab(), reloaded.get_vocab())


if __name__ == "__main__":
    unittest.main()
