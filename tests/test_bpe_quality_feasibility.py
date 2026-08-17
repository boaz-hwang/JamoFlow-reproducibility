from __future__ import annotations

import os
import unittest

import numpy as np
from bpe_quality_feasibility_core import (
    CANDIDATE_TRAIN_BYTE_BUDGETS,
    EFFECTIVE_BATCH_SIZE,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
    encode_stream_to_memmap,
    first_sequence_batch,
    projected_optimizer_steps,
    quality_role_contract,
)
from token_frontier_core import parse_role


class BpeQualityFeasibilityTest(unittest.TestCase):
    def test_quality_roles_cover_each_vocabulary_once(self) -> None:
        vocabularies = [parse_role(role)[0] for role in QUALITY_ROLES]
        self.assertEqual(len(vocabularies), len(set(vocabularies)))
        for role in QUALITY_ROLES:
            vocabulary, _ = parse_role(role)
            contract = quality_role_contract(role, vocabulary)
            self.assertEqual(
                contract["train_microbatch_size"]
                * contract["gradient_accumulation_steps"],
                EFFECTIVE_BATCH_SIZE,
            )

    def test_projection_is_monotonic_in_raw_byte_budget(self) -> None:
        values = [
            projected_optimizer_steps(100_000, budget)
            for budget in CANDIDATE_TRAIN_BYTE_BUDGETS
        ]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], values[-1])

    def test_first_sequence_batch_is_exact_and_does_not_copy_semantics(self) -> None:
        path = self._temp_path()
        try:
            raw = np.memmap(
                path,
                mode="w+",
                dtype="<i8",
                shape=(2 * SEQUENCE_LENGTH,),
            )
            raw[:] = np.arange(len(raw))
            batch = first_sequence_batch(raw, 2)
            self.assertEqual(batch.shape, (2, SEQUENCE_LENGTH))
            self.assertTrue(np.array_equal(batch.reshape(-1), np.arange(len(raw))))
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_stream_encoding_preserves_raw_bytes_and_target_denominator(self) -> None:
        class Encoding:
            def __init__(self, ids):
                self.ids = ids

        class ByteTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.assertFalse(add_special_tokens)
                return Encoding(list(text.encode("utf-8")))

            def decode(self, ids):
                return bytes(ids).decode("utf-8")

            def assertFalse(self, value):
                if value:
                    raise AssertionError

        raw = ("가나다라마바사\n" * 80).encode("utf-8")
        inventory, memory, path = encode_stream_to_memmap(
            raw,
            ByteTokenizer(),
            tuple(bytes([value]) for value in range(256)),
            first_batch_token_count=SEQUENCE_LENGTH,
        )
        try:
            self.assertEqual(inventory.complete_utf8_bytes, len(raw))
            self.assertEqual(inventory.token_count, len(raw))
            self.assertEqual(
                inventory.predicted_target_raw_bytes,
                inventory.full_sequence_count * (SEQUENCE_LENGTH - 1),
            )
            self.assertTrue(np.array_equal(np.asarray(memory), np.frombuffer(raw, np.uint8)))
        finally:
            del memory
            os.unlink(path)

    def _temp_path(self) -> str:
        import tempfile

        descriptor, path = tempfile.mkstemp(prefix="jamoflow-feasibility-test-")
        os.close(descriptor)
        return path
