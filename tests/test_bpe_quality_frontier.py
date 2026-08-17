from __future__ import annotations

import math
import unittest

import numpy as np
from bpe_quality_feasibility_core import (
    EFFECTIVE_BATCH_SIZE,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
)
from bpe_quality_frontier_core import (
    LEARNING_RATE,
    MINIMUM_LEARNING_RATE,
    array_sha256,
    bpb,
    cosine_learning_rate,
    deterministic_order,
    document_bootstrap_upper,
    encode_document_chunks,
    raw_target_bytes_by_sequence,
    role_training_contract,
    select_quality_frontier,
    total_optimizer_steps,
    warmup_steps,
)


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        if add_special_tokens:
            raise AssertionError("test tokenizer has no special tokens")
        return _Encoding(list(text.encode("utf-8")))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8")


class BpeQualityFrontierTest(unittest.TestCase):
    def test_order_is_a_deterministic_permutation(self) -> None:
        first = deterministic_order(101)
        second = deterministic_order(101)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(sorted(first.tolist()), list(range(101)))
        self.assertEqual(array_sha256(first), array_sha256(second))

    def test_optimizer_contract_and_cosine_endpoints(self) -> None:
        sequence_count = 100
        steps = total_optimizer_steps(sequence_count)
        warmup = warmup_steps(sequence_count)
        contract = role_training_contract("byte_bpe_v2048_d8", sequence_count)
        self.assertEqual(steps, math.ceil(sequence_count / EFFECTIVE_BATCH_SIZE))
        self.assertEqual(
            contract["train_microbatch_size"] * contract["gradient_accumulation_steps"],
            EFFECTIVE_BATCH_SIZE,
        )
        self.assertAlmostEqual(
            cosine_learning_rate(warmup - 1, steps, warmup), LEARNING_RATE
        )
        self.assertAlmostEqual(
            cosine_learning_rate(steps - 1, steps, warmup), MINIMUM_LEARNING_RATE
        )

    def test_document_chunks_predict_every_raw_byte_exactly_once(self) -> None:
        token_bytes = tuple(bytes((value,)) for value in range(256))
        pieces = (("가나다라마바사" * 40).encode("utf-8"), b"\nsecond document")
        inventory, chunks, chunk_documents, document_bytes = encode_document_chunks(
            pieces,
            _ByteTokenizer(),
            token_bytes,
        )
        self.assertEqual(inventory.document_count, len(pieces))
        self.assertGreater(inventory.chunk_count, len(pieces))
        self.assertTrue(
            np.array_equal(document_bytes, [len(value) for value in pieces])
        )
        reconstructed = [bytearray() for _ in pieces]
        for chunk, document_index in zip(chunks, chunk_documents, strict=True):
            for token_id in chunk[1:]:
                reconstructed[int(document_index)].extend(token_bytes[int(token_id)])
        self.assertEqual(tuple(bytes(value) for value in reconstructed), pieces)
        self.assertEqual(
            inventory.token_count_excluding_prefix,
            sum(len(value) for value in pieces),
        )

    def test_raw_target_denominator_uses_only_predicted_tokens(self) -> None:
        token_bytes = (b"a", b"bc", b"def")
        sequences = np.zeros((2, SEQUENCE_LENGTH), dtype=np.int64)
        sequences[0, 0] = 2
        sequences[0, 1:] = 1
        sequences[1, 0] = 1
        sequences[1, 1:] = 2
        result = raw_target_bytes_by_sequence(sequences, token_bytes)
        self.assertTrue(
            np.array_equal(
                result,
                [(SEQUENCE_LENGTH - 1) * 2, (SEQUENCE_LENGTH - 1) * 3],
            )
        )

    def test_bpb_and_document_bootstrap_share_raw_byte_denominator(self) -> None:
        nll = np.asarray([math.log(2.0) * 8, math.log(2.0) * 16], dtype=np.float64)
        raw = np.asarray([8, 16], dtype=np.int64)
        self.assertAlmostEqual(bpb(nll, raw), 1.0)
        point, lower, upper = document_bootstrap_upper(
            nll,
            nll.copy(),
            raw,
            repetitions=100,
            seed=7,
        )
        self.assertEqual((point, lower, upper), (0.0, 0.0, 0.0))

    def test_selection_rejects_fast_degraded_role_and_uses_fastest_qualified(
        self,
    ) -> None:
        raw = np.asarray([100, 100, 100, 100], dtype=np.int64)
        anchor_bpb = 1.0
        differences = {
            role: (0.0 if index == 0 else 0.005 if index == 5 else 0.020)
            for index, role in enumerate(QUALITY_ROLES)
        }
        document_nll = {
            role: np.full(
                len(raw),
                (anchor_bpb + differences[role]) * 100 * math.log(2.0),
                dtype=np.float64,
            )
            for role in QUALITY_ROLES
        }
        systems = {
            role: float(100 - index * 10) for index, role in enumerate(QUALITY_ROLES)
        }
        decision = select_quality_frontier(
            {role: anchor_bpb + differences[role] for role in QUALITY_ROLES},
            document_nll,
            raw,
            systems,
        )
        self.assertEqual(decision["calibration_quality_anchor"], QUALITY_ROLES[0])
        self.assertEqual(
            decision["quality_qualified_roles"],
            [QUALITY_ROLES[0], QUALITY_ROLES[-1]],
        )
        self.assertEqual(decision["development_bpe_comparator"], QUALITY_ROLES[-1])
        self.assertFalse(
            decision["comparisons"][QUALITY_ROLES[-2]]["quality_qualified"]
        )

    def test_invalid_schedule_coordinates_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            cosine_learning_rate(-1, 10, 1)
        with self.assertRaises(ValueError):
            deterministic_order(0)


if __name__ == "__main__":
    unittest.main()
