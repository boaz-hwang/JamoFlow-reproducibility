from dataclasses import replace
import random
import unittest

from jamoflow.publication_bpb import (
    RAW_BYTE_TOKENIZER_SHA256,
    build_publication_bpb_context_evidence,
    build_publication_bpb_document_plan,
    publication_bpb_scored_bytes,
    validate_publication_bpb_context_evidence,
)
from jamoflow.publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)


class PublicationBPBPlanTests(unittest.TestCase):
    def test_raw_plan_scores_every_byte_after_one_utf8_group_once(self) -> None:
        document = ("가나다라마바사 " * 80).encode("utf-8")
        plan = build_publication_bpb_document_plan(
            document,
            comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
        )
        self.assertEqual(plan.excluded_prefix_bytes, 3)
        self.assertEqual(plan.scored_bytes, len(document) - 3)
        self.assertEqual(
            sum(window.target_bytes for window in plan.windows),
            plan.scored_bytes,
        )
        self.assertTrue(
            all(window.source_bytes <= 512 for window in plan.windows)
        )
        self.assertTrue(
            all(window.target_bytes <= 256 for window in plan.windows)
        )
        self.assertTrue(
            all(window.context_bytes > 0 for window in plan.windows)
        )
        self.assertTrue(
            all(
                offset == len(document)
                or document[offset] & 0xC0 != 0x80
                for window in plan.windows
                for offset in (
                    window.context_start_byte,
                    window.target_start_byte,
                    window.target_end_byte,
                )
            )
        )
        for previous, current in zip(plan.windows, plan.windows[1:]):
            self.assertEqual(
                previous.target_end_byte,
                current.target_start_byte,
            )
        one_byte_bpe_plan = build_publication_bpb_document_plan(
            document,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            comparator_token_ids=tuple(document),
            comparator_token_bytes=tuple(bytes((value,)) for value in document),
        )
        self.assertEqual(
            plan.unit_lengths_sha256,
            one_byte_bpe_plan.unit_lengths_sha256,
        )

    def test_bpe_plan_uses_natural_units_without_splitting_a_token(self) -> None:
        document = b"a" * 900
        units = (b"a" * 3,) + (b"a" * 17,) * 52 + (b"a" * 13,)
        self.assertEqual(b"".join(units), document)
        plan = build_publication_bpb_document_plan(
            document,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
            comparator_token_ids=tuple(range(len(units))),
            comparator_token_bytes=units,
        )
        offsets = {0}
        cursor = 0
        for unit in units:
            cursor += len(unit)
            offsets.add(cursor)
        self.assertEqual(plan.excluded_prefix_bytes, 3)
        self.assertEqual(plan.scored_bytes, 897)
        for window in plan.windows:
            self.assertIn(window.context_start_byte, offsets)
            self.assertIn(window.target_start_byte, offsets)
            self.assertIn(window.target_end_byte, offsets)
            self.assertLessEqual(window.source_bytes, 512)
            self.assertLessEqual(window.target_bytes, 256)

    def test_bpe_plan_rejects_reconstruction_and_oversized_unit_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "require"):
            build_publication_bpb_document_plan(
                b"abcdef",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                comparator_token_bytes=(b"abc", b"def"),
            )
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            build_publication_bpb_document_plan(
                b"abcdef",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                comparator_token_ids=(1, 2),
                comparator_token_bytes=(b"abc", b"deg"),
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_publication_bpb_document_plan(
                b"a" * 300,
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                comparator_token_ids=(1, 2),
                comparator_token_bytes=(b"a", b"a" * 299),
            )

    def test_plan_identity_binds_natural_bpe_token_ids(self) -> None:
        document = b"abcdef"
        token_bytes = (b"ab", b"cd", b"ef")
        first = build_publication_bpb_document_plan(
            document,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            comparator_token_ids=(10, 11, 12),
            comparator_token_bytes=token_bytes,
        )
        second = build_publication_bpb_document_plan(
            document,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            comparator_token_ids=(12, 11, 10),
            comparator_token_bytes=token_bytes,
        )
        self.assertNotEqual(
            first.natural_token_ids_sha256,
            second.natural_token_ids_sha256,
        )
        self.assertNotEqual(first.plan_sha256, second.plan_sha256)

    def test_evidence_binds_pair_tokenizer_document_order_and_byte_counts(self) -> None:
        documents = (b"alpha beta", "한국어 문서".encode("utf-8"))
        units = tuple(
            tuple(bytes((value,)) for value in document)
            for document in documents
        )
        token_ids = tuple(tuple(document) for document in documents)
        evidence, plans = build_publication_bpb_context_evidence(
            documents,
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            tokenizer_sha256="b" * 64,
            comparator_token_ids_by_document=token_ids,
            comparator_token_bytes_by_document=units,
        )
        scored_bytes = publication_bpb_scored_bytes(plans)
        validate_publication_bpb_context_evidence(
            evidence,
            scored_bytes,
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_publication_bpb_context_evidence(
                evidence,
                tuple(value + 1 for value in scored_bytes),
            )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_publication_bpb_context_evidence(
                replace(evidence, comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY),
                scored_bytes,
            )

    def test_raw_evidence_uses_sealed_identity_and_reports_unscored_docs(self) -> None:
        evidence, plans = build_publication_bpb_context_evidence(
            (b"x", b"longer"),
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            tokenizer_sha256=RAW_BYTE_TOKENIZER_SHA256,
        )
        self.assertEqual(evidence.input_document_count, 2)
        self.assertEqual(evidence.scored_document_count, 1)
        self.assertEqual(evidence.unscored_document_count, 1)
        self.assertEqual(publication_bpb_scored_bytes(plans), (5,))

    def test_invalid_utf8_and_non_nfc_documents_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            build_publication_bpb_document_plan(
                b"\xffx",
                comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            )
        with self.assertRaisesRegex(ValueError, "NFC"):
            build_publication_bpb_document_plan(
                "가".encode("utf-8"),
                comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            )

    def test_seeded_random_unit_partitions_preserve_complete_target_cover(self) -> None:
        rng = random.Random(20_260_812)
        for document_length in (2, 17, 255, 256, 257, 511, 512, 513, 1_777):
            document = b"z" * document_length
            units = []
            remaining = document_length
            while remaining:
                length = min(remaining, rng.randint(1, 31))
                units.append(b"z" * length)
                remaining -= length
            plan = build_publication_bpb_document_plan(
                document,
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                comparator_token_ids=tuple(range(len(units))),
                comparator_token_bytes=tuple(units),
            )
            self.assertEqual(
                sum(window.target_bytes for window in plan.windows),
                plan.scored_bytes,
            )
            self.assertEqual(
                plan.scored_bytes,
                document_length - len(units[0]),
            )
            self.assertTrue(
                all(
                    0 < window.context_bytes
                    and 0 < window.target_bytes <= 256
                    and window.source_bytes <= 512
                    for window in plan.windows
                )
            )
if __name__ == "__main__":
    unittest.main()
