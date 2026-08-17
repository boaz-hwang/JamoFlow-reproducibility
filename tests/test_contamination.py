import unittest
import random

from jamoflow.contamination import (
    benchmark_fingerprint,
    canonicalize_contamination_text,
    compare_document_to_benchmark,
    IndexedContaminationDetector,
    scan_document_reference,
)


class ContaminationCanonicalizationTests(unittest.TestCase):
    def test_nfc_line_endings_and_whitespace_are_canonical(self) -> None:
        self.assertEqual(
            canonicalize_contamination_text("  가\r\n\t나다  "),
            "가 나다",
        )

    def test_short_or_low_information_input_is_not_eligible(self) -> None:
        short = benchmark_fingerprint("short", "짧은 제목")
        punctuation = benchmark_fingerprint("punctuation", "!" * 30)
        self.assertFalse(short.eligible_for_exact)
        self.assertFalse(punctuation.eligible_for_exact)


class ContaminationMatchingTests(unittest.TestCase):
    benchmark = "한국어 뉴스 제목이 웹 문서에 그대로 포함되었는지 확인하는 기준 문장입니다"

    def test_exact_match_handles_canonical_whitespace(self) -> None:
        document = f"문서 앞부분\n  {self.benchmark}\t문서 뒷부분"
        match = compare_document_to_benchmark(
            document,
            self.benchmark,
            benchmark_id="klue/ynat/17",
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.match_type, "exact_local_containment")
        self.assertEqual(match.shingle_jaccard, 1.0)
        self.assertNotIn(self.benchmark, match.to_dict().values())

    def test_one_character_change_is_verified_as_local_near_match(self) -> None:
        changed = self.benchmark.replace("기준", "검증")
        document = f"관련 없는 머리말 {changed} 관련 없는 꼬리말"
        match = compare_document_to_benchmark(
            document,
            self.benchmark,
            benchmark_id="klue/ynat/18",
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.match_type, "near_local_shingle")
        self.assertGreaterEqual(match.benchmark_shingle_coverage, 0.8)

    def test_unrelated_document_does_not_match(self) -> None:
        self.assertIsNone(
            compare_document_to_benchmark(
                "완전히 다른 내용의 충분히 긴 한국어 문서이며 겹치는 표현이 거의 없습니다",
                self.benchmark,
                benchmark_id="klue/ynat/19",
            )
        )

    def test_reference_scan_is_stable_and_contains_no_text(self) -> None:
        matches = scan_document_reference(
            f"앞 {self.benchmark} 뒤",
            {
                "b": "서로 무관한 다른 한국어 예시 문장으로 길이 조건만 만족시킵니다",
                "a": self.benchmark,
            },
        )
        self.assertEqual([match.benchmark_id for match in matches], ["a"])
        serialized = repr([match.to_dict() for match in matches])
        self.assertNotIn(self.benchmark, serialized)


class IndexedContaminationTests(unittest.TestCase):
    def _benchmarks(self) -> dict[str, str]:
        return {
            "exact": (
                "한국어 평가 문장이 웹 문서에 완전히 들어 있는지 확인하는 예시입니다"
            ),
            "near": (
                "형태가 조금 달라진 한국어 문장도 근접 중복으로 검출하는 기준입니다"
            ),
            "repetitive": "가나다라마바사" * 5,
            "short": "짧은 제목",
        }

    def test_indexed_scan_matches_reference_for_fixed_cases(self) -> None:
        benchmarks = self._benchmarks()
        detector = IndexedContaminationDetector(benchmarks)
        documents = (
            f"앞부분 {benchmarks['exact']} 뒷부분",
            "앞부분 "
            + benchmarks["near"].replace("근접 중복", "유사 중복")
            + " 뒷부분",
            f"반복 앞 {benchmarks['repetitive']} 반복 뒤",
            "서로 관련 없는 충분히 긴 한국어 문서이며 기준 표현과 겹치지 않습니다",
            "",
        )
        for document in documents:
            with self.subTest(document_length=len(document)):
                self.assertEqual(
                    detector.scan_document(document),
                    scan_document_reference(document, benchmarks),
                )

    def test_indexed_scan_is_reference_complete_under_seeded_mutations(self) -> None:
        rng = random.Random(20_260_812)
        alphabet = "가나다라마바사아자차카타파하0123456789"
        benchmarks = {
            f"benchmark-{index}": "".join(
                rng.choice(alphabet) for _ in range(32 + index)
            )
            for index in range(12)
        }
        detector = IndexedContaminationDetector(benchmarks)
        documents = []
        for benchmark in benchmarks.values():
            documents.append("머리말" + benchmark + "꼬리말")
            changed = list(benchmark)
            changed[len(changed) // 2] = rng.choice(alphabet)
            documents.append("머리말" + "".join(changed) + "꼬리말")
        documents.extend(
            "".join(rng.choice(alphabet) for _ in range(80))
            for _ in range(12)
        )
        for document in documents:
            self.assertEqual(
                detector.scan_document(document),
                scan_document_reference(document, benchmarks),
            )

    def test_index_metadata_contains_no_benchmark_text(self) -> None:
        benchmarks = self._benchmarks()
        metadata = IndexedContaminationDetector(benchmarks).public_metadata()
        serialized = repr(metadata)
        for text in benchmarks.values():
            self.assertNotIn(text, serialized)
        self.assertEqual(metadata["benchmark_count"], len(benchmarks))
        self.assertEqual(len(metadata["benchmark_manifest_sha256"]), 64)

    def test_index_rejects_empty_identity_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark identities"):
            IndexedContaminationDetector({})
        with self.assertRaisesRegex(ValueError, "benchmark identities"):
            IndexedContaminationDetector({"": "충분히 긴 한국어 기준 문장입니다"})


if __name__ == "__main__":
    unittest.main()
