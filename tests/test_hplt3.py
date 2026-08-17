from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jamoflow.corpus import load_records, partition_records
from jamoflow.hplt3 import (
    SourceMetadata,
    iter_zstd_jsonl_lines,
    sample_hplt_lines,
    validate_source_metadata,
    write_sample,
)


QUOTAS = {"train": 4_096, "calibration": 1_024, "test": 1_024}


def synthetic_lines(count: int = 600) -> list[bytes]:
    return [
        (
            json.dumps(
                {
                    "id": f"doc-{index}",
                    "text": (
                        f"문서 {index} 한국어 연구 자료와 English control을 포함한다. "
                        * 8
                    ),
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        for index in range(count)
    ]


class Hplt3PreparationTests(unittest.TestCase):
    def test_bottom_hash_sample_is_input_order_invariant(self) -> None:
        lines = synthetic_lines()
        first, first_stats = sample_hplt_lines(
            lines,
            quotas=QUOTAS,
            salt="test-salt",
            minimum_document_bytes=256,
            maximum_document_bytes=16_384,
        )
        second, second_stats = sample_hplt_lines(
            reversed(lines),
            quotas=QUOTAS,
            salt="test-salt",
            minimum_document_bytes=256,
            maximum_document_bytes=16_384,
        )
        self.assertEqual(first_stats.to_dict(), second_stats.to_dict())
        for split in QUOTAS:
            self.assertEqual(
                [candidate.digest for candidate in first[split]],
                [candidate.digest for candidate in second[split]],
            )
            available = sum(len(item.raw) for item in first[split]) + len(
                first[split]
            ) - 1
            self.assertGreaterEqual(available, QUOTAS[split])

    def test_filters_and_deduplicates_before_sampling(self) -> None:
        valid = synthetic_lines(600)
        lines = [
            b"not-json\n",
            json.dumps({"other": "field"}).encode() + b"\n",
            json.dumps({"text": ""}).encode() + b"\n",
            json.dumps({"text": "short"}).encode() + b"\n",
            *valid,
            valid[0],
        ]
        _, statistics = sample_hplt_lines(
            lines,
            quotas=QUOTAS,
            salt="test-salt",
            minimum_document_bytes=256,
            maximum_document_bytes=16_384,
        )
        self.assertEqual(statistics.invalid_json, 1)
        self.assertEqual(statistics.missing_text, 1)
        self.assertEqual(statistics.empty_text, 1)
        self.assertEqual(statistics.too_short, 1)
        self.assertEqual(statistics.exact_duplicates, 1)
        self.assertEqual(statistics.eligible_records, 600)

    def test_written_sample_preserves_hash_splits_without_overlap(self) -> None:
        selected, _ = sample_hplt_lines(
            synthetic_lines(),
            quotas=QUOTAS,
            salt="test-salt",
            minimum_document_bytes=256,
            maximum_document_bytes=16_384,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ko.jsonl"
            integrity = write_sample(output, selected)
            records = load_records([output], corpus_format="jsonl")
            partitions = partition_records(records)
            self.assertEqual(
                {split: len(values) for split, values in partitions.items()},
                {split: len(selected[split]) for split in QUOTAS},
            )
            identifiers = {
                split: {record.record_id for record in values}
                for split, values in partitions.items()
            }
            self.assertFalse(identifiers["train"] & identifiers["calibration"])
            self.assertFalse(identifiers["train"] & identifiers["test"])
            self.assertFalse(identifiers["calibration"] & identifiers["test"])
            self.assertEqual(integrity["total_records"], len(records))

    def test_zstandard_stream_reader_round_trip(self) -> None:
        import zstandard

        payload = b"".join(synthetic_lines(3))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.jsonl.zst"
            path.write_bytes(zstandard.ZstdCompressor().compress(payload))
            self.assertEqual(b"".join(iter_zstd_jsonl_lines(path)), payload)

    def test_remote_metadata_pin_is_strict(self) -> None:
        actual = SourceMetadata(
            content_length=123,
            etag='"abc"',
            last_modified="Monday",
        )
        expected = {
            "expected_bytes": 123,
            "etag": '"abc"',
            "last_modified": "Monday",
        }
        validate_source_metadata(actual, expected)
        with self.assertRaisesRegex(ValueError, "etag changed"):
            validate_source_metadata(actual, {**expected, "etag": '"def"'})


if __name__ == "__main__":
    unittest.main()
