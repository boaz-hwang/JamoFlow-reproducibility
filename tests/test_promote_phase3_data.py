from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jamoflow.hplt3 import sample_hplt_lines, write_sample


SCRIPT = Path(__file__).parents[1] / "scripts" / "promote_phase3_data.py"
SPEC = importlib.util.spec_from_file_location("promote_phase3_data", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def lines(count: int = 600) -> list[bytes]:
    return [
        (
            json.dumps(
                {"text": (f"한국어 문서 {index} English. " * 20)},
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        for index in range(count)
    ]


class Phase3DataPromotionTests(unittest.TestCase):
    def test_privacy_walk_rejects_content_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden promoted key"):
            MODULE.privacy_walk({"nested": {"text": "secret"}})

    def test_summary_reloads_exact_neural_streams(self) -> None:
        quotas = {"train": 4_096, "calibration": 1_024, "test": 1_024}
        selected, scan = sample_hplt_lines(
            lines(),
            quotas=quotas,
            salt="promotion-test",
            minimum_document_bytes=256,
            maximum_document_bytes=16_384,
        )
        with tempfile.TemporaryDirectory() as temporary:
            processed = Path(temporary) / "ko.jsonl"
            output = write_sample(processed, selected)
            integrity = {
                "created_at": "2026-08-10T00:00:00+00:00",
                "dataset_id": "synthetic",
                "source": {
                    "url": "https://example.test/source.zst",
                    "filename": "source.zst",
                    "bytes": 123,
                    "sha256": "a" * 64,
                    "etag": '"abc"',
                    "last_modified": "Monday",
                },
                "selection": {
                    "salt": "promotion-test",
                    "minimum_document_bytes": 256,
                    "maximum_document_bytes": 16_384,
                    "reserve_multiplier": 2.0,
                    "quotas": quotas,
                },
                "scan": scan.to_dict(),
                "output": output,
            }
            summary = MODULE.build_summary(
                integrity,
                processed,
                sequence_length=256,
            )
            self.assertTrue(
                summary["integrity"]["all_neural_stream_quotas_exact"]
            )
            self.assertFalse(
                summary["integrity"]["raw_or_processed_text_promoted"]
            )
            for split, quota in quotas.items():
                self.assertEqual(
                    summary["splits"][split]["neural_stream"]["selected_bytes"],
                    quota,
                )

    def test_summary_rejects_unclosed_scan_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            processed = Path(temporary) / "unused.jsonl"
            integrity = {
                "created_at": "now",
                "dataset_id": "bad",
                "source": {"bytes": 1, "sha256": "a" * 64},
                "selection": {"quotas": {}},
                "scan": {
                    "eligible_records": 1,
                    "empty_text": 0,
                    "too_short": 0,
                    "too_long": 0,
                    "exact_duplicates": 0,
                    "invalid_utf8": 0,
                    "parsed_records": 2,
                    "missing_text": 0,
                    "source_lines": 2,
                    "invalid_json": 0,
                },
                "output": {},
            }
            with self.assertRaisesRegex(ValueError, "accounting does not close"):
                MODULE.build_summary(integrity, processed)


if __name__ == "__main__":
    unittest.main()
