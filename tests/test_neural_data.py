import json
from pathlib import Path
import tempfile
import unittest

from jamoflow.corpus import load_records, partition_records
from jamoflow.neural_data import build_neural_stream


class NeuralDataTests(unittest.TestCase):
    def test_stream_uses_only_requested_hash_split_and_complete_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            rows = [
                {"text": f"문장 {index} alpha beta gamma"}
                for index in range(300)
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            train_ids = {
                record.record_id
                for record in partition_records(
                    load_records([path], corpus_format="jsonl")
                )["train"]
            }

            stream = build_neural_stream(
                path,
                language="ko",
                split="train",
                byte_limit=2048,
                sequence_length=64,
            )

            self.assertEqual(stream.selected_bytes % 64, 0)
            self.assertEqual(len(stream.codepoint_boundaries), len(stream.data))
            self.assertGreater(stream.sequence_count, 0)
            self.assertTrue(train_ids)
            self.assertEqual(stream.metadata()["language"], "ko")


if __name__ == "__main__":
    unittest.main()
