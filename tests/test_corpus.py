from pathlib import Path
import tempfile
import unittest

from jamoflow.corpus import (
    expand_input_paths,
    load_records,
    partition_records,
    split_for_record,
)


class CorpusTests(unittest.TestCase):
    def test_plain_records_are_deduplicated_and_stably_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_bytes("같은 줄\n같은 줄\n다른 line\n".encode("utf-8"))
            records = load_records([path])

        self.assertEqual(len(records), 2)
        self.assertEqual(split_for_record(records[0]), split_for_record(records[0]))
        partitions = partition_records(records)
        self.assertEqual(sum(len(value) for value in partitions.values()), 2)

    def test_invalid_utf8_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.txt"
            path.write_bytes(b"valid\n\xffbroken\n")
            records = load_records([path])

        invalid = [record for record in records if record.text is None]
        self.assertEqual(len(invalid), 1)
        self.assertTrue(invalid[0].error.startswith("utf8:"))

    def test_directory_and_file_record_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text("첫 줄\n둘째 줄\n", encoding="utf-8")
            (root / "b.txt").write_text("세 번째 문서\n", encoding="utf-8")
            (root / "ignored.bin").write_bytes(b"ignored")

            expanded = expand_input_paths([root])
            records = load_records([root], plain_record_unit="file")

        self.assertEqual([path.name for path in expanded], ["a.md", "b.txt"])
        self.assertEqual(len(records), 2)
        self.assertIn("둘째 줄", records[0].text)

    def test_directory_suffix_filter_is_explicit_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            markdown = root / "included.md"
            markdown.write_text("한글 문서", encoding="utf-8")
            (root / "excluded.txt").write_text("제외", encoding="utf-8")

            with_dot = expand_input_paths([root], include_suffixes=[".MD"])
            without_dot = expand_input_paths([root], include_suffixes=["md"])

        self.assertEqual(with_dot, [markdown])
        self.assertEqual(without_dot, [markdown])


if __name__ == "__main__":
    unittest.main()
