import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.prepare_leipzig import iter_sentences


class LeipzigPreparationTests(unittest.TestCase):
    def test_sentence_member_is_parsed_without_extracting_archive(self) -> None:
        payload = "1\t첫 문장\n2\tsecond\twith tab\n".encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "sample.tar.gz"
            with tarfile.open(archive_path, mode="w:gz") as archive:
                member = tarfile.TarInfo("sample/sample-sentences.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

            records = list(iter_sentences(archive_path))

        self.assertEqual(records, [("1", "첫 문장"), ("2", "second\twith tab")])


if __name__ == "__main__":
    unittest.main()
