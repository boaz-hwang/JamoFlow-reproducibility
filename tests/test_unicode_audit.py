import unittest

from jamoflow.corpus import Record, stable_record_id
from jamoflow.unicode_audit import audit_records


def record(text: str) -> Record:
    raw = text.encode("utf-8")
    return Record(stable_record_id(raw), "test", 1, raw, text)


class UnicodeAuditTests(unittest.TestCase):
    def test_hangul_jamo_and_mixed_scripts_are_separate(self) -> None:
        audit = audit_records([record("한글 abc ㅋㅋ 한")])

        self.assertEqual(audit.categories["hangul_syllable"], 2)
        self.assertEqual(audit.categories["hangul_compatibility_jamo"], 2)
        self.assertEqual(audit.categories["hangul_jamo"], 3)
        self.assertEqual(audit.categories["ascii_latin"], 3)
        self.assertEqual(audit.mixed_script_records, 1)
        self.assertEqual(audit.nfc_changed_records, 1)


if __name__ == "__main__":
    unittest.main()
