"""Unicode, normalization, and script-mixture corpus statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import unicodedata

from .corpus import Record


def is_hangul_syllable(codepoint: int) -> bool:
    return 0xAC00 <= codepoint <= 0xD7A3


def is_hangul_jamo(codepoint: int) -> bool:
    return 0x1100 <= codepoint <= 0x11FF


def is_hangul_compatibility_jamo(codepoint: int) -> bool:
    return 0x3130 <= codepoint <= 0x318F


def is_hangul_extended_jamo(codepoint: int) -> bool:
    return 0xA960 <= codepoint <= 0xA97F or 0xD7B0 <= codepoint <= 0xD7FF


def is_cjk_ideograph(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    )


def classify_character(character: str) -> str:
    codepoint = ord(character)
    category = unicodedata.category(character)

    if is_hangul_syllable(codepoint):
        return "hangul_syllable"
    if is_hangul_jamo(codepoint):
        return "hangul_jamo"
    if is_hangul_compatibility_jamo(codepoint):
        return "hangul_compatibility_jamo"
    if is_hangul_extended_jamo(codepoint):
        return "hangul_extended_jamo"
    if is_cjk_ideograph(codepoint):
        return "cjk_ideograph"
    if character.isascii() and character.isalpha():
        return "ascii_latin"
    if "LATIN" in unicodedata.name(character, "") and character.isalpha():
        return "nonascii_latin"
    if character.isdigit():
        return "digit"
    if character.isspace():
        return "whitespace"
    if category.startswith("P"):
        return "punctuation"
    if category.startswith("M"):
        return "combining_mark"
    if category.startswith("S"):
        return "symbol"
    if category.startswith("C"):
        return "control_or_unassigned"
    return "other"


def _script_for_category(category: str) -> str | None:
    if category.startswith("hangul_"):
        return "hangul"
    if category == "cjk_ideograph":
        return "cjk"
    if category in {"ascii_latin", "nonascii_latin"}:
        return "latin"
    return None


@dataclass(slots=True)
class UnicodeAudit:
    records_total: int = 0
    valid_unicode_records: int = 0
    invalid_records: int = 0
    raw_bytes: int = 0
    codepoints: int = 0
    nfc_exact_records: int = 0
    nfd_exact_records: int = 0
    nfc_changed_records: int = 0
    nfd_changed_records: int = 0
    nfc_codepoint_delta: int = 0
    nfd_codepoint_delta: int = 0
    mixed_script_records: int = 0
    categories: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)

    def update(self, record: Record) -> None:
        self.records_total += 1
        self.raw_bytes += len(record.raw)
        if record.text is None:
            self.invalid_records += 1
            self.errors[record.error or "unknown"] += 1
            return

        self.valid_unicode_records += 1
        text = record.text
        self.codepoints += len(text)

        nfc = unicodedata.normalize("NFC", text)
        nfd = unicodedata.normalize("NFD", text)
        if nfc == text:
            self.nfc_exact_records += 1
        else:
            self.nfc_changed_records += 1
        if nfd == text:
            self.nfd_exact_records += 1
        else:
            self.nfd_changed_records += 1
        self.nfc_codepoint_delta += len(nfc) - len(text)
        self.nfd_codepoint_delta += len(nfd) - len(text)

        scripts: set[str] = set()
        for character in text:
            category = classify_character(character)
            self.categories[category] += 1
            script = _script_for_category(category)
            if script is not None:
                scripts.add(script)
        if len(scripts) >= 2:
            self.mixed_script_records += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "records_total": self.records_total,
            "valid_unicode_records": self.valid_unicode_records,
            "invalid_records": self.invalid_records,
            "raw_bytes": self.raw_bytes,
            "codepoints": self.codepoints,
            "nfc_exact_records": self.nfc_exact_records,
            "nfd_exact_records": self.nfd_exact_records,
            "nfc_changed_records": self.nfc_changed_records,
            "nfd_changed_records": self.nfd_changed_records,
            "nfc_codepoint_delta": self.nfc_codepoint_delta,
            "nfd_codepoint_delta": self.nfd_codepoint_delta,
            "mixed_script_records": self.mixed_script_records,
            "categories": dict(sorted(self.categories.items())),
            "errors": dict(sorted(self.errors.items())),
        }


def audit_records(records: list[Record]) -> UnicodeAudit:
    audit = UnicodeAudit()
    for record in records:
        audit.update(record)
    return audit

