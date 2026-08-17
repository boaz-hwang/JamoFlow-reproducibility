"""Deterministic Korean normalization stress transforms and oracle units."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

import numpy as np

from .utf8 import codepoint_spans, prefix_boundary_mask


CONDITIONS = ("original", "nfc", "nfd", "compatibility_jamo")

_COMPAT_L = tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_COMPAT_V = tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_COMPAT_T = (
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
)


@dataclass(frozen=True, slots=True)
class PaddedNormalizationStream:
    """One normalized byte stream with masked terminal-only padding."""

    condition: str
    data: bytes
    codepoint_boundaries: bytes
    target_mask: np.ndarray
    sequence_length: int
    actual_transformed_bytes: int
    terminal_padding_bytes: int

    @property
    def sequence_count(self) -> int:
        return len(self.data) // self.sequence_length

    @property
    def scored_actual_target_bytes(self) -> int:
        return int(self.target_mask.sum())

    def metadata(self) -> dict[str, int | float | str]:
        values: dict[str, int | float | str] = {
            "condition": self.condition,
            "sequence_length": self.sequence_length,
            "sequence_count": self.sequence_count,
            "actual_transformed_bytes": self.actual_transformed_bytes,
            "padded_stream_bytes": len(self.data),
            "terminal_padding_bytes": self.terminal_padding_bytes,
            "scored_actual_target_bytes": self.scored_actual_target_bytes,
            "row_leading_unscored_actual_bytes": (
                self.actual_transformed_bytes - self.scored_actual_target_bytes
            ),
            "scored_actual_byte_fraction": (
                self.scored_actual_target_bytes
                / self.actual_transformed_bytes
            ),
        }
        return values


def compatibility_jamo_text(text: str) -> str:
    output: list[str] = []
    for character in text:
        codepoint = ord(character)
        if not 0xAC00 <= codepoint <= 0xD7A3:
            output.append(character)
            continue
        offset = codepoint - 0xAC00
        leading = offset // (21 * 28)
        vowel = (offset % (21 * 28)) // 28
        trailing = offset % 28
        output.append(_COMPAT_L[leading])
        output.append(_COMPAT_V[vowel])
        if trailing:
            output.append(_COMPAT_T[trailing])
    return "".join(output)


def transform_text(text: str, condition: str) -> str:
    if condition == "original":
        return text
    if condition == "nfc":
        return unicodedata.normalize("NFC", text)
    if condition == "nfd":
        return unicodedata.normalize("NFD", text)
    if condition == "compatibility_jamo":
        return compatibility_jamo_text(text)
    raise ValueError(f"unknown normalization condition: {condition}")


def padded_normalization_stream(
    source_text: str,
    condition: str,
    sequence_length: int,
) -> PaddedNormalizationStream:
    """Transform all source text and pad only the terminal incomplete row.

    The returned target mask follows the causal-LM convention used throughout
    JamoFlow: byte zero of every row is context-only and positions 1..L-1 are
    targets. Artificial terminal LF bytes are never scored.
    """

    if sequence_length <= 1:
        raise ValueError("sequence length must be greater than one")
    transformed = transform_text(source_text, condition).encode("utf-8")
    if not transformed:
        raise ValueError("normalization source must produce at least one byte")
    terminal_padding = (-len(transformed)) % sequence_length
    data = transformed + b"\n" * terminal_padding
    positions = np.arange(len(data), dtype=np.int64).reshape(
        -1, sequence_length
    )[:, 1:]
    target_mask = positions < len(transformed)
    boundaries = bytes(prefix_boundary_mask(data)[:-1])
    return PaddedNormalizationStream(
        condition=condition,
        data=data,
        codepoint_boundaries=boundaries,
        target_mask=target_mask,
        sequence_length=sequence_length,
        actual_transformed_bytes=len(transformed),
        terminal_padding_bytes=terminal_padding,
    )


def represented_source_prefix_length(
    source_text: str,
    condition: str,
    transformed_byte_prefix: int,
) -> int:
    """Find the longest source-codepoint prefix fully represented in bytes."""

    if transformed_byte_prefix < 0:
        raise ValueError("transformed byte prefix must be non-negative")
    low = 0
    high = len(source_text) + 1
    while low + 1 < high:
        midpoint = (low + high) // 2
        length = len(transform_text(source_text[:midpoint], condition).encode("utf-8"))
        if length <= transformed_byte_prefix:
            low = midpoint
        else:
            high = midpoint
    return low


def _is_leading_jamo(codepoint: int) -> bool:
    return 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97F


def _is_vowel_jamo(codepoint: int) -> bool:
    return 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6


def _is_trailing_jamo(codepoint: int) -> bool:
    return 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB


def oracle_hangul_unit_boundary_mask(data: bytes) -> np.ndarray:
    """Return a non-causal candidate mask grouping canonical L+V+(T)."""

    spans = codepoint_spans(data)
    mask = np.zeros(len(data), dtype=np.uint8)
    if len(data):
        mask[0] = 1
    index = 0
    while index < len(spans):
        span = spans[index]
        end = span.end
        codepoint = span.codepoint
        if (
            span.valid
            and codepoint is not None
            and _is_leading_jamo(codepoint)
            and index + 1 < len(spans)
            and spans[index + 1].valid
            and spans[index + 1].codepoint is not None
            and _is_vowel_jamo(int(spans[index + 1].codepoint))
        ):
            end = spans[index + 1].end
            index += 2
            if (
                index < len(spans)
                and spans[index].valid
                and spans[index].codepoint is not None
                and _is_trailing_jamo(int(spans[index].codepoint))
            ):
                end = spans[index].end
                index += 1
        else:
            index += 1
        if end < len(data):
            mask[end] = 1
    return mask


def count_precomposed_hangul(text: str) -> int:
    return sum(0xAC00 <= ord(character) <= 0xD7A3 for character in text)
