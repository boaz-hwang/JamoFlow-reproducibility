"""Deterministic byte streams for the Phase 1 neural experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .corpus import Record, SplitName, load_records, partition_records
from .utf8 import prefix_boundary_mask


@dataclass(frozen=True, slots=True)
class NeuralStream:
    language: str
    split: SplitName
    sequence_length: int
    data: bytes
    codepoint_boundaries: bytes
    available_bytes: int
    selected_records: int
    valid_records: int
    truncated_bytes: int

    @property
    def sequence_count(self) -> int:
        return len(self.data) // self.sequence_length

    @property
    def selected_bytes(self) -> int:
        return len(self.data)

    @property
    def sequence_starts_inside_codepoint(self) -> int:
        return sum(
            not self.codepoint_boundaries[index]
            for index in range(0, len(self.data), self.sequence_length)
        )

    def sequence_bytes(self, index: int) -> memoryview:
        if not 0 <= index < self.sequence_count:
            raise IndexError(index)
        start = index * self.sequence_length
        return memoryview(self.data)[start : start + self.sequence_length]

    def sequence_boundary_mask(self, index: int) -> memoryview:
        if not 0 <= index < self.sequence_count:
            raise IndexError(index)
        start = index * self.sequence_length
        return memoryview(self.codepoint_boundaries)[
            start : start + self.sequence_length
        ]

    def metadata(self) -> dict[str, int | str]:
        return {
            "language": self.language,
            "split": self.split,
            "sequence_length": self.sequence_length,
            "sequence_count": self.sequence_count,
            "available_bytes": self.available_bytes,
            "selected_bytes": self.selected_bytes,
            "selected_records": self.selected_records,
            "valid_records": self.valid_records,
            "truncated_bytes": self.truncated_bytes,
            "sequence_starts_inside_codepoint": (
                self.sequence_starts_inside_codepoint
            ),
        }


def _joined_record_prefix(
    records: Iterable[Record],
    byte_limit: int,
) -> tuple[bytes, int, int, int]:
    if byte_limit <= 0:
        raise ValueError("byte_limit must be positive")

    buffer = bytearray()
    selected_records = 0
    valid_records = 0
    available_bytes = 0
    first = True
    for record in records:
        if record.text is None:
            continue
        valid_records += 1
        separator = b"" if first else b"\n"
        first = False
        available_bytes += len(separator) + len(record.raw)
        if len(buffer) < byte_limit:
            remaining = byte_limit - len(buffer)
            chunk = separator + record.raw
            buffer.extend(chunk[:remaining])
            selected_records += 1

    return bytes(buffer), available_bytes, selected_records, valid_records


def build_neural_stream(
    path: str | Path,
    language: str,
    split: SplitName,
    byte_limit: int,
    sequence_length: int = 256,
) -> NeuralStream:
    """Build a fixed-length byte stream without leaking records across splits."""

    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")

    records = load_records(
        [path],
        corpus_format="jsonl",
        text_field="text",
        deduplicate=True,
    )
    selected = partition_records(records)[split]
    prefix, available, selected_records, valid_records = _joined_record_prefix(
        selected,
        byte_limit,
    )
    usable = len(prefix) - (len(prefix) % sequence_length)
    if usable == 0:
        raise ValueError(
            f"{language}/{split} has no complete {sequence_length}-byte sequence"
        )
    data = prefix[:usable]
    boundaries = bytes(prefix_boundary_mask(data)[:-1])
    return NeuralStream(
        language=language,
        split=split,
        sequence_length=sequence_length,
        data=data,
        codepoint_boundaries=boundaries,
        available_bytes=available,
        selected_records=selected_records,
        valid_records=valid_records,
        truncated_bytes=len(prefix) - usable,
    )
