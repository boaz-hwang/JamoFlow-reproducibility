"""Data and diagnostic helpers for Phase 2b artifact/mechanism controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .phase2_patching import CausalGridTrace, scheduled_targets
from .utf8 import prefix_boundary_mask


@dataclass(frozen=True, slots=True)
class AlignedPackedStream:
    data: bytes
    codepoint_boundaries: bytes
    sequence_length: int
    source_bytes: int
    raw_bytes_used: int
    inserted_newline_bytes: int
    dropped_tail_bytes: int

    @property
    def sequence_count(self) -> int:
        return len(self.data) // self.sequence_length

    def metadata(self) -> dict[str, int | float]:
        return {
            "sequence_length": self.sequence_length,
            "source_bytes": self.source_bytes,
            "packed_bytes": len(self.data),
            "boundary_mask_bytes": len(self.codepoint_boundaries),
            "raw_bytes_used": self.raw_bytes_used,
            "inserted_newline_bytes": self.inserted_newline_bytes,
            "dropped_tail_bytes": self.dropped_tail_bytes,
            "sequence_count": self.sequence_count,
            "inserted_fraction_of_packed_bytes": (
                self.inserted_newline_bytes / len(self.data)
            ),
            "inserted_per_raw_byte": (
                self.inserted_newline_bytes / self.raw_bytes_used
            ),
        }


def aligned_pack_stream(
    data: bytes,
    sequence_length: int = 256,
    *,
    end_boundary_mask: bytes | bytearray | np.ndarray | None = None,
    maximum_padding: int = 3,
) -> AlignedPackedStream:
    """Pack rows whose raw starts and ends are complete UTF-8 prefixes."""

    if sequence_length < 4:
        raise ValueError("sequence length must accommodate a UTF-8 codepoint")
    if not 0 <= maximum_padding < sequence_length:
        raise ValueError("maximum padding is outside the sequence range")
    if end_boundary_mask is None:
        global_boundaries = prefix_boundary_mask(data)
    else:
        supplied = np.asarray(end_boundary_mask, dtype=np.uint8)
        if supplied.shape != (len(data) + 1,):
            raise ValueError("end boundary mask must have len(data) + 1 entries")
        global_boundaries = supplied
    if not global_boundaries[0]:  # pragma: no cover - construction invariant
        raise AssertionError("an empty prefix must be a codepoint boundary")
    rows: list[bytes] = []
    start = 0
    raw_used = 0
    inserted = 0
    minimum_full_row = sequence_length - maximum_padding

    while len(data) - start >= minimum_full_row:
        maximum_end = min(start + sequence_length, len(data))
        end = maximum_end
        while end > start and not global_boundaries[end]:
            end -= 1
        raw = data[start:end]
        padding = sequence_length - len(raw)
        if padding > maximum_padding:
            break
        rows.append(raw + b"\n" * padding)
        raw_used += len(raw)
        inserted += padding
        start = end

    packed = b"".join(rows)
    boundaries = bytes(prefix_boundary_mask(packed)[:-1])
    return AlignedPackedStream(
        data=packed,
        codepoint_boundaries=boundaries,
        sequence_length=sequence_length,
        source_bytes=len(data),
        raw_bytes_used=raw_used,
        inserted_newline_bytes=inserted,
        dropped_tail_bytes=len(data) - start,
    )


def trace_diagnostics(
    traces: Sequence[CausalGridTrace],
    *,
    whitespace_masks: np.ndarray | None = None,
    punctuation_masks: np.ndarray | None = None,
) -> dict[str, float | int]:
    if not traces:
        raise ValueError("at least one trace is required")
    if whitespace_masks is not None and len(whitespace_masks) != len(traces):
        raise ValueError("whitespace masks and traces must have equal rows")
    if punctuation_masks is not None and len(punctuation_masks) != len(traces):
        raise ValueError("punctuation masks and traces must have equal rows")

    kinds = [kind for trace in traces for kind in trace.trigger_kinds]
    displacements = np.asarray(
        [value for trace in traces for value in trace.target_displacements],
        dtype=np.float64,
    )
    event_positions = [
        (row_index, position)
        for row_index, trace in enumerate(traces)
        for position, kind in zip(
            trace.boundaries[1:],
            trace.trigger_kinds,
            strict=True,
        )
        if kind == "event"
    ]
    event_count = sum(kind == "event" for kind in kinds)
    deadline_count = sum(kind == "deadline" for kind in kinds)
    final_count = sum(kind == "final" for kind in kinds)
    nonfinal = event_count + deadline_count

    result: dict[str, float | int] = {
        "examples": len(traces),
        "events": event_count,
        "deadlines": deadline_count,
        "final_boundaries": final_count,
        "nonfinal_boundaries": nonfinal,
        "event_trigger_fraction": event_count / nonfinal if nonfinal else math.nan,
        "mean_target_displacement_bytes": float(displacements.mean()),
        "median_target_displacement_bytes": float(np.median(displacements)),
        "p05_target_displacement_bytes": float(np.percentile(displacements, 5)),
        "p95_target_displacement_bytes": float(np.percentile(displacements, 95)),
        "minimum_target_displacement_bytes": int(displacements.min()),
        "maximum_target_displacement_bytes": int(displacements.max()),
    }
    if event_positions and whitespace_masks is not None:
        whitespace = sum(
            bool(whitespace_masks[row, position])
            for row, position in event_positions
        )
        result["selected_event_whitespace_count"] = whitespace
        result["selected_event_whitespace_rate"] = whitespace / event_count
    if event_positions and punctuation_masks is not None:
        punctuation = sum(
            bool(punctuation_masks[row, position])
            for row, position in event_positions
        )
        result["selected_event_punctuation_count"] = punctuation
        result["selected_event_punctuation_rate"] = punctuation / event_count
    return result


def offset_grid_displacements(
    boundary_rows: Sequence[Sequence[int]],
    sequence_length: int,
    patch_count: int,
) -> np.ndarray:
    targets = np.asarray(scheduled_targets(sequence_length, patch_count), dtype=np.int64)
    values = []
    for boundaries in boundary_rows:
        local = np.asarray(boundaries[1:], dtype=np.int64)
        if len(local) != len(targets):
            raise ValueError("boundary row has an unexpected patch count")
        values.extend((local - targets).tolist())
    return np.asarray(values, dtype=np.int64)
