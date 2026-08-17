"""Privacy-preserving helpers for the read-only ecological evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .corpus import expand_input_paths, iter_records, partition_records
from .phase2_patching import causal_window_grid_trace, padded_hf_patch_matrix
from .utf8 import prefix_boundary_mask


@dataclass(frozen=True, slots=True)
class EcologicalStream:
    """Aggregate-only view of a private Markdown test stream.

    The object intentionally retains neither source paths nor record IDs.
    ``data`` is held in memory for evaluation and must not be serialized.
    """

    data: bytes
    codepoint_boundaries: bytes
    sequence_length: int
    discovered_markdown_files: int
    nonempty_records: int
    duplicate_nonempty_records: int
    unique_records: int
    test_partition_records: int
    valid_test_records: int
    invalid_test_records: int
    joined_test_bytes: int
    discarded_tail_bytes: int

    @property
    def sequence_count(self) -> int:
        return len(self.data) // self.sequence_length

    def public_metadata(self) -> dict[str, int]:
        """Return only corpus-level counts safe for aggregate reporting."""

        values = asdict(self)
        del values["data"]
        del values["codepoint_boundaries"]
        values["selected_stream_bytes"] = len(self.data)
        values["sequence_count"] = self.sequence_count
        return values


def build_private_markdown_test_stream(
    root: str | Path,
    *,
    sequence_length: int = 256,
) -> EcologicalStream:
    """Read a Markdown vault without modifying it and select the hash-test split."""

    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    discovered = expand_input_paths([root_path], include_suffixes=[".md"])
    raw_records = list(
        iter_records(
            [root_path],
            corpus_format="plain",
            plain_record_unit="file",
            include_suffixes=[".md"],
        )
    )
    unique = []
    seen: set[str] = set()
    for record in raw_records:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        unique.append(record)

    selected = partition_records(unique)["test"]
    valid = [record for record in selected if record.text is not None]
    joined = b"\n".join(record.raw for record in valid)
    usable = len(joined) - len(joined) % sequence_length
    if usable == 0:
        raise ValueError("private test partition has no complete byte sequence")
    data = joined[:usable]
    boundaries = bytes(prefix_boundary_mask(data)[:-1])
    return EcologicalStream(
        data=data,
        codepoint_boundaries=boundaries,
        sequence_length=sequence_length,
        discovered_markdown_files=len(discovered),
        nonempty_records=len(raw_records),
        duplicate_nonempty_records=len(raw_records) - len(unique),
        unique_records=len(unique),
        test_partition_records=len(selected),
        valid_test_records=len(valid),
        invalid_test_records=len(selected) - len(valid),
        joined_test_bytes=len(joined),
        discarded_tail_bytes=len(joined) - usable,
    )


def whitespace_grid_patch_matrix(
    boundary_masks: np.ndarray,
    whitespace_masks: np.ndarray,
    *,
    patch_count: int,
) -> np.ndarray:
    """Build the exact-rate Phase 2b whitespace matrix for arbitrary rows."""

    if boundary_masks.ndim != 2 or boundary_masks.shape != whitespace_masks.shape:
        raise ValueError("boundary and whitespace masks must be equal matrices")
    sequence_length = int(boundary_masks.shape[1])
    rows = [
        causal_window_grid_trace(boundary, whitespace, patch_count).boundaries
        for boundary, whitespace in zip(
            boundary_masks,
            whitespace_masks,
            strict=True,
        )
    ]
    return padded_hf_patch_matrix(rows, sequence_length)


def stratum_bpb(
    sequence_nll_nats: np.ndarray,
    selected: np.ndarray,
    *,
    targets_per_sequence: int,
) -> float:
    """Convert selected per-sequence NLL values to byte-level bits."""

    import math

    losses = np.asarray(sequence_nll_nats, dtype=np.float64)
    mask = np.asarray(selected, dtype=bool)
    if losses.ndim != 1 or mask.ndim != 1 or losses.shape != mask.shape:
        raise ValueError("losses and selection must be equal vectors")
    if not mask.any():
        raise ValueError("stratum is empty")
    if targets_per_sequence <= 0:
        raise ValueError("targets_per_sequence must be positive")
    return float(losses[mask].mean()) / (targets_per_sequence * math.log(2))
