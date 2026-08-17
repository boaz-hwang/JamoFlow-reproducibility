"""Document-clustered inference for packed byte-stream experiments.

The neural training stream concatenates independently split documents and then
cuts fixed byte windows.  Windows from one document are correlated, while a
small number of windows cross the inserted record separator.  This module
reconstructs that layout without serializing document text or identifiers and
provides a crossed model-seed by shared-document bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .corpus import SplitName, load_records, partition_records


DOCUMENT_CLUSTER_MINIMUM_ELIGIBLE_FRACTION = 0.95
EXCLUDED_DOCUMENT_INDEX = -1


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentWindowMap:
    """Document assignment for every fixed-length sequence in one stream."""

    document_indices: np.ndarray
    sequence_length: int
    document_count: int
    eligible_document_count: int
    eligible_sequence_count: int
    excluded_sequence_count: int
    layout_sha256: str

    @property
    def sequence_count(self) -> int:
        return len(self.document_indices)

    @property
    def eligible_sequence_fraction(self) -> float:
        return self.eligible_sequence_count / self.sequence_count

    @property
    def coverage_pass(self) -> bool:
        return (
            self.eligible_sequence_fraction
            >= DOCUMENT_CLUSTER_MINIMUM_ELIGIBLE_FRACTION
        )

    def metadata(self) -> dict[str, int | float | bool | str]:
        return {
            "sequence_count": self.sequence_count,
            "sequence_length": self.sequence_length,
            "document_count": self.document_count,
            "eligible_document_count": self.eligible_document_count,
            "eligible_sequence_count": self.eligible_sequence_count,
            "excluded_sequence_count": self.excluded_sequence_count,
            "eligible_sequence_fraction": self.eligible_sequence_fraction,
            "minimum_required_eligible_sequence_fraction": (
                DOCUMENT_CLUSTER_MINIMUM_ELIGIBLE_FRACTION
            ),
            "eligible_sequence_fraction_pass": self.coverage_pass,
            "document_assignment_sha256": _array_sha256(
                self.document_indices
            ),
            "layout_sha256": self.layout_sha256,
            "excluded_index": EXCLUDED_DOCUMENT_INDEX,
        }


def document_window_map_from_spans(
    stream_length: int,
    sequence_length: int,
    document_spans: Sequence[tuple[int, int]],
    *,
    layout_sha256: str = "synthetic",
) -> DocumentWindowMap:
    """Assign a window only when all of its bytes lie in one document span."""

    if (
        stream_length <= 0
        or sequence_length <= 1
        or stream_length % sequence_length
    ):
        raise ValueError("stream must contain complete positive-length windows")
    previous_end = 0
    for start, end in document_spans:
        if not 0 <= start < end <= stream_length or start < previous_end:
            raise ValueError("document spans must be sorted and non-overlapping")
        previous_end = end

    sequence_count = stream_length // sequence_length
    assignments = np.full(
        sequence_count,
        EXCLUDED_DOCUMENT_INDEX,
        dtype=np.int32,
    )
    span_index = 0
    for sequence_index in range(sequence_count):
        window_start = sequence_index * sequence_length
        window_end = window_start + sequence_length
        while (
            span_index < len(document_spans)
            and document_spans[span_index][1] <= window_start
        ):
            span_index += 1
        if span_index == len(document_spans):
            break
        span_start, span_end = document_spans[span_index]
        if span_start <= window_start and window_end <= span_end:
            assignments[sequence_index] = span_index

    assignments.flags.writeable = False
    eligible = assignments >= 0
    eligible_documents = np.unique(assignments[eligible])
    return DocumentWindowMap(
        document_indices=assignments,
        sequence_length=sequence_length,
        document_count=len(document_spans),
        eligible_document_count=len(eligible_documents),
        eligible_sequence_count=int(eligible.sum()),
        excluded_sequence_count=int((~eligible).sum()),
        layout_sha256=layout_sha256,
    )


def reconstruct_document_window_map(
    path: str | Path,
    *,
    split: SplitName,
    byte_limit: int,
    sequence_length: int,
    expected_stream: bytes,
) -> DocumentWindowMap:
    """Independently reconstruct packed record spans and verify stream bytes."""

    if byte_limit <= 0:
        raise ValueError("byte limit must be positive")
    records = partition_records(
        load_records(
            [path],
            corpus_format="jsonl",
            text_field="text",
            deduplicate=True,
        )
    )[split]
    buffer = bytearray()
    raw_spans: list[tuple[int, int, str]] = []
    first = True
    for record in records:
        if record.text is None:
            continue
        separator = b"" if first else b"\n"
        first = False
        if len(buffer) >= byte_limit:
            break
        remaining = byte_limit - len(buffer)
        chunk_start = len(buffer)
        selected = (separator + record.raw)[:remaining]
        buffer.extend(selected)
        separator_bytes = min(len(separator), len(selected))
        raw_bytes = len(selected) - separator_bytes
        if raw_bytes > 0:
            start = chunk_start + separator_bytes
            raw_spans.append((start, start + raw_bytes, record.record_id))

    usable = len(buffer) - (len(buffer) % sequence_length)
    reconstructed = bytes(buffer[:usable])
    if reconstructed != expected_stream:
        raise ValueError("document layout reconstruction differs from neural stream")

    spans: list[tuple[int, int]] = []
    layout_digest = hashlib.sha256()
    layout_digest.update(b"JamoFlow-document-layout-v1\0")
    for start, end, record_id in raw_spans:
        clamped_end = min(end, usable)
        if start >= clamped_end:
            continue
        spans.append((start, clamped_end))
        layout_digest.update(record_id.encode("ascii"))
        layout_digest.update(np.asarray([start, clamped_end], dtype=np.int64).tobytes())
    return document_window_map_from_spans(
        usable,
        sequence_length,
        spans,
        layout_sha256=layout_digest.hexdigest(),
    )


def crossed_document_cluster_bootstrap_estimates(
    paired_sequence_differences_nats: Sequence[np.ndarray],
    document_indices: np.ndarray,
    *,
    targets_per_sequence: int,
    repetitions: int = 10_000,
    seed: int = 20_260_811,
    chunk_size: int = 128,
) -> np.ndarray:
    """Resample paired model seeds and shared documents with natural byte weight."""

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in paired_sequence_differences_nats
    ]
    indices = np.asarray(document_indices)
    if len(arrays) < 2:
        raise ValueError("document bootstrap needs at least two model seeds")
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(
        indices.dtype, np.integer
    ):
        raise ValueError("document indices must be a non-empty integer vector")
    if any(values.shape != indices.shape for values in arrays):
        raise ValueError("every seed must use the shared sequence/document map")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("paired differences must be finite")
    if targets_per_sequence <= 0 or repetitions <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap sizes and target count must be positive")

    eligible = indices >= 0
    unique_documents = np.unique(indices[eligible])
    if len(unique_documents) < 2:
        raise ValueError("document bootstrap needs at least two eligible documents")
    dense_indices = np.searchsorted(unique_documents, indices[eligible])
    document_count = len(unique_documents)
    sequence_counts = np.bincount(
        dense_indices,
        minlength=document_count,
    ).astype(np.int64)
    document_sums = np.stack(
        [
            np.bincount(
                dense_indices,
                weights=values[eligible],
                minlength=document_count,
            )
            for values in arrays
        ]
    )

    rng = np.random.default_rng(seed)
    seed_count = len(arrays)
    estimates = np.empty(repetitions, dtype=np.float64)
    scale = targets_per_sequence * math.log(2.0)
    for start in range(0, repetitions, chunk_size):
        size = min(chunk_size, repetitions - start)
        selected_seeds = rng.integers(
            0,
            seed_count,
            size=(size, seed_count),
        )
        selected_documents = rng.integers(
            0,
            document_count,
            size=(size, document_count),
        )
        sampled_sequences = sequence_counts[selected_documents].sum(axis=1)
        source_rates = np.empty((size, seed_count), dtype=np.float64)
        for source_seed, values in enumerate(document_sums):
            source_rates[:, source_seed] = (
                values[selected_documents].sum(axis=1)
                / (sampled_sequences * scale)
            )
        estimates[start : start + size] = np.take_along_axis(
            source_rates,
            selected_seeds,
            axis=1,
        ).mean(axis=1)
    return estimates


def document_cluster_contrast_summary(
    paired_sequence_differences_nats: Sequence[np.ndarray],
    window_map: DocumentWindowMap,
    *,
    targets_per_sequence: int,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    """Return point effects and a document-clustered crossed interval."""

    eligible = window_map.document_indices >= 0
    scale = targets_per_sequence * math.log(2.0)
    effects = tuple(
        float(np.asarray(values, dtype=np.float64)[eligible].sum())
        / (int(eligible.sum()) * scale)
        for values in paired_sequence_differences_nats
    )
    estimates = crossed_document_cluster_bootstrap_estimates(
        paired_sequence_differences_nats,
        window_map.document_indices,
        targets_per_sequence=targets_per_sequence,
        repetitions=repetitions,
        seed=seed,
    )
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {
        "seed_paired_effects_on_eligible_windows_bpb": list(effects),
        "mean_effect_on_eligible_windows_bpb": float(np.mean(effects)),
        "repetitions": repetitions,
        "seed": seed,
        "resampling_design": (
            "crossed model seeds x shared source documents; all eligible "
            "windows within sampled documents; target-byte weighted"
        ),
        "mean": float(estimates.mean()),
        "median": float(median),
        "lower": float(lower),
        "upper": float(upper),
        "eligible_sequence_fraction_pass": window_map.coverage_pass,
        "window_map": window_map.metadata(),
    }
