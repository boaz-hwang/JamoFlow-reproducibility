"""Patch-matrix construction and diagnostics for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .neural_model import DEFAULT_MODEL_SPEC, Phase1ModelSpec
from .neural_patching import (
    entropy_boundaries,
    fixed_byte_boundaries,
    fixed_codepoint_boundaries,
    hf_patch_lengths,
    validate_exact_rate,
)


POLICIES = (
    "fixed_byte",
    "fixed_codepoint",
    "entropy_full",
    "entropy_codepoint",
)


@dataclass(frozen=True, slots=True)
class PatchDiagnostics:
    examples: int
    data_patches: int
    mean_bytes_per_patch: float
    median_patch_length: float
    p95_patch_length: float
    maximum_patch_length: int
    internal_codepoint_boundaries: int
    total_noninitial_boundaries: int
    internal_codepoint_boundary_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def stream_arrays(data: bytes, boundary_mask: bytes, sequence_length: int):
    if len(data) != len(boundary_mask):
        raise ValueError("stream bytes and boundary mask must have equal length")
    if len(data) % sequence_length:
        raise ValueError("stream must contain complete fixed-length sequences")
    inputs = np.frombuffer(data, dtype=np.uint8).reshape(-1, sequence_length)
    boundaries = np.frombuffer(boundary_mask, dtype=np.uint8).reshape(
        -1, sequence_length
    )
    return inputs, boundaries


def fixed_patch_matrices(
    boundary_masks: np.ndarray,
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, np.ndarray]:
    if boundary_masks.ndim != 2 or boundary_masks.shape[1] != spec.sequence_length:
        raise ValueError("unexpected boundary mask shape")

    fixed_lengths = np.asarray(
        hf_patch_lengths(
            fixed_byte_boundaries(spec.sequence_length, spec.patch_stride),
            spec.sequence_length,
        ),
        dtype=np.uint16,
    )
    validate_exact_rate(fixed_lengths, spec.sequence_length, spec.patch_count)
    fixed_byte = np.broadcast_to(
        fixed_lengths,
        (len(boundary_masks), len(fixed_lengths)),
    ).copy()

    fixed_codepoint = np.empty_like(fixed_byte)
    for index, mask in enumerate(boundary_masks):
        boundaries = fixed_codepoint_boundaries(mask, spec.patch_count)
        lengths = hf_patch_lengths(boundaries, spec.sequence_length)
        validate_exact_rate(lengths, spec.sequence_length, spec.patch_count)
        fixed_codepoint[index] = lengths
    return {
        "fixed_byte": fixed_byte,
        "fixed_codepoint": fixed_codepoint,
    }


def entropy_patch_matrices(
    entropy_scores: np.ndarray,
    boundary_masks: np.ndarray,
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, np.ndarray]:
    if entropy_scores.shape != boundary_masks.shape:
        raise ValueError("entropy scores and boundary masks must have equal shape")
    if entropy_scores.ndim != 2 or entropy_scores.shape[1] != spec.sequence_length:
        raise ValueError("unexpected entropy score shape")

    shape = (len(entropy_scores), spec.patch_count + 1)
    full = np.empty(shape, dtype=np.uint16)
    codepoint = np.empty(shape, dtype=np.uint16)
    for index, (scores, mask) in enumerate(
        zip(entropy_scores, boundary_masks, strict=True)
    ):
        full_boundaries = entropy_boundaries(scores, spec.patch_count)
        codepoint_boundaries = entropy_boundaries(
            scores,
            spec.patch_count,
            candidate_mask=mask,
        )
        full_lengths = hf_patch_lengths(full_boundaries, spec.sequence_length)
        codepoint_lengths = hf_patch_lengths(
            codepoint_boundaries,
            spec.sequence_length,
        )
        validate_exact_rate(full_lengths, spec.sequence_length, spec.patch_count)
        validate_exact_rate(
            codepoint_lengths,
            spec.sequence_length,
            spec.patch_count,
        )
        full[index] = full_lengths
        codepoint[index] = codepoint_lengths
    return {
        "entropy_full": full,
        "entropy_codepoint": codepoint,
    }


def patch_boundaries_from_lengths(lengths: np.ndarray) -> np.ndarray:
    if lengths.ndim != 2 or lengths.shape[1] < 2:
        raise ValueError("patch lengths must be a two-dimensional HF matrix")
    # Cast before cumulative sums: NumPy promotes uint16 to uint64, whose
    # interaction with signed offsets can become float64 and whose subtraction
    # can silently underflow in displacement diagnostics.
    return np.cumsum(lengths[:, 1:].astype(np.int64), axis=1)[:, :-1]


def patch_diagnostics(
    patch_lengths: np.ndarray,
    boundary_masks: np.ndarray,
) -> PatchDiagnostics:
    if len(patch_lengths) != len(boundary_masks):
        raise ValueError("patch lengths and masks must have equal examples")
    data_lengths = patch_lengths[:, 1:]
    boundaries = patch_boundaries_from_lengths(patch_lengths)
    rows = np.arange(len(boundaries))[:, None]
    aligned = boundary_masks[rows, boundaries]
    internal = int((aligned == 0).sum())
    total = int(aligned.size)
    flat = data_lengths.reshape(-1).astype(np.float64)
    return PatchDiagnostics(
        examples=len(patch_lengths),
        data_patches=int(flat.size),
        mean_bytes_per_patch=float(flat.mean()),
        median_patch_length=float(np.median(flat)),
        p95_patch_length=float(np.percentile(flat, 95)),
        maximum_patch_length=int(flat.max()),
        internal_codepoint_boundaries=internal,
        total_noninitial_boundaries=total,
        internal_codepoint_boundary_rate=internal / total if total else math.nan,
    )


def selected_boundary_entropy(
    patch_lengths: np.ndarray,
    entropy_scores: np.ndarray,
) -> float:
    boundaries = patch_boundaries_from_lengths(patch_lengths)
    if len(boundaries) != len(entropy_scores):
        raise ValueError("patch lengths and entropy scores must have equal examples")
    rows = np.arange(len(boundaries))[:, None]
    return float(entropy_scores[rows, boundaries].mean())


def boundary_overlap(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if first.shape != second.shape:
        raise ValueError("patch matrices must have equal shape")
    first_boundaries = patch_boundaries_from_lengths(first)
    second_boundaries = patch_boundaries_from_lengths(second)
    overlaps = [
        len(set(left.tolist()) & set(right.tolist())) / len(left)
        for left, right in zip(first_boundaries, second_boundaries, strict=True)
    ]
    return float(np.mean(overlaps))


def concatenate_policy_matrices(
    matrices_by_language: Mapping[str, Mapping[str, np.ndarray]],
    languages: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return {
        policy: np.concatenate(
            [matrices_by_language[language][policy] for language in languages],
            axis=0,
        )
        for policy in POLICIES
    }
