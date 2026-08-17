"""Exact-rate patch policies for the Phase 1 controlled experiment."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def boundaries_to_lengths(
    boundaries: Iterable[int],
    sequence_length: int,
) -> tuple[int, ...]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    starts = sorted(set(boundaries))
    if not starts or starts[0] != 0:
        raise ValueError("boundaries must start at zero")
    if starts[-1] >= sequence_length:
        raise ValueError("boundaries must be inside the sequence")
    ends = [*starts[1:], sequence_length]
    lengths = tuple(end - start for start, end in zip(starts, ends, strict=True))
    if any(length <= 0 for length in lengths) or sum(lengths) != sequence_length:
        raise ValueError("boundaries do not form a complete positive partition")
    return lengths


def hf_patch_lengths(
    boundaries: Iterable[int],
    sequence_length: int,
) -> tuple[int, ...]:
    """Prepend the initial dummy patch expected by the HF BLT decoder shift."""

    return (1, *boundaries_to_lengths(boundaries, sequence_length))


def fixed_byte_boundaries(
    sequence_length: int = 256,
    stride: int = 6,
) -> tuple[int, ...]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    return tuple(range(0, sequence_length, stride))


def fixed_codepoint_boundaries(
    boundary_mask: Sequence[int | bool],
    patch_count: int,
) -> tuple[int, ...]:
    """Select an exact number of codepoint-aligned, near-uniform patches.

    This matched-count control is intentionally window-level rather than an
    online policy. It preserves candidate order and leaves enough candidates
    for every remaining target.
    """

    sequence_length = len(boundary_mask)
    if not 1 < patch_count <= sequence_length:
        raise ValueError("patch_count must be between 2 and sequence length")
    candidates = [
        index
        for index in range(1, sequence_length)
        if bool(boundary_mask[index])
    ]
    required = patch_count - 1
    if len(candidates) < required:
        raise ValueError(
            f"need {required} codepoint candidates, found {len(candidates)}"
        )

    chosen: list[int] = []
    lower = 0
    for target_number in range(1, patch_count):
        remaining_after = required - target_number
        upper = len(candidates) - remaining_after
        target = target_number * sequence_length / patch_count
        local = min(
            range(lower, upper),
            key=lambda index: (abs(candidates[index] - target), candidates[index]),
        )
        chosen.append(candidates[local])
        lower = local + 1
    return (0, *chosen)


def entropy_boundaries(
    entropy_scores: Sequence[float],
    patch_count: int,
    candidate_mask: Sequence[int | bool] | None = None,
) -> tuple[int, ...]:
    """Choose the highest-entropy exact-rate boundaries deterministically."""

    sequence_length = len(entropy_scores)
    if not 1 < patch_count <= sequence_length:
        raise ValueError("patch_count must be between 2 and sequence length")
    if candidate_mask is not None and len(candidate_mask) != sequence_length:
        raise ValueError("candidate mask and entropy scores must have equal length")

    candidates = [
        index
        for index in range(1, sequence_length)
        if candidate_mask is None or bool(candidate_mask[index])
    ]
    required = patch_count - 1
    if len(candidates) < required:
        raise ValueError(f"need {required} candidates, found {len(candidates)}")

    ranked = sorted(
        candidates,
        key=lambda index: (-float(entropy_scores[index]), index),
    )
    return (0, *sorted(ranked[:required]))


def validate_exact_rate(
    patch_lengths: Sequence[int],
    sequence_length: int,
    patch_count: int,
) -> None:
    if len(patch_lengths) != patch_count + 1:
        raise ValueError("HF patch lengths must include one initial dummy patch")
    if patch_lengths[0] != 1:
        raise ValueError("initial dummy patch must have length one")
    if sum(patch_lengths[1:]) != sequence_length:
        raise ValueError("data patch lengths must cover the sequence")
    if any(length <= 0 for length in patch_lengths):
        raise ValueError("all patch lengths must be positive")


def entropy_from_logits(logits: "object") -> "object":
    """Compute natural-log entropy while keeping torch an optional dependency."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in research env
        raise RuntimeError("torch is required for neural entropy scores") from exc
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def mean_patch_bytes(patch_lengths: Sequence[Sequence[int]]) -> float:
    total_bytes = sum(sum(lengths[1:]) for lengths in patch_lengths)
    total_patches = sum(len(lengths) - 1 for lengths in patch_lengths)
    return total_bytes / total_patches if total_patches else math.inf
