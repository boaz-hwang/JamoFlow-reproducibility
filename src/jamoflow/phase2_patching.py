"""Prefix-causal patch policies and padded matrices for Phase 2.

The functions in this module deliberately operate on already-observed prefix
state.  In particular, the grid policies never search backwards from a future
target or rank candidates using a complete sequence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import unicodedata
from typing import Iterable, Sequence

import numpy as np

from .neural_model import DEFAULT_MODEL_SPEC, Phase1ModelSpec
from .neural_patching import (
    boundaries_to_lengths,
    fixed_byte_boundaries,
    hf_patch_lengths,
)
from .utf8 import prefix_codepoint_predicate_mask


STRUCTURAL_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_eojeol_grid",
)
THRESHOLD_POLICIES = (
    "entropy_threshold_full",
    "entropy_threshold_codepoint",
)
PHASE2_POLICIES = (*STRUCTURAL_POLICIES, *THRESHOLD_POLICIES)


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    threshold_nats: float
    target_mean_patches: float
    mean_data_patches: float
    absolute_error: float
    iterations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VariablePatchDiagnostics:
    examples: int
    data_patches: int
    mean_data_patches: float
    minimum_data_patches: int
    maximum_data_patches: int
    mean_bytes_per_patch: float
    median_patch_length: float
    p95_patch_length: float
    maximum_patch_length: int
    padded_data_width: int
    padding_slots: int
    padding_slot_rate: float
    internal_codepoint_boundaries: int
    total_noninitial_boundaries: int
    internal_codepoint_boundary_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalGridTrace:
    boundaries: tuple[int, ...]
    trigger_kinds: tuple[str, ...]
    target_displacements: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlaceboCalibration:
    low_bit_threshold: int
    hash_bits: int
    target_event_trigger_fraction: float
    event_trigger_fraction: float
    absolute_error: float
    iterations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def scheduled_targets(
    sequence_length: int,
    patch_count: int,
) -> tuple[int, ...]:
    """Return the preregistered absolute grid targets ``ceil(j*n/k)``."""

    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")
    if not 1 < patch_count <= sequence_length:
        raise ValueError("patch_count must be between 2 and sequence length")
    return tuple(
        math.ceil(index * sequence_length / patch_count)
        for index in range(1, patch_count)
    )


def _validate_observation_horizon(
    observed_length: int,
    sequence_length: int,
) -> None:
    if not 0 < observed_length <= sequence_length:
        raise ValueError(
            "observed prefix length must be positive and no longer than the horizon"
        )


def causal_codepoint_grid_boundaries(
    boundary_mask: Sequence[int | bool],
    patch_count: int,
    *,
    sequence_length: int | None = None,
    require_complete: bool = True,
) -> tuple[int, ...]:
    """Emit each grid boundary at the first subsequently observed codepoint end.

    ``sequence_length`` may be longer than ``boundary_mask`` to test an online
    policy on a partial prefix.  With ``require_complete=False``, the returned
    tuple contains exactly the decisions possible from that prefix.  This is
    useful for mechanically checking prefix invariance.
    """

    observed_length = len(boundary_mask)
    horizon = observed_length if sequence_length is None else sequence_length
    _validate_observation_horizon(observed_length, horizon)
    targets = scheduled_targets(horizon, patch_count)
    emitted = [0]
    next_target = 0

    for position in range(1, observed_length):
        if next_target == len(targets):
            break
        if position >= targets[next_target] and bool(boundary_mask[position]):
            emitted.append(position)
            next_target += 1

    if require_complete:
        if observed_length != horizon:
            raise ValueError("a complete result requires the full sequence horizon")
        if next_target != len(targets):
            raise ValueError(
                f"emitted {len(emitted)} of {patch_count} required patches"
            )
    return tuple(emitted)


def compact_delimiter_mask(data: bytes) -> np.ndarray:
    """Mark prefix positions immediately after Unicode whitespace/punctuation.

    Entry ``t`` describes the prefix ``data[:t]`` and can therefore be used by
    a causal selector before consuming byte ``t``.  A delimiter that ends at
    the sequence end is omitted because no patch can start there.
    """

    def is_delimiter(codepoint: int) -> bool:
        character = chr(codepoint)
        return character.isspace() or unicodedata.category(character).startswith("P")

    markers = prefix_codepoint_predicate_mask(data, is_delimiter)
    return np.frombuffer(markers, dtype=np.uint8)[:-1]


def compact_whitespace_mask(data: bytes) -> np.ndarray:
    markers = prefix_codepoint_predicate_mask(
        data,
        lambda codepoint: chr(codepoint).isspace(),
    )
    return np.frombuffer(markers, dtype=np.uint8)[:-1]


def compact_punctuation_mask(data: bytes) -> np.ndarray:
    markers = prefix_codepoint_predicate_mask(
        data,
        lambda codepoint: unicodedata.category(chr(codepoint)).startswith("P"),
    )
    return np.frombuffer(markers, dtype=np.uint8)[:-1]


def causal_window_grid_trace(
    boundary_mask: Sequence[int | bool],
    event_mask: Sequence[int | bool],
    patch_count: int,
    *,
    window: int = 2,
    minimum_patch_length: int = 2,
    sequence_length: int | None = None,
    require_complete: bool = True,
) -> CausalGridTrace:
    """Trace the shared event-or-deadline policy used by C2 and placebo."""

    if len(boundary_mask) != len(event_mask):
        raise ValueError("boundary and event masks must have equal length")
    if window < 0:
        raise ValueError("window must be non-negative")
    if minimum_patch_length <= 0:
        raise ValueError("minimum_patch_length must be positive")

    observed_length = len(boundary_mask)
    horizon = observed_length if sequence_length is None else sequence_length
    _validate_observation_horizon(observed_length, horizon)
    targets = scheduled_targets(horizon, patch_count)
    emitted = [0]
    trigger_kinds: list[str] = []
    displacements: list[int] = []
    next_target = 0

    for position in range(1, observed_length):
        if next_target == len(targets):
            break
        if not bool(boundary_mask[position]):
            continue

        target = targets[next_target]
        final_target = next_target == len(targets) - 1
        event_trigger = (
            not final_target
            and bool(event_mask[position])
            and position >= target - window
            and position - emitted[-1] >= minimum_patch_length
        )
        deadline_trigger = not final_target and position >= target + window
        final_trigger = final_target and position >= target
        if event_trigger or deadline_trigger or final_trigger:
            emitted.append(position)
            trigger_kinds.append(
                "event" if event_trigger else "final" if final_trigger else "deadline"
            )
            displacements.append(position - target)
            next_target += 1

    if require_complete:
        if observed_length != horizon:
            raise ValueError("a complete result requires the full sequence horizon")
        if next_target != len(targets):
            raise ValueError(
                f"emitted {len(emitted)} of {patch_count} required patches"
            )
    return CausalGridTrace(
        boundaries=tuple(emitted),
        trigger_kinds=tuple(trigger_kinds),
        target_displacements=tuple(displacements),
    )


def causal_eojeol_grid_boundaries(
    boundary_mask: Sequence[int | bool],
    delimiter_mask: Sequence[int | bool],
    patch_count: int,
    *,
    window: int = 2,
    minimum_patch_length: int = 2,
    sequence_length: int | None = None,
    require_complete: bool = True,
) -> tuple[int, ...]:
    """Prefer an already-observed eojeol delimiter near each absolute target."""
    return causal_window_grid_trace(
        boundary_mask,
        delimiter_mask,
        patch_count,
        window=window,
        minimum_patch_length=minimum_patch_length,
        sequence_length=sequence_length,
        require_complete=require_complete,
    ).boundaries


def causal_offset_grid_boundaries(
    boundary_mask: Sequence[int | bool],
    patch_count: int,
    *,
    offset: int,
    sequence_length: int | None = None,
    require_complete: bool = True,
) -> tuple[int, ...]:
    """Use a fixed target offset, retaining an unshifted final target."""

    observed_length = len(boundary_mask)
    horizon = observed_length if sequence_length is None else sequence_length
    _validate_observation_horizon(observed_length, horizon)
    targets = scheduled_targets(horizon, patch_count)
    shifted = tuple(
        target if index == len(targets) - 1 else target + offset
        for index, target in enumerate(targets)
    )
    if any(target <= 0 or target >= horizon for target in shifted):
        raise ValueError("offset moves a target outside the sequence")
    emitted = [0]
    next_target = 0
    for position in range(1, observed_length):
        if next_target == len(shifted):
            break
        if position >= shifted[next_target] and bool(boundary_mask[position]):
            emitted.append(position)
            next_target += 1
    if require_complete:
        if observed_length != horizon:
            raise ValueError("a complete result requires the full sequence horizon")
        if next_target != len(shifted):
            raise ValueError(
                f"emitted {len(emitted)} of {patch_count} required patches"
            )
    return tuple(emitted)


def rolling_hash_event_mask(
    data: bytes,
    low_bit_threshold: int,
    *,
    hash_bits: int = 16,
) -> np.ndarray:
    """Return deterministic prefix-causal FNV-1a placebo events."""

    if not 1 <= hash_bits <= 32:
        raise ValueError("hash_bits must be between 1 and 32")
    modulus = 1 << hash_bits
    if not 0 <= low_bit_threshold <= modulus:
        raise ValueError("low-bit threshold is outside the hash range")
    output = np.zeros(len(data), dtype=np.uint8)
    state = 14_695_981_039_346_656_037
    mask64 = (1 << 64) - 1
    low_mask = modulus - 1
    for position, value in enumerate(data, start=1):
        state ^= value
        state = (state * 1_099_511_628_211) & mask64
        if position < len(data) and (state & low_mask) < low_bit_threshold:
            output[position] = 1
    return output


def event_trigger_fraction(traces: Sequence[CausalGridTrace]) -> float:
    event_count = sum(
        kind == "event"
        for trace in traces
        for kind in trace.trigger_kinds
    )
    nonfinal = sum(
        kind != "final"
        for trace in traces
        for kind in trace.trigger_kinds
    )
    return event_count / nonfinal if nonfinal else math.nan


def calibrate_placebo_threshold(
    inputs: np.ndarray,
    boundary_masks: np.ndarray,
    target_event_trigger_fraction: float,
    patch_count: int,
    *,
    hash_bits: int = 16,
    window: int = 2,
    minimum_patch_length: int = 2,
) -> PlaceboCalibration:
    """Match placebo early-trigger frequency on calibration inputs only."""

    if inputs.ndim != 2 or inputs.shape != boundary_masks.shape:
        raise ValueError("inputs and boundary masks must be equal matrices")
    if not 0 <= target_event_trigger_fraction <= 1:
        raise ValueError("target event fraction must lie in [0, 1]")
    maximum = 1 << hash_bits

    def realized(threshold: int) -> float:
        traces = [
            causal_window_grid_trace(
                boundary_mask,
                rolling_hash_event_mask(bytes(row), threshold, hash_bits=hash_bits),
                patch_count,
                window=window,
                minimum_patch_length=minimum_patch_length,
            )
            for row, boundary_mask in zip(inputs, boundary_masks, strict=True)
        ]
        return event_trigger_fraction(traces)

    low = 0
    high = maximum
    candidates = [(low, realized(low)), (high, realized(high))]
    iterations = 0
    while low + 1 < high:
        iterations += 1
        midpoint = (low + high) // 2
        fraction = realized(midpoint)
        candidates.append((midpoint, fraction))
        if fraction < target_event_trigger_fraction:
            low = midpoint
        else:
            high = midpoint
    threshold, fraction = min(
        candidates,
        key=lambda item: (
            abs(item[1] - target_event_trigger_fraction),
            item[0],
        ),
    )
    return PlaceboCalibration(
        low_bit_threshold=threshold,
        hash_bits=hash_bits,
        target_event_trigger_fraction=target_event_trigger_fraction,
        event_trigger_fraction=fraction,
        absolute_error=abs(fraction - target_event_trigger_fraction),
        iterations=iterations,
    )


def entropy_threshold_boundaries(
    entropy_scores: Sequence[float],
    threshold_nats: float,
    *,
    candidate_mask: Sequence[int | bool] | None = None,
    minimum_patch_length: int = 1,
    maximum_patch_length: int = 24,
) -> tuple[int, ...]:
    """Apply a causal entropy threshold with a starvation cap.

    When a candidate mask is supplied, both entropy and cap decisions wait for
    an allowed boundary.  Consequently a codepoint-preserving patch can exceed
    the byte cap by the remainder of the current UTF-8 codepoint.
    """

    sequence_length = len(entropy_scores)
    if sequence_length <= 0:
        raise ValueError("entropy scores must not be empty")
    if candidate_mask is not None and len(candidate_mask) != sequence_length:
        raise ValueError("candidate mask and entropy scores must have equal length")
    if minimum_patch_length <= 0:
        raise ValueError("minimum_patch_length must be positive")
    if maximum_patch_length < minimum_patch_length:
        raise ValueError("maximum patch length must cover the minimum")
    if not math.isfinite(threshold_nats):
        if not math.isinf(threshold_nats):
            raise ValueError("threshold must be finite or infinite")

    emitted = [0]
    for position in range(1, sequence_length):
        if candidate_mask is not None and not bool(candidate_mask[position]):
            continue
        length = position - emitted[-1]
        if length < minimum_patch_length:
            continue
        if (
            length >= maximum_patch_length
            or float(entropy_scores[position]) >= threshold_nats
        ):
            emitted.append(position)
    return tuple(emitted)


def _mean_threshold_patch_count(
    entropy_scores: np.ndarray,
    threshold_nats: float,
    candidate_masks: np.ndarray | None,
    minimum_patch_length: int,
    maximum_patch_length: int,
) -> float:
    total = 0
    for index, scores in enumerate(entropy_scores):
        mask = None if candidate_masks is None else candidate_masks[index]
        total += len(
            entropy_threshold_boundaries(
                scores,
                threshold_nats,
                candidate_mask=mask,
                minimum_patch_length=minimum_patch_length,
                maximum_patch_length=maximum_patch_length,
            )
        )
    return total / len(entropy_scores)


def calibrate_threshold(
    entropy_scores: np.ndarray,
    target_mean_patches: float,
    *,
    candidate_masks: np.ndarray | None = None,
    minimum_patch_length: int = 1,
    maximum_patch_length: int = 24,
    tolerance: float = 0.1,
    maximum_iterations: int = 32,
) -> ThresholdCalibration:
    """Calibrate only the scalar threshold to a requested mean patch rate."""

    if entropy_scores.ndim != 2 or not entropy_scores.size:
        raise ValueError("entropy scores must be a non-empty matrix")
    if candidate_masks is not None and candidate_masks.shape != entropy_scores.shape:
        raise ValueError("candidate masks and entropy scores must have equal shape")
    if not 1 <= target_mean_patches <= entropy_scores.shape[1]:
        raise ValueError("target mean patches is outside the sequence range")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    finite = entropy_scores[np.isfinite(entropy_scores)]
    if not finite.size:
        raise ValueError("entropy score matrix has no finite values")

    low = math.nextafter(float(finite.min()), -math.inf)
    high = math.nextafter(float(finite.max()), math.inf)
    low_count = _mean_threshold_patch_count(
        entropy_scores,
        low,
        candidate_masks,
        minimum_patch_length,
        maximum_patch_length,
    )
    high_count = _mean_threshold_patch_count(
        entropy_scores,
        high,
        candidate_masks,
        minimum_patch_length,
        maximum_patch_length,
    )
    if not high_count <= target_mean_patches <= low_count:
        raise ValueError(
            "target patch rate is unreachable: "
            f"available range is [{high_count:.6f}, {low_count:.6f}]"
        )

    candidates = [(low, low_count), (high, high_count)]
    iterations = 0
    for iterations in range(1, maximum_iterations + 1):
        midpoint = (low + high) / 2
        count = _mean_threshold_patch_count(
            entropy_scores,
            midpoint,
            candidate_masks,
            minimum_patch_length,
            maximum_patch_length,
        )
        candidates.append((midpoint, count))
        if count > target_mean_patches:
            low = midpoint
        else:
            high = midpoint
        if abs(count - target_mean_patches) <= tolerance:
            break

    threshold, mean_patches = min(
        candidates,
        key=lambda item: (
            abs(item[1] - target_mean_patches),
            -item[0],
        ),
    )
    error = abs(mean_patches - target_mean_patches)
    if error > tolerance:
        raise ValueError(
            f"closest calibrated rate {mean_patches:.6f} misses tolerance "
            f"{tolerance:.6f}"
        )
    return ThresholdCalibration(
        threshold_nats=threshold,
        target_mean_patches=target_mean_patches,
        mean_data_patches=mean_patches,
        absolute_error=error,
        iterations=iterations,
    )


def padded_hf_patch_matrix(
    boundary_rows: Iterable[Sequence[int]],
    sequence_length: int,
) -> np.ndarray:
    """Build a right-zero-padded matrix with HF's leading dummy patch."""

    rows = [
        (1, *boundaries_to_lengths(boundaries, sequence_length))
        for boundaries in boundary_rows
    ]
    if not rows:
        raise ValueError("at least one boundary row is required")
    width = max(len(row) for row in rows)
    matrix = np.zeros((len(rows), width), dtype=np.uint16)
    for index, row in enumerate(rows):
        matrix[index, : len(row)] = row
    validate_padded_patch_matrix(matrix, sequence_length)
    return matrix


def structural_patch_matrices(
    boundary_masks: np.ndarray,
    delimiter_masks: np.ndarray,
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, np.ndarray]:
    """Build the three seed-independent preregistered Phase 2 matrices."""

    expected_shape = (len(boundary_masks), spec.sequence_length)
    if boundary_masks.ndim != 2 or boundary_masks.shape != expected_shape:
        raise ValueError("boundary masks have an unexpected shape")
    if delimiter_masks.shape != boundary_masks.shape:
        raise ValueError("delimiter and boundary masks must have equal shape")

    fixed_row = np.asarray(
        hf_patch_lengths(
            fixed_byte_boundaries(spec.sequence_length, spec.patch_stride),
            spec.sequence_length,
        ),
        dtype=np.uint16,
    )
    fixed = np.broadcast_to(
        fixed_row,
        (len(boundary_masks), len(fixed_row)),
    ).copy()
    codepoint_rows: list[tuple[int, ...]] = []
    eojeol_rows: list[tuple[int, ...]] = []
    for index, (boundary_mask, delimiter_mask) in enumerate(
        zip(boundary_masks, delimiter_masks, strict=True)
    ):
        try:
            codepoint_rows.append(
                causal_codepoint_grid_boundaries(
                    boundary_mask,
                    spec.patch_count,
                )
            )
            eojeol_rows.append(
                causal_eojeol_grid_boundaries(
                    boundary_mask,
                    delimiter_mask,
                    spec.patch_count,
                )
            )
        except ValueError as exc:
            raise ValueError(f"cannot construct structural row {index}: {exc}") from exc

    codepoint = padded_hf_patch_matrix(codepoint_rows, spec.sequence_length)
    eojeol = padded_hf_patch_matrix(eojeol_rows, spec.sequence_length)
    if fixed.shape != codepoint.shape or codepoint.shape != eojeol.shape:
        raise AssertionError("exact-rate structural policies must have equal shapes")
    return {
        "fixed_byte_6": fixed,
        "causal_codepoint_grid": codepoint,
        "causal_eojeol_grid": eojeol,
    }


def threshold_patch_matrix(
    entropy_scores: np.ndarray,
    threshold_nats: float,
    *,
    candidate_masks: np.ndarray | None = None,
    minimum_patch_length: int = 1,
    maximum_patch_length: int = 24,
) -> np.ndarray:
    """Build a variable-rate matrix from a previously calibrated threshold."""

    if entropy_scores.ndim != 2 or not entropy_scores.size:
        raise ValueError("entropy scores must be a non-empty matrix")
    if candidate_masks is not None and candidate_masks.shape != entropy_scores.shape:
        raise ValueError("candidate masks and entropy scores must have equal shape")
    rows = []
    for index, scores in enumerate(entropy_scores):
        rows.append(
            entropy_threshold_boundaries(
                scores,
                threshold_nats,
                candidate_mask=(
                    None if candidate_masks is None else candidate_masks[index]
                ),
                minimum_patch_length=minimum_patch_length,
                maximum_patch_length=maximum_patch_length,
            )
        )
    return padded_hf_patch_matrix(rows, entropy_scores.shape[1])


def validate_padded_patch_matrix(
    patch_lengths: np.ndarray,
    sequence_length: int,
) -> None:
    if patch_lengths.ndim != 2 or not len(patch_lengths):
        raise ValueError("patch lengths must be a non-empty matrix")
    if patch_lengths.shape[1] < 2:
        raise ValueError("matrix must include a dummy and at least one data patch")
    for row in patch_lengths:
        if int(row[0]) != 1:
            raise ValueError("initial dummy patch must have length one")
        data = row[1:]
        zero_positions = np.flatnonzero(data == 0)
        if zero_positions.size:
            first_zero = int(zero_positions[0])
            if np.any(data[first_zero:] != 0):
                raise ValueError("zero padding must trail all positive patches")
            positive = data[:first_zero]
        else:
            positive = data
        if not len(positive) or np.any(positive <= 0):
            raise ValueError("every row needs positive data patch lengths")
        if int(positive.astype(np.int64).sum()) != sequence_length:
            raise ValueError("positive data patch lengths must cover the sequence")


def variable_patch_diagnostics(
    patch_lengths: np.ndarray,
    boundary_masks: np.ndarray | None = None,
) -> VariablePatchDiagnostics:
    """Summarize realized patches without treating zero padding as patches."""

    if patch_lengths.ndim != 2 or not len(patch_lengths):
        raise ValueError("patch lengths must be a non-empty matrix")
    sequence_length = int(patch_lengths[0, 1:].astype(np.int64).sum())
    validate_padded_patch_matrix(patch_lengths, sequence_length)
    if boundary_masks is not None:
        if boundary_masks.shape != (len(patch_lengths), sequence_length):
            raise ValueError("boundary masks have an unexpected shape")

    counts: list[int] = []
    lengths: list[int] = []
    internal = 0
    total_boundaries = 0
    for row_index, row in enumerate(patch_lengths):
        positive = row[1:][row[1:] > 0].astype(np.int64)
        counts.append(len(positive))
        lengths.extend(int(value) for value in positive)
        if boundary_masks is not None and len(positive) > 1:
            boundaries = np.cumsum(positive)[:-1]
            aligned = boundary_masks[row_index, boundaries]
            internal += int((aligned == 0).sum())
            total_boundaries += len(boundaries)

    padded_data_width = patch_lengths.shape[1] - 1
    padding_slots = len(patch_lengths) * padded_data_width - sum(counts)
    padded_slots = len(patch_lengths) * padded_data_width
    return VariablePatchDiagnostics(
        examples=len(patch_lengths),
        data_patches=sum(counts),
        mean_data_patches=float(np.mean(counts)),
        minimum_data_patches=min(counts),
        maximum_data_patches=max(counts),
        mean_bytes_per_patch=sequence_length / float(np.mean(counts)),
        median_patch_length=float(np.median(lengths)),
        p95_patch_length=float(np.percentile(lengths, 95)),
        maximum_patch_length=max(lengths),
        padded_data_width=padded_data_width,
        padding_slots=padding_slots,
        padding_slot_rate=padding_slots / padded_slots if padded_slots else 0.0,
        internal_codepoint_boundaries=internal,
        total_noninitial_boundaries=total_boundaries,
        internal_codepoint_boundary_rate=(
            internal / total_boundaries if total_boundaries else math.nan
        ),
    )
