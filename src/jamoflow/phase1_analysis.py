"""Statistical helpers for the preregistered Phase 1 paired experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .phase1 import patch_boundaries_from_lengths
from .utf8 import codepoint_spans


# Two-sided 95% Student-t critical values. Phase 1 has five paired seeds
# (four degrees of freedom), but keeping the small table makes the helper safe
# for sensitivity runs without adding SciPy as a research dependency.
_T_975 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
    21: 2.079614,
    22: 2.073873,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
    30: 2.042272,
}


@dataclass(frozen=True, slots=True)
class PairedInterval:
    count: int
    mean: float
    sample_standard_deviation: float
    standard_error: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    repetitions: int
    seed: int
    resampling_design: str
    mean: float
    median: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_t_interval(values: Sequence[float]) -> PairedInterval:
    """Return a two-sided 95% paired-t interval over seed-level effects."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError("paired interval needs at least two one-dimensional values")
    if not np.isfinite(array).all():
        raise ValueError("paired values must be finite")
    count = len(array)
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1))
    standard_error = sample_sd / math.sqrt(count)
    degrees = count - 1
    critical = _T_975.get(degrees, 1.959964)
    half_width = critical * standard_error
    return PairedInterval(
        count=count,
        mean=mean,
        sample_standard_deviation=sample_sd,
        standard_error=standard_error,
        lower=mean - half_width,
        upper=mean + half_width,
    )


def hierarchical_paired_bootstrap(
    paired_sequence_differences_nats: Sequence[np.ndarray],
    targets_per_sequence: int,
    repetitions: int = 10_000,
    seed: int = 20_260_810,
    chunk_size: int = 250,
) -> BootstrapInterval:
    """Bootstrap crossed seed-by-sequence paired BPB effects.

    Every seed evaluates the same held-out sequences. A replicate samples seeds
    with replacement and one shared sequence-index vector with replacement,
    then averages their crossing. This preserves policy pairing and the shared
    example-difficulty correlation across initializations.
    """

    if targets_per_sequence <= 0:
        raise ValueError("targets_per_sequence must be positive")
    if repetitions <= 0 or chunk_size <= 0:
        raise ValueError("repetitions and chunk size must be positive")
    arrays = [np.asarray(values, dtype=np.float64) for values in paired_sequence_differences_nats]
    if len(arrays) < 2:
        raise ValueError("hierarchical bootstrap needs at least two seeds")
    if any(values.ndim != 1 or len(values) == 0 for values in arrays):
        raise ValueError("each seed must have a non-empty one-dimensional array")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("sequence differences must be finite")
    sequence_counts = {len(values) for values in arrays}
    if len(sequence_counts) != 1:
        raise ValueError("crossed bootstrap requires the same sequences per seed")

    rng = np.random.default_rng(seed)
    seed_count = len(arrays)
    sequence_count = sequence_counts.pop()
    scale = targets_per_sequence * math.log(2)
    estimates = np.empty(repetitions, dtype=np.float64)

    for start in range(0, repetitions, chunk_size):
        size = min(chunk_size, repetitions - start)
        chosen_seeds = rng.integers(0, seed_count, size=(size, seed_count))
        chosen_sequences = rng.integers(
            0,
            sequence_count,
            size=(size, sequence_count),
        )
        source_means = np.empty((size, seed_count), dtype=np.float64)
        for source_seed, values in enumerate(arrays):
            source_means[:, source_seed] = values[chosen_sequences].mean(axis=1)
        estimates[start : start + size] = (
            np.take_along_axis(source_means, chosen_seeds, axis=1).mean(axis=1)
            / scale
        )

    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    return BootstrapInterval(
        repetitions=repetitions,
        seed=seed,
        resampling_design="crossed seeds x shared test sequences",
        mean=float(estimates.mean()),
        median=float(median),
        lower=float(lower),
        upper=float(upper),
    )


def numeric_summary(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("summary needs a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("summary values must be finite")
    return {
        "count": len(array),
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "sample_standard_deviation": (
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate_numeric_mappings(
    records: Sequence[Mapping[str, float | int]],
) -> dict[str, dict[str, float | int | list[float]]]:
    """Summarize numeric fields common to all flat mappings."""

    if not records:
        raise ValueError("at least one record is required")
    keys = set(records[0])
    for record in records[1:]:
        keys &= set(record)
    result: dict[str, dict[str, float | int | list[float]]] = {}
    for key in sorted(keys):
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            result[key] = numeric_summary([float(value) for value in values])
    return result


def _is_cjk_ideograph(codepoint: int) -> bool:
    ranges = (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F),
        (0x30000, 0x3134F),
        (0x31350, 0x323AF),
    )
    return any(start <= codepoint <= end for start, end in ranges)


def boundary_unicode_diagnostics(
    patch_lengths: np.ndarray,
    stream_data: bytes,
    sequence_length: int,
) -> dict[str, float | int]:
    """Classify noninitial patch starts that split valid UTF-8 codepoints."""

    if patch_lengths.ndim != 2:
        raise ValueError("patch lengths must be a two-dimensional matrix")
    if len(stream_data) != len(patch_lengths) * sequence_length:
        raise ValueError("stream length does not match the patch matrix")

    # 0=codepoint boundary, 1=precomposed Hangul syllable interior,
    # 2=CJK ideograph interior, 3=other valid codepoint interior. Positions in
    # an incomplete final codepoint remain 0 and are accounted as unclassified.
    interior_class = np.zeros(len(stream_data), dtype=np.uint8)
    for span in codepoint_spans(stream_data):
        if not span.valid or span.codepoint is None or span.end - span.start <= 1:
            continue
        if 0xAC00 <= span.codepoint <= 0xD7A3:
            value = 1
        elif _is_cjk_ideograph(span.codepoint):
            value = 2
        else:
            value = 3
        interior_class[span.start + 1 : span.end] = value

    relative = patch_boundaries_from_lengths(patch_lengths)
    offsets = np.arange(len(relative), dtype=np.int64)[:, None] * sequence_length
    selected = interior_class[offsets + relative]
    total = int(selected.size)
    hangul = int((selected == 1).sum())
    cjk = int((selected == 2).sum())
    other = int((selected == 3).sum())
    classified_internal = hangul + cjk + other

    return {
        "total_noninitial_boundaries": total,
        "inside_precomposed_hangul_syllable_count": hangul,
        "inside_precomposed_hangul_syllable_rate": hangul / total,
        "inside_cjk_ideograph_count": cjk,
        "inside_cjk_ideograph_rate": cjk / total,
        "inside_other_codepoint_count": other,
        "inside_other_codepoint_rate": other / total,
        "classified_internal_codepoint_count": classified_internal,
        "classified_internal_codepoint_rate": classified_internal / total,
    }


def nearest_boundary_displacement(
    first_patch_lengths: np.ndarray,
    second_patch_lengths: np.ndarray,
) -> dict[str, float | int]:
    """Return symmetric nearest-neighbor displacement between two segmentations."""

    if first_patch_lengths.shape != second_patch_lengths.shape:
        raise ValueError("patch matrices must have equal shapes")
    first = patch_boundaries_from_lengths(first_patch_lengths)
    second = patch_boundaries_from_lengths(second_patch_lengths)
    distances: list[int] = []
    nonzero: list[int] = []
    for left, right in zip(first, second, strict=True):
        for source, target in ((left, right), (right, left)):
            local = np.abs(source[:, None] - target[None, :]).min(axis=1)
            distances.extend(int(value) for value in local)
            nonzero.extend(int(value) for value in local if value)
    values = np.asarray(distances, dtype=np.float64)
    changed = np.asarray(nonzero, dtype=np.float64)
    return {
        "symmetric_boundary_observations": len(values),
        "exact_match_rate": float((values == 0).mean()),
        "mean_nearest_displacement_bytes": float(values.mean()),
        "median_nearest_displacement_bytes": float(np.median(values)),
        "p95_nearest_displacement_bytes": float(np.percentile(values, 95)),
        "changed_boundary_observations": len(changed),
        "mean_changed_nearest_displacement_bytes": (
            float(changed.mean()) if len(changed) else 0.0
        ),
        "p95_changed_nearest_displacement_bytes": (
            float(np.percentile(changed, 95)) if len(changed) else 0.0
        ),
    }
