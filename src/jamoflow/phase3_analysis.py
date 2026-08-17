"""Preregistered Korean strata and multiplicity helpers for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Phase3Stratum:
    name: str
    definition: str
    selected: np.ndarray

    def metadata(self) -> dict[str, int | float | str]:
        count = int(self.selected.sum())
        return {
            "name": self.name,
            "definition": self.definition,
            "sequences": count,
            "sequence_fraction": count / len(self.selected),
        }


def phase3_test_strata(
    stream_data: bytes,
    boundary_masks: np.ndarray,
    *,
    sequence_length: int = 512,
) -> tuple[dict[str, Phase3Stratum], dict[str, object]]:
    """Build the strata fixed in the Phase 3 protocol.

    UTF-8 fragments at arbitrary byte-window edges are ignored only for the
    descriptive text features. The global prefix mask separately preserves
    whether the byte window began inside a codepoint.
    """

    if sequence_length <= 0 or len(stream_data) % sequence_length:
        raise ValueError("stream must contain complete positive-length windows")
    sequence_count = len(stream_data) // sequence_length
    if boundary_masks.shape != (sequence_count, sequence_length):
        raise ValueError("boundary masks have an unexpected shape")

    hangul_byte_fraction = np.zeros(sequence_count, dtype=np.float64)
    latin_present = np.zeros(sequence_count, dtype=bool)
    newline_present = np.zeros(sequence_count, dtype=bool)
    whitespace_rate = np.zeros(sequence_count, dtype=np.float64)

    for index in range(sequence_count):
        start = index * sequence_length
        chunk = stream_data[start : start + sequence_length]
        text = chunk.decode("utf-8", errors="ignore")
        hangul_byte_fraction[index] = (
            sum(
                len(character.encode("utf-8"))
                for character in text
                if "\uac00" <= character <= "\ud7a3"
            )
            / sequence_length
        )
        latin_present[index] = any(
            ("A" <= character <= "Z") or ("a" <= character <= "z")
            for character in text
        )
        newline_present[index] = b"\n" in chunk or b"\r" in chunk
        whitespace_rate[index] = (
            sum(character.isspace() for character in text) / len(text)
            if text
            else 0.0
        )

    whitespace_order = np.lexsort((np.arange(sequence_count), whitespace_rate))
    whitespace_tercile = np.empty(sequence_count, dtype=np.int8)
    for label, indices in enumerate(np.array_split(whitespace_order, 3), start=1):
        whitespace_tercile[indices] = label

    raw: dict[str, tuple[str, np.ndarray]] = {
        "hangul_byte_fraction_lt_25": (
            "precomposed-Hangul UTF-8 bytes are less than 25% of window bytes",
            hangul_byte_fraction < 0.25,
        ),
        "hangul_byte_fraction_25_to_75": (
            "precomposed-Hangul UTF-8 bytes are at least 25% and below 75%",
            (hangul_byte_fraction >= 0.25) & (hangul_byte_fraction < 0.75),
        ),
        "hangul_byte_fraction_ge_75": (
            "precomposed-Hangul UTF-8 bytes are at least 75% of window bytes",
            hangul_byte_fraction >= 0.75,
        ),
        "ascii_latin_present": (
            "contains at least one ASCII Latin letter",
            latin_present,
        ),
        "ascii_latin_absent": (
            "contains no ASCII Latin letter",
            ~latin_present,
        ),
        "newline_present": (
            "contains carriage-return or line-feed byte",
            newline_present,
        ),
        "newline_absent": (
            "contains neither carriage-return nor line-feed byte",
            ~newline_present,
        ),
        "starts_at_codepoint_boundary": (
            "global UTF-8 parser is complete at byte-window start",
            boundary_masks[:, 0].astype(bool),
        ),
        "starts_inside_codepoint": (
            "byte window starts inside a global UTF-8 codepoint",
            ~boundary_masks[:, 0].astype(bool),
        ),
    }
    for tercile in range(1, 4):
        raw[f"whitespace_rate_t{tercile}"] = (
            f"stable-rank Unicode-whitespace-rate tercile {tercile}",
            whitespace_tercile == tercile,
        )

    strata = {
        name: Phase3Stratum(name, definition, selected)
        for name, (definition, selected) in raw.items()
    }
    metadata = {
        "sequence_count": sequence_count,
        "sequence_length": sequence_length,
        "classification_decode_errors": "ignored only at incomplete window edges",
        "hangul_byte_fraction": {
            "minimum": float(hangul_byte_fraction.min()),
            "median": float(np.median(hangul_byte_fraction)),
            "maximum": float(hangul_byte_fraction.max()),
        },
        "whitespace_rate": {
            "minimum": float(whitespace_rate.min()),
            "median": float(np.median(whitespace_rate)),
            "maximum": float(whitespace_rate.max()),
            "tercile_assignment": "stable rank by (rate, sequence index)",
        },
        "strata": {name: value.metadata() for name, value in strata.items()},
    }
    return strata, metadata


def empirical_nonnegative_bootstrap_tail(estimates: Sequence[float]) -> float:
    """Return the add-one bootstrap mass at a nonnegative effect.

    This percentile-tail diagnostic is not labeled an exact calibrated
    hypothesis-test p-value. Phase 3 uses it only to order the step-down
    percentile bounds; the gate is defined by those bounds, a minimum effect,
    and seed signs.
    """

    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap estimates must be a finite non-empty vector")
    return (int((values >= 0).sum()) + 1) / (len(values) + 1)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction."""

    maximum_iterations = 256
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    value = d
    for iteration in range(1, maximum_iterations + 1):
        doubled = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        value *= d * c
        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) <= epsilon:
            return value
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not (a > 0 and b > 0 and 0 <= x <= 1):
        raise ValueError("regularized beta requires a,b>0 and x in [0,1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        result = factor * _beta_continued_fraction(a, b, x) / a
    else:
        result = 1.0 - (
            factor * _beta_continued_fraction(b, a, 1.0 - x) / b
        )
    return min(1.0, max(0.0, result))


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """Return a Student-t CDF without adding a SciPy dependency."""

    if degrees_of_freedom <= 0 or not math.isfinite(value):
        raise ValueError("Student-t input must be finite with positive df")
    if value == 0:
        return 0.5
    ratio = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        ratio,
        degrees_of_freedom / 2.0,
        0.5,
    )
    return tail if value < 0 else 1.0 - tail


def paired_seed_lower_t_pvalue(values: Sequence[float]) -> float:
    """Return the one-sided paired-seed t p-value for a negative mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired-seed t test needs at least two finite values")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1))
    if standard_deviation == 0:
        if mean < 0:
            return 0.0
        if mean > 0:
            return 1.0
        return 0.5
    statistic = mean / (standard_deviation / math.sqrt(len(array)))
    return student_t_cdf(statistic, len(array) - 1)


def hierarchical_paired_bootstrap_estimates(
    paired_sequence_differences_nats: Sequence[np.ndarray],
    *,
    targets_per_sequence: int,
    repetitions: int = 10_000,
    seed: int = 20_260_810,
    chunk_size: int = 128,
) -> np.ndarray:
    """Return crossed seed-by-sequence paired BPB bootstrap replicates.

    Phase 3 evaluates every initialization on the same held-out windows, so
    seeds and sequences are crossed rather than sequences being nested inside
    a seed. Each replicate therefore draws one seed sample and one shared test-
    sequence sample, then averages their Cartesian crossing. Reusing the same
    sequence indices across selected seeds preserves cross-seed correlation in
    example difficulty.
    """

    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in paired_sequence_differences_nats
    ]
    if len(arrays) < 2:
        raise ValueError("hierarchical bootstrap needs at least two seeds")
    if any(values.ndim != 1 or not len(values) for values in arrays):
        raise ValueError("each seed must contain a non-empty vector")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("paired differences must be finite")
    sequence_counts = {len(values) for values in arrays}
    if len(sequence_counts) != 1:
        raise ValueError("crossed bootstrap requires the same sequences per seed")
    if targets_per_sequence <= 0 or repetitions <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap sizes and target count must be positive")

    rng = np.random.default_rng(seed)
    seed_count = len(arrays)
    sequence_count = sequence_counts.pop()
    scale = targets_per_sequence * np.log(2.0)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, chunk_size):
        size = min(chunk_size, repetitions - start)
        selected_seeds = rng.integers(0, seed_count, size=(size, seed_count))
        selected_sequences = rng.integers(
            0,
            sequence_count,
            size=(size, sequence_count),
        )
        source_means = np.empty((size, seed_count), dtype=np.float64)
        for source_seed, values in enumerate(arrays):
            source_means[:, source_seed] = values[selected_sequences].mean(
                axis=1
            )
        crossed_means = np.take_along_axis(
            source_means,
            selected_seeds,
            axis=1,
        ).mean(axis=1)
        estimates[start : start + size] = crossed_means / scale
    return estimates


def holm_step_down_adjusted_values(
    values: Mapping[str, float],
) -> dict[str, float]:
    """Apply the monotone Holm step-down transform to p-values in [0, 1]."""

    if not values:
        raise ValueError("at least one value is required")
    if any(not 0 <= value <= 1 for value in values.values()):
        raise ValueError("values must lie in [0, 1]")
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted
