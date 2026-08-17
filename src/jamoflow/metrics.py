"""Boundary-quality, structural, and reference runtime metrics."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Sequence

from .corpus import Record
from .entropy import PositionScore
from .patching import BoundaryPolicy
from .unicode_audit import is_cjk_ideograph, is_hangul_syllable
from .utf8 import codepoint_spans, scan_prefix_states


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    record: Record
    scores: tuple[PositionScore, ...]


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    record_ids: tuple[str, ...]
    ranked_positions: tuple[tuple[float, tuple[int, int]], ...]
    top_decile_keys: frozenset[tuple[int, int]]
    high_entropy_positions_by_record: tuple[tuple[int, ...], ...]
    codepoint_boundary_masks: tuple[bytes, ...]
    hangul_interior_masks: tuple[bytes, ...]
    cjk_interior_masks: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class PolicyMetrics:
    comparison_group: str
    role: str
    policy: str
    total_bytes: int
    patches: int
    average_patch_bytes: float
    median_patch_bytes: float
    p95_patch_bytes: float
    max_patch_bytes: int
    boundaries_per_kib: float
    mean_boundary_entropy_bits: float | None
    mean_boundary_surprisal_bits: float | None
    oracle_entropy_capture_ratio: float | None
    top_budget_overlap: float | None
    top_decile_entropy_recall: float | None
    mean_high_entropy_patch_lag_bytes: float | None
    p95_high_entropy_patch_lag_bytes: float | None
    boundaries_inside_codepoint: int
    boundaries_inside_codepoint_rate: float
    boundaries_inside_hangul_syllable: int
    boundaries_inside_hangul_syllable_rate: float
    boundaries_inside_cjk_ideograph: int
    boundaries_inside_cjk_ideograph_rate: float
    score_evaluations: int
    score_evaluations_per_byte: float
    policy_runtime_median_ns_per_byte: float
    policy_runtime_p95_ns_per_byte: float
    bootstrap_95: dict[str, tuple[float, float]] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil(percentile * len(ordered)) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


def _patch_lengths(data_length: int, boundaries: Sequence[int]) -> list[int]:
    if data_length == 0:
        return []
    ends = [*boundaries[1:], data_length]
    return [end - start for start, end in zip(boundaries, ends, strict=True)]


def build_evaluation_context(
    records: Sequence[ScoredRecord],
) -> EvaluationContext:
    all_positions: list[tuple[float, tuple[int, int]]] = []
    codepoint_masks: list[bytes] = []
    hangul_masks: list[bytes] = []
    cjk_masks: list[bytes] = []

    for record_index, scored in enumerate(records):
        data = scored.record.raw
        all_positions.extend(
            (scored.scores[position].entropy_bits, (record_index, position))
            for position in range(1, len(data))
        )

        states = scan_prefix_states(data)
        codepoint_mask = bytearray(len(data) + 1)
        for position, state in enumerate(states):
            codepoint_mask[position] = state.at_codepoint_boundary

        hangul_mask = bytearray(len(data) + 1)
        cjk_mask = bytearray(len(data) + 1)
        for span in codepoint_spans(data):
            if not span.valid or span.codepoint is None:
                continue
            if is_hangul_syllable(span.codepoint):
                hangul_mask[span.start + 1 : span.end] = b"\x01" * max(
                    0, span.end - span.start - 1
                )
            if is_cjk_ideograph(span.codepoint):
                cjk_mask[span.start + 1 : span.end] = b"\x01" * max(
                    0, span.end - span.start - 1
                )
        codepoint_masks.append(bytes(codepoint_mask))
        hangul_masks.append(bytes(hangul_mask))
        cjk_masks.append(bytes(cjk_mask))

    all_positions.sort(reverse=True)
    ranked_positions = tuple(all_positions)
    if ranked_positions:
        ascending_rank = math.ceil(0.90 * len(ranked_positions)) - 1
        descending_index = len(ranked_positions) - 1 - ascending_rank
        high_threshold = ranked_positions[descending_index][0]
        top_decile_keys = frozenset(
            key for entropy, key in ranked_positions if entropy >= high_threshold
        )
    else:
        top_decile_keys = frozenset()

    positions_by_record: list[list[int]] = [[] for _ in records]
    for record_index, position in top_decile_keys:
        positions_by_record[record_index].append(position)

    return EvaluationContext(
        record_ids=tuple(scored.record.record_id for scored in records),
        ranked_positions=ranked_positions,
        top_decile_keys=top_decile_keys,
        high_entropy_positions_by_record=tuple(
            tuple(sorted(positions)) for positions in positions_by_record
        ),
        codepoint_boundary_masks=tuple(codepoint_masks),
        hangul_interior_masks=tuple(hangul_masks),
        cjk_interior_masks=tuple(cjk_masks),
    )


def average_patch_bytes(
    records: Sequence[ScoredRecord],
    policy: BoundaryPolicy,
) -> float:
    total_bytes = 0
    total_patches = 0
    for scored in records:
        boundaries = policy.boundaries(scored.record.raw, scored.scores)
        total_bytes += len(scored.record.raw)
        total_patches += len(boundaries)
    if total_patches == 0:
        return math.inf
    return total_bytes / total_patches


def _runtime_samples(
    records: Sequence[ScoredRecord],
    policy: BoundaryPolicy,
    repeats: int,
) -> tuple[float, float]:
    total_bytes = sum(len(scored.record.raw) for scored in records)
    if total_bytes == 0:
        return (0.0, 0.0)

    for scored in records:
        policy.boundaries(scored.record.raw, scored.scores)

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for scored in records:
            policy.boundaries(scored.record.raw, scored.scores)
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / total_bytes)
    return statistics.median(samples), _percentile(samples, 0.95)


def make_bootstrap_weights(
    record_count: int,
    repeats: int,
    seed: int,
):
    """Create deterministic record-resampling weights.

    NumPy is an optional research dependency so the base audit remains usable
    with the Python standard library when bootstrap intervals are disabled.
    """

    if repeats <= 0:
        return None
    if record_count <= 0:
        raise ValueError("record_count must be positive")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "bootstrap intervals require the 'research' optional dependencies"
        ) from exc

    generator = np.random.default_rng(seed)
    probabilities = np.full(record_count, 1.0 / record_count)
    return generator.multinomial(
        record_count,
        probabilities,
        size=repeats,
    ).astype(np.int16)


def _bootstrap_ratio_interval(
    weights,
    numerators: Sequence[float | int],
    denominators: Sequence[float | int],
) -> tuple[float, float] | None:
    if weights is None:
        return None
    import numpy as np

    numerator_samples = weights @ np.asarray(numerators, dtype=np.float64)
    denominator_samples = weights @ np.asarray(denominators, dtype=np.float64)
    valid = denominator_samples > 0
    if not np.any(valid):
        return None
    ratios = numerator_samples[valid] / denominator_samples[valid]
    low, high = np.quantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def evaluate_policy(
    records: Sequence[ScoredRecord],
    policy: BoundaryPolicy,
    comparison_group: str,
    role: str,
    runtime_repeats: int = 7,
    context: EvaluationContext | None = None,
    bootstrap_weights=None,
) -> PolicyMetrics:
    context = context or build_evaluation_context(records)
    if context.record_ids != tuple(scored.record.record_id for scored in records):
        raise ValueError("evaluation context does not match scored records")
    total_bytes = sum(len(scored.record.raw) for scored in records)
    all_patch_lengths: list[int] = []
    selected_positions: set[tuple[int, int]] = set()
    boundary_entropies: list[float] = []
    boundary_surprisals: list[float] = []
    boundaries_inside_codepoint = 0
    boundaries_inside_hangul = 0
    boundaries_inside_cjk = 0
    total_noninitial_boundaries = 0
    score_evaluations = 0
    boundaries_by_record: list[tuple[int, ...]] = []
    record_bytes: list[int] = []
    record_patches: list[int] = []
    record_noninitial_boundaries: list[int] = []
    record_boundary_entropy_sum: list[float] = []
    record_codepoint_splits: list[int] = []
    record_score_evaluations: list[int] = []

    for record_index, scored in enumerate(records):
        data = scored.record.raw
        boundaries = policy.boundaries(data, scored.scores)
        boundaries_by_record.append(boundaries)
        record_bytes.append(len(data))
        record_patches.append(len(boundaries))
        all_patch_lengths.extend(_patch_lengths(len(data), boundaries))
        evaluations = policy.score_evaluations(data)
        score_evaluations += evaluations
        record_score_evaluations.append(evaluations)

        local_noninitial = 0
        local_entropy_sum = 0.0
        local_codepoint_splits = 0

        for boundary in boundaries:
            if boundary == 0:
                continue
            total_noninitial_boundaries += 1
            local_noninitial += 1
            key = (record_index, boundary)
            selected_positions.add(key)
            boundary_entropies.append(scored.scores[boundary].entropy_bits)
            local_entropy_sum += scored.scores[boundary].entropy_bits
            boundary_surprisals.append(scored.scores[boundary].surprisal_bits)
            if not context.codepoint_boundary_masks[record_index][boundary]:
                boundaries_inside_codepoint += 1
                local_codepoint_splits += 1
            if context.hangul_interior_masks[record_index][boundary]:
                boundaries_inside_hangul += 1
            if context.cjk_interior_masks[record_index][boundary]:
                boundaries_inside_cjk += 1
        record_noninitial_boundaries.append(local_noninitial)
        record_boundary_entropy_sum.append(local_entropy_sum)
        record_codepoint_splits.append(local_codepoint_splits)

    oracle_capture_ratio: float | None = None
    top_budget_overlap: float | None = None
    if total_noninitial_boundaries and context.ranked_positions:
        budget = min(total_noninitial_boundaries, len(context.ranked_positions))
        oracle = context.ranked_positions[:budget]
        oracle_keys = {key for _, key in oracle}
        oracle_entropy = sum(entropy for entropy, _ in oracle)
        selected_entropy = sum(boundary_entropies)
        if oracle_entropy > 0:
            oracle_capture_ratio = selected_entropy / oracle_entropy
        top_budget_overlap = len(selected_positions & oracle_keys) / budget

    top_decile_recall: float | None = None
    mean_lag: float | None = None
    p95_lag: float | None = None
    record_high_entropy_count = [0] * len(records)
    record_selected_high_entropy_count = [0] * len(records)
    record_lag_sum = [0] * len(records)
    if context.ranked_positions:
        high_entropy_keys = context.top_decile_keys
        if high_entropy_keys:
            top_decile_recall = len(selected_positions & high_entropy_keys) / len(
                high_entropy_keys
            )
            lags: list[int] = []
            for record_index, positions in enumerate(
                context.high_entropy_positions_by_record
            ):
                boundaries = boundaries_by_record[record_index]
                boundary_set = set(boundaries)
                record_high_entropy_count[record_index] = len(positions)
                record_selected_high_entropy_count[record_index] = sum(
                    position in boundary_set for position in positions
                )
                for position in positions:
                    preceding_index = bisect_right(boundaries, position) - 1
                    preceding_boundary = boundaries[max(preceding_index, 0)]
                    lags.append(position - preceding_boundary)
                    record_lag_sum[record_index] += position - preceding_boundary
            mean_lag = statistics.fmean(lags)
            p95_lag = _percentile(lags, 0.95)

    median_runtime, p95_runtime = _runtime_samples(
        records,
        policy,
        repeats=runtime_repeats,
    )
    patches = len(all_patch_lengths)
    denominator = max(total_noninitial_boundaries, 1)

    bootstrap_95 = None
    if bootstrap_weights is not None:
        interval_inputs = {
            "average_patch_bytes": (record_bytes, record_patches),
            "mean_boundary_entropy_bits": (
                record_boundary_entropy_sum,
                record_noninitial_boundaries,
            ),
            "top_decile_entropy_recall": (
                record_selected_high_entropy_count,
                record_high_entropy_count,
            ),
            "mean_high_entropy_patch_lag_bytes": (
                record_lag_sum,
                record_high_entropy_count,
            ),
            "boundaries_inside_codepoint_rate": (
                record_codepoint_splits,
                record_noninitial_boundaries,
            ),
            "score_evaluations_per_byte": (
                record_score_evaluations,
                record_bytes,
            ),
        }
        bootstrap_95 = {
            name: interval
            for name, (numerators, denominators) in interval_inputs.items()
            if (
                interval := _bootstrap_ratio_interval(
                    bootstrap_weights,
                    numerators,
                    denominators,
                )
            )
            is not None
        }

    return PolicyMetrics(
        comparison_group=comparison_group,
        role=role,
        policy=policy.name,
        total_bytes=total_bytes,
        patches=patches,
        average_patch_bytes=total_bytes / patches if patches else math.inf,
        median_patch_bytes=statistics.median(all_patch_lengths)
        if all_patch_lengths
        else math.inf,
        p95_patch_bytes=_percentile(all_patch_lengths, 0.95)
        if all_patch_lengths
        else math.inf,
        max_patch_bytes=max(all_patch_lengths, default=0),
        boundaries_per_kib=(total_noninitial_boundaries * 1024 / total_bytes)
        if total_bytes
        else 0.0,
        mean_boundary_entropy_bits=statistics.fmean(boundary_entropies)
        if boundary_entropies
        else None,
        mean_boundary_surprisal_bits=statistics.fmean(boundary_surprisals)
        if boundary_surprisals
        else None,
        oracle_entropy_capture_ratio=oracle_capture_ratio,
        top_budget_overlap=top_budget_overlap,
        top_decile_entropy_recall=top_decile_recall,
        mean_high_entropy_patch_lag_bytes=mean_lag,
        p95_high_entropy_patch_lag_bytes=p95_lag,
        boundaries_inside_codepoint=boundaries_inside_codepoint,
        boundaries_inside_codepoint_rate=boundaries_inside_codepoint / denominator,
        boundaries_inside_hangul_syllable=boundaries_inside_hangul,
        boundaries_inside_hangul_syllable_rate=boundaries_inside_hangul
        / denominator,
        boundaries_inside_cjk_ideograph=boundaries_inside_cjk,
        boundaries_inside_cjk_ideograph_rate=boundaries_inside_cjk / denominator,
        score_evaluations=score_evaluations,
        score_evaluations_per_byte=score_evaluations / total_bytes
        if total_bytes
        else 0.0,
        policy_runtime_median_ns_per_byte=median_runtime,
        policy_runtime_p95_ns_per_byte=p95_runtime,
        bootstrap_95=bootstrap_95,
    )
