"""Private-safe case selection and paired statistics for inference evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class InferenceCases:
    prompts: np.ndarray
    replay_continuations: np.ndarray
    candidate_rows: int
    unique_candidate_prompts: int
    unique_candidate_clusters: int | None
    selected_unique_clusters: int | None
    prompt_length: int
    continuation_length: int

    def public_metadata(self) -> dict[str, int]:
        metadata = {
            "selected_cases": len(self.prompts),
            "candidate_rows": self.candidate_rows,
            "unique_candidate_prompts": self.unique_candidate_prompts,
            "prompt_length_bytes": self.prompt_length,
            "continuation_length_bytes": self.continuation_length,
        }
        if self.unique_candidate_clusters is not None:
            metadata["unique_candidate_clusters"] = self.unique_candidate_clusters
            metadata["selected_unique_clusters"] = self.selected_unique_clusters
        return metadata


def _hangul_heavy(raw: bytes) -> bool:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in letters)
    return hangul / len(letters) >= 0.8


def select_inference_cases(
    inputs: np.ndarray,
    boundary_masks: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    case_count: int = 64,
    prompt_length: int = 128,
    continuation_length: int = 128,
) -> InferenceCases:
    """Select deterministic held-out prompt/replay pairs without public hashes."""

    if inputs.ndim != 2 or inputs.shape != boundary_masks.shape:
        raise ValueError("inputs and boundary masks must be equal matrices")
    clusters = None if cluster_ids is None else np.asarray(cluster_ids)
    if clusters is not None and (
        clusters.shape != (len(inputs),)
        or not np.issubdtype(clusters.dtype, np.integer)
        or np.any(clusters < 0)
    ):
        raise ValueError("cluster IDs must be one nonnegative integer per row")
    if case_count <= 0 or prompt_length <= 0 or continuation_length <= 0:
        raise ValueError("inference case dimensions must be positive")
    selected_length = prompt_length + continuation_length
    if selected_length > inputs.shape[1]:
        raise ValueError("inference case exceeds the available sequence length")

    candidates: list[tuple[bytes, bytes, int | None]] = []
    for row_index, (row, boundaries) in enumerate(
        zip(inputs, boundary_masks, strict=True)
    ):
        if not (
            bool(boundaries[0])
            and bool(boundaries[prompt_length])
            and (
                selected_length == inputs.shape[1]
                or bool(boundaries[selected_length])
            )
        ):
            continue
        prompt = bytes(row[:prompt_length])
        if not _hangul_heavy(prompt):
            continue
        continuation = bytes(row[prompt_length:selected_length])
        cluster = None if clusters is None else int(clusters[row_index])
        candidates.append((prompt, continuation, cluster))

    by_prompt: dict[bytes, tuple[bytes, int | None]] = {}
    for prompt, continuation, cluster in candidates:
        current = by_prompt.get(prompt)
        tie_break = -1 if cluster is None else cluster
        if current is None or (continuation, tie_break) < (
            current[0],
            -1 if current[1] is None else current[1],
        ):
            by_prompt[prompt] = (continuation, cluster)
    unique = [
        (prompt, continuation, cluster)
        for prompt, (continuation, cluster) in by_prompt.items()
    ]
    unique.sort(
        key=lambda item: (
            sha256(
                b"JamoFlow-actual-inference-v1\0" + item[0] + item[1]
            ).digest(),
            item,
        )
    )
    if clusters is None:
        chosen = unique[:case_count]
    else:
        chosen = []
        selected_clusters: set[int] = set()
        for item in unique:
            cluster = item[2]
            if cluster is None:
                raise AssertionError("cluster-aware selection lost a cluster ID")
            if cluster in selected_clusters:
                continue
            chosen.append(item)
            selected_clusters.add(cluster)
            if len(chosen) == case_count:
                break
    if len(chosen) < case_count:
        raise ValueError(
            f"need {case_count} unique inference cases, found {len(chosen)}"
        )
    prompts = np.stack(
        [np.frombuffer(prompt, dtype=np.uint8) for prompt, _, _ in chosen]
    )
    continuations = np.stack(
        [
            np.frombuffer(continuation, dtype=np.uint8)
            for _, continuation, _ in chosen
        ]
    )
    unique_candidate_clusters = (
        None
        if clusters is None
        else len({cluster for _, _, cluster in unique})
    )
    selected_unique_clusters = (
        None
        if clusters is None
        else len({cluster for _, _, cluster in chosen})
    )
    return InferenceCases(
        prompts=prompts,
        replay_continuations=continuations,
        candidate_rows=len(candidates),
        unique_candidate_prompts=len(unique),
        unique_candidate_clusters=unique_candidate_clusters,
        selected_unique_clusters=selected_unique_clusters,
        prompt_length=prompt_length,
        continuation_length=continuation_length,
    )


@dataclass(frozen=True, slots=True)
class PairedPromptLatency:
    prompt_count: int
    repetitions_per_prompt: int
    candidate_median_ms: float
    reference_median_ms: float
    median_latency_reduction: float
    mean_paired_prompt_reduction: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    bootstrap_percentile_95_lower: float
    bootstrap_percentile_95_upper: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def paired_prompt_latency(
    candidate_ms: np.ndarray,
    reference_ms: np.ndarray,
    *,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_811,
) -> PairedPromptLatency:
    """Estimate a ratio of prompt-level medians with paired prompt bootstrap."""

    candidate = np.asarray(candidate_ms, dtype=np.float64)
    reference = np.asarray(reference_ms, dtype=np.float64)
    if (
        candidate.ndim != 2
        or candidate.shape != reference.shape
        or not candidate.size
        or bootstrap_repetitions <= 0
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(reference))
        or np.any(candidate <= 0)
        or np.any(reference <= 0)
    ):
        raise ValueError("paired prompt timings must be equal positive matrices")
    candidate_by_prompt = np.median(candidate, axis=1)
    reference_by_prompt = np.median(reference, axis=1)

    def statistic(indices: np.ndarray) -> np.ndarray:
        candidate_median = np.median(candidate_by_prompt[indices], axis=-1)
        reference_median = np.median(reference_by_prompt[indices], axis=-1)
        return 1 - candidate_median / reference_median

    prompt_count = candidate.shape[0]
    point_indices = np.arange(prompt_count, dtype=np.int64)
    point = float(statistic(point_indices))
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(
        0,
        prompt_count,
        size=(bootstrap_repetitions, prompt_count),
    )
    bootstrap = statistic(sampled)
    paired_reductions = 1 - candidate_by_prompt / reference_by_prompt
    return PairedPromptLatency(
        prompt_count=prompt_count,
        repetitions_per_prompt=candidate.shape[1],
        candidate_median_ms=float(np.median(candidate_by_prompt)),
        reference_median_ms=float(np.median(reference_by_prompt)),
        median_latency_reduction=point,
        mean_paired_prompt_reduction=float(paired_reductions.mean()),
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        bootstrap_percentile_95_lower=float(np.percentile(bootstrap, 2.5)),
        bootstrap_percentile_95_upper=float(np.percentile(bootstrap, 97.5)),
    )


def latency_component_pass(
    summary: PairedPromptLatency | dict[str, Any],
    *,
    minimum_reduction: float = 0.10,
) -> bool:
    values = summary if isinstance(summary, dict) else summary.to_dict()
    return bool(
        float(values["median_latency_reduction"]) >= minimum_reduction
        and float(values["bootstrap_percentile_95_lower"]) > 0
    )


@dataclass(frozen=True, slots=True)
class MultiSeedPairedLatency:
    seed_order: tuple[int, ...]
    seed_count: int
    prompt_count: int
    repetitions_per_prompt: int
    candidate_median_ms: float
    reference_median_ms: float
    crossed_median_latency_reduction: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    bootstrap_design: str
    bootstrap_percentile_95_lower: float
    bootstrap_percentile_95_upper: float
    median_seed_point_reduction: float
    positive_seed_count: int
    per_seed: dict[str, dict[str, int | float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def multiseed_paired_latency(
    candidate_ms: np.ndarray,
    reference_ms: np.ndarray,
    seed_order: tuple[int, ...],
    *,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_812,
    chunk_size: int = 256,
) -> MultiSeedPairedLatency:
    """Crossed seed-by-prompt ratio with repetitions collapsed within prompt."""

    candidate = np.asarray(candidate_ms, dtype=np.float64)
    reference = np.asarray(reference_ms, dtype=np.float64)
    if (
        candidate.ndim != 3
        or candidate.shape != reference.shape
        or candidate.shape[0] != len(seed_order)
        or len(set(seed_order)) != len(seed_order)
        or candidate.shape[0] < 2
        or candidate.shape[1] < 2
        or candidate.shape[2] < 1
        or bootstrap_repetitions <= 0
        or chunk_size <= 0
        or not np.isfinite(candidate).all()
        or not np.isfinite(reference).all()
        or np.any(candidate <= 0)
        or np.any(reference <= 0)
    ):
        raise ValueError(
            "multi-seed timings must be positive equal seed-prompt-repetition arrays"
        )
    candidate_medians = np.median(candidate, axis=2)
    reference_medians = np.median(reference, axis=2)

    def ratio(candidate_values: np.ndarray, reference_values: np.ndarray) -> Any:
        return 1 - (
            np.median(candidate_values, axis=(-2, -1))
            / np.median(reference_values, axis=(-2, -1))
        )

    point = float(ratio(candidate_medians, reference_medians))
    rng = np.random.default_rng(bootstrap_seed)
    seed_count, prompt_count, repetitions = candidate.shape
    bootstrap = np.empty(bootstrap_repetitions, dtype=np.float64)
    for start in range(0, bootstrap_repetitions, chunk_size):
        size = min(chunk_size, bootstrap_repetitions - start)
        selected_seeds = rng.integers(
            0,
            seed_count,
            size=(size, seed_count),
        )
        selected_prompts = rng.integers(
            0,
            prompt_count,
            size=(size, prompt_count),
        )
        candidate_crossed = candidate_medians[
            selected_seeds[:, :, None],
            selected_prompts[:, None, :],
        ]
        reference_crossed = reference_medians[
            selected_seeds[:, :, None],
            selected_prompts[:, None, :],
        ]
        bootstrap[start : start + size] = ratio(
            candidate_crossed,
            reference_crossed,
        )

    per_seed = {
        str(seed): paired_prompt_latency(
            candidate[index],
            reference[index],
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed + index + 1,
        ).to_dict()
        for index, seed in enumerate(seed_order)
    }
    seed_points = np.asarray(
        [
            values["median_latency_reduction"]
            for values in per_seed.values()
        ],
        dtype=np.float64,
    )
    return MultiSeedPairedLatency(
        seed_order=seed_order,
        seed_count=seed_count,
        prompt_count=prompt_count,
        repetitions_per_prompt=repetitions,
        candidate_median_ms=float(np.median(candidate_medians)),
        reference_median_ms=float(np.median(reference_medians)),
        crossed_median_latency_reduction=point,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        bootstrap_design="crossed model seeds x shared prompts",
        bootstrap_percentile_95_lower=float(np.percentile(bootstrap, 2.5)),
        bootstrap_percentile_95_upper=float(np.percentile(bootstrap, 97.5)),
        median_seed_point_reduction=float(np.median(seed_points)),
        positive_seed_count=int((seed_points > 0).sum()),
        per_seed=per_seed,
    )


def multiseed_latency_component_pass(
    summary: MultiSeedPairedLatency | dict[str, Any],
    *,
    minimum_reduction: float = 0.10,
    minimum_positive_seeds: int = 4,
) -> bool:
    values = summary if isinstance(summary, dict) else summary.to_dict()
    return bool(
        float(values["crossed_median_latency_reduction"]) >= minimum_reduction
        and float(values["bootstrap_percentile_95_lower"]) > 0
        and float(values["median_seed_point_reduction"]) >= minimum_reduction
        and int(values["positive_seed_count"]) >= minimum_positive_seeds
    )


def timing_order_schedule(
    seed_order: tuple[int, ...],
    *,
    mode_count: int,
    prompt_count: int,
    repetitions: int,
    random_seed: int = 20_260_811,
) -> np.ndarray:
    """Return a balanced candidate-first flag for every timed paired trial."""

    if (
        len(seed_order) == 0
        or len(set(seed_order)) != len(seed_order)
        or mode_count <= 0
        or prompt_count <= 0
        or repetitions <= 0
    ):
        raise ValueError("timing schedule dimensions must be positive and unique")
    rng = np.random.default_rng(random_seed)
    output = np.empty(
        (len(seed_order), mode_count, prompt_count, repetitions),
        dtype=np.uint8,
    )
    for seed_index in range(len(seed_order)):
        for mode_index in range(mode_count):
            size = prompt_count * repetitions
            balanced = np.arange(size, dtype=np.uint8) % 2
            rng.shuffle(balanced)
            output[seed_index, mode_index] = balanced.reshape(
                prompt_count,
                repetitions,
            )
    return output


def verification_prefix_lengths(
    boundaries: tuple[int, ...],
    prompt_length: int,
    *,
    minimum_positions: int = 16,
    selection_seed: int = 20_260_811,
) -> tuple[int, ...]:
    """Cover every boundary transition and fill a deterministic prefix set."""

    if (
        not boundaries
        or boundaries[0] != 0
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        or boundaries[-1] >= prompt_length
        or prompt_length <= 0
        or minimum_positions <= 0
        or selection_seed < 0
    ):
        raise ValueError("verification boundaries or prompt length are malformed")
    selected = {1, prompt_length}
    for boundary in boundaries:
        for prefix_length in (boundary, boundary + 1, boundary + 2):
            if 1 <= prefix_length <= prompt_length:
                selected.add(prefix_length)
    if len(selected) < min(minimum_positions, prompt_length):
        candidates = [
            length
            for length in range(1, prompt_length + 1)
            if length not in selected
        ]
        candidates.sort(
            key=lambda length: sha256(
                b"JamoFlow-equivalence-position-v1\0"
                + int(selection_seed).to_bytes(8, "little", signed=False)
                + int(length).to_bytes(8, "little", signed=False)
            ).digest()
        )
        needed = min(minimum_positions, prompt_length) - len(selected)
        selected.update(candidates[:needed])
    return tuple(sorted(selected))
