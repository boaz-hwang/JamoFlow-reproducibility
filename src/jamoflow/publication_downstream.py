"""Publication-scale Korean downstream metrics and preregistered gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .publication_model_lock import (
    PUBLICATION_MODEL_KEYS,
    PublicationModelSnapshot,
    canonical_sha256,
    is_sha256,
    named_array_manifest_sha256,
    validate_publication_model_snapshot,
)

from .publication_protocol import (
    DOWNSTREAM_FAMILY_NONINFERIORITY_MARGIN_PP,
    DOWNSTREAM_FAMILY_ONE_SIDED_CONFIDENCE,
    DOWNSTREAM_MINIMUM_KOBEST_INFORMATIVE_TASKS,
    DOWNSTREAM_REFERENCE_FLOOR_ADVANTAGE_PP,
    DOWNSTREAM_REQUIRED_KLUE_INFORMATIVE_TASKS,
    DOWNSTREAM_TASK_GUARD_MARGIN_PP,
    PRIMARY_DOWNSTREAM_TASKS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_DOWNSTREAM_REFERENCE_KEYS,
    PUBLICATION_PRETRAIN_SEEDS,
)


@dataclass(frozen=True, slots=True)
class TaskPredictionComparison:
    task_key: str
    candidate_key: str
    reference_key: str
    gold: tuple[int, ...]
    train_majority_label: int
    candidate_by_seed: Mapping[int, tuple[int, ...]]
    reference_by_seed: Mapping[int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class TaskDownstreamResult:
    task_key: str
    family: str
    reference_key: str
    metric: str
    example_count: int
    uninformed_accuracy: float
    reference_mean_accuracy: float
    reference_floor_gap: float
    reference_floor_bootstrap_lower: float
    required_floor_advantage: float
    informative: bool
    candidate_mean_score: float
    reference_mean_score: float
    candidate_minus_reference: float
    task_guard_margin: float
    task_guard_pass: bool


@dataclass(frozen=True, slots=True)
class FamilyDownstreamResult:
    family: str
    informative_tasks: tuple[str, ...]
    mean_candidate_minus_reference: float
    bootstrap_one_sided_lower: float | None
    confidence: float
    noninferiority_margin: float
    seed_count_within_margin: int
    noninferiority_pass: bool


@dataclass(frozen=True, slots=True)
class PublicationDownstreamGate:
    candidate_key: str
    seed_order: tuple[int, ...]
    bootstrap_repetitions: int
    bootstrap_seed: int
    bootstrap_design: str
    model_snapshots: tuple[PublicationModelSnapshot, ...]
    case_manifest_sha256: str
    prediction_artifact_sha256: str
    prediction_arrays_sha256: str
    tasks: tuple[TaskDownstreamResult, ...]
    families: tuple[FamilyDownstreamResult, ...]
    informative_suite_pass: bool
    individual_task_guard_pass: bool
    family_noninferiority_pass: bool
    overall_pass: bool
    status: str
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def accuracy_score(gold: Sequence[int], predicted: Sequence[int]) -> float:
    gold_array, predicted_array = _validated_label_arrays(gold, predicted)
    return float(np.mean(gold_array == predicted_array))


def macro_f1_score(
    gold: Sequence[int],
    predicted: Sequence[int],
    *,
    label_count: int,
) -> float:
    """Macro F1 over a fixed label set, with zero division mapped to zero."""

    gold_array, predicted_array = _validated_label_arrays(gold, predicted)
    if label_count < 2:
        raise ValueError("macro F1 requires at least two labels")
    if (
        np.any(gold_array < 0)
        or np.any(gold_array >= label_count)
        or np.any(predicted_array < 0)
        or np.any(predicted_array >= label_count)
    ):
        raise ValueError("labels fall outside the fixed task label set")
    scores: list[float] = []
    for label in range(label_count):
        gold_positive = gold_array == label
        predicted_positive = predicted_array == label
        true_positive = int(np.count_nonzero(gold_positive & predicted_positive))
        false_positive = int(np.count_nonzero(~gold_positive & predicted_positive))
        false_negative = int(np.count_nonzero(gold_positive & ~predicted_positive))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _validated_label_arrays(
    gold: Sequence[int],
    predicted: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    gold_array = np.asarray(gold, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    if (
        gold_array.ndim != 1
        or not len(gold_array)
        or gold_array.shape != predicted_array.shape
    ):
        raise ValueError("gold and prediction labels must be equal non-empty vectors")
    return gold_array, predicted_array


def _score(
    metric: str,
    gold: np.ndarray,
    predicted: np.ndarray,
    *,
    label_count: int,
) -> float:
    if metric == "accuracy":
        return accuracy_score(gold, predicted)
    if metric == "macro_f1":
        return macro_f1_score(gold, predicted, label_count=label_count)
    raise ValueError("unsupported publication downstream metric")


def _validate_comparisons(
    comparisons: Mapping[str, TaskPredictionComparison],
    seed_order: tuple[int, ...],
) -> None:
    if set(comparisons) != set(PRIMARY_DOWNSTREAM_TASKS):
        raise ValueError("downstream gate requires the exact primary task suite")
    if seed_order != PUBLICATION_PRETRAIN_SEEDS:
        raise ValueError("downstream gate requires the paired publication seeds")
    candidate_keys = {
        comparison.candidate_key for comparison in comparisons.values()
    }
    if candidate_keys != {PUBLICATION_CANDIDATE_MODEL_KEY}:
        raise ValueError("downstream comparisons require the sealed candidate")
    for key, comparison in comparisons.items():
        spec = PRIMARY_DOWNSTREAM_TASKS[key]
        gold = np.asarray(comparison.gold, dtype=np.int64)
        if (
            comparison.task_key != key
            or not comparison.candidate_key
            or comparison.reference_key
            not in PUBLICATION_DOWNSTREAM_REFERENCE_KEYS
            or gold.ndim != 1
            or not len(gold)
            or comparison.train_majority_label not in range(spec.label_count)
            or np.any(gold < 0)
            or np.any(gold >= spec.label_count)
            or set(comparison.candidate_by_seed) != set(seed_order)
            or set(comparison.reference_by_seed) != set(seed_order)
        ):
            raise ValueError("invalid downstream task comparison")
        for predictions in (
            comparison.candidate_by_seed,
            comparison.reference_by_seed,
        ):
            for seed in seed_order:
                values = np.asarray(predictions[seed], dtype=np.int64)
                if (
                    values.shape != gold.shape
                    or np.any(values < 0)
                    or np.any(values >= spec.label_count)
                ):
                    raise ValueError("downstream predictions are not crossed")


def evaluate_publication_downstream_gate(
    comparisons: Mapping[str, TaskPredictionComparison],
    *,
    model_snapshots: Mapping[str, PublicationModelSnapshot],
    case_manifest_sha256: str,
    prediction_artifact_sha256: str,
    seed_order: tuple[int, ...] = PUBLICATION_PRETRAIN_SEEDS,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_814,
) -> PublicationDownstreamGate:
    """Evaluate floor, task-guard, and family noninferiority without pooling runs."""

    if bootstrap_repetitions <= 0:
        raise ValueError("downstream bootstrap repetitions must be positive")
    _validate_comparisons(comparisons, seed_order)
    candidate_key = next(iter(comparisons.values())).candidate_key
    used_model_keys = {
        candidate_key,
        *(comparison.reference_key for comparison in comparisons.values()),
    }
    if (
        set(model_snapshots) != used_model_keys
        or not all(
            is_sha256(value)
            for value in (case_manifest_sha256, prediction_artifact_sha256)
        )
    ):
        raise ValueError("downstream evidence requires exact model and case locks")
    for key, snapshot in model_snapshots.items():
        validate_publication_model_snapshot(snapshot)
        if snapshot.model_key != key:
            raise ValueError("downstream model snapshot key is inconsistent")
    ordered_snapshots = tuple(
        model_snapshots[key] for key in PUBLICATION_MODEL_KEYS if key in used_model_keys
    )
    prediction_arrays: dict[str, np.ndarray] = {}
    for key in sorted(comparisons):
        comparison = comparisons[key]
        prediction_arrays[f"{key}:gold"] = np.asarray(
            comparison.gold,
            dtype=np.int64,
        )
        prediction_arrays[f"{key}:train_majority_label"] = np.asarray(
            [comparison.train_majority_label],
            dtype=np.int64,
        )
        for seed in seed_order:
            prediction_arrays[f"{key}:candidate:{seed}"] = np.asarray(
                comparison.candidate_by_seed[seed],
                dtype=np.int64,
            )
            prediction_arrays[
                f"{key}:reference:{comparison.reference_key}:{seed}"
            ] = np.asarray(
                comparison.reference_by_seed[seed],
                dtype=np.int64,
            )
    prediction_arrays_sha256 = named_array_manifest_sha256(prediction_arrays)
    floor_margin = DOWNSTREAM_REFERENCE_FLOOR_ADVANTAGE_PP / 100.0
    task_margin = DOWNSTREAM_TASK_GUARD_MARGIN_PP / 100.0
    family_margin = DOWNSTREAM_FAMILY_NONINFERIORITY_MARGIN_PP / 100.0
    lower_quantile = 1.0 - DOWNSTREAM_FAMILY_ONE_SIDED_CONFIDENCE

    task_results: list[TaskDownstreamResult] = []
    point_differences: dict[str, dict[int, float]] = {}
    cached: dict[str, tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]] = {}
    for task_index, key in enumerate(sorted(comparisons)):
        comparison = comparisons[key]
        spec = PRIMARY_DOWNSTREAM_TASKS[key]
        gold = np.asarray(comparison.gold, dtype=np.int64)
        candidates = [
            np.asarray(comparison.candidate_by_seed[seed], dtype=np.int64)
            for seed in seed_order
        ]
        references = [
            np.asarray(comparison.reference_by_seed[seed], dtype=np.int64)
            for seed in seed_order
        ]
        cached[key] = (gold, candidates, references)
        majority_correct = gold == comparison.train_majority_label
        chance = 1.0 / spec.label_count
        uninformed = max(float(np.mean(majority_correct)), chance)
        reference_accuracies = [accuracy_score(gold, values) for values in references]
        reference_mean_accuracy = float(np.mean(reference_accuracies))
        reference_floor_gap = reference_mean_accuracy - uninformed

        floor_rng = np.random.default_rng(bootstrap_seed + task_index + 1)
        floor_estimates = np.empty(bootstrap_repetitions, dtype=np.float64)
        for repetition in range(bootstrap_repetitions):
            selected_seeds = floor_rng.integers(0, len(seed_order), len(seed_order))
            selected_examples = floor_rng.integers(0, len(gold), len(gold))
            reference_accuracy = float(
                np.mean(
                    [
                        np.mean(
                            references[index][selected_examples]
                            == gold[selected_examples]
                        )
                        for index in selected_seeds
                    ]
                )
            )
            sampled_majority = float(np.mean(majority_correct[selected_examples]))
            floor_estimates[repetition] = reference_accuracy - max(
                sampled_majority,
                chance,
            )
        floor_lower = float(np.quantile(floor_estimates, 0.025))
        informative = floor_lower > floor_margin

        candidate_scores = {
            seed: _score(
                spec.primary_metric,
                gold,
                candidates[index],
                label_count=spec.label_count,
            )
            for index, seed in enumerate(seed_order)
        }
        reference_scores = {
            seed: _score(
                spec.primary_metric,
                gold,
                references[index],
                label_count=spec.label_count,
            )
            for index, seed in enumerate(seed_order)
        }
        differences = {
            seed: candidate_scores[seed] - reference_scores[seed]
            for seed in seed_order
        }
        point_differences[key] = differences
        mean_difference = float(np.mean(list(differences.values())))
        task_results.append(
            TaskDownstreamResult(
                task_key=key,
                family=spec.family,
                reference_key=comparison.reference_key,
                metric=spec.primary_metric,
                example_count=len(gold),
                uninformed_accuracy=uninformed,
                reference_mean_accuracy=reference_mean_accuracy,
                reference_floor_gap=reference_floor_gap,
                reference_floor_bootstrap_lower=floor_lower,
                required_floor_advantage=floor_margin,
                informative=informative,
                candidate_mean_score=float(np.mean(list(candidate_scores.values()))),
                reference_mean_score=float(np.mean(list(reference_scores.values()))),
                candidate_minus_reference=mean_difference,
                task_guard_margin=task_margin,
                task_guard_pass=mean_difference >= -task_margin,
            )
        )

    task_results_by_key = {result.task_key: result for result in task_results}
    family_results: list[FamilyDownstreamResult] = []
    for family_index, family in enumerate(("kobest", "klue")):
        informative_keys = tuple(
            key
            for key in sorted(PRIMARY_DOWNSTREAM_TASKS)
            if PRIMARY_DOWNSTREAM_TASKS[key].family == family
            and task_results_by_key[key].informative
        )
        family_seed_differences = {
            seed: (
                float(
                    np.mean(
                        [point_differences[key][seed] for key in informative_keys]
                    )
                )
                if informative_keys
                else 0.0
            )
            for seed in seed_order
        }
        bootstrap_lower: float | None = None
        if informative_keys:
            family_rng = np.random.default_rng(
                bootstrap_seed + 100 + family_index
            )
            estimates = np.empty(bootstrap_repetitions, dtype=np.float64)
            for repetition in range(bootstrap_repetitions):
                selected_seeds = family_rng.integers(
                    0,
                    len(seed_order),
                    len(seed_order),
                )
                task_effects: list[float] = []
                for key in informative_keys:
                    spec = PRIMARY_DOWNSTREAM_TASKS[key]
                    gold, candidates, references = cached[key]
                    selected_examples = family_rng.integers(0, len(gold), len(gold))
                    effects = [
                        _score(
                            spec.primary_metric,
                            gold[selected_examples],
                            candidates[index][selected_examples],
                            label_count=spec.label_count,
                        )
                        - _score(
                            spec.primary_metric,
                            gold[selected_examples],
                            references[index][selected_examples],
                            label_count=spec.label_count,
                        )
                        for index in selected_seeds
                    ]
                    task_effects.append(float(np.mean(effects)))
                estimates[repetition] = float(np.mean(task_effects))
            bootstrap_lower = float(np.quantile(estimates, lower_quantile))
        seed_count_within = sum(
            value >= -family_margin for value in family_seed_differences.values()
        )
        family_results.append(
            FamilyDownstreamResult(
                family=family,
                informative_tasks=informative_keys,
                mean_candidate_minus_reference=float(
                    np.mean(list(family_seed_differences.values()))
                ),
                bootstrap_one_sided_lower=bootstrap_lower,
                confidence=DOWNSTREAM_FAMILY_ONE_SIDED_CONFIDENCE,
                noninferiority_margin=family_margin,
                seed_count_within_margin=seed_count_within,
                noninferiority_pass=bool(
                    bootstrap_lower is not None
                    and bootstrap_lower > -family_margin
                    and seed_count_within >= 2
                ),
            )
        )

    informative_counts = {
        family: sum(
            result.informative for result in task_results if result.family == family
        )
        for family in ("kobest", "klue")
    }
    informative_suite_pass = bool(
        informative_counts["kobest"]
        >= DOWNSTREAM_MINIMUM_KOBEST_INFORMATIVE_TASKS
        and informative_counts["klue"]
        == DOWNSTREAM_REQUIRED_KLUE_INFORMATIVE_TASKS
    )
    task_guard_pass = all(result.task_guard_pass for result in task_results)
    family_pass = all(result.noninferiority_pass for result in family_results)
    overall = informative_suite_pass and task_guard_pass and family_pass
    provisional = PublicationDownstreamGate(
        candidate_key=candidate_key,
        seed_order=seed_order,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        bootstrap_design=(
            "crossed model seeds x shared examples within task; "
            "independent example resamples across tasks"
        ),
        model_snapshots=ordered_snapshots,
        case_manifest_sha256=case_manifest_sha256,
        prediction_artifact_sha256=prediction_artifact_sha256,
        prediction_arrays_sha256=prediction_arrays_sha256,
        tasks=tuple(task_results),
        families=tuple(family_results),
        informative_suite_pass=informative_suite_pass,
        individual_task_guard_pass=task_guard_pass,
        family_noninferiority_pass=family_pass,
        overall_pass=overall,
        status="pass" if overall else "fail_downstream_quality",
        identity_sha256="",
    )
    payload = provisional.to_dict()
    payload.pop("identity_sha256")
    result = PublicationDownstreamGate(
        **{
            **provisional.to_dict(),
            "model_snapshots": ordered_snapshots,
            "tasks": tuple(task_results),
            "families": tuple(family_results),
            "identity_sha256": canonical_sha256(payload),
        }
    )
    validate_publication_downstream_gate(result)
    return result


def validate_publication_downstream_gate(
    evidence: PublicationDownstreamGate,
) -> None:
    if not isinstance(evidence, PublicationDownstreamGate):
        raise ValueError("publication downstream evidence is invalid")
    for snapshot in evidence.model_snapshots:
        validate_publication_model_snapshot(snapshot)
    task_by_key = {task.task_key: task for task in evidence.tasks}
    family_by_key = {family.family: family for family in evidence.families}
    if (
        len(task_by_key) != len(evidence.tasks)
        or len(family_by_key) != len(evidence.families)
    ):
        raise ValueError("publication downstream evidence is inconsistent")
    used_model_keys = {
        evidence.candidate_key,
        *(task.reference_key for task in evidence.tasks),
    }
    snapshot_keys = tuple(snapshot.model_key for snapshot in evidence.model_snapshots)
    expected_snapshot_keys = tuple(
        key for key in PUBLICATION_MODEL_KEYS if key in used_model_keys
    )
    if set(task_by_key) != set(PRIMARY_DOWNSTREAM_TASKS):
        raise ValueError("publication downstream evidence is inconsistent")
    for key, result in task_by_key.items():
        spec = PRIMARY_DOWNSTREAM_TASKS[key]
        numeric = (
            result.uninformed_accuracy,
            result.reference_mean_accuracy,
            result.reference_floor_gap,
            result.reference_floor_bootstrap_lower,
            result.required_floor_advantage,
            result.candidate_mean_score,
            result.reference_mean_score,
            result.candidate_minus_reference,
            result.task_guard_margin,
        )
        if (
            result.family != spec.family
            or result.metric != spec.primary_metric
            or result.reference_key not in PUBLICATION_DOWNSTREAM_REFERENCE_KEYS
            or result.example_count <= 0
            or not all(np.isfinite(value) for value in numeric)
            or any(
                not 0 <= value <= 1
                for value in (
                    result.uninformed_accuracy,
                    result.reference_mean_accuracy,
                    result.candidate_mean_score,
                    result.reference_mean_score,
                )
            )
            or not np.isclose(
                result.reference_floor_gap,
                result.reference_mean_accuracy - result.uninformed_accuracy,
            )
            or not np.isclose(
                result.candidate_minus_reference,
                result.candidate_mean_score - result.reference_mean_score,
            )
            or result.required_floor_advantage
            != DOWNSTREAM_REFERENCE_FLOOR_ADVANTAGE_PP / 100.0
            or result.task_guard_margin != DOWNSTREAM_TASK_GUARD_MARGIN_PP / 100.0
            or result.informative
            != (
                result.reference_floor_bootstrap_lower
                > result.required_floor_advantage
            )
            or result.task_guard_pass
            != (result.candidate_minus_reference >= -result.task_guard_margin)
        ):
            raise ValueError("publication downstream task evidence is inconsistent")
    if set(family_by_key) != {"kobest", "klue"}:
        raise ValueError("publication downstream evidence is inconsistent")
    for family, result in family_by_key.items():
        informative_tasks = tuple(
            key
            for key in sorted(PRIMARY_DOWNSTREAM_TASKS)
            if PRIMARY_DOWNSTREAM_TASKS[key].family == family
            and task_by_key[key].informative
        )
        expected_family_pass = bool(
            result.bootstrap_one_sided_lower is not None
            and result.bootstrap_one_sided_lower
            > -DOWNSTREAM_FAMILY_NONINFERIORITY_MARGIN_PP / 100.0
            and result.seed_count_within_margin >= 2
        )
        if (
            result.informative_tasks != informative_tasks
            or not np.isfinite(result.mean_candidate_minus_reference)
            or (
                result.bootstrap_one_sided_lower is not None
                and not np.isfinite(result.bootstrap_one_sided_lower)
            )
            or result.confidence != DOWNSTREAM_FAMILY_ONE_SIDED_CONFIDENCE
            or result.noninferiority_margin
            != DOWNSTREAM_FAMILY_NONINFERIORITY_MARGIN_PP / 100.0
            or result.seed_count_within_margin
            not in range(len(PUBLICATION_PRETRAIN_SEEDS) + 1)
            or result.noninferiority_pass != expected_family_pass
        ):
            raise ValueError("publication downstream family evidence is inconsistent")
    informative_counts = {
        family: sum(
            task.informative for task in evidence.tasks if task.family == family
        )
        for family in ("kobest", "klue")
    }
    informative_pass = bool(
        informative_counts["kobest"]
        >= DOWNSTREAM_MINIMUM_KOBEST_INFORMATIVE_TASKS
        and informative_counts["klue"]
        == DOWNSTREAM_REQUIRED_KLUE_INFORMATIVE_TASKS
    )
    guard_pass = all(task.task_guard_pass for task in evidence.tasks)
    family_pass = all(family.noninferiority_pass for family in evidence.families)
    overall = informative_pass and guard_pass and family_pass
    payload = evidence.to_dict()
    payload.pop("identity_sha256")
    if (
        evidence.candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or evidence.bootstrap_repetitions <= 0
        or evidence.bootstrap_seed < 0
        or evidence.bootstrap_design
        != (
            "crossed model seeds x shared examples within task; "
            "independent example resamples across tasks"
        )
        or snapshot_keys != expected_snapshot_keys
        or not all(
            is_sha256(value)
            for value in (
                evidence.case_manifest_sha256,
                evidence.prediction_artifact_sha256,
                evidence.prediction_arrays_sha256,
                evidence.identity_sha256,
            )
        )
        or evidence.informative_suite_pass != informative_pass
        or evidence.individual_task_guard_pass != guard_pass
        or evidence.family_noninferiority_pass != family_pass
        or evidence.overall_pass != overall
        or evidence.status
        != ("pass" if overall else "fail_downstream_quality")
        or evidence.identity_sha256 != canonical_sha256(payload)
    ):
        raise ValueError("publication downstream evidence is inconsistent")
