"""Learning-curve stability and capability floor for publication claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .publication_downstream import (
    PublicationDownstreamGate,
    validate_publication_downstream_gate,
)
from .publication_model_lock import (
    PUBLICATION_MODEL_KEYS,
    PublicationLearningCurveModelLock,
    canonical_sha256,
    is_sha256,
    named_array_manifest_sha256,
    validate_publication_learning_curve_model_lock,
)
from .publication_protocol import (
    PUBLICATION_BPE_VOCABULARY_CANDIDATES,
    PUBLICATION_BPB_NONINFERIORITY_MARGIN,
    PUBLICATION_BPB_ONE_SIDED_CONFIDENCE,
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)


PUBLICATION_LEARNING_CURVE_BYTES = (64_000_000, 128_000_000, 256_000_000)
PUBLICATION_DATA_ADEQUACY_BOOTSTRAP_REPETITIONS = 10_000
PUBLICATION_DATA_ADEQUACY_BOOTSTRAP_SEED = 20_260_819
PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS = 2


@dataclass(frozen=True, slots=True)
class PairDataStability:
    comparator_key: str
    previous_budget_bytes: int
    current_budget_bytes: int
    previous_candidate_mean_bpb: float
    current_candidate_mean_bpb: float
    previous_comparator_mean_bpb: float
    current_comparator_mean_bpb: float
    previous_paired_differences_bpb: tuple[float, ...]
    current_paired_differences_bpb: tuple[float, ...]
    previous_candidate_minus_comparator_bpb: float
    current_candidate_minus_comparator_bpb: float
    previous_bootstrap_one_sided_upper_bpb: float
    current_bootstrap_one_sided_upper_bpb: float
    bootstrap_repetitions: int
    previous_bootstrap_seed: int
    current_bootstrap_seed: int
    confidence: float
    margin_bpb: float
    previous_seed_count_within_margin: int
    current_seed_count_within_margin: int
    last_two_budget_noninferiority_pass: bool
    candidate_progress_bpb: float
    comparator_progress_bpb: float
    candidate_seed_count_nonworsening: int
    comparator_seed_count_nonworsening: int
    learning_progress_pass: bool
    overall_pass: bool


@dataclass(frozen=True, slots=True)
class PublicationDataAdequacy:
    seed_order: tuple[int, ...]
    learning_curve_bytes: tuple[int, ...]
    candidate_key: str
    raw_comparator_key: str
    bpe_vocabulary_sizes: tuple[int, ...]
    bpe_data_matched_keys: tuple[str, ...]
    learning_curve_model_locks: tuple[PublicationLearningCurveModelLock, ...]
    curve_artifact_sha256: str
    curve_arrays_sha256: str
    downstream_evidence_sha256: str
    downstream_gate_candidate_key: str
    downstream_gate_status: str
    downstream_informativeness_pass: bool
    pairs: tuple[PairDataStability, ...]
    overall_pass: bool
    status: str
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_curve(
    values: Mapping[int, tuple[float, ...]],
    budgets: tuple[int, ...],
    seed_order: tuple[int, ...],
) -> dict[int, np.ndarray]:
    if set(values) != set(budgets):
        raise ValueError("learning curve must contain every preregistered budget")
    output = {}
    for budget in budgets:
        array = np.asarray(values[budget], dtype=np.float64)
        if (
            array.shape != (len(seed_order),)
            or not np.isfinite(array).all()
            or np.any(array <= 0)
        ):
            raise ValueError("learning curve BPB values are malformed")
        output[budget] = array
    return output


def _paired_seed_bootstrap_upper(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> float:
    if (
        differences.shape != (len(PUBLICATION_PRETRAIN_SEEDS),)
        or not np.isfinite(differences).all()
        or repetitions <= 0
    ):
        raise ValueError("data-adequacy seed differences are malformed")
    rng = np.random.default_rng(seed)
    selected = rng.integers(
        0,
        len(differences),
        size=(repetitions, len(differences)),
    )
    estimates = differences[selected].mean(axis=1)
    return float(np.quantile(estimates, PUBLICATION_BPB_ONE_SIDED_CONFIDENCE))


def evaluate_publication_data_adequacy(
    curves: Mapping[str, Mapping[int, tuple[float, ...]]],
    *,
    candidate_key: str,
    raw_comparator_key: str,
    bpe_data_matched_keys: Mapping[int, str],
    downstream_gate: PublicationDownstreamGate,
    learning_curve_model_locks: Mapping[
        str,
        PublicationLearningCurveModelLock,
    ],
    curve_artifact_sha256: str,
    seed_order: tuple[int, ...] = PUBLICATION_PRETRAIN_SEEDS,
    learning_curve_bytes: tuple[int, ...] = PUBLICATION_LEARNING_CURVE_BYTES,
    bootstrap_repetitions: int = PUBLICATION_DATA_ADEQUACY_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = PUBLICATION_DATA_ADEQUACY_BOOTSTRAP_SEED,
) -> PublicationDataAdequacy:
    """Require two-budget quality retention and an informative task suite."""

    bpe_vocabulary_sizes = PUBLICATION_BPE_VOCABULARY_CANDIDATES
    if set(bpe_data_matched_keys) != set(bpe_vocabulary_sizes):
        raise ValueError("data adequacy requires both BPE vocabulary controls")
    ordered_bpe_keys = tuple(
        bpe_data_matched_keys[size] for size in bpe_vocabulary_sizes
    )
    identities = (candidate_key, raw_comparator_key, *ordered_bpe_keys)
    validate_publication_downstream_gate(downstream_gate)
    if set(learning_curve_model_locks) != set(identities):
        raise ValueError("data adequacy requires exact learning-curve model locks")
    for key, lock in learning_curve_model_locks.items():
        validate_publication_learning_curve_model_lock(lock)
        if lock.model_key != key or lock.budget_bytes != learning_curve_bytes:
            raise ValueError("learning-curve model lock does not match curve design")
    final_snapshots = {
        key: lock.final_snapshot for key, lock in learning_curve_model_locks.items()
    }
    downstream_snapshots = {
        snapshot.model_key: snapshot for snapshot in downstream_gate.model_snapshots
    }
    if (
        not is_sha256(curve_artifact_sha256)
        or any(
            final_snapshots[key] != snapshot
            for key, snapshot in downstream_snapshots.items()
        )
    ):
        raise ValueError("data adequacy model or curve artifact lock is inconsistent")
    if (
        seed_order != PUBLICATION_PRETRAIN_SEEDS
        or candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or raw_comparator_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
        or ordered_bpe_keys
        != tuple(
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size]
            for size in bpe_vocabulary_sizes
        )
        or not isinstance(downstream_gate, PublicationDownstreamGate)
        or downstream_gate.seed_order != seed_order
        or downstream_gate.candidate_key != candidate_key
        or len(set(identities)) != len(identities)
        or any(not identity for identity in identities)
        or tuple(sorted(learning_curve_bytes)) != learning_curve_bytes
        or len(learning_curve_bytes) < 3
        or any(
            right != left * 2
            for left, right in zip(
                learning_curve_bytes,
                learning_curve_bytes[1:],
            )
        )
        or set(curves) != set(identities)
        or bootstrap_repetitions <= 0
    ):
        raise ValueError(
            "data adequacy requires candidate, raw, both BPE controls, "
            "and doubling budgets"
        )
    validated = {
        key: _validated_curve(values, learning_curve_bytes, seed_order)
        for key, values in curves.items()
    }
    curve_arrays_sha256 = named_array_manifest_sha256(
        {
            f"{key}:{budget}": values[budget]
            for key, values in validated.items()
            for budget in learning_curve_bytes
        }
    )
    downstream_informativeness_pass = downstream_gate.informative_suite_pass
    previous_budget, current_budget = learning_curve_bytes[-2:]
    candidate_previous_values = validated[candidate_key][previous_budget]
    candidate_current_values = validated[candidate_key][current_budget]
    candidate_previous = float(candidate_previous_values.mean())
    candidate_current = float(candidate_current_values.mean())
    pair_results: list[PairDataStability] = []
    for comparator_index, comparator_key in enumerate(
        (raw_comparator_key, *ordered_bpe_keys)
    ):
        comparator_previous_values = validated[comparator_key][previous_budget]
        comparator_current_values = validated[comparator_key][current_budget]
        comparator_previous = float(comparator_previous_values.mean())
        comparator_current = float(comparator_current_values.mean())
        previous_differences = (
            candidate_previous_values - comparator_previous_values
        )
        current_differences = candidate_current_values - comparator_current_values
        previous_gap = float(previous_differences.mean())
        current_gap = float(current_differences.mean())
        pair_bootstrap_seed = bootstrap_seed + comparator_index * 2
        previous_upper = _paired_seed_bootstrap_upper(
            previous_differences,
            repetitions=bootstrap_repetitions,
            seed=pair_bootstrap_seed,
        )
        current_upper = _paired_seed_bootstrap_upper(
            current_differences,
            repetitions=bootstrap_repetitions,
            seed=pair_bootstrap_seed + 1,
        )
        previous_count = int(
            np.count_nonzero(
                previous_differences <= PUBLICATION_BPB_NONINFERIORITY_MARGIN
            )
        )
        current_count = int(
            np.count_nonzero(
                current_differences <= PUBLICATION_BPB_NONINFERIORITY_MARGIN
            )
        )
        noninferiority_pass = bool(
            previous_upper < PUBLICATION_BPB_NONINFERIORITY_MARGIN
            and current_upper < PUBLICATION_BPB_NONINFERIORITY_MARGIN
            and previous_count >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
            and current_count >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
        )
        candidate_progress = candidate_previous - candidate_current
        comparator_progress = comparator_previous - comparator_current
        candidate_seed_progress = (
            candidate_previous_values - candidate_current_values
        )
        comparator_seed_progress = (
            comparator_previous_values - comparator_current_values
        )
        candidate_nonworsening_count = int(
            np.count_nonzero(candidate_seed_progress >= 0)
        )
        comparator_nonworsening_count = int(
            np.count_nonzero(comparator_seed_progress >= 0)
        )
        progress_pass = bool(
            candidate_progress >= 0
            and comparator_progress >= 0
            and candidate_nonworsening_count
            >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
            and comparator_nonworsening_count
            >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
        )
        passed = progress_pass and noninferiority_pass
        pair_results.append(
            PairDataStability(
                comparator_key=comparator_key,
                previous_budget_bytes=previous_budget,
                current_budget_bytes=current_budget,
                previous_candidate_mean_bpb=candidate_previous,
                current_candidate_mean_bpb=candidate_current,
                previous_comparator_mean_bpb=comparator_previous,
                current_comparator_mean_bpb=comparator_current,
                previous_paired_differences_bpb=tuple(
                    float(value) for value in previous_differences
                ),
                current_paired_differences_bpb=tuple(
                    float(value) for value in current_differences
                ),
                previous_candidate_minus_comparator_bpb=previous_gap,
                current_candidate_minus_comparator_bpb=current_gap,
                previous_bootstrap_one_sided_upper_bpb=previous_upper,
                current_bootstrap_one_sided_upper_bpb=current_upper,
                bootstrap_repetitions=bootstrap_repetitions,
                previous_bootstrap_seed=pair_bootstrap_seed,
                current_bootstrap_seed=pair_bootstrap_seed + 1,
                confidence=PUBLICATION_BPB_ONE_SIDED_CONFIDENCE,
                margin_bpb=PUBLICATION_BPB_NONINFERIORITY_MARGIN,
                previous_seed_count_within_margin=previous_count,
                current_seed_count_within_margin=current_count,
                last_two_budget_noninferiority_pass=noninferiority_pass,
                candidate_progress_bpb=candidate_progress,
                comparator_progress_bpb=comparator_progress,
                candidate_seed_count_nonworsening=candidate_nonworsening_count,
                comparator_seed_count_nonworsening=(
                    comparator_nonworsening_count
                ),
                learning_progress_pass=progress_pass,
                overall_pass=passed,
            )
        )
    overall = bool(
        downstream_informativeness_pass
        and all(result.overall_pass for result in pair_results)
    )
    if overall:
        status = "pass"
    elif not downstream_informativeness_pass:
        status = "capability_undertrained"
    else:
        status = "last_two_budget_quality_unstable"
    ordered_curve_locks = tuple(
        learning_curve_model_locks[key] for key in PUBLICATION_MODEL_KEYS
    )
    provisional = PublicationDataAdequacy(
        seed_order=seed_order,
        learning_curve_bytes=learning_curve_bytes,
        candidate_key=candidate_key,
        raw_comparator_key=raw_comparator_key,
        bpe_vocabulary_sizes=bpe_vocabulary_sizes,
        bpe_data_matched_keys=ordered_bpe_keys,
        learning_curve_model_locks=ordered_curve_locks,
        curve_artifact_sha256=curve_artifact_sha256,
        curve_arrays_sha256=curve_arrays_sha256,
        downstream_evidence_sha256=downstream_gate.identity_sha256,
        downstream_gate_candidate_key=downstream_gate.candidate_key,
        downstream_gate_status=downstream_gate.status,
        downstream_informativeness_pass=downstream_informativeness_pass,
        pairs=tuple(pair_results),
        overall_pass=overall,
        status=status,
        identity_sha256="",
    )
    payload = provisional.to_dict()
    payload.pop("identity_sha256")
    result = PublicationDataAdequacy(
        **{
            **provisional.to_dict(),
            "learning_curve_model_locks": ordered_curve_locks,
            "pairs": tuple(pair_results),
            "identity_sha256": canonical_sha256(payload),
        }
    )
    validate_publication_data_adequacy(result)
    return result


def validate_publication_data_adequacy(
    evidence: PublicationDataAdequacy,
) -> None:
    if not isinstance(evidence, PublicationDataAdequacy):
        raise ValueError("publication data-adequacy evidence is invalid")
    for lock in evidence.learning_curve_model_locks:
        validate_publication_learning_curve_model_lock(lock)
    lock_keys = tuple(lock.model_key for lock in evidence.learning_curve_model_locks)
    expected_comparators = (
        PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
        *(
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size]
            for size in PUBLICATION_BPE_VOCABULARY_CANDIDATES
        ),
    )
    pair_keys = tuple(pair.comparator_key for pair in evidence.pairs)
    if pair_keys != expected_comparators:
        raise ValueError("publication data-adequacy pair order is inconsistent")
    previous_budget, current_budget = evidence.learning_curve_bytes[-2:]
    for pair in evidence.pairs:
        numeric = (
            pair.previous_candidate_mean_bpb,
            pair.current_candidate_mean_bpb,
            pair.previous_comparator_mean_bpb,
            pair.current_comparator_mean_bpb,
            pair.previous_candidate_minus_comparator_bpb,
            pair.current_candidate_minus_comparator_bpb,
            pair.previous_bootstrap_one_sided_upper_bpb,
            pair.current_bootstrap_one_sided_upper_bpb,
            pair.candidate_progress_bpb,
            pair.comparator_progress_bpb,
        )
        previous_differences = np.asarray(
            pair.previous_paired_differences_bpb,
            dtype=np.float64,
        )
        current_differences = np.asarray(
            pair.current_paired_differences_bpb,
            dtype=np.float64,
        )
        noninferiority_pass = bool(
            pair.previous_bootstrap_one_sided_upper_bpb
            < PUBLICATION_BPB_NONINFERIORITY_MARGIN
            and pair.current_bootstrap_one_sided_upper_bpb
            < PUBLICATION_BPB_NONINFERIORITY_MARGIN
            and pair.previous_seed_count_within_margin
            >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
            and pair.current_seed_count_within_margin
            >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
        )
        progress_pass = bool(
            pair.candidate_progress_bpb >= 0
            and pair.comparator_progress_bpb >= 0
            and pair.candidate_seed_count_nonworsening
            >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
            and pair.comparator_seed_count_nonworsening
            >= PUBLICATION_DATA_ADEQUACY_MINIMUM_SEEDS
        )
        if (
            pair.previous_budget_bytes != previous_budget
            or pair.current_budget_bytes != current_budget
            or previous_differences.shape
            != (len(PUBLICATION_PRETRAIN_SEEDS),)
            or current_differences.shape
            != (len(PUBLICATION_PRETRAIN_SEEDS),)
            or not np.isfinite(previous_differences).all()
            or not np.isfinite(current_differences).all()
            or not all(np.isfinite(value) for value in numeric)
            or any(
                value <= 0
                for value in (
                    pair.previous_candidate_mean_bpb,
                    pair.current_candidate_mean_bpb,
                    pair.previous_comparator_mean_bpb,
                    pair.current_comparator_mean_bpb,
                )
            )
            or not np.isclose(
                pair.previous_candidate_minus_comparator_bpb,
                pair.previous_candidate_mean_bpb
                - pair.previous_comparator_mean_bpb,
            )
            or not np.isclose(
                pair.current_candidate_minus_comparator_bpb,
                pair.current_candidate_mean_bpb
                - pair.current_comparator_mean_bpb,
            )
            or not np.isclose(
                pair.candidate_progress_bpb,
                pair.previous_candidate_mean_bpb
                - pair.current_candidate_mean_bpb,
            )
            or not np.isclose(
                pair.comparator_progress_bpb,
                pair.previous_comparator_mean_bpb
                - pair.current_comparator_mean_bpb,
            )
            or not np.isclose(
                pair.previous_candidate_minus_comparator_bpb,
                previous_differences.mean(),
            )
            or not np.isclose(
                pair.current_candidate_minus_comparator_bpb,
                current_differences.mean(),
            )
            or pair.confidence != PUBLICATION_BPB_ONE_SIDED_CONFIDENCE
            or pair.margin_bpb != PUBLICATION_BPB_NONINFERIORITY_MARGIN
            or pair.bootstrap_repetitions <= 0
            or pair.previous_bootstrap_seed < 0
            or pair.current_bootstrap_seed != pair.previous_bootstrap_seed + 1
            or pair.candidate_seed_count_nonworsening
            not in range(len(PUBLICATION_PRETRAIN_SEEDS) + 1)
            or pair.comparator_seed_count_nonworsening
            not in range(len(PUBLICATION_PRETRAIN_SEEDS) + 1)
            or pair.previous_seed_count_within_margin
            != int(
                np.count_nonzero(
                    previous_differences <= PUBLICATION_BPB_NONINFERIORITY_MARGIN
                )
            )
            or pair.current_seed_count_within_margin
            != int(
                np.count_nonzero(
                    current_differences <= PUBLICATION_BPB_NONINFERIORITY_MARGIN
                )
            )
            or pair.last_two_budget_noninferiority_pass != noninferiority_pass
            or pair.learning_progress_pass != progress_pass
            or pair.overall_pass != (noninferiority_pass and progress_pass)
        ):
            raise ValueError("publication data-adequacy pair is inconsistent")
    downstream_pass = evidence.downstream_informativeness_pass
    pairs_pass = all(pair.overall_pass for pair in evidence.pairs)
    overall = downstream_pass and pairs_pass
    if overall:
        status = "pass"
    elif not downstream_pass:
        status = "capability_undertrained"
    else:
        status = "last_two_budget_quality_unstable"
    payload = evidence.to_dict()
    payload.pop("identity_sha256")
    if (
        evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or tuple(sorted(evidence.learning_curve_bytes))
        != evidence.learning_curve_bytes
        or len(evidence.learning_curve_bytes) < 3
        or any(
            right != left * 2
            for left, right in zip(
                evidence.learning_curve_bytes,
                evidence.learning_curve_bytes[1:],
            )
        )
        or evidence.candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or evidence.raw_comparator_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
        or evidence.bpe_vocabulary_sizes
        != PUBLICATION_BPE_VOCABULARY_CANDIDATES
        or evidence.bpe_data_matched_keys != expected_comparators[1:]
        or lock_keys != PUBLICATION_MODEL_KEYS
        or any(
            lock.budget_bytes != evidence.learning_curve_bytes
            for lock in evidence.learning_curve_model_locks
        )
        or evidence.downstream_gate_candidate_key != evidence.candidate_key
        or not all(
            is_sha256(value)
            for value in (
                evidence.curve_artifact_sha256,
                evidence.curve_arrays_sha256,
                evidence.downstream_evidence_sha256,
                evidence.identity_sha256,
            )
        )
        or evidence.overall_pass != overall
        or evidence.status != status
        or evidence.identity_sha256 != canonical_sha256(payload)
    ):
        raise ValueError("publication data-adequacy evidence is inconsistent")
