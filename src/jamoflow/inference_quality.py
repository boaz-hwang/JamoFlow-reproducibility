"""Five-seed quality noninferiority required before actual inference timing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .document_inference import (
    DocumentWindowMap,
    document_cluster_contrast_summary,
)
from .phase1_analysis import paired_t_interval
from .phase3_analysis import hierarchical_paired_bootstrap_estimates


@dataclass(frozen=True, slots=True)
class InferenceQualityNoninferiority:
    seed_order: tuple[int, ...]
    candidate_policy: str
    reference_policy: str
    difference_direction: str
    paired_differences_bpb: tuple[float, ...]
    mean_difference_bpb: float
    noninferiority_margin_bpb: float
    paired_seed_t_95_lower_bpb: float
    paired_seed_t_95_upper_bpb: float
    hierarchical_bootstrap_repetitions: int
    hierarchical_bootstrap_seed: int
    hierarchical_bootstrap_95_lower_bpb: float
    hierarchical_bootstrap_95_upper_bpb: float
    document_cluster_bootstrap_95_lower_bpb: float
    document_cluster_bootstrap_95_upper_bpb: float
    document_cluster_coverage_pass: bool
    eligible_document_count: int
    eligible_sequence_fraction: float
    seed_count_within_margin: int
    required_seed_count_within_margin: int
    overall_pass: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["seed_order"] = list(self.seed_order)
        payload["paired_differences_bpb"] = list(
            self.paired_differences_bpb
        )
        return payload


def inference_quality_noninferiority(
    candidate_losses_nats: Mapping[int, np.ndarray],
    reference_losses_nats: Mapping[int, np.ndarray],
    *,
    seed_order: tuple[int, ...],
    candidate_policy: str,
    reference_policy: str,
    targets_per_sequence: int,
    document_window_map: DocumentWindowMap,
    margin_bpb: float = 0.010,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_813,
) -> InferenceQualityNoninferiority:
    """Use the paired-seed t upper bound as the preregistered quality gate."""

    if (
        len(seed_order) != 5
        or len(set(seed_order)) != len(seed_order)
        or set(candidate_losses_nats) != set(seed_order)
        or set(reference_losses_nats) != set(seed_order)
        or not candidate_policy
        or not reference_policy
        or targets_per_sequence <= 0
        or margin_bpb <= 0
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("inference quality requires a fixed five-seed paired design")
    scale = targets_per_sequence * math.log(2)
    differences_nats: list[np.ndarray] = []
    differences_bpb: list[float] = []
    expected_shape: tuple[int, ...] | None = None
    for seed in seed_order:
        candidate = np.asarray(candidate_losses_nats[seed], dtype=np.float64)
        reference = np.asarray(reference_losses_nats[seed], dtype=np.float64)
        if (
            candidate.ndim != 1
            or candidate.shape != reference.shape
            or not len(candidate)
            or not np.isfinite(candidate).all()
            or not np.isfinite(reference).all()
            or np.any(candidate < 0)
            or np.any(reference < 0)
        ):
            raise ValueError("quality losses must be equal finite nonnegative vectors")
        if expected_shape is None:
            expected_shape = candidate.shape
        elif candidate.shape != expected_shape:
            raise ValueError("all model seeds must use the same held-out sequences")
        difference = candidate - reference
        differences_nats.append(difference)
        differences_bpb.append(float(difference.mean()) / scale)

    interval = paired_t_interval(differences_bpb)
    bootstrap = hierarchical_paired_bootstrap_estimates(
        differences_nats,
        targets_per_sequence=targets_per_sequence,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    bootstrap_lower, bootstrap_upper = np.quantile(bootstrap, [0.025, 0.975])
    document_cluster = document_cluster_contrast_summary(
        differences_nats,
        document_window_map,
        targets_per_sequence=targets_per_sequence,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed + 100,
    )
    seed_count_within = int(
        sum(effect <= margin_bpb for effect in differences_bpb)
    )
    required_seed_count_within = 4
    passed = bool(
        interval.upper < margin_bpb
        and float(document_cluster["upper"]) < margin_bpb
        and bool(document_cluster["eligible_sequence_fraction_pass"])
        and seed_count_within >= required_seed_count_within
    )
    return InferenceQualityNoninferiority(
        seed_order=seed_order,
        candidate_policy=candidate_policy,
        reference_policy=reference_policy,
        difference_direction="candidate_minus_reference; lower favors candidate",
        paired_differences_bpb=tuple(differences_bpb),
        mean_difference_bpb=float(np.mean(differences_bpb)),
        noninferiority_margin_bpb=margin_bpb,
        paired_seed_t_95_lower_bpb=interval.lower,
        paired_seed_t_95_upper_bpb=interval.upper,
        hierarchical_bootstrap_repetitions=bootstrap_repetitions,
        hierarchical_bootstrap_seed=bootstrap_seed,
        hierarchical_bootstrap_95_lower_bpb=float(bootstrap_lower),
        hierarchical_bootstrap_95_upper_bpb=float(bootstrap_upper),
        document_cluster_bootstrap_95_lower_bpb=float(
            document_cluster["lower"]
        ),
        document_cluster_bootstrap_95_upper_bpb=float(
            document_cluster["upper"]
        ),
        document_cluster_coverage_pass=bool(
            document_cluster["eligible_sequence_fraction_pass"]
        ),
        eligible_document_count=document_window_map.eligible_document_count,
        eligible_sequence_fraction=(
            document_window_map.eligible_sequence_fraction
        ),
        seed_count_within_margin=seed_count_within,
        required_seed_count_within_margin=required_seed_count_within,
        overall_pass=passed,
        status="pass" if passed else "fail_quality_noninferiority",
    )
