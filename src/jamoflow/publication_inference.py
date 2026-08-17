"""Publication-scale actual-inference and final-value gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .data_adequacy import (
    PublicationDataAdequacy,
    validate_publication_data_adequacy,
)
from .inference_benchmark import MultiSeedPairedLatency
from .publication_bpb import (
    PublicationBPBContextEvidence,
    validate_publication_bpb_context_evidence,
)
from .publication_downstream import (
    PublicationDownstreamGate,
    validate_publication_downstream_gate,
)
from .publication_model_lock import (
    PublicationModelLockGraph,
    PublicationModelSnapshot,
    build_publication_model_lock_graph,
    canonical_sha256,
    is_sha256,
    named_array_manifest_sha256,
    publication_runtime_model_snapshots,
    validate_publication_model_lock_graph,
    validate_publication_model_snapshot,
)
from .publication_protocol import (
    ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS,
    ACTUAL_INFERENCE_MINIMUM_REDUCTION,
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_BPE_VOCABULARY_CANDIDATES,
    PUBLICATION_BPB_NONINFERIORITY_MARGIN,
    PUBLICATION_BPB_ONE_SIDED_CONFIDENCE,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)
from .publication_runtime import (
    PublicationRuntimeEvidence,
    validate_publication_runtime_evidence,
)


PUBLICATION_MINIMUM_SPEEDUP_SEEDS = 2
PUBLICATION_ENCODING_MAXIMUM_REGRESSION = 0.02
PUBLICATION_MINIMUM_ENCODING_SEEDS = 2
PUBLICATION_REQUIRED_VALID_OUTPUT_SEEDS = len(PUBLICATION_PRETRAIN_SEEDS)


@dataclass(frozen=True, slots=True)
class PublicationBPBNoninferiority:
    candidate_key: str
    comparator_key: str
    seed_order: tuple[int, ...]
    document_count: int
    scored_bytes: int
    scored_bytes_by_document: tuple[int, ...]
    paired_differences_bpb: tuple[float, ...]
    mean_difference_bpb: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    bootstrap_design: str
    bootstrap_one_sided_upper_bpb: float
    confidence: float
    margin_bpb: float
    seed_count_within_margin: int
    context_evidence: PublicationBPBContextEvidence
    candidate_snapshot: PublicationModelSnapshot
    comparator_snapshot: PublicationModelSnapshot
    loss_arrays_sha256: str
    overall_pass: bool
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationComparatorInferenceGate:
    candidate_key: str
    comparator_key: str
    comparator_family: str
    runtime_equivalence_pass: bool
    timing_integrity_pass: bool
    valid_output_contract_pass: bool
    runtime_evidence: PublicationRuntimeEvidence
    downstream_evidence: PublicationDownstreamGate
    downstream_noninferiority_pass: bool
    bpb: PublicationBPBNoninferiority
    controlled_replay_decode: MultiSeedPairedLatency
    controlled_seed_count_at_minimum_reduction: int
    controlled_replay_pass: bool
    free_running_end_to_end: MultiSeedPairedLatency
    free_seed_count_at_minimum_reduction: int
    free_running_pass: bool
    candidate_mean_valid_output_completion_rate: float
    comparator_mean_valid_output_completion_rate: float
    candidate_mean_replacement_free_rate: float
    comparator_mean_replacement_free_rate: float
    valid_output_seed_count_at_one: int
    replacement_seed_count_within_margin: int
    encoding_quality_pass: bool
    overall_pass: bool
    status: str
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationFinalValueGate:
    candidate_key: str
    raw_comparator_key: str
    bpe_vocabulary_sizes: tuple[int, ...]
    bpe_comparator_keys: tuple[str, ...]
    raw_comparator_pass: bool
    bpe_comparator_passes: tuple[bool, ...]
    all_bpe_comparators_pass: bool
    data_adequacy_pass: bool
    model_lock_graph: PublicationModelLockGraph
    raw_comparator_gate_sha256: str
    bpe_comparator_gate_sha256: tuple[str, ...]
    data_adequacy_sha256: str
    overall_pass: bool
    claim_level: str
    status: str
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def publication_bpb_noninferiority(
    candidate_losses_nats: Mapping[int, np.ndarray],
    reference_losses_nats: Mapping[int, np.ndarray],
    scored_bytes_by_document: np.ndarray,
    *,
    candidate_key: str,
    comparator_key: str,
    context_evidence: PublicationBPBContextEvidence,
    candidate_snapshot: PublicationModelSnapshot,
    comparator_snapshot: PublicationModelSnapshot,
    seed_order: tuple[int, ...] = PUBLICATION_PRETRAIN_SEEDS,
    bootstrap_repetitions: int = ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = 20_260_815,
) -> PublicationBPBNoninferiority:
    """Cross model seeds and shared documents with a byte-weighted BPB ratio."""

    bytes_array = np.asarray(scored_bytes_by_document, dtype=np.int64)
    validate_publication_model_snapshot(candidate_snapshot)
    validate_publication_model_snapshot(comparator_snapshot)
    if (
        candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or comparator_key
        not in {
            PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            *PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values(),
        }
        or seed_order != PUBLICATION_PRETRAIN_SEEDS
        or set(candidate_losses_nats) != set(seed_order)
        or set(reference_losses_nats) != set(seed_order)
        or bytes_array.ndim != 1
        or not len(bytes_array)
        or np.any(bytes_array <= 0)
        or bootstrap_repetitions <= 0
        or candidate_snapshot.model_key != candidate_key
        or comparator_snapshot.model_key != comparator_key
        or context_evidence.tokenizer_sha256
        != comparator_snapshot.tokenizer_sha256
    ):
        raise ValueError("publication BPB requires the exact crossed design")
    validate_publication_bpb_context_evidence(
        context_evidence,
        tuple(int(value) for value in bytes_array),
        candidate_key=candidate_key,
        comparator_key=comparator_key,
    )
    differences: list[np.ndarray] = []
    loss_arrays: dict[str, np.ndarray] = {"scored_bytes": bytes_array}
    for seed in seed_order:
        candidate = np.asarray(candidate_losses_nats[seed], dtype=np.float64)
        reference = np.asarray(reference_losses_nats[seed], dtype=np.float64)
        if (
            candidate.shape != bytes_array.shape
            or reference.shape != bytes_array.shape
            or not np.isfinite(candidate).all()
            or not np.isfinite(reference).all()
            or np.any(candidate < 0)
            or np.any(reference < 0)
        ):
            raise ValueError("publication document losses are malformed")
        loss_arrays[f"candidate:{seed}"] = candidate
        loss_arrays[f"comparator:{seed}"] = reference
        differences.append(candidate - reference)
    denominator = float(bytes_array.sum()) * math.log(2.0)
    paired = tuple(float(values.sum() / denominator) for values in differences)
    point = float(np.mean(paired))

    rng = np.random.default_rng(bootstrap_seed)
    seed_count = len(seed_order)
    document_count = len(bytes_array)
    estimates = np.empty(bootstrap_repetitions, dtype=np.float64)
    chunk_size = 128
    for start in range(0, bootstrap_repetitions, chunk_size):
        size = min(chunk_size, bootstrap_repetitions - start)
        selected_seeds = rng.integers(0, seed_count, size=(size, seed_count))
        selected_documents = rng.integers(
            0,
            document_count,
            size=(size, document_count),
        )
        source_numerators = np.empty((size, seed_count), dtype=np.float64)
        for source_seed, values in enumerate(differences):
            source_numerators[:, source_seed] = values[selected_documents].sum(
                axis=1
            )
        crossed_numerator = np.take_along_axis(
            source_numerators,
            selected_seeds,
            axis=1,
        ).mean(axis=1)
        sampled_denominator = (
            bytes_array[selected_documents].sum(axis=1) * math.log(2.0)
        )
        estimates[start : start + size] = crossed_numerator / sampled_denominator
    upper = float(np.quantile(estimates, PUBLICATION_BPB_ONE_SIDED_CONFIDENCE))
    seed_count_within = sum(
        value <= PUBLICATION_BPB_NONINFERIORITY_MARGIN for value in paired
    )
    passed = bool(
        upper < PUBLICATION_BPB_NONINFERIORITY_MARGIN
        and seed_count_within >= 2
    )
    provisional = PublicationBPBNoninferiority(
        candidate_key=candidate_key,
        comparator_key=comparator_key,
        seed_order=seed_order,
        document_count=document_count,
        scored_bytes=int(bytes_array.sum()),
        scored_bytes_by_document=tuple(int(value) for value in bytes_array),
        paired_differences_bpb=paired,
        mean_difference_bpb=point,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        bootstrap_design="crossed model seeds x shared documents; byte-weighted",
        bootstrap_one_sided_upper_bpb=upper,
        confidence=PUBLICATION_BPB_ONE_SIDED_CONFIDENCE,
        margin_bpb=PUBLICATION_BPB_NONINFERIORITY_MARGIN,
        seed_count_within_margin=seed_count_within,
        context_evidence=context_evidence,
        candidate_snapshot=candidate_snapshot,
        comparator_snapshot=comparator_snapshot,
        loss_arrays_sha256=named_array_manifest_sha256(loss_arrays),
        overall_pass=passed,
        identity_sha256="",
    )
    payload = provisional.to_dict()
    payload.pop("identity_sha256")
    result = PublicationBPBNoninferiority(
        **{
            **provisional.to_dict(),
            "context_evidence": context_evidence,
            "candidate_snapshot": candidate_snapshot,
            "comparator_snapshot": comparator_snapshot,
            "identity_sha256": canonical_sha256(payload),
        }
    )
    validate_publication_bpb_noninferiority(result)
    return result


def validate_publication_bpb_noninferiority(
    evidence: PublicationBPBNoninferiority,
) -> None:
    if not isinstance(evidence, PublicationBPBNoninferiority):
        raise ValueError("publication BPB evidence is invalid")
    validate_publication_model_snapshot(evidence.candidate_snapshot)
    validate_publication_model_snapshot(evidence.comparator_snapshot)
    validate_publication_bpb_context_evidence(
        evidence.context_evidence,
        evidence.scored_bytes_by_document,
        candidate_key=evidence.candidate_key,
        comparator_key=evidence.comparator_key,
    )
    differences = np.asarray(evidence.paired_differences_bpb, dtype=np.float64)
    expected_count = int(
        np.count_nonzero(differences <= PUBLICATION_BPB_NONINFERIORITY_MARGIN)
    )
    expected_pass = bool(
        evidence.bootstrap_one_sided_upper_bpb
        < PUBLICATION_BPB_NONINFERIORITY_MARGIN
        and expected_count >= 2
    )
    payload = evidence.to_dict()
    payload.pop("identity_sha256")
    if (
        evidence.candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or evidence.comparator_key
        not in {
            PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            *PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values(),
        }
        or evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or evidence.candidate_snapshot.model_key != evidence.candidate_key
        or evidence.comparator_snapshot.model_key != evidence.comparator_key
        or evidence.context_evidence.tokenizer_sha256
        != evidence.comparator_snapshot.tokenizer_sha256
        or evidence.document_count != len(evidence.scored_bytes_by_document)
        or evidence.document_count <= 0
        or evidence.scored_bytes != sum(evidence.scored_bytes_by_document)
        or any(value <= 0 for value in evidence.scored_bytes_by_document)
        or differences.shape != (len(PUBLICATION_PRETRAIN_SEEDS),)
        or not np.isfinite(differences).all()
        or not math.isclose(
            evidence.mean_difference_bpb,
            float(differences.mean()),
            rel_tol=0,
            abs_tol=1e-12,
        )
        or evidence.bootstrap_repetitions <= 0
        or evidence.bootstrap_seed < 0
        or evidence.bootstrap_design
        != "crossed model seeds x shared documents; byte-weighted"
        or not np.isfinite(evidence.bootstrap_one_sided_upper_bpb)
        or evidence.confidence != PUBLICATION_BPB_ONE_SIDED_CONFIDENCE
        or evidence.margin_bpb != PUBLICATION_BPB_NONINFERIORITY_MARGIN
        or evidence.seed_count_within_margin != expected_count
        or evidence.overall_pass != expected_pass
        or not is_sha256(evidence.loss_arrays_sha256)
        or not is_sha256(evidence.identity_sha256)
        or evidence.identity_sha256 != canonical_sha256(payload)
    ):
        raise ValueError("publication BPB evidence is inconsistent")


def _latency_gate(
    summary: MultiSeedPairedLatency,
) -> tuple[int, bool]:
    count_at_minimum = sum(
        float(values["median_latency_reduction"])
        >= ACTUAL_INFERENCE_MINIMUM_REDUCTION
        for values in summary.per_seed.values()
    )
    passed = bool(
        summary.crossed_median_latency_reduction
        >= ACTUAL_INFERENCE_MINIMUM_REDUCTION
        and summary.bootstrap_percentile_95_lower > 0
        and count_at_minimum >= PUBLICATION_MINIMUM_SPEEDUP_SEEDS
    )
    return count_at_minimum, passed


def evaluate_publication_comparator_inference_gate(
    *,
    runtime_evidence: PublicationRuntimeEvidence,
    downstream_gate: PublicationDownstreamGate,
    bpb: PublicationBPBNoninferiority,
) -> PublicationComparatorInferenceGate:
    validate_publication_runtime_evidence(runtime_evidence)
    validate_publication_bpb_noninferiority(bpb)
    validate_publication_downstream_gate(downstream_gate)
    candidate_key = runtime_evidence.candidate_key
    comparator_key = runtime_evidence.comparator_key
    comparator_family = runtime_evidence.comparator_family
    seed_order = runtime_evidence.seed_order
    runtime_candidate_snapshot, runtime_comparator_snapshot = (
        publication_runtime_model_snapshots(runtime_evidence.lineage)
    )
    downstream_candidate_snapshot = next(
        snapshot
        for snapshot in downstream_gate.model_snapshots
        if snapshot.model_key == PUBLICATION_CANDIDATE_MODEL_KEY
    )
    if (
        candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or (
            comparator_family == "raw_byte"
            and comparator_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
        )
        or (
            comparator_family == "standard_bpe"
            and comparator_key
            not in PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values()
        )
        or comparator_family not in {"raw_byte", "standard_bpe"}
        or bpb.seed_order != seed_order
        or bpb.candidate_key != candidate_key
        or bpb.comparator_key != comparator_key
        or bpb.candidate_snapshot != runtime_candidate_snapshot
        or bpb.comparator_snapshot != runtime_comparator_snapshot
        or not isinstance(downstream_gate, PublicationDownstreamGate)
        or downstream_gate.seed_order != seed_order
        or downstream_gate.candidate_key != candidate_key
        or downstream_candidate_snapshot != runtime_candidate_snapshot
    ):
        raise ValueError("publication inference comparator identity is invalid")
    downstream_noninferiority_pass = downstream_gate.overall_pass
    controlled = runtime_evidence.timing.controlled_replay_decode
    free = runtime_evidence.timing.free_running_end_to_end
    controlled_count, controlled_pass = _latency_gate(
        controlled,
    )
    free_count, free_pass = _latency_gate(
        free,
    )
    candidate_valid = np.asarray(
        runtime_evidence.valid_output.candidate_completion_rate_by_seed,
        dtype=np.float64,
    )
    reference_valid = np.asarray(
        runtime_evidence.valid_output.comparator_completion_rate_by_seed,
        dtype=np.float64,
    )
    candidate_replacement = np.asarray(
        runtime_evidence.valid_output.candidate_replacement_free_rate_by_seed,
        dtype=np.float64,
    )
    reference_replacement = np.asarray(
        runtime_evidence.valid_output.comparator_replacement_free_rate_by_seed,
        dtype=np.float64,
    )
    valid_seed_count = int(
        np.count_nonzero(
            (candidate_valid == 1.0) & (reference_valid == 1.0)
        )
    )
    replacement_seed_count = int(
        np.count_nonzero(
            candidate_replacement
            >= reference_replacement - PUBLICATION_ENCODING_MAXIMUM_REGRESSION
        )
    )
    encoding_pass = bool(
        valid_seed_count
        == PUBLICATION_REQUIRED_VALID_OUTPUT_SEEDS
        and float(candidate_replacement.mean())
        >= float(reference_replacement.mean())
        - PUBLICATION_ENCODING_MAXIMUM_REGRESSION
        and replacement_seed_count >= PUBLICATION_MINIMUM_ENCODING_SEEDS
    )
    overall = bool(
        runtime_evidence.overall_integrity_pass
        and downstream_noninferiority_pass
        and bpb.overall_pass
        and controlled_pass
        and free_pass
        and encoding_pass
    )
    provisional = PublicationComparatorInferenceGate(
        candidate_key=candidate_key,
        comparator_key=comparator_key,
        comparator_family=comparator_family,
        runtime_equivalence_pass=runtime_evidence.equivalence.overall_pass,
        timing_integrity_pass=runtime_evidence.timing.overall_pass,
        valid_output_contract_pass=runtime_evidence.valid_output.overall_pass,
        runtime_evidence=runtime_evidence,
        downstream_evidence=downstream_gate,
        downstream_noninferiority_pass=downstream_noninferiority_pass,
        bpb=bpb,
        controlled_replay_decode=controlled,
        controlled_seed_count_at_minimum_reduction=controlled_count,
        controlled_replay_pass=controlled_pass,
        free_running_end_to_end=free,
        free_seed_count_at_minimum_reduction=free_count,
        free_running_pass=free_pass,
        candidate_mean_valid_output_completion_rate=float(candidate_valid.mean()),
        comparator_mean_valid_output_completion_rate=float(reference_valid.mean()),
        candidate_mean_replacement_free_rate=float(candidate_replacement.mean()),
        comparator_mean_replacement_free_rate=float(reference_replacement.mean()),
        valid_output_seed_count_at_one=valid_seed_count,
        replacement_seed_count_within_margin=replacement_seed_count,
        encoding_quality_pass=encoding_pass,
        overall_pass=overall,
        status="pass" if overall else "fail_publication_comparator_gate",
        identity_sha256="",
    )
    payload = provisional.to_dict()
    payload.pop("identity_sha256")
    result = PublicationComparatorInferenceGate(
        **{
            **provisional.to_dict(),
            "runtime_evidence": runtime_evidence,
            "downstream_evidence": downstream_gate,
            "bpb": bpb,
            "controlled_replay_decode": controlled,
            "free_running_end_to_end": free,
            "identity_sha256": canonical_sha256(payload),
        }
    )
    validate_publication_comparator_inference_gate(result)
    return result


def validate_publication_comparator_inference_gate(
    gate: PublicationComparatorInferenceGate,
) -> None:
    if not isinstance(gate, PublicationComparatorInferenceGate):
        raise ValueError("publication comparator evidence is invalid")
    validate_publication_runtime_evidence(gate.runtime_evidence)
    validate_publication_downstream_gate(gate.downstream_evidence)
    validate_publication_bpb_noninferiority(gate.bpb)
    runtime_candidate, runtime_comparator = publication_runtime_model_snapshots(
        gate.runtime_evidence.lineage
    )
    downstream_candidate = next(
        snapshot
        for snapshot in gate.downstream_evidence.model_snapshots
        if snapshot.model_key == PUBLICATION_CANDIDATE_MODEL_KEY
    )
    controlled_count, controlled_pass = _latency_gate(
        gate.runtime_evidence.timing.controlled_replay_decode
    )
    free_count, free_pass = _latency_gate(
        gate.runtime_evidence.timing.free_running_end_to_end
    )
    output = gate.runtime_evidence.valid_output
    candidate_valid = np.asarray(
        output.candidate_completion_rate_by_seed,
        dtype=np.float64,
    )
    comparator_valid = np.asarray(
        output.comparator_completion_rate_by_seed,
        dtype=np.float64,
    )
    candidate_replacement = np.asarray(
        output.candidate_replacement_free_rate_by_seed,
        dtype=np.float64,
    )
    comparator_replacement = np.asarray(
        output.comparator_replacement_free_rate_by_seed,
        dtype=np.float64,
    )
    valid_count = int(
        np.count_nonzero((candidate_valid == 1.0) & (comparator_valid == 1.0))
    )
    replacement_count = int(
        np.count_nonzero(
            candidate_replacement
            >= comparator_replacement - PUBLICATION_ENCODING_MAXIMUM_REGRESSION
        )
    )
    encoding_pass = bool(
        valid_count == PUBLICATION_REQUIRED_VALID_OUTPUT_SEEDS
        and float(candidate_replacement.mean())
        >= float(comparator_replacement.mean())
        - PUBLICATION_ENCODING_MAXIMUM_REGRESSION
        and replacement_count >= PUBLICATION_MINIMUM_ENCODING_SEEDS
    )
    overall = bool(
        gate.runtime_evidence.overall_integrity_pass
        and gate.downstream_evidence.overall_pass
        and gate.bpb.overall_pass
        and controlled_pass
        and free_pass
        and encoding_pass
    )
    payload = gate.to_dict()
    payload.pop("identity_sha256")
    if (
        gate.candidate_key != gate.runtime_evidence.candidate_key
        or gate.comparator_key != gate.runtime_evidence.comparator_key
        or gate.comparator_family != gate.runtime_evidence.comparator_family
        or gate.runtime_equivalence_pass
        != gate.runtime_evidence.equivalence.overall_pass
        or gate.timing_integrity_pass != gate.runtime_evidence.timing.overall_pass
        or gate.valid_output_contract_pass
        != gate.runtime_evidence.valid_output.overall_pass
        or gate.downstream_noninferiority_pass
        != gate.downstream_evidence.overall_pass
        or gate.bpb.candidate_snapshot != runtime_candidate
        or gate.bpb.comparator_snapshot != runtime_comparator
        or downstream_candidate != runtime_candidate
        or gate.controlled_replay_decode
        != gate.runtime_evidence.timing.controlled_replay_decode
        or gate.controlled_seed_count_at_minimum_reduction != controlled_count
        or gate.controlled_replay_pass != controlled_pass
        or gate.free_running_end_to_end
        != gate.runtime_evidence.timing.free_running_end_to_end
        or gate.free_seed_count_at_minimum_reduction != free_count
        or gate.free_running_pass != free_pass
        or not math.isclose(
            gate.candidate_mean_valid_output_completion_rate,
            float(candidate_valid.mean()),
        )
        or not math.isclose(
            gate.comparator_mean_valid_output_completion_rate,
            float(comparator_valid.mean()),
        )
        or not math.isclose(
            gate.candidate_mean_replacement_free_rate,
            float(candidate_replacement.mean()),
        )
        or not math.isclose(
            gate.comparator_mean_replacement_free_rate,
            float(comparator_replacement.mean()),
        )
        or gate.valid_output_seed_count_at_one != valid_count
        or gate.replacement_seed_count_within_margin != replacement_count
        or gate.encoding_quality_pass != encoding_pass
        or gate.overall_pass != overall
        or gate.status
        != ("pass" if overall else "fail_publication_comparator_gate")
        or not is_sha256(gate.identity_sha256)
        or gate.identity_sha256 != canonical_sha256(payload)
    ):
        raise ValueError("publication comparator evidence is inconsistent")


def publication_final_value_gate(
    raw_gate: PublicationComparatorInferenceGate,
    bpe_gates: Mapping[int, PublicationComparatorInferenceGate],
    *,
    data_adequacy: PublicationDataAdequacy,
) -> PublicationFinalValueGate:
    vocabulary_sizes = PUBLICATION_BPE_VOCABULARY_CANDIDATES
    if set(bpe_gates) != set(vocabulary_sizes):
        raise ValueError("final value gate requires both BPE vocabulary controls")
    validate_publication_data_adequacy(data_adequacy)
    ordered_bpe_gates = tuple(bpe_gates[size] for size in vocabulary_sizes)
    for gate in (raw_gate, *ordered_bpe_gates):
        validate_publication_comparator_inference_gate(gate)
    shared_runtime_fields = (
        "seed_order",
        "candidate_checkpoint_sha256",
        "candidate_model_config_sha256",
        "candidate_tokenizer_sha256",
        "candidate_utf8_transition_sha256",
        "runtime_source_sha256",
        "timing_scope_audit_sha256",
        "case_manifest_sha256",
        "raw_prompt_array_sha256",
        "raw_replay_continuation_array_sha256",
        "candidate_prompt_unit_array_sha256",
        "candidate_replay_unit_array_sha256",
        "timing_scope_contract",
        "protocol_version",
    )
    for gate in (raw_gate, *ordered_bpe_gates):
        validate_publication_runtime_evidence(gate.runtime_evidence)
        if (
            gate.candidate_key != gate.runtime_evidence.candidate_key
            or gate.comparator_key != gate.runtime_evidence.comparator_key
            or gate.comparator_family
            != gate.runtime_evidence.comparator_family
        ):
            raise ValueError("final value gate runtime identity is inconsistent")
    raw_lineage = raw_gate.runtime_evidence.lineage
    if any(
        getattr(gate.runtime_evidence.lineage, field)
        != getattr(raw_lineage, field)
        for gate in ordered_bpe_gates
        for field in shared_runtime_fields
    ):
        raise ValueError(
            "final value gate requires one shared candidate and case lineage"
        )
    comparator_keys = (
        raw_gate.comparator_key,
        *(gate.comparator_key for gate in ordered_bpe_gates),
    )
    candidate_keys = {
        raw_gate.candidate_key,
        *(gate.candidate_key for gate in ordered_bpe_gates),
    }
    all_gates = (raw_gate, *ordered_bpe_gates)
    downstream_identity = raw_gate.downstream_evidence.identity_sha256
    if (
        any(
            gate.downstream_evidence.identity_sha256 != downstream_identity
            for gate in ordered_bpe_gates
        )
        or data_adequacy.downstream_evidence_sha256 != downstream_identity
    ):
        raise ValueError("final value gate requires one downstream evidence graph")
    runtime_pairs = {
        gate.comparator_key: publication_runtime_model_snapshots(
            gate.runtime_evidence.lineage
        )
        for gate in all_gates
    }
    bpb_pairs = {
        gate.comparator_key: (
            gate.bpb.candidate_snapshot,
            gate.bpb.comparator_snapshot,
        )
        for gate in all_gates
    }
    curve_final_snapshots = {
        lock.model_key: lock.final_snapshot
        for lock in data_adequacy.learning_curve_model_locks
    }
    downstream_snapshots = {
        snapshot.model_key: snapshot
        for snapshot in raw_gate.downstream_evidence.model_snapshots
    }
    model_lock_graph = build_publication_model_lock_graph(
        runtime_pairs=runtime_pairs,
        bpb_pairs=bpb_pairs,
        downstream_snapshots=downstream_snapshots,
        learning_curve_final_snapshots=curve_final_snapshots,
        runtime_evidence_sha256={
            gate.comparator_key: gate.runtime_evidence.identity_sha256
            for gate in all_gates
        },
        bpb_evidence_sha256={
            gate.comparator_key: gate.bpb.identity_sha256 for gate in all_gates
        },
        downstream_evidence_sha256=downstream_identity,
        learning_curve_evidence_sha256=data_adequacy.identity_sha256,
    )
    if (
        not isinstance(data_adequacy, PublicationDataAdequacy)
        or len(candidate_keys) != 1
        or data_adequacy.candidate_key != raw_gate.candidate_key
        or data_adequacy.raw_comparator_key != raw_gate.comparator_key
        or data_adequacy.bpe_data_matched_keys
        != tuple(gate.comparator_key for gate in ordered_bpe_gates)
        or tuple(gate.comparator_key for gate in ordered_bpe_gates)
        != tuple(
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size]
            for size in vocabulary_sizes
        )
        or raw_gate.comparator_family != "raw_byte"
        or any(
            gate.comparator_family != "standard_bpe"
            for gate in ordered_bpe_gates
        )
        or len(set(comparator_keys)) != len(comparator_keys)
    ):
        raise ValueError("final value gate requires distinct raw and BPE controls")
    bpe_passes = tuple(gate.overall_pass for gate in ordered_bpe_gates)
    all_bpe_pass = all(bpe_passes)
    data_adequacy_pass = data_adequacy.overall_pass
    overall = raw_gate.overall_pass and all_bpe_pass and data_adequacy_pass
    if overall:
        claim_level = "broad_korean_inference_efficiency_candidate"
        status = "pass"
    elif raw_gate.overall_pass and all_bpe_pass:
        claim_level = "mac_mechanism_scale_only"
        status = "fail_data_adequacy"
    elif raw_gate.overall_pass and any(bpe_passes):
        claim_level = "bpe_vocabulary_specific_only"
        status = "fail_bpe_vocabulary_robustness_gate"
    elif raw_gate.overall_pass:
        claim_level = "byte_latent_family_internal_only"
        status = "fail_broad_bpe_gate"
    else:
        claim_level = "no_positive_inference_efficiency_claim"
        status = "fail_final_value_gate"
    provisional = PublicationFinalValueGate(
        candidate_key=raw_gate.candidate_key,
        raw_comparator_key=raw_gate.comparator_key,
        bpe_vocabulary_sizes=vocabulary_sizes,
        bpe_comparator_keys=tuple(
            gate.comparator_key for gate in ordered_bpe_gates
        ),
        raw_comparator_pass=raw_gate.overall_pass,
        bpe_comparator_passes=bpe_passes,
        all_bpe_comparators_pass=all_bpe_pass,
        data_adequacy_pass=data_adequacy_pass,
        model_lock_graph=model_lock_graph,
        raw_comparator_gate_sha256=raw_gate.identity_sha256,
        bpe_comparator_gate_sha256=tuple(
            gate.identity_sha256 for gate in ordered_bpe_gates
        ),
        data_adequacy_sha256=data_adequacy.identity_sha256,
        overall_pass=overall,
        claim_level=claim_level,
        status=status,
        identity_sha256="",
    )
    payload = provisional.to_dict()
    payload.pop("identity_sha256")
    result = PublicationFinalValueGate(
        **{
            **provisional.to_dict(),
            "model_lock_graph": model_lock_graph,
            "identity_sha256": canonical_sha256(payload),
        }
    )
    validate_publication_final_value_gate(result)
    return result


def validate_publication_final_value_gate(
    gate: PublicationFinalValueGate,
) -> None:
    if not isinstance(gate, PublicationFinalValueGate):
        raise ValueError("publication final-value evidence is invalid")
    validate_publication_model_lock_graph(gate.model_lock_graph)
    all_bpe_pass = all(gate.bpe_comparator_passes)
    overall = gate.raw_comparator_pass and all_bpe_pass and gate.data_adequacy_pass
    if overall:
        claim_level = "broad_korean_inference_efficiency_candidate"
        status = "pass"
    elif gate.raw_comparator_pass and all_bpe_pass:
        claim_level = "mac_mechanism_scale_only"
        status = "fail_data_adequacy"
    elif gate.raw_comparator_pass and any(gate.bpe_comparator_passes):
        claim_level = "bpe_vocabulary_specific_only"
        status = "fail_bpe_vocabulary_robustness_gate"
    elif gate.raw_comparator_pass:
        claim_level = "byte_latent_family_internal_only"
        status = "fail_broad_bpe_gate"
    else:
        claim_level = "no_positive_inference_efficiency_claim"
        status = "fail_final_value_gate"
    payload = gate.to_dict()
    payload.pop("identity_sha256")
    if (
        gate.candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or gate.raw_comparator_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
        or gate.bpe_vocabulary_sizes != PUBLICATION_BPE_VOCABULARY_CANDIDATES
        or gate.bpe_comparator_keys
        != tuple(
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size]
            for size in PUBLICATION_BPE_VOCABULARY_CANDIDATES
        )
        or len(gate.bpe_comparator_passes)
        != len(PUBLICATION_BPE_VOCABULARY_CANDIDATES)
        or gate.all_bpe_comparators_pass != all_bpe_pass
        or gate.model_lock_graph.model_snapshots[0].model_key
        != gate.candidate_key
        or len(gate.bpe_comparator_gate_sha256)
        != len(PUBLICATION_BPE_VOCABULARY_CANDIDATES)
        or not all(
            is_sha256(value)
            for value in (
                gate.raw_comparator_gate_sha256,
                *gate.bpe_comparator_gate_sha256,
                gate.data_adequacy_sha256,
                gate.identity_sha256,
            )
        )
        or gate.overall_pass != overall
        or gate.claim_level != claim_level
        or gate.status != status
        or gate.identity_sha256 != canonical_sha256(payload)
    ):
        raise ValueError("publication final-value evidence is inconsistent")
