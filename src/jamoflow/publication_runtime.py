"""Identity-bound publication-scale actual-inference evidence.

This module accepts raw numeric evidence, reconstructs correctness, timing, and
valid-output properties, and returns immutable objects tied to one exact model
pair.  The final publication gate never accepts naked runtime booleans or
unbound latency arrays.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .actual_inference_protocol import timing_environment_eligible
from .inference_benchmark import (
    MultiSeedPairedLatency,
    multiseed_paired_latency,
    timing_order_schedule,
)
from .publication_reference import (
    PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
    PUBLICATION_AUXILIARY_KINDS,
    PUBLICATION_AUXILIARY_NONE,
    PublicationEntropyRouterBundle,
    PublicationRawReferenceDescriptor,
    validate_publication_entropy_router_bundle,
    validate_publication_entropy_router_bundle_family,
    validate_publication_raw_reference_descriptor,
)
from .publication_protocol import (
    ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS,
    ACTUAL_INFERENCE_FREE_RUNNING_MAXIMUM_STEPS,
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
    PUBLICATION_RUNTIME_COMPONENTS,
    PUBLICATION_RUNTIME_EQUIVALENCE_MINIMUM_VECTORS_PER_PAIR,
    PUBLICATION_RUNTIME_EQUIVALENCE_PATHS,
    PUBLICATION_RUNTIME_MEASURED_CASES,
    PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES,
    PUBLICATION_RUNTIME_MODES,
    PUBLICATION_RUNTIME_PROMPT_BYTES,
    PUBLICATION_RUNTIME_PROTOCOL_VERSION,
    PUBLICATION_RUNTIME_REPETITIONS,
    PUBLICATION_RUNTIME_ROLES,
    PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED,
    PUBLICATION_RUNTIME_TIMING_ORDER_SEED,
    PUBLICATION_RUNTIME_TIMING_SCOPE_CONTRACT,
    PUBLICATION_RUNTIME_WARMUP_CASES,
)


PUBLICATION_RUNTIME_EQUIVALENCE_RTOL = 2e-5
PUBLICATION_RUNTIME_EQUIVALENCE_ATOL = 2e-5
PUBLICATION_RUNTIME_DIAGNOSTICS = (
    "prompt_model_units",
    "emitted_output_bytes",
    "emitted_model_units",
    "decode_forward_steps",
    "runtime_observed_model_units",
    "overshoot_bytes",
    "valid_output_stop",
    "final_utf8_accept",
    "transition_trace_valid",
    "replacement_character_free",
    "output_codepoints",
    "router_observed_model_units",
    "router_cached_model_units",
    "router_scored_model_units",
    "router_forward_calls",
)


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _identity_payload(value: Any) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("identity_sha256")
    return payload


def _validate_model_pair(
    candidate_key: str,
    comparator_key: str,
    comparator_family: str,
) -> None:
    if (
        candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or comparator_family not in {"raw_byte", "standard_bpe"}
        or (
            comparator_family == "raw_byte"
            and comparator_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
        )
        or (
            comparator_family == "standard_bpe"
            and comparator_key
            not in PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values()
        )
    ):
        raise ValueError("publication runtime model-pair identity is invalid")


@dataclass(frozen=True, slots=True)
class PublicationRuntimeLineage:
    candidate_key: str
    comparator_key: str
    comparator_family: str
    seed_order: tuple[int, ...]
    candidate_checkpoint_sha256: tuple[str, ...]
    comparator_checkpoint_sha256: tuple[str, ...]
    candidate_model_config_sha256: tuple[str, ...]
    comparator_model_config_sha256: tuple[str, ...]
    raw_reference_descriptor: PublicationRawReferenceDescriptor | None
    comparator_auxiliary_kind: str
    comparator_auxiliary_bundles: tuple[PublicationEntropyRouterBundle, ...]
    comparator_auxiliary_checkpoint_sha256: tuple[str, ...]
    comparator_auxiliary_config_sha256: tuple[str, ...]
    comparator_auxiliary_calibration_sha256: tuple[str, ...]
    candidate_tokenizer_sha256: str
    comparator_tokenizer_sha256: str
    candidate_utf8_transition_sha256: str
    comparator_utf8_transition_sha256: str
    runtime_source_sha256: str
    timing_scope_audit_sha256: str
    case_manifest_sha256: str
    raw_prompt_array_sha256: str
    raw_replay_continuation_array_sha256: str
    candidate_prompt_unit_array_sha256: str
    comparator_prompt_unit_array_sha256: str
    candidate_replay_unit_array_sha256: str
    comparator_replay_unit_array_sha256: str
    unitization_audit_sha256: str
    timing_scope_contract: str
    protocol_version: int
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_runtime_lineage(
    *,
    candidate_key: str,
    comparator_key: str,
    comparator_family: str,
    candidate_checkpoint_sha256: Mapping[int, str],
    comparator_checkpoint_sha256: Mapping[int, str],
    candidate_model_config_sha256: Mapping[int, str],
    comparator_model_config_sha256: Mapping[int, str],
    raw_reference_descriptor: PublicationRawReferenceDescriptor | None = None,
    comparator_auxiliary_bundles: (
        Mapping[int, PublicationEntropyRouterBundle] | None
    ) = None,
    candidate_tokenizer_sha256: str,
    comparator_tokenizer_sha256: str,
    candidate_utf8_transition_sha256: str,
    comparator_utf8_transition_sha256: str,
    runtime_source_sha256: str,
    timing_scope_audit_sha256: str,
    case_manifest_sha256: str,
    raw_prompt_array_sha256: str,
    raw_replay_continuation_array_sha256: str,
    candidate_prompt_unit_array_sha256: str,
    comparator_prompt_unit_array_sha256: str,
    candidate_replay_unit_array_sha256: str,
    comparator_replay_unit_array_sha256: str,
    unitization_audit_sha256: str,
) -> PublicationRuntimeLineage:
    _validate_model_pair(candidate_key, comparator_key, comparator_family)
    if comparator_family == "raw_byte":
        if raw_reference_descriptor is None:
            raise ValueError("raw runtime requires its sealed reference descriptor")
        validate_publication_raw_reference_descriptor(raw_reference_descriptor)
        comparator_auxiliary_kind = raw_reference_descriptor.auxiliary_kind
    else:
        if raw_reference_descriptor is not None:
            raise ValueError("BPE runtime cannot bind a raw-reference descriptor")
        comparator_auxiliary_kind = PUBLICATION_AUXILIARY_NONE
    mappings = (
        candidate_checkpoint_sha256,
        comparator_checkpoint_sha256,
        candidate_model_config_sha256,
        comparator_model_config_sha256,
    )
    if any(set(values) != set(PUBLICATION_PRETRAIN_SEEDS) for values in mappings):
        raise ValueError("publication runtime lineage requires every paired seed")
    ordered = tuple(
        tuple(values[seed] for seed in PUBLICATION_PRETRAIN_SEEDS)
        for values in mappings
    )
    if comparator_auxiliary_kind == PUBLICATION_AUXILIARY_NONE:
        if comparator_auxiliary_bundles is not None:
            raise ValueError("runtime without an auxiliary model cannot bind one")
        ordered_bundles: tuple[PublicationEntropyRouterBundle, ...] = ()
        auxiliary_checkpoints: tuple[str, ...] = ()
        auxiliary_configurations: tuple[str, ...] = ()
        auxiliary_calibrations: tuple[str, ...] = ()
    else:
        if (
            comparator_auxiliary_bundles is None
            or set(comparator_auxiliary_bundles)
            != set(PUBLICATION_PRETRAIN_SEEDS)
            or raw_reference_descriptor is None
        ):
            raise ValueError("runtime auxiliary lineage requires every paired seed")
        ordered_bundles = tuple(
            comparator_auxiliary_bundles[seed]
            for seed in PUBLICATION_PRETRAIN_SEEDS
        )
        for seed, bundle in zip(
            PUBLICATION_PRETRAIN_SEEDS,
            ordered_bundles,
            strict=True,
        ):
            validate_publication_entropy_router_bundle(
                bundle,
                raw_reference_descriptor,
            )
            if bundle.seed != seed:
                raise ValueError("runtime auxiliary bundle seed order is inconsistent")
        validate_publication_entropy_router_bundle_family(
            ordered_bundles,
            raw_reference_descriptor,
        )
        auxiliary_checkpoints = tuple(
            bundle.router_checkpoint_state_sha256 for bundle in ordered_bundles
        )
        auxiliary_configurations = tuple(
            bundle.router_config_sha256 for bundle in ordered_bundles
        )
        auxiliary_calibrations = tuple(
            bundle.identity_sha256 for bundle in ordered_bundles
        )
    scalar_hashes = (
        candidate_tokenizer_sha256,
        comparator_tokenizer_sha256,
        candidate_utf8_transition_sha256,
        comparator_utf8_transition_sha256,
        runtime_source_sha256,
        timing_scope_audit_sha256,
        case_manifest_sha256,
        raw_prompt_array_sha256,
        raw_replay_continuation_array_sha256,
        candidate_prompt_unit_array_sha256,
        comparator_prompt_unit_array_sha256,
        candidate_replay_unit_array_sha256,
        comparator_replay_unit_array_sha256,
        unitization_audit_sha256,
        *auxiliary_checkpoints,
        *auxiliary_configurations,
        *auxiliary_calibrations,
    )
    if not all(_is_sha256(value) for group in ordered for value in group) or not all(
        _is_sha256(value) for value in scalar_hashes
    ) or (
        len(set(ordered[0])) != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(set(ordered[1])) != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(set(ordered[2])) != 1
        or len(set(ordered[3])) != 1
        or (
            comparator_auxiliary_kind == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
            and (
                len(set(auxiliary_checkpoints)) != len(auxiliary_checkpoints)
                or len(set(auxiliary_configurations)) != 1
                or len(set(auxiliary_calibrations)) != len(auxiliary_calibrations)
                or not set(ordered[1]).isdisjoint(auxiliary_checkpoints)
                or not set(auxiliary_checkpoints).isdisjoint(auxiliary_calibrations)
            )
        )
    ):
        raise ValueError("publication runtime lineage hashes are malformed")
    provisional = PublicationRuntimeLineage(
        candidate_key=candidate_key,
        comparator_key=comparator_key,
        comparator_family=comparator_family,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        candidate_checkpoint_sha256=ordered[0],
        comparator_checkpoint_sha256=ordered[1],
        candidate_model_config_sha256=ordered[2],
        comparator_model_config_sha256=ordered[3],
        raw_reference_descriptor=raw_reference_descriptor,
        comparator_auxiliary_kind=comparator_auxiliary_kind,
        comparator_auxiliary_bundles=ordered_bundles,
        comparator_auxiliary_checkpoint_sha256=auxiliary_checkpoints,
        comparator_auxiliary_config_sha256=auxiliary_configurations,
        comparator_auxiliary_calibration_sha256=auxiliary_calibrations,
        candidate_tokenizer_sha256=candidate_tokenizer_sha256,
        comparator_tokenizer_sha256=comparator_tokenizer_sha256,
        candidate_utf8_transition_sha256=candidate_utf8_transition_sha256,
        comparator_utf8_transition_sha256=comparator_utf8_transition_sha256,
        runtime_source_sha256=runtime_source_sha256,
        timing_scope_audit_sha256=timing_scope_audit_sha256,
        case_manifest_sha256=case_manifest_sha256,
        raw_prompt_array_sha256=raw_prompt_array_sha256,
        raw_replay_continuation_array_sha256=(
            raw_replay_continuation_array_sha256
        ),
        candidate_prompt_unit_array_sha256=(
            candidate_prompt_unit_array_sha256
        ),
        comparator_prompt_unit_array_sha256=(
            comparator_prompt_unit_array_sha256
        ),
        candidate_replay_unit_array_sha256=(
            candidate_replay_unit_array_sha256
        ),
        comparator_replay_unit_array_sha256=(
            comparator_replay_unit_array_sha256
        ),
        unitization_audit_sha256=unitization_audit_sha256,
        timing_scope_contract=PUBLICATION_RUNTIME_TIMING_SCOPE_CONTRACT,
        protocol_version=PUBLICATION_RUNTIME_PROTOCOL_VERSION,
        identity_sha256="",
    )
    lineage = PublicationRuntimeLineage(
        **{
            **provisional.to_dict(),
            "raw_reference_descriptor": raw_reference_descriptor,
            "comparator_auxiliary_bundles": ordered_bundles,
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_runtime_lineage(lineage)
    return lineage


def validate_publication_runtime_lineage(lineage: PublicationRuntimeLineage) -> None:
    _validate_model_pair(
        lineage.candidate_key,
        lineage.comparator_key,
        lineage.comparator_family,
    )
    hashes = (
        *lineage.candidate_checkpoint_sha256,
        *lineage.comparator_checkpoint_sha256,
        *lineage.candidate_model_config_sha256,
        *lineage.comparator_model_config_sha256,
        *lineage.comparator_auxiliary_checkpoint_sha256,
        *lineage.comparator_auxiliary_config_sha256,
        *lineage.comparator_auxiliary_calibration_sha256,
        lineage.candidate_tokenizer_sha256,
        lineage.comparator_tokenizer_sha256,
        lineage.candidate_utf8_transition_sha256,
        lineage.comparator_utf8_transition_sha256,
        lineage.runtime_source_sha256,
        lineage.timing_scope_audit_sha256,
        lineage.case_manifest_sha256,
        lineage.raw_prompt_array_sha256,
        lineage.raw_replay_continuation_array_sha256,
        lineage.candidate_prompt_unit_array_sha256,
        lineage.comparator_prompt_unit_array_sha256,
        lineage.candidate_replay_unit_array_sha256,
        lineage.comparator_replay_unit_array_sha256,
        lineage.unitization_audit_sha256,
        lineage.identity_sha256,
    )
    descriptor = lineage.raw_reference_descriptor
    if lineage.comparator_family == "raw_byte":
        if descriptor is None:
            raise ValueError("publication runtime lineage is inconsistent")
        validate_publication_raw_reference_descriptor(descriptor)
        expected_auxiliary_kind = descriptor.auxiliary_kind
    else:
        expected_auxiliary_kind = PUBLICATION_AUXILIARY_NONE
    if (
        lineage.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or any(
            len(values) != len(PUBLICATION_PRETRAIN_SEEDS)
            for values in (
                lineage.candidate_checkpoint_sha256,
                lineage.comparator_checkpoint_sha256,
                lineage.candidate_model_config_sha256,
                lineage.comparator_model_config_sha256,
            )
        )
        or len(set(lineage.candidate_checkpoint_sha256))
        != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(set(lineage.comparator_checkpoint_sha256))
        != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(set(lineage.candidate_model_config_sha256)) != 1
        or len(set(lineage.comparator_model_config_sha256)) != 1
        or lineage.comparator_auxiliary_kind not in PUBLICATION_AUXILIARY_KINDS
        or lineage.comparator_auxiliary_kind != expected_auxiliary_kind
        or (lineage.comparator_family != "raw_byte" and descriptor is not None)
        or (
            lineage.comparator_auxiliary_kind == PUBLICATION_AUXILIARY_NONE
            and (
                lineage.comparator_auxiliary_bundles
                or lineage.comparator_auxiliary_checkpoint_sha256
                or lineage.comparator_auxiliary_config_sha256
                or lineage.comparator_auxiliary_calibration_sha256
            )
        )
        or (
            lineage.comparator_auxiliary_kind
            == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
            and (
                lineage.comparator_family != "raw_byte"
                or descriptor is None
                or len(lineage.comparator_auxiliary_bundles)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(lineage.comparator_auxiliary_checkpoint_sha256)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(lineage.comparator_auxiliary_config_sha256)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(lineage.comparator_auxiliary_calibration_sha256)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(set(lineage.comparator_auxiliary_checkpoint_sha256))
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(set(lineage.comparator_auxiliary_config_sha256)) != 1
                or len(set(lineage.comparator_auxiliary_calibration_sha256))
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or not set(lineage.comparator_checkpoint_sha256).isdisjoint(
                    lineage.comparator_auxiliary_checkpoint_sha256
                )
                or not set(
                    lineage.comparator_auxiliary_checkpoint_sha256
                ).isdisjoint(lineage.comparator_auxiliary_calibration_sha256)
            )
        )
        or not all(_is_sha256(value) for value in hashes)
        or lineage.timing_scope_contract
        != PUBLICATION_RUNTIME_TIMING_SCOPE_CONTRACT
        or lineage.protocol_version != PUBLICATION_RUNTIME_PROTOCOL_VERSION
        or lineage.identity_sha256
        != _canonical_sha256(_identity_payload(lineage))
    ):
        raise ValueError("publication runtime lineage is inconsistent")
    if (
        descriptor is not None
        and lineage.comparator_auxiliary_kind
        == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
    ):
        validate_publication_entropy_router_bundle_family(
            lineage.comparator_auxiliary_bundles,
            descriptor,
        )
        for index, (seed, bundle) in enumerate(
            zip(
                PUBLICATION_PRETRAIN_SEEDS,
                lineage.comparator_auxiliary_bundles,
                strict=True,
            )
        ):
            validate_publication_entropy_router_bundle(bundle, descriptor)
            if (
                bundle.seed != seed
                or bundle.router_checkpoint_state_sha256
                != lineage.comparator_auxiliary_checkpoint_sha256[index]
                or bundle.router_config_sha256
                != lineage.comparator_auxiliary_config_sha256[index]
                or bundle.identity_sha256
                != lineage.comparator_auxiliary_calibration_sha256[index]
            ):
                raise ValueError("publication runtime lineage is inconsistent")


@dataclass(frozen=True, slots=True)
class PublicationRuntimeEquivalence:
    lineage_identity_sha256: str
    seed_order: tuple[int, ...]
    roles: tuple[str, ...]
    paths: tuple[str, ...]
    comparison_pairs: int
    logit_vectors: int
    maximum_absolute_error: float
    argmax_match_rate: float
    allclose_pass: bool
    rtol: float
    atol: float
    array_manifest_sha256: str
    overall_pass: bool
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_runtime_equivalence(
    lineage: PublicationRuntimeLineage,
    comparisons: Mapping[
        tuple[int, str, str],
        tuple[np.ndarray, np.ndarray],
    ],
) -> PublicationRuntimeEquivalence:
    validate_publication_runtime_lineage(lineage)
    expected_keys = {
        (seed, role, path)
        for seed in PUBLICATION_PRETRAIN_SEEDS
        for role in PUBLICATION_RUNTIME_ROLES
        for path in PUBLICATION_RUNTIME_EQUIVALENCE_PATHS
    }
    if set(comparisons) != expected_keys:
        raise ValueError("publication runtime equivalence design is incomplete")
    entries: list[dict[str, object]] = []
    allclose_pass = True
    argmax_matches = 0
    vector_count = 0
    maximum_error = 0.0
    for key in sorted(comparisons):
        full = np.asarray(comparisons[key][0])
        incremental = np.asarray(comparisons[key][1])
        if (
            full.shape != incremental.shape
            or full.ndim < 1
            or full.shape[-1] < 2
            or not np.issubdtype(full.dtype, np.floating)
            or not np.issubdtype(incremental.dtype, np.floating)
            or not np.isfinite(full).all()
            or not np.isfinite(incremental).all()
        ):
            raise ValueError("publication runtime equivalence logits are malformed")
        differences = np.abs(full.astype(np.float64) - incremental.astype(np.float64))
        maximum_error = max(maximum_error, float(differences.max(initial=0.0)))
        close = bool(
            np.allclose(
                full,
                incremental,
                rtol=PUBLICATION_RUNTIME_EQUIVALENCE_RTOL,
                atol=PUBLICATION_RUNTIME_EQUIVALENCE_ATOL,
            )
        )
        full_argmax = np.argmax(full, axis=-1)
        incremental_argmax = np.argmax(incremental, axis=-1)
        if (
            full_argmax.size
            < PUBLICATION_RUNTIME_EQUIVALENCE_MINIMUM_VECTORS_PER_PAIR
        ):
            raise ValueError(
                "publication runtime equivalence coverage is insufficient"
            )
        matches = int(np.count_nonzero(full_argmax == incremental_argmax))
        count = int(full_argmax.size)
        allclose_pass = allclose_pass and close
        argmax_matches += matches
        vector_count += count
        entries.append(
            {
                "seed": key[0],
                "role": key[1],
                "path": key[2],
                "shape": list(full.shape),
                "full_sha256": _array_sha256(full),
                "incremental_sha256": _array_sha256(incremental),
                "allclose": close,
                "argmax_matches": matches,
                "logit_vectors": count,
            }
        )
    argmax_rate = argmax_matches / vector_count
    passed = bool(allclose_pass and argmax_matches == vector_count)
    provisional = PublicationRuntimeEquivalence(
        lineage_identity_sha256=lineage.identity_sha256,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        roles=PUBLICATION_RUNTIME_ROLES,
        paths=PUBLICATION_RUNTIME_EQUIVALENCE_PATHS,
        comparison_pairs=len(entries),
        logit_vectors=vector_count,
        maximum_absolute_error=maximum_error,
        argmax_match_rate=argmax_rate,
        allclose_pass=allclose_pass,
        rtol=PUBLICATION_RUNTIME_EQUIVALENCE_RTOL,
        atol=PUBLICATION_RUNTIME_EQUIVALENCE_ATOL,
        array_manifest_sha256=_canonical_sha256(entries),
        overall_pass=passed,
        identity_sha256="",
    )
    result = PublicationRuntimeEquivalence(
        **{
            **provisional.to_dict(),
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_runtime_equivalence(result, lineage)
    return result


def validate_publication_runtime_equivalence(
    evidence: PublicationRuntimeEquivalence,
    lineage: PublicationRuntimeLineage,
) -> None:
    expected_pairs = (
        len(PUBLICATION_PRETRAIN_SEEDS)
        * len(PUBLICATION_RUNTIME_ROLES)
        * len(PUBLICATION_RUNTIME_EQUIVALENCE_PATHS)
    )
    if (
        evidence.lineage_identity_sha256 != lineage.identity_sha256
        or evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or evidence.roles != PUBLICATION_RUNTIME_ROLES
        or evidence.paths != PUBLICATION_RUNTIME_EQUIVALENCE_PATHS
        or evidence.comparison_pairs != expected_pairs
        or evidence.logit_vectors
        < expected_pairs
        * PUBLICATION_RUNTIME_EQUIVALENCE_MINIMUM_VECTORS_PER_PAIR
        or not np.isfinite(evidence.maximum_absolute_error)
        or evidence.maximum_absolute_error < 0
        or not 0 <= evidence.argmax_match_rate <= 1
        or evidence.rtol != PUBLICATION_RUNTIME_EQUIVALENCE_RTOL
        or evidence.atol != PUBLICATION_RUNTIME_EQUIVALENCE_ATOL
        or not _is_sha256(evidence.array_manifest_sha256)
        or not _is_sha256(evidence.identity_sha256)
        or evidence.overall_pass
        != bool(evidence.allclose_pass and evidence.argmax_match_rate == 1.0)
        or evidence.identity_sha256
        != _canonical_sha256(_identity_payload(evidence))
    ):
        raise ValueError("publication runtime equivalence evidence is inconsistent")


def publication_timing_array_keys() -> tuple[str, ...]:
    return tuple(
        f"{mode}__{component}__{role}"
        for mode in PUBLICATION_RUNTIME_MODES
        for component in PUBLICATION_RUNTIME_COMPONENTS
        for role in PUBLICATION_RUNTIME_ROLES
    )


@dataclass(frozen=True, slots=True)
class PublicationTimingEvidence:
    lineage_identity_sha256: str
    seed_order: tuple[int, ...]
    seed_execution_order_seed: int
    seed_execution_order: tuple[int, ...]
    prompt_count: int
    repetitions_per_prompt: int
    timing_order_seed: int
    schedule_sha256: str
    warmup_count: int
    warmup_order_seed: int
    warmup_schedule_sha256: str
    warmup_completion_sha256: str
    trial_artifact_sha256: str
    environment_sha256: str
    timing_arrays_sha256: str
    seed_execution_order_pass: bool
    schedule_pass: bool
    warmup_pass: bool
    environment_pass: bool
    component_identity_pass: bool
    controlled_replay_decode: MultiSeedPairedLatency
    free_running_end_to_end: MultiSeedPairedLatency
    overall_pass: bool
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_timing_evidence(
    lineage: PublicationRuntimeLineage,
    timing_arrays: Mapping[str, np.ndarray],
    timing_schedule: np.ndarray,
    warmup_schedule: np.ndarray,
    warmup_completion: np.ndarray,
    seed_execution_order: Sequence[int],
    environment_states: Mapping[int, Mapping[str, object]],
    trial_artifact_sha256: str,
) -> PublicationTimingEvidence:
    validate_publication_runtime_lineage(lineage)
    if not _is_sha256(trial_artifact_sha256):
        raise ValueError("publication timing artifact hash is malformed")
    expected_shape = (
        len(PUBLICATION_PRETRAIN_SEEDS),
        PUBLICATION_RUNTIME_MEASURED_CASES,
        PUBLICATION_RUNTIME_REPETITIONS,
    )
    if set(timing_arrays) != set(publication_timing_array_keys()):
        raise ValueError("publication timing array family is incomplete")
    arrays: dict[str, np.ndarray] = {}
    array_manifest: dict[str, str] = {}
    for key in publication_timing_array_keys():
        values = np.asarray(timing_arrays[key])
        if (
            values.shape != expected_shape
            or values.dtype != np.float64
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise ValueError("publication timing arrays must be positive float64")
        arrays[key] = values
        array_manifest[key] = _array_sha256(values)

    expected_schedule = timing_order_schedule(
        PUBLICATION_PRETRAIN_SEEDS,
        mode_count=len(PUBLICATION_RUNTIME_MODES),
        prompt_count=PUBLICATION_RUNTIME_MEASURED_CASES,
        repetitions=PUBLICATION_RUNTIME_REPETITIONS,
        random_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED,
    )
    schedule = np.asarray(timing_schedule)
    if schedule.shape != expected_schedule.shape or schedule.dtype != np.uint8:
        raise ValueError("publication timing schedule is malformed")
    schedule_pass = bool(np.array_equal(schedule, expected_schedule))

    expected_warmup_schedule = timing_order_schedule(
        PUBLICATION_PRETRAIN_SEEDS,
        mode_count=len(PUBLICATION_RUNTIME_MODES),
        prompt_count=PUBLICATION_RUNTIME_WARMUP_CASES,
        repetitions=1,
        random_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED + 1,
    )
    observed_warmup_schedule = np.asarray(warmup_schedule)
    if (
        observed_warmup_schedule.shape != expected_warmup_schedule.shape
        or observed_warmup_schedule.dtype != np.uint8
    ):
        raise ValueError("publication warmup schedule is malformed")
    expected_warmup_completion = np.ones(
        (
            len(PUBLICATION_PRETRAIN_SEEDS),
            len(PUBLICATION_RUNTIME_MODES),
            PUBLICATION_RUNTIME_WARMUP_CASES,
            len(PUBLICATION_RUNTIME_ROLES),
        ),
        dtype=np.uint8,
    )
    observed_warmup_completion = np.asarray(warmup_completion)
    if (
        observed_warmup_completion.shape != expected_warmup_completion.shape
        or observed_warmup_completion.dtype != np.uint8
    ):
        raise ValueError("publication warmup completion trace is malformed")
    warmup_pass = bool(
        np.array_equal(observed_warmup_schedule, expected_warmup_schedule)
        and np.array_equal(
            observed_warmup_completion,
            expected_warmup_completion,
        )
    )

    expected_seed_execution_order = tuple(
        PUBLICATION_PRETRAIN_SEEDS[index]
        for index in np.random.default_rng(
            PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED
        ).permutation(len(PUBLICATION_PRETRAIN_SEEDS))
    )
    observed_seed_execution_order = tuple(int(value) for value in seed_execution_order)
    seed_execution_order_pass = bool(
        observed_seed_execution_order == expected_seed_execution_order
    )

    if set(environment_states) != set(PUBLICATION_PRETRAIN_SEEDS):
        raise ValueError("publication timing environments require every seed")
    environment_pass = True
    environment_payload: dict[str, object] = {}
    for seed in PUBLICATION_PRETRAIN_SEEDS:
        state = environment_states[seed]
        if set(state) != {"start", "end"}:
            raise ValueError("publication timing environment endpoints differ")
        start = state["start"]
        end = state["end"]
        if not isinstance(start, Mapping) or not isinstance(end, Mapping):
            raise ValueError("publication timing environment state is malformed")
        start_pass = timing_environment_eligible(start)
        end_pass = timing_environment_eligible(end)
        environment_pass = environment_pass and start_pass and end_pass
        environment_payload[str(seed)] = {
            "start": start,
            "end": end,
            "start_pass": start_pass,
            "end_pass": end_pass,
        }

    component_pass = True
    for mode in PUBLICATION_RUNTIME_MODES:
        for role in PUBLICATION_RUNTIME_ROLES:
            total = arrays[f"{mode}__end_to_end_ms__{role}"]
            parts = (
                arrays[f"{mode}__ttft_ms__{role}"]
                + arrays[f"{mode}__decode_ms__{role}"]
            )
            component_pass = component_pass and bool(
                np.allclose(total, parts, rtol=0, atol=1e-9)
            )

    controlled = multiseed_paired_latency(
        arrays["controlled_replay__decode_ms__candidate"],
        arrays["controlled_replay__decode_ms__reference"],
        PUBLICATION_PRETRAIN_SEEDS,
        bootstrap_repetitions=ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS,
        bootstrap_seed=20_260_816,
    )
    free = multiseed_paired_latency(
        arrays["free_running_utf8_greedy__end_to_end_ms__candidate"],
        arrays["free_running_utf8_greedy__end_to_end_ms__reference"],
        PUBLICATION_PRETRAIN_SEEDS,
        bootstrap_repetitions=ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS,
        bootstrap_seed=20_260_817,
    )
    passed = bool(
        seed_execution_order_pass
        and schedule_pass
        and warmup_pass
        and environment_pass
        and component_pass
    )
    provisional = PublicationTimingEvidence(
        lineage_identity_sha256=lineage.identity_sha256,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        seed_execution_order_seed=PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED,
        seed_execution_order=observed_seed_execution_order,
        prompt_count=PUBLICATION_RUNTIME_MEASURED_CASES,
        repetitions_per_prompt=PUBLICATION_RUNTIME_REPETITIONS,
        timing_order_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED,
        schedule_sha256=_array_sha256(schedule),
        warmup_count=PUBLICATION_RUNTIME_WARMUP_CASES,
        warmup_order_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED + 1,
        warmup_schedule_sha256=_array_sha256(observed_warmup_schedule),
        warmup_completion_sha256=_array_sha256(observed_warmup_completion),
        trial_artifact_sha256=trial_artifact_sha256,
        environment_sha256=_canonical_sha256(environment_payload),
        timing_arrays_sha256=_canonical_sha256(array_manifest),
        seed_execution_order_pass=seed_execution_order_pass,
        schedule_pass=schedule_pass,
        warmup_pass=warmup_pass,
        environment_pass=environment_pass,
        component_identity_pass=component_pass,
        controlled_replay_decode=controlled,
        free_running_end_to_end=free,
        overall_pass=passed,
        identity_sha256="",
    )
    result = PublicationTimingEvidence(
        **{
            **provisional.to_dict(),
            "controlled_replay_decode": controlled,
            "free_running_end_to_end": free,
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_timing_evidence(result, lineage)
    return result


def _validate_publication_latency_summary(
    summary: MultiSeedPairedLatency,
    *,
    bootstrap_seed: int,
) -> None:
    if not isinstance(summary, MultiSeedPairedLatency):
        raise ValueError("publication latency summary has the wrong type")
    expected_per_seed = {str(seed) for seed in PUBLICATION_PRETRAIN_SEEDS}
    finite_values = (
        summary.candidate_median_ms,
        summary.reference_median_ms,
        summary.crossed_median_latency_reduction,
        summary.bootstrap_percentile_95_lower,
        summary.bootstrap_percentile_95_upper,
        summary.median_seed_point_reduction,
    )
    if (
        summary.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or summary.seed_count != len(PUBLICATION_PRETRAIN_SEEDS)
        or summary.prompt_count != PUBLICATION_RUNTIME_MEASURED_CASES
        or summary.repetitions_per_prompt != PUBLICATION_RUNTIME_REPETITIONS
        or summary.bootstrap_repetitions
        != ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS
        or summary.bootstrap_seed != bootstrap_seed
        or summary.bootstrap_design != "crossed model seeds x shared prompts"
        or not all(np.isfinite(value) for value in finite_values)
        or summary.candidate_median_ms <= 0
        or summary.reference_median_ms <= 0
        or summary.bootstrap_percentile_95_lower
        > summary.bootstrap_percentile_95_upper
        or not 0 <= summary.positive_seed_count <= len(PUBLICATION_PRETRAIN_SEEDS)
        or set(summary.per_seed) != expected_per_seed
    ):
        raise ValueError("publication latency summary is inconsistent")
    expected_keys = {
        "prompt_count",
        "repetitions_per_prompt",
        "candidate_median_ms",
        "reference_median_ms",
        "median_latency_reduction",
        "mean_paired_prompt_reduction",
        "bootstrap_repetitions",
        "bootstrap_seed",
        "bootstrap_percentile_95_lower",
        "bootstrap_percentile_95_upper",
    }
    for index, seed in enumerate(PUBLICATION_PRETRAIN_SEEDS):
        values = summary.per_seed[str(seed)]
        if set(values) != expected_keys:
            raise ValueError("publication per-seed latency fields are incomplete")
        numeric = tuple(values[key] for key in expected_keys - {
            "prompt_count",
            "repetitions_per_prompt",
            "bootstrap_repetitions",
            "bootstrap_seed",
        })
        if (
            values["prompt_count"] != PUBLICATION_RUNTIME_MEASURED_CASES
            or values["repetitions_per_prompt"]
            != PUBLICATION_RUNTIME_REPETITIONS
            or values["bootstrap_repetitions"]
            != ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS
            or values["bootstrap_seed"] != bootstrap_seed + index + 1
            or not all(np.isfinite(value) for value in numeric)
            or values["candidate_median_ms"] <= 0
            or values["reference_median_ms"] <= 0
            or values["bootstrap_percentile_95_lower"]
            > values["bootstrap_percentile_95_upper"]
        ):
            raise ValueError("publication per-seed latency summary is inconsistent")


def validate_publication_timing_evidence(
    evidence: PublicationTimingEvidence,
    lineage: PublicationRuntimeLineage,
) -> None:
    expected_schedule = timing_order_schedule(
        PUBLICATION_PRETRAIN_SEEDS,
        mode_count=len(PUBLICATION_RUNTIME_MODES),
        prompt_count=PUBLICATION_RUNTIME_MEASURED_CASES,
        repetitions=PUBLICATION_RUNTIME_REPETITIONS,
        random_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED,
    )
    expected_warmup_schedule = timing_order_schedule(
        PUBLICATION_PRETRAIN_SEEDS,
        mode_count=len(PUBLICATION_RUNTIME_MODES),
        prompt_count=PUBLICATION_RUNTIME_WARMUP_CASES,
        repetitions=1,
        random_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED + 1,
    )
    expected_warmup_completion = np.ones(
        (
            len(PUBLICATION_PRETRAIN_SEEDS),
            len(PUBLICATION_RUNTIME_MODES),
            PUBLICATION_RUNTIME_WARMUP_CASES,
            len(PUBLICATION_RUNTIME_ROLES),
        ),
        dtype=np.uint8,
    )
    expected_seed_execution_order = tuple(
        PUBLICATION_PRETRAIN_SEEDS[index]
        for index in np.random.default_rng(
            PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED
        ).permutation(len(PUBLICATION_PRETRAIN_SEEDS))
    )
    _validate_publication_latency_summary(
        evidence.controlled_replay_decode,
        bootstrap_seed=20_260_816,
    )
    _validate_publication_latency_summary(
        evidence.free_running_end_to_end,
        bootstrap_seed=20_260_817,
    )
    if (
        evidence.lineage_identity_sha256 != lineage.identity_sha256
        or evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or evidence.seed_execution_order_seed
        != PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED
        or evidence.seed_execution_order != expected_seed_execution_order
        or evidence.prompt_count != PUBLICATION_RUNTIME_MEASURED_CASES
        or evidence.repetitions_per_prompt != PUBLICATION_RUNTIME_REPETITIONS
        or evidence.timing_order_seed != PUBLICATION_RUNTIME_TIMING_ORDER_SEED
        or evidence.schedule_sha256 != _array_sha256(expected_schedule)
        or evidence.warmup_count != PUBLICATION_RUNTIME_WARMUP_CASES
        or evidence.warmup_order_seed
        != PUBLICATION_RUNTIME_TIMING_ORDER_SEED + 1
        or evidence.warmup_schedule_sha256
        != _array_sha256(expected_warmup_schedule)
        or evidence.warmup_completion_sha256
        != _array_sha256(expected_warmup_completion)
        or not all(
            _is_sha256(value)
            for value in (
                evidence.schedule_sha256,
                evidence.warmup_schedule_sha256,
                evidence.warmup_completion_sha256,
                evidence.trial_artifact_sha256,
                evidence.environment_sha256,
                evidence.timing_arrays_sha256,
                evidence.identity_sha256,
            )
        )
        or evidence.overall_pass
        != bool(
            evidence.seed_execution_order_pass
            and evidence.schedule_pass
            and evidence.warmup_pass
            and evidence.environment_pass
            and evidence.component_identity_pass
        )
        or not evidence.seed_execution_order_pass
        or not evidence.schedule_pass
        or not evidence.warmup_pass
        or evidence.identity_sha256
        != _canonical_sha256(_identity_payload(evidence))
    ):
        raise ValueError("publication timing evidence is inconsistent")


def publication_output_diagnostic_keys() -> tuple[str, ...]:
    return tuple(
        f"{mode}__{diagnostic}__{role}"
        for mode in PUBLICATION_RUNTIME_MODES
        for diagnostic in PUBLICATION_RUNTIME_DIAGNOSTICS
        for role in PUBLICATION_RUNTIME_ROLES
    )


@dataclass(frozen=True, slots=True)
class PublicationValidOutputEvidence:
    lineage_identity_sha256: str
    seed_order: tuple[int, ...]
    minimum_output_bytes: int
    maximum_generation_steps: int
    candidate_maximum_unit_bytes: int
    comparator_maximum_unit_bytes: int
    trial_artifact_sha256: str
    output_trace_artifact_sha256: str
    output_trace_audit_sha256: str
    diagnostic_arrays_sha256: str
    deterministic_diagnostics_pass: bool
    router_execution_pass: bool
    controlled_contract_pass: bool
    free_running_contract_pass: bool
    candidate_completion_rate_by_seed: tuple[float, ...]
    comparator_completion_rate_by_seed: tuple[float, ...]
    candidate_replacement_free_rate_by_seed: tuple[float, ...]
    comparator_replacement_free_rate_by_seed: tuple[float, ...]
    overall_pass: bool
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _role_unit_bound_pass(
    emitted_bytes: np.ndarray,
    emitted_units: np.ndarray,
    maximum_unit_bytes: int,
    *,
    exact_byte_units: bool,
) -> bool:
    if exact_byte_units:
        return bool(np.array_equal(emitted_bytes, emitted_units))
    return bool(
        np.all(emitted_bytes >= emitted_units)
        and np.all(emitted_bytes <= emitted_units * maximum_unit_bytes)
    )


def build_publication_valid_output_evidence(
    lineage: PublicationRuntimeLineage,
    diagnostic_arrays: Mapping[str, np.ndarray],
    *,
    candidate_maximum_unit_bytes: int = 1,
    comparator_maximum_unit_bytes: int,
    trial_artifact_sha256: str,
    output_trace_artifact_sha256: str,
    output_trace_audit_sha256: str,
) -> PublicationValidOutputEvidence:
    validate_publication_runtime_lineage(lineage)
    if (
        candidate_maximum_unit_bytes != 1
        or comparator_maximum_unit_bytes <= 0
        or (
            lineage.comparator_family == "raw_byte"
            and comparator_maximum_unit_bytes != 1
        )
        or not all(
            _is_sha256(value)
            for value in (
                trial_artifact_sha256,
                output_trace_artifact_sha256,
                output_trace_audit_sha256,
            )
        )
    ):
        raise ValueError("publication valid-output unit-byte bounds are invalid")
    expected_shape = (
        len(PUBLICATION_PRETRAIN_SEEDS),
        PUBLICATION_RUNTIME_MEASURED_CASES,
        PUBLICATION_RUNTIME_REPETITIONS,
    )
    if set(diagnostic_arrays) != set(publication_output_diagnostic_keys()):
        raise ValueError("publication output diagnostic family is incomplete")
    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, str] = {}
    deterministic = True
    router_execution_pass = True
    for key in publication_output_diagnostic_keys():
        values = np.asarray(diagnostic_arrays[key])
        if values.shape != expected_shape or not np.issubdtype(
            values.dtype,
            np.integer,
        ):
            raise ValueError("publication output diagnostics must be integer arrays")
        values = values.astype(np.int64, copy=False)
        arrays[key] = values
        manifest[key] = _array_sha256(values)
        deterministic = deterministic and bool(
            np.all(values == values[:, :, :1])
        )

    controlled_pass = True
    free_pass = True
    completion: dict[str, list[float]] = {
        role: [] for role in PUBLICATION_RUNTIME_ROLES
    }
    replacement: dict[str, list[float]] = {
        role: [] for role in PUBLICATION_RUNTIME_ROLES
    }
    for mode in PUBLICATION_RUNTIME_MODES:
        for role in PUBLICATION_RUNTIME_ROLES:
            values = {
                name: arrays[f"{mode}__{name}__{role}"]
                for name in PUBLICATION_RUNTIME_DIAGNOSTICS
            }
            prompt_units = values["prompt_model_units"]
            emitted_bytes = values["emitted_output_bytes"]
            emitted_units = values["emitted_model_units"]
            steps = values["decode_forward_steps"]
            observed = values["runtime_observed_model_units"]
            overshoot = values["overshoot_bytes"]
            stopped = values["valid_output_stop"]
            accepted = values["final_utf8_accept"]
            transitions = values["transition_trace_valid"]
            replacement_free = values["replacement_character_free"]
            codepoints = values["output_codepoints"]
            router_observed = values["router_observed_model_units"]
            router_cached = values["router_cached_model_units"]
            router_scored = values["router_scored_model_units"]
            router_forward_calls = values["router_forward_calls"]
            binary = bool(
                np.isin(stopped, (0, 1)).all()
                and np.isin(accepted, (0, 1)).all()
                and np.isin(transitions, (0, 1)).all()
                and np.isin(replacement_free, (0, 1)).all()
            )
            common = bool(
                binary
                and np.all(prompt_units > 0)
                and np.all(prompt_units <= PUBLICATION_RUNTIME_PROMPT_BYTES)
                and np.all(emitted_bytes > 0)
                and np.all(emitted_units > 0)
                and np.all(
                    emitted_units <= ACTUAL_INFERENCE_FREE_RUNNING_MAXIMUM_STEPS
                )
                and np.array_equal(steps, emitted_units - 1)
                and np.all(steps >= 0)
                and np.array_equal(observed, prompt_units + steps)
                and np.all(observed > 0)
                and np.all(overshoot >= 0)
                and np.all(codepoints > 0)
                and np.all(codepoints <= emitted_bytes)
            )
            exact_bytes = role == "candidate" or lineage.comparator_family == "raw_byte"
            maximum_unit_bytes = (
                candidate_maximum_unit_bytes
                if role == "candidate"
                else comparator_maximum_unit_bytes
            )
            common = common and _role_unit_bound_pass(
                emitted_bytes,
                emitted_units,
                maximum_unit_bytes,
                exact_byte_units=exact_bytes,
            )
            if exact_bytes:
                common = common and bool(
                    np.all(prompt_units == PUBLICATION_RUNTIME_PROMPT_BYTES)
                )
            router_expected = bool(
                role == "reference"
                and lineage.comparator_auxiliary_kind
                == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
            )
            if router_expected:
                router_pass = bool(
                    np.array_equal(router_observed, observed)
                    and np.array_equal(router_cached, observed)
                    and np.array_equal(router_scored, observed)
                    and np.array_equal(router_forward_calls, steps + 1)
                    and np.all(router_forward_calls > 0)
                )
            else:
                router_pass = bool(
                    np.all(router_observed == 0)
                    and np.all(router_cached == 0)
                    and np.all(router_scored == 0)
                    and np.all(router_forward_calls == 0)
                )
            router_execution_pass = router_execution_pass and router_pass
            common = common and router_pass
            completed = (
                (stopped == 1)
                & (accepted == 1)
                & (transitions == 1)
                & (emitted_bytes >= PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES)
            )
            if mode == "controlled_replay":
                mode_pass = bool(
                    common
                    and np.all(
                        emitted_bytes == PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES
                    )
                    and np.all(overshoot == 0)
                    and np.all(completed)
                )
                controlled_pass = controlled_pass and mode_pass
            else:
                mode_pass = bool(
                    common
                    and np.array_equal(
                        overshoot,
                        emitted_bytes - PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES,
                    )
                    and np.all(completed)
                    and (
                        not exact_bytes
                        or np.all(
                            emitted_bytes
                            <= PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES + 3
                        )
                    )
                )
                free_pass = free_pass and mode_pass
                completion[role] = [
                    float(completed[index].mean())
                    for index in range(len(PUBLICATION_PRETRAIN_SEEDS))
                ]
                replacement[role] = [
                    float(replacement_free[index].mean())
                    for index in range(len(PUBLICATION_PRETRAIN_SEEDS))
                ]

    passed = bool(
        deterministic
        and router_execution_pass
        and controlled_pass
        and free_pass
    )
    provisional = PublicationValidOutputEvidence(
        lineage_identity_sha256=lineage.identity_sha256,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        minimum_output_bytes=PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES,
        maximum_generation_steps=ACTUAL_INFERENCE_FREE_RUNNING_MAXIMUM_STEPS,
        candidate_maximum_unit_bytes=candidate_maximum_unit_bytes,
        comparator_maximum_unit_bytes=comparator_maximum_unit_bytes,
        trial_artifact_sha256=trial_artifact_sha256,
        output_trace_artifact_sha256=output_trace_artifact_sha256,
        output_trace_audit_sha256=output_trace_audit_sha256,
        diagnostic_arrays_sha256=_canonical_sha256(manifest),
        deterministic_diagnostics_pass=deterministic,
        router_execution_pass=router_execution_pass,
        controlled_contract_pass=controlled_pass,
        free_running_contract_pass=free_pass,
        candidate_completion_rate_by_seed=tuple(completion["candidate"]),
        comparator_completion_rate_by_seed=tuple(completion["reference"]),
        candidate_replacement_free_rate_by_seed=tuple(replacement["candidate"]),
        comparator_replacement_free_rate_by_seed=tuple(replacement["reference"]),
        overall_pass=passed,
        identity_sha256="",
    )
    result = PublicationValidOutputEvidence(
        **{
            **provisional.to_dict(),
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_valid_output_evidence(result, lineage)
    return result


def validate_publication_valid_output_evidence(
    evidence: PublicationValidOutputEvidence,
    lineage: PublicationRuntimeLineage,
) -> None:
    rates = (
        evidence.candidate_completion_rate_by_seed,
        evidence.comparator_completion_rate_by_seed,
        evidence.candidate_replacement_free_rate_by_seed,
        evidence.comparator_replacement_free_rate_by_seed,
    )
    if (
        evidence.lineage_identity_sha256 != lineage.identity_sha256
        or evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or evidence.minimum_output_bytes != PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES
        or evidence.maximum_generation_steps
        != ACTUAL_INFERENCE_FREE_RUNNING_MAXIMUM_STEPS
        or evidence.candidate_maximum_unit_bytes != 1
        or evidence.comparator_maximum_unit_bytes <= 0
        or (
            lineage.comparator_family == "raw_byte"
            and evidence.comparator_maximum_unit_bytes != 1
        )
        or not all(
            _is_sha256(value)
            for value in (
                evidence.trial_artifact_sha256,
                evidence.output_trace_artifact_sha256,
                evidence.output_trace_audit_sha256,
            )
        )
        or any(len(values) != len(PUBLICATION_PRETRAIN_SEEDS) for values in rates)
        or any(
            not np.isfinite(value) or not 0 <= value <= 1
            for values in rates
            for value in values
        )
        or not _is_sha256(evidence.diagnostic_arrays_sha256)
        or not _is_sha256(evidence.identity_sha256)
        or evidence.overall_pass
        != bool(
            evidence.deterministic_diagnostics_pass
            and evidence.router_execution_pass
            and evidence.controlled_contract_pass
            and evidence.free_running_contract_pass
        )
        or evidence.identity_sha256
        != _canonical_sha256(_identity_payload(evidence))
    ):
        raise ValueError("publication valid-output evidence is inconsistent")


@dataclass(frozen=True, slots=True)
class PublicationRuntimeEvidence:
    candidate_key: str
    comparator_key: str
    comparator_family: str
    seed_order: tuple[int, ...]
    lineage: PublicationRuntimeLineage
    equivalence: PublicationRuntimeEquivalence
    timing: PublicationTimingEvidence
    valid_output: PublicationValidOutputEvidence
    overall_integrity_pass: bool
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_runtime_evidence(
    lineage: PublicationRuntimeLineage,
    equivalence: PublicationRuntimeEquivalence,
    timing: PublicationTimingEvidence,
    valid_output: PublicationValidOutputEvidence,
) -> PublicationRuntimeEvidence:
    validate_publication_runtime_lineage(lineage)
    validate_publication_runtime_equivalence(equivalence, lineage)
    validate_publication_timing_evidence(timing, lineage)
    validate_publication_valid_output_evidence(valid_output, lineage)
    if timing.trial_artifact_sha256 != valid_output.trial_artifact_sha256:
        raise ValueError("publication runtime trial artifacts are inconsistent")
    passed = bool(
        equivalence.overall_pass
        and timing.overall_pass
        and valid_output.overall_pass
    )
    provisional = PublicationRuntimeEvidence(
        candidate_key=lineage.candidate_key,
        comparator_key=lineage.comparator_key,
        comparator_family=lineage.comparator_family,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        lineage=lineage,
        equivalence=equivalence,
        timing=timing,
        valid_output=valid_output,
        overall_integrity_pass=passed,
        identity_sha256="",
    )
    result = PublicationRuntimeEvidence(
        **{
            **provisional.to_dict(),
            "lineage": lineage,
            "equivalence": equivalence,
            "timing": timing,
            "valid_output": valid_output,
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_runtime_evidence(result)
    return result


def validate_publication_runtime_evidence(evidence: PublicationRuntimeEvidence) -> None:
    validate_publication_runtime_lineage(evidence.lineage)
    validate_publication_runtime_equivalence(evidence.equivalence, evidence.lineage)
    validate_publication_timing_evidence(evidence.timing, evidence.lineage)
    validate_publication_valid_output_evidence(evidence.valid_output, evidence.lineage)
    expected_pass = bool(
        evidence.equivalence.overall_pass
        and evidence.timing.overall_pass
        and evidence.valid_output.overall_pass
    )
    if (
        evidence.candidate_key != evidence.lineage.candidate_key
        or evidence.comparator_key != evidence.lineage.comparator_key
        or evidence.comparator_family != evidence.lineage.comparator_family
        or evidence.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or evidence.timing.trial_artifact_sha256
        != evidence.valid_output.trial_artifact_sha256
        or evidence.overall_integrity_pass != expected_pass
        or not _is_sha256(evidence.identity_sha256)
        or evidence.identity_sha256
        != _canonical_sha256(_identity_payload(evidence))
    ):
        raise ValueError("publication runtime evidence is inconsistent")
