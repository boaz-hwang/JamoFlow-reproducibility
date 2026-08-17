"""Cryptographic model locks shared by every publication evaluation path.

Model-family names are labels, not evidence.  This module binds each label to
the exact seed checkpoints, model configurations, tokenizer, and UTF-8
transition table used by an evaluation.  It also builds the final graph that
proves BPB, downstream, learning-curve, and runtime evidence refer to the same
four model families.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_BPE_VOCABULARY_CANDIDATES,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
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


PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION = 3
PUBLICATION_MODEL_KEYS = (
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
    *(
        PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size]
        for size in PUBLICATION_BPE_VOCABULARY_CANDIDATES
    ),
)
PUBLICATION_COMPARATOR_KEYS = PUBLICATION_MODEL_KEYS[1:]


def canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def named_array_manifest_sha256(
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Hash names, canonical dtypes, shapes, and C-order bytes."""

    if not arrays or any(not isinstance(name, str) or not name for name in arrays):
        raise ValueError("publication array manifest requires named arrays")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        values = np.asarray(arrays[name])
        if values.dtype.hasobject:
            raise ValueError("publication array manifest forbids object arrays")
        canonical_dtype = values.dtype.newbyteorder("<")
        values = np.ascontiguousarray(
            values.astype(canonical_dtype, copy=False)
        )
        encoded_name = name.encode("utf-8")
        encoded_dtype = values.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(8, "little"))
        digest.update(encoded_dtype)
        digest.update(values.ndim.to_bytes(8, "little"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.nbytes.to_bytes(8, "little"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _identity_payload(value: Any) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("identity_sha256")
    return payload


@dataclass(frozen=True, slots=True)
class PublicationModelSnapshot:
    model_key: str
    seed_order: tuple[int, ...]
    checkpoint_sha256: tuple[str, ...]
    model_config_sha256: tuple[str, ...]
    raw_reference_descriptor: PublicationRawReferenceDescriptor | None
    auxiliary_kind: str
    auxiliary_bundles: tuple[PublicationEntropyRouterBundle, ...]
    auxiliary_checkpoint_sha256: tuple[str, ...]
    auxiliary_config_sha256: tuple[str, ...]
    auxiliary_calibration_sha256: tuple[str, ...]
    tokenizer_sha256: str
    utf8_transition_sha256: str
    protocol_version: int
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_model_snapshot(
    *,
    model_key: str,
    checkpoint_sha256: Mapping[int, str],
    model_config_sha256: Mapping[int, str],
    raw_reference_descriptor: PublicationRawReferenceDescriptor | None = None,
    auxiliary_bundles: Mapping[int, PublicationEntropyRouterBundle] | None = None,
    tokenizer_sha256: str,
    utf8_transition_sha256: str,
) -> PublicationModelSnapshot:
    if (
        model_key not in PUBLICATION_MODEL_KEYS
        or set(checkpoint_sha256) != set(PUBLICATION_PRETRAIN_SEEDS)
        or set(model_config_sha256) != set(PUBLICATION_PRETRAIN_SEEDS)
    ):
        raise ValueError("publication model snapshot has an invalid model or seed set")
    if model_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY:
        if raw_reference_descriptor is None:
            raise ValueError("raw-reference snapshot requires its sealed descriptor")
        validate_publication_raw_reference_descriptor(raw_reference_descriptor)
        auxiliary_kind = raw_reference_descriptor.auxiliary_kind
    else:
        if raw_reference_descriptor is not None:
            raise ValueError("only the raw reference may bind a selection descriptor")
        auxiliary_kind = PUBLICATION_AUXILIARY_NONE
    checkpoints = tuple(
        checkpoint_sha256[seed] for seed in PUBLICATION_PRETRAIN_SEEDS
    )
    configurations = tuple(
        model_config_sha256[seed] for seed in PUBLICATION_PRETRAIN_SEEDS
    )
    if auxiliary_kind == PUBLICATION_AUXILIARY_NONE:
        if auxiliary_bundles is not None:
            raise ValueError("model without an auxiliary component cannot bind auxiliary hashes")
        ordered_bundles: tuple[PublicationEntropyRouterBundle, ...] = ()
        auxiliary_checkpoints: tuple[str, ...] = ()
        auxiliary_configurations: tuple[str, ...] = ()
        auxiliary_calibrations: tuple[str, ...] = ()
    else:
        if (
            auxiliary_bundles is None
            or set(auxiliary_bundles) != set(PUBLICATION_PRETRAIN_SEEDS)
            or raw_reference_descriptor is None
        ):
            raise ValueError("auxiliary model snapshot requires every paired seed")
        ordered_bundles = tuple(
            auxiliary_bundles[seed] for seed in PUBLICATION_PRETRAIN_SEEDS
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
                raise ValueError("auxiliary bundle seed order is inconsistent")
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
    hashes = (
        *checkpoints,
        *configurations,
        *auxiliary_checkpoints,
        *auxiliary_configurations,
        *auxiliary_calibrations,
        tokenizer_sha256,
        utf8_transition_sha256,
    )
    if (
        not all(is_sha256(value) for value in hashes)
        or len(set(checkpoints)) != len(checkpoints)
        or len(set(configurations)) != 1
        or (
            auxiliary_kind == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
            and (
                len(set(auxiliary_checkpoints)) != len(auxiliary_checkpoints)
                or len(set(auxiliary_configurations)) != 1
                or len(set(auxiliary_calibrations)) != len(auxiliary_calibrations)
                or not set(checkpoints).isdisjoint(auxiliary_checkpoints)
                or not set(auxiliary_checkpoints).isdisjoint(auxiliary_calibrations)
            )
        )
    ):
        raise ValueError("publication model snapshot hashes are invalid")
    provisional = PublicationModelSnapshot(
        model_key=model_key,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        checkpoint_sha256=checkpoints,
        model_config_sha256=configurations,
        raw_reference_descriptor=raw_reference_descriptor,
        auxiliary_kind=auxiliary_kind,
        auxiliary_bundles=ordered_bundles,
        auxiliary_checkpoint_sha256=auxiliary_checkpoints,
        auxiliary_config_sha256=auxiliary_configurations,
        auxiliary_calibration_sha256=auxiliary_calibrations,
        tokenizer_sha256=tokenizer_sha256,
        utf8_transition_sha256=utf8_transition_sha256,
        protocol_version=PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION,
        identity_sha256="",
    )
    snapshot = PublicationModelSnapshot(
        **{
            **provisional.to_dict(),
            "raw_reference_descriptor": raw_reference_descriptor,
            "auxiliary_bundles": ordered_bundles,
            "identity_sha256": canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_model_snapshot(snapshot)
    return snapshot


def validate_publication_model_snapshot(
    snapshot: PublicationModelSnapshot,
) -> None:
    if not isinstance(snapshot, PublicationModelSnapshot):
        raise ValueError("publication model snapshot is inconsistent")
    hashes = (
        *snapshot.checkpoint_sha256,
        *snapshot.model_config_sha256,
        *snapshot.auxiliary_checkpoint_sha256,
        *snapshot.auxiliary_config_sha256,
        *snapshot.auxiliary_calibration_sha256,
        snapshot.tokenizer_sha256,
        snapshot.utf8_transition_sha256,
        snapshot.identity_sha256,
    )
    descriptor = snapshot.raw_reference_descriptor
    if snapshot.model_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY:
        if descriptor is None:
            raise ValueError("publication model snapshot is inconsistent")
        validate_publication_raw_reference_descriptor(descriptor)
        expected_auxiliary_kind = descriptor.auxiliary_kind
    else:
        expected_auxiliary_kind = PUBLICATION_AUXILIARY_NONE
    if (
        snapshot.model_key not in PUBLICATION_MODEL_KEYS
        or snapshot.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or len(snapshot.checkpoint_sha256) != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(snapshot.model_config_sha256) != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(set(snapshot.checkpoint_sha256))
        != len(PUBLICATION_PRETRAIN_SEEDS)
        or len(set(snapshot.model_config_sha256)) != 1
        or snapshot.auxiliary_kind not in PUBLICATION_AUXILIARY_KINDS
        or snapshot.auxiliary_kind != expected_auxiliary_kind
        or (
            snapshot.model_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
            and descriptor is not None
        )
        or (
            snapshot.auxiliary_kind == PUBLICATION_AUXILIARY_NONE
            and (
                snapshot.auxiliary_bundles
                or snapshot.auxiliary_checkpoint_sha256
                or snapshot.auxiliary_config_sha256
                or snapshot.auxiliary_calibration_sha256
            )
        )
        or (
            snapshot.auxiliary_kind == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
            and (
                snapshot.model_key != PUBLICATION_RAW_COMPARATOR_MODEL_KEY
                or descriptor is None
                or len(snapshot.auxiliary_bundles)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(snapshot.auxiliary_checkpoint_sha256)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(snapshot.auxiliary_config_sha256)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(snapshot.auxiliary_calibration_sha256)
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(set(snapshot.auxiliary_checkpoint_sha256))
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or len(set(snapshot.auxiliary_config_sha256)) != 1
                or len(set(snapshot.auxiliary_calibration_sha256))
                != len(PUBLICATION_PRETRAIN_SEEDS)
                or not set(snapshot.checkpoint_sha256).isdisjoint(
                    snapshot.auxiliary_checkpoint_sha256
                )
                or not set(snapshot.auxiliary_checkpoint_sha256).isdisjoint(
                    snapshot.auxiliary_calibration_sha256
                )
            )
        )
        or not all(is_sha256(value) for value in hashes)
        or snapshot.protocol_version != PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION
        or snapshot.identity_sha256
        != canonical_sha256(_identity_payload(snapshot))
    ):
        raise ValueError("publication model snapshot is inconsistent")
    if (
        descriptor is not None
        and snapshot.auxiliary_kind == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
    ):
        validate_publication_entropy_router_bundle_family(
            snapshot.auxiliary_bundles,
            descriptor,
        )
        for seed, bundle in zip(
            PUBLICATION_PRETRAIN_SEEDS,
            snapshot.auxiliary_bundles,
            strict=True,
        ):
            validate_publication_entropy_router_bundle(bundle, descriptor)
            if (
                bundle.seed != seed
                or bundle.router_checkpoint_state_sha256
                != snapshot.auxiliary_checkpoint_sha256[
                    PUBLICATION_PRETRAIN_SEEDS.index(seed)
                ]
                or bundle.router_config_sha256
                != snapshot.auxiliary_config_sha256[
                    PUBLICATION_PRETRAIN_SEEDS.index(seed)
                ]
                or bundle.identity_sha256
                != snapshot.auxiliary_calibration_sha256[
                    PUBLICATION_PRETRAIN_SEEDS.index(seed)
                ]
            ):
                raise ValueError("publication model snapshot is inconsistent")


def publication_runtime_model_snapshots(
    lineage: object,
) -> tuple[PublicationModelSnapshot, PublicationModelSnapshot]:
    """Reconstruct canonical snapshots from a validated runtime lineage."""

    seed_order = tuple(getattr(lineage, "seed_order"))
    if seed_order != PUBLICATION_PRETRAIN_SEEDS:
        raise ValueError("publication runtime lineage has the wrong seed order")

    def mapping(attribute: str) -> dict[int, str]:
        values = tuple(getattr(lineage, attribute))
        if len(values) != len(seed_order):
            raise ValueError("publication runtime lineage snapshot is incomplete")
        return dict(zip(seed_order, values, strict=True))

    candidate = build_publication_model_snapshot(
        model_key=getattr(lineage, "candidate_key"),
        checkpoint_sha256=mapping("candidate_checkpoint_sha256"),
        model_config_sha256=mapping("candidate_model_config_sha256"),
        tokenizer_sha256=getattr(lineage, "candidate_tokenizer_sha256"),
        utf8_transition_sha256=getattr(
            lineage,
            "candidate_utf8_transition_sha256",
        ),
    )
    comparator = build_publication_model_snapshot(
        model_key=getattr(lineage, "comparator_key"),
        checkpoint_sha256=mapping("comparator_checkpoint_sha256"),
        model_config_sha256=mapping("comparator_model_config_sha256"),
        raw_reference_descriptor=getattr(
            lineage,
            "raw_reference_descriptor",
        ),
        auxiliary_bundles=(
            dict(
                zip(
                    seed_order,
                    tuple(getattr(lineage, "comparator_auxiliary_bundles")),
                    strict=True,
                )
            )
            if getattr(lineage, "comparator_auxiliary_kind")
            != PUBLICATION_AUXILIARY_NONE
            else None
        ),
        tokenizer_sha256=getattr(lineage, "comparator_tokenizer_sha256"),
        utf8_transition_sha256=getattr(
            lineage,
            "comparator_utf8_transition_sha256",
        ),
    )
    return candidate, comparator


@dataclass(frozen=True, slots=True)
class PublicationLearningCurveModelLock:
    model_key: str
    seed_order: tuple[int, ...]
    budget_bytes: tuple[int, ...]
    snapshots: tuple[PublicationModelSnapshot, ...]
    protocol_version: int
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def final_snapshot(self) -> PublicationModelSnapshot:
        return self.snapshots[-1]


def build_publication_learning_curve_model_lock(
    *,
    model_key: str,
    budget_bytes: tuple[int, ...],
    snapshots_by_budget: Mapping[int, PublicationModelSnapshot],
) -> PublicationLearningCurveModelLock:
    if (
        model_key not in PUBLICATION_MODEL_KEYS
        or not budget_bytes
        or tuple(sorted(budget_bytes)) != budget_bytes
        or len(set(budget_bytes)) != len(budget_bytes)
        or any(value <= 0 for value in budget_bytes)
        or set(snapshots_by_budget) != set(budget_bytes)
    ):
        raise ValueError("publication learning-curve model lock is malformed")
    snapshots = tuple(snapshots_by_budget[budget] for budget in budget_bytes)
    for snapshot in snapshots:
        validate_publication_model_snapshot(snapshot)
        if snapshot.model_key != model_key:
            raise ValueError("learning-curve snapshot has the wrong model key")
    first = snapshots[0]
    invariant_fields = (
        "model_config_sha256",
        "raw_reference_descriptor",
        "auxiliary_kind",
        "auxiliary_config_sha256",
        "tokenizer_sha256",
        "utf8_transition_sha256",
        "seed_order",
        "protocol_version",
    )
    if any(
        getattr(snapshot, field) != getattr(first, field)
        for snapshot in snapshots[1:]
        for field in invariant_fields
    ):
        raise ValueError("learning-curve architecture changed across budgets")
    checkpoints = tuple(
        checkpoint
        for snapshot in snapshots
        for checkpoint in snapshot.checkpoint_sha256
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("learning-curve checkpoint was reused across budget or seed")
    if first.auxiliary_kind != PUBLICATION_AUXILIARY_NONE:
        auxiliary_checkpoints = tuple(
            checkpoint
            for snapshot in snapshots
            for checkpoint in snapshot.auxiliary_checkpoint_sha256
        )
        if len(set(auxiliary_checkpoints)) != len(auxiliary_checkpoints):
            raise ValueError(
                "learning-curve auxiliary checkpoint was reused across budget or seed"
            )
        auxiliary_calibrations = tuple(
            calibration
            for snapshot in snapshots
            for calibration in snapshot.auxiliary_calibration_sha256
        )
        if len(set(auxiliary_calibrations)) != len(auxiliary_calibrations):
            raise ValueError(
                "learning-curve auxiliary calibration was reused across budget or seed"
            )
        bundle_fields = (
            "router_checkpoint_artifact_sha256",
            "router_checkpoint_state_sha256",
            "router_report_artifact_sha256",
            "threshold_cache_artifact_sha256",
            "threshold_diagnostics_artifact_sha256",
            "train_patch_matrix_sha256",
            "calibration_patch_matrix_sha256",
            "test_patch_matrix_sha256",
            "identity_sha256",
        )
        if any(
            len(
                {
                    getattr(bundle, field)
                    for snapshot in snapshots
                    for bundle in snapshot.auxiliary_bundles
                }
            )
            != len(snapshots) * len(PUBLICATION_PRETRAIN_SEEDS)
            for field in bundle_fields
        ):
            raise ValueError(
                "learning-curve auxiliary artifact was reused across budget or seed"
            )
    provisional = PublicationLearningCurveModelLock(
        model_key=model_key,
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        budget_bytes=budget_bytes,
        snapshots=snapshots,
        protocol_version=PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION,
        identity_sha256="",
    )
    lock = PublicationLearningCurveModelLock(
        **{
            **provisional.to_dict(),
            "snapshots": snapshots,
            "identity_sha256": canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_learning_curve_model_lock(lock)
    return lock


def validate_publication_learning_curve_model_lock(
    lock: PublicationLearningCurveModelLock,
) -> None:
    if not isinstance(lock, PublicationLearningCurveModelLock):
        raise ValueError("publication learning-curve model lock is invalid")
    if (
        lock.model_key not in PUBLICATION_MODEL_KEYS
        or not lock.budget_bytes
        or tuple(sorted(lock.budget_bytes)) != lock.budget_bytes
        or len(set(lock.budget_bytes)) != len(lock.budget_bytes)
        or any(value <= 0 for value in lock.budget_bytes)
        or len(lock.snapshots) != len(lock.budget_bytes)
    ):
        raise ValueError("publication learning-curve model lock is inconsistent")
    for snapshot in lock.snapshots:
        validate_publication_model_snapshot(snapshot)
        if snapshot.model_key != lock.model_key:
            raise ValueError("publication learning-curve model lock is inconsistent")
    first = lock.snapshots[0]
    invariant_fields = (
        "model_config_sha256",
        "raw_reference_descriptor",
        "auxiliary_kind",
        "auxiliary_config_sha256",
        "tokenizer_sha256",
        "utf8_transition_sha256",
        "seed_order",
        "protocol_version",
    )
    if any(
        getattr(snapshot, field) != getattr(first, field)
        for snapshot in lock.snapshots[1:]
        for field in invariant_fields
    ):
        raise ValueError("publication learning-curve model lock is inconsistent")
    checkpoints = tuple(
        checkpoint
        for snapshot in lock.snapshots
        for checkpoint in snapshot.checkpoint_sha256
    )
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("publication learning-curve model lock is inconsistent")
    if first.auxiliary_kind != PUBLICATION_AUXILIARY_NONE:
        auxiliary_checkpoints = tuple(
            checkpoint
            for snapshot in lock.snapshots
            for checkpoint in snapshot.auxiliary_checkpoint_sha256
        )
        auxiliary_calibrations = tuple(
            calibration
            for snapshot in lock.snapshots
            for calibration in snapshot.auxiliary_calibration_sha256
        )
        if (
            len(set(auxiliary_checkpoints)) != len(auxiliary_checkpoints)
            or len(set(auxiliary_calibrations)) != len(auxiliary_calibrations)
        ):
            raise ValueError("publication learning-curve model lock is inconsistent")
        bundle_fields = (
            "router_checkpoint_artifact_sha256",
            "router_checkpoint_state_sha256",
            "router_report_artifact_sha256",
            "threshold_cache_artifact_sha256",
            "threshold_diagnostics_artifact_sha256",
            "train_patch_matrix_sha256",
            "calibration_patch_matrix_sha256",
            "test_patch_matrix_sha256",
            "identity_sha256",
        )
        if any(
            len(
                {
                    getattr(bundle, field)
                    for snapshot in lock.snapshots
                    for bundle in snapshot.auxiliary_bundles
                }
            )
            != len(lock.snapshots) * len(PUBLICATION_PRETRAIN_SEEDS)
            for field in bundle_fields
        ):
            raise ValueError("publication learning-curve model lock is inconsistent")
    if (
        lock.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or lock.protocol_version != PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION
        or not is_sha256(lock.identity_sha256)
        or lock.identity_sha256 != canonical_sha256(_identity_payload(lock))
    ):
        raise ValueError("publication learning-curve model lock is inconsistent")


@dataclass(frozen=True, slots=True)
class PublicationModelLockGraph:
    seed_order: tuple[int, ...]
    model_keys: tuple[str, ...]
    comparator_keys: tuple[str, ...]
    model_snapshots: tuple[PublicationModelSnapshot, ...]
    runtime_evidence_sha256: tuple[str, ...]
    bpb_evidence_sha256: tuple[str, ...]
    downstream_evidence_sha256: str
    learning_curve_evidence_sha256: str
    protocol_version: int
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_model_lock_graph(
    *,
    runtime_pairs: Mapping[
        str,
        tuple[PublicationModelSnapshot, PublicationModelSnapshot],
    ],
    bpb_pairs: Mapping[
        str,
        tuple[PublicationModelSnapshot, PublicationModelSnapshot],
    ],
    downstream_snapshots: Mapping[str, PublicationModelSnapshot],
    learning_curve_final_snapshots: Mapping[str, PublicationModelSnapshot],
    runtime_evidence_sha256: Mapping[str, str],
    bpb_evidence_sha256: Mapping[str, str],
    downstream_evidence_sha256: str,
    learning_curve_evidence_sha256: str,
) -> PublicationModelLockGraph:
    comparator_keys = PUBLICATION_COMPARATOR_KEYS
    if (
        set(runtime_pairs) != set(comparator_keys)
        or set(bpb_pairs) != set(comparator_keys)
        or set(runtime_evidence_sha256) != set(comparator_keys)
        or set(bpb_evidence_sha256) != set(comparator_keys)
        or set(learning_curve_final_snapshots) != set(PUBLICATION_MODEL_KEYS)
        or not downstream_snapshots
        or PUBLICATION_CANDIDATE_MODEL_KEY not in downstream_snapshots
        or not set(downstream_snapshots).issubset(PUBLICATION_MODEL_KEYS)
    ):
        raise ValueError("publication model graph has incomplete evaluation roles")
    for mapping in (
        runtime_evidence_sha256,
        bpb_evidence_sha256,
    ):
        if not all(is_sha256(value) for value in mapping.values()):
            raise ValueError("publication model graph evidence hashes are invalid")
    if not all(
        is_sha256(value)
        for value in (
            downstream_evidence_sha256,
            learning_curve_evidence_sha256,
        )
    ):
        raise ValueError("publication model graph evidence hashes are invalid")

    final = dict(learning_curve_final_snapshots)
    for key, snapshot in final.items():
        validate_publication_model_snapshot(snapshot)
        if snapshot.model_key != key:
            raise ValueError("publication model graph snapshot key is invalid")
    for comparator_key in comparator_keys:
        runtime_candidate, runtime_comparator = runtime_pairs[comparator_key]
        bpb_candidate, bpb_comparator = bpb_pairs[comparator_key]
        for snapshot in (
            runtime_candidate,
            runtime_comparator,
            bpb_candidate,
            bpb_comparator,
        ):
            validate_publication_model_snapshot(snapshot)
        if (
            runtime_candidate != final[PUBLICATION_CANDIDATE_MODEL_KEY]
            or bpb_candidate != final[PUBLICATION_CANDIDATE_MODEL_KEY]
            or runtime_comparator != final[comparator_key]
            or bpb_comparator != final[comparator_key]
        ):
            raise ValueError("publication model graph contains checkpoint drift")
    for key, snapshot in downstream_snapshots.items():
        validate_publication_model_snapshot(snapshot)
        if snapshot != final[key]:
            raise ValueError("publication downstream model differs from final model")

    snapshots = tuple(final[key] for key in PUBLICATION_MODEL_KEYS)
    provisional = PublicationModelLockGraph(
        seed_order=PUBLICATION_PRETRAIN_SEEDS,
        model_keys=PUBLICATION_MODEL_KEYS,
        comparator_keys=PUBLICATION_COMPARATOR_KEYS,
        model_snapshots=snapshots,
        runtime_evidence_sha256=tuple(
            runtime_evidence_sha256[key] for key in comparator_keys
        ),
        bpb_evidence_sha256=tuple(
            bpb_evidence_sha256[key] for key in comparator_keys
        ),
        downstream_evidence_sha256=downstream_evidence_sha256,
        learning_curve_evidence_sha256=learning_curve_evidence_sha256,
        protocol_version=PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION,
        identity_sha256="",
    )
    graph = PublicationModelLockGraph(
        **{
            **provisional.to_dict(),
            "model_snapshots": snapshots,
            "identity_sha256": canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_model_lock_graph(graph)
    return graph


def validate_publication_model_lock_graph(
    graph: PublicationModelLockGraph,
) -> None:
    if not isinstance(graph, PublicationModelLockGraph):
        raise ValueError("publication model lock graph is invalid")
    for snapshot in graph.model_snapshots:
        validate_publication_model_snapshot(snapshot)
    if (
        graph.seed_order != PUBLICATION_PRETRAIN_SEEDS
        or graph.model_keys != PUBLICATION_MODEL_KEYS
        or graph.comparator_keys != PUBLICATION_COMPARATOR_KEYS
        or len(graph.model_snapshots) != len(PUBLICATION_MODEL_KEYS)
        or tuple(snapshot.model_key for snapshot in graph.model_snapshots)
        != PUBLICATION_MODEL_KEYS
        or len(graph.runtime_evidence_sha256) != len(PUBLICATION_COMPARATOR_KEYS)
        or len(graph.bpb_evidence_sha256) != len(PUBLICATION_COMPARATOR_KEYS)
        or len(set(graph.runtime_evidence_sha256))
        != len(PUBLICATION_COMPARATOR_KEYS)
        or len(set(graph.bpb_evidence_sha256))
        != len(PUBLICATION_COMPARATOR_KEYS)
        or not all(
            is_sha256(value)
            for value in (
                *graph.runtime_evidence_sha256,
                *graph.bpb_evidence_sha256,
                graph.downstream_evidence_sha256,
                graph.learning_curve_evidence_sha256,
                graph.identity_sha256,
            )
        )
        or graph.protocol_version != PUBLICATION_MODEL_LOCK_PROTOCOL_VERSION
        or graph.identity_sha256 != canonical_sha256(_identity_payload(graph))
    ):
        raise ValueError("publication model lock graph is inconsistent")
