"""Sealed raw-reference and entropy-router identities for publication runs.

The compact experiment chooses one raw-byte reference before publication-scale
training.  That choice is authoritative: publication workers may not relabel an
entropy policy as structural in order to omit its router.  This module turns the
selection artifact into a validated descriptor and, when needed, binds every
seed's router checkpoint and calibration/cache lineage into a structured bundle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from .compute_conversion import CONVERSION_RATES, conversion_policy
from .phase3 import PHASE3_POLICIES, THRESHOLD_POLICIES
from .publication_protocol import PUBLICATION_PRETRAIN_SEEDS


PUBLICATION_REFERENCE_PROTOCOL_VERSION = 1
PUBLICATION_ROUTER_BUNDLE_PROTOCOL_VERSION = 1
PUBLICATION_AUXILIARY_NONE = "none"
PUBLICATION_AUXILIARY_ENTROPY_ROUTER = "entropy_router"
PUBLICATION_AUXILIARY_KINDS = (
    PUBLICATION_AUXILIARY_NONE,
    PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
)
PUBLICATION_CONVERSION_REFERENCE_POLICIES = tuple(
    conversion_policy("codepoint", rate) for rate in CONVERSION_RATES
)
PUBLICATION_RAW_REFERENCE_POLICIES = (
    *PHASE3_POLICIES,
    *PUBLICATION_CONVERSION_REFERENCE_POLICIES,
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


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _identity_payload(value: Any) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("identity_sha256")
    return payload


def _reference_expectation(policy: str) -> tuple[str, str, int, bool]:
    if policy in PHASE3_POLICIES:
        return "phase3", policy, 86, policy in THRESHOLD_POLICIES
    if policy in PUBLICATION_CONVERSION_REFERENCE_POLICIES:
        rate = int(policy.rsplit("_", 1)[-1])
        return "compute_conversion", "causal_codepoint_grid", rate, False
    raise ValueError("publication raw-reference policy is not preregistered")


def publication_auxiliary_kind_for_policy(policy: str) -> str:
    _, _, _, requires_router = _reference_expectation(policy)
    return (
        PUBLICATION_AUXILIARY_ENTROPY_ROUTER
        if requires_router
        else PUBLICATION_AUXILIARY_NONE
    )


@dataclass(frozen=True, slots=True)
class PublicationRawReferenceDescriptor:
    policy: str
    runtime_policy: str
    model_family: str
    patch_count: int
    requires_entropy_router: bool
    auxiliary_kind: str
    selection_sha256: str
    phase3_initial_summary_sha256: str
    conversion_initial_summary_sha256: str
    protocol_version: int
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_raw_reference_descriptor(
    selection: Mapping[str, Any],
    *,
    selection_sha256: str,
) -> PublicationRawReferenceDescriptor:
    """Validate the compact selection and derive, never accept, aux identity."""

    reference = selection.get("reference")
    reference_selection = selection.get("reference_selection")
    phase3_summary = selection.get("phase3_initial_summary")
    conversion_summary = selection.get("conversion_initial_summary")
    if (
        selection.get("schema_version") != 1
        or selection.get("selection_uses_latency") is not False
        or tuple(selection.get("seed_order", ())) != PUBLICATION_PRETRAIN_SEEDS
        or not isinstance(reference, Mapping)
        or not isinstance(reference_selection, Mapping)
        or not isinstance(phase3_summary, Mapping)
        or not isinstance(conversion_summary, Mapping)
        or not is_sha256(selection_sha256)
    ):
        raise ValueError("publication raw-reference selection is malformed")
    policy = reference.get("policy")
    if not isinstance(policy, str):
        raise ValueError("publication raw-reference selection has no policy")
    model_family, runtime_policy, patch_count, requires_router = (
        _reference_expectation(policy)
    )
    if (
        reference.get("runtime_policy") != runtime_policy
        or reference.get("model_family") != model_family
        or reference.get("patch_count") != patch_count
        or reference.get("requires_entropy_router") is not requires_router
        or reference_selection.get("selected_policy") != policy
    ):
        raise ValueError("publication raw-reference descriptor contradicts selection")
    phase3_sha256 = phase3_summary.get("sha256")
    conversion_sha256 = conversion_summary.get("sha256")
    if not is_sha256(phase3_sha256) or not is_sha256(conversion_sha256):
        raise ValueError("publication raw-reference source summaries are unsealed")
    auxiliary_kind = (
        PUBLICATION_AUXILIARY_ENTROPY_ROUTER
        if requires_router
        else PUBLICATION_AUXILIARY_NONE
    )
    provisional = PublicationRawReferenceDescriptor(
        policy=policy,
        runtime_policy=runtime_policy,
        model_family=model_family,
        patch_count=patch_count,
        requires_entropy_router=requires_router,
        auxiliary_kind=auxiliary_kind,
        selection_sha256=selection_sha256,
        phase3_initial_summary_sha256=str(phase3_sha256),
        conversion_initial_summary_sha256=str(conversion_sha256),
        protocol_version=PUBLICATION_REFERENCE_PROTOCOL_VERSION,
        identity_sha256="",
    )
    descriptor = PublicationRawReferenceDescriptor(
        **{
            **provisional.to_dict(),
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_raw_reference_descriptor(descriptor)
    return descriptor


def validate_publication_raw_reference_descriptor(
    descriptor: PublicationRawReferenceDescriptor,
) -> None:
    if not isinstance(descriptor, PublicationRawReferenceDescriptor):
        raise ValueError("publication raw-reference descriptor is inconsistent")
    try:
        model_family, runtime_policy, patch_count, requires_router = (
            _reference_expectation(descriptor.policy)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "publication raw-reference descriptor is inconsistent"
        ) from exc
    expected_auxiliary = (
        PUBLICATION_AUXILIARY_ENTROPY_ROUTER
        if requires_router
        else PUBLICATION_AUXILIARY_NONE
    )
    if (
        not isinstance(descriptor.policy, str)
        or not isinstance(descriptor.runtime_policy, str)
        or not isinstance(descriptor.model_family, str)
        or not isinstance(descriptor.patch_count, int)
        or isinstance(descriptor.patch_count, bool)
        or not isinstance(descriptor.requires_entropy_router, bool)
        or not isinstance(descriptor.auxiliary_kind, str)
        or descriptor.runtime_policy != runtime_policy
        or descriptor.model_family != model_family
        or descriptor.patch_count != patch_count
        or descriptor.requires_entropy_router is not requires_router
        or descriptor.auxiliary_kind != expected_auxiliary
        or not is_sha256(descriptor.selection_sha256)
        or not is_sha256(descriptor.phase3_initial_summary_sha256)
        or not is_sha256(descriptor.conversion_initial_summary_sha256)
        or descriptor.protocol_version != PUBLICATION_REFERENCE_PROTOCOL_VERSION
        or not is_sha256(descriptor.identity_sha256)
        or descriptor.identity_sha256
        != _canonical_sha256(_identity_payload(descriptor))
    ):
        raise ValueError("publication raw-reference descriptor is inconsistent")


def entropy_policy_definition_sha256(policy: str) -> str:
    if policy not in THRESHOLD_POLICIES:
        raise ValueError("router bundle requires an entropy policy")
    return _canonical_sha256(
        {
            "policy": policy,
            "causal_next_byte_entropy": True,
            "candidate_positions": (
                "all_byte_positions"
                if policy == "entropy_threshold_full"
                else "utf8_codepoint_boundaries"
            ),
            "threshold_selected_on": "calibration_only",
            "maximum_patch_length_required": True,
        }
    )


@dataclass(frozen=True, slots=True)
class PublicationEntropyRouterBundle:
    seed: int
    reference_descriptor_identity_sha256: str
    policy: str
    runtime_policy: str
    router_checkpoint_artifact_sha256: str
    router_checkpoint_state_sha256: str
    router_report_artifact_sha256: str
    router_config_sha256: str
    router_training_stream_sha256: str
    calibration_stream_sha256: str
    test_stream_sha256: str
    threshold_nats: float
    maximum_patch_length: int
    policy_definition_sha256: str
    threshold_cache_artifact_sha256: str
    threshold_diagnostics_artifact_sha256: str
    train_patch_matrix_sha256: str
    calibration_patch_matrix_sha256: str
    test_patch_matrix_sha256: str
    protocol_version: int
    identity_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_publication_entropy_router_bundle(
    *,
    seed: int,
    descriptor: PublicationRawReferenceDescriptor,
    router_checkpoint_artifact_sha256: str,
    router_checkpoint_state_sha256: str,
    router_report_artifact_sha256: str,
    router_config_sha256: str,
    router_training_stream_sha256: str,
    calibration_stream_sha256: str,
    test_stream_sha256: str,
    threshold_nats: float,
    maximum_patch_length: int,
    threshold_cache_artifact_sha256: str,
    threshold_diagnostics_artifact_sha256: str,
    train_patch_matrix_sha256: str,
    calibration_patch_matrix_sha256: str,
    test_patch_matrix_sha256: str,
) -> PublicationEntropyRouterBundle:
    validate_publication_raw_reference_descriptor(descriptor)
    if (
        descriptor.auxiliary_kind != PUBLICATION_AUXILIARY_ENTROPY_ROUTER
        or seed not in PUBLICATION_PRETRAIN_SEEDS
        or not isinstance(threshold_nats, (int, float))
        or isinstance(threshold_nats, bool)
        or not math.isfinite(float(threshold_nats))
        or not isinstance(maximum_patch_length, int)
        or isinstance(maximum_patch_length, bool)
        or maximum_patch_length <= 0
    ):
        raise ValueError("publication entropy-router bundle is malformed")
    hashes = (
        router_checkpoint_artifact_sha256,
        router_checkpoint_state_sha256,
        router_report_artifact_sha256,
        router_config_sha256,
        router_training_stream_sha256,
        calibration_stream_sha256,
        test_stream_sha256,
        threshold_cache_artifact_sha256,
        threshold_diagnostics_artifact_sha256,
        train_patch_matrix_sha256,
        calibration_patch_matrix_sha256,
        test_patch_matrix_sha256,
    )
    if not all(is_sha256(value) for value in hashes):
        raise ValueError("publication entropy-router bundle hashes are malformed")
    provisional = PublicationEntropyRouterBundle(
        seed=seed,
        reference_descriptor_identity_sha256=descriptor.identity_sha256,
        policy=descriptor.policy,
        runtime_policy=descriptor.runtime_policy,
        router_checkpoint_artifact_sha256=router_checkpoint_artifact_sha256,
        router_checkpoint_state_sha256=router_checkpoint_state_sha256,
        router_report_artifact_sha256=router_report_artifact_sha256,
        router_config_sha256=router_config_sha256,
        router_training_stream_sha256=router_training_stream_sha256,
        calibration_stream_sha256=calibration_stream_sha256,
        test_stream_sha256=test_stream_sha256,
        threshold_nats=float(threshold_nats),
        maximum_patch_length=maximum_patch_length,
        policy_definition_sha256=entropy_policy_definition_sha256(
            descriptor.policy
        ),
        threshold_cache_artifact_sha256=threshold_cache_artifact_sha256,
        threshold_diagnostics_artifact_sha256=(
            threshold_diagnostics_artifact_sha256
        ),
        train_patch_matrix_sha256=train_patch_matrix_sha256,
        calibration_patch_matrix_sha256=calibration_patch_matrix_sha256,
        test_patch_matrix_sha256=test_patch_matrix_sha256,
        protocol_version=PUBLICATION_ROUTER_BUNDLE_PROTOCOL_VERSION,
        identity_sha256="",
    )
    bundle = PublicationEntropyRouterBundle(
        **{
            **provisional.to_dict(),
            "identity_sha256": _canonical_sha256(_identity_payload(provisional)),
        }
    )
    validate_publication_entropy_router_bundle(bundle, descriptor)
    return bundle


def validate_publication_entropy_router_bundle(
    bundle: PublicationEntropyRouterBundle,
    descriptor: PublicationRawReferenceDescriptor,
) -> None:
    validate_publication_raw_reference_descriptor(descriptor)
    if not isinstance(bundle, PublicationEntropyRouterBundle):
        raise ValueError("publication entropy-router bundle is inconsistent")
    hashes = (
        bundle.reference_descriptor_identity_sha256,
        bundle.router_checkpoint_artifact_sha256,
        bundle.router_checkpoint_state_sha256,
        bundle.router_report_artifact_sha256,
        bundle.router_config_sha256,
        bundle.router_training_stream_sha256,
        bundle.calibration_stream_sha256,
        bundle.test_stream_sha256,
        bundle.policy_definition_sha256,
        bundle.threshold_cache_artifact_sha256,
        bundle.threshold_diagnostics_artifact_sha256,
        bundle.train_patch_matrix_sha256,
        bundle.calibration_patch_matrix_sha256,
        bundle.test_patch_matrix_sha256,
        bundle.identity_sha256,
    )
    if (
        descriptor.auxiliary_kind != PUBLICATION_AUXILIARY_ENTROPY_ROUTER
        or bundle.seed not in PUBLICATION_PRETRAIN_SEEDS
        or bundle.reference_descriptor_identity_sha256
        != descriptor.identity_sha256
        or bundle.policy != descriptor.policy
        or bundle.runtime_policy != descriptor.runtime_policy
        or not isinstance(bundle.threshold_nats, (int, float))
        or isinstance(bundle.threshold_nats, bool)
        or not math.isfinite(float(bundle.threshold_nats))
        or not isinstance(bundle.maximum_patch_length, int)
        or isinstance(bundle.maximum_patch_length, bool)
        or bundle.maximum_patch_length <= 0
        or bundle.policy_definition_sha256
        != entropy_policy_definition_sha256(bundle.policy)
        or not all(is_sha256(value) for value in hashes)
        or len(set(hashes[1:-1])) != len(hashes[1:-1])
        or bundle.protocol_version != PUBLICATION_ROUTER_BUNDLE_PROTOCOL_VERSION
        or bundle.identity_sha256
        != _canonical_sha256(_identity_payload(bundle))
    ):
        raise ValueError("publication entropy-router bundle is inconsistent")


def validate_publication_entropy_router_bundle_family(
    bundles: tuple[PublicationEntropyRouterBundle, ...],
    descriptor: PublicationRawReferenceDescriptor,
) -> None:
    validate_publication_raw_reference_descriptor(descriptor)
    if len(bundles) != len(PUBLICATION_PRETRAIN_SEEDS):
        raise ValueError("publication entropy-router bundle family is inconsistent")
    for seed, bundle in zip(PUBLICATION_PRETRAIN_SEEDS, bundles, strict=True):
        validate_publication_entropy_router_bundle(bundle, descriptor)
        if bundle.seed != seed:
            raise ValueError("publication entropy-router bundle family is inconsistent")
    common_fields = (
        "reference_descriptor_identity_sha256",
        "policy",
        "runtime_policy",
        "router_config_sha256",
        "router_training_stream_sha256",
        "calibration_stream_sha256",
        "test_stream_sha256",
        "maximum_patch_length",
        "policy_definition_sha256",
        "protocol_version",
    )
    unique_fields = (
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
    if not (
        all(
            len({getattr(bundle, field) for bundle in bundles}) == 1
            for field in common_fields
        )
        and all(
            len({getattr(bundle, field) for bundle in bundles}) == len(bundles)
            for field in unique_fields
        )
    ):
        raise ValueError("publication entropy-router bundle family is inconsistent")
