"""Content-free selection and router bundles for publication contract tests."""

from __future__ import annotations

import hashlib

from jamoflow.publication_protocol import PUBLICATION_PRETRAIN_SEEDS
from jamoflow.publication_reference import (
    PublicationEntropyRouterBundle,
    PublicationRawReferenceDescriptor,
    build_publication_entropy_router_bundle,
    build_publication_raw_reference_descriptor,
)


def content_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def make_reference_descriptor(
    policy: str = "entropy_threshold_full",
) -> PublicationRawReferenceDescriptor:
    if policy in {"causal_codepoint_grid_64", "causal_codepoint_grid_72"}:
        rate = int(policy.rsplit("_", 1)[-1])
        family = "compute_conversion"
        runtime_policy = "causal_codepoint_grid"
        patch_count = rate
    else:
        family = "phase3"
        runtime_policy = policy
        patch_count = 86
    requires_router = policy in {
        "entropy_threshold_full",
        "entropy_threshold_codepoint",
    }
    selection = {
        "schema_version": 1,
        "selection_uses_latency": False,
        "seed_order": list(PUBLICATION_PRETRAIN_SEEDS),
        "phase3_initial_summary": {"sha256": content_hash("phase3-summary")},
        "conversion_initial_summary": {
            "sha256": content_hash("conversion-summary")
        },
        "reference": {
            "policy": policy,
            "runtime_policy": runtime_policy,
            "model_family": family,
            "patch_count": patch_count,
            "requires_entropy_router": requires_router,
        },
        "reference_selection": {"selected_policy": policy},
    }
    return build_publication_raw_reference_descriptor(
        selection,
        selection_sha256=content_hash(f"selection:{policy}"),
    )


def make_router_bundles(
    descriptor: PublicationRawReferenceDescriptor,
    *,
    variant: str = "final",
) -> dict[int, PublicationEntropyRouterBundle]:
    return {
        seed: build_publication_entropy_router_bundle(
            seed=seed,
            descriptor=descriptor,
            router_checkpoint_artifact_sha256=content_hash(
                f"router-checkpoint-artifact:{variant}:{seed}"
            ),
            router_checkpoint_state_sha256=content_hash(
                f"router-checkpoint-state:{variant}:{seed}"
            ),
            router_report_artifact_sha256=content_hash(
                f"router-report:{variant}:{seed}"
            ),
            router_config_sha256=content_hash("router-config"),
            router_training_stream_sha256=content_hash(
                f"router-training-stream:{variant}"
            ),
            calibration_stream_sha256=content_hash("calibration-stream"),
            test_stream_sha256=content_hash("test-stream"),
            threshold_nats=1.0 + seed / 1_000_000,
            maximum_patch_length=24,
            threshold_cache_artifact_sha256=content_hash(
                f"threshold-cache:{variant}:{seed}"
            ),
            threshold_diagnostics_artifact_sha256=content_hash(
                f"threshold-diagnostics:{variant}:{seed}"
            ),
            train_patch_matrix_sha256=content_hash(
                f"train-patches:{variant}:{seed}"
            ),
            calibration_patch_matrix_sha256=content_hash(
                f"calibration-patches:{variant}:{seed}"
            ),
            test_patch_matrix_sha256=content_hash(
                f"test-patches:{variant}:{seed}"
            ),
        )
        for seed in PUBLICATION_PRETRAIN_SEEDS
    }
