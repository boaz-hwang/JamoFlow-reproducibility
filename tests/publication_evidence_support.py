"""Synthetic model locks shared by publication evidence unit tests."""

from __future__ import annotations

from typing import Iterable

from jamoflow.publication_model_lock import (
    PublicationLearningCurveModelLock,
    PublicationModelSnapshot,
    build_publication_learning_curve_model_lock,
    build_publication_model_snapshot,
    publication_runtime_model_snapshots,
)
from jamoflow.publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)
from tests.publication_runtime_support import content_hash, make_lineage
from tests.publication_reference_support import make_router_bundles


def final_model_snapshots(
    *,
    candidate_variant: str = "shared",
) -> dict[str, PublicationModelSnapshot]:
    pairs = {}
    for family, comparator_key in (
        ("raw_byte", PUBLICATION_RAW_COMPARATOR_MODEL_KEY),
        ("standard_bpe", PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]),
        ("standard_bpe", PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]),
    ):
        pairs[comparator_key] = publication_runtime_model_snapshots(
            make_lineage(
                family,
                comparator_key=comparator_key,
                candidate_variant=candidate_variant,
            )
        )
    candidate = next(iter(pairs.values()))[0]
    if any(pair[0] != candidate for pair in pairs.values()):
        raise AssertionError("synthetic candidate snapshot drifted across controls")
    return {
        PUBLICATION_CANDIDATE_MODEL_KEY: candidate,
        **{key: pair[1] for key, pair in pairs.items()},
    }


def downstream_evidence_kwargs(
    reference_keys: Iterable[str] = (
        PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
    ),
    *,
    candidate_variant: str = "shared",
) -> dict[str, object]:
    snapshots = final_model_snapshots(candidate_variant=candidate_variant)
    keys = {PUBLICATION_CANDIDATE_MODEL_KEY, *reference_keys}
    return {
        "model_snapshots": {key: snapshots[key] for key in keys},
        "case_manifest_sha256": content_hash("downstream-cases"),
        "prediction_artifact_sha256": content_hash(
            f"downstream-predictions:{candidate_variant}"
        ),
    }


def learning_curve_model_locks(
    budget_bytes: tuple[int, ...],
    *,
    candidate_variant: str = "shared",
) -> dict[str, PublicationLearningCurveModelLock]:
    finals = final_model_snapshots(candidate_variant=candidate_variant)
    output = {}
    for key, final in finals.items():
        snapshots_by_budget = {}
        for budget in budget_bytes[:-1]:
            snapshots_by_budget[budget] = build_publication_model_snapshot(
                model_key=key,
                checkpoint_sha256={
                    seed: content_hash(f"curve:{key}:{budget}:{seed}")
                    for seed in PUBLICATION_PRETRAIN_SEEDS
                },
                model_config_sha256=dict(
                    zip(
                        PUBLICATION_PRETRAIN_SEEDS,
                        final.model_config_sha256,
                        strict=True,
                    )
                ),
                raw_reference_descriptor=final.raw_reference_descriptor,
                auxiliary_bundles=(
                    make_router_bundles(
                        final.raw_reference_descriptor,
                        variant=f"curve-{key}-{budget}",
                    )
                    if final.auxiliary_bundles
                    and final.raw_reference_descriptor is not None
                    else None
                ),
                tokenizer_sha256=final.tokenizer_sha256,
                utf8_transition_sha256=final.utf8_transition_sha256,
            )
        snapshots_by_budget[budget_bytes[-1]] = final
        output[key] = build_publication_learning_curve_model_lock(
            model_key=key,
            budget_bytes=budget_bytes,
            snapshots_by_budget=snapshots_by_budget,
        )
    return output


def data_adequacy_evidence_kwargs(
    budget_bytes: tuple[int, ...],
    *,
    candidate_variant: str = "shared",
) -> dict[str, object]:
    return {
        "learning_curve_model_locks": learning_curve_model_locks(
            budget_bytes,
            candidate_variant=candidate_variant,
        ),
        "curve_artifact_sha256": content_hash(
            f"learning-curves:{candidate_variant}:{budget_bytes}"
        ),
    }
