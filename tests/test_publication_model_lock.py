import unittest
from dataclasses import replace

import numpy as np

from jamoflow.publication_model_lock import (
    PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
    PUBLICATION_COMPARATOR_KEYS,
    PUBLICATION_MODEL_KEYS,
    build_publication_learning_curve_model_lock,
    build_publication_model_lock_graph,
    build_publication_model_snapshot,
    named_array_manifest_sha256,
    publication_runtime_model_snapshots,
    validate_publication_learning_curve_model_lock,
    validate_publication_model_lock_graph,
    validate_publication_model_snapshot,
)
from jamoflow.publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)
from jamoflow.publication_reference import (
    build_publication_entropy_router_bundle,
)
from tests.publication_runtime_support import content_hash, make_lineage
from tests.publication_reference_support import (
    make_reference_descriptor,
    make_router_bundles,
)


class PublicationModelLockTests(unittest.TestCase):
    def _runtime_pairs(self):
        pairs = {}
        for family, comparator_key in (
            ("raw_byte", PUBLICATION_RAW_COMPARATOR_MODEL_KEY),
            ("standard_bpe", PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]),
            ("standard_bpe", PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]),
        ):
            pairs[comparator_key] = publication_runtime_model_snapshots(
                make_lineage(family, comparator_key=comparator_key)
            )
        return pairs

    def _final_snapshots(self):
        pairs = self._runtime_pairs()
        snapshots = {
            PUBLICATION_CANDIDATE_MODEL_KEY: next(iter(pairs.values()))[0],
        }
        snapshots.update(
            {key: pair[1] for key, pair in pairs.items()}
        )
        return snapshots

    def _curve_lock(self, final, budgets=(64, 128, 256)):
        snapshots = {}
        for budget in budgets[:-1]:
            snapshots[budget] = build_publication_model_snapshot(
                model_key=final.model_key,
                checkpoint_sha256={
                    seed: content_hash(
                        f"curve:{final.model_key}:{budget}:{seed}"
                    )
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
                        variant=f"curve-{budget}",
                    )
                    if final.auxiliary_bundles
                    and final.raw_reference_descriptor is not None
                    else None
                ),
                tokenizer_sha256=final.tokenizer_sha256,
                utf8_transition_sha256=final.utf8_transition_sha256,
            )
        snapshots[budgets[-1]] = final
        return build_publication_learning_curve_model_lock(
            model_key=final.model_key,
            budget_bytes=budgets,
            snapshots_by_budget=snapshots,
        )

    def test_runtime_lineage_reconstructs_exact_model_snapshots(self) -> None:
        candidate, comparator = publication_runtime_model_snapshots(make_lineage())
        validate_publication_model_snapshot(candidate)
        validate_publication_model_snapshot(comparator)
        self.assertEqual(candidate.model_key, PUBLICATION_CANDIDATE_MODEL_KEY)
        self.assertEqual(comparator.model_key, PUBLICATION_RAW_COMPARATOR_MODEL_KEY)
        self.assertEqual(len(set(candidate.model_config_sha256)), 1)
        self.assertEqual(
            comparator.auxiliary_kind,
            PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
        )
        self.assertEqual(
            len(comparator.auxiliary_checkpoint_sha256),
            len(PUBLICATION_PRETRAIN_SEEDS),
        )

    def test_entropy_router_is_required_to_be_seed_distinct_and_raw_only(self) -> None:
        checkpoints = {
            seed: content_hash(f"main:{seed}")
            for seed in PUBLICATION_PRETRAIN_SEEDS
        }
        configurations = {
            seed: content_hash("main-config")
            for seed in PUBLICATION_PRETRAIN_SEEDS
        }
        descriptor = make_reference_descriptor("entropy_threshold_full")
        bundles = make_router_bundles(descriptor)
        cloned_router = {
            seed: bundles[PUBLICATION_PRETRAIN_SEEDS[0]]
            for seed in PUBLICATION_PRETRAIN_SEEDS
        }
        with self.assertRaisesRegex(ValueError, "seed order"):
            build_publication_model_snapshot(
                model_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
                checkpoint_sha256=checkpoints,
                model_config_sha256=configurations,
                raw_reference_descriptor=descriptor,
                auxiliary_bundles=cloned_router,
                tokenizer_sha256="b" * 64,
                utf8_transition_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "only the raw reference"):
            build_publication_model_snapshot(
                model_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                checkpoint_sha256=checkpoints,
                model_config_sha256=configurations,
                raw_reference_descriptor=descriptor,
                auxiliary_bundles=bundles,
                tokenizer_sha256="b" * 64,
                utf8_transition_sha256="c" * 64,
            )

        structural = make_reference_descriptor("fixed_byte_6")
        with self.assertRaisesRegex(ValueError, "without an auxiliary"):
            build_publication_model_snapshot(
                model_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
                checkpoint_sha256=checkpoints,
                model_config_sha256=configurations,
                raw_reference_descriptor=structural,
                auxiliary_bundles=bundles,
                tokenizer_sha256="b" * 64,
                utf8_transition_sha256="c" * 64,
            )

    def test_array_manifest_binds_name_dtype_shape_and_content(self) -> None:
        base = {
            "bytes": np.asarray([2], dtype=np.int64),
            "loss": np.asarray([[1, 2]], dtype=np.int64),
        }
        digest = named_array_manifest_sha256(base)
        self.assertEqual(
            digest,
            named_array_manifest_sha256(dict(reversed(tuple(base.items())))),
        )
        variants = (
            {"bytes": base["bytes"], "other": base["loss"]},
            {"bytes": base["bytes"], "loss": base["loss"].astype(np.float64)},
            {"bytes": base["bytes"], "loss": base["loss"].reshape(2, 1)},
            {
                "bytes": base["bytes"],
                "loss": np.asarray([[1, 3]], dtype=np.int64),
            },
        )
        self.assertTrue(
            all(named_array_manifest_sha256(values) != digest for values in variants)
        )

    def test_snapshot_rejects_cloned_seed_or_config_drift(self) -> None:
        checkpoints = {
            seed: content_hash(f"checkpoint:{seed}")
            for seed in PUBLICATION_PRETRAIN_SEEDS
        }
        configurations = {
            seed: content_hash("config") for seed in PUBLICATION_PRETRAIN_SEEDS
        }
        with self.assertRaisesRegex(ValueError, "hashes"):
            build_publication_model_snapshot(
                model_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                checkpoint_sha256={seed: "a" * 64 for seed in checkpoints},
                model_config_sha256=configurations,
                tokenizer_sha256="b" * 64,
                utf8_transition_sha256="c" * 64,
            )
        configurations[PUBLICATION_PRETRAIN_SEEDS[-1]] = content_hash("other")
        with self.assertRaisesRegex(ValueError, "hashes"):
            build_publication_model_snapshot(
                model_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                checkpoint_sha256=checkpoints,
                model_config_sha256=configurations,
                tokenizer_sha256="b" * 64,
                utf8_transition_sha256="c" * 64,
            )

    def test_snapshot_identity_detects_nested_tampering(self) -> None:
        snapshot = self._final_snapshots()[PUBLICATION_CANDIDATE_MODEL_KEY]
        tampered = replace(
            snapshot,
            checkpoint_sha256=("f" * 64, *snapshot.checkpoint_sha256[1:]),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_publication_model_snapshot(tampered)

    def test_learning_curve_lock_requires_architecture_and_checkpoint_progress(
        self,
    ) -> None:
        final = self._final_snapshots()[PUBLICATION_CANDIDATE_MODEL_KEY]
        lock = self._curve_lock(final)
        validate_publication_learning_curve_model_lock(lock)
        self.assertEqual(lock.final_snapshot, final)

        reused = dict(zip(lock.budget_bytes, lock.snapshots, strict=True))
        reused[128] = reused[64]
        with self.assertRaisesRegex(ValueError, "reused"):
            build_publication_learning_curve_model_lock(
                model_key=final.model_key,
                budget_bytes=lock.budget_bytes,
                snapshots_by_budget=reused,
            )

        raw_final = self._final_snapshots()[PUBLICATION_RAW_COMPARATOR_MODEL_KEY]
        raw_lock = self._curve_lock(raw_final)
        auxiliary_reused = dict(
            zip(raw_lock.budget_bytes, raw_lock.snapshots, strict=True)
        )
        changed = auxiliary_reused[128]
        changed_bundles = dict(
            zip(
                PUBLICATION_PRETRAIN_SEEDS,
                changed.auxiliary_bundles,
                strict=True,
            )
        )
        current_bundle = changed_bundles[PUBLICATION_PRETRAIN_SEEDS[1]]
        previous_bundle = auxiliary_reused[64].auxiliary_bundles[0]
        changed_bundles[PUBLICATION_PRETRAIN_SEEDS[1]] = (
            build_publication_entropy_router_bundle(
                seed=current_bundle.seed,
                descriptor=changed.raw_reference_descriptor,
                router_checkpoint_artifact_sha256=(
                    previous_bundle.router_checkpoint_artifact_sha256
                ),
                router_checkpoint_state_sha256=(
                    previous_bundle.router_checkpoint_state_sha256
                ),
                router_report_artifact_sha256=(
                    current_bundle.router_report_artifact_sha256
                ),
                router_config_sha256=current_bundle.router_config_sha256,
                router_training_stream_sha256=(
                    current_bundle.router_training_stream_sha256
                ),
                calibration_stream_sha256=(
                    current_bundle.calibration_stream_sha256
                ),
                test_stream_sha256=current_bundle.test_stream_sha256,
                threshold_nats=current_bundle.threshold_nats,
                maximum_patch_length=current_bundle.maximum_patch_length,
                threshold_cache_artifact_sha256=(
                    current_bundle.threshold_cache_artifact_sha256
                ),
                threshold_diagnostics_artifact_sha256=(
                    current_bundle.threshold_diagnostics_artifact_sha256
                ),
                train_patch_matrix_sha256=(
                    current_bundle.train_patch_matrix_sha256
                ),
                calibration_patch_matrix_sha256=(
                    current_bundle.calibration_patch_matrix_sha256
                ),
                test_patch_matrix_sha256=(
                    current_bundle.test_patch_matrix_sha256
                ),
            )
        )
        auxiliary_reused[128] = build_publication_model_snapshot(
            model_key=changed.model_key,
            checkpoint_sha256=dict(
                zip(
                    PUBLICATION_PRETRAIN_SEEDS,
                    changed.checkpoint_sha256,
                    strict=True,
                )
            ),
            model_config_sha256=dict(
                zip(
                    PUBLICATION_PRETRAIN_SEEDS,
                    changed.model_config_sha256,
                    strict=True,
                )
            ),
            raw_reference_descriptor=changed.raw_reference_descriptor,
            auxiliary_bundles=changed_bundles,
            tokenizer_sha256=changed.tokenizer_sha256,
            utf8_transition_sha256=changed.utf8_transition_sha256,
        )
        with self.assertRaisesRegex(ValueError, "auxiliary checkpoint was reused"):
            build_publication_learning_curve_model_lock(
                model_key=raw_final.model_key,
                budget_bytes=raw_lock.budget_bytes,
                snapshots_by_budget=auxiliary_reused,
            )

    def test_final_graph_rejects_cross_evaluation_checkpoint_stitching(self) -> None:
        pairs = self._runtime_pairs()
        final = self._final_snapshots()
        evidence_hashes = {
            key: content_hash(f"runtime:{key}")
            for key in PUBLICATION_COMPARATOR_KEYS
        }
        bpb_hashes = {
            key: content_hash(f"bpb:{key}")
            for key in PUBLICATION_COMPARATOR_KEYS
        }
        graph = build_publication_model_lock_graph(
            runtime_pairs=pairs,
            bpb_pairs=pairs,
            downstream_snapshots={
                key: final[key]
                for key in (
                    PUBLICATION_CANDIDATE_MODEL_KEY,
                    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                )
            },
            learning_curve_final_snapshots=final,
            runtime_evidence_sha256=evidence_hashes,
            bpb_evidence_sha256=bpb_hashes,
            downstream_evidence_sha256=content_hash("downstream"),
            learning_curve_evidence_sha256=content_hash("curves"),
        )
        validate_publication_model_lock_graph(graph)
        self.assertEqual(graph.model_keys, PUBLICATION_MODEL_KEYS)

        drifted_pair = publication_runtime_model_snapshots(
            make_lineage(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                candidate_variant="drifted",
            )
        )
        drifted = dict(pairs)
        drifted[PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]] = drifted_pair
        with self.assertRaisesRegex(ValueError, "checkpoint drift"):
            build_publication_model_lock_graph(
                runtime_pairs=pairs,
                bpb_pairs=drifted,
                downstream_snapshots={
                    PUBLICATION_CANDIDATE_MODEL_KEY: final[
                        PUBLICATION_CANDIDATE_MODEL_KEY
                    ]
                },
                learning_curve_final_snapshots=final,
                runtime_evidence_sha256=evidence_hashes,
                bpb_evidence_sha256=bpb_hashes,
                downstream_evidence_sha256=content_hash("downstream"),
                learning_curve_evidence_sha256=content_hash("curves"),
            )


if __name__ == "__main__":
    unittest.main()
