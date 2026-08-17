import math
import unittest
from dataclasses import replace

import numpy as np

from jamoflow.data_adequacy import evaluate_publication_data_adequacy
from jamoflow.publication_bpb import (
    RAW_BYTE_TOKENIZER_SHA256,
    build_publication_bpb_context_evidence,
    publication_bpb_scored_bytes,
)
from jamoflow.publication_inference import (
    evaluate_publication_comparator_inference_gate,
    publication_bpb_noninferiority,
    publication_final_value_gate,
    validate_publication_comparator_inference_gate,
    validate_publication_final_value_gate,
)
from jamoflow.publication_model_lock import publication_runtime_model_snapshots
from jamoflow.publication_downstream import (
    TaskPredictionComparison,
    evaluate_publication_downstream_gate,
)
from jamoflow.publication_protocol import (
    PRIMARY_DOWNSTREAM_TASKS,
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)
from tests.publication_runtime_support import make_runtime_evidence
from tests.publication_evidence_support import (
    data_adequacy_evidence_kwargs,
    downstream_evidence_kwargs,
)


class PublicationInferenceGateTests(unittest.TestCase):
    def _downstream(self, *, candidate_variant: str = "shared"):
        comparisons = {}
        for key, spec in PRIMARY_DOWNSTREAM_TASKS.items():
            gold = tuple(range(spec.label_count)) * 6
            comparisons[key] = TaskPredictionComparison(
                task_key=key,
                candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                reference_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                gold=gold,
                train_majority_label=0,
                candidate_by_seed={
                    seed: gold for seed in PUBLICATION_PRETRAIN_SEEDS
                },
                reference_by_seed={
                    seed: gold for seed in PUBLICATION_PRETRAIN_SEEDS
                },
            )
        return evaluate_publication_downstream_gate(
            comparisons,
            bootstrap_repetitions=100,
            **downstream_evidence_kwargs(
                candidate_variant=candidate_variant,
            ),
        )

    def _adequacy(self):
        curves = {
            key: {
                64_000_000: (2.30 + offset, 2.31 + offset, 2.29 + offset),
                128_000_000: (2.10 + offset, 2.11 + offset, 2.09 + offset),
                256_000_000: (1.96 + offset, 1.97 + offset, 1.95 + offset),
            }
            for key, offset in {
                PUBLICATION_CANDIDATE_MODEL_KEY: 0.0,
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: 0.005,
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: -0.004,
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: -0.005,
            }.items()
        }
        return evaluate_publication_data_adequacy(
            curves,
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            bpe_data_matched_keys={
                16_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
                32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            },
            downstream_gate=self._downstream(),
            bootstrap_repetitions=100,
            **data_adequacy_evidence_kwargs(
                (64_000_000, 128_000_000, 256_000_000)
            ),
        )

    def _bpb(
        self,
        effect: float = 0.004,
        *,
        candidate_key: str = PUBLICATION_CANDIDATE_MODEL_KEY,
        comparator_key: str = PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
        candidate_variant: str = "shared",
    ):
        documents = tuple(
            (f"문서 {index:02d} ".encode("utf-8") + b"x" * 100)
            for index in range(32)
        )
        family = (
            "raw_byte"
            if comparator_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY
            else "standard_bpe"
        )
        snapshots = publication_runtime_model_snapshots(
            make_runtime_evidence(
                family,
                comparator_key=comparator_key,
                candidate_variant=candidate_variant,
            ).lineage
        )
        if comparator_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY:
            context_evidence, plans = build_publication_bpb_context_evidence(
                documents,
                candidate_key=candidate_key,
                comparator_key=comparator_key,
                tokenizer_sha256=RAW_BYTE_TOKENIZER_SHA256,
            )
        else:
            comparator_units = tuple(
                tuple(bytes((value,)) for value in document)
                for document in documents
            )
            comparator_token_ids = tuple(tuple(document) for document in documents)
            context_evidence, plans = build_publication_bpb_context_evidence(
                documents,
                candidate_key=candidate_key,
                comparator_key=comparator_key,
                tokenizer_sha256=snapshots[1].tokenizer_sha256,
                comparator_token_ids_by_document=comparator_token_ids,
                comparator_token_bytes_by_document=comparator_units,
            )
        scored_bytes = np.asarray(
            publication_bpb_scored_bytes(plans),
            dtype=np.int64,
        )
        reference = {
            seed: np.full(len(scored_bytes), 40.0)
            for seed in PUBLICATION_PRETRAIN_SEEDS
        }
        candidate = {
            seed: values + effect * math.log(2.0) * scored_bytes
            for seed, values in reference.items()
        }
        return publication_bpb_noninferiority(
            candidate,
            reference,
            scored_bytes,
            candidate_key=candidate_key,
            comparator_key=comparator_key,
            context_evidence=context_evidence,
            candidate_snapshot=snapshots[0],
            comparator_snapshot=snapshots[1],
            bootstrap_repetitions=200,
        )

    def _gate(
        self,
        family: str,
        *,
        comparator_key: str | None = None,
        candidate_latency: float = 8.0,
        downstream_pass: bool = True,
        bpb_effect: float = 0.004,
        runtime_pass: bool = True,
        valid_output_rate: float = 1.0,
        candidate_variant: str = "shared",
    ):
        resolved_comparator_key = comparator_key or (
            PUBLICATION_RAW_COMPARATOR_MODEL_KEY
            if family == "raw_byte"
            else PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]
        )
        runtime_evidence = make_runtime_evidence(
            family,
            comparator_key=resolved_comparator_key,
            candidate_variant=candidate_variant,
            candidate_decode_ms=candidate_latency,
            pass_equivalence=runtime_pass,
            valid_output_rate=valid_output_rate,
        )
        return evaluate_publication_comparator_inference_gate(
            runtime_evidence=runtime_evidence,
            downstream_gate=(
                self._downstream(candidate_variant=candidate_variant)
                if downstream_pass
                else self._uninformative_downstream(
                    candidate_variant=candidate_variant
                )
            ),
            bpb=self._bpb(
                bpb_effect,
                comparator_key=resolved_comparator_key,
                candidate_variant=candidate_variant,
            ),
        )

    def _uninformative_downstream(self, *, candidate_variant: str = "shared"):
        comparisons = {}
        for key, spec in PRIMARY_DOWNSTREAM_TASKS.items():
            gold = tuple(range(spec.label_count)) * 6
            majority = (0,) * len(gold)
            comparisons[key] = TaskPredictionComparison(
                task_key=key,
                candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                reference_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                gold=gold,
                train_majority_label=0,
                candidate_by_seed={
                    seed: majority for seed in PUBLICATION_PRETRAIN_SEEDS
                },
                reference_by_seed={
                    seed: majority for seed in PUBLICATION_PRETRAIN_SEEDS
                },
            )
        return evaluate_publication_downstream_gate(
            comparisons,
            bootstrap_repetitions=100,
            **downstream_evidence_kwargs(
                candidate_variant=candidate_variant,
            ),
        )

    def test_bpb_gate_accepts_constant_effect_inside_margin(self) -> None:
        result = self._bpb(0.004)
        self.assertTrue(result.overall_pass)
        self.assertAlmostEqual(result.mean_difference_bpb, 0.004)
        self.assertLess(result.bootstrap_one_sided_upper_bpb, 0.010)

    def test_bpb_gate_rejects_regression_beyond_margin(self) -> None:
        result = self._bpb(0.020)
        self.assertFalse(result.overall_pass)

    def test_comparator_gate_requires_quality_and_two_actual_speedups(self) -> None:
        result = self._gate("raw_byte")
        self.assertTrue(result.overall_pass)
        self.assertTrue(result.controlled_replay_pass)
        self.assertTrue(result.free_running_pass)
        self.assertEqual(result.controlled_seed_count_at_minimum_reduction, 3)

        slow = self._gate("standard_bpe", candidate_latency=11.0)
        self.assertFalse(slow.overall_pass)
        self.assertFalse(slow.controlled_replay_pass)

        floor = self._gate("standard_bpe", downstream_pass=False)
        self.assertFalse(floor.overall_pass)

        invalid_runtime = self._gate("standard_bpe", runtime_pass=False)
        self.assertFalse(invalid_runtime.overall_pass)

        incomplete_output = self._gate(
            "standard_bpe",
            valid_output_rate=2 / 3,
        )
        self.assertFalse(incomplete_output.overall_pass)
        self.assertFalse(incomplete_output.encoding_quality_pass)

    def test_final_gate_distinguishes_internal_from_broad_claim(self) -> None:
        raw = self._gate("raw_byte")
        fast_16k = self._gate(
            "standard_bpe",
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
        )
        slow_32k = self._gate(
            "standard_bpe",
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            candidate_latency=11.0,
        )
        internal = publication_final_value_gate(
            raw,
            {16_000: fast_16k, 32_000: slow_32k},
            data_adequacy=self._adequacy(),
        )
        self.assertFalse(internal.overall_pass)
        self.assertEqual(internal.claim_level, "bpe_vocabulary_specific_only")

        fast_32k = self._gate(
            "standard_bpe",
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )
        broad = publication_final_value_gate(
            raw,
            {16_000: fast_16k, 32_000: fast_32k},
            data_adequacy=self._adequacy(),
        )
        self.assertTrue(broad.overall_pass)
        self.assertEqual(
            broad.claim_level,
            "broad_korean_inference_efficiency_candidate",
        )

        undertrained = publication_final_value_gate(
            raw,
            {16_000: fast_16k, 32_000: fast_32k},
            data_adequacy=self._inadequate_data(),
        )
        self.assertFalse(undertrained.overall_pass)
        self.assertEqual(undertrained.claim_level, "mac_mechanism_scale_only")

    def test_final_gate_requires_both_predeclared_bpe_vocabularies(self) -> None:
        raw = self._gate("raw_byte")
        bpe = self._gate(
            "standard_bpe",
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )
        with self.assertRaisesRegex(ValueError, "both BPE"):
            publication_final_value_gate(
                raw,
                {32_000: bpe},
                data_adequacy=self._adequacy(),
            )

    def test_final_gate_rejects_data_from_another_identity_graph(self) -> None:
        raw = self._gate("raw_byte")
        bpe_gates = {
            16_000: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
            ),
            32_000: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            ),
        }
        mismatched = replace(
            self._adequacy(),
            raw_comparator_key="different_raw_family",
        )
        with self.assertRaisesRegex(ValueError, "data-adequacy"):
            publication_final_value_gate(
                raw,
                bpe_gates,
                data_adequacy=mismatched,
            )

        with self.assertRaisesRegex(ValueError, "data-adequacy"):
            publication_final_value_gate(
                raw,
                bpe_gates,
                data_adequacy=True,  # type: ignore[arg-type]
            )

    def test_final_gate_rejects_swapped_bpe_vocabulary_roles(self) -> None:
        raw = self._gate("raw_byte")
        swapped = {
            16_000: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            ),
            32_000: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
            ),
        }
        with self.assertRaisesRegex(ValueError, "distinct raw and BPE"):
            publication_final_value_gate(
                raw,
                swapped,
                data_adequacy=self._adequacy(),
            )

    def test_final_gate_rejects_candidate_checkpoint_drift(self) -> None:
        raw = self._gate("raw_byte")
        bpe_gates = {
            16_000: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
            ),
            32_000: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                candidate_variant="different-checkpoint-family",
            ),
        }
        with self.assertRaisesRegex(ValueError, "shared candidate"):
            publication_final_value_gate(
                raw,
                bpe_gates,
                data_adequacy=self._adequacy(),
            )

    def _inadequate_data(self):
        curves = {
            key: {
                64_000_000: (2.30 + offset, 2.31 + offset, 2.29 + offset),
                128_000_000: (2.10 + offset, 2.11 + offset, 2.09 + offset),
                256_000_000: (2.12 + offset, 2.13 + offset, 2.11 + offset),
            }
            for key, offset in {
                PUBLICATION_CANDIDATE_MODEL_KEY: 0.0,
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: 0.005,
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: -0.004,
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: -0.005,
            }.items()
        }
        return evaluate_publication_data_adequacy(
            curves,
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            bpe_data_matched_keys={
                16_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
                32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            },
            downstream_gate=self._downstream(),
            bootstrap_repetitions=100,
            **data_adequacy_evidence_kwargs(
                (64_000_000, 128_000_000, 256_000_000)
            ),
        )

    def test_comparator_gate_rejects_cross_experiment_identity_stitching(self) -> None:
        with self.assertRaisesRegex(ValueError, "downstream evidence"):
            mismatched_downstream = replace(
                self._downstream(),
                candidate_key="different_candidate",
            )
            evaluate_publication_comparator_inference_gate(
                runtime_evidence=make_runtime_evidence("raw_byte"),
                downstream_gate=mismatched_downstream,
                bpb=self._bpb(
                    comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY
                ),
            )

    def test_comparator_and_final_booleans_cannot_be_replaced(self) -> None:
        raw = self._gate("raw_byte")
        tampered_raw = replace(raw, overall_pass=False)
        with self.assertRaisesRegex(ValueError, "comparator evidence"):
            validate_publication_comparator_inference_gate(tampered_raw)

        bpe_gates = {
            size: self._gate(
                "standard_bpe",
                comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size],
            )
            for size in (16_000, 32_000)
        }
        final = publication_final_value_gate(
            raw,
            bpe_gates,
            data_adequacy=self._adequacy(),
        )
        tampered_final = replace(final, overall_pass=False)
        with self.assertRaisesRegex(ValueError, "final-value evidence"):
            validate_publication_final_value_gate(tampered_final)

    def test_bpb_checkpoint_drift_is_rejected_before_latency_gate(self) -> None:
        runtime = make_runtime_evidence("raw_byte")
        drifted_bpb = self._bpb(candidate_variant="different-checkpoint")
        with self.assertRaisesRegex(ValueError, "identity"):
            evaluate_publication_comparator_inference_gate(
                runtime_evidence=runtime,
                downstream_gate=self._downstream(),
                bpb=drifted_bpb,
            )


if __name__ == "__main__":
    unittest.main()
