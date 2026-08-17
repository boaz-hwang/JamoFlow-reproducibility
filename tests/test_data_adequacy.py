import unittest
from dataclasses import replace

from jamoflow.data_adequacy import (
    evaluate_publication_data_adequacy,
    validate_publication_data_adequacy,
)
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
from tests.publication_evidence_support import (
    data_adequacy_evidence_kwargs,
    downstream_evidence_kwargs,
)


class PublicationDataAdequacyTests(unittest.TestCase):
    def _downstream_gate(
        self,
        *,
        informative: bool = True,
    ):
        comparisons = {}
        for key, spec in PRIMARY_DOWNSTREAM_TASKS.items():
            gold = tuple(range(spec.label_count)) * 6
            reference = gold if informative else (0,) * len(gold)
            comparisons[key] = TaskPredictionComparison(
                task_key=key,
                candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                reference_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                gold=gold,
                train_majority_label=0,
                candidate_by_seed={
                    seed: reference for seed in PUBLICATION_PRETRAIN_SEEDS
                },
                reference_by_seed={
                    seed: reference for seed in PUBLICATION_PRETRAIN_SEEDS
                },
            )
        return evaluate_publication_downstream_gate(
            comparisons,
            bootstrap_repetitions=100,
            **downstream_evidence_kwargs(),
        )

    def _curves(self) -> dict:
        return {
            PUBLICATION_CANDIDATE_MODEL_KEY: {
                64_000_000: (2.30, 2.31, 2.29),
                128_000_000: (2.10, 2.11, 2.09),
                256_000_000: (1.96, 1.97, 1.95),
            },
            PUBLICATION_RAW_COMPARATOR_MODEL_KEY: {
                64_000_000: (2.305, 2.315, 2.295),
                128_000_000: (2.105, 2.115, 2.095),
                256_000_000: (1.965, 1.975, 1.955),
            },
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: {
                64_000_000: (2.295, 2.305, 2.285),
                128_000_000: (2.095, 2.105, 2.085),
                256_000_000: (1.955, 1.965, 1.945),
            },
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: {
                64_000_000: (2.296, 2.306, 2.286),
                128_000_000: (2.096, 2.106, 2.086),
                256_000_000: (1.956, 1.966, 1.946),
            },
        }

    def _evaluate(self, curves: dict, *, floor: bool = True):
        return evaluate_publication_data_adequacy(
            curves,
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            bpe_data_matched_keys={
                16_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
                32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            },
            downstream_gate=self._downstream_gate(informative=floor),
            bootstrap_repetitions=200,
            **data_adequacy_evidence_kwargs((64_000_000, 128_000_000, 256_000_000)),
        )

    def test_stable_learning_curves_and_capability_floor_pass(self) -> None:
        result = self._evaluate(self._curves())
        self.assertTrue(result.overall_pass)
        self.assertTrue(
            all(pair.last_two_budget_noninferiority_pass for pair in result.pairs)
        )

    def test_order_reversal_inside_noninferiority_margin_is_allowed(self) -> None:
        curves = self._curves()
        curves[PUBLICATION_RAW_COMPARATOR_MODEL_KEY][128_000_000] = (
            2.104,
            2.114,
            2.094,
        )
        curves[PUBLICATION_RAW_COMPARATOR_MODEL_KEY][256_000_000] = (
            1.956,
            1.966,
            1.946,
        )
        result = self._evaluate(curves)
        self.assertTrue(result.overall_pass)
        self.assertLess(
            result.pairs[0].previous_candidate_minus_comparator_bpb,
            0,
        )
        self.assertGreater(
            result.pairs[0].current_candidate_minus_comparator_bpb,
            0,
        )

    def test_near_zero_gap_is_valid_quality_retention(self) -> None:
        curves = self._curves()
        curves[PUBLICATION_RAW_COMPARATOR_MODEL_KEY][128_000_000] = (
            2.101,
            2.111,
            2.091,
        )
        curves[PUBLICATION_RAW_COMPARATOR_MODEL_KEY][256_000_000] = (
            1.961,
            1.971,
            1.951,
        )
        result = self._evaluate(curves)
        self.assertTrue(result.overall_pass)
        self.assertTrue(result.pairs[0].last_two_budget_noninferiority_pass)

    def test_margin_violation_at_either_recent_budget_is_unstable(self) -> None:
        curves = self._curves()
        curves[PUBLICATION_RAW_COMPARATOR_MODEL_KEY][128_000_000] = (
            2.08,
            2.09,
            2.07,
        )
        result = self._evaluate(curves)
        self.assertFalse(result.overall_pass)
        self.assertEqual(result.status, "last_two_budget_quality_unstable")
        self.assertFalse(result.pairs[0].last_two_budget_noninferiority_pass)

    def test_nonimproving_model_is_unstable_even_when_gap_is_safe(self) -> None:
        curves = self._curves()
        curves[PUBLICATION_RAW_COMPARATOR_MODEL_KEY][256_000_000] = (
            2.109,
            2.119,
            2.099,
        )
        result = self._evaluate(curves)
        self.assertFalse(result.overall_pass)
        self.assertFalse(result.pairs[0].learning_progress_pass)

    def test_downstream_floor_failure_is_capability_undertrained(self) -> None:
        result = self._evaluate(self._curves(), floor=False)
        self.assertFalse(result.overall_pass)
        self.assertEqual(result.status, "capability_undertrained")

    def test_missing_budget_or_model_is_rejected(self) -> None:
        curves = self._curves()
        curves[PUBLICATION_CANDIDATE_MODEL_KEY].pop(64_000_000)
        with self.assertRaisesRegex(ValueError, "every preregistered"):
            self._evaluate(curves)

    def test_single_bpe_control_cannot_enter_data_adequacy_gate(self) -> None:
        curves = self._curves()
        curves.pop(PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000])
        with self.assertRaisesRegex(ValueError, "both BPE"):
            evaluate_publication_data_adequacy(
                curves,
                candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
                bpe_data_matched_keys={
                    32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]
                },
                downstream_gate=self._downstream_gate(),
                **data_adequacy_evidence_kwargs(
                    (64_000_000, 128_000_000, 256_000_000)
                ),
            )

    def test_fixed_1024m_extension_uses_its_last_two_budgets(self) -> None:
        budgets = (256_000_000, 512_000_000, 1_024_000_000)
        curves = {
            key: {
                budget: tuple(
                    2.0
                    - 0.1 * budget_index
                    + offset
                    + 0.001 * seed_index
                    for seed_index in range(3)
                )
                for budget_index, budget in enumerate(budgets)
            }
            for key, offset in {
                PUBLICATION_CANDIDATE_MODEL_KEY: 0.0,
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: 0.004,
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: -0.004,
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: -0.005,
            }.items()
        }
        result = evaluate_publication_data_adequacy(
            curves,
            candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
            raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            bpe_data_matched_keys={
                16_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
                32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            },
            downstream_gate=self._downstream_gate(),
            learning_curve_bytes=budgets,
            bootstrap_repetitions=200,
            **data_adequacy_evidence_kwargs(budgets),
        )
        self.assertTrue(result.overall_pass)
        self.assertTrue(
            all(pair.previous_budget_bytes == 512_000_000 for pair in result.pairs)
        )
        self.assertTrue(
            all(pair.current_budget_bytes == 1_024_000_000 for pair in result.pairs)
        )

    def test_downstream_candidate_identity_must_match_curves(self) -> None:
        mismatched = replace(
            self._downstream_gate(),
            candidate_key="other",
        )
        with self.assertRaisesRegex(ValueError, "downstream"):
            evaluate_publication_data_adequacy(
                self._curves(),
                candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
                bpe_data_matched_keys={
                    16_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
                    32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                },
                downstream_gate=mismatched,
                bootstrap_repetitions=20,
                **data_adequacy_evidence_kwargs(
                    (64_000_000, 128_000_000, 256_000_000)
                ),
            )

    def test_data_adequacy_rejects_unsealed_bpe_alias(self) -> None:
        curves = self._curves()
        values = curves.pop(PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000])
        curves["posthoc_bpe_16000"] = values
        with self.assertRaisesRegex(ValueError, "learning-curve model locks"):
            evaluate_publication_data_adequacy(
                curves,
                candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
                raw_comparator_key=PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
                bpe_data_matched_keys={
                    16_000: "posthoc_bpe_16000",
                    32_000: PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
                },
                downstream_gate=self._downstream_gate(),
                bootstrap_repetitions=20,
                **data_adequacy_evidence_kwargs(
                    (64_000_000, 128_000_000, 256_000_000)
                ),
            )

    def test_curve_manifest_cannot_be_replaced_after_evaluation(self) -> None:
        result = self._evaluate(self._curves())
        tampered = replace(result, curve_arrays_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "data-adequacy evidence"):
            validate_publication_data_adequacy(tampered)


if __name__ == "__main__":
    unittest.main()
