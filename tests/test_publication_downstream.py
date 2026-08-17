import unittest
from dataclasses import replace

from jamoflow.publication_downstream import (
    TaskPredictionComparison,
    accuracy_score,
    evaluate_publication_downstream_gate,
    macro_f1_score,
    validate_publication_downstream_gate,
)
from jamoflow.publication_protocol import (
    PRIMARY_DOWNSTREAM_TASKS,
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
)
from tests.publication_evidence_support import downstream_evidence_kwargs


class PublicationDownstreamMetricTests(unittest.TestCase):
    def test_accuracy_and_fixed_label_macro_f1(self) -> None:
        gold = (0, 0, 1, 1)
        predicted = (0, 1, 1, 1)
        self.assertEqual(accuracy_score(gold, predicted), 0.75)
        self.assertAlmostEqual(
            macro_f1_score(gold, predicted, label_count=2),
            (2 / 3 + 4 / 5) / 2,
        )

    def test_macro_f1_counts_absent_fixed_label_as_zero(self) -> None:
        self.assertAlmostEqual(
            macro_f1_score((0, 0), (0, 0), label_count=2),
            0.5,
        )


class PublicationDownstreamGateTests(unittest.TestCase):
    def _comparisons(self, *, reference_mode: str = "perfect") -> dict:
        comparisons = {}
        for key, spec in PRIMARY_DOWNSTREAM_TASKS.items():
            gold = tuple(range(spec.label_count)) * 6
            if reference_mode == "majority":
                reference = (0,) * len(gold)
            else:
                reference = gold
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
                    seed: reference for seed in PUBLICATION_PRETRAIN_SEEDS
                },
            )
        return comparisons

    def test_equal_high_quality_models_pass(self) -> None:
        result = evaluate_publication_downstream_gate(
            self._comparisons(),
            bootstrap_repetitions=100,
            **downstream_evidence_kwargs(),
        )
        self.assertTrue(result.overall_pass)
        self.assertTrue(result.informative_suite_pass)
        self.assertTrue(result.family_noninferiority_pass)

    def test_equal_models_at_majority_floor_do_not_pass(self) -> None:
        comparisons = self._comparisons(reference_mode="majority")
        for key, comparison in tuple(comparisons.items()):
            comparisons[key] = TaskPredictionComparison(
                task_key=comparison.task_key,
                candidate_key=comparison.candidate_key,
                reference_key=comparison.reference_key,
                gold=comparison.gold,
                train_majority_label=comparison.train_majority_label,
                candidate_by_seed=comparison.reference_by_seed,
                reference_by_seed=comparison.reference_by_seed,
            )
        result = evaluate_publication_downstream_gate(
            comparisons,
            bootstrap_repetitions=100,
            **downstream_evidence_kwargs(),
        )
        self.assertFalse(result.overall_pass)
        self.assertFalse(result.informative_suite_pass)

    def test_large_single_task_regression_fails_guard(self) -> None:
        comparisons = self._comparisons()
        target = comparisons["kobest_boolq"]
        wrong = tuple(1 - value for value in target.gold)
        comparisons["kobest_boolq"] = TaskPredictionComparison(
            task_key=target.task_key,
            candidate_key=target.candidate_key,
            reference_key=target.reference_key,
            gold=target.gold,
            train_majority_label=target.train_majority_label,
            candidate_by_seed={seed: wrong for seed in PUBLICATION_PRETRAIN_SEEDS},
            reference_by_seed=target.reference_by_seed,
        )
        result = evaluate_publication_downstream_gate(
            comparisons,
            bootstrap_repetitions=100,
            **downstream_evidence_kwargs(),
        )
        self.assertFalse(result.overall_pass)
        self.assertFalse(result.individual_task_guard_pass)

    def test_gate_rejects_task_or_seed_subset(self) -> None:
        comparisons = self._comparisons()
        comparisons.pop("klue_nli")
        with self.assertRaisesRegex(ValueError, "exact primary"):
            evaluate_publication_downstream_gate(
                comparisons,
                bootstrap_repetitions=10,
                **downstream_evidence_kwargs(),
            )

    def test_gate_rejects_mixed_candidate_identities(self) -> None:
        comparisons = self._comparisons()
        target = comparisons["kobest_boolq"]
        comparisons["kobest_boolq"] = TaskPredictionComparison(
            task_key=target.task_key,
            candidate_key="different_candidate",
            reference_key=target.reference_key,
            gold=target.gold,
            train_majority_label=target.train_majority_label,
            candidate_by_seed=target.candidate_by_seed,
            reference_by_seed=target.reference_by_seed,
        )
        with self.assertRaisesRegex(ValueError, "sealed candidate"):
            evaluate_publication_downstream_gate(
                comparisons,
                bootstrap_repetitions=10,
                **downstream_evidence_kwargs(),
            )

    def test_gate_rejects_unsealed_reference_role(self) -> None:
        comparisons = self._comparisons()
        comparisons["kobest_boolq"] = replace(
            comparisons["kobest_boolq"],
            reference_key="posthoc_reference",
        )
        with self.assertRaisesRegex(ValueError, "invalid downstream"):
            evaluate_publication_downstream_gate(
                comparisons,
                bootstrap_repetitions=10,
                **downstream_evidence_kwargs(),
            )

    def test_prediction_manifest_cannot_be_replaced_after_evaluation(self) -> None:
        result = evaluate_publication_downstream_gate(
            self._comparisons(),
            bootstrap_repetitions=20,
            **downstream_evidence_kwargs(),
        )
        tampered = replace(result, prediction_arrays_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "downstream evidence"):
            validate_publication_downstream_gate(tampered)


if __name__ == "__main__":
    unittest.main()
