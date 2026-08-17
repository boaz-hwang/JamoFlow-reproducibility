import unittest

from compositional_head_core import (
    BASE_ROLE,
    BODY_PARAMETER_COUNT,
    HEAD_PARAMETER_BUDGET,
    ROLE_ORDER,
    ROLE_SPECS,
    analytical_head_multiply_adds_per_position,
    balanced_role_schedule,
    parameter_fraction_from_baseline,
    paired_latency_comparison,
    parse_role,
    preflight_decision,
)


class CompositionalHeadCoreTests(unittest.TestCase):
    def test_role_grid_and_parameter_budgets(self) -> None:
        self.assertEqual(len(ROLE_ORDER), 13)
        self.assertEqual(ROLE_SPECS[BASE_ROLE].expected_parameters, 19_667_328)
        self.assertEqual(HEAD_PARAMETER_BUDGET, 786_432)
        self.assertEqual(BODY_PARAMETER_COUNT, 18_880_896)
        for vocabulary_size in (8_192, 16_000, 32_000):
            generic = ROLE_SPECS[f"generic_code_v{vocabulary_size}"]
            hangul = ROLE_SPECS[f"hangul_code_v{vocabulary_size}"]
            self.assertEqual(generic.expected_parameters, 19_667_328)
            self.assertEqual(hangul.expected_parameters, generic.expected_parameters)
            self.assertEqual(generic.head_parameters, HEAD_PARAMETER_BUDGET)
        self.assertEqual(ROLE_SPECS["low_rank_v8192"].rank, 92)
        self.assertEqual(ROLE_SPECS["low_rank_v16000"].rank, 48)
        self.assertEqual(ROLE_SPECS["low_rank_v32000"].rank, 24)

    def test_role_parser_is_exact(self) -> None:
        self.assertEqual(parse_role(BASE_ROLE), ("dense", 2_048))
        self.assertEqual(parse_role("hangul_code_v8192"), ("hangul_code", 8_192))
        for invalid in ("dense_v2048x", "generic_code_v4096", "other_v8192"):
            with self.assertRaises(ValueError):
                parse_role(invalid)

    def test_codebook_head_cost_grows_only_with_gather(self) -> None:
        baseline = analytical_head_multiply_adds_per_position(BASE_ROLE)
        self.assertEqual(baseline, 786_432)
        self.assertEqual(
            analytical_head_multiply_adds_per_position("hangul_code_v8192"),
            786_432 + 16 * 8_192,
        )
        self.assertLess(
            analytical_head_multiply_adds_per_position("hangul_code_v32000"),
            analytical_head_multiply_adds_per_position("dense_v32000"),
        )

    def test_parameter_differences_are_explicit(self) -> None:
        self.assertEqual(parameter_fraction_from_baseline("hangul_code_v32000"), 0.0)
        self.assertAlmostEqual(
            parameter_fraction_from_baseline("low_rank_v8192"),
            2_560 / 19_667_328,
        )
        self.assertAlmostEqual(
            parameter_fraction_from_baseline("low_rank_v32000"),
            -9_216 / 19_667_328,
        )

    def test_paired_latency_collapses_repetitions_within_prompt(self) -> None:
        import numpy as np

        baseline = np.asarray([[100.0, 101.0, 99.0], [120.0, 121.0, 119.0]])
        candidate = baseline * 0.8
        baseline_steps = np.asarray([[10, 10, 10], [12, 12, 12]])
        candidate_steps = np.asarray([[8, 8, 8], [9, 9, 9]])
        row = paired_latency_comparison(
            candidate,
            baseline,
            candidate_steps,
            baseline_steps,
            bootstrap_seed=3,
            bootstrap_repetitions=100,
        )
        self.assertAlmostEqual(row["end_to_end_reduction"], 0.2)
        self.assertEqual(row["positive_prompt_count"], 2)
        self.assertGreater(row["continuation_step_reduction"], 0.2)

    def test_decision_selects_smallest_jointly_passing_code_vocabulary(self) -> None:
        comparisons = {}
        correctness = {role: True for role in ROLE_ORDER}
        for role in ROLE_ORDER:
            if role == BASE_ROLE:
                continue
            comparisons[role] = {
                "bootstrap_95_lower": 0.01,
                "continuation_step_reduction": 0.2,
                "end_to_end_reduction": 0.11,
                "positive_prompt_count": 30,
                "prompt_count": 36,
            }
        comparisons["generic_code_v8192"]["end_to_end_reduction"] = 0.09
        decision = preflight_decision(comparisons, correctness)
        self.assertEqual(decision["selected_vocabulary_size"], 16_000)
        self.assertEqual(decision["selected_candidate_role"], "hangul_code_v16000")

    def test_decision_does_not_fallback_to_one_code_assignment(self) -> None:
        comparisons = {}
        correctness = {role: True for role in ROLE_ORDER}
        for role in ROLE_ORDER:
            if role == BASE_ROLE:
                continue
            comparisons[role] = {
                "bootstrap_95_lower": 0.01,
                "continuation_step_reduction": 0.2,
                "end_to_end_reduction": 0.11,
                "positive_prompt_count": 30,
                "prompt_count": 36,
            }
        for vocabulary_size in (8_192, 16_000, 32_000):
            comparisons[f"hangul_code_v{vocabulary_size}"]["bootstrap_95_lower"] = -0.01
        self.assertIsNone(
            preflight_decision(comparisons, correctness)["selected_vocabulary_size"]
        )

    def test_schedule_is_a_rotating_permutation(self) -> None:
        schedules = [balanced_role_schedule(index, 0) for index in range(len(ROLE_ORDER))]
        self.assertTrue(all(set(row) == set(ROLE_ORDER) for row in schedules))
        self.assertEqual(len({row[0] for row in schedules}), len(ROLE_ORDER))

    def test_schedule_balances_every_position_over_measured_trials(self) -> None:
        schedules = [
            balanced_role_schedule(case_index, repetition)
            for case_index in range(36)
            for repetition in range(3)
        ]
        for position in range(len(ROLE_ORDER)):
            counts = {
                role: sum(row[position] == role for row in schedules)
                for role in ROLE_ORDER
            }
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
