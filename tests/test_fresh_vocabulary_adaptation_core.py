from __future__ import annotations

import math
import unittest

import numpy as np
from fresh_vocabulary_adaptation_core import (
    HEAD_MINIMUM_LEARNING_RATE,
    HEAD_PEAK_LEARNING_RATE,
    ROLES,
    adaptation_decision,
    head_learning_rate,
    inplace_stage_contract,
    role_definition,
)


class FreshVocabularyAdaptationCoreTest(unittest.TestCase):
    def test_roles_are_exact_and_ordinary_dense(self) -> None:
        self.assertEqual(len(ROLES), 4)
        self.assertEqual(role_definition("dense2k_joint")["vocabulary_size"], 2_048)
        for role in ROLES[1:]:
            row = role_definition(role)
            self.assertEqual(row["vocabulary_size"], 8_192)
            self.assertEqual(
                row["initialization"],
                "untied_uniform_input_byte_weighted_output",
            )

    def test_stage_boundary_is_first_complete_batch_crossing_sixty_percent(
        self,
    ) -> None:
        raw = np.ones(100, dtype=np.int64)
        contract = inplace_stage_contract(raw)
        self.assertEqual(contract["total_optimizer_steps"], 4)
        self.assertEqual(contract["stage_one_optimizer_steps"], 2)
        self.assertEqual(contract["stage_one_raw_target_bytes"], 64)
        self.assertEqual(contract["stage_two_optimizer_steps"], 2)
        self.assertAlmostEqual(contract["stage_one_realized_raw_fraction"], 0.64)

    def test_learning_rate_follows_raw_progress_and_rewarms_stage_two(self) -> None:
        standard_early = head_learning_rate(
            "dense8k_standard_joint",
            cumulative_raw_target_bytes=1,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=None,
        )
        standard_peak = head_learning_rate(
            "dense8k_standard_joint",
            cumulative_raw_target_bytes=5,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=None,
        )
        standard_end = head_learning_rate(
            "dense8k_standard_joint",
            cumulative_raw_target_bytes=100,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=None,
        )
        self.assertLess(standard_early, standard_peak)
        self.assertAlmostEqual(standard_peak, HEAD_PEAK_LEARNING_RATE)
        self.assertAlmostEqual(standard_end, HEAD_MINIMUM_LEARNING_RATE)

        stage_peak = head_learning_rate(
            "dense8k_inplace_two_stage",
            cumulative_raw_target_bytes=30,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=60,
        )
        stage_end = head_learning_rate(
            "dense8k_inplace_two_stage",
            cumulative_raw_target_bytes=60,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=60,
        )
        stage_two_first = head_learning_rate(
            "dense8k_inplace_two_stage",
            cumulative_raw_target_bytes=61,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=60,
        )
        final = head_learning_rate(
            "dense8k_inplace_two_stage",
            cumulative_raw_target_bytes=100,
            total_raw_target_bytes=100,
            stage_one_raw_target_bytes=60,
        )
        self.assertAlmostEqual(stage_peak, HEAD_PEAK_LEARNING_RATE)
        self.assertAlmostEqual(stage_end, HEAD_PEAK_LEARNING_RATE)
        self.assertLess(stage_two_first, HEAD_PEAK_LEARNING_RATE)
        self.assertAlmostEqual(final, HEAD_MINIMUM_LEARNING_RATE)

    @staticmethod
    def _nll_for_bpb(value: float, raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.float64) * math.log(2.0) * value

    def test_method_and_deployment_gate_can_pass(self) -> None:
        raw = np.full(32, 1_000, dtype=np.int64)
        values = {
            "dense2k_joint": 1.400,
            "dense8k_standard_joint": 1.405,
            "dense8k_inplace_two_stage": 1.403,
            "dense8k_update_geometry": 1.399,
        }
        decision = adaptation_decision(
            {role: self._nll_for_bpb(values[role], raw) for role in ROLES}, raw
        )
        self.assertTrue(decision["actual_inference_preflight_authorized"])
        self.assertEqual(
            decision["selected_dense8k_role_for_actual_preflight"],
            "dense8k_update_geometry",
        )
        self.assertTrue(decision["optimizer_geometry_method_supported"])
        self.assertTrue(decision["fresh_multiseed_method_confirmation_authorized"])

    def test_deployment_opportunity_does_not_imply_method_novelty(self) -> None:
        raw = np.full(32, 1_000, dtype=np.int64)
        values = {
            "dense2k_joint": 1.400,
            "dense8k_standard_joint": 1.401,
            "dense8k_inplace_two_stage": 1.402,
            "dense8k_update_geometry": 1.4015,
        }
        decision = adaptation_decision(
            {role: self._nll_for_bpb(values[role], raw) for role in ROLES}, raw
        )
        self.assertEqual(
            decision["status"], "deployment_opportunity_without_optimizer_novelty"
        )
        self.assertEqual(
            decision["selected_dense8k_role_for_actual_preflight"],
            "dense8k_standard_joint",
        )
        self.assertFalse(decision["optimizer_geometry_method_supported"])

    def test_quality_failure_stops_actual_preflight(self) -> None:
        raw = np.full(32, 1_000, dtype=np.int64)
        values = {role: 1.421 for role in ROLES}
        values["dense2k_joint"] = 1.400
        decision = adaptation_decision(
            {role: self._nll_for_bpb(values[role], raw) for role in ROLES}, raw
        )
        self.assertEqual(decision["status"], "no_quality_qualified_dense8k")
        self.assertIsNone(decision["selected_dense8k_role_for_actual_preflight"])
        self.assertFalse(decision["actual_inference_preflight_authorized"])


if __name__ == "__main__":
    unittest.main()
