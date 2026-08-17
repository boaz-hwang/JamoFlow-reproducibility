from __future__ import annotations

import json
import math
import unittest

import numpy as np

from jamoflow.document_inference import document_window_map_from_spans
from jamoflow.inference_final_quality_v2 import (
    FINAL_LOGICAL_ROLES,
    FINAL_SEEDS,
    final_quality_gate_v2,
    resolve_final_evaluation_roles,
)
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
    build_selection_decision_v2,
    build_independent_calibration_recomputation_v2,
    build_selection_lock_v2,
)
from jamoflow.phase3 import PHASE3_POLICIES


def selection_lock_fixture(
    *,
    conversion_reference: bool = False,
    broad_futile: bool = False,
) -> dict:
    values = {
        seed: {policy: 1.5 for policy in CALIBRATION_POLICY_ORDER}
        for seed in INITIAL_SEEDS
    }
    for seed in INITIAL_SEEDS:
        values[seed]["causal_codepoint_grid"] = 1.40
        values[seed]["causal_whitespace_grid_64"] = 1.405
        if conversion_reference:
            for policy in PHASE3_POLICIES:
                values[seed][policy] = 2.0
            values[seed]["causal_codepoint_grid_64"] = 1.0
            values[seed]["causal_whitespace_grid_64"] = 1.005
        else:
            values[seed]["spacebyte_spacelike"] = 1.2 if broad_futile else 1.399
    decision = build_selection_decision_v2(values)
    replay = build_independent_calibration_recomputation_v2(
        values,
        nll_array_sha256_by_seed_policy={
            seed: {policy: "4" * 64 for policy in CALIBRATION_POLICY_ORDER}
            for seed in INITIAL_SEEDS
        },
        evaluator_git_commit="a" * 40,
        verification_git_commit="b" * 40,
        environment_sha256="c" * 64,
        implementation_manifest_sha256="d" * 64,
    )
    return build_selection_lock_v2(
        decision,
        plan_sha256="1" * 64,
        calibration_evidence_manifest_sha256="2" * 64,
        final_test_seal_sha256="3" * 64,
        initial_model_identity_lock_sha256="4" * 64,
        independent_calibration_recomputation=replay,
    )


class InferenceFinalQualityV2Tests(unittest.TestCase):
    def test_roles_are_derived_only_from_the_selection_lock(self) -> None:
        roles = resolve_final_evaluation_roles(selection_lock_fixture())
        self.assertEqual(roles["evaluation_role_order"], list(FINAL_LOGICAL_ROLES))
        self.assertEqual(
            roles["logical_roles"]["broad_reference"]["policy"],
            "spacebyte_spacelike",
        )
        self.assertEqual(
            roles["logical_roles"]["same_rate_codepoint_control"]["policy"],
            "causal_codepoint_grid_64",
        )
        self.assertEqual(len(roles["unique_models"]), 4)

    def test_conversion_reference_and_control_share_one_physical_artifact(self) -> None:
        roles = resolve_final_evaluation_roles(
            selection_lock_fixture(conversion_reference=True)
        )
        self.assertEqual(
            roles["role_to_artifact_role"]["broad_reference"],
            "same_rate_codepoint_control",
        )
        self.assertEqual(len(roles["unique_models"]), 3)

    def test_broad_futility_keeps_the_primary_three_roles_only(self) -> None:
        roles = resolve_final_evaluation_roles(
            selection_lock_fixture(broad_futile=True)
        )
        self.assertEqual(
            roles["evaluation_role_order"],
            [
                "candidate",
                "matched_efficiency_baseline",
                "same_rate_codepoint_control",
            ],
        )
        self.assertEqual(
            roles["broad_reference"]["evaluation_status"],
            "not_authorized_calibration_futility",
        )
        self.assertNotIn("broad_reference", roles["logical_roles"])

    def _quality_fixture(self, *, matched_gap_bpb: float = 0.0) -> tuple:
        sequence_count = 20
        control = np.full(sequence_count, 100.0, dtype=np.float32)
        mechanism_delta = np.float32(-0.005 * 511 * math.log(2.0))
        candidate = (control + mechanism_delta).astype(np.float32)
        matched_delta = np.float32(matched_gap_bpb * 511 * math.log(2.0))
        matched = (candidate - matched_delta).astype(np.float32)
        losses = {
            "candidate": {seed: candidate.copy() for seed in FINAL_SEEDS},
            "matched_efficiency_baseline": {
                seed: matched.copy() for seed in FINAL_SEEDS
            },
            "same_rate_codepoint_control": {
                seed: control.copy() for seed in FINAL_SEEDS
            },
            "broad_reference": {
                seed: candidate.copy() for seed in FINAL_SEEDS
            },
        }
        roles = resolve_final_evaluation_roles(selection_lock_fixture())[
            "logical_roles"
        ]
        window_map = document_window_map_from_spans(
            sequence_count * 512,
            512,
            [
                (index * 512, (index + 1) * 512)
                for index in range(sequence_count)
            ],
        )
        return losses, roles, window_map

    def test_final_gate_requires_both_reference_quality_and_mechanism(self) -> None:
        losses, roles, window_map = self._quality_fixture()
        summary = final_quality_gate_v2(
            losses,
            role_descriptors=roles,
            document_window_map=window_map,
            bootstrap_repetitions=100,
        )
        self.assertTrue(summary["overall_pass"])
        self.assertTrue(summary["actual_timing_authorized"])
        self.assertTrue(
            summary["mechanism_candidate_vs_same_rate_codepoint"]["overall_pass"]
        )
        self.assertEqual(summary, json.loads(json.dumps(summary)))

        losses, roles, window_map = self._quality_fixture(
            matched_gap_bpb=0.020
        )
        failed = final_quality_gate_v2(
            losses,
            role_descriptors=roles,
            document_window_map=window_map,
            bootstrap_repetitions=100,
        )
        self.assertFalse(failed["overall_pass"])
        self.assertFalse(failed["actual_timing_authorized"])

    def test_loss_dtype_and_role_set_are_fail_closed(self) -> None:
        losses, roles, window_map = self._quality_fixture()
        losses["candidate"][FINAL_SEEDS[0]] = losses["candidate"][
            FINAL_SEEDS[0]
        ].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            final_quality_gate_v2(
                losses,
                role_descriptors=roles,
                document_window_map=window_map,
                bootstrap_repetitions=10,
            )


if __name__ == "__main__":
    unittest.main()
