from __future__ import annotations

from copy import deepcopy
import math
import unittest

from jamoflow.compute_conversion import CONVERSION_POLICIES
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
    build_selection_decision_v2,
    build_independent_calibration_recomputation_v2,
    build_selection_lock_v2,
    validate_selection_decision_v2,
    validate_selection_lock_v2,
)
from jamoflow.phase3 import PHASE3_POLICIES


def calibration_fixture() -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for offset, seed in enumerate(INITIAL_SEEDS):
        output[seed] = {
            "fixed_byte_6": 1.50 + offset * 0.001,
            "causal_codepoint_grid": 1.40 + offset * 0.001,
            "causal_whitespace_grid": 1.41 + offset * 0.001,
            "spacebyte_spacelike": 1.399 + offset * 0.001,
            "entropy_threshold_full": 1.60 + offset * 0.001,
            "entropy_threshold_codepoint": 1.70 + offset * 0.001,
            "causal_codepoint_grid_64": 1.45 + offset * 0.001,
            "causal_whitespace_grid_64": 1.405 + offset * 0.001,
            "causal_codepoint_grid_72": 1.43 + offset * 0.001,
            "causal_whitespace_grid_72": 1.401 + offset * 0.001,
        }
    return output


def recomputation_fixture(decision: dict) -> dict:
    values = {
        int(seed): row
        for seed, row in decision["calibration_bpb_by_seed_policy"].items()
    }
    hashes = {
        seed: {policy: f"{index:064x}" for index, policy in enumerate(
            CALIBRATION_POLICY_ORDER, start=1
        )}
        for seed in INITIAL_SEEDS
    }
    return build_independent_calibration_recomputation_v2(
        values,
        nll_array_sha256_by_seed_policy=hashes,
        evaluator_git_commit="a" * 40,
        verification_git_commit="b" * 40,
        environment_sha256="c" * 64,
        implementation_manifest_sha256="d" * 64,
    )


class InferenceSelectionV2Tests(unittest.TestCase):
    def test_rate_and_reference_are_one_calibration_only_decision(self) -> None:
        decision = build_selection_decision_v2(calibration_fixture())
        self.assertEqual(decision["rate_selection"]["selected_rate"], 64)
        self.assertEqual(decision["candidate"]["policy"], "causal_whitespace_grid_64")
        self.assertEqual(decision["reference"]["policy"], "spacebyte_spacelike")
        self.assertEqual(
            decision["matched_efficiency_baseline"]["policy"],
            "causal_codepoint_grid",
        )
        self.assertEqual(
            decision["confirmation_plan"]["phase3_reference"]["policies"],
            ["spacebyte_spacelike"],
        )
        self.assertEqual(
            decision["selection_uses"],
            {
                "calibration": True,
                "final_test": False,
                "historical_screening_test": False,
                "latency": False,
            },
        )
        validate_selection_decision_v2(decision)

    def test_72_is_used_only_when_64_fails(self) -> None:
        values = calibration_fixture()
        for seed in INITIAL_SEEDS:
            values[seed]["causal_whitespace_grid_64"] = (
                values[seed]["causal_codepoint_grid"] + 0.011
            )
        decision = build_selection_decision_v2(values)
        self.assertEqual(decision["rate_selection"]["selected_rate"], 72)
        self.assertEqual(decision["candidate"]["patch_count"], 72)

    def test_no_rate_is_terminal_and_has_no_candidate_or_reference(self) -> None:
        values = calibration_fixture()
        for seed in INITIAL_SEEDS:
            for rate in (64, 72):
                values[seed][f"causal_whitespace_grid_{rate}"] = (
                    values[seed]["causal_codepoint_grid"] + 0.02
                )
        decision = build_selection_decision_v2(values)
        self.assertEqual(decision["status"], "terminal_no_rate")
        self.assertIsNone(decision["candidate"])
        self.assertIsNone(decision["reference"])
        self.assertIsNone(decision["confirmation_plan"])

    def test_broad_reference_futility_does_not_block_narrow_confirmation(self) -> None:
        values = calibration_fixture()
        for seed in INITIAL_SEEDS:
            values[seed]["spacebyte_spacelike"] = 1.20
        decision = build_selection_decision_v2(values)
        self.assertEqual(
            decision["status"],
            "locked_pending_confirmation_and_new_final_test",
        )
        self.assertFalse(
            decision["broad_reference_calibration_screen"]["pass"]
        )
        self.assertEqual(
            decision["broad_reference_evaluation_status"],
            "not_authorized_calibration_futility",
        )
        self.assertIsNotNone(decision["confirmation_plan"]["compute_conversion"])
        self.assertIsNone(decision["confirmation_plan"]["phase3_reference"])
        self.assertEqual(
            decision["reference"]["policy"],
            "spacebyte_spacelike",
        )

    def test_reference_exact_tie_uses_fixed_phase3_order(self) -> None:
        values = calibration_fixture()
        for seed in INITIAL_SEEDS:
            for policy in PHASE3_POLICIES:
                values[seed][policy] = 1.0
            for policy in CONVERSION_POLICIES:
                if "whitespace" not in policy:
                    values[seed][policy] = 1.0
            values[seed]["causal_whitespace_grid_64"] = 1.0
        decision = build_selection_decision_v2(values)
        self.assertEqual(decision["reference"]["policy"], "fixed_byte_6")
        self.assertIsNone(decision["confirmation_plan"]["phase3_reference"])

    def test_selected_rate_codepoint_reference_reuses_conversion_confirmation(self) -> None:
        values = calibration_fixture()
        for seed in INITIAL_SEEDS:
            for policy in PHASE3_POLICIES:
                values[seed][policy] = 2.0
            values[seed]["causal_codepoint_grid_64"] = 1.0
            values[seed]["causal_whitespace_grid_64"] = 1.005
        decision = build_selection_decision_v2(values)
        self.assertEqual(
            decision["reference"]["policy"],
            "causal_codepoint_grid_64",
        )
        self.assertIsNone(decision["confirmation_plan"]["phase3_reference"])
        self.assertEqual(
            decision["confirmation_plan"]["compute_conversion"],
            {
                "authorization_kind": "compute_conversion_confirmation_v2",
                "policies": [
                    "causal_codepoint_grid_64",
                    "causal_whitespace_grid_64",
                ],
                "selected_rate": 64,
                "seeds": [57721, 65537],
            },
        )

    def test_wrong_policy_set_nonfinite_and_tampered_decision_fail_closed(self) -> None:
        values = calibration_fixture()
        del values[1729][CALIBRATION_POLICY_ORDER[-1]]
        with self.assertRaisesRegex(ValueError, "policy set is not exact"):
            build_selection_decision_v2(values)
        values = calibration_fixture()
        values[1729][CALIBRATION_POLICY_ORDER[-1]] = math.nan
        with self.assertRaisesRegex(ValueError, "BPB is invalid"):
            build_selection_decision_v2(values)
        decision = build_selection_decision_v2(calibration_fixture())
        tampered = deepcopy(decision)
        tampered["candidate"]["patch_count"] = 72
        with self.assertRaisesRegex(ValueError, "canonical reconstruction"):
            validate_selection_decision_v2(tampered)

    def test_decision_identity_is_mapping_order_invariant(self) -> None:
        values = calibration_fixture()
        reversed_values = {
            seed: dict(reversed(tuple(values[seed].items())))
            for seed in reversed(INITIAL_SEEDS)
        }
        first = build_selection_decision_v2(values)
        second = build_selection_decision_v2(reversed_values)
        self.assertEqual(first, second)
        self.assertEqual(len(first["decision_sha256"]), 64)

    def test_lock_binds_plan_calibration_evidence_and_final_test(self) -> None:
        decision = build_selection_decision_v2(calibration_fixture())
        lock = build_selection_lock_v2(
            decision,
            plan_sha256="1" * 64,
            calibration_evidence_manifest_sha256="2" * 64,
            final_test_seal_sha256="3" * 64,
            initial_model_identity_lock_sha256="4" * 64,
            independent_calibration_recomputation=recomputation_fixture(
                decision
            ),
        )
        validate_selection_lock_v2(lock)
        tampered = deepcopy(lock)
        tampered["final_test_seal_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            validate_selection_lock_v2(tampered)
        tampered = deepcopy(lock)
        tampered["decision"]["candidate"]["patch_count"] = 72
        with self.assertRaisesRegex(ValueError, "canonical reconstruction"):
            validate_selection_lock_v2(tampered)


if __name__ == "__main__":
    unittest.main()
