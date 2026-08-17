from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import balanced_200m_failure_analysis_core as MODULE


class Balanced200MFailureAnalysisTests(unittest.TestCase):
    def _receipt(self) -> dict:
        payload = {
            "schema_version": 1,
            "kind": MODULE.VERIFICATION_KIND,
            "protocol_id": MODULE.PROTOCOL_ID,
            "verification_base_git_commit": "1" * 40,
            "plan_artifact_sha256": "2" * 64,
            "plan_sha256": "3" * 64,
            "training_summary_artifact_sha256": "4" * 64,
            "training_summary_sha256": "5" * 64,
            "training_summary_git_commit": "6" * 40,
            "sealed_verifier_sha256": "7" * 64,
            "transcript_sha256": "8" * 64,
            "checkpoint_replay_roles": ["c86", "w72"],
            "independent_checkpoint_replay_pass": True,
            "quality_status": "balanced_200m_quality_fail",
            "quality": {
                "quality_screen_pass": False,
                "actual_timing_authorized": False,
            },
            "actual_timing_authorized": False,
            "claim_boundary": {
                "one_seed_mechanism_screen": True,
                "sufficiently_trained_llm_claimed": False,
                "actual_incremental_timing_executed": False,
                "verification_replays_full_calibration_forward": True,
            },
        }
        return {**payload, "receipt_sha256": MODULE.canonical_sha256(payload)}

    def test_paired_effect_definition_is_exact(self) -> None:
        c86 = np.asarray([10.0, 12.0], dtype=np.float32)
        w72 = np.asarray([11.0, 10.0], dtype=np.float32)
        effects = MODULE.paired_bpb_effects(c86, w72)
        expected = np.asarray([1.0, -2.0]) / (511 * np.log(2.0))
        np.testing.assert_allclose(effects, expected, rtol=0, atol=0)

    def test_block_bootstrap_is_deterministic_and_drops_only_tail(self) -> None:
        effects = np.linspace(-0.1, 0.2, 137, dtype=np.float64)
        first = MODULE.contiguous_block_bootstrap(
            effects, block_size=8, repetitions=500, seed=17
        )
        second = MODULE.contiguous_block_bootstrap(
            effects, block_size=8, repetitions=500, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first["used_sequences"], 136)
        self.assertEqual(first["dropped_tail_sequences"], 1)
        self.assertLessEqual(first["lower"], first["upper"])

    def test_spearman_and_quintiles_handle_ties(self) -> None:
        feature = np.repeat(np.arange(5), 4)
        effects = feature.astype(np.float64)
        self.assertEqual(MODULE.spearman_correlation(feature, effects), 1.0)
        rows = MODULE.equal_count_quintiles(feature, effects)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["examples"] == 4 for row in rows))

    def test_density_heuristic_has_exact_anchors(self) -> None:
        self.assertEqual(MODULE.linear_density_heuristic(0.0242, 72), 0.0242)
        self.assertEqual(MODULE.linear_density_heuristic(0.0242, 86), 0.0)
        self.assertAlmostEqual(
            MODULE.linear_density_heuristic(0.0242, 80), 0.010371428571428573
        )

    def test_verification_receipt_round_trip_and_tamper(self) -> None:
        receipt = self._receipt()
        MODULE.validate_verification_receipt(receipt)
        tampered = deepcopy(receipt)
        tampered["actual_timing_authorized"] = True
        payload = dict(tampered)
        payload.pop("receipt_sha256")
        tampered["receipt_sha256"] = MODULE.canonical_sha256(payload)
        with self.assertRaisesRegex(ValueError, "receipt differs"):
            MODULE.validate_verification_receipt(tampered)


if __name__ == "__main__":
    unittest.main()

