import hashlib
import json
from pathlib import Path
import unittest

from scripts.scalar_representation_core import canonical_json_sha256


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/scalar-representation-opportunity-v1/summary.json"
EXPECTED_FILE_SHA256 = (
    "5bd1fce05e209842189580e32328ec383c0741d15cbc22519fb023268ccb8c0a"
)
EXPECTED_CANONICAL_SHA256 = (
    "7c60212d640d1ea8521183eac5a8daea8a03584bf64c1b3ad71e480619070bcd"
)


class ScalarRepresentationResultTest(unittest.TestCase):
    def test_result_passes_only_construction_gate_and_bpe_is_shorter(self):
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(RESULT.read_bytes()).hexdigest(),
            EXPECTED_FILE_SHA256,
        )
        claimed = payload.pop("summary_sha256")
        self.assertEqual(claimed, EXPECTED_CANONICAL_SHA256)
        self.assertEqual(canonical_json_sha256(payload), claimed)
        decision = payload["decision"]
        self.assertTrue(decision["pass"])
        self.assertTrue(all(decision["checks"].values()))
        self.assertEqual(
            decision["status"],
            "random_weight_representation_construction_authorized",
        )

        counts = payload["metrics"]["sequential_step_comparison"]
        self.assertEqual(counts["raw_byte"], 8_000_000)
        self.assertEqual(counts["generic_unicode_scalar"], 3_330_977)
        self.assertEqual(counts["hangul_hybrid"], 3_392_568)
        self.assertEqual(counts["byte_bpe_16000"], 1_533_938)
        self.assertEqual(counts["byte_bpe_32000"], 1_388_745)
        self.assertLess(counts["byte_bpe_32000"], counts["generic_unicode_scalar"])
        self.assertLess(counts["byte_bpe_16000"], counts["hangul_hybrid"])

        for size in ("16000", "32000"):
            row = payload["metrics"]["bpe"][size]
            self.assertTrue(row["roundtrip_identity"])
            self.assertTrue(row["raw_token_bytes_identity"])
            self.assertTrue(row["deterministic_replicate_json_identity"])
            self.assertTrue(row["structural_audit"]["overall_pass"])

        opportunity = payload["metrics"]["dense_matmul_opportunity"]
        self.assertAlmostEqual(
            opportunity["generic_unicode_scalar"]["reduction_relative_to_w72"],
            0.36621631111314457,
        )
        self.assertAlmostEqual(
            opportunity["hangul_scalar_otherwise_raw_byte"][
                "reduction_relative_to_w72"
            ],
            0.36252034520487064,
        )
        self.assertFalse(
            payload["claim_boundary"]["actual_latency_or_memory_evidence"]
        )
        self.assertFalse(payload["claim_boundary"]["matched_quality_evidence"])
        self.assertTrue(
            payload["interpretation"][
                "hangul_advantage_over_generic_scalar_not_yet_established"
            ]
        )

    def test_conditional_dependence_and_low_in_domain_oov_are_locked(self):
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        inventory = payload["metrics"]["scalar_inventory"]
        self.assertEqual(inventory["train"]["unique_scalars"], 7_006)
        self.assertEqual(
            inventory["calibration"]["unseen_scalar_occurrences"], 149
        )
        self.assertLess(
            inventory["calibration"]["unseen_scalar_occurrence_rate"],
            0.001,
        )
        for split in ("train", "calibration"):
            row = payload["metrics"]["hangul_dependence"][split]
            entropy = row["entropy_bits"]
            self.assertAlmostEqual(
                entropy["conditional_chain_total"], entropy["joint"]
            )
            self.assertGreater(
                entropy["independent_excess_total_correlation"], 1.2
            )


if __name__ == "__main__":
    unittest.main()
