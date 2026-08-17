import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/target-block-kernel-v1/summary.json"
INVALIDATION = ROOT / "results/target-block-kernel-v1/invalidation.json"
V2_SUMMARY = ROOT / "results/target-block-kernel-v2/summary.json"


class TargetBlockKernelResultTest(unittest.TestCase):
    def test_v1_result_is_preserved_but_explicitly_invalidated(self):
        summary_bytes = SUMMARY.read_bytes()
        summary = json.loads(summary_bytes)
        invalidation = json.loads(INVALIDATION.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(summary_bytes).hexdigest(),
            "3fec9807b873f9fb1b6c2f6b58ba31d3d0f0d2ee27414a576b1f9b34ad0bfb4a",
        )
        self.assertEqual(
            invalidation["invalidated_result"]["artifact_sha256"],
            hashlib.sha256(summary_bytes).hexdigest(),
        )
        self.assertEqual(
            summary["status"], "full_speculative_runtime_authorized"
        )
        self.assertFalse(
            invalidation["decision"]["full_speculative_runtime_authorized"]
        )
        self.assertFalse(
            invalidation["decision"]["publication_or_engineering_claim_authorized"]
        )

    def test_v2_inference_mode_result_authorizes_only_the_rollback_prototype(self):
        summary_bytes = V2_SUMMARY.read_bytes()
        summary = json.loads(summary_bytes)
        self.assertEqual(
            hashlib.sha256(summary_bytes).hexdigest(),
            "d47dd7d312492cdeb602d48a69656e82e7ae6b9c83eb8d4e7073976eda1b113d",
        )
        self.assertEqual(summary["kind"], "target_block_kernel_summary_v2")
        self.assertTrue(summary["provenance"]["runtime"]["torch_inference_mode"])
        self.assertEqual(summary["status"], "full_speculative_runtime_authorized")
        self.assertTrue(
            summary["aggregate"]["gates"]["full_speculative_runtime_authorized"]
        )
        self.assertGreater(
            summary["aggregate"]["weighted_micro"]["case_bootstrap_95_interval"][
                "lower"
            ],
            0.20,
        )
        self.assertGreater(
            summary["aggregate"]["perfect_hangul_whole_path"][
                "case_bootstrap_95_interval"
            ]["lower"],
            0.10,
        )
        self.assertGreater(
            summary["aggregate"]["fixed_independent_projection"][
                "case_bootstrap_95_interval"
            ]["lower"],
            0.10,
        )
        self.assertEqual(
            summary["case_context"]["artifact_sha256"],
            "e1e90398c344bab91933b714e6903a9b15abde60ac94553160c0d71d2f168b79",
        )
        self.assertFalse(summary["claim_boundary"]["actual_speculative_rollback_implemented"])
        self.assertFalse(summary["claim_boundary"]["quality_or_final_test_claimed"])


if __name__ == "__main__":
    unittest.main()
