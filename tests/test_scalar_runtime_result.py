import json
import unittest

from scripts.scalar_runtime_protocol import OUTPUT_PATH, RUNTIME_ROLES, canonical_sha256


class ScalarRuntimeResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))

    def test_result_is_complete_correct_and_parameter_matched(self):
        summary = self.summary
        self.assertEqual(summary["kind"], "scalar_runtime_preflight_summary_v1")
        unsigned = dict(summary)
        sealed = unsigned.pop("summary_sha256")
        self.assertEqual(canonical_sha256(unsigned), sealed)
        self.assertEqual(set(summary["metrics"]["correctness"]), set(RUNTIME_ROLES))
        self.assertTrue(
            all(row["pass"] for row in summary["metrics"]["correctness"].values())
        )
        target = summary["metrics"]["parameter_counts"]["byte_w72"]
        for count in summary["metrics"]["parameter_counts"].values():
            self.assertLessEqual(abs(count / target - 1), 0.0025)

    def test_scalar_speedup_is_real_but_strong_bpe_gate_stops_training(self):
        summary = self.summary
        comparisons = summary["metrics"]["comparisons"]
        self.assertGreater(
            comparisons["generic_unicode_scalar_vs_byte_w72"][
                "bootstrap_percentile_95_lower"
            ],
            0.4,
        )
        self.assertGreater(
            comparisons["hangul_hybrid_vs_byte_w72"][
                "bootstrap_percentile_95_lower"
            ],
            0.4,
        )
        self.assertLess(
            comparisons["generic_unicode_scalar_vs_byte_bpe_32000"][
                "bootstrap_percentile_95_upper"
            ],
            -0.9,
        )
        self.assertLess(
            comparisons["hangul_hybrid_vs_generic_unicode_scalar"][
                "bootstrap_percentile_95_upper"
            ],
            -0.05,
        )
        self.assertEqual(
            summary["decision"]["authorized_one_seed_quality_candidates"],
            [],
        )
        self.assertFalse(summary["decision"]["pass"])
        self.assertEqual(
            summary["decision"]["status"],
            "scalar_runtime_branch_stopped",
        )


if __name__ == "__main__":
    unittest.main()
