import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/speculative-w72-preflight-v1/summary.json"


class SpeculativeW72ResultTest(unittest.TestCase):
    def test_exact_runtime_improves_but_does_not_pass_the_locked_gate(self):
        payload = RESULT.read_bytes()
        summary = json.loads(payload)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "106f951e00f37d25bb6d9e97907549250630019fceb5175fa9197b74eae5f44e",
        )
        self.assertEqual(summary["status"], "multi_byte_branch_stopped")
        aggregate = summary["aggregate"]
        self.assertTrue(aggregate["correctness"]["all_outputs_exact"])
        self.assertEqual(aggregate["correctness"]["output_comparisons"], 128)
        self.assertGreater(aggregate["end_to_end"]["reduction"], 0)
        self.assertGreater(
            aggregate["end_to_end"]["prompt_bootstrap_95_interval"]["lower"], 0
        )
        self.assertEqual(aggregate["end_to_end"]["positive_prompt_count"], 110)
        self.assertFalse(aggregate["gates"]["passes"]["point_reduction"])
        self.assertFalse(aggregate["gates"]["passes"]["bootstrap_lower_bound"])
        self.assertTrue(aggregate["gates"]["passes"]["prompt_direction"])
        self.assertFalse(
            aggregate["gates"]["multi_seed_generic_comparator_authorized"]
        )
        self.assertFalse(
            summary["claim_boundary"]["final_or_publication_efficiency_claimed"]
        )


if __name__ == "__main__":
    unittest.main()
