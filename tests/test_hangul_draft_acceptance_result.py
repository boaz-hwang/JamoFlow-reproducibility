import hashlib
import json
from pathlib import Path
import unittest

from scripts.hangul_draft_acceptance_core import ARCHITECTURES


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/hangul-draft-acceptance-v1/summary.json"
EXPECTED_ARTIFACT_SHA256 = (
    "0e31dada00ca04835432f8f35b2e438b225dbed58e4ee12f262aff081fcd3591"
)


class HangulDraftAcceptanceResultTest(unittest.TestCase):
    def test_authoritative_result_identity_and_negative_gate(self):
        raw = RESULT.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), EXPECTED_ARTIFACT_SHA256)
        summary = json.loads(raw)
        unsigned = dict(summary)
        claimed = unsigned.pop("summary_sha256")
        canonical = (
            json.dumps(
                unsigned,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(claimed, hashlib.sha256(canonical).hexdigest())
        self.assertEqual(summary["status"], "hangul_prototype_not_authorized")
        self.assertFalse(summary["gates"]["overall_hangul_prototype_authorized"])
        self.assertFalse(summary["gates"]["generic_control_prototype_authorized"])
        self.assertEqual(
            summary["gates"]["recommended_next_stage"],
            "stop_multi_byte_draft_branch",
        )

    def test_all_heads_failed_and_independent_was_strongest(self):
        summary = json.loads(RESULT.read_text(encoding="utf-8"))
        rows = summary["architecture_summary"]
        self.assertEqual(set(rows), set(ARCHITECTURES))
        self.assertEqual(summary["data"]["free_attempt_count"], 14_422)
        self.assertEqual(summary["data"]["free_target_hangul_rate"], 1.0)
        self.assertTrue(
            all(
                not row["pass"]
                for row in summary["gates"]["systems_feasibility"].values()
            )
        )
        best = max(
            rows,
            key=lambda architecture: rows[architecture][
                "median_free_complete_pair_acceptance"
            ],
        )
        self.assertEqual(best, "generic_independent_utf8")
        self.assertAlmostEqual(
            rows[best]["median_free_complete_pair_acceptance"],
            0.2437942033005131,
        )
        self.assertAlmostEqual(
            rows["hangul_conditional_components"][
                "median_free_complete_pair_acceptance"
            ],
            0.1770212175842463,
        )
        self.assertLess(
            summary["gates"]["primary_korean_specificity"]["ci_lower"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
