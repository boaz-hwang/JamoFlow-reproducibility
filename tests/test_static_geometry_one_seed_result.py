import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/static-geometry-one-seed-v1/summary.json"
EXPECTED_ARTIFACT_SHA256 = (
    "f8a5276dc2c907ed477462633d3798fb2115a399288bfd24647b84e73cd65be0"
)


def _canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class StaticGeometryOneSeedResultTest(unittest.TestCase):
    def test_latency_passes_but_quality_stops_the_static_branch(self):
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(RESULT.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )
        claimed = payload.pop("summary_sha256")
        self.assertEqual(hashlib.sha256(_canonical_bytes(payload)).hexdigest(), claimed)

        self.assertEqual(payload["status"], "one_seed_static_control_stopped")
        decision = payload["decision"]
        self.assertFalse(decision["quality_pass"])
        self.assertTrue(decision["actual_latency_pass"])
        self.assertFalse(decision["multi_seed_static_control_authorized"])
        self.assertFalse(decision["conditional_local_compute_research_authorized"])

        quality = payload["quality"]
        self.assertFalse(quality["overall_pass"])
        self.assertAlmostEqual(quality["mean_difference_bpb"], 0.09560098394219652)
        self.assertAlmostEqual(
            quality["document_bootstrap"]["one_sided_95_upper_bpb"],
            0.09673958919597228,
        )
        self.assertGreater(
            quality["mean_difference_bpb"],
            quality["noninferiority_margin_bpb"],
        )
        self.assertTrue(quality["passes"]["document_coverage"])
        self.assertFalse(quality["passes"]["mean_difference"])
        self.assertFalse(quality["passes"]["one_sided_document_upper"])

        timing = payload["actual_timing"]
        self.assertTrue(timing["overall_pass"])
        controlled = timing["modes"]["controlled_replay"]
        free = timing["modes"]["free_running_utf8_greedy"]
        self.assertAlmostEqual(controlled["end_to_end_reduction"], 0.24306843632330832)
        self.assertAlmostEqual(free["end_to_end_reduction"], 0.22840732225717642)
        self.assertEqual(controlled["positive_prompt_count"], 64)
        self.assertEqual(free["positive_prompt_count"], 64)
        self.assertTrue(all(controlled["passes"].values()))
        self.assertTrue(all(free["passes"].values()))
        self.assertTrue(
            all(row["argmax_exact"] == row["argmax_comparisons"]
                for row in timing["correctness"].values())
        )

        boundary = payload["claim_boundary"]
        self.assertTrue(boundary["calibration_only"])
        self.assertFalse(boundary["test_or_final_evidence_read"])
        self.assertFalse(boundary["publication_quality_or_efficiency_claimed"])
        self.assertFalse(boundary["static_geometry_novelty_claimed"])
        self.assertEqual(
            payload["raw_evidence"]["timing_artifact_sha256"],
            "b8430a8f113420823ad3cbd9665158c9b45fcc864ed4aff12229c16207c02446",
        )


if __name__ == "__main__":
    unittest.main()
