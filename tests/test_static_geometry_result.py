import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/static-geometry-preflight-v1/summary.json"
EXPECTED_ARTIFACT_SHA256 = (
    "0735291749e1305835a9dd09a4a22293e240cf9cc9f3e656fe1a69a001b0c352"
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


class StaticGeometryResultTest(unittest.TestCase):
    def test_only_the_predeclared_second_geometry_is_authorized(self):
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(RESULT.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )
        claimed = payload.pop("summary_sha256")
        self.assertEqual(hashlib.sha256(_canonical_bytes(payload)).hexdigest(), claimed)

        aggregate = payload["aggregate"]
        self.assertEqual(payload["status"], "one_seed_static_control_authorized")
        self.assertEqual(
            aggregate["selection"]["selected_candidate"],
            "thin160_e1_d1_g384x9",
        )
        self.assertTrue(aggregate["selection"]["one_seed_training_authorized"])
        selected = aggregate["rows"]["thin160_e1_d1_g384x9"]
        self.assertTrue(selected["overall_pass"])
        self.assertAlmostEqual(selected["end_to_end_reduction"], 0.24416677210329818)
        self.assertAlmostEqual(
            selected["prompt_bootstrap_95_interval"]["lower"],
            0.19201726937498576,
        )
        self.assertEqual(selected["positive_prompt_count"], 32)
        self.assertTrue(all(selected["passes"].values()))

        conservative = aggregate["rows"]["thin128_e1_d2_g384x9"]
        aggressive = aggregate["rows"]["thin128_e1_d1_g384x9"]
        self.assertFalse(conservative["overall_pass"])
        self.assertFalse(aggressive["overall_pass"])
        self.assertFalse(aggressive["passes"]["point_reduction"])
        self.assertGreater(
            aggressive["analytical_flop_reduction"],
            selected["analytical_flop_reduction"],
        )
        self.assertLess(
            aggressive["end_to_end_reduction"],
            selected["end_to_end_reduction"],
        )

        self.assertFalse(payload["claim_boundary"]["quality_or_publication_efficiency_claimed"])
        self.assertFalse(payload["claim_boundary"]["static_geometry_is_novelty_claimed"])
        self.assertEqual(
            payload["raw_evidence"]["artifact_sha256"],
            "631c4bbc2a70ae26bb19bf2f8b3509e374dfa80af5789a4af531c63b77d7a2b1",
        )


if __name__ == "__main__":
    unittest.main()
