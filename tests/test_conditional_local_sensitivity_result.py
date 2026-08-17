import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/conditional-local-frozen-sensitivity-v1/summary.json"
EXPECTED_ARTIFACT_SHA256 = (
    "5f48ab269d44de01eef1205636784daaee88d381e81b7854b3fe8735e4c552fb"
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


class ConditionalLocalSensitivityResultTest(unittest.TestCase):
    def test_all_frozen_candidates_fail_without_hangul_claim(self):
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(RESULT.read_bytes()).hexdigest(),
            EXPECTED_ARTIFACT_SHA256,
        )
        claimed = payload.pop("summary_sha256")
        self.assertEqual(hashlib.sha256(_canonical_bytes(payload)).hexdigest(), claimed)

        self.assertEqual(
            payload["status"], "conditional_branch_not_advanced_by_frozen_screen"
        )
        aggregate = payload["aggregate"]
        self.assertIsNone(aggregate["selection"]["selected_pair"])
        self.assertFalse(
            aggregate["selection"]["actual_runtime_prototype_authorized"]
        )
        self.assertTrue(all(not row["overall_pass"] for row in aggregate["rows"].values()))
        self.assertTrue(
            all(not row["passes"]["mean_risk_margin"] for row in aggregate["rows"].values())
        )
        self.assertTrue(
            all(
                not row["passes"]["document_upper_risk_margin"]
                for row in aggregate["rows"].values()
            )
        )

        least_damaging = aggregate["rows"]["hangul_prefix__decoder__second_mlp"]
        self.assertAlmostEqual(
            least_damaging["mean_difference_bpb"], 0.19883165396901972
        )
        self.assertAlmostEqual(
            least_damaging["document_bootstrap"]["one_sided_95_upper_bpb"],
            0.19996730037160249,
        )
        self.assertGreater(
            least_damaging["mean_difference_bpb"],
            aggregate["thresholds"]["risk_margin_bpb"],
        )

        generic = payload["route_rates"]["utf8_incomplete"]
        hangul = payload["route_rates"]["hangul_prefix"]
        self.assertAlmostEqual(generic, 0.583054875)
        self.assertAlmostEqual(hangul, 0.575361125)
        self.assertAlmostEqual(hangul / generic, 0.9868044152790936)

        interpretation = aggregate["interpretation"]
        self.assertFalse(interpretation["hangul_specific_effect_identified"])
        self.assertFalse(
            interpretation["trained_conditional_model_falsified_on_failure"]
        )
        boundary = payload["claim_boundary"]
        self.assertFalse(boundary["hangul_specific_effect_claimed"])
        self.assertFalse(boundary["publication_quality_or_efficiency_claimed"])
        self.assertFalse(boundary["this_stream_reused_as_confirmatory_trained_evaluation"])


if __name__ == "__main__":
    unittest.main()
