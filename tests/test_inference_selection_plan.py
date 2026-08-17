from __future__ import annotations

from copy import deepcopy
import unittest

from jamoflow.inference_selection_plan import (
    build_selection_plan_v2,
    validate_selection_plan_v2,
)


def plan_fixture() -> dict:
    return build_selection_plan_v2(
        plan_git_commit="a" * 40,
        final_test_manifest_sha256="1" * 64,
        final_test_seal_sha256="2" * 64,
        final_test_payload_sha256="3" * 64,
        phase3_all_initial_summary_sha256="4" * 64,
        phase3_primary_summary_sha256="5" * 64,
        source_artifact_sha256="6" * 64,
        source_integrity_artifact_sha256="7" * 64,
        calibration_stream_sha256="8" * 64,
        calibration_sequence_count=15_625,
    )


class InferenceSelectionPlanTests(unittest.TestCase):
    def test_plan_is_result_blind_and_canonical(self) -> None:
        plan = plan_fixture()
        validate_selection_plan_v2(plan)
        encoded = repr(plan).lower()
        self.assertNotIn("test_bpb", encoded)
        self.assertNotIn("latency_ms", encoded)
        self.assertFalse(plan["selection_rules"]["final_test_input"])
        self.assertFalse(
            plan["historical_screening"]["all_initial_summary"][
                "authorizes_selection"
            ]
        )
        self.assertEqual(
            plan["historical_screening"]["primary_summary"]["path"],
            "results/phase3-primary-five-seed/summary.json",
        )

    def test_plan_rejects_rule_path_hash_and_unknown_field_tampering(self) -> None:
        for mutate in (
            lambda value: value["selection_rules"].__setitem__(
                "rate_margin_bpb", 0.02
            ),
            lambda value: value["execution_paths"].__setitem__(
                "selection_lock", "alternate.json"
            ),
            lambda value: value["final_test"].__setitem__(
                "evaluated_at_plan", True
            ),
            lambda value: value.__setitem__("test_metric", 1.0),
        ):
            plan = deepcopy(plan_fixture())
            mutate(plan)
            with self.assertRaises(ValueError):
                validate_selection_plan_v2(plan)

    def test_plan_requires_positive_calibration_count_and_valid_hashes(self) -> None:
        with self.assertRaisesRegex(ValueError, "count must be positive"):
            build_selection_plan_v2(
                plan_git_commit="a" * 40,
                final_test_manifest_sha256="1" * 64,
                final_test_seal_sha256="2" * 64,
                final_test_payload_sha256="3" * 64,
                phase3_all_initial_summary_sha256="4" * 64,
                phase3_primary_summary_sha256="5" * 64,
                source_artifact_sha256="6" * 64,
                source_integrity_artifact_sha256="7" * 64,
                calibration_stream_sha256="8" * 64,
                calibration_sequence_count=0,
            )


if __name__ == "__main__":
    unittest.main()
