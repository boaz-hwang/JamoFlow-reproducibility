import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "select_phase3_inference_comparator.py"
)
SPEC = importlib.util.spec_from_file_location(
    "select_phase3_inference_comparator",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase3InferenceComparatorSelectionTests(unittest.TestCase):
    def test_reference_uses_lowest_mean_quality_without_latency(self) -> None:
        order = ("fixed", "learned", "same_rate")
        selected, details = MODULE.select_reference(
            {"fixed": 2.0, "learned": 1.98, "same_rate": 1.99},
            order,
        )
        self.assertEqual(selected, "learned")
        self.assertNotIn("latency", details["criterion"])
        self.assertIn("calibration", details["criterion"])

    def test_reference_exact_tie_uses_preregistered_order(self) -> None:
        order = ("fixed", "learned", "same_rate")
        selected, details = MODULE.select_reference(
            {policy: 2.0 for policy in order},
            order,
        )
        self.assertEqual(selected, "fixed")
        self.assertEqual(details["candidate_order"], list(order))

    def test_reference_selection_rejects_missing_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "one finite value"):
            MODULE.select_reference(
                {"fixed": 2.0},
                ("fixed", "learned"),
            )


if __name__ == "__main__":
    unittest.main()
