import unittest

from scripts.summarize_stage2 import select_metric


class Stage2SummaryTests(unittest.TestCase):
    def test_metric_selection_requires_exact_identity(self) -> None:
        report = {
            "policy_metrics": [
                {"comparison_group": "fixed", "role": "rule", "value": 1},
                {"comparison_group": "fixed", "role": "candidate", "value": 2},
            ]
        }

        selected = select_metric(report, "fixed", "candidate")

        self.assertEqual(selected["value"], 2)
        with self.assertRaises(ValueError):
            select_metric(report, "missing", "candidate")


if __name__ == "__main__":
    unittest.main()
