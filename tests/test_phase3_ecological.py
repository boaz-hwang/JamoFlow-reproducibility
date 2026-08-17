import importlib.util
from pathlib import Path
import unittest

import numpy as np


RUN_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_phase3_ecological.py"
RUN_SPEC = importlib.util.spec_from_file_location("run_phase3_ecological", RUN_SCRIPT)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
RUN_MODULE = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN_MODULE)

PROMOTE_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "promote_phase3_ecological.py"
)
PROMOTE_SPEC = importlib.util.spec_from_file_location(
    "promote_phase3_ecological", PROMOTE_SCRIPT
)
assert PROMOTE_SPEC is not None and PROMOTE_SPEC.loader is not None
PROMOTE_MODULE = importlib.util.module_from_spec(PROMOTE_SPEC)
PROMOTE_SPEC.loader.exec_module(PROMOTE_MODULE)


class Phase3EcologicalTests(unittest.TestCase):
    def test_bootstrap_summary_contains_no_raw_replicates(self) -> None:
        summary = RUN_MODULE._bootstrap_summary(
            [
                np.asarray([1.0, -1.0, 0.5]),
                np.asarray([0.0, -0.5, 0.25]),
            ],
            repetitions=100,
            seed=1,
        )
        self.assertEqual(summary["repetitions"], 100)
        self.assertEqual(
            set(summary),
            {
                "repetitions",
                "seed",
                "resampling_design",
                "mean",
                "median",
                "lower",
                "upper",
            },
        )

    def test_privacy_walk_rejects_content_hashes_and_paths(self) -> None:
        with self.assertRaises(ValueError):
            PROMOTE_MODULE._walk({"content": "private"})
        with self.assertRaises(ValueError):
            PROMOTE_MODULE._walk({"sha256": "abc"})
        with self.assertRaises(ValueError):
            PROMOTE_MODULE._walk({"nested": "../vault"})

    def test_privacy_walk_accepts_only_aggregate_values(self) -> None:
        PROMOTE_MODULE._walk(
            {
                "source_label": "private convenience sample",
                "valid_test_records": 10,
                "bpb": {"mean": 2.1, "values": [2.0, 2.2]},
            }
        )


if __name__ == "__main__":
    unittest.main()
