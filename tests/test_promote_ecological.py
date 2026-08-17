import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "promote_phase2_ecological.py"
SPEC = importlib.util.spec_from_file_location("promote_phase2_ecological", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load ecological promotion script")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EcologicalPromotionTests(unittest.TestCase):
    def test_privacy_walk_rejects_paths_and_content_keys(self) -> None:
        with self.assertRaises(ValueError):
            MODULE._walk({"path": "/private/file.md"})
        with self.assertRaises(ValueError):
            MODULE._walk({"nested": {"content": "private text"}})
        with self.assertRaises(ValueError):
            MODULE._walk({"nested": "../vault"})

    def test_privacy_walk_accepts_aggregate_counts(self) -> None:
        MODULE._walk(
            {
                "source_label": "private convenience sample",
                "valid_test_records": 10,
                "bpb": [2.1, 2.2],
            }
        )


if __name__ == "__main__":
    unittest.main()
