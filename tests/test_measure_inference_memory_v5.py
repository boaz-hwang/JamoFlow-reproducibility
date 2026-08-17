from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "measure_inference_memory_v5.py"
SPEC = importlib.util.spec_from_file_location("measure_inference_memory_v5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MeasureInferenceMemoryV5Tests(unittest.TestCase):
    def test_worker_has_no_cli_and_one_fixed_unit_order(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("argparse", imports)
        order = MODULE._unit_order()
        self.assertEqual(len(order), 10)
        self.assertEqual(order[0], ("candidate", 1729))
        self.assertEqual(order[-1], ("reference", 65537))

    def test_receipt_prefix_cannot_skip_a_missing_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {
                "plan_sha256": "a" * 64,
                "timing_pair": {
                    "roles": {
                        "candidate": {
                            "model_identity_sha256": "b" * 64,
                            "parameter_bytes_float32": 400,
                        },
                        "reference": {
                            "model_identity_sha256": "c" * 64,
                            "parameter_bytes_float32": 400,
                        },
                    }
                },
            }
            authorization = {
                "models": [
                    {
                        "identity_sha256": "b" * 64,
                        "descriptor": {"requires_entropy_router": False},
                        "seeds": {
                            str(seed): {
                                "checkpoint": {"state_sha256": "d" * 64},
                                "auxiliary": {"kind": "none"},
                            }
                            for seed in (1729, 2718, 31415, 57721, 65537)
                        },
                    },
                    {
                        "identity_sha256": "c" * 64,
                        "descriptor": {"requires_entropy_router": False},
                        "seeds": {
                            str(seed): {
                                "checkpoint": {"state_sha256": "e" * 64},
                                "auxiliary": {"kind": "none"},
                            }
                            for seed in (1729, 2718, 31415, 57721, 65537)
                        },
                    },
                ]
            }
            with (
                mock.patch.object(MODULE, "MEMORY_ROOT", root),
                mock.patch.object(
                    MODULE, "_tracked_history_exists", return_value=False
                ),
            ):
                self.assertEqual(
                    MODULE._next_unit(plan=plan, authorization=authorization),
                    ("candidate", 1729),
                )
                later = MODULE._receipt_path("candidate", 2718)
                later.parent.mkdir(parents=True)
                later.write_text("{}")
                with self.assertRaisesRegex(ValueError, "complete prefix"):
                    MODULE._next_unit(plan=plan, authorization=authorization)

    def test_memory_process_inventory_fails_when_current_pid_was_not_parsed(
        self,
    ) -> None:
        snapshot = mock.Mock(returncode=0, stdout="999 1 unrelated\n")
        with mock.patch.object(MODULE.subprocess, "run", return_value=snapshot):
            with self.assertRaisesRegex(RuntimeError, "another neural/MPS"):
                MODULE._require_no_conflicting_neural_processes()

    def test_memory_requires_all_five_timing_receipts_before_first_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(MODULE, "SESSION_RECEIPT_ROOT", root / "receipts"),
                mock.patch.object(MODULE, "ARTIFACT_ROOT", root / "artifacts"),
                mock.patch.object(MODULE, "PLAN_PATH", root / "plan.json"),
                mock.patch.object(
                    MODULE,
                    "_tracked_head_identity",
                    return_value={
                        "git_commit": "a" * 40,
                        "path": "plan.json",
                        "sha256": "b" * 64,
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "five timing sessions"):
                    MODULE._require_timing_campaign_complete(
                        plan={"plan_sha256": "c" * 64},
                        current_commit="d" * 40,
                    )


if __name__ == "__main__":
    unittest.main()
