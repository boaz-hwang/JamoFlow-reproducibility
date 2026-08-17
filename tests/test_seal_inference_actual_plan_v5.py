from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "seal_inference_actual_plan_v5.py"
SPEC = importlib.util.spec_from_file_location("seal_inference_actual_plan_v5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SealInferenceActualPlanV5Tests(unittest.TestCase):
    def test_plan_sealer_has_fixed_paths_and_no_cli_or_latency_input(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn("argparse", imports)
        self.assertEqual(
            {value for value in strings if "latency" in value.lower()},
            {"latency_metrics_inspected"},
        )
        self.assertEqual(
            MODULE.PLAN_PATH.as_posix(),
            "results/phase3-inference-actual-v5r3/plan.json",
        )

    def test_case_artifact_is_deterministic_and_tamper_detected(self) -> None:
        prompts = np.arange(72 * 128, dtype=np.uint8).reshape(72, 128)
        continuations = np.flip(prompts, axis=1).copy()
        first = MODULE._npz_bytes(prompts, continuations)
        second = MODULE._npz_bytes(prompts, continuations)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.npz"
            path.write_bytes(first)
            self.assertEqual(
                MODULE._validate_case_artifact(
                    path,
                    prompts=prompts,
                    continuations=continuations,
                ),
                __import__("hashlib").sha256(first).hexdigest(),
            )
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "cases differ"):
                MODULE._validate_case_artifact(
                    path,
                    prompts=prompts,
                    continuations=continuations,
                )

    def test_v5_implementation_cannot_follow_final_evaluator(self) -> None:
        evaluator_commit = "a" * 40
        with mock.patch.object(
            MODULE,
            "_tracked_head_identity",
            return_value={
                "git_commit": "b" * 40,
                "path": "fixture",
                "sha256": "b" * 64,
            },
        ), mock.patch.object(
            MODULE, "_require_ancestor", side_effect=ValueError("order")
        ):
            with self.assertRaisesRegex(ValueError, "order"):
                MODULE._require_implementation_not_after_evaluator(
                    evaluator_commit,
                    erratum={
                        "allowed_post_evaluator_files": {
                            path: {}
                            for path in MODULE.POST_FINAL_CORRECTNESS_REVISION_FILES
                        }
                    },
                )


if __name__ == "__main__":
    unittest.main()
