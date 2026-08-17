from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from foldable_jamo_residual_core import (
    RESIDUAL_ROLES,
    expected_parameter_counts,
    role_definition,
)
from foldable_jamo_residual_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    selection_rule,
)
from run_foldable_jamo_residual import _load_nll, _optimizer, _paths
from vocabulary_transfer_baseline_core import build_target_graph
from foldable_jamo_residual_core import install_foldable_residual
from foldable_jamo_residual_core import build_residual_assignment
from vocabulary_transfer_probe_core import TARGET_VOCABULARY_SIZE


class FoldableJamoResidualTest(unittest.TestCase):
    def test_role_namespaces_and_parameter_counts_are_architecture_matched(self) -> None:
        for role in RESIDUAL_ROLES:
            report, checkpoints, nlls, folded = _paths(role)
            self.assertEqual(set(checkpoints), {0, 32, 128, 512})
            self.assertEqual(set(nlls), set(checkpoints))
            self.assertTrue(str(report).endswith(f"workers/{role}.json"))
            self.assertTrue(str(folded).endswith(f"{role}-step-0512.pt"))
            counts = expected_parameter_counts(role)
            self.assertGreater(counts["training_total"], counts["deployed"])
            self.assertEqual(
                counts["training_total"],
                counts["deployed"] + counts["training_only_residual"],
            )
        self.assertEqual(
            expected_parameter_counts("untied_jamo"),
            expected_parameter_counts("untied_shuffled_jamo"),
        )
        self.assertEqual(
            expected_parameter_counts("tied_jamo"),
            expected_parameter_counts("tied_generic_surface"),
        )

    def test_optimizer_opens_dense_body_rows_and_same_cost_residual(self) -> None:
        pieces = tuple(f"x{index}".encode() for index in range(TARGET_VOCABULARY_SIZE))
        exposures = np.arange(TARGET_VOCABULARY_SIZE, dtype=np.int64)
        for role in ("untied_jamo", "tied_jamo"):
            base = build_target_graph(role_definition(role)["base_initializer_role"])
            assignment = build_residual_assignment(pieces, exposures, kind="jamo")
            model = install_foldable_residual(
                base, assignment, tied=role.startswith("tied_")
            )
            optimizer = _optimizer(model)
            self.assertEqual(len(optimizer.param_groups), 3)
            self.assertEqual(optimizer.param_groups[-1]["schedule_kind"], "head")
            self.assertEqual(
                len(optimizer.param_groups[-1]["params"]),
                2 if role.startswith("tied_") else 4,
            )
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_nll_archive_schema_distinguishes_final_document_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intermediate = root / "intermediate.npz"
            final = root / "final.npz"
            np.savez_compressed(
                intermediate,
                contiguous_nll_nats=np.ones(3, dtype=np.float32),
                contiguous_raw_target_bytes=np.ones(3, dtype=np.int64),
            )
            np.savez_compressed(
                final,
                contiguous_nll_nats=np.ones(3, dtype=np.float32),
                contiguous_raw_target_bytes=np.ones(3, dtype=np.int64),
                document_nll_nats=np.ones(2, dtype=np.float64),
                document_raw_bytes=np.ones(2, dtype=np.int64),
            )
            self.assertEqual(len(_load_nll(intermediate, final=False)), 2)
            self.assertEqual(len(_load_nll(final, final=True)), 4)
            with self.assertRaisesRegex(RuntimeError, "key set"):
                _load_nll(intermediate, final=True)

    def test_selection_rule_has_no_result_dependent_fallback(self) -> None:
        rule = selection_rule()
        self.assertIsNone(rule["threshold_or_role_fallback"])
        self.assertEqual(
            rule["minimum_jamo_advantage_over_generic_and_shuffle_bpb"], 0.002
        )
        self.assertTrue(rule["requires_contiguous_and_document_point_advantage"])
        self.assertTrue(rule["document_bootstrap_upper_must_be_nonpositive"])

    def test_implementation_manifest_is_unique_and_complete(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        self.assertTrue(all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS))

    def test_plan_sealer_has_no_role_seed_or_threshold_cli(self) -> None:
        source = (ROOT / "scripts/seal_foldable_jamo_residual_plan.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                for call in calls
            )
        )
        self.assertNotIn("--threshold", source)
        self.assertNotIn("--role", source)
        self.assertNotIn("--seed", source)


if __name__ == "__main__":
    unittest.main()
