from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from foldable_jamo_residual_core import RESIDUAL_SLOT_COUNT
from foldable_jamo_residual_protocol import PLAN_PATH as PARENT_PLAN_PATH
from foldable_jamo_residual_protocol import read_json
from foldable_multihash_update_audit_core import (
    select_update_matched_control,
    update_geometry,
)
from run_foldable_multihash_update_audit import (
    _bucket_alignment,
    _sequence_view,
    _validate_parent_plan_for_historical_replay,
)
from seal_foldable_multihash_update_audit_plan import IMPLEMENTATION_PATHS

ROOT = Path(__file__).resolve().parents[1]


class FoldableMultihashUpdateAuditTest(unittest.TestCase):
    def test_bucket_alignment_excludes_and_reports_zero_direct_rows(self) -> None:
        rows = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        tables = np.ones((RESIDUAL_SLOT_COUNT, 2, 2), dtype=np.float32)
        assignments = np.zeros((4, RESIDUAL_SLOT_COUNT), dtype=np.int64)
        result = _bucket_alignment(rows, tables, assignments)
        self.assertEqual(result["total_row_count"], 4)
        self.assertEqual(result["nonzero_row_gradient_count"], 3)
        self.assertEqual(result["zero_row_gradient_count"], 1)
        self.assertEqual(result["aligned_pair_count"], 3 * RESIDUAL_SLOT_COUNT)
        self.assertTrue(
            all(row["aligned_row_count"] == 3 for row in result["per_slot"])
        )

    def test_bucket_alignment_requires_one_nonzero_direct_row(self) -> None:
        rows = np.zeros((4, 2), dtype=np.float32)
        tables = np.ones((RESIDUAL_SLOT_COUNT, 2, 2), dtype=np.float32)
        assignments = np.zeros((4, RESIDUAL_SLOT_COUNT), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "no nonzero row gradient"):
            _bucket_alignment(rows, tables, assignments)

    def test_flat_token_memmap_is_viewed_as_complete_sequences(self) -> None:
        flat = np.arange(4 * 512, dtype=np.int64)
        viewed = _sequence_view(flat, 4)
        self.assertEqual(viewed.shape, (4, 512))
        self.assertTrue(np.array_equal(viewed[2], flat[1024:1536]))

    def test_historical_parent_allows_only_the_two_result_doc_amendments(self) -> None:
        _validate_parent_plan_for_historical_replay(read_json(PARENT_PLAN_PATH))

    def test_implementation_manifest_is_unique_and_quality_blind(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        for relative in IMPLEMENTATION_PATHS:
            self.assertTrue((ROOT / relative).is_file(), relative)
        for relative in (
            "scripts/seal_foldable_multihash_update_audit_plan.py",
            "scripts/run_foldable_multihash_update_audit.py",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            for forbidden in ("contiguous_bpb", "document_bpb", "test_nll", "latency"):
                self.assertNotIn(forbidden, source)

    def test_projection_and_orthogonal_component_are_reconstructed(self) -> None:
        dense = np.asarray(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, -1.0]], dtype=np.float32
        )
        orthogonal = np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [-1.0, 1.0], [0.2, 0.4]], dtype=np.float32
        )
        projection = float(
            (orthogonal.astype(np.float64) * dense).sum()
            / np.square(dense.astype(np.float64)).sum()
        )
        orthogonal = orthogonal - np.float32(projection) * dense
        candidate = np.float32(4.0) * dense + orthogonal
        result = update_geometry(
            dense,
            candidate.astype(np.float32),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        self.assertAlmostEqual(result["projection_multiplier"], 4.0, places=6)
        self.assertGreater(result["orthogonal_fraction_of_candidate"], 0.0)
        self.assertEqual(len(result["exposure_strata"]), 4)

    def test_control_uses_only_input_and_output_projection(self) -> None:
        dense = np.asarray(
            [[1.0, 2.0], [2.0, 1.0], [1.0, -1.0], [-2.0, 1.0]],
            dtype=np.float32,
        )
        exposure = np.asarray([3, 4, 5, 6], dtype=np.int64)
        geometry = {
            "input": update_geometry(dense, dense * np.float32(3.0), exposure),
            "output": update_geometry(dense, dense * np.float32(5.0), exposure),
        }
        control = select_update_matched_control(geometry)
        self.assertEqual(control["input_multiplier"], 3.0)
        self.assertEqual(control["output_multiplier"], 5.0)
        self.assertFalse(control["quality_metric_used"])

    def test_out_of_range_multiplier_fails_closed(self) -> None:
        dense = np.asarray(
            [[1.0, 2.0], [2.0, 1.0], [1.0, -1.0], [-2.0, 1.0]],
            dtype=np.float32,
        )
        exposure = np.asarray([3, 4, 5, 6], dtype=np.int64)
        geometry = update_geometry(dense, dense * np.float32(17.0), exposure)
        with self.assertRaisesRegex(ValueError, "safety range"):
            select_update_matched_control({"input": geometry, "output": geometry})


if __name__ == "__main__":
    unittest.main()
