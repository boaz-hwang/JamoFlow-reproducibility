from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_inference_final_authorization_v2 import selection_lock_fixture


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_phase3.py"
SPEC = importlib.util.spec_from_file_location("run_phase3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase3RunnerEvidenceTests(unittest.TestCase):
    def test_selected_reference_attempt_requires_exact_live_marker(self) -> None:
        lock = selection_lock_fixture(broad_futile=False)
        policies = tuple(
            lock["decision"]["confirmation_plan"]["phase3_reference"][
                "policies"
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            os.chdir(root)
            try:
                artifact_root = Path("artifacts/phase3")
                completion = root / "results/completion.json"
                with (
                    mock.patch.object(
                        MODULE,
                        "PHASE3_REFERENCE_COMPLETION_PATH",
                        completion,
                    ),
                    mock.patch.object(MODULE, "_git_path_history", return_value=""),
                ):
                    active, completed = MODULE._start_selected_reference_attempt(
                        artifact_root=artifact_root,
                        selection_lock=lock,
                        selection_lock_artifact_sha256="a" * 64,
                        run_git_commit="b" * 40,
                        seeds=MODULE.CONFIRMATION_ONLY_SEEDS,
                        policies=policies,
                    )
                    self.assertTrue(active.is_file())
                    active.write_text("rotated", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "active attempt differs"):
                        MODULE._start_selected_reference_attempt(
                            artifact_root=artifact_root,
                            selection_lock=lock,
                            selection_lock_artifact_sha256="a" * 64,
                            run_git_commit="b" * 40,
                            seeds=MODULE.CONFIRMATION_ONLY_SEEDS,
                            policies=policies,
                        )
                    self.assertFalse(completed.exists())
            finally:
                os.chdir(previous)

    def test_parser_keeps_primary_and_selection_authorizations_separate(self) -> None:
        parser = MODULE.build_parser()
        selected = parser.parse_args(
            [
                "--seeds",
                "57721",
                "65537",
                "--policies",
                "spacebyte_spacelike",
                "--selection-lock",
                "results/phase3-inference-selection-v2/selection-lock.json",
            ]
        )
        self.assertIsNone(selected.authorization_summary)
        self.assertEqual(
            selected.selection_lock,
            Path("results/phase3-inference-selection-v2/selection-lock.json"),
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("load_selected_reference_authorization_v3", source)
        self.assertNotIn("PHASE3_PRIMARY_SUMMARY_PATH", source)

    def test_atomic_json_writer_never_overwrites_partial_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            partial = path.with_suffix(".json.part")
            partial.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forensic recovery"):
                MODULE._write_json(path, {"complete": True})
            self.assertEqual(partial.read_text(encoding="utf-8"), "partial")
            self.assertFalse(path.exists())

    def test_entropy_cache_provenance_binds_selected_reference_authorization(
        self,
    ) -> None:
        import numpy as np

        values = {split: np.zeros((1, 4), dtype=np.int64) for split in MODULE.SPLITS}
        boundaries = {
            split: np.zeros((1, 4), dtype=np.bool_) for split in MODULE.SPLITS
        }
        binding = {
            "kind": "selected_phase3_reference_training_evidence_v4",
            "schema_version": 4,
        }
        provenance = MODULE._threshold_cache_provenance(
            57721,
            "a" * 64,
            values,
            boundaries,
            evidence_binding=binding,
        )
        self.assertEqual(provenance["evidence_binding"], binding)


if __name__ == "__main__":
    unittest.main()
