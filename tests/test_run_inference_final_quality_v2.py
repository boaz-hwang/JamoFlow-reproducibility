from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.test_inference_final_authorization_v2 import (
    InferenceFinalAuthorizationV2Tests,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_inference_final_quality_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_inference_final_quality_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RunInferenceFinalQualityV2Tests(unittest.TestCase):
    def test_runner_has_fixed_paths_no_cli_and_no_historical_or_timing_input(self) -> None:
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
        for value in strings:
            self.assertNotIn("-test-nll", value)
            self.assertNotIn("latency", value.lower())
            self.assertNotIn("timing.json", value)
        self.assertEqual(
            MODULE.EVIDENCE_PATH.as_posix(),
            "results/phase3-inference-final-v2/evidence-manifest.json",
        )

    def test_unit_publish_is_no_clobber_and_removes_only_exact_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nll = root / "unit.npz"
            receipt = root / "unit.json"
            MODULE._publish_unit_pair(
                nll_path=nll,
                nll_bytes=b"nll",
                receipt_path=receipt,
                receipt_bytes=b"receipt",
            )
            self.assertEqual(nll.read_bytes(), b"nll")
            self.assertEqual(receipt.read_bytes(), b"receipt")
            self.assertFalse((root / "unit.npz.part").exists())
            self.assertFalse((root / "unit.json.part").exists())
            with self.assertRaises(FileExistsError):
                MODULE._publish_unit_pair(
                    nll_path=nll,
                    nll_bytes=b"other",
                    receipt_path=receipt,
                    receipt_bytes=b"other",
                )

    def test_partial_or_staged_unit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nll = root / "unit.npz"
            receipt = root / "unit.json"
            nll.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "partial final unit"):
                MODULE._validate_completed_unit(
                    receipt_path=receipt,
                    nll_path=nll,
                    authorization={},
                    selection_lock={},
                    session_plan={},
                    matrix_sha256="a" * 64,
                    auxiliary_execution={"kind": "none"},
                )
            nll.unlink()
            (root / "unit.npz.part").write_bytes(b"stage")
            with self.assertRaisesRegex(ValueError, "forensic review"):
                MODULE._validate_completed_unit(
                    receipt_path=receipt,
                    nll_path=nll,
                    authorization={},
                    selection_lock={},
                    session_plan={},
                    matrix_sha256="a" * 64,
                    auxiliary_execution={"kind": "none"},
                )

    def test_completed_unit_is_checked_only_after_model_and_matrix(self) -> None:
        lock, authorization = InferenceFinalAuthorizationV2Tests()._authorization()
        unit_index, role, seed = MODULE.authorized_unit_order(authorization)[0]
        order: list[str] = []

        def load_model(*args, **kwargs):
            del args, kwargs
            order.append("model")
            return object()

        def matrix(*args, **kwargs):
            del args, kwargs
            order.append("matrix")
            import numpy as np

            return np.ones((62_500, 65), dtype=np.int16), {"kind": "none"}

        def completed(*args, **kwargs):
            del args, kwargs
            order.append("resume")
            return ({"receipt_sha256": "a" * 64}, {"path": "x", "sha256": "b" * 64})

        with (
            mock.patch.object(MODULE, "_load_main_model", side_effect=load_model),
            mock.patch.object(MODULE, "_matrix_for_unit", side_effect=matrix),
            mock.patch.object(
                MODULE,
                "_validate_completed_unit",
                side_effect=completed,
            ),
            mock.patch.object(MODULE, "_release"),
            mock.patch.object(MODULE, "_array_sha256", return_value="c" * 64),
        ):
            MODULE._evaluate_unit(
                unit_index=unit_index,
                artifact_role=role,
                seed=seed,
                authorization=authorization,
                authorization_artifact_sha256="d" * 64,
                selection_lock=lock,
                session_plan={},
                inputs=object(),
                boundaries=object(),
                structural_matrices={},
            )
        self.assertEqual(order, ["model", "matrix", "resume"])

    def test_active_session_can_resume_only_with_exact_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / ".active"
            unit = root / "unit.npz"
            plan = {
                "evaluator_git_commit": "a" * 40,
                "session_id": "b" * 64,
                "session_plan_sha256": "c" * 64,
            }
            with (
                mock.patch.object(MODULE, "ACTIVE_SENTINEL", sentinel),
                mock.patch.object(
                    MODULE,
                    "_unit_artifact_paths",
                    return_value={unit},
                ),
            ):
                MODULE._start_or_resume_active_session(plan, {})
                first = sentinel.read_bytes()
                MODULE._start_or_resume_active_session(plan, {})
                self.assertEqual(sentinel.read_bytes(), first)
                sentinel.write_bytes(b"different-session")
                with self.assertRaisesRegex(ValueError, "sentinel differs"):
                    MODULE._start_or_resume_active_session(plan, {})
                sentinel.unlink()
                unit.write_bytes(b"orphan")
                with self.assertRaisesRegex(ValueError, "without their active"):
                    MODULE._start_or_resume_active_session(plan, {})

    def test_artifact_namespace_rejects_unknown_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / "alternate-result.npz"
            unknown.write_bytes(b"unexpected")
            with (
                mock.patch.object(MODULE, "FINAL_ARTIFACT_ROOT", root),
                mock.patch.object(MODULE, "SESSION_PLAN_PATH", root / "session.json"),
                mock.patch.object(MODULE, "ACTIVE_SENTINEL", root / ".active"),
                mock.patch.object(MODULE, "_unit_artifact_paths", return_value=set()),
            ):
                with self.assertRaisesRegex(ValueError, "namespace"):
                    MODULE._validate_artifact_namespace({})

    def test_session_file_lock_excludes_a_second_live_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session-plan.json"
            session.write_bytes(b"sealed-plan")
            with (
                mock.patch.object(MODULE, "FINAL_ARTIFACT_ROOT", root),
                mock.patch.object(MODULE, "SESSION_PLAN_PATH", session),
            ):
                with MODULE._exclusive_session_process_lock():
                    with self.assertRaisesRegex(RuntimeError, "another final"):
                        with MODULE._exclusive_session_process_lock():
                            self.fail("a second live evaluator acquired the lock")
                with MODULE._exclusive_session_process_lock():
                    pass

    def test_artifact_parent_or_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside = base / "outside"
            outside.mkdir()
            root = base / "artifacts"
            root.mkdir()
            (root / "seed-1729").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                MODULE._require_unsymlinked_path_within_root(
                    root / "seed-1729" / "unit.npz",
                    root,
                )
            linked_root = base / "linked-root"
            linked_root.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                MODULE._require_unsymlinked_path_within_root(
                    linked_root / "unit.npz",
                    linked_root,
                )

    def test_deleted_tracked_final_evidence_blocks_a_new_session(self) -> None:
        missing = Path("results/nonexistent-final-evidence-for-test.json")
        with (
            mock.patch.object(MODULE, "_git_path_history", return_value=("a" * 40,)),
            self.assertRaisesRegex(ValueError, "published then deleted"),
        ):
            MODULE._require_not_deleted_tracked_artifact(missing)


if __name__ == "__main__":
    unittest.main()
