from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from jamoflow.inference_final_quality_v2 import resolve_final_evaluation_roles
from tests.test_inference_final_authorization_v2 import (
    model_fixture,
    selection_lock_fixture,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "seal_inference_final_quality_lock_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seal_inference_final_quality_lock_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SealInferenceFinalQualityLockV2Tests(unittest.TestCase):
    def test_lock_sealer_has_one_output_and_no_timing_or_cli_input(self) -> None:
        self.assertEqual(
            MODULE.OUTPUT_PATH.as_posix(),
            "results/phase3-inference-final-v2/summary.json",
        )
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
            self.assertNotIn("latency", value.lower())
            self.assertNotIn("timing.json", value)
            self.assertNotIn("-test-nll", value)

    def test_structural_receipt_matrix_is_independently_reconstructed(self) -> None:
        lock = selection_lock_fixture()
        model = model_fixture(resolve_final_evaluation_roles(lock)["unique_models"][0])
        descriptor = model["descriptor"]
        matrix = np.zeros(
            (62_500, int(descriptor["patch_count"]) + 1),
            dtype=np.uint16,
        )
        matrix[:, 0] = 1
        matrix[:, 1] = 512
        receipt = {
            "artifact_role": model["artifact_role"],
            "auxiliary_execution": {"kind": "none"},
            "nll": {"array_sha256": "1" * 64},
            "patch_matrix_sha256": "0" * 64,
            "receipt_sha256": "2" * 64,
            "seed": 1729,
        }
        with (
            mock.patch.object(
                MODULE,
                "_load_verified_main_checkpoint",
                return_value=object(),
            ),
            mock.patch.object(MODULE, "_release"),
        ):
            with self.assertRaisesRegex(ValueError, "patch matrix/router"):
                MODULE._recompute_verified_nll_arrays(
                    authorization={"models": [model]},
                    evidence={"receipts": [receipt]},
                    inputs=np.zeros((1, 1), dtype=np.uint8),
                    boundaries=np.zeros((1, 1), dtype=np.uint8),
                    structural_matrices={descriptor["policy"]: matrix},
                    stored_arrays={
                        receipt["receipt_sha256"]: np.zeros(
                            62_500, dtype=np.float32
                        )
                    },
                )

    def test_checkpoint_file_hash_is_checked_before_state_load(self) -> None:
        lock = selection_lock_fixture()
        model = model_fixture(resolve_final_evaluation_roles(lock)["unique_models"][0])
        with (
            mock.patch.object(MODULE, "hash_file", return_value="0" * 64),
            mock.patch.object(MODULE, "build_main_model") as build_model,
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint artifact"):
                MODULE._load_verified_main_checkpoint(model, 1729)
        build_model.assert_not_called()

    def test_quality_sealer_rejects_authorized_implementation_drift(self) -> None:
        authorization = {
            "implementation_sha256": {
                "src/jamoflow/inference_final_quality_v2.py": "a" * 64,
            }
        }
        with mock.patch.object(
            MODULE,
            "_tracked_identity",
            return_value={
                "git_commit": "b" * 40,
                "path": "src/jamoflow/inference_final_quality_v2.py",
                "sha256": "c" * 64,
            },
        ):
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                MODULE._verify_implementation(authorization)

    def test_deleted_quality_lock_history_blocks_reseal(self) -> None:
        missing = Path("results/nonexistent-final-quality-lock-for-test.json")
        with mock.patch.object(
            MODULE, "_git_path_history", return_value=("a" * 40,)
        ):
            with self.assertRaisesRegex(ValueError, "was deleted"):
                MODULE._require_never_published(missing)

    def test_arbitrary_stored_nll_fails_independent_model_forward(self) -> None:
        lock = selection_lock_fixture()
        model = model_fixture(resolve_final_evaluation_roles(lock)["unique_models"][0])
        descriptor = model["descriptor"]
        matrix = np.zeros(
            (62_500, int(descriptor["patch_count"]) + 1),
            dtype=np.uint16,
        )
        matrix[:, 0] = 1
        matrix[:, 1] = 512
        recomputed = np.full(62_500, 3.0, dtype=np.float32)
        stored = np.full(62_500, 4.0, dtype=np.float32)
        receipt_sha = "2" * 64
        receipt = {
            "artifact_role": model["artifact_role"],
            "auxiliary_execution": {"kind": "none"},
            "nll": {"array_sha256": MODULE._array_sha256(stored)},
            "patch_matrix_sha256": MODULE._array_sha256(matrix),
            "receipt_sha256": receipt_sha,
            "seed": 1729,
        }
        with (
            mock.patch.object(
                MODULE,
                "_load_verified_main_checkpoint",
                return_value=object(),
            ),
            mock.patch.object(
                MODULE,
                "evaluate_main_model",
                return_value=(0.0, recomputed),
            ),
            mock.patch.object(MODULE, "_release"),
        ):
            with self.assertRaisesRegex(ValueError, "independent model forward"):
                MODULE._recompute_verified_nll_arrays(
                    authorization={"models": [model]},
                    evidence={"receipts": [receipt]},
                    inputs=np.zeros((62_500, 512), dtype=np.uint8),
                    boundaries=np.zeros((62_500, 512), dtype=np.uint8),
                    structural_matrices={descriptor["policy"]: matrix},
                    stored_arrays={receipt_sha: stored},
                )

    def test_verifier_rejects_float64_before_any_cast(self) -> None:
        lock = selection_lock_fixture()
        model = model_fixture(resolve_final_evaluation_roles(lock)["unique_models"][0])
        descriptor = model["descriptor"]
        matrix = np.zeros(
            (62_500, int(descriptor["patch_count"]) + 1),
            dtype=np.uint16,
        )
        matrix[:, 0] = 1
        matrix[:, 1] = 512
        stored = np.zeros(62_500, dtype=np.float32)
        receipt_sha = "2" * 64
        receipt = {
            "artifact_role": model["artifact_role"],
            "auxiliary_execution": {"kind": "none"},
            "nll": {"array_sha256": MODULE._array_sha256(stored)},
            "patch_matrix_sha256": MODULE._array_sha256(matrix),
            "receipt_sha256": receipt_sha,
            "seed": 1729,
        }
        with (
            mock.patch.object(
                MODULE,
                "_load_verified_main_checkpoint",
                return_value=object(),
            ),
            mock.patch.object(
                MODULE,
                "evaluate_main_model",
                return_value=(0.0, stored.astype(np.float64)),
            ),
            mock.patch.object(MODULE, "_release"),
        ):
            with self.assertRaisesRegex(ValueError, "dtype differs"):
                MODULE._recompute_verified_nll_arrays(
                    authorization={"models": [model]},
                    evidence={"receipts": [receipt]},
                    inputs=np.zeros((62_500, 512), dtype=np.uint8),
                    boundaries=np.zeros((62_500, 512), dtype=np.uint8),
                    structural_matrices={descriptor["policy"]: matrix},
                    stored_arrays={receipt_sha: stored},
                )

    def test_entropy_execution_rebuilds_router_score_and_matrix_hashes(self) -> None:
        lock = selection_lock_fixture(
            broad_futile=False,
            broad_policy="entropy_threshold_codepoint",
        )
        broad = resolve_final_evaluation_roles(lock)["unique_models"][-1]
        model = model_fixture(broad)
        auxiliary = model["seeds"]["1729"]["auxiliary"]
        hashes_by_path = {
            auxiliary[path_key]: auxiliary[hash_key]
            for path_key, hash_key in (
                ("router_checkpoint_path", "router_checkpoint_artifact_sha256"),
                ("router_report_path", "router_report_artifact_sha256"),
                ("threshold_cache_path", "threshold_cache_artifact_sha256"),
                (
                    "threshold_diagnostics_path",
                    "threshold_diagnostics_artifact_sha256",
                ),
            )
        }
        scores = np.asarray([[0.0, 0.5, 2.0, 0.1]], dtype=np.float32)
        matrix = np.asarray([[1, 2, 2]], dtype=np.uint16)
        router = mock.Mock()
        threshold_builder = mock.Mock(return_value=matrix)
        with (
            mock.patch.object(
                MODULE,
                "hash_file",
                side_effect=lambda path: hashes_by_path[path.as_posix()],
            ),
            mock.patch.object(MODULE, "resolve_device", return_value="mps"),
            mock.patch.object(MODULE, "build_router", return_value=router),
            mock.patch.object(MODULE.torch, "load", return_value={}),
            mock.patch.object(
                MODULE,
                "parameter_count",
                return_value=auxiliary["router_parameter_count"],
            ),
            mock.patch.object(
                MODULE,
                "_state_sha256",
                return_value=auxiliary["router_checkpoint_state_sha256"],
            ),
            mock.patch.object(MODULE, "router_entropy_scores", return_value=scores),
            mock.patch.object(
                MODULE,
                "threshold_patch_matrix",
                threshold_builder,
            ),
            mock.patch.object(MODULE, "_release"),
        ):
            rebuilt, execution = MODULE._reconstruct_entropy_execution(
                model_identity=model,
                seed=1729,
                inputs=np.zeros((1, 4), dtype=np.uint8),
                boundaries=np.ones((1, 4), dtype=np.uint8),
            )
        np.testing.assert_array_equal(rebuilt, matrix)
        self.assertEqual(
            execution["router_scores_sha256"], MODULE._array_sha256(scores)
        )
        self.assertEqual(
            execution["final_matrix_sha256"], MODULE._array_sha256(matrix)
        )
        threshold = threshold_builder.call_args.args[1]
        candidate_masks = threshold_builder.call_args.kwargs["candidate_masks"]
        maximum = threshold_builder.call_args.kwargs["maximum_patch_length"]
        self.assertEqual(threshold, auxiliary["threshold_nats"])
        np.testing.assert_array_equal(
            candidate_masks,
            np.ones((1, 4), dtype=np.uint8),
        )
        self.assertEqual(maximum, 24)

    def test_entropy_receipt_full_recompute_passes_and_bundle_tamper_fails(self) -> None:
        lock = selection_lock_fixture(
            broad_futile=False,
            broad_policy="entropy_threshold_codepoint",
        )
        broad = resolve_final_evaluation_roles(lock)["unique_models"][-1]
        model = model_fixture(broad)
        matrix = np.zeros((62_500, 87), dtype=np.uint16)
        matrix[:, 0] = 1
        matrix[:, 1] = 512
        values = np.full(62_500, 2.5, dtype=np.float32)
        receipt_sha = "2" * 64
        auxiliary = model["seeds"]["1729"]["auxiliary"]
        execution = {
            "final_matrix_sha256": MODULE._array_sha256(matrix),
            "kind": "entropy_router",
            "locked_bundle_sha256": MODULE.canonical_sha256(auxiliary),
            "router_scores_sha256": "3" * 64,
        }
        receipt = {
            "artifact_role": model["artifact_role"],
            "auxiliary_execution": execution,
            "nll": {"array_sha256": MODULE._array_sha256(values)},
            "patch_matrix_sha256": MODULE._array_sha256(matrix),
            "receipt_sha256": receipt_sha,
            "seed": 1729,
        }
        with (
            mock.patch.object(
                MODULE,
                "_load_verified_main_checkpoint",
                return_value=object(),
            ),
            mock.patch.object(
                MODULE,
                "_reconstruct_entropy_execution",
                return_value=(matrix, execution),
            ),
            mock.patch.object(
                MODULE,
                "evaluate_main_model",
                return_value=(0.0, values),
            ),
            mock.patch.object(MODULE, "_release"),
        ):
            rebuilt = MODULE._recompute_verified_nll_arrays(
                authorization={"models": [model]},
                evidence={"receipts": [receipt]},
                inputs=np.zeros((62_500, 512), dtype=np.uint8),
                boundaries=np.ones((62_500, 512), dtype=np.uint8),
                structural_matrices={},
                stored_arrays={receipt_sha: values},
            )
            np.testing.assert_array_equal(rebuilt[receipt_sha], values)

            tampered_model = copy.deepcopy(model)
            tampered_model["seeds"]["1729"]["auxiliary"][
                "threshold_nats"
            ] += 1e-6
            tampered_execution = dict(execution)
            tampered_execution["locked_bundle_sha256"] = MODULE.canonical_sha256(
                tampered_model["seeds"]["1729"]["auxiliary"]
            )
            with mock.patch.object(
                MODULE,
                "_reconstruct_entropy_execution",
                return_value=(matrix, tampered_execution),
            ):
                with self.assertRaisesRegex(ValueError, "matrix/router"):
                    MODULE._recompute_verified_nll_arrays(
                        authorization={"models": [tampered_model]},
                        evidence={"receipts": [receipt]},
                        inputs=np.zeros((62_500, 512), dtype=np.uint8),
                        boundaries=np.ones((62_500, 512), dtype=np.uint8),
                        structural_matrices={},
                        stored_arrays={receipt_sha: values},
                    )

    def test_active_or_runtime_drift_blocks_quality_lock(self) -> None:
        runtime = {
            "batch_size": 64,
            "device": "mps",
            "mps_available": True,
            "numpy": "2.5.2",
            "python": "3.13.11",
            "torch": "2.13.0",
            "transformers": "5.14.1",
        }
        session = {
            "final_context": {"stream_sha256": "a" * 64},
            "runtime": runtime,
            "session_plan_sha256": "b" * 64,
        }
        evidence = {
            "final_context": session["final_context"],
            "runtime": runtime,
            "session_plan": {"sha256": session["session_plan_sha256"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / ".active"
            sentinel.write_bytes(b"active")
            with (
                mock.patch.object(MODULE, "ACTIVE_SENTINEL", sentinel),
                mock.patch.object(MODULE, "_runtime_identity", return_value=runtime),
                mock.patch.object(MODULE, "resolve_device", return_value="mps"),
            ):
                with self.assertRaisesRegex(ValueError, "session is incomplete"):
                    MODULE._require_completed_session(session, evidence)
            sentinel.unlink()
            drifted = copy.deepcopy(evidence)
            drifted["runtime"]["torch"] = "different"
            with (
                mock.patch.object(MODULE, "ACTIVE_SENTINEL", sentinel),
                mock.patch.object(MODULE, "_runtime_identity", return_value=runtime),
                mock.patch.object(MODULE, "resolve_device", return_value="mps"),
            ):
                with self.assertRaisesRegex(ValueError, "session is incomplete"):
                    MODULE._require_completed_session(session, drifted)

    def test_quality_lock_requires_exact_complete_artifact_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session-plan.json"
            receipt = root / "seed-1729" / "candidate-receipt.json"
            nll = root / "seed-1729" / "candidate-nll.npz"
            receipt.parent.mkdir(parents=True)
            for path in (session, receipt, nll):
                path.write_bytes(b"sealed")
            evidence = {
                "receipt_artifacts": [{"path": receipt.as_posix()}],
                "receipts": [{"nll": {"artifact_path": nll.as_posix()}}],
            }
            with (
                mock.patch.object(MODULE, "FINAL_ARTIFACT_ROOT", root),
                mock.patch.object(MODULE, "SESSION_PATH", session),
            ):
                MODULE._require_canonical_artifact_namespace(evidence)
                (root / "alternate.part").write_bytes(b"partial")
                with self.assertRaisesRegex(ValueError, "namespace"):
                    MODULE._require_canonical_artifact_namespace(evidence)

    def test_quality_verifier_uses_the_same_exclusive_session_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "session-plan.json"
            session.write_bytes(b"sealed-plan")
            with (
                mock.patch.object(MODULE, "FINAL_ARTIFACT_ROOT", root),
                mock.patch.object(MODULE, "SESSION_PATH", session),
            ):
                with MODULE._exclusive_session_process_lock():
                    with self.assertRaisesRegex(RuntimeError, "another final"):
                        with MODULE._exclusive_session_process_lock():
                            self.fail("a concurrent verifier acquired the lock")


if __name__ == "__main__":
    unittest.main()
