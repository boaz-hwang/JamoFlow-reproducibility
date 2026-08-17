from __future__ import annotations

import ast
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "reconstruct_inference_calibration_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconstruct_inference_calibration_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReconstructInferenceCalibrationV2Tests(unittest.TestCase):
    def test_npz_payload_preserves_exact_float32_vector(self) -> None:
        values = np.asarray([1.25, 2.5, 3.75], dtype=np.float32)
        with np.load(io.BytesIO(MODULE._npz_bytes(values)), allow_pickle=False) as data:
            self.assertEqual(data.files, ["sequence_nll_nats"])
            self.assertTrue(np.array_equal(data["sequence_nll_nats"], values))
            self.assertEqual(data["sequence_nll_nats"].dtype, np.float32)

    def test_resume_refuses_partial_receipt_pair_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partial calibration evidence"):
                MODULE._load_existing_receipt(
                    receipt_path=receipt,
                    nll_path=root / "missing.npz",
                    plan={},
                    plan_artifact_sha256="a" * 64,
                    evaluator_git_commit="b" * 40,
                    report_path=root / "report.json",
                    checkpoint_path=root / "checkpoint.pt",
                    matrix=np.ones((1, 1), dtype=np.int64),
                    auxiliary={"kind": "none"},
                    inputs=np.ones((1, 1), dtype=np.uint8),
                    boundaries=np.ones((1, 1), dtype=np.bool_),
                    stream_sha256="c" * 64,
                    initial_model_identity_lock_sha256="d" * 64,
                )

    def test_selection_evaluator_has_no_test_or_latency_artifact_path(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        forbidden_fragments = (
            "-test-nll",
            "actual-inference",
            "final-test-v1/ko.jsonl",
            "latency_ms",
        )
        for value in strings:
            self.assertFalse(
                any(fragment in value for fragment in forbidden_fragments),
                f"selection evaluator can access forbidden result path: {value}",
            )

    def test_complete_receipt_is_replayed_and_forged_nll_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            report.write_text("{}", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            paths = {
                "training_report": str(report),
                "checkpoint": str(checkpoint),
                "nll": str(root / "nll.npz"),
                "receipt": str(root / "receipt.json"),
            }
            existing_receipt = {
                "calibration": {"nll_array_sha256": "1" * 64}
            }
            forged = np.asarray([1.0], dtype=np.float32)
            replayed = np.asarray([2.0], dtype=np.float32)
            replay = {
                "auxiliary": {"kind": "none"},
                "bpb": 1.0,
                "checkpoint_artifact_sha256": "2" * 64,
                "checkpoint_path": str(checkpoint),
                "checkpoint_state_sha256": "3" * 64,
                "losses": replayed,
                "nll_array_sha256": "4" * 64,
                "parameter_count": 1,
                "report_artifact_sha256": "5" * 64,
                "report_bpb": 1.0,
                "report_path": str(report),
                "spec_sha256": "6" * 64,
            }
            with (
                mock.patch.object(
                    MODULE, "expected_evidence_paths", return_value=paths
                ),
                mock.patch.object(
                    MODULE,
                    "_load_existing_receipt",
                    return_value=(existing_receipt, forged),
                ),
                mock.patch.object(
                    MODULE,
                    "replay_calibration_unit",
                    return_value=replay,
                ) as causal_forward,
            ):
                with self.assertRaisesRegex(
                    ValueError, "fails causal-forward replay"
                ):
                    MODULE._evaluate_receipt(
                        seed=1729,
                        policy="fixed_byte_6",
                        inputs=np.ones((1, 1), dtype=np.uint8),
                        boundaries=np.ones((1, 1), dtype=np.bool_),
                        stream_sha256="7" * 64,
                        matrix=np.ones((1, 1), dtype=np.uint16),
                        auxiliary={"kind": "none"},
                        plan={"calibration_evaluator": {"batch_size": 1}},
                        plan_artifact_sha256="8" * 64,
                        evaluator_git_commit="a" * 40,
                        initial_model_identity_lock_sha256="9" * 64,
                        identity_lock={},
                        device="mps",
                    )
            causal_forward.assert_called_once()

    def test_fresh_receipt_validates_before_publishing_both_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            report.write_text("{}", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            paths = {
                "training_report": str(report),
                "checkpoint": str(checkpoint),
                "nll": str(root / "nll.npz"),
                "receipt": str(root / "receipt.json"),
            }
            losses = np.asarray([2.0], dtype=np.float32)
            replay = {
                "auxiliary": {"kind": "none"},
                "bpb": 1.0,
                "checkpoint_artifact_sha256": "2" * 64,
                "checkpoint_path": str(checkpoint),
                "checkpoint_state_sha256": "3" * 64,
                "losses": losses,
                "nll_array_sha256": "4" * 64,
                "parameter_count": 19_596_096,
                "report_artifact_sha256": "5" * 64,
                "report_bpb": 1.0,
                "report_path": str(report),
                "spec_sha256": "6" * 64,
            }
            sealed = {"receipt_sha256": "7" * 64}
            events: list[str] = []

            def validate(*args, **kwargs):
                del args, kwargs
                events.append("validate")

            def publish(path, payload):
                del payload
                events.append(f"publish:{Path(path).name}")

            with (
                mock.patch.object(MODULE, "expected_evidence_paths", return_value=paths),
                mock.patch.object(MODULE, "_load_existing_receipt", return_value=None),
                mock.patch.object(MODULE, "replay_calibration_unit", return_value=replay),
                mock.patch.object(
                    MODULE,
                    "_model_spec",
                    return_value=mock.Mock(patch_count=86),
                ),
                mock.patch.object(MODULE, "hash_file", return_value="8" * 64),
                mock.patch.object(MODULE, "seal_calibration_receipt", return_value=sealed),
                mock.patch.object(MODULE, "validate_calibration_receipt", side_effect=validate),
                mock.patch.object(MODULE, "publish_no_clobber", side_effect=publish),
            ):
                receipt = MODULE._evaluate_receipt(
                    seed=1729,
                    policy="fixed_byte_6",
                    inputs=np.ones((1, 512), dtype=np.uint8),
                    boundaries=np.ones((1, 512), dtype=np.bool_),
                    stream_sha256="9" * 64,
                    matrix=np.ones((1, 86), dtype=np.uint16),
                    auxiliary={"kind": "none"},
                    plan={"calibration_evaluator": {"batch_size": 1}},
                    plan_artifact_sha256="a" * 64,
                    evaluator_git_commit="b" * 40,
                    initial_model_identity_lock_sha256="c" * 64,
                    identity_lock={},
                    device="mps",
                )
            self.assertEqual(receipt, sealed)
            self.assertEqual(
                events,
                ["validate", "publish:nll.npz", "publish:receipt.json"],
            )


if __name__ == "__main__":
    unittest.main()
