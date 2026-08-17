from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "reconstruct_inference_confirmation_calibration_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconstruct_inference_confirmation_calibration_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReconstructInferenceConfirmationCalibrationV2Tests(unittest.TestCase):
    def test_completion_history_requires_one_immutable_publication(self) -> None:
        path = Path("results/completion.json")
        commit = "a" * 40
        with mock.patch.object(MODULE, "_git_path_history", return_value=(commit,)):
            MODULE._require_single_publication_history(path, commit)
        for history in ((), (commit, "b" * 40)):
            with (
                self.subTest(history=history),
                mock.patch.object(MODULE, "_git_path_history", return_value=history),
                self.assertRaisesRegex(ValueError, "exactly once"),
            ):
                MODULE._require_single_publication_history(path, commit)

    def test_evaluator_has_no_historical_or_final_metric_artifact_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8").lower()
        for forbidden in (
            "-test-nll",
            "phase3-compute-conversion/confirmation-summary",
            "phase3-inference-final-v2/summary",
            "benchmark_phase3",
        ):
            self.assertNotIn(forbidden, source)

    def test_npz_payload_preserves_exact_float32_vector(self) -> None:
        values = np.asarray([1.0, 2.5, 7.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.npz"
            path.write_bytes(MODULE._npz_bytes(values))
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive.files, ["sequence_nll_nats"])
                self.assertTrue(np.array_equal(archive["sequence_nll_nats"], values))

    def test_partial_or_staged_receipt_pair_blocks_before_model_loading(self) -> None:
        cases = ("nll_only", "receipt_only", "staged")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt = root / "receipt.json"
                nll = root / "nll.npz"
                if case == "nll_only":
                    nll.write_bytes(b"partial")
                elif case == "receipt_only":
                    receipt.write_text("{}", encoding="utf-8")
                else:
                    receipt.with_suffix(".json.preparing").write_bytes(b"partial")
                with self.assertRaisesRegex(
                    ValueError,
                    "partial confirmation|forensic review",
                ):
                    MODULE._load_existing_receipt(
                        receipt_path=receipt,
                        nll_path=nll,
                        lock={},
                        lock_artifact_sha256="a" * 64,
                        evaluator_git_commit="b" * 40,
                        artifact_role="candidate",
                        descriptor={},
                        seed=57721,
                        report_path=root / "report.json",
                        checkpoint_path=root / "checkpoint.pt",
                        matrix=np.zeros((1, 1), dtype=np.int16),
                        auxiliary={"kind": "none"},
                    )

    def test_existing_receipt_still_requires_forward_and_rejects_forged_nll(
        self,
    ) -> None:
        existing_losses = np.zeros(
            MODULE.CALIBRATION_SEQUENCE_COUNT, dtype=np.float32
        )
        replay_losses = existing_losses.copy()
        replay_losses[0] = np.float32(1.0)
        existing_receipt = {"sealed": True}
        replay = {
            "bpb": 0.0,
            "checkpoint_artifact_sha256": "1" * 64,
            "checkpoint_state_sha256": "2" * 64,
            "losses": replay_losses,
            "nll_array_sha256": "3" * 64,
            "report_artifact_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            report.write_text("{}", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            with (
                mock.patch.object(
                    MODULE,
                    "expected_confirmation_paths",
                    return_value={
                        "receipt": str(root / "receipt.json"),
                        "nll": str(root / "nll.npz"),
                    },
                ),
                mock.patch.object(
                    MODULE,
                    "expected_model_paths",
                    return_value={
                        "training_report": str(report),
                        "checkpoint": str(checkpoint),
                    },
                ),
                mock.patch.object(
                    MODULE,
                    "_load_existing_receipt",
                    return_value=(existing_receipt, existing_losses),
                ),
                mock.patch.object(
                    MODULE,
                    "replay_confirmation_unit",
                    return_value=replay,
                ) as forward,
                mock.patch.object(
                    MODULE,
                    "build_confirmation_calibration_receipt",
                    return_value=existing_receipt,
                ),
                mock.patch.object(
                    MODULE, "validate_confirmation_calibration_receipt"
                ),
                mock.patch.object(MODULE, "hash_file", return_value="5" * 64),
            ):
                with self.assertRaisesRegex(ValueError, "fails causal replay"):
                    MODULE._evaluate_receipt(
                        lock={},
                        lock_artifact_sha256="a" * 64,
                        artifact_role="candidate",
                        descriptor={"policy": "candidate"},
                        seed=57721,
                        inputs=np.zeros((1, 512), dtype=np.uint8),
                        boundaries=np.zeros((1, 512), dtype=bool),
                        stream_sha256="b" * 64,
                        matrix=np.zeros((1, 2), dtype=np.int16),
                        auxiliary={"kind": "none"},
                        evaluator_git_commit="c" * 40,
                        device="mps",
                    )
            forward.assert_called_once()

    def test_fresh_receipt_is_validated_before_any_publication(self) -> None:
        losses = np.zeros(MODULE.CALIBRATION_SEQUENCE_COUNT, dtype=np.float32)
        replay = {
            "bpb": 0.0,
            "checkpoint_artifact_sha256": "1" * 64,
            "checkpoint_state_sha256": "2" * 64,
            "losses": losses,
            "nll_array_sha256": "3" * 64,
            "report_artifact_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            checkpoint = root / "checkpoint.pt"
            report.write_text("{}", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            with (
                mock.patch.object(
                    MODULE,
                    "expected_confirmation_paths",
                    return_value={
                        "receipt": str(root / "receipt.json"),
                        "nll": str(root / "nll.npz"),
                    },
                ),
                mock.patch.object(
                    MODULE,
                    "expected_model_paths",
                    return_value={
                        "training_report": str(report),
                        "checkpoint": str(checkpoint),
                    },
                ),
                mock.patch.object(MODULE, "_load_existing_receipt", return_value=None),
                mock.patch.object(MODULE, "replay_confirmation_unit", return_value=replay),
                mock.patch.object(
                    MODULE,
                    "build_confirmation_calibration_receipt",
                    return_value={"invalid": True},
                ),
                mock.patch.object(
                    MODULE,
                    "validate_confirmation_calibration_receipt",
                    side_effect=ValueError("invalid receipt"),
                ),
                mock.patch.object(MODULE, "hash_file", return_value="5" * 64),
                mock.patch.object(MODULE, "publish_no_clobber") as publish,
            ):
                with self.assertRaisesRegex(ValueError, "invalid receipt"):
                    MODULE._evaluate_receipt(
                        lock={},
                        lock_artifact_sha256="a" * 64,
                        artifact_role="candidate",
                        descriptor={"policy": "candidate"},
                        seed=57721,
                        inputs=np.zeros((1, 512), dtype=np.uint8),
                        boundaries=np.zeros((1, 512), dtype=bool),
                        stream_sha256="b" * 64,
                        matrix=np.zeros((1, 2), dtype=np.int16),
                        auxiliary={"kind": "none"},
                        evaluator_git_commit="c" * 40,
                        device="mps",
                    )
            publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
