from __future__ import annotations

import ast
import importlib.util
import hashlib
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "seal_inference_selection_lock_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seal_inference_selection_lock_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SealInferenceSelectionLockV2Tests(unittest.TestCase):
    def test_paths_are_single_canonical_outputs(self) -> None:
        self.assertEqual(
            str(MODULE.OUTPUT),
            "results/phase3-inference-selection-v2/selection-lock.json",
        )
        self.assertEqual(
            str(MODULE.EVIDENCE),
            "results/phase3-inference-selection-v2/calibration-evidence.json",
        )

    def test_lock_builder_has_no_test_metric_or_latency_input(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        forbidden = ("test_bpb", "test-nll", "latency_ms", "timing.json")
        for value in strings:
            self.assertFalse(any(token in value for token in forbidden))

    def test_independent_replay_rejects_self_consistent_forged_nll(self) -> None:
        evidence = {
            "receipts": {
                "1": {
                    "policy": {
                        "auxiliary": {"kind": "none"},
                        "calibration": {"nll_array_sha256": "1" * 64},
                        "checkpoint": {
                            "artifact_sha256": "2" * 64,
                            "state_sha256": "3" * 64,
                        },
                        "training_report": {"artifact_sha256": "4" * 64},
                    }
                }
            }
        }
        replay = {
            "auxiliary": {"kind": "none"},
            "bpb": 1.0,
            "checkpoint_artifact_sha256": "2" * 64,
            "checkpoint_state_sha256": "3" * 64,
            "losses": np.asarray([2.0], dtype=np.float32),
            "nll_array_sha256": "5" * 64,
            "report_artifact_sha256": "4" * 64,
        }
        with (
            mock.patch.object(MODULE, "INITIAL_SEEDS", (1,)),
            mock.patch.object(MODULE, "CALIBRATION_POLICY_ORDER", ("policy",)),
            mock.patch.object(
                MODULE,
                "load_calibration_context",
                return_value=(
                    b"x",
                    np.ones((1, 1), dtype=np.uint8),
                    np.ones((1, 1), dtype=np.bool_),
                    {"policy": np.ones((1, 1), dtype=np.uint16)},
                ),
            ),
            mock.patch.object(
                MODULE,
                "reconstruct_entropy_matrices",
                return_value=({}, {}),
            ),
            mock.patch.object(
                MODULE, "replay_calibration_unit", return_value=replay
            ) as causal_forward,
            mock.patch.object(
                MODULE,
                "model_spec",
                return_value=mock.Mock(patch_count=1),
            ),
        ):
            plan = {"calibration_evaluator": {"input_stream_sha256": "0" * 64}}
            with mock.patch.object(
                MODULE.hashlib,
                "sha256",
                return_value=mock.Mock(hexdigest=mock.Mock(return_value="0" * 64)),
            ):
                with self.assertRaisesRegex(
                    ValueError, "fails independent causal replay"
                ):
                    MODULE._independent_replay(
                        plan=plan,
                        evidence=evidence,
                        identity={},
                        device="mps",
                    )
        causal_forward.assert_called_once()

    def test_successful_replay_executes_exactly_thirty_causal_forwards(self) -> None:
        seeds = (1, 2, 3)
        policies = tuple(f"policy-{index}" for index in range(10))
        inputs = np.ones((2, 512), dtype=np.uint8)
        boundaries = np.ones((2, 512), dtype=np.bool_)
        matrix = np.ones((2, 1), dtype=np.uint16)
        stream = b"calibration"
        array_sha = MODULE.array_sha256
        stream_sha = hashlib.sha256(stream).hexdigest()
        receipts = {}
        for seed in seeds:
            receipts[str(seed)] = {}
            for policy in policies:
                receipts[str(seed)][policy] = {
                    "auxiliary": {"kind": "none"},
                    "calibration": {
                        "boundaries_sha256": array_sha(boundaries),
                        "bpb": 1.0,
                        "count": 2,
                        "dtype": "float32",
                        "inputs_sha256": array_sha(inputs),
                        "matrix_sha256": array_sha(matrix),
                        "nll_array_sha256": "1" * 64,
                        "predicted_bytes": 1022,
                        "report_bpb": 1.0,
                        "stream_sha256": stream_sha,
                    },
                    "checkpoint": {
                        "artifact_sha256": "2" * 64,
                        "path": "checkpoint.pt",
                        "state_sha256": "3" * 64,
                    },
                    "model": {
                        "global_max_position_embeddings": 1_032,
                        "parameters": 19_596_096,
                        "spec_sha256": "6" * 64,
                    },
                    "patch_count": 1,
                    "training_report": {
                        "artifact_sha256": "4" * 64,
                        "path": "report.json",
                    },
                }
        replay = {
            "auxiliary": {"kind": "none"},
            "bpb": 1.0,
            "checkpoint_artifact_sha256": "2" * 64,
            "checkpoint_path": "checkpoint.pt",
            "checkpoint_state_sha256": "3" * 64,
            "losses": np.ones(2, dtype=np.float32),
            "nll_array_sha256": "1" * 64,
            "parameter_count": 19_596_096,
            "report_artifact_sha256": "4" * 64,
            "report_bpb": 1.0,
            "report_path": "report.json",
            "spec_sha256": "6" * 64,
        }
        with (
            mock.patch.object(MODULE, "INITIAL_SEEDS", seeds),
            mock.patch.object(MODULE, "CALIBRATION_POLICY_ORDER", policies),
            mock.patch.object(
                MODULE,
                "load_calibration_context",
                return_value=(
                    stream,
                    inputs,
                    boundaries,
                    {policy: matrix for policy in policies},
                ),
            ),
            mock.patch.object(MODULE, "reconstruct_entropy_matrices", return_value=({}, {})),
            mock.patch.object(MODULE, "replay_calibration_unit", return_value=replay) as forward,
            mock.patch.object(MODULE, "model_spec", return_value=mock.Mock(patch_count=1)),
        ):
            bpb, hashes = MODULE._independent_replay(
                plan={"calibration_evaluator": {"input_stream_sha256": stream_sha}},
                evidence={"receipts": receipts},
                identity={},
                device="mps",
            )
        self.assertEqual(forward.call_count, 30)
        self.assertEqual(sum(len(row) for row in bpb.values()), 30)
        self.assertEqual(sum(len(row) for row in hashes.values()), 30)

    def test_deleted_selection_lock_history_blocks_reseal(self) -> None:
        completed = mock.Mock(stdout="a" * 40)
        with mock.patch.object(
            MODULE,
            "OUTPUT",
            Path("results/nonexistent-selection-lock-for-test.json"),
        ), mock.patch.object(MODULE.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "was deleted"):
                MODULE._require_output_never_published()

    def test_double_replay_chronology_binds_evaluator_and_artifact_commits(self) -> None:
        expected = "1" * 64
        identity = {
            "calibration_selection_implementation": {
                "file_order": ["src/selector.py"],
                "sha256_by_path": {"src/selector.py": expected},
            }
        }
        evidence = {"evaluator_git_commit": "b" * 40}

        def tracked(path: Path) -> dict[str, str]:
            commit = "a" * 40 if path == MODULE.IDENTITY else "c" * 40
            return {"git_commit": commit, "path": str(path), "sha256": "2" * 64}

        with (
            mock.patch.object(MODULE, "_tracked_head_identity", side_effect=tracked),
            mock.patch.object(MODULE, "_require_ancestor") as ancestor,
            mock.patch.object(MODULE, "_git_blob_sha256", return_value=expected),
        ):
            MODULE._verify_double_replay_chronology(
                identity=identity,
                evidence=evidence,
                verification_commit="d" * 40,
            )
        self.assertEqual(ancestor.call_count, 3)

        with mock.patch.object(
            MODULE,
            "_tracked_head_identity",
            return_value={
                "git_commit": "b" * 40,
                "path": "evidence.json",
                "sha256": "2" * 64,
            },
        ):
            with self.assertRaisesRegex(ValueError, "not committed after"):
                MODULE._verify_double_replay_chronology(
                    identity=identity,
                    evidence=evidence,
                    verification_commit="d" * 40,
                )


if __name__ == "__main__":
    unittest.main()
