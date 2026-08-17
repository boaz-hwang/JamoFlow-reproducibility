from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    script = ROOT / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MODULE = _load_script("summarize_phase3_ood")
RUNNER = _load_script("run_phase3_ood")


class Phase3OodTests(unittest.TestCase):
    def test_runner_requires_same_clean_commit_at_start_and_end(self) -> None:
        with mock.patch.object(RUNNER, "_git_commit", return_value="a" * 40), mock.patch.object(
            RUNNER,
            "_git_status",
            return_value="",
        ):
            self.assertEqual(RUNNER._clean_git_commit(), "a" * 40)
            RUNNER._require_unchanged_clean_git("a" * 40)
        with mock.patch.object(RUNNER, "_git_commit", return_value="a" * 40), mock.patch.object(
            RUNNER,
            "_git_status",
            return_value=" M docs/drift.md\n",
        ):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                RUNNER._clean_git_commit()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                RUNNER._require_unchanged_clean_git("a" * 40)

    def test_ood_gate_requires_both_margins(self) -> None:
        passed = MODULE.ood_gate(
            {
                "whitespace_minus_codepoint": 0.019,
                "whitespace_minus_fixed": -0.1,
            }
        )
        failed = MODULE.ood_gate(
            {
                "whitespace_minus_codepoint": 0.021,
                "whitespace_minus_fixed": -0.1,
            }
        )
        self.assertTrue(passed["pass"])
        self.assertFalse(failed["pass"])

    def test_ood_margin_is_inclusive(self) -> None:
        gate = MODULE.ood_gate(
            {
                "whitespace_minus_codepoint": 0.020,
                "whitespace_minus_fixed": 0.020,
            }
        )
        self.assertTrue(gate["pass"])

    def test_manifest_requires_invocation_coverage_for_each_result(self) -> None:
        manifest = {
            "schema_version": 1,
            "seeds": list(MODULE.INITIAL_SEEDS),
            "policies": list(MODULE.POLICIES),
            "model_spec": MODULE.PHASE3_MODEL_SPEC.to_dict(),
            "global_max_position_embeddings": MODULE.GLOBAL_POSITION_LIMIT,
            "requested_byte_limit": 100_000_000,
            "invocations": [
                {
                    "seeds": list(MODULE.INITIAL_SEEDS),
                    "policies": list(MODULE.POLICIES),
                }
            ],
        }
        MODULE._validate_manifest_design(manifest, MODULE.INITIAL_SEEDS)
        manifest["invocations"][0]["policies"] = [MODULE.F, MODULE.C]
        with self.assertRaisesRegex(ValueError, "no invocation"):
            MODULE._validate_manifest_design(manifest, MODULE.INITIAL_SEEDS)

    def test_confirmation_report_requires_exact_clean_authorized_invocation(self) -> None:
        authorization = {"authorization_kind": "phase3-confirmation", "hash": "a"}
        report = {
            "schema_version": 2,
            "git_commit": "c" * 40,
            "git_worktree_clean_at_start": True,
            "authorization": authorization,
        }
        manifest = {
            "invocations": [
                {
                    "seeds": [57721, 65537],
                    "policies": list(MODULE.POLICIES),
                    "git_commit": "c" * 40,
                    "git_worktree_clean_at_start": True,
                    "authorization": authorization,
                }
            ]
        }
        MODULE._validate_confirmation_report_binding(
            report,
            manifest,
            seed=57721,
            policy=MODULE.W,
            authorization=authorization,
        )
        report["authorization"] = {"authorization_kind": "other"}
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            MODULE._validate_confirmation_report_binding(
                report,
                manifest,
                seed=57721,
                policy=MODULE.W,
                authorization=authorization,
            )

    def test_runner_rejects_stale_completed_result(self) -> None:
        examples = 2
        nll = np.full(
            examples,
            MODULE.TARGETS_PER_SEQUENCE * math.log(2.0),
            dtype=np.float64,
        )
        report = {
            "schema_version": 2,
            "seed": 1729,
            "policy": MODULE.F,
            "parameters": MODULE.EXPECTED_PARAMETERS,
            "model_spec": MODULE.PHASE3_MODEL_SPEC.to_dict(),
            "global_max_position_embeddings": MODULE.GLOBAL_POSITION_LIMIT,
            "stream_selected_sha256": "stream",
            "source_file_sha256": "source",
            "patch_matrix_sha256": "patch",
            "training_report_artifact_sha256": "training-report",
            "checkpoint_artifact_sha256": "checkpoint",
            "checkpoint_state_sha256": "state",
            "training_report_state_sha256": "state",
            "git_commit": "c" * 40,
            "git_worktree_clean_at_start": True,
            "authorization": None,
            "evaluation": {
                "examples": examples,
                "predicted_bytes": examples * MODULE.TARGETS_PER_SEQUENCE,
                "bpb": 1.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            loss_path = Path(directory) / "loss.npz"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            np.savez_compressed(loss_path, sequence_nll_nats=nll)
            kwargs = {
                "seed": 1729,
                "policy": MODULE.F,
                "expected_examples": examples,
                "stream_sha256": "stream",
                "source_sha256": "source",
                "patch_matrix_sha256": "patch",
                "training_report_sha256": "training-report",
                "checkpoint_file_sha256": "checkpoint",
                "trained_state_sha256": "state",
                "git_commit": "c" * 40,
                "authorization": None,
            }
            RUNNER._validate_completed_result(report_path, loss_path, **kwargs)
            kwargs["stream_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "stale OOD result"):
                RUNNER._validate_completed_result(
                    report_path,
                    loss_path,
                    **kwargs,
                )
            kwargs["stream_sha256"] = "stream"
            kwargs["authorization"] = {
                "authorization_kind": "unexpected"
            }
            with self.assertRaisesRegex(ValueError, "stale OOD result"):
                RUNNER._validate_completed_result(
                    report_path,
                    loss_path,
                    **kwargs,
                )


if __name__ == "__main__":
    unittest.main()
