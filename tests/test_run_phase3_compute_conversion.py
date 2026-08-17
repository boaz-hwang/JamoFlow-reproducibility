import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from jamoflow.compute_conversion import CONVERSION_POLICIES, conversion_policy
from jamoflow.inference_selection_plan import PHASE3_PRIMARY_SUMMARY_PATH
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
    build_independent_calibration_recomputation_v2,
    build_selection_decision_v2,
    build_selection_lock_v2,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_phase3_compute_conversion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_phase3_compute_conversion",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase3ComputeConversionRunnerTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _calibration_fixture(self) -> dict[int, dict[str, float]]:
        rows = {
            seed: {policy: 1.5 for policy in CALIBRATION_POLICY_ORDER}
            for seed in INITIAL_SEEDS
        }
        for seed in INITIAL_SEEDS:
            rows[seed]["causal_codepoint_grid"] = 1.4
            rows[seed]["causal_whitespace_grid_64"] = 1.405
            rows[seed]["spacebyte_spacelike"] = 1.399
        return rows

    def test_initial_stage_requires_the_complete_preregistered_design(self) -> None:
        self.assertIsNone(
            MODULE._validate_stage(
                "initial",
                MODULE.INITIAL_SEEDS,
                CONVERSION_POLICIES,
                None,
            )
        )
        with self.assertRaisesRegex(ValueError, "all preregistered"):
            MODULE._validate_stage(
                "initial",
                MODULE.INITIAL_SEEDS[:2],
                CONVERSION_POLICIES,
                None,
            )

    def test_default_primary_summary_matches_the_sealed_plan(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "--stage",
                "initial",
                "--seeds",
                *map(str, MODULE.INITIAL_SEEDS),
                "--policies",
                *CONVERSION_POLICIES,
            ]
        )
        self.assertEqual(args.primary_summary, PHASE3_PRIMARY_SUMMARY_PATH)

    def test_runner_requires_same_clean_commit_at_start_and_end(self) -> None:
        with mock.patch.object(
            MODULE, "_git_commit", return_value="a" * 40
        ), mock.patch.object(MODULE, "_git_status", return_value=""):
            self.assertEqual(MODULE._clean_git_commit(), "a" * 40)
            MODULE._require_unchanged_clean_git("a" * 40)
        with mock.patch.object(
            MODULE, "_git_commit", return_value="a" * 40
        ), mock.patch.object(
            MODULE, "_git_status", return_value=" M protocol.py\n"
        ):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                MODULE._clean_git_commit()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                MODULE._require_unchanged_clean_git("a" * 40)

    def test_confirmation_attempt_resume_is_exact_and_missing_active_is_terminal(
        self,
    ) -> None:
        decision = build_selection_decision_v2(self._calibration_fixture())
        lock = build_selection_lock_v2(
            decision,
            plan_sha256="1" * 64,
            calibration_evidence_manifest_sha256="2" * 64,
            final_test_seal_sha256="3" * 64,
            initial_model_identity_lock_sha256="4" * 64,
            independent_calibration_recomputation=(
                build_independent_calibration_recomputation_v2(
                    self._calibration_fixture(),
                    nll_array_sha256_by_seed_policy={
                        seed: {
                            policy: "5" * 64
                            for policy in CALIBRATION_POLICY_ORDER
                        }
                        for seed in INITIAL_SEEDS
                    },
                    evaluator_git_commit="a" * 40,
                    verification_git_commit="b" * 40,
                    environment_sha256="c" * 64,
                    implementation_manifest_sha256="d" * 64,
                )
            ),
        )
        policies = tuple(
            lock["decision"]["confirmation_plan"]["compute_conversion"][
                "policies"
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            os.chdir(root)
            try:
                artifact_root = Path("artifacts/phase3-compute-conversion")
                completion = root / "results/completion.json"
                with (
                    mock.patch.object(
                        MODULE,
                        "COMPUTE_CONFIRMATION_COMPLETION_PATH",
                        completion,
                    ),
                    mock.patch.object(MODULE, "_git_path_history", return_value=""),
                ):
                    active, completed = MODULE._start_confirmation_attempt(
                        artifact_root=artifact_root,
                        selection_lock=lock,
                        selection_lock_artifact_sha256="e" * 64,
                        run_git_commit="f" * 40,
                        seeds=MODULE.CONFIRMATION_SEEDS,
                        policies=policies,
                    )
                    self.assertTrue(active.is_file())
                    resumed = MODULE._start_confirmation_attempt(
                        artifact_root=artifact_root,
                        selection_lock=lock,
                        selection_lock_artifact_sha256="e" * 64,
                        run_git_commit="f" * 40,
                        seeds=MODULE.CONFIRMATION_SEEDS,
                        policies=policies,
                    )
                    self.assertEqual(resumed, (active, completed))
                    active.unlink()
                    report = (
                        Path("runs/phase3-compute-conversion")
                        / f"seed-{MODULE.CONFIRMATION_SEEDS[0]}"
                        / f"{policies[0]}.json"
                    )
                    report.parent.mkdir(parents=True)
                    report.write_text("{}", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "without their active"):
                        MODULE._start_confirmation_attempt(
                            artifact_root=artifact_root,
                            selection_lock=lock,
                            selection_lock_artifact_sha256="e" * 64,
                            run_git_commit="f" * 40,
                            seeds=MODULE.CONFIRMATION_SEEDS,
                            policies=policies,
                        )
            finally:
                os.chdir(previous)

    def test_evidence_identity_is_canonical(self) -> None:
        first = {"stage": "initial", "seeds": [1729, 2718, 31415]}
        second = dict(reversed(tuple(first.items())))
        self.assertEqual(
            MODULE._canonical_sha256(first),
            MODULE._canonical_sha256(second),
        )
        self.assertIn(
            "calibration_loss_artifact_sha256",
            MODULE.CONVERSION_REPORT_KEYS,
        )
        self.assertIn("evidence_binding", MODULE.CONVERSION_REPORT_KEYS)

    def test_confirmation_stage_is_bound_to_the_selected_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            decision = build_selection_decision_v2(self._calibration_fixture())
            lock = build_selection_lock_v2(
                decision,
                plan_sha256="1" * 64,
                calibration_evidence_manifest_sha256="2" * 64,
                final_test_seal_sha256="3" * 64,
                initial_model_identity_lock_sha256="4" * 64,
                independent_calibration_recomputation=(
                    build_independent_calibration_recomputation_v2(
                        self._calibration_fixture(),
                        nll_array_sha256_by_seed_policy={
                            seed: {
                                policy: "5" * 64
                                for policy in CALIBRATION_POLICY_ORDER
                            }
                            for seed in INITIAL_SEEDS
                        },
                        evaluator_git_commit="a" * 40,
                        verification_git_commit="b" * 40,
                        environment_sha256="c" * 64,
                        implementation_manifest_sha256="d" * 64,
                    )
                ),
            )
            self._write_json(
                summary,
                lock,
            )
            policies = (
                conversion_policy("codepoint", 64),
                conversion_policy("whitespace", 64),
            )
            loaded = MODULE._validate_stage(
                "confirmation",
                MODULE.CONFIRMATION_SEEDS,
                policies,
                summary,
            )
            self.assertEqual(
                loaded["decision"]["rate_selection"]["selected_rate"],
                64,
            )
            with self.assertRaisesRegex(ValueError, "selection-v2 lock"):
                MODULE._validate_stage(
                    "confirmation",
                    MODULE.CONFIRMATION_SEEDS,
                    tuple(reversed(policies)),
                    summary,
                )

    def test_primary_gate_must_pass_with_complete_integrity(self) -> None:
        payload = {
            "seeds": [*MODULE.INITIAL_SEEDS, *MODULE.CONFIRMATION_SEEDS],
            "policies": [
                "fixed_byte_6",
                "causal_codepoint_grid",
                "causal_whitespace_grid",
            ],
            "gate_i": {"overall_pass": True},
            "gate_j": {"overall_pass": True},
            "integrity": {"all_integrity_checks_pass": True},
            "confirmation_authorization": {
                "authorization_kind": "phase3_corrected_gate_i_confirmation_v1"
            },
            "ood": {
                "gate_i_ood_guard": {"pass": True},
                "integrity": {"all_integrity_checks_pass": True},
            },
            "targets_per_sequence": 511,
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            self._write_json(summary, payload)
            self.assertEqual(MODULE._load_primary_gate(summary), payload)
            payload["gate_j"]["overall_pass"] = False
            self._write_json(summary, payload)
            with self.assertRaisesRegex(ValueError, "Gate J"):
                MODULE._load_primary_gate(summary)

    def test_primary_gate_rejects_initial_only_seed_set(self) -> None:
        payload = {
            "seeds": list(MODULE.INITIAL_SEEDS),
            "policies": [
                "fixed_byte_6",
                "causal_codepoint_grid",
                "causal_whitespace_grid",
            ],
            "gate_i": {"overall_pass": True},
            "gate_j": {"overall_pass": True},
            "integrity": {"all_integrity_checks_pass": True},
            "confirmation_authorization": {
                "authorization_kind": "phase3_corrected_gate_i_confirmation_v1"
            },
            "ood": {
                "gate_i_ood_guard": {"pass": True},
                "integrity": {"all_integrity_checks_pass": True},
            },
            "targets_per_sequence": 511,
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            self._write_json(summary, payload)
            with self.assertRaisesRegex(ValueError, "five-seed Gate J"):
                MODULE._load_primary_gate(summary)

    def test_partial_result_is_never_silently_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            self._write_json(report, {})
            with self.assertRaisesRegex(ValueError, "partial conversion result"):
                MODULE._completed_result_valid(
                    report,
                    root / "checkpoint.pt",
                    root / "calibration-losses.npz",
                    root / "losses.npz",
                    seed=1729,
                    policy=conversion_policy("codepoint", 64),
                    inputs={},
                    boundaries={},
                    matrices={},
                    evidence_binding={"identity_sha256": "a" * 64},
                )

    def test_partial_staging_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.with_suffix(".json.part").write_text(
                "partial",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "staging artifact"):
                MODULE._completed_result_valid(
                    report,
                    root / "checkpoint.pt",
                    root / "calibration-losses.npz",
                    root / "losses.npz",
                    seed=1729,
                    policy=conversion_policy("codepoint", 64),
                    inputs={},
                    boundaries={},
                    matrices={},
                    evidence_binding={"identity_sha256": "a" * 64},
                )


if __name__ == "__main__":
    unittest.main()
