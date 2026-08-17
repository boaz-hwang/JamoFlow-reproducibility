from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT / "src", SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
SCRIPT = SCRIPTS / "balanced_200m_trained_core.py"
SPEC = importlib.util.spec_from_file_location("balanced_200m_trained_core", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Balanced200MTrainedScreenTests(unittest.TestCase):
    def _data(self) -> dict:
        return {
            "source_path": "data/processed/hplt3-korean-phase3/ko.jsonl",
            "source_sha256": "1" * 64,
            "integrity_path": "data/processed/hplt3-korean-phase3/integrity.json",
            "integrity_sha256": "2" * 64,
            "sequence_length": MODULE.SEQUENCE_LENGTH,
            "nominal_train_bytes": MODULE.NOMINAL_TRAIN_BYTES,
            "available_train_sequences": MODULE.AVAILABLE_TRAIN_SEQUENCES,
            "used_train_sequences": MODULE.TRAIN_SEQUENCES,
            "used_train_bytes": MODULE.TRAIN_BYTES,
            "dropped_train_sequences": (
                MODULE.AVAILABLE_TRAIN_SEQUENCES - MODULE.TRAIN_SEQUENCES
            ),
            "inputs_array_sha256": "3" * 64,
            "training_order_seed": MODULE.TRAINING_ORDER_SEED,
            "training_order_array_sha256": "4" * 64,
            "training_patch_matrix_sha256": {"c86": "5" * 64, "w72": "6" * 64},
            "preflight_examples": MODULE.PREFLIGHT_EXAMPLES,
            "preflight_selection": (
                "first 96 complete 512-byte sequences in canonical train stream"
            ),
            "preflight_inputs_array_sha256": "7" * 64,
            "preflight_patch_matrix_sha256": {"c86": "8" * 64, "w72": "9" * 64},
            "calibration_bytes": MODULE.CALIBRATION_BYTES,
            "calibration_examples": MODULE.CALIBRATION_BYTES // MODULE.SEQUENCE_LENGTH,
            "calibration_inputs_array_sha256": "a" * 64,
            "calibration_patch_matrix_sha256": {"c86": "b" * 64, "w72": "c" * 64},
            "historical_test_or_final_metric_used": False,
        }

    def _plan(self) -> dict:
        payload = {
            "schema_version": 1,
            "kind": "balanced_200m_trained_screen_plan_v1",
            "protocol_id": MODULE.PROTOCOL_ID,
            "status": "sealed_before_batch8_preflight_and_training",
            "git_commit_before_plan": "d" * 40,
            "model": {
                "target_millions": MODULE.TARGET,
                "expected_parameter_count": MODULE.EXPECTED_PARAMETER_COUNT,
                "spec": MODULE.large_scale_model_spec(MODULE.TARGET, 86).to_dict(),
                "model_seed": MODULE.MODEL_SEED,
                "global_position_limit": MODULE.GLOBAL_POSITION_LIMIT,
                "model_state_sha256": "e" * 64,
            },
            "roles": {
                "order": list(MODULE.ROLE_ORDER),
                "c86": {"policy": "causal_codepoint_grid", "patch_count": 86},
                "w72": {"policy": "causal_whitespace_grid", "patch_count": 72},
            },
            "data": self._data(),
            "optimizer": MODULE.optimizer_contract(),
            "preflight": {
                "warmup_updates": MODULE.PREFLIGHT_WARMUP_UPDATES,
                "measurement_updates": MODULE.PREFLIGHT_MEASUREMENT_UPDATES,
                "maximum_memory_fraction": MODULE.MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
                "maximum_hours_per_role": MODULE.MAXIMUM_HOURS_PER_ROLE,
                "maximum_hours_per_pair": MODULE.MAXIMUM_HOURS_PER_PAIR,
            },
            "quality_gate": {
                "calibration_bpb_margin": MODULE.QUALITY_MARGIN_BPB,
                "seed_count": 1,
                "historical_test_used_for_gate": False,
                "actual_timing_requires_quality_pass": True,
            },
            "trained_timing_gate": MODULE.trained_timing_contract(),
            "upstream": {
                "resource_summary_path": MODULE.RESOURCE_SUMMARY_PATH.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "resource_summary_artifact_sha256": "f" * 64,
                "resource_summary_sha256": "0" * 64,
                "scale_plan_path": MODULE.SCALE_PLAN_PATH.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "scale_plan_artifact_sha256": "4" * 64,
                "scale_plan_sha256": "5" * 64,
                "scale_summary_path": MODULE.SCALE_SUMMARY_PATH.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "scale_summary_artifact_sha256": "1" * 64,
                "scale_summary_sha256": "2" * 64,
            },
            "environment": {"device": "mps"},
            "implementation_sha256": {
                path: "3" * 64 for path in MODULE.IMPLEMENTATION_PATHS
            },
            "outputs": {
                "active_path": MODULE.ACTIVE_PATH.relative_to(MODULE.ROOT).as_posix(),
                "training_active_path": MODULE.TRAINING_ACTIVE_PATH.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "artifact_root": MODULE.ARTIFACT_ROOT.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "preflight_summary_path": MODULE.PREFLIGHT_OUTPUT_PATH.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "training_summary_path": MODULE.TRAINING_OUTPUT_PATH.relative_to(
                    MODULE.ROOT
                ).as_posix(),
                "training_reports": {
                    role: MODULE.training_report_path(role)
                    .relative_to(MODULE.ROOT)
                    .as_posix()
                    for role in MODULE.ROLE_ORDER
                },
                "checkpoints": {
                    role: MODULE.checkpoint_path(role)
                    .relative_to(MODULE.ROOT)
                    .as_posix()
                    for role in MODULE.ROLE_ORDER
                },
                "calibration_nll": {
                    role: MODULE.calibration_nll_path(role)
                    .relative_to(MODULE.ROOT)
                    .as_posix()
                    for role in MODULE.ROLE_ORDER
                },
            },
            "claim_boundary": {
                "one_seed_mechanism_screen": True,
                "sufficiently_trained_llm_claimed": False,
                "training_starts_only_after_committed_preflight_pass": True,
                "quality_claimed_before_training": False,
            },
        }
        return {**payload, "plan_sha256": MODULE.canonical_sha256(payload)}

    @staticmethod
    def _preflight_report(hours: float = 6.0) -> dict:
        return {
            "completed": True,
            "finite": True,
            "optimizer_state_initialized": True,
            "recommended_max_memory_bytes": 100,
            "maximum_driver_allocated_bytes": 60,
            "measurement": {"projected_hours": hours},
        }

    def test_exact_budget_uses_249984_sequences_and_7812_updates(self) -> None:
        self.assertEqual(MODULE.AVAILABLE_TRAIN_SEQUENCES, 250_000)
        self.assertEqual(MODULE.TRAIN_SEQUENCES, 249_984)
        self.assertEqual(MODULE.TOTAL_UPDATES, 7_812)
        self.assertEqual(MODULE.TRAIN_BYTES, 127_991_808)
        self.assertEqual(MODULE.PREFLIGHT_EXAMPLES, 96)

    def test_projection_and_resource_boundaries_are_exact(self) -> None:
        projection = MODULE.project_preflight((2.0, 4.0))
        self.assertEqual(projection["median_update_seconds"], 3.0)
        self.assertAlmostEqual(projection["projected_hours"], 6.51)
        report = self._preflight_report(MODULE.MAXIMUM_HOURS_PER_ROLE)
        self.assertTrue(MODULE.preflight_pass(report))
        report["measurement"]["projected_hours"] += 1e-6
        self.assertFalse(MODULE.preflight_pass(report))
        report = self._preflight_report()
        report["maximum_driver_allocated_bytes"] = 76
        self.assertFalse(MODULE.preflight_pass(report))

    def test_pair_gate_requires_both_roles_and_24_hours(self) -> None:
        reports = {role: self._preflight_report(12.0) for role in MODULE.ROLE_ORDER}
        aggregate = MODULE.summarize_preflight(reports)
        self.assertTrue(aggregate["overall_preflight_pass"])
        reports["w72"] = self._preflight_report(12.000001)
        self.assertFalse(MODULE.summarize_preflight(reports)["overall_preflight_pass"])

    def test_quality_gate_uses_only_float32_calibration_nll(self) -> None:
        count = MODULE.CALIBRATION_BYTES // MODULE.SEQUENCE_LENGTH
        c86 = np.full(count, 511 * np.log(2), dtype=np.float32)
        w72_pass = np.full(count, 511 * np.log(2) * 1.0099, dtype=np.float32)
        passed = MODULE.summarize_training_quality({"c86": c86, "w72": w72_pass})
        self.assertTrue(passed["quality_screen_pass"])
        w72_fail = np.full(count, 511 * np.log(2) * 1.0101, dtype=np.float32)
        failed = MODULE.summarize_training_quality({"c86": c86, "w72": w72_fail})
        self.assertFalse(failed["quality_screen_pass"])
        with self.assertRaises(ValueError):
            MODULE.summarize_training_quality(
                {"c86": c86.astype(np.float64), "w72": w72_pass}
            )

    def test_plan_validation_rejects_role_data_and_output_rotation(self) -> None:
        plan = self._plan()
        MODULE.validate_plan(
            plan, current_environment={"device": "mps"}, verify_implementation=False
        )
        for path, replacement in (
            (("roles", "w72", "patch_count"), 71),
            (("data", "preflight_examples"), 32),
            (("outputs", "training_summary_path"), "results/alternate.json"),
        ):
            changed = deepcopy(plan)
            cursor = changed
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = replacement
            payload = dict(changed)
            payload.pop("plan_sha256")
            changed["plan_sha256"] = MODULE.canonical_sha256(payload)
            with self.assertRaises(ValueError):
                MODULE.validate_plan(changed, verify_implementation=False)

    def test_worker_entrypoints_import_before_any_mps_work(self) -> None:
        for script in (
            "run_balanced_200m_preflight.py",
            "run_balanced_200m_training.py",
        ):
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / script), "--help"],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
