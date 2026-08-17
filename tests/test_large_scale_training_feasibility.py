from __future__ import annotations

import importlib.util
import sys
import unittest
from copy import deepcopy
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "large_scale_training_feasibility_core.py"
SPEC = importlib.util.spec_from_file_location("large_scale_training_feasibility_core", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SCRIPTS = str(Path(__file__).parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
SPEC.loader.exec_module(MODULE)


class LargeScaleTrainingFeasibilityTests(unittest.TestCase):
    def _plan(self) -> dict:
        return {
            "plan_sha256": "a" * 64,
            "environment": {"device": "mps"},
            "models": {
                str(target): {
                    "expected_parameter_count": MODULE.EXPECTED_PARAMETERS[target],
                    "model_state_sha256": chr(98 + index) * 64,
                }
                for index, target in enumerate(MODULE.TARGET_ORDER)
            },
            "training_data": {
                "inputs_array_sha256": "f" * 64,
                "patch_matrix_sha256": {"c86": "1" * 64, "w72": "2" * 64},
            },
        }

    def _report(
        self,
        target: int,
        regime: str,
        role: str,
        *,
        update_seconds: tuple[float, float] = (1.0, 1.0),
        completed: bool = True,
    ) -> dict:
        plan = self._plan()
        snapshots = [
            {
                "stage": "model_resident",
                "current_allocated_bytes": 10,
                "driver_allocated_bytes": 20,
            },
            {
                "stage": "measurement_update_1",
                "current_allocated_bytes": 30,
                "driver_allocated_bytes": 40,
            },
        ]
        report = {
            "schema_version": 1,
            "kind": "large_scale_training_feasibility_worker_v1",
            "protocol_id": MODULE.PROTOCOL_ID,
            "target_millions": target,
            "regime": regime,
            "role": role,
            "runner_git_commit": "9" * 40,
            "plan_sha256": plan["plan_sha256"],
            "plan_artifact_sha256": "8" * 64,
            "parameter_count": MODULE.EXPECTED_PARAMETERS[target],
            "model_state_sha256": plan["models"][str(target)]["model_state_sha256"],
            "patch_matrix_sha256": plan["training_data"]["patch_matrix_sha256"][role],
            "training_data_sha256": plan["training_data"]["inputs_array_sha256"],
            "memory_cap_enforced": True,
            "memory_snapshots": snapshots,
            "maximum_driver_allocated_bytes": 40,
            "recommended_max_memory_bytes": 100,
            "optimizer_state_initialized": completed,
            "measurement": MODULE.projected_training(update_seconds) if completed else None,
            "finite": completed,
            "completed": completed,
            "failure": None,
            "environment_start": plan["environment"],
            "environment_end": plan["environment"],
        }
        if not completed:
            report["failure"] = {
                "category": "RuntimeError",
                "message": "out of memory",
                "returncode": 1,
                "stage": "warmup_update",
            }
        return report

    def _reports(self, *, seconds: float = 1.0) -> dict:
        return {
            MODULE.worker_id(target, regime, role): self._report(
                target, regime, role, update_seconds=(seconds, seconds)
            )
            for target, regime, role in MODULE.worker_order()
        }

    def test_worker_order_is_exact_and_checkpointing_only_targets_1600(self) -> None:
        self.assertEqual(len(MODULE.worker_order()), 10)
        self.assertEqual(
            MODULE.worker_order()[-2:],
            (
                (1600, MODULE.CHECKPOINTED_REGIME, "c86"),
                (1600, MODULE.CHECKPOINTED_REGIME, "w72"),
            ),
        )
        with self.assertRaises(ValueError):
            MODULE.worker_id(800, MODULE.CHECKPOINTED_REGIME, "c86")

    def test_projection_reconstructs_exact_update_budget(self) -> None:
        result = MODULE.projected_training((2.0, 4.0))
        self.assertEqual(result["median_update_seconds"], 3.0)
        row = result["by_source_byte_budget"]["64000000"]
        self.assertEqual(row["optimizer_updates"], 31_250)
        self.assertEqual(row["projected_source_bytes"], 64_000_000)
        self.assertAlmostEqual(row["projected_hours_per_model"], 26.0416666667)
        with self.assertRaises(ValueError):
            MODULE.projected_training((1.0, float("nan")))

    def test_worker_validation_rebuilds_projection_memory_and_identity(self) -> None:
        plan = self._plan()
        report = self._report(200, MODULE.STANDARD_REGIME, "c86")
        MODULE.validate_worker_report(
            report,
            plan=plan,
            plan_artifact_sha256="8" * 64,
            runner_git_commit="9" * 40,
            target=200,
            regime=MODULE.STANDARD_REGIME,
            role="c86",
        )
        altered = deepcopy(report)
        altered["measurement"]["median_update_seconds"] += 1
        with self.assertRaisesRegex(ValueError, "successful measurement"):
            MODULE.validate_worker_report(
                altered,
                plan=plan,
                plan_artifact_sha256="8" * 64,
                runner_git_commit="9" * 40,
                target=200,
                regime=MODULE.STANDARD_REGIME,
                role="c86",
            )
        altered = deepcopy(report)
        altered["maximum_driver_allocated_bytes"] = 80
        with self.assertRaises(ValueError):
            MODULE.validate_worker_report(
                altered,
                plan=plan,
                plan_artifact_sha256="8" * 64,
                runner_git_commit="9" * 40,
                target=200,
                regime=MODULE.STANDARD_REGIME,
                role="c86",
            )

    def test_primary_standard_pass_is_selected_without_lower_target_fallback(self) -> None:
        aggregate = MODULE.summarize_reports(self._reports(seconds=1.0))
        self.assertTrue(aggregate["primary_1600_resource_feasible"])
        self.assertEqual(
            aggregate["primary_1600_selected_regime"], MODULE.STANDARD_REGIME
        )
        self.assertFalse(aggregate["lower_target_fallback_authorized"])
        self.assertFalse(aggregate["global_heavy_architecture_pivot_required"])

    def test_checkpointed_1600_can_rescue_only_the_fixed_primary(self) -> None:
        reports = self._reports(seconds=1.0)
        for role in MODULE.ROLE_ORDER:
            reports[MODULE.worker_id(1600, MODULE.STANDARD_REGIME, role)] = self._report(
                1600, MODULE.STANDARD_REGIME, role, completed=False
            )
        aggregate = MODULE.summarize_reports(reports)
        self.assertTrue(aggregate["primary_1600_resource_feasible"])
        self.assertEqual(
            aggregate["primary_1600_selected_regime"], MODULE.CHECKPOINTED_REGIME
        )

    def test_lower_resource_pass_cannot_replace_failed_1600_endpoint(self) -> None:
        reports = self._reports(seconds=1.0)
        for regime in MODULE.REGIME_ORDER:
            for role in MODULE.ROLE_ORDER:
                reports[MODULE.worker_id(1600, regime, role)] = self._report(
                    1600, regime, role, completed=False
                )
        aggregate = MODULE.summarize_reports(reports)
        self.assertEqual(
            aggregate["largest_standard_resource_feasible_target_millions"], 800
        )
        self.assertFalse(aggregate["primary_1600_resource_feasible"])
        self.assertFalse(aggregate["lower_target_fallback_authorized"])
        self.assertTrue(aggregate["global_heavy_architecture_pivot_required"])

    def test_120_hour_role_budget_is_fail_closed(self) -> None:
        seconds_at_limit = 120.0 * 3600 / 31_250
        report = self._report(
            1600,
            MODULE.STANDARD_REGIME,
            "c86",
            update_seconds=(seconds_at_limit, seconds_at_limit),
        )
        self.assertTrue(MODULE.resource_pass(report))
        report["measurement"] = MODULE.projected_training(
            (seconds_at_limit + 0.001, seconds_at_limit + 0.001)
        )
        self.assertFalse(MODULE.resource_pass(report))


if __name__ == "__main__":
    unittest.main()
