from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = str(Path(__file__).parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
SCRIPT = Path(SCRIPTS) / "global_heavy_schedule_core.py"
SPEC = importlib.util.spec_from_file_location("global_heavy_schedule_core", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GlobalHeavyScheduleBridgeTests(unittest.TestCase):
    def test_protocol_uses_post_import_failure_v2_namespace(self) -> None:
        self.assertEqual(
            MODULE.PROTOCOL_ID, "jamoflow-global-heavy-schedule-bridge-v2"
        )
        self.assertTrue(str(MODULE.PLAN_PATH).endswith("bridge-v2.json"))

    def test_runner_entrypoint_imports_before_plan_execution(self) -> None:
        runner = Path(SCRIPTS) / "run_global_heavy_schedule_bridge.py"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            f"{Path(__file__).parents[1] / 'src'}:{SCRIPTS}"
        )
        completed = subprocess.run(
            (sys.executable, str(runner), "--help"),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def _correctness(self) -> dict:
        comparisons = MODULE.CORRECTNESS_PROMPTS * MODULE.CONTINUATION_BYTES
        return {
            "argmax_comparisons": comparisons,
            "argmax_exact": comparisons,
            "boundary_prefix_comparisons": comparisons,
            "boundary_trace_exact": True,
            "cache_diagnostics_exact": True,
            "maximum_normalized_logit_error": 0.1,
            "offline_boundary_prefix_exact": True,
        }

    def _reports(self) -> list[dict]:
        return [
            {
                "session_id": session,
                "parameter_count": MODULE.EXPECTED_PARAMETER_COUNT,
                "global_parameter_count": MODULE.EXPECTED_GLOBAL_PARAMETER_COUNT,
                "global_parameter_share": MODULE.EXPECTED_GLOBAL_PARAMETER_SHARE,
                "same_model_object_for_both_schedules": True,
                "correctness": {
                    role: self._correctness() for role in MODULE.SCHEDULE_ORDER
                },
                "maximum_driver_allocated_bytes": 40,
                "recommended_max_memory_bytes": 100,
                "environment_start": {"device": "mps"},
                "environment_end": {"device": "mps"},
            }
            for session in MODULE.SESSION_ORDER
        ]

    def _timings(self, reduction: float = 0.12) -> np.ndarray:
        values = np.empty(
            (
                len(MODULE.SESSION_ORDER),
                MODULE.MEASURED_PROMPTS,
                MODULE.INNER_REPETITIONS,
                len(MODULE.SCHEDULE_ORDER),
            ),
            dtype=np.float64,
        )
        values[:, :, :, MODULE.SCHEDULE_ORDER.index("c86")] = 100.0
        values[:, :, :, MODULE.SCHEDULE_ORDER.index("w72")] = 100.0 * (
            1 - reduction
        )
        return values

    def test_exact_model_contract_is_smaller_than_balanced_50m(self) -> None:
        contract = MODULE.global_heavy_model_contract()
        self.assertEqual(contract["expected_parameter_count"], 46_644_640)
        self.assertEqual(contract["expected_global_parameter_count"], 42_813_440)
        self.assertGreater(contract["expected_global_parameter_share"], 0.91)
        self.assertLess(
            contract["expected_parameter_count"],
            contract["comparison_balanced_parameter_count"],
        )

    def test_role_order_balances_each_session(self) -> None:
        for session in range(len(MODULE.SESSION_ORDER)):
            counts = [0, 0]
            for prompt in range(MODULE.MEASURED_PROMPTS):
                for repetition in range(MODULE.INNER_REPETITIONS):
                    counts[MODULE.role_order(session, prompt, repetition)[0]] += 1
            self.assertEqual(counts, [24, 24])

    @patch.object(MODULE, "mechanism_arrays")
    @patch.object(MODULE, "load_case_arrays")
    def test_fixed_gate_passes_only_with_every_clause(
        self, load_cases, mechanisms
    ) -> None:
        load_cases.return_value = (
            np.zeros((20, 128), dtype=np.uint8),
            np.zeros((20, 128), dtype=np.uint8),
            None,
            None,
            None,
        )
        counts = np.zeros((20, 2), dtype=np.int64)
        counts[:, 0] = 43
        counts[:, 1] = 36
        mechanisms.return_value = (counts, np.zeros((20, 2, 32), dtype=np.uint8))
        result = MODULE.summarize(self._timings(), self._reports())
        self.assertTrue(result["overall_threshold_pass"])
        self.assertEqual(result["status"], "global_heavy_10_percent_headroom_detected")
        failed = MODULE.summarize(self._timings(reduction=0.09), self._reports())
        self.assertFalse(failed["overall_threshold_pass"])
        altered = self._reports()
        altered[0] = deepcopy(altered[0])
        altered[0]["correctness"]["c86"]["argmax_exact"] -= 1
        failed = MODULE.summarize(self._timings(), altered)
        self.assertFalse(failed["gates"]["evidence_valid"])

    def test_correctness_rejects_relaxed_error_or_missing_comparison(self) -> None:
        row = self._correctness()
        self.assertTrue(MODULE._correctness_pass(row))
        row["maximum_normalized_logit_error"] = 1.0001
        self.assertFalse(MODULE._correctness_pass(row))
        row = self._correctness()
        row["argmax_comparisons"] -= 1
        self.assertFalse(MODULE._correctness_pass(row))


if __name__ == "__main__":
    unittest.main()
