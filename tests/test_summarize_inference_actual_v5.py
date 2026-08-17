from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.test_inference_actual_v5 import _plan_fixture


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_inference_actual_v5.py"
SPEC = importlib.util.spec_from_file_location("summarize_inference_actual_v5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _correctness(
    plan: dict,
    authorization: dict,
    *,
    comparison_contract: str = "mps_backend",
) -> dict:
    active_atol = 2e-5 if comparison_contract == "cpu_semantic" else 1e-4
    output = {}
    for seed in (1729, 2718, 31415, 57721, 65537):
        output[str(seed)] = {}
        for role in ("candidate", "reference"):
            model_id = plan["timing_pair"]["roles"][role]["model_identity_sha256"]
            model = next(
                item for item in authorization["models"]
                if item["identity_sha256"] == model_id
            )
            entropy = model["descriptor"]["requires_entropy_router"]
            output[str(seed)][role] = {
                "atol": active_atol,
                "boundary_trace_sha256": "a" * 64,
                "comparison_contract": comparison_contract,
                "entropy_router_argmax_exact_comparisons": 18_360 if entropy else 0,
                "entropy_router_position_comparisons": 18_360 if entropy else 0,
                "entropy_router_tolerance_tie_argmax_comparisons": 0,
                "main_full_causal_argmax_exact_comparisons": 18_360,
                "main_full_causal_position_comparisons": 18_360,
                "main_full_causal_tolerance_tie_argmax_comparisons": 0,
                "main_parallel_argmax_exact_comparisons": 9_216,
                "main_parallel_position_comparisons": 9_216,
                "main_parallel_tolerance_tie_argmax_comparisons": 0,
                "maximum_main_absolute_logit_error": 1e-6,
                "maximum_main_nominal_normalized_tolerance_ratio": 0.1,
                "maximum_main_normalized_tolerance_ratio": 0.1,
                "maximum_main_probability_total_variation": 1e-7,
                "maximum_router_absolute_entropy_error": 1e-6 if entropy else 0.0,
                "maximum_router_absolute_logit_error": 1e-6 if entropy else 0.0,
                "maximum_router_nominal_entropy_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_nominal_logit_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_normalized_entropy_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_normalized_logit_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_probability_total_variation": 1e-7 if entropy else 0.0,
                "main_nominal_tolerance_violation_elements": 0,
                "nominal_atol": 2e-5,
                "nominal_rtol": 2e-5,
                "pass": True,
                "probability_total_variation_limit": 1e-5,
                "router_entropy_nominal_tolerance_violation_elements": 0,
                "router_logit_nominal_tolerance_violation_elements": 0,
                "rtol": 2e-5,
            }
    return output


def _free_correctness(plan: dict, authorization: dict) -> dict:
    output = {}
    for seed in (1729, 2718, 31415, 57721, 65537):
        output[str(seed)] = {}
        for role in ("candidate", "reference"):
            model_id = plan["timing_pair"]["roles"][role][
                "model_identity_sha256"
            ]
            model = next(
                item
                for item in authorization["models"]
                if item["identity_sha256"] == model_id
            )
            entropy = model["descriptor"]["requires_entropy_router"]
            output[str(seed)][role] = {
                "atol": 1e-4,
                "boundary_trace_sha256": "b" * 64,
                "comparison_contract": "mps_backend",
                "entropy_router_argmax_exact_comparisons": 16_320 if entropy else 0,
                "entropy_router_position_comparisons": 16_320 if entropy else 0,
                "entropy_router_tolerance_tie_argmax_comparisons": 0,
                "greedy_byte_argmax_comparisons": 8_192,
                "main_full_causal_argmax_exact_comparisons": 16_320,
                "main_full_causal_position_comparisons": 16_320,
                "main_full_causal_tolerance_tie_argmax_comparisons": 0,
                "main_parallel_argmax_exact_comparisons": 8_192,
                "main_parallel_position_comparisons": 8_192,
                "main_parallel_tolerance_tie_argmax_comparisons": 0,
                "maximum_main_absolute_logit_error": 1e-6,
                "maximum_main_nominal_normalized_tolerance_ratio": 0.1,
                "maximum_main_normalized_tolerance_ratio": 0.1,
                "maximum_main_probability_total_variation": 1e-7,
                "maximum_router_absolute_entropy_error": 1e-6 if entropy else 0.0,
                "maximum_router_absolute_logit_error": 1e-6 if entropy else 0.0,
                "maximum_router_nominal_entropy_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_nominal_logit_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_normalized_entropy_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_normalized_logit_tolerance_ratio": 0.1 if entropy else 0.0,
                "maximum_router_probability_total_variation": 1e-7 if entropy else 0.0,
                "main_nominal_tolerance_violation_elements": 0,
                "nominal_atol": 2e-5,
                "nominal_rtol": 2e-5,
                "pass": True,
                "probability_total_variation_limit": 1e-5,
                "router_entropy_nominal_tolerance_violation_elements": 0,
                "router_logit_nominal_tolerance_violation_elements": 0,
                "rtol": 2e-5,
            }
    return output


class SummarizeInferenceActualV5Tests(unittest.TestCase):
    def test_summary_has_fixed_path_no_cli_and_rejects_compact_v4(self) -> None:
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
        self.assertFalse(any("phase3-actual-inference/summary" in value for value in strings))
        self.assertEqual(
            MODULE.OUTPUT_PATH.as_posix(),
            "results/phase3-inference-actual-v5r3/summary.json",
        )

    def test_correctness_requires_every_controlled_position_and_router_condition(self) -> None:
        authorization, _, plan = _plan_fixture()
        values = _correctness(plan, authorization)
        MODULE._validate_correctness(
            values,
            authorization=authorization,
            plan=plan,
        )
        values["1729"]["candidate"]["main_parallel_position_comparisons"] -= 1
        with self.assertRaisesRegex(ValueError, "correctness evidence"):
            MODULE._validate_correctness(
                values,
                authorization=authorization,
                plan=plan,
            )

    def test_correctness_rejects_self_attested_error_outside_tolerance(self) -> None:
        authorization, _, plan = _plan_fixture()
        values = _correctness(plan, authorization)
        values["1729"]["candidate"][
            "maximum_main_normalized_tolerance_ratio"
        ] = 1.000001
        with self.assertRaisesRegex(ValueError, "correctness evidence"):
            MODULE._validate_correctness(
                values,
                authorization=authorization,
                plan=plan,
            )

    def test_correctness_accepts_explicit_tolerance_tie_partition(self) -> None:
        authorization, _, plan = _plan_fixture()
        values = _correctness(plan, authorization)
        row = values["2718"]["reference"]
        row["main_parallel_argmax_exact_comparisons"] -= 1
        row["main_parallel_tolerance_tie_argmax_comparisons"] = 1

        MODULE._validate_correctness(
            values,
            authorization=authorization,
            plan=plan,
        )

    def test_mps_correctness_preserves_nominal_violation_as_diagnostic(self) -> None:
        authorization, _, plan = _plan_fixture()
        values = _correctness(plan, authorization)
        row = values["65537"]["reference"]
        row["maximum_main_nominal_normalized_tolerance_ratio"] = 1.05
        row["main_nominal_tolerance_violation_elements"] = 1

        MODULE._validate_correctness(
            values,
            authorization=authorization,
            plan=plan,
            comparison_contract="mps_backend",
        )

    def test_cpu_semantic_correctness_rejects_nominal_violation(self) -> None:
        authorization, _, plan = _plan_fixture()
        values = _correctness(
            plan,
            authorization,
            comparison_contract="cpu_semantic",
        )
        values["65537"]["reference"][
            "main_nominal_tolerance_violation_elements"
        ] = 1

        with self.assertRaisesRegex(ValueError, "correctness evidence"):
            MODULE._validate_correctness(
                values,
                authorization=authorization,
                plan=plan,
                comparison_contract="cpu_semantic",
            )

    def test_mps_correctness_rejects_probability_tv_outside_bound(self) -> None:
        authorization, _, plan = _plan_fixture()
        values = _correctness(plan, authorization)
        values["2718"]["candidate"][
            "maximum_main_probability_total_variation"
        ] = 1.0001e-5

        with self.assertRaisesRegex(ValueError, "correctness evidence"):
            MODULE._validate_correctness(
                values,
                authorization=authorization,
                plan=plan,
                comparison_contract="mps_backend",
            )

    def test_timing_schema_contains_counters_for_every_mode_and_role(self) -> None:
        keys = MODULE._expected_timing_keys()
        self.assertEqual(len(keys), 68)
        for mode in ("controlled_replay", "free_running_utf8_greedy"):
            for role in ("candidate", "reference"):
                self.assertIn(
                    f"{mode}__counter_explicit_device_synchronizations_inside_timing__{role}",
                    keys,
                )
                self.assertIn(f"{mode}__end_to_end_ms__{role}", keys)

    def test_free_path_correctness_count_is_reconstructed_from_outputs(self) -> None:
        authorization, _, plan = _plan_fixture()
        lengths = np.full((5, 2, 64, 5), 128, dtype=np.int64)
        rows = _free_correctness(plan, authorization)
        MODULE._validate_free_path_correctness(
            rows,
            output_lengths=lengths,
            authorization=authorization,
            plan=plan,
        )
        rows["1729"]["candidate"]["main_full_causal_position_comparisons"] -= 1
        with self.assertRaisesRegex(ValueError, "free-path correctness evidence"):
            MODULE._validate_free_path_correctness(
                rows,
                output_lengths=lengths,
                authorization=authorization,
                plan=plan,
            )

    def test_free_path_rejects_self_attested_error_outside_tolerance(self) -> None:
        authorization, _, plan = _plan_fixture()
        lengths = np.full((5, 2, 64, 5), 128, dtype=np.int64)
        rows = _free_correctness(plan, authorization)
        rows["2718"]["reference"][
            "maximum_main_normalized_tolerance_ratio"
        ] = 2.0
        with self.assertRaisesRegex(ValueError, "free-path correctness evidence"):
            MODULE._validate_free_path_correctness(
                rows,
                output_lengths=lengths,
                authorization=authorization,
                plan=plan,
            )

    def test_boundary_trace_must_be_identical_across_sessions(self) -> None:
        authorization, _, plan = _plan_fixture()
        controlled = _correctness(plan, authorization)
        free = _free_correctness(plan, authorization)
        reports = [
            {
                "correctness": copy.deepcopy(controlled),
                "cpu_semantic_correctness": _correctness(
                    plan,
                    authorization,
                    comparison_contract="cpu_semantic",
                ),
                "free_path_correctness": copy.deepcopy(free),
            }
            for _ in range(5)
        ]
        MODULE._validate_boundary_trace_stability(reports)
        reports[-1]["free_path_correctness"]["65537"]["candidate"][
            "boundary_trace_sha256"
        ] = "c" * 64
        with self.assertRaisesRegex(ValueError, "changed across sessions"):
            MODULE._validate_boundary_trace_stability(reports)

    def test_numerical_summary_discloses_nominal_mps_violations(self) -> None:
        authorization, _, plan = _plan_fixture()
        controlled = _correctness(plan, authorization)
        controlled["65537"]["reference"][
            "main_nominal_tolerance_violation_elements"
        ] = 1
        controlled["65537"]["reference"][
            "maximum_main_nominal_normalized_tolerance_ratio"
        ] = 1.05
        reports = [
            {
                "correctness": copy.deepcopy(controlled),
                "cpu_semantic_correctness": _correctness(
                    plan,
                    authorization,
                    comparison_contract="cpu_semantic",
                ),
                "free_path_correctness": _free_correctness(plan, authorization),
            }
            for _ in range(5)
        ]

        summary = MODULE._numerical_correctness_summary(reports)

        self.assertEqual(
            summary["mps_controlled_replay"][
                "main_nominal_tolerance_violation_elements"
            ],
            5,
        )
        self.assertEqual(
            summary["cpu_semantic_controlled_replay"][
                "main_nominal_tolerance_violation_elements"
            ],
            0,
        )

    def test_fresh_process_identity_allows_pid_reuse_but_not_token_reuse(self) -> None:
        _, _, plan = _plan_fixture()
        reports = []
        for index in range(5):
            reports.append(
                {
                    "environment": {
                        "device": "mps",
                        "mps_available": True,
                        **plan["runtime_environment_contract"],
                    },
                    "process": {
                        "pid": 4242,
                        "start_token_sha256": f"{index + 1:064x}",
                    },
                }
            )
        self.assertEqual(
            len(MODULE._validate_fresh_session_environments(reports, plan=plan)),
            5,
        )
        reports[-1]["process"]["start_token_sha256"] = reports[0]["process"][
            "start_token_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "fresh processes"):
            MODULE._validate_fresh_session_environments(reports, plan=plan)

    def test_deleted_summary_history_forbids_resealing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    MODULE, "OUTPUT_PATH", Path(directory) / "summary.json"
                ),
                mock.patch.object(
                    MODULE, "_tracked_history_exists", return_value=True
                ),
                mock.patch.object(
                    MODULE, "MACHINE_LOCK_PATH", Path(directory) / "machine.lock"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "forbids resealing"):
                    MODULE.run()

    def test_order_diagnostics_are_descriptive_and_balanced(self) -> None:
        sessions = []
        for _ in range(5):
            row = {}
            for mode in ("controlled_replay", "free_running_utf8_greedy"):
                for component in ("ttft_ms", "decode_ms", "end_to_end_ms"):
                    row[f"{mode}__{component}__candidate"] = np.full(
                        (5, 64, 5), 9.0
                    )
                    row[f"{mode}__{component}__reference"] = np.full(
                        (5, 64, 5), 10.0
                    )
            sessions.append(row)
        diagnostics = MODULE._execution_order_diagnostics(sessions)
        controlled = diagnostics["controlled_replay"]["end_to_end_ms"]
        self.assertTrue(controlled["descriptive_only"])
        self.assertEqual(
            controlled["candidate_first_trial_count"],
            controlled["reference_first_trial_count"],
        )
        self.assertAlmostEqual(
            controlled["candidate_first_median_paired_reduction"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
