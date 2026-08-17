from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_CASE_PATH,
    ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
    ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
    ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES,
    ACTUAL_INFERENCE_V5_MEASURED_CASES,
    ACTUAL_INFERENCE_V5_REPETITIONS,
    ACTUAL_INFERENCE_V5_ROLES,
    ACTUAL_INFERENCE_V5_SESSIONS,
    RUNTIME_COUNTER_NAMES,
    actual_efficiency_component_pass,
    assert_workspace_path_no_symlinks,
    build_actual_inference_plan_v5,
    session_schedule,
    three_way_paired_latency,
    validate_actual_inference_plan_v5,
    validate_free_output_bytes,
    validate_isolated_memory_receipt,
    validate_runtime_counter_arrays,
)
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_SEEDS,
    canonical_sha256,
)
from jamoflow.inference_final_quality_lock_v2 import (
    FINAL_QUALITY_LOCK_KIND,
    PRIMARY_TIMING_AUTHORIZATION_KEY,
)
from jamoflow.incremental_blt import structural_prefix_boundaries
from tests.test_inference_final_authorization_v2 import (
    InferenceFinalAuthorizationV2Tests,
    digest,
)


def _quality_lock(authorization: dict, authorization_artifact_sha: str) -> dict:
    candidate = next(
        model for model in authorization["models"]
        if model["artifact_role"] == "candidate"
    )
    reference = next(
        model for model in authorization["models"]
        if model["artifact_role"] == "matched_efficiency_baseline"
    )
    matched_gate = {"overall_pass": True}
    mechanism_gate = {"overall_pass": False}
    gate = {
        "actual_timing_authorized": True,
        "broad_actual_timing_authorized": False,
        "broad_candidate_vs_strongest_reference": None,
        "bootstrap_repetitions": 10_000,
        "bootstrap_seed": 20_260_814,
        "candidate_vs_matched_efficiency_baseline": matched_gate,
        "evaluated_role_order": [
            "candidate",
            "matched_efficiency_baseline",
            "same_rate_codepoint_control",
        ],
        "mechanism_candidate_vs_same_rate_codepoint": mechanism_gate,
        "mechanism_timing_authorized": False,
        "matched_quality_timing_authorized": True,
        "overall_pass": False,
        "seed_order": list(FINAL_SEEDS),
        "sequence_count": 62_500,
        "status": "fail_final_quality_v2",
        "targets_per_sequence": 511,
    }
    gate_sha = canonical_sha256(gate)
    model_order = [model["identity_sha256"] for model in authorization["models"]]
    nll_artifacts = []
    per_receipt = []
    for model in authorization["models"]:
        for seed in FINAL_SEEDS:
            artifact_role = model["artifact_role"]
            receipt_sha = digest(f"receipt/{artifact_role}/{seed}")
            array_sha = digest(f"array/{artifact_role}/{seed}")
            nll_artifacts.append(
                {
                    "array_sha256": array_sha,
                    "artifact_path": (
                        f"artifacts/phase3-inference-final-v2/seed-{seed}/"
                        f"{artifact_role}-nll.npz"
                    ),
                    "artifact_sha256": digest(f"nll/{artifact_role}/{seed}"),
                    "artifact_role": artifact_role,
                    "receipt_sha256": receipt_sha,
                    "seed": seed,
                }
            )
            per_receipt.append(
                {"array_sha256": array_sha, "receipt_sha256": receipt_sha}
            )
    pair = {
        "authorized": True,
        "criterion": "sealed final 0.010 BPB noninferiority",
        "left_artifact_role": candidate["artifact_role"],
        "left_logical_role": "candidate",
        "left_model_identity_sha256": candidate["identity_sha256"],
        "quality_gate_sha256": gate_sha,
        "right_artifact_role": reference["artifact_role"],
        "right_logical_role": "matched_efficiency_baseline",
        "right_model_identity_sha256": reference["identity_sha256"],
    }
    control = dict(pair)
    control.update(
        {
            "authorized": False,
            "left_logical_role": "candidate",
            "right_artifact_role": "same_rate_codepoint_control",
            "right_logical_role": "same_rate_codepoint_control",
            "right_model_identity_sha256": next(
                model["identity_sha256"]
                for model in authorization["models"]
                if model["artifact_role"] == "same_rate_codepoint_control"
            ),
        }
    )
    payload = {
        "authorization_artifact": {
            "git_commit": "d" * 40,
            "path": FINAL_AUTHORIZATION_PATH,
            "sha256": authorization_artifact_sha,
        },
        "broad_reference_policy": None,
        "document_window_map": {},
        "evidence_artifact": {
            "git_commit": "c" * 40,
            "path": "results/phase3-inference-final-v2/evidence-manifest.json",
            "sha256": digest("evidence-artifact"),
        },
        "evidence_manifest_sha256": digest("evidence-manifest"),
        "final_quality_gate": gate,
        "final_quality_gate_sha256": gate_sha,
        "independent_nll_recomputation": {
            "batch_size": 64,
            "comparison": "bitwise_equal_float32_array_sha256",
            "device": "mps",
            "pass": True,
            "per_receipt": per_receipt,
            "runtime": {"device": "mps"},
            "was_predeclared_before_first_final_loss": True,
        },
        "kind": FINAL_QUALITY_LOCK_KIND,
        "model_identity_order": model_order,
        "nll_artifacts": nll_artifacts,
        "primary_publication_timing_authorized": True,
        "primary_timing_authorization_key": PRIMARY_TIMING_AUTHORIZATION_KEY,
        "protocol_version": 2,
        "quality_lock_base_git_commit": "b" * 40,
        "quality_lock_path": FINAL_QUALITY_LOCK_PATH,
        "schema_version": 2,
        "seed_order": list(FINAL_SEEDS),
        "selection_lock_sha256": digest("selection-lock"),
        "session_plan": {
            "path": "artifacts/phase3-inference-final-v2/session-plan.json",
            "sha256": digest("session-plan"),
        },
        "status": "pass_matched_quality_only",
        "timing_authorizations": {
            PRIMARY_TIMING_AUTHORIZATION_KEY: pair,
            "candidate_vs_same_rate_codepoint_control": control,
        },
    }
    payload["quality_lock_sha256"] = canonical_sha256(payload)
    return payload


def _plan_fixture() -> tuple[dict, dict, dict]:
    _, authorization = InferenceFinalAuthorizationV2Tests()._authorization()
    auth_artifact_sha = digest("authorization-artifact")
    quality = _quality_lock(authorization, auth_artifact_sha)
    plan = build_actual_inference_plan_v5(
        quality_lock=quality,
        authorization=authorization,
        quality_lock_artifact={
            "git_commit": "e" * 40,
            "path": FINAL_QUALITY_LOCK_PATH,
            "sha256": digest("quality-lock-artifact"),
        },
        authorization_artifact={
            "git_commit": "d" * 40,
            "path": FINAL_AUTHORIZATION_PATH,
            "sha256": auth_artifact_sha,
        },
        case_context={
            "artifact_path": ACTUAL_INFERENCE_V5_CASE_PATH,
            "artifact_sha256": digest("cases-artifact"),
            "case_selection_sha256": digest("case-selection"),
            "continuation_array_sha256": digest("continuations"),
            "document_assignment_sha256": digest("documents"),
            "prompt_array_sha256": digest("prompts"),
            "selected_unique_documents": 72,
            "total_cases": 72,
        },
        implementation_sha256={
            path: digest(path)
            for path in ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER
        },
        implementation_order=ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
        plan_base_git_commit="f" * 40,
        runtime_environment_contract={
            "hardware": {
                "chip": "Apple Test Chip",
                "machine_model": "MacTest1,1",
                "memory_bytes": 64 * 1024**3,
                "os_build": "25A000",
            },
            "machine": "arm64",
            "packages": {
                "numpy": "2.5.2",
                "tokenizers": "0.22.2",
                "torch": "2.13.0",
                "transformers": "5.14.1",
                "zstandard": "0.25.0",
            },
            "platform": "macOS-test",
            "python": "3.13.11",
            "system": "Darwin",
        },
    )
    return authorization, quality, plan


class InferenceActualV5Tests(unittest.TestCase):
    def test_workspace_path_rejects_a_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            link = root / "link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                assert_workspace_path_no_symlinks(link / "unit.json")

    def test_actual_v5_implementation_manifest_is_complete_and_unique(self) -> None:
        self.assertEqual(
            len(ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER),
            len(set(ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER)),
        )
        self.assertTrue(
            all(
                Path(path).is_file()
                for path in ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER
            )
        )
        package = {
            path.as_posix() for path in Path("src/jamoflow").glob("*.py")
        }
        self.assertTrue(package <= set(ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER))

    def test_plan_binds_exact_quality_authorized_pair_and_sessions(self) -> None:
        authorization, quality, plan = _plan_fixture()
        validate_actual_inference_plan_v5(
            plan,
            quality_lock=quality,
            authorization=authorization,
        )
        self.assertEqual(plan["protocol_version"], 5)
        self.assertEqual(plan["protocol_revision"], 3)
        self.assertEqual(
            plan["timing_pair"]["authorization_key"],
            PRIMARY_TIMING_AUTHORIZATION_KEY,
        )
        self.assertEqual(
            tuple(row["session_id"] for row in plan["session_schedules"]),
            ACTUAL_INFERENCE_V5_SESSIONS,
        )
        self.assertEqual(
            plan["protocol"]["statistical_unit"],
            "session x model-seed x prompt",
        )
        self.assertTrue(plan["protocol"]["repetitions_are_within_cell_only"])
        self.assertEqual(plan["protocol"]["patching_horizon"], 512)
        self.assertEqual(plan["protocol"]["global_position_limit"], 1_032)
        self.assertTrue(plan["case_selection_contract"]["sealed_after_final_quality"])
        self.assertTrue(
            plan["case_selection_contract"][
                "outcome_sensitive_logic_matches_pre_final_evaluator_commit"
            ]
        )
        self.assertEqual(
            plan["case_selection_contract"]["post_final_correctness_revision"],
            "data/manifests/phase3-inference-actual-v5r3-device-identity-erratum.json",
        )
        self.assertEqual(
            plan["protocol"]["correctness_contract"]["cpu_semantic"],
            {
                "atol": 2e-5,
                "device": "cpu",
                "required_for_every_seed_role_session": True,
                "rtol": 2e-5,
            },
        )
        self.assertEqual(
            plan["protocol"]["correctness_contract"]["mps_backend"][
                "maximum_probability_total_variation"
            ],
            1e-5,
        )
        self.assertFalse(
            plan["protocol"]["correctness_contract"][
                "third_tolerance_relaxation_allowed"
            ]
        )
        self.assertFalse(plan["case_selection_contract"]["uses_prior_latency"])
        self.assertFalse(
            plan["case_selection_contract"]["case_selection_uses_final_nll"]
        )

    def test_plan_rejects_unauthorized_or_rotated_primary_pair(self) -> None:
        authorization, quality, _ = _plan_fixture()
        quality["primary_publication_timing_authorized"] = False
        quality["quality_lock_sha256"] = canonical_sha256(
            {key: value for key, value in quality.items() if key != "quality_lock_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "timing status"):
            build_actual_inference_plan_v5(
                quality_lock=quality,
                authorization=authorization,
                quality_lock_artifact={
                    "git_commit": "e" * 40,
                    "path": FINAL_QUALITY_LOCK_PATH,
                    "sha256": digest("quality-lock-artifact"),
                },
                authorization_artifact={
                    "git_commit": "d" * 40,
                    "path": FINAL_AUTHORIZATION_PATH,
                    "sha256": digest("authorization-artifact"),
                },
                case_context={
                    "artifact_path": ACTUAL_INFERENCE_V5_CASE_PATH,
                    "artifact_sha256": digest("cases-artifact"),
                    "case_selection_sha256": digest("case-selection"),
                    "continuation_array_sha256": digest("continuations"),
                    "document_assignment_sha256": digest("documents"),
                    "prompt_array_sha256": digest("prompts"),
                    "selected_unique_documents": 72,
                    "total_cases": 72,
                },
                implementation_sha256={
                    path: digest(path)
                    for path in ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER
                },
                implementation_order=ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
                plan_base_git_commit="f" * 40,
                runtime_environment_contract={
                    "hardware": {
                        "chip": "Apple Test Chip",
                        "machine_model": "MacTest1,1",
                        "memory_bytes": 64 * 1024**3,
                        "os_build": "25A000",
                    },
                    "machine": "arm64",
                    "packages": {
                        "numpy": "2.5.2",
                        "tokenizers": "0.22.2",
                        "torch": "2.13.0",
                        "transformers": "5.14.1",
                        "zstandard": "0.25.0",
                    },
                    "platform": "macOS-test",
                    "python": "3.13.11",
                    "system": "Darwin",
                },
            )

    def test_session_schedules_are_balanced_distinct_and_deterministic(self) -> None:
        hashes = set()
        for index in range(5):
            first = session_schedule(index)
            second = session_schedule(index)
            np.testing.assert_array_equal(
                first["candidate_first"], second["candidate_first"]
            )
            np.testing.assert_array_equal(first["seed_order"], second["seed_order"])
            self.assertEqual(set(first["seed_order"].tolist()), set(FINAL_SEEDS))
            row = first["candidate_first"]
            self.assertEqual(int(row.sum()) * 2, row.size)
            hashes.add(hashlib.sha256(row.tobytes()).hexdigest())
        self.assertEqual(len(hashes), 5)

    def test_patching_horizon_is_training_geometry_not_position_capacity(self) -> None:
        observed = b"a" * 258
        trained = structural_prefix_boundaries(
            observed,
            "causal_codepoint_grid",
            horizon=512,
            patch_count=86,
            fixed_stride=6,
        )
        wrong_capacity_horizon = structural_prefix_boundaries(
            observed,
            "causal_codepoint_grid",
            horizon=1_032,
            patch_count=86,
            fixed_stride=6,
        )
        self.assertNotEqual(trained, wrong_capacity_horizon)
        self.assertEqual(_plan_fixture()[2]["protocol"]["patching_horizon"], 512)

    def test_plan_rejects_an_incomplete_implementation_manifest(self) -> None:
        authorization, quality, plan = _plan_fixture()
        plan["implementation_sha256"].pop(
            "scripts/benchmark_inference_actual_v5.py"
        )
        plan["plan_sha256"] = canonical_sha256(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "lineage"):
            validate_actual_inference_plan_v5(
                plan,
                quality_lock=quality,
                authorization=authorization,
            )

    def test_plan_rejects_a_quality_lock_without_independent_nll_evidence(self) -> None:
        authorization, quality, plan = _plan_fixture()
        quality["independent_nll_recomputation"] = {}
        quality["quality_lock_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in quality.items()
                if key != "quality_lock_sha256"
            }
        )
        plan["quality_lock_sha256"] = quality["quality_lock_sha256"]
        plan["plan_sha256"] = canonical_sha256(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "independent evidence"):
            validate_actual_inference_plan_v5(
                plan,
                quality_lock=quality,
                authorization=authorization,
            )

    def test_three_way_bootstrap_collapses_repetitions_and_passes_speedup(self) -> None:
        shape = (5, 5, 64, 5)
        reference = np.full(shape, 10.0)
        candidate = np.full(shape, 8.5)
        candidate[..., 0] = 1000.0
        candidate[..., 1] = 0.1
        summary = three_way_paired_latency(
            candidate,
            reference,
            bootstrap_repetitions=200,
        )
        self.assertAlmostEqual(summary.crossed_median_latency_reduction, 0.15)
        self.assertEqual(summary.positive_session_count, 5)
        self.assertEqual(summary.target_session_count, 5)
        self.assertEqual(summary.positive_seed_count, 5)
        self.assertTrue(actual_efficiency_component_pass(summary))

    def test_three_way_gate_requires_session_and_seed_stability(self) -> None:
        summary = {
            "bootstrap_percentile_95_lower": 0.01,
            "crossed_median_latency_reduction": 0.12,
            "positive_seed_count": 5,
            "positive_session_count": 4,
            "target_session_count": 3,
            "median_seed_point_reduction": 0.12,
        }
        self.assertFalse(actual_efficiency_component_pass(summary))
        summary["positive_session_count"] = 5
        summary["positive_seed_count"] = 3
        self.assertFalse(actual_efficiency_component_pass(summary))

    def test_three_way_gate_rejects_each_remaining_threshold_independently(self) -> None:
        passing = {
            "bootstrap_percentile_95_lower": 0.01,
            "crossed_median_latency_reduction": 0.12,
            "positive_seed_count": 5,
            "positive_session_count": 5,
            "target_session_count": 3,
            "median_seed_point_reduction": 0.12,
        }
        self.assertTrue(actual_efficiency_component_pass(passing))
        for key, failing_value in (
            ("bootstrap_percentile_95_lower", 0.0),
            ("crossed_median_latency_reduction", 0.099999),
            ("target_session_count", 2),
            ("median_seed_point_reduction", 0.099999),
        ):
            failing = dict(passing)
            failing[key] = failing_value
            self.assertFalse(
                actual_efficiency_component_pass(failing),
                msg=f"gate accepted failing {key}",
            )

    def _counters(self, emitted: np.ndarray, *, entropy: bool, free: bool) -> dict:
        consume = emitted.astype(np.int64) - 1
        observed = 128 + consume
        counters = {
            name: np.zeros(emitted.shape, dtype=np.int64)
            for name in RUNTIME_COUNTER_NAMES
        }
        counters["parallel_prefill_calls"].fill(1)
        counters["main_consume_calls"][:] = consume
        counters["selector_observed_bytes"][:] = observed
        counters["explicit_device_synchronizations_inside_timing"].fill(2)
        counters["device_to_host_readbacks_inside_timing"][:] = (
            (emitted if free else 0) + (1 + consume if entropy else 0)
        )
        if entropy:
            counters["router_forward_calls"][:] = 1 + consume
            counters["router_scored_bytes"][:] = observed
        if free:
            for name in (
                "argmax_calls",
                "utf8_mask_calls",
                "utf8_dfa_advances",
                "stop_checks",
            ):
                counters[name][:] = emitted
        return counters

    def test_runtime_counters_distinguish_structural_and_entropy_paths(self) -> None:
        emitted = np.full((64, 5), 128, dtype=np.int64)
        structural = self._counters(emitted, entropy=False, free=False)
        validate_runtime_counter_arrays(
            structural,
            requires_entropy_router=False,
            mode="controlled_replay",
            emitted_output_bytes=emitted,
        )
        entropy = self._counters(emitted, entropy=True, free=True)
        validate_runtime_counter_arrays(
            entropy,
            requires_entropy_router=True,
            mode="free_running_utf8_greedy",
            emitted_output_bytes=emitted,
        )
        entropy["router_scored_bytes"][0, 0] -= 1
        with self.assertRaisesRegex(ValueError, "router"):
            validate_runtime_counter_arrays(
                entropy,
                requires_entropy_router=True,
                mode="free_running_utf8_greedy",
                emitted_output_bytes=emitted,
            )

    def _outputs(self) -> tuple[np.ndarray, np.ndarray]:
        prefix = (
            len(ACTUAL_INFERENCE_V5_SESSIONS),
            len(FINAL_SEEDS),
            len(ACTUAL_INFERENCE_V5_ROLES),
            ACTUAL_INFERENCE_V5_MEASURED_CASES,
            ACTUAL_INFERENCE_V5_REPETITIONS,
        )
        values = np.zeros((*prefix, ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES), dtype=np.uint8)
        values[..., :ACTUAL_INFERENCE_V5_CONTINUATION_BYTES] = ord("a")
        lengths = np.full(prefix, ACTUAL_INFERENCE_V5_CONTINUATION_BYTES, dtype=np.int64)
        return values, lengths

    def test_raw_outputs_are_replayed_and_committed(self) -> None:
        outputs, lengths = self._outputs()
        evidence = validate_free_output_bytes(outputs, lengths)
        self.assertEqual(evidence["output_count"], 5 * 5 * 2 * 64 * 5)
        self.assertTrue(evidence["all_outputs_strict_utf8"])
        self.assertEqual(evidence["replacement_character_free_count"], evidence["output_count"])

    def test_output_must_stop_at_first_boundary_and_match_across_sessions(self) -> None:
        outputs, lengths = self._outputs()
        lengths[0, 0, 0, 0, 0] = 129
        outputs[0, 0, 0, 0, 0, 128] = ord("b")
        with self.assertRaisesRegex(ValueError, "first valid boundary"):
            validate_free_output_bytes(outputs, lengths)

        outputs, lengths = self._outputs()
        outputs[1, 0, 0, 0, 0, 0] = ord("b")
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_free_output_bytes(outputs, lengths)

        outputs, lengths = self._outputs()
        outputs[0, 0, 0, 0, 0, 130] = 1
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_free_output_bytes(outputs, lengths)

    def test_isolated_memory_receipt_requires_real_peak_ordering(self) -> None:
        receipt = {
            "backend": "isolated-process-ru_maxrss-macos",
            "checkpoint_state_sha256": digest("checkpoint"),
            "mps_snapshots": {
                "after_inference_current_bytes": 350,
                "after_inference_driver_bytes": 450,
                "after_load_current_bytes": 300,
                "after_load_driver_bytes": 400,
                "after_release_current_bytes": 10,
                "after_release_driver_bytes": 20,
                "baseline_current_bytes": 0,
                "baseline_driver_bytes": 0,
            },
            "measurement_git_commit": "a" * 40,
            "model_identity_sha256": digest("model"),
            "parameter_bytes": 200,
            "plan_sha256": digest("plan"),
            "process_rss": {
                "after_inference_bytes": 400,
                "after_model_load_bytes": 300,
                "baseline_bytes": 100,
                "high_water_bytes": 400,
                "unit": "bytes_on_macos",
            },
            "receipt_sha256": "0" * 64,
            "resettable_peak_supported": False,
            "role": "candidate",
            "router_checkpoint_state_sha256": None,
            "seed": 1729,
            "workload": {
                "case_artifact_sha256": digest("cases"),
                "continuation_bytes": 128,
                "measured_cases": 64,
                "mode": "free_running_utf8_greedy",
                "prompt_bytes": 128,
                "prompt_array_sha256": digest("prompts"),
                "repetitions": 1,
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
        validate_isolated_memory_receipt(
            receipt,
            role="candidate",
            model_identity_sha256=digest("model"),
            seed=1729,
            plan_sha256=digest("plan"),
            expected_checkpoint_state_sha256=digest("checkpoint"),
            expected_router_checkpoint_state_sha256=None,
            expected_parameter_bytes=200,
        )
        receipt["process_rss"]["after_inference_bytes"] = 299
        receipt["process_rss"]["high_water_bytes"] = 299
        receipt["receipt_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "ordering"):
            validate_isolated_memory_receipt(
                receipt,
                role="candidate",
                model_identity_sha256=digest("model"),
                seed=1729,
                plan_sha256=digest("plan"),
                expected_checkpoint_state_sha256=digest("checkpoint"),
                expected_router_checkpoint_state_sha256=None,
                expected_parameter_bytes=200,
            )


if __name__ == "__main__":
    unittest.main()
