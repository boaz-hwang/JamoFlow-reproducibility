import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_phase3_actual_inference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_phase3_actual_inference",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def latency_component(reduction: float = 0.2) -> dict:
    return {
        "crossed_median_latency_reduction": reduction,
        "bootstrap_percentile_95_lower": 0.1,
        "median_seed_point_reduction": reduction,
        "positive_seed_count": 5,
    }


class ActualInferenceSummaryTests(unittest.TestCase):
    def _reports(
        self,
        candidate_replacement_free: int = 63,
        *,
        candidate_valid: int = 64,
    ) -> dict:
        reports = {}
        for seed in MODULE.SEEDS:
            reports[seed] = {"generation": {}}
            for role, valid, replacement_free in (
                ("candidate", candidate_valid, candidate_replacement_free),
                ("reference", 64, 64),
            ):
                reports[seed]["generation"][role] = {
                    "continuations": 64,
                    "valid_utf8_count": valid,
                    "valid_utf8_rate": valid / 64,
                    "replacement_character_free_count": replacement_free,
                    "replacement_character_free_rate": replacement_free / 64,
                }
        return reports

    def test_encoding_guard_allows_at_most_two_point_regression(self) -> None:
        result = MODULE.valid_output_guard_summary(self._reports(63))
        self.assertTrue(result["overall_pass"])
        result = MODULE.valid_output_guard_summary(self._reports(62))
        self.assertFalse(result["overall_pass"])
        result = MODULE.valid_output_guard_summary(
            self._reports(64, candidate_valid=63)
        )
        self.assertFalse(result["overall_pass"])

    def test_compact_gate_requires_both_actual_latency_modes(self) -> None:
        latency = {
            "controlled_replay": {"decode_ms": latency_component()},
            "free_running_utf8_greedy": {
                "end_to_end_ms": latency_component()
            },
        }
        gate = MODULE.compact_actual_inference_gate(
            {"overall_pass": True},
            latency,
            {"overall_pass": True},
            correctness_pass=True,
            protocol_pass=True,
        )
        self.assertTrue(gate["overall_pass"])
        latency["free_running_utf8_greedy"]["end_to_end_ms"][
            "crossed_median_latency_reduction"
        ] = 0.09
        gate = MODULE.compact_actual_inference_gate(
            {"overall_pass": True},
            latency,
            {"overall_pass": True},
            correctness_pass=True,
            protocol_pass=True,
        )
        self.assertFalse(gate["overall_pass"])

    def test_timing_schema_is_complete(self) -> None:
        self.assertEqual(len(MODULE._expected_array_keys()), 56)

    def test_protocol_repetition_count_is_shared_with_runner(self) -> None:
        self.assertEqual(MODULE.REPETITIONS, 5)

    def test_patch_diagnostic_uses_only_runtime_observed_bytes(self) -> None:
        arrays = {}
        for mode in MODULE.MODES:
            for role in MODULE.ROLES:
                arrays[f"{mode}__global_patches__{role}"] = np.ones(
                    (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS),
                    dtype=np.int64,
                )
                emitted = 128 if mode == "controlled_replay" else 130
                arrays[f"{mode}__runtime_observed_bytes__{role}"] = np.full(
                    (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS),
                    128 + emitted - 1,
                    dtype=np.int64,
                )
                arrays[f"{mode}__emitted_output_bytes__{role}"] = np.full(
                    (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS),
                    emitted,
                    dtype=np.int64,
                )
                arrays[f"{mode}__overshoot_bytes__{role}"] = np.full(
                    (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS),
                    emitted - 128,
                    dtype=np.int64,
                )
                arrays[f"{mode}__mps_current_bytes__{role}"] = np.zeros(
                    (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS),
                    dtype=np.int64,
                )
                arrays[f"{mode}__mps_driver_bytes__{role}"] = np.zeros(
                    (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS),
                    dtype=np.int64,
                )
        summary = MODULE._patch_and_memory_summary(arrays)
        for mode in MODULE.MODES:
            for role in MODULE.ROLES:
                values = summary[mode][role]
                expected = 255.0 if mode == "controlled_replay" else 257.0
                self.assertEqual(values["median_runtime_observed_bytes"], expected)
                self.assertEqual(values["median_bytes_per_global_patch"], expected)

    def test_encoding_rates_remain_probabilities(self) -> None:
        result = MODULE.valid_output_guard_summary(self._reports(63))
        for role in MODULE.ROLES:
            values = result["by_role"][role]
            self.assertGreaterEqual(values["valid_utf8_rate"], 0.0)
            self.assertLessEqual(values["valid_utf8_rate"], 1.0)
            self.assertGreaterEqual(
                values["replacement_character_free_rate"],
                0.0,
            )
            self.assertLessEqual(
                values["replacement_character_free_rate"],
                1.0,
            )

    def test_throughput_diagnostics_use_actual_emitted_units(self) -> None:
        shape = (len(MODULE.SEEDS), MODULE.MEASURED_CASES, MODULE.REPETITIONS)
        arrays = {}
        for mode in MODULE.MODES:
            for role in MODULE.ROLES:
                emitted = 128 if mode == "controlled_replay" else 130
                arrays[f"{mode}__emitted_output_bytes__{role}"] = np.full(
                    shape,
                    emitted,
                    dtype=np.int64,
                )
                arrays[f"{mode}__output_codepoints__{role}"] = np.full(
                    shape,
                    64,
                    dtype=np.int64,
                )
                arrays[f"{mode}__decode_ms__{role}"] = np.full(
                    shape,
                    10.0,
                )
                arrays[f"{mode}__end_to_end_ms__{role}"] = np.full(
                    shape,
                    20.0,
                )
        result = MODULE._throughput_diagnostics(arrays)
        free = result["free_running_utf8_greedy"]["candidate"]
        self.assertEqual(
            free["emitted_bytes_per_decode_second"]["median"],
            13_000.0,
        )
        self.assertEqual(
            free["unicode_codepoints_per_decode_second"]["median"],
            6_400.0,
        )

    def test_in_progress_seed_blocks_stale_artifact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            seed = MODULE.SEEDS[0]
            (run_root / f"seed-{seed}.in-progress.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                MODULE._load_seed(
                    seed,
                    0,
                    run_root,
                    run_root,
                    {},
                    run_root / "quality.json",
                    run_root / "selection.json",
                    np.zeros((len(MODULE.SEEDS), 1, 1, 1), dtype=np.uint8),
                    object(),
                )


if __name__ == "__main__":
    unittest.main()
