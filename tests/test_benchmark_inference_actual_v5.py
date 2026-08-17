from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_inference_actual_v5.py"
SPEC = importlib.util.spec_from_file_location("benchmark_inference_actual_v5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class MainDiagnostics:
    observed_bytes: int
    emitted_data_patches: int
    local_encoder_cached_bytes: int
    global_cached_patches: int
    local_decoder_cached_bytes: int
    boundaries: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeCounters:
    parallel_prefill_calls: int
    main_consume_calls: int
    selector_observed_bytes: int
    router_forward_calls: int
    router_scored_bytes: int


class FakeRuntime:
    def __init__(self, entropy: bool = False) -> None:
        self.count = 0
        self.prefills = 0
        self.consumes = 0
        self.entropy = entropy

    @property
    def diagnostics(self) -> MainDiagnostics:
        return MainDiagnostics(
            observed_bytes=self.count,
            emitted_data_patches=1,
            local_encoder_cached_bytes=self.count,
            global_cached_patches=1,
            local_decoder_cached_bytes=self.count,
            boundaries=(0,),
        )

    @property
    def runtime_counters(self) -> RuntimeCounters:
        return RuntimeCounters(
            parallel_prefill_calls=self.prefills,
            main_consume_calls=self.consumes,
            selector_observed_bytes=self.count,
            router_forward_calls=(self.prefills + self.consumes if self.entropy else 0),
            router_scored_bytes=(self.count if self.entropy else 0),
        )

    def _logits(self) -> torch.Tensor:
        logits = torch.zeros((1, 256))
        logits[0, ord("a")] = 2
        return logits

    def prefill_parallel(self, prompt: bytes) -> torch.Tensor:
        self.count = len(prompt)
        self.prefills += 1
        return self._logits()

    def consume(self, value: int) -> torch.Tensor:
        del value
        self.count += 1
        self.consumes += 1
        return self._logits()


class FakeBundle:
    def __init__(self, entropy: bool = False) -> None:
        self.descriptor = {"fixture": True}
        self.patch_count = 1
        self.requires_entropy_router = entropy
        self.device = "cpu"
        self.runtime_policy = "fixture"

    def runtime(self) -> FakeRuntime:
        return FakeRuntime(self.requires_entropy_router)


class BenchmarkInferenceActualV5Tests(unittest.TestCase):
    def test_logit_comparison_classifies_only_tolerance_ambiguous_argmax(self) -> None:
        left = torch.zeros((1, 256), dtype=torch.float32)
        right = torch.zeros((1, 256), dtype=torch.float32)
        left[0, 130] = 6.526310920715332
        left[0, 160] = 6.526309967041016
        right[0, 130] = 6.526309967041016
        right[0, 160] = 6.526310443878174

        comparison = MODULE._compare_logits(left, right)

        self.assertEqual(comparison.exact_argmax_count, 0)
        self.assertEqual(comparison.tolerance_tie_argmax_count, 1)
        self.assertLess(comparison.maximum_normalized_tolerance_ratio, 1.0)

    def test_logit_comparison_rejects_stable_argmax_mismatch(self) -> None:
        left = torch.zeros((1, 256), dtype=torch.float32)
        right = torch.zeros((1, 256), dtype=torch.float32)
        left[0, 130] = 1.0
        right[0, 160] = 1.0

        with self.assertRaises(AssertionError):
            MODULE._compare_logits(left, right)

    def test_runner_has_fixed_paths_no_cli_or_v4_quality_input(self) -> None:
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
        self.assertFalse(any("phase3-inference-quality/summary" in value for value in strings))
        self.assertEqual(MODULE.PLAN_PATH.as_posix(), "results/phase3-inference-actual-v5r3/plan.json")

    def test_mps_device_family_accepts_indexed_device(self) -> None:
        self.assertTrue("mps:0".startswith("mps"))
        with mock.patch.object(MODULE.torch.mps, "synchronize") as synchronize:
            MODULE._synchronize("mps:0")
        synchronize.assert_called_once_with()

    def test_mps_contract_accepts_one_nominal_violation_only_with_small_tv(self) -> None:
        left = torch.zeros((1, 256), dtype=torch.float32)
        right = torch.zeros((1, 256), dtype=torch.float32)
        left[0, 17] = 2.1e-5

        comparison = MODULE._compare_logits(
            left,
            right,
            contract="mps_backend",
        )

        self.assertEqual(comparison.nominal_tolerance_violation_elements, 1)
        self.assertGreater(
            comparison.maximum_nominal_normalized_tolerance_ratio,
            1.0,
        )
        self.assertLessEqual(
            comparison.maximum_probability_total_variation,
            1e-5,
        )

    def test_cpu_contract_rejects_nominal_violation(self) -> None:
        left = torch.zeros((1, 256), dtype=torch.float32)
        right = torch.zeros((1, 256), dtype=torch.float32)
        left[0, 17] = 2.1e-5

        with self.assertRaises(AssertionError):
            MODULE._compare_logits(
                left,
                right,
                contract="cpu_semantic",
            )

    def test_mps_contract_rejects_probability_mass_drift_inside_logit_envelope(
        self,
    ) -> None:
        left = torch.empty((1, 256), dtype=torch.float32)
        left[0, :128] = 5e-5
        left[0, 128:] = -5e-5
        right = torch.zeros((1, 256), dtype=torch.float32)

        with self.assertRaisesRegex(AssertionError, "probability-distribution"):
            MODULE._compare_logits(
                left,
                right,
                contract="mps_backend",
            )

    def test_controlled_and_free_trials_expose_exact_timed_counters(self) -> None:
        prompt = b"p" * 128
        continuation = b"x" * 128
        masks = MODULE._utf8_mask_cache("cpu")
        controlled = MODULE._run_trial(
            FakeBundle(False),
            prompt,
            continuation,
            "controlled_replay",
            masks,
        )
        self.assertEqual(controlled.emitted_output_bytes, 128)
        self.assertEqual(controlled.runtime_observed_bytes, 255)
        self.assertEqual(controlled.counters["router_forward_calls"], 0)
        self.assertEqual(controlled.counters["main_consume_calls"], 127)

        free = MODULE._run_trial(
            FakeBundle(True),
            prompt,
            continuation,
            "free_running_utf8_greedy",
            masks,
        )
        self.assertEqual(free.generated, b"a" * 128)
        self.assertEqual(free.counters["router_forward_calls"], 128)
        self.assertEqual(free.counters["router_scored_bytes"], 255)
        self.assertEqual(free.counters["argmax_calls"], 128)

    def test_free_verifier_rejects_strict_utf8_bytes_that_are_not_greedy_argmax(
        self,
    ) -> None:
        prompts = np.full((1, 128), ord("p"), dtype=np.uint8)
        outputs = np.zeros((1, 131), dtype=np.uint8)
        outputs[0, :128] = ord("b")
        lengths = np.asarray([128], dtype=np.int64)

        def full_logits(_bundle, observed, _boundaries):
            logits = torch.zeros((len(observed), 256))
            logits[:, ord("a")] = 2
            return logits

        with (
            mock.patch.object(
                MODULE, "structural_prefix_boundaries", return_value=(0,)
            ),
            mock.patch.object(MODULE, "full_main_logits", side_effect=full_logits),
            mock.patch.object(
                MODULE,
                "model_spec_for_descriptor",
                return_value=SimpleNamespace(patch_stride=1),
            ),
        ):
            bundle = FakeBundle(False)
            bundle.device = "mps"
            with (
                mock.patch.object(MODULE, "_synchronize"),
                self.assertRaisesRegex(AssertionError, "masked greedy byte"),
            ):
                MODULE._verify_free_bundle(
                    bundle,
                    prompts,
                    outputs,
                    lengths,
                    MODULE._utf8_mask_cache("cpu"),
                )

    def test_session_prefix_rejects_partial_or_out_of_order_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            artifact_root = root / "artifacts"
            with (
                mock.patch.object(MODULE, "RUN_ROOT", run_root),
                mock.patch.object(MODULE, "ARTIFACT_ROOT", artifact_root),
                mock.patch.object(
                    MODULE, "SESSION_RECEIPT_ROOT", root / "receipts"
                ),
                mock.patch.object(MODULE, "_tracked_history_exists", return_value=False),
            ):
                self.assertEqual(
                    MODULE._next_session({"plan_sha256": "a" * 64}),
                    "session-01",
                )
                paths = MODULE._session_paths("session-01")
                paths["report"].parent.mkdir(parents=True)
                paths["report"].write_text("{}")
                with self.assertRaisesRegex(ValueError, "partial"):
                    MODULE._next_session({"plan_sha256": "a" * 64})

    def test_process_lock_excludes_a_second_live_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(MODULE, "ARTIFACT_ROOT", root),
                mock.patch.object(MODULE, "PROCESS_LOCK_PATH", root / ".lock"),
                mock.patch.object(MODULE, "_assert_no_symlink_namespace"),
            ):
                with MODULE._exclusive_process_lock():
                    with self.assertRaisesRegex(RuntimeError, "another"):
                        with MODULE._exclusive_process_lock():
                            self.fail("second live actual runner acquired the lock")


if __name__ == "__main__":
    unittest.main()
