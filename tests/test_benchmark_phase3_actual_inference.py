from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "benchmark_phase3_actual_inference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_phase3_actual_inference",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class Diagnostics:
    observed_bytes: int
    emitted_data_patches: int
    local_encoder_cached_bytes: int
    global_cached_patches: int
    local_decoder_cached_bytes: int
    boundaries: tuple[int, ...]


class FakeRuntime:
    def __init__(self) -> None:
        self.count = 0

    @property
    def diagnostics(self) -> Diagnostics:
        return Diagnostics(
            observed_bytes=self.count,
            emitted_data_patches=1,
            local_encoder_cached_bytes=self.count,
            global_cached_patches=1,
            local_decoder_cached_bytes=self.count,
            boundaries=(0,),
        )

    def _logits(self) -> torch.Tensor:
        logits = torch.zeros((1, 256))
        logits[0, 7] = 1
        return logits

    def prefill_parallel(self, prompt: bytes) -> torch.Tensor:
        self.count = len(prompt)
        return self._logits()

    def consume(self, value: int) -> torch.Tensor:
        self.count += 1
        return self._logits()


class FakeBundle:
    def runtime(self) -> FakeRuntime:
        return FakeRuntime()


class SequenceRuntime(FakeRuntime):
    def __init__(self, values: tuple[int, ...]) -> None:
        super().__init__()
        self.values = values
        self.position = 0

    def _logits(self) -> torch.Tensor:
        logits = torch.zeros((1, 256))
        logits[0, self.values[self.position]] = 2
        return logits

    def consume(self, value: int) -> torch.Tensor:
        self.count += 1
        self.position += 1
        return self._logits()


class SequenceBundle:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values

    def runtime(self) -> SequenceRuntime:
        return SequenceRuntime(self.values)


class ActualInferenceBenchmarkTests(unittest.TestCase):
    def test_controlled_trial_consumes_exact_replay_without_output(self) -> None:
        result = MODULE._run_trial(
            FakeBundle(),
            b"prompt",
            b"truth",
            "controlled_replay",
            "cpu",
        )
        self.assertIsNone(result.generated)
        self.assertEqual(result.observed_bytes, 10)
        self.assertEqual(result.emitted_output_bytes, 5)
        self.assertEqual(result.decode_forward_steps, 4)
        self.assertGreater(result.ttft_ms, 0)
        self.assertGreater(result.decode_ms, 0)
        self.assertAlmostEqual(
            result.end_to_end_ms,
            result.ttft_ms + result.decode_ms,
        )

    def test_free_trial_includes_greedy_feedback(self) -> None:
        result = MODULE._run_trial(
            FakeBundle(),
            b"prompt",
            b"xxxxx",
            "free_running_utf8_greedy",
            "cpu",
            MODULE._utf8_mask_cache("cpu"),
        )
        self.assertEqual(result.generated, bytes([7]) * 5)
        self.assertEqual(result.observed_bytes, 10)
        self.assertEqual(result.decode_forward_steps, 4)
        self.assertEqual(result.overshoot_bytes, 0)
        self.assertTrue(result.valid_output_stop)

    def test_trial_rejects_zero_output_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            MODULE._run_trial(
                FakeBundle(),
                b"prompt",
                b"",
                "controlled_replay",
                "cpu",
            )

    def test_one_output_byte_needs_no_decode_forward(self) -> None:
        result = MODULE._run_trial(
            FakeBundle(),
            b"prompt",
            b"x",
            "free_running_utf8_greedy",
            "cpu",
            MODULE._utf8_mask_cache("cpu"),
        )
        self.assertEqual(result.generated, bytes([7]))
        self.assertEqual(result.decode_forward_steps, 0)
        self.assertEqual(result.observed_bytes, len(b"prompt"))

    def test_free_trial_stops_at_first_valid_boundary_after_target(self) -> None:
        result = MODULE._run_trial(
            SequenceBundle((0xE2, 0x82, 0xAC)),
            b"prompt",
            b"x",
            "free_running_utf8_greedy",
            "cpu",
            MODULE._utf8_mask_cache("cpu"),
        )
        self.assertEqual(result.generated, "€".encode("utf-8"))
        self.assertEqual(result.emitted_output_bytes, 3)
        self.assertEqual(result.overshoot_bytes, 2)
        self.assertEqual(result.decode_forward_steps, 2)
        self.assertEqual(result.observed_bytes, len(b"prompt") + 2)

    def test_free_trial_masks_an_illegal_greedy_byte(self) -> None:
        result = MODULE._run_trial(
            SequenceBundle((0xFF,)),
            b"prompt",
            b"x",
            "free_running_utf8_greedy",
            "cpu",
            MODULE._utf8_mask_cache("cpu"),
        )
        self.assertNotEqual(result.generated, b"\xff")
        self.assertTrue(result.valid_output_stop)

    def test_timing_schema_has_every_mode_component_and_role(self) -> None:
        arrays = MODULE._timing_arrays(4, 3)
        self.assertEqual(set(arrays), MODULE._expected_array_keys())
        self.assertTrue(all(values.shape == (4, 3) for values in arrays.values()))

    def test_missing_system_command_is_recorded_not_raised(self) -> None:
        result = MODULE._command_snapshot(
            ["jamoflow-command-that-does-not-exist"]
        )
        self.assertIsNone(result["returncode"])
        self.assertTrue(result["stderr"])


if __name__ == "__main__":
    unittest.main()
