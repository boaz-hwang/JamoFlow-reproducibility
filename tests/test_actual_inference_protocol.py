import unittest

import numpy as np

from jamoflow.actual_inference_protocol import (
    CONTINUATION_BYTES,
    FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES,
    MODES,
    OUTPUT_DIAGNOSTICS,
    PROMPT_BYTES,
    REPETITIONS,
    ROLES,
    decode_forward_steps,
    free_running_maximum_output_bytes,
    reconstruct_valid_completion_metrics,
    runtime_observed_bytes,
    timing_environment_eligible,
    valid_output_overshoot,
    validate_output_diagnostic_arrays,
)


def session_state(
    *,
    power: str = "Now drawing from 'AC Power'",
    thermal: str = (
        "No thermal warning level has been recorded\n"
        "No performance warning level has been recorded"
    ),
    power_mode: str = "0",
) -> dict:
    return {
        "power": {"returncode": 0, "stdout": power},
        "thermal": {"returncode": 0, "stdout": thermal},
        "settings": {
            "returncode": 0,
            "stdout": f"Battery Power:\n powermode 0\nAC Power:\n powermode {power_mode}",
        },
    }


class ActualInferenceProtocolTests(unittest.TestCase):
    def test_controlled_horizon_is_exact_and_valid_output_can_overshoot(self) -> None:
        self.assertEqual(REPETITIONS, 5)
        self.assertEqual(decode_forward_steps(CONTINUATION_BYTES), 127)
        self.assertEqual(
            runtime_observed_bytes(PROMPT_BYTES, CONTINUATION_BYTES),
            255,
        )
        self.assertEqual(FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES, 3)
        self.assertEqual(free_running_maximum_output_bytes(), 131)
        self.assertEqual(valid_output_overshoot(130), 2)

    def test_one_output_byte_requires_no_feedback_forward(self) -> None:
        self.assertEqual(decode_forward_steps(1), 0)
        self.assertEqual(runtime_observed_bytes(PROMPT_BYTES, 1), PROMPT_BYTES)

    def test_invalid_byte_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "output byte count"):
            decode_forward_steps(0)
        with self.assertRaisesRegex(ValueError, "prompt byte count"):
            runtime_observed_bytes(0, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            valid_output_overshoot(132)

    def test_output_diagnostics_enforce_mode_specific_horizons(self) -> None:
        shape = (2, 3)
        arrays = {}
        for mode in MODES:
            for role in ROLES:
                emitted = 128 if mode == "controlled_replay" else 130
                values = {
                    "emitted_output_bytes": emitted,
                    "decode_forward_steps": emitted - 1,
                    "runtime_observed_bytes": PROMPT_BYTES + emitted - 1,
                    "overshoot_bytes": (
                        0 if mode == "controlled_replay" else emitted - 128
                    ),
                    "valid_output_stop": 1,
                    "replacement_character_free": 1,
                    "valid_jamo_transition": 1,
                    "output_codepoints": 43,
                }
                for diagnostic in OUTPUT_DIAGNOSTICS:
                    arrays[f"{mode}__{diagnostic}__{role}"] = np.full(
                        shape,
                        values[diagnostic],
                        dtype=np.int32,
                    )
                arrays[f"{mode}__global_patches__{role}"] = np.ones(
                    shape,
                    dtype=np.int32,
                )
        validate_output_diagnostic_arrays(arrays, expected_shape=shape)
        reconstructed = reconstruct_valid_completion_metrics(
            arrays,
            "candidate",
        )
        self.assertEqual(reconstructed["mean_emitted_bytes"], 130.0)
        self.assertEqual(reconstructed["maximum_overshoot_bytes"], 2)

        tampered = {key: value.copy() for key, value in arrays.items()}
        tampered[
            "free_running_utf8_greedy__runtime_observed_bytes__candidate"
        ][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "identities"):
            validate_output_diagnostic_arrays(tampered, expected_shape=shape)

        tampered = {key: value.copy() for key, value in arrays.items()}
        tampered["controlled_replay__global_patches__candidate"][0, 0] = 999
        with self.assertRaisesRegex(ValueError, "global-patch"):
            validate_output_diagnostic_arrays(tampered, expected_shape=shape)

    def test_timing_environment_requires_ac_and_no_warnings(self) -> None:
        self.assertTrue(timing_environment_eligible(session_state()))
        self.assertFalse(
            timing_environment_eligible(
                session_state(power="Now drawing from 'Battery Power'")
            )
        )
        self.assertFalse(
            timing_environment_eligible(
                session_state(thermal="Thermal warning level: 1")
            )
        )
        self.assertFalse(
            timing_environment_eligible(session_state(power_mode="1"))
        )
        self.assertFalse(timing_environment_eligible({}))


if __name__ == "__main__":
    unittest.main()
