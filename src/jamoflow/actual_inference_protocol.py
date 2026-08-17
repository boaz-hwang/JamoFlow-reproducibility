"""Single source of truth for compact actual-inference evidence."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ACTUAL_INFERENCE_PROTOCOL_VERSION = 4
ACTUAL_INFERENCE_SELECTION_ALGORITHM = (
    "JamoFlow-actual-inference-v4-one-prompt-per-document-valid-output"
)
SEEDS = (1729, 2718, 31415, 57721, 65537)
MODES = ("controlled_replay", "free_running_utf8_greedy")
ROLES = ("candidate", "reference")
COMPONENTS = ("ttft_ms", "decode_ms", "end_to_end_ms")
OUTPUT_DIAGNOSTICS = (
    "emitted_output_bytes",
    "decode_forward_steps",
    "runtime_observed_bytes",
    "overshoot_bytes",
    "valid_output_stop",
    "replacement_character_free",
    "valid_jamo_transition",
    "output_codepoints",
)
PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES = 3
FREE_RUNNING_UTF8_CONSTRAINT = (
    "shared_strict_rfc3629_transition_mask_no_horizon_closure"
)
WARMUP_CASES = 8
MEASURED_CASES = 64
REPETITIONS = 5
CORRECTNESS_CONTINUATION_BYTES = 16
TIMING_ORDER_SEED = 20_260_811
SEED_EXECUTION_ORDER_SEED = 20_260_814
TIME_TO_OUTPUT_SEMANTICS = {
    "controlled_replay": (
        "prompt prefill predicts source byte 1; exactly N-1 decode forwards "
        "score N identical held-out source bytes; no unused next-logit forward"
    ),
    "free_running_utf8_greedy": (
        "shared strict UTF-8 transition masking; stop at the first scalar "
        "boundary after emitting at least N bytes; N-1 through N+2 feedback "
        "forwards for N through N+3 emitted bytes; no horizon-closure forcing "
        "and no unused next-logit forward"
    ),
}


def decode_forward_steps(output_bytes: int = CONTINUATION_BYTES) -> int:
    if output_bytes <= 0:
        raise ValueError("output byte count must be positive")
    return output_bytes - 1


def runtime_observed_bytes(
    prompt_bytes: int = PROMPT_BYTES,
    output_bytes: int = CONTINUATION_BYTES,
) -> int:
    if prompt_bytes <= 0:
        raise ValueError("prompt byte count must be positive")
    return prompt_bytes + decode_forward_steps(output_bytes)


def free_running_maximum_output_bytes(
    minimum_output_bytes: int = CONTINUATION_BYTES,
) -> int:
    if minimum_output_bytes <= 0:
        raise ValueError("minimum output byte count must be positive")
    return minimum_output_bytes + FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES


def valid_output_overshoot(
    emitted_output_bytes: int,
    minimum_output_bytes: int = CONTINUATION_BYTES,
) -> int:
    maximum = free_running_maximum_output_bytes(minimum_output_bytes)
    if not minimum_output_bytes <= emitted_output_bytes <= maximum:
        raise ValueError("valid-output byte count is outside the UTF-8 bound")
    return emitted_output_bytes - minimum_output_bytes


def validate_output_diagnostic_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_shape: tuple[int, ...],
    prompt_bytes: int = PROMPT_BYTES,
    minimum_output_bytes: int = CONTINUATION_BYTES,
) -> None:
    """Validate per-trial time-to-output identities for every role and mode."""

    maximum_output = free_running_maximum_output_bytes(minimum_output_bytes)
    for mode in MODES:
        for role in ROLES:
            values: dict[str, np.ndarray] = {}
            for diagnostic in OUTPUT_DIAGNOSTICS:
                key = f"{mode}__{diagnostic}__{role}"
                if key not in arrays:
                    raise ValueError(f"missing output diagnostic array: {key}")
                value = np.asarray(arrays[key])
                if value.shape != expected_shape or not np.issubdtype(
                    value.dtype,
                    np.integer,
                ):
                    raise ValueError(f"malformed output diagnostic array: {key}")
                values[diagnostic] = value.astype(np.int64, copy=False)

            emitted = values["emitted_output_bytes"]
            steps = values["decode_forward_steps"]
            observed = values["runtime_observed_bytes"]
            overshoot = values["overshoot_bytes"]
            stopped = values["valid_output_stop"]
            replacement_free = values["replacement_character_free"]
            jamo_valid = values["valid_jamo_transition"]
            codepoints = values["output_codepoints"]
            if (
                np.any(stopped != 1)
                or not np.isin(replacement_free, (0, 1)).all()
                or not np.isin(jamo_valid, (0, 1)).all()
                or np.any(codepoints <= 0)
                or np.any(steps != emitted - 1)
                or np.any(observed != prompt_bytes + steps)
            ):
                raise ValueError(f"time-to-output identities differ: {mode}/{role}")
            if mode == "controlled_replay":
                valid = bool(
                    np.all(emitted == minimum_output_bytes)
                    and np.all(overshoot == 0)
                )
            else:
                valid = bool(
                    np.all(emitted >= minimum_output_bytes)
                    and np.all(emitted <= maximum_output)
                    and np.array_equal(
                        overshoot,
                        emitted - minimum_output_bytes,
                    )
                )
            if not valid:
                raise ValueError(f"output horizon contract differs: {mode}/{role}")
            patch_key = f"{mode}__global_patches__{role}"
            if patch_key in arrays:
                patches = np.asarray(arrays[patch_key])
                if (
                    patches.shape != expected_shape
                    or not np.issubdtype(patches.dtype, np.integer)
                    or np.any(patches <= 0)
                    or np.any(patches > observed)
                ):
                    raise ValueError(
                        f"global-patch diagnostics exceed observed bytes: "
                        f"{mode}/{role}"
                    )


def reconstruct_valid_completion_metrics(
    arrays: Mapping[str, np.ndarray],
    role: str,
    *,
    minimum_output_bytes: int = CONTINUATION_BYTES,
) -> dict[str, int | float]:
    """Reconstruct content-free free-running aggregates from trial arrays."""

    if role not in ROLES:
        raise ValueError("unknown actual-inference role")
    mode = "free_running_utf8_greedy"
    diagnostics = {
        name: np.asarray(arrays[f"{mode}__{name}__{role}"])
        for name in OUTPUT_DIAGNOSTICS
    }
    shapes = {value.shape for value in diagnostics.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("free-running diagnostics must be prompt-by-repetition")
    if any(
        not np.all(value == value[:, :1])
        for value in diagnostics.values()
    ):
        raise ValueError("deterministic greedy diagnostics changed across repeats")

    emitted = diagnostics["emitted_output_bytes"][:, 0].astype(np.int64)
    overshoot = diagnostics["overshoot_bytes"][:, 0].astype(np.int64)
    replacement = diagnostics["replacement_character_free"][:, 0].astype(
        np.int64
    )
    jamo = diagnostics["valid_jamo_transition"][:, 0].astype(np.int64)
    codepoints = diagnostics["output_codepoints"][:, 0].astype(np.int64)
    byte_rates = emitted.astype(np.float64) / codepoints
    count = len(emitted)
    return {
        "continuations": count,
        "minimum_completion_bytes": minimum_output_bytes,
        "valid_utf8_count": count,
        "valid_utf8_rate": 1.0,
        "replacement_character_free_count": int(replacement.sum()),
        "replacement_character_free_rate": float(replacement.mean()),
        "valid_jamo_transition_count": int(jamo.sum()),
        "valid_jamo_transition_rate": float(jamo.mean()),
        "minimum_emitted_bytes": int(emitted.min()),
        "mean_emitted_bytes": float(emitted.mean()),
        "median_emitted_bytes": float(np.median(emitted)),
        "maximum_emitted_bytes": int(emitted.max()),
        "minimum_overshoot_bytes": int(overshoot.min()),
        "mean_overshoot_bytes": float(overshoot.mean()),
        "median_overshoot_bytes": float(np.median(overshoot)),
        "maximum_overshoot_bytes": int(overshoot.max()),
        "mean_bytes_per_codepoint_valid_utf8": float(byte_rates.mean()),
        "median_bytes_per_codepoint_valid_utf8": float(np.median(byte_rates)),
    }


def timing_environment_eligible(state: Mapping[str, Any]) -> bool:
    """Require AC power, default power mode, and no thermal warning."""

    power = state.get("power")
    thermal = state.get("thermal")
    settings = state.get("settings")
    if not all(
        isinstance(item, Mapping) for item in (power, thermal, settings)
    ):
        return False
    power_output = str(power.get("stdout", ""))
    thermal_output = str(thermal.get("stdout", ""))
    settings_output = str(settings.get("stdout", ""))
    if "AC Power:" not in settings_output:
        return False
    ac_settings = settings_output.split("AC Power:", maxsplit=1)[1]
    power_modes = [
        fields[1]
        for line in ac_settings.splitlines()
        if len(fields := line.split()) == 2
        and fields[0] in {"lowpowermode", "powermode"}
    ]
    return bool(
        power.get("returncode") == 0
        and "Now drawing from 'AC Power'" in power_output
        and thermal.get("returncode") == 0
        and settings.get("returncode") == 0
        and power_modes
        and all(mode == "0" for mode in power_modes)
        and "No thermal warning level has been recorded" in thermal_output
        and "No performance warning level has been recorded" in thermal_output
    )
