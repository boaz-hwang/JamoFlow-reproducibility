"""Pure contracts and statistics for trained fresh-vocabulary actual inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PROTOCOL_ID = "jamoflow-fresh-vocabulary-actual-one-seed-v1"
ROLES = ("candidate", "reference")
MODES = ("controlled_replay", "free_running_utf8_greedy")

PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
WARMUP_CASES = 8
MEASURED_CASES = 64
REPETITIONS = 5

BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_841
MINIMUM_END_TO_END_POINT_REDUCTION = 0.10
MINIMUM_BOOTSTRAP_LOWER_REDUCTION = 0.0
MINIMUM_POSITIVE_PROMPTS = 48

TIMING_COMPONENTS = (
    "tokenizer_ms",
    "ttft_ms",
    "decode_ms",
    "model_loop_ms",
    "end_to_end_ms",
)


def balanced_role_order(
    case_index: int,
    repetition_index: int,
    mode_index: int,
) -> tuple[int, int]:
    """Alternate which physical model is timed first in every paired cell."""

    if (
        not 0 <= case_index < MEASURED_CASES
        or not 0 <= repetition_index < REPETITIONS
        or not 0 <= mode_index < len(MODES)
    ):
        raise ValueError("fresh actual schedule coordinate differs")
    first = (case_index + repetition_index + mode_index) % len(ROLES)
    return (first, 1 - first)


def paired_prompt_bootstrap(
    candidate_prompt_medians: np.ndarray,
    reference_prompt_medians: np.ndarray,
    *,
    seed_offset: int,
) -> tuple[float, float]:
    candidate = np.asarray(candidate_prompt_medians, dtype=np.float64)
    reference = np.asarray(reference_prompt_medians, dtype=np.float64)
    if (
        candidate.shape != (MEASURED_CASES,)
        or reference.shape != candidate.shape
        or not np.isfinite(candidate).all()
        or not np.isfinite(reference).all()
        or np.any(candidate <= 0)
        or np.any(reference <= 0)
        or not isinstance(seed_offset, int)
        or seed_offset < 0
    ):
        raise ValueError("fresh actual bootstrap inputs differ")
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    chunk = 256
    for start in range(0, BOOTSTRAP_REPETITIONS, chunk):
        size = min(chunk, BOOTSTRAP_REPETITIONS - start)
        rows = rng.integers(
            0,
            MEASURED_CASES,
            size=(size, MEASURED_CASES),
        )
        estimates[start : start + size] = 1.0 - (
            np.median(candidate[rows], axis=1)
            / np.median(reference[rows], axis=1)
        )
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _timing_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if (
        array.dtype != np.float64
        or array.shape != expected
        or not np.isfinite(array).all()
        or np.any(array <= 0)
    ):
        raise ValueError("fresh actual timing array differs")
    return array


def _count_array(value: np.ndarray, *, minimum: int) -> np.ndarray:
    array = np.asarray(value)
    expected = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
    if (
        array.dtype != np.int16
        or array.shape != expected
        or np.any(array < minimum)
    ):
        raise ValueError("fresh actual count array differs")
    return array


def _correctness_pass(
    correctness: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> bool:
    if set(correctness) != set(ROLES):
        return False
    required = {
        "argmax_comparisons",
        "argmax_exact",
        "cases",
        "comparisons",
        "maximum_normalized_tolerance_ratio",
        "pass",
    }
    for role in ROLES:
        if set(correctness[role]) != set(MODES):
            return False
        for mode in MODES:
            row = correctness[role][mode]
            maximum = float(row.get("maximum_normalized_tolerance_ratio", math.nan))
            comparisons = int(row.get("comparisons", -1))
            if (
                set(row) != required
                or row.get("pass") is not True
                or int(row.get("cases", -1)) != WARMUP_CASES
                or comparisons <= 0
                or int(row.get("argmax_comparisons", -1)) != comparisons
                or int(row.get("argmax_exact", -1)) != comparisons
                or not math.isfinite(maximum)
                or not 0.0 <= maximum <= 1.0
            ):
                return False
    return True


def summarize_actual_preflight(
    *,
    timing: Mapping[str, np.ndarray],
    output_token_count: np.ndarray,
    output_raw_byte_count: np.ndarray,
    correctness: Mapping[str, Mapping[str, Mapping[str, Any]]],
    maximum_output_bytes_by_role: Mapping[str, int],
) -> dict[str, Any]:
    if set(timing) != set(TIMING_COMPONENTS):
        raise ValueError("fresh actual timing component set differs")
    arrays = {name: _timing_array(timing[name]) for name in TIMING_COMPONENTS}
    tokens = _count_array(output_token_count, minimum=1)
    raw_bytes = _count_array(output_raw_byte_count, minimum=CONTINUATION_BYTES)
    if set(maximum_output_bytes_by_role) != set(ROLES):
        raise ValueError("fresh actual maximum output role set differs")
    for role_index, role in enumerate(ROLES):
        maximum = int(maximum_output_bytes_by_role[role])
        if maximum < CONTINUATION_BYTES or np.any(raw_bytes[..., role_index] > maximum):
            raise ValueError("fresh actual output byte bound differs")
        if np.any(raw_bytes[MODES.index("controlled_replay"), ..., role_index] != CONTINUATION_BYTES):
            raise ValueError("fresh actual controlled byte count differs")

    correctness_pass = _correctness_pass(correctness)
    end_to_end = arrays["end_to_end_ms"]
    cell_medians = np.median(end_to_end, axis=2)
    modes: dict[str, Any] = {}
    for mode_index, mode in enumerate(MODES):
        candidate = cell_medians[mode_index, :, ROLES.index("candidate")]
        reference = cell_medians[mode_index, :, ROLES.index("reference")]
        point = 1.0 - float(np.median(candidate)) / float(np.median(reference))
        lower, upper = paired_prompt_bootstrap(
            candidate,
            reference,
            seed_offset=mode_index,
        )
        prompt_effects = 1.0 - candidate / reference
        positive = int(np.count_nonzero(prompt_effects > 0.0))
        passes = {
            "correctness": correctness_pass,
            "point_reduction": point >= MINIMUM_END_TO_END_POINT_REDUCTION,
            "bootstrap_lower_positive": lower > MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "prompt_direction": positive >= MINIMUM_POSITIVE_PROMPTS,
        }
        component_metrics: dict[str, Any] = {}
        for component in TIMING_COMPONENTS:
            values = arrays[component][mode_index]
            candidate_component = np.median(
                values[..., ROLES.index("candidate")], axis=1
            )
            reference_component = np.median(
                values[..., ROLES.index("reference")], axis=1
            )
            component_metrics[component] = {
                "candidate_median_ms": float(np.median(candidate_component)),
                "reference_median_ms": float(np.median(reference_component)),
                "median_reduction": 1.0
                - float(np.median(candidate_component))
                / float(np.median(reference_component)),
            }
        modes[mode] = {
            "end_to_end_reduction": point,
            "paired_prompt_bootstrap_95_interval": {
                "lower": lower,
                "upper": upper,
            },
            "positive_prompt_count": positive,
            "component_metrics": component_metrics,
            "candidate_output_tokens_median": float(
                np.median(tokens[mode_index, ..., ROLES.index("candidate")])
            ),
            "reference_output_tokens_median": float(
                np.median(tokens[mode_index, ..., ROLES.index("reference")])
            ),
            "candidate_output_bytes_median": float(
                np.median(raw_bytes[mode_index, ..., ROLES.index("candidate")])
            ),
            "reference_output_bytes_median": float(
                np.median(raw_bytes[mode_index, ..., ROLES.index("reference")])
            ),
            "passes": passes,
            "overall_pass": bool(all(passes.values())),
        }
    overall = correctness_pass and all(row["overall_pass"] for row in modes.values())
    return {
        "modes": modes,
        "correctness": {
            "independent_checkpoint_replay": correctness,
            "overall_pass": correctness_pass,
        },
        "gate": {
            "minimum_end_to_end_point_reduction": MINIMUM_END_TO_END_POINT_REDUCTION,
            "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "requires_both_modes": True,
            "overall_pass": overall,
        },
        "status": (
            "pass_trained_actual_e2e_preflight"
            if overall
            else "fail_trained_actual_e2e_preflight"
        ),
        "multiseed_confirmation_authorized": overall,
    }


def validate_strict_token_replay(
    token_ids: Sequence[int],
    *,
    token_bytes: Sequence[bytes],
    next_state_indices: Sequence[Sequence[int]],
    minimum_bytes: int = CONTINUATION_BYTES,
) -> tuple[bytes, int]:
    """Independently replay a free-running token trace through the strict DFA."""

    if not token_ids or minimum_bytes <= 0:
        raise ValueError("fresh actual token replay is empty")
    state_index = 0
    output = bytearray()
    for token_id in token_ids:
        if not isinstance(token_id, int) or not 0 <= token_id < len(token_bytes):
            raise ValueError("fresh actual replay token differs")
        next_state = int(next_state_indices[state_index][token_id])
        if next_state < 0:
            raise ValueError("fresh actual replay violates strict UTF-8")
        output.extend(token_bytes[token_id])
        state_index = next_state
    if len(output) < minimum_bytes or state_index != 0:
        raise ValueError("fresh actual replay stop state differs")
    bytes(output).decode("utf-8", errors="strict")
    return bytes(output), state_index
