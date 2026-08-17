"""Causal patch reconstruction and structural generation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from typing import Iterable, Sequence

import numpy as np

from .neural_patching import fixed_byte_boundaries, hf_patch_lengths
from .phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_window_grid_trace,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
)
from .utf8 import (
    StrictUtf8State,
    advance_strict_utf8,
    prefix_boundary_mask,
    strict_utf8_allowed_ranges,
    strict_utf8_state,
)


GENERATION_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
DECODING_MODES = ("greedy", "sampled")


@dataclass(frozen=True, slots=True)
class PromptSelection:
    prompts: np.ndarray
    candidate_count: int
    unique_candidate_count: int
    prompt_length: int

    def public_metadata(self) -> dict[str, int]:
        return {
            "selected_prompts": len(self.prompts),
            "candidate_prompts": self.candidate_count,
            "unique_candidate_prompts": self.unique_candidate_count,
            "prompt_length_bytes": self.prompt_length,
        }


@dataclass(frozen=True, slots=True)
class ContinuationMetrics:
    continuations: int
    continuation_bytes: int
    valid_utf8_count: int
    valid_utf8_rate: float
    replacement_character_free_count: int
    replacement_character_free_rate: float
    valid_jamo_transition_count: int
    valid_jamo_transition_rate: float
    mean_bytes_per_codepoint_valid_utf8: float | None
    median_bytes_per_codepoint_valid_utf8: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidCompletionMetrics:
    """Aggregate diagnostics for variable-length time-to-valid completions."""

    continuations: int
    minimum_completion_bytes: int
    valid_utf8_count: int
    valid_utf8_rate: float
    replacement_character_free_count: int
    replacement_character_free_rate: float
    valid_jamo_transition_count: int
    valid_jamo_transition_rate: float
    minimum_emitted_bytes: int
    mean_emitted_bytes: float
    median_emitted_bytes: float
    maximum_emitted_bytes: int
    minimum_overshoot_bytes: int
    mean_overshoot_bytes: float
    median_overshoot_bytes: float
    maximum_overshoot_bytes: int
    mean_bytes_per_codepoint_valid_utf8: float | None
    median_bytes_per_codepoint_valid_utf8: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Utf8FailureMetrics:
    continuations: int
    continuation_bytes: int
    strict_valid_count: int
    strict_valid_rate: float
    illegal_transition_count: int
    illegal_transition_rate: float
    incomplete_terminal_scalar_count: int
    incomplete_terminal_scalar_rate: float
    mean_legal_prefix_bytes: float
    median_legal_prefix_bytes: float
    mean_legal_prefix_fraction: float
    median_legal_prefix_fraction: float
    mean_closed_codepoint_prefix_bytes: float
    median_closed_codepoint_prefix_bytes: float
    mean_closed_codepoint_prefix_fraction: float
    median_closed_codepoint_prefix_fraction: float
    mean_first_illegal_byte_position: float | None
    median_first_illegal_byte_position: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _hangul_heavy(raw: bytes) -> bool:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    precomposed = sum("\uac00" <= character <= "\ud7a3" for character in letters)
    return precomposed / len(letters) >= 0.8


def select_generation_prompts(
    inputs: np.ndarray,
    boundary_masks: np.ndarray,
    *,
    prompt_count: int = 256,
    prompt_length: int = 128,
) -> PromptSelection:
    """Select deterministic, valid, Hangul-heavy held-out byte prompts."""

    if inputs.ndim != 2 or inputs.shape != boundary_masks.shape:
        raise ValueError("inputs and boundary masks must be equal matrices")
    if prompt_count <= 0 or not 0 < prompt_length < inputs.shape[1]:
        raise ValueError("invalid prompt count or length")

    candidates: list[bytes] = []
    for row, boundaries in zip(inputs, boundary_masks, strict=True):
        if not bool(boundaries[0]) or not bool(boundaries[prompt_length]):
            continue
        raw = bytes(row[:prompt_length])
        if _hangul_heavy(raw):
            candidates.append(raw)
    unique = sorted(
        set(candidates),
        key=lambda raw: (sha256(raw).digest(), raw),
    )
    if len(unique) < prompt_count:
        raise ValueError(
            f"need {prompt_count} unique generation prompts, found {len(unique)}"
        )
    selected = np.stack(
        [np.frombuffer(raw, dtype=np.uint8) for raw in unique[:prompt_count]]
    )
    return PromptSelection(
        prompts=selected,
        candidate_count=len(candidates),
        unique_candidate_count=len(unique),
        prompt_length=prompt_length,
    )


def generation_patch_matrix(
    inputs: np.ndarray,
    policy: str,
    *,
    horizon: int = 256,
    patch_count: int = 43,
    fixed_stride: int = 6,
) -> np.ndarray:
    """Rebuild a prefix-causal BLT matrix for the currently observed bytes."""

    if policy not in GENERATION_POLICIES:
        raise ValueError(f"unknown generation policy: {policy}")
    if inputs.ndim != 2 or not len(inputs):
        raise ValueError("generation inputs must be a non-empty matrix")
    observed = int(inputs.shape[1])
    if not 0 < observed <= horizon:
        raise ValueError("observed prefix exceeds the fixed horizon")

    rows: list[tuple[int, ...]] = []
    for row in inputs:
        raw = bytes(row)
        if policy == "fixed_byte_6":
            boundaries = fixed_byte_boundaries(observed, fixed_stride)
        else:
            codepoints = prefix_boundary_mask(raw)[:-1]
            if policy == "causal_codepoint_grid":
                boundaries = causal_codepoint_grid_boundaries(
                    codepoints,
                    patch_count,
                    sequence_length=horizon,
                    require_complete=False,
                )
            else:
                whitespace = compact_whitespace_mask(raw)
                boundaries = causal_window_grid_trace(
                    codepoints,
                    whitespace,
                    patch_count,
                    sequence_length=horizon,
                    require_complete=False,
                ).boundaries
        rows.append(boundaries)
    return padded_hf_patch_matrix(rows, observed)


def _utf8_transition(
    remaining: int,
    lower: int,
    upper: int,
    value: int,
) -> tuple[int, int, int, bool]:
    """Advance one strict UTF-8 DFA byte."""

    state = advance_strict_utf8(
        StrictUtf8State(remaining, lower, upper),
        value,
    )
    return state.remaining, state.lower, state.upper, state.valid


def _utf8_constraint_state(data: bytes) -> tuple[int, int, int, bool]:
    """Return remaining continuations, next-byte range, and prefix validity."""

    state = strict_utf8_state(data)
    return state.remaining, state.lower, state.upper, state.valid


def utf8_allowed_next_bytes(
    prefix: bytes,
    *,
    remaining_bytes_after_choice: int,
) -> np.ndarray:
    """Return byte candidates preserving strict UTF-8 and final-horizon closure."""

    state = strict_utf8_state(prefix)
    allowed = np.zeros(256, dtype=bool)
    for lower, upper in strict_utf8_allowed_ranges(
        state,
        remaining_bytes_after_choice=remaining_bytes_after_choice,
    ):
        allowed[lower : upper + 1] = True
    return allowed


def top_p_sample(
    logits: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 0.8,
    top_p: float = 0.95,
    allowed: np.ndarray | None = None,
) -> int:
    """Sample one byte with deterministic tie-breaking and CPU float64 math."""

    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (256,) or not np.isfinite(values).all():
        raise ValueError("logits must be 256 finite values")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("invalid temperature or top-p")
    if allowed is not None:
        mask = np.asarray(allowed, dtype=bool)
        if mask.shape != (256,) or not mask.any():
            raise ValueError("allowed-byte mask is empty or malformed")
        values = np.where(mask, values, -np.inf)

    scaled = values / temperature
    finite = np.isfinite(scaled)
    maximum = float(scaled[finite].max())
    weights = np.zeros(256, dtype=np.float64)
    weights[finite] = np.exp(scaled[finite] - maximum)
    probabilities = weights / weights.sum()
    byte_ids = np.arange(256)
    order = np.lexsort((byte_ids, -probabilities))
    ordered_probabilities = probabilities[order]
    cutoff = int(np.searchsorted(np.cumsum(ordered_probabilities), top_p)) + 1
    kept = order[:cutoff]
    kept_probabilities = probabilities[kept]
    kept_probabilities /= kept_probabilities.sum()
    return int(rng.choice(kept, p=kept_probabilities))


def greedy_byte(logits: np.ndarray, allowed: np.ndarray | None = None) -> int:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (256,) or not np.isfinite(values).all():
        raise ValueError("logits must be 256 finite values")
    if allowed is not None:
        mask = np.asarray(allowed, dtype=bool)
        if mask.shape != (256,) or not mask.any():
            raise ValueError("allowed-byte mask is empty or malformed")
        values = np.where(mask, values, -np.inf)
    return int(np.argmax(values))


def valid_conjoining_jamo_transitions(text: str) -> bool:
    """Validate broad Hangul conjoining-Jamo sequences as L+V+optional T."""

    def kind(codepoint: int) -> str | None:
        if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97F:
            return "L"
        if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
            return "V"
        if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
            return "T"
        return None

    state: str | None = None
    for character in text:
        current = kind(ord(character))
        if current == "L":
            if state == "L":
                return False
            state = "L"
        elif current == "V":
            if state != "L":
                return False
            state = "LV"
        elif current == "T":
            if state != "LV":
                return False
            state = None
        else:
            if state == "L":
                return False
            state = None
    return state != "L"


def continuation_diagnostic_arrays(
    continuations: Iterable[bytes],
) -> tuple[dict[str, np.ndarray], int]:
    """Return non-content per-continuation structural diagnostics."""

    values = list(continuations)
    if not values:
        raise ValueError("at least one continuation is required")
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("continuations must have equal byte lengths")
    count = len(values)
    strict_valid = np.zeros(count, dtype=np.uint8)
    replacement_free = np.zeros(count, dtype=np.uint8)
    jamo_valid = np.zeros(count, dtype=np.uint8)
    bytes_per_codepoint = np.full(count, np.nan, dtype=np.float64)
    for index, raw in enumerate(values):
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        strict_valid[index] = 1
        if "\ufffd" not in text:
            replacement_free[index] = 1
        if valid_conjoining_jamo_transitions(text):
            jamo_valid[index] = 1
        if text:
            bytes_per_codepoint[index] = len(raw) / len(text)
    return (
        {
            "strict_valid": strict_valid,
            "replacement_character_free": replacement_free,
            "valid_jamo_transition": jamo_valid,
            "bytes_per_codepoint": bytes_per_codepoint,
        },
        next(iter(lengths)),
    )


def continuation_metrics_from_diagnostics(
    diagnostics: dict[str, np.ndarray],
    continuation_bytes: int,
) -> ContinuationMetrics:
    """Reconstruct aggregate structural metrics from numeric diagnostics."""

    expected = {
        "strict_valid",
        "replacement_character_free",
        "valid_jamo_transition",
        "bytes_per_codepoint",
    }
    if set(diagnostics) != expected or continuation_bytes < 0:
        raise ValueError("malformed continuation diagnostics")
    arrays = {key: np.asarray(value) for key, value in diagnostics.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 1:
        raise ValueError("continuation diagnostics must be equal vectors")
    count = len(arrays["strict_valid"])
    if not count:
        raise ValueError("continuation diagnostics must not be empty")
    for key in (
        "strict_valid",
        "replacement_character_free",
        "valid_jamo_transition",
    ):
        if not np.isin(arrays[key], (0, 1)).all():
            raise ValueError(f"continuation diagnostic is not binary: {key}")
    strict = arrays["strict_valid"].astype(bool)
    replacement = arrays["replacement_character_free"].astype(bool)
    jamo = arrays["valid_jamo_transition"].astype(bool)
    if np.any(replacement & ~strict) or np.any(jamo & ~strict):
        raise ValueError("invalid continuations cannot pass structural diagnostics")
    byte_rates = arrays["bytes_per_codepoint"].astype(np.float64)
    if np.any(~strict & ~np.isnan(byte_rates)):
        raise ValueError("invalid continuations cannot have codepoint rates")
    available = byte_rates[np.isfinite(byte_rates)]
    if np.any(available <= 0) or (
        continuation_bytes > 0 and np.any(strict & ~np.isfinite(byte_rates))
    ):
        raise ValueError("valid continuations need positive codepoint rates")
    if continuation_bytes > 0 and len(available):
        codepoint_counts = continuation_bytes / available
        if (
            np.any(available < 1)
            or np.any(available > 4)
            or np.any(codepoint_counts < 1)
            or not np.allclose(
                codepoint_counts,
                np.rint(codepoint_counts),
                rtol=0,
                atol=1e-10,
            )
        ):
            raise ValueError("bytes/codepoint diagnostics are not realizable")
    return ContinuationMetrics(
        continuations=count,
        continuation_bytes=continuation_bytes,
        valid_utf8_count=int(strict.sum()),
        valid_utf8_rate=float(strict.mean()),
        replacement_character_free_count=int(replacement.sum()),
        replacement_character_free_rate=float(replacement.mean()),
        valid_jamo_transition_count=int(jamo.sum()),
        valid_jamo_transition_rate=float(jamo.mean()),
        mean_bytes_per_codepoint_valid_utf8=(
            float(available.mean()) if len(available) else None
        ),
        median_bytes_per_codepoint_valid_utf8=(
            float(np.median(available)) if len(available) else None
        ),
    )


def continuation_metrics(continuations: Iterable[bytes]) -> ContinuationMetrics:
    diagnostics, continuation_bytes = continuation_diagnostic_arrays(
        continuations
    )
    return continuation_metrics_from_diagnostics(
        diagnostics,
        continuation_bytes,
    )


def valid_completion_metrics(
    continuations: Iterable[bytes],
    *,
    minimum_completion_bytes: int,
) -> ValidCompletionMetrics:
    """Summarize variable-length completions crossing a common byte target.

    This is intentionally distinct from :func:`continuation_metrics`, whose
    fixed-length contract is required by the historical generation-validity
    artifacts.  Time-to-valid inference can stop one to three bytes after a
    byte target in order to close the current UTF-8 scalar.
    """

    values = list(continuations)
    if not values or minimum_completion_bytes <= 0:
        raise ValueError("valid completion metrics need outputs and a target")
    lengths = np.asarray([len(value) for value in values], dtype=np.int64)
    if np.any(lengths < minimum_completion_bytes):
        raise ValueError("a completion stopped before the minimum byte target")

    valid = 0
    replacement_free = 0
    jamo_valid = 0
    byte_rates: list[float] = []
    for raw in values:
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        valid += 1
        if "\ufffd" not in text:
            replacement_free += 1
        if valid_conjoining_jamo_transitions(text):
            jamo_valid += 1
        if text:
            byte_rates.append(len(raw) / len(text))

    count = len(values)
    overshoot = lengths - minimum_completion_bytes
    available = np.asarray(byte_rates, dtype=np.float64)
    return ValidCompletionMetrics(
        continuations=count,
        minimum_completion_bytes=minimum_completion_bytes,
        valid_utf8_count=valid,
        valid_utf8_rate=valid / count,
        replacement_character_free_count=replacement_free,
        replacement_character_free_rate=replacement_free / count,
        valid_jamo_transition_count=jamo_valid,
        valid_jamo_transition_rate=jamo_valid / count,
        minimum_emitted_bytes=int(lengths.min()),
        mean_emitted_bytes=float(lengths.mean()),
        median_emitted_bytes=float(np.median(lengths)),
        maximum_emitted_bytes=int(lengths.max()),
        minimum_overshoot_bytes=int(overshoot.min()),
        mean_overshoot_bytes=float(overshoot.mean()),
        median_overshoot_bytes=float(np.median(overshoot)),
        maximum_overshoot_bytes=int(overshoot.max()),
        mean_bytes_per_codepoint_valid_utf8=(
            float(available.mean()) if len(available) else None
        ),
        median_bytes_per_codepoint_valid_utf8=(
            float(np.median(available)) if len(available) else None
        ),
    )


def utf8_failure_diagnostic_arrays(
    prompts: Iterable[bytes],
    continuations: Iterable[bytes],
) -> tuple[dict[str, np.ndarray], int]:
    """Return non-content DFA failure diagnostics for each continuation."""

    prefix_values = list(prompts)
    output_values = list(continuations)
    if not output_values or len(prefix_values) != len(output_values):
        raise ValueError("prompts and continuations need equal non-zero counts")
    lengths = {len(value) for value in output_values}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError("continuations need one positive fixed byte length")
    continuation_bytes = next(iter(lengths))

    count = len(output_values)
    failure_category = np.empty(count, dtype=np.uint8)
    legal_prefix_lengths = np.empty(count, dtype=np.int64)
    closed_prefix_lengths = np.empty(count, dtype=np.int64)
    illegal_positions = np.full(count, -1, dtype=np.int64)
    for index, (prompt, continuation) in enumerate(zip(
        prefix_values,
        output_values,
        strict=True,
    )):
        remaining, lower, upper, valid = _utf8_constraint_state(prompt)
        if not valid or remaining:
            raise ValueError("each generation prompt must end at a UTF-8 boundary")
        illegal_position: int | None = None
        legal_prefix = 0
        closed_prefix = 0
        for position, value in enumerate(continuation):
            remaining, lower, upper, valid = _utf8_transition(
                remaining,
                lower,
                upper,
                value,
            )
            if not valid:
                illegal_position = position
                break
            legal_prefix = position + 1
            if remaining == 0:
                closed_prefix = position + 1
        legal_prefix_lengths[index] = legal_prefix
        closed_prefix_lengths[index] = closed_prefix
        if illegal_position is not None:
            failure_category[index] = 1
            illegal_positions[index] = illegal_position
            continue
        if remaining:
            failure_category[index] = 2
        else:
            failure_category[index] = 0
            try:
                continuation.decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:  # pragma: no cover - invariant
                raise AssertionError("DFA and strict decoder disagree") from error
    return (
        {
            "failure_category": failure_category,
            "legal_prefix_bytes": legal_prefix_lengths,
            "closed_codepoint_prefix_bytes": closed_prefix_lengths,
            "first_illegal_byte_position": illegal_positions,
        },
        continuation_bytes,
    )


def utf8_failure_metrics_from_diagnostics(
    diagnostics: dict[str, np.ndarray],
    continuation_bytes: int,
) -> Utf8FailureMetrics:
    """Reconstruct aggregate DFA metrics from per-continuation diagnostics."""

    expected = {
        "failure_category",
        "legal_prefix_bytes",
        "closed_codepoint_prefix_bytes",
        "first_illegal_byte_position",
    }
    if set(diagnostics) != expected or continuation_bytes <= 0:
        raise ValueError("malformed UTF-8 failure diagnostics")
    arrays = {key: np.asarray(value) for key, value in diagnostics.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or not shapes or len(next(iter(shapes))) != 1:
        raise ValueError("UTF-8 failure diagnostics must be equal vectors")
    count = len(arrays["failure_category"])
    if not count:
        raise ValueError("UTF-8 failure diagnostics must not be empty")
    numeric = {
        key: np.asarray(arrays[key], dtype=np.float64)
        for key in expected
    }
    if any(
        not np.isfinite(value).all()
        or not np.equal(value, np.floor(value)).all()
        for value in numeric.values()
    ):
        raise ValueError("UTF-8 failure diagnostics must contain finite integers")
    categories = numeric["failure_category"].astype(np.int64)
    legal = numeric["legal_prefix_bytes"].astype(np.int64)
    closed = numeric["closed_codepoint_prefix_bytes"].astype(np.int64)
    illegal_positions = numeric["first_illegal_byte_position"].astype(np.int64)
    if (
        not np.isin(categories, (0, 1, 2)).all()
        or np.any(legal < 0)
        or np.any(legal > continuation_bytes)
        or np.any(closed < 0)
        or np.any(closed > legal)
    ):
        raise ValueError("UTF-8 failure diagnostics contain invalid ranges")
    illegal_mask = categories == 1
    if (
        np.any(illegal_positions[illegal_mask] < 0)
        or np.any(illegal_positions[illegal_mask] >= continuation_bytes)
        or np.any(illegal_positions[~illegal_mask] != -1)
        or np.any(legal[illegal_mask] != illegal_positions[illegal_mask])
        or np.any(legal[~illegal_mask] != continuation_bytes)
        or np.any(closed[categories == 0] != continuation_bytes)
        or np.any(closed[categories == 2] >= continuation_bytes)
    ):
        raise ValueError("UTF-8 failure categories and positions disagree")

    valid_count = int((categories == 0).sum())
    illegal_count = int(illegal_mask.sum())
    incomplete_count = int((categories == 2).sum())
    if valid_count + illegal_count + incomplete_count != count:
        raise AssertionError("UTF-8 failure categories do not partition outputs")
    legal_values = legal.astype(np.float64)
    closed_values = closed.astype(np.float64)
    illegal_values = illegal_positions[illegal_mask].astype(np.float64)
    return Utf8FailureMetrics(
        continuations=count,
        continuation_bytes=continuation_bytes,
        strict_valid_count=valid_count,
        strict_valid_rate=valid_count / count,
        illegal_transition_count=illegal_count,
        illegal_transition_rate=illegal_count / count,
        incomplete_terminal_scalar_count=incomplete_count,
        incomplete_terminal_scalar_rate=incomplete_count / count,
        mean_legal_prefix_bytes=float(legal_values.mean()),
        median_legal_prefix_bytes=float(np.median(legal_values)),
        mean_legal_prefix_fraction=float(
            legal_values.mean() / continuation_bytes
        ),
        median_legal_prefix_fraction=float(
            np.median(legal_values) / continuation_bytes
        ),
        mean_closed_codepoint_prefix_bytes=float(closed_values.mean()),
        median_closed_codepoint_prefix_bytes=float(np.median(closed_values)),
        mean_closed_codepoint_prefix_fraction=float(
            closed_values.mean() / continuation_bytes
        ),
        median_closed_codepoint_prefix_fraction=float(
            np.median(closed_values) / continuation_bytes
        ),
        mean_first_illegal_byte_position=(
            float(illegal_values.mean()) if len(illegal_values) else None
        ),
        median_first_illegal_byte_position=(
            float(np.median(illegal_values)) if len(illegal_values) else None
        ),
    )


def utf8_failure_metrics(
    prompts: Iterable[bytes],
    continuations: Iterable[bytes],
) -> Utf8FailureMetrics:
    diagnostics, continuation_bytes = utf8_failure_diagnostic_arrays(
        prompts,
        continuations,
    )
    return utf8_failure_metrics_from_diagnostics(
        diagnostics,
        continuation_bytes,
    )


def sampling_generators(seed: int, prompt_count: int) -> list[np.random.Generator]:
    if prompt_count <= 0:
        raise ValueError("prompt count must be positive")
    return [
        np.random.default_rng(np.random.SeedSequence([20_260_810, seed, index]))
        for index in range(prompt_count)
    ]
