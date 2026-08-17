"""Exact greedy and forced-proposal MLX transactions for the isolated preflight."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


@dataclass(frozen=True)
class ForcedSpeculativeTrace:
    token_ids: tuple[int, ...]
    target_forward_calls: int
    full_accept_cycles: int
    immediate_reject_cycles: int
    partial_accept_cycles: int
    final_cache_offset: int


def cache_offsets(prompt_cache: Sequence[Any]) -> tuple[int, ...]:
    offsets = tuple(int(entry.offset) for entry in prompt_cache)
    if not offsets or len(set(offsets)) != 1:
        raise ValueError("MLX cache layer offsets differ")
    return offsets


def _evaluated_logits(model: Any, token_ids: Sequence[int], prompt_cache=None) -> mx.array:
    if not token_ids:
        raise ValueError("model call requires at least one token")
    values = mx.array([list(token_ids)], dtype=mx.uint32)
    logits = model(values, cache=prompt_cache)
    mx.eval(logits)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(token_ids):
        raise ValueError("MLX model returned unexpected logits shape")
    return logits


def _argmax_rows(logits: mx.array) -> tuple[int, ...]:
    values = mx.argmax(logits[0], axis=-1)
    mx.eval(values)
    return tuple(int(value) for value in values.tolist())


def compare_logits(reference: mx.array, candidate: mx.array, *, atol: float, rtol: float) -> dict[str, Any]:
    reference_np = np.asarray(reference, dtype=np.float32)
    candidate_np = np.asarray(candidate, dtype=np.float32)
    if reference_np.shape != candidate_np.shape or reference_np.ndim != 3:
        raise ValueError("MLX comparison logits shape differs")
    difference = np.abs(candidate_np - reference_np)
    tolerance = float(atol) + float(rtol) * np.abs(reference_np)
    normalized = difference / tolerance
    finite = bool(
        np.isfinite(reference_np).all()
        and np.isfinite(candidate_np).all()
        and np.isfinite(normalized).all()
    )
    reference_argmax = np.argmax(reference_np, axis=-1)
    candidate_argmax = np.argmax(candidate_np, axis=-1)
    argmax_exact = bool(np.array_equal(reference_argmax, candidate_argmax))
    maximum_absolute = float(np.max(difference))
    maximum_normalized = float(np.max(normalized))
    return {
        "argmax_exact": argmax_exact,
        "comparison_positions": int(reference_np.shape[1]),
        "finite": finite,
        "maximum_absolute_error": maximum_absolute,
        "maximum_normalized_error": maximum_normalized,
        "pass": bool(finite and argmax_exact and maximum_normalized <= 1.0),
    }


def _decision_equivalence_summary(comparison: dict[str, Any]) -> dict[str, Any]:
    """Separate greedy semantic equivalence from all-logit numeric diagnostics."""

    return {
        "argmax_exact": bool(comparison["argmax_exact"]),
        "comparison_positions": int(comparison["comparison_positions"]),
        "decision_equivalence_pass": bool(
            comparison["finite"] and comparison["argmax_exact"]
        ),
        "finite": bool(comparison["finite"]),
        "maximum_absolute_error": float(comparison["maximum_absolute_error"]),
        "maximum_normalized_error": float(comparison["maximum_normalized_error"]),
        "numeric_tolerance_pass": bool(comparison["pass"]),
    }


def full_vs_incremental_equivalence(
    model: Any,
    token_ids: Sequence[int],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(token_ids) < 2:
        raise ValueError("full/cache equivalence requires at least two tokens")
    full = _evaluated_logits(model, token_ids)
    prompt_cache = make_prompt_cache(model)
    rows = []
    for token_id in token_ids:
        rows.append(_evaluated_logits(model, (int(token_id),), prompt_cache))
    incremental = mx.concatenate(rows, axis=1)
    mx.eval(incremental)
    if cache_offsets(prompt_cache)[0] != len(token_ids):
        raise ValueError("incremental cache offset differs")
    return compare_logits(full, incremental, atol=atol, rtol=rtol)


def prefill_decode_equivalence(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    maximum_tokens: int,
    atol: float,
    rtol: float,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Compare the actual parallel-prefill/cached-decode path to full prefixes."""

    if len(prompt_ids) < 2 or maximum_tokens <= 0:
        raise ValueError("prefill/decode equivalence inputs differ")
    full_prompt = _evaluated_logits(model, prompt_ids)
    prompt_cache = make_prompt_cache(model)
    prefill = _evaluated_logits(model, prompt_ids[:-1], prompt_cache)
    last = _evaluated_logits(model, (int(prompt_ids[-1]),), prompt_cache)
    cached_prompt = mx.concatenate([prefill, last], axis=1)
    mx.eval(cached_prompt)
    comparisons = [
        compare_logits(full_prompt, cached_prompt, atol=atol, rtol=rtol)
    ]

    context = [int(value) for value in prompt_ids]
    output: list[int] = []
    cached_next = last
    for output_index in range(maximum_tokens):
        token_id = _argmax_rows(cached_next)[-1]
        output.append(token_id)
        if output_index + 1 == maximum_tokens:
            break
        context.append(token_id)
        cached_next = _evaluated_logits(model, (token_id,), prompt_cache)
        full_next = _evaluated_logits(model, context)[:, -1:, :]
        comparisons.append(
            compare_logits(full_next, cached_next, atol=atol, rtol=rtol)
        )

    expected_offset = len(prompt_ids) + maximum_tokens - 1
    if cache_offsets(prompt_cache)[0] != expected_offset:
        raise ValueError("prefill/decode cache lag differs")
    result = {
        "argmax_exact": bool(all(item["argmax_exact"] for item in comparisons)),
        "comparison_positions": int(
            sum(item["comparison_positions"] for item in comparisons)
        ),
        "decision_equivalence_pass": bool(
            all(item["finite"] and item["argmax_exact"] for item in comparisons)
        ),
        "finite": bool(all(item["finite"] for item in comparisons)),
        "maximum_absolute_error": float(
            max(item["maximum_absolute_error"] for item in comparisons)
        ),
        "maximum_normalized_error": float(
            max(item["maximum_normalized_error"] for item in comparisons)
        ),
        "numeric_tolerance_pass": bool(all(item["pass"] for item in comparisons)),
    }
    return result, tuple(output)


def rollback_equivalence(
    model: Any,
    prefix_ids: Sequence[int],
    speculative_ids: Sequence[int],
    *,
    keep_speculative_tokens: int,
    correction_token_id: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if (
        not prefix_ids
        or not speculative_ids
        or keep_speculative_tokens < 0
        or keep_speculative_tokens >= len(speculative_ids)
    ):
        raise ValueError("rollback test inputs differ")
    rolled_cache = make_prompt_cache(model)
    _evaluated_logits(model, prefix_ids, rolled_cache)
    _evaluated_logits(model, speculative_ids, rolled_cache)
    trim = len(speculative_ids) - keep_speculative_tokens
    trimmed = trim_prompt_cache(rolled_cache, trim)
    if trimmed != trim:
        raise ValueError("MLX cache did not trim the requested token count")
    rolled_logits = _evaluated_logits(model, (correction_token_id,), rolled_cache)

    fresh_cache = make_prompt_cache(model)
    fresh_sequence = (
        tuple(int(value) for value in prefix_ids)
        + tuple(int(value) for value in speculative_ids[:keep_speculative_tokens])
        + (int(correction_token_id),)
    )
    fresh_logits = _evaluated_logits(model, fresh_sequence, fresh_cache)[:, -1:, :]
    expected_offset = len(fresh_sequence)
    if (
        cache_offsets(rolled_cache)[0] != expected_offset
        or cache_offsets(fresh_cache)[0] != expected_offset
    ):
        raise ValueError("rollback cache offset differs from fresh cache")
    result = compare_logits(fresh_logits, rolled_logits, atol=atol, rtol=rtol)
    result["comparison_positions"] = 1
    return _decision_equivalence_summary(result)


def greedy_generate(model: Any, prompt_ids: Sequence[int], *, maximum_tokens: int) -> tuple[int, ...]:
    if not prompt_ids or maximum_tokens <= 0:
        raise ValueError("greedy generation inputs differ")
    prompt_cache = make_prompt_cache(model)
    if len(prompt_ids) > 1:
        _evaluated_logits(model, prompt_ids[:-1], prompt_cache)
    last_token = int(prompt_ids[-1])
    output: list[int] = []
    for _ in range(maximum_tokens):
        logits = _evaluated_logits(model, (last_token,), prompt_cache)
        next_token = _argmax_rows(logits)[-1]
        output.append(next_token)
        last_token = next_token
    if cache_offsets(prompt_cache)[0] != len(prompt_ids) + maximum_tokens - 1:
        raise ValueError("greedy generation cache lag differs")
    return tuple(output)


ProposalProvider = Callable[[tuple[int, ...], int, int], tuple[int, ...]]


def forced_speculative_generate(
    model: Any,
    prompt_ids: Sequence[int],
    *,
    maximum_tokens: int,
    maximum_draft_tokens: int,
    proposal_provider: ProposalProvider,
) -> ForcedSpeculativeTrace:
    if not prompt_ids or maximum_tokens <= 0 or maximum_draft_tokens <= 0:
        raise ValueError("forced speculative inputs differ")
    prompt_cache = make_prompt_cache(model)
    if len(prompt_ids) > 1:
        _evaluated_logits(model, prompt_ids[:-1], prompt_cache)
    last_token = int(prompt_ids[-1])
    emitted: list[int] = []
    target_forward_calls = 0
    full_accept_cycles = 0
    immediate_reject_cycles = 0
    partial_accept_cycles = 0

    while len(emitted) < maximum_tokens:
        remaining = maximum_tokens - len(emitted)
        context = tuple(int(value) for value in prompt_ids) + tuple(emitted)
        proposed = tuple(
            int(value)
            for value in proposal_provider(context, remaining, len(emitted))
        )[: min(maximum_draft_tokens, remaining)]
        if not proposed:
            logits = _evaluated_logits(model, (last_token,), prompt_cache)
            target_forward_calls += 1
            next_token = _argmax_rows(logits)[-1]
            emitted.append(next_token)
            last_token = next_token
            continue

        block = (last_token,) + proposed
        logits = _evaluated_logits(model, block, prompt_cache)
        target_forward_calls += 1
        predicted = _argmax_rows(logits)
        accepted = 0
        while accepted < len(proposed) and proposed[accepted] == predicted[accepted]:
            accepted += 1

        room = maximum_tokens - len(emitted)
        accepted_to_emit = min(accepted, room)
        emitted.extend(proposed[:accepted_to_emit])
        if accepted_to_emit < accepted:
            trim_prompt_cache(prompt_cache, len(proposed) - accepted_to_emit)
            break

        if accepted == len(proposed):
            full_accept_cycles += 1
            if len(emitted) < maximum_tokens:
                bonus = predicted[len(proposed)]
                emitted.append(bonus)
                last_token = bonus
            else:
                trimmed = trim_prompt_cache(prompt_cache, 1)
                if trimmed != 1:
                    raise ValueError("terminal full-accept cache trim differs")
                last_token = emitted[-1]
        else:
            if accepted == 0:
                immediate_reject_cycles += 1
            else:
                partial_accept_cycles += 1
            trim = len(proposed) - accepted
            trimmed = trim_prompt_cache(prompt_cache, trim)
            if trimmed != trim:
                raise ValueError("speculative rejection cache trim differs")
            correction = predicted[accepted]
            emitted.append(correction)
            last_token = correction

    output = tuple(emitted[:maximum_tokens])
    expected_offset = len(prompt_ids) + len(output) - 1
    final_offset = cache_offsets(prompt_cache)[0]
    if final_offset != expected_offset:
        raise ValueError("forced speculative final cache lag differs")
    return ForcedSpeculativeTrace(
        token_ids=output,
        target_forward_calls=target_forward_calls,
        full_accept_cycles=full_accept_cycles,
        immediate_reject_cycles=immediate_reject_cycles,
        partial_accept_cycles=partial_accept_cycles,
        final_cache_offset=final_offset,
    )
