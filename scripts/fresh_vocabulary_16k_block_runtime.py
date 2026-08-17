"""Exact target-side block execution for the trained 16K BPE model."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import torch
from benchmark_fresh_vocabulary_actual import RoleBundle
from scalar_runtime_core import maximum_normalized_error


class IncrementalBpeBlockDecoder:
    """A fresh-cache decoder that can advance several known tokens at once."""

    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.cache = None
        self.observed_tokens = 0

    def prefill_parallel(self, token_ids: Sequence[int]) -> torch.Tensor:
        if self.observed_tokens or not token_ids:
            raise ValueError("16K target-block prefill state differs")
        values = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        output = self.model(input_ids=values, use_cache=True, logits_to_keep=1)
        self.cache = output.past_key_values
        self.observed_tokens = len(token_ids)
        return output.logits[:, -1, :].float()

    def consume_block(self, token_ids: Sequence[int]) -> torch.Tensor:
        values = tuple(int(value) for value in token_ids)
        if self.cache is None or not values:
            raise RuntimeError("16K target-block decoder must be prefixed")
        inputs = torch.tensor([list(values)], dtype=torch.long, device=self.device)
        output = self.model(
            input_ids=inputs,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=len(values),
        )
        self.cache = output.past_key_values
        self.observed_tokens += len(values)
        logits = output.logits.float()
        if logits.shape[:2] != (1, len(values)):
            raise AssertionError("16K target-block logit shape differs")
        return logits


def _timed_prompt_ids(bundle: RoleBundle, prompt: bytes) -> tuple[int, ...]:
    text = prompt.decode("utf-8", errors="strict")
    values = tuple(
        int(value)
        for value in bundle.tokenizer.encode(text, add_special_tokens=False).ids
    )
    if not values:
        raise AssertionError("16K target-block tokenizer returned no IDs")
    return values


def _advance_state(bundle: RoleBundle, state_index: int, token_id: int) -> int:
    next_state = bundle.transitions.next_state_indices[state_index][token_id]
    if next_state < 0:
        raise AssertionError("16K target-block trace violates strict UTF-8")
    return next_state


def verify_block_sequence(
    bundle: RoleBundle,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    *,
    block_size: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare every block-cache logit with a no-cache full forward."""

    prompt = tuple(int(value) for value in prompt_ids)
    output = tuple(int(value) for value in output_ids)
    if not prompt or not output or block_size <= 0:
        raise ValueError("16K target-block correctness sequence differs")
    device = next(bundle.model.parameters()).device
    values = torch.tensor([list(prompt + output)], dtype=torch.long, device=device)
    with torch.inference_mode():
        full = bundle.model(input_ids=values, use_cache=False).logits.float()
        runtime = IncrementalBpeBlockDecoder(bundle.model)
        logits = runtime.prefill_parallel(prompt)
        expected = full[:, len(prompt) - 1]
        maximum = maximum_normalized_error(logits, expected, rtol=rtol, atol=atol)
        comparisons = 1
        argmax_exact = int(torch.equal(logits.argmax(dim=-1), expected.argmax(dim=-1)))
        next_index = 1
        decode_calls = 0
        while next_index < len(output):
            count = min(block_size, len(output) - next_index)
            block_inputs = output[next_index - 1 : next_index + count - 1]
            logits = runtime.consume_block(block_inputs)
            decode_calls += 1
            start = len(prompt) + next_index - 1
            expected = full[:, start : start + count]
            maximum = max(
                maximum,
                maximum_normalized_error(logits, expected, rtol=rtol, atol=atol),
            )
            actual_argmax = logits.argmax(dim=-1)
            expected_argmax = expected.argmax(dim=-1)
            argmax_exact += int(torch.count_nonzero(actual_argmax == expected_argmax))
            comparisons += count
            next_index += count
    expected_observed = len(prompt) + len(output) - 1
    if (
        argmax_exact != comparisons
        or runtime.observed_tokens != expected_observed
        or not math.isfinite(maximum)
        or maximum > 1.0
    ):
        raise AssertionError("16K target-block cache/full correctness differs")
    return {
        "comparisons": comparisons,
        "argmax_comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "decode_calls": decode_calls,
        "maximum_normalized_tolerance_ratio": maximum,
        "pass": True,
    }


def run_perfect_block_trial(
    bundle: RoleBundle,
    prompt: bytes,
    expected_prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    output_raw: bytes,
    *,
    mode: str,
    block_size: int,
    continuation_bytes: int,
) -> tuple[dict[str, float], tuple[int, ...], bytes, dict[str, int]]:
    """Time a zero-draft-cost, known-correct target block upper bound."""

    if mode not in {"controlled_replay", "free_running_utf8_greedy"}:
        raise ValueError("16K target-block mode differs")
    expected = tuple(int(value) for value in output_ids)
    if not expected or block_size <= 0:
        raise ValueError("16K target-block output contract differs")
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    prompt_ids = _timed_prompt_ids(bundle, prompt)
    encoded = time.perf_counter_ns()
    generated_ids: list[int] = []
    generated = bytearray()
    state_index = 0
    decode_calls = 0
    with torch.inference_mode():
        runtime = IncrementalBpeBlockDecoder(bundle.model)
        logits = runtime.prefill_parallel(prompt_ids)
        if mode == "free_running_utf8_greedy":
            first_token = int(
                logits.masked_fill(~bundle.masks[state_index], -torch.inf)
                .argmax(dim=-1)
                .item()
            )
            if first_token != expected[0]:
                raise AssertionError("16K target-block first proposal differs")
        else:
            _ = int(logits.argmax(dim=-1).item())
            first_token = expected[0]
        generated_ids.append(first_token)
        generated.extend(bundle.token_bytes[first_token])
        state_index = _advance_state(bundle, state_index, first_token)
        first = time.perf_counter_ns()
        next_index = 1
        while next_index < len(expected):
            count = min(block_size, len(expected) - next_index)
            block_inputs = expected[next_index - 1 : next_index + count - 1]
            logits = runtime.consume_block(block_inputs)
            decode_calls += 1
            expected_block = expected[next_index : next_index + count]
            if mode == "free_running_utf8_greedy":
                states: list[int] = []
                cursor = state_index
                for token_id in expected_block:
                    states.append(cursor)
                    cursor = _advance_state(bundle, cursor, token_id)
                masks = torch.stack([bundle.masks[value] for value in states])
                predicted = tuple(
                    int(value)
                    for value in logits[0]
                    .masked_fill(~masks, -torch.inf)
                    .argmax(dim=-1)
                    .cpu()
                    .tolist()
                )
                if predicted != expected_block:
                    raise AssertionError("16K target-block perfect proposal differs")
            else:
                _ = logits[0].argmax(dim=-1).cpu().tolist()
            for token_id in expected_block:
                generated_ids.append(token_id)
                generated.extend(bundle.token_bytes[token_id])
                state_index = _advance_state(bundle, state_index, token_id)
            next_index += count
        torch.mps.synchronize()
        model_finished = time.perf_counter_ns()
    raw = bytes(generated)
    raw.decode("utf-8", errors="strict")
    finished = time.perf_counter_ns()
    if tuple(prompt_ids) != tuple(expected_prompt_ids):
        raise AssertionError("16K target-block prompt tokenization drifted")
    if tuple(generated_ids) != expected or raw != output_raw:
        raise AssertionError("16K target-block output differs")
    if (
        len(raw) < continuation_bytes
        or state_index != 0
        or runtime.observed_tokens != len(prompt_ids) + len(expected) - 1
    ):
        raise AssertionError("16K target-block runtime accounting differs")
    metrics = {
        "tokenizer_ms": (encoded - started) / 1_000_000,
        "ttft_ms": (first - started) / 1_000_000,
        "decode_ms": (finished - first) / 1_000_000,
        "model_loop_ms": (model_finished - encoded) / 1_000_000,
        "end_to_end_ms": (finished - started) / 1_000_000,
    }
    if any(not math.isfinite(value) or value <= 0 for value in metrics.values()):
        raise AssertionError("16K target-block timing metric differs")
    return (
        metrics,
        tuple(generated_ids),
        raw,
        {
            "prefill_calls": 1,
            "decode_calls": decode_calls,
            "target_forward_calls": 1 + decode_calls,
            "perfect_draft_tokens": len(expected),
        },
    )
