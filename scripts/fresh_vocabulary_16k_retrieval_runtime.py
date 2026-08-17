"""Exact same-tokenizer retrieval drafting for the trained 16K BPE target."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from benchmark_fresh_vocabulary_actual import RoleBundle
from fresh_vocabulary_16k_block_runtime import IncrementalBpeBlockDecoder
from fresh_vocabulary_16k_retrieval_core import (
    MAXIMUM_DRAFT_TOKENS,
    CompactBackoffTable,
    hybrid_retrieval_proposal,
    prompt_lookup_proposal,
)

RETRIEVAL_ROLES = (
    "baseline_ar",
    "prompt_lookup_block_4",
    "corpus_ngram_block_4",
    "hybrid_retrieval_block_4",
)
PRIMARY_RETRIEVAL_ROLE = "hybrid_retrieval_block_4"


@dataclass(slots=True)
class RetrievalCounters:
    prefill_calls: int = 1
    target_decode_calls: int = 0
    proposal_attempts: int = 0
    proposal_tokens: int = 0
    accepted_draft_tokens: int = 0
    full_accept_cycles: int = 0
    rejection_cycles: int = 0
    no_draft_steps: int = 0
    correction_tokens: int = 0
    bonus_tokens: int = 0
    cropped_input_tokens: int = 0
    corpus_ngram_proposals: int = 0
    prompt_lookup_proposals: int = 0
    draft_lookup_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        output = asdict(self)
        output["target_forward_calls"] = self.prefill_calls + self.target_decode_calls
        return output


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    token_ids: tuple[int, ...]
    raw: bytes
    state_index: int
    observed_tokens: int
    counters: dict[str, int]


@dataclass(slots=True)
class BpeBlockTransaction:
    runtime: TransactionalIncrementalBpeDecoder
    start_observed: int
    inputs: tuple[int, ...]
    logits: torch.Tensor
    _closed: bool = False

    def finish(self, keep: int) -> None:
        if self._closed:
            raise RuntimeError("16K retrieval transaction is already closed")
        if not 0 <= int(keep) <= len(self.inputs):
            raise ValueError("16K retrieval transaction keep count differs")
        target = self.start_observed + int(keep)
        self.runtime._crop_to(target)
        self._closed = True


class TransactionalIncrementalBpeDecoder(IncrementalBpeBlockDecoder):
    """DynamicCache decoder with an explicit crop-to-verified-prefix operation."""

    def _cache_length(self) -> int:
        if self.cache is None or not hasattr(self.cache, "get_seq_length"):
            raise RuntimeError("16K retrieval target cache does not expose its length")
        return int(self.cache.get_seq_length())

    def _crop_to(self, observed_tokens: int) -> None:
        if (
            self.cache is None
            or not hasattr(self.cache, "crop")
            or not 0 <= observed_tokens <= self.observed_tokens
        ):
            raise RuntimeError("16K retrieval target cache cannot be cropped")
        self.cache.crop(int(observed_tokens))
        self.observed_tokens = int(observed_tokens)
        if self._cache_length() != self.observed_tokens:
            raise AssertionError("16K retrieval target cache crop length differs")

    def prefill_parallel(self, token_ids: Sequence[int]) -> torch.Tensor:
        logits = super().prefill_parallel(token_ids)
        if self._cache_length() != self.observed_tokens:
            raise AssertionError("16K retrieval prefill cache length differs")
        return logits

    def consume_transaction(self, token_ids: Sequence[int]) -> BpeBlockTransaction:
        values = tuple(int(value) for value in token_ids)
        if not 1 <= len(values) <= MAXIMUM_DRAFT_TOKENS + 1:
            raise ValueError("16K retrieval target block length differs")
        start = self.observed_tokens
        logits = self.consume_block(values)
        if self._cache_length() != start + len(values):
            raise AssertionError("16K retrieval target block cache length differs")
        return BpeBlockTransaction(
            runtime=self,
            start_observed=start,
            inputs=values,
            logits=logits[0],
        )


def _proposal(
    role: str,
    table: CompactBackoffTable,
    history: Sequence[int],
) -> tuple[tuple[int, ...], str]:
    if role == "prompt_lookup_block_4":
        values = prompt_lookup_proposal(history)
        return values, "prompt_lookup" if values else "none"
    if role == "corpus_ngram_block_4":
        values = table.propose(history)
        return values, "corpus_ngram" if values else "none"
    if role == PRIMARY_RETRIEVAL_ROLE:
        return hybrid_retrieval_proposal(table, history)
    if role == "baseline_ar":
        return (), "none"
    raise ValueError("16K retrieval runtime role differs")


def _target_token(
    bundle: RoleBundle,
    logits: torch.Tensor,
    state_index: int,
    expected_token: int,
    *,
    mode: str,
) -> int:
    if mode == "controlled_replay":
        _ = int(logits.argmax(dim=-1).item())
        return int(expected_token)
    if mode != "free_running_utf8_greedy":
        raise ValueError("16K retrieval runtime mode differs")
    target = int(
        logits.masked_fill(~bundle.masks[state_index], -torch.inf)
        .argmax(dim=-1)
        .item()
    )
    if target != int(expected_token):
        raise AssertionError("16K retrieval target greedy trace differs")
    return target


def _append_token(
    bundle: RoleBundle,
    token_id: int,
    generated_ids: list[int],
    generated: bytearray,
    state_index: int,
    *,
    continuation_bytes: int,
    maximum_output_bytes: int,
) -> tuple[int, bool]:
    token = int(token_id)
    next_state = bundle.transitions.next_state_indices[state_index][token]
    if next_state < 0:
        raise AssertionError("16K retrieval verifier emitted invalid UTF-8")
    generated_ids.append(token)
    generated.extend(bundle.token_bytes[token])
    if len(generated) > maximum_output_bytes:
        raise AssertionError("16K retrieval output exceeded its fixed bound")
    stop = len(generated) >= continuation_bytes and next_state == 0
    if len(generated) == maximum_output_bytes and not stop:
        raise AssertionError("16K retrieval output did not close at its bound")
    return next_state, stop


def decode_retrieval_after_prefill(
    bundle: RoleBundle,
    runtime: TransactionalIncrementalBpeDecoder,
    initial_logits: torch.Tensor,
    prompt_ids: Sequence[int],
    expected_ids: Sequence[int],
    expected_raw: bytes,
    table: CompactBackoffTable,
    *,
    role: str,
    mode: str,
    continuation_bytes: int,
    maximum_output_bytes: int,
) -> RetrievalTrace:
    expected = tuple(int(value) for value in expected_ids)
    prompt = tuple(int(value) for value in prompt_ids)
    if (
        role not in RETRIEVAL_ROLES
        or mode not in {"controlled_replay", "free_running_utf8_greedy"}
        or not prompt
        or not expected
        or len(expected_raw) < continuation_bytes
    ):
        raise ValueError("16K retrieval decode contract differs")
    counters = RetrievalCounters()
    generated_ids: list[int] = []
    generated = bytearray()
    state_index = 0
    next_index = 0
    first = _target_token(
        bundle,
        initial_logits[0],
        state_index,
        expected[next_index],
        mode=mode,
    )
    state_index, stop = _append_token(
        bundle,
        first,
        generated_ids,
        generated,
        state_index,
        continuation_bytes=continuation_bytes,
        maximum_output_bytes=maximum_output_bytes,
    )
    next_index += 1
    history = list(prompt) + [first]
    while not stop:
        if next_index >= len(expected):
            raise AssertionError("16K retrieval expected trace ended before UTF-8 stop")
        lookup_started = time.perf_counter_ns()
        proposal, source = _proposal(role, table, history)
        counters.draft_lookup_ns += time.perf_counter_ns() - lookup_started
        pending = generated_ids[-1]
        if not proposal:
            counters.no_draft_steps += 1
            transaction = runtime.consume_transaction((pending,))
            counters.target_decode_calls += 1
            target = _target_token(
                bundle,
                transaction.logits[0],
                state_index,
                expected[next_index],
                mode=mode,
            )
            state_index, stop = _append_token(
                bundle,
                target,
                generated_ids,
                generated,
                state_index,
                continuation_bytes=continuation_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
            next_index += 1
            history.append(target)
            transaction.finish(1)
            continue

        if len(proposal) > MAXIMUM_DRAFT_TOKENS:
            raise AssertionError("16K retrieval proposal exceeds block 4")
        counters.proposal_attempts += 1
        counters.proposal_tokens += len(proposal)
        if source == "corpus_ngram":
            counters.corpus_ngram_proposals += 1
        elif source == "prompt_lookup":
            counters.prompt_lookup_proposals += 1
        else:
            raise AssertionError("16K retrieval nonempty proposal source differs")
        inputs = (pending,) + tuple(int(value) for value in proposal)
        transaction = runtime.consume_transaction(inputs)
        counters.target_decode_calls += 1
        accepted = 0
        mismatch = False
        for draft_index, draft_token in enumerate(proposal):
            target = _target_token(
                bundle,
                transaction.logits[draft_index],
                state_index,
                expected[next_index],
                mode=mode,
            )
            if int(draft_token) != target:
                mismatch = True
                counters.rejection_cycles += 1
                counters.correction_tokens += 1
                counters.cropped_input_tokens += len(inputs) - (1 + accepted)
                state_index, stop = _append_token(
                    bundle,
                    target,
                    generated_ids,
                    generated,
                    state_index,
                    continuation_bytes=continuation_bytes,
                    maximum_output_bytes=maximum_output_bytes,
                )
                next_index += 1
                history.append(target)
                transaction.finish(1 + accepted)
                break
            state_index, stop = _append_token(
                bundle,
                target,
                generated_ids,
                generated,
                state_index,
                continuation_bytes=continuation_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
            accepted += 1
            counters.accepted_draft_tokens += 1
            next_index += 1
            history.append(target)
            if stop:
                # Keep pending and every accepted token except the terminal token.
                keep = accepted
                counters.cropped_input_tokens += len(inputs) - keep
                transaction.finish(keep)
                break
            if next_index >= len(expected):
                raise AssertionError("16K retrieval expected trace lacks a UTF-8 stop")
        if mismatch or stop:
            continue

        if accepted != len(proposal):
            raise AssertionError("16K retrieval acceptance loop ended inconsistently")
        counters.full_accept_cycles += 1
        bonus = _target_token(
            bundle,
            transaction.logits[len(proposal)],
            state_index,
            expected[next_index],
            mode=mode,
        )
        counters.bonus_tokens += 1
        state_index, stop = _append_token(
            bundle,
            bonus,
            generated_ids,
            generated,
            state_index,
            continuation_bytes=continuation_bytes,
            maximum_output_bytes=maximum_output_bytes,
        )
        next_index += 1
        history.append(bonus)
        transaction.finish(len(inputs))

    raw = bytes(generated)
    raw.decode("utf-8", errors="strict")
    if tuple(generated_ids) != expected or raw != expected_raw or state_index != 0:
        raise AssertionError("16K retrieval output differs from its target trace")
    expected_observed = len(prompt) + len(expected) - 1
    if runtime.observed_tokens != expected_observed:
        raise AssertionError("16K retrieval final cache/output lag differs")
    counters_payload = counters.to_dict()
    if (
        counters_payload["target_forward_calls"]
        != counters.prefill_calls + counters.target_decode_calls
        or counters.corpus_ngram_proposals + counters.prompt_lookup_proposals
        != counters.proposal_attempts
    ):
        raise AssertionError("16K retrieval counter identity differs")
    return RetrievalTrace(
        token_ids=tuple(generated_ids),
        raw=raw,
        state_index=state_index,
        observed_tokens=runtime.observed_tokens,
        counters=counters_payload,
    )


def _timed_prompt_ids(bundle: RoleBundle, prompt: bytes) -> tuple[int, ...]:
    text = prompt.decode("utf-8", errors="strict")
    values = tuple(
        int(value)
        for value in bundle.tokenizer.encode(text, add_special_tokens=False).ids
    )
    if not values:
        raise AssertionError("16K retrieval tokenizer returned no IDs")
    return values


def run_retrieval_trial(
    bundle: RoleBundle,
    prompt: bytes,
    expected_prompt_ids: Sequence[int],
    expected_ids: Sequence[int],
    expected_raw: bytes,
    table: CompactBackoffTable,
    *,
    role: str,
    mode: str,
    continuation_bytes: int,
    maximum_output_bytes: int,
) -> tuple[dict[str, float], RetrievalTrace]:
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    prompt_ids = _timed_prompt_ids(bundle, prompt)
    encoded = time.perf_counter_ns()
    with torch.inference_mode():
        runtime = TransactionalIncrementalBpeDecoder(bundle.model)
        logits = runtime.prefill_parallel(prompt_ids)
        prefetched = time.perf_counter_ns()
        trace = decode_retrieval_after_prefill(
            bundle,
            runtime,
            logits,
            prompt_ids,
            expected_ids,
            expected_raw,
            table,
            role=role,
            mode=mode,
            continuation_bytes=continuation_bytes,
            maximum_output_bytes=maximum_output_bytes,
        )
        torch.mps.synchronize()
        model_finished = time.perf_counter_ns()
    finished = time.perf_counter_ns()
    if tuple(prompt_ids) != tuple(int(value) for value in expected_prompt_ids):
        raise AssertionError("16K retrieval prompt tokenization drifted")
    metrics = {
        "tokenizer_ms": (encoded - started) / 1_000_000,
        "ttft_ms": (prefetched - started) / 1_000_000,
        "decode_ms": (finished - prefetched) / 1_000_000,
        "model_loop_ms": (model_finished - encoded) / 1_000_000,
        "draft_lookup_ms": trace.counters["draft_lookup_ns"] / 1_000_000,
        "end_to_end_ms": (finished - started) / 1_000_000,
    }
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        raise AssertionError("16K retrieval timing metric differs")
    for key in ("tokenizer_ms", "ttft_ms", "decode_ms", "model_loop_ms", "end_to_end_ms"):
        if metrics[key] <= 0:
            raise AssertionError("16K retrieval positive timing metric differs")
    return metrics, trace
