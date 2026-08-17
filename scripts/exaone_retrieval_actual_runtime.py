"""Timed ordinary and exact retrieval generation for the EXAONE actual study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter_ns

import mlx.core as mx
from exaone_actual_runtime import (
    ExaoneRuntimeBundle,
    _prompt_text,
    _validate_decoded_sequence,
)
from exaone_retrieval_data import hybrid_retrieval_proposal
from large_model_retrieval_preflight import token_sequence_sha256
from mlx_retrieval_runtime import forced_speculative_generate, greedy_generate


@dataclass(frozen=True, slots=True)
class ActualGenerationTrial:
    accepted_draft_tokens: int
    bonus_tokens: int
    corpus_accepted_draft_tokens: int
    corpus_proposal_calls: int
    corpus_proposed_tokens: int
    correction_tokens: int
    decoded_utf8_sha256: str
    detokenization_ns: int
    elapsed_ns: int
    final_cache_offset: int
    full_accept_cycles: int
    generation_ns: int
    immediate_reject_cycles: int
    no_proposal_calls: int
    output_token_ids: tuple[int, ...]
    output_token_sha256: str
    partial_accept_cycles: int
    prompt_accepted_draft_tokens: int
    prompt_proposal_calls: int
    prompt_proposed_tokens: int
    prompt_token_count: int
    proposal_attempts: int
    proposed_tokens: int
    target_generation_forward_calls: int
    target_prefill_forward_calls: int
    tokenization_ns: int


def _timing_parts(
    *, started: int, tokenized: int, generated: int, stopped: int
) -> tuple[int, int, int, int]:
    values = (
        tokenized - started,
        generated - tokenized,
        stopped - generated,
        stopped - started,
    )
    if any(value <= 0 for value in values) or sum(values[:3]) != values[3]:
        raise ValueError("EXAONE actual timing decomposition differs")
    return values


def run_actual_baseline_trial(
    bundle: ExaoneRuntimeBundle,
    prompt_ids: Sequence[int],
    *,
    output_tokens: int,
) -> ActualGenerationTrial:
    if output_tokens <= 0:
        raise ValueError("EXAONE actual baseline token count differs")
    prompt_text = _prompt_text(bundle.tokenizer, prompt_ids)
    mx.synchronize()
    started = perf_counter_ns()
    encoded_prompt = tuple(
        int(value)
        for value in bundle.tokenizer.encode(prompt_text, add_special_tokens=False)
    )
    tokenized = perf_counter_ns()
    output = greedy_generate(bundle.model, encoded_prompt, maximum_tokens=output_tokens)
    generated = perf_counter_ns()
    decoded = bundle.tokenizer.decode(
        encoded_prompt + output,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    mx.synchronize()
    stopped = perf_counter_ns()
    tokenization_ns, generation_ns, detokenization_ns, elapsed_ns = _timing_parts(
        started=started,
        tokenized=tokenized,
        generated=generated,
        stopped=stopped,
    )
    decoded_hash = _validate_decoded_sequence(
        bundle.tokenizer, encoded_prompt, output, decoded
    )
    expected_offset = len(encoded_prompt) + len(output) - 1
    return ActualGenerationTrial(
        accepted_draft_tokens=0,
        bonus_tokens=0,
        corpus_accepted_draft_tokens=0,
        corpus_proposal_calls=0,
        corpus_proposed_tokens=0,
        correction_tokens=0,
        decoded_utf8_sha256=decoded_hash,
        detokenization_ns=detokenization_ns,
        elapsed_ns=elapsed_ns,
        final_cache_offset=expected_offset,
        full_accept_cycles=0,
        generation_ns=generation_ns,
        immediate_reject_cycles=0,
        no_proposal_calls=output_tokens,
        output_token_ids=output,
        output_token_sha256=token_sequence_sha256(output),
        partial_accept_cycles=0,
        prompt_accepted_draft_tokens=0,
        prompt_proposal_calls=0,
        prompt_proposed_tokens=0,
        prompt_token_count=len(encoded_prompt),
        proposal_attempts=0,
        proposed_tokens=0,
        target_generation_forward_calls=output_tokens,
        target_prefill_forward_calls=1 if len(encoded_prompt) > 1 else 0,
        tokenization_ns=tokenization_ns,
    )


def _accepted_prefix(proposal: Sequence[int], output: Sequence[int]) -> int:
    accepted = 0
    for proposed, actual in zip(proposal, output):
        if int(proposed) != int(actual):
            break
        accepted += 1
    return accepted


def run_actual_candidate_trial(
    bundle: ExaoneRuntimeBundle,
    prompt_ids: Sequence[int],
    *,
    output_tokens: int,
    maximum_draft_tokens: int,
) -> ActualGenerationTrial:
    if bundle.table is None:
        raise ValueError("EXAONE actual candidate table is missing")
    if output_tokens <= 0 or maximum_draft_tokens <= 0:
        raise ValueError("EXAONE actual candidate token limits differ")
    prompt_text = _prompt_text(bundle.tokenizer, prompt_ids)
    events: list[tuple[int, tuple[int, ...], str]] = []

    def provider(
        history: tuple[int, ...], remaining: int, emitted: int
    ) -> tuple[int, ...]:
        proposal, source = hybrid_retrieval_proposal(bundle.table, history)
        proposal = tuple(proposal[: min(maximum_draft_tokens, remaining)])
        events.append((emitted, proposal, source))
        return proposal

    mx.synchronize()
    started = perf_counter_ns()
    encoded_prompt = tuple(
        int(value)
        for value in bundle.tokenizer.encode(prompt_text, add_special_tokens=False)
    )
    tokenized = perf_counter_ns()
    trace = forced_speculative_generate(
        bundle.model,
        encoded_prompt,
        maximum_tokens=output_tokens,
        maximum_draft_tokens=maximum_draft_tokens,
        proposal_provider=provider,
    )
    generated = perf_counter_ns()
    decoded = bundle.tokenizer.decode(
        encoded_prompt + trace.token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    mx.synchronize()
    stopped = perf_counter_ns()
    tokenization_ns, generation_ns, detokenization_ns, elapsed_ns = _timing_parts(
        started=started,
        tokenized=tokenized,
        generated=generated,
        stopped=stopped,
    )
    decoded_hash = _validate_decoded_sequence(
        bundle.tokenizer, encoded_prompt, trace.token_ids, decoded
    )

    corpus_calls = 0
    prompt_calls = 0
    corpus_proposed_tokens = 0
    prompt_proposed_tokens = 0
    corpus_accepted_tokens = 0
    prompt_accepted_tokens = 0
    no_proposal_calls = 0
    proposed_tokens = 0
    accepted_tokens = 0
    full_cycles = 0
    immediate_cycles = 0
    partial_cycles = 0
    correction_tokens = 0
    bonus_tokens = 0
    for emitted, proposal, source in events:
        if source == "corpus_ngram":
            corpus_calls += 1
            corpus_proposed_tokens += len(proposal)
        elif source == "prompt_lookup":
            prompt_calls += 1
            prompt_proposed_tokens += len(proposal)
        elif source == "none":
            no_proposal_calls += 1
        else:
            raise ValueError("EXAONE actual proposal source differs")
        if not proposal:
            if source != "none":
                raise ValueError("EXAONE empty proposal source differs")
            continue
        proposed_tokens += len(proposal)
        accepted = _accepted_prefix(proposal, trace.token_ids[emitted:])
        accepted_tokens += accepted
        if source == "corpus_ngram":
            corpus_accepted_tokens += accepted
        elif source == "prompt_lookup":
            prompt_accepted_tokens += accepted
        if accepted == len(proposal):
            full_cycles += 1
            if emitted + len(proposal) < len(trace.token_ids):
                bonus_tokens += 1
        elif accepted == 0:
            immediate_cycles += 1
            correction_tokens += 1
        else:
            partial_cycles += 1
            correction_tokens += 1

    proposal_attempts = corpus_calls + prompt_calls
    if (
        len(events) != trace.target_forward_calls
        or proposal_attempts + no_proposal_calls != trace.target_forward_calls
        or full_cycles != trace.full_accept_cycles
        or immediate_cycles != trace.immediate_reject_cycles
        or partial_cycles != trace.partial_accept_cycles
        or full_cycles + immediate_cycles + partial_cycles != proposal_attempts
        or accepted_tokens > proposed_tokens
        or trace.final_cache_offset != len(encoded_prompt) + len(trace.token_ids) - 1
    ):
        raise ValueError("EXAONE actual candidate counter identity differs")

    return ActualGenerationTrial(
        accepted_draft_tokens=accepted_tokens,
        bonus_tokens=bonus_tokens,
        corpus_accepted_draft_tokens=corpus_accepted_tokens,
        corpus_proposal_calls=corpus_calls,
        corpus_proposed_tokens=corpus_proposed_tokens,
        correction_tokens=correction_tokens,
        decoded_utf8_sha256=decoded_hash,
        detokenization_ns=detokenization_ns,
        elapsed_ns=elapsed_ns,
        final_cache_offset=trace.final_cache_offset,
        full_accept_cycles=full_cycles,
        generation_ns=generation_ns,
        immediate_reject_cycles=immediate_cycles,
        no_proposal_calls=no_proposal_calls,
        output_token_ids=trace.token_ids,
        output_token_sha256=token_sequence_sha256(trace.token_ids),
        partial_accept_cycles=partial_cycles,
        prompt_accepted_draft_tokens=prompt_accepted_tokens,
        prompt_proposal_calls=prompt_calls,
        prompt_proposed_tokens=prompt_proposed_tokens,
        prompt_token_count=len(encoded_prompt),
        proposal_attempts=proposal_attempts,
        proposed_tokens=proposed_tokens,
        target_generation_forward_calls=trace.target_forward_calls,
        target_prefill_forward_calls=1 if len(encoded_prompt) > 1 else 0,
        tokenization_ns=tokenization_ns,
    )
