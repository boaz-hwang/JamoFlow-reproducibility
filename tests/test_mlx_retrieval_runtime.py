from __future__ import annotations

import mlx.core as mx

from mlx_retrieval_runtime import (
    forced_speculative_generate,
    full_vs_incremental_equivalence,
    greedy_generate,
    prefill_decode_equivalence,
    rollback_equivalence,
)


class _FakeCache:
    def __init__(self):
        self.tokens: list[int] = []
        self.offset = 0

    def is_trimmable(self):
        return True

    def trim(self, count):
        count = min(self.offset, int(count))
        if count:
            self.tokens = self.tokens[:-count]
        self.offset -= count
        return count


class _FakeModel:
    vocab_size = 32

    def make_cache(self):
        return [_FakeCache()]

    def __call__(self, values, cache=None):
        rows = []
        for token_id in values.tolist()[0]:
            token_id = int(token_id)
            logits = [-100.0] * self.vocab_size
            logits[(token_id + 1) % self.vocab_size] = 100.0
            rows.append(logits)
            if cache is not None:
                cache[0].tokens.append(token_id)
                cache[0].offset += 1
        return mx.array([rows], dtype=mx.float32)


class _ShapeSensitiveFakeModel(_FakeModel):
    def __call__(self, values, cache=None):
        rows = []
        cached = cache is not None
        for token_id in values.tolist()[0]:
            token_id = int(token_id)
            logits = [-100.0] * self.vocab_size
            logits[(token_id + 1) % self.vocab_size] = 100.0
            logits[(token_id + 2) % self.vocab_size] = 1.0 if cached else 0.0
            rows.append(logits)
            if cache is not None:
                cache[0].tokens.append(token_id)
                cache[0].offset += 1
        return mx.array([rows], dtype=mx.float32)


def test_full_incremental_and_rollback_are_exact() -> None:
    model = _FakeModel()
    full = full_vs_incremental_equivalence(
        model, (1, 2, 3, 4), atol=1e-6, rtol=1e-6
    )
    assert full["pass"] is True
    assert full["argmax_exact"] is True
    assert full["maximum_normalized_error"] == 0.0

    rollback = rollback_equivalence(
        model,
        (1, 2, 3),
        (4, 5, 6),
        keep_speculative_tokens=1,
        correction_token_id=9,
        atol=1e-6,
        rtol=1e-6,
    )
    assert rollback["decision_equivalence_pass"] is True
    assert rollback["numeric_tolerance_pass"] is True
    assert rollback["argmax_exact"] is True

    decode, output = prefill_decode_equivalence(
        model,
        (1, 2, 3, 4),
        maximum_tokens=8,
        atol=1e-6,
        rtol=1e-6,
    )
    assert decode["decision_equivalence_pass"] is True
    assert decode["numeric_tolerance_pass"] is True
    assert decode["comparison_positions"] == 4 + 8 - 1
    assert output == tuple(range(5, 13))


def test_greedy_and_all_forced_paths_match() -> None:
    model = _FakeModel()
    prompt = (1, 2)
    maximum_tokens = 8
    baseline = greedy_generate(model, prompt, maximum_tokens=maximum_tokens)
    assert baseline == tuple(range(3, 11))

    def full(_context, remaining, output_index):
        return baseline[output_index : output_index + min(3, remaining)]

    def reject(_context, remaining, output_index):
        width = min(3, remaining)
        return tuple((baseline[output_index] + 1) % model.vocab_size for _ in range(width))

    def partial(_context, remaining, output_index):
        if remaining < 2:
            return ()
        width = min(3, remaining)
        return (baseline[output_index],) + tuple(
            (baseline[output_index + 1] + 1) % model.vocab_size
            for _ in range(width - 1)
        )

    full_trace = forced_speculative_generate(
        model,
        prompt,
        maximum_tokens=maximum_tokens,
        maximum_draft_tokens=3,
        proposal_provider=full,
    )
    reject_trace = forced_speculative_generate(
        model,
        prompt,
        maximum_tokens=maximum_tokens,
        maximum_draft_tokens=3,
        proposal_provider=reject,
    )
    partial_trace = forced_speculative_generate(
        model,
        prompt,
        maximum_tokens=maximum_tokens,
        maximum_draft_tokens=3,
        proposal_provider=partial,
    )
    assert full_trace.token_ids == reject_trace.token_ids == partial_trace.token_ids == baseline
    assert full_trace.full_accept_cycles > 0
    assert reject_trace.immediate_reject_cycles > 0
    assert partial_trace.partial_accept_cycles > 0
    assert full_trace.final_cache_offset == len(prompt) + maximum_tokens - 1
    assert reject_trace.final_cache_offset == len(prompt) + maximum_tokens - 1
    assert partial_trace.final_cache_offset == len(prompt) + maximum_tokens - 1


def test_decision_equivalence_is_distinct_from_all_logit_numeric_diagnostic() -> None:
    result, output = prefill_decode_equivalence(
        _ShapeSensitiveFakeModel(),
        (1, 2, 3, 4),
        maximum_tokens=4,
        atol=1e-6,
        rtol=1e-6,
    )
    assert output == (5, 6, 7, 8)
    assert result["argmax_exact"] is True
    assert result["finite"] is True
    assert result["decision_equivalence_pass"] is True
    assert result["numeric_tolerance_pass"] is False
    assert result["maximum_normalized_error"] > 1.0


def test_terminal_full_accept_keeps_lagging_cache() -> None:
    model = _FakeModel()
    prompt = (4, 5)
    baseline = greedy_generate(model, prompt, maximum_tokens=3)

    trace = forced_speculative_generate(
        model,
        prompt,
        maximum_tokens=3,
        maximum_draft_tokens=3,
        proposal_provider=lambda _context, _remaining, _index: baseline,
    )
    assert trace.token_ids == baseline
    assert trace.final_cache_offset == len(prompt) + len(baseline) - 1
