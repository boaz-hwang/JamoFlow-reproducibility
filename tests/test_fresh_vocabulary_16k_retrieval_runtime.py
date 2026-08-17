import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.fresh_vocabulary_16k_retrieval_core import (
    CompactBackoffTable,
    OrderTable,
    pack_context,
)
from scripts.fresh_vocabulary_16k_retrieval_runtime import (
    decode_retrieval_after_prefill,
)


def _table(first_context=7, first_next=8):
    rows = {}
    for order, context, token in (
        (1, (first_context,), first_next),
        (2, (71, 72), 73),
        (3, (81, 82, 83), 84),
    ):
        rows[order] = OrderTable(
            order=order,
            contexts=np.asarray([pack_context(context)], dtype=np.uint64),
            next_tokens=np.asarray([token], dtype=np.uint16),
            best_counts=np.asarray([9], dtype=np.uint32),
            total_counts=np.asarray([10], dtype=np.uint32),
        )
    result = CompactBackoffTable(rows)
    result.validate()
    return result


class _FakeTransaction:
    def __init__(self, runtime, inputs, logits):
        self.runtime = runtime
        self.inputs = tuple(inputs)
        self.logits = logits
        self.start = runtime.observed_tokens
        self.closed = False
        runtime.observed_tokens += len(inputs)

    def finish(self, keep):
        if self.closed:
            raise RuntimeError("transaction already closed")
        self.runtime.observed_tokens = self.start + int(keep)
        self.closed = True


class _FakeRuntime:
    def __init__(self, prompt_length, expected):
        self.prompt_length = prompt_length
        self.expected = tuple(expected)
        self.observed_tokens = prompt_length
        self.finished_keeps = []

    def _logits(self, output_index):
        row = torch.full((16_000,), -100.0)
        token = self.expected[output_index] if output_index < len(self.expected) else 0
        row[token] = 100.0
        return row

    def consume_transaction(self, inputs):
        output_index = self.observed_tokens - self.prompt_length + 1
        logits = torch.stack(
            [self._logits(output_index + offset) for offset in range(len(inputs))]
        )
        transaction = _FakeTransaction(self, inputs, logits)
        original_finish = transaction.finish

        def finish(keep):
            self.finished_keeps.append(int(keep))
            original_finish(keep)

        transaction.finish = finish
        return transaction


def _bundle():
    token_bytes = tuple(
        bytes((value,)) if value < 128 else b"x" for value in range(16_000)
    )
    transitions = SimpleNamespace(
        next_state_indices=(tuple(0 for _ in range(16_000)),),
    )
    masks = (torch.ones(16_000, dtype=torch.bool),)
    return SimpleNamespace(
        transitions=transitions,
        token_bytes=token_bytes,
        masks=masks,
    )


def _initial_logits(first):
    logits = torch.full((1, 16_000), -100.0)
    logits[0, first] = 100.0
    return logits


class RetrievalRuntimeTest(unittest.TestCase):
    def test_prompt_lookup_full_accept_emits_bonus_and_preserves_lag(self):
        prompt = (10, 20, 30, 40)
        expected = (10, 20, 30, 40, 50)
        runtime = _FakeRuntime(len(prompt), expected)
        trace = decode_retrieval_after_prefill(
            _bundle(),
            runtime,
            _initial_logits(expected[0]),
            prompt,
            expected,
            bytes(expected),
            _table(),
            role="prompt_lookup_block_4",
            mode="free_running_utf8_greedy",
            continuation_bytes=len(expected),
            maximum_output_bytes=len(expected) + 3,
        )
        self.assertEqual(trace.token_ids, expected)
        self.assertEqual(trace.counters["full_accept_cycles"], 1)
        self.assertEqual(trace.counters["accepted_draft_tokens"], 3)
        self.assertEqual(trace.counters["bonus_tokens"], 1)
        self.assertEqual(trace.counters["target_forward_calls"], 2)
        self.assertEqual(runtime.finished_keeps, [4])
        self.assertEqual(runtime.observed_tokens, len(prompt) + len(expected) - 1)

    def test_corpus_rejection_crops_rejected_token_and_emits_correction(self):
        prompt = (5, 6)
        expected = (7, 8, 9, 10)
        runtime = _FakeRuntime(len(prompt), expected)
        trace = decode_retrieval_after_prefill(
            _bundle(),
            runtime,
            _initial_logits(expected[0]),
            prompt,
            expected,
            bytes(expected),
            _table(first_context=7, first_next=99),
            role="corpus_ngram_block_4",
            mode="free_running_utf8_greedy",
            continuation_bytes=len(expected),
            maximum_output_bytes=len(expected) + 3,
        )
        self.assertEqual(trace.token_ids, expected)
        self.assertEqual(trace.counters["rejection_cycles"], 1)
        self.assertEqual(trace.counters["correction_tokens"], 1)
        self.assertGreaterEqual(trace.counters["no_draft_steps"], 1)
        self.assertEqual(runtime.finished_keeps[0], 1)
        self.assertEqual(runtime.observed_tokens, len(prompt) + len(expected) - 1)

    def test_utf8_stop_inside_accepted_block_crops_terminal_token(self):
        prompt = (10, 20, 30, 40)
        expected = (10, 20, 30)
        runtime = _FakeRuntime(len(prompt), expected)
        trace = decode_retrieval_after_prefill(
            _bundle(),
            runtime,
            _initial_logits(expected[0]),
            prompt,
            expected,
            bytes(expected),
            _table(),
            role="prompt_lookup_block_4",
            mode="free_running_utf8_greedy",
            continuation_bytes=len(expected),
            maximum_output_bytes=len(expected) + 3,
        )
        self.assertEqual(trace.token_ids, expected)
        self.assertEqual(trace.counters["accepted_draft_tokens"], 2)
        self.assertEqual(trace.counters["bonus_tokens"], 0)
        self.assertEqual(runtime.finished_keeps, [2])
        self.assertEqual(runtime.observed_tokens, len(prompt) + len(expected) - 1)

    def test_controlled_replay_uses_fixed_target_tokens_but_same_transaction(self):
        prompt = (10, 20, 30, 40)
        expected = (10, 20, 30, 40, 50)
        runtime = _FakeRuntime(len(prompt), expected)
        trace = decode_retrieval_after_prefill(
            _bundle(),
            runtime,
            _initial_logits(99),
            prompt,
            expected,
            bytes(expected),
            _table(),
            role="prompt_lookup_block_4",
            mode="controlled_replay",
            continuation_bytes=len(expected),
            maximum_output_bytes=len(expected) + 3,
        )
        self.assertEqual(trace.token_ids, expected)


if __name__ == "__main__":
    unittest.main()
