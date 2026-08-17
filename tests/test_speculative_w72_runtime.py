import unittest
from dataclasses import dataclass

import torch
from torch import nn

from scripts.hangul_draft_acceptance_core import (
    DeviceHangulTables,
    build_head,
    pair_index,
    propose_pairs,
)
from scripts.speculative_w72_runtime import (
    IndependentProposalEngine,
    generate_baseline,
    generate_speculative,
)
from jamoflow.utf8 import strict_utf8_allowed_ranges, strict_utf8_reachable_states


class _FakeTargetModel:
    def __init__(self):
        self.model = type("Base", (), {})()
        self.model.local_decoder = type("Decoder", (), {})()
        self.model.local_decoder.norm = nn.Identity()


@dataclass(frozen=True)
class _FakeDiagnostics:
    observed_bytes: int


class _FakeTransaction:
    def __init__(self, runtime, values):
        self.runtime = runtime
        self.start = runtime.position
        self.values = values
        self.logits = torch.cat(
            [runtime.logits(self.start + offset + 1) for offset in range(len(values))],
            dim=0,
        )
        self.hidden = torch.stack(
            [runtime.hidden(self.start + offset) for offset in range(len(values))]
        )
        self.closed = False

    def finish(self, keep):
        if self.closed:
            raise RuntimeError("already closed")
        self.closed = True
        self.runtime.position = self.start + keep
        if keep:
            return self.logits[keep - 1 : keep], self.hidden[keep - 1 : keep]
        return self.logits[:0], self.hidden[:0]


class _FakeRuntime:
    def __init__(self, target: bytes):
        self.target = target
        self.position = 0
        self.prompt_bytes = 0
        self.model = _FakeTargetModel()

    @property
    def diagnostics(self):
        return _FakeDiagnostics(self.prompt_bytes + self.position)

    def hidden(self, position):
        row = torch.zeros(192)
        row[position % 192] = 1
        return row

    def logits(self, position):
        logits = torch.full((1, 256), -100.0)
        logits[0, self.target[position] if position < len(self.target) else 0] = 100.0
        return logits

    def prefill_parallel(self, prompt):
        self.prompt_bytes = len(prompt)
        self.model.model.local_decoder.norm(self.hidden(0).reshape(1, 1, -1))
        return self.logits(0)

    def consume(self, value):
        if value != self.target[self.position]:
            raise AssertionError("fake target consumed a non-greedy byte")
        self.position += 1
        self.model.model.local_decoder.norm(
            self.hidden(self.position).reshape(1, 1, -1)
        )
        return self.logits(self.position)

    def consume_block_transaction(self, values):
        if values[0] != self.target[self.position]:
            raise AssertionError("fake block did not start with the pending target byte")
        return _FakeTransaction(self, values)


def _masks():
    output = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool)
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        output[state] = mask
    return output


class SpeculativeW72RuntimeTest(unittest.TestCase):
    def test_exact_state_machine_covers_first_second_and_full_acceptance(self):
        target = ("가" * 4).encode("utf-8")
        wrong_second = "걀".encode("utf-8")
        self.assertNotEqual(wrong_second[1], target[1])
        proposals = iter(
            (
                pair_index(wrong_second[1], wrong_second[2]),
                pair_index(target[1], target[2] + 1),
                pair_index(target[1], target[2]),
            )
        )

        def proposal(_head, _hidden, _lead, _tables):
            return torch.tensor([next(proposals)], dtype=torch.long)

        baseline = generate_baseline(
            _FakeRuntime(target), b"prompt", _masks(),
            minimum_output_bytes=9, maximum_output_bytes=12,
        )
        speculative = generate_speculative(
            _FakeRuntime(target), None, b"prompt", _masks(),
            DeviceHangulTables.build("cpu"),
            minimum_output_bytes=9, maximum_output_bytes=12,
            proposal=proposal,
        )
        self.assertEqual(speculative.generated, baseline.generated)
        self.assertEqual(speculative.diagnostics, baseline.diagnostics)
        self.assertEqual(speculative.counters["first_mismatches"], 1)
        self.assertEqual(speculative.counters["second_mismatches"], 1)
        self.assertEqual(speculative.counters["complete_pair_accepts"], 1)

    def test_optimized_independent_proposal_matches_original_candidate_order(self):
        torch.manual_seed(7)
        head = build_head("generic_independent_utf8").eval()
        engine = IndependentProposalEngine(head, "cpu")
        tables = DeviceHangulTables.build("cpu")
        with torch.inference_mode():
            for lead_value in range(4):
                hidden = torch.randn(17, 192)
                lead = torch.full((17,), lead_value, dtype=torch.long)
                expected = propose_pairs(head, hidden, lead, tables)
                actual = torch.tensor(
                    [
                        engine.propose(hidden[index : index + 1], lead_value)
                        for index in range(len(hidden))
                    ],
                    dtype=torch.long,
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_retry_third_is_valid_for_the_corrected_hangul_prefix(self):
        torch.manual_seed(13)
        head = build_head("generic_independent_utf8").eval()
        engine = IndependentProposalEngine(head, "cpu")
        with torch.inference_mode():
            hidden = torch.randn(1, 192)
            for lead_value in range(4):
                _, third_logits = engine.propose_with_context(hidden, lead_value)
                allowed_pairs = engine.allowed_pairs[lead_value]
                for second_index in torch.unique(
                    allowed_pairs // 64, sorted=True
                ).tolist():
                    third = engine.retry_third(
                        third_logits,
                        lead_value,
                        0x80 + int(second_index),
                    )
                    self.assertIsNotNone(third)
                    pair = int(second_index) * 64 + int(third) - 0x80
                    self.assertIn(pair, allowed_pairs.tolist())


if __name__ == "__main__":
    unittest.main()
