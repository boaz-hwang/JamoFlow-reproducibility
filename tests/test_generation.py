import unittest

import numpy as np

from jamoflow.generation import (
    continuation_diagnostic_arrays,
    continuation_metrics,
    continuation_metrics_from_diagnostics,
    generation_patch_matrix,
    greedy_byte,
    select_generation_prompts,
    top_p_sample,
    utf8_allowed_next_bytes,
    utf8_failure_diagnostic_arrays,
    utf8_failure_metrics,
    utf8_failure_metrics_from_diagnostics,
    valid_completion_metrics,
    valid_conjoining_jamo_transitions,
)
from jamoflow.phase1 import patch_boundaries_from_lengths
from jamoflow.utf8 import prefix_boundary_mask


class GenerationTests(unittest.TestCase):
    def test_prompt_selection_is_valid_deduplicated_and_deterministic(self) -> None:
        prompts = [
            ("한" * 42 + suffix).encode("utf-8")
            for suffix in ("AA", "BB", "CC")
        ]
        self.assertTrue(all(len(prompt) == 128 for prompt in prompts))
        rows = [prompt + prompt for prompt in (prompts[2], prompts[0], prompts[1], prompts[0])]
        inputs = np.stack([np.frombuffer(row, dtype=np.uint8) for row in rows])
        masks = np.stack(
            [np.frombuffer(prefix_boundary_mask(row)[:-1], dtype=np.uint8) for row in rows]
        )
        first = select_generation_prompts(inputs, masks, prompt_count=3)
        second = select_generation_prompts(inputs[::-1], masks[::-1], prompt_count=3)
        np.testing.assert_array_equal(first.prompts, second.prompts)
        self.assertEqual(first.candidate_count, 4)
        self.assertEqual(first.unique_candidate_count, 3)

    def test_generation_patch_prefix_decisions_do_not_change(self) -> None:
        raw = ("한국어 문서 테스트\n" * 30).encode("utf-8")[:256]
        raw = raw.ljust(256, b" ")
        full = np.frombuffer(raw, dtype=np.uint8)[None, :]
        prefix = full[:, :128]
        for policy in ("causal_codepoint_grid", "causal_whitespace_grid"):
            partial_matrix = generation_patch_matrix(prefix, policy)
            full_matrix = generation_patch_matrix(full, policy)
            partial_starts = patch_boundaries_from_lengths(partial_matrix)[0]
            full_starts = patch_boundaries_from_lengths(full_matrix)[0]
            np.testing.assert_array_equal(
                partial_starts,
                full_starts[full_starts < 128],
            )

    def test_utf8_mask_enforces_scalar_ranges_and_horizon_closure(self) -> None:
        last = utf8_allowed_next_bytes(b"", remaining_bytes_after_choice=0)
        self.assertTrue(last[:0x80].all())
        self.assertFalse(last[0x80:].any())

        after_e0 = utf8_allowed_next_bytes(
            b"\xe0",
            remaining_bytes_after_choice=1,
        )
        self.assertFalse(after_e0[0x9F])
        self.assertTrue(after_e0[0xA0])
        self.assertTrue(after_e0[0xBF])

        invalid = utf8_allowed_next_bytes(
            b"\xff",
            remaining_bytes_after_choice=3,
        )
        self.assertFalse(invalid.any())

    def test_sampling_and_greedy_respect_allowed_mask(self) -> None:
        logits = np.arange(256, dtype=np.float64)
        allowed = np.zeros(256, dtype=bool)
        allowed[[3, 7]] = True
        self.assertEqual(greedy_byte(logits, allowed), 7)
        rng = np.random.default_rng(1)
        observed = {
            top_p_sample(logits, rng, temperature=1.0, top_p=1.0, allowed=allowed)
            for _ in range(50)
        }
        self.assertTrue(observed <= {3, 7})

    def test_jamo_and_continuation_metrics(self) -> None:
        self.assertTrue(valid_conjoining_jamo_transitions("한각국"))
        self.assertTrue(valid_conjoining_jamo_transitions("ㅋㅋ"))
        self.assertFalse(valid_conjoining_jamo_transitions("ᄀX"))
        self.assertFalse(valid_conjoining_jamo_transitions("ᅡ"))
        metrics = continuation_metrics(["ABC".encode(), b"\xffAA"])
        self.assertEqual(metrics.valid_utf8_count, 1)
        self.assertEqual(metrics.valid_utf8_rate, 0.5)
        self.assertEqual(metrics.valid_jamo_transition_rate, 0.5)

    def test_variable_valid_completion_metrics_preserve_overshoot(self) -> None:
        metrics = valid_completion_metrics(
            [b"A" * 128, ("한" * 43).encode("utf-8")],
            minimum_completion_bytes=128,
        )
        self.assertEqual(metrics.valid_utf8_rate, 1.0)
        self.assertEqual(metrics.minimum_emitted_bytes, 128)
        self.assertEqual(metrics.maximum_emitted_bytes, 129)
        self.assertEqual(metrics.maximum_overshoot_bytes, 1)
        with self.assertRaisesRegex(ValueError, "before"):
            valid_completion_metrics(
                [b"short"],
                minimum_completion_bytes=128,
            )

    def test_continuation_diagnostics_reconstruct_and_reject_tampering(self) -> None:
        values = [b"ABC", b"\xffAA"]
        diagnostics, continuation_bytes = continuation_diagnostic_arrays(values)
        self.assertEqual(
            continuation_metrics_from_diagnostics(
                diagnostics,
                continuation_bytes,
            ),
            continuation_metrics(values),
        )
        tampered = {key: value.copy() for key, value in diagnostics.items()}
        tampered["strict_valid"][0] = 2
        with self.assertRaises(ValueError):
            continuation_metrics_from_diagnostics(
                tampered,
                continuation_bytes,
            )

    def test_utf8_failure_taxonomy_partitions_outputs(self) -> None:
        metrics = utf8_failure_metrics(
            [b"A", b"B", b"C"],
            [b"xy", b"\xffz", b"\xe2\x82"],
        )
        self.assertEqual(metrics.strict_valid_count, 1)
        self.assertEqual(metrics.illegal_transition_count, 1)
        self.assertEqual(metrics.incomplete_terminal_scalar_count, 1)
        self.assertAlmostEqual(metrics.mean_legal_prefix_bytes, 4 / 3)
        self.assertAlmostEqual(metrics.mean_legal_prefix_fraction, 2 / 3)
        self.assertAlmostEqual(metrics.mean_closed_codepoint_prefix_bytes, 2 / 3)
        self.assertAlmostEqual(
            metrics.mean_closed_codepoint_prefix_fraction,
            1 / 3,
        )
        self.assertEqual(metrics.mean_first_illegal_byte_position, 0.0)

    def test_utf8_failure_taxonomy_requires_closed_prompts(self) -> None:
        with self.assertRaises(ValueError):
            utf8_failure_metrics([b"\xe2"], [b"AB"])

    def test_utf8_failure_diagnostics_reconstruct_and_reject_fractions(self) -> None:
        prompts = [b"A", b"B", b"C"]
        continuations = [b"xy", b"\xffz", b"\xe2\x82"]
        diagnostics, continuation_bytes = utf8_failure_diagnostic_arrays(
            prompts,
            continuations,
        )
        self.assertEqual(
            utf8_failure_metrics_from_diagnostics(
                diagnostics,
                continuation_bytes,
            ),
            utf8_failure_metrics(prompts, continuations),
        )
        tampered = {key: value.copy() for key, value in diagnostics.items()}
        tampered["failure_category"] = tampered["failure_category"].astype(
            np.float64
        )
        tampered["failure_category"][0] = 0.5
        with self.assertRaises(ValueError):
            utf8_failure_metrics_from_diagnostics(
                tampered,
                continuation_bytes,
            )


if __name__ == "__main__":
    unittest.main()
