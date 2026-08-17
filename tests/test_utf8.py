import unittest

from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    advance_strict_utf8,
    codepoint_spans,
    compile_strict_utf8_token_transitions,
    prefix_boundary_mask,
    prefix_codepoint_predicate_mask,
    scan_prefix_states,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
    strict_utf8_state,
)


class Utf8Tests(unittest.TestCase):
    def test_prefix_boundaries_for_multibyte_text(self) -> None:
        data = "한A🙂".encode("utf-8")
        states = scan_prefix_states(data)
        boundary_positions = [
            index for index, state in enumerate(states) if state.at_codepoint_boundary
        ]

        self.assertEqual(boundary_positions, [0, 3, 4, 8])
        self.assertEqual(states[3].completed_codepoint, ord("한"))
        self.assertEqual(states[4].completed_codepoint, ord("A"))
        self.assertEqual(states[8].completed_codepoint, ord("🙂"))

    def test_codepoint_spans_recover_valid_text(self) -> None:
        text = "한A🙂"
        spans = codepoint_spans(text.encode("utf-8"))

        self.assertEqual([span.codepoint for span in spans], [ord(char) for char in text])
        self.assertTrue(all(span.valid for span in spans))

    def test_invalid_byte_is_reported(self) -> None:
        spans = codepoint_spans(b"A\xffB")
        self.assertEqual([span.valid for span in spans], [True, False, True])

    def test_compact_boundary_mask_matches_prefix_states(self) -> None:
        samples = [
            "한A🙂".encode("utf-8"),
            b"A\xffB",
            b"\xe1\x80A",
            b"\xf0\x9f\x99",
        ]
        for data in samples:
            with self.subTest(data=data):
                expected = bytearray(
                    state.at_codepoint_boundary for state in scan_prefix_states(data)
                )
                self.assertEqual(prefix_boundary_mask(data), expected)

    def test_compact_predicate_mask_matches_completed_codepoints(self) -> None:
        samples = ["한 A🙂".encode("utf-8"), b"A\xff B", b"\xe1\x80A"]
        for data in samples:
            with self.subTest(data=data):
                expected = bytearray(
                    state.completed_codepoint == ord(" ")
                    for state in scan_prefix_states(data)
                )
                actual = prefix_codepoint_predicate_mask(
                    data,
                    lambda codepoint: codepoint == ord(" "),
                )
                self.assertEqual(actual, expected)

    def test_strict_dfa_rejects_overlong_surrogate_and_out_of_range(self) -> None:
        for data in (b"\xc0\x80", b"\xed\xa0\x80", b"\xf4\x90\x80\x80"):
            with self.subTest(data=data):
                self.assertFalse(strict_utf8_state(data).valid)
        self.assertTrue(strict_utf8_state("한🙂".encode()).at_codepoint_boundary)
        self.assertTrue(strict_utf8_state(b"\xe2\x82").valid)
        self.assertFalse(strict_utf8_state(b"\xe2\x82").at_codepoint_boundary)

    def test_streaming_allowed_ranges_cover_every_reachable_transition(self) -> None:
        states = strict_utf8_reachable_states()
        self.assertEqual(len(states), 8)
        self.assertEqual(states[0], STRICT_UTF8_INITIAL_STATE)
        for state in states:
            ranges = strict_utf8_allowed_ranges(state)
            allowed = {
                value
                for lower, upper in ranges
                for value in range(lower, upper + 1)
            }
            self.assertTrue(allowed)
            for value in range(256):
                advanced = advance_strict_utf8(state, value)
                self.assertEqual(advanced.valid, value in allowed)
                if advanced.valid:
                    self.assertIn(advanced, states)

    def test_finite_budget_prevents_an_unclosed_scalar(self) -> None:
        ranges = strict_utf8_allowed_ranges(
            STRICT_UTF8_INITIAL_STATE,
            remaining_bytes_after_choice=0,
        )
        self.assertEqual(ranges, ((0, 0x7F),))

    def test_token_transition_table_handles_split_and_complete_scalars(self) -> None:
        token_bytes = tuple(bytes((value,)) for value in range(256)) + (
            "あ".encode("utf-8"),
        )
        table = compile_strict_utf8_token_transitions(token_bytes)
        initial = table.states.index(STRICT_UTF8_INITIAL_STATE)
        after_lead = table.transition(initial, 0xE3)
        self.assertIsNotNone(after_lead)
        assert after_lead is not None
        after_first_continuation = table.transition(after_lead, 0x81)
        self.assertIsNotNone(after_first_continuation)
        assert after_first_continuation is not None
        self.assertEqual(
            table.transition(after_first_continuation, 0x82),
            initial,
        )
        self.assertEqual(table.transition(initial, 256), initial)
        self.assertIsNone(table.transition(initial, 0x81))
        self.assertIsNone(table.transition(initial, 0xFF))
        self.assertEqual(table.maximum_token_bytes, 3)
        self.assertEqual(len(table.token_bytes_sha256), 64)
        self.assertEqual(len(table.transition_table_sha256), 64)

    def test_token_transition_table_matches_bytewise_dfa_exhaustively(self) -> None:
        token_bytes = tuple(bytes((value,)) for value in range(256)) + (
            "한글".encode("utf-8"),
        )
        table = compile_strict_utf8_token_transitions(token_bytes)
        state_indices = {state: index for index, state in enumerate(table.states)}
        for state_index, state in enumerate(table.states):
            for token_id, values in enumerate(token_bytes):
                advanced = state
                for value in values:
                    advanced = advance_strict_utf8(advanced, value)
                    if not advanced.valid:
                        break
                expected = state_indices.get(advanced)
                self.assertEqual(
                    table.transition(state_index, token_id),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
