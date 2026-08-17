import unittest

import numpy as np

from scripts.fresh_vocabulary_16k_retrieval_core import (
    MAXIMUM_TABLE_ENTRIES,
    CompactBackoffTable,
    OrderTable,
    build_compact_backoff_table,
    committed_token_count,
    hybrid_retrieval_proposal,
    pack_context,
    prompt_lookup_proposal,
    proposal_acceptance,
    table_from_arrays,
)


def _manual_table() -> CompactBackoffTable:
    rows = {}
    for order, context, token in (
        (1, (7,), 8),
        (2, (6, 7), 9),
        (3, (5, 6, 7), 10),
    ):
        rows[order] = OrderTable(
            order=order,
            contexts=np.asarray([pack_context(context)], dtype=np.uint64),
            next_tokens=np.asarray([token], dtype=np.uint16),
            best_counts=np.asarray([8], dtype=np.uint32),
            total_counts=np.asarray([10], dtype=np.uint32),
        )
    table = CompactBackoffTable(rows)
    table.validate()
    return table


class RetrievalDraftCoreTest(unittest.TestCase):
    def test_prompt_lookup_prefers_longest_then_earliest_match(self):
        history = (1, 2, 3, 8, 1, 2, 3, 9, 1, 2, 3)
        self.assertEqual(prompt_lookup_proposal(history), (8, 1, 2))
        self.assertEqual(prompt_lookup_proposal((4, 5, 6)), ())

    def test_backoff_draft_recurses_and_round_trips_arrays(self):
        table = _manual_table()
        self.assertEqual(table.next_token((5, 6, 7)), 10)
        self.assertEqual(table.next_token((1, 6, 7)), 9)
        self.assertEqual(table.next_token((1, 2, 7)), 8)
        self.assertEqual(table.propose((5, 6, 7)), (10,))
        restored = table_from_arrays(table.to_arrays())
        self.assertEqual(restored.propose((5, 6, 7)), (10,))

    def test_hybrid_uses_dictionary_then_prompt_then_none(self):
        table = _manual_table()
        self.assertEqual(
            hybrid_retrieval_proposal(table, (5, 6, 7)),
            ((10,), "corpus_ngram"),
        )
        self.assertEqual(
            hybrid_retrieval_proposal(table, (1, 2, 1, 2)),
            ((1, 2), "prompt_lookup"),
        )
        self.assertEqual(hybrid_retrieval_proposal(table, (1, 2, 3)), ((), "none"))

    def test_compact_builder_selects_deterministic_high_confidence_rows(self):
        # Each order has at least one >=5-count deterministic continuation.
        tokens = np.asarray(([1, 2, 3, 4] * 12) + ([9, 8, 7, 6] * 7), dtype=np.int64)
        table = build_compact_backoff_table(
            tokens,
            maximum_entries=MAXIMUM_TABLE_ENTRIES,
        )
        table.validate()
        self.assertLessEqual(table.entry_count, MAXIMUM_TABLE_ENTRIES)
        self.assertEqual(table.next_token((1, 2, 3)), 4)
        self.assertEqual(table.next_token((9, 8, 7)), 6)
        self.assertEqual(table_from_arrays(table.to_arrays()).entry_count, table.entry_count)

    def test_acceptance_and_committed_count_include_correction_or_bonus(self):
        self.assertEqual(proposal_acceptance((1, 2, 3), (1, 2, 9)), 2)
        self.assertEqual(proposal_acceptance((1, 2), (9, 2)), 0)
        self.assertEqual(committed_token_count(3, 0), 1)
        self.assertEqual(committed_token_count(3, 2), 3)
        self.assertEqual(committed_token_count(3, 3), 4)


if __name__ == "__main__":
    unittest.main()
