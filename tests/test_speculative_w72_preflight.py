import unittest

import numpy as np

from scripts.speculative_w72_preflight_core import (
    COUNTER_KEYS,
    PROMPT_COUNT,
    REPETITIONS,
    summarize_speculative_preflight,
)


class SpeculativeW72PreflightTest(unittest.TestCase):
    @staticmethod
    def _inputs(speculative_ms: float = 70.0):
        timings = np.empty((PROMPT_COUNT, REPETITIONS, 2), dtype=np.float64)
        timings[..., 0] = 100.0
        timings[..., 1] = speculative_ms
        lengths = np.full(PROMPT_COUNT, 129, dtype=np.int64)
        counters = np.zeros((PROMPT_COUNT, len(COUNTER_KEYS)), dtype=np.int64)
        col = {key: COUNTER_KEYS.index(key) for key in COUNTER_KEYS}
        counters[:, col["emitted_bytes"]] = 129
        counters[:, col["sequential_target_calls"]] = 35
        counters[:, col["target_block_calls"]] = 55
        counters[:, col["draft_head_calls"]] = 40
        counters[:, col["first_draft_accepts"]] = 20
        counters[:, col["first_mismatches"]] = 20
        counters[:, col["complete_pair_accepts"]] = 12
        counters[:, col["second_mismatches"]] = 8
        counters[:, col["retry_block_calls"]] = 20
        counters[:, col["retry_third_accepts"]] = 3
        counters[:, col["retry_third_mismatches"]] = 17
        return timings, lengths, counters

    def test_summary_requires_actual_twenty_percent_with_ten_percent_lower(self):
        timings, lengths, counters = self._inputs()
        summary = summarize_speculative_preflight(
            timings_ms=timings,
            output_lengths=lengths,
            speculative_counters=counters,
            correctness={
                "all_outputs_exact": True,
                "cache_comparisons": PROMPT_COUNT,
                "output_comparisons": PROMPT_COUNT,
                "output_hash_root_sha256": "a" * 64,
            },
            minimum_point_reduction=0.20,
            minimum_lower_bound=0.10,
            minimum_positive_prompts=96,
        )
        self.assertTrue(summary["gates"]["multi_seed_generic_comparator_authorized"])
        self.assertAlmostEqual(summary["end_to_end"]["reduction"], 0.30)
        slow, lengths, counters = self._inputs(speculative_ms=85.0)
        failed = summarize_speculative_preflight(
            timings_ms=slow,
            output_lengths=lengths,
            speculative_counters=counters,
            correctness=summary["correctness"],
            minimum_point_reduction=0.20,
            minimum_lower_bound=0.10,
            minimum_positive_prompts=96,
        )
        self.assertFalse(failed["gates"]["multi_seed_generic_comparator_authorized"])


if __name__ == "__main__":
    unittest.main()
