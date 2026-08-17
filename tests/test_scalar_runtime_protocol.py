import json
import unittest

import numpy as np

from scripts.scalar_runtime_protocol import (
    IMPLEMENTATION_PATHS,
    MEASURED_CASES,
    PLAN_PATH,
    REPETITIONS,
    ROOT,
    RUNTIME_ROLES,
    array_sha256,
    comparison_summary,
    reconstruct_cases,
    role_schedule,
    schedule_sha256,
    validate_plan,
)
from scripts.summarize_scalar_runtime_preflight import _candidate_decision


class ScalarRuntimeProtocolTest(unittest.TestCase):
    def test_case_selection_is_deterministic_document_and_bpe_aligned(self):
        first_prompts, first_continuations, first = reconstruct_cases()
        second_prompts, second_continuations, second = reconstruct_cases()
        np.testing.assert_array_equal(first_prompts, second_prompts)
        np.testing.assert_array_equal(first_continuations, second_continuations)
        self.assertEqual(first, second)
        self.assertEqual(first_prompts.shape, (40, 128))
        self.assertEqual(first_continuations.shape, (40, 128))
        self.assertEqual(first["selected_cases"], 40)
        self.assertGreaterEqual(first["bpe_boundary_eligible_cases"], 40)
        self.assertEqual(first["prompt_array_sha256"], array_sha256(first_prompts))
        self.assertEqual(
            first["continuation_array_sha256"],
            array_sha256(first_continuations),
        )

    def test_schedule_is_deterministic_and_role_balanced(self):
        orders = [
            role_schedule(prompt, repetition)
            for prompt in range(MEASURED_CASES)
            for repetition in range(REPETITIONS)
        ]
        self.assertEqual(len({tuple(order) for order in orders}), 5)
        for order in orders:
            self.assertEqual(set(order), set(RUNTIME_ROLES))
        first_counts = {role: 0 for role in RUNTIME_ROLES}
        last_counts = {role: 0 for role in RUNTIME_ROLES}
        for order in orders:
            first_counts[order[0]] += 1
            last_counts[order[-1]] += 1
        self.assertLessEqual(max(first_counts.values()) - min(first_counts.values()), 1)
        self.assertLessEqual(max(last_counts.values()) - min(last_counts.values()), 1)
        self.assertEqual(len(schedule_sha256()), 64)

    def test_prompt_paired_statistic_does_not_count_repetitions_as_prompts(self):
        reference = np.full((32, 3), 10.0, dtype=np.float64)
        candidate = np.full((32, 3), 8.0, dtype=np.float64)
        result = comparison_summary(candidate, reference)
        self.assertEqual(result["prompt_count"], 32)
        self.assertEqual(result["repetitions_per_prompt"], 3)
        self.assertEqual(result["positive_prompt_count"], 32)
        self.assertAlmostEqual(result["median_latency_reduction"], 0.2)

    def test_candidate_gate_rejects_fast_but_bpe_uncompetitive_graph(self):
        passing = {
            "bootstrap_percentile_95_lower": 0.11,
            "median_latency_reduction": 0.15,
            "positive_prompt_count": 30,
        }
        comparisons = {
            "generic_unicode_scalar_vs_byte_w72": dict(passing),
            "generic_unicode_scalar_vs_byte_bpe_32000": {
                **passing,
                "bootstrap_percentile_95_lower": -0.11,
            },
            "generic_unicode_scalar_vs_byte_bpe_16000": dict(passing),
        }
        decision = _candidate_decision(
            "generic_unicode_scalar",
            comparisons,
            True,
        )
        self.assertFalse(decision["pass"])
        self.assertFalse(
            decision["checks"][
                "bpe32_lower_bound_not_worse_than_minus_10_percent"
            ]
        )

    def test_implementation_manifest_is_unique_and_complete(self):
        self.assertEqual(len(set(IMPLEMENTATION_PATHS)), len(IMPLEMENTATION_PATHS))
        for relative in IMPLEMENTATION_PATHS:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_sealed_plan_validates_when_present(self):
        if not PLAN_PATH.exists():
            self.skipTest("plan is created only after the implementation commit")
        validate_plan(json.loads(PLAN_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
