import math
import unittest

import numpy as np

from scripts.conditional_local_sensitivity_core import (
    CANDIDATE_ORDER,
    PAIR_ORDER,
    PREOUTCOME_ROUTE_GEOMETRY,
    ROUTE_ORDER,
    candidate_definition,
    summarize_frozen_sensitivity,
)


def _losses(effect_bpb, count=96):
    baseline = np.full(count, 600.0, dtype=np.float32)
    delta = np.float32(effect_bpb * 511 * math.log(2.0))
    return baseline, baseline + delta


class ConditionalLocalSensitivityTests(unittest.TestCase):
    def test_preoutcome_route_geometry_records_near_matched_overlap(self):
        geometry = PREOUTCOME_ROUTE_GEOMETRY
        self.assertEqual(geometry["total_positions"], 8_000_000)
        self.assertTrue(geometry["hangul_is_subset_of_utf8_incomplete"])
        generic = geometry["utf8_incomplete_easy_positions"]
        hangul = geometry["hangul_prefix_easy_positions"]
        self.assertAlmostEqual(generic / geometry["total_positions"], 0.583054875)
        self.assertAlmostEqual(hangul / geometry["total_positions"], 0.575361125)
        self.assertAlmostEqual(hangul / generic, 0.9868044152790936)

    def test_factorial_contract_is_complete_and_ordered_by_expected_savings(self):
        self.assertEqual(len(CANDIDATE_ORDER), 8)
        self.assertEqual(
            CANDIDATE_ORDER[:2],
            tuple(f"{route}__{PAIR_ORDER[0]}" for route in ROUTE_ORDER),
        )
        definitions = [candidate_definition(name) for name in CANDIDATE_ORDER]
        self.assertEqual(
            {(row["route_policy"], row["components"], row["operator"]) for row in definitions},
            {
                (route, components, operator)
                for route in ROUTE_ORDER
                for components in ("decoder", "encoder_decoder")
                for operator in ("second_mlp", "second_layer_kv")
            },
        )

    def test_selection_requires_both_routes_and_uses_first_passing_pair(self):
        baseline, passing = _losses(0.01)
        _, failing = _losses(0.03)
        candidates = {name: passing.copy() for name in CANDIDATE_ORDER}
        documents = np.repeat(np.arange(24, dtype=np.int32), 4)
        summary = summarize_frozen_sensitivity(
            candidate_losses_nats=candidates,
            baseline_losses_nats=baseline,
            document_indices=documents,
            route_rates={"utf8_incomplete": 0.6, "hangul_prefix": 0.5},
            eligible_sequence_fraction=1.0,
        )
        self.assertEqual(summary["selection"]["selected_pair"], PAIR_ORDER[0])
        self.assertTrue(summary["selection"]["actual_runtime_prototype_authorized"])

        candidates[f"hangul_prefix__{PAIR_ORDER[0]}"] = failing
        summary = summarize_frozen_sensitivity(
            candidate_losses_nats=candidates,
            baseline_losses_nats=baseline,
            document_indices=documents,
            route_rates={"utf8_incomplete": 0.6, "hangul_prefix": 0.5},
            eligible_sequence_fraction=1.0,
        )
        self.assertEqual(summary["selection"]["selected_pair"], PAIR_ORDER[1])

    def test_no_pair_pass_stops_without_route_or_margin_fallback(self):
        baseline, failing = _losses(0.03)
        summary = summarize_frozen_sensitivity(
            candidate_losses_nats={name: failing.copy() for name in CANDIDATE_ORDER},
            baseline_losses_nats=baseline,
            document_indices=np.repeat(np.arange(24, dtype=np.int32), 4),
            route_rates={"utf8_incomplete": 0.6, "hangul_prefix": 0.5},
            eligible_sequence_fraction=1.0,
        )
        self.assertIsNone(summary["selection"]["selected_pair"])
        self.assertFalse(summary["selection"]["actual_runtime_prototype_authorized"])
        self.assertEqual(
            summary["selection"]["status"],
            "conditional_branch_not_advanced_by_frozen_screen",
        )
        self.assertFalse(
            summary["interpretation"]["trained_conditional_model_falsified_on_failure"]
        )

    def test_missing_candidate_and_low_route_rate_fail_closed(self):
        baseline, passing = _losses(0.01)
        candidates = {name: passing.copy() for name in CANDIDATE_ORDER}
        candidates.pop(CANDIDATE_ORDER[-1])
        with self.assertRaisesRegex(ValueError, "candidate NLL keys"):
            summarize_frozen_sensitivity(
                candidate_losses_nats=candidates,
                baseline_losses_nats=baseline,
                document_indices=np.repeat(np.arange(24, dtype=np.int32), 4),
                route_rates={"utf8_incomplete": 0.6, "hangul_prefix": 0.5},
                eligible_sequence_fraction=1.0,
            )

        candidates[CANDIDATE_ORDER[-1]] = passing.copy()
        summary = summarize_frozen_sensitivity(
            candidate_losses_nats=candidates,
            baseline_losses_nats=baseline,
            document_indices=np.repeat(np.arange(24, dtype=np.int32), 4),
            route_rates={"utf8_incomplete": 0.6, "hangul_prefix": 0.2},
            eligible_sequence_fraction=1.0,
        )
        self.assertIsNone(summary["selection"]["selected_pair"])


if __name__ == "__main__":
    unittest.main()
