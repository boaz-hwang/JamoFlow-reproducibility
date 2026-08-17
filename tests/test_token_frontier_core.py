from __future__ import annotations

import gc
import unittest

from scalar_runtime_core import model_parameter_count

from token_frontier_core import (
    DEPTHS,
    FRONTIER_SPECS,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    RUNTIME_ROLES,
    VOCABULARY_SIZES,
    analytical_parameters,
    balanced_role_schedule,
    build_frontier_model,
    parse_role,
    role_name,
)


class TokenFrontierCoreTest(unittest.TestCase):
    def test_frontier_specs_cover_grid_and_match_parameters(self) -> None:
        self.assertEqual(len(FRONTIER_SPECS), len(VOCABULARY_SIZES) * len(DEPTHS))
        self.assertEqual(len(FRONTIER_SPECS), 18)
        for role, spec in FRONTIER_SPECS.items():
            self.assertEqual(parse_role(role), (spec.vocabulary_size, spec.layers))
            self.assertEqual(role_name(spec.vocabulary_size, spec.layers), role)
            self.assertEqual(
                spec.expected_parameters,
                analytical_parameters(
                    spec.vocabulary_size,
                    spec.hidden_size,
                    spec.intermediate_size,
                    spec.layers,
                ),
            )
            self.assertLessEqual(
                abs(spec.expected_parameters / PARAMETER_TARGET - 1),
                PARAMETER_RELATIVE_TOLERANCE,
            )
            self.assertEqual(spec.hidden_size % spec.attention_heads, 0)

    def test_schedule_is_a_permutation_and_balanced_over_108_trials(self) -> None:
        position_counts = {role: [0] * len(RUNTIME_ROLES) for role in RUNTIME_ROLES}
        for case_index in range(36):
            for repetition in range(3):
                schedule = balanced_role_schedule(case_index, repetition)
                self.assertEqual(set(schedule), set(RUNTIME_ROLES))
                self.assertEqual(len(schedule), len(set(schedule)))
                for position, role in enumerate(schedule):
                    position_counts[role][position] += 1
        for counts in position_counts.values():
            self.assertEqual(counts, [6] * len(RUNTIME_ROLES))

    def test_analytical_counts_equal_instantiated_tied_llama_graphs(self) -> None:
        for role in RUNTIME_ROLES:
            model = build_frontier_model(role, seed=20_260_814)
            self.assertEqual(
                model_parameter_count(model), FRONTIER_SPECS[role].expected_parameters
            )
            del model
            gc.collect()
