from __future__ import annotations

import ast
import unittest

from run_vocabulary_transfer_baseline import _paths, _stage_one_optimizer
from vocabulary_transfer_baseline_core import (
    BASELINE_ROLES,
    PROBE_STEPS,
    build_target_graph,
    expected_parameter_count,
    role_definition,
    state_mapping_sha256,
)
from vocabulary_transfer_baseline_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    initialization_identities,
    selection_rule,
)


class VocabularyTransferBaselineTest(unittest.TestCase):
    def test_all_roles_have_expected_dense_graph_and_namespace(self) -> None:
        for role in BASELINE_ROLES:
            model = build_target_graph(role)
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                expected_parameter_count(role),
            )
            tied = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
            self.assertEqual(tied, role_definition(role)["tied"])
            report, checkpoints, nlls = _paths(role)
            self.assertEqual(set(checkpoints), set(PROBE_STEPS))
            self.assertEqual(set(nlls), set(PROBE_STEPS))
            self.assertTrue(str(report).endswith(f"workers/{role}.json"))

    def test_stage_one_optimizer_opens_only_the_tied_lexical_matrix(self) -> None:
        model = build_target_graph("tied_uniform_no_norm_two_stage")
        optimizer = _stage_one_optimizer(model)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(trainable), 1)
        self.assertIs(trainable[0], model.model.embed_tokens.weight)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["schedule_kind"], "head")

    def test_real_initialization_identity_binds_exact_metadata(self) -> None:
        audits, states, metadata = initialization_identities()
        self.assertEqual(set(audits), set(BASELINE_ROLES))
        self.assertEqual(set(states), set(BASELINE_ROLES))
        self.assertEqual(metadata["hangul_token_count"], 1_448)
        self.assertEqual(metadata["replacement_character_token_count"], 400)
        self.assertEqual(min(metadata["decoded_character_lengths"]), 1)
        self.assertEqual(max(metadata["decoded_character_lengths"]), 5)
        self.assertEqual(
            states["tied_uniform_no_norm_all"],
            states["tied_uniform_no_norm_two_stage"],
        )
        self.assertEqual(
            states["tied_random_native_all"],
            states["tied_random_native_two_stage"],
        )
        for role, audit in audits.items():
            self.assertTrue(audit["copied_rows_exact"])
            self.assertTrue(audit["target_pieces_reconstruct_exactly"])
            self.assertEqual(audit["shared_token_count"], 2_048)
            self.assertEqual(audit["new_token_count"], 6_144)
            model = build_target_graph(role)
            self.assertIsInstance(state_mapping_sha256(model.state_dict()), str)

    def test_selection_rule_has_no_result_dependent_fallback(self) -> None:
        rule = selection_rule()
        self.assertIsNone(rule["korean_specific_fallback"])
        self.assertFalse(rule["result_dependent_role_addition"])
        self.assertFalse(rule["step_zero_or_step_fifty_can_select"])
        self.assertTrue(rule["requires_both_gates"])
        self.assertTrue(rule["preserve_best_qualified_tied_and_untied_pareto_roles"])

    def test_implementation_manifest_is_unique_and_complete(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        self.assertTrue(all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS))

    def test_plan_sealer_has_no_role_or_threshold_cli(self) -> None:
        source = (ROOT / "scripts/seal_vocabulary_transfer_baseline_plan.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                for call in calls
            )
        )
        self.assertNotIn("--role", source)
        self.assertNotIn("--threshold", source)


if __name__ == "__main__":
    unittest.main()
