from __future__ import annotations

import ast
import unittest

from compositional_quality_core import state_subset_sha256
from run_vocabulary_transfer_probe import _optimizer, _paths
from vocabulary_transfer_probe_core import (
    PROBE_STEPS,
    TRANSFER_ROLES,
    build_target_graph,
    expected_parameter_count,
    state_mapping_sha256,
)
from vocabulary_transfer_probe_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    base_checkpoint_state,
    initialization_identities,
    parent_anchor,
    selection_rule,
)


class VocabularyTransferProbeTest(unittest.TestCase):
    def test_all_roles_share_the_exact_dense_8k_runtime_graph(self) -> None:
        for role in TRANSFER_ROLES:
            model = build_target_graph(role)
            optimizer = _optimizer(model)
            self.assertEqual(
                {group["schedule_kind"] for group in optimizer.param_groups},
                {"body", "head"},
            )
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                expected_parameter_count(role),
            )
            self.assertEqual(
                state_mapping_sha256(model.state_dict()),
                state_subset_sha256(model, transformer_body_only=False),
            )
            tied = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
            self.assertEqual(tied, role.startswith("tied_"))

    def test_namespace_is_exact_and_korean_free(self) -> None:
        self.assertEqual(len(TRANSFER_ROLES), 7)
        self.assertFalse(any("hangul" in role or "jamo" in role for role in TRANSFER_ROLES))
        for role in TRANSFER_ROLES:
            report, checkpoints, nlls = _paths(role)
            self.assertEqual(set(checkpoints), set(PROBE_STEPS))
            self.assertEqual(set(nlls), set(PROBE_STEPS))
            self.assertTrue(str(report).endswith(f"workers/{role}.json"))

    def test_real_initialization_identities_bind_source_and_both_heads(self) -> None:
        self.assertEqual(
            state_mapping_sha256(base_checkpoint_state()),
            parent_anchor()["checkpoint_state_sha256"],
        )
        audits, states = initialization_identities()
        self.assertEqual(set(audits), set(TRANSFER_ROLES))
        self.assertEqual(set(states), set(TRANSFER_ROLES))
        self.assertEqual(len(set(states.values())), len(TRANSFER_ROLES))
        self.assertEqual(
            len({row["decomposition_sha256"] for row in audits.values()}),
            1,
        )
        for role, row in audits.items():
            self.assertEqual(row["shared_token_count"], 2_048)
            self.assertEqual(row["new_token_count"], 6_144)
            self.assertEqual(row["maximum_constituent_count"], 8)
            self.assertEqual(row["mean_constituent_count"], 1.884033203125)
            self.assertEqual(
                row["decomposition_kind"],
                "canonical_target_merge_tree_cut_at_source_vocab",
            )
            self.assertEqual(row["tied_input_output"], role.startswith("tied_"))

    def test_selection_rule_has_no_fallback(self) -> None:
        rule = selection_rule()
        self.assertIsNone(rule["korean_specific_fallback"])
        self.assertTrue(rule["requires_both_gates"])
        self.assertTrue(
            set(rule["random_control_by_candidate"].values()).isdisjoint(
                rule["candidate_pool"]
            )
        )

    def test_implementation_manifest_is_unique_and_complete(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        self.assertTrue(all((ROOT / path).is_file() for path in IMPLEMENTATION_PATHS))

    def test_plan_sealer_has_no_metric_cli_or_result_dependent_role(self) -> None:
        source = (ROOT / "scripts/seal_vocabulary_transfer_probe_plan.py").read_text(
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
        self.assertNotIn("hangul", source.lower())
        self.assertNotIn("jamo", source.lower())


if __name__ == "__main__":
    unittest.main()
