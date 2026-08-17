from __future__ import annotations

import unittest
from pathlib import Path

from foldable_multihash_mechanism_core import NEW_ROLES
from foldable_multihash_mechanism_protocol import (
    AUDIT_RESULT_PATH,
    IMPLEMENTATION_PATHS,
    read_json,
    role_definition,
)

ROOT = Path(__file__).resolve().parents[1]


class FoldableMultihashMechanismProtocolTest(unittest.TestCase):
    def test_roles_are_exact_and_deploy_as_dense(self) -> None:
        self.assertEqual(
            NEW_ROLES,
            (
                "update_matched_dense",
                "stratified_generic_shuffle",
                "balanced_random_multihash",
            ),
        )
        for role in NEW_ROLES:
            self.assertEqual(
                role_definition(role)["deployed_graph"],
                "ordinary_untied_dense_bpe_8192",
            )

    def test_implementation_manifest_is_unique_and_complete(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        for path in IMPLEMENTATION_PATHS:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_plan_and_runtime_do_not_consume_test_or_latency(self) -> None:
        for path in (
            "scripts/seal_foldable_multihash_mechanism_plan.py",
            "scripts/run_foldable_multihash_mechanism.py",
            "scripts/summarize_foldable_multihash_mechanism.py",
        ):
            source = (ROOT / path).read_text(encoding="utf-8").lower()
            for forbidden in ("test_nll", "test_bpb", "latency"):
                self.assertNotIn(forbidden, source)

    def test_fixed_update_multipliers_equal_the_sealed_audit(self) -> None:
        audit = read_json(AUDIT_RESULT_PATH)
        selected = audit["evidence"]["selected_control"]
        definition = role_definition("update_matched_dense")
        self.assertEqual(
            definition["post_adamw_new_row_input_multiplier"],
            selected["input_multiplier"],
        )
        self.assertEqual(
            definition["post_adamw_new_row_output_multiplier"],
            selected["output_multiplier"],
        )


if __name__ == "__main__":
    unittest.main()
