from __future__ import annotations

import copy
import unittest

from fresh_vocabulary_adaptation_core import ROLES, role_definition
from fresh_vocabulary_adaptation_protocol import (
    IMPLEMENTATION_PATHS,
    canonical_sha256,
    decision_contract,
    training_contract,
    validate_plan,
)


def _inventories() -> dict[str, object]:
    common_tokens = {
        "raw_stream_bytes": 128_000_000,
        "complete_utf8_bytes": 128_000_000,
        "trailing_incomplete_utf8_bytes": 0,
        "token_count": 1,
        "full_sequence_count": 1,
        "dropped_token_count": 0,
        "predicted_target_raw_bytes": 100,
        "token_ids_sha256": "1" * 64,
        "first_batch_token_count": 16_384,
        "first_batch_sha256": "2" * 64,
    }
    base = {
        "train_tokens": {
            **common_tokens,
            "full_sequence_count": 10,
            "predicted_target_raw_bytes": 1_000,
        },
        "total_optimizer_steps": 1,
    }
    target = {
        "train_tokens": {
            **common_tokens,
            "full_sequence_count": 8,
            "predicted_target_raw_bytes": 999,
        },
        "total_optimizer_steps": 1,
        "inplace_stage": {
            "boundary_rule": "first_complete_effective_batch_reaching_60pct_raw_target_bytes",
            "requested_stage_one_raw_fraction": 0.6,
            "stage_one_optimizer_steps": 1,
            "stage_one_raw_target_bytes": 600,
            "stage_one_realized_raw_fraction": 0.6,
            "stage_two_optimizer_steps": 1,
            "stage_two_raw_target_bytes": 399,
            "total_optimizer_steps": 2,
            "total_raw_target_bytes": 999,
        },
    }
    return {"2048": base, "8192": target}


class FreshVocabularyAdaptationProtocolTest(unittest.TestCase):
    def test_implementation_paths_are_unique(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))

    def test_training_contract_keeps_equal_raw_stream_and_different_steps(self) -> None:
        contract = training_contract(_inventories())
        self.assertEqual(set(contract), set(ROLES))
        for role in ROLES:
            self.assertEqual(contract[role]["raw_stream_bytes"], 128_000_000)
            self.assertEqual(
                contract[role]["ordering"],
                "sealed_rank_order_contiguous_no_permutation",
            )
        self.assertIn("inplace_stage", contract["dense8k_inplace_two_stage"])
        self.assertNotIn("inplace_stage", contract["dense8k_standard_joint"])

    def test_static_plan_validation_rejects_result_dependent_rule_change(self) -> None:
        inventories = _inventories()
        plan = {
            "schema_version": 1,
            "kind": "fresh_vocabulary_adaptation_one_seed_plan_v1",
            "protocol_id": "jamoflow-fresh-vocabulary-adaptation-one-seed-v1",
            "status": "sealed_before_fresh_training",
            "git_commit_before_plan": "a" * 40,
            "dependencies": {},
            "environment": {},
            "implementation_sha256": {},
            "tokenizers": {},
            "roles": {role: role_definition(role) for role in ROLES},
            "initialization": {},
            "inventories": inventories,
            "document_common": {},
            "training": training_contract(inventories),
            "decision": decision_contract(),
            "claim_boundary": {},
            "output_path": "results/fresh-vocabulary-adaptation-one-seed-v1/summary.json",
        }
        plan["plan_sha256"] = canonical_sha256(plan)
        validate_plan(plan, verify_derived=False)
        changed = copy.deepcopy(plan)
        changed["decision"]["quality_noninferiority_margin_bpb"] = 0.02
        unsigned = dict(changed)
        unsigned.pop("plan_sha256")
        changed["plan_sha256"] = canonical_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "plan contract"):
            validate_plan(changed, verify_derived=False)


if __name__ == "__main__":
    unittest.main()
