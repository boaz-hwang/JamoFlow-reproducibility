from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from compositional_head_preflight_protocol import hash_file
from fresh_vocabulary_16k_core import (
    ROLES,
    VOCABULARY_SIZES,
    expected_parameter_count,
    inplace_stage_contract,
    role_definition,
)
from fresh_vocabulary_16k_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    canonical_sha256,
    decision_contract,
    dependency_identity,
    implementation_identity,
    selected_tokenizer_identity,
    training_contract,
    validate_plan,
    verified_fresh_streams,
)


def _inventories() -> dict[str, object]:
    output = {}
    for size in VOCABULARY_SIZES:
        raw = __import__("numpy").full(100, 511, dtype="int64")
        row = {
            "train_tokens": {
                "full_sequence_count": 100,
                "predicted_target_raw_bytes": 51_100,
                "token_ids_sha256": f"{size:064x}",
            },
            "total_optimizer_steps": 4,
        }
        if size == 16_000:
            row["inplace_stage"] = inplace_stage_contract(raw)
        output[str(size)] = row
    return output


def _plan() -> dict[str, object]:
    inventories = _inventories()
    initialization = {
        "eightk_initial_state_bitwise_matches_fresh_v1": True,
        "parameter_count_by_role": {
            role: expected_parameter_count(role_definition(role)["vocabulary_size"])
            for role in ROLES
        },
    }
    plan = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_quality_one_seed_plan_v1",
        "protocol_id": "jamoflow-fresh-vocabulary-16k-quality-one-seed-v1",
        "status": "sealed_before_fresh_v2_training",
        "git_commit_before_plan": "1" * 40,
        "dependencies": {},
        "environment": {},
        "implementation_sha256": {},
        "tokenizers": {},
        "roles": {role: role_definition(role) for role in ROLES},
        "initialization": initialization,
        "inventories": inventories,
        "document_common": {},
        "training": training_contract(inventories),
        "decision": decision_contract(),
        "claim_boundary": {},
        "output_path": "results/fresh-vocabulary-16k-quality-one-seed-v1/summary.json",
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


class FreshVocabulary16KProtocolTests(unittest.TestCase):
    def test_plan_contract_rejects_role_decision_or_hash_rotation(self) -> None:
        plan = _plan()
        validate_plan(plan, verify_derived=False)
        for label in ("role", "decision", "hash"):
            changed = copy.deepcopy(plan)
            if label == "role":
                changed["roles"][ROLES[-1]]["vocabulary_size"] = 8_192
            elif label == "decision":
                changed["decision"]["quality_noninferiority_margin_bpb"] = 0.02
            else:
                changed["plan_sha256"] = "0" * 64
            if label != "hash":
                unsigned = dict(changed)
                unsigned.pop("plan_sha256")
                changed["plan_sha256"] = canonical_sha256(unsigned)
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_plan(changed, verify_derived=False)

    def test_training_contract_uses_exact_vocabulary_microbatches(self) -> None:
        contract = training_contract(_inventories())
        self.assertEqual(contract["dense2k_joint_v2"]["train_microbatch_size"], 32)
        self.assertEqual(
            contract["dense8k_update_geometry_v2"]["train_microbatch_size"], 8
        )
        for role in (
            "dense16k_standard_joint",
            "dense16k_inplace_two_stage",
            "dense16k_update_geometry",
        ):
            self.assertEqual(contract[role]["train_microbatch_size"], 4)
            self.assertEqual(contract[role]["evaluation_batch_size"], 8)
        self.assertIn("inplace_stage", contract["dense16k_inplace_two_stage"])
        self.assertNotIn("inplace_stage", contract["dense16k_standard_joint"])

    def test_decision_has_no_fallback_candidate(self) -> None:
        contract = decision_contract()
        self.assertEqual(
            contract["actual_candidate_is_fixed_before_training"],
            "dense16k_update_geometry",
        )
        self.assertTrue(contract["no_result_dependent_fallback_role"])
        self.assertEqual(
            set(contract["actual_requires_candidate_noninferior_to_every_anchor"]),
            {"dense2k_joint_v2", "dense8k_update_geometry_v2"},
        )
        self.assertEqual(len(contract["actual_requires_candidate_beats_every_16k_control"]), 2)

    def test_fresh_v2_streams_match_the_seal(self) -> None:
        streams = verified_fresh_streams()
        self.assertEqual(len(streams["train"].data), 128_000_000)
        self.assertEqual(len(streams["calibration"].data), 8_000_000)
        self.assertEqual(streams["train"].selected_records, 5_637)
        self.assertEqual(streams["calibration"].selected_records, 357)

    def test_dependencies_and_selected_tokenizers_are_current(self) -> None:
        dependencies = dependency_identity()
        self.assertEqual(
            dependencies["fresh_v2_seal"]["sha256"],
            "c7ceeb3290db5e1d0b905494d15b54874f22f53da3281f155a6e2e11437bbe9e",
        )
        for row in dependencies.values():
            self.assertEqual(hash_file(ROOT / row["path"]), row["sha256"])
        tokenizers = selected_tokenizer_identity()
        self.assertEqual(set(tokenizers), {"2048", "8192", "16000"})
        self.assertEqual(
            {row["vocabulary_size"] for row in tokenizers.values()},
            {2_048, 8_192, 16_000},
        )

    def test_implementation_manifest_is_unique_and_complete(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        identity = implementation_identity()
        self.assertEqual(tuple(identity), IMPLEMENTATION_PATHS)
        for path, expected in identity.items():
            self.assertTrue((ROOT / path).is_file(), path)
            self.assertEqual(hash_file(ROOT / path), expected)

    def test_protocol_does_not_name_sealed_final_test_or_result_metric(self) -> None:
        paths = [
            ROOT / "scripts/fresh_vocabulary_16k_core.py",
            ROOT / "scripts/fresh_vocabulary_16k_protocol.py",
            ROOT / "scripts/seal_fresh_vocabulary_16k_plan.py",
        ]
        forbidden = ("hplt3-korean-final-test-v1/ko.jsonl", "test_bpb", "free_e2e")
        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertFalse(any(value in source for value in forbidden))


if __name__ == "__main__":
    unittest.main()
