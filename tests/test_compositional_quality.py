from __future__ import annotations

from pathlib import Path
import unittest

from compositional_quality_core import (
    QUALITY_ROLES,
    QUALITY_SPECS,
    build_quality_model,
    deterministic_order,
    quality_decision,
    state_subset_sha256,
    training_contract,
)
from compositional_quality_protocol import (
    IMPLEMENTATION_PATHS,
    ROOT,
    resource_contract,
    selection_rule,
)
from compositional_head_preflight_protocol import load_tokenizers


def _contrast(value: float, upper: float | None = None) -> dict[str, float]:
    return {
        "contiguous_bpb_difference": value,
        "document_bpb_difference": value,
        "bootstrap_95_upper": value if upper is None else upper,
    }


class CompositionalQualityTests(unittest.TestCase):
    def test_role_grid_has_exact_parameter_controls(self) -> None:
        self.assertEqual(len(QUALITY_ROLES), 6)
        self.assertEqual(QUALITY_SPECS["dense_v2048"].expected_parameters, 19_667_328)
        self.assertEqual(QUALITY_SPECS["dense_v8192"].expected_parameters, 22_026_624)
        self.assertEqual(QUALITY_SPECS["low_rank_v8192"].expected_parameters, 19_669_888)
        for role in (
            "generic_code_v8192",
            "shuffled_hangul_code_v8192",
            "hangul_code_v8192",
        ):
            self.assertEqual(QUALITY_SPECS[role].expected_parameters, 19_667_328)

    def test_shuffled_and_hangul_models_share_body_and_initial_code_rows(self) -> None:
        import torch

        table = load_tokenizers()[8_192][1]
        hangul = build_quality_model("hangul_code_v8192", token_bytes=table, seed=31)
        shuffled = build_quality_model(
            "shuffled_hangul_code_v8192", token_bytes=table, seed=31
        )
        self.assertEqual(
            state_subset_sha256(hangul, transformer_body_only=True),
            state_subset_sha256(shuffled, transformer_body_only=True),
        )
        self.assertTrue(
            torch.equal(
                hangul.factorized_vocabulary.weight,
                shuffled.factorized_vocabulary.weight,
            )
        )
        self.assertFalse(
            torch.equal(
                hangul.factorized_vocabulary.code_indices,
                shuffled.factorized_vocabulary.code_indices,
            )
        )
        output = shuffled(input_ids=torch.tensor([[0, 1]]), use_cache=False)
        self.assertEqual(tuple(output.logits.shape), (1, 2, 8_192))

    def test_training_order_and_contract_are_deterministic(self) -> None:
        first = deterministic_order(101)
        second = deterministic_order(101)
        self.assertTrue((first == second).all())
        contract = training_contract("hangul_code_v8192", 53_590)
        self.assertEqual(contract["train_microbatch_size"], 8)
        self.assertEqual(contract["gradient_accumulation_steps"], 4)
        self.assertEqual(contract["total_optimizer_steps"], 1_675)

    def test_quality_decision_requires_all_primary_contrasts(self) -> None:
        passing = {
            "hangul_vs_dense_2k": _contrast(0.005, upper=0.009),
            "hangul_vs_generic": _contrast(-0.003, upper=-0.0001),
            "hangul_vs_shuffled": _contrast(-0.004, upper=-0.0002),
            "hangul_vs_low_rank": _contrast(0.001, upper=0.0015),
            "generic_vs_dense_2k": _contrast(0.006, upper=0.009),
        }
        decision = quality_decision(passing)
        self.assertTrue(decision["overall_pass"])
        self.assertTrue(decision["trained_actual_inference_authorized"])
        for failed_key in (
            "hangul_vs_dense_2k",
            "hangul_vs_generic",
            "hangul_vs_shuffled",
            "hangul_vs_low_rank",
        ):
            changed = {key: dict(value) for key, value in passing.items()}
            changed[failed_key]["bootstrap_95_upper"] = 0.02
            self.assertFalse(quality_decision(changed)["overall_pass"])

    def test_generic_only_result_is_not_candidate_fallback(self) -> None:
        values = {
            "hangul_vs_dense_2k": _contrast(0.02),
            "hangul_vs_generic": _contrast(0.003),
            "hangul_vs_shuffled": _contrast(0.003),
            "hangul_vs_low_rank": _contrast(0.003),
            "generic_vs_dense_2k": _contrast(0.005, upper=0.009),
        }
        decision = quality_decision(values)
        self.assertEqual(
            decision["status"],
            "generic_factorization_only_requires_novelty_reassessment",
        )
        self.assertFalse(decision["trained_actual_inference_authorized"])
        self.assertTrue(decision["no_result_dependent_candidate_fallback"])

    def test_protocol_manifest_and_gates_are_exact(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        self.assertEqual(
            [path for path in IMPLEMENTATION_PATHS if not (ROOT / path).is_file()], []
        )
        self.assertEqual(resource_contract()["safety_factor"], 1.25)
        self.assertEqual(
            selection_rule()["minimum_korean_advantage_bpb"], 0.002
        )
        self.assertIsNone(selection_rule()["candidate_fallback"])


if __name__ == "__main__":
    unittest.main()
