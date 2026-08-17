from __future__ import annotations

import unittest

import numpy as np
import torch

from foldable_jamo_residual_core import (
    ASSIGNMENT_KINDS,
    BASELINE_ROLE_BY_ARCHITECTURE,
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    RESIDUAL_ROLES,
    RESIDUAL_SLOT_COUNT,
    FoldableVocabularyResidual,
    audit_residual_assignment,
    build_folded_dense_model,
    build_residual_assignment,
    expected_parameter_counts,
    install_foldable_residual,
    residual_decision,
    role_definition,
)
from vocabulary_transfer_baseline_core import build_target_graph
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    TARGET_VOCABULARY_SIZE,
)


def _pieces() -> tuple[bytes, ...]:
    return tuple(("가" + str(index)).encode("utf-8") for index in range(TARGET_VOCABULARY_SIZE))


def _exposures() -> np.ndarray:
    return (np.arange(TARGET_VOCABULARY_SIZE, dtype=np.int64) % 257).copy()


class FoldableJamoResidualCoreTest(unittest.TestCase):
    def test_role_and_checkpoint_contract_is_fixed(self) -> None:
        self.assertEqual(len(RESIDUAL_ROLES), 6)
        self.assertEqual(PROBE_STEPS, (0, 32, 128, 512))
        self.assertEqual(FINAL_PROBE_STEP, 512)
        self.assertEqual(set(ASSIGNMENT_KINDS), {"generic_surface", "shuffled_jamo", "jamo"})
        self.assertEqual(
            {role_definition(role)["base_initializer_role"] for role in RESIDUAL_ROLES},
            set(BASELINE_ROLE_BY_ARCHITECTURE.values()),
        )

    def test_shuffled_assignment_preserves_matched_slots_and_breaks_alignment(self) -> None:
        pieces = _pieces()
        exposures = _exposures()
        true = build_residual_assignment(pieces, exposures, kind="jamo")
        shuffled = build_residual_assignment(pieces, exposures, kind="shuffled_jamo")
        generic = build_residual_assignment(pieces, exposures, kind="generic_surface")
        self.assertEqual(true.shape, (TARGET_VOCABULARY_SIZE, RESIDUAL_SLOT_COUNT))
        self.assertTrue(np.array_equal(true[:, :6], shuffled[:, :6]))
        self.assertTrue(np.array_equal(true[:, 12], shuffled[:, 12]))
        changed = np.any(
            true[BASE_VOCABULARY_SIZE:] != shuffled[BASE_VOCABULARY_SIZE:], axis=1
        )
        self.assertGreater(float(changed.mean()), 0.50)
        for slot in range(6, 12):
            self.assertTrue(
                np.array_equal(
                    np.sort(true[BASE_VOCABULARY_SIZE:, slot]),
                    np.sort(shuffled[BASE_VOCABULARY_SIZE:, slot]),
                )
            )
            true_weighted = np.bincount(
                true[BASE_VOCABULARY_SIZE:, slot],
                weights=exposures[BASE_VOCABULARY_SIZE:],
                minlength=128,
            )
            shuffled_weighted = np.bincount(
                shuffled[BASE_VOCABULARY_SIZE:, slot],
                weights=exposures[BASE_VOCABULARY_SIZE:],
                minlength=128,
            )
            self.assertTrue(np.array_equal(true_weighted, shuffled_weighted))
        self.assertFalse(np.array_equal(generic[:, 6:12], true[:, 6:12]))
        audit = audit_residual_assignment(pieces, exposures, kind="shuffled_jamo")
        self.assertGreater(audit.stratum_count or 0, 0)
        self.assertGreater(audit.changed_new_row_fraction_vs_true_jamo, 0.50)

    def test_zero_residual_is_exact_and_old_rows_are_permanently_masked(self) -> None:
        generator = torch.Generator().manual_seed(7)
        input_weight = torch.randn(TARGET_VOCABULARY_SIZE, 4, generator=generator)
        output_weight = torch.randn(TARGET_VOCABULARY_SIZE, 4, generator=generator)
        assignment = build_residual_assignment(_pieces(), _exposures(), kind="jamo")
        module = FoldableVocabularyResidual(
            input_weight, output_weight, assignment, tied=False
        )
        self.assertTrue(module.residuals_are_exact_zero())
        self.assertTrue(torch.equal(module.effective_input_weight(), input_weight))
        self.assertTrue(torch.equal(module.effective_output_weight(), output_weight))
        with torch.no_grad():
            module.input_residual.fill_(1.0)
            assert module.output_residual is not None
            module.output_residual.fill_(2.0)
        self.assertTrue(
            torch.equal(
                module.effective_input_weight()[:BASE_VOCABULARY_SIZE],
                input_weight[:BASE_VOCABULARY_SIZE],
            )
        )
        self.assertFalse(
            torch.equal(
                module.effective_input_weight()[BASE_VOCABULARY_SIZE:],
                input_weight[BASE_VOCABULARY_SIZE:],
            )
        )

    def test_materialized_model_is_an_ordinary_dense_graph(self) -> None:
        role = "tied_jamo"
        dense = build_target_graph(role_definition(role)["base_initializer_role"])
        initial = dense.model.embed_tokens.weight.detach().clone()
        assignment = build_residual_assignment(_pieces(), _exposures(), kind="jamo")
        training_model = install_foldable_residual(dense, assignment, tied=True)
        self.assertTrue(training_model.foldable_residual.residuals_are_exact_zero())
        with torch.no_grad():
            training_model.foldable_residual.input_residual.normal_(mean=0.0, std=0.01)
        deployed = build_folded_dense_model(training_model, role)
        self.assertFalse(hasattr(deployed, "foldable_residual"))
        self.assertEqual(
            sum(parameter.numel() for parameter in deployed.parameters()),
            expected_parameter_counts(role)["deployed"],
        )
        self.assertTrue(
            torch.equal(
                deployed.model.embed_tokens.weight[:BASE_VOCABULARY_SIZE],
                initial[:BASE_VOCABULARY_SIZE],
            )
        )
        ids = torch.tensor([[1, 2048, 4096, 8191]], dtype=torch.long)
        training_model.eval()
        deployed.eval()
        with torch.inference_mode():
            left = training_model(input_ids=ids, use_cache=False).logits
            right = deployed(input_ids=ids, use_cache=False).logits
        self.assertTrue(torch.equal(left, right))

    def test_decision_requires_true_jamo_to_beat_all_controls(self) -> None:
        raw = np.asarray([100, 100, 100, 100], dtype=np.int64)
        roles = {
            *RESIDUAL_ROLES,
            "untied_base",
            "tied_base",
        }
        contiguous = {role: 1.47 for role in roles}
        document = {role: np.full(4, 100.0, dtype=np.float64) for role in roles}
        for architecture in ("untied", "tied"):
            candidate = f"{architecture}_jamo"
            contiguous[candidate] = 1.44
            document[candidate] = np.full(4, 98.0, dtype=np.float64)
        decision = residual_decision(contiguous, document, raw, anchor_bpb=1.43)
        self.assertEqual(
            decision["qualified_jamo_roles"], ["untied_jamo", "tied_jamo"]
        )
        document["tied_shuffled_jamo"] = np.full(4, 97.0, dtype=np.float64)
        failed = residual_decision(contiguous, document, raw, anchor_bpb=1.43)
        self.assertEqual(failed["qualified_jamo_roles"], ["untied_jamo"])


if __name__ == "__main__":
    unittest.main()
