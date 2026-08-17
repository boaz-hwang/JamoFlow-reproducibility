from __future__ import annotations

import unittest

import torch
from vocabulary_transfer_baseline_core import (
    BASE_VOCABULARY_SIZE,
    BASELINE_ROLES,
    FINAL_PROBE_STEP,
    PROBE_STEPS,
    TARGET_VOCABULARY_SIZE,
    TWO_STAGE_BOUNDARY,
    SourceTokenMetadata,
    baseline_closure_decision,
    initialize_target_weights,
    mask_old_row_gradient,
    restore_old_rows,
    role_definition,
)


def _pieces_and_decompositions():
    base = tuple(bytes((index // 256, index % 256)) for index in range(BASE_VOCABULARY_SIZE))
    target = base + tuple(
        base[index // BASE_VOCABULARY_SIZE] + base[index % BASE_VOCABULARY_SIZE]
        for index in range(TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE)
    )
    rows = tuple(
        (index,)
        if index < BASE_VOCABULARY_SIZE
        else (
            (index - BASE_VOCABULARY_SIZE) // BASE_VOCABULARY_SIZE,
            (index - BASE_VOCABULARY_SIZE) % BASE_VOCABULARY_SIZE,
        )
        for index in range(TARGET_VOCABULARY_SIZE)
    )
    return base, target, rows


def _metadata() -> SourceTokenMetadata:
    lengths = tuple(1 + index % 5 for index in range(BASE_VOCABULARY_SIZE))
    mask = tuple(index < 100 for index in range(BASE_VOCABULARY_SIZE))
    return SourceTokenMetadata(
        decoded_character_lengths=lengths,
        decoded_character_lengths_sha256="1" * 64,
        hangul_token_mask=mask,
        hangul_token_mask_sha256="2" * 64,
        hangul_token_count=sum(mask),
        replacement_character_token_count=0,
        length_definition="test",
        script_definition="test",
    )


class VocabularyTransferBaselineCoreTest(unittest.TestCase):
    def test_role_set_and_stage_boundary_are_frozen(self) -> None:
        self.assertEqual(len(BASELINE_ROLES), 9)
        self.assertEqual(PROBE_STEPS, (0, 32, 50, 128, 307, 512))
        self.assertEqual(FINAL_PROBE_STEP, 512)
        self.assertEqual(TWO_STAGE_BOUNDARY, 307)
        self.assertEqual(
            sum(role_definition(role)["training_schedule"].startswith("new_rows") for role in BASELINE_ROLES),
            2,
        )

    def test_bil_uses_midpoint_median_and_decoded_character_weights(self) -> None:
        base, target, rows = _pieces_and_decompositions()
        metadata = _metadata()
        base_weight = torch.zeros(BASE_VOCABULARY_SIZE, 2, dtype=torch.float32)
        base_weight[:, 0] = torch.arange(1, BASE_VOCABULARY_SIZE + 1, dtype=torch.float32)
        generator = torch.Generator().manual_seed(7)
        random_input = torch.randn(TARGET_VOCABULARY_SIZE, 2, generator=generator)
        random_output = torch.randn(TARGET_VOCABULARY_SIZE, 2, generator=generator)
        input_weight, output_weight, audit = initialize_target_weights(
            "untied_bil_hangul_median_char_out",
            base_weight=base_weight,
            target_input_random_weight=random_input,
            target_output_random_weight=random_output,
            base_pieces=base,
            target_pieces=target,
            decompositions=rows,
            metadata=metadata,
        )
        self.assertEqual(audit.input_target_norm, 50.5)
        first_new = BASE_VOCABULARY_SIZE + 1
        expected = (base_weight[0] * 1.0 + base_weight[1] * 2.0) / 3.0
        self.assertTrue(torch.allclose(output_weight[first_new], expected, rtol=0.0, atol=2e-7))
        self.assertAlmostEqual(float(input_weight[first_new].norm()), 50.5, places=5)
        self.assertTrue(torch.equal(input_weight[:BASE_VOCABULARY_SIZE], base_weight))
        self.assertTrue(torch.equal(output_weight[:BASE_VOCABULARY_SIZE], base_weight))
        self.assertTrue(audit.copied_rows_exact)

    def test_eeve_output_copies_first_constituent_and_staging_does_not_change_init(self) -> None:
        base, target, rows = _pieces_and_decompositions()
        metadata = _metadata()
        generator = torch.Generator().manual_seed(11)
        base_weight = torch.randn(BASE_VOCABULARY_SIZE, 3, generator=generator)
        random_input = torch.randn(TARGET_VOCABULARY_SIZE, 3, generator=generator)
        random_output = torch.randn(TARGET_VOCABULARY_SIZE, 3, generator=generator)
        _, eeve_output, _ = initialize_target_weights(
            "untied_eeve_uniform_in_first_out",
            base_weight=base_weight,
            target_input_random_weight=random_input,
            target_output_random_weight=random_output,
            base_pieces=base,
            target_pieces=target,
            decompositions=rows,
            metadata=metadata,
        )
        self.assertTrue(torch.equal(eeve_output[BASE_VOCABULARY_SIZE + 1], base_weight[0]))
        all_input, all_output, _ = initialize_target_weights(
            "tied_uniform_no_norm_all",
            base_weight=base_weight,
            target_input_random_weight=random_input,
            target_output_random_weight=random_output,
            base_pieces=base,
            target_pieces=target,
            decompositions=rows,
            metadata=metadata,
        )
        staged_input, staged_output, _ = initialize_target_weights(
            "tied_uniform_no_norm_two_stage",
            base_weight=base_weight,
            target_input_random_weight=random_input,
            target_output_random_weight=random_output,
            base_pieces=base,
            target_pieces=target,
            decompositions=rows,
            metadata=metadata,
        )
        self.assertTrue(torch.equal(all_input, staged_input))
        self.assertTrue(torch.equal(all_output, staged_output))
        self.assertEqual(all_input.data_ptr(), all_output.data_ptr())
        self.assertEqual(staged_input.data_ptr(), staged_output.data_ptr())

    def test_stage_one_masks_and_restores_old_rows_under_adamw(self) -> None:
        weight = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(10, 2))
        copied = weight[:4].detach().clone()
        before_new = weight[4:].detach().clone()
        optimizer = torch.optim.AdamW([weight], lr=0.1, weight_decay=0.1)
        weight.grad = torch.ones_like(weight)
        mask_old_row_gradient(weight, base_size=4)
        self.assertTrue(torch.equal(weight.grad[:4], torch.zeros_like(weight.grad[:4])))
        optimizer.step()
        restore_old_rows(weight, copied, base_size=4)
        self.assertTrue(torch.equal(weight[:4].detach(), copied))
        self.assertFalse(torch.equal(weight[4:].detach(), before_new))

    def test_decision_preserves_tied_and_untied_pareto_roles(self) -> None:
        values = {
            "untied_random_hangul_median_input_native_output": 1.56,
            "untied_bil_hangul_median_char_out": 1.46,
            "untied_bil_global_median_char_out": 1.47,
            "untied_bil_hangul_median_uniform_out": 1.48,
            "untied_eeve_uniform_in_first_out": 1.49,
            "tied_random_native_all": 1.58,
            "tied_uniform_no_norm_all": 1.47,
            "tied_random_native_two_stage": 1.57,
            "tied_uniform_no_norm_two_stage": 1.45,
        }
        decision = baseline_closure_decision(values, anchor_bpb=1.43)
        self.assertTrue(decision["korean_stage_authorized"])
        self.assertEqual(decision["best_untied_pareto_role"], "untied_bil_hangul_median_char_out")
        self.assertEqual(decision["best_tied_pareto_role"], "tied_uniform_no_norm_two_stage")
        self.assertFalse(decision["selection_uses_step_zero_or_step_fifty"])


if __name__ == "__main__":
    unittest.main()
