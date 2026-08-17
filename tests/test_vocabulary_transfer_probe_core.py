from __future__ import annotations

import unittest

import torch
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    TARGET_VOCABULARY_SIZE,
    TRANSFER_ROLES,
    initialize_target_weights,
    transfer_probe_decision,
)


class VocabularyTransferProbeCoreTest(unittest.TestCase):
    def test_full_initialization_copies_shared_and_reconstructs_all(self) -> None:
        base = tuple(bytes((index // 256, index % 256)) for index in range(BASE_VOCABULARY_SIZE))
        target = base + tuple(
            base[index // BASE_VOCABULARY_SIZE]
            + base[index % BASE_VOCABULARY_SIZE]
            for index in range(TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE)
        )
        table = tuple(
            (index,)
            if index < BASE_VOCABULARY_SIZE
            else (
                (index - BASE_VOCABULARY_SIZE) // BASE_VOCABULARY_SIZE,
                (index - BASE_VOCABULARY_SIZE) % BASE_VOCABULARY_SIZE,
            )
            for index in range(TARGET_VOCABULARY_SIZE)
        )
        self.assertEqual(len(table), TARGET_VOCABULARY_SIZE)
        self.assertTrue(all(b"".join(base[token] for token in row) == piece for row, piece in zip(table, target)))
        generator = torch.Generator().manual_seed(7)
        base_weight = torch.randn(BASE_VOCABULARY_SIZE, 4, generator=generator)
        target_input_random = torch.randn(TARGET_VOCABULARY_SIZE, 4, generator=generator)
        target_output_random = torch.randn(TARGET_VOCABULARY_SIZE, 4, generator=generator)
        for role in TRANSFER_ROLES:
            input_weight, output_weight, audit = initialize_target_weights(
                role,
                base_weight=base_weight,
                target_input_random_weight=target_input_random,
                target_output_random_weight=target_output_random,
                base_pieces=base,
                target_pieces=target,
                decompositions=table,
            )
            self.assertTrue(torch.equal(input_weight[:BASE_VOCABULARY_SIZE], base_weight))
            self.assertTrue(torch.equal(output_weight[:BASE_VOCABULARY_SIZE], base_weight))
            self.assertEqual(audit.shared_token_count, BASE_VOCABULARY_SIZE)
            self.assertEqual(audit.new_token_count, TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE)
            self.assertTrue(audit.exact_reconstruction)
            self.assertEqual(audit.tied_input_output, role.startswith("tied_"))
            self.assertEqual(
                input_weight.data_ptr() == output_weight.data_ptr(),
                role.startswith("tied_"),
            )

    def test_probe_decision_is_fail_closed(self) -> None:
        passing = {
            "tied_random_norm": 1.52,
            "tied_uniform_norm": 1.48,
            "tied_byte_weighted_norm": 1.49,
            "tied_last_subpiece": 1.50,
            "untied_random_norm": 1.51,
            "untied_uniform_in_uniform_out": 1.47,
            "untied_uniform_in_byte_weighted_out": 1.46,
        }
        decision = transfer_probe_decision(passing, anchor_bpb=1.43)
        self.assertTrue(decision["full_cpt_authorized"])
        self.assertEqual(
            decision["selected_composed_initializer"],
            "untied_uniform_in_byte_weighted_out",
        )
        failing = {
            "tied_random_norm": 1.50,
            "tied_uniform_norm": 1.495,
            "tied_byte_weighted_norm": 1.496,
            "tied_last_subpiece": 1.497,
            "untied_random_norm": 1.50,
            "untied_uniform_in_uniform_out": 1.495,
            "untied_uniform_in_byte_weighted_out": 1.496,
        }
        decision = transfer_probe_decision(failing, anchor_bpb=1.43)
        self.assertFalse(decision["full_cpt_authorized"])
        self.assertIsNone(decision["selected_composed_initializer"])


if __name__ == "__main__":
    unittest.main()
