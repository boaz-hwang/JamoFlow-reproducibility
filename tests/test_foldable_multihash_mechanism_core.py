from __future__ import annotations

import unittest
from collections import Counter

import numpy as np
import torch
from foldable_jamo_residual_core import CODEBOOK_SIZE, RESIDUAL_SLOT_COUNT
from foldable_multihash_mechanism_core import (
    BALANCED_RANDOM_SEED,
    INPUT_UPDATE_MULTIPLIER,
    balanced_random_assignment,
    mechanism_decision,
    scale_new_row_update_,
    stratified_generic_shuffle,
)
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    TARGET_VOCABULARY_SIZE,
)


def _generic_assignment() -> np.ndarray:
    rows = np.arange(TARGET_VOCABULARY_SIZE, dtype=np.int64)[:, None]
    slots = np.arange(RESIDUAL_SLOT_COUNT, dtype=np.int64)[None, :]
    return ((rows * 17 + slots * 31) % CODEBOOK_SIZE).astype(np.int64)


class FoldableMultihashMechanismCoreTest(unittest.TestCase):
    def test_stratified_shuffle_preserves_each_stratum_code_multiset(self) -> None:
        generic = _generic_assignment()
        token_bytes = tuple(
            b"a" * (1 + row % 3) for row in range(TARGET_VOCABULARY_SIZE)
        )
        exposure = (np.arange(TARGET_VOCABULARY_SIZE) % 4).astype(np.int64)
        shuffled, audit = stratified_generic_shuffle(
            generic, token_bytes, exposure
        )
        new_rows = np.arange(BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
        lengths = np.asarray([len(value) for value in token_bytes])
        for length in np.unique(lengths[new_rows]):
            at_length = new_rows[lengths[new_rows] == length]
            for count in np.unique(exposure[at_length]):
                rows = at_length[exposure[at_length] == count]
                self.assertEqual(
                    Counter(map(tuple, generic[rows].tolist())),
                    Counter(map(tuple, shuffled[rows].tolist())),
                )
        self.assertEqual(shuffled.shape, generic.shape)
        self.assertTrue(
            np.array_equal(
                shuffled[:BASE_VOCABULARY_SIZE],
                generic[:BASE_VOCABULARY_SIZE],
            )
        )
        self.assertGreater(audit["non_singleton_row_count"], 0)

    def test_balanced_random_uses_each_bucket_exactly_48_times(self) -> None:
        generic = _generic_assignment()
        balanced, audit = balanced_random_assignment(generic)
        for slot in range(RESIDUAL_SLOT_COUNT):
            counts = np.bincount(
                balanced[BASE_VOCABULARY_SIZE:, slot], minlength=CODEBOOK_SIZE
            )
            self.assertTrue(np.all(counts == 48))
        self.assertEqual(audit["occupancy_per_bucket"], 48)
        self.assertEqual(audit["seed"], BALANCED_RANDOM_SEED)

    def test_post_adamw_scaling_changes_only_new_rows(self) -> None:
        weight = torch.arange(
            TARGET_VOCABULARY_SIZE * 2, dtype=torch.float32
        ).reshape(TARGET_VOCABULARY_SIZE, 2)
        old = weight[:BASE_VOCABULARY_SIZE].clone()
        before = weight[BASE_VOCABULARY_SIZE:].clone()
        weight.add_(2.0)
        scale_new_row_update_(weight, before, INPUT_UPDATE_MULTIPLIER)
        self.assertTrue(torch.equal(weight[:BASE_VOCABULARY_SIZE], old + 2.0))
        expected = before + 2.0 * INPUT_UPDATE_MULTIPLIER
        self.assertTrue(torch.allclose(weight[BASE_VOCABULARY_SIZE:], expected))

    def test_primary_gate_has_no_random_fallback(self) -> None:
        roles = (
            "untied_base",
            "untied_generic_surface",
            "update_matched_dense",
            "stratified_generic_shuffle",
            "balanced_random_multihash",
        )
        raw = np.asarray([100, 100, 100, 100], dtype=np.int64)
        # Generic loses to the update control while both random controls are much better.
        per_document_bpb = {
            "untied_base": 1.10,
            "untied_generic_surface": 1.08,
            "update_matched_dense": 1.07,
            "stratified_generic_shuffle": 1.00,
            "balanced_random_multihash": 0.99,
        }
        nll = {
            role: np.full(4, value * 100 * np.log(2.0), dtype=np.float64)
            for role, value in per_document_bpb.items()
        }
        decision = mechanism_decision(
            {role: per_document_bpb[role] for role in roles},
            nll,
            raw,
            anchor_bpb=1.0,
        )
        self.assertFalse(decision["fresh_korean_multiseed_stage_authorized"])
        self.assertIsNone(decision["threshold_or_role_fallback"])
        self.assertEqual(
            decision["status"],
            "generic_surface_stopped_random_opportunity_requires_new_protocol",
        )

    def test_primary_and_surface_gates_pass_only_with_predeclared_margins(self) -> None:
        raw = np.asarray([100, 100, 100, 100], dtype=np.int64)
        per_document_bpb = {
            "untied_base": 1.10,
            "untied_generic_surface": 1.00,
            "update_matched_dense": 1.02,
            "stratified_generic_shuffle": 1.01,
            "balanced_random_multihash": 1.01,
        }
        nll = {
            role: np.full(4, value * 100 * np.log(2.0), dtype=np.float64)
            for role, value in per_document_bpb.items()
        }
        decision = mechanism_decision(
            per_document_bpb,
            nll,
            raw,
            anchor_bpb=0.99,
        )
        self.assertEqual(decision["status"], "foldable_multihash_mechanism_pass")
        self.assertTrue(decision["fresh_korean_multiseed_stage_authorized"])
        self.assertTrue(decision["surface_assignment_supported"])


if __name__ == "__main__":
    unittest.main()
