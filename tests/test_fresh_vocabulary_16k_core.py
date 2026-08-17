from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_core import (
    ANCHOR_ROLES,
    BASE_VOCABULARY_SIZE,
    CANDIDATE_ROLE,
    INPUT_UPDATE_MULTIPLIER,
    OUTPUT_UPDATE_MULTIPLIER,
    ROLES,
    TARGET_VOCABULARY_SIZE,
    batch_raw_target_bytes,
    build_canonical_decomposition_table,
    build_transferred_model,
    expected_parameter_count,
    head_learning_rate,
    inplace_stage_contract,
    quality_decision,
    role_definition,
)
from vocabulary_transfer_probe_core import (
    build_canonical_bpe_decomposition_table,
    build_transferred_model as build_old_transferred_model,
    state_mapping_sha256,
)
from vocabulary_transfer_probe_protocol import base_checkpoint_state


def _losses(raw: np.ndarray, target_bpb: float) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64) * math.log(2.0) * target_bpb


class FreshVocabulary16KCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizers = load_tokenizers()
        cls.base_state = base_checkpoint_state()

    def test_role_contract_has_one_fixed_candidate_and_exact_multipliers(self) -> None:
        self.assertEqual(len(ROLES), 5)
        self.assertEqual(CANDIDATE_ROLE, "dense16k_update_geometry")
        for role in ROLES:
            definition = role_definition(role)
            self.assertIn(
                definition["vocabulary_size"],
                (2_048, 8_192, 16_000),
            )
        for role in ("dense8k_update_geometry_v2", CANDIDATE_ROLE):
            scaling = role_definition(role)["post_adamw_new_row_scaling"]
            self.assertEqual(scaling["input_multiplier"], INPUT_UPDATE_MULTIPLIER)
            self.assertEqual(scaling["output_multiplier"], OUTPUT_UPDATE_MULTIPLIER)
            self.assertFalse(scaling["validation_metric_used"])

    def test_parameter_contract_counts_untied_16k_head(self) -> None:
        self.assertEqual(expected_parameter_count(2_048), 19_667_328)
        self.assertEqual(expected_parameter_count(8_192), 25_172_352)
        self.assertEqual(expected_parameter_count(16_000), 31_168_896)

    def test_generic_8k_transfer_is_bitwise_compatible_with_prior_builder(self) -> None:
        base_tokenizer, base_pieces = self.tokenizers[2_048]
        target_tokenizer, target_pieces = self.tokenizers[8_192]
        generic_rows = build_canonical_decomposition_table(
            base_tokenizer,
            target_tokenizer,
            base_pieces,
            target_pieces,
        )
        old_rows = build_canonical_bpe_decomposition_table(
            base_tokenizer,
            target_tokenizer,
            base_pieces,
            target_pieces,
        )
        self.assertEqual(generic_rows, old_rows)
        generic, generic_audit = build_transferred_model(
            8_192,
            base_state=self.base_state,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=generic_rows,
        )
        old, old_audit = build_old_transferred_model(
            "untied_uniform_in_byte_weighted_out",
            base_state=self.base_state,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=old_rows,
        )
        self.assertEqual(
            state_mapping_sha256(generic.state_dict()),
            state_mapping_sha256(old.state_dict()),
        )
        self.assertEqual(
            generic_audit.initialized_input_weight_sha256,
            old_audit.initialized_input_weight_sha256,
        )
        self.assertEqual(
            generic_audit.initialized_output_weight_sha256,
            old_audit.initialized_output_weight_sha256,
        )

    def test_16k_transfer_is_exact_untied_extension(self) -> None:
        base_tokenizer, base_pieces = self.tokenizers[2_048]
        target_tokenizer, target_pieces = self.tokenizers[TARGET_VOCABULARY_SIZE]
        rows = build_canonical_decomposition_table(
            base_tokenizer,
            target_tokenizer,
            base_pieces,
            target_pieces,
        )
        model, audit = build_transferred_model(
            TARGET_VOCABULARY_SIZE,
            base_state=self.base_state,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=rows,
        )
        self.assertEqual(len(rows), TARGET_VOCABULARY_SIZE)
        self.assertTrue(audit.exact_reconstruction)
        self.assertEqual(audit.shared_token_count, BASE_VOCABULARY_SIZE)
        self.assertEqual(audit.new_token_count, TARGET_VOCABULARY_SIZE - 2_048)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            expected_parameter_count(TARGET_VOCABULARY_SIZE),
        )
        self.assertNotEqual(
            model.model.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr()
        )
        self.assertTrue(
            torch.equal(
                model.model.embed_tokens.weight[:2_048],
                self.base_state["model.embed_tokens.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.lm_head.weight[:2_048],
                self.base_state["lm_head.weight"],
            )
        )

    def test_raw_progress_schedule_and_stage_boundary(self) -> None:
        raw = np.full(101, 511, dtype=np.int64)
        batches = batch_raw_target_bytes(raw)
        self.assertEqual(len(batches), 4)
        self.assertEqual(int(batches.sum()), int(raw.sum()))
        stage = inplace_stage_contract(raw)
        self.assertGreaterEqual(stage["stage_one_realized_raw_fraction"], 0.60)
        self.assertEqual(stage["total_optimizer_steps"], 4)
        self.assertGreater(
            head_learning_rate(
                "dense16k_standard_joint",
                cumulative_raw_target_bytes=100,
                total_raw_target_bytes=10_000,
                stage_one_raw_target_bytes=None,
            ),
            0.0,
        )
        with self.assertRaises(ValueError):
            head_learning_rate(
                "dense16k_standard_joint",
                cumulative_raw_target_bytes=100,
                total_raw_target_bytes=10_000,
                stage_one_raw_target_bytes=6_000,
            )

    def test_quality_gate_requires_both_anchors_and_both_controls(self) -> None:
        raw = np.asarray([100, 120, 140, 160, 180, 200], dtype=np.int64)
        values = {
            "dense2k_joint_v2": _losses(raw, 1.400),
            "dense8k_update_geometry_v2": _losses(raw, 1.390),
            "dense16k_standard_joint": _losses(raw, 1.405),
            "dense16k_inplace_two_stage": _losses(raw, 1.410),
            "dense16k_update_geometry": _losses(raw, 1.395),
        }
        decision = quality_decision(values, raw)
        self.assertEqual(decision["status"], "pass_16k_quality_for_actual_preflight")
        self.assertTrue(decision["actual_inference_preflight_authorized"])
        self.assertEqual(set(decision["candidate_noninferiority_vs_each_anchor"]), set(ANCHOR_ROLES))
        self.assertTrue(decision["cross_vocabulary_geometry_supported"])

        anchor_failure = dict(values)
        anchor_failure[CANDIDATE_ROLE] = _losses(raw, 1.402)
        failed = quality_decision(anchor_failure, raw)
        self.assertEqual(failed["status"], "fail_16k_anchor_noninferiority")
        self.assertFalse(failed["actual_inference_preflight_authorized"])

        control_failure = dict(values)
        control_failure["dense16k_standard_joint"] = _losses(raw, 1.396)
        failed = quality_decision(control_failure, raw)
        self.assertEqual(failed["status"], "fail_16k_method_controls")
        self.assertFalse(failed["actual_inference_preflight_authorized"])

    def test_decision_rejects_missing_role_or_float_raw_bytes(self) -> None:
        raw = np.asarray([100, 120], dtype=np.int64)
        values = {role: _losses(raw, 1.4) for role in ROLES}
        changed = dict(values)
        changed.pop(ROLES[-1])
        with self.assertRaises(ValueError):
            quality_decision(changed, raw)
        with self.assertRaises(ValueError):
            quality_decision(values, raw.astype(np.float64))


if __name__ == "__main__":
    unittest.main()
