import unittest

import numpy as np
import torch

from scripts.hangul_draft_acceptance_core import (
    ARCHITECTURES,
    HANGUL_TABLES,
    HEAD_TRAINING_SEEDS,
    PRIMARY_HANGUL_DRAFT,
    DeviceHangulTables,
    array_sha256,
    build_head,
    evaluate_gates,
    hangul_codepoint,
    hangul_components,
    pair_bytes,
    pair_index,
    paired_prompt_bootstrap,
    proposal_metrics,
    propose_pairs,
    trainable_parameter_count,
    training_loss,
)
from scripts.run_hangul_draft_acceptance_preflight import (
    _hangul_locations,
    _select_prompts,
)
from jamoflow.utf8 import prefix_boundary_mask


class HangulDraftAcceptanceCoreTest(unittest.TestCase):
    def test_hangul_tables_cover_every_precomposed_syllable(self):
        self.assertEqual(len(HANGUL_TABLES.lead), 11_172)
        self.assertEqual(set(HANGUL_TABLES.lead.tolist()), {0, 1, 2, 3})
        self.assertEqual(
            len(set(zip(HANGUL_TABLES.lead, HANGUL_TABLES.pair, strict=True))),
            11_172,
        )
        for codepoint in (0xAC00, 0xAC01, 0xB098, 0xD7A3):
            components = hangul_components(codepoint)
            self.assertEqual(hangul_codepoint(*components), codepoint)
            raw = chr(codepoint).encode("utf-8")
            self.assertEqual(pair_bytes(pair_index(raw[1], raw[2])), (raw[1], raw[2]))

    def test_heads_are_parameter_matched_and_propose_valid_hangul(self):
        torch.manual_seed(7)
        hidden = torch.randn(16, 192)
        lead = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
        tables = DeviceHangulTables.build("cpu")
        counts = {}
        for architecture in ARCHITECTURES:
            model = build_head(architecture).eval()
            counts[architecture] = trainable_parameter_count(model)
            with torch.inference_mode():
                pairs = propose_pairs(model, hidden, lead, tables)
            self.assertEqual(pairs.shape, (16,))
            for row, pair in enumerate(pairs.tolist()):
                second, third = pair_bytes(pair)
                raw = bytes((0xEA + int(lead[row]), second, third))
                codepoint = ord(raw.decode("utf-8"))
                self.assertTrue(0xAC00 <= codepoint <= 0xD7A3)
        self.assertLess(max(counts.values()) / min(counts.values()), 1.25)
        self.assertEqual(
            counts,
            {
                "generic_independent_utf8": 41_728,
                "generic_joint_utf8": 42_733,
                "hangul_parallel_components": 42_468,
                "hangul_conditional_components": 39_604,
            },
        )

    def test_training_loss_is_finite_for_every_head(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = build_head(architecture)
                hidden = torch.randn(8, 192)
                onset = torch.arange(8, dtype=torch.long) % 19
                vowel = torch.arange(8, dtype=torch.long) % 21
                coda = torch.arange(8, dtype=torch.long) % 28
                codepoint = 0xAC00 + (onset * 21 + vowel) * 28 + coda
                second = ((codepoint >> 6) & 0x3F).long()
                third = (codepoint & 0x3F).long()
                lead = ((0xE0 | (codepoint >> 12)) - 0xEA).long()
                loss = training_loss(
                    model,
                    hidden,
                    lead,
                    second,
                    third,
                    onset,
                    vowel,
                    coda,
                )
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()

    def test_proposal_metrics_counts_prefix_acceptance(self):
        target_second = np.asarray([1, 2, 3, 4], dtype=np.uint8)
        target_third = np.asarray([5, 6, 7, 8], dtype=np.uint8)
        prediction = np.asarray(
            [1 * 64 + 5, 2 * 64 + 0, 0 * 64 + 7, 4 * 64 + 8],
            dtype=np.int64,
        )
        result = proposal_metrics(
            prediction,
            target_second,
            target_third,
            np.asarray([True, True, False, True], dtype=np.bool_),
        )
        self.assertEqual(result["first_continuation_acceptance"], 0.75)
        self.assertEqual(result["complete_pair_acceptance"], 0.5)
        self.assertEqual(result["mean_accepted_suffix_bytes"], 1.25)
        self.assertAlmostEqual(
            result["complete_pair_acceptance_when_target_hangul"], 2 / 3
        )

    def test_prompt_bootstrap_is_paired_and_deterministic(self):
        prompt = np.repeat(np.arange(4, dtype=np.int64), 3)
        left = np.asarray([True] * 9 + [False] * 3, dtype=np.bool_)
        right = np.asarray([False, True, False] * 4, dtype=np.bool_)
        first = paired_prompt_bootstrap(
            left,
            right,
            prompt,
            repetitions=1000,
            seed=19,
        )
        second = paired_prompt_bootstrap(
            left,
            right,
            prompt,
            repetitions=1000,
            seed=19,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(
            first["point_difference"], float(left.mean() - right.mean())
        )

    @staticmethod
    def _gate_rows(conditional_acceptance: float = 0.50):
        rows = {}
        for architecture in ARCHITECTURES:
            acceptance = (
                conditional_acceptance if architecture == PRIMARY_HANGUL_DRAFT else 0.47
            )
            rows[architecture] = {
                "free_attempt_count": 3000,
                "median_free_complete_pair_acceptance": acceptance,
                "median_free_mean_accepted_suffix_bytes": 1.1,
                "per_seed_free_complete_pair_acceptance": {
                    str(seed): acceptance for seed in HEAD_TRAINING_SEEDS
                },
                "median_head_latency_ms": 0.5,
            }
        return rows

    def test_gate_requires_systems_and_korean_specificity(self):
        specificity = {
            "point_difference": 0.03,
            "ci_lower": 0.01,
            "ci_upper": 0.05,
            "prompt_count": 128,
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 1,
        }
        arguments = {
            "minimum_attempts": 2000,
            "minimum_complete_pair_acceptance": 0.40,
            "minimum_mean_accepted_suffix_bytes": 0.90,
            "minimum_per_seed_complete_pair_acceptance": 0.35,
            "maximum_median_head_latency_ms": 1.0,
            "minimum_specificity_acceptance_gain": 0.01,
        }
        passed = evaluate_gates(self._gate_rows(), specificity, **arguments)
        self.assertTrue(passed["overall_hangul_prototype_authorized"])
        self.assertTrue(passed["generic_control_prototype_authorized"])
        self.assertEqual(
            passed["recommended_next_stage"], "hangul_exact_verifier_prototype"
        )
        failed = evaluate_gates(
            self._gate_rows(conditional_acceptance=0.30),
            specificity,
            **arguments,
        )
        self.assertFalse(failed["overall_hangul_prototype_authorized"])
        self.assertTrue(failed["generic_control_prototype_authorized"])
        self.assertEqual(
            failed["recommended_next_stage"],
            "generic_joint_exact_verifier_diagnostic_only",
        )

    def test_array_hash_binds_dtype_and_shape(self):
        base = np.arange(6, dtype=np.int64).reshape(2, 3)
        self.assertNotEqual(array_sha256(base), array_sha256(base.astype(np.float64)))
        self.assertNotEqual(array_sha256(base), array_sha256(base.reshape(3, 2)))

    def test_teacher_context_locations_do_not_cross_rows(self):
        rows = [
            ("A가나다라마바사" * 40).encode("utf-8")[:512],
            ("B하호후히헤" * 50).encode("utf-8")[:512],
        ]
        # Complete rows with ASCII so shape and strict boundaries stay fixed.
        rows = [row + b"x" * (512 - len(row)) for row in rows]
        data = b"".join(rows)
        inputs = np.frombuffer(data, dtype=np.uint8).reshape(2, 512)
        boundaries = np.frombuffer(
            bytes(prefix_boundary_mask(data)[:-1]), dtype=np.uint8
        ).reshape(2, 512)
        locations = _hangul_locations(inputs, boundaries)
        self.assertGreater(len(locations), 100)
        self.assertTrue(bool(np.all(locations[:, 1] >= 1)))
        self.assertTrue(bool(np.all(locations[:, 1] <= 509)))
        for row, start in locations:
            scalar = bytes(inputs[row, start : start + 3]).decode("utf-8")
            self.assertTrue(0xAC00 <= ord(scalar) <= 0xD7A3)

    def test_prompt_selection_is_deterministic_disjoint_and_boundary_aligned(self):
        data = ("가나다라마바사아자차카타파하 한국어 실험 문장. " * 500).encode("utf-8")
        first, first_offsets = _select_prompts(
            data,
            prompt_bytes=128,
            count=12,
            minimum_hangul_share=0.8,
        )
        second, second_offsets = _select_prompts(
            data,
            prompt_bytes=128,
            count=12,
            minimum_hangul_share=0.8,
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_offsets, second_offsets)
        for left_index, left in enumerate(first_offsets):
            self.assertEqual(bytes(first[left_index]), data[left : left + 128])
            bytes(first[left_index]).decode("utf-8")
            for right in first_offsets[left_index + 1 :]:
                self.assertTrue(left + 128 <= right or right + 128 <= left)


if __name__ == "__main__":
    unittest.main()
