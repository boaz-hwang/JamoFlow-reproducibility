import unittest

import numpy as np

from jamoflow.compute_conversion import (
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
    conversion_policy,
    initial_conversion_gate,
    select_rate_from_calibration,
)
from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.phase3 import PHASE3_MODEL_SPEC
from jamoflow.utf8 import prefix_boundary_mask


class ComputeConversionTests(unittest.TestCase):
    def _masks(self) -> tuple[np.ndarray, np.ndarray]:
        raw = ("한국어 compute conversion 문장입니다. " * 100).encode("utf-8")
        data = raw[: PHASE3_MODEL_SPEC.sequence_length]
        boundaries = np.frombuffer(
            prefix_boundary_mask(data)[:-1],
            dtype=np.uint8,
        ).reshape(1, -1)
        whitespace = compact_whitespace_mask(data).reshape(1, -1)
        return boundaries, whitespace

    def test_rate_specs_keep_parameters_geometry_except_patch_count(self) -> None:
        for rate in CONVERSION_RATES:
            spec = conversion_model_spec(rate)
            self.assertEqual(spec.patch_count, rate)
            original = PHASE3_MODEL_SPEC.to_dict()
            converted = spec.to_dict()
            converted["patch_count"] = original["patch_count"]
            self.assertEqual(converted, original)

    def test_conversion_matrices_are_exact_rate_and_unicode_safe(self) -> None:
        boundaries, whitespace = self._masks()
        for rate in CONVERSION_RATES:
            matrices = conversion_patch_matrices(
                boundaries,
                whitespace,
                rate=rate,
            )
            self.assertEqual(
                set(matrices),
                {
                    conversion_policy("codepoint", rate),
                    conversion_policy("whitespace", rate),
                },
            )
            for matrix in matrices.values():
                self.assertEqual(int((matrix[0, 1:] > 0).sum()), rate)
                self.assertEqual(int(matrix[0, 1:].sum()), 512)

    def test_calibration_selection_prefers_64_then_72(self) -> None:
        seeds = (1729, 2718, 31415)
        primary = {seed: 2.0 for seed in seeds}
        values = {
            seed: {
                conversion_policy("whitespace", 64): 2.02,
                conversion_policy("whitespace", 72): 2.005,
            }
            for seed in seeds
        }
        selection = select_rate_from_calibration(values, primary)
        self.assertEqual(selection.selected_rate, 72)
        for seed in seeds[:2]:
            values[seed][conversion_policy("whitespace", 64)] = 2.005
        selection = select_rate_from_calibration(values, primary)
        self.assertEqual(selection.selected_rate, 64)

    def test_initial_gate_requires_quality_conversion_and_same_rate_signal(self) -> None:
        seeds = (1729, 2718, 31415)
        primary = {seed: 2.0 for seed in seeds}
        values = {
            seed: {
                conversion_policy("whitespace", 72): 2.005,
                conversion_policy("codepoint", 72): 2.010,
            }
            for seed in seeds
        }
        result = initial_conversion_gate(
            values,
            primary,
            selected_rate=72,
        )
        self.assertTrue(result["overall_pass"])
        values[31415][conversion_policy("whitespace", 72)] = 2.03
        values[31415][conversion_policy("codepoint", 72)] = 2.02
        result = initial_conversion_gate(
            values,
            primary,
            selected_rate=72,
        )
        self.assertFalse(result["overall_pass"])


if __name__ == "__main__":
    unittest.main()
