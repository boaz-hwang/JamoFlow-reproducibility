import unittest

import numpy as np

from jamoflow.phase2_analysis import gate_effect_checks, korean_test_strata
from jamoflow.utf8 import prefix_boundary_mask


class Phase2AnalysisTests(unittest.TestCase):
    def test_korean_strata_are_deterministic_and_quartiles_partition(self) -> None:
        rows = [
            ("한글 문장 " * 8).encode("utf-8")[:64],
            ("한글ABC 123 " * 8).encode("utf-8")[:64],
            ("漢字 한글 " * 8).encode("utf-8")[:64],
            ("ㄱ 가 " * 12).encode("utf-8")[:64],
        ]
        rows = [row + b" " * (64 - len(row)) for row in rows]
        data = b"".join(rows)
        masks = np.frombuffer(prefix_boundary_mask(data)[:-1], dtype=np.uint8).reshape(
            -1, 64
        )
        strata, metadata = korean_test_strata(data, masks, sequence_length=64)

        self.assertTrue(strata["latin_mixed"].selected[1])
        self.assertTrue(strata["digit_mixed"].selected[1])
        self.assertTrue(strata["hanja_mixed"].selected[2])
        self.assertTrue(strata["compatibility_jamo_present"].selected[3])
        self.assertTrue(strata["modern_jamo_present"].selected[3])
        quartile_total = sum(
            strata[f"whitespace_density_q{index}"].selected.astype(int)
            for index in range(1, 5)
        )
        np.testing.assert_array_equal(quartile_total, np.ones(4, dtype=int))
        self.assertEqual(metadata["sequence_count"], 4)

    def test_gate_effect_checks_use_all_preregistered_conditions(self) -> None:
        passed = gate_effect_checks(
            [-0.004, -0.005, -0.006, -0.003, 0.001],
            maximum_mean=-0.003,
            interval_upper=-0.0001,
        )
        self.assertTrue(passed["primary_effect_pass"])
        failed = gate_effect_checks(
            [-0.004, -0.005, -0.006, 0.001, 0.001],
            maximum_mean=-0.003,
            interval_upper=0.0001,
        )
        self.assertFalse(failed["primary_effect_pass"])


if __name__ == "__main__":
    unittest.main()
