from __future__ import annotations

import inspect
import subprocess
import unittest

import numpy as np

import benchmark_compositional_head_preflight as benchmark
from compositional_head_core import ROLE_ORDER
from compositional_head_preflight_protocol import (
    ACTIVE_PATH,
    IMPLEMENTATION_PATHS,
    MEASURED_CASES,
    REPORT_PATH,
    REPETITIONS,
    RESULT_PATH,
    ROOT,
    TIMING_PATH,
    VOCABULARY_SIZES,
    array_sha256,
    assignment_audits,
    canonical_sha256,
    experiment_contract,
)


class CompositionalHeadPreflightProtocolTests(unittest.TestCase):
    def test_implementation_manifest_is_complete_and_unique(self) -> None:
        self.assertEqual(len(IMPLEMENTATION_PATHS), len(set(IMPLEMENTATION_PATHS)))
        missing = [path for path in IMPLEMENTATION_PATHS if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])

    def test_evidence_is_tracked_but_active_sentinel_is_ignored(self) -> None:
        for path in (REPORT_PATH, TIMING_PATH, RESULT_PATH):
            result = subprocess.run(
                ("git", "check-ignore", "-q", str(path.relative_to(ROOT))),
                cwd=ROOT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, path)
        active = subprocess.run(
            ("git", "check-ignore", "-q", str(ACTIVE_PATH.relative_to(ROOT))),
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(active.returncode, 0)

    def test_sealed_contract_has_exact_grid_and_gate(self) -> None:
        contract = experiment_contract()
        self.assertEqual(contract["role_order"], list(ROLE_ORDER))
        self.assertEqual(contract["vocabulary_order"], list(VOCABULARY_SIZES))
        self.assertEqual(contract["measured_cases"], MEASURED_CASES)
        self.assertEqual(contract["repetitions"], REPETITIONS)
        self.assertEqual(contract["minimum_end_to_end_reduction"], 0.10)
        self.assertEqual(contract["minimum_step_reduction"], 0.10)
        self.assertFalse(contract["tokenizer_inside_model_timer"])

    def test_timer_cannot_encode_or_retokenize_cases(self) -> None:
        source = inspect.getsource(benchmark._timed_trial)
        self.assertNotIn("encode_case", source)
        self.assertNotIn("Tokenizer", source)
        self.assertIn("argmax", source)
        self.assertIn("mps.synchronize", source)

    def test_hashes_bind_shape_dtype_values_and_mapping(self) -> None:
        values = np.arange(12, dtype=np.float64).reshape(3, 4)
        changed = values.copy()
        changed[-1, -1] += 1
        self.assertNotEqual(array_sha256(values), array_sha256(changed))
        self.assertNotEqual(array_sha256(values), array_sha256(values.astype(np.float32)))
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))

    def test_real_tokenizer_assignments_are_collision_free(self) -> None:
        audits = assignment_audits()
        expected = {
            f"{kind}_v{size}"
            for size in VOCABULARY_SIZES
            for kind in ("generic_code", "hangul_code")
        }
        self.assertEqual(set(audits), expected)
        for row in audits.values():
            self.assertEqual(row["unique_code_tuples"], row["vocabulary_size"])
            self.assertEqual(row["codebook_count"], 16)
            self.assertEqual(row["codebook_size"], 128)


if __name__ == "__main__":
    unittest.main()
