from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from jamoflow.inference_actual_v5 import RUNTIME_COUNTER_NAMES


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_inference_actual_v5r3_counter_fix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_inference_actual_v5r3_counter_fix_test_module", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SummaryCounterFixTests(unittest.TestCase):
    def test_full_session_is_validated_as_exact_five_seed_slices(self) -> None:
        emitted = np.full((5, 64, 5), 128, dtype=np.int64)
        counters = {
            name: np.zeros((5, 64, 5), dtype=np.int64)
            for name in RUNTIME_COUNTER_NAMES
        }
        validator = mock.Mock()

        MODULE.validate_session_runtime_counter_arrays(
            counters,
            requires_entropy_router=False,
            mode="controlled_replay",
            emitted_output_bytes=emitted,
            base_validator=validator,
        )

        self.assertEqual(validator.call_count, 5)
        for call in validator.call_args_list:
            self.assertEqual(call.kwargs["emitted_output_bytes"].shape, (64, 5))
            self.assertEqual(
                {value.shape for value in call.args[0].values()}, {(64, 5)}
            )

    def test_wrong_seed_or_repetition_shape_fails_before_base_validator(self) -> None:
        emitted = np.full((4, 64, 5), 128, dtype=np.int64)
        counters = {
            name: np.zeros((4, 64, 5), dtype=np.int64)
            for name in RUNTIME_COUNTER_NAMES
        }
        validator = mock.Mock()

        with self.assertRaisesRegex(ValueError, "corrected session runtime counter shape"):
            MODULE.validate_session_runtime_counter_arrays(
                counters,
                requires_entropy_router=False,
                mode="controlled_replay",
                emitted_output_bytes=emitted,
                base_validator=validator,
            )
        validator.assert_not_called()

    def test_counter_shape_rotation_fails_before_base_validator(self) -> None:
        emitted = np.full((5, 64, 5), 128, dtype=np.int64)
        counters = {
            name: np.zeros((5, 64, 5), dtype=np.int64)
            for name in RUNTIME_COUNTER_NAMES
        }
        counters[next(iter(RUNTIME_COUNTER_NAMES))] = np.zeros(
            (5, 64, 4), dtype=np.int64
        )
        validator = mock.Mock()

        with self.assertRaisesRegex(ValueError, "corrected session runtime counter shape"):
            MODULE.validate_session_runtime_counter_arrays(
                counters,
                requires_entropy_router=False,
                mode="controlled_replay",
                emitted_output_bytes=emitted,
                base_validator=validator,
            )
        validator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
