from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from jamoflow.inference_selection_v2 import CONFIRMATION_SEEDS, INITIAL_SEEDS


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "seal_inference_selection_plan_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seal_inference_selection_plan_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SealInferenceSelectionPlanV2Tests(unittest.TestCase):
    def _summary(self, seeds: tuple[int, ...], *, completed: bool) -> dict:
        payload = {
            "seeds": list(seeds),
            "policies": [
                "fixed_byte_6",
                "causal_codepoint_grid",
                "causal_whitespace_grid",
            ],
            "integrity": {"all_integrity_checks_pass": True},
            "targets_per_sequence": 511,
        }
        if completed:
            payload.update(
                {
                    "confirmation_authorization": {
                        "authorization_kind": (
                            "phase3_corrected_gate_i_confirmation_v1"
                        )
                    },
                    "gate_i": {"overall_pass": True},
                    "gate_j": {"overall_pass": True},
                    "ood": {
                        "gate_i_ood_guard": {"pass": True},
                        "integrity": {"all_integrity_checks_pass": True},
                    },
                }
            )
        return payload

    def test_primary_identity_requires_completed_five_seed_gate_j_and_ood(self) -> None:
        seeds = (*INITIAL_SEEDS, *CONFIRMATION_SEEDS)
        policies = (
            "fixed_byte_6",
            "causal_codepoint_grid",
            "causal_whitespace_grid",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            with mock.patch.object(MODULE, "_require_tracked_head_blob"):
                path.write_text(
                    json.dumps(self._summary(seeds, completed=True)),
                    encoding="utf-8",
                )
                MODULE._load_summary_identity(
                    path,
                    seeds=seeds,
                    policies=policies,
                    require_completed_confirmation=True,
                )

                path.write_text(
                    json.dumps(self._summary(INITIAL_SEEDS, completed=True)),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "identity is invalid"):
                    MODULE._load_summary_identity(
                        path,
                        seeds=seeds,
                        policies=policies,
                        require_completed_confirmation=True,
                    )

                failed = self._summary(seeds, completed=True)
                failed["gate_j"] = {"overall_pass": False}
                path.write_text(json.dumps(failed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Gate J and OOD"):
                    MODULE._load_summary_identity(
                        path,
                        seeds=seeds,
                        policies=policies,
                        require_completed_confirmation=True,
                    )


if __name__ == "__main__":
    unittest.main()
