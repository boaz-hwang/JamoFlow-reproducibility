import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jamoflow.compute_conversion import conversion_policy
from jamoflow.document_inference import document_window_map_from_spans


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_phase3_compute_conversion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_phase3_compute_conversion",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase3ComputeConversionSummaryTests(unittest.TestCase):
    def _window_map(self):
        return document_window_map_from_spans(
            32 * 512,
            512,
            tuple((start, start + 8 * 512) for start in range(0, 32 * 512, 8 * 512)),
        )

    def _evidence(
        self,
        effects: list[float],
        *,
        rate: int = 72,
    ) -> tuple[dict, dict]:
        whitespace = conversion_policy("whitespace", rate)
        codepoint = conversion_policy("codepoint", rate)
        reports: dict = {}
        losses: dict = {}
        scale = MODULE.TARGETS_PER_SEQUENCE * math.log(2)
        for seed, effect in zip(MODULE.ALL_SEEDS, effects, strict=True):
            reports[seed] = {
                whitespace: {"evaluation": {"test": {"bpb": 2.0 + effect}}},
                codepoint: {"evaluation": {"test": {"bpb": 2.0}}},
            }
            losses[seed] = {
                whitespace: np.full(32, 10.0 + effect * scale),
                codepoint: np.full(32, 10.0),
            }
        return reports, losses

    def test_confirmation_gate_passes_stable_same_rate_gain(self) -> None:
        reports, losses = self._evidence([-0.004] * 5)
        result = MODULE.confirmation_same_rate_summary(
            reports,
            losses,
            MODULE.ALL_SEEDS,
            72,
            document_window_map=self._window_map(),
            repetitions=100,
        )
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["negative_seed_count"], 5)
        self.assertLess(result["hierarchical_bootstrap_95_interval"]["upper"], 0)

    def test_confirmation_gate_rejects_insufficient_seed_signs(self) -> None:
        reports, losses = self._evidence(
            [-0.006, -0.006, -0.006, 0.001, 0.001]
        )
        result = MODULE.confirmation_same_rate_summary(
            reports,
            losses,
            MODULE.ALL_SEEDS,
            72,
            document_window_map=self._window_map(),
            repetitions=100,
        )
        self.assertFalse(result["overall_pass"])
        self.assertEqual(result["negative_seed_count"], 3)

    def test_confirmation_inference_rejects_seed_subsets(self) -> None:
        reports, losses = self._evidence([-0.004] * 5)
        with self.assertRaisesRegex(ValueError, "all five"):
            MODULE.confirmation_same_rate_summary(
                reports,
                losses,
                MODULE.ALL_SEEDS[:3],
                72,
                document_window_map=self._window_map(),
                repetitions=10,
            )

    def test_primary_dependency_requires_five_seed_gate_j_and_ood(self) -> None:
        payload = {
            "confirmation_authorization": {
                "authorization_kind": "phase3_corrected_gate_i_confirmation_v1"
            },
            "gate_i": {"overall_pass": True},
            "gate_j": {"overall_pass": True},
            "integrity": {"all_integrity_checks_pass": True},
            "ood": {
                "gate_i_ood_guard": {"pass": True},
                "integrity": {"all_integrity_checks_pass": True},
            },
            "policies": list(MODULE.PRIMARY_POLICIES),
            "seeds": list(MODULE.ALL_SEEDS),
            "targets_per_sequence": MODULE.TARGETS_PER_SEQUENCE,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "primary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = {
                "primary_gate_summary_sha256": MODULE._sha256(path),
                "primary_gate_i": payload["gate_i"],
            }
            MODULE._validate_primary_summary(payload, path, manifest)
            payload["seeds"] = list(MODULE.INITIAL_SEEDS)
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest["primary_gate_summary_sha256"] = MODULE._sha256(path)
            with self.assertRaisesRegex(ValueError, "five-seed Gate J"):
                MODULE._validate_primary_summary(payload, path, manifest)

    def test_confirmation_reads_rate_from_selection_v2_decision(self) -> None:
        reports, _ = self._evidence([-0.004] * 5, rate=72)
        policies = (
            conversion_policy("codepoint", 72),
            conversion_policy("whitespace", 72),
        )
        selection = {
            "decision": {
                "rate_selection": {
                    "selected_rate": 72,
                    "status": "selected",
                }
            }
        }
        rate_selection, gate = MODULE._confirmation_historical_screening_gate(
            selection,
            reports,
            {seed: 2.0 for seed in MODULE.INITIAL_SEEDS},
            72,
            policies,
        )
        self.assertEqual(rate_selection["selected_rate"], 72)
        self.assertTrue(gate["overall_pass"])
        with self.assertRaisesRegex(ValueError, "canonical decision"):
            MODULE._confirmation_historical_screening_gate(
                {"calibration_rate_selection": rate_selection},
                reports,
                {seed: 2.0 for seed in MODULE.INITIAL_SEEDS},
                72,
                policies,
            )

    def test_parser_locks_primary_summary_and_bootstrap_repetitions(self) -> None:
        args = MODULE.build_parser().parse_args(["--stage", "initial"])
        self.assertEqual(
            args.primary_summary,
            MODULE.PHASE3_PRIMARY_SUMMARY_PATH,
        )
        self.assertEqual(
            args.bootstrap_repetitions,
            MODULE.CONFIRMATION_BOOTSTRAP_REPETITIONS,
        )
        with self.assertRaises(SystemExit):
            MODULE.build_parser().parse_args(
                [
                    "--stage",
                    "confirmation",
                    "--bootstrap-repetitions",
                    "100",
                ]
            )


if __name__ == "__main__":
    unittest.main()
