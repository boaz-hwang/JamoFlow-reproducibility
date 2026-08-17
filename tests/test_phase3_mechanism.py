from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.phase3 import PHASE3_MODEL_SPEC, structural_patch_matrices
from jamoflow.phase3_mechanism import (
    DELAYED_POLICY,
    INITIAL_SEEDS,
    MECHANISM_POLICIES,
    PLACEBO_POLICY,
    array_sha256,
    build_mechanism_patch_matrices,
    mechanism_cache_provenance,
    merge_mechanism_manifest,
    validate_mechanism_execution_gate,
)
from jamoflow.utf8 import prefix_boundary_mask


ROOT = Path(__file__).parents[1]
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_phase3_mechanism",
    ROOT / "scripts" / "summarize_phase3_mechanism.py",
)
assert SUMMARY_SPEC is not None and SUMMARY_SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
SUMMARY_SPEC.loader.exec_module(SUMMARY)


class Phase3MechanismTests(unittest.TestCase):
    def test_execution_gate_requires_i_then_j(self) -> None:
        summary = {
            "gate_i": {"status": "pass", "overall_pass": True},
            "gate_j": {"status": "fail", "overall_pass": False},
        }
        initial = validate_mechanism_execution_gate(
            summary, (1729, 2718, 31415), quick=False
        )
        self.assertEqual(initial["required_gate"], "gate_i")
        with self.assertRaises(ValueError):
            validate_mechanism_execution_gate(
                summary, (57721, 65537), quick=False
            )
        summary["gate_j"] = {"status": "pass", "overall_pass": True}
        confirmation = validate_mechanism_execution_gate(
            summary, (57721, 65537), quick=False
        )
        self.assertEqual(confirmation["required_gate"], "gate_j")

    def test_quick_gate_bypass_is_not_evidence(self) -> None:
        result = validate_mechanism_execution_gate(
            None, (1729,), quick=True
        )
        self.assertFalse(result["evidence_eligible"])
        self.assertEqual(result["status"], "quick_smoke_bypass")

    def test_manifest_merge_preserves_gate_invocations(self) -> None:
        base = {
            "phase": "phase3_mechanism",
            "created_at": "first",
            "git_commit": "a",
            "device": "cpu",
            "platform": "test",
            "versions": {},
            "seeds": [1729],
            "policies": [DELAYED_POLICY, PLACEBO_POLICY],
            "quick_smoke_only": False,
            "language": "ko",
            "limits": {"train": 1},
            "global_max_position_embeddings": 1032,
            "model_spec": {"width": 384},
            "optimization_spec": {"batch": 32},
            "streams": {"train": {"sha": "x"}},
            "gate_authorization": {"required_gate": "gate_i"},
        }
        first = merge_mechanism_manifest(None, base)
        later = dict(base)
        later.update(
            {
                "created_at": "later",
                "git_commit": "b",
                "seeds": [57721],
                "gate_authorization": {"required_gate": "gate_j"},
            }
        )
        merged = merge_mechanism_manifest(first, later)
        self.assertEqual(merged["seeds"], [1729, 57721])
        self.assertEqual(len(merged["invocations"]), 2)
        self.assertEqual(
            merged["invocations"][1]["gate_authorization"]["required_gate"],
            "gate_j",
        )

    def test_mechanism_cache_provenance_tracks_all_source_matrices(self) -> None:
        spec = replace(PHASE3_MODEL_SPEC, sequence_length=4)
        inputs = {
            split: np.arange(8, dtype=np.uint8).reshape(2, 4)
            for split in ("train", "calibration", "test")
        }
        boundaries = {
            split: np.ones((2, 4), dtype=np.uint8) for split in inputs
        }
        whitespace = {
            split: np.zeros((2, 4), dtype=np.uint8) for split in inputs
        }
        first = mechanism_cache_provenance(
            inputs, boundaries, whitespace, spec
        )
        whitespace["test"][1, 2] = 1
        second = mechanism_cache_provenance(
            inputs, boundaries, whitespace, spec
        )
        self.assertNotEqual(first, second)

    def test_summary_manifest_requires_evidentiary_checkpointed_invocation(
        self,
    ) -> None:
        manifest = {
            "phase": "phase3_mechanism",
            "quick_smoke_only": False,
            "model_spec": PHASE3_MODEL_SPEC.to_dict(),
            "optimization_spec": SUMMARY.PHASE3_OPTIMIZATION_SPEC.to_dict(),
            "limits": SUMMARY.FULL_LIMITS,
            "policies": list(MECHANISM_POLICIES),
            "seeds": list(INITIAL_SEEDS),
            "invocations": [
                {
                    "seeds": list(INITIAL_SEEDS),
                    "policies": list(MECHANISM_POLICIES),
                    "save_checkpoints": True,
                    "gate_authorization": {
                        "required_gate": "gate_i",
                        "evidence_eligible": True,
                    },
                }
            ],
        }
        SUMMARY._validate_manifest_execution(manifest, INITIAL_SEEDS)
        manifest["invocations"][0]["save_checkpoints"] = False
        with self.assertRaisesRegex(ValueError, "evidentiary invocation"):
            SUMMARY._validate_manifest_execution(manifest, INITIAL_SEEDS)

    def test_summary_preserves_historical_and_confirmation_authorization(self) -> None:
        manifest = {
            "invocations": [
                {
                    "seeds": list(INITIAL_SEEDS),
                    "primary_summary_sha256": "historical",
                    "gate_authorization": {"required_gate": "gate_i"},
                },
                {
                    "seeds": [57721, 65537],
                    "primary_summary_sha256": "confirmation",
                    "gate_authorization": {"required_gate": "gate_j"},
                },
            ]
        }
        initial = SUMMARY._validate_authorization_summary_lineage(
            manifest,
            INITIAL_SEEDS,
            historical_initial_sha256="historical",
            current_primary_sha256="corrected-initial",
        )
        self.assertTrue(initial["historical_initial_authorization_match"])
        self.assertIsNone(initial["confirmation_gate_j_authorization_match"])

        final = SUMMARY._validate_authorization_summary_lineage(
            manifest,
            SUMMARY.ALL_SEEDS,
            historical_initial_sha256="historical",
            current_primary_sha256="confirmation",
        )
        self.assertTrue(final["confirmation_gate_j_authorization_match"])

        with self.assertRaisesRegex(ValueError, "historical"):
            SUMMARY._validate_authorization_summary_lineage(
                manifest,
                INITIAL_SEEDS,
                historical_initial_sha256="different",
                current_primary_sha256="corrected-initial",
            )

    def test_initial_reanalysis_is_separate_from_progression(self) -> None:
        failed = SUMMARY._mechanism_reanalysis_authorization(
            {"gate_i": {"status": "fail_quality", "overall_pass": False}},
            INITIAL_SEEDS,
        )
        self.assertTrue(failed["evidence_eligible"])
        self.assertFalse(failed["current_primary_gate_pass"])
        self.assertFalse(failed["confirmation_progression_authorized"])

        passed = SUMMARY._mechanism_reanalysis_authorization(
            {"gate_i": {"status": "pass", "overall_pass": True}},
            INITIAL_SEEDS,
        )
        self.assertTrue(passed["confirmation_progression_authorized"])

        with self.assertRaisesRegex(ValueError, "finalized"):
            SUMMARY._mechanism_reanalysis_authorization(
                {"gate_i": {"status": "pending", "overall_pass": None}},
                INITIAL_SEEDS,
            )

    def test_confirmation_reanalysis_still_requires_current_gate_j(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate_j did not pass"):
            SUMMARY._mechanism_reanalysis_authorization(
                {"gate_j": {"status": "fail", "overall_pass": False}},
                SUMMARY.ALL_SEEDS,
            )

    def test_primary_summary_is_bound_to_loaded_w_evidence(self) -> None:
        shared_manifest = {
            "model_spec": {"width": 384},
            "optimization_spec": {"batch": 32},
            "limits": {"train": 1},
            "streams": {"test": {"sha256": "stream"}},
        }
        primary = {
            "seeds": [1729],
            "run_manifest": dict(shared_manifest),
            "integrity": {
                "all_integrity_checks_pass": True,
                "by_seed": {
                    "1729": {
                        "checkpoint_state_sha256": {
                            SUMMARY.WHITESPACE_POLICY: "state"
                        },
                        "loss_artifact_sha256": {
                            SUMMARY.WHITESPACE_POLICY: "loss"
                        },
                    }
                },
            },
        }
        loss_hashes = {1729: {SUMMARY.WHITESPACE_POLICY: "loss"}}
        state_hashes = {
            "1729": {SUMMARY.WHITESPACE_POLICY: "state"}
        }
        SUMMARY._validate_primary_summary_context(
            primary,
            shared_manifest,
            (1729,),
            loss_hashes,
            state_hashes,
        )
        state_hashes["1729"][SUMMARY.WHITESPACE_POLICY] = "different"
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            SUMMARY._validate_primary_summary_context(
                primary,
                shared_manifest,
                (1729,),
                loss_hashes,
                state_hashes,
            )

    def test_patch_controls_are_exact_rate_and_rebuild_w(self) -> None:
        spec = replace(
            PHASE3_MODEL_SPEC,
            sequence_length=48,
            patch_count=8,
            patch_stride=6,
        )
        raw = ("한국어 문장과 English 123. " * 40).encode("utf-8")
        data = raw[: 48 * 9]
        inputs = np.frombuffer(data, dtype=np.uint8).reshape(9, 48)
        boundaries = np.frombuffer(
            prefix_boundary_mask(data)[:-1], dtype=np.uint8
        ).reshape(9, 48)
        whitespace = compact_whitespace_mask(data).reshape(9, 48)
        split_inputs = {
            "train": inputs[:4],
            "calibration": inputs[4:7],
            "test": inputs[7:],
        }
        split_boundaries = {
            "train": boundaries[:4],
            "calibration": boundaries[4:7],
            "test": boundaries[7:],
        }
        split_whitespace = {
            "train": whitespace[:4],
            "calibration": whitespace[4:7],
            "test": whitespace[7:],
        }
        matrices, diagnostics = build_mechanism_patch_matrices(
            split_inputs,
            split_boundaries,
            split_whitespace,
            spec,
        )
        for split in split_inputs:
            for policy in (DELAYED_POLICY, PLACEBO_POLICY):
                matrix = matrices[split][policy]
                self.assertEqual(matrix.shape[1], spec.patch_count + 1)
                self.assertTrue(np.all((matrix[:, 1:] > 0).sum(axis=1) == 8))
                self.assertTrue(np.all(matrix[:, 1:].sum(axis=1) == 48))
            primary = structural_patch_matrices(
                split_boundaries[split],
                split_whitespace[split],
                np.zeros_like(split_whitespace[split]),
                spec,
            )["causal_whitespace_grid"]
            self.assertEqual(
                diagnostics["whitespace_reference"][split]["matrix_sha256"],
                array_sha256(primary),
            )
        calibration = diagnostics["placebo_calibration"]
        self.assertAlmostEqual(
            calibration["target_event_trigger_fraction"],
            diagnostics["calibration_target"]["value"],
        )

    def test_gate_m_uses_stricter_final_rule(self) -> None:
        def contrast(
            mean: float,
            negative: int,
            upper: float = -0.001,
            adjusted_pvalue: float = 0.01,
        ):
            return {
                "paired_t_95_interval": {"mean": mean},
                "negative_seed_count": negative,
                "hierarchical_bootstrap_95_interval": {"upper": upper},
                "document_cluster_bootstrap_95_interval": {
                    "upper": upper,
                    "eligible_sequence_fraction_pass": True,
                },
                "holm_mechanism_family": {
                    "holm_adjusted_seed_t_pvalue": adjusted_pvalue,
                },
            }

        initial = {
            name: contrast(-0.0025, 2) for name in SUMMARY.CONTRASTS
        }
        self.assertTrue(
            SUMMARY.gate_m_summary(
                initial, seed_count=3, integrity_pass=True
            )["overall_pass"]
        )
        final = {name: contrast(-0.0035, 4) for name in SUMMARY.CONTRASTS}
        self.assertTrue(
            SUMMARY.gate_m_summary(
                final, seed_count=5, integrity_pass=True
            )["overall_pass"]
        )
        final[next(iter(final))] = contrast(-0.0035, 4, upper=0.0001)
        self.assertFalse(
            SUMMARY.gate_m_summary(
                final, seed_count=5, integrity_pass=True
            )["overall_pass"]
        )
        final = {name: contrast(-0.0035, 4) for name in SUMMARY.CONTRASTS}
        final[next(iter(final))] = contrast(
            -0.0035,
            4,
            adjusted_pvalue=0.051,
        )
        self.assertFalse(
            SUMMARY.gate_m_summary(
                final, seed_count=5, integrity_pass=True
            )["overall_pass"]
        )


if __name__ == "__main__":
    unittest.main()
