from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_phase3.py"
SPEC = importlib.util.spec_from_file_location("summarize_phase3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def contrast(mean: float, negative_count: int) -> dict[str, object]:
    return {
        "whitespace_minus_codepoint": {
            "paired_t_95_interval": {"mean": mean},
            "negative_seed_count": negative_count,
            "document_cluster_bootstrap_95_interval": {
                "upper": -0.001,
                "eligible_sequence_fraction_pass": True,
            },
        }
    }


def final_contrasts(
    *,
    codepoint_mean: float = -0.004,
    fixed_mean: float = -0.004,
    negative_count: int = 4,
    bootstrap_upper: float = -0.001,
    document_upper: float = -0.001,
    document_coverage_pass: bool = True,
    adjusted_pvalue: float = 0.04,
) -> dict[str, object]:
    return {
        "whitespace_minus_codepoint": {
            "paired_t_95_interval": {"mean": codepoint_mean},
            "negative_seed_count": negative_count,
            "hierarchical_bootstrap_95_interval": {
                "upper": bootstrap_upper
            },
            "document_cluster_bootstrap_95_interval": {
                "upper": document_upper,
                "eligible_sequence_fraction_pass": document_coverage_pass,
            },
            "holm_primary_family": {
                "holm_adjusted_seed_t_pvalue": adjusted_pvalue
            },
        },
        "whitespace_minus_fixed": {
            "paired_t_95_interval": {"mean": fixed_mean},
            "negative_seed_count": negative_count,
            "hierarchical_bootstrap_95_interval": {
                "upper": bootstrap_upper
            },
            "document_cluster_bootstrap_95_interval": {
                "upper": document_upper,
                "eligible_sequence_fraction_pass": document_coverage_pass,
            },
            "holm_primary_family": {
                "holm_adjusted_seed_t_pvalue": adjusted_pvalue
            },
        },
    }


class Phase3SummaryTests(unittest.TestCase):
    def _manifest(self) -> dict[str, object]:
        return {
            "quick_smoke_only": False,
            "language": "ko",
            "model_spec": MODULE.PHASE3_MODEL_SPEC.to_dict(),
            "optimization_spec": MODULE.PHASE3_OPTIMIZATION_SPEC.to_dict(),
            "limits": MODULE.FULL_LIMITS,
            "global_max_position_embeddings": MODULE.GLOBAL_POSITION_LIMIT,
            "seeds": list(MODULE.INITIAL_SEEDS),
            "policies": [MODULE.F, MODULE.C, MODULE.W],
            "invocations": [
                {
                    "seeds": list(MODULE.INITIAL_SEEDS),
                    "policies": [MODULE.F, MODULE.C, MODULE.W],
                }
            ],
            "streams": {
                "test": {"selected_stream_sha256": "test-hash"},
            },
        }

    def test_design_validation_rejects_cherry_picked_seeds(self) -> None:
        manifest = self._manifest()
        with self.assertRaisesRegex(ValueError, "preregistered"):
            MODULE._validate_requested_design(
                manifest,
                (1729, 2718, 57721),
                (MODULE.F, MODULE.C, MODULE.W),
                test_stream_sha256="test-hash",
            )

    def test_design_validation_accepts_preregistered_initial_design(self) -> None:
        MODULE._validate_requested_design(
            self._manifest(),
            MODULE.INITIAL_SEEDS,
            (MODULE.F, MODULE.C, MODULE.W),
            test_stream_sha256="test-hash",
        )

    def test_design_validation_requires_pair_invocation(self) -> None:
        manifest = self._manifest()
        manifest["invocations"] = [
            {
                "seeds": [MODULE.INITIAL_SEEDS[0]],
                "policies": [MODULE.F, MODULE.C, MODULE.W],
            }
        ]
        with self.assertRaisesRegex(ValueError, "lacks invocation"):
            MODULE._validate_requested_design(
                manifest,
                MODULE.INITIAL_SEEDS,
                (MODULE.F, MODULE.C, MODULE.W),
                test_stream_sha256="test-hash",
            )

    def test_checkpoint_state_hash_detects_tensor_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.pt"
            torch.save({"weight": torch.tensor([1.0, 2.0])}, checkpoint)
            first = MODULE._checkpoint_state_sha256(checkpoint)
            torch.save({"weight": torch.tensor([1.0, 3.0])}, checkpoint)
            second = MODULE._checkpoint_state_sha256(checkpoint)
        self.assertNotEqual(first, second)

    def test_ood_summary_is_bound_to_primary_checkpoint_hashes(self) -> None:
        seeds = (1729,)
        checkpoint_hashes = {
            1729: {
                MODULE.F: "fixed-state",
                MODULE.C: "codepoint-state",
                MODULE.W: "whitespace-state",
            }
        }
        ood = {
            "seeds": [1729],
            "policies": [MODULE.F, MODULE.C, MODULE.W],
            "integrity": {
                "all_integrity_checks_pass": True,
                "checkpoint_state_sha256": {
                    "1729": dict(checkpoint_hashes[1729]),
                },
            },
            "gate_i_ood_guard": {"pass": True},
        }
        self.assertTrue(
            MODULE._validate_ood_summary(ood, seeds, checkpoint_hashes)
        )
        ood["integrity"]["checkpoint_state_sha256"]["1729"][MODULE.W] = (
            "different"
        )
        with self.assertRaisesRegex(ValueError, "checkpoints differ"):
            MODULE._validate_ood_summary(ood, seeds, checkpoint_hashes)

    def test_ood_summary_requires_composite_integrity_flag(self) -> None:
        ood = {
            "seeds": [1729],
            "policies": [MODULE.F, MODULE.C, MODULE.W],
            "integrity": {"all_integrity_checks_pass": False},
            "gate_i_ood_guard": {"pass": True},
        }
        hashes = {
            1729: {MODULE.F: "f", MODULE.C: "c", MODULE.W: "w"}
        }
        with self.assertRaisesRegex(ValueError, "integrity checks"):
            MODULE._validate_ood_summary(ood, (1729,), hashes)

    def test_final_ood_summary_must_share_confirmation_authorization(self) -> None:
        seeds = MODULE.CONFIRMATION_SEEDS
        hashes = {
            seed: {MODULE.F: f"f-{seed}", MODULE.C: f"c-{seed}", MODULE.W: f"w-{seed}"}
            for seed in seeds
        }
        authorization = {
            "authorization_kind": "phase3_corrected_gate_i_confirmation_v1",
            "summary_artifact_sha256": "a" * 64,
        }
        ood = {
            "seeds": list(seeds),
            "policies": [MODULE.F, MODULE.C, MODULE.W],
            "confirmation_authorization": authorization,
            "integrity": {
                "all_integrity_checks_pass": True,
                "checkpoint_state_sha256": {
                    str(seed): dict(hashes[seed]) for seed in seeds
                },
            },
            "gate_i_ood_guard": {"pass": True},
        }
        self.assertTrue(
            MODULE._validate_ood_summary(
                ood,
                seeds,
                hashes,
                confirmation_authorization=authorization,
            )
        )
        with self.assertRaisesRegex(ValueError, "authorization differ"):
            MODULE._validate_ood_summary(
                ood,
                seeds,
                hashes,
                confirmation_authorization={
                    **authorization,
                    "summary_artifact_sha256": "b" * 64,
                },
            )

    def test_gate_i_waits_for_ood_after_quality_pass(self) -> None:
        gate = MODULE.gate_i_summary(
            contrast(-0.003, 2), ood_guard_pass=None
        )
        self.assertTrue(gate["quality_component_pass"])
        self.assertIsNone(gate["overall_pass"])
        self.assertEqual(gate["status"], "pending_ood_guard")

    def test_gate_i_quality_failure_is_final_without_ood(self) -> None:
        gate = MODULE.gate_i_summary(
            contrast(-0.001, 3), ood_guard_pass=None
        )
        self.assertFalse(gate["quality_component_pass"])
        self.assertFalse(gate["overall_pass"])
        self.assertEqual(gate["status"], "fail_quality")

    def test_gate_i_requires_document_cluster_support_and_coverage(self) -> None:
        interval_failure = contrast(-0.003, 3)
        interval_failure["whitespace_minus_codepoint"][
            "document_cluster_bootstrap_95_interval"
        ]["upper"] = 0.0001
        self.assertFalse(
            MODULE.gate_i_summary(
                interval_failure,
                ood_guard_pass=True,
            )["overall_pass"]
        )

        coverage_failure = copy.deepcopy(contrast(-0.003, 3))
        coverage_failure["whitespace_minus_codepoint"][
            "document_cluster_bootstrap_95_interval"
        ]["eligible_sequence_fraction_pass"] = False
        self.assertFalse(
            MODULE.gate_i_summary(
                coverage_failure,
                ood_guard_pass=True,
            )["overall_pass"]
        )

    def test_gate_i_combines_quality_and_ood(self) -> None:
        passed = MODULE.gate_i_summary(
            contrast(-0.003, 3), ood_guard_pass=True
        )
        failed = MODULE.gate_i_summary(
            contrast(-0.003, 3), ood_guard_pass=False
        )
        self.assertTrue(passed["overall_pass"])
        self.assertEqual(passed["status"], "pass")
        self.assertFalse(failed["overall_pass"])
        self.assertEqual(failed["status"], "fail_ood_guard")

    def test_contrasts_require_both_policies(self) -> None:
        available = MODULE._available_contrasts(
            (MODULE.F, MODULE.C, MODULE.W)
        )
        self.assertEqual(
            set(available),
            {
                "whitespace_minus_codepoint",
                "whitespace_minus_fixed",
                "codepoint_minus_fixed",
            },
        )

    def test_gate_j_requires_exactly_five_seeds(self) -> None:
        gate = MODULE.gate_j_summary(
            final_contrasts(), seed_count=3, ood_guard_pass=True
        )
        self.assertIsNone(gate["overall_pass"])
        self.assertIn("exactly_five", gate["status"])

    def test_gate_j_combines_both_primary_contrasts_and_ood(self) -> None:
        passed = MODULE.gate_j_summary(
            final_contrasts(), seed_count=5, ood_guard_pass=True
        )
        self.assertTrue(passed["overall_pass"])

        failed_quality = MODULE.gate_j_summary(
            final_contrasts(fixed_mean=-0.002),
            seed_count=5,
            ood_guard_pass=True,
        )
        self.assertFalse(failed_quality["overall_pass"])
        self.assertEqual(failed_quality["status"], "fail_primary_quality")

        failed_interval = MODULE.gate_j_summary(
            final_contrasts(bootstrap_upper=0.0001),
            seed_count=5,
            ood_guard_pass=True,
        )
        self.assertFalse(failed_interval["overall_pass"])

        failed_document_interval = MODULE.gate_j_summary(
            final_contrasts(document_upper=0.0001),
            seed_count=5,
            ood_guard_pass=True,
        )
        self.assertFalse(failed_document_interval["overall_pass"])

        failed_document_coverage = MODULE.gate_j_summary(
            final_contrasts(document_coverage_pass=False),
            seed_count=5,
            ood_guard_pass=True,
        )
        self.assertFalse(failed_document_coverage["overall_pass"])

        failed_multiplicity = MODULE.gate_j_summary(
            final_contrasts(adjusted_pvalue=0.051),
            seed_count=5,
            ood_guard_pass=True,
        )
        self.assertFalse(failed_multiplicity["overall_pass"])

        failed_ood = MODULE.gate_j_summary(
            final_contrasts(), seed_count=5, ood_guard_pass=False
        )
        self.assertFalse(failed_ood["overall_pass"])
        self.assertEqual(failed_ood["status"], "fail_ood_guard")


if __name__ == "__main__":
    unittest.main()
