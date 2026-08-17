import copy
import importlib.util
from pathlib import Path
import unittest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_phase3_inference_quality.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_phase3_inference_quality",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Phase3InferenceQualitySummaryTests(unittest.TestCase):
    def _summaries(self, reference_family: str = "phase3") -> tuple[dict, dict, dict]:
        rate = 64
        candidate = "causal_whitespace_grid_64"
        reference = (
            "causal_codepoint_grid_64"
            if reference_family == "compute_conversion"
            else "causal_codepoint_grid"
        )
        selection = {
            "selection_uses_latency": False,
            "status": "locked_before_latency_pending_five_seed_quality",
            "seed_order": [1729, 2718, 31415],
            "candidate": {"policy": candidate, "patch_count": rate},
            "reference": {
                "policy": reference,
                "model_family": reference_family,
            },
        }
        phase3 = {
            "seeds": list(MODULE.SEEDS),
            "policies": ["causal_codepoint_grid"],
            "integrity": {"all_integrity_checks_pass": True},
            "gate_j": {"overall_pass": True},
        }
        conversion = {
            "stage": "confirmation",
            "seeds": list(MODULE.SEEDS),
            "policies": [
                "causal_codepoint_grid_64",
                "causal_whitespace_grid_64",
            ],
            "calibration_rate_selection": {"selected_rate": rate},
            "integrity": {"all_integrity_checks_pass": True},
            "initial_conversion_gate": {"overall_pass": True},
            "confirmation_same_rate_gate": {"overall_pass": True},
        }
        return selection, phase3, conversion

    def test_quality_summary_accepts_locked_phase3_reference(self) -> None:
        values = self._summaries()
        candidate, reference, family = MODULE._validate_summaries(*values)
        self.assertEqual(candidate, "causal_whitespace_grid_64")
        self.assertEqual(reference, "causal_codepoint_grid")
        self.assertEqual(family, "phase3")

    def test_quality_summary_accepts_selected_same_rate_reference(self) -> None:
        values = self._summaries("compute_conversion")
        _, reference, family = MODULE._validate_summaries(*values)
        self.assertEqual(reference, "causal_codepoint_grid_64")
        self.assertEqual(family, "compute_conversion")

    def test_quality_summary_rejects_latency_informed_selection(self) -> None:
        selection, phase3, conversion = self._summaries()
        selection["selection_uses_latency"] = True
        with self.assertRaisesRegex(ValueError, "before timing"):
            MODULE._validate_summaries(selection, phase3, conversion)

    def test_final_summaries_must_preserve_locked_initial_evidence(self) -> None:
        selection, _, _ = self._summaries()
        selection["reference"]["policy"] = "entropy_threshold_full"
        selection["candidate"]["policy"] = "causal_whitespace_grid_64"
        selection["candidate"]["patch_count"] = 64
        phase3_fields = {
            field: {
                policy: f"{field}-{policy}"
                for policy in (
                    *MODULE.PRIMARY_PHASE3_POLICIES,
                    "entropy_threshold_full",
                )
            }
            for field in MODULE.ARTIFACT_EVIDENCE_FIELDS
        }
        locked_phase3 = {
            "run_manifest": {
                key: f"shared-{key}"
                for key in (
                    "source_artifact",
                    "source_integrity_artifact",
                    "model_spec",
                    "optimization_spec",
                    "limits",
                    "streams",
                )
            },
            "integrity": {
                "by_seed": {
                    str(seed): {
                        **phase3_fields,
                        "router_and_threshold_cache": {"hash": f"router-{seed}"},
                    }
                    for seed in MODULE.INITIAL_SEEDS
                }
            },
        }
        conversion_artifacts = {
            policy: {"hash": policy}
            for policy in (
                "causal_whitespace_grid_64",
                "causal_codepoint_grid_64",
            )
        }
        locked_conversion = {
            "integrity": {
                "source_context": {"stream": "shared"},
                "by_seed": {
                    str(seed): {
                        "conversion_artifacts": conversion_artifacts,
                    }
                    for seed in MODULE.INITIAL_SEEDS
                },
            }
        }
        final_phase3 = {
            "run_manifest": locked_phase3["run_manifest"],
            "integrity": {
                "by_seed": copy.deepcopy(
                    locked_phase3["integrity"]["by_seed"]
                )
            },
        }
        final_conversion = {
            "integrity": locked_conversion["integrity"],
        }
        MODULE._validate_locked_initial_evidence(
            selection,
            locked_phase3,
            locked_conversion,
            final_phase3,
            final_conversion,
        )

        final_phase3["integrity"]["by_seed"]["1729"] = {
            **final_phase3["integrity"]["by_seed"]["1729"],
            "checkpoint_state_sha256": {
                **phase3_fields["checkpoint_state_sha256"],
                "entropy_threshold_full": "changed",
            },
        }
        with self.assertRaisesRegex(ValueError, "locked Phase 3 evidence changed"):
            MODULE._validate_locked_initial_evidence(
                selection,
                locked_phase3,
                locked_conversion,
                final_phase3,
                final_conversion,
            )

    def test_locked_conversion_evidence_change_is_rejected(self) -> None:
        selection, _, _ = self._summaries("compute_conversion")
        phase3 = {
            "run_manifest": {
                key: f"shared-{key}"
                for key in (
                    "source_artifact",
                    "source_integrity_artifact",
                    "model_spec",
                    "optimization_spec",
                    "limits",
                    "streams",
                )
            },
            "integrity": {
                "by_seed": {
                    str(seed): {
                        field: {
                            policy: f"{field}-{policy}"
                            for policy in MODULE.PRIMARY_PHASE3_POLICIES
                        }
                        for field in MODULE.ARTIFACT_EVIDENCE_FIELDS
                    }
                    for seed in MODULE.INITIAL_SEEDS
                }
            },
        }
        conversion = {
            "integrity": {
                "source_context": {"stream": "shared"},
                "by_seed": {
                    str(seed): {
                        "conversion_artifacts": {
                            "causal_whitespace_grid_64": {"hash": "w"},
                            "causal_codepoint_grid_64": {"hash": "c"},
                        }
                    }
                    for seed in MODULE.INITIAL_SEEDS
                },
            }
        }
        changed = {
            "integrity": {
                **conversion["integrity"],
                "by_seed": {
                    **conversion["integrity"]["by_seed"],
                    "1729": {
                        "conversion_artifacts": {
                            "causal_whitespace_grid_64": {"hash": "changed"},
                            "causal_codepoint_grid_64": {"hash": "c"},
                        }
                    },
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "locked conversion evidence changed"):
            MODULE._validate_locked_initial_evidence(
                selection,
                phase3,
                conversion,
                phase3,
                changed,
            )


if __name__ == "__main__":
    unittest.main()
