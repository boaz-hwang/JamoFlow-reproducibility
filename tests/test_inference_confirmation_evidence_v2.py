from __future__ import annotations

import copy
import json
import unittest

from jamoflow.inference_confirmation_evidence_v2 import (
    CALIBRATION_SEQUENCE_COUNT,
    CALIBRATION_TARGETS_PER_SEQUENCE,
    COMPUTE_CONFIRMATION_COMPLETION_PATH,
    CONFIRMATION_SEEDS,
    PHASE3_REFERENCE_COMPLETION_PATH,
    build_confirmation_training_completion,
    build_confirmation_calibration_receipt,
    build_confirmation_evidence_manifest,
    expected_confirmation_paths,
    required_confirmation_models,
    validate_confirmation_calibration_receipt,
    validate_confirmation_evidence_manifest,
    validate_receipts_against_training_completions,
    validate_confirmation_training_completion,
    validate_training_report_against_completion,
)
from jamoflow.inference_final_authorization_v2 import (
    canonical_sha256,
    expected_model_paths,
    expected_router_paths,
)
from tests.test_inference_final_authorization_v2 import (
    digest,
    selection_lock_fixture,
)
from tests.test_inference_selection_plan import plan_fixture


class InferenceConfirmationEvidenceV2Tests(unittest.TestCase):
    def _compute_completion(self, lock: dict) -> dict:
        policies = tuple(
            lock["decision"]["confirmation_plan"]["compute_conversion"][
                "policies"
            ]
        )
        descriptors = {
            model["descriptor"]["policy"]: model["descriptor"]
            for model in required_confirmation_models(lock)
        }
        units = {}
        for seed in CONFIRMATION_SEEDS:
            units[seed] = {}
            for policy in policies:
                descriptor = descriptors[policy]
                paths = expected_model_paths(descriptor, seed)
                auxiliary = {"kind": "none"}
                if descriptor["requires_entropy_router"]:
                    router = expected_router_paths(seed)
                    auxiliary = {
                        "kind": "entropy_router_artifacts",
                        "router_checkpoint_artifact_sha256": digest(
                            f"router-checkpoint/{seed}"
                        ),
                        "router_checkpoint_path": router["router_checkpoint"],
                        "router_checkpoint_state_sha256": digest(
                            f"router-state/{seed}"
                        ),
                        "router_report_artifact_sha256": digest(
                            f"router-report/{seed}"
                        ),
                        "router_report_path": router["router_report"],
                        "threshold_cache_artifact_sha256": digest(
                            f"threshold-cache/{seed}"
                        ),
                        "threshold_cache_path": router["threshold_cache"],
                        "threshold_diagnostics_artifact_sha256": digest(
                            f"threshold-diagnostics/{seed}"
                        ),
                        "threshold_diagnostics_path": router[
                            "threshold_diagnostics"
                        ],
                    }
                units[seed][policy] = {
                    "auxiliary": auxiliary,
                    "checkpoint_artifact_sha256": digest(
                        f"completion-checkpoint/{seed}/{policy}"
                    ),
                    "checkpoint_path": paths["checkpoint"],
                    "checkpoint_state_sha256": digest(
                        f"completion-state/{seed}/{policy}"
                    ),
                    "training_report_artifact_sha256": digest(
                        f"completion-report/{seed}/{policy}"
                    ),
                    "training_report_path": paths["training_report"],
                }
        return build_confirmation_training_completion(
            selection_lock=lock,
            selection_lock_artifact_sha256=digest("selection-artifact"),
            family="compute_conversion",
            run_git_commit="a" * 40,
            run_manifest={
                "artifact_sha256": digest("conversion-run-manifest"),
                "path": "runs/phase3-compute-conversion/manifest.json",
            },
            implementation_manifest_sha256=digest("implementation-manifest"),
            environment_sha256=digest("environment"),
            units=units,
        )

    def _phase3_completion(self, lock: dict) -> dict:
        policies = tuple(
            lock["decision"]["confirmation_plan"]["phase3_reference"][
                "policies"
            ]
        )
        descriptors = {
            model["descriptor"]["policy"]: model["descriptor"]
            for model in required_confirmation_models(lock)
        }
        units = {}
        for seed in CONFIRMATION_SEEDS:
            units[seed] = {}
            for policy in policies:
                descriptor = descriptors[policy]
                paths = expected_model_paths(descriptor, seed)
                auxiliary = {"kind": "none"}
                if descriptor["requires_entropy_router"]:
                    router = expected_router_paths(seed)
                    auxiliary = {
                        "kind": "entropy_router_artifacts",
                        "router_checkpoint_artifact_sha256": digest(
                            f"router-checkpoint/{seed}"
                        ),
                        "router_checkpoint_path": router["router_checkpoint"],
                        "router_checkpoint_state_sha256": digest(
                            f"router-state/{seed}"
                        ),
                        "router_report_artifact_sha256": digest(
                            f"router-report/{seed}"
                        ),
                        "router_report_path": router["router_report"],
                        "threshold_cache_artifact_sha256": digest(
                            f"threshold-cache/{seed}"
                        ),
                        "threshold_cache_path": router["threshold_cache"],
                        "threshold_diagnostics_artifact_sha256": digest(
                            f"threshold-diagnostics/{seed}"
                        ),
                        "threshold_diagnostics_path": router[
                            "threshold_diagnostics"
                        ],
                    }
                units[seed][policy] = {
                    "auxiliary": auxiliary,
                    "checkpoint_artifact_sha256": digest(
                        f"phase3-completion-checkpoint/{seed}/{policy}"
                    ),
                    "checkpoint_path": paths["checkpoint"],
                    "checkpoint_state_sha256": digest(
                        f"phase3-completion-state/{seed}/{policy}"
                    ),
                    "training_report_artifact_sha256": digest(
                        f"phase3-completion-report/{seed}/{policy}"
                    ),
                    "training_report_path": paths["training_report"],
                }
        return build_confirmation_training_completion(
            selection_lock=lock,
            selection_lock_artifact_sha256=digest("selection-artifact"),
            family="phase3_reference",
            run_git_commit="e" * 40,
            run_manifest={
                "artifact_sha256": digest("phase3-run-manifest"),
                "path": "runs/phase3/manifest.json",
            },
            implementation_manifest_sha256=digest("implementation-manifest"),
            environment_sha256=digest("environment"),
            units=units,
        )

    def _receipt(
        self,
        lock: dict,
        model: dict,
        seed: int,
        *,
        selection_artifact: str,
        evaluator_commit: str,
        stream_sha256: str,
    ) -> dict:
        descriptor = model["descriptor"]
        model_paths = expected_model_paths(descriptor, seed)
        evidence_paths = expected_confirmation_paths(
            model["artifact_role"], seed
        )
        return build_confirmation_calibration_receipt(
            selection_lock=lock,
            selection_lock_artifact_sha256=selection_artifact,
            artifact_role=model["artifact_role"],
            descriptor=descriptor,
            seed=seed,
            evaluator_git_commit=evaluator_commit,
            training_report={
                "artifact_sha256": digest(
                    f"confirmation-report/{model['artifact_role']}/{seed}"
                ),
                "path": model_paths["training_report"],
            },
            checkpoint={
                "artifact_sha256": digest(
                    f"confirmation-checkpoint/{model['artifact_role']}/{seed}"
                ),
                "path": model_paths["checkpoint"],
                "state_sha256": digest(
                    f"confirmation-state/{model['artifact_role']}/{seed}"
                ),
            },
            auxiliary={"kind": "none"},
            calibration={
                "boundaries_sha256": digest("calibration-boundaries"),
                "bpb": 1.5 + seed / 1_000_000,
                "count": CALIBRATION_SEQUENCE_COUNT,
                "dtype": "float32",
                "inputs_sha256": digest("calibration-inputs"),
                "matrix_sha256": digest(
                    f"matrix/{model['artifact_role']}/{seed}"
                ),
                "nll_array_sha256": digest(
                    f"nll-array/{model['artifact_role']}/{seed}"
                ),
                "nll_artifact_path": evidence_paths["nll"],
                "nll_artifact_sha256": digest(
                    f"nll-artifact/{model['artifact_role']}/{seed}"
                ),
                "predicted_bytes": (
                    CALIBRATION_SEQUENCE_COUNT
                    * CALIBRATION_TARGETS_PER_SEQUENCE
                ),
                "stream_sha256": stream_sha256,
            },
        )

    def _manifest(self, *, broad_futile: bool = True) -> tuple[dict, dict, dict]:
        plan = plan_fixture()
        lock = selection_lock_fixture(broad_futile=broad_futile)
        selection_artifact = digest("selection-artifact")
        evaluator_commit = "b" * 40
        models = required_confirmation_models(lock)
        receipts = {
            seed: {
                model["artifact_role"]: self._receipt(
                    lock,
                    model,
                    seed,
                    selection_artifact=selection_artifact,
                    evaluator_commit=evaluator_commit,
                    stream_sha256=plan["calibration_evaluator"][
                        "input_stream_sha256"
                    ],
                )
                for model in models
            }
            for seed in CONFIRMATION_SEEDS
        }
        training_completions = {
            "compute_conversion": {
                "artifact": {
                    "git_commit": "c" * 40,
                    "path": COMPUTE_CONFIRMATION_COMPLETION_PATH.as_posix(),
                    "sha256": digest("compute-completion-artifact"),
                },
                "completion_sha256": digest("compute-completion"),
                "run_git_commit": "d" * 40,
            }
        }
        if isinstance(
            lock["decision"]["confirmation_plan"].get("phase3_reference"),
            dict,
        ):
            training_completions["phase3_reference"] = {
                "artifact": {
                    "git_commit": "e" * 40,
                    "path": PHASE3_REFERENCE_COMPLETION_PATH.as_posix(),
                    "sha256": digest("phase3-completion-artifact"),
                },
                "completion_sha256": digest("phase3-completion"),
                "run_git_commit": "f" * 40,
            }
        manifest = build_confirmation_evidence_manifest(
            plan=plan,
            plan_artifact_sha256=lock["plan_sha256"],
            initial_calibration_evidence_artifact_sha256=lock[
                "calibration_evidence_manifest_sha256"
            ],
            initial_calibration_evidence_payload_sha256=digest(
                "initial-calibration-manifest-payload"
            ),
            selection_lock=lock,
            selection_lock_artifact_sha256=selection_artifact,
            evaluator_git_commit=evaluator_commit,
            training_completions=training_completions,
            receipts=receipts,
        )
        return plan, lock, manifest

    def test_manifest_round_trip_has_no_test_or_latency_authority(self) -> None:
        plan, lock, manifest = self._manifest()
        validate_confirmation_evidence_manifest(
            manifest,
            plan=plan,
            selection_lock=lock,
        )
        encoded = repr(manifest).lower()
        self.assertNotIn("test_bpb", encoded)
        self.assertNotIn("test-nll", encoded)
        self.assertNotIn("latency", encoded)
        self.assertEqual(len(manifest["model_artifact_role_order"]), 3)

    def test_receipt_rejects_seed_role_and_lock_rotation(self) -> None:
        plan, lock, manifest = self._manifest()
        receipt = manifest["receipts"][str(CONFIRMATION_SEEDS[0])][
            "candidate"
        ]
        validate_confirmation_calibration_receipt(
            receipt,
            selection_lock=lock,
        )
        tampered = copy.deepcopy(receipt)
        tampered["selection_lock_payload_sha256"] = digest("other-lock")
        with self.assertRaises(ValueError):
            validate_confirmation_calibration_receipt(
                tampered,
                selection_lock=lock,
            )
        with self.assertRaisesRegex(ValueError, "role differs"):
            build_confirmation_calibration_receipt(
                selection_lock=lock,
                selection_lock_artifact_sha256=receipt[
                    "selection_lock_artifact_sha256"
                ],
                artifact_role="broad_reference",
                descriptor=receipt["descriptor"],
                seed=receipt["seed"],
                evaluator_git_commit=receipt["evaluator_git_commit"],
                training_report=receipt["training_report"],
                checkpoint=receipt["checkpoint"],
                auxiliary=receipt["auxiliary"],
                calibration=receipt["calibration"],
            )

    def test_manifest_rejects_missing_model_and_wrong_calibration_stream(self) -> None:
        plan, lock, manifest = self._manifest()
        receipts = {
            seed: copy.deepcopy(manifest["receipts"][str(seed)])
            for seed in CONFIRMATION_SEEDS
        }
        receipts[CONFIRMATION_SEEDS[0]].pop("same_rate_codepoint_control")
        with self.assertRaisesRegex(ValueError, "role set"):
            build_confirmation_evidence_manifest(
                plan=plan,
                plan_artifact_sha256=manifest["plan_artifact_sha256"],
                initial_calibration_evidence_artifact_sha256=manifest[
                    "initial_calibration_evidence_artifact_sha256"
                ],
                initial_calibration_evidence_payload_sha256=manifest[
                    "initial_calibration_evidence_payload_sha256"
                ],
                selection_lock=lock,
                selection_lock_artifact_sha256=manifest[
                    "selection_lock_artifact_sha256"
                ],
                evaluator_git_commit=manifest["evaluator_git_commit"],
                training_completions=manifest["training_completions"],
                receipts=receipts,
            )

        _, _, manifest = self._manifest()
        receipts = {
            seed: copy.deepcopy(manifest["receipts"][str(seed)])
            for seed in CONFIRMATION_SEEDS
        }
        receipt = receipts[CONFIRMATION_SEEDS[0]]["candidate"]
        receipt["calibration"]["stream_sha256"] = digest("wrong-stream")
        receipt["receipt_sha256"] = digest("rehashed-placeholder")
        with self.assertRaises(ValueError):
            build_confirmation_evidence_manifest(
                plan=plan,
                plan_artifact_sha256=manifest["plan_artifact_sha256"],
                initial_calibration_evidence_artifact_sha256=manifest[
                    "initial_calibration_evidence_artifact_sha256"
                ],
                initial_calibration_evidence_payload_sha256=manifest[
                    "initial_calibration_evidence_payload_sha256"
                ],
                selection_lock=lock,
                selection_lock_artifact_sha256=manifest[
                    "selection_lock_artifact_sha256"
                ],
                evaluator_git_commit=manifest["evaluator_git_commit"],
                training_completions=manifest["training_completions"],
                receipts=receipts,
            )

    def test_eligible_broad_model_is_mandatory(self) -> None:
        _, lock, manifest = self._manifest(broad_futile=False)
        self.assertEqual(len(required_confirmation_models(lock)), 4)
        self.assertIn("broad_reference", manifest["model_artifact_role_order"])

    def test_training_completion_survives_sorted_json_and_rejects_rotation(self) -> None:
        lock = selection_lock_fixture()
        completion = self._compute_completion(lock)
        round_tripped = json.loads(json.dumps(completion, sort_keys=True))
        validate_confirmation_training_completion(
            round_tripped,
            selection_lock=lock,
        )
        tampered = copy.deepcopy(round_tripped)
        policy = tampered["policy_order"][0]
        tampered["units"][str(CONFIRMATION_SEEDS[0])][policy][
            "checkpoint_path"
        ] = "artifacts/rotated.pt"
        unsigned = {
            key: value
            for key, value in tampered.items()
            if key != "completion_sha256"
        }
        tampered["completion_sha256"] = canonical_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "unit is malformed"):
            validate_confirmation_training_completion(
                tampered,
                selection_lock=lock,
            )

    def test_training_report_run_and_historical_gate_are_completion_bound(self) -> None:
        lock = selection_lock_fixture()
        completion = self._compute_completion(lock)
        seed = CONFIRMATION_SEEDS[0]
        policy = completion["policy_order"][0]
        historical = digest("historical-primary")
        binding_payload = {
            "device": "mps",
            "git_commit": completion["run_git_commit"],
            "git_worktree_clean_at_start": True,
            "policies": completion["policy_order"],
            "primary_summary_sha256": historical,
            "schema_version": 1,
            "seeds": completion["seed_order"],
            "selection_plan_sha256": lock["plan_sha256"],
            "selection_summary_sha256": completion[
                "selection_lock_artifact_sha256"
            ],
            "stage": "confirmation",
        }
        report = {
            "evidence_binding": {
                **binding_payload,
                "identity_sha256": canonical_sha256(binding_payload),
            },
            "policy": policy,
            "seed": seed,
        }
        validate_training_report_against_completion(
            completion=completion,
            report=report,
            seed=seed,
            policy=policy,
            selection_lock=lock,
            historical_primary_summary_sha256=historical,
        )
        for key, value in (
            ("git_commit", "f" * 40),
            ("primary_summary_sha256", digest("rotated-primary")),
        ):
            tampered = copy.deepcopy(report)
            tampered["evidence_binding"][key] = value
            unsigned = {
                name: item
                for name, item in tampered["evidence_binding"].items()
                if name != "identity_sha256"
            }
            tampered["evidence_binding"]["identity_sha256"] = canonical_sha256(
                unsigned
            )
            with self.assertRaisesRegex(ValueError, "binding differs"):
                validate_training_report_against_completion(
                    completion=completion,
                    report=tampered,
                    seed=seed,
                    policy=policy,
                    selection_lock=lock,
                    historical_primary_summary_sha256=historical,
                )

    def test_phase3_report_run_commit_is_completion_bound(self) -> None:
        lock = selection_lock_fixture(broad_futile=False)
        completion = self._phase3_completion(lock)
        seed = CONFIRMATION_SEEDS[0]
        policy = completion["policy_order"][0]
        report = {
            "evidence_binding": {
                "authorization": {"kind": "locked-reference"},
                "device": "mps",
                "git_worktree_clean_at_start": True,
                "kind": "selected_phase3_reference_training_evidence_v4",
                "run_git_commit": completion["run_git_commit"],
                "schema_version": 4,
            },
            "policy": policy,
            "seed": seed,
        }
        validate_training_report_against_completion(
            completion=completion,
            report=report,
            seed=seed,
            policy=policy,
            selection_lock=lock,
            historical_primary_summary_sha256=digest("historical-primary"),
        )
        report["evidence_binding"]["run_git_commit"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "binding differs"):
            validate_training_report_against_completion(
                completion=completion,
                report=report,
                seed=seed,
                policy=policy,
                selection_lock=lock,
                historical_primary_summary_sha256=digest("historical-primary"),
            )

    def test_entropy_completion_round_trip_and_router_rotation(self) -> None:
        lock = selection_lock_fixture(
            broad_futile=False,
            broad_policy="entropy_threshold_full",
        )
        completion = self._phase3_completion(lock)
        round_tripped = json.loads(json.dumps(completion, sort_keys=True))
        validate_confirmation_training_completion(
            round_tripped,
            selection_lock=lock,
        )
        policy = round_tripped["policy_order"][0]
        auxiliary = round_tripped["units"][str(CONFIRMATION_SEEDS[0])][
            policy
        ]["auxiliary"]
        auxiliary["threshold_cache_path"] = "artifacts/rotated-cache.npz"
        unsigned = {
            key: value
            for key, value in round_tripped.items()
            if key != "completion_sha256"
        }
        round_tripped["completion_sha256"] = canonical_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "unit is malformed"):
            validate_confirmation_training_completion(
                round_tripped,
                selection_lock=lock,
            )

    def test_training_completion_must_match_replayed_receipts(self) -> None:
        _, lock, manifest = self._manifest()
        models = required_confirmation_models(lock)
        role_by_policy = {
            model["descriptor"]["policy"]: model["artifact_role"]
            for model in models
        }
        policies = tuple(
            lock["decision"]["confirmation_plan"]["compute_conversion"][
                "policies"
            ]
        )
        units = {}
        for seed in CONFIRMATION_SEEDS:
            units[seed] = {}
            for policy in policies:
                receipt = manifest["receipts"][str(seed)][role_by_policy[policy]]
                units[seed][policy] = {
                    "auxiliary": {"kind": "none"},
                    "checkpoint_artifact_sha256": receipt["checkpoint"][
                        "artifact_sha256"
                    ],
                    "checkpoint_path": receipt["checkpoint"]["path"],
                    "checkpoint_state_sha256": receipt["checkpoint"][
                        "state_sha256"
                    ],
                    "training_report_artifact_sha256": receipt[
                        "training_report"
                    ]["artifact_sha256"],
                    "training_report_path": receipt["training_report"]["path"],
                }
        completion = build_confirmation_training_completion(
            selection_lock=lock,
            selection_lock_artifact_sha256=manifest[
                "selection_lock_artifact_sha256"
            ],
            family="compute_conversion",
            run_git_commit="a" * 40,
            run_manifest={
                "artifact_sha256": digest("conversion-run-manifest"),
                "path": "runs/phase3-compute-conversion/manifest.json",
            },
            implementation_manifest_sha256=digest("implementation-manifest"),
            environment_sha256=digest("environment"),
            units=units,
        )
        validate_receipts_against_training_completions(
            selection_lock=lock,
            receipts=manifest["receipts"],
            completions={"compute_conversion": completion},
        )
        tampered = copy.deepcopy(completion)
        policy = policies[0]
        tampered["units"][str(CONFIRMATION_SEEDS[0])][policy][
            "checkpoint_state_sha256"
        ] = digest("rotated-state")
        unsigned = {
            key: value
            for key, value in tampered.items()
            if key != "completion_sha256"
        }
        tampered["completion_sha256"] = canonical_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "replay differs"):
            validate_receipts_against_training_completions(
                selection_lock=lock,
                receipts=manifest["receipts"],
                completions={"compute_conversion": tampered},
            )


if __name__ == "__main__":
    unittest.main()
