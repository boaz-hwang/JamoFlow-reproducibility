from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from jamoflow.inference_confirmation_evidence_v2 import (
    COMPUTE_CONFIRMATION_COMPLETION_PATH,
    PHASE3_REFERENCE_COMPLETION_PATH,
)

from jamoflow.inference_final_authorization_v2 import (
    CONFIRMATION_EVIDENCE_PATH,
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_SEEDS,
    FINAL_TEST_MANIFEST_PATH,
    FINAL_TEST_OUTPUT_PATH,
    FINAL_TEST_SEAL_PATH,
    HISTORICAL_PRIMARY_SUMMARY_PATH,
    IMPLEMENTATION_FILE_ORDER,
    EVALUATION_PACKAGE_FILE_ORDER,
    SELECTION_EVIDENCE_PATH,
    SELECTION_LOCK_PATH,
    SELECTION_PLAN_PATH,
    build_final_evaluation_authorization_v2,
    build_final_model_identity,
    canonical_sha256,
    validate_final_evaluation_authorization_v2,
    validate_final_model_identity,
)
from jamoflow.inference_final_quality_v2 import resolve_final_evaluation_roles
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
    build_selection_decision_v2,
    build_independent_calibration_recomputation_v2,
    build_selection_lock_v2,
)
from jamoflow.phase3 import PHASE3_OPTIMIZATION_SPEC
from jamoflow.phase3 import PHASE3_MODEL_SPEC
from jamoflow.publication_reference import entropy_policy_definition_sha256


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def selection_lock_fixture(
    *,
    broad_futile: bool = True,
    broad_policy: str = "spacebyte_spacelike",
) -> dict:
    values = {
        seed: {policy: 1.5 for policy in CALIBRATION_POLICY_ORDER}
        for seed in INITIAL_SEEDS
    }
    for seed in INITIAL_SEEDS:
        values[seed]["causal_codepoint_grid"] = 1.4
        values[seed]["causal_codepoint_grid_64"] = 1.4
        values[seed]["causal_whitespace_grid_64"] = 1.405
        values[seed][broad_policy] = 1.2 if broad_futile else 1.399
    decision = build_selection_decision_v2(values)
    replay = build_independent_calibration_recomputation_v2(
        values,
        nll_array_sha256_by_seed_policy={
            seed: {policy: "4" * 64 for policy in CALIBRATION_POLICY_ORDER}
            for seed in INITIAL_SEEDS
        },
        evaluator_git_commit="a" * 40,
        verification_git_commit="b" * 40,
        environment_sha256="c" * 64,
        implementation_manifest_sha256="d" * 64,
    )
    return build_selection_lock_v2(
        decision,
        plan_sha256="1" * 64,
        calibration_evidence_manifest_sha256="2" * 64,
        final_test_seal_sha256="3" * 64,
        initial_model_identity_lock_sha256="4" * 64,
        independent_calibration_recomputation=replay,
    )


def model_fixture(model: dict) -> dict:
    descriptor = {
        key: model[key]
        for key in (
            "model_family",
            "patch_count",
            "policy",
            "requires_entropy_router",
            "runtime_policy",
        )
    }
    seed_evidence = {}
    root = "phase3" if descriptor["model_family"] == "phase3" else "phase3-compute-conversion"
    for seed in FINAL_SEEDS:
        policy = descriptor["policy"]
        auxiliary = {"kind": "none"}
        if descriptor["requires_entropy_router"]:
            auxiliary = {
                "calibration_stream_sha256": digest("calibration-stream"),
                "candidate_mask": (
                    "none"
                    if policy == "entropy_threshold_full"
                    else "codepoint"
                ),
                "kind": "entropy_router",
                "maximum_patch_length": 24,
                "policy": policy,
                "policy_definition_sha256": entropy_policy_definition_sha256(
                    policy
                ),
                "router_checkpoint_artifact_sha256": digest(
                    f"router-checkpoint-artifact/{seed}"
                ),
                "router_checkpoint_path": (
                    f"artifacts/phase3/seed-{seed}/router.pt"
                ),
                "router_checkpoint_state_sha256": digest(
                    f"router-checkpoint-state/{seed}"
                ),
                "router_config_sha256": canonical_sha256(
                    PHASE3_MODEL_SPEC.to_dict()
                ),
                "router_parameter_count": 2_016_960,
                "router_report_artifact_sha256": digest(
                    f"router-report/{seed}"
                ),
                "router_report_path": f"runs/phase3/seed-{seed}/router.json",
                "router_training_stream_sha256": digest("router-train-stream"),
                "seed": seed,
                "threshold_cache_artifact_sha256": digest(
                    f"threshold-cache/{seed}"
                ),
                "threshold_cache_path": (
                    f"artifacts/phase3/seed-{seed}/threshold-patches.npz"
                ),
                "threshold_diagnostics_artifact_sha256": digest(
                    f"threshold-diagnostics/{seed}"
                ),
                "threshold_diagnostics_path": (
                    f"runs/phase3/seed-{seed}/threshold-patch-diagnostics.json"
                ),
                "threshold_nats": 1.25 + seed / 1_000_000,
            }
        seed_evidence[seed] = {
            "auxiliary": auxiliary,
            "checkpoint": {
                "artifact_sha256": digest(
                    f"checkpoint-artifact/{model['artifact_role']}/{seed}"
                ),
                "path": f"artifacts/{root}/seed-{seed}/{policy}.pt",
                "state_sha256": digest(
                    f"checkpoint-state/{model['artifact_role']}/{seed}"
                ),
            },
            "seed": seed,
            "training": {
                "evidence_binding_sha256": digest(
                    f"training-binding/{model['artifact_role']}/{seed}"
                ),
                "global_max_position_embeddings": 1_032,
                "initialization_sha256": digest(
                    f"initialization/{model['artifact_role']}/{seed}"
                ),
                "optimization_spec_sha256": canonical_sha256(
                    PHASE3_OPTIMIZATION_SPEC.to_dict()
                ),
                "run_manifest_artifact_sha256": digest(
                    f"run-manifest/{model['artifact_role']}"
                ),
                "source_artifact_sha256": digest("source-artifact"),
                "source_integrity_artifact_sha256": digest(
                    "source-integrity"
                ),
                "steps": 7_813,
                "train_examples": 250_000,
                "train_patch_matrix_sha256": digest(
                    f"train-matrix/{model['artifact_role']}/"
                    + (str(seed) if descriptor["requires_entropy_router"] else "common")
                ),
                "train_predicted_bytes": 127_750_000,
                "train_stream_sha256": digest("train-stream"),
                "training_order_sha256": digest(
                    f"training-order/{model['artifact_role']}/{seed}"
                ),
            },
            "training_report": {
                "artifact_sha256": digest(
                    f"training-report/{model['artifact_role']}/{seed}"
                ),
                "path": f"runs/{root}/seed-{seed}/{policy}.json",
            },
        }
    return build_final_model_identity(
        artifact_role=model["artifact_role"],
        descriptor=descriptor,
        seed_evidence=seed_evidence,
        parameter_count=FINAL_MAIN_PARAMETER_COUNT,
    )


class InferenceFinalAuthorizationV2Tests(unittest.TestCase):
    def _authorization(self, *, broad_futile: bool = True) -> tuple[dict, dict]:
        lock = selection_lock_fixture(broad_futile=broad_futile)
        roles = resolve_final_evaluation_roles(lock)
        models = [model_fixture(model) for model in roles["unique_models"]]
        selection_artifact = digest("selection-lock-artifact")
        recomputation = {
            "comparison": "bitwise_float32_nll_hash_equal",
            "device": "mps",
            "model_artifact_role_order": [
                model["artifact_role"] for model in roles["unique_models"]
            ],
            "receipt_count": len(roles["unique_models"]) * 2,
            "replay_by_seed_role": {
                str(seed): {
                    model["artifact_role"]: {
                        "checkpoint_state_sha256": digest(
                            f"checkpoint-state/{model['artifact_role']}/{seed}"
                        ),
                        "matrix_sha256": digest(
                            f"replay-matrix/{seed}/{model['artifact_role']}"
                        ),
                        "nll_array_sha256": digest(
                            f"replay-nll/{seed}/{model['artifact_role']}"
                        ),
                        "receipt_sha256": digest(
                            f"replay-receipt/{seed}/{model['artifact_role']}"
                        ),
                    }
                    for model in roles["unique_models"]
                }
                for seed in FINAL_SEEDS[3:]
            },
            "seed_order": list(FINAL_SEEDS[3:]),
            "status": "pass",
            "verification_git_commit": "a" * 40,
        }
        recomputation["recomputation_sha256"] = canonical_sha256(
            recomputation
        )
        matched = next(
            model
            for model in models
            if model["artifact_role"] == "matched_efficiency_baseline"
        )
        historical_phase3 = {
            "artifact": {
                "git_commit": "6" * 40,
                "path": HISTORICAL_PRIMARY_SUMMARY_PATH,
                "sha256": digest("historical-primary-five-seed"),
            },
            "by_seed_policy": {
                str(seed): {
                    policy: (
                        {
                            "checkpoint_artifact_sha256": matched["seeds"][
                                str(seed)
                            ]["checkpoint"]["artifact_sha256"],
                            "checkpoint_state_sha256": matched["seeds"][str(seed)][
                                "checkpoint"
                            ]["state_sha256"],
                            "training_report_artifact_sha256": matched["seeds"][
                                str(seed)
                            ]["training_report"]["artifact_sha256"],
                        }
                        if policy == "causal_codepoint_grid"
                        else {
                            "checkpoint_artifact_sha256": digest(
                                f"historical-checkpoint/{seed}/{policy}"
                            ),
                            "checkpoint_state_sha256": digest(
                                f"historical-state/{seed}/{policy}"
                            ),
                            "training_report_artifact_sha256": digest(
                                f"historical-report/{seed}/{policy}"
                            ),
                        }
                    )
                    for policy in (
                        "fixed_byte_6",
                        "causal_codepoint_grid",
                        "causal_whitespace_grid",
                    )
                }
                for seed in FINAL_SEEDS[3:]
            },
            "policy_order": [
                "fixed_byte_6",
                "causal_codepoint_grid",
                "causal_whitespace_grid",
            ],
            "provenance_scope": "historical_preselection_five_seed_evidence",
            "seed_order": list(FINAL_SEEDS[3:]),
            "status": "integrity_verified",
        }
        historical_phase3["anchor_sha256"] = canonical_sha256(
            historical_phase3
        )
        training_completions = {
            "compute_conversion": {
                "artifact": {
                    "git_commit": "c" * 40,
                    "path": COMPUTE_CONFIRMATION_COMPLETION_PATH.as_posix(),
                    "sha256": digest("compute-confirmation-completion-artifact"),
                },
                "completion_sha256": digest("compute-confirmation-completion"),
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
                    "sha256": digest("phase3-confirmation-completion-artifact"),
                },
                "completion_sha256": digest("phase3-confirmation-completion"),
                "run_git_commit": "f" * 40,
            }
        authorization = build_final_evaluation_authorization_v2(
            selection_lock=lock,
            upstream_artifacts={
                "calibration_evidence": {
                    "git_commit": "8" * 40,
                    "path": SELECTION_EVIDENCE_PATH,
                    "sha256": lock["calibration_evidence_manifest_sha256"],
                },
                "selection_lock": {
                    "git_commit": "9" * 40,
                    "path": SELECTION_LOCK_PATH,
                    "sha256": selection_artifact,
                },
                "selection_plan": {
                    "git_commit": "7" * 40,
                    "path": SELECTION_PLAN_PATH,
                    "sha256": lock["plan_sha256"],
                },
            },
            confirmation_evidence={
                "artifact": {
                    "git_commit": "b" * 40,
                    "path": CONFIRMATION_EVIDENCE_PATH,
                    "sha256": digest("confirmation-evidence-artifact"),
                },
                "complete": True,
                "integrity_pass": True,
                "independent_recomputation": recomputation,
                "historical_primary_phase3_provenance": historical_phase3,
                "manifest_sha256": digest("confirmation-manifest-payload"),
                "model_artifact_role_order": [
                    model["artifact_role"] for model in roles["unique_models"]
                ],
                "receipt_commitments_by_seed_role": copy.deepcopy(
                    recomputation["replay_by_seed_role"]
                ),
                "seed_order": list(FINAL_SEEDS[3:]),
                "selection_lock_artifact_sha256": selection_artifact,
                "selection_lock_payload_sha256": lock["lock_sha256"],
                "training_completions": training_completions,
            },
            final_test={
                "evaluation_stream_bytes": 32_000_000,
                "evaluation_stream_sha256": digest("final-stream"),
                "manifest": {
                    "git_commit": "5" * 40,
                    "path": FINAL_TEST_MANIFEST_PATH,
                    "sha256": digest("final-manifest"),
                },
                "output_jsonl": {
                    "path": FINAL_TEST_OUTPUT_PATH,
                    "sha256": digest("final-jsonl"),
                },
                "seal": {
                    "git_commit": "6" * 40,
                    "path": FINAL_TEST_SEAL_PATH,
                    "sha256": lock["final_test_seal_sha256"],
                },
                "seal_payload_sha256": digest("final-seal-payload"),
                "sequence_count": 62_500,
                "sequence_length": 512,
            },
            models=models,
            implementation_sha256={
                path: digest(f"implementation/{path}")
                for path in IMPLEMENTATION_FILE_ORDER
            },
            authorization_git_commit="a" * 40,
        )
        return lock, authorization

    def test_authorization_round_trip_binds_all_models_and_no_test_gate(self) -> None:
        lock, authorization = self._authorization()
        validate_final_evaluation_authorization_v2(
            authorization,
            selection_lock=lock,
        )
        serialized_round_trip = json.loads(
            json.dumps(authorization, sort_keys=True)
        )
        validate_final_evaluation_authorization_v2(
            serialized_round_trip,
            selection_lock=lock,
        )
        self.assertEqual(len(authorization["models"]), 3)
        self.assertNotIn("historical_screening", repr(authorization))
        self.assertNotIn("test_bpb", repr(authorization))
        self.assertNotIn("latency", repr(authorization))

    def test_implementation_manifest_covers_the_complete_package_tree(self) -> None:
        actual = {
            path.as_posix()
            for path in Path("src/jamoflow").glob("*.py")
        }
        self.assertEqual(
            set(EVALUATION_PACKAGE_FILE_ORDER),
            actual
            - {
                "src/jamoflow/inference_actual_v5.py",
                "src/jamoflow/inference_actual_runtime_v5.py",
            },
        )
        self.assertEqual(
            len(IMPLEMENTATION_FILE_ORDER),
            len(set(IMPLEMENTATION_FILE_ORDER)),
        )

    def test_model_identity_rejects_seed_reuse_and_structural_router(self) -> None:
        lock = selection_lock_fixture()
        model = resolve_final_evaluation_roles(lock)["unique_models"][0]
        identity = model_fixture(model)
        rows = {
            seed: copy.deepcopy(identity["seeds"][str(seed)])
            for seed in FINAL_SEEDS
        }
        rows[FINAL_SEEDS[1]]["checkpoint"]["state_sha256"] = rows[
            FINAL_SEEDS[0]
        ]["checkpoint"]["state_sha256"]
        with self.assertRaisesRegex(ValueError, "reused across seeds"):
            build_final_model_identity(
                artifact_role=identity["artifact_role"],
                descriptor=identity["descriptor"],
                seed_evidence=rows,
                parameter_count=FINAL_MAIN_PARAMETER_COUNT,
            )
        rows = {
            seed: copy.deepcopy(identity["seeds"][str(seed)])
            for seed in FINAL_SEEDS
        }
        rows[FINAL_SEEDS[0]]["auxiliary"] = {
            "kind": "entropy_router"
        }
        with self.assertRaisesRegex(ValueError, "cannot bind a router"):
            build_final_model_identity(
                artifact_role=identity["artifact_role"],
                descriptor=identity["descriptor"],
                seed_evidence=rows,
                parameter_count=FINAL_MAIN_PARAMETER_COUNT,
            )

    def test_confirmation_schema_cannot_smuggle_screening_result(self) -> None:
        lock, authorization = self._authorization()
        confirmation = copy.deepcopy(authorization["confirmation_evidence"])
        confirmation["historical_test_pass"] = True
        with self.assertRaisesRegex(ValueError, "confirmation evidence"):
            build_final_evaluation_authorization_v2(
                selection_lock=lock,
                upstream_artifacts=authorization["upstream_artifacts"],
                confirmation_evidence=confirmation,
                final_test=authorization["final_test"],
                models=authorization["models"],
                implementation_sha256=authorization["implementation_sha256"],
                authorization_git_commit=authorization[
                    "authorization_git_commit"
                ],
            )

    def test_recomputation_cannot_disconnect_from_receipt_commitment(self) -> None:
        lock, authorization = self._authorization()
        confirmation = copy.deepcopy(authorization["confirmation_evidence"])
        replay = confirmation["independent_recomputation"]
        replay["replay_by_seed_role"][str(FINAL_SEEDS[3])]["candidate"][
            "nll_array_sha256"
        ] = digest("disconnected-replay")
        replay["recomputation_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in replay.items()
                if key != "recomputation_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "replay role set differs"):
            build_final_evaluation_authorization_v2(
                selection_lock=lock,
                upstream_artifacts=authorization["upstream_artifacts"],
                confirmation_evidence=confirmation,
                final_test=authorization["final_test"],
                models=authorization["models"],
                implementation_sha256=authorization["implementation_sha256"],
                authorization_git_commit=authorization[
                    "authorization_git_commit"
                ],
            )

    def test_historical_phase3_anchor_cannot_rotate_a_model_checkpoint(self) -> None:
        lock, authorization = self._authorization()
        confirmation = copy.deepcopy(authorization["confirmation_evidence"])
        historical = confirmation["historical_primary_phase3_provenance"]
        historical["by_seed_policy"][str(FINAL_SEEDS[3])][
            "causal_codepoint_grid"
        ]["checkpoint_state_sha256"] = digest("rotated-historical-state")
        historical["anchor_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in historical.items()
                if key != "anchor_sha256"
            }
        )
        with self.assertRaisesRegex(ValueError, "anchor differs from model"):
            build_final_evaluation_authorization_v2(
                selection_lock=lock,
                upstream_artifacts=authorization["upstream_artifacts"],
                confirmation_evidence=confirmation,
                final_test=authorization["final_test"],
                models=authorization["models"],
                implementation_sha256=authorization["implementation_sha256"],
                authorization_git_commit=authorization[
                    "authorization_git_commit"
                ],
            )

    def test_entropy_reference_requires_exact_seed_router_bundle(self) -> None:
        lock = selection_lock_fixture(
            broad_futile=False,
            broad_policy="entropy_threshold_codepoint",
        )
        roles = resolve_final_evaluation_roles(lock)
        broad = roles["unique_models"][-1]
        identity = model_fixture(broad)
        validate_final_model_identity(identity)
        rows = {
            seed: copy.deepcopy(identity["seeds"][str(seed)])
            for seed in FINAL_SEEDS
        }
        rows[FINAL_SEEDS[0]]["auxiliary"]["candidate_mask"] = "none"
        with self.assertRaisesRegex(ValueError, "router bundle is malformed"):
            build_final_model_identity(
                artifact_role=identity["artifact_role"],
                descriptor=identity["descriptor"],
                seed_evidence=rows,
                parameter_count=FINAL_MAIN_PARAMETER_COUNT,
            )

    def test_eligible_broad_reference_cannot_be_dropped(self) -> None:
        lock, authorization = self._authorization(broad_futile=False)
        self.assertEqual(len(authorization["models"]), 4)
        with self.assertRaisesRegex(ValueError, "model set is incomplete"):
            build_final_evaluation_authorization_v2(
                selection_lock=lock,
                upstream_artifacts=authorization["upstream_artifacts"],
                confirmation_evidence=authorization["confirmation_evidence"],
                final_test=authorization["final_test"],
                models=authorization["models"][:-1],
                implementation_sha256=authorization["implementation_sha256"],
                authorization_git_commit=authorization[
                    "authorization_git_commit"
                ],
            )

    def test_nested_tamper_invalidates_authorization_hash(self) -> None:
        lock, authorization = self._authorization()
        tampered = copy.deepcopy(authorization)
        tampered["models"][0]["seeds"][str(FINAL_SEEDS[0])]["checkpoint"][
            "state_sha256"
        ] = digest("tampered")
        with self.assertRaises(ValueError):
            validate_final_evaluation_authorization_v2(
                tampered,
                selection_lock=lock,
            )


if __name__ == "__main__":
    unittest.main()
