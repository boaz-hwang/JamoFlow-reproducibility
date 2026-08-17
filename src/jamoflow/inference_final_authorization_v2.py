"""Fail-closed authorization for the one sealed final inference evaluation.

The authorization is created only after calibration-only selection and every
required confirmation run are complete.  It binds the exact five-seed model
artifacts, the still-unopened final-test seal, and the evaluator implementation
before any final-test loss is computed.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .compute_conversion import conversion_model_spec
from .inference_final_quality_v2 import (
    FINAL_BOOTSTRAP_REPETITIONS,
    FINAL_BOOTSTRAP_SEED,
    FINAL_SEEDS,
    resolve_final_evaluation_roles,
)
from .inference_selection_v2 import validate_selection_lock_v2
from .inference_selection_v2 import PRIMARY_CONFIRMED_POLICIES
from .phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    THRESHOLD_POLICIES,
)


FINAL_AUTHORIZATION_KIND = "phase3_inference_final_evaluation_authorization_v2"
FINAL_AUTHORIZATION_PROTOCOL_VERSION = 2
FINAL_EVALUATION_PROTOCOL_ID = "jamoflow-sealed-final-quality-v2"
FINAL_TEST_MANIFEST_PATH = "data/manifests/hplt3-korean-final-test-v1.json"
FINAL_TEST_SEAL_PATH = "data/seals/hplt3-korean-final-test-v1.json"
FINAL_TEST_OUTPUT_PATH = "data/processed/hplt3-korean-final-test-v1/ko.jsonl"
SELECTION_PLAN_PATH = "data/manifests/phase3-inference-selection-plan-v2.json"
SELECTION_EVIDENCE_PATH = (
    "results/phase3-inference-selection-v2/calibration-evidence.json"
)
SELECTION_LOCK_PATH = "results/phase3-inference-selection-v2/selection-lock.json"
CONFIRMATION_EVIDENCE_PATH = (
    "results/phase3-inference-confirmation-v2/calibration-evidence.json"
)
HISTORICAL_PRIMARY_SUMMARY_PATH = (
    "results/phase3-primary-five-seed/summary.json"
)
FINAL_AUTHORIZATION_PATH = (
    "results/phase3-inference-final-v2/post-confirmation-authorization.json"
)
FINAL_EVIDENCE_PATH = "results/phase3-inference-final-v2/evidence-manifest.json"
FINAL_QUALITY_LOCK_PATH = "results/phase3-inference-final-v2/summary.json"
FINAL_ARTIFACT_ROOT = "artifacts/phase3-inference-final-v2"
FINAL_TEST_STREAM_BYTES = 32_000_000
FINAL_TEST_SEQUENCE_LENGTH = 512
FINAL_TEST_SEQUENCE_COUNT = 62_500
FINAL_TEST_TARGETS_PER_SEQUENCE = 511
FINAL_MAIN_PARAMETER_COUNT = 19_596_096
FINAL_ROUTER_PARAMETER_COUNT = 2_016_960
FINAL_TRAIN_EXAMPLE_COUNT = 250_000
FINAL_TRAIN_PREDICTED_BYTES = 127_750_000
FINAL_TRAIN_STEP_COUNT = 7_813

EVALUATION_PACKAGE_FILE_ORDER = (
    "src/jamoflow/__init__.py",
    "src/jamoflow/__main__.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/cli.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/contamination.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/cost.py",
    "src/jamoflow/data_adequacy.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/downstream_data.py",
    "src/jamoflow/ecological.py",
    "src/jamoflow/entropy.py",
    "src/jamoflow/generation.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/hplt3_final_test.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/incremental_token.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/inference_calibration_evidence.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/inference_confirmation_evidence_v2.py",
    "src/jamoflow/inference_confirmation_replay_v2.py",
    "src/jamoflow/inference_final_authorization_v2.py",
    "src/jamoflow/inference_final_quality_evidence_v2.py",
    "src/jamoflow/inference_final_quality_lock_v2.py",
    "src/jamoflow/inference_final_quality_v2.py",
    "src/jamoflow/inference_initial_model_identity_v2.py",
    "src/jamoflow/inference_quality.py",
    "src/jamoflow/inference_selection_plan.py",
    "src/jamoflow/inference_selection_v2.py",
    "src/jamoflow/metrics.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_patching.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/normalization.py",
    "src/jamoflow/patching.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase1_analysis.py",
    "src/jamoflow/phase2_analysis.py",
    "src/jamoflow/phase2_controls.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/phase3_analysis.py",
    "src/jamoflow/phase3_confirmation.py",
    "src/jamoflow/phase3_mechanism.py",
    "src/jamoflow/publication_bpb.py",
    "src/jamoflow/publication_bpe.py",
    "src/jamoflow/publication_downstream.py",
    "src/jamoflow/publication_inference.py",
    "src/jamoflow/publication_model_lock.py",
    "src/jamoflow/publication_protocol.py",
    "src/jamoflow/publication_reference.py",
    "src/jamoflow/publication_runtime.py",
    "src/jamoflow/publication_scale.py",
    "src/jamoflow/report.py",
    "src/jamoflow/unicode_audit.py",
    "src/jamoflow/utf8.py",
)

IMPLEMENTATION_FILE_ORDER = (
    "scripts/run_phase3.py",
    "scripts/run_phase3_compute_conversion.py",
    "scripts/reconstruct_inference_confirmation_calibration_v2.py",
    "scripts/seal_inference_post_confirmation_authorization_v2.py",
    "scripts/run_inference_final_quality_v2.py",
    "scripts/seal_inference_final_quality_lock_v2.py",
    *EVALUATION_PACKAGE_FILE_ORDER,
)


def canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _canonical_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "model_family",
        "patch_count",
        "policy",
        "requires_entropy_router",
        "runtime_policy",
    }
    if set(value) != keys:
        raise ValueError("final model descriptor has an unexpected schema")
    descriptor = {key: value[key] for key in sorted(keys)}
    if (
        descriptor["model_family"] not in {"phase3", "compute_conversion"}
        or not isinstance(descriptor["patch_count"], int)
        or isinstance(descriptor["patch_count"], bool)
        or descriptor["patch_count"] <= 0
        or not isinstance(descriptor["policy"], str)
        or not descriptor["policy"]
        or not isinstance(descriptor["runtime_policy"], str)
        or not descriptor["runtime_policy"]
        or not isinstance(descriptor["requires_entropy_router"], bool)
    ):
        raise ValueError("final model descriptor is malformed")
    return descriptor


def expected_model_paths(
    descriptor: Mapping[str, Any],
    seed: int,
) -> dict[str, str]:
    descriptor = _canonical_descriptor(descriptor)
    if seed not in FINAL_SEEDS:
        raise ValueError("final model path requires a preregistered seed")
    family = descriptor["model_family"]
    policy = descriptor["policy"]
    root = "phase3" if family == "phase3" else "phase3-compute-conversion"
    return {
        "checkpoint": f"artifacts/{root}/seed-{seed}/{policy}.pt",
        "training_report": f"runs/{root}/seed-{seed}/{policy}.json",
    }


def expected_router_paths(seed: int) -> dict[str, str]:
    if seed not in FINAL_SEEDS:
        raise ValueError("final router path requires a preregistered seed")
    return {
        "router_checkpoint": f"artifacts/phase3/seed-{seed}/router.pt",
        "router_report": f"runs/phase3/seed-{seed}/router.json",
        "threshold_cache": f"artifacts/phase3/seed-{seed}/threshold-patches.npz",
        "threshold_diagnostics": (
            f"runs/phase3/seed-{seed}/threshold-patch-diagnostics.json"
        ),
    }


def validate_final_auxiliary_bundle(
    auxiliary: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    requires_router = descriptor["requires_entropy_router"]
    if not requires_router:
        if dict(auxiliary) != {"kind": "none"}:
            raise ValueError("structural final model cannot bind a router")
        return {"kind": "none"}
    keys = {
        "calibration_stream_sha256",
        "candidate_mask",
        "kind",
        "maximum_patch_length",
        "policy",
        "policy_definition_sha256",
        "router_checkpoint_artifact_sha256",
        "router_checkpoint_path",
        "router_checkpoint_state_sha256",
        "router_config_sha256",
        "router_parameter_count",
        "router_report_artifact_sha256",
        "router_report_path",
        "router_training_stream_sha256",
        "seed",
        "threshold_cache_artifact_sha256",
        "threshold_cache_path",
        "threshold_diagnostics_artifact_sha256",
        "threshold_diagnostics_path",
        "threshold_nats",
    }
    if set(auxiliary) != keys:
        raise ValueError("entropy final model has an incomplete router bundle")
    expected_paths = expected_router_paths(seed)
    expected_mask = (
        "none"
        if descriptor["policy"] == "entropy_threshold_full"
        else "codepoint"
    )
    hashes = tuple(
        auxiliary[key]
        for key in (
            "calibration_stream_sha256",
            "policy_definition_sha256",
            "router_checkpoint_artifact_sha256",
            "router_checkpoint_state_sha256",
            "router_config_sha256",
            "router_report_artifact_sha256",
            "router_training_stream_sha256",
            "threshold_cache_artifact_sha256",
            "threshold_diagnostics_artifact_sha256",
        )
    )
    if (
        descriptor["policy"] not in THRESHOLD_POLICIES
        or auxiliary["kind"] != "entropy_router"
        or auxiliary["policy"] != descriptor["policy"]
        or auxiliary["seed"] != seed
        or auxiliary["candidate_mask"] != expected_mask
        or auxiliary["maximum_patch_length"] != 24
        or auxiliary["router_parameter_count"] != FINAL_ROUTER_PARAMETER_COUNT
        or not isinstance(auxiliary["threshold_nats"], (int, float))
        or isinstance(auxiliary["threshold_nats"], bool)
        or not math.isfinite(float(auxiliary["threshold_nats"]))
        or auxiliary["router_checkpoint_path"]
        != expected_paths["router_checkpoint"]
        or auxiliary["router_report_path"] != expected_paths["router_report"]
        or auxiliary["threshold_cache_path"]
        != expected_paths["threshold_cache"]
        or auxiliary["threshold_diagnostics_path"]
        != expected_paths["threshold_diagnostics"]
        or not all(is_sha256(value) for value in hashes)
    ):
        raise ValueError("entropy final model router bundle is malformed")
    return dict(auxiliary)


def model_spec_sha256_for_descriptor(descriptor: Mapping[str, Any]) -> str:
    if descriptor["model_family"] == "phase3":
        if descriptor["patch_count"] != PHASE3_MODEL_SPEC.patch_count:
            raise ValueError("Phase 3 descriptor patch count differs")
        spec = PHASE3_MODEL_SPEC
    else:
        spec = conversion_model_spec(int(descriptor["patch_count"]))
    return canonical_sha256(spec.to_dict())


def build_final_model_identity(
    *,
    artifact_role: str,
    descriptor: Mapping[str, Any],
    seed_evidence: Mapping[int, Mapping[str, Any]],
    parameter_count: int,
) -> dict[str, Any]:
    descriptor = _canonical_descriptor(descriptor)
    if (
        not isinstance(artifact_role, str)
        or not artifact_role
        or tuple(sorted(seed_evidence)) != FINAL_SEEDS
        or not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count != FINAL_MAIN_PARAMETER_COUNT
    ):
        raise ValueError("final model identity has an invalid role or seed set")
    rows: dict[str, Any] = {}
    for seed in FINAL_SEEDS:
        evidence = seed_evidence[seed]
        if set(evidence) != {
            "auxiliary",
            "checkpoint",
            "seed",
            "training",
            "training_report",
        }:
            raise ValueError("final model seed evidence has an unexpected schema")
        checkpoint = evidence["checkpoint"]
        report = evidence["training_report"]
        training = evidence["training"]
        if (
            not isinstance(checkpoint, Mapping)
            or set(checkpoint) != {"artifact_sha256", "path", "state_sha256"}
            or not isinstance(report, Mapping)
            or set(report) != {"artifact_sha256", "path"}
            or not isinstance(training, Mapping)
            or set(training)
            != {
                "evidence_binding_sha256",
                "global_max_position_embeddings",
                "initialization_sha256",
                "optimization_spec_sha256",
                "run_manifest_artifact_sha256",
                "source_artifact_sha256",
                "source_integrity_artifact_sha256",
                "steps",
                "train_examples",
                "train_patch_matrix_sha256",
                "train_predicted_bytes",
                "train_stream_sha256",
                "training_order_sha256",
            }
            or evidence["seed"] != seed
        ):
            raise ValueError("final model seed evidence is malformed")
        expected = expected_model_paths(descriptor, seed)
        if (
            checkpoint["path"] != expected["checkpoint"]
            or report["path"] != expected["training_report"]
            or not all(
                is_sha256(value)
                for value in (
                    checkpoint["artifact_sha256"],
                    checkpoint["state_sha256"],
                    report["artifact_sha256"],
                )
            )
            or not isinstance(evidence["auxiliary"], Mapping)
        ):
            raise ValueError("final model seed evidence paths or hashes differ")
        training_hashes = tuple(
            training[key]
            for key in (
                "evidence_binding_sha256",
                "initialization_sha256",
                "optimization_spec_sha256",
                "run_manifest_artifact_sha256",
                "source_artifact_sha256",
                "source_integrity_artifact_sha256",
                "train_patch_matrix_sha256",
                "train_stream_sha256",
                "training_order_sha256",
            )
        )
        if (
            not all(is_sha256(value) for value in training_hashes)
            or training["optimization_spec_sha256"]
            != canonical_sha256(PHASE3_OPTIMIZATION_SPEC.to_dict())
            or training["global_max_position_embeddings"]
            != PHASE3_MODEL_SPEC.sequence_length * 2 + 8
            or training["train_examples"] != FINAL_TRAIN_EXAMPLE_COUNT
            or training["train_predicted_bytes"] != FINAL_TRAIN_PREDICTED_BYTES
            or training["steps"] != FINAL_TRAIN_STEP_COUNT
        ):
            raise ValueError("final model training lineage is malformed")
        rows[str(seed)] = {
            "auxiliary": validate_final_auxiliary_bundle(
                evidence["auxiliary"], descriptor, seed
            ),
            "checkpoint": dict(checkpoint),
            "seed": seed,
            "training": dict(training),
            "training_report": dict(report),
        }
    checkpoints = tuple(
        rows[str(seed)]["checkpoint"]["artifact_sha256"] for seed in FINAL_SEEDS
    )
    states = tuple(
        rows[str(seed)]["checkpoint"]["state_sha256"] for seed in FINAL_SEEDS
    )
    if len(set(checkpoints)) != len(checkpoints) or len(set(states)) != len(states):
        raise ValueError("final model checkpoint was reused across seeds")
    if descriptor["requires_entropy_router"]:
        router_artifacts = tuple(
            rows[str(seed)]["auxiliary"]["router_checkpoint_artifact_sha256"]
            for seed in FINAL_SEEDS
        )
        router_states = tuple(
            rows[str(seed)]["auxiliary"]["router_checkpoint_state_sha256"]
            for seed in FINAL_SEEDS
        )
        router_configs = tuple(
            rows[str(seed)]["auxiliary"]["router_config_sha256"]
            for seed in FINAL_SEEDS
        )
        if (
            len(set(router_artifacts)) != len(router_artifacts)
            or len(set(router_states)) != len(router_states)
            or len(set(router_configs)) != 1
            or not set(checkpoints).isdisjoint(router_artifacts)
            or not set(states).isdisjoint(router_states)
        ):
            raise ValueError("final router identity is not seed-distinct and invariant")
    common_training_fields = (
        "optimization_spec_sha256",
        "run_manifest_artifact_sha256",
        "source_artifact_sha256",
        "source_integrity_artifact_sha256",
        "train_stream_sha256",
    )
    for field in common_training_fields:
        if len({rows[str(seed)]["training"][field] for seed in FINAL_SEEDS}) != 1:
            raise ValueError("final model training lineage is not seed invariant")
    train_matrices = {
        rows[str(seed)]["training"]["train_patch_matrix_sha256"]
        for seed in FINAL_SEEDS
    }
    if (
        descriptor["requires_entropy_router"]
        and len(train_matrices) != len(FINAL_SEEDS)
    ) or (not descriptor["requires_entropy_router"] and len(train_matrices) != 1):
        raise ValueError("final model train-matrix seed contract differs")
    for field in (
        "evidence_binding_sha256",
        "initialization_sha256",
        "training_order_sha256",
    ):
        if len({rows[str(seed)]["training"][field] for seed in FINAL_SEEDS}) != len(
            FINAL_SEEDS
        ):
            raise ValueError("final model seed training identity was reused")
    payload = {
        "artifact_role": artifact_role,
        "descriptor": descriptor,
        "model_spec_sha256": model_spec_sha256_for_descriptor(descriptor),
        "parameter_count": parameter_count,
        "seed_order": list(FINAL_SEEDS),
        "seeds": rows,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def validate_final_model_identity(identity: Mapping[str, Any]) -> None:
    if not isinstance(identity, Mapping) or set(identity) != {
        "artifact_role",
        "descriptor",
        "identity_sha256",
        "model_spec_sha256",
        "parameter_count",
        "seed_order",
        "seeds",
    }:
        raise ValueError("final model identity has an unexpected schema")
    descriptor = _canonical_descriptor(identity["descriptor"])
    rows = identity["seeds"]
    if (
        tuple(identity["seed_order"]) != FINAL_SEEDS
        or not isinstance(rows, Mapping)
        or set(rows) != {str(seed) for seed in FINAL_SEEDS}
    ):
        raise ValueError("final model identity seed order differs")
    rebuilt = build_final_model_identity(
        artifact_role=identity["artifact_role"],
        descriptor=descriptor,
        seed_evidence={seed: rows[str(seed)] for seed in FINAL_SEEDS},
        parameter_count=identity["parameter_count"],
    )
    if dict(identity) != rebuilt:
        raise ValueError("final model identity is not canonical")


def _validate_artifact_identity(
    value: Mapping[str, Any],
    *,
    expected_path: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"git_commit", "path", "sha256"}
        or value.get("path") != expected_path
        or not is_sha256(value.get("sha256"))
        or not isinstance(value.get("git_commit"), str)
        or len(value["git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in value["git_commit"]
        )
    ):
        raise ValueError(f"final authorization artifact differs: {expected_path}")
    return {
        "git_commit": str(value["git_commit"]),
        "path": expected_path,
        "sha256": str(value["sha256"]),
    }


def _validate_untracked_artifact_identity(
    value: Mapping[str, Any],
    *,
    expected_path: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or value.get("path") != expected_path
        or not is_sha256(value.get("sha256"))
    ):
        raise ValueError(f"final authorization artifact differs: {expected_path}")
    return {"path": expected_path, "sha256": str(value["sha256"])}


def build_final_evaluation_authorization_v2(
    *,
    selection_lock: Mapping[str, Any],
    upstream_artifacts: Mapping[str, Mapping[str, Any]],
    confirmation_evidence: Mapping[str, Any],
    final_test: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    implementation_sha256: Mapping[str, str],
    authorization_git_commit: str,
) -> dict[str, Any]:
    validate_selection_lock_v2(selection_lock)
    roles = resolve_final_evaluation_roles(selection_lock)
    if set(upstream_artifacts) != {
        "calibration_evidence",
        "selection_lock",
        "selection_plan",
    }:
        raise ValueError("final authorization upstream set is incomplete")
    upstream = {
        "calibration_evidence": _validate_artifact_identity(
            upstream_artifacts["calibration_evidence"],
            expected_path=SELECTION_EVIDENCE_PATH,
        ),
        "selection_lock": _validate_artifact_identity(
            upstream_artifacts["selection_lock"],
            expected_path=SELECTION_LOCK_PATH,
        ),
        "selection_plan": _validate_artifact_identity(
            upstream_artifacts["selection_plan"],
            expected_path=SELECTION_PLAN_PATH,
        ),
    }
    if (
        upstream["calibration_evidence"]["sha256"]
        != selection_lock["calibration_evidence_manifest_sha256"]
        or upstream["selection_plan"]["sha256"] != selection_lock["plan_sha256"]
    ):
        raise ValueError("final authorization selection lineage differs")

    expected_artifact_roles = tuple(
        model["artifact_role"] for model in roles["unique_models"]
    )
    recomputation = confirmation_evidence.get("independent_recomputation", {})
    replay_rows = recomputation.get("replay_by_seed_role", {})
    receipt_commitments = confirmation_evidence.get(
        "receipt_commitments_by_seed_role", {}
    )
    historical_phase3 = confirmation_evidence.get(
        "historical_primary_phase3_provenance", {}
    )
    training_completions = confirmation_evidence.get("training_completions", {})
    required_completion_families = ["compute_conversion"]
    if isinstance(
        selection_lock["decision"]["confirmation_plan"].get("phase3_reference"),
        Mapping,
    ):
        required_completion_families.append("phase3_reference")
    completion_paths = {
        "compute_conversion": (
            "results/phase3-inference-confirmation-v2/"
            "compute-conversion-training-completion.json"
        ),
        "phase3_reference": (
            "results/phase3-inference-confirmation-v2/"
            "phase3-reference-training-completion.json"
        ),
    }
    if (
        not isinstance(confirmation_evidence, Mapping)
        or set(confirmation_evidence)
        != {
            "artifact",
            "complete",
            "integrity_pass",
            "independent_recomputation",
            "historical_primary_phase3_provenance",
            "manifest_sha256",
            "model_artifact_role_order",
            "receipt_commitments_by_seed_role",
            "seed_order",
            "selection_lock_artifact_sha256",
            "selection_lock_payload_sha256",
            "training_completions",
        }
        or confirmation_evidence["complete"] is not True
        or confirmation_evidence["integrity_pass"] is not True
        or tuple(confirmation_evidence["seed_order"]) != FINAL_SEEDS[3:]
        or tuple(confirmation_evidence["model_artifact_role_order"])
        != expected_artifact_roles
        or confirmation_evidence["selection_lock_artifact_sha256"]
        != upstream["selection_lock"]["sha256"]
        or confirmation_evidence["selection_lock_payload_sha256"]
        != selection_lock["lock_sha256"]
        or not is_sha256(confirmation_evidence["manifest_sha256"])
        or not isinstance(training_completions, Mapping)
        or len(training_completions) != len(required_completion_families)
        or set(training_completions) != set(required_completion_families)
        or not isinstance(recomputation, Mapping)
        or set(recomputation)
        != {
            "comparison",
            "device",
            "model_artifact_role_order",
            "receipt_count",
            "recomputation_sha256",
            "replay_by_seed_role",
            "seed_order",
            "status",
            "verification_git_commit",
        }
        or recomputation.get("comparison")
        != "bitwise_float32_nll_hash_equal"
        or recomputation.get("device") != "mps"
        or recomputation.get("status") != "pass"
        or tuple(recomputation.get("seed_order", ())) != FINAL_SEEDS[3:]
        or tuple(recomputation.get("model_artifact_role_order", ()))
        != expected_artifact_roles
        or recomputation.get("receipt_count")
        != len(expected_artifact_roles) * len(FINAL_SEEDS[3:])
        or recomputation.get("verification_git_commit")
        != authorization_git_commit
        or not is_sha256(recomputation.get("recomputation_sha256"))
        or recomputation.get("recomputation_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in recomputation.items()
                if key != "recomputation_sha256"
            }
        )
        or set(replay_rows) != {str(seed) for seed in FINAL_SEEDS[3:]}
        or set(receipt_commitments)
        != {str(seed) for seed in FINAL_SEEDS[3:]}
        or not isinstance(historical_phase3, Mapping)
        or set(historical_phase3)
        != {
            "anchor_sha256",
            "artifact",
            "by_seed_policy",
            "policy_order",
            "provenance_scope",
            "seed_order",
            "status",
        }
        or tuple(historical_phase3.get("policy_order", ()))
        != PRIMARY_CONFIRMED_POLICIES
        or historical_phase3.get("provenance_scope")
        != "historical_preselection_five_seed_evidence"
        or historical_phase3.get("status") != "integrity_verified"
        or tuple(historical_phase3.get("seed_order", ())) != FINAL_SEEDS[3:]
        or set(historical_phase3.get("by_seed_policy", {}))
        != {str(seed) for seed in FINAL_SEEDS[3:]}
        or not is_sha256(historical_phase3.get("anchor_sha256"))
        or historical_phase3.get("anchor_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in historical_phase3.items()
                if key != "anchor_sha256"
            }
        )
    ):
        raise ValueError("final authorization confirmation evidence is incomplete")
    normalized_training_completions: dict[str, dict[str, Any]] = {}
    for family in required_completion_families:
        row = training_completions[family]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"artifact", "completion_sha256", "run_git_commit"}
            or not is_sha256(row.get("completion_sha256"))
            or not isinstance(row.get("run_git_commit"), str)
            or len(row["run_git_commit"]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in row["run_git_commit"]
            )
        ):
            raise ValueError("final authorization training completion differs")
        normalized_training_completions[family] = {
            "artifact": _validate_artifact_identity(
                row["artifact"],
                expected_path=completion_paths[family],
            ),
            "completion_sha256": row["completion_sha256"],
            "run_git_commit": row["run_git_commit"],
        }
    for seed in FINAL_SEEDS[3:]:
        seed_rows = replay_rows[str(seed)]
        committed_rows = receipt_commitments[str(seed)]
        if (
            not isinstance(seed_rows, Mapping)
            or len(seed_rows) != len(expected_artifact_roles)
            or set(seed_rows) != set(expected_artifact_roles)
            or not isinstance(committed_rows, Mapping)
            or set(committed_rows) != set(expected_artifact_roles)
            or dict(seed_rows) != dict(committed_rows)
        ):
            raise ValueError("final authorization replay role set differs")
        for role in expected_artifact_roles:
            row = seed_rows[role]
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "checkpoint_state_sha256",
                    "matrix_sha256",
                    "nll_array_sha256",
                    "receipt_sha256",
                }
                or not all(is_sha256(value) for value in row.values())
            ):
                raise ValueError("final authorization replay receipt differs")
    confirmation = {
        **dict(confirmation_evidence),
        "artifact": _validate_artifact_identity(
            confirmation_evidence["artifact"],
            expected_path=CONFIRMATION_EVIDENCE_PATH,
        ),
        "historical_primary_phase3_provenance": {
            **dict(historical_phase3),
            "artifact": _validate_artifact_identity(
                historical_phase3["artifact"],
                expected_path=HISTORICAL_PRIMARY_SUMMARY_PATH,
            ),
        },
        "training_completions": normalized_training_completions,
    }
    for seed in FINAL_SEEDS[3:]:
        policy_rows = historical_phase3["by_seed_policy"][str(seed)]
        if not isinstance(policy_rows, Mapping) or set(policy_rows) != set(
            PRIMARY_CONFIRMED_POLICIES
        ):
            raise ValueError("historical Phase3 policy set differs")
        for historical_row in policy_rows.values():
            if (
                not isinstance(historical_row, Mapping)
                or set(historical_row)
                != {
                    "checkpoint_artifact_sha256",
                    "checkpoint_state_sha256",
                    "training_report_artifact_sha256",
                }
                or not all(is_sha256(value) for value in historical_row.values())
            ):
                raise ValueError("historical Phase3 identity differs")

    if not isinstance(final_test, Mapping) or set(final_test) != {
        "evaluation_stream_bytes",
        "evaluation_stream_sha256",
        "manifest",
        "output_jsonl",
        "seal",
        "seal_payload_sha256",
        "sequence_count",
        "sequence_length",
    }:
        raise ValueError("final-test authorization schema differs")
    sealed_test = {
        **dict(final_test),
        "manifest": _validate_artifact_identity(
            final_test["manifest"], expected_path=FINAL_TEST_MANIFEST_PATH
        ),
        "seal": _validate_artifact_identity(
            final_test["seal"], expected_path=FINAL_TEST_SEAL_PATH
        ),
        "output_jsonl": _validate_untracked_artifact_identity(
            final_test["output_jsonl"], expected_path=FINAL_TEST_OUTPUT_PATH
        ),
    }
    if (
        sealed_test["seal"]["sha256"] != selection_lock["final_test_seal_sha256"]
        or sealed_test["evaluation_stream_bytes"] != FINAL_TEST_STREAM_BYTES
        or sealed_test["sequence_length"] != FINAL_TEST_SEQUENCE_LENGTH
        or sealed_test["sequence_count"] != FINAL_TEST_SEQUENCE_COUNT
        or not is_sha256(sealed_test["evaluation_stream_sha256"])
        or not is_sha256(sealed_test["seal_payload_sha256"])
    ):
        raise ValueError("final-test authorization identity differs")

    expected_models = roles["unique_models"]
    if len(models) != len(expected_models):
        raise ValueError("final authorization model set is incomplete")
    canonical_models: list[dict[str, Any]] = []
    for expected, supplied in zip(expected_models, models, strict=True):
        validate_final_model_identity(supplied)
        expected_descriptor = {
            key: expected[key]
            for key in (
                "model_family",
                "patch_count",
                "policy",
                "requires_entropy_router",
                "runtime_policy",
            )
        }
        if (
            supplied["artifact_role"] != expected["artifact_role"]
            or supplied["descriptor"] != _canonical_descriptor(expected_descriptor)
        ):
            raise ValueError("final authorization model role differs from selection")
        canonical_models.append(dict(supplied))
    if len({model["identity_sha256"] for model in canonical_models}) != len(
        canonical_models
    ):
        raise ValueError("final authorization reused a model identity across roles")
    model_by_role = {
        model["artifact_role"]: model for model in canonical_models
    }
    for seed in FINAL_SEEDS[3:]:
        for role in expected_artifact_roles:
            if (
                replay_rows[str(seed)][role]["checkpoint_state_sha256"]
                != model_by_role[role]["seeds"][str(seed)]["checkpoint"][
                    "state_sha256"
                ]
            ):
                raise ValueError(
                    "final authorization replay checkpoint differs from model"
                )
    for model in canonical_models:
        policy = model["descriptor"]["policy"]
        if (
            model["descriptor"]["model_family"] != "phase3"
            or policy not in PRIMARY_CONFIRMED_POLICIES
        ):
            continue
        for seed in FINAL_SEEDS[3:]:
            historical_row = historical_phase3["by_seed_policy"][str(seed)][
                policy
            ]
            model_row = model["seeds"][str(seed)]
            if (
                historical_row["checkpoint_artifact_sha256"]
                != model_row["checkpoint"]["artifact_sha256"]
                or historical_row["checkpoint_state_sha256"]
                != model_row["checkpoint"]["state_sha256"]
                or historical_row["training_report_artifact_sha256"]
                != model_row["training_report"]["artifact_sha256"]
            ):
                raise ValueError(
                    "historical Phase3 anchor differs from model"
                )

    if (
        len(implementation_sha256) != len(IMPLEMENTATION_FILE_ORDER)
        or set(implementation_sha256) != set(IMPLEMENTATION_FILE_ORDER)
        or not all(is_sha256(value) for value in implementation_sha256.values())
        or not isinstance(authorization_git_commit, str)
        or len(authorization_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in authorization_git_commit)
    ):
        raise ValueError("final authorization implementation identity differs")
    payload = {
        "authorization_git_commit": authorization_git_commit,
        "confirmation_evidence": confirmation,
        "evaluation_contract": {
            "artifact_root": FINAL_ARTIFACT_ROOT,
            "bootstrap_repetitions": FINAL_BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": FINAL_BOOTSTRAP_SEED,
            "device": "mps",
            "evaluation_role_order": roles["evaluation_role_order"],
            "logical_roles": roles["logical_roles"],
            "evidence_output_path": FINAL_EVIDENCE_PATH,
            "quality_lock_path": FINAL_QUALITY_LOCK_PATH,
            "protocol_id": FINAL_EVALUATION_PROTOCOL_ID,
            "role_to_artifact_role": roles["role_to_artifact_role"],
            "seed_order": list(FINAL_SEEDS),
            "sequence_count": FINAL_TEST_SEQUENCE_COUNT,
            "sequence_length": FINAL_TEST_SEQUENCE_LENGTH,
            "targets_per_sequence": FINAL_TEST_TARGETS_PER_SEQUENCE,
        },
        "final_test": sealed_test,
        "implementation_sha256": {
            path: implementation_sha256[path]
            for path in IMPLEMENTATION_FILE_ORDER
        },
        "kind": FINAL_AUTHORIZATION_KIND,
        "models": canonical_models,
        "protocol_version": FINAL_AUTHORIZATION_PROTOCOL_VERSION,
        "schema_version": 2,
        "selection_decision_sha256": selection_lock["decision"]["decision_sha256"],
        "selection_lock_sha256": selection_lock["lock_sha256"],
        "upstream_artifacts": upstream,
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def validate_final_evaluation_authorization_v2(
    authorization: Mapping[str, Any],
    *,
    selection_lock: Mapping[str, Any],
) -> None:
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "authorization_git_commit",
        "authorization_sha256",
        "confirmation_evidence",
        "evaluation_contract",
        "final_test",
        "implementation_sha256",
        "kind",
        "models",
        "protocol_version",
        "schema_version",
        "selection_decision_sha256",
        "selection_lock_sha256",
        "upstream_artifacts",
    }:
        raise ValueError("final evaluation authorization has an unexpected schema")
    unsigned = {
        key: value
        for key, value in authorization.items()
        if key != "authorization_sha256"
    }
    if (
        authorization.get("kind") != FINAL_AUTHORIZATION_KIND
        or authorization.get("schema_version") != 2
        or authorization.get("protocol_version")
        != FINAL_AUTHORIZATION_PROTOCOL_VERSION
        or not is_sha256(authorization.get("authorization_sha256"))
        or authorization["authorization_sha256"]
        != canonical_sha256(unsigned)
    ):
        raise ValueError("final evaluation authorization identity differs")
    rebuilt = build_final_evaluation_authorization_v2(
        selection_lock=selection_lock,
        upstream_artifacts=authorization["upstream_artifacts"],
        confirmation_evidence=authorization["confirmation_evidence"],
        final_test=authorization["final_test"],
        models=authorization["models"],
        implementation_sha256=authorization["implementation_sha256"],
        authorization_git_commit=authorization["authorization_git_commit"],
    )
    if dict(authorization) != rebuilt:
        raise ValueError("final evaluation authorization is not canonical")
