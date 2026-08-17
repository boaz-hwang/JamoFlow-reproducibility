"""Calibration-only evidence for the two post-selection confirmation seeds.

This layer intentionally has no historical-test, final-test, or latency input.
It proves only that every physical model already fixed by the selection lock has
an exact, independently reconstructed calibration receipt for both confirmation
seeds.  Metric values are descriptive and never change the locked role set.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from .inference_final_authorization_v2 import (
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_SEEDS,
    canonical_sha256,
    expected_model_paths,
    expected_router_paths,
    is_sha256,
    model_spec_sha256_for_descriptor,
    validate_final_auxiliary_bundle,
)
from .inference_final_quality_v2 import resolve_final_evaluation_roles
from .inference_selection_plan import validate_selection_plan_v2
from .inference_selection_v2 import validate_selection_lock_v2


CONFIRMATION_SEEDS = FINAL_SEEDS[3:]
CONFIRMATION_RECEIPT_KIND = "phase3_confirmation_calibration_receipt_v2"
CONFIRMATION_MANIFEST_KIND = "phase3_confirmation_calibration_evidence_v2"
CONFIRMATION_EVALUATOR_PROTOCOL = "jamoflow-confirmation-calibration-v2"
CONFIRMATION_ARTIFACT_ROOT = Path(
    "artifacts/phase3-inference-confirmation-v2/calibration"
)
CONFIRMATION_RESULT_PATH = Path(
    "results/phase3-inference-confirmation-v2/calibration-evidence.json"
)
COMPUTE_CONFIRMATION_COMPLETION_PATH = Path(
    "results/phase3-inference-confirmation-v2/"
    "compute-conversion-training-completion.json"
)
PHASE3_REFERENCE_COMPLETION_PATH = Path(
    "results/phase3-inference-confirmation-v2/"
    "phase3-reference-training-completion.json"
)
CONFIRMATION_TRAINING_COMPLETION_KIND = (
    "phase3_inference_confirmation_training_completion_v2"
)
CONFIRMATION_TRAINING_PROTOCOL = "jamoflow-confirmation-training-completion-v2"
CALIBRATION_SEQUENCE_COUNT = 15_625
CALIBRATION_SEQUENCE_LENGTH = 512
CALIBRATION_TARGETS_PER_SEQUENCE = 511


def confirmation_completion_path(family: str) -> Path:
    if family == "compute_conversion":
        return COMPUTE_CONFIRMATION_COMPLETION_PATH
    if family == "phase3_reference":
        return PHASE3_REFERENCE_COMPLETION_PATH
    raise ValueError("unknown confirmation training family")


def confirmation_run_manifest_path(family: str) -> Path:
    if family == "compute_conversion":
        return Path("runs/phase3-compute-conversion/manifest.json")
    if family == "phase3_reference":
        return Path("runs/phase3/manifest.json")
    raise ValueError("unknown confirmation training family")


def required_confirmation_completion_families(
    selection_lock: Mapping[str, Any],
) -> tuple[str, ...]:
    validate_selection_lock_v2(selection_lock)
    families = ["compute_conversion"]
    if isinstance(
        selection_lock["decision"]["confirmation_plan"].get(
            "phase3_reference"
        ),
        Mapping,
    ):
        families.append("phase3_reference")
    return tuple(families)


def _expected_confirmation_policies(
    selection_lock: Mapping[str, Any], family: str
) -> tuple[str, ...]:
    plan = selection_lock["decision"]["confirmation_plan"]
    if family == "compute_conversion":
        return tuple(plan["compute_conversion"]["policies"])
    if family == "phase3_reference":
        reference = plan.get("phase3_reference")
        if not isinstance(reference, Mapping):
            raise ValueError("selection lock has no Phase3 reference confirmation")
        return tuple(reference["policies"])
    raise ValueError("unknown confirmation training family")


def build_confirmation_training_completion(
    *,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    family: str,
    run_git_commit: str,
    run_manifest: Mapping[str, str],
    implementation_manifest_sha256: str,
    environment_sha256: str,
    units: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_selection_lock_v2(selection_lock)
    policies = _expected_confirmation_policies(selection_lock, family)
    if (
        not is_sha256(selection_lock_artifact_sha256)
        or not _is_git_commit(run_git_commit)
        or not is_sha256(implementation_manifest_sha256)
        or not is_sha256(environment_sha256)
        or not isinstance(run_manifest, Mapping)
        or set(run_manifest) != {"artifact_sha256", "path"}
        or not is_sha256(run_manifest.get("artifact_sha256"))
        or not isinstance(run_manifest.get("path"), str)
        or tuple(sorted(units)) != CONFIRMATION_SEEDS
    ):
        raise ValueError("confirmation training completion identity differs")
    normalized: dict[str, dict[str, Any]] = {}
    descriptors = {
        model["descriptor"]["policy"]: model["descriptor"]
        for model in required_confirmation_models(selection_lock)
    }
    unit_keys = {
        "auxiliary",
        "checkpoint_artifact_sha256",
        "checkpoint_path",
        "checkpoint_state_sha256",
        "training_report_artifact_sha256",
        "training_report_path",
    }
    expected_family = (
        "compute_conversion" if family == "compute_conversion" else "phase3"
    )
    if run_manifest.get("path") != confirmation_run_manifest_path(family).as_posix():
        raise ValueError("confirmation training run-manifest path differs")
    for seed in CONFIRMATION_SEEDS:
        row = units[seed]
        if not isinstance(row, Mapping) or (
            len(row) != len(policies) or set(row) != set(policies)
        ):
            raise ValueError("confirmation training completion policy set differs")
        normalized[str(seed)] = {}
        for policy in policies:
            unit = row[policy]
            descriptor = descriptors.get(policy)
            if descriptor is None or descriptor["model_family"] != expected_family:
                raise ValueError("confirmation completion policy lacks a descriptor")
            expected_paths = expected_model_paths(descriptor, seed)
            auxiliary = unit.get("auxiliary") if isinstance(unit, Mapping) else None
            auxiliary_valid = dict(auxiliary) == {"kind": "none"} if isinstance(
                auxiliary, Mapping
            ) else False
            if isinstance(auxiliary, Mapping) and auxiliary.get("kind") == (
                "entropy_router_artifacts"
            ):
                expected_router = expected_router_paths(seed)
                auxiliary_valid = (
                    descriptor["requires_entropy_router"] is True
                    and set(auxiliary)
                    == {
                        "kind",
                        "router_checkpoint_artifact_sha256",
                        "router_checkpoint_path",
                        "router_checkpoint_state_sha256",
                        "router_report_artifact_sha256",
                        "router_report_path",
                        "threshold_cache_artifact_sha256",
                        "threshold_cache_path",
                        "threshold_diagnostics_artifact_sha256",
                        "threshold_diagnostics_path",
                    }
                    and all(
                        is_sha256(value)
                        for key, value in auxiliary.items()
                        if key.endswith("sha256")
                    )
                    and all(
                        isinstance(value, str) and value
                        for key, value in auxiliary.items()
                        if key.endswith("path")
                    )
                    and auxiliary["router_checkpoint_path"]
                    == expected_router["router_checkpoint"]
                    and auxiliary["router_report_path"]
                    == expected_router["router_report"]
                    and auxiliary["threshold_cache_path"]
                    == expected_router["threshold_cache"]
                    and auxiliary["threshold_diagnostics_path"]
                    == expected_router["threshold_diagnostics"]
                )
            if descriptor["requires_entropy_router"] != (
                isinstance(auxiliary, Mapping)
                and auxiliary.get("kind") == "entropy_router_artifacts"
            ):
                auxiliary_valid = False
            if (
                not isinstance(unit, Mapping)
                or set(unit) != unit_keys
                or not all(
                    is_sha256(unit.get(key))
                    for key in (
                        "checkpoint_artifact_sha256",
                        "checkpoint_state_sha256",
                        "training_report_artifact_sha256",
                    )
                )
                or not isinstance(unit.get("checkpoint_path"), str)
                or not isinstance(unit.get("training_report_path"), str)
                or unit.get("checkpoint_path") != expected_paths["checkpoint"]
                or unit.get("training_report_path")
                != expected_paths["training_report"]
                or not auxiliary_valid
            ):
                raise ValueError("confirmation training unit is malformed")
            normalized[str(seed)][policy] = dict(unit)
    payload = {
        "complete": True,
        "device": "mps",
        "environment_sha256": environment_sha256,
        "family": family,
        "git_worktree_clean_at_end": True,
        "git_worktree_clean_at_start": True,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "kind": CONFIRMATION_TRAINING_COMPLETION_KIND,
        "policy_order": list(policies),
        "protocol": CONFIRMATION_TRAINING_PROTOCOL,
        "run_git_commit": run_git_commit,
        "run_manifest": dict(run_manifest),
        "schema_version": 2,
        "seed_order": list(CONFIRMATION_SEEDS),
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_payload_sha256": selection_lock["lock_sha256"],
        "units": normalized,
    }
    payload["completion_sha256"] = canonical_sha256(payload)
    return payload


def validate_confirmation_training_completion(
    completion: Mapping[str, Any],
    *,
    selection_lock: Mapping[str, Any],
) -> None:
    expected_keys = {
        "complete",
        "completion_sha256",
        "device",
        "environment_sha256",
        "family",
        "git_worktree_clean_at_end",
        "git_worktree_clean_at_start",
        "implementation_manifest_sha256",
        "kind",
        "policy_order",
        "protocol",
        "run_git_commit",
        "run_manifest",
        "schema_version",
        "seed_order",
        "selection_lock_artifact_sha256",
        "selection_lock_payload_sha256",
        "units",
    }
    if not isinstance(completion, Mapping) or set(completion) != expected_keys:
        raise ValueError("confirmation training completion schema differs")
    unsigned = {
        key: value for key, value in completion.items()
        if key != "completion_sha256"
    }
    if (
        completion.get("kind") != CONFIRMATION_TRAINING_COMPLETION_KIND
        or completion.get("protocol") != CONFIRMATION_TRAINING_PROTOCOL
        or completion.get("schema_version") != 2
        or completion.get("complete") is not True
        or completion.get("device") != "mps"
        or completion.get("git_worktree_clean_at_start") is not True
        or completion.get("git_worktree_clean_at_end") is not True
        or completion.get("completion_sha256") != canonical_sha256(unsigned)
    ):
        raise ValueError("confirmation training completion is invalid")
    rebuilt = build_confirmation_training_completion(
        selection_lock=selection_lock,
        selection_lock_artifact_sha256=completion[
            "selection_lock_artifact_sha256"
        ],
        family=completion["family"],
        run_git_commit=completion["run_git_commit"],
        run_manifest=completion["run_manifest"],
        implementation_manifest_sha256=completion[
            "implementation_manifest_sha256"
        ],
        environment_sha256=completion["environment_sha256"],
        units={int(seed): row for seed, row in completion["units"].items()},
    )
    if dict(completion) != rebuilt:
        raise ValueError("confirmation training completion is not canonical")


def validate_training_report_against_completion(
    *,
    completion: Mapping[str, Any],
    report: Mapping[str, Any],
    seed: int,
    policy: str,
    selection_lock: Mapping[str, Any],
    historical_primary_summary_sha256: str,
) -> None:
    """Bind an ignored training report to its tracked completion receipt."""

    validate_confirmation_training_completion(
        completion,
        selection_lock=selection_lock,
    )
    if (
        seed not in CONFIRMATION_SEEDS
        or policy not in completion["policy_order"]
        or report.get("seed") != seed
        or report.get("policy") != policy
        or not is_sha256(historical_primary_summary_sha256)
    ):
        raise ValueError("confirmation report identity differs from completion")
    binding = report.get("evidence_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("confirmation report lacks its execution binding")
    if completion["family"] == "compute_conversion":
        expected_keys = {
            "device",
            "git_commit",
            "git_worktree_clean_at_start",
            "identity_sha256",
            "policies",
            "primary_summary_sha256",
            "schema_version",
            "seeds",
            "selection_plan_sha256",
            "selection_summary_sha256",
            "stage",
        }
        unsigned = {
            key: value
            for key, value in binding.items()
            if key != "identity_sha256"
        }
        valid = (
            set(binding) == expected_keys
            and binding.get("identity_sha256") == canonical_sha256(unsigned)
            and binding.get("device") == "mps"
            and binding.get("git_commit") == completion["run_git_commit"]
            and binding.get("git_worktree_clean_at_start") is True
            and binding.get("policies") == completion["policy_order"]
            and binding.get("primary_summary_sha256")
            == historical_primary_summary_sha256
            and binding.get("schema_version") == 1
            and binding.get("seeds") == completion["seed_order"]
            and binding.get("selection_plan_sha256")
            == selection_lock["plan_sha256"]
            and binding.get("selection_summary_sha256")
            == completion["selection_lock_artifact_sha256"]
            and binding.get("stage") == "confirmation"
        )
    elif completion["family"] == "phase3_reference":
        valid = (
            set(binding)
            == {
                "authorization",
                "device",
                "git_worktree_clean_at_start",
                "kind",
                "run_git_commit",
                "schema_version",
            }
            and isinstance(binding.get("authorization"), Mapping)
            and binding.get("device") == "mps"
            and binding.get("git_worktree_clean_at_start") is True
            and binding.get("kind")
            == "selected_phase3_reference_training_evidence_v4"
            and binding.get("run_git_commit") == completion["run_git_commit"]
            and binding.get("schema_version") == 4
        )
    else:  # guarded by the canonical completion validator above
        valid = False
    if not valid:
        raise ValueError("confirmation report execution binding differs")


def validate_receipts_against_training_completions(
    *,
    selection_lock: Mapping[str, Any],
    receipts: Mapping[int | str, Mapping[str, Mapping[str, Any]]],
    completions: Mapping[str, Mapping[str, Any]],
) -> None:
    """Bind every prospective training unit to its replayed calibration receipt."""

    required_families = required_confirmation_completion_families(selection_lock)
    if len(completions) != len(required_families) or set(completions) != set(
        required_families
    ):
        raise ValueError("confirmation completion family set differs")
    for completion in completions.values():
        validate_confirmation_training_completion(
            completion,
            selection_lock=selection_lock,
        )
    models = required_confirmation_models(selection_lock)
    used: set[tuple[str, int, str]] = set()
    for seed in CONFIRMATION_SEEDS:
        seed_receipts = receipts.get(seed, receipts.get(str(seed)))
        if not isinstance(seed_receipts, Mapping):
            raise ValueError("confirmation replay receipt seed set differs")
        for model in models:
            descriptor = model["descriptor"]
            policy = descriptor["policy"]
            family = (
                "compute_conversion"
                if descriptor["model_family"] == "compute_conversion"
                else "phase3_reference"
            )
            completion = completions.get(family)
            if completion is None or policy not in completion["policy_order"]:
                continue
            unit = completion["units"][str(seed)][policy]
            receipt = seed_receipts[model["artifact_role"]]
            if (
                receipt["checkpoint"]
                != {
                    "artifact_sha256": unit["checkpoint_artifact_sha256"],
                    "path": unit["checkpoint_path"],
                    "state_sha256": unit["checkpoint_state_sha256"],
                }
                or receipt["training_report"]
                != {
                    "artifact_sha256": unit[
                        "training_report_artifact_sha256"
                    ],
                    "path": unit["training_report_path"],
                }
            ):
                raise ValueError("confirmation replay differs from training completion")
            completion_auxiliary = unit["auxiliary"]
            receipt_auxiliary = receipt["auxiliary"]
            if completion_auxiliary["kind"] == "none":
                if receipt_auxiliary != {"kind": "none"}:
                    raise ValueError("structural completion gained a router")
            else:
                if receipt_auxiliary.get("kind") != "entropy_router":
                    raise ValueError("entropy completion lost its router")
                for stem in (
                    "router_checkpoint",
                    "router_report",
                    "threshold_cache",
                    "threshold_diagnostics",
                ):
                    if (
                        completion_auxiliary[f"{stem}_path"]
                        != receipt_auxiliary[f"{stem}_path"]
                        or completion_auxiliary[f"{stem}_artifact_sha256"]
                        != receipt_auxiliary[f"{stem}_artifact_sha256"]
                    ):
                        raise ValueError(
                            "entropy completion differs from calibration replay"
                        )
                if (
                    completion_auxiliary["router_checkpoint_state_sha256"]
                    != receipt_auxiliary["router_checkpoint_state_sha256"]
                ):
                    raise ValueError("entropy completion router state differs")
            used.add((family, seed, policy))
    expected = {
        (family, seed, policy)
        for family, completion in completions.items()
        for seed in CONFIRMATION_SEEDS
        for policy in completion["policy_order"]
    }
    if used != expected:
        raise ValueError("confirmation completion contains an unused or missing unit")


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def required_confirmation_models(
    selection_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return the exact unique physical-model order fixed before confirmation."""

    roles = resolve_final_evaluation_roles(selection_lock)
    models: list[dict[str, Any]] = []
    for model in roles["unique_models"]:
        models.append(
            {
                "artifact_role": model["artifact_role"],
                "descriptor": {
                    key: model[key]
                    for key in (
                        "model_family",
                        "patch_count",
                        "policy",
                        "requires_entropy_router",
                        "runtime_policy",
                    )
                },
            }
        )
    return tuple(models)


def expected_confirmation_paths(
    artifact_role: str,
    seed: int,
) -> dict[str, str]:
    if seed not in CONFIRMATION_SEEDS or not artifact_role:
        raise ValueError("confirmation path requires an exact role and seed")
    root = CONFIRMATION_ARTIFACT_ROOT / f"seed-{seed}"
    return {
        "nll": str(root / f"{artifact_role}-nll.npz"),
        "receipt": str(root / f"{artifact_role}-receipt.json"),
    }


def build_confirmation_calibration_receipt(
    *,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    artifact_role: str,
    descriptor: Mapping[str, Any],
    seed: int,
    evaluator_git_commit: str,
    training_report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    validate_selection_lock_v2(selection_lock)
    required = {
        model["artifact_role"]: model["descriptor"]
        for model in required_confirmation_models(selection_lock)
    }
    if artifact_role not in required or dict(descriptor) != required[artifact_role]:
        raise ValueError("confirmation receipt role differs from the selection lock")
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation receipt seed is not preregistered")
    model_paths = expected_model_paths(descriptor, seed)
    evidence_paths = expected_confirmation_paths(artifact_role, seed)
    if (
        not is_sha256(selection_lock_artifact_sha256)
        or not _is_git_commit(evaluator_git_commit)
        or not isinstance(training_report, Mapping)
        or set(training_report) != {"artifact_sha256", "path"}
        or training_report["path"] != model_paths["training_report"]
        or not is_sha256(training_report["artifact_sha256"])
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"artifact_sha256", "path", "state_sha256"}
        or checkpoint["path"] != model_paths["checkpoint"]
        or not is_sha256(checkpoint["artifact_sha256"])
        or not is_sha256(checkpoint["state_sha256"])
    ):
        raise ValueError("confirmation receipt model artifacts are malformed")
    canonical_auxiliary = validate_final_auxiliary_bundle(
        auxiliary,
        descriptor,
        seed,
    )
    calibration_keys = {
        "boundaries_sha256",
        "bpb",
        "count",
        "dtype",
        "inputs_sha256",
        "matrix_sha256",
        "nll_array_sha256",
        "nll_artifact_path",
        "nll_artifact_sha256",
        "predicted_bytes",
        "stream_sha256",
    }
    hashes = (
        calibration.get("boundaries_sha256"),
        calibration.get("inputs_sha256"),
        calibration.get("matrix_sha256"),
        calibration.get("nll_array_sha256"),
        calibration.get("nll_artifact_sha256"),
        calibration.get("stream_sha256"),
    )
    if (
        not isinstance(calibration, Mapping)
        or set(calibration) != calibration_keys
        or calibration.get("dtype") != "float32"
        or calibration.get("count") != CALIBRATION_SEQUENCE_COUNT
        or calibration.get("predicted_bytes")
        != CALIBRATION_SEQUENCE_COUNT * CALIBRATION_TARGETS_PER_SEQUENCE
        or calibration.get("nll_artifact_path") != evidence_paths["nll"]
        or not all(is_sha256(value) for value in hashes)
        or not isinstance(calibration.get("bpb"), (int, float))
        or isinstance(calibration.get("bpb"), bool)
        or not math.isfinite(float(calibration.get("bpb")))
        or float(calibration.get("bpb")) < 0
    ):
        raise ValueError("confirmation calibration evidence is malformed")
    payload = {
        "artifact_role": artifact_role,
        "auxiliary": canonical_auxiliary,
        "calibration": dict(calibration),
        "checkpoint": dict(checkpoint),
        "complete": True,
        "descriptor": dict(descriptor),
        "device": "mps",
        "evaluator_git_commit": evaluator_git_commit,
        "evaluator_protocol": CONFIRMATION_EVALUATOR_PROTOCOL,
        "kind": CONFIRMATION_RECEIPT_KIND,
        "model": {
            "parameter_count": FINAL_MAIN_PARAMETER_COUNT,
            "spec_sha256": model_spec_sha256_for_descriptor(descriptor),
        },
        "schema_version": 2,
        "seed": seed,
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_payload_sha256": selection_lock["lock_sha256"],
        "training_report": dict(training_report),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_confirmation_calibration_receipt(
    receipt: Mapping[str, Any],
    *,
    selection_lock: Mapping[str, Any],
) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "artifact_role",
        "auxiliary",
        "calibration",
        "checkpoint",
        "complete",
        "descriptor",
        "device",
        "evaluator_git_commit",
        "evaluator_protocol",
        "kind",
        "model",
        "receipt_sha256",
        "schema_version",
        "seed",
        "selection_lock_artifact_sha256",
        "selection_lock_payload_sha256",
        "training_report",
    }:
        raise ValueError("confirmation receipt is not the sealed schema")
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("kind") != CONFIRMATION_RECEIPT_KIND
        or receipt.get("schema_version") != 2
        or receipt.get("complete") is not True
        or receipt.get("device") != "mps"
        or receipt.get("evaluator_protocol") != CONFIRMATION_EVALUATOR_PROTOCOL
        or not is_sha256(receipt.get("receipt_sha256"))
        or receipt["receipt_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError("confirmation receipt identity is invalid")
    rebuilt = build_confirmation_calibration_receipt(
        selection_lock=selection_lock,
        selection_lock_artifact_sha256=receipt[
            "selection_lock_artifact_sha256"
        ],
        artifact_role=receipt["artifact_role"],
        descriptor=receipt["descriptor"],
        seed=receipt["seed"],
        evaluator_git_commit=receipt["evaluator_git_commit"],
        training_report=receipt["training_report"],
        checkpoint=receipt["checkpoint"],
        auxiliary=receipt["auxiliary"],
        calibration=receipt["calibration"],
    )
    if dict(receipt) != rebuilt:
        raise ValueError("confirmation receipt is not canonical")


def build_confirmation_evidence_manifest(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    initial_calibration_evidence_artifact_sha256: str,
    initial_calibration_evidence_payload_sha256: str,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    evaluator_git_commit: str,
    training_completions: Mapping[str, Mapping[str, Any]],
    receipts: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_selection_plan_v2(plan)
    validate_selection_lock_v2(selection_lock)
    models = required_confirmation_models(selection_lock)
    roles = tuple(model["artifact_role"] for model in models)
    required_completions = required_confirmation_completion_families(
        selection_lock
    )
    if (
        not all(
            is_sha256(value)
            for value in (
                plan_artifact_sha256,
                initial_calibration_evidence_artifact_sha256,
                initial_calibration_evidence_payload_sha256,
                selection_lock_artifact_sha256,
            )
        )
        or not _is_git_commit(evaluator_git_commit)
        or plan_artifact_sha256 != selection_lock["plan_sha256"]
        or initial_calibration_evidence_artifact_sha256
        != selection_lock["calibration_evidence_manifest_sha256"]
        or tuple(sorted(receipts)) != CONFIRMATION_SEEDS
        or not isinstance(training_completions, Mapping)
        or len(training_completions) != len(required_completions)
        or set(training_completions) != set(required_completions)
    ):
        raise ValueError("confirmation manifest upstream identity differs")
    normalized_completions: dict[str, dict[str, Any]] = {}
    for family in required_completions:
        row = training_completions[family]
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"artifact", "completion_sha256", "run_git_commit"}
            or not isinstance(row["artifact"], Mapping)
            or set(row["artifact"]) != {"git_commit", "path", "sha256"}
            or row["artifact"]["path"]
            != confirmation_completion_path(family).as_posix()
            or not _is_git_commit(row["artifact"]["git_commit"])
            or not is_sha256(row["artifact"]["sha256"])
            or not is_sha256(row["completion_sha256"])
            or not _is_git_commit(row["run_git_commit"])
        ):
            raise ValueError("confirmation training completion evidence differs")
        normalized_completions[family] = dict(row)
    normalized: dict[str, dict[str, Any]] = {}
    for seed in CONFIRMATION_SEEDS:
        row = receipts[seed]
        if not isinstance(row, Mapping) or (
            len(row) != len(roles) or set(row) != set(roles)
        ):
            raise ValueError("confirmation manifest role set differs")
        normalized[str(seed)] = {}
        for role in roles:
            receipt = row[role]
            validate_confirmation_calibration_receipt(
                receipt,
                selection_lock=selection_lock,
            )
            if (
                receipt["seed"] != seed
                or receipt["artifact_role"] != role
                or receipt["selection_lock_artifact_sha256"]
                != selection_lock_artifact_sha256
                or receipt["evaluator_git_commit"] != evaluator_git_commit
                or receipt["calibration"]["stream_sha256"]
                != plan["calibration_evaluator"]["input_stream_sha256"]
                or receipt["calibration"]["count"]
                != plan["calibration_evaluator"]["sequence_count"]
            ):
                raise ValueError("confirmation manifest receipt was rotated")
            normalized[str(seed)][role] = dict(receipt)
    payload = {
        "calibration": {
            "predicted_bytes_per_sequence": CALIBRATION_TARGETS_PER_SEQUENCE,
            "sequence_count": CALIBRATION_SEQUENCE_COUNT,
            "sequence_length": CALIBRATION_SEQUENCE_LENGTH,
            "stream_sha256": plan["calibration_evaluator"][
                "input_stream_sha256"
            ],
        },
        "complete": True,
        "device": "mps",
        "evaluator_git_commit": evaluator_git_commit,
        "evaluator_protocol": CONFIRMATION_EVALUATOR_PROTOCOL,
        "initial_calibration_evidence_artifact_sha256": (
            initial_calibration_evidence_artifact_sha256
        ),
        "initial_calibration_evidence_payload_sha256": (
            initial_calibration_evidence_payload_sha256
        ),
        "integrity_pass": True,
        "kind": CONFIRMATION_MANIFEST_KIND,
        "model_artifact_role_order": list(roles),
        "models": list(models),
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_payload_sha256": plan["plan_sha256"],
        "receipts": normalized,
        "schema_version": 2,
        "seed_order": list(CONFIRMATION_SEEDS),
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_payload_sha256": selection_lock["lock_sha256"],
        "training_completions": normalized_completions,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def validate_confirmation_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "calibration",
        "complete",
        "device",
        "evaluator_git_commit",
        "evaluator_protocol",
        "initial_calibration_evidence_artifact_sha256",
        "initial_calibration_evidence_payload_sha256",
        "integrity_pass",
        "kind",
        "manifest_sha256",
        "model_artifact_role_order",
        "models",
        "plan_artifact_sha256",
        "plan_payload_sha256",
        "receipts",
        "schema_version",
        "seed_order",
        "selection_lock_artifact_sha256",
        "selection_lock_payload_sha256",
        "training_completions",
    }:
        raise ValueError("confirmation manifest is not the sealed schema")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("kind") != CONFIRMATION_MANIFEST_KIND
        or manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("integrity_pass") is not True
        or manifest.get("device") != "mps"
        or manifest.get("evaluator_protocol") != CONFIRMATION_EVALUATOR_PROTOCOL
        or not is_sha256(manifest.get("manifest_sha256"))
        or manifest["manifest_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError("confirmation manifest identity is invalid")
    rebuilt = build_confirmation_evidence_manifest(
        plan=plan,
        plan_artifact_sha256=manifest["plan_artifact_sha256"],
        initial_calibration_evidence_artifact_sha256=manifest[
            "initial_calibration_evidence_artifact_sha256"
        ],
        initial_calibration_evidence_payload_sha256=manifest[
            "initial_calibration_evidence_payload_sha256"
        ],
        selection_lock=selection_lock,
        selection_lock_artifact_sha256=manifest[
            "selection_lock_artifact_sha256"
        ],
        evaluator_git_commit=manifest["evaluator_git_commit"],
        training_completions=manifest["training_completions"],
        receipts={
            seed: manifest["receipts"][str(seed)]
            for seed in CONFIRMATION_SEEDS
        },
    )
    if dict(manifest) != rebuilt:
        raise ValueError("confirmation manifest is not canonical")
