"""Schemas for checkpoint-reconstructed calibration evidence used by selection v2."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .compute_conversion import CONVERSION_POLICIES, conversion_model_spec
from .inference_selection_plan import validate_selection_plan_v2
from .inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
)
from .phase3 import PHASE3_MODEL_SPEC, PHASE3_POLICIES, THRESHOLD_POLICIES


RECEIPT_KIND = "phase3_calibration_receipt_v2"
MANIFEST_KIND = "phase3_calibration_evidence_manifest_v2"
EVIDENCE_PROTOCOL = "jamoflow-calibration-evaluator-v2"
EVIDENCE_ROOT = Path("artifacts/phase3-inference-selection-v2/calibration")

_RECEIPT_KEYS = {
    "auxiliary",
    "calibration",
    "checkpoint",
    "complete",
    "device",
    "evaluator_git_commit",
    "evaluator_protocol",
    "initial_model_identity_lock_sha256",
    "kind",
    "model",
    "model_family",
    "patch_count",
    "plan_artifact_sha256",
    "policy",
    "receipt_sha256",
    "schema_version",
    "seed",
    "training_report",
}
_CALIBRATION_KEYS = {
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
    "report_bpb",
    "stream_sha256",
}
_ARTIFACT_KEYS = {"artifact_sha256", "path"}
_CHECKPOINT_KEYS = {"artifact_sha256", "path", "state_sha256"}
_MODEL_KEYS = {
    "global_max_position_embeddings",
    "parameters",
    "spec_sha256",
}
_ENTROPY_AUXILIARY_KEYS = {
    "cache_artifact_sha256",
    "cache_path",
    "candidate_mask",
    "diagnostics_artifact_sha256",
    "diagnostics_path",
    "kind",
    "maximum_patch_length",
    "router_checkpoint_artifact_sha256",
    "router_checkpoint_path",
    "router_report_artifact_sha256",
    "router_report_path",
    "router_scores_sha256",
    "router_state_sha256",
    "threshold_nats",
}
_MANIFEST_KEYS = {
    "calibration",
    "complete",
    "device",
    "evaluator_git_commit",
    "evaluator_protocol",
    "initial_model_identity_lock_sha256",
    "kind",
    "manifest_sha256",
    "plan_artifact_sha256",
    "plan_payload_sha256",
    "policy_order",
    "receipts",
    "schema_version",
    "seed_order",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _policy_identity(policy: str) -> tuple[str, int]:
    if policy in PHASE3_POLICIES:
        return "phase3", PHASE3_MODEL_SPEC.patch_count
    if policy not in CONVERSION_POLICIES:
        raise ValueError("calibration receipt policy is outside the sealed design")
    return "compute_conversion", int(policy.rsplit("_", 1)[1])


def expected_evidence_paths(
    seed: int,
    policy: str,
) -> dict[str, str]:
    family, _ = _policy_identity(policy)
    if family == "phase3":
        run_root = Path("runs/phase3")
        artifact_root = Path("artifacts/phase3")
    else:
        run_root = Path("runs/phase3-compute-conversion")
        artifact_root = Path("artifacts/phase3-compute-conversion")
    return {
        "training_report": str(run_root / f"seed-{seed}" / f"{policy}.json"),
        "checkpoint": str(artifact_root / f"seed-{seed}" / f"{policy}.pt"),
        "nll": str(EVIDENCE_ROOT / f"seed-{seed}" / f"{policy}-nll.npz"),
        "receipt": str(
            EVIDENCE_ROOT / f"seed-{seed}" / f"{policy}-receipt.json"
        ),
    }


def seal_calibration_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    if "receipt_sha256" in unsigned:
        raise ValueError("unsealed calibration receipt already has a hash")
    receipt = {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}
    validate_calibration_receipt(receipt)
    return receipt


def validate_calibration_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("calibration receipt is not the sealed schema")
    training_report = receipt.get("training_report")
    checkpoint = receipt.get("checkpoint")
    model = receipt.get("model")
    calibration = receipt.get("calibration")
    auxiliary = receipt.get("auxiliary")
    if not all(
        isinstance(value, Mapping)
        for value in (
            training_report,
            checkpoint,
            model,
            calibration,
            auxiliary,
        )
    ):
        raise ValueError("calibration receipt sections must be objects")
    if (
        set(training_report) != _ARTIFACT_KEYS
        or set(checkpoint) != _CHECKPOINT_KEYS
        or set(model) != _MODEL_KEYS
        or set(calibration) != _CALIBRATION_KEYS
    ):
        raise ValueError("calibration receipt nested schema differs")
    seed = receipt.get("seed")
    policy = receipt.get("policy")
    if seed not in INITIAL_SEEDS or policy not in CALIBRATION_POLICY_ORDER:
        raise ValueError("calibration receipt seed/policy is not preregistered")
    family, patch_count = _policy_identity(policy)
    paths = expected_evidence_paths(seed, policy)
    expected_spec = (
        PHASE3_MODEL_SPEC
        if family == "phase3"
        else conversion_model_spec(patch_count)
    )
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    hashes = (
        receipt.get("plan_artifact_sha256"),
        receipt.get("initial_model_identity_lock_sha256"),
        receipt.get("receipt_sha256"),
        training_report.get("artifact_sha256"),
        checkpoint.get("artifact_sha256"),
        checkpoint.get("state_sha256"),
        model.get("spec_sha256"),
        calibration.get("boundaries_sha256"),
        calibration.get("inputs_sha256"),
        calibration.get("matrix_sha256"),
        calibration.get("nll_array_sha256"),
        calibration.get("nll_artifact_sha256"),
        calibration.get("stream_sha256"),
    )
    if (
        receipt.get("kind") != RECEIPT_KIND
        or receipt.get("schema_version") != 2
        or receipt.get("complete") is not True
        or receipt.get("evaluator_protocol") != EVIDENCE_PROTOCOL
        or receipt.get("device") != "mps"
        or not _is_git_commit(receipt.get("evaluator_git_commit"))
        or receipt.get("model_family") != family
        or receipt.get("patch_count") != patch_count
        or not all(_is_sha256(value) for value in hashes)
        or receipt["receipt_sha256"] != canonical_sha256(unsigned)
        or training_report.get("path") != paths["training_report"]
        or checkpoint.get("path") != paths["checkpoint"]
        or calibration.get("nll_artifact_path") != paths["nll"]
        or model.get("spec_sha256") != canonical_sha256(expected_spec.to_dict())
        or not isinstance(model.get("parameters"), int)
        or isinstance(model.get("parameters"), bool)
        or model.get("parameters") <= 0
        or model.get("global_max_position_embeddings")
        != PHASE3_MODEL_SPEC.sequence_length * 2 + 8
        or calibration.get("dtype") != "float32"
        or not isinstance(calibration.get("count"), int)
        or isinstance(calibration.get("count"), bool)
        or calibration.get("count") <= 0
        or calibration.get("predicted_bytes")
        != calibration.get("count") * (PHASE3_MODEL_SPEC.sequence_length - 1)
        or not isinstance(calibration.get("bpb"), (int, float))
        or isinstance(calibration.get("bpb"), bool)
        or not math.isfinite(float(calibration.get("bpb")))
        or float(calibration.get("bpb")) < 0
        or not isinstance(calibration.get("report_bpb"), (int, float))
        or isinstance(calibration.get("report_bpb"), bool)
        or not math.isfinite(float(calibration.get("report_bpb")))
        or not math.isclose(
            float(calibration.get("bpb")),
            float(calibration.get("report_bpb")),
            rel_tol=0,
            abs_tol=1e-7,
        )
    ):
        raise ValueError("calibration receipt identity/content is invalid")
    if policy in THRESHOLD_POLICIES:
        expected_candidate_mask = (
            "none" if policy == "entropy_threshold_full" else "codepoint"
        )
        if (
            set(auxiliary) != _ENTROPY_AUXILIARY_KEYS
            or auxiliary.get("kind") != "entropy_router"
            or auxiliary.get("candidate_mask") != expected_candidate_mask
            or auxiliary.get("maximum_patch_length") != 24
            or auxiliary.get("cache_path")
            != f"artifacts/phase3/seed-{seed}/threshold-patches.npz"
            or auxiliary.get("diagnostics_path")
            != (
                f"runs/phase3/seed-{seed}/"
                "threshold-patch-diagnostics.json"
            )
            or auxiliary.get("router_checkpoint_path")
            != f"artifacts/phase3/seed-{seed}/router.pt"
            or auxiliary.get("router_report_path")
            != f"runs/phase3/seed-{seed}/router.json"
            or not isinstance(auxiliary.get("threshold_nats"), (int, float))
            or isinstance(auxiliary.get("threshold_nats"), bool)
            or not math.isfinite(float(auxiliary.get("threshold_nats")))
            or not all(
                _is_sha256(auxiliary.get(key))
                for key in (
                    "cache_artifact_sha256",
                    "diagnostics_artifact_sha256",
                    "router_checkpoint_artifact_sha256",
                    "router_report_artifact_sha256",
                    "router_scores_sha256",
                    "router_state_sha256",
                )
            )
        ):
            raise ValueError("entropy calibration receipt lacks its router bundle")
    elif dict(auxiliary) != {"kind": "none"}:
        raise ValueError("structural calibration receipt must not claim an auxiliary")
    if plan is not None:
        validate_selection_plan_v2(plan)
        if (
            calibration.get("stream_sha256")
            != plan["calibration_evaluator"]["input_stream_sha256"]
            or calibration.get("count")
            != plan["calibration_evaluator"]["sequence_count"]
            or receipt.get("evaluator_protocol")
            != plan["calibration_evaluator"]["evaluator_protocol"]
            or receipt.get("device")
            != plan["calibration_evaluator"]["device"]
        ):
            raise ValueError("calibration receipt differs from the selection plan")


def build_calibration_evidence_manifest(
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    evaluator_git_commit: str,
    initial_model_identity_lock_sha256: str,
    receipts: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    validate_selection_plan_v2(plan)
    if (
        not _is_sha256(plan_artifact_sha256)
        or not _is_sha256(initial_model_identity_lock_sha256)
        or not _is_git_commit(evaluator_git_commit)
    ):
        raise ValueError("calibration evidence identity is malformed")
    if tuple(sorted(receipts)) != INITIAL_SEEDS:
        raise ValueError("calibration evidence requires the exact initial seeds")
    normalized: dict[str, dict[str, Any]] = {}
    for seed in INITIAL_SEEDS:
        row = receipts[seed]
        if set(row) != set(CALIBRATION_POLICY_ORDER):
            raise ValueError("calibration evidence policy set is not exact")
        normalized[str(seed)] = {}
        for policy in CALIBRATION_POLICY_ORDER:
            receipt = dict(row[policy])
            validate_calibration_receipt(receipt, plan=plan)
            if (
                receipt["seed"] != seed
                or receipt["policy"] != policy
                or receipt["plan_artifact_sha256"] != plan_artifact_sha256
                or receipt["evaluator_git_commit"] != evaluator_git_commit
                or receipt["initial_model_identity_lock_sha256"]
                != initial_model_identity_lock_sha256
            ):
                raise ValueError("calibration receipt identity was rotated")
            normalized[str(seed)][policy] = receipt
    unsigned: dict[str, Any] = {
        "calibration": {
            "predicted_bytes_per_sequence": 511,
            "sequence_count": plan["calibration_evaluator"]["sequence_count"],
            "stream_sha256": plan["calibration_evaluator"]["input_stream_sha256"],
        },
        "complete": True,
        "device": plan["calibration_evaluator"]["device"],
        "evaluator_git_commit": evaluator_git_commit,
        "evaluator_protocol": EVIDENCE_PROTOCOL,
        "initial_model_identity_lock_sha256": (
            initial_model_identity_lock_sha256
        ),
        "kind": MANIFEST_KIND,
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_payload_sha256": plan["plan_sha256"],
        "policy_order": list(CALIBRATION_POLICY_ORDER),
        "receipts": normalized,
        "schema_version": 2,
        "seed_order": list(INITIAL_SEEDS),
    }
    manifest = {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}
    validate_calibration_evidence_manifest(manifest, plan=plan)
    return manifest


def validate_calibration_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    validate_selection_plan_v2(plan)
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("calibration evidence manifest is not the sealed schema")
    calibration = manifest.get("calibration")
    receipts = manifest.get("receipts")
    if not isinstance(calibration, Mapping) or not isinstance(receipts, Mapping):
        raise ValueError("calibration evidence manifest sections are malformed")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("kind") != MANIFEST_KIND
        or manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("device") != "mps"
        or manifest.get("evaluator_protocol") != EVIDENCE_PROTOCOL
        or not _is_git_commit(manifest.get("evaluator_git_commit"))
        or not _is_sha256(
            manifest.get("initial_model_identity_lock_sha256")
        )
        or not _is_sha256(manifest.get("plan_artifact_sha256"))
        or manifest.get("plan_payload_sha256") != plan["plan_sha256"]
        or tuple(manifest.get("seed_order", ())) != INITIAL_SEEDS
        or tuple(manifest.get("policy_order", ()))
        != CALIBRATION_POLICY_ORDER
        or not _is_sha256(manifest.get("manifest_sha256"))
        or manifest["manifest_sha256"] != canonical_sha256(unsigned)
        or dict(calibration)
        != {
            "predicted_bytes_per_sequence": 511,
            "sequence_count": plan["calibration_evaluator"]["sequence_count"],
            "stream_sha256": plan["calibration_evaluator"]["input_stream_sha256"],
        }
        or tuple(sorted(int(key) for key in receipts)) != INITIAL_SEEDS
    ):
        raise ValueError("calibration evidence manifest identity is invalid")
    for seed in INITIAL_SEEDS:
        row = receipts.get(str(seed))
        if not isinstance(row, Mapping) or set(row) != set(
            CALIBRATION_POLICY_ORDER
        ):
            raise ValueError("calibration evidence manifest policy set differs")
        for policy in CALIBRATION_POLICY_ORDER:
            receipt = row[policy]
            if not isinstance(receipt, Mapping):
                raise ValueError("calibration evidence receipt is malformed")
            validate_calibration_receipt(receipt, plan=plan)
            if (
                receipt.get("seed") != seed
                or receipt.get("policy") != policy
                or receipt.get("plan_artifact_sha256")
                != manifest["plan_artifact_sha256"]
                or receipt.get("evaluator_git_commit")
                != manifest["evaluator_git_commit"]
                or receipt.get("initial_model_identity_lock_sha256")
                != manifest["initial_model_identity_lock_sha256"]
            ):
                raise ValueError("calibration evidence manifest receipt was rotated")


def calibration_bpb_matrix(
    manifest: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[int, dict[str, float]]:
    validate_calibration_evidence_manifest(manifest, plan=plan)
    return {
        seed: {
            policy: float(
                manifest["receipts"][str(seed)][policy]["calibration"]["bpb"]
            )
            for policy in CALIBRATION_POLICY_ORDER
        }
        for seed in INITIAL_SEEDS
    }
