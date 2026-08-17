"""One-session evidence schema for the sealed Korean final-quality evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .inference_final_authorization_v2 import (
    FINAL_ARTIFACT_ROOT,
    FINAL_AUTHORIZATION_KIND,
    FINAL_EVALUATION_PROTOCOL_ID,
    FINAL_SEEDS,
    FINAL_TEST_SEQUENCE_COUNT,
    FINAL_TEST_SEQUENCE_LENGTH,
    FINAL_TEST_TARGETS_PER_SEQUENCE,
    canonical_sha256,
    is_sha256,
    validate_final_evaluation_authorization_v2,
)


FINAL_EVIDENCE_PROTOCOL_VERSION = 2
FINAL_SESSION_KIND = "phase3_inference_final_quality_session_v2"
FINAL_RECEIPT_KIND = "phase3_inference_final_quality_receipt_v2"
FINAL_EVIDENCE_KIND = "phase3_inference_final_quality_evidence_v2"
FINAL_EVALUATION_BATCH_SIZE = 64
FINAL_NLL_VERIFICATION_COMPARISON = "bitwise_equal_float32_array_sha256"
FINAL_SESSION_PATH = str(Path(FINAL_ARTIFACT_ROOT) / "session-plan.json")


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def final_session_id(
    *,
    authorization_artifact_sha256: str,
    authorization_payload_sha256: str,
    final_test_seal_payload_sha256: str,
) -> str:
    if not all(
        is_sha256(value)
        for value in (
            authorization_artifact_sha256,
            authorization_payload_sha256,
            final_test_seal_payload_sha256,
        )
    ):
        raise ValueError("final session requires sealed SHA-256 identities")
    return canonical_sha256(
        {
            "authorization_artifact_sha256": authorization_artifact_sha256,
            "authorization_payload_sha256": authorization_payload_sha256,
            "final_test_seal_payload_sha256": final_test_seal_payload_sha256,
            "protocol_id": FINAL_EVALUATION_PROTOCOL_ID,
        }
    )


def _canonical_runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "batch_size",
        "device",
        "mps_available",
        "numpy",
        "python",
        "torch",
        "transformers",
    }
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != keys
        or runtime["batch_size"] != FINAL_EVALUATION_BATCH_SIZE
        or runtime["device"] != "mps"
        or runtime["mps_available"] is not True
        or any(
            not isinstance(runtime[key], str) or not runtime[key]
            for key in ("numpy", "python", "torch", "transformers")
        )
    ):
        raise ValueError("final evaluation runtime identity differs")
    return {key: runtime[key] for key in sorted(keys)}


def _canonical_final_context(
    final_context: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    context_keys = {
        "boundaries_sha256",
        "document_assignment_sha256",
        "document_layout_sha256",
        "eligible_sequence_count",
        "inputs_sha256",
        "stream_bytes",
        "stream_sha256",
    }
    if (
        not isinstance(final_context, Mapping)
        or set(final_context) != context_keys
        or final_context["stream_bytes"]
        != FINAL_TEST_SEQUENCE_COUNT * FINAL_TEST_SEQUENCE_LENGTH
        or final_context["stream_sha256"]
        != authorization["final_test"]["evaluation_stream_sha256"]
        or not isinstance(final_context["eligible_sequence_count"], int)
        or isinstance(final_context["eligible_sequence_count"], bool)
        or not 0 < final_context["eligible_sequence_count"]
        <= FINAL_TEST_SEQUENCE_COUNT
        or not all(
            is_sha256(final_context[key])
            for key in (
                "boundaries_sha256",
                "document_assignment_sha256",
                "document_layout_sha256",
                "inputs_sha256",
                "stream_sha256",
            )
        )
    ):
        raise ValueError("final receipt stream/document context differs")
    return dict(final_context)


def authorized_unit_order(
    authorization: Mapping[str, Any],
) -> tuple[tuple[int, str, int], ...]:
    units: list[tuple[int, str, int]] = []
    index = 0
    for model in authorization["models"]:
        for seed in FINAL_SEEDS:
            units.append((index, model["artifact_role"], seed))
            index += 1
    return tuple(units)


def expected_final_evidence_paths(
    artifact_role: str,
    seed: int,
) -> dict[str, str]:
    if not artifact_role or seed not in FINAL_SEEDS:
        raise ValueError("final evidence path requires an authorized role and seed")
    root = Path(FINAL_ARTIFACT_ROOT) / f"seed-{seed}"
    return {
        "nll": str(root / f"{artifact_role}-nll.npz"),
        "receipt": str(root / f"{artifact_role}-receipt.json"),
    }


def build_final_quality_session_plan(
    *,
    authorization: Mapping[str, Any],
    authorization_artifact_sha256: str,
    authorization_git_commit: str,
    selection_lock: Mapping[str, Any],
    evaluator_git_commit: str,
    runtime: Mapping[str, Any],
    final_context: Mapping[str, Any],
) -> dict[str, Any]:
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    if (
        not is_sha256(authorization_artifact_sha256)
        or not _is_git_commit(authorization_git_commit)
        or not _is_git_commit(evaluator_git_commit)
    ):
        raise ValueError("final session Git/artifact identity differs")
    session_id = final_session_id(
        authorization_artifact_sha256=authorization_artifact_sha256,
        authorization_payload_sha256=authorization["authorization_sha256"],
        final_test_seal_payload_sha256=authorization["final_test"][
            "seal_payload_sha256"
        ],
    )
    units = authorized_unit_order(authorization)
    payload = {
        "authorization": {
            "artifact_sha256": authorization_artifact_sha256,
            "git_commit": authorization_git_commit,
            "payload_sha256": authorization["authorization_sha256"],
        },
        "evaluator_git_commit": evaluator_git_commit,
        "evaluator_implementation_sha256": dict(
            authorization["implementation_sha256"]
        ),
        "evaluator_protocol": FINAL_EVALUATION_PROTOCOL_ID,
        "final_context": _canonical_final_context(final_context, authorization),
        "final_test_seal_artifact_sha256": authorization["final_test"]["seal"][
            "sha256"
        ],
        "final_test_seal_payload_sha256": authorization["final_test"][
            "seal_payload_sha256"
        ],
        "kind": FINAL_SESSION_KIND,
        "runtime": _canonical_runtime_identity(runtime),
        "schema_version": 2,
        "session_id": session_id,
        "status": "planned_fixed_units_before_evaluation",
        "unit_order": [
            {
                "artifact_role": role,
                "nll_path": expected_final_evidence_paths(role, seed)["nll"],
                "receipt_path": expected_final_evidence_paths(role, seed)[
                    "receipt"
                ],
                "seed": seed,
                "unit_index": index,
            }
            for index, role, seed in units
        ],
        "verification_contract": {
            "comparison": FINAL_NLL_VERIFICATION_COMPARISON,
            "device": "mps",
            "independent_second_model_forward_required": True,
            "verifier_batch_size": FINAL_EVALUATION_BATCH_SIZE,
        },
    }
    payload["session_plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_final_quality_session_plan(
    session_plan: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
) -> None:
    if not isinstance(session_plan, Mapping) or set(session_plan) != {
        "authorization",
        "evaluator_git_commit",
        "evaluator_implementation_sha256",
        "evaluator_protocol",
        "final_context",
        "final_test_seal_artifact_sha256",
        "final_test_seal_payload_sha256",
        "kind",
        "runtime",
        "schema_version",
        "session_id",
        "session_plan_sha256",
        "status",
        "unit_order",
        "verification_contract",
    }:
        raise ValueError("final session plan is not the sealed schema")
    unsigned = {
        key: value
        for key, value in session_plan.items()
        if key != "session_plan_sha256"
    }
    if (
        session_plan.get("kind") != FINAL_SESSION_KIND
        or session_plan.get("schema_version") != 2
        or not is_sha256(session_plan.get("session_plan_sha256"))
        or session_plan["session_plan_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError("final session plan identity differs")
    rebuilt = build_final_quality_session_plan(
        authorization=authorization,
        authorization_artifact_sha256=session_plan["authorization"][
            "artifact_sha256"
        ],
        authorization_git_commit=session_plan["authorization"]["git_commit"],
        selection_lock=selection_lock,
        evaluator_git_commit=session_plan["evaluator_git_commit"],
        runtime=session_plan["runtime"],
        final_context=session_plan["final_context"],
    )
    if dict(session_plan) != rebuilt:
        raise ValueError("final session plan is not canonical")


def _authorized_model(
    authorization: Mapping[str, Any],
    artifact_role: str,
) -> Mapping[str, Any]:
    matches = [
        model
        for model in authorization["models"]
        if model["artifact_role"] == artifact_role
    ]
    if len(matches) != 1:
        raise ValueError("final evidence role is not an authorized unique model")
    return matches[0]


def build_final_quality_receipt(
    *,
    authorization: Mapping[str, Any],
    authorization_artifact_sha256: str,
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
    unit_index: int,
    artifact_role: str,
    seed: int,
    patch_matrix_sha256: str,
    auxiliary_execution: Mapping[str, Any],
    nll: Mapping[str, Any],
) -> dict[str, Any]:
    if authorization.get("kind") != FINAL_AUTHORIZATION_KIND:
        raise ValueError("final receipt requires a post-confirmation authorization")
    validate_final_quality_session_plan(
        session_plan,
        authorization=authorization,
        selection_lock=selection_lock,
    )
    model = _authorized_model(authorization, artifact_role)
    expected_units = authorized_unit_order(authorization)
    expected_session = final_session_id(
        authorization_artifact_sha256=authorization_artifact_sha256,
        authorization_payload_sha256=authorization["authorization_sha256"],
        final_test_seal_payload_sha256=authorization["final_test"][
            "seal_payload_sha256"
        ],
    )
    if (
        unit_index < 0
        or unit_index >= len(expected_units)
        or expected_units[unit_index] != (unit_index, artifact_role, seed)
        or not is_sha256(authorization_artifact_sha256)
        or session_plan["session_id"] != expected_session
        or session_plan["authorization"]["artifact_sha256"]
        != authorization_artifact_sha256
    ):
        raise ValueError("final receipt unit/session identity differs")
    seed_evidence = model["seeds"][str(seed)]
    final_context = _canonical_final_context(
        session_plan["final_context"],
        authorization,
    )
    if not is_sha256(patch_matrix_sha256):
        raise ValueError("final receipt stream/document context differs")
    locked_auxiliary = seed_evidence["auxiliary"]
    if locked_auxiliary["kind"] == "none":
        if dict(auxiliary_execution) != {"kind": "none"}:
            raise ValueError("structural final receipt cannot execute a router")
        canonical_auxiliary = {"kind": "none"}
    else:
        if (
            not isinstance(auxiliary_execution, Mapping)
            or set(auxiliary_execution)
            != {
                "final_matrix_sha256",
                "kind",
                "locked_bundle_sha256",
                "router_scores_sha256",
            }
            or auxiliary_execution["kind"] != "entropy_router"
            or auxiliary_execution["locked_bundle_sha256"]
            != canonical_sha256(locked_auxiliary)
            or auxiliary_execution["final_matrix_sha256"]
            != patch_matrix_sha256
            or not is_sha256(auxiliary_execution["router_scores_sha256"])
        ):
            raise ValueError("entropy final receipt execution differs from its bundle")
        canonical_auxiliary = dict(auxiliary_execution)
    nll_keys = {
        "array_sha256",
        "artifact_path",
        "artifact_sha256",
        "bpb",
        "count",
        "dtype",
        "predicted_bytes",
    }
    paths = expected_final_evidence_paths(artifact_role, seed)
    if (
        not isinstance(nll, Mapping)
        or set(nll) != nll_keys
        or nll["artifact_path"] != paths["nll"]
        or nll["dtype"] != "float32"
        or nll["count"] != FINAL_TEST_SEQUENCE_COUNT
        or nll["predicted_bytes"]
        != FINAL_TEST_SEQUENCE_COUNT * FINAL_TEST_TARGETS_PER_SEQUENCE
        or not is_sha256(nll["array_sha256"])
        or not is_sha256(nll["artifact_sha256"])
        or not isinstance(nll["bpb"], (int, float))
        or isinstance(nll["bpb"], bool)
        or not math.isfinite(float(nll["bpb"]))
        or float(nll["bpb"]) < 0
    ):
        raise ValueError("final receipt NLL evidence is malformed")
    payload = {
        "artifact_role": artifact_role,
        "authorization_artifact_sha256": authorization_artifact_sha256,
        "authorization_payload_sha256": authorization["authorization_sha256"],
        "auxiliary_execution": canonical_auxiliary,
        "checkpoint": dict(seed_evidence["checkpoint"]),
        "complete": True,
        "evaluator_git_commit": session_plan["evaluator_git_commit"],
        "evaluator_protocol": FINAL_EVALUATION_PROTOCOL_ID,
        "final_context": final_context,
        "final_test_seal_artifact_sha256": authorization["final_test"]["seal"][
            "sha256"
        ],
        "final_test_seal_payload_sha256": authorization["final_test"][
            "seal_payload_sha256"
        ],
        "kind": FINAL_RECEIPT_KIND,
        "model_identity_sha256": model["identity_sha256"],
        "nll": dict(nll),
        "patch_matrix_sha256": patch_matrix_sha256,
        "schema_version": 2,
        "seed": seed,
        "session_id": session_plan["session_id"],
        "session_plan_sha256": session_plan["session_plan_sha256"],
        "runtime": dict(session_plan["runtime"]),
        "unit_index": unit_index,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_final_quality_receipt(
    receipt: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
) -> None:
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "artifact_role",
        "authorization_artifact_sha256",
        "authorization_payload_sha256",
        "auxiliary_execution",
        "checkpoint",
        "complete",
        "evaluator_git_commit",
        "evaluator_protocol",
        "final_context",
        "final_test_seal_artifact_sha256",
        "final_test_seal_payload_sha256",
        "kind",
        "model_identity_sha256",
        "nll",
        "patch_matrix_sha256",
        "receipt_sha256",
        "schema_version",
        "seed",
        "session_id",
        "session_plan_sha256",
        "runtime",
        "unit_index",
    }:
        raise ValueError("final quality receipt is not the sealed schema")
    unsigned = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("kind") != FINAL_RECEIPT_KIND
        or receipt.get("schema_version") != 2
        or receipt.get("complete") is not True
        or not is_sha256(receipt.get("receipt_sha256"))
        or receipt["receipt_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError("final quality receipt identity differs")
    rebuilt = build_final_quality_receipt(
        authorization=authorization,
        authorization_artifact_sha256=receipt[
            "authorization_artifact_sha256"
        ],
        selection_lock=selection_lock,
        session_plan=session_plan,
        unit_index=receipt["unit_index"],
        artifact_role=receipt["artifact_role"],
        seed=receipt["seed"],
        patch_matrix_sha256=receipt["patch_matrix_sha256"],
        auxiliary_execution=receipt["auxiliary_execution"],
        nll=receipt["nll"],
    )
    if dict(receipt) != rebuilt:
        raise ValueError("final quality receipt is not canonical")


def build_final_quality_evidence_manifest(
    *,
    authorization: Mapping[str, Any],
    authorization_artifact_sha256: str,
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    receipt_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    validate_final_quality_session_plan(
        session_plan,
        authorization=authorization,
        selection_lock=selection_lock,
    )
    units = authorized_unit_order(authorization)
    expected_session = final_session_id(
        authorization_artifact_sha256=authorization_artifact_sha256,
        authorization_payload_sha256=authorization["authorization_sha256"],
        final_test_seal_payload_sha256=authorization["final_test"][
            "seal_payload_sha256"
        ],
    )
    if (
        len(receipts) != len(units)
        or len(receipt_artifacts) != len(units)
        or not is_sha256(authorization_artifact_sha256)
        or session_plan["authorization"]["artifact_sha256"]
        != authorization_artifact_sha256
        or session_plan["session_id"] != expected_session
    ):
        raise ValueError("final evidence manifest unit/session set differs")
    normalized: list[dict[str, Any]] = []
    normalized_receipt_artifacts: list[dict[str, str]] = []
    receipt_hashes: set[str] = set()
    receipt_artifact_hashes: set[str] = set()
    nll_hashes: set[str] = set()
    for expected, receipt, receipt_artifact in zip(
        units,
        receipts,
        receipt_artifacts,
        strict=True,
    ):
        validate_final_quality_receipt(
            receipt,
            authorization=authorization,
            selection_lock=selection_lock,
            session_plan=session_plan,
        )
        expected_receipt_path = expected_final_evidence_paths(
            expected[1], expected[2]
        )["receipt"]
        if (
            (receipt["unit_index"], receipt["artifact_role"], receipt["seed"])
            != expected
            or receipt["authorization_artifact_sha256"]
            != authorization_artifact_sha256
            or receipt["evaluator_git_commit"]
            != session_plan["evaluator_git_commit"]
            or receipt["session_id"] != session_plan["session_id"]
            or receipt["session_plan_sha256"]
            != session_plan["session_plan_sha256"]
            or receipt["final_context"] != session_plan["final_context"]
            or receipt["runtime"] != session_plan["runtime"]
            or receipt["receipt_sha256"] in receipt_hashes
            or receipt["nll"]["artifact_sha256"] in nll_hashes
            or not isinstance(receipt_artifact, Mapping)
            or set(receipt_artifact) != {"path", "sha256"}
            or receipt_artifact["path"] != expected_receipt_path
            or not is_sha256(receipt_artifact["sha256"])
            or receipt_artifact["sha256"] in receipt_artifact_hashes
        ):
            raise ValueError("final evidence receipt order/identity was rotated")
        receipt_hashes.add(receipt["receipt_sha256"])
        receipt_artifact_hashes.add(receipt_artifact["sha256"])
        nll_hashes.add(receipt["nll"]["artifact_sha256"])
        normalized.append(dict(receipt))
        normalized_receipt_artifacts.append(dict(receipt_artifact))
    payload = {
        "authorization": {
            "artifact_sha256": authorization_artifact_sha256,
            "payload_sha256": authorization["authorization_sha256"],
        },
        "complete": True,
        "evaluator_git_commit": session_plan["evaluator_git_commit"],
        "evaluator_protocol": FINAL_EVALUATION_PROTOCOL_ID,
        "final_context": dict(session_plan["final_context"]),
        "final_test": dict(authorization["final_test"]),
        "integrity_pass": True,
        "kind": FINAL_EVIDENCE_KIND,
        "model_identity_order": [
            model["identity_sha256"] for model in authorization["models"]
        ],
        "receipt_artifacts": normalized_receipt_artifacts,
        "receipts": normalized,
        "role_to_artifact_role": dict(
            authorization["evaluation_contract"]["role_to_artifact_role"]
        ),
        "schema_version": 2,
        "seed_order": list(FINAL_SEEDS),
        "session_id": session_plan["session_id"],
        "session_plan": {
            "path": FINAL_SESSION_PATH,
            "sha256": session_plan["session_plan_sha256"],
        },
        "runtime": dict(session_plan["runtime"]),
        "unit_order": [
            {"artifact_role": role, "seed": seed, "unit_index": index}
            for index, role, seed in units
        ],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def validate_final_quality_evidence_manifest(
    manifest: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "authorization",
        "complete",
        "evaluator_git_commit",
        "evaluator_protocol",
        "final_context",
        "final_test",
        "integrity_pass",
        "kind",
        "manifest_sha256",
        "model_identity_order",
        "receipt_artifacts",
        "receipts",
        "role_to_artifact_role",
        "schema_version",
        "seed_order",
        "session_id",
        "session_plan",
        "runtime",
        "unit_order",
    }:
        raise ValueError("final evidence manifest is not the sealed schema")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("kind") != FINAL_EVIDENCE_KIND
        or manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or manifest.get("integrity_pass") is not True
        or not is_sha256(manifest.get("manifest_sha256"))
        or manifest["manifest_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError("final evidence manifest identity differs")
    rebuilt = build_final_quality_evidence_manifest(
        authorization=authorization,
        authorization_artifact_sha256=manifest["authorization"][
            "artifact_sha256"
        ],
        selection_lock=selection_lock,
        session_plan=session_plan,
        receipts=manifest["receipts"],
        receipt_artifacts=manifest["receipt_artifacts"],
    )
    if dict(manifest) != rebuilt:
        raise ValueError("final evidence manifest is not canonical")
