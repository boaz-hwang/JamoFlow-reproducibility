"""Immutable quality lock derived only from sealed-final receipt arrays."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .document_inference import DocumentWindowMap
from .inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_EVIDENCE_PATH,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_SEEDS,
    canonical_sha256,
    is_sha256,
    validate_final_evaluation_authorization_v2,
)
from .inference_final_quality_evidence_v2 import (
    FINAL_NLL_VERIFICATION_COMPARISON,
    FINAL_SESSION_PATH,
    validate_final_quality_evidence_manifest,
    validate_final_quality_session_plan,
)
from .inference_final_quality_v2 import (
    BROAD_REFERENCE_ROLE,
    FINAL_BOOTSTRAP_REPETITIONS,
    FINAL_BOOTSTRAP_SEED,
    final_quality_gate_v2,
)


FINAL_QUALITY_LOCK_KIND = "phase3_inference_final_quality_lock_v2"
FINAL_QUALITY_LOCK_PROTOCOL_VERSION = 2
PRIMARY_TIMING_AUTHORIZATION_KEY = (
    "candidate_vs_matched_efficiency_baseline"
)


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _tracked_artifact(
    value: Mapping[str, Any],
    *,
    expected_path: str,
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"git_commit", "path", "sha256"}
        or value["path"] != expected_path
        or not is_sha256(value["sha256"])
        or not _git_commit(value["git_commit"])
    ):
        raise ValueError(f"final quality artifact differs: {expected_path}")
    return dict(value)


def _model_for_artifact_role(
    authorization: Mapping[str, Any],
    artifact_role: str,
) -> Mapping[str, Any]:
    matches = [
        model
        for model in authorization["models"]
        if model["artifact_role"] == artifact_role
    ]
    if len(matches) != 1:
        raise ValueError("final quality role has no unique physical model")
    return matches[0]


def _validate_document_map(
    document_window_map: DocumentWindowMap,
    final_context: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = document_window_map.metadata()
    if (
        document_window_map.sequence_count != 62_500
        or document_window_map.sequence_length != 512
        or not document_window_map.coverage_pass
        or metadata["document_assignment_sha256"]
        != final_context["document_assignment_sha256"]
        or document_window_map.layout_sha256
        != final_context["document_layout_sha256"]
        or document_window_map.eligible_sequence_count
        != final_context["eligible_sequence_count"]
    ):
        raise ValueError("final quality document map differs from evidence")
    return metadata


def _validated_arrays(
    evidence: Mapping[str, Any],
    arrays_by_receipt_sha256: Mapping[str, np.ndarray],
) -> dict[str, dict[int, np.ndarray]]:
    receipts = evidence["receipts"]
    expected_hashes = tuple(receipt["receipt_sha256"] for receipt in receipts)
    if tuple(arrays_by_receipt_sha256) != expected_hashes:
        raise ValueError("final quality NLL array order/set differs from receipts")
    arrays: dict[str, dict[int, np.ndarray]] = {}
    for receipt in receipts:
        values = np.asarray(arrays_by_receipt_sha256[receipt["receipt_sha256"]])
        if (
            values.dtype != np.float32
            or values.shape != (62_500,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or _array_sha256(values) != receipt["nll"]["array_sha256"]
        ):
            raise ValueError("final quality NLL array differs from its receipt")
        arrays.setdefault(receipt["artifact_role"], {})[receipt["seed"]] = values
    if any(tuple(sorted(row)) != FINAL_SEEDS for row in arrays.values()):
        raise ValueError("final quality physical model has an incomplete seed set")
    return arrays


def _pair_authorization(
    *,
    authorization: Mapping[str, Any],
    left_role: str,
    right_role: str,
    authorized: bool,
    criterion: str,
    quality_gate_sha256: str,
) -> dict[str, Any]:
    aliases = authorization["evaluation_contract"]["role_to_artifact_role"]
    left_artifact_role = aliases[left_role]
    right_artifact_role = aliases[right_role]
    left = _model_for_artifact_role(authorization, left_artifact_role)
    right = _model_for_artifact_role(authorization, right_artifact_role)
    if left["identity_sha256"] == right["identity_sha256"]:
        raise ValueError("final timing pair aliases one physical model")
    return {
        "authorized": bool(authorized),
        "criterion": criterion,
        "left_artifact_role": left_artifact_role,
        "left_model_identity_sha256": left["identity_sha256"],
        "left_logical_role": left_role,
        "quality_gate_sha256": quality_gate_sha256,
        "right_artifact_role": right_artifact_role,
        "right_model_identity_sha256": right["identity_sha256"],
        "right_logical_role": right_role,
    }


def build_final_quality_lock_v2(
    *,
    authorization: Mapping[str, Any],
    authorization_artifact: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_artifact: Mapping[str, Any],
    quality_lock_base_git_commit: str,
    document_window_map: DocumentWindowMap,
    arrays_by_receipt_sha256: Mapping[str, np.ndarray],
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
    validate_final_quality_evidence_manifest(
        evidence,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
    )
    auth_artifact = _tracked_artifact(
        authorization_artifact,
        expected_path=FINAL_AUTHORIZATION_PATH,
    )
    final_evidence_artifact = _tracked_artifact(
        evidence_artifact,
        expected_path=FINAL_EVIDENCE_PATH,
    )
    if (
        not _git_commit(quality_lock_base_git_commit)
        or evidence["authorization"]["artifact_sha256"]
        != auth_artifact["sha256"]
        or session_plan["authorization"]["artifact_sha256"]
        != auth_artifact["sha256"]
    ):
        raise ValueError("final quality upstream identity differs")
    document_metadata = _validate_document_map(
        document_window_map,
        evidence["final_context"],
    )
    physical_arrays = _validated_arrays(evidence, arrays_by_receipt_sha256)
    role_order = tuple(
        authorization["evaluation_contract"]["evaluation_role_order"]
    )
    aliases = authorization["evaluation_contract"]["role_to_artifact_role"]
    losses_by_role = {
        role: physical_arrays[aliases[role]] for role in role_order
    }
    role_descriptors = {
        role: authorization["evaluation_contract"]["logical_roles"][role]
        for role in role_order
    }
    gate = final_quality_gate_v2(
        losses_by_role,
        role_descriptors=role_descriptors,
        document_window_map=document_window_map,
        targets_per_sequence=511,
        bootstrap_repetitions=FINAL_BOOTSTRAP_REPETITIONS,
        bootstrap_seed=FINAL_BOOTSTRAP_SEED,
    )
    gate_sha256 = canonical_sha256(gate)
    matched_pass = bool(
        gate["candidate_vs_matched_efficiency_baseline"]["overall_pass"]
    )
    mechanism_pass = bool(
        gate["mechanism_candidate_vs_same_rate_codepoint"]["overall_pass"]
    )
    timing = {
        "candidate_vs_matched_efficiency_baseline": _pair_authorization(
            authorization=authorization,
            left_role="candidate",
            right_role="matched_efficiency_baseline",
            authorized=matched_pass,
            criterion="sealed final 0.010 BPB noninferiority",
            quality_gate_sha256=gate_sha256,
        ),
        "candidate_vs_same_rate_codepoint_control": _pair_authorization(
            authorization=authorization,
            left_role="candidate",
            right_role="same_rate_codepoint_control",
            authorized=mechanism_pass,
            criterion="sealed final preregistered W-minus-C mechanism replication",
            quality_gate_sha256=gate_sha256,
        ),
    }
    broad_gate = gate["broad_candidate_vs_strongest_reference"]
    if BROAD_REFERENCE_ROLE in role_order:
        timing["candidate_vs_broad_reference"] = _pair_authorization(
            authorization=authorization,
            left_role="candidate",
            right_role=BROAD_REFERENCE_ROLE,
            authorized=bool(broad_gate["overall_pass"]),
            criterion="sealed final broad-reference 0.010 BPB noninferiority",
            quality_gate_sha256=gate_sha256,
        )
    # The user's publication-value criterion is measured inference efficiency at
    # matched quality.  The W-vs-same-rate-C mechanism contrast remains a
    # preregistered attribution gate, but it must not suppress the primary
    # candidate-vs-C86 timing experiment once noninferiority is established.
    primary_authorized = matched_pass
    status = (
        "pass_full_final_quality_v2"
        if bool(gate["overall_pass"])
        else (
            "pass_matched_quality_only"
            if matched_pass
            else "fail_final_quality_v2"
        )
    )
    payload = {
        "authorization_artifact": auth_artifact,
        "broad_reference_policy": (
            role_descriptors[BROAD_REFERENCE_ROLE]["policy"]
            if BROAD_REFERENCE_ROLE in role_order
            else None
        ),
        "document_window_map": document_metadata,
        "evidence_artifact": final_evidence_artifact,
        "evidence_manifest_sha256": evidence["manifest_sha256"],
        "final_quality_gate": gate,
        "final_quality_gate_sha256": gate_sha256,
        "kind": FINAL_QUALITY_LOCK_KIND,
        "independent_nll_recomputation": {
            "batch_size": session_plan["runtime"]["batch_size"],
            "comparison": FINAL_NLL_VERIFICATION_COMPARISON,
            "device": session_plan["runtime"]["device"],
            "pass": True,
            "per_receipt": [
                {
                    "array_sha256": _array_sha256(
                        arrays_by_receipt_sha256[receipt["receipt_sha256"]]
                    ),
                    "receipt_sha256": receipt["receipt_sha256"],
                }
                for receipt in evidence["receipts"]
            ],
            "runtime": dict(session_plan["runtime"]),
            "was_predeclared_before_first_final_loss": (
                session_plan["verification_contract"]
                == {
                    "comparison": FINAL_NLL_VERIFICATION_COMPARISON,
                    "device": "mps",
                    "independent_second_model_forward_required": True,
                    "verifier_batch_size": session_plan["runtime"]["batch_size"],
                }
            ),
        },
        "model_identity_order": evidence["model_identity_order"],
        "nll_artifacts": [
            {
                "array_sha256": receipt["nll"]["array_sha256"],
                "artifact_path": receipt["nll"]["artifact_path"],
                "artifact_sha256": receipt["nll"]["artifact_sha256"],
                "artifact_role": receipt["artifact_role"],
                "receipt_sha256": receipt["receipt_sha256"],
                "seed": receipt["seed"],
            }
            for receipt in evidence["receipts"]
        ],
        "primary_publication_timing_authorized": primary_authorized,
        "primary_timing_authorization_key": PRIMARY_TIMING_AUTHORIZATION_KEY,
        "protocol_version": FINAL_QUALITY_LOCK_PROTOCOL_VERSION,
        "quality_lock_base_git_commit": quality_lock_base_git_commit,
        "quality_lock_path": FINAL_QUALITY_LOCK_PATH,
        "schema_version": 2,
        "seed_order": list(FINAL_SEEDS),
        "selection_lock_sha256": selection_lock["lock_sha256"],
        "session_plan": {
            "path": FINAL_SESSION_PATH,
            "sha256": session_plan["session_plan_sha256"],
        },
        "status": status,
        "timing_authorizations": timing,
    }
    payload["quality_lock_sha256"] = canonical_sha256(payload)
    return payload


def validate_final_quality_lock_v2(
    quality_lock: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    document_window_map: DocumentWindowMap,
    arrays_by_receipt_sha256: Mapping[str, np.ndarray],
) -> None:
    if not isinstance(quality_lock, Mapping) or set(quality_lock) != {
        "authorization_artifact",
        "broad_reference_policy",
        "document_window_map",
        "evidence_artifact",
        "evidence_manifest_sha256",
        "final_quality_gate",
        "final_quality_gate_sha256",
        "independent_nll_recomputation",
        "kind",
        "model_identity_order",
        "nll_artifacts",
        "primary_publication_timing_authorized",
        "primary_timing_authorization_key",
        "protocol_version",
        "quality_lock_base_git_commit",
        "quality_lock_path",
        "quality_lock_sha256",
        "schema_version",
        "seed_order",
        "selection_lock_sha256",
        "session_plan",
        "status",
        "timing_authorizations",
    }:
        raise ValueError("final quality lock is not the sealed schema")
    unsigned = {
        key: value
        for key, value in quality_lock.items()
        if key != "quality_lock_sha256"
    }
    if (
        quality_lock.get("kind") != FINAL_QUALITY_LOCK_KIND
        or quality_lock.get("schema_version") != 2
        or quality_lock.get("protocol_version")
        != FINAL_QUALITY_LOCK_PROTOCOL_VERSION
        or not is_sha256(quality_lock.get("quality_lock_sha256"))
        or quality_lock["quality_lock_sha256"] != canonical_sha256(unsigned)
    ):
        raise ValueError("final quality lock identity differs")
    rebuilt = build_final_quality_lock_v2(
        authorization=authorization,
        authorization_artifact=quality_lock["authorization_artifact"],
        selection_lock=selection_lock,
        session_plan=session_plan,
        evidence=evidence,
        evidence_artifact=quality_lock["evidence_artifact"],
        quality_lock_base_git_commit=quality_lock[
            "quality_lock_base_git_commit"
        ],
        document_window_map=document_window_map,
        arrays_by_receipt_sha256=arrays_by_receipt_sha256,
    )
    if dict(quality_lock) != rebuilt:
        raise ValueError("final quality lock is not canonical")
