"""Fail-closed authorization lineage for Phase 3 confirmation seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .inference_selection_plan import PLAN_KIND, validate_selection_plan_v2
from .inference_selection_v2 import (
    SELECTION_LOCK_KIND,
    validate_selection_lock_v2,
)


INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_ONLY_SEEDS = (57721, 65537)
PRIMARY_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
AUTHORIZATION_KIND = "phase3_corrected_gate_i_confirmation_v1"
SELECTED_REFERENCE_AUTHORIZATION_KIND = (
    "selected_phase3_reference_confirmation_v2"
)
SELECTED_REFERENCE_AUTHORIZATION_KIND_V3 = (
    "selection_lock_selected_phase3_reference_confirmation_v3"
)
SELECTABLE_REFERENCE_POLICIES = (
    "spacebyte_spacelike",
    "entropy_threshold_full",
    "entropy_threshold_codepoint",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def confirmation_authorization_record(
    summary: Mapping[str, Any],
    *,
    summary_artifact_sha256: str,
    expected_source_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate corrected initial evidence and return its sealed identity."""

    if not _is_sha256(summary_artifact_sha256):
        raise ValueError("confirmation authorization summary SHA-256 is invalid")
    source_manifest = summary.get("source_manifest")
    source_manifest_sha256 = (
        source_manifest.get("sha256")
        if isinstance(source_manifest, Mapping)
        else None
    )
    if not _is_sha256(source_manifest_sha256):
        raise ValueError("confirmation authorization lacks source manifest SHA-256")
    if (
        expected_source_manifest_sha256 is not None
        and source_manifest_sha256 != expected_source_manifest_sha256
    ):
        raise ValueError(
            "confirmation authorization was not computed from the current "
            "pre-confirmation manifest"
        )
    if tuple(summary.get("seeds", ())) != INITIAL_SEEDS:
        raise ValueError("confirmation authorization needs exactly the initial seeds")
    if tuple(summary.get("policies", ())) != PRIMARY_POLICIES:
        raise ValueError("confirmation authorization needs exactly F/C/W evidence")
    if summary.get("gate_i", {}).get("overall_pass") is not True:
        raise ValueError("confirmation authorization requires corrected Gate I pass")
    if summary.get("integrity", {}).get("all_integrity_checks_pass") is not True:
        raise ValueError("confirmation authorization integrity is incomplete")
    summary_git_commit = summary.get("summary_git_commit")
    if summary_git_commit is not None and not isinstance(summary_git_commit, str):
        raise ValueError("confirmation authorization git commit is invalid")
    return {
        "authorization_kind": AUTHORIZATION_KIND,
        "summary_artifact_sha256": summary_artifact_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "summary_git_commit": summary_git_commit,
        "authorized_gate": "gate_i",
        "authorized_gate_status": "pass",
        "summary_seeds": list(INITIAL_SEEDS),
        "summary_policies": list(PRIMARY_POLICIES),
    }


def load_confirmation_authorization(
    summary_path: Path,
    *,
    expected_source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = (
        file_sha256(expected_source_manifest_path)
        if expected_source_manifest_path is not None
        else None
    )
    return confirmation_authorization_record(
        summary,
        summary_artifact_sha256=file_sha256(summary_path),
        expected_source_manifest_sha256=expected,
    )


def load_run_confirmation_authorization(
    summary_path: Path,
    manifest_path: Path,
    *,
    seeds: Sequence[int],
    policies: Sequence[str],
) -> dict[str, Any]:
    """Authorize a first invocation or an exact, already-recorded resume."""

    authorization = load_confirmation_authorization(summary_path)
    current_manifest_sha256 = file_sha256(manifest_path)
    if authorization["source_manifest_sha256"] == current_manifest_sha256:
        return authorization

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matching = [
        invocation
        for invocation in manifest.get("invocations", ())
        if isinstance(invocation, Mapping)
        and tuple(invocation.get("seeds", ())) == tuple(seeds)
        and tuple(invocation.get("policies", ())) == tuple(policies)
        and invocation.get("authorization") == authorization
    ]
    if not matching:
        raise ValueError(
            "confirmation authorization was not computed from the current "
            "pre-confirmation manifest and no exact authorized resume exists"
        )
    return authorization


def validate_confirmation_request(
    seeds: Sequence[int],
    policies: Sequence[str],
) -> None:
    """Keep the first confirmation invocation exactly preregistered."""

    if tuple(seeds) != CONFIRMATION_ONLY_SEEDS:
        raise ValueError(
            "primary confirmation must request both confirmation seeds in order"
        )
    if tuple(policies) != PRIMARY_POLICIES:
        raise ValueError("primary confirmation must request exactly F/C/W in order")


def validate_confirmation_invocations(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    """Require every confirmation seed/policy pair to share the sealed gate."""

    invocations = manifest.get("invocations")
    if not isinstance(invocations, list):
        raise ValueError("confirmation manifest lacks invocation provenance")
    expected = dict(authorization)
    for seed in CONFIRMATION_ONLY_SEEDS:
        for policy in PRIMARY_POLICIES:
            matches = [
                invocation
                for invocation in invocations
                if isinstance(invocation, Mapping)
                and seed in invocation.get("seeds", ())
                and policy in invocation.get("policies", ())
            ]
            if not matches:
                raise ValueError(
                    f"confirmation manifest lacks invocation for {seed}/{policy}"
                )
            if not any(invocation.get("authorization") == expected for invocation in matches):
                raise ValueError(
                    "confirmation invocation is not bound to the corrected Gate I "
                    f"summary: {seed}/{policy}"
                )


def selected_reference_authorization_record(
    selection_lock: Mapping[str, Any],
    selection_plan: Mapping[str, Any],
    primary_summary: Mapping[str, Any],
    *,
    selection_lock_artifact_sha256: str,
    selection_plan_artifact_sha256: str,
    primary_summary_artifact_sha256: str,
) -> dict[str, Any]:
    """Build the exact authorization for one calibration-selected S/E/EC policy."""

    validate_selection_lock_v2(selection_lock)
    validate_selection_plan_v2(selection_plan)
    hashes = (
        selection_lock_artifact_sha256,
        selection_plan_artifact_sha256,
        primary_summary_artifact_sha256,
    )
    if not all(_is_sha256(value) for value in hashes):
        raise ValueError("selected-reference authorization hashes are invalid")
    if (
        selection_lock.get("kind") != SELECTION_LOCK_KIND
        or selection_plan.get("kind") != PLAN_KIND
        or selection_lock.get("plan_sha256") != selection_plan_artifact_sha256
        or selection_plan.get("historical_screening", {})
        .get("primary_summary", {})
        .get("sha256")
        != primary_summary_artifact_sha256
    ):
        raise ValueError("selected-reference lock and plan lineage differ")

    expected_seeds = (*INITIAL_SEEDS, *CONFIRMATION_ONLY_SEEDS)
    primary_authorization = primary_summary.get("confirmation_authorization")
    ood = primary_summary.get("ood")
    if (
        tuple(primary_summary.get("seeds", ())) != expected_seeds
        or tuple(primary_summary.get("policies", ())) != PRIMARY_POLICIES
        or primary_summary.get("integrity", {}).get(
            "all_integrity_checks_pass"
        )
        is not True
        or primary_summary.get("gate_i", {}).get("overall_pass") is not True
        or primary_summary.get("gate_j", {}).get("overall_pass") is not True
        or not isinstance(primary_authorization, Mapping)
        or primary_authorization.get("authorization_kind") != AUTHORIZATION_KIND
        or not isinstance(ood, Mapping)
        or ood.get("gate_i_ood_guard", {}).get("pass") is not True
        or ood.get("integrity", {}).get("all_integrity_checks_pass") is not True
    ):
        raise ValueError(
            "selected-reference confirmation requires completed five-seed Gate J/OOD"
        )

    decision = selection_lock.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("selected-reference selection decision is missing")
    plan = decision.get("confirmation_plan")
    reference = decision.get("reference")
    reference_confirmation = (
        plan.get("phase3_reference") if isinstance(plan, Mapping) else None
    )
    if (
        decision.get("status")
        != "locked_pending_confirmation_and_new_final_test"
        or not isinstance(reference, Mapping)
        or not isinstance(reference_confirmation, Mapping)
        or reference_confirmation.get("authorization_kind")
        != SELECTED_REFERENCE_AUTHORIZATION_KIND
    ):
        raise ValueError("selection lock has no Phase 3 reference confirmation")
    policies = tuple(reference_confirmation.get("policies", ()))
    seeds = tuple(reference_confirmation.get("seeds", ()))
    policy = reference.get("policy")
    expected_auxiliary = (
        "entropy_router" if policy in SELECTABLE_REFERENCE_POLICIES[1:] else "none"
    )
    if (
        policies != (policy,)
        or policy not in SELECTABLE_REFERENCE_POLICIES
        or seeds != CONFIRMATION_ONLY_SEEDS
        or reference.get("model_family") != "phase3"
        or reference.get("patch_count") != 86
        or reference.get("runtime_policy") != policy
        or reference.get("requires_entropy_router")
        != (expected_auxiliary == "entropy_router")
        or reference_confirmation.get("required_auxiliary") != expected_auxiliary
    ):
        raise ValueError("selected-reference confirmation descriptor is inconsistent")

    return {
        "authorization_kind": SELECTED_REFERENCE_AUTHORIZATION_KIND,
        "calibration_evidence_manifest_sha256": selection_lock[
            "calibration_evidence_manifest_sha256"
        ],
        "final_test_seal_sha256": selection_lock["final_test_seal_sha256"],
        "policies": [policy],
        "primary_gate_j_summary_sha256": primary_summary_artifact_sha256,
        "required_auxiliary": expected_auxiliary,
        "seeds": list(CONFIRMATION_ONLY_SEEDS),
        "selection_decision_sha256": decision["decision_sha256"],
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_sha256": selection_lock["lock_sha256"],
        "selection_plan_artifact_sha256": selection_plan_artifact_sha256,
        "selection_plan_sha256": selection_plan["plan_sha256"],
    }


def load_selected_reference_authorization(
    selection_lock_path: Path,
    selection_plan_path: Path,
    primary_summary_path: Path,
) -> dict[str, Any]:
    for path in (
        selection_lock_path,
        selection_plan_path,
        primary_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection_lock = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    selection_plan = json.loads(selection_plan_path.read_text(encoding="utf-8"))
    primary_summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    authorization = selected_reference_authorization_record(
        selection_lock,
        selection_plan,
        primary_summary,
        selection_lock_artifact_sha256=file_sha256(selection_lock_path),
        selection_plan_artifact_sha256=file_sha256(selection_plan_path),
        primary_summary_artifact_sha256=file_sha256(primary_summary_path),
    )
    calibration_evidence_path = Path(
        selection_plan["execution_paths"]["calibration_evidence"]
    )
    final_test_seal_path = Path(selection_plan["final_test"]["seal_path"])
    if (
        not calibration_evidence_path.is_file()
        or file_sha256(calibration_evidence_path)
        != authorization["calibration_evidence_manifest_sha256"]
        or not final_test_seal_path.is_file()
        or file_sha256(final_test_seal_path)
        != authorization["final_test_seal_sha256"]
    ):
        raise ValueError("selected-reference transitive evidence lineage differs")
    return authorization


def validate_selected_reference_request(
    seeds: Sequence[int],
    policies: Sequence[str],
    authorization: Mapping[str, Any],
) -> None:
    if tuple(seeds) != CONFIRMATION_ONLY_SEEDS:
        raise ValueError("selected reference requires both confirmation seeds in order")
    if (
        authorization.get("authorization_kind")
        != SELECTED_REFERENCE_AUTHORIZATION_KIND
        or tuple(authorization.get("seeds", ())) != CONFIRMATION_ONLY_SEEDS
        or tuple(authorization.get("policies", ())) != tuple(policies)
        or len(tuple(policies)) != 1
        or tuple(policies)[0] not in SELECTABLE_REFERENCE_POLICIES
    ):
        raise ValueError("selected reference request differs from the selection lock")


def validate_selected_reference_invocation(
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    """Require one exact pre-training invocation for every selected seed/policy."""

    seeds = tuple(authorization.get("seeds", ()))
    policies = tuple(authorization.get("policies", ()))
    validate_selected_reference_request(seeds, policies, authorization)
    matches = [
        invocation
        for invocation in manifest.get("invocations", ())
        if isinstance(invocation, Mapping)
        and tuple(invocation.get("seeds", ())) == seeds
        and tuple(invocation.get("policies", ())) == policies
        and invocation.get("authorization") == dict(authorization)
    ]
    if len(matches) != 1:
        raise ValueError(
            "selected-reference manifest needs exactly one authorized invocation"
        )


def selected_reference_authorization_record_v3(
    selection_lock: Mapping[str, Any],
    selection_plan: Mapping[str, Any],
    *,
    selection_lock_artifact_sha256: str,
    selection_plan_artifact_sha256: str,
    calibration_evidence_artifact_sha256: str,
    final_test_seal_artifact_sha256: str,
) -> dict[str, Any]:
    """Authorize only the policy fixed by calibration, with no test gate input."""

    validate_selection_lock_v2(selection_lock)
    validate_selection_plan_v2(selection_plan)
    hashes = (
        selection_lock_artifact_sha256,
        selection_plan_artifact_sha256,
        calibration_evidence_artifact_sha256,
        final_test_seal_artifact_sha256,
    )
    if not all(_is_sha256(value) for value in hashes):
        raise ValueError("selected-reference v3 authorization hashes are invalid")
    if (
        selection_lock.get("kind") != SELECTION_LOCK_KIND
        or selection_plan.get("kind") != PLAN_KIND
        or selection_lock.get("plan_sha256") != selection_plan_artifact_sha256
        or selection_lock.get("calibration_evidence_manifest_sha256")
        != calibration_evidence_artifact_sha256
        or selection_lock.get("final_test_seal_sha256")
        != final_test_seal_artifact_sha256
        or selection_plan.get("plan_sha256") is None
    ):
        raise ValueError("selected-reference v3 lock and evidence lineage differ")
    decision = selection_lock.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("selected-reference v3 selection decision is missing")
    confirmation_plan = decision.get("confirmation_plan")
    reference = decision.get("reference")
    reference_confirmation = (
        confirmation_plan.get("phase3_reference")
        if isinstance(confirmation_plan, Mapping)
        else None
    )
    policy = reference.get("policy") if isinstance(reference, Mapping) else None
    required_auxiliary = (
        "entropy_router" if policy in SELECTABLE_REFERENCE_POLICIES[1:] else "none"
    )
    if (
        decision.get("status")
        != "locked_pending_confirmation_and_new_final_test"
        or not isinstance(reference, Mapping)
        or not isinstance(reference_confirmation, Mapping)
        or reference_confirmation.get("authorization_kind")
        != SELECTED_REFERENCE_AUTHORIZATION_KIND
        or tuple(reference_confirmation.get("seeds", ()))
        != CONFIRMATION_ONLY_SEEDS
        or tuple(reference_confirmation.get("policies", ())) != (policy,)
        or reference_confirmation.get("required_auxiliary")
        != required_auxiliary
        or policy not in SELECTABLE_REFERENCE_POLICIES
        or reference.get("model_family") != "phase3"
        or reference.get("patch_count") != 86
        or reference.get("runtime_policy") != policy
        or reference.get("requires_entropy_router")
        != (required_auxiliary == "entropy_router")
    ):
        raise ValueError("selected-reference v3 descriptor is inconsistent")
    return {
        "authorization_kind": SELECTED_REFERENCE_AUTHORIZATION_KIND_V3,
        "calibration_evidence_artifact_sha256": (
            calibration_evidence_artifact_sha256
        ),
        "final_test_seal_artifact_sha256": final_test_seal_artifact_sha256,
        "policies": [policy],
        "required_auxiliary": required_auxiliary,
        "result_inputs": {
            "calibration_selection": True,
            "final_test": False,
            "historical_screening_test": False,
            "latency": False,
        },
        "seeds": list(CONFIRMATION_ONLY_SEEDS),
        "selection_decision_sha256": decision["decision_sha256"],
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_sha256": selection_lock["lock_sha256"],
        "selection_plan_artifact_sha256": selection_plan_artifact_sha256,
        "selection_plan_sha256": selection_plan["plan_sha256"],
    }


def load_selected_reference_authorization_v3(
    selection_lock_path: Path,
    selection_plan_path: Path,
) -> dict[str, Any]:
    for path in (selection_lock_path, selection_plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection_lock = json.loads(selection_lock_path.read_text(encoding="utf-8"))
    selection_plan = json.loads(selection_plan_path.read_text(encoding="utf-8"))
    calibration_path = Path(
        selection_plan["execution_paths"]["calibration_evidence"]
    )
    final_test_seal_path = Path(selection_plan["final_test"]["seal_path"])
    for path in (calibration_path, final_test_seal_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return selected_reference_authorization_record_v3(
        selection_lock,
        selection_plan,
        selection_lock_artifact_sha256=file_sha256(selection_lock_path),
        selection_plan_artifact_sha256=file_sha256(selection_plan_path),
        calibration_evidence_artifact_sha256=file_sha256(calibration_path),
        final_test_seal_artifact_sha256=file_sha256(final_test_seal_path),
    )


def validate_selected_reference_request_v3(
    seeds: Sequence[int],
    policies: Sequence[str],
    authorization: Mapping[str, Any],
) -> None:
    if (
        tuple(seeds) != CONFIRMATION_ONLY_SEEDS
        or authorization.get("authorization_kind")
        != SELECTED_REFERENCE_AUTHORIZATION_KIND_V3
        or tuple(authorization.get("seeds", ())) != CONFIRMATION_ONLY_SEEDS
        or tuple(authorization.get("policies", ())) != tuple(policies)
        or len(tuple(policies)) != 1
        or tuple(policies)[0] not in SELECTABLE_REFERENCE_POLICIES
        or authorization.get("result_inputs")
        != {
            "calibration_selection": True,
            "final_test": False,
            "historical_screening_test": False,
            "latency": False,
        }
    ):
        raise ValueError("selected reference request differs from the v3 lock")
