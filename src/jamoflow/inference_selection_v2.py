"""Canonical calibration-only rate and comparator selection for inference v2."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Mapping

from .compute_conversion import (
    CONVERSION_POLICIES,
    CONVERSION_RATES,
    conversion_policy,
    select_rate_from_calibration,
)
from .phase3 import PHASE3_POLICIES


INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_SEEDS = (57721, 65537)
C86 = "causal_codepoint_grid"
PRIMARY_CONFIRMED_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
CALIBRATION_POLICY_ORDER = (*PHASE3_POLICIES, *CONVERSION_POLICIES)
SELECTION_LOCK_KIND = "phase3_inference_selection_lock_v2"
BROAD_REFERENCE_CALIBRATION_FUTILITY_MARGIN_BPB = 0.010
BROAD_REFERENCE_CALIBRATION_FUTILITY_MINIMUM_SEEDS = 2
CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER = (
    "scripts/seal_inference_initial_model_identity_v2.py",
    "scripts/reconstruct_inference_calibration_v2.py",
    "scripts/seal_inference_selection_lock_v2.py",
    "scripts/run_phase3.py",
    "scripts/run_phase3_compute_conversion.py",
    "scripts/reconstruct_inference_confirmation_calibration_v2.py",
    "scripts/seal_inference_post_confirmation_authorization_v2.py",
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
    "pyproject.toml",
)
SELECTION_RECOMPUTATION_PROTOCOL = (
    "jamoflow-selection-lock-calibration-replay-v1"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _canonical_calibration_payload(
    calibration_bpb: Mapping[int, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    values = _validate_calibration_matrix(calibration_bpb)
    return {
        str(seed): {
            policy: values[seed][policy]
            for policy in CALIBRATION_POLICY_ORDER
        }
        for seed in INITIAL_SEEDS
    }


def build_independent_calibration_recomputation_v2(
    calibration_bpb: Mapping[int, Mapping[str, float]],
    *,
    nll_array_sha256_by_seed_policy: Mapping[int, Mapping[str, str]],
    evaluator_git_commit: str,
    verification_git_commit: str,
    environment_sha256: str,
    implementation_manifest_sha256: str,
) -> dict[str, object]:
    """Seal the second, lock-time causal-forward replay of all 30 units."""

    canonical_bpb = _canonical_calibration_payload(calibration_bpb)
    if not (_is_git_commit(evaluator_git_commit) and _is_git_commit(verification_git_commit)):
        raise ValueError("selection replay Git identities are malformed")
    if not (
        _is_sha256(environment_sha256)
        and _is_sha256(implementation_manifest_sha256)
    ):
        raise ValueError("selection replay environment/implementation is malformed")
    if tuple(sorted(nll_array_sha256_by_seed_policy)) != INITIAL_SEEDS:
        raise ValueError("selection replay NLL seed set is not exact")
    nll_hashes: dict[str, dict[str, str]] = {}
    for seed in INITIAL_SEEDS:
        row = nll_array_sha256_by_seed_policy[seed]
        if not isinstance(row, Mapping) or set(row) != set(CALIBRATION_POLICY_ORDER):
            raise ValueError("selection replay NLL policy set is not exact")
        if not all(_is_sha256(row[policy]) for policy in CALIBRATION_POLICY_ORDER):
            raise ValueError("selection replay NLL hash is malformed")
        nll_hashes[str(seed)] = {
            policy: row[policy] for policy in CALIBRATION_POLICY_ORDER
        }
    unsigned: dict[str, object] = {
        "bpb_matrix_sha256": sha256(
            _canonical_json_bytes(canonical_bpb)
        ).hexdigest(),
        "comparison": "bitwise_float32_array_equal",
        "device": "mps",
        "environment_sha256": environment_sha256,
        "evaluator_git_commit": evaluator_git_commit,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "nll_array_sha256_by_seed_policy": nll_hashes,
        "receipt_count": len(INITIAL_SEEDS) * len(CALIBRATION_POLICY_ORDER),
        "recomputation_protocol": SELECTION_RECOMPUTATION_PROTOCOL,
        "status": "pass",
        "verification_git_commit": verification_git_commit,
    }
    return {
        **unsigned,
        "verification_sha256": sha256(
            _canonical_json_bytes(unsigned)
        ).hexdigest(),
    }


def validate_independent_calibration_recomputation_v2(
    replay: Mapping[str, object],
    *,
    decision: Mapping[str, object],
) -> None:
    expected_keys = {
        "bpb_matrix_sha256",
        "comparison",
        "device",
        "environment_sha256",
        "evaluator_git_commit",
        "implementation_manifest_sha256",
        "nll_array_sha256_by_seed_policy",
        "receipt_count",
        "recomputation_protocol",
        "status",
        "verification_git_commit",
        "verification_sha256",
    }
    if not isinstance(replay, Mapping) or set(replay) != expected_keys:
        raise ValueError("selection replay is not the sealed schema")
    unsigned = {
        key: value for key, value in replay.items()
        if key != "verification_sha256"
    }
    if (
        replay.get("comparison") != "bitwise_float32_array_equal"
        or replay.get("device") != "mps"
        or replay.get("recomputation_protocol")
        != SELECTION_RECOMPUTATION_PROTOCOL
        or replay.get("status") != "pass"
        or replay.get("receipt_count")
        != len(INITIAL_SEEDS) * len(CALIBRATION_POLICY_ORDER)
        or not _is_git_commit(replay.get("evaluator_git_commit"))
        or not _is_git_commit(replay.get("verification_git_commit"))
        or not _is_sha256(replay.get("environment_sha256"))
        or not _is_sha256(replay.get("implementation_manifest_sha256"))
        or not _is_sha256(replay.get("bpb_matrix_sha256"))
        or replay.get("verification_sha256")
        != sha256(_canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise ValueError("selection replay identity is invalid")
    calibration = decision.get("calibration_bpb_by_seed_policy")
    if not isinstance(calibration, Mapping):
        raise ValueError("selection replay lacks the decision calibration matrix")
    restored = {
        int(seed): row
        for seed, row in calibration.items()
        if isinstance(seed, str) and seed.isdigit() and isinstance(row, Mapping)
    }
    canonical_bpb = _canonical_calibration_payload(restored)  # type: ignore[arg-type]
    if replay["bpb_matrix_sha256"] != sha256(
        _canonical_json_bytes(canonical_bpb)
    ).hexdigest():
        raise ValueError("selection replay BPB matrix differs from the decision")
    hashes = replay.get("nll_array_sha256_by_seed_policy")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != {str(seed) for seed in INITIAL_SEEDS}
    ):
        raise ValueError("selection replay NLL seed set differs")
    for seed in INITIAL_SEEDS:
        row = hashes.get(str(seed))
        if not isinstance(row, Mapping) or set(row) != set(CALIBRATION_POLICY_ORDER):
            raise ValueError("selection replay NLL policy set differs")
        if not all(_is_sha256(row.get(policy)) for policy in CALIBRATION_POLICY_ORDER):
            raise ValueError("selection replay NLL hashes are invalid")


def _descriptor(policy: str, *, selected_rate: int) -> dict[str, object]:
    if policy in PHASE3_POLICIES:
        return {
            "model_family": "phase3",
            "patch_count": 86,
            "policy": policy,
            "requires_entropy_router": policy
            in {"entropy_threshold_full", "entropy_threshold_codepoint"},
            "runtime_policy": policy,
        }
    codepoint = conversion_policy("codepoint", selected_rate)
    whitespace = conversion_policy("whitespace", selected_rate)
    if policy not in (codepoint, whitespace):
        raise ValueError("selection descriptor policy is outside the locked pool")
    return {
        "model_family": "compute_conversion",
        "patch_count": selected_rate,
        "policy": policy,
        "requires_entropy_router": False,
        "runtime_policy": (
            "causal_codepoint_grid" if policy == codepoint else "causal_whitespace_grid"
        ),
    }


def _validate_calibration_matrix(
    calibration_bpb: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    if tuple(sorted(calibration_bpb)) != INITIAL_SEEDS:
        raise ValueError("selection-v2 requires exactly the three initial seeds")
    validated: dict[int, dict[str, float]] = {}
    expected = set(CALIBRATION_POLICY_ORDER)
    for seed in INITIAL_SEEDS:
        row = calibration_bpb[seed]
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(
                f"selection-v2 calibration policy set is not exact for seed {seed}"
            )
        values: dict[str, float] = {}
        for policy in CALIBRATION_POLICY_ORDER:
            value = row[policy]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    f"selection-v2 calibration BPB is invalid: {seed}/{policy}"
                )
            values[policy] = float(value)
        validated[seed] = values
    return validated


def build_selection_decision_v2(
    calibration_bpb: Mapping[int, Mapping[str, float]],
) -> dict[str, object]:
    """Select one rate and one reference without accepting test/latency inputs."""

    values = _validate_calibration_matrix(calibration_bpb)
    primary_codepoint = {seed: values[seed][C86] for seed in INITIAL_SEEDS}
    conversion = {
        seed: {
            policy: values[seed][policy]
            for policy in CONVERSION_POLICIES
        }
        for seed in INITIAL_SEEDS
    }
    rate_selection = select_rate_from_calibration(conversion, primary_codepoint)
    base: dict[str, object] = {
        "algorithm_id": "jamoflow-inference-selection-v2",
        "calibration_bpb_by_seed_policy": {
            str(seed): {
                policy: values[seed][policy]
                for policy in CALIBRATION_POLICY_ORDER
            }
            for seed in INITIAL_SEEDS
        },
        "calibration_policy_order": list(CALIBRATION_POLICY_ORDER),
        "confirmation_seed_order": list(CONFIRMATION_SEEDS),
        "rate_order": list(CONVERSION_RATES),
        "rate_selection": rate_selection.to_dict(),
        "schema_version": 2,
        "seed_order": list(INITIAL_SEEDS),
        "selection_uses": {
            "calibration": True,
            "final_test": False,
            "historical_screening_test": False,
            "latency": False,
        },
    }
    selected_rate = rate_selection.selected_rate
    if selected_rate is None:
        payload = {
            **base,
            "candidate": None,
            "confirmation_plan": None,
            "matched_efficiency_baseline": _descriptor(C86, selected_rate=64),
            "reference": None,
            "reference_selection": None,
            "status": "terminal_no_rate",
        }
        payload["decision_sha256"] = sha256(_canonical_json_bytes(payload)).hexdigest()
        return payload

    selected_codepoint = conversion_policy("codepoint", selected_rate)
    selected_whitespace = conversion_policy("whitespace", selected_rate)
    reference_order = (*PHASE3_POLICIES, selected_codepoint)
    means = {
        policy: _mean([values[seed][policy] for seed in INITIAL_SEEDS])
        for policy in reference_order
    }
    reference = min(
        reference_order,
        key=lambda policy: (means[policy], reference_order.index(policy)),
    )
    candidate_mean = _mean(
        [values[seed][selected_whitespace] for seed in INITIAL_SEEDS]
    )
    c86_mean = _mean([values[seed][C86] for seed in INITIAL_SEEDS])
    reference_confirmation = None
    if (
        reference in PHASE3_POLICIES
        and reference not in PRIMARY_CONFIRMED_POLICIES
    ):
        reference_confirmation = {
            "authorization_kind": "selected_phase3_reference_confirmation_v2",
            "policies": [reference],
            "required_auxiliary": (
                "entropy_router"
                if reference
                in {"entropy_threshold_full", "entropy_threshold_codepoint"}
                else "none"
            ),
            "seeds": list(CONFIRMATION_SEEDS),
        }
    candidate_minus_reference = [
        values[seed][selected_whitespace] - values[seed][reference]
        for seed in INITIAL_SEEDS
    ]
    reference_screen_mean = _mean(candidate_minus_reference)
    reference_screen_within = sum(
        effect <= BROAD_REFERENCE_CALIBRATION_FUTILITY_MARGIN_BPB
        for effect in candidate_minus_reference
    )
    reference_screen_pass = bool(
        reference_screen_mean <= BROAD_REFERENCE_CALIBRATION_FUTILITY_MARGIN_BPB
        and reference_screen_within
        >= BROAD_REFERENCE_CALIBRATION_FUTILITY_MINIMUM_SEEDS
    )
    confirmation_plan = {
        "compute_conversion": {
            "authorization_kind": "compute_conversion_confirmation_v2",
            "policies": [selected_codepoint, selected_whitespace],
            "selected_rate": selected_rate,
            "seeds": list(CONFIRMATION_SEEDS),
        },
        "phase3_reference": (
            reference_confirmation if reference_screen_pass else None
        ),
    }
    payload = {
        **base,
        "candidate": {
            **_descriptor(selected_whitespace, selected_rate=selected_rate),
            "initial_mean_calibration_bpb": candidate_mean,
        },
        "confirmation_plan": confirmation_plan,
        "initial_candidate_minus_c86_mean_calibration_bpb": (
            candidate_mean - c86_mean
        ),
        "initial_candidate_minus_reference_mean_calibration_bpb": (
            candidate_mean - means[reference]
        ),
        "matched_efficiency_baseline": {
            **_descriptor(C86, selected_rate=selected_rate),
            "initial_mean_calibration_bpb": c86_mean,
        },
        "reference": {
            **_descriptor(reference, selected_rate=selected_rate),
            "initial_mean_calibration_bpb": means[reference],
        },
        "reference_selection": {
            "candidate_order": list(reference_order),
            "criterion": "lowest initial-three-seed mean calibration BPB",
            "mean_calibration_bpb": means,
            "selected_policy": reference,
            "tie_break": "first policy in fixed candidate order on exact tie",
        },
        "broad_reference_calibration_screen": {
            "candidate_minus_reference_effects_bpb": (
                candidate_minus_reference
            ),
            "margin_bpb": BROAD_REFERENCE_CALIBRATION_FUTILITY_MARGIN_BPB,
            "mean_candidate_minus_reference_bpb": reference_screen_mean,
            "minimum_seed_count_within_margin": (
                BROAD_REFERENCE_CALIBRATION_FUTILITY_MINIMUM_SEEDS
            ),
            "pass": reference_screen_pass,
            "seed_count_within_margin": reference_screen_within,
        },
        "broad_reference_evaluation_status": (
            "eligible_pending_confirmation"
            if reference_screen_pass
            else "not_authorized_calibration_futility"
        ),
        "status": "locked_pending_confirmation_and_new_final_test",
    }
    payload["decision_sha256"] = sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def validate_selection_decision_v2(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("selection-v2 decision must be an object")
    calibration = payload.get("calibration_bpb_by_seed_policy")
    if not isinstance(calibration, Mapping):
        raise ValueError("selection-v2 decision lacks calibration evidence")
    restored: dict[int, Mapping[str, float]] = {}
    for key, row in calibration.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(row, Mapping):
            raise ValueError("selection-v2 calibration evidence is malformed")
        restored[int(key)] = row  # type: ignore[assignment]
    expected = build_selection_decision_v2(restored)
    if dict(payload) != expected:
        raise ValueError("selection-v2 decision is not the canonical reconstruction")


def build_selection_lock_v2(
    decision: Mapping[str, object],
    *,
    plan_sha256: str,
    calibration_evidence_manifest_sha256: str,
    final_test_seal_sha256: str,
    initial_model_identity_lock_sha256: str,
    independent_calibration_recomputation: Mapping[str, object],
) -> dict[str, object]:
    """Bind the canonical decision to pre-existing plan/evidence/test identities."""

    validate_selection_decision_v2(decision)
    hashes = (
        plan_sha256,
        calibration_evidence_manifest_sha256,
        final_test_seal_sha256,
        initial_model_identity_lock_sha256,
    )
    if not all(_is_sha256(value) for value in hashes):
        raise ValueError("selection-v2 lock identities must be SHA-256 digests")
    validate_independent_calibration_recomputation_v2(
        independent_calibration_recomputation,
        decision=decision,
    )
    payload: dict[str, object] = {
        "calibration_evidence_manifest_sha256": (
            calibration_evidence_manifest_sha256
        ),
        "decision": dict(decision),
        "final_test_seal_sha256": final_test_seal_sha256,
        "independent_calibration_recomputation": dict(
            independent_calibration_recomputation
        ),
        "initial_model_identity_lock_sha256": (
            initial_model_identity_lock_sha256
        ),
        "kind": SELECTION_LOCK_KIND,
        "plan_sha256": plan_sha256,
        "schema_version": 2,
    }
    payload["lock_sha256"] = sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def validate_selection_lock_v2(payload: Mapping[str, object]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "calibration_evidence_manifest_sha256",
        "decision",
        "final_test_seal_sha256",
        "independent_calibration_recomputation",
        "initial_model_identity_lock_sha256",
        "kind",
        "lock_sha256",
        "plan_sha256",
        "schema_version",
    }:
        raise ValueError("selection-v2 lock is not the sealed schema")
    if (
        payload.get("kind") != SELECTION_LOCK_KIND
        or payload.get("schema_version") != 2
        or not all(
            _is_sha256(payload.get(key))
            for key in (
                "calibration_evidence_manifest_sha256",
                "final_test_seal_sha256",
                "lock_sha256",
                "initial_model_identity_lock_sha256",
                "plan_sha256",
            )
        )
        or not isinstance(payload.get("decision"), Mapping)
        or not isinstance(
            payload.get("independent_calibration_recomputation"), Mapping
        )
    ):
        raise ValueError("selection-v2 lock identity is invalid")
    validate_selection_decision_v2(payload["decision"])  # type: ignore[arg-type]
    validate_independent_calibration_recomputation_v2(
        payload["independent_calibration_recomputation"],  # type: ignore[arg-type]
        decision=payload["decision"],  # type: ignore[arg-type]
    )
    unsigned = {key: value for key, value in payload.items() if key != "lock_sha256"}
    if payload["lock_sha256"] != sha256(_canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError("selection-v2 lock hash does not match its payload")
