#!/usr/bin/env python3
"""Seal exact five-seed model bundles before opening the new final test."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import (
    publish_no_clobber,
    validate_seal_envelope,
)
from jamoflow.inference_calibration_evidence import (
    validate_calibration_evidence_manifest,
)
from jamoflow.inference_confirmation_evidence_v2 import (
    CONFIRMATION_SEEDS,
    confirmation_completion_path,
    required_confirmation_completion_families,
    required_confirmation_models,
    validate_confirmation_evidence_manifest,
    validate_confirmation_training_completion,
    validate_receipts_against_training_completions,
    validate_training_report_against_completion,
)
from jamoflow.inference_confirmation_replay_v2 import (
    confirmation_entropy_matrices_and_auxiliary,
    load_confirmation_calibration_context,
    replay_confirmation_unit,
    validate_confirmation_replay_against_receipt,
)
from jamoflow.inference_final_authorization_v2 import (
    CONFIRMATION_EVIDENCE_PATH,
    FINAL_ARTIFACT_ROOT,
    FINAL_AUTHORIZATION_PATH,
    FINAL_EVIDENCE_PATH,
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_ROUTER_PARAMETER_COUNT,
    FINAL_SEEDS,
    FINAL_TEST_MANIFEST_PATH,
    FINAL_TEST_OUTPUT_PATH,
    FINAL_TEST_SEAL_PATH,
    IMPLEMENTATION_FILE_ORDER,
    SELECTION_EVIDENCE_PATH,
    SELECTION_LOCK_PATH,
    SELECTION_PLAN_PATH,
    build_final_evaluation_authorization_v2,
    build_final_model_identity,
    canonical_sha256,
    expected_model_paths,
    is_sha256,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_final_quality_v2 import resolve_final_evaluation_roles
from jamoflow.inference_initial_model_identity_v2 import (
    INITIAL_MODEL_IDENTITY_LOCK_PATH,
    runtime_environment_v2,
    validate_current_implementation_v2,
    validate_initial_model_identity_lock_v2,
    validate_selection_lock_identity_binding_v2,
)
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_selection_plan import validate_selection_plan_v2
from jamoflow.inference_selection_plan import PHASE3_PRIMARY_SUMMARY_PATH
from jamoflow.inference_selection_v2 import (
    INITIAL_SEEDS,
    PRIMARY_CONFIRMED_POLICIES,
    build_selection_decision_v2,
    validate_selection_lock_v2,
)
from jamoflow.neural_model import build_main_model, build_router, parameter_count
from jamoflow.neural_training import resolve_device
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    THRESHOLD_POLICIES,
)
from jamoflow.publication_reference import entropy_policy_definition_sha256
from jamoflow.phase3_confirmation import (
    SELECTABLE_REFERENCE_POLICIES,
    selected_reference_authorization_record_v3,
)


PLAN_PATH = Path(SELECTION_PLAN_PATH)
INITIAL_EVIDENCE_PATH = Path(SELECTION_EVIDENCE_PATH)
LOCK_PATH = Path(SELECTION_LOCK_PATH)
CONFIRMATION_PATH = Path(CONFIRMATION_EVIDENCE_PATH)
FINAL_MANIFEST_PATH = Path(FINAL_TEST_MANIFEST_PATH)
FINAL_SEAL_PATH = Path(FINAL_TEST_SEAL_PATH)
FINAL_OUTPUT_PATH = Path(FINAL_TEST_OUTPUT_PATH)
OUTPUT_PATH = Path(FINAL_AUTHORIZATION_PATH)
PHASE3_MANIFEST_PATH = Path("runs/phase3/manifest.json")
CONVERSION_MANIFEST_PATH = Path("runs/phase3-compute-conversion/manifest.json")
HISTORICAL_PRIMARY_SUMMARY_PATH = Path(PHASE3_PRIMARY_SUMMARY_PATH)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _require_clean_root() -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(root).resolve() != Path.cwd().resolve() or _git_status().strip():
        raise ValueError("final authorization requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("final authorization requires a Git commit")
    return commit


def _git_path_history(path: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def _require_single_publication_history(path: Path, git_commit: str) -> None:
    if _git_path_history(path) != (git_commit,):
        raise ValueError("confirmation artifact was not published exactly once")


def _require_never_published(path: Path) -> None:
    if path.exists() or _git_path_history(path):
        raise ValueError(
            f"final-evaluation artifact was already published or deleted: {path}"
        )


def _tracked_identity(path: Path) -> dict[str, str]:
    blob = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-list", "-1", "HEAD", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    commit_value = commit.stdout.strip()
    if (
        blob.returncode != 0
        or commit.returncode != 0
        or len(commit_value) != 40
        or not path.is_file()
        or path.read_bytes() != blob.stdout
    ):
        raise ValueError(f"final authorization input is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit_value,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(blob.stdout).hexdigest(),
    }


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"final authorization Git order differs: {label}")


def _require_strict_ancestor(ancestor: str, descendant: str, label: str) -> None:
    _require_ancestor(ancestor, descendant, label)
    if ancestor == descendant:
        raise ValueError(f"final authorization Git order is not strict: {label}")


def _git_blob_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"confirmation execution commit lacks {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _validate_initial_double_replay_chronology(
    *,
    initial_model_identity: Mapping[str, Any],
    initial_evidence: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, str]],
) -> None:
    replay = selection_lock["independent_calibration_recomputation"]
    evaluator_commit = initial_evidence["evaluator_git_commit"]
    verification_commit = replay["verification_git_commit"]
    if replay["evaluator_git_commit"] != evaluator_commit:
        raise ValueError("selection replay evaluator differs from calibration evidence")
    if evaluator_commit == identities["calibration_evidence"]["git_commit"]:
        raise ValueError("calibration evidence was not committed after its evaluator")
    if verification_commit == identities["selection_lock"]["git_commit"]:
        raise ValueError("selection lock was not committed after its verifier")
    for ancestor, descendant, label in (
        (
            identities["initial_model_identity"]["git_commit"],
            evaluator_commit,
            "initial identity -> calibration evaluator",
        ),
        (
            evaluator_commit,
            identities["calibration_evidence"]["git_commit"],
            "calibration evaluator -> evidence artifact",
        ),
        (
            identities["calibration_evidence"]["git_commit"],
            verification_commit,
            "calibration evidence -> selection verifier",
        ),
        (
            verification_commit,
            identities["selection_lock"]["git_commit"],
            "selection verifier -> selection lock artifact",
        ),
    ):
        _require_ancestor(ancestor, descendant, label)
    implementation = initial_model_identity[
        "calibration_selection_implementation"
    ]
    for path in implementation["file_order"]:
        expected = implementation["sha256_by_path"][path]
        if (
            _git_blob_sha256(evaluator_commit, path) != expected
            or _git_blob_sha256(verification_commit, path) != expected
        ):
            raise ValueError(f"initial double-replay implementation differs: {path}")


def _verify_prospective_execution_commit(
    *,
    run_commit: str,
    selection_lock_git_commit: str,
    confirmation_evaluator_git_commit: str,
    implementation: Mapping[str, Any],
) -> None:
    _require_ancestor(
        selection_lock_git_commit,
        run_commit,
        "selection lock -> confirmation run",
    )
    _require_ancestor(
        run_commit,
        confirmation_evaluator_git_commit,
        "confirmation run -> calibration evaluator",
    )
    for path in implementation["file_order"]:
        if (
            _git_blob_sha256(run_commit, path)
            != implementation["sha256_by_path"][path]
            or _git_blob_sha256(confirmation_evaluator_git_commit, path)
            != implementation["sha256_by_path"][path]
        ):
            raise ValueError(
                f"confirmation execution implementation differs: {path}"
            )


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_spec(descriptor: Mapping[str, Any]):
    if descriptor["model_family"] == "phase3":
        return PHASE3_MODEL_SPEC
    from jamoflow.compute_conversion import conversion_model_spec

    return conversion_model_spec(int(descriptor["patch_count"]))


def _run_manifest_context(
    descriptor: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = (
        PHASE3_MANIFEST_PATH
        if descriptor["model_family"] == "phase3"
        else CONVERSION_MANIFEST_PATH
    )
    manifest = _read_json(path)
    source = (
        manifest
        if descriptor["model_family"] == "phase3"
        else manifest.get("source_context", {})
    )
    source_artifact = source.get("source_artifact", {})
    integrity_artifact = source.get("source_integrity_artifact", {})
    train_stream = source.get("streams", {}).get("train", {})
    hashes = (
        source_artifact.get("sha256"),
        integrity_artifact.get("sha256"),
        train_stream.get("selected_stream_sha256"),
    )
    if (
        manifest.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or not all(is_sha256(value) for value in hashes)
    ):
        raise ValueError("final model run-manifest lineage is malformed")
    return path, manifest, {
        "source_artifact_sha256": source_artifact["sha256"],
        "source_integrity_artifact_sha256": integrity_artifact["sha256"],
        "train_stream_sha256": train_stream["selected_stream_sha256"],
    }


def _training_identity(
    *,
    descriptor: Mapping[str, Any],
    seed: int,
    report_path: Path,
    checkpoint_state_sha256: str,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    expected_selected_reference_authorization: Mapping[str, Any] | None,
    selection_lock_git_commit: str,
    confirmation_evaluator_git_commit: str,
    prospective_implementation: Mapping[str, Any],
    historical_phase3_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    report = _read_json(report_path)
    manifest_path, manifest, source = _run_manifest_context(descriptor)
    spec = _model_spec(descriptor)
    training = report.get("training", {})
    patch_hash = report.get("patch_matrix_sha256", {}).get("train")
    evidence_binding = report.get("evidence_binding")
    if (
        descriptor["model_family"] == "phase3"
        and descriptor["policy"] in PRIMARY_CONFIRMED_POLICIES
        and seed in CONFIRMATION_SEEDS
    ):
        historical_row = historical_phase3_anchor["by_seed_policy"][str(seed)][
            descriptor["policy"]
        ]
        if (
            historical_row["checkpoint_artifact_sha256"]
            != hash_file(Path(expected_model_paths(descriptor, seed)["checkpoint"]))
            or historical_row["checkpoint_state_sha256"]
            != checkpoint_state_sha256
            or historical_row["training_report_artifact_sha256"]
            != hash_file(report_path)
        ):
            raise ValueError(
                "historical Phase3 checkpoint differs from its sealed summary"
            )
        evidence_binding = {
            "anchor_sha256": historical_phase3_anchor["anchor_sha256"],
            "kind": "historical_phase3_five_seed_provenance_v1",
            "policy": descriptor["policy"],
            "seed": seed,
        }
    elif evidence_binding is None:
        evidence_binding = {
            "kind": "legacy_phase3_training_binding_v1",
            "manifest_artifact_sha256": hash_file(manifest_path),
            "policy": descriptor["policy"],
            "report_artifact_sha256": hash_file(report_path),
            "seed": seed,
            **source,
        }
    if descriptor["model_family"] == "compute_conversion":
        if not isinstance(evidence_binding, Mapping):
            raise ValueError("conversion model lacks its training authorization")
        unsigned_binding = {
            key: value
            for key, value in evidence_binding.items()
            if key != "identity_sha256"
        }
        expected_stage = "initial" if seed in (1729, 2718, 31415) else "confirmation"
        expected_selection = (
            None
            if expected_stage == "initial"
            else selection_lock_artifact_sha256
        )
        expected_policies = (
            [
                "causal_codepoint_grid_64",
                "causal_whitespace_grid_64",
                "causal_codepoint_grid_72",
                "causal_whitespace_grid_72",
            ]
            if expected_stage == "initial"
            else selection_lock["decision"]["confirmation_plan"][
                "compute_conversion"
            ]["policies"]
        )
        expected_seeds = (
            [1729, 2718, 31415]
            if expected_stage == "initial"
            else [57721, 65537]
        )
        if (
            set(evidence_binding)
            != {
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
            or evidence_binding["identity_sha256"]
            != canonical_sha256(unsigned_binding)
            or evidence_binding["stage"] != expected_stage
            or evidence_binding["device"] != "mps"
            or evidence_binding["git_worktree_clean_at_start"] is not True
            or evidence_binding["schema_version"] != 1
            or evidence_binding["policies"] != expected_policies
            or evidence_binding["seeds"] != expected_seeds
            or evidence_binding["selection_plan_sha256"]
            != selection_lock["plan_sha256"]
            or evidence_binding["selection_summary_sha256"]
            != expected_selection
        ):
            raise ValueError("conversion training authorization differs")
        matching_invocations = [
            invocation
            for invocation in manifest.get("invocations", ())
            if isinstance(invocation, Mapping)
            and invocation.get("stage") == expected_stage
            and invocation.get("seeds") == expected_seeds
            and invocation.get("policies") == expected_policies
            and invocation.get("evidence_binding") == evidence_binding
            and invocation.get("git_commit") == evidence_binding["git_commit"]
        ]
        stage_invocations = [
            invocation
            for invocation in manifest.get("invocations", ())
            if isinstance(invocation, Mapping)
            and invocation.get("stage") == expected_stage
        ]
        if (
            not matching_invocations
            or len(matching_invocations) != len(stage_invocations)
        ):
            raise ValueError(
                "conversion training lacks an exact non-conflicting invocation"
            )
        if expected_stage == "confirmation":
            _verify_prospective_execution_commit(
                run_commit=evidence_binding["git_commit"],
                selection_lock_git_commit=selection_lock_git_commit,
                confirmation_evaluator_git_commit=(
                    confirmation_evaluator_git_commit
                ),
                implementation=prospective_implementation,
            )
    elif (
        descriptor["policy"] in SELECTABLE_REFERENCE_POLICIES
        and seed in (57721, 65537)
    ):
        expected = expected_selected_reference_authorization
        if (
            expected is None
            or not isinstance(evidence_binding, Mapping)
            or not isinstance(evidence_binding.get("run_git_commit"), str)
            or len(evidence_binding["run_git_commit"]) != 40
            or any(
                character not in "0123456789abcdef"
                for character in evidence_binding["run_git_commit"]
            )
            or evidence_binding
            != {
                "authorization": dict(expected),
                "device": "mps",
                "git_worktree_clean_at_start": True,
                "kind": "selected_phase3_reference_training_evidence_v4",
                "run_git_commit": evidence_binding.get("run_git_commit"),
                "schema_version": 4,
            }
        ):
            raise ValueError("selected-reference training authorization differs")
        matching_invocations = [
            invocation
            for invocation in manifest.get("invocations", ())
            if isinstance(invocation, Mapping)
            and invocation.get("seeds") == [57721, 65537]
            and invocation.get("policies") == [descriptor["policy"]]
            and invocation.get("authorization") == dict(expected)
            and invocation.get("git_commit")
            == evidence_binding["run_git_commit"]
            and invocation.get("force") is False
            and invocation.get("save_checkpoints") is True
        ]
        if len(matching_invocations) != 1:
            raise ValueError("selected-reference training lacks one exact invocation")
        _verify_prospective_execution_commit(
            run_commit=evidence_binding["run_git_commit"],
            selection_lock_git_commit=selection_lock_git_commit,
            confirmation_evaluator_git_commit=confirmation_evaluator_git_commit,
            implementation=prospective_implementation,
        )
    if (
        report.get("seed") != seed
        or report.get("policy") != descriptor["policy"]
        or report.get("model_spec") != spec.to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("parameters") != FINAL_MAIN_PARAMETER_COUNT
        or report.get("trained_state_sha256") != checkpoint_state_sha256
        or manifest.get("global_max_position_embeddings")
        != PHASE3_MODEL_SPEC.sequence_length * 2 + 8
        or not is_sha256(report.get("initialization_sha256"))
        or not is_sha256(report.get("training_order_sha256"))
        or not is_sha256(patch_hash)
        or not isinstance(evidence_binding, Mapping)
        or training.get("examples") != 250_000
        or training.get("predicted_bytes") != 127_750_000
        or training.get("steps") != 7_813
    ):
        raise ValueError(f"final training report lineage differs: {seed}")
    return {
        "evidence_binding_sha256": canonical_sha256(
            {
                "policy": descriptor["policy"],
                "report_artifact_sha256": hash_file(report_path),
                "report_evidence_binding": evidence_binding,
                "seed": seed,
            }
        ),
        "global_max_position_embeddings": 1_032,
        "initialization_sha256": report["initialization_sha256"],
        "optimization_spec_sha256": canonical_sha256(
            PHASE3_OPTIMIZATION_SPEC.to_dict()
        ),
        "run_manifest_artifact_sha256": hash_file(manifest_path),
        **source,
        "steps": training["steps"],
        "train_examples": training["examples"],
        "train_patch_matrix_sha256": patch_hash,
        "train_predicted_bytes": training["predicted_bytes"],
        "training_order_sha256": report["training_order_sha256"],
    }


def _normalize_initial_auxiliary(
    receipt: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    auxiliary = receipt["auxiliary"]
    if descriptor["policy"] not in THRESHOLD_POLICIES:
        if dict(auxiliary) != {"kind": "none"}:
            raise ValueError("structural initial receipt unexpectedly has a router")
        return {"kind": "none"}
    manifest = _read_json(PHASE3_MANIFEST_PATH)
    expected_mask = (
        "none"
        if descriptor["policy"] == "entropy_threshold_full"
        else "codepoint"
    )
    if (
        auxiliary.get("kind") != "entropy_router"
        or auxiliary.get("candidate_mask") != expected_mask
        or auxiliary.get("maximum_patch_length") != 24
    ):
        raise ValueError("initial entropy receipt router contract differs")
    return {
        "calibration_stream_sha256": receipt["calibration"]["stream_sha256"],
        "candidate_mask": expected_mask,
        "kind": "entropy_router",
        "maximum_patch_length": 24,
        "policy": descriptor["policy"],
        "policy_definition_sha256": entropy_policy_definition_sha256(
            descriptor["policy"]
        ),
        "router_checkpoint_artifact_sha256": auxiliary[
            "router_checkpoint_artifact_sha256"
        ],
        "router_checkpoint_path": auxiliary["router_checkpoint_path"],
        "router_checkpoint_state_sha256": auxiliary["router_state_sha256"],
        "router_config_sha256": canonical_sha256(PHASE3_MODEL_SPEC.to_dict()),
        "router_parameter_count": FINAL_ROUTER_PARAMETER_COUNT,
        "router_report_artifact_sha256": auxiliary[
            "router_report_artifact_sha256"
        ],
        "router_report_path": auxiliary["router_report_path"],
        "router_training_stream_sha256": manifest["streams"]["train"][
            "selected_stream_sha256"
        ],
        "seed": receipt["seed"],
        "threshold_cache_artifact_sha256": auxiliary["cache_artifact_sha256"],
        "threshold_cache_path": auxiliary["cache_path"],
        "threshold_diagnostics_artifact_sha256": auxiliary[
            "diagnostics_artifact_sha256"
        ],
        "threshold_diagnostics_path": auxiliary["diagnostics_path"],
        "threshold_nats": auxiliary["threshold_nats"],
    }


def _verify_auxiliary_artifacts(
    auxiliary: Mapping[str, Any],
    seed: int,
    *,
    expected_evidence_binding: Mapping[str, Any] | None,
) -> None:
    if auxiliary["kind"] == "none":
        return
    checks = (
        ("router_checkpoint_path", "router_checkpoint_artifact_sha256"),
        ("router_report_path", "router_report_artifact_sha256"),
        ("threshold_cache_path", "threshold_cache_artifact_sha256"),
        (
            "threshold_diagnostics_path",
            "threshold_diagnostics_artifact_sha256",
        ),
    )
    for path_key, hash_key in checks:
        if hash_file(Path(auxiliary[path_key])) != auxiliary[hash_key]:
            raise ValueError(f"final router artifact differs: {seed}/{path_key}")
    router_report = _read_json(Path(auxiliary["router_report_path"]))
    threshold_diagnostics = _read_json(
        Path(auxiliary["threshold_diagnostics_path"])
    )
    if (
        expected_evidence_binding is not None
        and (
            router_report.get("evidence_binding")
            != expected_evidence_binding
            or threshold_diagnostics.get("_provenance", {}).get(
                "evidence_binding"
            )
            != expected_evidence_binding
        )
    ):
        raise ValueError(f"final router training authorization differs: {seed}")
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    router.load_state_dict(
        torch.load(
            Path(auxiliary["router_checkpoint_path"]),
            map_location="cpu",
            weights_only=True,
        )
    )
    state = _state_sha256(router)
    if (
        parameter_count(router) != FINAL_ROUTER_PARAMETER_COUNT
        or state != auxiliary["router_checkpoint_state_sha256"]
    ):
        raise ValueError(f"final router state differs: {seed}")
    del router
    gc.collect()


def _seed_evidence(
    *,
    receipt: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    seed: int,
    initial: bool,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    expected_selected_reference_authorization: Mapping[str, Any] | None,
    selection_lock_git_commit: str,
    confirmation_evaluator_git_commit: str,
    prospective_implementation: Mapping[str, Any],
    historical_phase3_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = receipt["checkpoint"]
    training_report = receipt["training_report"]
    if hash_file(Path(checkpoint["path"])) != checkpoint["artifact_sha256"]:
        raise ValueError(f"final checkpoint artifact differs: {seed}")
    if hash_file(Path(training_report["path"])) != training_report["artifact_sha256"]:
        raise ValueError(f"final training report artifact differs: {seed}")
    model = build_main_model(
        _model_spec(descriptor),
        seed=seed,
        global_max_position_embeddings=1_032,
    )
    model.load_state_dict(
        torch.load(Path(checkpoint["path"]), map_location="cpu", weights_only=True)
    )
    state = _state_sha256(model)
    if parameter_count(model) != FINAL_MAIN_PARAMETER_COUNT or state != checkpoint[
        "state_sha256"
    ]:
        raise ValueError(f"final checkpoint state differs: {seed}")
    del model
    gc.collect()
    auxiliary = (
        _normalize_initial_auxiliary(receipt, descriptor)
        if initial
        else dict(receipt["auxiliary"])
    )
    report_payload = _read_json(Path(training_report["path"]))
    expected_router_binding = (
        report_payload.get("evidence_binding")
        if expected_selected_reference_authorization is not None
        and descriptor["requires_entropy_router"]
        and seed in (57721, 65537)
        else None
    )
    _verify_auxiliary_artifacts(
        auxiliary,
        seed,
        expected_evidence_binding=expected_router_binding,
    )
    return {
        "auxiliary": auxiliary,
        "checkpoint": dict(checkpoint),
        "seed": seed,
        "training": _training_identity(
            descriptor=descriptor,
            seed=seed,
            report_path=Path(training_report["path"]),
            checkpoint_state_sha256=state,
            selection_lock=selection_lock,
            selection_lock_artifact_sha256=selection_lock_artifact_sha256,
            expected_selected_reference_authorization=(
                expected_selected_reference_authorization
            ),
            selection_lock_git_commit=selection_lock_git_commit,
            confirmation_evaluator_git_commit=(
                confirmation_evaluator_git_commit
            ),
            prospective_implementation=prospective_implementation,
            historical_phase3_anchor=historical_phase3_anchor,
        ),
        "training_report": dict(training_report),
    }


def _build_models(
    *,
    selection_lock: Mapping[str, Any],
    initial_evidence: Mapping[str, Any],
    confirmation_evidence: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    expected_selected_reference_authorization: Mapping[str, Any] | None,
    selection_lock_git_commit: str,
    confirmation_evaluator_git_commit: str,
    prospective_implementation: Mapping[str, Any],
    historical_phase3_anchor: Mapping[str, Any],
) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model in resolve_final_evaluation_roles(selection_lock)["unique_models"]:
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
        rows: dict[int, dict[str, Any]] = {}
        for seed in FINAL_SEEDS:
            initial = seed in INITIAL_SEEDS
            receipt = (
                initial_evidence["receipts"][str(seed)][descriptor["policy"]]
                if initial
                else confirmation_evidence["receipts"][str(seed)][
                    model["artifact_role"]
                ]
            )
            if initial:
                if (
                    receipt["model_family"] != descriptor["model_family"]
                    or receipt["patch_count"] != descriptor["patch_count"]
                    or receipt["policy"] != descriptor["policy"]
                ):
                    raise ValueError("initial model receipt differs from locked role")
            elif receipt["descriptor"] != descriptor:
                raise ValueError("confirmation model receipt differs from locked role")
            rows[seed] = _seed_evidence(
                receipt=receipt,
                descriptor=descriptor,
                seed=seed,
                initial=initial,
                selection_lock=selection_lock,
                selection_lock_artifact_sha256=selection_lock_artifact_sha256,
                expected_selected_reference_authorization=(
                    expected_selected_reference_authorization
                ),
                selection_lock_git_commit=selection_lock_git_commit,
                confirmation_evaluator_git_commit=(
                    confirmation_evaluator_git_commit
                ),
                prospective_implementation=prospective_implementation,
                historical_phase3_anchor=historical_phase3_anchor,
            )
        models.append(
            build_final_model_identity(
                artifact_role=model["artifact_role"],
                descriptor=descriptor,
                seed_evidence=rows,
                parameter_count=FINAL_MAIN_PARAMETER_COUNT,
            )
        )
    return models


def _historical_phase3_anchor(
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    planned = plan["historical_screening"]["primary_summary"]
    identity = _tracked_identity(HISTORICAL_PRIMARY_SUMMARY_PATH)
    summary = _read_json(HISTORICAL_PRIMARY_SUMMARY_PATH)
    integrity = summary.get("integrity", {})
    if (
        planned.get("path") != HISTORICAL_PRIMARY_SUMMARY_PATH.as_posix()
        or planned.get("sha256") != identity["sha256"]
        or summary.get("seeds") != list(FINAL_SEEDS)
        or tuple(summary.get("policies", ())) != PRIMARY_CONFIRMED_POLICIES
        or integrity.get("all_integrity_checks_pass") is not True
    ):
        raise ValueError("historical C86 summary is not the plan-sealed evidence")
    rows: dict[str, dict[str, dict[str, str]]] = {}
    for seed in CONFIRMATION_SEEDS:
        row = integrity.get("by_seed", {}).get(str(seed), {})
        rows[str(seed)] = {}
        for policy in PRIMARY_CONFIRMED_POLICIES:
            values = {
                "checkpoint_artifact_sha256": row.get(
                    "checkpoint_artifact_sha256", {}
                ).get(policy),
                "checkpoint_state_sha256": row.get(
                    "checkpoint_state_sha256", {}
                ).get(policy),
                "training_report_artifact_sha256": row.get(
                    "training_report_artifact_sha256", {}
                ).get(policy),
            }
            if not all(is_sha256(value) for value in values.values()):
                raise ValueError(
                    "historical Phase3 summary lacks physical model identity"
                )
            rows[str(seed)][policy] = values
    payload = {
        "artifact": identity,
        "by_seed_policy": rows,
        "policy_order": list(PRIMARY_CONFIRMED_POLICIES),
        "provenance_scope": "historical_preselection_five_seed_evidence",
        "seed_order": list(CONFIRMATION_SEEDS),
        "status": "integrity_verified",
    }
    payload["anchor_sha256"] = canonical_sha256(payload)
    return payload


def _validate_final_output(final_seal: Mapping[str, Any]) -> dict[str, Any]:
    output = final_seal["payload"]["output"]
    if (
        not FINAL_OUTPUT_PATH.is_file()
        or hash_file(FINAL_OUTPUT_PATH) != output["full_jsonl_sha256"]
        or FINAL_OUTPUT_PATH.stat().st_size != output["full_jsonl_bytes"]
        or output["evaluation_stream_bytes"] != 32_000_000
        or output["sequence_length"] != 512
        or output["sequence_count"] != 62_500
    ):
        raise ValueError("sealed final-test output differs before authorization")
    return output


def _post_publish_status_is_clean() -> bool:
    lines = [line for line in _git_status().splitlines() if line.strip()]
    allowed = {f"?? {OUTPUT_PATH.as_posix()}"}
    return not lines or set(lines) <= allowed


def _selected_reference_authorization_if_required(
    *,
    selection_lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    reference = selection_lock["decision"]["reference"]
    if (
        reference["model_family"] != "phase3"
        or reference["policy"] not in SELECTABLE_REFERENCE_POLICIES
        or selection_lock["decision"].get(
            "broad_reference_evaluation_status"
        )
        != "eligible_pending_confirmation"
    ):
        return None
    phase3_plan = selection_lock["decision"]["confirmation_plan"].get(
        "phase3_reference"
    )
    if not isinstance(phase3_plan, Mapping):
        raise ValueError("eligible broad reference lacks its typed confirmation plan")
    return selected_reference_authorization_record_v3(
        selection_lock,
        plan,
        selection_lock_artifact_sha256=identities["selection_lock"]["sha256"],
        selection_plan_artifact_sha256=identities["selection_plan"]["sha256"],
        calibration_evidence_artifact_sha256=identities[
            "calibration_evidence"
        ]["sha256"],
        final_test_seal_artifact_sha256=identities["final_seal"]["sha256"],
    )


def _validate_confirmation_training_completions(
    *,
    plan: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    selection_lock_identity: Mapping[str, str],
    initial_model_identity: Mapping[str, Any],
    confirmation_evidence: Mapping[str, Any],
    confirmation_evaluator_git_commit: str,
) -> dict[str, dict[str, Any]]:
    """Independently bind committed training completions to replay evidence."""

    required = required_confirmation_completion_families(selection_lock)
    projected = confirmation_evidence.get("training_completions")
    if (
        not isinstance(projected, Mapping)
        or len(projected) != len(required)
        or set(projected) != set(required)
    ):
        raise ValueError("confirmation evidence completion set differs")
    implementation = initial_model_identity[
        "calibration_selection_implementation"
    ]
    completions: dict[str, dict[str, Any]] = {}
    for family in required:
        path = confirmation_completion_path(family)
        artifact = _tracked_identity(path)
        _require_single_publication_history(path, artifact["git_commit"])
        completion = _read_json(path)
        validate_confirmation_training_completion(
            completion,
            selection_lock=selection_lock,
        )
        expected_projection = {
            "artifact": artifact,
            "completion_sha256": completion["completion_sha256"],
            "run_git_commit": completion["run_git_commit"],
        }
        if dict(projected[family]) != expected_projection:
            raise ValueError("confirmation completion projection was rotated")
        if (
            completion["selection_lock_artifact_sha256"]
            != selection_lock_identity["sha256"]
            or completion["selection_lock_payload_sha256"]
            != selection_lock["lock_sha256"]
            or completion["implementation_manifest_sha256"]
            != implementation["manifest_sha256"]
            or completion["environment_sha256"]
            != implementation["environment_sha256"]
        ):
            raise ValueError("confirmation completion lineage differs")
        run_commit = completion["run_git_commit"]
        run_manifest = completion["run_manifest"]
        if hash_file(Path(run_manifest["path"])) != run_manifest["artifact_sha256"]:
            raise ValueError("confirmation completion run manifest changed")
        _require_ancestor(
            selection_lock_identity["git_commit"],
            run_commit,
            f"selection lock -> {family} run",
        )
        _require_ancestor(
            run_commit,
            artifact["git_commit"],
            f"{family} run -> completion receipt",
        )
        if run_commit == artifact["git_commit"]:
            raise ValueError("confirmation completion was not committed after its run")
        _require_ancestor(
            artifact["git_commit"],
            confirmation_evaluator_git_commit,
            f"{family} completion -> confirmation evaluator",
        )
        for implementation_path in implementation["file_order"]:
            if (
                _git_blob_sha256(run_commit, implementation_path)
                != implementation["sha256_by_path"][implementation_path]
            ):
                raise ValueError(
                    f"confirmation run implementation differs: {implementation_path}"
                )
        for seed in CONFIRMATION_SEEDS:
            for policy, unit in completion["units"][str(seed)].items():
                if (
                    hash_file(Path(unit["checkpoint_path"]))
                    != unit["checkpoint_artifact_sha256"]
                    or hash_file(Path(unit["training_report_path"]))
                    != unit["training_report_artifact_sha256"]
                ):
                    raise ValueError("confirmation completion model artifact changed")
                validate_training_report_against_completion(
                    completion=completion,
                    report=_read_json(Path(unit["training_report_path"])),
                    seed=seed,
                    policy=policy,
                    selection_lock=selection_lock,
                    historical_primary_summary_sha256=plan[
                        "historical_screening"
                    ]["primary_summary"]["sha256"],
                )
                auxiliary = unit["auxiliary"]
                if auxiliary["kind"] == "entropy_router_artifacts":
                    for stem in (
                        "router_checkpoint",
                        "router_report",
                        "threshold_cache",
                        "threshold_diagnostics",
                    ):
                        if (
                            hash_file(Path(auxiliary[f"{stem}_path"]))
                            != auxiliary[f"{stem}_artifact_sha256"]
                        ):
                            raise ValueError(
                                "confirmation completion router artifact changed"
                            )
        completions[family] = completion
    validate_receipts_against_training_completions(
        selection_lock=selection_lock,
        receipts=confirmation_evidence["receipts"],
        completions=completions,
    )
    return {family: dict(projected[family]) for family in required}


def _validate_stored_confirmation_nll(
    *,
    receipt: Mapping[str, Any],
    replay: Mapping[str, Any],
    seed: int,
    role: str,
) -> None:
    calibration = receipt["calibration"]
    nll_path = Path(calibration["nll_artifact_path"])
    if (
        not nll_path.is_file()
        or nll_path.is_symlink()
        or hash_file(nll_path) != calibration["nll_artifact_sha256"]
    ):
        raise ValueError(f"confirmation NLL artifact differs: {seed}/{role}")
    with np.load(nll_path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError(f"confirmation NLL schema differs: {seed}/{role}")
        stored_nll = archive["sequence_nll_nats"]
    replay_nll = replay["losses"]
    if (
        stored_nll.dtype != np.float32
        or stored_nll.shape != replay_nll.shape
        or not np.isfinite(stored_nll).all()
        or np.any(stored_nll < 0)
        or _array_sha256(stored_nll) != calibration["nll_array_sha256"]
        or not np.array_equal(stored_nll, replay_nll)
    ):
        raise ValueError(f"confirmation NLL fails independent replay: {seed}/{role}")


def _independent_confirmation_recomputation(
    *,
    plan: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    confirmation_evidence: Mapping[str, Any],
    verification_git_commit: str,
) -> dict[str, Any]:
    """Replay every confirmation checkpoint without trusting stored NLL files."""

    device = resolve_device("mps")
    stream, inputs, boundaries, shared_matrices = (
        load_confirmation_calibration_context(plan)
    )
    models = required_confirmation_models(selection_lock)
    required_policies = tuple(
        model["descriptor"]["policy"] for model in models
    )
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in CONFIRMATION_SEEDS:
        entropy_matrices, entropy_auxiliary = (
            confirmation_entropy_matrices_and_auxiliary(
                seed=seed,
                required_policies=required_policies,
                inputs=inputs,
                boundaries=boundaries,
                plan=plan,
                device=device,
            )
        )
        matrices = {**shared_matrices, **entropy_matrices}
        rows[str(seed)] = {}
        for model in models:
            role = model["artifact_role"]
            descriptor = model["descriptor"]
            policy = descriptor["policy"]
            matrix = matrices.get(policy)
            if matrix is None:
                raise ValueError(
                    f"independent confirmation matrix is missing: {seed}/{role}"
                )
            receipt = confirmation_evidence["receipts"][str(seed)][role]
            replay = replay_confirmation_unit(
                descriptor=descriptor,
                seed=seed,
                inputs=inputs,
                boundaries=boundaries,
                matrix=matrix,
                auxiliary=entropy_auxiliary.get(policy, {"kind": "none"}),
                device=device,
            )
            validate_confirmation_replay_against_receipt(
                receipt=receipt,
                replay=replay,
                stream=stream,
                inputs=inputs,
                boundaries=boundaries,
                matrix=matrix,
            )
            _validate_stored_confirmation_nll(
                receipt=receipt,
                replay=replay,
                seed=seed,
                role=role,
            )
            rows[str(seed)][role] = {
                "checkpoint_state_sha256": replay[
                    "checkpoint_state_sha256"
                ],
                "matrix_sha256": receipt["calibration"]["matrix_sha256"],
                "nll_array_sha256": replay["nll_array_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
            del replay
        del matrices, entropy_matrices, entropy_auxiliary
        gc.collect()
        torch.mps.empty_cache()
    payload = {
        "comparison": "bitwise_float32_nll_hash_equal",
        "device": "mps",
        "model_artifact_role_order": [
            model["artifact_role"] for model in models
        ],
        "receipt_count": len(models) * len(CONFIRMATION_SEEDS),
        "replay_by_seed_role": rows,
        "seed_order": list(CONFIRMATION_SEEDS),
        "status": "pass",
        "verification_git_commit": verification_git_commit,
    }
    payload["recomputation_sha256"] = canonical_sha256(payload)
    return payload


def _main_locked() -> int:
    base_commit = _require_clean_root()
    tracked_forbidden = (
        OUTPUT_PATH,
        Path(FINAL_EVIDENCE_PATH),
        Path(FINAL_QUALITY_LOCK_PATH),
    )
    for path in tracked_forbidden:
        _require_never_published(path)
    ignored_forbidden = (
        Path(FINAL_ARTIFACT_ROOT),
    )
    if any(path.exists() for path in ignored_forbidden):
        raise ValueError("final evaluation artifacts predate their authorization")

    identities = {
        "selection_plan": _tracked_identity(PLAN_PATH),
        "initial_model_identity": _tracked_identity(
            Path(INITIAL_MODEL_IDENTITY_LOCK_PATH)
        ),
        "calibration_evidence": _tracked_identity(INITIAL_EVIDENCE_PATH),
        "selection_lock": _tracked_identity(LOCK_PATH),
        "confirmation_evidence": _tracked_identity(CONFIRMATION_PATH),
        "final_manifest": _tracked_identity(FINAL_MANIFEST_PATH),
        "final_seal": _tracked_identity(FINAL_SEAL_PATH),
    }
    _require_single_publication_history(
        CONFIRMATION_PATH,
        identities["confirmation_evidence"]["git_commit"],
    )
    ordered = (
        (identities["final_seal"]["git_commit"], identities["selection_plan"]["git_commit"], "final seal -> plan"),
        (identities["selection_plan"]["git_commit"], identities["initial_model_identity"]["git_commit"], "plan -> initial model identity"),
        (identities["initial_model_identity"]["git_commit"], identities["calibration_evidence"]["git_commit"], "initial model identity -> initial evidence"),
        (identities["calibration_evidence"]["git_commit"], identities["selection_lock"]["git_commit"], "initial evidence -> lock"),
        (identities["selection_lock"]["git_commit"], identities["confirmation_evidence"]["git_commit"], "lock -> confirmation evidence"),
        (identities["confirmation_evidence"]["git_commit"], base_commit, "confirmation evidence -> authorization base"),
    )
    for ancestor, descendant, label in ordered:
        _require_ancestor(ancestor, descendant, label)

    plan = _read_json(PLAN_PATH)
    initial_evidence = _read_json(INITIAL_EVIDENCE_PATH)
    initial_model_identity = _read_json(
        Path(INITIAL_MODEL_IDENTITY_LOCK_PATH)
    )
    selection_lock = _read_json(LOCK_PATH)
    confirmation_evidence = _read_json(CONFIRMATION_PATH)
    final_seal = _read_json(FINAL_SEAL_PATH)
    validate_selection_plan_v2(plan)
    validate_calibration_evidence_manifest(initial_evidence, plan=plan)
    validate_initial_model_identity_lock_v2(initial_model_identity)
    validate_selection_lock_v2(selection_lock)
    validate_selection_lock_identity_binding_v2(
        selection_lock, initial_model_identity
    )
    _validate_initial_double_replay_chronology(
        initial_model_identity=initial_model_identity,
        initial_evidence=initial_evidence,
        selection_lock=selection_lock,
        identities=identities,
    )
    validate_current_implementation_v2(
        initial_model_identity,
        sha256_by_path={
            path: _tracked_identity(Path(path))["sha256"]
            for path in initial_model_identity[
                "calibration_selection_implementation"
            ]["file_order"]
        },
        environment=runtime_environment_v2(),
    )
    validate_confirmation_evidence_manifest(
        confirmation_evidence,
        plan=plan,
        selection_lock=selection_lock,
    )
    validate_seal_envelope(final_seal)
    confirmation_evaluator_commit = confirmation_evidence[
        "evaluator_git_commit"
    ]
    _require_ancestor(
        identities["selection_lock"]["git_commit"],
        confirmation_evaluator_commit,
        "selection lock -> confirmation evaluator",
    )
    _require_strict_ancestor(
        confirmation_evaluator_commit,
        identities["confirmation_evidence"]["git_commit"],
        "confirmation evaluator -> confirmation evidence",
    )
    prospective_implementation = initial_model_identity[
        "calibration_selection_implementation"
    ]
    for path in prospective_implementation["file_order"]:
        if (
            _git_blob_sha256(confirmation_evaluator_commit, path)
            != prospective_implementation["sha256_by_path"][path]
        ):
            raise ValueError(
                f"confirmation evaluator implementation differs: {path}"
            )
    decision = build_selection_decision_v2(
        {
            seed: {
                policy: initial_evidence["receipts"][str(seed)][policy][
                    "calibration"
                ]["bpb"]
                for policy in initial_evidence["policy_order"]
            }
            for seed in INITIAL_SEEDS
        }
    )
    if (
        selection_lock["decision"] != decision
        or selection_lock["plan_sha256"] != identities["selection_plan"]["sha256"]
        or selection_lock["calibration_evidence_manifest_sha256"]
        != identities["calibration_evidence"]["sha256"]
        or selection_lock["final_test_seal_sha256"]
        != identities["final_seal"]["sha256"]
        or selection_lock["initial_model_identity_lock_sha256"]
        != identities["initial_model_identity"]["sha256"]
        or initial_evidence["initial_model_identity_lock_sha256"]
        != identities["initial_model_identity"]["sha256"]
    ):
        raise ValueError("final authorization selection lineage does not reconstruct")
    final_output = _validate_final_output(final_seal)
    historical_phase3_anchor = _historical_phase3_anchor(plan=plan)
    selected_reference_authorization = (
        _selected_reference_authorization_if_required(
            selection_lock=selection_lock,
            plan=plan,
            identities=identities,
        )
    )
    training_completions = _validate_confirmation_training_completions(
        plan=plan,
        selection_lock=selection_lock,
        selection_lock_identity=identities["selection_lock"],
        initial_model_identity=initial_model_identity,
        confirmation_evidence=confirmation_evidence,
        confirmation_evaluator_git_commit=confirmation_evaluator_commit,
    )
    models = _build_models(
        selection_lock=selection_lock,
        initial_evidence=initial_evidence,
        confirmation_evidence=confirmation_evidence,
        selection_lock_artifact_sha256=identities["selection_lock"][
            "sha256"
        ],
        expected_selected_reference_authorization=(
            selected_reference_authorization
        ),
        selection_lock_git_commit=identities["selection_lock"]["git_commit"],
        confirmation_evaluator_git_commit=confirmation_evaluator_commit,
        prospective_implementation=prospective_implementation,
        historical_phase3_anchor=historical_phase3_anchor,
    )
    independent_recomputation = _independent_confirmation_recomputation(
        plan=plan,
        selection_lock=selection_lock,
        confirmation_evidence=confirmation_evidence,
        verification_git_commit=base_commit,
    )
    implementation = {
        path: _tracked_identity(Path(path))["sha256"]
        for path in IMPLEMENTATION_FILE_ORDER
    }
    confirmation_projection = {
        "artifact": identities["confirmation_evidence"],
        "complete": confirmation_evidence["complete"],
        "integrity_pass": confirmation_evidence["integrity_pass"],
        "historical_primary_phase3_provenance": historical_phase3_anchor,
        "independent_recomputation": independent_recomputation,
        "manifest_sha256": confirmation_evidence["manifest_sha256"],
        "model_artifact_role_order": confirmation_evidence[
            "model_artifact_role_order"
        ],
        "seed_order": confirmation_evidence["seed_order"],
        "training_completions": training_completions,
        "receipt_commitments_by_seed_role": {
            str(seed): {
                role: {
                    "checkpoint_state_sha256": confirmation_evidence[
                        "receipts"
                    ][str(seed)][role]["checkpoint"]["state_sha256"],
                    "matrix_sha256": confirmation_evidence["receipts"][
                        str(seed)
                    ][role]["calibration"]["matrix_sha256"],
                    "nll_array_sha256": confirmation_evidence["receipts"][
                        str(seed)
                    ][role]["calibration"]["nll_array_sha256"],
                    "receipt_sha256": confirmation_evidence["receipts"][
                        str(seed)
                    ][role]["receipt_sha256"],
                }
                for role in confirmation_evidence["model_artifact_role_order"]
            }
            for seed in CONFIRMATION_SEEDS
        },
        "selection_lock_artifact_sha256": confirmation_evidence[
            "selection_lock_artifact_sha256"
        ],
        "selection_lock_payload_sha256": confirmation_evidence[
            "selection_lock_payload_sha256"
        ],
    }
    authorization = build_final_evaluation_authorization_v2(
        selection_lock=selection_lock,
        upstream_artifacts={
            "calibration_evidence": identities["calibration_evidence"],
            "selection_lock": identities["selection_lock"],
            "selection_plan": identities["selection_plan"],
        },
        confirmation_evidence=confirmation_projection,
        final_test={
            "evaluation_stream_bytes": final_output["evaluation_stream_bytes"],
            "evaluation_stream_sha256": final_output[
                "evaluation_stream_sha256"
            ],
            "manifest": identities["final_manifest"],
            "output_jsonl": {
                "path": FINAL_OUTPUT_PATH.as_posix(),
                "sha256": final_output["full_jsonl_sha256"],
            },
            "seal": identities["final_seal"],
            "seal_payload_sha256": final_seal["payload_sha256"],
            "sequence_count": final_output["sequence_count"],
            "sequence_length": final_output["sequence_length"],
        },
        models=models,
        implementation_sha256=implementation,
        authorization_git_commit=base_commit,
    )
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    serialized = (
        json.dumps(
            authorization,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    publish_no_clobber(OUTPUT_PATH, serialized)
    if _git_commit() != base_commit or not _post_publish_status_is_clean():
        raise RuntimeError("repository changed while sealing final authorization")
    print(
        json.dumps(
            {
                "authorization_sha256": authorization[
                    "authorization_sha256"
                ],
                "model_identity_order": [
                    model["identity_sha256"] for model in models
                ],
                "output": OUTPUT_PATH.as_posix(),
                "status": "sealed_pending_commit_no_final_metric_read",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    with publication_mps_exclusive():
        return _main_locked()


if __name__ == "__main__":
    raise SystemExit(main())
