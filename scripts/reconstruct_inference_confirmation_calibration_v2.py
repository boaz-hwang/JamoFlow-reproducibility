#!/usr/bin/env python3
"""Reconstruct calibration-only evidence for every locked confirmation model."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.compute_conversion import (
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
)
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.inference_calibration_evidence import (
    calibration_bpb_matrix,
    validate_calibration_evidence_manifest,
)
from jamoflow.inference_confirmation_evidence_v2 import (
    CALIBRATION_SEQUENCE_COUNT,
    CONFIRMATION_RESULT_PATH,
    CONFIRMATION_SEEDS,
    build_confirmation_calibration_receipt,
    build_confirmation_evidence_manifest,
    confirmation_completion_path,
    expected_confirmation_paths,
    required_confirmation_completion_families,
    required_confirmation_models,
    validate_confirmation_calibration_receipt,
    validate_confirmation_training_completion,
    validate_receipts_against_training_completions,
    validate_training_report_against_completion,
)
from jamoflow.inference_confirmation_replay_v2 import (
    confirmation_entropy_matrices_and_auxiliary,
    load_confirmation_calibration_context,
    replay_confirmation_unit,
)
from jamoflow.inference_final_authorization_v2 import (
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_ROUTER_PARAMETER_COUNT,
    canonical_sha256,
    expected_model_paths,
)
from jamoflow.inference_initial_model_identity_v2 import (
    INITIAL_MODEL_IDENTITY_LOCK_PATH,
    runtime_environment_v2,
    validate_current_implementation_v2,
    validate_initial_model_identity_lock_v2,
    validate_selection_lock_identity_binding_v2,
)
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_selection_plan import validate_selection_plan_v2
from jamoflow.inference_selection_v2 import (
    build_selection_decision_v2,
    validate_selection_lock_v2,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    build_router,
    parameter_count,
)
from jamoflow.neural_training import (
    evaluate_main_model,
    resolve_device,
    router_entropy_scores,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    calibrate_threshold,
    compact_whitespace_mask,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    THRESHOLD_POLICIES,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.publication_reference import entropy_policy_definition_sha256


PLAN_PATH = Path("data/manifests/phase3-inference-selection-plan-v2.json")
INITIAL_EVIDENCE_PATH = Path(
    "results/phase3-inference-selection-v2/calibration-evidence.json"
)
SELECTION_LOCK_PATH = Path(
    "results/phase3-inference-selection-v2/selection-lock.json"
)
SOURCE_PATH = Path("data/processed/hplt3-korean-phase3/ko.jsonl")
INTEGRITY_PATH = Path("data/processed/hplt3-korean-phase3/integrity.json")
PHASE3_MANIFEST_PATH = Path("runs/phase3/manifest.json")
CALIBRATION_BYTES = 8_000_000
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise ValueError("confirmation reconstruction requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("confirmation reconstruction requires a Git commit")
    return commit


def _require_result_never_published(path: Path) -> None:
    history = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if path.exists() or history:
        raise ValueError(
            f"confirmation evidence was already published or deleted: {path}"
        )


def _tracked_head_sha256(path: Path) -> str:
    return _tracked_head_identity(path)["sha256"]


def _tracked_head_identity(path: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not path.is_file() or path.read_bytes() != result.stdout:
        raise ValueError(f"confirmation input is not the exact HEAD blob: {path}")
    commit = subprocess.run(
        ["git", "rev-list", "-1", "HEAD", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"confirmation input has no tracked commit: {path}")
    return {
        "git_commit": commit,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


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
        raise ValueError("confirmation completion was not published exactly once")


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"confirmation chronology differs: {label}")


def _git_blob_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"confirmation implementation is absent at {commit}: {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _release(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def _npz_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, sequence_nll_nats=array)
    return output.getvalue()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_locked_context() -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
]:
    plan_sha256 = _tracked_head_sha256(PLAN_PATH)
    initial_sha256 = _tracked_head_sha256(INITIAL_EVIDENCE_PATH)
    lock_sha256 = _tracked_head_sha256(SELECTION_LOCK_PATH)
    identity_path = Path(INITIAL_MODEL_IDENTITY_LOCK_PATH)
    identity_sha256 = _tracked_head_sha256(identity_path)
    plan = _read_json(PLAN_PATH)
    initial = _read_json(INITIAL_EVIDENCE_PATH)
    lock = _read_json(SELECTION_LOCK_PATH)
    identity = _read_json(identity_path)
    validate_selection_plan_v2(plan)
    validate_calibration_evidence_manifest(initial, plan=plan)
    validate_selection_lock_v2(lock)
    validate_initial_model_identity_lock_v2(identity)
    validate_selection_lock_identity_binding_v2(lock, identity)
    validate_current_implementation_v2(
        identity,
        sha256_by_path={
            path: _tracked_head_sha256(Path(path))
            for path in identity["calibration_selection_implementation"][
                "file_order"
            ]
        },
        environment=runtime_environment_v2(),
    )
    decision = build_selection_decision_v2(
        calibration_bpb_matrix(initial, plan=plan)
    )
    if (
        lock["plan_sha256"] != plan_sha256
        or lock["calibration_evidence_manifest_sha256"] != initial_sha256
        or lock["initial_model_identity_lock_sha256"] != identity_sha256
        or initial["initial_model_identity_lock_sha256"] != identity_sha256
        or lock["decision"] != decision
    ):
        raise ValueError("confirmation selection lock does not canonically reconstruct")
    return plan, plan_sha256, initial, initial_sha256, lock, lock_sha256, identity


def _load_training_completions(
    *,
    plan: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    initial_model_identity: Mapping[str, Any],
    evaluator_git_commit: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load the exact committed completion receipts before any model replay."""

    required = required_confirmation_completion_families(selection_lock)
    implementation = initial_model_identity[
        "calibration_selection_implementation"
    ]
    selection_identity = _tracked_head_identity(SELECTION_LOCK_PATH)
    completions: dict[str, dict[str, Any]] = {}
    projection: dict[str, dict[str, Any]] = {}
    for family in required:
        path = confirmation_completion_path(family)
        artifact = _tracked_head_identity(path)
        _require_single_publication_history(path, artifact["git_commit"])
        completion = _read_json(path)
        validate_confirmation_training_completion(
            completion,
            selection_lock=selection_lock,
        )
        run_commit = completion["run_git_commit"]
        if (
            completion["selection_lock_artifact_sha256"]
            != selection_lock_artifact_sha256
            or completion["selection_lock_payload_sha256"]
            != selection_lock["lock_sha256"]
            or completion["implementation_manifest_sha256"]
            != implementation["manifest_sha256"]
            or completion["environment_sha256"]
            != implementation["environment_sha256"]
        ):
            raise ValueError("confirmation completion differs from its sealed inputs")
        run_manifest = completion["run_manifest"]
        run_manifest_path = Path(run_manifest["path"])
        if (
            not run_manifest_path.is_file()
            or hash_file(run_manifest_path) != run_manifest["artifact_sha256"]
        ):
            raise ValueError("confirmation run manifest changed after completion")
        _require_ancestor(
            selection_identity["git_commit"],
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
            evaluator_git_commit,
            f"{family} completion -> calibration evaluator",
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
        projection[family] = {
            "artifact": artifact,
            "completion_sha256": completion["completion_sha256"],
            "run_git_commit": run_commit,
        }
    return completions, projection


def _load_calibration_context(
    plan: Mapping[str, Any],
) -> tuple[bytes, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if (
        hash_file(SOURCE_PATH) != plan["initial_design"]["source_artifact_sha256"]
        or hash_file(INTEGRITY_PATH)
        != plan["initial_design"]["source_integrity_artifact_sha256"]
    ):
        raise ValueError("confirmation calibration source differs from the plan")
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    if (
        hashlib.sha256(stream.data).hexdigest()
        != plan["calibration_evaluator"]["input_stream_sha256"]
        or len(inputs) != CALIBRATION_SEQUENCE_COUNT
        or len(inputs) != plan["calibration_evaluator"]["sequence_count"]
    ):
        raise ValueError("confirmation calibration stream differs from the plan")
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    matrices = structural_patch_matrices(boundaries, whitespace, spacelike)
    for rate in CONVERSION_RATES:
        matrices.update(
            conversion_patch_matrices(boundaries, whitespace, rate=rate)
        )
    return stream.data, inputs, boundaries, matrices


def _entropy_matrices_and_auxiliary(
    seed: int,
    required_policies: tuple[str, ...],
    inputs: np.ndarray,
    boundaries: np.ndarray,
    plan: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    entropy_policies = tuple(
        policy for policy in required_policies if policy in THRESHOLD_POLICIES
    )
    if not entropy_policies:
        return {}, {}
    artifact_root = Path("artifacts/phase3") / f"seed-{seed}"
    run_root = Path("runs/phase3") / f"seed-{seed}"
    router_checkpoint = artifact_root / "router.pt"
    router_report_path = run_root / "router.json"
    cache_path = artifact_root / "threshold-patches.npz"
    diagnostics_path = run_root / "threshold-patch-diagnostics.json"
    for path in (
        router_checkpoint,
        router_report_path,
        cache_path,
        diagnostics_path,
        PHASE3_MANIFEST_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    router_report = _read_json(router_report_path)
    phase3_manifest = _read_json(PHASE3_MANIFEST_PATH)
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    router.load_state_dict(
        torch.load(router_checkpoint, map_location="cpu", weights_only=True)
    )
    router_state_sha256 = _state_sha256(router)
    if (
        router_report.get("seed") != seed
        or router_report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or router_report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or router_report.get("parameters") != FINAL_ROUTER_PARAMETER_COUNT
        or parameter_count(router) != FINAL_ROUTER_PARAMETER_COUNT
        or router_report.get("trained_state_sha256") != router_state_sha256
    ):
        raise ValueError(f"confirmation router identity differs for seed {seed}")
    scores = router_entropy_scores(router, inputs, device)
    matrices: dict[str, np.ndarray] = {}
    calibrations: dict[str, Any] = {}
    for policy in entropy_policies:
        candidate_mask = (
            None if policy == "entropy_threshold_full" else boundaries
        )
        calibration = calibrate_threshold(
            scores,
            PHASE3_MODEL_SPEC.patch_count,
            candidate_masks=candidate_mask,
            maximum_patch_length=24,
        )
        calibrations[policy] = calibration
        matrices[policy] = threshold_patch_matrix(
            scores,
            calibration.threshold_nats,
            candidate_masks=candidate_mask,
            maximum_patch_length=24,
        )
    del scores
    _release(router, device)

    diagnostics = _read_json(diagnostics_path)
    provenance = diagnostics.get("_provenance", {})
    if (
        provenance.get("kind") != "phase3_threshold_patch_cache"
        or provenance.get("seed") != seed
        or provenance.get("router_state_sha256") != router_state_sha256
        or provenance.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or provenance.get("maximum_patch_length") != 24
        or provenance.get("splits", {}).get("calibration", {}).get(
            "inputs_sha256"
        )
        != _array_sha256(inputs)
        or provenance.get("splits", {}).get("calibration", {}).get(
            "boundaries_sha256"
        )
        != _array_sha256(boundaries)
    ):
        raise ValueError(f"confirmation router provenance differs for seed {seed}")
    auxiliary: dict[str, dict[str, Any]] = {}
    with np.load(cache_path, allow_pickle=False) as cache:
        for policy in entropy_policies:
            matrix = matrices[policy]
            key = f"calibration__{policy}"
            calibration = calibrations[policy]
            expected_diagnostics = {
                **variable_patch_diagnostics(matrix, boundaries).to_dict(),
                "matrix_sha256": _array_sha256(matrix),
            }
            if (
                key not in cache.files
                or not np.array_equal(cache[key], matrix)
                or diagnostics.get("calibration", {}).get(policy)
                != calibration.to_dict()
                or diagnostics.get("splits", {}).get("calibration", {}).get(
                    policy
                )
                != expected_diagnostics
            ):
                raise ValueError(
                    f"confirmation router matrix differs for {seed}/{policy}"
                )
            auxiliary[policy] = {
                "calibration_stream_sha256": plan["calibration_evaluator"][
                    "input_stream_sha256"
                ],
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
                "router_checkpoint_artifact_sha256": hash_file(
                    router_checkpoint
                ),
                "router_checkpoint_path": str(router_checkpoint),
                "router_checkpoint_state_sha256": router_state_sha256,
                "router_config_sha256": canonical_sha256(
                    PHASE3_MODEL_SPEC.to_dict()
                ),
                "router_parameter_count": FINAL_ROUTER_PARAMETER_COUNT,
                "router_report_artifact_sha256": hash_file(router_report_path),
                "router_report_path": str(router_report_path),
                "router_training_stream_sha256": phase3_manifest["streams"][
                    "train"
                ]["selected_stream_sha256"],
                "seed": seed,
                "threshold_cache_artifact_sha256": hash_file(cache_path),
                "threshold_cache_path": str(cache_path),
                "threshold_diagnostics_artifact_sha256": hash_file(
                    diagnostics_path
                ),
                "threshold_diagnostics_path": str(diagnostics_path),
                "threshold_nats": calibration.threshold_nats,
            }
    return matrices, auxiliary


def _model_spec(descriptor: Mapping[str, Any]):
    if descriptor["model_family"] == "phase3":
        return PHASE3_MODEL_SPEC
    return conversion_model_spec(int(descriptor["patch_count"]))


def _load_existing_receipt(
    *,
    receipt_path: Path,
    nll_path: Path,
    lock: Mapping[str, Any],
    lock_artifact_sha256: str,
    evaluator_git_commit: str,
    artifact_role: str,
    descriptor: Mapping[str, Any],
    seed: int,
    report_path: Path,
    checkpoint_path: Path,
    matrix: np.ndarray,
    auxiliary: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray] | None:
    stages = (
        receipt_path.with_suffix(receipt_path.suffix + ".preparing"),
        nll_path.with_suffix(nll_path.suffix + ".preparing"),
    )
    if any(path.exists() for path in stages):
        raise ValueError(
            f"staged confirmation evidence requires forensic review: {receipt_path.parent}"
        )
    if receipt_path.exists() != nll_path.exists():
        raise ValueError(f"partial confirmation evidence exists: {receipt_path.parent}")
    if not receipt_path.exists():
        return None
    receipt = _read_json(receipt_path)
    validate_confirmation_calibration_receipt(receipt, selection_lock=lock)
    with np.load(nll_path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError("resumed confirmation NLL has unexpected keys")
        nll = archive["sequence_nll_nats"]
    if (
        nll.dtype != np.float32
        or nll.shape != (CALIBRATION_SEQUENCE_COUNT,)
        or not np.isfinite(nll).all()
        or np.any(nll < 0)
    ):
        raise ValueError("resumed confirmation NLL is invalid")
    reconstructed_bpb = math.fsum(float(value) for value in nll) / (
        len(nll) * (PHASE3_MODEL_SPEC.sequence_length - 1) * math.log(2)
    )
    model = build_main_model(
        _model_spec(descriptor),
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state_sha256 = _state_sha256(model)
    del model
    calibration = receipt["calibration"]
    if (
        receipt["selection_lock_artifact_sha256"] != lock_artifact_sha256
        or receipt["evaluator_git_commit"] != evaluator_git_commit
        or receipt["artifact_role"] != artifact_role
        or receipt["descriptor"] != descriptor
        or receipt["seed"] != seed
        or receipt["training_report"]["artifact_sha256"]
        != hash_file(report_path)
        or receipt["checkpoint"]["artifact_sha256"]
        != hash_file(checkpoint_path)
        or receipt["checkpoint"]["state_sha256"] != state_sha256
        or receipt["auxiliary"] != auxiliary
        or calibration["matrix_sha256"] != _array_sha256(matrix)
        or calibration["nll_artifact_sha256"] != hash_file(nll_path)
        or calibration["nll_array_sha256"] != _array_sha256(nll)
        or not math.isclose(
            float(calibration["bpb"]),
            reconstructed_bpb,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("resumed confirmation receipt identity differs")
    return receipt, np.ascontiguousarray(nll)


def _evaluate_receipt(
    *,
    lock: Mapping[str, Any],
    lock_artifact_sha256: str,
    artifact_role: str,
    descriptor: Mapping[str, Any],
    seed: int,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    stream_sha256: str,
    matrix: np.ndarray,
    auxiliary: Mapping[str, Any],
    evaluator_git_commit: str,
    device: str,
) -> dict[str, Any]:
    evidence_paths = expected_confirmation_paths(artifact_role, seed)
    model_paths = expected_model_paths(descriptor, seed)
    receipt_path = Path(evidence_paths["receipt"])
    nll_path = Path(evidence_paths["nll"])
    report_path = Path(model_paths["training_report"])
    checkpoint_path = Path(model_paths["checkpoint"])
    for path in (report_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = _load_existing_receipt(
        receipt_path=receipt_path,
        nll_path=nll_path,
        lock=lock,
        lock_artifact_sha256=lock_artifact_sha256,
        evaluator_git_commit=evaluator_git_commit,
        artifact_role=artifact_role,
        descriptor=descriptor,
        seed=seed,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        matrix=matrix,
        auxiliary=auxiliary,
    )
    print(
        f"confirmation calibration {seed}/{artifact_role}: mandatory checkpoint replay",
        flush=True,
    )
    replay = replay_confirmation_unit(
        descriptor=descriptor,
        seed=seed,
        inputs=inputs,
        boundaries=boundaries,
        matrix=matrix,
        auxiliary=auxiliary,
        device=device,
    )
    losses = replay["losses"]
    bpb = replay["bpb"]
    nll_bytes = _npz_bytes(losses)
    nll_artifact_sha256 = (
        hash_file(nll_path)
        if existing is not None
        else hashlib.sha256(nll_bytes).hexdigest()
    )
    receipt = build_confirmation_calibration_receipt(
        selection_lock=lock,
        selection_lock_artifact_sha256=lock_artifact_sha256,
        artifact_role=artifact_role,
        descriptor=descriptor,
        seed=seed,
        evaluator_git_commit=evaluator_git_commit,
        training_report={
            "artifact_sha256": replay["report_artifact_sha256"],
            "path": str(report_path),
        },
        checkpoint={
            "artifact_sha256": replay["checkpoint_artifact_sha256"],
            "path": str(checkpoint_path),
            "state_sha256": replay["checkpoint_state_sha256"],
        },
        auxiliary=auxiliary,
        calibration={
            "boundaries_sha256": _array_sha256(boundaries),
            "bpb": bpb,
            "count": len(losses),
            "dtype": str(losses.dtype),
            "inputs_sha256": _array_sha256(inputs),
            "matrix_sha256": _array_sha256(matrix),
            "nll_array_sha256": _array_sha256(losses),
            "nll_artifact_path": str(nll_path),
            "nll_artifact_sha256": nll_artifact_sha256,
            "predicted_bytes": len(losses)
            * (PHASE3_MODEL_SPEC.sequence_length - 1),
            "stream_sha256": stream_sha256,
        },
    )
    validate_confirmation_calibration_receipt(receipt, selection_lock=lock)
    if existing is not None:
        existing_receipt, existing_losses = existing
        if not np.array_equal(existing_losses, losses) or existing_receipt != receipt:
            raise ValueError(
                f"resumed confirmation evidence fails causal replay: {seed}/{artifact_role}"
            )
        print(
            json.dumps(
                {
                    "artifact_role": artifact_role,
                    "receipt_sha256": existing_receipt["receipt_sha256"],
                    "seed": seed,
                    "status": "exact_resume_replay_verified",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return existing_receipt
    publish_no_clobber(nll_path, nll_bytes)
    publish_no_clobber(receipt_path, _json_bytes(receipt))
    print(
        json.dumps(
            {
                "artifact_role": artifact_role,
                "receipt_sha256": receipt["receipt_sha256"],
                "seed": seed,
                "status": "complete",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt


def _main_locked() -> int:
    evaluator_commit = _require_clean_root()
    _require_result_never_published(CONFIRMATION_RESULT_PATH)
    (
        plan,
        plan_artifact_sha256,
        initial,
        initial_artifact_sha256,
        lock,
        lock_artifact_sha256,
        initial_model_identity,
    ) = _load_locked_context()
    completions, completion_projection = _load_training_completions(
        plan=plan,
        selection_lock=lock,
        selection_lock_artifact_sha256=lock_artifact_sha256,
        initial_model_identity=initial_model_identity,
        evaluator_git_commit=evaluator_commit,
    )
    if plan["calibration_evaluator"]["device"] != "mps":
        raise ValueError("selection plan does not authorize MPS reconstruction")
    device = resolve_device("mps")
    stream, inputs, boundaries, shared_matrices = (
        load_confirmation_calibration_context(plan)
    )
    stream_sha256 = hashlib.sha256(stream).hexdigest()
    models = required_confirmation_models(lock)
    required_policies = tuple(model["descriptor"]["policy"] for model in models)
    receipts: dict[int, dict[str, dict[str, Any]]] = {}
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
        receipts[seed] = {}
        for model in models:
            role = model["artifact_role"]
            descriptor = model["descriptor"]
            policy = descriptor["policy"]
            if policy not in matrices:
                raise ValueError(f"confirmation matrix is missing: {role}/{policy}")
            receipts[seed][role] = _evaluate_receipt(
                lock=lock,
                lock_artifact_sha256=lock_artifact_sha256,
                artifact_role=role,
                descriptor=descriptor,
                seed=seed,
                inputs=inputs,
                boundaries=boundaries,
                stream_sha256=stream_sha256,
                matrix=matrices[policy],
                auxiliary=entropy_auxiliary.get(policy, {"kind": "none"}),
                evaluator_git_commit=evaluator_commit,
                device=device,
            )
        del matrices, entropy_matrices, entropy_auxiliary
        gc.collect()
        torch.mps.empty_cache()
    if _git_commit() != evaluator_commit or _git_status().strip():
        raise RuntimeError("Git HEAD/worktree changed during confirmation reconstruction")
    validate_receipts_against_training_completions(
        selection_lock=lock,
        receipts=receipts,
        completions=completions,
    )
    manifest = build_confirmation_evidence_manifest(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        initial_calibration_evidence_artifact_sha256=initial_artifact_sha256,
        initial_calibration_evidence_payload_sha256=initial["manifest_sha256"],
        selection_lock=lock,
        selection_lock_artifact_sha256=lock_artifact_sha256,
        evaluator_git_commit=evaluator_commit,
        training_completions=completion_projection,
        receipts=receipts,
    )
    publish_no_clobber(CONFIRMATION_RESULT_PATH, _json_bytes(manifest))
    if _git_commit() != evaluator_commit:
        raise RuntimeError("Git HEAD changed while sealing confirmation evidence")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(CONFIRMATION_RESULT_PATH),
                "status": "complete_pending_commit",
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
