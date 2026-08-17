#!/usr/bin/env python3
"""Run the preregistered 19.6M-parameter Korean Phase 3 experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.neural_data import NeuralStream, build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    build_router,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import (
    evaluate_main_model,
    evaluate_router,
    resolve_device,
    router_entropy_scores,
    shuffled_indices,
    train_main_model,
    train_router,
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
    PHASE3_POLICIES,
    STRUCTURAL_POLICIES,
    THRESHOLD_POLICIES,
    merge_phase3_manifest,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.phase3_confirmation import (
    CONFIRMATION_ONLY_SEEDS,
    SELECTED_REFERENCE_AUTHORIZATION_KIND_V3,
    load_selected_reference_authorization_v3,
    load_run_confirmation_authorization,
    validate_confirmation_request,
    validate_selected_reference_request_v3,
)
from jamoflow.inference_selection_plan import (
    SELECTION_LOCK_PATH,
    SELECTION_PLAN_PATH,
)
from jamoflow.inference_initial_model_identity_v2 import (
    INITIAL_MODEL_IDENTITY_LOCK_PATH,
    runtime_environment_v2,
    validate_current_implementation_v2,
    validate_initial_model_identity_lock_v2,
    validate_selection_lock_identity_binding_v2,
)
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.inference_confirmation_evidence_v2 import (
    PHASE3_REFERENCE_COMPLETION_PATH,
    build_confirmation_training_completion,
    validate_confirmation_training_completion,
)


SPLITS = ("train", "calibration", "test")
DEFAULT_SEEDS = (1729, 2718, 31415)
KNOWN_SEEDS = (*DEFAULT_SEEDS, 57721, 65537)
FULL_LIMITS = {
    "train": 128_000_000,
    "calibration": 8_000_000,
    "test": 16_000_000,
}
QUICK_LIMITS = {
    "train": 131_072,
    "calibration": 32_768,
    "test": 32_768,
}
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
MAIN_REPORT_KEYS = {
    "seed",
    "policy",
    "parameters",
    "initialization_sha256",
    "trained_state_sha256",
    "training_order_sha256",
    "patch_matrix_sha256",
    "patch_diagnostics",
    "training",
    "evaluation",
    "model_spec",
    "optimization_spec",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise ValueError(f"partial JSON artifact requires forensic recovery: {temporary}")
    with temporary.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
    temporary.replace(path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _clean_git_commit() -> str:
    commit = _git_commit()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if not commit or status.returncode != 0 or status.stdout.strip():
        raise ValueError("Phase 3 evidence requires a clean committed worktree")
    return commit


def _git_status() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("Phase 3 Git status failed closed")
    return result.stdout


def _require_unchanged_clean_git(expected_commit: str) -> None:
    if _git_commit() != expected_commit or _git_status().strip():
        raise RuntimeError("Git HEAD/worktree changed during Phase 3 evidence execution")


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


def _git_path_history(path: Path) -> str:
    return subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _start_selected_reference_attempt(
    *,
    artifact_root: Path,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    run_git_commit: str,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> tuple[Path, Path]:
    completion_path = Path(PHASE3_REFERENCE_COMPLETION_PATH)
    if completion_path.exists() or _git_path_history(completion_path):
        raise ValueError("selected-reference completion was already published")
    active = artifact_root / ".publication-selected-reference-active.json"
    completed = artifact_root / ".publication-selected-reference-completed.json"
    payload = {
        "family": "phase3_reference",
        "policies": list(policies),
        "run_git_commit": run_git_commit,
        "seeds": list(seeds),
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_payload_sha256": selection_lock["lock_sha256"],
    }
    expected = _json_bytes(payload)
    if completed.exists():
        raise ValueError("selected-reference completed marker requires forensic review")
    if active.exists():
        if active.is_symlink() or active.read_bytes() != expected:
            raise ValueError("selected-reference active attempt differs")
    else:
        target_paths = [
            path
            for seed in seeds
            for policy in policies
            for path in (
                Path("runs/phase3") / f"seed-{seed}" / f"{policy}.json",
                artifact_root / f"seed-{seed}" / f"{policy}.pt",
                artifact_root / f"seed-{seed}" / f"{policy}-test-nll.npz",
            )
        ]
        if any(path.exists() for path in target_paths):
            raise ValueError(
                "selected-reference artifacts exist without their active attempt"
            )
        publish_no_clobber(active, expected)
    return active, completed


def _selected_reference_auxiliary(
    *, seed: int, policy: str, run_root: Path, artifact_root: Path
) -> dict[str, Any]:
    if policy not in THRESHOLD_POLICIES:
        return {"kind": "none"}
    report_root = run_root / f"seed-{seed}"
    artifact_seed = artifact_root / f"seed-{seed}"
    router_report_path = report_root / "router.json"
    router_checkpoint_path = artifact_seed / "router.pt"
    cache_path = artifact_seed / "threshold-patches.npz"
    diagnostics_path = report_root / "threshold-patch-diagnostics.json"
    router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
    return {
        "kind": "entropy_router_artifacts",
        "router_checkpoint_artifact_sha256": _sha256_file(
            router_checkpoint_path
        ),
        "router_checkpoint_path": str(router_checkpoint_path),
        "router_checkpoint_state_sha256": router_report[
            "trained_state_sha256"
        ],
        "router_report_artifact_sha256": _sha256_file(router_report_path),
        "router_report_path": str(router_report_path),
        "threshold_cache_artifact_sha256": _sha256_file(cache_path),
        "threshold_cache_path": str(cache_path),
        "threshold_diagnostics_artifact_sha256": _sha256_file(
            diagnostics_path
        ),
        "threshold_diagnostics_path": str(diagnostics_path),
    }


def _complete_selected_reference_attempt(
    *,
    active: Path,
    completed: Path,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    identity: Mapping[str, Any],
    run_git_commit: str,
    manifest_path: Path,
    run_root: Path,
    artifact_root: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    units: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        units[seed] = {}
        for policy in policies:
            report_path = run_root / f"seed-{seed}" / f"{policy}.json"
            checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            units[seed][policy] = {
                "auxiliary": _selected_reference_auxiliary(
                    seed=seed,
                    policy=policy,
                    run_root=run_root,
                    artifact_root=artifact_root,
                ),
                "checkpoint_artifact_sha256": _sha256_file(checkpoint_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_state_sha256": report["trained_state_sha256"],
                "training_report_artifact_sha256": _sha256_file(report_path),
                "training_report_path": str(report_path),
            }
    implementation = identity["calibration_selection_implementation"]
    completion = build_confirmation_training_completion(
        selection_lock=selection_lock,
        selection_lock_artifact_sha256=selection_lock_artifact_sha256,
        family="phase3_reference",
        run_git_commit=run_git_commit,
        run_manifest={
            "artifact_sha256": _sha256_file(manifest_path),
            "path": str(manifest_path),
        },
        implementation_manifest_sha256=implementation["manifest_sha256"],
        environment_sha256=implementation["environment_sha256"],
        units=units,
    )
    validate_confirmation_training_completion(
        completion, selection_lock=selection_lock
    )
    publish_no_clobber(Path(PHASE3_REFERENCE_COMPLETION_PATH), _json_bytes(completion))
    publish_no_clobber(
        completed,
        _json_bytes(
            {
                "completion_sha256": completion["completion_sha256"],
                "family": "phase3_reference",
            }
        ),
    )
    active.unlink()
    return completion


def _require_tracked_head_artifact(path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if (
        result.returncode != 0
        or not path.is_file()
        or path.read_bytes() != result.stdout
    ):
        raise ValueError(f"Phase 3 authorization is not the exact HEAD blob: {path}")
    return _sha256_file(path)


def _sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_artifact_metadata(data_root: Path) -> dict[str, dict[str, Any]]:
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
    if not source_path.exists() or not integrity_path.exists():
        raise FileNotFoundError("Phase 3 processed source or integrity file is missing")
    source_hash = _sha256_file(source_path)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (
        integrity.get("dataset_id") != "hplt3-korean-phase3"
        or integrity.get("output", {}).get("output_bytes")
        != source_path.stat().st_size
        or integrity.get("output", {}).get("output_sha256") != source_hash
    ):
        raise ValueError("Phase 3 processed source differs from integrity metadata")
    return {
        "source_artifact": {
            "filename": "ko.jsonl",
            "bytes": source_path.stat().st_size,
            "sha256": source_hash,
        },
        "source_integrity_artifact": {
            "filename": "integrity.json",
            "bytes": integrity_path.stat().st_size,
            "sha256": _sha256_file(integrity_path),
        },
    }


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _structural_cache_provenance(
    boundaries: dict[str, np.ndarray],
    whitespace: dict[str, np.ndarray],
    spacelike: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "phase3_structural_patch_cache",
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "splits": {
            split: {
                "boundaries_sha256": _array_sha256(boundaries[split]),
                "whitespace_sha256": _array_sha256(whitespace[split]),
                "spacelike_sha256": _array_sha256(spacelike[split]),
            }
            for split in SPLITS
        },
    }


def _threshold_cache_provenance(
    seed: int,
    router_state_sha256: str,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    *,
    evidence_binding: dict[str, Any] | None = None,
    requested_policies: tuple[str, ...] = THRESHOLD_POLICIES,
) -> dict[str, Any]:
    provenance = {
        "schema_version": 1,
        "kind": "phase3_threshold_patch_cache",
        "seed": seed,
        "router_state_sha256": router_state_sha256,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "maximum_patch_length": 24,
        "splits": {
            split: {
                "inputs_sha256": _array_sha256(inputs[split]),
                "boundaries_sha256": _array_sha256(boundaries[split]),
            }
            for split in SPLITS
        },
    }
    if evidence_binding is not None:
        provenance["evidence_binding"] = evidence_binding
        provenance["requested_policies"] = list(requested_policies)
    return provenance


def _cache_provenance_matches(
    diagnostics_path: Path,
    expected: dict[str, Any],
) -> bool:
    try:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return diagnostics.get("_provenance") == expected


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cpu_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to("cpu")
        for name, value in model.state_dict().items()
    }


def _save_torch_state(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise ValueError(f"partial checkpoint requires forensic recovery: {temporary}")
    with temporary.open("xb") as output:
        torch.save(state, output)
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise ValueError(f"partial NPZ artifact requires forensic recovery: {temporary}")
    with temporary.open("xb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _load_streams(
    data_root: Path,
    limits: dict[str, int],
) -> tuple[
    dict[str, NeuralStream],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    path = data_root / "ko.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    streams: dict[str, NeuralStream] = {}
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    spacelike: dict[str, np.ndarray] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            path,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=limits[split],
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        split_whitespace = compact_whitespace_mask(stream.data).reshape(
            -1, stream.sequence_length
        )
        split_spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(
            -1, stream.sequence_length
        )
        streams[split] = stream
        inputs[split] = split_inputs
        boundaries[split] = split_boundaries
        whitespace[split] = split_whitespace
        spacelike[split] = split_spacelike
        print(
            f"data ko/{split}: {stream.selected_bytes:,} bytes, "
            f"{stream.sequence_count:,} sequences",
            flush=True,
        )
    return streams, inputs, boundaries, whitespace, spacelike


def _matrix_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as loaded:
        matrices = {split: {} for split in SPLITS}
        for key in loaded.files:
            split, policy = key.split("__", 1)
            matrices[split][policy] = loaded[key]
    return matrices


def _save_matrix_cache(
    path: Path,
    matrices: dict[str, dict[str, np.ndarray]],
    policies: tuple[str, ...],
) -> None:
    _save_npz(
        path,
        **{
            f"{split}__{policy}": matrices[split][policy]
            for split in SPLITS
            for policy in policies
        },
    )


def _structural_matrices(
    boundaries: dict[str, np.ndarray],
    whitespace: dict[str, np.ndarray],
    spacelike: dict[str, np.ndarray],
    artifact_root: Path,
    run_root: Path,
    *,
    force: bool,
) -> dict[str, dict[str, np.ndarray]]:
    cache_path = artifact_root / "structural-patches.npz"
    diagnostics_path = run_root / "structural-patch-diagnostics.json"
    provenance = _structural_cache_provenance(
        boundaries,
        whitespace,
        spacelike,
    )
    if (
        cache_path.exists()
        and diagnostics_path.exists()
        and not force
        and _cache_provenance_matches(diagnostics_path, provenance)
    ):
        matrices = _matrix_cache(cache_path)
        if all(
            set(matrices[split]) == set(STRUCTURAL_POLICIES)
            for split in SPLITS
        ):
            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            for split in SPLITS:
                for policy in STRUCTURAL_POLICIES:
                    validate_padded_patch_matrix(
                        matrices[split][policy],
                        PHASE3_MODEL_SPEC.sequence_length,
                    )
                    expected = {
                        **variable_patch_diagnostics(
                            matrices[split][policy],
                            boundaries[split],
                        ).to_dict(),
                        "matrix_sha256": _array_sha256(
                            matrices[split][policy]
                        ),
                    }
                    if diagnostics.get(split, {}).get(policy) != expected:
                        raise ValueError(
                            f"structural cache content mismatch: {split}/{policy}"
                        )
            print("loaded Phase 3 structural patch cache", flush=True)
            return matrices
    elif cache_path.exists() and diagnostics_path.exists() and not force:
        print("ignoring stale Phase 3 structural patch cache", flush=True)

    matrices: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {"_provenance": provenance}
    for split in SPLITS:
        print(f"constructing structural patches for {split}", flush=True)
        matrices[split] = structural_patch_matrices(
            boundaries[split],
            whitespace[split],
            spacelike[split],
        )
        diagnostics[split] = {
            policy: {
                **variable_patch_diagnostics(
                    matrices[split][policy], boundaries[split]
                ).to_dict(),
                "matrix_sha256": _array_sha256(matrices[split][policy]),
            }
            for policy in STRUCTURAL_POLICIES
        }
    _save_matrix_cache(cache_path, matrices, STRUCTURAL_POLICIES)
    _write_json(diagnostics_path, diagnostics)
    return matrices


def _ensure_router(
    seed: int,
    inputs: dict[str, np.ndarray],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    *,
    force: bool,
    evidence_binding: dict[str, Any] | None = None,
) -> Any:
    checkpoint = artifact_directory / "router.pt"
    report_path = run_directory / "router.json"
    partials = (
        checkpoint.with_suffix(checkpoint.suffix + ".part"),
        report_path.with_suffix(report_path.suffix + ".part"),
    )
    if any(path.exists() for path in partials):
        raise ValueError(f"seed {seed}: partial router staging artifact exists")
    if (
        evidence_binding is not None
        and checkpoint.exists() != report_path.exists()
    ):
        raise ValueError(f"seed {seed}: partial router result requires recovery")
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    initialization_sha256 = _state_dict_sha256(router)
    if checkpoint.exists() and report_path.exists() and not force:
        router.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("seed") != seed
            or report.get("parameters") != parameter_count(router)
            or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
            or report.get("optimization_spec")
            != PHASE3_OPTIMIZATION_SPEC.to_dict()
            or report.get("initialization_sha256") != initialization_sha256
            or report.get("training_order_sha256")
            != _array_sha256(shuffled_indices(len(inputs["train"]), seed))
            or (
                evidence_binding is not None
                and report.get("evidence_binding") != evidence_binding
            )
        ):
            raise ValueError(f"seed {seed}: stale router report")
        expected_hash = report.get("trained_state_sha256")
        if expected_hash is None:
            raise ValueError(
                f"seed {seed}: router report lacks trained-state provenance"
            )
        if _state_dict_sha256(router) != expected_hash:
            raise ValueError(f"seed {seed}: router checkpoint hash mismatch")
        print(f"seed {seed}: loaded Phase 3 router", flush=True)
        return router

    order = shuffled_indices(len(inputs["train"]), seed)
    print(
        f"seed {seed}: training {parameter_count(router):,}-parameter router",
        flush=True,
    )
    training = train_router(
        router,
        inputs["train"],
        order,
        device,
        PHASE3_OPTIMIZATION_SPEC,
    )
    evaluations = {
        split: evaluate_router(router, inputs[split], device).to_dict()
        for split in ("calibration", "test")
    }
    trained_state_sha256 = _state_dict_sha256(router)
    _save_torch_state(checkpoint, _cpu_state_dict(router))
    report = {
        "seed": seed,
        "parameters": parameter_count(router),
        "initialization_sha256": initialization_sha256,
        "trained_state_sha256": trained_state_sha256,
        "training_order_sha256": _array_sha256(order),
        "training": training.to_dict(),
        "evaluation": evaluations,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
    }
    if evidence_binding is not None:
        report["evidence_binding"] = evidence_binding
    _write_json(report_path, report)
    return router


def _threshold_matrices(
    seed: int,
    router: Any,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    *,
    force: bool,
    evidence_binding: dict[str, Any] | None = None,
    requested_policies: tuple[str, ...] = THRESHOLD_POLICIES,
) -> dict[str, dict[str, np.ndarray]]:
    if (
        not requested_policies
        or len(set(requested_policies)) != len(requested_policies)
        or set(requested_policies) - set(THRESHOLD_POLICIES)
    ):
        raise ValueError("threshold request must be a nonempty exact policy subset")
    cache_path = artifact_directory / "threshold-patches.npz"
    diagnostics_path = run_directory / "threshold-patch-diagnostics.json"
    partials = (
        cache_path.with_suffix(cache_path.suffix + ".part"),
        diagnostics_path.with_suffix(diagnostics_path.suffix + ".part"),
    )
    if any(path.exists() for path in partials):
        raise ValueError(f"seed {seed}: partial threshold staging artifact exists")
    if (
        evidence_binding is not None
        and cache_path.exists() != diagnostics_path.exists()
    ):
        raise ValueError(f"seed {seed}: partial threshold result requires recovery")
    provenance = _threshold_cache_provenance(
        seed,
        _state_dict_sha256(router),
        inputs,
        boundaries,
        evidence_binding=evidence_binding,
        requested_policies=requested_policies,
    )
    if (
        cache_path.exists()
        and diagnostics_path.exists()
        and not force
        and _cache_provenance_matches(diagnostics_path, provenance)
    ):
        matrices = _matrix_cache(cache_path)
        if all(
            set(matrices[split]) == set(requested_policies)
            for split in SPLITS
        ):
            diagnostics = json.loads(
                diagnostics_path.read_text(encoding="utf-8")
            )
            for split in SPLITS:
                for policy in requested_policies:
                    validate_padded_patch_matrix(
                        matrices[split][policy],
                        PHASE3_MODEL_SPEC.sequence_length,
                    )
                    expected = {
                        **variable_patch_diagnostics(
                            matrices[split][policy],
                            boundaries[split],
                        ).to_dict(),
                        "matrix_sha256": _array_sha256(
                            matrices[split][policy]
                        ),
                    }
                    if diagnostics.get("splits", {}).get(split, {}).get(
                        policy
                    ) != expected:
                        raise ValueError(
                            f"threshold cache content mismatch: "
                            f"{seed}/{split}/{policy}"
                        )
            print(f"seed {seed}: loaded entropy patch cache", flush=True)
            return matrices
    elif (
        evidence_binding is not None
        and cache_path.exists()
        and diagnostics_path.exists()
        and not force
    ):
        raise ValueError(f"seed {seed}: threshold evidence binding differs")
    elif cache_path.exists() and diagnostics_path.exists() and not force:
        print(f"seed {seed}: ignoring stale entropy patch cache", flush=True)

    print(f"seed {seed}: scoring calibration entropy", flush=True)
    calibration_scores = router_entropy_scores(
        router, inputs["calibration"], device
    )
    calibrations = {}
    if "entropy_threshold_full" in requested_policies:
        calibrations["entropy_threshold_full"] = calibrate_threshold(
            calibration_scores,
            PHASE3_MODEL_SPEC.patch_count,
            maximum_patch_length=24,
        )
    if "entropy_threshold_codepoint" in requested_policies:
        calibrations["entropy_threshold_codepoint"] = calibrate_threshold(
            calibration_scores,
            PHASE3_MODEL_SPEC.patch_count,
            candidate_masks=boundaries["calibration"],
            maximum_patch_length=24,
        )
    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    diagnostics: dict[str, Any] = {
        "_provenance": provenance,
        "seed": seed,
        "calibration": {
            policy: calibration.to_dict()
            for policy, calibration in calibrations.items()
        },
        "splits": {},
    }
    for split in SPLITS:
        print(f"seed {seed}: scoring/building entropy patches for {split}", flush=True)
        scores = (
            calibration_scores
            if split == "calibration"
            else router_entropy_scores(router, inputs[split], device)
        )
        if "entropy_threshold_full" in calibrations:
            matrices[split]["entropy_threshold_full"] = threshold_patch_matrix(
                scores,
                calibrations["entropy_threshold_full"].threshold_nats,
                maximum_patch_length=24,
            )
        if "entropy_threshold_codepoint" in calibrations:
            matrices[split]["entropy_threshold_codepoint"] = (
                threshold_patch_matrix(
                    scores,
                    calibrations[
                        "entropy_threshold_codepoint"
                    ].threshold_nats,
                    candidate_masks=boundaries[split],
                    maximum_patch_length=24,
                )
            )
        diagnostics["splits"][split] = {
            policy: {
                **variable_patch_diagnostics(
                    matrices[split][policy], boundaries[split]
                ).to_dict(),
                "matrix_sha256": _array_sha256(matrices[split][policy]),
            }
            for policy in requested_policies
        }
        if split != "calibration":
            del scores
    del calibration_scores
    _save_matrix_cache(cache_path, matrices, requested_policies)
    _write_json(diagnostics_path, diagnostics)
    return matrices


def _policy_complete(
    report: Path,
    checkpoint: Path,
    *,
    save_checkpoints: bool,
) -> bool:
    return report.exists() and (checkpoint.exists() or not save_checkpoints)


def _validate_completed_policy(
    report_path: Path,
    checkpoint_path: Path,
    losses_path: Path,
    *,
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    save_checkpoints: bool,
    evidence_binding: dict[str, Any] | None = None,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization_hash = _state_dict_sha256(model)
    order_hash = _array_sha256(shuffled_indices(len(inputs["train"]), seed))
    expected_report_keys = set(MAIN_REPORT_KEYS)
    if evidence_binding is not None:
        expected_report_keys.add("evidence_binding")
    if (
        set(report) != expected_report_keys
        or report.get("seed") != seed
        or report.get("policy") != policy
        or report.get("parameters") != parameter_count(model)
        or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("initialization_sha256") != initialization_hash
        or report.get("training_order_sha256") != order_hash
        or (
            evidence_binding is not None
            and report.get("evidence_binding") != evidence_binding
        )
    ):
        raise ValueError(
            f"stale Phase 3 result for {seed}/{policy}; rerun with --force"
        )
    for split in SPLITS:
        matrix = matrices[split][policy]
        validate_padded_patch_matrix(
            matrix,
            PHASE3_MODEL_SPEC.sequence_length,
        )
        if report.get("patch_matrix_sha256", {}).get(split) != _array_sha256(
            matrix
        ) or report.get("patch_diagnostics", {}).get(split) != (
            variable_patch_diagnostics(matrix, boundaries[split]).to_dict()
        ):
            raise ValueError(
                f"stale Phase 3 matrix for {seed}/{policy}/{split}; "
                "rerun with --force"
            )
    if save_checkpoints:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        if _state_dict_sha256(model) != report.get("trained_state_sha256"):
            raise ValueError(
                f"stale Phase 3 checkpoint for {seed}/{policy}; "
                "rerun with --force"
            )
    del model
    with np.load(losses_path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError(f"unexpected Phase 3 loss keys: {losses_path}")
        stored = archive["sequence_nll_nats"]
        if stored.dtype != np.float32:
            raise ValueError(f"unexpected Phase 3 loss dtype: {losses_path}")
        losses = stored.astype(np.float64)
    if (
        losses.shape != (len(inputs["test"]),)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError(f"invalid Phase 3 losses: {losses_path}")
    evaluation = report.get("evaluation", {}).get("test", {})
    predicted_bytes = len(losses) * (PHASE3_MODEL_SPEC.sequence_length - 1)
    expected_bpb = float(losses.sum()) / (predicted_bytes * np.log(2))
    if (
        evaluation.get("examples") != len(losses)
        or evaluation.get("predicted_bytes") != predicted_bytes
        or not isinstance(evaluation.get("bpb"), (int, float))
        or not np.isclose(
            float(evaluation["bpb"]),
            expected_bpb,
            rtol=0,
            atol=1e-7,
        )
    ):
        raise ValueError(
            f"stale Phase 3 loss report for {seed}/{policy}; "
            "rerun with --force"
        )


def _train_policy(
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    *,
    force: bool,
    save_checkpoints: bool,
    evidence_binding: dict[str, Any] | None = None,
) -> None:
    report_path = run_directory / f"{policy}.json"
    checkpoint = artifact_directory / f"{policy}.pt"
    losses_path = artifact_directory / f"{policy}-test-nll.npz"
    partials = (
        report_path.with_suffix(report_path.suffix + ".part"),
        checkpoint.with_suffix(checkpoint.suffix + ".part"),
        losses_path.with_suffix(losses_path.suffix + ".part"),
    )
    if any(path.exists() for path in partials):
        raise ValueError(
            f"partial Phase 3 staging artifact exists for {seed}/{policy}"
        )
    artifact_presence = (
        report_path.exists(),
        checkpoint.exists() if save_checkpoints else True,
        losses_path.exists(),
    )
    if any(artifact_presence) and not all(artifact_presence) and not force:
        raise ValueError(
            f"partial Phase 3 result requires forensic recovery: {seed}/{policy}"
        )
    if (
        _policy_complete(
            report_path,
            checkpoint,
            save_checkpoints=save_checkpoints,
        )
        and losses_path.exists()
        and not force
    ):
        _validate_completed_policy(
            report_path,
            checkpoint,
            losses_path,
            seed=seed,
            policy=policy,
            inputs=inputs,
            boundaries=boundaries,
            matrices=matrices,
            save_checkpoints=save_checkpoints,
            evidence_binding=evidence_binding,
        )
        print(f"seed {seed}/{policy}: already complete", flush=True)
        return

    for split in SPLITS:
        validate_padded_patch_matrix(
            matrices[split][policy], PHASE3_MODEL_SPEC.sequence_length
        )
    order = shuffled_indices(len(inputs["train"]), seed)
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization_sha256 = _state_dict_sha256(model)
    print(
        f"seed {seed}/{policy}: training {parameter_count(model):,} parameters; "
        f"matrix width={matrices['train'][policy].shape[1]}",
        flush=True,
    )
    training = train_main_model(
        model,
        inputs["train"],
        matrices["train"][policy],
        order,
        device,
        PHASE3_OPTIMIZATION_SPEC,
    )

    evaluations: dict[str, Any] = {}
    test_losses: np.ndarray | None = None
    for split in ("calibration", "test"):
        summary, sequence_nll = evaluate_main_model(
            model,
            inputs[split],
            matrices[split][policy],
            device,
            batch_size=PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
            return_sequence_nll=split == "test",
        )
        evaluations[split] = summary.to_dict()
        if sequence_nll is not None:
            test_losses = sequence_nll
    if test_losses is None:
        raise AssertionError("test sequence losses were not produced")
    _save_npz(losses_path, sequence_nll_nats=test_losses)
    trained_state_sha256 = _state_dict_sha256(model)
    if save_checkpoints:
        _save_torch_state(checkpoint, _cpu_state_dict(model))
    report = {
        "seed": seed,
        "policy": policy,
        "parameters": parameter_count(model),
        "initialization_sha256": initialization_sha256,
        "trained_state_sha256": trained_state_sha256,
        "training_order_sha256": _array_sha256(order),
        "patch_matrix_sha256": {
            split: _array_sha256(matrices[split][policy])
            for split in SPLITS
        },
        "patch_diagnostics": {
            split: variable_patch_diagnostics(
                matrices[split][policy], boundaries[split]
            ).to_dict()
            for split in SPLITS
        },
        "training": training.to_dict(),
        "evaluation": evaluations,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
    }
    if evidence_binding is not None:
        report["evidence_binding"] = evidence_binding
    _write_json(report_path, report)
    _release_model(model, device)


def _run_locked(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    policies = tuple(args.policies)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or set(seeds) - set(KNOWN_SEEDS)
    ):
        raise ValueError("Phase 3 needs unique preregistered seeds")
    unknown = set(policies) - set(PHASE3_POLICIES)
    if unknown or len(set(policies)) != len(policies):
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    if not policies:
        raise ValueError("at least one policy is required")
    run_git_commit = _clean_git_commit()

    limits = dict(QUICK_LIMITS if args.quick else FULL_LIMITS)
    device = resolve_device(args.device)
    run_root = Path(
        args.run_root or ("runs/phase3-smoke" if args.quick else "runs/phase3")
    )
    artifact_root = Path(
        args.artifact_root
        or ("artifacts/phase3-smoke" if args.quick else "artifacts/phase3")
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}; seeds: {seeds}; policies: {policies}", flush=True)

    manifest_path = run_root / "manifest.json"
    authorization = None
    evidence_binding = None
    selection_lock: dict[str, Any] | None = None
    initial_model_identity: dict[str, Any] | None = None
    selected_reference_mode = False
    selection_lock_argument = getattr(args, "selection_lock", None)
    if set(seeds) & set(CONFIRMATION_ONLY_SEEDS):
        if args.quick:
            raise ValueError("confirmation seeds cannot be run as quick evidence")
        if not manifest_path.is_file():
            raise ValueError("confirmation requires the pre-confirmation run manifest")
        if selection_lock_argument is not None:
            if args.authorization_summary is not None:
                raise ValueError(
                    "primary and selected-reference authorizations are mutually exclusive"
                )
            if args.force or args.no_checkpoints:
                raise ValueError(
                    "selected-reference evidence forbids --force and --no-checkpoints"
                )
            selection_lock_path = Path(selection_lock_argument)
            selection_plan_path = Path(SELECTION_PLAN_PATH)
            if selection_lock_path != Path(SELECTION_LOCK_PATH):
                raise ValueError("selected reference requires the canonical selection lock")
            for path in (
                selection_lock_path,
                selection_plan_path,
            ):
                _require_tracked_head_artifact(path)
            authorization = load_selected_reference_authorization_v3(
                selection_lock_path,
                selection_plan_path,
            )
            selection_plan = json.loads(
                selection_plan_path.read_text(encoding="utf-8")
            )
            for path in (
                Path(selection_plan["execution_paths"]["calibration_evidence"]),
                Path(selection_plan["final_test"]["seal_path"]),
                Path(INITIAL_MODEL_IDENTITY_LOCK_PATH),
            ):
                _require_tracked_head_artifact(path)
            selection_lock = json.loads(
                selection_lock_path.read_text(encoding="utf-8")
            )
            initial_model_identity = json.loads(
                Path(INITIAL_MODEL_IDENTITY_LOCK_PATH).read_text(
                    encoding="utf-8"
                )
            )
            validate_initial_model_identity_lock_v2(initial_model_identity)
            if selection_lock["initial_model_identity_lock_sha256"] != (
                _sha256_file(Path(INITIAL_MODEL_IDENTITY_LOCK_PATH))
            ):
                raise ValueError("selected reference initial model identity differs")
            validate_selection_lock_identity_binding_v2(
                selection_lock, initial_model_identity
            )
            validate_current_implementation_v2(
                initial_model_identity,
                sha256_by_path={
                    path: _require_tracked_head_artifact(Path(path))
                    for path in initial_model_identity[
                        "calibration_selection_implementation"
                    ]["file_order"]
                },
                environment=runtime_environment_v2(),
            )
            validate_selected_reference_request_v3(seeds, policies, authorization)
            evidence_binding = {
                "authorization": authorization,
                "device": device,
                "git_worktree_clean_at_start": True,
                "kind": "selected_phase3_reference_training_evidence_v4",
                "run_git_commit": run_git_commit,
                "schema_version": 4,
            }
            selected_reference_mode = True
        else:
            validate_confirmation_request(seeds, policies)
            if args.authorization_summary is None:
                raise ValueError(
                    "confirmation seeds require --authorization-summary from corrected Gate I"
                )
            authorization = load_run_confirmation_authorization(
                Path(args.authorization_summary),
                manifest_path,
                seeds=seeds,
                policies=policies,
            )
    elif args.authorization_summary is not None or selection_lock_argument is not None:
        raise ValueError("authorization inputs are only valid for confirmation seeds")

    if selected_reference_mode and (
        device != "mps"
        or Path(args.data_root) != Path("data/processed/hplt3-korean-phase3")
        or run_root != Path("runs/phase3")
        or artifact_root != Path("artifacts/phase3")
    ):
        raise ValueError(
            "publication selected-reference confirmation requires canonical roots and Apple MPS"
        )

    data_root = Path(args.data_root)
    source_artifacts = _source_artifact_metadata(data_root)
    streams, inputs, boundaries, whitespace, spacelike = _load_streams(
        data_root, limits
    )
    structural = _structural_matrices(
        boundaries,
        whitespace,
        spacelike,
        artifact_root,
        run_root,
        force=args.force,
    )
    current_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick_smoke_only": bool(args.quick),
        "git_commit": run_git_commit,
        "git_worktree_clean_at_start": True,
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "language": "ko",
        "seeds": list(seeds),
        "policies": list(policies),
        "limits": limits,
        **source_artifacts,
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "force": bool(args.force),
        "save_checkpoints": not args.no_checkpoints,
        "authorization": authorization,
        "streams": {
            split: {
                **streams[split].metadata(),
                "selected_stream_sha256": _sha256_bytes(streams[split].data),
            }
            for split in SPLITS
        },
    }
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    if authorization is not None and authorization.get("authorization_kind") == (
        SELECTED_REFERENCE_AUTHORIZATION_KIND_V3
    ):
        if not isinstance(existing_manifest, dict):
            raise ValueError("selected reference requires an existing Phase 3 manifest")
        matching_invocations = [
            invocation
            for invocation in existing_manifest.get("invocations", ())
            if isinstance(invocation, dict)
            and tuple(invocation.get("seeds", ())) == seeds
            and tuple(invocation.get("policies", ())) == policies
            and invocation.get("authorization") == authorization
            and invocation.get("git_commit") == run_git_commit
            and invocation.get("force") is False
            and invocation.get("save_checkpoints") is True
        ]
        if len(matching_invocations) > 1:
            raise ValueError("selected-reference manifest has duplicate invocations")
        target_paths: list[Path] = []
        for seed in seeds:
            seed_run = run_root / f"seed-{seed}"
            seed_artifact = artifact_root / f"seed-{seed}"
            for policy in policies:
                target_paths.extend(
                    (
                        seed_run / f"{policy}.json",
                        seed_artifact / f"{policy}.pt",
                        seed_artifact / f"{policy}-test-nll.npz",
                    )
                )
            if authorization.get("required_auxiliary") == "entropy_router":
                target_paths.extend(
                    (
                        seed_run / "router.json",
                        seed_artifact / "router.pt",
                        seed_run / "threshold-patch-diagnostics.json",
                        seed_artifact / "threshold-patches.npz",
                    )
                )
        target_paths.extend(
            path.with_suffix(path.suffix + ".part")
            for path in tuple(target_paths)
        )
        if any(path.exists() for path in target_paths) and not matching_invocations:
            raise ValueError(
                "preexisting selected-reference artifacts lack exact authorization"
            )
        if matching_invocations:
            manifest = existing_manifest
        else:
            manifest = merge_phase3_manifest(existing_manifest, current_manifest)
            _write_json(manifest_path, manifest)
    else:
        manifest = merge_phase3_manifest(existing_manifest, current_manifest)
        _write_json(manifest_path, manifest)

    active: Path | None = None
    completed: Path | None = None
    if selected_reference_mode:
        if selection_lock is None or initial_model_identity is None:
            raise AssertionError("selected-reference identity context disappeared")
        active, completed = _start_selected_reference_attempt(
            artifact_root=artifact_root,
            selection_lock=selection_lock,
            selection_lock_artifact_sha256=_sha256_file(
                Path(SELECTION_LOCK_PATH)
            ),
            run_git_commit=run_git_commit,
            seeds=seeds,
            policies=policies,
        )

    needs_thresholds = bool(set(policies) & set(THRESHOLD_POLICIES))
    for seed in seeds:
        seed_run = run_root / f"seed-{seed}"
        seed_artifact = artifact_root / f"seed-{seed}"
        seed_run.mkdir(parents=True, exist_ok=True)
        seed_artifact.mkdir(parents=True, exist_ok=True)
        matrices = {split: dict(structural[split]) for split in SPLITS}
        if needs_thresholds:
            router = _ensure_router(
                seed,
                inputs,
                device,
                seed_artifact,
                seed_run,
                force=args.force,
                evidence_binding=evidence_binding,
            )
            threshold = _threshold_matrices(
                seed,
                router,
                inputs,
                boundaries,
                device,
                seed_artifact,
                seed_run,
                force=args.force,
                evidence_binding=evidence_binding,
                requested_policies=tuple(
                    policy for policy in policies if policy in THRESHOLD_POLICIES
                ),
            )
            for split in SPLITS:
                matrices[split].update(threshold[split])
            _release_model(router, device)

        for policy in policies:
            _train_policy(
                seed,
                policy,
                inputs,
                boundaries,
                matrices,
                device,
                seed_artifact,
                seed_run,
                force=args.force,
                save_checkpoints=not args.no_checkpoints,
                evidence_binding=evidence_binding,
            )

    _require_unchanged_clean_git(run_git_commit)
    if selected_reference_mode:
        if (
            active is None
            or completed is None
            or selection_lock is None
            or initial_model_identity is None
        ):
            raise AssertionError("selected-reference completion context disappeared")
        completion = _complete_selected_reference_attempt(
            active=active,
            completed=completed,
            selection_lock=selection_lock,
            selection_lock_artifact_sha256=_sha256_file(
                Path(SELECTION_LOCK_PATH)
            ),
            identity=initial_model_identity,
            run_git_commit=run_git_commit,
            manifest_path=manifest_path,
            run_root=run_root,
            artifact_root=artifact_root,
            seeds=seeds,
            policies=policies,
        )
        print(
            json.dumps(
                {
                    "completion_sha256": completion["completion_sha256"],
                    "status": "complete_pending_receipt_commit",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(f"completed Phase 3 runs under {run_root}", flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    with publication_mps_exclusive():
        return _run_locked(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--run-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(PHASE3_POLICIES),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--authorization-summary", type=Path)
    parser.add_argument("--selection-lock", type=Path)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
