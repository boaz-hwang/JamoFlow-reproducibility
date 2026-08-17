#!/usr/bin/env python3
"""Validate and aggregate Phase 3 quality runs without promoting text."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from jamoflow.document_inference import (
    DocumentWindowMap,
    document_cluster_contrast_summary,
    reconstruct_document_window_map,
)
from jamoflow.neural_data import NeuralStream, build_neural_stream
from jamoflow.neural_model import build_main_model, build_router, parameter_count
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
    STRUCTURAL_POLICIES,
    THRESHOLD_POLICIES,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.phase3_analysis import (
    empirical_nonnegative_bootstrap_tail,
    hierarchical_paired_bootstrap_estimates,
    holm_step_down_adjusted_values,
    paired_seed_lower_t_pvalue,
    phase3_test_strata,
)
from jamoflow.phase3_confirmation import (
    CONFIRMATION_ONLY_SEEDS,
    load_confirmation_authorization,
    validate_confirmation_invocations,
)
from jamoflow.neural_training import shuffled_indices


F = "fixed_byte_6"
C = "causal_codepoint_grid"
W = "causal_whitespace_grid"
S = "spacebyte_spacelike"
E = "entropy_threshold_full"
EC = "entropy_threshold_codepoint"
CONTRASTS = {
    "whitespace_minus_codepoint": (W, C),
    "whitespace_minus_fixed": (W, F),
    "codepoint_minus_fixed": (C, F),
    "entropy_codepoint_minus_entropy_full": (EC, E),
    "whitespace_minus_entropy_full": (W, E),
    "whitespace_minus_entropy_codepoint": (W, EC),
    "spacebyte_minus_whitespace": (S, W),
}
PRIMARY_FAMILY = (
    "whitespace_minus_codepoint",
    "whitespace_minus_fixed",
)
SPLITS = ("train", "calibration", "test")
TARGETS_PER_SEQUENCE = PHASE3_MODEL_SPEC.sequence_length - 1
INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_SEEDS = (*INITIAL_SEEDS, 57721, 65537)
FULL_LIMITS = {
    "train": 128_000_000,
    "calibration": 8_000_000,
    "test": 16_000_000,
}
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
MAIN_PARAMETERS = 19_596_096
ROUTER_PARAMETERS = 2_016_960
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"unexpected checkpoint entry in {path}")
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _reconstruct_data(
    manifest: dict[str, Any],
    data_root: Path,
) -> tuple[
    dict[str, NeuralStream],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
    if not source_path.exists() or not integrity_path.exists():
        raise FileNotFoundError("Phase 3 source or integrity artifact is missing")
    source_hash = _sha256(source_path)
    expected_source_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": source_hash,
    }
    expected_integrity_artifact = {
        "filename": "integrity.json",
        "bytes": integrity_path.stat().st_size,
        "sha256": _sha256(integrity_path),
    }
    if manifest.get("source_artifact") != expected_source_artifact:
        raise ValueError("Phase 3 source artifact differs from the manifest")
    if manifest.get("source_integrity_artifact") != expected_integrity_artifact:
        raise ValueError("Phase 3 integrity artifact differs from the manifest")
    source_integrity = _read_json(integrity_path)
    if (
        source_integrity.get("dataset_id") != "hplt3-korean-phase3"
        or source_integrity.get("output", {}).get("output_bytes")
        != source_path.stat().st_size
        or source_integrity.get("output", {}).get("output_sha256")
        != source_hash
    ):
        raise ValueError("Phase 3 source fails its processed-data integrity record")

    streams: dict[str, NeuralStream] = {}
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    spacelike: dict[str, np.ndarray] = {}
    expected_streams: dict[str, Any] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            source_path,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=FULL_LIMITS[split],
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        streams[split] = stream
        inputs[split] = split_inputs
        boundaries[split] = split_boundaries
        whitespace[split] = compact_whitespace_mask(stream.data).reshape(
            split_inputs.shape
        )
        spacelike[split] = spacebyte_causal_prefix_mask(stream.data).reshape(
            split_inputs.shape
        )
        expected_streams[split] = {
            **stream.metadata(),
            "selected_stream_sha256": hashlib.sha256(stream.data).hexdigest(),
        }
    if manifest.get("streams") != expected_streams:
        raise ValueError("Phase 3 streams differ from independent reconstruction")
    return (
        streams,
        inputs,
        boundaries,
        whitespace,
        spacelike,
        {
            "source_artifact": expected_source_artifact,
            "source_integrity_artifact": expected_integrity_artifact,
            "processed_integrity_dataset_id": source_integrity["dataset_id"],
            "all_split_streams_match_manifest": True,
        },
    )


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


def _reconstruct_structural_matrices(
    run_root: Path,
    artifact_root: Path,
    boundaries: dict[str, np.ndarray],
    whitespace: dict[str, np.ndarray],
    spacelike: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    cache_path = artifact_root / "structural-patches.npz"
    diagnostics_path = run_root / "structural-patch-diagnostics.json"
    diagnostics = _read_json(diagnostics_path)
    expected_provenance = _structural_cache_provenance(
        boundaries,
        whitespace,
        spacelike,
    )
    if diagnostics.get("_provenance") != expected_provenance:
        raise ValueError("Phase 3 structural cache provenance is stale")
    reconstructed: dict[str, dict[str, np.ndarray]] = {}
    expected_diagnostics: dict[str, Any] = {"_provenance": expected_provenance}
    for split in SPLITS:
        reconstructed[split] = structural_patch_matrices(
            boundaries[split],
            whitespace[split],
            spacelike[split],
        )
        expected_diagnostics[split] = {
            policy: {
                **variable_patch_diagnostics(
                    reconstructed[split][policy],
                    boundaries[split],
                ).to_dict(),
                "matrix_sha256": _array_sha256(
                    reconstructed[split][policy]
                ),
            }
            for policy in STRUCTURAL_POLICIES
        }
    if diagnostics != expected_diagnostics:
        raise ValueError("Phase 3 structural diagnostics differ from reconstruction")
    with np.load(cache_path, allow_pickle=False) as archive:
        expected_keys = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in STRUCTURAL_POLICIES
        }
        if set(archive.files) != expected_keys:
            raise ValueError("Phase 3 structural cache keys mismatch")
        for split in SPLITS:
            for policy in STRUCTURAL_POLICIES:
                cached = archive[f"{split}__{policy}"]
                if cached.dtype != np.uint16 or not np.array_equal(
                    cached,
                    reconstructed[split][policy],
                ):
                    raise ValueError(
                        f"Phase 3 structural cache mismatch: {split}/{policy}"
                    )
    return reconstructed, {
        "cache_artifact_sha256": _sha256(cache_path),
        "diagnostics_artifact_sha256": _sha256(diagnostics_path),
        "matrix_sha256": {
            split: {
                policy: expected_diagnostics[split][policy]["matrix_sha256"]
                for policy in STRUCTURAL_POLICIES
            }
            for split in SPLITS
        },
        "all_matrices_match_independent_reconstruction": True,
    }


def _load_threshold_matrices(
    seed: int,
    run_root: Path,
    artifact_root: Path,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    run_directory = run_root / f"seed-{seed}"
    artifact_directory = artifact_root / f"seed-{seed}"
    router_report_path = run_directory / "router.json"
    router_checkpoint_path = artifact_directory / "router.pt"
    diagnostics_path = run_directory / "threshold-patch-diagnostics.json"
    cache_path = artifact_directory / "threshold-patches.npz"
    for path in (
        router_report_path,
        router_checkpoint_path,
        diagnostics_path,
        cache_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    router_report = _read_json(router_report_path)
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    initialization_hash = _state_dict_sha256(router)
    if (
        router_report.get("seed") != seed
        or router_report.get("parameters") != ROUTER_PARAMETERS
        or router_report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or router_report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or router_report.get("initialization_sha256") != initialization_hash
        or router_report.get("training_order_sha256")
        != _array_sha256(shuffled_indices(len(inputs["train"]), seed))
    ):
        raise ValueError(f"Phase 3 router report mismatch for seed {seed}")
    router_training = router_report.get("training", {})
    if (
        router_training.get("examples") != len(inputs["train"])
        or router_training.get("steps")
        != math.ceil(
            len(inputs["train"]) / PHASE3_OPTIMIZATION_SPEC.router_batch_size
        )
        or any(
            router_report.get("evaluation", {}).get(split, {}).get("examples")
            != len(inputs[split])
            for split in ("calibration", "test")
        )
    ):
        raise ValueError(f"Phase 3 router report counts mismatch for seed {seed}")
    state_hash = _checkpoint_state_sha256(router_checkpoint_path)
    if state_hash != router_report.get("trained_state_sha256"):
        raise ValueError(f"Phase 3 router checkpoint mismatch for seed {seed}")
    if parameter_count(router) != ROUTER_PARAMETERS:
        raise ValueError("Phase 3 router parameter count changed")
    del router

    diagnostics = _read_json(diagnostics_path)
    expected_provenance = {
        "schema_version": 1,
        "kind": "phase3_threshold_patch_cache",
        "seed": seed,
        "router_state_sha256": state_hash,
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
    if diagnostics.get("_provenance") != expected_provenance:
        raise ValueError(f"Phase 3 threshold cache provenance mismatch: {seed}")
    if diagnostics.get("seed") != seed or set(
        diagnostics.get("calibration", {})
    ) != set(THRESHOLD_POLICIES):
        raise ValueError(f"Phase 3 threshold diagnostics mismatch: {seed}")
    if any(
        not isinstance(
            diagnostics["calibration"][policy].get("threshold_nats"),
            (int, float),
        )
        or not math.isfinite(
            float(diagnostics["calibration"][policy]["threshold_nats"])
        )
        for policy in THRESHOLD_POLICIES
    ):
        raise ValueError(f"Phase 3 threshold calibration mismatch: {seed}")
    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    with np.load(cache_path, allow_pickle=False) as archive:
        expected_keys = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in THRESHOLD_POLICIES
        }
        if set(archive.files) != expected_keys:
            raise ValueError(f"Phase 3 threshold cache keys mismatch: {seed}")
        for split in SPLITS:
            for policy in THRESHOLD_POLICIES:
                matrix = archive[f"{split}__{policy}"]
                if matrix.dtype != np.uint16:
                    raise ValueError("Phase 3 threshold matrix dtype mismatch")
                matrix = matrix.copy()
                validate_padded_patch_matrix(
                    matrix,
                    PHASE3_MODEL_SPEC.sequence_length,
                )
                expected_diagnostic = {
                    **variable_patch_diagnostics(
                        matrix,
                        boundaries[split],
                    ).to_dict(),
                    "matrix_sha256": _array_sha256(matrix),
                }
                if diagnostics.get("splits", {}).get(split, {}).get(policy) != (
                    expected_diagnostic
                ):
                    raise ValueError(
                        f"Phase 3 threshold diagnostics mismatch: "
                        f"{seed}/{split}/{policy}"
                    )
                matrices[split][policy] = matrix
    return matrices, {
        "router_checkpoint_state_sha256": state_hash,
        "router_checkpoint_artifact_sha256": _sha256(router_checkpoint_path),
        "router_report_artifact_sha256": _sha256(router_report_path),
        "threshold_cache_artifact_sha256": _sha256(cache_path),
        "threshold_diagnostics_artifact_sha256": _sha256(diagnostics_path),
        "cache_provenance_matches_current_router_and_streams": True,
        "selector_reconstruction_scope": (
            "cache lineage and matrix diagnostics; full entropy logits are not "
            "recomputed by the quality summarizer"
        ),
    }


def _load_runs(
    run_root: Path,
    artifact_root: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    structural_matrices: dict[str, dict[str, np.ndarray]],
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, np.ndarray]],
    dict[str, Any],
]:
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    evidence: dict[str, Any] = {}
    for seed in seeds:
        reports[seed] = {}
        losses[seed] = {}
        evidence[str(seed)] = {
            "loss_artifact_sha256": {},
            "training_report_artifact_sha256": {},
            "checkpoint_state_sha256": {},
            "checkpoint_artifact_sha256": {},
        }
        initial_model = build_main_model(
            PHASE3_MODEL_SPEC,
            seed=seed,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        if parameter_count(initial_model) != MAIN_PARAMETERS:
            raise ValueError("Phase 3 main-model parameter count changed")
        initialization_hash = _state_dict_sha256(initial_model)
        del initial_model
        order_hash = _array_sha256(shuffled_indices(len(inputs["train"]), seed))
        matrices = {
            split: dict(structural_matrices[split])
            for split in SPLITS
        }
        threshold_matrices: dict[str, dict[str, np.ndarray]] | None = None
        if set(policies) & set(THRESHOLD_POLICIES):
            threshold_matrices, threshold_evidence = _load_threshold_matrices(
                seed,
                run_root,
                artifact_root,
                inputs,
                boundaries,
            )
            for split in SPLITS:
                matrices[split].update(threshold_matrices[split])
            evidence[str(seed)]["router_and_threshold_cache"] = threshold_evidence
        for policy in policies:
            report_path = run_root / f"seed-{seed}" / f"{policy}.json"
            loss_path = (
                artifact_root
                / f"seed-{seed}"
                / f"{policy}-test-nll.npz"
            )
            checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
            if (
                not report_path.exists()
                or not loss_path.exists()
                or not checkpoint_path.exists()
            ):
                raise FileNotFoundError(
                    f"Phase 3 evidence is incomplete for seed {seed}/{policy}"
                )
            report = _read_json(report_path)
            if (
                set(report) != MAIN_REPORT_KEYS
                or report.get("seed") != seed
                or report.get("policy") != policy
                or report.get("parameters") != MAIN_PARAMETERS
                or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
                or report.get("optimization_spec")
                != PHASE3_OPTIMIZATION_SPEC.to_dict()
                or report.get("initialization_sha256") != initialization_hash
                or report.get("training_order_sha256") != order_hash
            ):
                raise ValueError(f"main training report mismatch in {report_path}")
            for split in SPLITS:
                matrix = matrices[split][policy]
                validate_padded_patch_matrix(
                    matrix,
                    PHASE3_MODEL_SPEC.sequence_length,
                )
                if report.get("patch_matrix_sha256", {}).get(split) != (
                    _array_sha256(matrix)
                ):
                    raise ValueError(
                        f"patch matrix mismatch for {seed}/{policy}/{split}"
                    )
                expected_diagnostics = variable_patch_diagnostics(
                    matrix,
                    boundaries[split],
                ).to_dict()
                if report.get("patch_diagnostics", {}).get(split) != (
                    expected_diagnostics
                ):
                    raise ValueError(
                        f"patch diagnostics mismatch for {seed}/{policy}/{split}"
                    )
            with np.load(loss_path, allow_pickle=False) as archive:
                if archive.files != ["sequence_nll_nats"]:
                    raise ValueError(f"unexpected loss keys in {loss_path}")
                stored_values = archive["sequence_nll_nats"]
                if stored_values.dtype != np.float32:
                    raise ValueError(f"unexpected loss dtype in {loss_path}")
                values = stored_values.astype(np.float64)
            if (
                values.ndim != 1
                or len(values) != len(inputs["test"])
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError(f"invalid sequence losses in {loss_path}")
            evaluation = report["evaluation"]["test"]
            predicted_bytes = len(values) * TARGETS_PER_SEQUENCE
            total_nll = float(values.sum())
            if (
                evaluation.get("examples") != len(values)
                or evaluation.get("predicted_bytes") != predicted_bytes
            ):
                raise ValueError(f"predicted-byte count mismatch in {report_path}")
            reconstructed_nll = total_nll / predicted_bytes
            reconstructed_bpb = total_nll / (
                predicted_bytes * math.log(2)
            )
            if (
                not math.isclose(
                    reconstructed_nll,
                    float(evaluation["nll_nats"]),
                    abs_tol=1e-7,
                )
                or not math.isclose(
                    reconstructed_bpb,
                    float(evaluation["bpb"]),
                    abs_tol=1e-7,
                )
            ):
                raise ValueError(f"absolute loss/report mismatch in {report_path}")
            checkpoint_hash = _checkpoint_state_sha256(checkpoint_path)
            if checkpoint_hash != report.get("trained_state_sha256"):
                raise ValueError(f"checkpoint state hash mismatch in {report_path}")
            reports[seed][policy] = report
            losses[seed][policy] = values
            evidence[str(seed)]["loss_artifact_sha256"][policy] = _sha256(
                loss_path
            )
            evidence[str(seed)]["training_report_artifact_sha256"][policy] = (
                _sha256(report_path)
            )
            evidence[str(seed)]["checkpoint_state_sha256"][policy] = (
                checkpoint_hash
            )
            evidence[str(seed)]["checkpoint_artifact_sha256"][policy] = (
                _sha256(checkpoint_path)
            )
        if threshold_matrices is not None:
            del threshold_matrices
    return reports, losses, evidence


def _quality_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        policy: numeric_summary(
            [
                reports[seed][policy]["evaluation"]["test"]["bpb"]
                for seed in seeds
            ]
        )
        for policy in policies
    }


def _calibration_quality_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        policy: numeric_summary(
            [
                reports[seed][policy]["evaluation"]["calibration"]["bpb"]
                for seed in seeds
            ]
        )
        for policy in policies
    }


def _validate_requested_design(
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    *,
    test_stream_sha256: str,
) -> None:
    if seeds not in (INITIAL_SEEDS, CONFIRMATION_SEEDS):
        raise ValueError("seeds must be the preregistered initial three or final five")
    if len(set(policies)) != len(policies):
        raise ValueError("policies must be unique")
    if not {F, C, W} <= set(policies):
        raise ValueError("Phase 3 summary requires all F/C/W primary policies")
    if not set(policies) <= set(PHASE3_POLICIES):
        raise ValueError("summary requested an unknown Phase 3 policy")
    if manifest.get("quick_smoke_only"):
        raise ValueError("quick smoke manifest cannot support Phase 3 evidence")
    if manifest.get("language") != "ko":
        raise ValueError("Phase 3 manifest language mismatch")
    if manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("manifest model spec mismatch")
    if manifest.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict():
        raise ValueError("manifest optimization spec mismatch")
    if manifest.get("limits") != FULL_LIMITS:
        raise ValueError("manifest byte limits mismatch")
    if manifest.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT:
        raise ValueError("manifest global position limit mismatch")
    if len(manifest.get("seeds", [])) != len(set(manifest.get("seeds", []))):
        raise ValueError("Phase 3 manifest contains duplicate seeds")
    if len(manifest.get("policies", [])) != len(
        set(manifest.get("policies", []))
    ):
        raise ValueError("Phase 3 manifest contains duplicate policies")
    if not set(seeds) <= set(manifest.get("seeds", [])):
        raise ValueError("requested seeds are absent from the run manifest")
    if not set(policies) <= set(manifest.get("policies", [])):
        raise ValueError("requested policies are absent from the run manifest")
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("Phase 3 manifest lacks invocation provenance")
    for seed in seeds:
        for policy in policies:
            if not any(
                isinstance(invocation, dict)
                and seed in invocation.get("seeds", [])
                and policy in invocation.get("policies", [])
                for invocation in invocations
            ):
                raise ValueError(
                    f"Phase 3 manifest lacks invocation for {seed}/{policy}"
                )
    expected_test_hash = manifest["streams"]["test"]["selected_stream_sha256"]
    if test_stream_sha256 != expected_test_hash:
        raise ValueError("test stream hash differs from the run manifest")


def _validate_report_counts(
    reports: dict[int, dict[str, dict[str, Any]]],
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> None:
    train_examples = int(manifest["streams"]["train"]["sequence_count"])
    calibration_examples = int(
        manifest["streams"]["calibration"]["sequence_count"]
    )
    test_examples = int(manifest["streams"]["test"]["sequence_count"])
    expected_steps = math.ceil(
        train_examples / PHASE3_OPTIMIZATION_SPEC.batch_size
    )
    expected_by_split = {
        "train": train_examples,
        "calibration": calibration_examples,
        "test": test_examples,
    }
    for seed in seeds:
        for policy in policies:
            report = reports[seed][policy]
            training = report["training"]
            if (
                training["examples"] != train_examples
                or training["steps"] != expected_steps
                or training["predicted_bytes"]
                != train_examples * TARGETS_PER_SEQUENCE
            ):
                raise ValueError(f"training-count mismatch for seed {seed}/{policy}")
            for split, examples in expected_by_split.items():
                if report["patch_diagnostics"][split]["examples"] != examples:
                    raise ValueError(
                        f"patch example-count mismatch for {split}/seed "
                        f"{seed}/{policy}"
                    )
            for split, examples in (
                ("calibration", calibration_examples),
                ("test", test_examples),
            ):
                evaluation = report["evaluation"][split]
                if (
                    evaluation["examples"] != examples
                    or evaluation["predicted_bytes"]
                    != examples * TARGETS_PER_SEQUENCE
                ):
                    raise ValueError(
                        f"evaluation-count mismatch for {split}/seed "
                        f"{seed}/{policy}"
                    )


def _available_contrasts(policies: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    available = set(policies)
    return {
        name: pair
        for name, pair in CONTRASTS.items()
        if set(pair) <= available
    }


def _contrast_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    document_window_map: DocumentWindowMap,
    *,
    repetitions: int,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("Phase 3 contrast inference requires at least two seeds")
    result: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(
        _available_contrasts(policies).items()
    ):
        reported = [
            reports[seed][left]["evaluation"]["test"]["bpb"]
            - reports[seed][right]["evaluation"]["test"]["bpb"]
            for seed in seeds
        ]
        sequence_differences = [
            losses[seed][left] - losses[seed][right]
            for seed in seeds
        ]
        for seed, expected, sequence_values in zip(
            seeds, reported, sequence_differences, strict=True
        ):
            reconstructed = float(sequence_values.mean()) / (
                TARGETS_PER_SEQUENCE * math.log(2)
            )
            if not math.isclose(expected, reconstructed, abs_tol=2e-5):
                raise ValueError(
                    f"loss/report mismatch for {name}/seed-{seed}: "
                    f"{expected} versus {reconstructed}"
                )
        estimates = hierarchical_paired_bootstrap_estimates(
            sequence_differences,
            targets_per_sequence=TARGETS_PER_SEQUENCE,
            repetitions=repetitions,
            seed=20_260_810 + index,
        )
        lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
        document_cluster = document_cluster_contrast_summary(
            sequence_differences,
            document_window_map,
            targets_per_sequence=TARGETS_PER_SEQUENCE,
            repetitions=repetitions,
            seed=20_260_910 + index,
        )
        result[name] = {
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(seeds),
            "paired_differences_bpb": reported,
            "negative_seed_count": int(sum(value < 0 for value in reported)),
            "paired_t_95_interval": paired_t_interval(reported).to_dict(),
            "hierarchical_bootstrap_95_interval": {
                "repetitions": repetitions,
                "seed": 20_260_810 + index,
                "resampling_design": "crossed seeds x shared test sequences",
                "mean": float(estimates.mean()),
                "median": float(median),
                "lower": float(lower),
                "upper": float(upper),
            },
            "document_cluster_bootstrap_95_interval": document_cluster,
            "bootstrap_nonnegative_tail": (
                empirical_nonnegative_bootstrap_tail(estimates)
            ),
            "paired_seed_one_sided_t_pvalue": (
                paired_seed_lower_t_pvalue(reported)
            ),
        }

    present_primary = [name for name in PRIMARY_FAMILY if name in result]
    if len(present_primary) == len(PRIMARY_FAMILY):
        raw_pvalues = {
            name: result[name]["paired_seed_one_sided_t_pvalue"]
            for name in present_primary
        }
        adjusted = holm_step_down_adjusted_values(raw_pvalues)
        ordered = sorted(
            present_primary,
            key=lambda name: (raw_pvalues[name], name),
        )
        family_size = len(ordered)
        for rank, name in enumerate(ordered):
            result[name]["holm_primary_family"] = {
                "rank": rank + 1,
                "family_size": family_size,
                "test": "one-sided paired-seed Student-t",
                "raw_one_sided_seed_t_pvalue": raw_pvalues[name],
                "holm_adjusted_seed_t_pvalue": adjusted[name],
                "rejects_at_familywise_alpha_0_05": adjusted[name] <= 0.05,
                "bootstrap_nonnegative_tail_diagnostic": result[name][
                    "bootstrap_nonnegative_tail"
                ],
            }
    return result


def _stratum_summary(
    stream_data: bytes,
    boundary_masks: np.ndarray,
    losses: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    strata, metadata = phase3_test_strata(
        stream_data,
        boundary_masks,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    scale = TARGETS_PER_SEQUENCE * math.log(2)
    contrasts: dict[str, Any] = {}
    for name, (left, right) in _available_contrasts(policies).items():
        contrast_strata: dict[str, Any] = {}
        for stratum_name, stratum in strata.items():
            selected = stratum.selected
            if not selected.any():
                contrast_strata[stratum_name] = {
                    **stratum.metadata(),
                    "status": "empty",
                }
                continue
            effects = [
                float(
                    (
                        losses[seed][left][selected]
                        - losses[seed][right][selected]
                    ).mean()
                )
                / scale
                for seed in seeds
            ]
            contrast_strata[stratum_name] = {
                **stratum.metadata(),
                "status": "estimated",
                "seed_order": list(seeds),
                "paired_seed_effects_bpb": effects,
                "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            }
        contrasts[name] = contrast_strata
    return {
        "definitions_and_counts": metadata,
        "contrasts": contrasts,
        "guardrail": (
            "Overlapping descriptive strata do not replace the full-test "
            "primary endpoint and are not used for discovery claims."
        ),
    }


def _integrity_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, np.ndarray]],
    evidence: dict[str, Any],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    initialization_ok = True
    order_ok = True
    for seed in seeds:
        initializations = {
            reports[seed][policy]["initialization_sha256"]
            for policy in policies
        }
        orders = {
            reports[seed][policy]["training_order_sha256"]
            for policy in policies
        }
        initialization_ok &= len(initializations) == 1
        order_ok &= len(orders) == 1
        by_seed[str(seed)] = {
            "identical_initialization_across_policies": len(initializations) == 1,
            "initialization_sha256": sorted(initializations),
            "identical_training_order_across_policies": len(orders) == 1,
            "training_order_sha256": sorted(orders),
            **evidence[str(seed)],
        }

    expected_examples = len(next(iter(next(iter(losses.values())).values())))
    loss_shapes_ok = all(
        len(losses[seed][policy]) == expected_examples
        for seed in seeds
        for policy in policies
    )
    exact_rate_policies = tuple(policy for policy in (F, C, W) if policy in policies)
    exact_rate_ok = all(
        reports[seed][policy]["patch_diagnostics"][split][
            "minimum_data_patches"
        ]
        == 86
        and reports[seed][policy]["patch_diagnostics"][split][
            "maximum_data_patches"
        ]
        == 86
        and reports[seed][policy]["patch_diagnostics"][split][
            "mean_data_patches"
        ]
        == 86.0
        and reports[seed][policy]["patch_diagnostics"][split]["padding_slots"]
        == 0
        for seed in seeds
        for policy in exact_rate_policies
        for split in SPLITS
    )
    structural_policies = tuple(
        policy for policy in (F, C, W, S) if policy in policies
    )
    structural_hashes = {
        policy: {
            split: sorted(
                {
                    reports[seed][policy]["patch_matrix_sha256"][split]
                    for seed in seeds
                }
            )
            for split in SPLITS
        }
        for policy in structural_policies
    }
    structural_seed_independent = all(
        len(values) == 1
        for policy in structural_hashes.values()
        for values in policy.values()
    )
    return {
        "all_seeds_paired_initialization": initialization_ok,
        "all_seeds_paired_training_order": order_ok,
        "all_loss_vectors_same_shape": loss_shapes_ok,
        "all_checkpoint_state_hashes_match_reports": True,
        "all_checkpoint_artifact_hashes_recorded": True,
        "all_training_report_artifact_hashes_recorded": True,
        "all_test_loss_artifact_hashes_recorded": True,
        "all_test_metrics_reconstructed_from_sequence_losses": True,
        "all_patch_matrices_and_diagnostics_match_current_evidence": True,
        "test_sequence_losses_per_policy": expected_examples,
        "fcw_exactly_86_patches_all_splits": exact_rate_ok,
        "structural_matrices_seed_independent": structural_seed_independent,
        "structural_patch_hashes": structural_hashes,
        "by_seed": by_seed,
        "all_integrity_checks_pass": bool(
            initialization_ok
            and order_ok
            and loss_shapes_ok
            and exact_rate_ok
            and structural_seed_independent
        ),
    }


def gate_i_summary(
    contrasts: dict[str, Any],
    *,
    ood_guard_pass: bool | None,
) -> dict[str, Any]:
    primary = contrasts["whitespace_minus_codepoint"]
    mean_effect = primary["paired_t_95_interval"]["mean"]
    negative_count = primary["negative_seed_count"]
    document_cluster = primary["document_cluster_bootstrap_95_interval"]
    document_upper = float(document_cluster["upper"])
    document_coverage_pass = bool(
        document_cluster["eligible_sequence_fraction_pass"]
    )
    quality_pass = bool(
        mean_effect <= -0.002
        and negative_count >= 2
        and document_coverage_pass
        and document_upper < 0
    )
    if not quality_pass:
        status = "fail_quality"
        overall: bool | None = False
    elif ood_guard_pass is None:
        status = "pending_ood_guard"
        overall = None
    elif ood_guard_pass:
        status = "pass"
        overall = True
    else:
        status = "fail_ood_guard"
        overall = False
    return {
        "status": status,
        "overall_pass": overall,
        "mean_whitespace_minus_codepoint_bpb": mean_effect,
        "maximum_mean_bpb": -0.002,
        "mean_threshold_pass": mean_effect <= -0.002,
        "negative_seed_count": negative_count,
        "required_negative_seed_count": 2,
        "negative_seed_count_pass": negative_count >= 2,
        "document_cluster_95_upper_bpb": document_upper,
        "document_cluster_95_upper_below_zero": document_upper < 0,
        "document_cluster_coverage_pass": document_coverage_pass,
        "quality_component_pass": quality_pass,
        "ood_guard_pass": ood_guard_pass,
    }


def gate_j_summary(
    contrasts: dict[str, Any],
    *,
    seed_count: int,
    ood_guard_pass: bool | None,
) -> dict[str, Any]:
    """Evaluate the preregistered five-seed method-evidence gate."""

    if seed_count != 5:
        return {
            "status": "not_evaluated_requires_exactly_five_seeds",
            "overall_pass": None,
            "seed_count": seed_count,
            "required_seed_count": 5,
        }
    required = {
        "whitespace_minus_codepoint",
        "whitespace_minus_fixed",
    }
    if not required <= set(contrasts):
        return {
            "status": "not_evaluated_missing_primary_policies",
            "overall_pass": None,
            "seed_count": seed_count,
            "required_contrasts": sorted(required),
        }

    by_contrast: dict[str, Any] = {}
    for name in sorted(required):
        contrast = contrasts[name]
        mean_effect = float(contrast["paired_t_95_interval"]["mean"])
        negative_count = int(contrast["negative_seed_count"])
        holm = contrast.get("holm_primary_family")
        if holm is None:
            raise ValueError(f"missing Holm result for Gate J contrast: {name}")
        bootstrap_upper = float(
            contrast["hierarchical_bootstrap_95_interval"]["upper"]
        )
        document_cluster = contrast[
            "document_cluster_bootstrap_95_interval"
        ]
        document_upper = float(document_cluster["upper"])
        document_coverage_pass = bool(
            document_cluster["eligible_sequence_fraction_pass"]
        )
        adjusted_pvalue = float(holm["holm_adjusted_seed_t_pvalue"])
        by_contrast[name] = {
            "mean_bpb": mean_effect,
            "maximum_mean_bpb": -0.003,
            "mean_threshold_pass": mean_effect <= -0.003,
            "negative_seed_count": negative_count,
            "required_negative_seed_count": 4,
            "negative_seed_count_pass": negative_count >= 4,
            "bootstrap_95_upper_bpb": bootstrap_upper,
            "bootstrap_95_upper_below_zero": bootstrap_upper < 0,
            "document_cluster_95_upper_bpb": document_upper,
            "document_cluster_95_upper_below_zero": document_upper < 0,
            "document_cluster_coverage_pass": document_coverage_pass,
            "holm_adjusted_seed_t_pvalue": adjusted_pvalue,
            "holm_adjusted_seed_t_pvalue_at_most_0_05": (
                adjusted_pvalue <= 0.05
            ),
        }

    primary_pass = all(
        values["mean_threshold_pass"]
        and values["negative_seed_count_pass"]
        and values["bootstrap_95_upper_below_zero"]
        and values["document_cluster_95_upper_below_zero"]
        and values["document_cluster_coverage_pass"]
        and values["holm_adjusted_seed_t_pvalue_at_most_0_05"]
        for values in by_contrast.values()
    )
    if not primary_pass:
        status = "fail_primary_quality"
        overall: bool | None = False
    elif ood_guard_pass is None:
        status = "pending_ood_guard"
        overall = None
    elif ood_guard_pass:
        status = "pass"
        overall = True
    else:
        status = "fail_ood_guard"
        overall = False
    return {
        "status": status,
        "overall_pass": overall,
        "seed_count": seed_count,
        "required_seed_count": 5,
        "by_contrast": by_contrast,
        "primary_quality_component_pass": primary_pass,
        "ood_guard_pass": ood_guard_pass,
    }


def _validate_ood_summary(
    ood: dict[str, Any],
    seeds: tuple[int, ...],
    checkpoint_hashes: dict[int, dict[str, str]],
    *,
    confirmation_authorization: dict[str, Any] | None = None,
) -> bool:
    """Bind the OOD guard to the exact primary checkpoints summarized here."""

    if tuple(ood.get("seeds", [])) != seeds:
        raise ValueError("OOD summary seeds do not match primary summary")
    if set(ood.get("policies", [])) != {F, C, W}:
        raise ValueError("OOD summary policies do not match F/C/W")
    integrity = ood.get("integrity", {})
    if integrity.get("all_integrity_checks_pass") is not True:
        raise ValueError("OOD summary integrity checks are incomplete")
    ood_checkpoint_hashes = integrity.get("checkpoint_state_sha256")
    expected = {
        str(seed): {
            policy: checkpoint_hashes[seed][policy]
            for policy in (F, C, W)
        }
        for seed in seeds
    }
    if ood_checkpoint_hashes != expected:
        raise ValueError("OOD summary checkpoints differ from primary evidence")
    if set(seeds) & set(CONFIRMATION_ONLY_SEEDS):
        if confirmation_authorization is None:
            raise ValueError("final primary evidence lacks confirmation authorization")
        if ood.get("confirmation_authorization") != confirmation_authorization:
            raise ValueError("OOD and primary confirmation authorization differ")
    gate = ood.get("gate_i_ood_guard", {})
    if not isinstance(gate.get("pass"), bool):
        raise ValueError("OOD summary gate result is missing")
    return gate["pass"]


def _write_observations(
    path: Path,
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "seed",
                "policy",
                "test_bpb",
                "calibration_bpb",
                "train_elapsed_seconds",
                "train_bytes_per_second",
                "test_mean_data_patches",
                "test_mean_bytes_per_patch",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for seed in seeds:
            for policy in policies:
                report = reports[seed][policy]
                writer.writerow(
                    {
                        "seed": seed,
                        "policy": policy,
                        "test_bpb": report["evaluation"]["test"]["bpb"],
                        "calibration_bpb": report["evaluation"]["calibration"]["bpb"],
                        "train_elapsed_seconds": report["training"]["elapsed_seconds"],
                        "train_bytes_per_second": report["training"]["bytes_per_second"],
                        "test_mean_data_patches": report["patch_diagnostics"]["test"]["mean_data_patches"],
                        "test_mean_bytes_per_patch": report["patch_diagnostics"]["test"]["mean_bytes_per_patch"],
                    }
                )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    policies = tuple(args.policies)
    if len(seeds) < 2:
        raise ValueError("summarization requires at least two seeds")
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    confirmation_authorization = None
    if set(seeds) & set(CONFIRMATION_ONLY_SEEDS):
        if args.confirmation_authorization_summary is None:
            raise ValueError(
                "five-seed summary requires --confirmation-authorization-summary"
            )
        confirmation_authorization = load_confirmation_authorization(
            Path(args.confirmation_authorization_summary)
        )
        validate_confirmation_invocations(manifest, confirmation_authorization)
    elif args.confirmation_authorization_summary is not None:
        raise ValueError(
            "confirmation authorization summary is only valid for five-seed evidence"
        )
    (
        streams,
        inputs,
        boundaries,
        whitespace,
        spacelike,
        source_context,
    ) = _reconstruct_data(manifest, Path(args.data_root))
    test_stream = streams["test"]
    test_boundaries = boundaries["test"]
    document_window_map = reconstruct_document_window_map(
        Path(args.data_root) / "ko.jsonl",
        split="test",
        byte_limit=FULL_LIMITS["test"],
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        expected_stream=test_stream.data,
    )
    _validate_requested_design(
        manifest,
        seeds,
        policies,
        test_stream_sha256=hashlib.sha256(test_stream.data).hexdigest(),
    )
    structural_matrices, structural_context = _reconstruct_structural_matrices(
        run_root,
        artifact_root,
        boundaries,
        whitespace,
        spacelike,
    )
    reports, losses, evidence = _load_runs(
        run_root,
        artifact_root,
        seeds,
        policies,
        inputs,
        boundaries,
        structural_matrices,
    )
    _validate_report_counts(reports, manifest, seeds, policies)
    expected_examples = manifest["streams"]["test"]["sequence_count"]
    if any(
        len(losses[seed][policy]) != expected_examples
        for seed in seeds
        for policy in policies
    ):
        raise ValueError("loss vectors differ from manifest test sequence count")
    contrasts = _contrast_summary(
        reports,
        losses,
        seeds,
        policies,
        document_window_map,
        repetitions=args.bootstrap_repetitions,
    )
    integrity = _integrity_summary(
        reports,
        losses,
        evidence,
        seeds,
        policies,
    )
    integrity.update(
        {
            "source_and_all_streams_match_independent_reconstruction": True,
            "structural_cache_matches_independent_reconstruction": True,
            "source_context": source_context,
            "structural_cache_context": structural_context,
            "document_window_map": document_window_map.metadata(),
            "document_cluster_coverage_pass": document_window_map.coverage_pass,
        }
    )
    if not integrity["all_integrity_checks_pass"]:
        raise ValueError("Phase 3 integrity checks failed")

    ood_guard_pass: bool | None = None
    ood = None
    if args.ood_summary is not None:
        ood = _read_json(args.ood_summary)
        checkpoint_hashes = {
            seed: evidence[str(seed)]["checkpoint_state_sha256"]
            for seed in seeds
        }
        ood_guard_pass = _validate_ood_summary(
            ood,
            seeds,
            checkpoint_hashes,
            confirmation_authorization=confirmation_authorization,
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "run_manifest": manifest,
        "confirmation_authorization": confirmation_authorization,
        "seeds": list(seeds),
        "policies": list(policies),
        "targets_per_sequence": TARGETS_PER_SEQUENCE,
        "quality": _quality_summary(reports, seeds, policies),
        "calibration_quality": _calibration_quality_summary(
            reports,
            seeds,
            policies,
        ),
        "contrasts": contrasts,
        "strata": _stratum_summary(
            test_stream.data,
            test_boundaries,
            losses,
            seeds,
            policies,
        ),
        "integrity": integrity,
        "gate_i": gate_i_summary(
            contrasts,
            ood_guard_pass=ood_guard_pass,
        ),
        "gate_j": gate_j_summary(
            contrasts,
            seed_count=len(seeds),
            ood_guard_pass=ood_guard_pass,
        ),
        "ood": ood,
        "interpretation_guardrail": (
            "This summary evaluates Phase 3 quality and Gate I only. It does "
            "not establish generation speed, CUDA latency, or publication-scale "
            "generality."
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_observations(
        output_root / "observations.csv", reports, seeds, policies
    )
    print(json.dumps(summary["gate_i"], indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase3")
    parser.add_argument("--artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--data-root", default="data/processed/hplt3-korean-phase3"
    )
    parser.add_argument(
        "--output-root",
        default="results/phase3-primary-clustered",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--ood-summary", type=Path)
    parser.add_argument("--confirmation-authorization-summary", type=Path)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
