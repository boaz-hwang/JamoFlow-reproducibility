#!/usr/bin/env python3
"""Reconstruct and summarize the preregistered Phase 3 compute conversion."""

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
from jamoflow.compute_conversion import (
    CONVERSION_POLICIES,
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
    conversion_policy,
    initial_conversion_gate,
    select_rate_from_calibration,
)
from jamoflow.inference_selection_plan import (
    PHASE3_PRIMARY_SUMMARY_PATH,
    validate_selection_plan_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import shuffled_indices
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import PHASE3_MODEL_SPEC, PHASE3_OPTIMIZATION_SPEC
from jamoflow.phase3_analysis import (
    empirical_nonnegative_bootstrap_tail,
    hierarchical_paired_bootstrap_estimates,
    holm_step_down_adjusted_values,
    paired_seed_lower_t_pvalue,
)


INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_ONLY_SEEDS = (57721, 65537)
ALL_SEEDS = (*INITIAL_SEEDS, *CONFIRMATION_ONLY_SEEDS)
CONFIRMATION_BOOTSTRAP_REPETITIONS = 10_000
SPLITS = ("train", "calibration", "test")
PRIMARY_CODEPOINT = "causal_codepoint_grid"
PRIMARY_POLICIES = (
    "fixed_byte_6",
    PRIMARY_CODEPOINT,
    "causal_whitespace_grid",
)
TARGETS_PER_SEQUENCE = PHASE3_MODEL_SPEC.sequence_length - 1
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
SELECTION_PLAN_PATH = Path(
    "data/manifests/phase3-inference-selection-plan-v2.json"
)
REPORT_KEYS = {
    "schema_version",
    "seed",
    "policy",
    "rate",
    "parameters",
    "initialization_sha256",
    "trained_state_sha256",
    "training_order_sha256",
    "checkpoint_artifact_sha256",
    "calibration_loss_artifact_sha256",
    "loss_artifact_sha256",
    "evidence_binding",
    "patch_matrix_sha256",
    "patch_diagnostics",
    "training",
    "evaluation",
    "model_spec",
    "optimization_spec",
    "global_max_position_embeddings",
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


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    if not state:
        raise ValueError("checkpoint state dict is empty")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint has an unexpected entry")
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dict: {path}")
    return _state_dict_sha256(state)


def _initialization_sha256(seed: int, rate: int) -> tuple[str, int]:
    model = build_main_model(
        conversion_model_spec(rate),
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    count = parameter_count(model)
    state_hash = _state_dict_sha256(model.state_dict())
    del model
    return state_hash, count


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_primary_summary(
    primary: dict[str, Any],
    primary_path: Path,
    manifest: dict[str, Any],
) -> None:
    authorization = primary.get("confirmation_authorization")
    ood = primary.get("ood")
    if (
        tuple(primary.get("seeds", [])) != ALL_SEEDS
        or tuple(primary.get("policies", [])) != PRIMARY_POLICIES
        or primary.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or primary.get("gate_i", {}).get("overall_pass") is not True
        or primary.get("gate_j", {}).get("overall_pass") is not True
        or primary.get("targets_per_sequence") != TARGETS_PER_SEQUENCE
        or not isinstance(authorization, dict)
        or authorization.get("authorization_kind")
        != "phase3_corrected_gate_i_confirmation_v1"
        or not isinstance(ood, dict)
        or ood.get("gate_i_ood_guard", {}).get("pass") is not True
        or ood.get("integrity", {}).get("all_integrity_checks_pass") is not True
    ):
        raise ValueError(
            "conversion summary requires passing five-seed Gate J and OOD evidence"
        )
    if (
        manifest.get("primary_gate_summary_sha256") != _sha256(primary_path)
        or manifest.get("primary_gate_i") != primary["gate_i"]
    ):
        raise ValueError("conversion manifest is not bound to the primary summary")


def _reconstruct_context(
    manifest: dict[str, Any],
    primary: dict[str, Any],
    data_root: Path,
    artifact_root: Path,
    run_root: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    DocumentWindowMap,
]:
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
    source_context = manifest.get("source_context", {})
    source_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    integrity_artifact = {
        "filename": "integrity.json",
        "bytes": integrity_path.stat().st_size,
        "sha256": _sha256(integrity_path),
    }
    if (
        source_context.get("source_artifact") != source_artifact
        or source_context.get("source_integrity_artifact") != integrity_artifact
    ):
        raise ValueError("conversion source artifacts do not reconstruct")

    primary_manifest = manifest.get("primary_gate_i")
    if not isinstance(primary_manifest, dict):
        raise ValueError("conversion manifest lacks primary gate lineage")
    limits = primary["run_manifest"]["limits"]
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    reconstructed_streams: dict[str, Any] = {}
    test_stream_data: bytes | None = None
    for split in SPLITS:
        stream = build_neural_stream(
            source_path,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=int(limits[split]),
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        split_whitespace = compact_whitespace_mask(stream.data).reshape(
            split_inputs.shape
        )
        reconstructed_streams[split] = {
            "selected_stream_sha256": hashlib.sha256(stream.data).hexdigest(),
            "inputs_sha256": _array_sha256(split_inputs),
            "boundaries_sha256": _array_sha256(split_boundaries),
            "whitespace_sha256": _array_sha256(split_whitespace),
            "sequence_count": len(split_inputs),
        }
        inputs[split] = split_inputs
        boundaries[split] = split_boundaries
        whitespace[split] = split_whitespace
        if split == "test":
            test_stream_data = stream.data
    reconstructed_context = {
        "source_artifact": source_artifact,
        "source_integrity_artifact": integrity_artifact,
        "streams": reconstructed_streams,
    }
    if source_context != reconstructed_context:
        raise ValueError("conversion streams do not independently reconstruct")

    matrices: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in SPLITS
    }
    for split in SPLITS:
        for rate in CONVERSION_RATES:
            matrices[split].update(
                conversion_patch_matrices(
                    boundaries[split],
                    whitespace[split],
                    rate=rate,
                )
            )
    expected_arrays = {
        f"{split}__{policy}": matrices[split][policy]
        for split in SPLITS
        for policy in CONVERSION_POLICIES
    }
    cache_path = artifact_root / "patches.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_arrays) or any(
            not np.array_equal(archive[key], value)
            for key, value in expected_arrays.items()
        ):
            raise ValueError("conversion matrix cache differs from reconstruction")
    diagnostics_path = run_root / "patch-diagnostics.json"
    expected_diagnostics = {
        "cache_artifact_sha256": _sha256(cache_path),
        "splits": {
            split: {
                policy: {
                    **variable_patch_diagnostics(
                        matrices[split][policy], boundaries[split]
                    ).to_dict(),
                    "matrix_sha256": _array_sha256(matrices[split][policy]),
                }
                for policy in CONVERSION_POLICIES
            }
            for split in SPLITS
        },
    }
    if _read_json(diagnostics_path) != expected_diagnostics:
        raise ValueError("conversion matrix diagnostics do not reconstruct")
    if test_stream_data is None:
        raise AssertionError("conversion reconstruction lacks test stream")
    document_window_map = reconstruct_document_window_map(
        source_path,
        split="test",
        byte_limit=int(limits["test"]),
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        expected_stream=test_stream_data,
    )
    return (
        inputs,
        boundaries,
        matrices,
        reconstructed_context,
        document_window_map,
    )


def _selected_confirmation_context(
    selection_path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], int, tuple[str, str]]:
    selection = _read_json(selection_path)
    validate_selection_lock_v2(selection)
    decision = selection["decision"]
    rate = decision.get("rate_selection", {}).get("selected_rate")
    confirmation = decision.get("confirmation_plan", {})
    compute_confirmation = (
        confirmation.get("compute_conversion", {})
        if isinstance(confirmation, dict)
        else {}
    )
    if (
        decision.get("status")
        != "locked_pending_confirmation_and_new_final_test"
        or rate not in CONVERSION_RATES
        or compute_confirmation.get("authorization_kind")
        != "compute_conversion_confirmation_v2"
        or compute_confirmation.get("selected_rate") != rate
        or tuple(compute_confirmation.get("seeds", ()))
        != CONFIRMATION_ONLY_SEEDS
    ):
        raise ValueError("confirmation requires the canonical selection-v2 lock")
    policies = (
        conversion_policy("codepoint", rate),
        conversion_policy("whitespace", rate),
    )
    if tuple(compute_confirmation.get("policies", ())) != policies:
        raise ValueError("selection-v2 confirmation plan policies differ")
    selection_hash = _sha256(selection_path)
    matching = [
        invocation
        for invocation in manifest.get("invocations", [])
        if isinstance(invocation, dict)
        and invocation.get("stage") == "confirmation"
        and tuple(invocation.get("seeds", [])) == CONFIRMATION_ONLY_SEEDS
        and tuple(invocation.get("policies", [])) == policies
        and invocation.get("selection_summary_sha256") == selection_hash
    ]
    if not matching:
        raise ValueError("manifest lacks the selected confirmation invocation")
    return selection, int(rate), policies


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    stage: str,
    primary: dict[str, Any],
    primary_path: Path,
    selection_path: Path | None,
) -> tuple[tuple[int, ...], tuple[str, ...], dict[str, Any] | None, int | None]:
    selection_plan = _read_json(SELECTION_PLAN_PATH)
    validate_selection_plan_v2(selection_plan)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("selection_plan") != str(SELECTION_PLAN_PATH)
        or manifest.get("selection_plan_sha256")
        != _sha256(SELECTION_PLAN_PATH)
        or tuple(manifest.get("rates", [])) != CONVERSION_RATES
        or tuple(manifest.get("policies", [])) != CONVERSION_POLICIES
        or manifest.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or manifest.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT
        or manifest.get("model_specs")
        != {
            str(rate): conversion_model_spec(rate).to_dict()
            for rate in CONVERSION_RATES
        }
    ):
        raise ValueError("compute-conversion manifest design mismatch")
    if (
        selection_plan["historical_screening"]["primary_summary"]["sha256"]
        != _sha256(primary_path)
    ):
        raise ValueError("compute-conversion primary summary differs from the plan")
    _validate_primary_summary(primary, primary_path, manifest)
    initial_invocation = any(
        isinstance(invocation, dict)
        and invocation.get("stage") == "initial"
        and tuple(invocation.get("seeds", [])) == INITIAL_SEEDS
        and tuple(invocation.get("policies", [])) == CONVERSION_POLICIES
        for invocation in manifest.get("invocations", [])
    )
    if not initial_invocation:
        raise ValueError("manifest lacks the complete initial invocation")
    if stage == "initial":
        if selection_path is not None:
            raise ValueError("initial summary does not accept a selection summary")
        return INITIAL_SEEDS, CONVERSION_POLICIES, None, None
    if stage != "confirmation" or selection_path is None:
        raise ValueError("confirmation summary requires --selection-summary")
    selection, rate, policies = _selected_confirmation_context(
        selection_path,
        manifest,
    )
    return ALL_SEEDS, policies, selection, rate


def _load_primary_codepoint(
    primary: dict[str, Any],
    seeds: tuple[int, ...],
    primary_run_root: Path,
    primary_artifact_root: Path,
) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    calibration: dict[int, float] = {}
    test: dict[int, float] = {}
    evidence: dict[str, Any] = {}
    for seed in seeds:
        report_path = (
            primary_run_root / f"seed-{seed}" / f"{PRIMARY_CODEPOINT}.json"
        )
        loss_path = (
            primary_artifact_root
            / f"seed-{seed}"
            / f"{PRIMARY_CODEPOINT}-test-nll.npz"
        )
        checkpoint_path = (
            primary_artifact_root
            / f"seed-{seed}"
            / f"{PRIMARY_CODEPOINT}.pt"
        )
        recorded = primary["integrity"]["by_seed"][str(seed)]
        checkpoint_state_hash = _checkpoint_state_sha256(checkpoint_path)
        if (
            _sha256(report_path)
            != recorded["training_report_artifact_sha256"][PRIMARY_CODEPOINT]
            or _sha256(loss_path)
            != recorded["loss_artifact_sha256"][PRIMARY_CODEPOINT]
            or _sha256(checkpoint_path)
            != recorded["checkpoint_artifact_sha256"][PRIMARY_CODEPOINT]
            or checkpoint_state_hash
            != recorded["checkpoint_state_sha256"][PRIMARY_CODEPOINT]
        ):
            raise ValueError(f"primary C86 evidence mismatch for seed {seed}")
        report = _read_json(report_path)
        with np.load(loss_path, allow_pickle=False) as archive:
            if archive.files != ["sequence_nll_nats"]:
                raise ValueError("primary C86 loss keys mismatch")
            stored = archive["sequence_nll_nats"]
            if stored.dtype != np.float32:
                raise ValueError("primary C86 loss dtype mismatch")
            losses = stored.astype(np.float64)
        if (
            losses.ndim != 1
            or not len(losses)
            or not np.isfinite(losses).all()
            or np.any(losses < 0)
        ):
            raise ValueError("primary C86 loss vector is invalid")
        expected_bpb = float(losses.sum()) / (
            len(losses) * TARGETS_PER_SEQUENCE * math.log(2)
        )
        if (
            report.get("seed") != seed
            or report.get("policy") != PRIMARY_CODEPOINT
            or not math.isclose(
                expected_bpb,
                float(report["evaluation"]["test"]["bpb"]),
                abs_tol=1e-7,
            )
        ):
            raise ValueError(f"primary C86 metrics mismatch for seed {seed}")
        calibration[seed] = float(report["evaluation"]["calibration"]["bpb"])
        test[seed] = float(report["evaluation"]["test"]["bpb"])
        evidence[str(seed)] = {
            "training_report_artifact_sha256": _sha256(report_path),
            "loss_artifact_sha256": _sha256(loss_path),
            "checkpoint_artifact_sha256": _sha256(checkpoint_path),
            "checkpoint_state_sha256": checkpoint_state_hash,
        }
    return calibration, test, evidence


def _load_conversion_runs(
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    run_root: Path,
    artifact_root: Path,
    manifest: dict[str, Any],
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, np.ndarray]],
    dict[str, Any],
]:
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    evidence: dict[str, Any] = {}
    expected_training_steps = math.ceil(
        len(inputs["train"]) / PHASE3_OPTIMIZATION_SPEC.batch_size
    )
    for seed in seeds:
        reports[seed] = {}
        losses[seed] = {}
        evidence[str(seed)] = {}
        initialization_by_rate = {
            rate: _initialization_sha256(seed, rate)
            for rate in sorted({_rate_from_policy(policy) for policy in policies})
        }
        order_hash = _array_sha256(shuffled_indices(len(inputs["train"]), seed))
        for policy in policies:
            expected_report_stage = (
                "initial" if seed in INITIAL_SEEDS else "confirmation"
            )
            expected_binding_seeds = (
                INITIAL_SEEDS
                if expected_report_stage == "initial"
                else CONFIRMATION_ONLY_SEEDS
            )
            expected_binding_policies = (
                CONVERSION_POLICIES
                if expected_report_stage == "initial"
                else policies
            )
            rate = _rate_from_policy(policy)
            report_path = run_root / f"seed-{seed}" / f"{policy}.json"
            checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
            calibration_loss_path = (
                artifact_root
                / f"seed-{seed}"
                / f"{policy}-calibration-nll.npz"
            )
            loss_path = (
                artifact_root / f"seed-{seed}" / f"{policy}-test-nll.npz"
            )
            if not all(
                path.exists()
                for path in (
                    report_path,
                    checkpoint_path,
                    calibration_loss_path,
                    loss_path,
                )
            ):
                raise FileNotFoundError(
                    f"conversion evidence incomplete for seed {seed}/{policy}"
                )
            report = _read_json(report_path)
            init_hash, expected_parameters = initialization_by_rate[rate]
            checkpoint_state_hash = _checkpoint_state_sha256(checkpoint_path)
            if (
                set(report) != REPORT_KEYS
                or report.get("schema_version") != 2
                or report.get("seed") != seed
                or report.get("policy") != policy
                or report.get("rate") != rate
                or report.get("parameters") != expected_parameters
                or report.get("model_spec") != conversion_model_spec(rate).to_dict()
                or report.get("optimization_spec")
                != PHASE3_OPTIMIZATION_SPEC.to_dict()
                or report.get("global_max_position_embeddings")
                != GLOBAL_POSITION_LIMIT
                or report.get("initialization_sha256") != init_hash
                or report.get("training_order_sha256") != order_hash
            ):
                raise ValueError(f"conversion report mismatch: {seed}/{policy}")
            binding = report.get("evidence_binding")
            binding_payload = (
                {
                    key: value
                    for key, value in binding.items()
                    if key != "identity_sha256"
                }
                if isinstance(binding, dict)
                else {}
            )
            if (
                not isinstance(binding, dict)
                or set(binding)
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
                or binding.get("identity_sha256")
                != _canonical_sha256(binding_payload)
                or binding.get("stage") != expected_report_stage
                or binding.get("schema_version") != 1
                or binding.get("git_worktree_clean_at_start") is not True
                or tuple(binding.get("seeds", ())) != expected_binding_seeds
                or tuple(binding.get("policies", ()))
                != expected_binding_policies
                or not isinstance(binding.get("device"), str)
                or not binding.get("device")
                or not isinstance(binding.get("git_commit"), str)
                or len(binding.get("git_commit")) != 40
                or binding.get("primary_summary_sha256")
                != manifest.get("primary_gate_summary_sha256")
                or binding.get("selection_plan_sha256")
                != manifest.get("selection_plan_sha256")
                or (
                    expected_report_stage == "initial"
                    and binding.get("selection_summary_sha256") is not None
                )
                or not any(
                    isinstance(invocation, dict)
                    and seed in invocation.get("seeds", [])
                    and policy in invocation.get("policies", [])
                    and invocation.get("stage") == expected_report_stage
                    and invocation.get("evidence_binding") == binding
                    and invocation.get("git_commit") == binding.get("git_commit")
                    and invocation.get("selection_summary_sha256")
                    == binding.get("selection_summary_sha256")
                    and invocation.get("git_worktree_clean_at_start") is True
                    for invocation in manifest.get("invocations", [])
                )
            ):
                raise ValueError(
                    f"conversion report invocation binding mismatch: {seed}/{policy}"
                )
            training = report.get("training", {})
            if (
                training.get("examples") != len(inputs["train"])
                or training.get("steps") != expected_training_steps
                or training.get("predicted_bytes")
                != len(inputs["train"]) * TARGETS_PER_SEQUENCE
            ):
                raise ValueError(f"conversion training count mismatch: {seed}/{policy}")
            for split in SPLITS:
                matrix = matrices[split][policy]
                validate_padded_patch_matrix(
                    matrix,
                    PHASE3_MODEL_SPEC.sequence_length,
                )
                diagnostics = variable_patch_diagnostics(
                    matrix,
                    boundaries[split],
                ).to_dict()
                if (
                    report.get("patch_matrix_sha256", {}).get(split)
                    != _array_sha256(matrix)
                    or report.get("patch_diagnostics", {}).get(split)
                    != diagnostics
                    or diagnostics["minimum_data_patches"] != rate
                    or diagnostics["maximum_data_patches"] != rate
                    or diagnostics["padding_slots"] != 0
                ):
                    raise ValueError(
                        f"conversion matrix mismatch: {seed}/{policy}/{split}"
                    )
            for split in ("calibration", "test"):
                evaluation = report.get("evaluation", {}).get(split, {})
                if (
                    evaluation.get("examples") != len(inputs[split])
                    or evaluation.get("predicted_bytes")
                    != len(inputs[split]) * TARGETS_PER_SEQUENCE
                    or not math.isfinite(float(evaluation.get("bpb", math.nan)))
                ):
                    raise ValueError(
                        f"conversion evaluation count mismatch: {seed}/{policy}/{split}"
                    )
            if (
                report.get("checkpoint_artifact_sha256")
                != _sha256(checkpoint_path)
                or report.get("calibration_loss_artifact_sha256")
                != _sha256(calibration_loss_path)
                or report.get("loss_artifact_sha256") != _sha256(loss_path)
                or report.get("trained_state_sha256")
                != checkpoint_state_hash
            ):
                raise ValueError(f"conversion artifact hash mismatch: {seed}/{policy}")
            with np.load(loss_path, allow_pickle=False) as archive:
                if archive.files != ["sequence_nll_nats"]:
                    raise ValueError("conversion loss keys mismatch")
                stored = archive["sequence_nll_nats"]
                if stored.dtype != np.float32:
                    raise ValueError("conversion loss dtype mismatch")
                values = stored.astype(np.float64)
            with np.load(calibration_loss_path, allow_pickle=False) as archive:
                if archive.files != ["sequence_nll_nats"]:
                    raise ValueError("conversion calibration-loss keys mismatch")
                stored = archive["sequence_nll_nats"]
                if stored.dtype != np.float32:
                    raise ValueError("conversion calibration-loss dtype mismatch")
                calibration_values = stored.astype(np.float64)
            if (
                values.shape != (len(inputs["test"]),)
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError("conversion loss vector is invalid")
            if (
                calibration_values.shape != (len(inputs["calibration"]),)
                or not np.isfinite(calibration_values).all()
                or np.any(calibration_values < 0)
            ):
                raise ValueError("conversion calibration-loss vector is invalid")
            expected_bpb = float(values.sum()) / (
                len(values) * TARGETS_PER_SEQUENCE * math.log(2)
            )
            if not math.isclose(
                expected_bpb,
                float(report["evaluation"]["test"]["bpb"]),
                abs_tol=1e-7,
            ):
                raise ValueError(f"conversion BPB mismatch: {seed}/{policy}")
            calibration_bpb = float(calibration_values.sum()) / (
                len(calibration_values) * TARGETS_PER_SEQUENCE * math.log(2)
            )
            if not math.isclose(
                calibration_bpb,
                float(report["evaluation"]["calibration"]["bpb"]),
                abs_tol=1e-7,
            ):
                raise ValueError(
                    f"conversion calibration BPB mismatch: {seed}/{policy}"
                )
            reports[seed][policy] = report
            losses[seed][policy] = values
            evidence[str(seed)][policy] = {
                "training_report_artifact_sha256": _sha256(report_path),
                "checkpoint_artifact_sha256": _sha256(checkpoint_path),
                "checkpoint_state_sha256": checkpoint_state_hash,
                "calibration_loss_artifact_sha256": _sha256(
                    calibration_loss_path
                ),
                "loss_artifact_sha256": _sha256(loss_path),
                "evidence_binding": binding,
            }
    return reports, losses, evidence


def _rate_from_policy(policy: str) -> int:
    try:
        rate = int(policy.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"malformed conversion policy: {policy}") from error
    if rate not in CONVERSION_RATES or policy not in CONVERSION_POLICIES:
        raise ValueError(f"unknown conversion policy: {policy}")
    return rate


def _quality_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        policy: {
            "calibration_bpb": numeric_summary(
                [
                    float(reports[seed][policy]["evaluation"]["calibration"]["bpb"])
                    for seed in seeds
                ]
            ),
            "test_bpb": numeric_summary(
                [
                    float(reports[seed][policy]["evaluation"]["test"]["bpb"])
                    for seed in seeds
                ]
            ),
        }
        for policy in policies
    }


def confirmation_same_rate_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    rate: int,
    *,
    document_window_map: DocumentWindowMap,
    repetitions: int = 10_000,
) -> dict[str, Any]:
    if seeds != ALL_SEEDS:
        raise ValueError("confirmation inference requires all five fixed seeds")
    left = conversion_policy("whitespace", rate)
    right = conversion_policy("codepoint", rate)
    effects = [
        float(reports[seed][left]["evaluation"]["test"]["bpb"])
        - float(reports[seed][right]["evaluation"]["test"]["bpb"])
        for seed in seeds
    ]
    sequence_differences = [
        losses[seed][left] - losses[seed][right]
        for seed in seeds
    ]
    for expected, values in zip(effects, sequence_differences, strict=True):
        reconstructed = float(values.mean()) / (
            TARGETS_PER_SEQUENCE * math.log(2)
        )
        if not math.isclose(expected, reconstructed, abs_tol=2e-5):
            raise ValueError("confirmation paired losses do not reconstruct")
    estimates = hierarchical_paired_bootstrap_estimates(
        sequence_differences,
        targets_per_sequence=TARGETS_PER_SEQUENCE,
        repetitions=repetitions,
        seed=20_260_811,
    )
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    document_cluster = document_cluster_contrast_summary(
        sequence_differences,
        document_window_map,
        targets_per_sequence=TARGETS_PER_SEQUENCE,
        repetitions=repetitions,
        seed=20_260_911,
    )
    raw_pvalue = paired_seed_lower_t_pvalue(effects)
    adjusted_pvalue = holm_step_down_adjusted_values(
        {"selected_whitespace_minus_codepoint": raw_pvalue}
    )["selected_whitespace_minus_codepoint"]
    mean_effect = float(np.mean(effects))
    negative_count = int(sum(value < 0 for value in effects))
    passed = bool(
        mean_effect <= -0.003
        and negative_count >= 4
        and float(upper) < 0
        and bool(document_cluster["eligible_sequence_fraction_pass"])
        and float(document_cluster["upper"]) < 0
        and adjusted_pvalue <= 0.05
    )
    return {
        "status": "pass" if passed else "fail_same_rate_confirmation",
        "overall_pass": passed,
        "left_policy": left,
        "right_policy": right,
        "difference_direction": "left_minus_right; negative favors whitespace",
        "seed_order": list(seeds),
        "paired_differences_bpb": effects,
        "mean_bpb": mean_effect,
        "maximum_mean_bpb": -0.003,
        "mean_threshold_pass": mean_effect <= -0.003,
        "negative_seed_count": negative_count,
        "required_negative_seed_count": 4,
        "negative_seed_count_pass": negative_count >= 4,
        "paired_t_95_interval": paired_t_interval(effects).to_dict(),
        "hierarchical_bootstrap_95_interval": {
            "repetitions": repetitions,
            "seed": 20_260_811,
            "resampling_design": "crossed seeds x shared test sequences",
            "mean": float(estimates.mean()),
            "median": float(median),
            "lower": float(lower),
            "upper": float(upper),
        },
        "bootstrap_95_upper_below_zero": float(upper) < 0,
        "document_cluster_bootstrap_95_interval": document_cluster,
        "document_cluster_95_upper_below_zero": (
            float(document_cluster["upper"]) < 0
        ),
        "document_cluster_coverage_pass": bool(
            document_cluster["eligible_sequence_fraction_pass"]
        ),
        "bootstrap_nonnegative_tail": empirical_nonnegative_bootstrap_tail(
            estimates
        ),
        "paired_seed_one_sided_t_pvalue": raw_pvalue,
        "holm_family_size": 1,
        "holm_adjusted_seed_t_pvalue": adjusted_pvalue,
        "holm_adjusted_seed_t_pvalue_at_most_0_05": adjusted_pvalue <= 0.05,
    }


def _confirmation_historical_screening_gate(
    selection: dict[str, Any],
    reports: dict[int, dict[str, dict[str, Any]]],
    primary_test: dict[int, float],
    selected_rate: int,
    policies: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, object]]:
    """Reconstruct the exposed-test screen without making it a selection input."""

    decision = selection.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("confirmation selection lacks a canonical decision")
    rate_selection = decision.get("rate_selection")
    if (
        not isinstance(rate_selection, dict)
        or rate_selection.get("selected_rate") != selected_rate
        or policies
        != (
            conversion_policy("codepoint", selected_rate),
            conversion_policy("whitespace", selected_rate),
        )
    ):
        raise ValueError("confirmation selection rate/policy identity differs")
    initial_test_bpb = {
        seed: {
            policy: float(
                reports[seed][policy]["evaluation"]["test"]["bpb"]
            )
            for policy in policies
        }
        for seed in INITIAL_SEEDS
    }
    gate = initial_conversion_gate(
        initial_test_bpb,
        primary_test,
        selected_rate=selected_rate,
    )
    return rate_selection, gate


def _integrity_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    evidence: dict[str, Any],
    primary_evidence: dict[str, Any],
    source_context: dict[str, Any],
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
            "identical_training_order_across_policies": len(orders) == 1,
            "conversion_artifacts": evidence[str(seed)],
        }
    return {
        "all_integrity_checks_pass": bool(initialization_ok and order_ok),
        "all_seeds_paired_initialization": initialization_ok,
        "all_seeds_paired_training_order": order_ok,
        "all_source_streams_and_matrices_independently_reconstructed": True,
        "all_report_counts_and_test_metrics_reconstructed": True,
        "all_checkpoint_states_and_artifact_hashes_match": True,
        "all_conversion_rates_exact_on_all_splits": True,
        "source_context": source_context,
        "primary_codepoint_evidence": primary_evidence,
        "by_seed": by_seed,
    }


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
            fieldnames=(
                "seed",
                "policy",
                "rate",
                "calibration_bpb",
                "test_bpb",
                "train_elapsed_seconds",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            for policy in policies:
                report = reports[seed][policy]
                writer.writerow(
                    {
                        "seed": seed,
                        "policy": policy,
                        "rate": report["rate"],
                        "calibration_bpb": report["evaluation"]["calibration"]["bpb"],
                        "test_bpb": report["evaluation"]["test"]["bpb"],
                        "train_elapsed_seconds": report["training"]["elapsed_seconds"],
                    }
                )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    primary_path = Path(args.primary_summary)
    selection_path = (
        Path(args.selection_summary) if args.selection_summary is not None else None
    )
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    primary = _read_json(primary_path)
    seeds, policies, selection, selected_rate = _validate_manifest(
        manifest,
        stage=args.stage,
        primary=primary,
        primary_path=primary_path,
        selection_path=selection_path,
    )
    (
        inputs,
        boundaries,
        matrices,
        source_context,
        document_window_map,
    ) = _reconstruct_context(
        manifest,
        primary,
        Path(args.data_root),
        artifact_root,
        run_root,
    )
    reports, losses, evidence = _load_conversion_runs(
        seeds,
        policies,
        inputs,
        boundaries,
        matrices,
        run_root,
        artifact_root,
        manifest,
    )
    primary_calibration, primary_test, primary_evidence = _load_primary_codepoint(
        primary,
        INITIAL_SEEDS,
        Path(args.primary_run_root),
        Path(args.primary_artifact_root),
    )
    integrity = _integrity_summary(
        reports,
        seeds,
        policies,
        evidence,
        primary_evidence,
        source_context,
    )
    rate_selection = None
    initial_gate = None
    confirmation_gate = None
    if args.stage == "initial":
        calibration_bpb = {
            seed: {
                policy: float(
                    reports[seed][policy]["evaluation"]["calibration"]["bpb"]
                )
                for policy in policies
            }
            for seed in seeds
        }
        rate_selection = select_rate_from_calibration(
            calibration_bpb,
            primary_calibration,
        ).to_dict()
        rate = rate_selection["selected_rate"]
        if rate is None:
            initial_gate = {
                "status": "fail_no_calibration_rate",
                "overall_pass": False,
                "selected_rate": None,
            }
        else:
            test_bpb = {
                seed: {
                    policy: float(
                        reports[seed][policy]["evaluation"]["test"]["bpb"]
                    )
                    for policy in policies
                }
                for seed in seeds
            }
            initial_gate = initial_conversion_gate(
                test_bpb,
                primary_test,
                selected_rate=int(rate),
            )
    else:
        if selection is None or selected_rate is None:
            raise AssertionError("validated confirmation lacks a selection")
        rate_selection, initial_gate = _confirmation_historical_screening_gate(
            selection,
            reports,
            primary_test,
            selected_rate,
            policies,
        )
        confirmation_gate = confirmation_same_rate_summary(
            reports,
            losses,
            seeds,
            selected_rate,
            document_window_map=document_window_map,
            repetitions=CONFIRMATION_BOOTSTRAP_REPETITIONS,
        )

    output_root = Path(args.output_root)
    summary_path = output_root / f"{args.stage}-summary.json"
    summary = {
        "schema_version": 2,
        "stage": args.stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "primary_summary": {
            "path": str(primary_path),
            "sha256": _sha256(primary_path),
        },
        "selection_summary": (
            {
                "path": str(selection_path),
                "sha256": _sha256(selection_path),
            }
            if selection_path is not None
            else None
        ),
        "seeds": list(seeds),
        "policies": list(policies),
        "quality": _quality_summary(reports, seeds, policies),
        "calibration_rate_selection": rate_selection,
        "initial_conversion_gate": initial_gate,
        "confirmation_same_rate_gate": confirmation_gate,
        "evidence_role": "historical_development_screening_only",
        "authorization_scope": {
            "authorizes_actual_timing": False,
            "authorizes_final_claim": False,
            "authorizes_final_test_evaluation": False,
            "authorizes_selection": False,
        },
        "integrity": integrity,
        "document_cluster_inference": document_window_map.metadata(),
        "interpretation_guardrail": (
            "This summary evaluates quality-preserving global-rate conversion. "
            "It is not evidence of actual autoregressive latency improvement."
        ),
    }
    _write_json(summary_path, summary)
    _write_observations(
        output_root / f"{args.stage}-observations.csv",
        reports,
        seeds,
        policies,
    )
    print(
        json.dumps(
            confirmation_gate if confirmation_gate is not None else initial_gate,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("initial", "confirmation"), required=True)
    parser.add_argument("--selection-summary")
    parser.add_argument(
        "--primary-summary",
        default=PHASE3_PRIMARY_SUMMARY_PATH,
    )
    parser.add_argument("--primary-run-root", default="runs/phase3")
    parser.add_argument("--primary-artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--run-root", default="runs/phase3-compute-conversion")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-compute-conversion",
    )
    parser.add_argument(
        "--output-root",
        default="results/phase3-compute-conversion",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=CONFIRMATION_BOOTSTRAP_REPETITIONS,
        choices=(CONFIRMATION_BOOTSTRAP_REPETITIONS,),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
