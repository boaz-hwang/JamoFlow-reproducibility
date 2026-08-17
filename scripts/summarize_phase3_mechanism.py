#!/usr/bin/env python3
"""Validate and summarize preregistered Phase 3 mechanism controls."""

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
from jamoflow.neural_training import shuffled_indices
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.phase3 import PHASE3_MODEL_SPEC, PHASE3_OPTIMIZATION_SPEC
from jamoflow.phase3_analysis import (
    empirical_nonnegative_bootstrap_tail,
    hierarchical_paired_bootstrap_estimates,
    holm_step_down_adjusted_values,
    paired_seed_lower_t_pvalue,
)
from jamoflow.phase3_mechanism import (
    ALL_SEEDS,
    DELAYED_POLICY,
    INITIAL_SEEDS,
    MECHANISM_POLICIES,
    PLACEBO_POLICY,
    WHITESPACE_POLICY,
    array_sha256,
    build_mechanism_patch_matrices,
    mechanism_cache_provenance,
    validate_mechanism_execution_gate,
)


SPLITS = ("train", "calibration", "test")
FULL_LIMITS = {
    "train": 128_000_000,
    "calibration": 8_000_000,
    "test": 16_000_000,
}
TARGETS_PER_SEQUENCE = PHASE3_MODEL_SPEC.sequence_length - 1
CONTRASTS = {
    "whitespace_minus_delayed2": (WHITESPACE_POLICY, DELAYED_POLICY),
    "whitespace_minus_placebo": (WHITESPACE_POLICY, PLACEBO_POLICY),
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


def _validate_manifest_execution(
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    if manifest.get("phase") != "phase3_mechanism":
        raise ValueError("mechanism manifest phase mismatch")
    if manifest.get("quick_smoke_only"):
        raise ValueError("refusing to promote quick mechanism controls")
    if manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("mechanism manifest model spec mismatch")
    if manifest.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict():
        raise ValueError("mechanism manifest optimization spec mismatch")
    if manifest.get("limits") != FULL_LIMITS:
        raise ValueError("mechanism manifest byte limits mismatch")
    if tuple(manifest.get("policies", [])) != MECHANISM_POLICIES:
        raise ValueError("mechanism manifest policy mismatch")
    if len(manifest.get("seeds", [])) != len(set(manifest.get("seeds", []))):
        raise ValueError("mechanism manifest contains duplicate seeds")
    if not set(seeds) <= set(manifest.get("seeds", [])):
        raise ValueError("mechanism manifest does not cover requested seeds")
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("mechanism manifest lacks invocation provenance")
    for seed in seeds:
        required_gate = "gate_i" if seed in INITIAL_SEEDS else "gate_j"
        matching = [
            invocation
            for invocation in invocations
            if isinstance(invocation, dict)
            and seed in invocation.get("seeds", [])
            and set(MECHANISM_POLICIES)
            <= set(invocation.get("policies", []))
            and invocation.get("save_checkpoints") is True
            and invocation.get("gate_authorization", {}).get("required_gate")
            == required_gate
            and invocation.get("gate_authorization", {}).get(
                "evidence_eligible"
            )
            is True
        ]
        if not matching:
            raise ValueError(
                f"mechanism manifest lacks an evidentiary invocation for seed {seed}"
            )


def _validate_authorization_summary_lineage(
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
    *,
    historical_initial_sha256: str,
    current_primary_sha256: str,
) -> dict[str, Any]:
    """Separate original initial authorization from corrected reanalysis."""

    invocations = manifest.get("invocations", [])
    initial_matches = [
        invocation
        for invocation in invocations
        if isinstance(invocation, dict)
        and set(INITIAL_SEEDS) <= set(invocation.get("seeds", []))
        and invocation.get("gate_authorization", {}).get("required_gate")
        == "gate_i"
        and invocation.get("primary_summary_sha256")
        == historical_initial_sha256
    ]
    if not initial_matches:
        raise ValueError(
            "mechanism initial controls are not bound to the historical Gate I"
        )
    confirmation_matches: list[dict[str, Any]] = []
    if seeds == ALL_SEEDS:
        confirmation_matches = [
            invocation
            for invocation in invocations
            if isinstance(invocation, dict)
            and {57721, 65537} <= set(invocation.get("seeds", []))
            and invocation.get("gate_authorization", {}).get("required_gate")
            == "gate_j"
            and invocation.get("primary_summary_sha256")
            == current_primary_sha256
        ]
        if not confirmation_matches:
            raise ValueError(
                "mechanism confirmation is not bound to the current Gate J"
            )
    return {
        "historical_initial_gate_i_summary_sha256": historical_initial_sha256,
        "historical_initial_authorization_match": True,
        "current_reanalysis_summary_sha256": current_primary_sha256,
        "confirmation_gate_j_authorization_match": (
            True if seeds == ALL_SEEDS else None
        ),
    }


def _mechanism_reanalysis_authorization(
    primary_summary: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Separate analysis of existing controls from authorization to train more."""

    if seeds == INITIAL_SEEDS:
        gate = primary_summary.get("gate_i")
        if not isinstance(gate, dict) or not isinstance(
            gate.get("overall_pass"),
            bool,
        ):
            raise ValueError(
                "corrected initial Gate I must be finalized before mechanism reanalysis"
            )
        current_pass = bool(gate["overall_pass"])
        return {
            "status": "historical_execution_current_reanalysis",
            "evidence_eligible": True,
            "execution_required_gate": "historical_gate_i",
            "current_primary_gate": "gate_i",
            "current_primary_gate_status": gate.get("status"),
            "current_primary_gate_pass": current_pass,
            "confirmation_progression_authorized": current_pass,
            "requested_seeds": list(seeds),
        }

    authorization = validate_mechanism_execution_gate(
        primary_summary,
        seeds,
        quick=False,
    )
    return {
        **authorization,
        "execution_required_gate": authorization["required_gate"],
        "current_primary_gate": authorization["required_gate"],
        "current_primary_gate_status": authorization.get(
            "required_gate_status"
        ),
        "current_primary_gate_pass": True,
        "confirmation_progression_authorized": True,
    }


def _reconstruct_context(
    manifest: dict[str, Any],
    data_root: Path,
) -> tuple[
    dict[str, NeuralStream],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    source = data_root / "ko.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    limits = manifest.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(SPLITS):
        raise ValueError("mechanism manifest limits are invalid")
    streams: dict[str, NeuralStream] = {}
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            source,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=int(limits[split]),
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        recorded = manifest.get("streams", {}).get(split, {})
        for key, value in stream.metadata().items():
            if recorded.get(key) != value:
                raise ValueError(
                    f"mechanism stream metadata mismatch: {split}/{key}"
                )
        stream_hash = hashlib.sha256(stream.data).hexdigest()
        if recorded.get("selected_stream_sha256") != stream_hash:
            raise ValueError(f"mechanism stream hash mismatch: {split}")
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
    matrices, diagnostics = build_mechanism_patch_matrices(
        inputs,
        boundaries,
        whitespace,
    )
    diagnostics["_provenance"] = mechanism_cache_provenance(
        inputs,
        boundaries,
        whitespace,
    )
    return streams, inputs, boundaries, matrices, diagnostics


def _load_report_and_loss(
    report_path: Path,
    loss_path: Path,
    *,
    seed: int,
    policy: str,
    expected_test_examples: int,
) -> tuple[dict[str, Any], np.ndarray, str]:
    report = _read_json(report_path)
    if report.get("seed") != seed or report.get("policy") != policy:
        raise ValueError(f"run identity mismatch in {report_path}")
    if report.get("parameters") != 19_596_096:
        raise ValueError(f"parameter-count mismatch in {report_path}")
    if report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError(f"model-spec mismatch in {report_path}")
    if report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict():
        raise ValueError(f"optimization-spec mismatch in {report_path}")
    with np.load(loss_path) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError(f"unexpected loss keys in {loss_path}")
        losses = archive["sequence_nll_nats"].astype(np.float64)
    if (
        losses.shape != (expected_test_examples,)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError(f"invalid sequence losses in {loss_path}")
    evaluation = report.get("evaluation", {}).get("test", {})
    predicted_bytes = expected_test_examples * TARGETS_PER_SEQUENCE
    if (
        evaluation.get("examples") != expected_test_examples
        or evaluation.get("predicted_bytes") != predicted_bytes
    ):
        raise ValueError(f"test evaluation count mismatch in {report_path}")
    reconstructed = float(losses.sum()) / (predicted_bytes * math.log(2))
    reported = float(report["evaluation"]["test"]["bpb"])
    if not math.isclose(reconstructed, reported, abs_tol=2e-5):
        raise ValueError(
            f"loss/report mismatch for seed {seed}/{policy}: "
            f"{reconstructed} versus {reported}"
        )
    file_hash = _sha256(loss_path)
    expected_hash = report.get("test_loss_file_sha256")
    if expected_hash is not None and expected_hash != file_hash:
        raise ValueError(f"loss hash mismatch for seed {seed}/{policy}")
    return report, losses, file_hash


def _load_runs(
    primary_run_root: Path,
    primary_artifact_root: Path,
    control_run_root: Path,
    control_artifact_root: Path,
    seeds: tuple[int, ...],
    expected_test_examples: int,
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, str]],
]:
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    hashes: dict[int, dict[str, str]] = {}
    for seed in seeds:
        reports[seed] = {}
        losses[seed] = {}
        hashes[seed] = {}
        report, values, file_hash = _load_report_and_loss(
            primary_run_root / f"seed-{seed}" / f"{WHITESPACE_POLICY}.json",
            primary_artifact_root
            / f"seed-{seed}"
            / f"{WHITESPACE_POLICY}-test-nll.npz",
            seed=seed,
            policy=WHITESPACE_POLICY,
            expected_test_examples=expected_test_examples,
        )
        reports[seed][WHITESPACE_POLICY] = report
        losses[seed][WHITESPACE_POLICY] = values
        hashes[seed][WHITESPACE_POLICY] = file_hash
        for policy in MECHANISM_POLICIES:
            report, values, file_hash = _load_report_and_loss(
                control_run_root / f"seed-{seed}" / f"{policy}.json",
                control_artifact_root
                / f"seed-{seed}"
                / f"{policy}-test-nll.npz",
                seed=seed,
                policy=policy,
                expected_test_examples=expected_test_examples,
            )
            reports[seed][policy] = report
            losses[seed][policy] = values
            hashes[seed][policy] = file_hash
    return reports, losses, hashes


def _contrast_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    document_window_map: DocumentWindowMap,
    *,
    repetitions: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        effects = [
            float(reports[seed][left]["evaluation"]["test"]["bpb"])
            - float(reports[seed][right]["evaluation"]["test"]["bpb"])
            for seed in seeds
        ]
        sequence_differences = [
            losses[seed][left] - losses[seed][right] for seed in seeds
        ]
        for seed, expected, values in zip(
            seeds, effects, sequence_differences, strict=True
        ):
            reconstructed = float(values.mean()) / (
                TARGETS_PER_SEQUENCE * math.log(2)
            )
            if not math.isclose(expected, reconstructed, abs_tol=2e-5):
                raise ValueError(
                    f"contrast reconstruction failed for {name}/seed-{seed}"
                )
        estimates = hierarchical_paired_bootstrap_estimates(
            sequence_differences,
            targets_per_sequence=TARGETS_PER_SEQUENCE,
            repetitions=repetitions,
            seed=20_260_830 + index,
        )
        lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
        document_cluster = document_cluster_contrast_summary(
            sequence_differences,
            document_window_map,
            targets_per_sequence=TARGETS_PER_SEQUENCE,
            repetitions=repetitions,
            seed=20_260_930 + index,
        )
        result[name] = {
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(seeds),
            "paired_differences_bpb": effects,
            "negative_seed_count": int(sum(value < 0 for value in effects)),
            "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            "hierarchical_bootstrap_95_interval": {
                "repetitions": repetitions,
                "seed": 20_260_830 + index,
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
                paired_seed_lower_t_pvalue(effects)
            ),
        }

    raw = {
        name: values["paired_seed_one_sided_t_pvalue"]
        for name, values in result.items()
    }
    adjusted = holm_step_down_adjusted_values(raw)
    ordered = sorted(raw, key=lambda name: (raw[name], name))
    for rank, name in enumerate(ordered):
        result[name]["holm_mechanism_family"] = {
            "rank": rank + 1,
            "family_size": len(ordered),
            "test": "one-sided paired-seed Student-t",
            "raw_one_sided_seed_t_pvalue": raw[name],
            "holm_adjusted_seed_t_pvalue": adjusted[name],
            "rejects_at_familywise_alpha_0_05": adjusted[name] <= 0.05,
            "bootstrap_nonnegative_tail_diagnostic": result[name][
                "bootstrap_nonnegative_tail"
            ],
        }
    return result


def _validate_report_context(
    reports: dict[int, dict[str, dict[str, Any]]],
    streams: dict[str, NeuralStream],
    diagnostics: dict[str, Any],
    seeds: tuple[int, ...],
) -> dict[str, str]:
    expected_examples = {
        split: streams[split].sequence_count for split in SPLITS
    }
    stream_hashes = {
        split: hashlib.sha256(streams[split].data).hexdigest()
        for split in SPLITS
    }
    expected_steps = math.ceil(
        expected_examples["train"] / PHASE3_OPTIMIZATION_SPEC.batch_size
    )
    expected_orders: dict[str, str] = {}
    for seed in seeds:
        expected_order = array_sha256(
            shuffled_indices(expected_examples["train"], seed)
        )
        expected_orders[str(seed)] = expected_order
        for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES):
            report = reports[seed][policy]
            if report.get("training_order_sha256") != expected_order:
                raise ValueError(
                    f"training order differs from reconstruction: {seed}/{policy}"
                )
            training = report.get("training", {})
            if (
                training.get("examples") != expected_examples["train"]
                or training.get("steps") != expected_steps
                or training.get("predicted_bytes")
                != expected_examples["train"] * TARGETS_PER_SEQUENCE
            ):
                raise ValueError(f"training count mismatch: {seed}/{policy}")
            for split in ("calibration", "test"):
                evaluation = report.get("evaluation", {}).get(split, {})
                if (
                    evaluation.get("examples") != expected_examples[split]
                    or evaluation.get("predicted_bytes")
                    != expected_examples[split] * TARGETS_PER_SEQUENCE
                ):
                    raise ValueError(
                        f"evaluation count mismatch: {split}/{seed}/{policy}"
                    )
            for split in SPLITS:
                patch = report.get("patch_diagnostics", {}).get(split, {})
                if (
                    patch.get("examples") != expected_examples[split]
                    or patch.get("minimum_data_patches") != 86
                    or patch.get("maximum_data_patches") != 86
                    or patch.get("mean_data_patches") != 86.0
                    or patch.get("padding_slots") != 0
                ):
                    raise ValueError(
                        f"patch count mismatch: {split}/{seed}/{policy}"
                    )
                expected_matrix_hash = (
                    diagnostics["whitespace_reference"][split]["matrix_sha256"]
                    if policy == WHITESPACE_POLICY
                    else diagnostics["splits"][split][policy]["matrix_sha256"]
                )
                if (
                    report.get("patch_matrix_sha256", {}).get(split)
                    != expected_matrix_hash
                ):
                    raise ValueError(
                        f"patch matrix differs from reconstruction: "
                        f"{split}/{seed}/{policy}"
                    )
            if (
                policy in MECHANISM_POLICIES
                and report.get("stream_selected_sha256") != stream_hashes
            ):
                raise ValueError(
                    f"control stream provenance mismatch: {seed}/{policy}"
                )
    return expected_orders


def _checkpoint_integrity(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    primary_artifact_root: Path,
    control_artifact_root: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    state_hashes: dict[str, dict[str, str]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    for seed in seeds:
        state_hashes[str(seed)] = {}
        artifact_hashes[str(seed)] = {}
        for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES):
            root = (
                primary_artifact_root
                if policy == WHITESPACE_POLICY
                else control_artifact_root
            )
            checkpoint = root / f"seed-{seed}" / f"{policy}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            artifact_hash = _sha256(checkpoint)
            state_hash = _checkpoint_state_sha256(checkpoint)
            report = reports[seed][policy]
            if state_hash != report.get("trained_state_sha256"):
                raise ValueError(f"checkpoint state mismatch: {seed}/{policy}")
            if (
                policy in MECHANISM_POLICIES
                and artifact_hash != report.get("checkpoint_artifact_sha256")
            ):
                raise ValueError(
                    f"checkpoint artifact mismatch: {seed}/{policy}"
                )
            state_hashes[str(seed)][policy] = state_hash
            artifact_hashes[str(seed)][policy] = artifact_hash
    return state_hashes, artifact_hashes


def _validate_primary_summary_context(
    primary_summary: dict[str, Any],
    mechanism_manifest: dict[str, Any],
    seeds: tuple[int, ...],
    loss_hashes: dict[int, dict[str, str]],
    checkpoint_state_hashes: dict[str, dict[str, str]],
) -> None:
    integrity = primary_summary.get("integrity", {})
    if integrity.get("all_integrity_checks_pass") is not True:
        raise ValueError("primary summary integrity is incomplete")
    primary_manifest = primary_summary.get("run_manifest", {})
    for key in ("model_spec", "optimization_spec", "limits", "streams"):
        if primary_manifest.get(key) != mechanism_manifest.get(key):
            raise ValueError(
                f"primary and mechanism manifests differ: {key}"
            )
    if not set(seeds) <= set(primary_summary.get("seeds", [])):
        raise ValueError("primary summary does not cover mechanism seeds")
    for seed in seeds:
        primary_seed = integrity.get("by_seed", {}).get(str(seed), {})
        if (
            primary_seed.get("checkpoint_state_sha256", {}).get(
                WHITESPACE_POLICY
            )
            != checkpoint_state_hashes[str(seed)][WHITESPACE_POLICY]
        ):
            raise ValueError(
                f"primary summary checkpoint mismatch for seed {seed}"
            )
        if (
            primary_seed.get("loss_artifact_sha256", {}).get(
                WHITESPACE_POLICY
            )
            != loss_hashes[seed][WHITESPACE_POLICY]
        ):
            raise ValueError(f"primary summary loss mismatch for seed {seed}")


def _integrity_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, np.ndarray]],
    loss_hashes: dict[int, dict[str, str]],
    diagnostics: dict[str, Any],
    seeds: tuple[int, ...],
    checkpoint_state_hashes: dict[str, dict[str, str]],
    checkpoint_artifact_hashes: dict[str, dict[str, str]],
    expected_orders: dict[str, str],
    diagnostics_match_reconstruction: bool,
    primary_summary_matches_evidence: bool,
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    initializations_ok = True
    orders_ok = True
    rates_ok = True
    report_matrix_hashes_ok = True
    loss_shapes_ok = True
    expected_loss_count = len(losses[seeds[0]][WHITESPACE_POLICY])
    for seed in seeds:
        seed_reports = reports[seed]
        initializations = {
            report["initialization_sha256"] for report in seed_reports.values()
        }
        orders = {
            report["training_order_sha256"] for report in seed_reports.values()
        }
        seed_rates_ok = all(
            report["patch_diagnostics"][split]["mean_data_patches"] == 86.0
            for report in seed_reports.values()
            for split in SPLITS
        )
        seed_matrix_ok = all(
            seed_reports[policy]["patch_matrix_sha256"][split]
            == (
                diagnostics["whitespace_reference"][split]["matrix_sha256"]
                if policy == WHITESPACE_POLICY
                else diagnostics["splits"][split][policy]["matrix_sha256"]
            )
            for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES)
            for split in SPLITS
        )
        seed_loss_shapes_ok = all(
            len(losses[seed][policy]) == expected_loss_count
            for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES)
        )
        initializations_ok &= len(initializations) == 1
        orders_ok &= len(orders) == 1
        rates_ok &= seed_rates_ok
        report_matrix_hashes_ok &= seed_matrix_ok
        loss_shapes_ok &= seed_loss_shapes_ok
        by_seed[str(seed)] = {
            "identical_initialization": len(initializations) == 1,
            "identical_training_order": len(orders) == 1,
            "all_policy_split_rates_exactly_86": seed_rates_ok,
            "all_patch_hashes_match_seed_independent_diagnostics": seed_matrix_ok,
            "all_loss_vectors_same_shape": seed_loss_shapes_ok,
            "training_order_matches_seeded_reconstruction": (
                next(iter(orders)) == expected_orders[str(seed)]
                if len(orders) == 1
                else False
            ),
            "loss_artifact_sha256": loss_hashes[seed],
            "checkpoint_state_sha256": checkpoint_state_hashes[str(seed)],
            "checkpoint_artifact_sha256": checkpoint_artifact_hashes[
                str(seed)
            ],
        }
    seed_independent = all(
        len(
            {
                reports[seed][policy]["patch_matrix_sha256"][split]
                for seed in seeds
            }
        )
        == 1
        for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES)
        for split in SPLITS
    )
    overall = bool(
        initializations_ok
        and orders_ok
        and rates_ok
        and report_matrix_hashes_ok
        and loss_shapes_ok
        and seed_independent
        and diagnostics_match_reconstruction
        and primary_summary_matches_evidence
    )
    return {
        "all_initializations_paired": initializations_ok,
        "all_training_orders_paired": orders_ok,
        "all_compared_rates_exactly_86": rates_ok,
        "all_report_matrix_hashes_match_diagnostics": report_matrix_hashes_ok,
        "all_loss_vectors_same_shape": loss_shapes_ok,
        "patch_matrices_seed_independent": seed_independent,
        "all_checkpoint_states_match_reports": True,
        "all_control_checkpoint_artifacts_match_reports": True,
        "diagnostics_match_independent_reconstruction": (
            diagnostics_match_reconstruction
        ),
        "primary_summary_matches_loaded_evidence": (
            primary_summary_matches_evidence
        ),
        "test_sequence_losses_per_policy": expected_loss_count,
        "by_seed": by_seed,
        "all_integrity_checks_pass": overall,
    }


def gate_m_summary(
    contrasts: dict[str, Any],
    *,
    seed_count: int,
    integrity_pass: bool,
) -> dict[str, Any]:
    """Evaluate the preregistered initial-three or final-five attribution gate."""

    if seed_count not in (3, 5):
        return {
            "status": "not_evaluated_requires_three_or_five_seeds",
            "overall_pass": None,
            "seed_count": seed_count,
        }
    final = seed_count == 5
    maximum_mean = -0.003 if final else -0.002
    required_negative = 4 if final else 2
    by_contrast: dict[str, Any] = {}
    for name in CONTRASTS:
        contrast = contrasts[name]
        mean_effect = float(contrast["paired_t_95_interval"]["mean"])
        negative_count = int(contrast["negative_seed_count"])
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
        holm = contrast["holm_mechanism_family"]
        checks = {
            "mean_bpb": mean_effect,
            "maximum_mean_bpb": maximum_mean,
            "mean_threshold_pass": mean_effect <= maximum_mean,
            "negative_seed_count": negative_count,
            "required_negative_seed_count": required_negative,
            "negative_seed_count_pass": negative_count >= required_negative,
            "bootstrap_95_upper_bpb": bootstrap_upper,
            "bootstrap_95_upper_below_zero": bootstrap_upper < 0,
            "document_cluster_95_upper_bpb": document_upper,
            "document_cluster_95_upper_below_zero": document_upper < 0,
            "document_cluster_coverage_pass": document_coverage_pass,
            "holm_adjusted_seed_t_pvalue": holm[
                "holm_adjusted_seed_t_pvalue"
            ],
            "holm_adjusted_seed_t_pvalue_at_most_0_05": (
                holm["holm_adjusted_seed_t_pvalue"] <= 0.05
            ),
        }
        checks["contrast_pass"] = bool(
            checks["mean_threshold_pass"]
            and checks["negative_seed_count_pass"]
            and checks["document_cluster_95_upper_below_zero"]
            and checks["document_cluster_coverage_pass"]
            and (
                not final
                or (
                    checks["bootstrap_95_upper_below_zero"]
                    and checks["holm_adjusted_seed_t_pvalue_at_most_0_05"]
                )
            )
        )
        by_contrast[name] = checks
    contrasts_pass = all(value["contrast_pass"] for value in by_contrast.values())
    overall = bool(integrity_pass and contrasts_pass)
    return {
        "status": "pass" if overall else "fail",
        "overall_pass": overall,
        "stage": "final_five_seed" if final else "initial_three_seed",
        "seed_count": seed_count,
        "integrity_pass": integrity_pass,
        "by_contrast": by_contrast,
        "claim_if_pass": (
            "observed whitespace association survives delayed-phase and "
            "rate-matched causal-event controls"
        ),
        "claim_if_fail": (
            "describe W only as a deterministic relocation heuristic in this "
            "geometry"
        ),
    }


def _write_observations(
    path: Path,
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "seed",
                "source",
                "policy",
                "test_bpb",
                "calibration_bpb",
                "training_seconds",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES):
                report = reports[seed][policy]
                writer.writerow(
                    {
                        "seed": seed,
                        "source": (
                            "phase3_primary_reused"
                            if policy == WHITESPACE_POLICY
                            else "phase3_mechanism_new"
                        ),
                        "policy": policy,
                        "test_bpb": report["evaluation"]["test"]["bpb"],
                        "calibration_bpb": report["evaluation"]["calibration"][
                            "bpb"
                        ],
                        "training_seconds": report["training"]["elapsed_seconds"],
                    }
                )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    if seeds not in (INITIAL_SEEDS, ALL_SEEDS):
        raise ValueError(
            "mechanism summary requires the preregistered initial 3 or final 5 seeds"
        )
    primary_summary_path = Path(args.primary_summary)
    primary_summary = _read_json(primary_summary_path)
    historical_summary_path = Path(args.historical_authorization_summary)
    historical_summary = _read_json(historical_summary_path)
    if (
        historical_summary.get("gate_i", {}).get("overall_pass") is not True
        or historical_summary.get("integrity", {}).get(
            "all_integrity_checks_pass"
        )
        is not True
    ):
        raise ValueError("historical Gate I authorization artifact is invalid")
    authorization = _mechanism_reanalysis_authorization(
        primary_summary,
        seeds,
    )
    control_run_root = Path(args.control_run_root)
    manifest_path = control_run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    _validate_manifest_execution(manifest, seeds)
    authorization_lineage = _validate_authorization_summary_lineage(
        manifest,
        seeds,
        historical_initial_sha256=_sha256(historical_summary_path),
        current_primary_sha256=_sha256(primary_summary_path),
    )
    streams, inputs, boundaries, matrices, reconstructed_diagnostics = (
        _reconstruct_context(manifest, Path(args.data_root))
    )
    document_window_map = reconstruct_document_window_map(
        Path(args.data_root) / "ko.jsonl",
        split="test",
        byte_limit=FULL_LIMITS["test"],
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        expected_stream=streams["test"].data,
    )

    reports, losses, hashes = _load_runs(
        Path(args.primary_run_root),
        Path(args.primary_artifact_root),
        control_run_root,
        Path(args.control_artifact_root),
        seeds,
        streams["test"].sequence_count,
    )
    diagnostics_path = control_run_root / "mechanism-patch-diagnostics.json"
    diagnostics = _read_json(diagnostics_path)
    diagnostics_match_reconstruction = diagnostics == reconstructed_diagnostics
    if not diagnostics_match_reconstruction:
        raise ValueError("mechanism diagnostics differ from reconstruction")
    for split in SPLITS:
        for policy in MECHANISM_POLICIES:
            expected_hash = diagnostics["splits"][split][policy][
                "matrix_sha256"
            ]
            if array_sha256(matrices[split][policy]) != expected_hash:
                raise ValueError(
                    f"mechanism matrix reconstruction failed: {split}/{policy}"
                )
    expected_orders = _validate_report_context(
        reports,
        streams,
        diagnostics,
        seeds,
    )
    checkpoint_state_hashes, checkpoint_artifact_hashes = (
        _checkpoint_integrity(
            reports,
            seeds,
            Path(args.primary_artifact_root),
            Path(args.control_artifact_root),
        )
    )
    _validate_primary_summary_context(
        primary_summary,
        manifest,
        seeds,
        hashes,
        checkpoint_state_hashes,
    )
    primary_summary_matches_evidence = True
    contrasts = _contrast_summary(
        reports,
        losses,
        seeds,
        document_window_map,
        repetitions=args.bootstrap_repetitions,
    )
    integrity = _integrity_summary(
        reports,
        losses,
        hashes,
        diagnostics,
        seeds,
        checkpoint_state_hashes,
        checkpoint_artifact_hashes,
        expected_orders,
        diagnostics_match_reconstruction,
        primary_summary_matches_evidence,
    )
    if not integrity["all_integrity_checks_pass"]:
        raise ValueError("Phase 3 mechanism integrity checks failed")
    gate_m = gate_m_summary(
        contrasts,
        seed_count=len(seeds),
        integrity_pass=integrity["all_integrity_checks_pass"],
    )
    gate_m["current_primary_gate"] = authorization["current_primary_gate"]
    gate_m["current_primary_gate_pass"] = authorization[
        "current_primary_gate_pass"
    ]
    gate_m["progression_authorized"] = bool(
        gate_m["overall_pass"]
        and authorization["confirmation_progression_authorized"]
    )
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "scope": "Phase 3 conditional mechanism attribution controls",
        "seeds": list(seeds),
        "policies": [WHITESPACE_POLICY, *MECHANISM_POLICIES],
        "run_manifest": manifest,
        "source": {
            "primary_summary_sha256": _sha256(primary_summary_path),
            "historical_authorization_summary_sha256": _sha256(
                historical_summary_path
            ),
            "mechanism_manifest_sha256": _sha256(manifest_path),
            "mechanism_diagnostics_sha256": _sha256(diagnostics_path),
            "git_commit_at_control_run_start": manifest.get("git_commit"),
        },
        "gate_authorization_revalidated": authorization,
        "authorization_summary_lineage": authorization_lineage,
        "quality": {
            policy: numeric_summary(
                [
                    reports[seed][policy]["evaluation"]["test"]["bpb"]
                    for seed in seeds
                ]
            )
            for policy in (WHITESPACE_POLICY, *MECHANISM_POLICIES)
        },
        "contrasts": contrasts,
        "integrity": integrity,
        "document_cluster_inference": document_window_map.metadata(),
        "gate_m": gate_m,
        "diagnostics": diagnostics,
        "interpretation_guardrail": (
            "Gate M attributes a paired W effect beyond two specified controls. "
            "Reanalysis of historically authorized initial controls does not "
            "authorize confirmation unless the corrected current primary gate "
            "also passes. Gate M does not identify Korean morphology, optimal "
            "segmentation, or general learned-router superiority."
        ),
    }
    output_root = Path(args.output_root)
    _write_json(output_root / "summary.json", summary)
    _write_observations(output_root / "observations.csv", reports, seeds)
    print(json.dumps(gate_m, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run-root", default="runs/phase3")
    parser.add_argument("--primary-artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--primary-summary",
        default="results/phase3-primary-clustered/summary.json",
    )
    parser.add_argument(
        "--historical-authorization-summary",
        default="results/phase3-primary/summary.json",
    )
    parser.add_argument("--control-run-root", default="runs/phase3-mechanism")
    parser.add_argument(
        "--control-artifact-root", default="artifacts/phase3-mechanism"
    )
    parser.add_argument(
        "--output-root",
        default="results/phase3-mechanism-clustered",
    )
    parser.add_argument(
        "--data-root", default="data/processed/hplt3-korean-phase3"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
