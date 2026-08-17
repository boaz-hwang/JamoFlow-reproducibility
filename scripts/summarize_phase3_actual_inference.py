#!/usr/bin/env python3
"""Reconstruct the five-seed actual-inference benchmark and evaluate its gate."""

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

from jamoflow.actual_inference_protocol import (
    ACTUAL_INFERENCE_PROTOCOL_VERSION,
    ACTUAL_INFERENCE_SELECTION_ALGORITHM,
    COMPONENTS,
    CONTINUATION_BYTES,
    CORRECTNESS_CONTINUATION_BYTES,
    FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES,
    FREE_RUNNING_UTF8_CONSTRAINT,
    MEASURED_CASES,
    MODES,
    OUTPUT_DIAGNOSTICS,
    PROMPT_BYTES,
    REPETITIONS,
    ROLES,
    SEED_EXECUTION_ORDER_SEED,
    SEEDS,
    TIME_TO_OUTPUT_SEMANTICS,
    TIMING_ORDER_SEED,
    WARMUP_CASES,
    decode_forward_steps,
    free_running_maximum_output_bytes,
    reconstruct_valid_completion_metrics,
    runtime_observed_bytes,
    timing_environment_eligible,
    validate_output_diagnostic_arrays,
)
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.incremental_blt import INCREMENTAL_ENTROPY_POLICIES
from jamoflow.inference_benchmark import (
    multiseed_latency_component_pass,
    multiseed_paired_latency,
    select_inference_cases,
    timing_order_schedule,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.phase3 import PHASE3_MODEL_SPEC


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


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"unexpected checkpoint entry: {path}")
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


def _expected_array_keys() -> set[str]:
    return {
        f"{mode}__{metric}__{role}"
        for mode in MODES
        for role in ROLES
        for metric in (
            *COMPONENTS,
            "global_patches",
            *OUTPUT_DIAGNOSTICS,
            "mps_current_bytes",
            "mps_driver_bytes",
        )
    }


def _validate_manifest(
    manifest: dict[str, Any],
    selection_path: Path,
    quality_path: Path,
) -> None:
    protocol = manifest.get("protocol", {})
    if (
        manifest.get("schema_version") != ACTUAL_INFERENCE_PROTOCOL_VERSION
        or manifest.get("quick_smoke_only") is not False
        or manifest.get("evidence_eligible") is not True
        or tuple(manifest.get("seeds", [])) != SEEDS
        or manifest.get("selection", {}).get("sha256") != _sha256(selection_path)
        or manifest.get("quality_summary", {}).get("sha256")
        != _sha256(quality_path)
        or protocol.get("prompt_bytes") != PROMPT_BYTES
        or protocol.get("continuation_bytes") != CONTINUATION_BYTES
        or protocol.get("warmup_cases") != WARMUP_CASES
        or protocol.get("measured_cases") != MEASURED_CASES
        or protocol.get("repetitions_per_prompt") != REPETITIONS
        or tuple(protocol.get("modes", [])) != MODES
        or tuple(protocol.get("components", [])) != COMPONENTS
        or protocol.get("correctness_continuation_bytes")
        != CORRECTNESS_CONTINUATION_BYTES
        or protocol.get("timing_order_seed") != TIMING_ORDER_SEED
        or protocol.get("parallel_prefill_only_in_timing") is not True
        or protocol.get(
            "selector_router_cache_and_synchronization_inside_timing"
        )
        is not True
        or protocol.get("time_to_output_semantics")
        != TIME_TO_OUTPUT_SEMANTICS
        or protocol.get("controlled_replay_decode_forward_steps")
        != decode_forward_steps(CONTINUATION_BYTES)
        or protocol.get("controlled_replay_emitted_output_bytes")
        != CONTINUATION_BYTES
        or protocol.get("controlled_replay_runtime_observed_bytes")
        != runtime_observed_bytes()
        or protocol.get("free_running_minimum_output_bytes")
        != CONTINUATION_BYTES
        or protocol.get("free_running_maximum_output_bytes")
        != free_running_maximum_output_bytes()
        or protocol.get("free_running_maximum_overshoot_bytes")
        != FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES
        or protocol.get("free_running_utf8_constraint")
        != FREE_RUNNING_UTF8_CONSTRAINT
        or protocol.get("utf8_dfa_mask_compilation_outside_timing")
        is not True
        or protocol.get("utf8_mask_argmax_state_and_stop_checks_inside_timing")
        is not True
        or protocol.get("session_start_timing_environment_eligible")
        is not True
        or not timing_environment_eligible(manifest.get("session_start", {}))
    ):
        raise ValueError("actual-inference manifest is not evidentiary")
    if manifest.get("environment", {}).get("device") != "mps":
        raise ValueError("compact actual-inference evidence must be the MPS run")
    expected_order = timing_order_schedule(
        SEEDS,
        mode_count=len(MODES),
        prompt_count=MEASURED_CASES,
        repetitions=REPETITIONS,
        random_seed=TIMING_ORDER_SEED,
    )
    if protocol.get("timing_schedule_sha256") != _array_sha256(expected_order):
        raise ValueError("actual-inference timing order does not reconstruct")
    expected_warmup_order = timing_order_schedule(
        SEEDS,
        mode_count=len(MODES),
        prompt_count=WARMUP_CASES,
        repetitions=1,
        random_seed=TIMING_ORDER_SEED + 1,
    )
    if protocol.get("warmup_schedule_sha256") != _array_sha256(
        expected_warmup_order
    ):
        raise ValueError("actual-inference warmup order does not reconstruct")
    expected_seed_order = tuple(
        SEEDS[index]
        for index in np.random.default_rng(
            SEED_EXECUTION_ORDER_SEED
        ).permutation(len(SEEDS))
    )
    if (
        protocol.get("seed_execution_order_seed")
        != SEED_EXECUTION_ORDER_SEED
        or tuple(manifest.get("seed_execution_order", []))
        != expected_seed_order
    ):
        raise ValueError("actual-inference seed order does not reconstruct")


def _validate_upstream(
    manifest: dict[str, Any],
    selection_path: Path,
    quality_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selection = _read_json(selection_path)
    quality = _read_json(quality_path)
    if (
        quality.get("selection", {}).get("sha256") != _sha256(selection_path)
        or quality.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or quality.get("quality_noninferiority", {}).get("overall_pass") is not True
        or manifest.get("roles")
        != {role: selection[role] for role in ROLES}
    ):
        raise ValueError("actual-inference upstream quality lineage failed")
    phase3_item = quality["phase3_confirmation_summary"]
    phase3_path = Path(phase3_item["path"])
    if _sha256(phase3_path) != phase3_item["sha256"]:
        raise ValueError("Phase 3 confirmation summary changed")
    phase3 = _read_json(phase3_path)
    if (
        tuple(phase3.get("seeds", [])) != SEEDS
        or phase3.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or phase3.get("gate_j", {}).get("overall_pass") is not True
    ):
        raise ValueError("actual-inference upstream Gate J failed")
    return selection, quality, phase3


def _reconstruct_cases(
    phase3: dict[str, Any],
    manifest: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    phase3_manifest = phase3["run_manifest"]
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
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
        phase3_manifest.get("source_artifact") != source_artifact
        or phase3_manifest.get("source_integrity_artifact") != integrity_artifact
    ):
        raise ValueError("actual-inference source artifacts do not reconstruct")
    stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=int(phase3_manifest["limits"]["test"]),
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    selected_stream_hash = hashlib.sha256(stream.data).hexdigest()
    if (
        selected_stream_hash
        != phase3_manifest["streams"]["test"]["selected_stream_sha256"]
    ):
        raise ValueError("actual-inference test stream does not reconstruct")
    document_window_map = reconstruct_document_window_map(
        source_path,
        split="test",
        byte_limit=int(phase3_manifest["limits"]["test"]),
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        expected_stream=stream.data,
    )
    document_contained = document_window_map.document_indices >= 0
    cases = select_inference_cases(
        inputs[document_contained],
        boundaries[document_contained],
        cluster_ids=document_window_map.document_indices[document_contained],
        case_count=WARMUP_CASES + MEASURED_CASES,
        prompt_length=PROMPT_BYTES,
        continuation_length=CONTINUATION_BYTES,
    )
    context = {
        "source_artifact": source_artifact,
        "source_integrity_artifact": integrity_artifact,
        "test_stream_sha256": selected_stream_hash,
        "test_sequence_count": len(inputs),
        "selection_algorithm": ACTUAL_INFERENCE_SELECTION_ALGORITHM,
        "document_window_map": document_window_map.metadata(),
        **cases.public_metadata(),
    }
    if manifest.get("case_context") != context:
        raise ValueError("actual-inference case selection metadata differs")
    return context


def _policy_roots(
    descriptor: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    if descriptor["model_family"] == "compute_conversion":
        return Path(args.conversion_run_root), Path(args.conversion_artifact_root)
    if descriptor["model_family"] == "phase3":
        return Path(args.phase3_run_root), Path(args.phase3_artifact_root)
    raise ValueError("unknown inference model family")


def _validate_checkpoint_provenance(
    report: dict[str, Any],
    selection: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    recorded_roles = report.get("checkpoint_provenance", {})
    if set(recorded_roles) != set(ROLES):
        raise ValueError(f"checkpoint provenance roles differ for seed {seed}")
    for role in ROLES:
        descriptor = selection[role]
        policy = descriptor["policy"]
        run_root, artifact_root = _policy_roots(descriptor, args)
        training_report_path = run_root / f"seed-{seed}" / f"{policy}.json"
        checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
        recorded = recorded_roles[role]
        actual = {
            "training_report_artifact_sha256": _sha256(training_report_path),
            "checkpoint_artifact_sha256": _sha256(checkpoint_path),
            "checkpoint_state_sha256": _checkpoint_state_sha256(checkpoint_path),
        }
        if any(recorded.get(key) != value for key, value in actual.items()):
            raise ValueError(f"checkpoint changed after timing: {seed}/{role}")
        router_output = None
        if descriptor["runtime_policy"] in INCREMENTAL_ENTROPY_POLICIES:
            router_paths = {
                "router_checkpoint_artifact_sha256": (
                    artifact_root / f"seed-{seed}" / "router.pt"
                ),
                "router_report_artifact_sha256": (
                    run_root / f"seed-{seed}" / "router.json"
                ),
                "threshold_cache_artifact_sha256": (
                    artifact_root / f"seed-{seed}" / "threshold-patches.npz"
                ),
                "threshold_diagnostics_artifact_sha256": (
                    run_root
                    / f"seed-{seed}"
                    / "threshold-patch-diagnostics.json"
                ),
            }
            router_recorded = recorded.get("router")
            if not isinstance(router_recorded, dict):
                raise ValueError("timed learned reference lacks router provenance")
            router_output = {
                key: _sha256(path) for key, path in router_paths.items()
            }
            router_output["router_checkpoint_state_sha256"] = (
                _checkpoint_state_sha256(
                    artifact_root / f"seed-{seed}" / "router.pt"
                )
            )
            if any(
                router_recorded.get(key) != value
                for key, value in router_output.items()
            ):
                raise ValueError(f"router changed after timing: {seed}/{role}")
        elif recorded.get("router") is not None:
            raise ValueError("structural runtime unexpectedly records a router")
        output[role] = {**actual, "router": router_output}
    return output


def _load_seed(
    seed: int,
    seed_index: int,
    run_root: Path,
    artifact_root: Path,
    selection: dict[str, Any],
    quality_path: Path,
    selection_path: Path,
    order: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    report_path = run_root / f"seed-{seed}.json"
    timing_path = artifact_root / f"seed-{seed}-timings.npz"
    in_progress_path = run_root / f"seed-{seed}.in-progress.json"
    if in_progress_path.exists():
        raise ValueError(f"actual-inference seed is incomplete: {seed}")
    report = _read_json(report_path)
    if (
        report.get("seed") != seed
        or report.get("selection_sha256") != _sha256(selection_path)
        or report.get("quality_summary_sha256") != _sha256(quality_path)
        or report.get("timing_schedule_sha256")
        != _array_sha256(order[seed_index])
        or report.get("timing_artifact_sha256") != _sha256(timing_path)
        or report.get("session_start_timing_environment_eligible") is not True
        or not timing_environment_eligible(report.get("session_start", {}))
        or report.get("session_end_timing_environment_eligible") is not True
        or not timing_environment_eligible(report.get("session_end", {}))
        or set(report.get("correctness", {})) != set(ROLES)
        or any(
            value.get("pass") is not True
            or value.get("argmax_match_rate") != 1.0
            or value.get("full_prefix_boundary_position_comparisons", 0) <= 0
            or value.get(
                "parallel_prefill_and_continuation_comparisons",
                0,
            )
            <= 0
            for value in report.get("correctness", {}).values()
        )
    ):
        raise ValueError(f"actual-inference report failed: seed {seed}")
    checkpoint_evidence = _validate_checkpoint_provenance(
        report,
        selection,
        seed,
        args,
    )
    arrays: dict[str, np.ndarray] = {}
    with np.load(timing_path, allow_pickle=False) as archive:
        if set(archive.files) != _expected_array_keys():
            raise ValueError(f"timing array keys differ: seed {seed}")
        for key in archive.files:
            values = archive[key]
            if (
                values.shape != (MEASURED_CASES, REPETITIONS)
                or report.get("timing_array_sha256", {}).get(key)
                != _array_sha256(values)
            ):
                raise ValueError(f"timing array lineage differs: {seed}/{key}")
            if "_ms__" in key:
                if (
                    values.dtype != np.float64
                    or not np.isfinite(values).all()
                    or np.any(values <= 0)
                ):
                    raise ValueError(f"latency values invalid: {seed}/{key}")
            elif "global_patches" in key:
                if (
                    not np.issubdtype(values.dtype, np.integer)
                    or np.any(values <= 0)
                    or np.any(
                        values
                        > runtime_observed_bytes(
                            PROMPT_BYTES,
                            free_running_maximum_output_bytes(),
                        )
                    )
                ):
                    raise ValueError(f"global patch values invalid: {seed}/{key}")
            elif any(
                f"__{diagnostic}__" in key
                for diagnostic in OUTPUT_DIAGNOSTICS
            ):
                if not np.issubdtype(values.dtype, np.integer):
                    raise ValueError(
                        f"output diagnostic values invalid: {seed}/{key}"
                    )
            elif (
                values.dtype != np.int64
                or np.any(values < 0)
            ):
                raise ValueError(f"MPS memory values invalid: {seed}/{key}")
            arrays[key] = values.copy()
    validate_output_diagnostic_arrays(
        arrays,
        expected_shape=(MEASURED_CASES, REPETITIONS),
    )
    for mode in MODES:
        for role in ROLES:
            end_to_end = arrays[f"{mode}__end_to_end_ms__{role}"]
            component_sum = (
                arrays[f"{mode}__ttft_ms__{role}"]
                + arrays[f"{mode}__decode_ms__{role}"]
            )
            if not np.allclose(end_to_end, component_sum, rtol=0, atol=1e-9):
                raise ValueError(f"timing components do not add: {seed}/{mode}/{role}")
    generation = report.get("generation", {})
    if set(generation) != set(ROLES):
        raise ValueError(f"free-running generation metrics invalid: seed {seed}")
    for role in ROLES:
        expected_generation = reconstruct_valid_completion_metrics(
            arrays,
            role,
        )
        recorded_generation = generation[role]
        if (
            any(
                recorded_generation.get(key) != value
                for key, value in expected_generation.items()
            )
            or recorded_generation.get("utf8_constraint")
            != FREE_RUNNING_UTF8_CONSTRAINT
            or recorded_generation.get("all_stops_at_strict_utf8_boundary")
            is not True
            or recorded_generation.get(
                "greedy_outputs_identical_across_repetitions"
            )
            is not True
        ):
            raise ValueError(
                f"free-running generation metrics differ: seed {seed}/{role}"
            )
    evidence = {
        "timing_report_artifact_sha256": _sha256(report_path),
        "timing_array_artifact_sha256": _sha256(timing_path),
        "checkpoint_evidence": checkpoint_evidence,
    }
    return arrays, report, evidence


def valid_output_guard_summary(
    reports: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    aggregates: dict[str, dict[str, float | int]] = {}
    for role in ROLES:
        continuations = sum(
            int(reports[seed]["generation"][role]["continuations"])
            for seed in SEEDS
        )
        valid = sum(
            int(reports[seed]["generation"][role]["valid_utf8_count"])
            for seed in SEEDS
        )
        replacement_free = sum(
            int(
                reports[seed]["generation"][role][
                    "replacement_character_free_count"
                ]
            )
            for seed in SEEDS
        )
        aggregates[role] = {
            "continuations": continuations,
            "valid_utf8_count": valid,
            "valid_utf8_rate": valid / continuations,
            "replacement_character_free_count": replacement_free,
            "replacement_character_free_rate": replacement_free / continuations,
        }
    valid_difference = (
        float(aggregates["candidate"]["valid_utf8_rate"])
        - float(aggregates["reference"]["valid_utf8_rate"])
    )
    replacement_difference = (
        float(aggregates["candidate"]["replacement_character_free_rate"])
        - float(aggregates["reference"]["replacement_character_free_rate"])
    )
    replacement_seed_differences = {
        str(seed): (
            reports[seed]["generation"]["candidate"][
                "replacement_character_free_rate"
            ]
            - reports[seed]["generation"]["reference"][
                "replacement_character_free_rate"
            ]
        )
        for seed in SEEDS
    }
    replacement_seed_count_within_margin = sum(
        value >= -0.02 for value in replacement_seed_differences.values()
    )
    all_outputs_strict_valid = all(
        values["valid_utf8_count"] == values["continuations"]
        for values in aggregates.values()
    )
    passed = bool(
        all_outputs_strict_valid
        and replacement_difference >= -0.02
        and replacement_seed_count_within_margin >= 4
    )
    return {
        "overall_pass": passed,
        "status": "pass" if passed else "fail_valid_output_guard",
        "by_role": aggregates,
        "shared_utf8_constraint": FREE_RUNNING_UTF8_CONSTRAINT,
        "all_outputs_strict_utf8_valid": all_outputs_strict_valid,
        "candidate_minus_reference_valid_utf8_rate": valid_difference,
        "candidate_minus_reference_replacement_character_free_rate": (
            replacement_difference
        ),
        "candidate_minus_reference_replacement_rate_by_seed": (
            replacement_seed_differences
        ),
        "replacement_seed_count_within_margin": (
            replacement_seed_count_within_margin
        ),
        "minimum_seed_count_within_margin": 4,
        "maximum_allowed_regression": 0.02,
    }


def _latency_summaries(
    arrays: dict[str, np.ndarray],
    *,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    bootstrap_seed = 20_260_820
    for mode in MODES:
        output[mode] = {}
        for component in COMPONENTS:
            candidate = arrays[f"{mode}__{component}__candidate"]
            reference = arrays[f"{mode}__{component}__reference"]
            output[mode][component] = multiseed_paired_latency(
                candidate,
                reference,
                SEEDS,
                bootstrap_repetitions=bootstrap_repetitions,
                bootstrap_seed=bootstrap_seed,
            ).to_dict()
            bootstrap_seed += 1
    return output


def _patch_and_memory_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in MODES:
        output[mode] = {}
        for role in ROLES:
            patches = arrays[f"{mode}__global_patches__{role}"].astype(
                np.float64
            )
            observed = arrays[
                f"{mode}__runtime_observed_bytes__{role}"
            ].astype(np.float64)
            emitted = arrays[
                f"{mode}__emitted_output_bytes__{role}"
            ].astype(np.float64)
            overshoot = arrays[f"{mode}__overshoot_bytes__{role}"].astype(
                np.float64
            )
            current = arrays[f"{mode}__mps_current_bytes__{role}"]
            driver = arrays[f"{mode}__mps_driver_bytes__{role}"]
            output[mode][role] = {
                "median_runtime_observed_bytes": float(np.median(observed)),
                "median_emitted_output_bytes": float(np.median(emitted)),
                "mean_overshoot_bytes": float(np.mean(overshoot)),
                "maximum_overshoot_bytes": int(np.max(overshoot)),
                "median_emitted_global_patches": float(np.median(patches)),
                "mean_emitted_global_patches": float(np.mean(patches)),
                "median_bytes_per_global_patch": float(
                    np.median(observed / patches)
                ),
                "maximum_mps_current_allocated_bytes": int(current.max()),
                "maximum_mps_driver_allocated_bytes": int(driver.max()),
            }
    return output


def _throughput_diagnostics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Report unit-normalized rates without replacing time-to-output gates."""

    def collapsed(values: np.ndarray) -> dict[str, float]:
        by_seed_prompt = np.median(values.astype(np.float64), axis=2)
        return {
            "median": float(np.median(by_seed_prompt)),
            "p05": float(np.percentile(by_seed_prompt, 5)),
            "p95": float(np.percentile(by_seed_prompt, 95)),
        }

    output: dict[str, Any] = {}
    for mode in MODES:
        output[mode] = {}
        for role in ROLES:
            emitted = arrays[
                f"{mode}__emitted_output_bytes__{role}"
            ].astype(np.float64)
            codepoints = arrays[f"{mode}__output_codepoints__{role}"].astype(
                np.float64
            )
            decode = arrays[f"{mode}__decode_ms__{role}"]
            end_to_end = arrays[f"{mode}__end_to_end_ms__{role}"]
            output[mode][role] = {
                "decode_ms_per_emitted_byte": collapsed(decode / emitted),
                "emitted_bytes_per_decode_second": collapsed(
                    emitted * 1_000.0 / decode
                ),
                "unicode_codepoints_per_decode_second": collapsed(
                    codepoints * 1_000.0 / decode
                ),
                "end_to_end_ms_per_minimum_valid_byte": collapsed(
                    end_to_end / CONTINUATION_BYTES
                ),
                "minimum_valid_bytes_per_end_to_end_second": collapsed(
                    CONTINUATION_BYTES * 1_000.0 / end_to_end
                ),
                "gate_role": (
                    "diagnostic_only; raw time-to-output latency remains primary"
                ),
            }
    return output


def compact_actual_inference_gate(
    quality_gate: dict[str, Any],
    latency: dict[str, Any],
    utf8: dict[str, Any],
    *,
    correctness_pass: bool,
    protocol_pass: bool,
) -> dict[str, Any]:
    controlled = multiseed_latency_component_pass(
        latency["controlled_replay"]["decode_ms"]
    )
    free_running = multiseed_latency_component_pass(
        latency["free_running_utf8_greedy"]["end_to_end_ms"]
    )
    checks = {
        "incremental_equivalence": correctness_pass,
        "five_seed_quality_noninferiority": quality_gate.get("overall_pass")
        is True,
        "controlled_replay_decode_latency": controlled,
        "free_running_end_to_end_latency": free_running,
        "free_running_valid_output_and_replacement_guard": utf8.get("overall_pass")
        is True,
        "runtime_includes_selector_router_cache_and_sync": protocol_pass,
    }
    passed = all(checks.values())
    return {
        "status": "pass" if passed else "fail_compact_actual_inference",
        "overall_pass": passed,
        "checks": checks,
        "scope": "19.6M-parameter Apple MPS compact actual-inference gate",
        "publication_final_value_gate": (
            "pending downstream noninferiority and publication-scale replication"
            if passed
            else "blocked by compact actual-inference failure"
        ),
    }


def _write_observations(
    path: Path,
    latency: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "mode",
                "component",
                "seed",
                "candidate_median_ms",
                "reference_median_ms",
                "latency_reduction",
                "bootstrap_95_lower",
                "bootstrap_95_upper",
            ),
        )
        writer.writeheader()
        for mode in MODES:
            for component in COMPONENTS:
                summary = latency[mode][component]
                for seed in SEEDS:
                    values = summary["per_seed"][str(seed)]
                    writer.writerow(
                        {
                            "mode": mode,
                            "component": component,
                            "seed": seed,
                            "candidate_median_ms": values[
                                "candidate_median_ms"
                            ],
                            "reference_median_ms": values[
                                "reference_median_ms"
                            ],
                            "latency_reduction": values[
                                "median_latency_reduction"
                            ],
                            "bootstrap_95_lower": values[
                                "bootstrap_percentile_95_lower"
                            ],
                            "bootstrap_95_upper": values[
                                "bootstrap_percentile_95_upper"
                            ],
                        }
                    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    selection_path = Path(args.selection)
    quality_path = Path(args.quality_summary)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    _validate_manifest(manifest, selection_path, quality_path)
    selection, quality, phase3 = _validate_upstream(
        manifest,
        selection_path,
        quality_path,
    )
    case_context = _reconstruct_cases(
        phase3,
        manifest,
        Path(args.data_root),
    )
    order = timing_order_schedule(
        SEEDS,
        mode_count=len(MODES),
        prompt_count=MEASURED_CASES,
        repetitions=REPETITIONS,
        random_seed=TIMING_ORDER_SEED,
    )
    seed_arrays: dict[int, dict[str, np.ndarray]] = {}
    reports: dict[int, dict[str, Any]] = {}
    evidence: dict[str, Any] = {}
    for seed_index, seed in enumerate(SEEDS):
        seed_arrays[seed], reports[seed], evidence[str(seed)] = _load_seed(
            seed,
            seed_index,
            run_root,
            artifact_root,
            selection,
            quality_path,
            selection_path,
            order,
            args,
        )
    arrays = {
        key: np.stack([seed_arrays[seed][key] for seed in SEEDS])
        for key in _expected_array_keys()
    }
    latency = _latency_summaries(
        arrays,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    utf8 = valid_output_guard_summary(reports)
    correctness_pass = all(
        reports[seed]["correctness"][role]["pass"] is True
        for seed in SEEDS
        for role in ROLES
    )
    protocol_pass = bool(
        manifest["protocol"][
            "selector_router_cache_and_synchronization_inside_timing"
        ]
        and manifest["protocol"]["parallel_prefill_only_in_timing"]
        and manifest["protocol"][
            "utf8_mask_argmax_state_and_stop_checks_inside_timing"
        ]
        and manifest["protocol"]["free_running_utf8_constraint"]
        == FREE_RUNNING_UTF8_CONSTRAINT
    )
    compact_gate = compact_actual_inference_gate(
        quality["quality_noninferiority"],
        latency,
        utf8,
        correctness_pass=correctness_pass,
        protocol_pass=protocol_pass,
    )
    output_root = Path(args.output_root)
    summary = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
        },
        "quality_summary": {
            "path": str(quality_path),
            "sha256": _sha256(quality_path),
        },
        "candidate": selection["candidate"],
        "reference": selection["reference"],
        "seeds": list(SEEDS),
        "latency": latency,
        "unit_normalized_throughput_diagnostics": _throughput_diagnostics(
            arrays
        ),
        "patch_and_memory_diagnostics": _patch_and_memory_summary(arrays),
        "free_running_valid_output_guard": utf8,
        "compact_actual_inference_gate": compact_gate,
        "integrity": {
            "all_integrity_checks_pass": True,
            "all_actual_checkpoints_and_timing_arrays_reconstructed": True,
            "all_incremental_equivalence_checks_pass": correctness_pass,
            "same_cases_crossed_over_all_seeds_and_policies": True,
            "case_context": case_context,
            "by_seed": evidence,
        },
        "interpretation_guardrail": (
            "A compact pass establishes an actual 19.6M MPS result only. "
            "Efficient-inference publication claims remain blocked until "
            "downstream and publication-scale gates pass."
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_observations(output_root / "observations.csv", latency)
    print(json.dumps(compact_gate, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="runs/phase3-actual-inference",
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-actual-inference",
    )
    parser.add_argument(
        "--selection",
        default="results/phase3-inference-selection/selection.json",
    )
    parser.add_argument(
        "--quality-summary",
        default="results/phase3-inference-quality/summary.json",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--phase3-run-root", default="runs/phase3")
    parser.add_argument("--phase3-artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--conversion-run-root",
        default="runs/phase3-compute-conversion",
    )
    parser.add_argument(
        "--conversion-artifact-root",
        default="artifacts/phase3-compute-conversion",
    )
    parser.add_argument(
        "--output-root",
        default="results/phase3-actual-inference",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
