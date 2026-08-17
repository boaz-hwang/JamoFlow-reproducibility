#!/usr/bin/env python3
"""Run Phase 2 duplicate, aligned-packing, and mechanism controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

import numpy as np
import torch

from jamoflow.neural_data import NeuralStream, build_neural_stream
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import (
    DEFAULT_OPTIMIZATION_SPEC,
    evaluate_main_model,
    resolve_device,
    shuffled_indices,
    train_main_model,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_controls import (
    aligned_pack_stream,
    offset_grid_displacements,
    trace_diagnostics,
)
from jamoflow.phase2_patching import (
    calibrate_placebo_threshold,
    causal_eojeol_grid_boundaries,
    causal_offset_grid_boundaries,
    causal_window_grid_trace,
    compact_punctuation_mask,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
    rolling_hash_event_mask,
    structural_patch_matrices,
    variable_patch_diagnostics,
)


SPLITS = ("train", "calibration", "test")
SEEDS = (1729, 2718, 31415, 57721, 65537)
ALIGNED_SEEDS = (1729, 2718, 31415)
MECHANISM_POLICIES = (
    "causal_grid_early2",
    "causal_grid_delayed2",
    "causal_placebo_grid",
    "causal_whitespace_grid",
)
PUNCTUATION_POLICY = "causal_punctuation_grid"
FULL_LIMITS = {
    "train": 11_000_000,
    "calibration": 1_000_000,
    "test": 1_000_000,
}
QUICK_LIMITS = {
    "train": 65_536,
    "calibration": 32_768,
    "test": 32_768,
}
GLOBAL_POSITION_LIMIT = DEFAULT_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
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


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _cpu_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to("cpu")
        for name, value in model.state_dict().items()
    }


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
    streams: dict[str, NeuralStream] = {}
    inputs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    punctuation: dict[str, np.ndarray] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            data_root / "ko.jsonl",
            language="ko",
            split=split,
            byte_limit=limits[split],
            sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_masks = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        shape = split_inputs.shape
        streams[split] = stream
        inputs[split] = split_inputs
        masks[split] = split_masks
        whitespace[split] = compact_whitespace_mask(stream.data).reshape(shape)
        punctuation[split] = compact_punctuation_mask(stream.data).reshape(shape)
        print(
            f"data ko/{split}: {stream.selected_bytes:,} bytes, "
            f"{stream.sequence_count:,} sequences",
            flush=True,
        )
    return streams, inputs, masks, whitespace, punctuation


def _traced_matrix(
    inputs: np.ndarray,
    masks: np.ndarray,
    event_masks: np.ndarray | Callable[[int, np.ndarray], np.ndarray],
    whitespace: np.ndarray,
    punctuation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    traces = []
    for index, row in enumerate(inputs):
        events = (
            event_masks(index, row)
            if callable(event_masks)
            else event_masks[index]
        )
        traces.append(
            causal_window_grid_trace(
                masks[index],
                events,
                DEFAULT_MODEL_SPEC.patch_count,
            )
        )
    matrix = padded_hf_patch_matrix(
        [trace.boundaries for trace in traces],
        DEFAULT_MODEL_SPEC.sequence_length,
    )
    diagnostics = {
        **trace_diagnostics(
            traces,
            whitespace_masks=whitespace,
            punctuation_masks=punctuation,
        ),
        **{
            f"patch_{key}": value
            for key, value in variable_patch_diagnostics(matrix, masks).to_dict().items()
        },
        "matrix_sha256": _array_sha256(matrix),
    }
    return matrix, diagnostics


def _offset_matrix(
    masks: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = [
        causal_offset_grid_boundaries(
            mask,
            DEFAULT_MODEL_SPEC.patch_count,
            offset=offset,
        )
        for mask in masks
    ]
    matrix = padded_hf_patch_matrix(rows, DEFAULT_MODEL_SPEC.sequence_length)
    displacements = offset_grid_displacements(
        rows,
        DEFAULT_MODEL_SPEC.sequence_length,
        DEFAULT_MODEL_SPEC.patch_count,
    ).astype(np.float64)
    return matrix, {
        **{
            f"patch_{key}": value
            for key, value in variable_patch_diagnostics(matrix, masks).to_dict().items()
        },
        "mean_target_displacement_bytes": float(displacements.mean()),
        "median_target_displacement_bytes": float(np.median(displacements)),
        "p05_target_displacement_bytes": float(np.percentile(displacements, 5)),
        "p95_target_displacement_bytes": float(np.percentile(displacements, 95)),
        "minimum_target_displacement_bytes": int(displacements.min()),
        "maximum_target_displacement_bytes": int(displacements.max()),
        "matrix_sha256": _array_sha256(matrix),
    }


def _build_mechanism_matrices(
    inputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    whitespace: dict[str, np.ndarray],
    punctuation: dict[str, np.ndarray],
    run_root: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], tuple[str, ...]]:
    delimiter = {
        split: np.maximum(whitespace[split], punctuation[split])
        for split in SPLITS
    }
    calibration_traces = [
        causal_window_grid_trace(
            mask,
            events,
            DEFAULT_MODEL_SPEC.patch_count,
        )
        for mask, events in zip(
            masks["calibration"],
            delimiter["calibration"],
            strict=True,
        )
    ]
    c2_calibration_diagnostics = trace_diagnostics(
        calibration_traces,
        whitespace_masks=whitespace["calibration"],
        punctuation_masks=punctuation["calibration"],
    )
    target_event_fraction = float(
        c2_calibration_diagnostics["event_trigger_fraction"]
    )
    placebo_calibration = calibrate_placebo_threshold(
        inputs["calibration"],
        masks["calibration"],
        target_event_fraction,
        DEFAULT_MODEL_SPEC.patch_count,
    )
    punctuation_share = float(
        c2_calibration_diagnostics.get("selected_event_punctuation_rate", 0.0)
    )
    train_punctuation = punctuation_share > 0.5
    trained_policies = (
        *MECHANISM_POLICIES,
        *((PUNCTUATION_POLICY,) if train_punctuation else ()),
    )

    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    diagnostics: dict[str, Any] = {
        "placebo_calibration": placebo_calibration.to_dict(),
        "c2_calibration_trigger_diagnostics": c2_calibration_diagnostics,
        "punctuation_training_rule": {
            "selected_event_punctuation_rate": punctuation_share,
            "threshold_strictly_above": 0.5,
            "train_punctuation_policy": train_punctuation,
        },
        "trained_policies": list(trained_policies),
        "splits": {},
    }
    for split in SPLITS:
        split_diagnostics: dict[str, Any] = {}
        matrices[split]["causal_grid_early2"], split_diagnostics[
            "causal_grid_early2"
        ] = _offset_matrix(masks[split], -2)
        matrices[split]["causal_grid_delayed2"], split_diagnostics[
            "causal_grid_delayed2"
        ] = _offset_matrix(masks[split], 2)
        matrices[split]["causal_placebo_grid"], split_diagnostics[
            "causal_placebo_grid"
        ] = _traced_matrix(
            inputs[split],
            masks[split],
            lambda _index, row: rolling_hash_event_mask(
                bytes(row),
                placebo_calibration.low_bit_threshold,
                hash_bits=placebo_calibration.hash_bits,
            ),
            whitespace[split],
            punctuation[split],
        )
        matrices[split]["causal_whitespace_grid"], split_diagnostics[
            "causal_whitespace_grid"
        ] = _traced_matrix(
            inputs[split],
            masks[split],
            whitespace[split],
            whitespace[split],
            punctuation[split],
        )
        punctuation_matrix, punctuation_diagnostics = _traced_matrix(
            inputs[split],
            masks[split],
            punctuation[split],
            whitespace[split],
            punctuation[split],
        )
        split_diagnostics[PUNCTUATION_POLICY] = punctuation_diagnostics
        if train_punctuation:
            matrices[split][PUNCTUATION_POLICY] = punctuation_matrix

        c2_matrix, c2_diagnostics = _traced_matrix(
            inputs[split],
            masks[split],
            delimiter[split],
            whitespace[split],
            punctuation[split],
        )
        split_diagnostics["causal_eojeol_grid_reference"] = c2_diagnostics
        # The reference is diagnostic only; its hash must agree with the
        # independently built primary C2 matrix when summarized.
        split_diagnostics["causal_eojeol_grid_reference"][
            "matrix_sha256"
        ] = _array_sha256(c2_matrix)
        diagnostics["splits"][split] = split_diagnostics
        print(f"constructed Phase 2b matrices for {split}", flush=True)

    _write_json(run_root / "mechanism-patch-diagnostics.json", diagnostics)
    return matrices, trained_policies


def _train_control(
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    device: str,
    run_directory: Path,
    artifact_directory: Path,
    force: bool,
) -> None:
    report_path = run_directory / f"{policy}.json"
    checkpoint_path = artifact_directory / f"{policy}.pt"
    loss_path = artifact_directory / f"{policy}-test-nll.npz"
    if report_path.exists() and checkpoint_path.exists() and loss_path.exists() and not force:
        print(f"seed {seed}/{policy}: already complete", flush=True)
        return

    order = shuffled_indices(len(inputs["train"]), seed)
    model = build_main_model(
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization_sha256 = _state_dict_sha256(model)
    print(f"seed {seed}/{policy}: training mechanism control", flush=True)
    training = train_main_model(
        model,
        inputs["train"],
        matrices["train"][policy],
        order,
        device,
        DEFAULT_OPTIMIZATION_SPEC,
    )
    evaluations: dict[str, Any] = {}
    test_nll: np.ndarray | None = None
    for split in ("calibration", "test"):
        evaluation, local_nll = evaluate_main_model(
            model,
            inputs[split],
            matrices[split][policy],
            device,
            batch_size=DEFAULT_OPTIMIZATION_SPEC.evaluation_batch_size,
            return_sequence_nll=split == "test",
        )
        evaluations[split] = evaluation.to_dict()
        if local_nll is not None:
            test_nll = local_nll
    if test_nll is None:
        raise AssertionError("test evaluation did not return sequence NLL")

    torch.save(_cpu_state_dict(model), checkpoint_path)
    np.savez_compressed(loss_path, ko=test_nll)
    _write_json(
        report_path,
        {
            "seed": seed,
            "policy": policy,
            "language": "ko",
            "parameters": parameter_count(model),
            "initialization_sha256": initialization_sha256,
            "training_order_sha256": _array_sha256(order),
            "patch_matrix_sha256": {
                split: _array_sha256(matrices[split][policy])
                for split in SPLITS
            },
            "patch_diagnostics": {
                split: variable_patch_diagnostics(
                    matrices[split][policy],
                    masks[split],
                ).to_dict()
                for split in SPLITS
            },
            "training": training.to_dict(),
            "evaluation": evaluations,
        },
    )
    print(
        f"seed {seed}/{policy}: test BPB={evaluations['test']['bpb']:.6f}",
        flush=True,
    )
    _release_model(model, device)


def _run_duplicate(
    inputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    c1_matrices: dict[str, np.ndarray],
    device: str,
    run_root: Path,
    artifact_root: Path,
    primary_run_root: Path,
    primary_artifact_root: Path,
    force: bool,
) -> None:
    report_path = run_root / "duplicate-seed-1729.json"
    checkpoint_path = artifact_root / "duplicate-seed-1729.pt"
    loss_path = artifact_root / "duplicate-seed-1729-test-nll.npz"
    if report_path.exists() and checkpoint_path.exists() and loss_path.exists() and not force:
        print("exact duplicate: already complete", flush=True)
        return

    seed = 1729
    order = shuffled_indices(len(inputs["train"]), seed)
    model = build_main_model(
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization_sha256 = _state_dict_sha256(model)
    print("training exact C1 duplicate for seed 1729", flush=True)
    training = train_main_model(
        model,
        inputs["train"],
        c1_matrices["train"],
        order,
        device,
        DEFAULT_OPTIMIZATION_SPEC,
    )
    evaluation, duplicate_nll = evaluate_main_model(
        model,
        inputs["test"],
        c1_matrices["test"],
        device,
        batch_size=DEFAULT_OPTIMIZATION_SPEC.evaluation_batch_size,
        return_sequence_nll=True,
    )
    if duplicate_nll is None:
        raise AssertionError("duplicate evaluation did not return sequence NLL")
    duplicate_state = _cpu_state_dict(model)
    original_checkpoint = (
        primary_artifact_root / "seed-1729" / "causal_codepoint_grid.pt"
    )
    original_state = torch.load(
        original_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    maximum_parameter_difference = max(
        float((duplicate_state[name] - original_state[name]).abs().max())
        for name in duplicate_state
    )
    original_report = _read_json(
        primary_run_root / "seed-1729" / "causal_codepoint_grid.json"
    )
    with np.load(
        primary_artifact_root
        / "seed-1729"
        / "causal_codepoint_grid-test-nll.npz"
    ) as archive:
        original_nll = archive["ko"].astype(np.float64)
    sequence_difference = duplicate_nll.astype(np.float64) - original_nll

    torch.save(duplicate_state, checkpoint_path)
    np.savez_compressed(loss_path, ko=duplicate_nll)
    _write_json(
        report_path,
        {
            "seed": seed,
            "policy": "causal_codepoint_grid_exact_duplicate",
            "initialization_sha256": initialization_sha256,
            "training_order_sha256": _array_sha256(order),
            "patch_matrix_sha256": {
                split: _array_sha256(c1_matrices[split]) for split in SPLITS
            },
            "training": training.to_dict(),
            "evaluation": evaluation.to_dict(),
            "comparison_to_primary": {
                "primary_test_bpb": original_report["evaluation"]["test"]["bpb"],
                "duplicate_test_bpb": evaluation.bpb,
                "test_bpb_difference": (
                    evaluation.bpb
                    - original_report["evaluation"]["test"]["bpb"]
                ),
                "maximum_parameter_absolute_difference": maximum_parameter_difference,
                "maximum_sequence_nll_absolute_difference_nats": float(
                    np.abs(sequence_difference).max()
                ),
                "mean_sequence_nll_absolute_difference_nats": float(
                    np.abs(sequence_difference).mean()
                ),
                "primary_checkpoint_sha256": _file_sha256(original_checkpoint),
            },
        },
    )
    print(
        "exact duplicate BPB difference="
        f"{evaluation.bpb - original_report['evaluation']['test']['bpb']:+.9f}",
        flush=True,
    )
    _release_model(model, device)


def _aligned_arrays(
    streams: dict[str, NeuralStream],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    inputs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, Any] = {}
    for split in SPLITS:
        packed = aligned_pack_stream(streams[split].data)
        split_inputs, split_masks = stream_arrays(
            packed.data,
            packed.codepoint_boundaries,
            packed.sequence_length,
        )
        delimiters = np.maximum(
            compact_whitespace_mask(packed.data),
            compact_punctuation_mask(packed.data),
        ).reshape(split_inputs.shape)
        structural = structural_patch_matrices(split_masks, delimiters)
        inputs[split] = split_inputs
        masks[split] = split_masks
        matrices[split] = {
            "aligned_fixed_byte_6": structural["fixed_byte_6"],
            "aligned_causal_codepoint_grid": structural[
                "causal_codepoint_grid"
            ],
        }
        metadata[split] = packed.metadata()
        if not np.all(split_masks[:, 0]):
            raise AssertionError("aligned rows must start at codepoint boundaries")
    return inputs, masks, matrices, metadata


def run(args: argparse.Namespace) -> int:
    limits = QUICK_LIMITS if args.quick else FULL_LIMITS
    seeds = (SEEDS[0],) if args.quick else SEEDS
    aligned_seeds = (ALIGNED_SEEDS[0],) if args.quick else ALIGNED_SEEDS
    device = resolve_device(args.device)
    run_root = Path(
        args.run_root
        or ("runs/phase2-controls-smoke" if args.quick else "runs/phase2-controls")
    )
    artifact_root = Path(
        args.artifact_root
        or (
            "artifacts/phase2-controls-smoke"
            if args.quick
            else "artifacts/phase2-controls"
        )
    )
    primary_run_root = Path(
        args.primary_run_root
        or ("runs/phase2-smoke" if args.quick else "runs/phase2")
    )
    primary_artifact_root = Path(
        args.primary_artifact_root
        or ("artifacts/phase2-smoke" if args.quick else "artifacts/phase2")
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    streams, inputs, masks, whitespace, punctuation = _load_streams(
        Path(args.data_root),
        limits,
    )
    delimiter = {
        split: np.maximum(whitespace[split], punctuation[split])
        for split in SPLITS
    }
    structural = {
        split: structural_patch_matrices(masks[split], delimiter[split])
        for split in SPLITS
    }
    c1_matrices = {
        split: structural[split]["causal_codepoint_grid"]
        for split in SPLITS
    }
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick_smoke_only": args.quick,
        "git_commit": _git_commit(),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "limits": limits,
        "mechanism_seeds": list(seeds),
        "aligned_seeds": list(aligned_seeds),
        "primary_run_root": str(primary_run_root),
        "primary_artifact_root": str(primary_artifact_root),
        "model_spec": DEFAULT_MODEL_SPEC.to_dict(),
        "optimization_spec": DEFAULT_OPTIMIZATION_SPEC.to_dict(),
    }

    if not args.skip_mechanism:
        mechanism, mechanism_policies = _build_mechanism_matrices(
            inputs,
            masks,
            whitespace,
            punctuation,
            run_root,
        )
        manifest["mechanism_policies"] = list(mechanism_policies)
        for seed in seeds:
            seed_run = run_root / f"mechanism-seed-{seed}"
            seed_artifact = artifact_root / f"mechanism-seed-{seed}"
            seed_run.mkdir(parents=True, exist_ok=True)
            seed_artifact.mkdir(parents=True, exist_ok=True)
            for policy in mechanism_policies:
                _train_control(
                    seed,
                    policy,
                    inputs,
                    masks,
                    mechanism,
                    device,
                    seed_run,
                    seed_artifact,
                    args.force,
                )

    if not args.skip_duplicate:
        _run_duplicate(
            inputs,
            masks,
            c1_matrices,
            device,
            run_root,
            artifact_root,
            primary_run_root,
            primary_artifact_root,
            args.force,
        )

    if not args.skip_aligned:
        aligned_inputs, aligned_masks, aligned_matrices, aligned_metadata = (
            _aligned_arrays(streams)
        )
        manifest["aligned_packing"] = aligned_metadata
        for seed in aligned_seeds:
            seed_run = run_root / f"aligned-seed-{seed}"
            seed_artifact = artifact_root / f"aligned-seed-{seed}"
            seed_run.mkdir(parents=True, exist_ok=True)
            seed_artifact.mkdir(parents=True, exist_ok=True)
            for policy in (
                "aligned_fixed_byte_6",
                "aligned_causal_codepoint_grid",
            ):
                _train_control(
                    seed,
                    policy,
                    aligned_inputs,
                    aligned_masks,
                    aligned_matrices,
                    device,
                    seed_run,
                    seed_artifact,
                    args.force,
                )

    _write_json(run_root / "manifest.json", manifest)
    print(f"completed Phase 2 controls under {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--run-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--primary-run-root")
    parser.add_argument("--primary-artifact-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-mechanism", action="store_true")
    parser.add_argument("--skip-duplicate", action="store_true")
    parser.add_argument("--skip-aligned", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
