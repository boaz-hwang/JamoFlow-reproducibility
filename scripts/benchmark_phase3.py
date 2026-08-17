#!/usr/bin/env python3
"""Benchmark preregistered Phase 3 Korean patch policies end to end."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import random
import resource
import subprocess
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from jamoflow.cost import variable_patch_flop_summary
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    build_router,
    parameter_count,
    research_versions,
)
from jamoflow.neural_patching import (
    entropy_from_logits,
    fixed_byte_boundaries,
    hf_patch_lengths,
)
from jamoflow.neural_training import resolve_device, synchronize
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_window_grid_trace,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
    STRUCTURAL_POLICIES,
    THRESHOLD_POLICIES,
    spacebyte_boundaries,
    spacebyte_causal_prefix_mask,
)


GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
SPLITS = ("train", "calibration", "test")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_verified_state(model: Any, path: Path, expected_hash: str) -> Any:
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actual_hash = _state_dict_sha256(model)
    if actual_hash != expected_hash:
        raise ValueError(f"checkpoint state hash mismatch: {path}")
    return model


def _timing_summary(values_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "repetitions": len(values),
        "median_ms": float(np.median(values)),
        "p05_ms": float(np.percentile(values, 5)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(values.mean()),
        "sample_standard_deviation_ms": float(values.std(ddof=1)),
        "measurements_ms": [float(value) for value in values],
    }


def benchmark_interleaved(
    functions: dict[str, Callable[[], Any] | Sequence[Callable[[], Any]]],
    *,
    warmup_rounds: int,
    repetitions: int,
    seed: int,
    device: str | None,
) -> dict[str, dict[str, Any]]:
    """Time methods on a shared, balanced sequence of input batches."""

    if warmup_rounds < 1 or repetitions < 1:
        raise ValueError("warmup rounds and repetitions must be positive")
    if not functions:
        raise ValueError("at least one benchmark function is required")
    batched: dict[str, tuple[Callable[[], Any], ...]] = {}
    for name, value in functions.items():
        local = (value,) if callable(value) else tuple(value)
        if not local or not all(callable(function) for function in local):
            raise ValueError(f"{name} must provide non-empty callables")
        batched[name] = local
    batch_counts = {len(local) for local in batched.values()}
    if len(batch_counts) != 1:
        raise ValueError("all methods must provide the same input batches")
    input_batch_count = batch_counts.pop()
    names = list(batched)
    rng = random.Random(seed)

    def balanced_schedule(rounds: int) -> list[int]:
        schedule: list[int] = []
        while len(schedule) < rounds:
            cycle = list(range(input_batch_count))
            rng.shuffle(cycle)
            schedule.extend(cycle)
        return schedule[:rounds]

    def sync() -> None:
        if device is not None:
            synchronize(device)

    with torch.inference_mode():
        for batch_id in balanced_schedule(warmup_rounds):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                batched[name][batch_id]()
                sync()
        measurements = {name: [] for name in names}
        measured_batch_ids = balanced_schedule(repetitions)
        for batch_id in measured_batch_ids:
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                sync()
                started = time.perf_counter_ns()
                batched[name][batch_id]()
                sync()
                measurements[name].append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
    batch_measurement_counts = {
        str(batch_id): measured_batch_ids.count(batch_id)
        for batch_id in range(input_batch_count)
    }
    output: dict[str, dict[str, Any]] = {}
    for name, values in measurements.items():
        summary = _timing_summary(values)
        summary["input_batches"] = input_batch_count
        summary["input_batch_measurement_counts"] = batch_measurement_counts
        summary["measurement_input_batch_ids"] = measured_batch_ids
        output[name] = summary
    return output


def _trim_matrix(matrix: np.ndarray) -> np.ndarray:
    used = np.flatnonzero(np.any(matrix != 0, axis=0))
    if not used.size:
        raise ValueError("patch matrix contains no positive values")
    return matrix[:, : int(used[-1]) + 1]


def _fixed_lengths(example_count: int) -> np.ndarray:
    row = np.asarray(
        hf_patch_lengths(
            fixed_byte_boundaries(
                PHASE3_MODEL_SPEC.sequence_length,
                PHASE3_MODEL_SPEC.patch_stride,
            ),
            PHASE3_MODEL_SPEC.sequence_length,
        ),
        dtype=np.uint16,
    )
    return np.broadcast_to(row, (example_count, len(row))).copy()


def _structural_selector(
    policy: str,
    boundaries: np.ndarray,
    whitespace: np.ndarray,
    spacelike: np.ndarray,
) -> np.ndarray:
    if boundaries.ndim != 2 or not len(boundaries):
        raise ValueError("boundary masks must be a non-empty matrix")
    if whitespace.shape != boundaries.shape or spacelike.shape != boundaries.shape:
        raise ValueError("all selector masks must have equal shape")
    if policy == "fixed_byte_6":
        return _fixed_lengths(len(boundaries))
    if policy == "causal_codepoint_grid":
        rows = [
            causal_codepoint_grid_boundaries(
                row, PHASE3_MODEL_SPEC.patch_count
            )
            for row in boundaries
        ]
    elif policy == "causal_whitespace_grid":
        rows = [
            causal_window_grid_trace(
                row,
                local_whitespace,
                PHASE3_MODEL_SPEC.patch_count,
            ).boundaries
            for row, local_whitespace in zip(
                boundaries, whitespace, strict=True
            )
        ]
    elif policy == "spacebyte_spacelike":
        rows = [spacebyte_boundaries(row) for row in spacelike]
    else:
        raise ValueError(f"not a structural Phase 3 policy: {policy}")
    return padded_hf_patch_matrix(rows, PHASE3_MODEL_SPEC.sequence_length)


def _aligned_entropy_scores(logits: torch.Tensor) -> np.ndarray:
    entropies = entropy_from_logits(logits)
    aligned = torch.zeros_like(entropies)
    aligned[:, 1:] = entropies[:, :-1]
    return aligned.float().cpu().numpy()


def _entropy_selector(
    scores: np.ndarray,
    threshold_nats: float,
    candidate_masks: np.ndarray | None,
) -> np.ndarray:
    return threshold_patch_matrix(
        scores,
        threshold_nats,
        candidate_masks=candidate_masks,
        maximum_patch_length=24,
    )


def _patch_counts(matrix: np.ndarray) -> np.ndarray:
    return (matrix[:, 1:] > 0).sum(axis=1).astype(np.int64)


def _benchmark_index_matrix(
    example_count: int,
    maximum_batch_size: int,
    timing_batches: int,
    seed: int,
) -> np.ndarray:
    if timing_batches < 1:
        raise ValueError("timing_batches must be positive")
    required = maximum_batch_size * timing_batches
    if not 0 < required <= example_count:
        raise ValueError("timing batches must fit the test examples")
    selected = np.random.default_rng(seed).permutation(example_count)[:required]
    return selected.reshape(timing_batches, maximum_batch_size)


def _device_patches(matrix: np.ndarray, device: str) -> torch.Tensor:
    trimmed = _trim_matrix(matrix)
    return torch.from_numpy(trimmed.astype(np.int64, copy=False)).to(device)


def _structural_pipeline(
    policy: str,
    model: Any,
    input_ids: torch.Tensor,
    boundaries: np.ndarray,
    whitespace: np.ndarray,
    spacelike: np.ndarray,
    device: str,
) -> Any:
    matrix = _structural_selector(policy, boundaries, whitespace, spacelike)
    return model(
        input_ids=input_ids,
        patch_lengths=_device_patches(matrix, device),
        use_cache=False,
    )


def _entropy_pipeline(
    policy: str,
    router: Any,
    model: Any,
    input_ids: torch.Tensor,
    boundaries: np.ndarray,
    threshold_nats: float,
    device: str,
) -> Any:
    _, _, logits = router(input_ids, patch_size=None, use_cache=False)
    scores = _aligned_entropy_scores(logits)
    candidates = boundaries if policy == "entropy_threshold_codepoint" else None
    matrix = _entropy_selector(scores, threshold_nats, candidates)
    return model(
        input_ids=input_ids,
        patch_lengths=_device_patches(matrix, device),
        use_cache=False,
    )


def _system_value(command: list[str]) -> str | None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _memory_snapshot(device: str) -> dict[str, int | str | None]:
    snapshot: dict[str, int | str | None] = {
        "ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "ru_maxrss_unit": "bytes_on_macos; kilobytes_on_linux",
    }
    if device == "mps":
        snapshot.update(
            {
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                "mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
            }
        )
    elif device.startswith("cuda"):
        snapshot.update(
            {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
            }
        )
    return snapshot


def _load_patch_matrices(
    artifact_root: Path,
    seed: int,
    example_count: int,
    quality_summary: dict[str, Any],
) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    structural_path = artifact_root / "structural-patches.npz"
    threshold_path = artifact_root / f"seed-{seed}" / "threshold-patches.npz"
    expected_structural_hash = quality_summary["integrity"][
        "structural_cache_context"
    ]["cache_artifact_sha256"]
    expected_threshold_hash = quality_summary["integrity"]["by_seed"][
        str(seed)
    ]["router_and_threshold_cache"]["threshold_cache_artifact_sha256"]
    if (
        _file_sha256(structural_path) != expected_structural_hash
        or _file_sha256(threshold_path) != expected_threshold_hash
    ):
        raise ValueError("Phase 3 cost patch cache differs from quality evidence")
    with np.load(structural_path, allow_pickle=False) as loaded:
        expected_keys = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in STRUCTURAL_POLICIES
        }
        if set(loaded.files) != expected_keys:
            raise ValueError("structural cost cache has unexpected keys")
        for policy in STRUCTURAL_POLICIES:
            matrix = loaded[f"test__{policy}"]
            if matrix.dtype != np.uint16:
                raise ValueError("structural cost matrix dtype mismatch")
            matrices[policy] = matrix.copy()
    with np.load(threshold_path, allow_pickle=False) as loaded:
        expected_keys = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in THRESHOLD_POLICIES
        }
        if set(loaded.files) != expected_keys:
            raise ValueError("threshold cost cache has unexpected keys")
        for policy in THRESHOLD_POLICIES:
            matrix = loaded[f"test__{policy}"]
            if matrix.dtype != np.uint16:
                raise ValueError("threshold cost matrix dtype mismatch")
            matrices[policy] = matrix.copy()
    for policy, matrix in matrices.items():
        if len(matrix) != example_count:
            raise ValueError(f"{policy} patch matrix has unexpected row count")
        validate_padded_patch_matrix(
            matrix, PHASE3_MODEL_SPEC.sequence_length
        )
    return matrices


def run(args: argparse.Namespace) -> int:
    if args.repetitions < 30:
        raise ValueError("Phase 3 benchmark requires at least 30 repetitions")
    if args.timing_batches < 8:
        raise ValueError("Phase 3 benchmark requires at least 8 timing batches")
    if len(set(args.batch_size)) != len(args.batch_size):
        raise ValueError("batch sizes must be unique")
    if any(batch_size <= 0 for batch_size in args.batch_size):
        raise ValueError("batch sizes must be positive")

    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    quality_path = Path(args.quality_summary)
    quality_summary = _read_json(quality_path)
    if (
        quality_summary.get("integrity", {}).get("all_integrity_checks_pass")
        is not True
        or set(quality_summary.get("policies", [])) != set(PHASE3_POLICIES)
        or int(args.seed) not in quality_summary.get("seeds", [])
    ):
        raise ValueError("cost benchmark needs complete shared-seed quality evidence")
    quality_integrity = quality_summary["integrity"]
    quality_lineage = quality_integrity["by_seed"][str(args.seed)]
    threshold_lineage = quality_lineage["router_and_threshold_cache"]
    expected_structural_hash = quality_integrity["structural_cache_context"][
        "cache_artifact_sha256"
    ]
    expected_threshold_hash = threshold_lineage[
        "threshold_cache_artifact_sha256"
    ]
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("quick_smoke_only"):
        raise ValueError("smoke runs cannot support the confirmatory cost benchmark")
    seed = int(args.seed)
    if seed not in manifest["seeds"]:
        raise ValueError(f"seed {seed} is not in the Phase 3 manifest")
    device = resolve_device(args.device)

    quality_manifest = quality_summary.get("run_manifest", {})
    for key in (
        "quick_smoke_only",
        "language",
        "limits",
        "source_artifact",
        "source_integrity_artifact",
        "global_max_position_embeddings",
        "model_spec",
        "optimization_spec",
        "streams",
    ):
        if manifest.get(key) != quality_manifest.get(key):
            raise ValueError(f"cost/quality manifest invariant mismatch: {key}")
    if (
        manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or manifest.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or manifest.get("global_max_position_embeddings")
        != GLOBAL_POSITION_LIMIT
    ):
        raise ValueError("cost benchmark model design mismatch")
    source_path = Path(args.data_root) / "ko.jsonl"
    integrity_path = Path(args.data_root) / "integrity.json"
    expected_source_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _file_sha256(source_path),
    }
    expected_integrity_artifact = {
        "filename": "integrity.json",
        "bytes": integrity_path.stat().st_size,
        "sha256": _file_sha256(integrity_path),
    }
    if (
        manifest.get("source_artifact") != expected_source_artifact
        or manifest.get("source_integrity_artifact")
        != expected_integrity_artifact
    ):
        raise ValueError("cost benchmark source artifact mismatch")

    stream = build_neural_stream(
        Path(args.data_root) / "ko.jsonl",
        language="ko",
        split="test",
        byte_limit=int(manifest["limits"]["test"]),
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    stream_sha256 = hashlib.sha256(stream.data).hexdigest()
    expected_stream_sha256 = manifest["streams"]["test"][
        "selected_stream_sha256"
    ]
    if stream_sha256 != expected_stream_sha256:
        raise ValueError("Phase 3 test stream hash differs from the run manifest")
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    matrices = _load_patch_matrices(
        artifact_root,
        seed,
        len(inputs),
        quality_summary,
    )

    seed_run = run_root / f"seed-{seed}"
    seed_artifact = artifact_root / f"seed-{seed}"
    reports = {
        policy: _read_json(seed_run / f"{policy}.json")
        for policy in PHASE3_POLICIES
    }
    models: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    for policy in PHASE3_POLICIES:
        report = reports[policy]
        report_path = seed_run / f"{policy}.json"
        checkpoint_path = seed_artifact / f"{policy}.pt"
        if (
            report.get("seed") != seed
            or report.get("policy") != policy
            or report.get("parameters") != 19_596_096
            or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
            or report.get("optimization_spec")
            != PHASE3_OPTIMIZATION_SPEC.to_dict()
            or _file_sha256(report_path)
            != quality_lineage["training_report_artifact_sha256"][policy]
            or _file_sha256(checkpoint_path)
            != quality_lineage["checkpoint_artifact_sha256"][policy]
        ):
            raise ValueError(f"cost checkpoint lineage mismatch: {policy}")
        expected_matrix_hash = report["patch_matrix_sha256"]["test"]
        if _array_sha256(matrices[policy]) != expected_matrix_hash:
            raise ValueError(f"{policy} test patch matrix hash mismatch")
        model = build_main_model(
            PHASE3_MODEL_SPEC,
            seed=seed,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        expected_state_hash = report["trained_state_sha256"]
        _load_verified_state(
            model,
            checkpoint_path,
            expected_state_hash,
        )
        if expected_state_hash != quality_lineage["checkpoint_state_sha256"][
            policy
        ]:
            raise ValueError(f"cost checkpoint state differs from quality: {policy}")
        checkpoint_hashes[policy] = expected_state_hash
        models[policy] = model.to(device).eval()

    router_report = _read_json(seed_run / "router.json")
    expected_router_hash = router_report.get("trained_state_sha256")
    if expected_router_hash is None:
        raise ValueError("router report lacks trained-state provenance")
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    _load_verified_state(
        router,
        seed_artifact / "router.pt",
        expected_router_hash,
    )
    if (
        expected_router_hash
        != threshold_lineage["router_checkpoint_state_sha256"]
        or _file_sha256(seed_artifact / "router.pt")
        != threshold_lineage["router_checkpoint_artifact_sha256"]
        or _file_sha256(seed_run / "router.json")
        != threshold_lineage["router_report_artifact_sha256"]
        or _file_sha256(seed_run / "threshold-patch-diagnostics.json")
        != threshold_lineage["threshold_diagnostics_artifact_sha256"]
    ):
        raise ValueError("cost router lineage differs from quality evidence")
    router.to(device).eval()
    thresholds_report = _read_json(seed_run / "threshold-patch-diagnostics.json")
    thresholds = {
        policy: float(
            thresholds_report["calibration"][policy]["threshold_nats"]
        )
        for policy in THRESHOLD_POLICIES
    }
    memory_after_load = _memory_snapshot(device)

    by_batch: dict[str, Any] = {}
    analytical: dict[str, Any] = {}
    selector_integrity: dict[str, Any] = {}
    selection_seed = seed + 30_000
    benchmark_index_matrix = _benchmark_index_matrix(
        len(inputs),
        max(args.batch_size),
        args.timing_batches,
        selection_seed,
    )
    for batch_size in args.batch_size:
        payloads: list[dict[str, Any]] = []
        selector_integrity[str(batch_size)] = {}
        for timing_batch_id, timing_row in enumerate(benchmark_index_matrix):
            local_indices = timing_row[:batch_size]
            local_inputs = inputs[local_indices]
            local_boundaries = boundaries[local_indices]
            local_whitespace = whitespace[local_indices]
            local_spacelike = spacelike[local_indices]
            input_ids = torch.from_numpy(
                local_inputs.astype(np.int64, copy=False)
            ).to(device)
            cached_local = {
                policy: _trim_matrix(matrix[local_indices])
                for policy, matrix in matrices.items()
            }
            device_patches = {
                policy: _device_patches(matrix, device)
                for policy, matrix in cached_local.items()
            }

            structural_selected = {
                policy: _structural_selector(
                    policy,
                    local_boundaries,
                    local_whitespace,
                    local_spacelike,
                )
                for policy in STRUCTURAL_POLICIES
            }
            with torch.inference_mode():
                _, _, selector_logits = router(
                    input_ids, patch_size=None, use_cache=False
                )
                synchronize(device)
                selector_scores = _aligned_entropy_scores(selector_logits)
            threshold_selected = {
                policy: _entropy_selector(
                    selector_scores,
                    thresholds[policy],
                    (
                        local_boundaries
                        if policy == "entropy_threshold_codepoint"
                        else None
                    ),
                )
                for policy in THRESHOLD_POLICIES
            }
            selected = {**structural_selected, **threshold_selected}
            local_integrity = {
                policy: {
                    "matches_cached_evaluation_matrix": bool(
                        np.array_equal(
                            _trim_matrix(selected[policy]),
                            cached_local[policy],
                        )
                    ),
                    "selected_matrix_sha256": _array_sha256(
                        _trim_matrix(selected[policy])
                    ),
                    "cached_matrix_sha256": _array_sha256(
                        cached_local[policy]
                    ),
                }
                for policy in PHASE3_POLICIES
            }
            selector_integrity[str(batch_size)][str(timing_batch_id)] = (
                local_integrity
            )
            mismatches = [
                policy
                for policy, values in local_integrity.items()
                if not values["matches_cached_evaluation_matrix"]
            ]
            if mismatches:
                raise ValueError(
                    f"batch size {batch_size}, timing batch {timing_batch_id} "
                    f"selectors differ from cached matrices: {mismatches}"
                )
            payloads.append(
                {
                    "input_ids": input_ids,
                    "boundaries": local_boundaries,
                    "whitespace": local_whitespace,
                    "spacelike": local_spacelike,
                    "selector_scores": selector_scores,
                    "device_patches": device_patches,
                }
            )

        main_functions = {
            f"main_only/{policy}": [
                (
                    lambda policy=policy, payload=payload: models[policy](
                        input_ids=payload["input_ids"],
                        patch_lengths=payload["device_patches"][policy],
                        use_cache=False,
                    )
                )
                for payload in payloads
            ]
            for policy in PHASE3_POLICIES
        }
        component_functions: dict[
            str,
            Sequence[Callable[[], Any]],
        ] = {
            **main_functions,
            "router_only": [
                (
                    lambda payload=payload: router(
                        payload["input_ids"], patch_size=None, use_cache=False
                    )
                )
                for payload in payloads
            ],
        }
        component_timings = benchmark_interleaved(
            component_functions,
            warmup_rounds=args.warmup_rounds,
            repetitions=args.repetitions,
            seed=seed + batch_size,
            device=device,
        )

        selector_functions: dict[
            str,
            Sequence[Callable[[], Any]],
        ] = {
            f"selector_only/{policy}": [
                (
                    lambda policy=policy, payload=payload: _structural_selector(
                        policy,
                        payload["boundaries"],
                        payload["whitespace"],
                        payload["spacelike"],
                    )
                )
                for payload in payloads
            ]
            for policy in STRUCTURAL_POLICIES
        }
        selector_functions.update(
            {
                f"selector_only/{policy}": [
                    (
                        lambda policy=policy, payload=payload: _entropy_selector(
                            payload["selector_scores"],
                            thresholds[policy],
                            (
                                payload["boundaries"]
                                if policy == "entropy_threshold_codepoint"
                                else None
                            ),
                        )
                    )
                    for payload in payloads
                ]
                for policy in THRESHOLD_POLICIES
            }
        )
        selector_timings = benchmark_interleaved(
            selector_functions,
            warmup_rounds=args.warmup_rounds,
            repetitions=args.repetitions,
            seed=seed + 10_000 + batch_size,
            device=None,
        )

        pipeline_functions: dict[
            str,
            Sequence[Callable[[], Any]],
        ] = {
            "end_to_end/fixed_byte_6": main_functions["main_only/fixed_byte_6"]
        }
        for policy in (
            "causal_codepoint_grid",
            "causal_whitespace_grid",
            "spacebyte_spacelike",
        ):
            pipeline_functions[f"end_to_end/{policy}"] = [
                (
                    lambda policy=policy, payload=payload: _structural_pipeline(
                        policy,
                        models[policy],
                        payload["input_ids"],
                        payload["boundaries"],
                        payload["whitespace"],
                        payload["spacelike"],
                        device,
                    )
                )
                for payload in payloads
            ]
        for policy in THRESHOLD_POLICIES:
            pipeline_functions[f"end_to_end/{policy}"] = [
                (
                    lambda policy=policy, payload=payload: _entropy_pipeline(
                        policy,
                        router,
                        models[policy],
                        payload["input_ids"],
                        payload["boundaries"],
                        thresholds[policy],
                        device,
                    )
                )
                for payload in payloads
            ]
        pipeline_timings = benchmark_interleaved(
            pipeline_functions,
            warmup_rounds=args.warmup_rounds,
            repetitions=args.repetitions,
            seed=seed + 20_000 + batch_size,
            device=device,
        )
        byte_count = batch_size * PHASE3_MODEL_SPEC.sequence_length
        by_batch[str(batch_size)] = {
            "component_timings": component_timings,
            "selector_only_cpu_timings": selector_timings,
            "direct_pipeline_timings": pipeline_timings,
            "derived_throughput_input_bytes_per_second": {
                name: byte_count / (values["median_ms"] / 1000)
                for name, values in pipeline_timings.items()
            },
        }
        analytical[str(batch_size)] = {
            policy: variable_patch_flop_summary(
                _patch_counts(matrix),
                batch_size=batch_size,
                include_router=policy in THRESHOLD_POLICIES,
                spec=PHASE3_MODEL_SPEC,
            )
            for policy, matrix in matrices.items()
        }
        print(f"benchmarked Phase 3 batch size {batch_size}", flush=True)

    comparisons: dict[str, Any] = {}
    for learned_policy in THRESHOLD_POLICIES:
        comparisons[learned_policy] = {}
        for candidate in STRUCTURAL_POLICIES:
            comparisons[learned_policy][candidate] = {
                "ideal_unpadded_flop_reduction": 1
                - analytical["1"][candidate][
                    "ideal_unpadded_mean_flops_per_sequence"
                ]
                / analytical["1"][learned_policy][
                    "ideal_unpadded_mean_flops_per_sequence"
                ],
                "implemented_batch_max_flop_reduction": {
                    batch: 1
                    - analytical[batch][candidate][
                        "implemented_batch_max_mean_flops_per_sequence"
                    ]
                    / analytical[batch][learned_policy][
                        "implemented_batch_max_mean_flops_per_sequence"
                    ]
                    for batch in analytical
                },
                "direct_latency_reduction": {
                    batch: 1
                    - by_batch[batch]["direct_pipeline_timings"][
                        f"end_to_end/{candidate}"
                    ]["median_ms"]
                    / by_batch[batch]["direct_pipeline_timings"][
                        f"end_to_end/{learned_policy}"
                    ]["median_ms"]
                    for batch in by_batch
                },
            }

    main_parameters = parameter_count(models["fixed_byte_6"])
    router_parameters = parameter_count(router)
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_git_commit": _git_commit(),
        "scope": (
            "Teacher-forced 512-byte Korean windows; direct learned-policy "
            "timings include router, entropy, device-to-host transfer, CPU "
            "selector, patch upload, and main model; not incremental generation"
        ),
        "seed": seed,
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "mac_hardware_model": _system_value(["sysctl", "-n", "hw.model"]),
        "versions": research_versions(),
        "parameters": {
            "main": main_parameters,
            "router": router_parameters,
            "main_parameter_bytes_fp32": main_parameters * 4,
            "router_parameter_bytes_fp32": router_parameters * 4,
            "per_policy_resident_parameter_bytes_fp32": {
                policy: (main_parameters + (
                    router_parameters if policy in THRESHOLD_POLICIES else 0
                )) * 4
                for policy in PHASE3_POLICIES
            },
        },
        "protocol": {
            "language": "ko",
            "sequence_input_bytes": PHASE3_MODEL_SPEC.sequence_length,
            "predicted_bytes_per_sequence": PHASE3_MODEL_SPEC.sequence_length - 1,
            "batch_sizes": args.batch_size,
            "timing_batches_per_batch_size": args.timing_batches,
            "warmup_rounds": args.warmup_rounds,
            "repetitions": args.repetitions,
            "randomized_interleaving": True,
            "balanced_shared_input_batch_schedule": True,
            "device_synchronization_around_each_device_measurement": True,
            "selector_only_timing_excludes_device_synchronization": True,
            "inputs_preloaded_on_device": True,
            "threshold_nats_from_calibration": thresholds,
            "timing_batch_selection": (
                "One seeded random permutation of held-out sequence indices is "
                "reshaped into disjoint timing batches. Every policy uses the "
                "same balanced batch schedule; each batch size uses the nested "
                "prefix of every timing-batch row."
            ),
            "timing_batch_selection_seed": selection_seed,
            "timing_sequence_indices_sha256": _array_sha256(
                benchmark_index_matrix.astype(np.int64, copy=False)
            ),
            "cost_comparison": (
                "Choose min(E, EC) by preregistered mean quality, then compare "
                "W against that learned policy without reselecting by cost."
            ),
        },
        "integrity": {
            "shared_seed_quality_summary_path": str(quality_path),
            "shared_seed_quality_summary_sha256": _file_sha256(quality_path),
            "run_manifest_sha256": _file_sha256(manifest_path),
            "source_artifact": expected_source_artifact,
            "source_integrity_artifact": expected_integrity_artifact,
            "test_stream_sha256": stream_sha256,
            "checkpoint_state_sha256": checkpoint_hashes,
            "checkpoint_artifact_sha256": quality_lineage[
                "checkpoint_artifact_sha256"
            ],
            "training_report_artifact_sha256": quality_lineage[
                "training_report_artifact_sha256"
            ],
            "router_state_sha256": expected_router_hash,
            "router_and_threshold_cache": threshold_lineage,
            "structural_cache_artifact_sha256": expected_structural_hash,
            "threshold_cache_artifact_sha256": expected_threshold_hash,
            "selector_reconstruction": selector_integrity,
            "all_selector_reconstructions_match": all(
                values[policy]["matches_cached_evaluation_matrix"]
                for batch_size_values in selector_integrity.values()
                for values in batch_size_values.values()
                for policy in PHASE3_POLICIES
            ),
        },
        "analytical_flops": analytical,
        "measurements": by_batch,
        "comparisons_vs_learned_router": comparisons,
        "memory_after_all_comparison_models_loaded": memory_after_load,
        "memory_after_benchmark": _memory_snapshot(device),
        "limitations": [
            "MPS measurements do not predict CUDA serving latency.",
            "Teacher-forced windows do not measure sequential autoregressive decoding.",
            (
                "The measured allocator footprint includes all six comparison "
                "models; per-method parameter bytes are reported separately."
            ),
            (
                "Structural prefix masks are treated as streaming parser state "
                "already available before each byte."
            ),
            "Selector CPU timings exclude mask construction from raw UTF-8 bytes.",
            (
                "Timing p95 is over device/runtime repetitions on a fixed seeded "
                "set of held-out batches; it is not a population input-latency "
                "quantile."
            ),
            "Analytical FLOPs omit non-matmul operations explicitly listed in jamoflow.cost.",
        ],
    }
    _write_json(Path(args.output), output)
    print(f"wrote Phase 3 cost benchmark to {args.output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase3")
    parser.add_argument("--artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--quality-summary",
        default="results/phase3-all-policies/summary.json",
    )
    parser.add_argument(
        "--data-root", default="data/processed/hplt3-korean-phase3"
    )
    parser.add_argument("--output", default="runs/phase3/cost-benchmark.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument("--timing-batches", type=int, default=8)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.batch_size is None:
        parsed.batch_size = [1, 8, 32, 64]
    raise SystemExit(run(parsed))
