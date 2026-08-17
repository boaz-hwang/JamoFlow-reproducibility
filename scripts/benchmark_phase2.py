#!/usr/bin/env python3
"""Benchmark padding-aware Phase 2 Korean patching cost."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import random
import resource
import subprocess
import time
from typing import Any, Callable

import numpy as np
import torch

from jamoflow.cost import variable_patch_flop_summary
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    build_router,
    parameter_count,
    research_versions,
)
from jamoflow.neural_patching import entropy_from_logits
from jamoflow.neural_training import resolve_device, synchronize
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_eojeol_grid_boundaries,
    compact_punctuation_mask,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
    threshold_patch_matrix,
)


POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_eojeol_grid",
    "causal_whitespace_grid",
    "entropy_threshold_full",
)
GLOBAL_POSITION_LIMIT = DEFAULT_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_state(model: Any, path: Path) -> Any:
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _timing_summary(values_ms: list[float]) -> dict[str, Any]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "repetitions": len(values),
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "mean_ms": float(values.mean()),
        "sample_standard_deviation_ms": float(values.std(ddof=1)),
        "measurements_ms": [float(value) for value in values],
    }


def benchmark_interleaved(
    functions: dict[str, Callable[[], Any]],
    device: str,
    warmup_rounds: int,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    if warmup_rounds < 1 or repetitions < 1:
        raise ValueError("warmup rounds and repetitions must be positive")
    names = list(functions)
    rng = random.Random(seed)
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                functions[name]()
                synchronize(device)
        measurements = {name: [] for name in names}
        for _ in range(repetitions):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                synchronize(device)
                started = time.perf_counter_ns()
                functions[name]()
                synchronize(device)
                measurements[name].append(
                    (time.perf_counter_ns() - started) / 1_000_000
                )
    return {
        name: _timing_summary(values)
        for name, values in measurements.items()
    }


def _trim_matrix(matrix: np.ndarray) -> np.ndarray:
    used = np.flatnonzero(np.any(matrix != 0, axis=0))
    if not used.size:
        raise ValueError("patch matrix contains no positive values")
    return matrix[:, : int(used[-1]) + 1]


def _fixed_lengths(example_count: int) -> np.ndarray:
    row = np.asarray(
        [1, *([6] * 42), 4],
        dtype=np.uint16,
    )
    return np.broadcast_to(row, (example_count, len(row))).copy()


def _codepoint_lengths(masks: np.ndarray) -> np.ndarray:
    return padded_hf_patch_matrix(
        [
            causal_codepoint_grid_boundaries(mask, DEFAULT_MODEL_SPEC.patch_count)
            for mask in masks
        ],
        DEFAULT_MODEL_SPEC.sequence_length,
    )


def _event_lengths(masks: np.ndarray, events: np.ndarray) -> np.ndarray:
    return padded_hf_patch_matrix(
        [
            causal_eojeol_grid_boundaries(
                mask,
                local_events,
                DEFAULT_MODEL_SPEC.patch_count,
            )
            for mask, local_events in zip(masks, events, strict=True)
        ],
        DEFAULT_MODEL_SPEC.sequence_length,
    )


def _patch_counts(matrix: np.ndarray) -> np.ndarray:
    return (matrix[:, 1:] > 0).sum(axis=1).astype(np.int64)


def _device_patches(matrix: np.ndarray, device: str) -> torch.Tensor:
    trimmed = _trim_matrix(matrix)
    return torch.from_numpy(trimmed.astype(np.int64, copy=False)).to(device)


def _structural_pipeline(
    model: Any,
    input_ids: torch.Tensor,
    masks: np.ndarray,
    events: np.ndarray | None,
    device: str,
) -> None:
    matrix = _codepoint_lengths(masks) if events is None else _event_lengths(masks, events)
    model(
        input_ids=input_ids,
        patch_lengths=_device_patches(matrix, device),
        use_cache=False,
    )


def _entropy_pipeline(
    router: Any,
    model: Any,
    input_ids: torch.Tensor,
    threshold_nats: float,
    device: str,
) -> None:
    _, _, logits = router(input_ids, patch_size=None, use_cache=False)
    entropies = entropy_from_logits(logits)
    aligned = torch.zeros_like(entropies)
    aligned[:, 1:] = entropies[:, :-1]
    scores = aligned.float().cpu().numpy()
    matrix = threshold_patch_matrix(
        scores,
        threshold_nats,
        maximum_patch_length=24,
    )
    model(
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
    return snapshot


def run(args: argparse.Namespace) -> int:
    if args.repetitions < 30:
        raise ValueError("benchmark requires at least 30 repetitions")
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    control_artifact_root = Path(args.control_artifact_root)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    seed = int(args.seed or manifest["seeds"][0])
    if seed not in manifest["seeds"]:
        raise ValueError(f"seed {seed} is not in the Phase 2 manifest")
    device = resolve_device(args.device)

    stream = build_neural_stream(
        Path(args.data_root) / "ko.jsonl",
        language="ko",
        split="test",
        byte_limit=int(manifest["limits"]["test"]),
        sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
    )
    inputs, masks = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    punctuation = compact_punctuation_mask(stream.data).reshape(inputs.shape)
    delimiter = np.maximum(whitespace, punctuation)
    seed_artifact = artifact_root / f"seed-{seed}"
    control_seed_artifact = control_artifact_root / f"mechanism-seed-{seed}"

    router = _load_state(build_router(seed=seed), seed_artifact / "router.pt")
    router.to(device).eval()
    checkpoint_paths = {
        "fixed_byte_6": seed_artifact / "fixed_byte_6.pt",
        "causal_codepoint_grid": seed_artifact / "causal_codepoint_grid.pt",
        "causal_eojeol_grid": seed_artifact / "causal_eojeol_grid.pt",
        "causal_whitespace_grid": control_seed_artifact / "causal_whitespace_grid.pt",
        "entropy_threshold_full": seed_artifact / "entropy_threshold_full.pt",
    }
    models = {
        policy: _load_state(
            build_main_model(
                seed=seed,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            ),
            path,
        ).to(device).eval()
        for policy, path in checkpoint_paths.items()
    }
    threshold_report = json.loads(
        (run_root / f"seed-{seed}" / "threshold-patch-diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    threshold_nats = float(
        threshold_report["calibration"]["entropy_threshold_full"]["threshold_nats"]
    )
    with np.load(seed_artifact / "threshold-patches.npz") as cache:
        entropy_matrix = cache["test__entropy_threshold_full"].astype(np.uint16)
    full_matrices = {
        "fixed_byte_6": _fixed_lengths(len(inputs)),
        "causal_codepoint_grid": _codepoint_lengths(masks),
        "causal_eojeol_grid": _event_lengths(masks, delimiter),
        "causal_whitespace_grid": _event_lengths(masks, whitespace),
        "entropy_threshold_full": entropy_matrix,
    }
    memory_after_load = _memory_snapshot(device)

    by_batch: dict[str, Any] = {}
    analytical: dict[str, Any] = {}
    for batch_size in args.batch_size:
        if batch_size > len(inputs):
            raise ValueError(f"batch size {batch_size} exceeds test sequences")
        local_inputs = inputs[:batch_size]
        local_masks = masks[:batch_size]
        local_whitespace = whitespace[:batch_size]
        local_delimiter = delimiter[:batch_size]
        input_ids = torch.from_numpy(
            local_inputs.astype(np.int64, copy=False)
        ).to(device)
        device_patches = {
            policy: _device_patches(matrix[:batch_size], device)
            for policy, matrix in full_matrices.items()
        }
        main_functions = {
            f"main_only/{policy}": (
                lambda policy=policy: models[policy](
                    input_ids=input_ids,
                    patch_lengths=device_patches[policy],
                    use_cache=False,
                )
            )
            for policy in POLICIES
        }
        component_functions: dict[str, Callable[[], Any]] = {
            **main_functions,
            "router_only": lambda: router(
                input_ids,
                patch_size=None,
                use_cache=False,
            ),
        }
        component_timings = benchmark_interleaved(
            component_functions,
            device,
            args.warmup_rounds,
            args.repetitions,
            seed + batch_size,
        )
        pipeline_functions: dict[str, Callable[[], Any]] = {
            "end_to_end/fixed_byte_6": main_functions["main_only/fixed_byte_6"],
            "end_to_end/causal_codepoint_grid": lambda: _structural_pipeline(
                models["causal_codepoint_grid"],
                input_ids,
                local_masks,
                None,
                device,
            ),
            "end_to_end/causal_eojeol_grid": lambda: _structural_pipeline(
                models["causal_eojeol_grid"],
                input_ids,
                local_masks,
                local_delimiter,
                device,
            ),
            "end_to_end/causal_whitespace_grid": lambda: _structural_pipeline(
                models["causal_whitespace_grid"],
                input_ids,
                local_masks,
                local_whitespace,
                device,
            ),
            "end_to_end/entropy_threshold_full": lambda: _entropy_pipeline(
                router,
                models["entropy_threshold_full"],
                input_ids,
                threshold_nats,
                device,
            ),
        }
        pipeline_timings = benchmark_interleaved(
            pipeline_functions,
            device,
            args.warmup_rounds,
            args.repetitions,
            seed + 10_000 + batch_size,
        )
        byte_count = batch_size * DEFAULT_MODEL_SPEC.sequence_length
        by_batch[str(batch_size)] = {
            "component_timings": component_timings,
            "direct_pipeline_timings": pipeline_timings,
            "derived_throughput_bytes_per_second": {
                name: byte_count / (values["median_ms"] / 1000)
                for name, values in pipeline_timings.items()
            },
        }
        analytical[str(batch_size)] = {
            policy: variable_patch_flop_summary(
                _patch_counts(matrix),
                batch_size=batch_size,
                include_router=policy == "entropy_threshold_full",
            )
            for policy, matrix in full_matrices.items()
        }
        print(f"benchmarked Phase 2 batch size {batch_size}", flush=True)

    comparisons: dict[str, Any] = {}
    for candidate in ("causal_codepoint_grid", "causal_eojeol_grid", "causal_whitespace_grid"):
        candidate_latency = by_batch["1"]["direct_pipeline_timings"][
            f"end_to_end/{candidate}"
        ]["median_ms"]
        entropy_latency = by_batch["1"]["direct_pipeline_timings"][
            "end_to_end/entropy_threshold_full"
        ]["median_ms"]
        candidate_ideal = analytical["1"][candidate][
            "ideal_unpadded_mean_flops_per_sequence"
        ]
        entropy_ideal = analytical["1"]["entropy_threshold_full"][
            "ideal_unpadded_mean_flops_per_sequence"
        ]
        comparisons[candidate] = {
            "ideal_unpadded_flop_reduction_vs_entropy_full": (
                1 - candidate_ideal / entropy_ideal
            ),
            "batch1_direct_latency_reduction_vs_entropy_full": (
                1 - candidate_latency / entropy_latency
            ),
            "implemented_batch_max_flop_reduction_vs_entropy_full": {
                batch: 1
                - analytical[batch][candidate][
                    "implemented_batch_max_mean_flops_per_sequence"
                ]
                / analytical[batch]["entropy_threshold_full"][
                    "implemented_batch_max_mean_flops_per_sequence"
                ]
                for batch in analytical
            },
        }

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Teacher-forced 256-byte Korean windows; selector/router/transfer/upload "
            "included in direct pipeline; not incremental generation latency"
        ),
        "seed": seed,
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "mac_hardware_model": _system_value(["sysctl", "-n", "hw.model"]),
        "versions": research_versions(),
        "parameters": {
            "main": parameter_count(models["fixed_byte_6"]),
            "router": parameter_count(router),
            "entropy_end_to_end": (
                parameter_count(models["fixed_byte_6"]) + parameter_count(router)
            ),
        },
        "protocol": {
            "language": "ko",
            "sequence_bytes": DEFAULT_MODEL_SPEC.sequence_length,
            "batch_sizes": args.batch_size,
            "warmup_rounds": args.warmup_rounds,
            "repetitions": args.repetitions,
            "randomized_interleaving": True,
            "device_synchronization_around_each_measurement": True,
            "inputs_preloaded_on_device": True,
            "threshold_nats_from_calibration": threshold_nats,
        },
        "analytical_flops": analytical,
        "measurements": by_batch,
        "candidate_comparisons_vs_entropy_full": comparisons,
        "memory_after_model_load": memory_after_load,
        "memory_after_benchmark": _memory_snapshot(device),
        "limitations": [
            "MPS measurements do not predict CUDA serving latency.",
            "Teacher-forced windows do not measure sequential autoregressive decoding.",
            "Candidate masks represent parser state already available to a streaming decoder.",
            "The entropy path includes the preregistered MPS-to-CPU score transfer and Python/NumPy threshold selector.",
            "Analytical FLOPs omit non-matmul operations explicitly listed in jamoflow.cost.",
        ],
    }
    _write_json(Path(args.output), output)
    print(f"wrote Phase 2 cost benchmark to {args.output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase2")
    parser.add_argument("--artifact-root", default="artifacts/phase2")
    parser.add_argument("--control-artifact-root", default="artifacts/phase2-controls")
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--output", default="runs/phase2/cost-benchmark.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, action="append", default=None)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.batch_size is None:
        parsed.batch_size = [1, 8, 64]
    raise SystemExit(run(parsed))
