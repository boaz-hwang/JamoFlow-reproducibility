#!/usr/bin/env python3
"""Measure Phase 1 BLT, router, boundary, and end-to-end inference cost."""

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

from jamoflow.cost import end_to_end_flop_summary
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    build_router,
    parameter_count,
    research_versions,
)
from jamoflow.neural_patching import (
    entropy_boundaries,
    fixed_byte_boundaries,
    fixed_codepoint_boundaries,
    hf_patch_lengths,
)
from jamoflow.neural_training import resolve_device, synchronize
from jamoflow.phase1 import POLICIES, stream_arrays


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
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                measurements[name].append(elapsed_ms)
    return {
        name: _timing_summary(values)
        for name, values in measurements.items()
    }


def _entropy_lengths(
    scores: np.ndarray,
    masks: np.ndarray,
    codepoint_only: bool,
) -> np.ndarray:
    output = np.empty(
        (len(scores), DEFAULT_MODEL_SPEC.patch_count + 1),
        dtype=np.int64,
    )
    for index, row in enumerate(scores):
        boundaries = entropy_boundaries(
            row,
            DEFAULT_MODEL_SPEC.patch_count,
            candidate_mask=masks[index] if codepoint_only else None,
        )
        output[index] = hf_patch_lengths(
            boundaries,
            DEFAULT_MODEL_SPEC.sequence_length,
        )
    return output


def _fixed_codepoint_lengths(masks: np.ndarray) -> np.ndarray:
    output = np.empty(
        (len(masks), DEFAULT_MODEL_SPEC.patch_count + 1),
        dtype=np.int64,
    )
    for index, mask in enumerate(masks):
        output[index] = hf_patch_lengths(
            fixed_codepoint_boundaries(mask, DEFAULT_MODEL_SPEC.patch_count),
            DEFAULT_MODEL_SPEC.sequence_length,
        )
    return output


def _entropy_pipeline(
    router: Any,
    model: Any,
    input_ids: torch.Tensor,
    masks: np.ndarray,
    device: str,
    codepoint_only: bool,
) -> None:
    prediction_entropies, _, _ = router(
        input_ids,
        patch_size=None,
        use_cache=False,
    )
    aligned = torch.zeros_like(prediction_entropies)
    aligned[:, 1:] = prediction_entropies[:, :-1]
    # The primary matched-rate implementation is NumPy/Python. The transfer,
    # synchronization, top-k selection, and patch upload are deliberately part
    # of this direct pipeline measurement.
    scores = aligned.float().cpu().numpy()
    lengths = _entropy_lengths(scores, masks, codepoint_only)
    patches = torch.from_numpy(lengths).to(device)
    model(input_ids=input_ids, patch_lengths=patches, use_cache=False)


def _fixed_codepoint_pipeline(
    model: Any,
    input_ids: torch.Tensor,
    masks: np.ndarray,
    device: str,
) -> None:
    lengths = _fixed_codepoint_lengths(masks)
    patches = torch.from_numpy(lengths).to(device)
    model(input_ids=input_ids, patch_lengths=patches, use_cache=False)


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
        raise ValueError("the preregistered benchmark requires at least 30 repetitions")
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    seed = int(args.seed or manifest["seeds"][0])
    if seed not in manifest["seeds"]:
        raise ValueError(f"seed {seed} is not in the Phase 1 manifest")
    device = resolve_device(args.device)
    seed_artifact = artifact_root / f"seed-{seed}"

    test_limit = int(manifest["limits"]["test"])
    stream = build_neural_stream(
        Path(args.data_root) / "ko.jsonl",
        language="ko",
        split="test",
        byte_limit=test_limit,
        sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
    )
    inputs, masks = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    cache = np.load(seed_artifact / "patches.npz")
    patch_matrices = {
        policy: cache[f"test__ko__{policy}"].astype(np.int64)
        for policy in POLICIES
    }

    router = _load_state(build_router(seed=seed), seed_artifact / "router.pt")
    router.to(device).eval()
    models = {
        policy: _load_state(
            build_main_model(seed=seed),
            seed_artifact / f"{policy}.pt",
        ).to(device).eval()
        for policy in POLICIES
    }
    memory_after_load = _memory_snapshot(device)

    by_batch: dict[str, Any] = {}
    for batch_size in args.batch_size:
        if batch_size > len(inputs):
            raise ValueError(f"batch size {batch_size} exceeds test sequences")
        input_ids = torch.from_numpy(
            inputs[:batch_size].astype(np.int64, copy=False)
        ).to(device)
        local_masks = masks[:batch_size]
        device_patches = {
            policy: torch.from_numpy(patch_matrices[policy][:batch_size]).to(device)
            for policy in POLICIES
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
            "end_to_end/fixed_byte": main_functions["main_only/fixed_byte"],
            "end_to_end/fixed_codepoint": lambda: _fixed_codepoint_pipeline(
                models["fixed_codepoint"],
                input_ids,
                local_masks,
                device,
            ),
            "end_to_end/entropy_full": lambda: _entropy_pipeline(
                router,
                models["entropy_full"],
                input_ids,
                local_masks,
                device,
                False,
            ),
            "end_to_end/entropy_codepoint": lambda: _entropy_pipeline(
                router,
                models["entropy_codepoint"],
                input_ids,
                local_masks,
                device,
                True,
            ),
        }
        pipeline_timings = benchmark_interleaved(
            pipeline_functions,
            device,
            args.warmup_rounds,
            args.repetitions,
            seed + 10_000 + batch_size,
        )
        tokens = batch_size * DEFAULT_MODEL_SPEC.sequence_length
        by_batch[str(batch_size)] = {
            "component_timings": component_timings,
            "direct_pipeline_timings": pipeline_timings,
            "derived_throughput_bytes_per_second": {
                name: tokens / (values["median_ms"] / 1000)
                for name, values in pipeline_timings.items()
            },
        }
        print(f"benchmarked batch size {batch_size}", flush=True)

    analytical = end_to_end_flop_summary()
    batch_one = by_batch["1"]["direct_pipeline_timings"]
    fixed_latency = batch_one["end_to_end/fixed_codepoint"]["median_ms"]
    entropy_latency = batch_one["end_to_end/entropy_full"]["median_ms"]
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Teacher-forced 256-byte Korean test windows; not incremental "
            "autoregressive generation latency"
        ),
        "seed": seed,
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "mac_hardware_model": _system_value(["sysctl", "-n", "hw.model"]),
        "versions": research_versions(),
        "parameters": {
            "main": parameter_count(models["fixed_byte"]),
            "router": parameter_count(router),
            "entropy_end_to_end": (
                parameter_count(models["fixed_byte"]) + parameter_count(router)
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
        },
        "analytical_flops": analytical,
        "measurements": by_batch,
        "batch1_fixed_codepoint_reduction_vs_entropy_full": (
            1 - fixed_latency / entropy_latency
        ),
        "memory_after_model_load": memory_after_load,
        "memory_after_benchmark": _memory_snapshot(device),
        "limitations": [
            "MPS measurements do not predict CUDA serving latency.",
            "The matched-rate top-k policies are window-level and non-streaming.",
            "The direct entropy pipeline includes MPS-to-CPU score transfer and the preregistered Python/NumPy selector.",
            "Peak per-condition MPS allocation cannot be reset with the available API; only process snapshots are recorded.",
            "Analytical FLOPs omit the explicitly listed non-matmul operations.",
        ],
    }
    _write_json(Path(args.output), output)
    print(f"wrote Phase 1 cost benchmark to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase1")
    parser.add_argument("--artifact-root", default="artifacts/phase1")
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--output", default="runs/phase1/cost-benchmark.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        action="append",
        default=None,
        help="May be repeated; defaults to 1, 8, and 64.",
    )
    parser.add_argument("--warmup-rounds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.batch_size is None:
        parsed.batch_size = [1, 8, 64]
    raise SystemExit(run(parsed))
