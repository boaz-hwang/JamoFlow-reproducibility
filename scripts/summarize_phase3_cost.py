#!/usr/bin/env python3
"""Validate Phase 3 quality/cost evidence and evaluate Pareto Gate K."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping

import numpy as np

from jamoflow.cost import variable_patch_flop_summary
from jamoflow.phase2_patching import validate_padded_patch_matrix
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_POLICIES,
    STRUCTURAL_POLICIES,
    THRESHOLD_POLICIES,
)


F = "fixed_byte_6"
C = "causal_codepoint_grid"
W = "causal_whitespace_grid"
S = "spacebyte_spacelike"
E = "entropy_threshold_full"
EC = "entropy_threshold_codepoint"
LEARNED_POLICY_ORDER = (E, EC)
EXPECTED_BATCHES = [1, 8, 32, 64]
EXPECTED_SEED = 1729
EXPECTED_TIMING_BATCHES = 8
SPLITS = ("train", "calibration", "test")
TIMING_SECTIONS = (
    "component_timings",
    "selector_only_cpu_timings",
    "direct_pipeline_timings",
)
TIMING_BOOTSTRAP_RESAMPLES = 10_000
TIMING_BOOTSTRAP_SEED = 20_260_811


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


def _assert_equivalent(actual: Any, expected: Any, path: str) -> None:
    """Compare reconstructed evidence, allowing only tiny float roundoff."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"mapping mismatch at {path}")
        for key in expected:
            _assert_equivalent(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"list mismatch at {path}")
        for index, value in enumerate(expected):
            _assert_equivalent(actual[index], value, f"{path}[{index}]")
        return
    if (
        isinstance(expected, (float, np.floating))
        and not isinstance(expected, bool)
    ):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError) as error:
            raise ValueError(f"numeric mismatch at {path}") from error
        if not math.isfinite(actual_float) or not math.isclose(
            actual_float,
            float(expected),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(f"numeric mismatch at {path}")
        return
    if actual != expected:
        raise ValueError(f"value mismatch at {path}")


def _expected_timing_names(section: str) -> tuple[str, ...]:
    if section == "component_timings":
        return tuple(f"main_only/{policy}" for policy in PHASE3_POLICIES) + (
            "router_only",
        )
    if section == "selector_only_cpu_timings":
        return tuple(f"selector_only/{policy}" for policy in PHASE3_POLICIES)
    if section == "direct_pipeline_timings":
        return tuple(f"end_to_end/{policy}" for policy in PHASE3_POLICIES)
    raise ValueError(f"unknown timing section: {section}")


def _balanced_measurement_schedule(
    *,
    input_batches: int,
    warmup_rounds: int,
    repetitions: int,
    method_count: int,
    seed: int,
) -> list[int]:
    """Reproduce benchmark_interleaved's RNG consumption and measured rows."""

    rng = random.Random(seed)

    def balanced(rounds: int) -> list[int]:
        schedule: list[int] = []
        while len(schedule) < rounds:
            cycle = list(range(input_batches))
            rng.shuffle(cycle)
            schedule.extend(cycle)
        return schedule[:rounds]

    for _ in balanced(warmup_rounds):
        order = list(range(method_count))
        rng.shuffle(order)
    measured = balanced(repetitions)
    for _ in measured:
        order = list(range(method_count))
        rng.shuffle(order)
    return measured


def paired_input_batch_stability(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    seed: int,
    resamples: int = TIMING_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Bootstrap the mean of paired per-input-batch median reductions."""

    candidate_ids = [int(value) for value in candidate["measurement_input_batch_ids"]]
    reference_ids = [int(value) for value in reference["measurement_input_batch_ids"]]
    if candidate_ids != reference_ids:
        raise ValueError("paired timing methods used different input-batch schedules")
    candidate_values = np.asarray(candidate["measurements_ms"], dtype=np.float64)
    reference_values = np.asarray(reference["measurements_ms"], dtype=np.float64)
    if (
        len(candidate_values) != len(candidate_ids)
        or len(reference_values) != len(reference_ids)
        or not np.all(np.isfinite(candidate_values))
        or not np.all(np.isfinite(reference_values))
        or np.any(candidate_values <= 0)
        or np.any(reference_values <= 0)
    ):
        raise ValueError("invalid paired timing samples")
    batch_ids = sorted(set(candidate_ids))
    reductions = []
    ids = np.asarray(candidate_ids, dtype=np.int64)
    for batch_id in batch_ids:
        selected = ids == batch_id
        candidate_median = float(np.median(candidate_values[selected]))
        reference_median = float(np.median(reference_values[selected]))
        reductions.append(1 - candidate_median / reference_median)
    reduction_array = np.asarray(reductions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(reduction_array),
        size=(resamples, len(reduction_array)),
    )
    bootstrap = reduction_array[sampled].mean(axis=1)
    return {
        "estimand": "mean paired per-input-batch median latency reduction",
        "input_batch_ids": batch_ids,
        "input_batch_count": len(batch_ids),
        "per_input_batch_median_reductions": [
            float(value) for value in reduction_array
        ],
        "mean_reduction": float(reduction_array.mean()),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_percentile_95_interval": {
            "lower": float(np.percentile(bootstrap, 2.5)),
            "upper": float(np.percentile(bootstrap, 97.5)),
        },
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_test_patch_matrices(
    artifact_root: Path,
    benchmark: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    structural_path = artifact_root / "structural-patches.npz"
    threshold_path = artifact_root / f"seed-{EXPECTED_SEED}" / "threshold-patches.npz"
    integrity = benchmark["integrity"]
    if (
        _sha256(structural_path)
        != integrity["structural_cache_artifact_sha256"]
        or _sha256(threshold_path)
        != integrity["threshold_cache_artifact_sha256"]
    ):
        raise ValueError("cost cache artifacts differ from benchmark lineage")

    matrices: dict[str, np.ndarray] = {}
    with np.load(structural_path, allow_pickle=False) as loaded:
        expected = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in STRUCTURAL_POLICIES
        }
        if set(loaded.files) != expected:
            raise ValueError("structural cost cache keys mismatch")
        for policy in STRUCTURAL_POLICIES:
            matrix = loaded[f"test__{policy}"]
            if matrix.dtype != np.uint16:
                raise ValueError("structural cost cache dtype mismatch")
            matrices[policy] = matrix.copy()
    with np.load(threshold_path, allow_pickle=False) as loaded:
        expected = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in THRESHOLD_POLICIES
        }
        if set(loaded.files) != expected:
            raise ValueError("threshold cost cache keys mismatch")
        for policy in THRESHOLD_POLICIES:
            matrix = loaded[f"test__{policy}"]
            if matrix.dtype != np.uint16:
                raise ValueError("threshold cost cache dtype mismatch")
            matrices[policy] = matrix.copy()
    row_counts = {len(matrix) for matrix in matrices.values()}
    if len(row_counts) != 1:
        raise ValueError("cost matrices do not share a test row count")
    for matrix in matrices.values():
        validate_padded_patch_matrix(matrix, PHASE3_MODEL_SPEC.sequence_length)
    return matrices


def _reconstruct_analytical_flops(
    matrices: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        str(batch_size): {
            policy: variable_patch_flop_summary(
                (matrix[:, 1:] > 0).sum(axis=1).astype(np.int64),
                batch_size=batch_size,
                include_router=policy in THRESHOLD_POLICIES,
                spec=PHASE3_MODEL_SPEC,
            )
            for policy, matrix in matrices.items()
        }
        for batch_size in EXPECTED_BATCHES
    }


def _validate_timing_evidence(benchmark: Mapping[str, Any]) -> None:
    protocol = benchmark["protocol"]
    measurements = benchmark["measurements"]
    expected_batches = {str(value) for value in EXPECTED_BATCHES}
    if set(measurements) != expected_batches:
        raise ValueError("cost timing batch keys mismatch")
    timing_batch_count = int(protocol["timing_batches_per_batch_size"])
    repetitions = int(protocol["repetitions"])
    warmup_rounds = int(protocol["warmup_rounds"])
    benchmark_seed = int(benchmark["seed"])
    for batch_key, batch_values in measurements.items():
        batch_size = int(batch_key)
        if set(batch_values) != {
            *TIMING_SECTIONS,
            "derived_throughput_input_bytes_per_second",
        }:
            raise ValueError(f"unexpected timing sections for batch {batch_key}")
        for section_index, section in enumerate(TIMING_SECTIONS):
            timings = batch_values[section]
            names = _expected_timing_names(section)
            if set(timings) != set(names):
                raise ValueError(f"timing method set mismatch: {batch_key}/{section}")
            seed_offset = (0, 10_000, 20_000)[section_index]
            expected_ids = _balanced_measurement_schedule(
                input_batches=timing_batch_count,
                warmup_rounds=warmup_rounds,
                repetitions=repetitions,
                method_count=len(names),
                seed=benchmark_seed + seed_offset + batch_size,
            )
            expected_counts = {
                str(batch_id): expected_ids.count(batch_id)
                for batch_id in range(timing_batch_count)
            }
            if max(expected_counts.values()) - min(expected_counts.values()) > 1:
                raise ValueError("timing schedule is not balanced")
            for name in names:
                values = timings[name]
                samples = [float(value) for value in values["measurements_ms"]]
                if (
                    len(samples) != repetitions
                    or not np.all(np.isfinite(samples))
                    or np.any(np.asarray(samples) <= 0)
                ):
                    raise ValueError(f"invalid timing samples: {batch_key}/{name}")
                expected_summary = {
                    **_timing_summary(samples),
                    "input_batches": timing_batch_count,
                    "input_batch_measurement_counts": expected_counts,
                    "measurement_input_batch_ids": expected_ids,
                }
                _assert_equivalent(
                    values,
                    expected_summary,
                    f"measurements.{batch_key}.{section}.{name}",
                )

        throughput = batch_values["derived_throughput_input_bytes_per_second"]
        expected_throughput = {
            name: batch_size * PHASE3_MODEL_SPEC.sequence_length
            / (values["median_ms"] / 1000)
            for name, values in batch_values["direct_pipeline_timings"].items()
        }
        _assert_equivalent(
            throughput,
            expected_throughput,
            f"measurements.{batch_key}.derived_throughput",
        )


def _reconstruct_comparisons(
    benchmark: Mapping[str, Any],
    analytical: Mapping[str, Any],
) -> dict[str, Any]:
    by_batch = benchmark["measurements"]
    return {
        learned_policy: {
            candidate: {
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
            for candidate in STRUCTURAL_POLICIES
        }
        for learned_policy in THRESHOLD_POLICIES
    }


def _validate_quality_lineage(
    benchmark: Mapping[str, Any],
    quality: Mapping[str, Any],
    gate_j: Mapping[str, Any],
    *,
    quality_path: Path,
    gate_j_path: Path,
    run_root: Path,
) -> None:
    benchmark_integrity = benchmark["integrity"]
    quality_integrity = quality["integrity"]
    seed_lineage = quality_integrity["by_seed"][str(EXPECTED_SEED)]
    if (
        benchmark_integrity["shared_seed_quality_summary_sha256"]
        != _sha256(quality_path)
        or benchmark_integrity["checkpoint_state_sha256"]
        != seed_lineage["checkpoint_state_sha256"]
        or benchmark_integrity["checkpoint_artifact_sha256"]
        != seed_lineage["checkpoint_artifact_sha256"]
        or benchmark_integrity["training_report_artifact_sha256"]
        != seed_lineage["training_report_artifact_sha256"]
        or benchmark_integrity["router_and_threshold_cache"]
        != seed_lineage["router_and_threshold_cache"]
        or benchmark_integrity["source_artifact"]
        != quality["run_manifest"]["source_artifact"]
        or benchmark_integrity["source_integrity_artifact"]
        != quality["run_manifest"]["source_integrity_artifact"]
    ):
        raise ValueError("cost benchmark differs from shared quality lineage")
    manifest_path = run_root / "manifest.json"
    if (
        _sha256(manifest_path) != benchmark_integrity["run_manifest_sha256"]
        or _read_json(manifest_path) != quality["run_manifest"]
    ):
        raise ValueError("cost run manifest differs from shared quality evidence")

    gate_integrity = gate_j["integrity"]
    invariant_keys = (
        "quick_smoke_only",
        "language",
        "limits",
        "source_artifact",
        "source_integrity_artifact",
        "global_max_position_embeddings",
        "model_spec",
        "optimization_spec",
        "streams",
    )
    for key in invariant_keys:
        if gate_j["run_manifest"].get(key) != quality["run_manifest"].get(key):
            raise ValueError(f"Gate J/shared quality manifest mismatch: {key}")
    common_seeds = sorted(
        set(int(seed) for seed in quality["seeds"])
        & set(int(seed) for seed in gate_j["seeds"])
    )
    if not common_seeds or EXPECTED_SEED not in common_seeds:
        raise ValueError("Gate J and quality evidence lack common cost seed")
    for seed in common_seeds:
        quality_seed = quality_integrity["by_seed"][str(seed)]
        gate_seed = gate_integrity["by_seed"][str(seed)]
        for key in (
            "checkpoint_state_sha256",
            "checkpoint_artifact_sha256",
            "training_report_artifact_sha256",
        ):
            for policy in (F, C, W):
                if gate_seed[key][policy] != quality_seed[key][policy]:
                    raise ValueError(
                        f"Gate J/shared quality checkpoint mismatch: {seed}/{policy}"
                    )
    if not gate_j_path.exists():
        raise FileNotFoundError(gate_j_path)


def select_learned_policy(quality: Mapping[str, Any]) -> str:
    """Select E versus EC by mean quality only, with E as the fixed tie-break."""

    missing = set(LEARNED_POLICY_ORDER) - set(quality)
    if missing:
        raise ValueError(f"missing learned quality policies: {sorted(missing)}")
    return min(
        LEARNED_POLICY_ORDER,
        key=lambda policy: float(quality[policy]["mean"]),
    )


def pareto_summary(
    quality_means: Mapping[str, float],
    analytical_costs: Mapping[str, float],
) -> dict[str, Any]:
    if set(quality_means) != set(analytical_costs):
        raise ValueError("quality and cost policies must match")
    by_policy: dict[str, Any] = {}
    for policy in quality_means:
        quality = float(quality_means[policy])
        cost = float(analytical_costs[policy])
        dominators = []
        for other in quality_means:
            if other == policy:
                continue
            other_quality = float(quality_means[other])
            other_cost = float(analytical_costs[other])
            weakly_better = other_quality <= quality and other_cost <= cost
            strictly_better = other_quality < quality or other_cost < cost
            if weakly_better and strictly_better:
                dominators.append(other)
        by_policy[policy] = {
            "mean_test_bpb": quality,
            "ideal_unpadded_flops_per_sequence": cost,
            "dominated": bool(dominators),
            "dominators": sorted(dominators),
        }
    return {
        "cost_axis": (
            "ideal unpadded dense-matmul forward FLOPs including router for E/EC"
        ),
        "quality_axis": "mean held-out HPLT3 test BPB over the shared seed set",
        "by_policy": by_policy,
    }


def gate_k_summary(
    benchmark: Mapping[str, Any],
    quality_summary: Mapping[str, Any],
    *,
    gate_j_pass: bool | None,
) -> dict[str, Any]:
    quality = quality_summary["quality"]
    calibration_quality = quality_summary.get("calibration_quality", {})
    missing = set(PHASE3_POLICIES) - set(quality)
    missing_calibration = set(PHASE3_POLICIES) - set(calibration_quality)
    if missing or missing_calibration:
        raise ValueError(
            "quality summary lacks policies: "
            f"test={sorted(missing)}, calibration={sorted(missing_calibration)}"
        )
    counts = {int(quality[policy]["count"]) for policy in PHASE3_POLICIES}
    calibration_counts = {
        int(calibration_quality[policy]["count"])
        for policy in PHASE3_POLICIES
    }
    if (
        len(counts) != 1
        or next(iter(counts)) < 3
        or calibration_counts != counts
    ):
        raise ValueError("all Phase 3 policies need the same >=3 quality seeds")

    selected_learned = select_learned_policy(calibration_quality)
    quality_means = {
        policy: float(quality[policy]["mean"])
        for policy in PHASE3_POLICIES
    }
    analytical_costs = {
        policy: float(
            benchmark["analytical_flops"]["1"][policy][
                "ideal_unpadded_mean_flops_per_sequence"
            ]
        )
        for policy in PHASE3_POLICIES
    }
    pareto = pareto_summary(quality_means, analytical_costs)

    comparison = benchmark["comparisons_vs_learned_router"][selected_learned][W]
    analytical_reduction = 1 - analytical_costs[W] / analytical_costs[
        selected_learned
    ]
    if not math.isclose(
        analytical_reduction,
        float(comparison["ideal_unpadded_flop_reduction"]),
        abs_tol=1e-12,
    ):
        raise ValueError("stored analytical comparison is inconsistent")

    latency_reductions: dict[str, float] = {}
    latency_stability: dict[str, Any] = {}
    qualifying_latency_batches: list[str] = []
    for batch in ("1", "8"):
        timings = benchmark["measurements"][batch]["direct_pipeline_timings"]
        recomputed = 1 - float(timings[f"end_to_end/{W}"]["median_ms"]) / float(
            timings[f"end_to_end/{selected_learned}"]["median_ms"]
        )
        stored = float(comparison["direct_latency_reduction"][batch])
        if not math.isclose(recomputed, stored, abs_tol=1e-12):
            raise ValueError(f"stored batch-{batch} latency comparison is inconsistent")
        latency_reductions[batch] = recomputed
        stability = paired_input_batch_stability(
            timings[f"end_to_end/{W}"],
            timings[f"end_to_end/{selected_learned}"],
            seed=TIMING_BOOTSTRAP_SEED + int(batch),
        )
        latency_stability[batch] = stability
        if (
            recomputed >= 0.10
            and stability["bootstrap_percentile_95_interval"]["lower"] > 0
        ):
            qualifying_latency_batches.append(batch)

    quality_difference = quality_means[W] - quality_means[selected_learned]
    h2_quality_pass = quality_difference <= 0.010
    analytical_pass = analytical_reduction >= 0.10
    latency_pass = bool(qualifying_latency_batches)
    whitespace_nondominated = not pareto["by_policy"][W]["dominated"]
    components_pass = (
        h2_quality_pass
        and analytical_pass
        and latency_pass
        and whitespace_nondominated
    )

    if gate_j_pass is None:
        status = "pending_gate_j"
        overall: bool | None = None
    elif not gate_j_pass:
        status = "fail_gate_j"
        overall = False
    elif components_pass:
        status = "pass"
        overall = True
    else:
        status = "fail_pareto_evidence"
        overall = False
    return {
        "status": status,
        "overall_pass": overall,
        "gate_j_pass": gate_j_pass,
        "learned_policy_selection_rule": (
            "minimum mean calibration BPB over shared seeds; fixed tie-break "
            "E before EC; held-out test BPB is opened only for evaluation"
        ),
        "selected_learned_policy": selected_learned,
        "learned_policy_calibration_means_bpb": {
            policy: float(calibration_quality[policy]["mean"])
            for policy in LEARNED_POLICY_ORDER
        },
        "learned_policy_quality_means_bpb": {
            policy: quality_means[policy] for policy in LEARNED_POLICY_ORDER
        },
        "whitespace_minus_selected_learned_bpb": quality_difference,
        "h2_quality_maximum_regression_bpb": 0.010,
        "h2_quality_pass": h2_quality_pass,
        "analytical_flop_reduction": analytical_reduction,
        "analytical_reduction_at_least_10_percent": analytical_pass,
        "direct_latency_reduction": latency_reductions,
        "paired_input_batch_latency_stability": latency_stability,
        "qualifying_latency_batches": qualifying_latency_batches,
        "batch1_or_batch8_latency_reduction_at_least_10_percent_with_positive_paired_bootstrap_lower_bound": latency_pass,
        "whitespace_nondominated_with_spacebyte_included": whitespace_nondominated,
        "pareto": pareto,
    }


def _compact_timings(measurements: Mapping[str, Any]) -> dict[str, Any]:
    return {
        batch: {
            section: {
                name: {
                    key: value
                    for key, value in values.items()
                    if key
                    not in ("measurements_ms", "measurement_input_batch_ids")
                }
                for name, values in batch_values[section].items()
            }
            for section in (
                "component_timings",
                "selector_only_cpu_timings",
                "direct_pipeline_timings",
            )
        }
        for batch, batch_values in measurements.items()
    }


def run(args: argparse.Namespace) -> int:
    benchmark_path = Path(args.benchmark)
    quality_path = Path(args.quality_summary)
    gate_j_path = Path(args.gate_j_summary)
    artifact_root = Path(args.artifact_root)
    run_root = Path(args.run_root)
    benchmark = _read_json(benchmark_path)
    quality = _read_json(quality_path)
    gate_j_source = _read_json(gate_j_path)

    protocol = benchmark["protocol"]
    if benchmark.get("schema_version") != 1:
        raise ValueError("unexpected Phase 3 cost benchmark schema")
    if benchmark["seed"] != EXPECTED_SEED:
        raise ValueError("expected preregistered cost seed 1729")
    if protocol["batch_sizes"] != EXPECTED_BATCHES:
        raise ValueError("unexpected Phase 3 benchmark batch sizes")
    if (
        protocol["repetitions"] < 30
        or protocol["warmup_rounds"] < 1
        or protocol["timing_batches_per_batch_size"]
        != EXPECTED_TIMING_BATCHES
        or protocol["sequence_input_bytes"]
        != PHASE3_MODEL_SPEC.sequence_length
        or protocol["predicted_bytes_per_sequence"]
        != PHASE3_MODEL_SPEC.sequence_length - 1
        or protocol["timing_batch_selection_seed"] != EXPECTED_SEED + 30_000
        or protocol["randomized_interleaving"] is not True
        or protocol["balanced_shared_input_batch_schedule"] is not True
        or protocol["device_synchronization_around_each_device_measurement"]
        is not True
        or protocol["inputs_preloaded_on_device"] is not True
    ):
        raise ValueError("Phase 3 timing budget is insufficient")
    if (
        set(quality.get("policies", [])) != set(PHASE3_POLICIES)
        or quality["integrity"]["all_integrity_checks_pass"] is not True
    ):
        raise ValueError("shared-seed quality integrity checks failed")
    if gate_j_source["integrity"]["all_integrity_checks_pass"] is not True:
        raise ValueError("Gate J quality integrity checks failed")

    _validate_quality_lineage(
        benchmark,
        quality,
        gate_j_source,
        quality_path=quality_path,
        gate_j_path=gate_j_path,
        run_root=run_root,
    )

    selector = benchmark["integrity"]["selector_reconstruction"]
    if set(selector) != {str(value) for value in EXPECTED_BATCHES}:
        raise ValueError("selector reconstruction batch keys mismatch")
    selector_checks = []
    for batch in selector.values():
        if set(batch) != {
            str(value) for value in range(EXPECTED_TIMING_BATCHES)
        }:
            raise ValueError("selector reconstruction timing-row keys mismatch")
        for timing_row in batch.values():
            if set(timing_row) != set(PHASE3_POLICIES):
                raise ValueError("selector reconstruction policy keys mismatch")
            selector_checks.extend(
                bool(values["matches_cached_evaluation_matrix"])
                and values["selected_matrix_sha256"]
                == values["cached_matrix_sha256"]
                for values in timing_row.values()
            )
    if (
        not selector_checks
        or not all(selector_checks)
        or benchmark["integrity"]["all_selector_reconstructions_match"]
        is not True
    ):
        raise ValueError("benchmark selector reconstruction failed")

    matrices = _load_test_patch_matrices(artifact_root, benchmark)
    expected_index_matrix = np.random.default_rng(
        protocol["timing_batch_selection_seed"]
    ).permutation(len(next(iter(matrices.values()))))[
        : max(EXPECTED_BATCHES) * EXPECTED_TIMING_BATCHES
    ].reshape(EXPECTED_TIMING_BATCHES, max(EXPECTED_BATCHES))
    if (
        _array_sha256(expected_index_matrix.astype(np.int64, copy=False))
        != protocol["timing_sequence_indices_sha256"]
    ):
        raise ValueError("timing sequence selection hash mismatch")

    reconstructed_analytical = _reconstruct_analytical_flops(matrices)
    _assert_equivalent(
        benchmark["analytical_flops"],
        reconstructed_analytical,
        "analytical_flops",
    )
    _validate_timing_evidence(benchmark)
    reconstructed_comparisons = _reconstruct_comparisons(
        benchmark,
        reconstructed_analytical,
    )
    _assert_equivalent(
        benchmark["comparisons_vs_learned_router"],
        reconstructed_comparisons,
        "comparisons_vs_learned_router",
    )

    gate_j_pass = gate_j_source["gate_j"]["overall_pass"]
    if gate_j_pass not in (True, False, None):
        raise ValueError("invalid Gate J status")
    gate_k = gate_k_summary(
        benchmark,
        quality,
        gate_j_pass=gate_j_pass,
    )
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "scope": benchmark["scope"],
        "sources": {
            "benchmark": str(benchmark_path),
            "benchmark_sha256": _sha256(benchmark_path),
            "shared_seed_quality_summary": str(quality_path),
            "shared_seed_quality_summary_sha256": _sha256(quality_path),
            "gate_j_summary": str(gate_j_path),
            "gate_j_summary_sha256": _sha256(gate_j_path),
            "run_manifest": str(run_root / "manifest.json"),
            "run_manifest_sha256": _sha256(run_root / "manifest.json"),
            "structural_patch_cache": str(
                artifact_root / "structural-patches.npz"
            ),
            "structural_patch_cache_sha256": _sha256(
                artifact_root / "structural-patches.npz"
            ),
            "threshold_patch_cache": str(
                artifact_root
                / f"seed-{EXPECTED_SEED}"
                / "threshold-patches.npz"
            ),
            "threshold_patch_cache_sha256": _sha256(
                artifact_root
                / f"seed-{EXPECTED_SEED}"
                / "threshold-patches.npz"
            ),
        },
        "environment": {
            key: benchmark[key]
            for key in (
                "seed",
                "device",
                "platform",
                "processor",
                "mac_hardware_model",
                "versions",
                "parameters",
            )
        },
        "protocol": protocol,
        "integrity": {
            "all_integrity_checks_pass": True,
            "quality_and_gate_j_lineage_reconstructed": True,
            "timing_summaries_reconstructed_from_raw_samples": True,
            "timing_schedules_reconstructed_from_preregistered_seeds": True,
            "analytical_flops_reconstructed_from_patch_caches": True,
            "comparisons_reconstructed": True,
            "selector_subset_hashes_self_consistent": True,
        },
        "quality": quality["quality"],
        "analytical_flops": reconstructed_analytical,
        "timing_summaries_without_raw_samples": _compact_timings(
            benchmark["measurements"]
        ),
        "comparisons_vs_learned_router": reconstructed_comparisons,
        "gate_k": gate_k,
        "memory_after_all_comparison_models_loaded": benchmark[
            "memory_after_all_comparison_models_loaded"
        ],
        "memory_after_benchmark": benchmark["memory_after_benchmark"],
        "limitations": benchmark["limitations"],
    }
    _write_json(Path(args.output), output)
    print(json.dumps(gate_k, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default="runs/phase3/cost-benchmark.json"
    )
    parser.add_argument(
        "--quality-summary", default="results/phase3-all-policies/summary.json"
    )
    parser.add_argument(
        "--gate-j-summary", default="results/phase3-final/summary.json"
    )
    parser.add_argument("--run-root", default="runs/phase3")
    parser.add_argument("--artifact-root", default="artifacts/phase3")
    parser.add_argument("--output", default="results/phase3-cost/summary.json")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
