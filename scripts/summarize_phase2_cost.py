#!/usr/bin/env python3
"""Validate and promote compact Phase 2 cost aggregates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_timings(source: dict[str, Any]) -> dict[str, Any]:
    return {
        batch: {
            section: {
                name: {
                    key: value
                    for key, value in values.items()
                    if key != "measurements_ms"
                }
                for name, values in batch_values[section].items()
            }
            for section in ("component_timings", "direct_pipeline_timings")
        }
        for batch, batch_values in source.items()
    }


def run(args: argparse.Namespace) -> int:
    benchmark_path = Path(args.benchmark)
    primary_path = Path(args.primary_summary)
    controls_path = Path(args.controls_summary)
    benchmark = _read_json(benchmark_path)
    primary = _read_json(primary_path)
    controls = _read_json(controls_path)
    protocol = benchmark["protocol"]
    if benchmark["seed"] != 1729:
        raise ValueError("expected preregistered benchmark seed 1729")
    if protocol["batch_sizes"] != [1, 8, 64]:
        raise ValueError("unexpected benchmark batch sizes")
    if protocol["repetitions"] < 100 or protocol["warmup_rounds"] < 10:
        raise ValueError("benchmark does not meet the fixed timing budget")

    quality_contrasts = {
        "causal_codepoint_grid": primary["contrasts"][
            "causal_codepoint_minus_entropy_full"
        ]["paired_t_95_interval"]["mean"],
        "causal_eojeol_grid": primary["contrasts"][
            "causal_eojeol_minus_entropy_full"
        ]["paired_t_95_interval"]["mean"],
        "causal_whitespace_grid": (
            primary["contrasts"]["causal_eojeol_minus_entropy_full"][
                "paired_t_95_interval"
            ]["mean"]
            - controls["contrasts"]["eojeol_minus_whitespace"][
                "paired_t_95_interval"
            ]["mean"]
        ),
    }
    gates: dict[str, Any] = {}
    for policy, comparison in benchmark[
        "candidate_comparisons_vs_entropy_full"
    ].items():
        quality = quality_contrasts[policy]
        analytical = comparison["ideal_unpadded_flop_reduction_vs_entropy_full"]
        latency = comparison["batch1_direct_latency_reduction_vs_entropy_full"]
        padding_aware = comparison[
            "implemented_batch_max_flop_reduction_vs_entropy_full"
        ]["64"]
        passed = (
            quality <= 0.015
            and analytical >= 0.10
            and latency >= 0.10
            and padding_aware >= 0.10
        )
        gates[policy] = {
            "quality_difference_vs_entropy_full_bpb": quality,
            "quality_within_0_015_harm_margin": quality <= 0.015,
            "ideal_unpadded_flop_reduction": analytical,
            "ideal_reduction_at_least_10_percent": analytical >= 0.10,
            "batch1_direct_latency_reduction": latency,
            "latency_reduction_at_least_10_percent": latency >= 0.10,
            "batch64_padding_aware_flop_reduction": padding_aware,
            "padding_aware_reduction_at_least_10_percent": padding_aware >= 0.10,
            "gate_f_pass": passed,
        }

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": benchmark["scope"],
        "source": {
            "benchmark": str(benchmark_path),
            "benchmark_sha256": _sha256(benchmark_path),
            "primary_summary": str(primary_path),
            "primary_summary_sha256": _sha256(primary_path),
            "controls_summary": str(controls_path),
            "controls_summary_sha256": _sha256(controls_path),
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
        "analytical_flops": benchmark["analytical_flops"],
        "timing_summaries_without_raw_samples": _compact_timings(
            benchmark["measurements"]
        ),
        "candidate_comparisons_vs_entropy_full": benchmark[
            "candidate_comparisons_vs_entropy_full"
        ],
        "decision_gate_f": {
            "status": "pass" if all(item["gate_f_pass"] for item in gates.values()) else "fail",
            "by_candidate": gates,
        },
        "memory_after_model_load": benchmark["memory_after_model_load"],
        "memory_after_benchmark": benchmark["memory_after_benchmark"],
        "limitations": benchmark["limitations"],
    }
    _write_json(Path(args.output), output)
    print(f"wrote validated Phase 2 cost aggregates to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="runs/phase2/cost-benchmark.json")
    parser.add_argument(
        "--primary-summary",
        default="results/phase2-korean-primary/summary.json",
    )
    parser.add_argument(
        "--controls-summary",
        default="results/phase2-controls/summary.json",
    )
    parser.add_argument("--output", default="results/phase2-cost/summary.json")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
