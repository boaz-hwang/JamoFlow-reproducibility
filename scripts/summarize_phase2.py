#!/usr/bin/env python3
"""Validate and aggregate the preregistered Korean Phase 2 primary runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import (
    aggregate_numeric_mappings,
    hierarchical_paired_bootstrap,
    numeric_summary,
    paired_t_interval,
)
from jamoflow.phase2_analysis import gate_effect_checks, korean_test_strata
from jamoflow.phase2_patching import PHASE2_POLICIES, STRUCTURAL_POLICIES


CONTRASTS = {
    "causal_codepoint_minus_fixed_byte": (
        "causal_codepoint_grid",
        "fixed_byte_6",
    ),
    "causal_eojeol_minus_causal_codepoint": (
        "causal_eojeol_grid",
        "causal_codepoint_grid",
    ),
    "entropy_codepoint_minus_entropy_full": (
        "entropy_threshold_codepoint",
        "entropy_threshold_full",
    ),
    "causal_codepoint_minus_entropy_full": (
        "causal_codepoint_grid",
        "entropy_threshold_full",
    ),
    "causal_eojeol_minus_entropy_full": (
        "causal_eojeol_grid",
        "entropy_threshold_full",
    ),
}


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


def _load_complete_runs(
    run_root: Path,
    artifact_root: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    routers: dict[int, dict[str, Any]] = {}
    threshold_diagnostics: dict[int, dict[str, Any]] = {}
    missing: list[str] = []
    for seed in seeds:
        run_directory = run_root / f"seed-{seed}"
        artifact_directory = artifact_root / f"seed-{seed}"
        router_path = run_directory / "router.json"
        diagnostic_path = run_directory / "threshold-patch-diagnostics.json"
        if router_path.exists():
            routers[seed] = _read_json(router_path)
        else:
            missing.append(str(router_path))
        if diagnostic_path.exists():
            threshold_diagnostics[seed] = _read_json(diagnostic_path)
        else:
            missing.append(str(diagnostic_path))

        reports[seed] = {}
        losses[seed] = {}
        for policy in policies:
            report_path = run_directory / f"{policy}.json"
            loss_path = artifact_directory / f"{policy}-test-nll.npz"
            if not report_path.exists():
                missing.append(str(report_path))
                continue
            if not loss_path.exists():
                missing.append(str(loss_path))
                continue
            report = _read_json(report_path)
            if (
                report.get("seed") != seed
                or report.get("policy") != policy
                or report.get("language") != "ko"
            ):
                raise ValueError(f"identity mismatch in {report_path}")
            with np.load(loss_path) as archive:
                if archive.files != ["ko"]:
                    raise ValueError(f"unexpected loss keys in {loss_path}")
                losses[seed][policy] = archive["ko"].astype(np.float64)
            reports[seed][policy] = report
    if missing:
        raise FileNotFoundError(
            "Phase 2 primary run is incomplete; missing:\n" + "\n".join(missing)
        )
    return reports, losses, routers, threshold_diagnostics


def _quality_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        policy: numeric_summary(
            [reports[seed][policy]["evaluation"]["test"]["bpb"] for seed in seeds]
        )
        for policy in policies
    }


def _contrast_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
    repetitions: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    targets_per_sequence = 255
    for contrast_index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        reported = [
            reports[seed][left]["evaluation"]["test"]["bpb"]
            - reports[seed][right]["evaluation"]["test"]["bpb"]
            for seed in seeds
        ]
        sequence_differences = [
            losses[seed][left] - losses[seed][right]
            for seed in seeds
        ]
        for seed, report_value, sequence_values in zip(
            seeds,
            reported,
            sequence_differences,
            strict=True,
        ):
            reconstructed = float(sequence_values.mean()) / (
                targets_per_sequence * math.log(2)
            )
            if not math.isclose(report_value, reconstructed, abs_tol=2e-5):
                raise ValueError(
                    f"loss/report mismatch for {name}/seed-{seed}: "
                    f"{report_value} versus {reconstructed}"
                )
        interval = paired_t_interval(reported)
        bootstrap = hierarchical_paired_bootstrap(
            sequence_differences,
            targets_per_sequence=targets_per_sequence,
            repetitions=repetitions,
            seed=20_260_810 + contrast_index,
        )
        result[name] = {
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(seeds),
            "paired_differences_bpb": reported,
            "negative_seed_count": sum(value < 0 for value in reported),
            "paired_t_95_interval": interval.to_dict(),
            "hierarchical_bootstrap_95_interval": bootstrap.to_dict(),
        }
    return result


def _stratum_summary(
    stream_data: bytes,
    masks: np.ndarray,
    losses: dict[int, dict[str, np.ndarray]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    strata, metadata = korean_test_strata(stream_data, masks)
    scale = 255 * math.log(2)
    contrasts: dict[str, Any] = {}
    for name, (left, right) in CONTRASTS.items():
        contrast_strata: dict[str, Any] = {}
        for stratum_name, stratum in strata.items():
            selected = stratum.selected
            if not selected.any():
                contrast_strata[stratum_name] = {
                    **stratum.metadata(),
                    "status": "empty_in_primary_test",
                }
                continue
            effects = [
                float((losses[seed][left][selected] - losses[seed][right][selected]).mean())
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
            "Strata overlap and do not replace the full-test primary endpoint; "
            "no multiplicity-adjusted discovery claim is made from these slices."
        ),
    }


def _training_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    routers: dict[int, dict[str, Any]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "main_by_policy": {
            policy: aggregate_numeric_mappings(
                [reports[seed][policy]["training"] for seed in seeds]
            )
            for policy in policies
        },
        "router": {
            "parameters": numeric_summary(
                [routers[seed]["parameters"] for seed in seeds]
            ),
            "training": aggregate_numeric_mappings(
                [routers[seed]["training"] for seed in seeds]
            ),
            "calibration": aggregate_numeric_mappings(
                [routers[seed]["evaluation"]["calibration"] for seed in seeds]
            ),
            "test": aggregate_numeric_mappings(
                [routers[seed]["evaluation"]["test"] for seed in seeds]
            ),
        },
    }


def _patch_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    threshold_diagnostics: dict[int, dict[str, Any]],
    structural_diagnostics: dict[str, Any],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    by_split = {
        split: {
            policy: aggregate_numeric_mappings(
                [reports[seed][policy]["patch_diagnostics"][split] for seed in seeds]
            )
            for policy in policies
        }
        for split in ("train", "calibration", "test")
    }
    calibrations = {
        policy: aggregate_numeric_mappings(
            [
                threshold_diagnostics[seed]["calibration"][policy]
                for seed in seeds
            ]
        )
        for policy in (
            "entropy_threshold_full",
            "entropy_threshold_codepoint",
        )
    }
    return {
        "by_split_policy": by_split,
        "threshold_calibration": calibrations,
        "seed_independent_structural_diagnostics": structural_diagnostics,
    }


def _integrity_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    all_initialization_paired = True
    all_orders_paired = True
    for seed in seeds:
        initializations = {
            reports[seed][policy]["initialization_sha256"] for policy in policies
        }
        orders = {
            reports[seed][policy]["training_order_sha256"] for policy in policies
        }
        initialization_paired = len(initializations) == 1
        order_paired = len(orders) == 1
        all_initialization_paired &= initialization_paired
        all_orders_paired &= order_paired
        by_seed[str(seed)] = {
            "identical_initialization_across_policies": initialization_paired,
            "initialization_sha256": sorted(initializations),
            "identical_training_order_across_policies": order_paired,
            "training_order_sha256": sorted(orders),
        }

    structural_hashes = {
        policy: {
            split: sorted(
                {
                    reports[seed][policy]["patch_matrix_sha256"][split]
                    for seed in seeds
                }
            )
            for split in ("train", "calibration", "test")
        }
        for policy in STRUCTURAL_POLICIES
    }
    structural_seed_independent = all(
        len(values) == 1
        for policy_values in structural_hashes.values()
        for values in policy_values.values()
    )
    return {
        "all_seeds_have_identical_initialization_across_policies": (
            all_initialization_paired
        ),
        "all_seeds_have_identical_training_order_across_policies": all_orders_paired,
        "structural_matrices_are_seed_independent": structural_seed_independent,
        "by_seed": by_seed,
        "structural_patch_hashes": structural_hashes,
    }


def _gate_summary(
    contrasts: dict[str, Any],
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    codepoint = contrasts["causal_codepoint_minus_fixed_byte"]
    eojeol = contrasts["causal_eojeol_minus_causal_codepoint"]
    gate_d_primary = gate_effect_checks(
        codepoint["paired_differences_bpb"],
        maximum_mean=-0.003,
        interval_upper=codepoint["paired_t_95_interval"]["upper"],
    )
    gate_e_primary = gate_effect_checks(
        eojeol["paired_differences_bpb"],
        maximum_mean=-0.003,
        interval_upper=eojeol["paired_t_95_interval"]["upper"],
    )
    exact_eojeol_rate = all(
        reports[seed][policy]["patch_diagnostics"][split]["mean_data_patches"]
        == 43.0
        for seed in seeds
        for policy in ("causal_codepoint_grid", "causal_eojeol_grid")
        for split in ("train", "calibration", "test")
    )
    c1_entropy = contrasts["causal_codepoint_minus_entropy_full"]
    c2_entropy = contrasts["causal_eojeol_minus_entropy_full"]
    quality_candidates = {
        "causal_codepoint_grid": {
            "mean_difference_vs_entropy_full_bpb": c1_entropy[
                "paired_t_95_interval"
            ]["mean"],
            "within_0_015_harm_margin": c1_entropy["paired_t_95_interval"]["mean"]
            <= 0.015,
        },
        "causal_eojeol_grid": {
            "mean_difference_vs_entropy_full_bpb": c2_entropy[
                "paired_t_95_interval"
            ]["mean"],
            "within_0_015_harm_margin": c2_entropy["paired_t_95_interval"]["mean"]
            <= 0.015,
        },
    }
    return {
        "gate_d_causal_replication": {
            "status": "pending_aligned_packing_control",
            "primary_arbitrary_packing": gate_d_primary,
            "aligned_packing_direction": None,
        },
        "gate_e_korean_eojeol_value": {
            "status": "pending_ecological_or_external_diagnostic",
            "primary_effect": gate_e_primary,
            "exact_patch_count_match": exact_eojeol_rate,
            "external_regression_check": None,
        },
        "gate_f_parameter_free_pareto": {
            "status": "pending_cost_benchmark",
            "quality_component": quality_candidates,
            "analytical_cost_component": None,
            "batch1_latency_component": None,
            "padding_aware_component": None,
        },
        "gate_g_normalization_robustness": {
            "status": "pending_normalization_experiment"
        },
        "gate_h_scale_up": {
            "status": "pending_duplicate_alignment_validity_and_cost_controls"
        },
    }


def _write_observations_csv(
    path: Path,
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "policy",
                "test_bpb",
                "test_nll_nats",
                "predicted_bytes",
                "training_seconds",
                "train_mean_data_patches",
                "train_padded_data_width",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            for policy in policies:
                report = reports[seed][policy]
                evaluation = report["evaluation"]["test"]
                diagnostics = report["patch_diagnostics"]["train"]
                writer.writerow(
                    {
                        "seed": seed,
                        "policy": policy,
                        "test_bpb": f"{evaluation['bpb']:.12f}",
                        "test_nll_nats": f"{evaluation['nll_nats']:.12f}",
                        "predicted_bytes": evaluation["predicted_bytes"],
                        "training_seconds": f"{report['training']['elapsed_seconds']:.6f}",
                        "train_mean_data_patches": (
                            f"{diagnostics['mean_data_patches']:.9f}"
                        ),
                        "train_padded_data_width": diagnostics["padded_data_width"],
                    }
                )


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("quick_smoke_only"):
        raise ValueError("refusing to promote a quick smoke run as Phase 2 results")
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    policies = tuple(manifest["policies"])
    if seeds != (1729, 2718, 31415, 57721, 65537):
        raise ValueError(f"unexpected preregistered seeds: {seeds}")
    if policies != PHASE2_POLICIES:
        raise ValueError(f"unexpected Phase 2 policy order: {policies}")

    reports, losses, routers, threshold_diagnostics = _load_complete_runs(
        run_root,
        artifact_root,
        seeds,
        policies,
    )
    contrasts = _contrast_summary(
        reports,
        losses,
        seeds,
        args.bootstrap_repetitions,
    )
    test_stream = build_neural_stream(
        Path(args.data_root) / "ko.jsonl",
        language="ko",
        split="test",
        byte_limit=int(manifest["limits"]["test"]),
        sequence_length=int(manifest["model_spec"]["sequence_length"]),
    )
    _, test_masks = stream_arrays(
        test_stream.data,
        test_stream.codepoint_boundaries,
        test_stream.sequence_length,
    )
    structural_path = run_root / "structural-patch-diagnostics.json"
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 2 primary Korean-only compact-BLT experiment",
        "source": {
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": _sha256(manifest_path),
            "artifact_root": str(artifact_root),
            "git_commit_at_run_start": manifest.get("git_commit"),
        },
        "design": {
            "language": "ko",
            "seeds": list(seeds),
            "policies": list(policies),
            "test_sequences": test_stream.sequence_count,
            "predicted_bytes_per_sequence": 255,
        },
        "integrity": _integrity_summary(reports, seeds, policies),
        "quality": _quality_summary(reports, seeds, policies),
        "contrasts": contrasts,
        "korean_test_strata": _stratum_summary(
            test_stream.data,
            test_masks,
            losses,
            seeds,
        ),
        "training_and_router": _training_summary(
            reports,
            routers,
            seeds,
            policies,
        ),
        "patch_diagnostics": _patch_summary(
            reports,
            threshold_diagnostics,
            _read_json(structural_path),
            seeds,
            policies,
        ),
        "decision_gates": _gate_summary(contrasts, reports, seeds),
        "pending_controls": [
            "exact duplicate",
            "aligned packing",
            "normalization and Hangul-unit robustness",
            "generation UTF-8 validity",
            "padding-aware analytical and measured cost",
            "read-only ecological Korean diagnostic",
        ],
    }
    _write_json(output_root / "summary.json", summary)
    _write_observations_csv(
        output_root / "observations.csv",
        reports,
        seeds,
        policies,
    )
    print(f"wrote validated Phase 2 primary aggregates to {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase2")
    parser.add_argument("--artifact-root", default="artifacts/phase2")
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--output-root", default="results/phase2-korean-primary")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
