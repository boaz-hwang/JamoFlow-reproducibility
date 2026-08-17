#!/usr/bin/env python3
"""Validate and aggregate Phase 2b mechanism and artifact controls."""

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

from jamoflow.phase1_analysis import (
    hierarchical_paired_bootstrap,
    numeric_summary,
    paired_t_interval,
)
from jamoflow.phase2_analysis import gate_effect_checks


SEEDS = (1729, 2718, 31415, 57721, 65537)
ALIGNED_SEEDS = (1729, 2718, 31415)
CONTROL_POLICIES = (
    "causal_grid_early2",
    "causal_grid_delayed2",
    "causal_placebo_grid",
    "causal_whitespace_grid",
)
CONTRASTS = {
    "eojeol_minus_delayed2": (
        ("primary", "causal_eojeol_grid"),
        ("control", "causal_grid_delayed2"),
    ),
    "eojeol_minus_placebo": (
        ("primary", "causal_eojeol_grid"),
        ("control", "causal_placebo_grid"),
    ),
    "eojeol_minus_whitespace": (
        ("primary", "causal_eojeol_grid"),
        ("control", "causal_whitespace_grid"),
    ),
    "whitespace_minus_delayed2": (
        ("control", "causal_whitespace_grid"),
        ("control", "causal_grid_delayed2"),
    ),
    "early2_minus_codepoint": (
        ("control", "causal_grid_early2"),
        ("primary", "causal_codepoint_grid"),
    ),
    "delayed2_minus_codepoint": (
        ("control", "causal_grid_delayed2"),
        ("primary", "causal_codepoint_grid"),
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


def _load_report_and_loss(
    report_path: Path,
    loss_path: Path,
    seed: int,
    policy: str,
) -> tuple[dict[str, Any], np.ndarray]:
    report = _read_json(report_path)
    if report.get("seed") != seed or report.get("policy") != policy:
        raise ValueError(f"identity mismatch in {report_path}")
    with np.load(loss_path) as archive:
        if archive.files != ["ko"]:
            raise ValueError(f"unexpected loss keys in {loss_path}")
        losses = archive["ko"].astype(np.float64)
    return report, losses


def _load_runs(
    primary_run_root: Path,
    primary_artifact_root: Path,
    control_run_root: Path,
    control_artifact_root: Path,
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, np.ndarray]],
]:
    primary_reports: dict[int, dict[str, dict[str, Any]]] = {}
    primary_losses: dict[int, dict[str, np.ndarray]] = {}
    control_reports: dict[int, dict[str, dict[str, Any]]] = {}
    control_losses: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        primary_reports[seed] = {}
        primary_losses[seed] = {}
        for policy in ("causal_codepoint_grid", "causal_eojeol_grid"):
            report, loss = _load_report_and_loss(
                primary_run_root / f"seed-{seed}" / f"{policy}.json",
                primary_artifact_root
                / f"seed-{seed}"
                / f"{policy}-test-nll.npz",
                seed,
                policy,
            )
            primary_reports[seed][policy] = report
            primary_losses[seed][policy] = loss

        control_reports[seed] = {}
        control_losses[seed] = {}
        for policy in CONTROL_POLICIES:
            report, loss = _load_report_and_loss(
                control_run_root
                / f"mechanism-seed-{seed}"
                / f"{policy}.json",
                control_artifact_root
                / f"mechanism-seed-{seed}"
                / f"{policy}-test-nll.npz",
                seed,
                policy,
            )
            control_reports[seed][policy] = report
            control_losses[seed][policy] = loss
    return primary_reports, primary_losses, control_reports, control_losses


def _contrast_summary(
    primary_reports: dict[int, dict[str, dict[str, Any]]],
    primary_losses: dict[int, dict[str, np.ndarray]],
    control_reports: dict[int, dict[str, dict[str, Any]]],
    control_losses: dict[int, dict[str, np.ndarray]],
    repetitions: int,
) -> dict[str, Any]:
    report_sources = {"primary": primary_reports, "control": control_reports}
    loss_sources = {"primary": primary_losses, "control": control_losses}
    result: dict[str, Any] = {}
    scale = 255 * math.log(2)
    for contrast_index, (name, (left_key, right_key)) in enumerate(
        CONTRASTS.items()
    ):
        left_source, left_policy = left_key
        right_source, right_policy = right_key
        effects = [
            report_sources[left_source][seed][left_policy]["evaluation"]["test"][
                "bpb"
            ]
            - report_sources[right_source][seed][right_policy]["evaluation"]["test"][
                "bpb"
            ]
            for seed in SEEDS
        ]
        sequence_differences = [
            loss_sources[left_source][seed][left_policy]
            - loss_sources[right_source][seed][right_policy]
            for seed in SEEDS
        ]
        for seed, effect, values in zip(
            SEEDS,
            effects,
            sequence_differences,
            strict=True,
        ):
            reconstructed = float(values.mean()) / scale
            if not math.isclose(effect, reconstructed, abs_tol=2e-5):
                raise ValueError(
                    f"loss/report mismatch for {name}/seed-{seed}: "
                    f"{effect} versus {reconstructed}"
                )
        result[name] = {
            "left": {"source": left_source, "policy": left_policy},
            "right": {"source": right_source, "policy": right_policy},
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(SEEDS),
            "paired_differences_bpb": effects,
            "negative_seed_count": sum(value < 0 for value in effects),
            "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            "hierarchical_bootstrap_95_interval": hierarchical_paired_bootstrap(
                sequence_differences,
                targets_per_sequence=255,
                repetitions=repetitions,
                seed=20_260_820 + contrast_index,
            ).to_dict(),
        }
    return result


def _aligned_summary(
    run_root: Path,
) -> dict[str, Any]:
    effects: list[float] = []
    observations: list[dict[str, float | int]] = []
    for seed in ALIGNED_SEEDS:
        directory = run_root / f"aligned-seed-{seed}"
        fixed = _read_json(directory / "aligned_fixed_byte_6.json")
        codepoint = _read_json(directory / "aligned_causal_codepoint_grid.json")
        effect = (
            codepoint["evaluation"]["test"]["bpb"]
            - fixed["evaluation"]["test"]["bpb"]
        )
        effects.append(effect)
        observations.append(
            {
                "seed": seed,
                "fixed_byte_bpb": fixed["evaluation"]["test"]["bpb"],
                "causal_codepoint_bpb": codepoint["evaluation"]["test"]["bpb"],
                "difference_bpb": effect,
            }
        )
        if fixed["initialization_sha256"] != codepoint["initialization_sha256"]:
            raise ValueError(f"aligned initialization mismatch for seed {seed}")
        if fixed["training_order_sha256"] != codepoint["training_order_sha256"]:
            raise ValueError(f"aligned order mismatch for seed {seed}")
    return {
        "observations": observations,
        "paired_t_95_interval": paired_t_interval(effects).to_dict(),
        "all_seed_directions_negative": all(value < 0 for value in effects),
        "mean_direction_negative": float(np.mean(effects)) < 0,
    }


def _integrity_summary(
    primary_reports: dict[int, dict[str, dict[str, Any]]],
    control_reports: dict[int, dict[str, dict[str, Any]]],
    mechanism_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    all_initializations = True
    all_orders = True
    all_exact_rate = True
    for seed in SEEDS:
        reports = [
            primary_reports[seed]["causal_codepoint_grid"],
            primary_reports[seed]["causal_eojeol_grid"],
            *[control_reports[seed][policy] for policy in CONTROL_POLICIES],
        ]
        initialization_hashes = {report["initialization_sha256"] for report in reports}
        order_hashes = {report["training_order_sha256"] for report in reports}
        exact_rate = all(
            report["patch_diagnostics"][split]["mean_data_patches"] == 43.0
            for report in reports
            for split in ("train", "calibration", "test")
        )
        all_initializations &= len(initialization_hashes) == 1
        all_orders &= len(order_hashes) == 1
        all_exact_rate &= exact_rate
        by_seed[str(seed)] = {
            "identical_initialization": len(initialization_hashes) == 1,
            "identical_training_order": len(order_hashes) == 1,
            "all_policy_split_rates_exactly_43": exact_rate,
        }

    primary_c2_hashes = {
        split: {
            primary_reports[seed]["causal_eojeol_grid"]["patch_matrix_sha256"][split]
            for seed in SEEDS
        }
        for split in ("train", "calibration", "test")
    }
    reference_matches = all(
        len(primary_c2_hashes[split]) == 1
        and next(iter(primary_c2_hashes[split]))
        == mechanism_diagnostics["splits"][split][
            "causal_eojeol_grid_reference"
        ]["matrix_sha256"]
        for split in ("train", "calibration", "test")
    )
    return {
        "all_initializations_paired": all_initializations,
        "all_training_orders_paired": all_orders,
        "all_compared_rates_exactly_43": all_exact_rate,
        "rebuilt_c2_matrix_matches_primary": reference_matches,
        "by_seed": by_seed,
    }


def _gate_summary(
    contrasts: dict[str, Any],
    aligned: dict[str, Any],
    duplicate: dict[str, Any],
    primary_summary: dict[str, Any],
) -> dict[str, Any]:
    delayed = contrasts["eojeol_minus_delayed2"]
    placebo = contrasts["eojeol_minus_placebo"]
    delayed_checks = gate_effect_checks(
        delayed["paired_differences_bpb"],
        maximum_mean=-0.003,
        interval_upper=delayed["paired_t_95_interval"]["upper"],
    )
    placebo_checks = gate_effect_checks(
        placebo["paired_differences_bpb"],
        maximum_mean=-0.003,
        interval_upper=placebo["paired_t_95_interval"]["upper"],
    )
    primary_codepoint_effect = abs(
        primary_summary["contrasts"]["causal_codepoint_minus_fixed_byte"][
            "paired_t_95_interval"
        ]["mean"]
    )
    duplicate_difference = abs(
        duplicate["comparison_to_primary"]["test_bpb_difference"]
    )
    duplicate_ratio = duplicate_difference / primary_codepoint_effect
    duplicate_ok = (
        duplicate_difference <= 0.001
        and duplicate_ratio <= 0.5
    )
    gate_d_pass = (
        primary_summary["decision_gates"]["gate_d_causal_replication"][
            "primary_arbitrary_packing"
        ]["primary_effect_pass"]
        and aligned["mean_direction_negative"]
        and duplicate_ok
    )
    mechanism_pass = bool(
        delayed_checks["primary_effect_pass"]
        and placebo_checks["primary_effect_pass"]
    )
    return {
        "gate_d_causal_replication": {
            "status": "pass" if gate_d_pass else "fail",
            "pass": gate_d_pass,
            "aligned_direction_negative": aligned["mean_direction_negative"],
            "duplicate_noise_ok": duplicate_ok,
            "duplicate_absolute_bpb_difference": duplicate_difference,
            "duplicate_to_primary_effect_ratio": duplicate_ratio,
        },
        "gate_e_strengthened_mechanism_component": {
            "status": "pass" if mechanism_pass else "fail",
            "pass": mechanism_pass,
            "eojeol_vs_delayed2": delayed_checks,
            "eojeol_vs_placebo": placebo_checks,
            "full_gate_e_status": "pending_ecological_external_regression_check",
        },
    }


def _write_observations_csv(
    path: Path,
    primary_reports: dict[int, dict[str, dict[str, Any]]],
    control_reports: dict[int, dict[str, dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "source", "policy", "test_bpb", "training_seconds"),
        )
        writer.writeheader()
        for seed in SEEDS:
            for source, reports in (
                ("primary_reused", primary_reports),
                ("phase2b_new", control_reports),
            ):
                for policy, report in reports[seed].items():
                    writer.writerow(
                        {
                            "seed": seed,
                            "source": source,
                            "policy": policy,
                            "test_bpb": f"{report['evaluation']['test']['bpb']:.12f}",
                            "training_seconds": f"{report['training']['elapsed_seconds']:.6f}",
                        }
                    )


def run(args: argparse.Namespace) -> int:
    primary_run_root = Path(args.primary_run_root)
    primary_artifact_root = Path(args.primary_artifact_root)
    control_run_root = Path(args.control_run_root)
    control_artifact_root = Path(args.control_artifact_root)
    output_root = Path(args.output_root)
    manifest_path = control_run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("quick_smoke_only"):
        raise ValueError("refusing to promote quick controls as full evidence")
    if tuple(manifest["mechanism_seeds"]) != SEEDS:
        raise ValueError("mechanism seed mismatch")
    if tuple(manifest["aligned_seeds"]) != ALIGNED_SEEDS:
        raise ValueError("aligned seed mismatch")

    primary_reports, primary_losses, control_reports, control_losses = _load_runs(
        primary_run_root,
        primary_artifact_root,
        control_run_root,
        control_artifact_root,
    )
    contrasts = _contrast_summary(
        primary_reports,
        primary_losses,
        control_reports,
        control_losses,
        args.bootstrap_repetitions,
    )
    mechanism_path = control_run_root / "mechanism-patch-diagnostics.json"
    mechanism = _read_json(mechanism_path)
    duplicate = _read_json(control_run_root / "duplicate-seed-1729.json")
    aligned = _aligned_summary(control_run_root)
    primary_summary = _read_json(
        Path(args.primary_summary)
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 2b post-primary mechanism and artifact controls",
        "source": {
            "control_manifest": str(manifest_path),
            "control_manifest_sha256": _sha256(manifest_path),
            "git_commit_at_run_start": manifest.get("git_commit"),
        },
        "design": {
            "mechanism_seeds": list(SEEDS),
            "aligned_seeds": list(ALIGNED_SEEDS),
            "new_control_policies": list(CONTROL_POLICIES),
            "reused_primary_policies": [
                "causal_codepoint_grid",
                "causal_eojeol_grid",
            ],
        },
        "integrity": _integrity_summary(
            primary_reports,
            control_reports,
            mechanism,
        ),
        "quality": {
            "primary_reused": {
                policy: numeric_summary(
                    [
                        primary_reports[seed][policy]["evaluation"]["test"]["bpb"]
                        for seed in SEEDS
                    ]
                )
                for policy in ("causal_codepoint_grid", "causal_eojeol_grid")
            },
            "controls": {
                policy: numeric_summary(
                    [
                        control_reports[seed][policy]["evaluation"]["test"]["bpb"]
                        for seed in SEEDS
                    ]
                )
                for policy in CONTROL_POLICIES
            },
        },
        "contrasts": contrasts,
        "mechanism_patch_diagnostics": mechanism,
        "duplicate_control": duplicate,
        "aligned_packing": {
            "packing_metadata": manifest["aligned_packing"],
            "quality": aligned,
        },
        "decision_gates": _gate_summary(
            contrasts,
            aligned,
            duplicate,
            primary_summary,
        ),
    }
    _write_json(output_root / "summary.json", summary)
    _write_observations_csv(
        output_root / "observations.csv",
        primary_reports,
        control_reports,
    )
    print(f"wrote validated Phase 2 control aggregates to {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run-root", default="runs/phase2")
    parser.add_argument("--primary-artifact-root", default="artifacts/phase2")
    parser.add_argument("--control-run-root", default="runs/phase2-controls")
    parser.add_argument("--control-artifact-root", default="artifacts/phase2-controls")
    parser.add_argument(
        "--primary-summary",
        default="results/phase2-korean-primary/summary.json",
    )
    parser.add_argument("--output-root", default="results/phase2-controls")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
