#!/usr/bin/env python3
"""Validate and aggregate the preregistered Phase 1 neural experiment."""

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
    aggregate_numeric_mappings,
    boundary_unicode_diagnostics,
    hierarchical_paired_bootstrap,
    nearest_boundary_displacement,
    numeric_summary,
    paired_t_interval,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import patch_boundaries_from_lengths, stream_arrays


CONTRASTS = {
    "entropy_codepoint_minus_entropy_full": (
        "entropy_codepoint",
        "entropy_full",
    ),
    "fixed_codepoint_minus_fixed_byte": (
        "fixed_codepoint",
        "fixed_byte",
    ),
    "fixed_codepoint_minus_entropy_full": (
        "fixed_codepoint",
        "entropy_full",
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
    languages: tuple[str, ...],
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, dict[str, np.ndarray]]],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    losses: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    routers: dict[int, dict[str, Any]] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    missing: list[str] = []

    for seed in seeds:
        run_directory = run_root / f"seed-{seed}"
        artifact_directory = artifact_root / f"seed-{seed}"
        router_path = run_directory / "router.json"
        diagnostic_path = run_directory / "patch-diagnostics.json"
        if not router_path.exists():
            missing.append(str(router_path))
        else:
            routers[seed] = _read_json(router_path)
        if not diagnostic_path.exists():
            missing.append(str(diagnostic_path))
        else:
            diagnostics[seed] = _read_json(diagnostic_path)

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
            if report["seed"] != seed or report["policy"] != policy:
                raise ValueError(f"identity mismatch in {report_path}")
            archive = np.load(loss_path)
            if set(archive.files) != set(languages):
                raise ValueError(f"language mismatch in {loss_path}")
            reports[seed][policy] = report
            losses[seed][policy] = {
                language: archive[language].astype(np.float64)
                for language in languages
            }

    if missing:
        raise FileNotFoundError(
            "Phase 1 is incomplete; missing:\n" + "\n".join(missing)
        )
    return reports, losses, routers, diagnostics


def _quality_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {"by_policy_language": {}, "overall": {}}
    for policy in policies:
        result["by_policy_language"][policy] = {}
        overall_values: list[float] = []
        for language in languages:
            values = [
                reports[seed][policy]["evaluation"]["test"][language]["bpb"]
                for seed in seeds
            ]
            result["by_policy_language"][policy][language] = numeric_summary(values)

        for seed in seeds:
            language_reports = reports[seed][policy]["evaluation"]["test"]
            total_nll = sum(
                language_reports[language]["nll_nats"]
                * language_reports[language]["predicted_bytes"]
                for language in languages
            )
            total_targets = sum(
                language_reports[language]["predicted_bytes"]
                for language in languages
            )
            overall_values.append(total_nll / total_targets / math.log(2))
        result["overall"][policy] = numeric_summary(overall_values)
    return result


def _contrast_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    losses: dict[int, dict[str, dict[str, np.ndarray]]],
    seeds: tuple[int, ...],
    languages: tuple[str, ...],
    repetitions: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    targets_per_sequence = 255
    for name, (left, right) in CONTRASTS.items():
        language_result: dict[str, Any] = {}
        for language_index, language in enumerate(languages):
            reported_differences = [
                reports[seed][left]["evaluation"]["test"][language]["bpb"]
                - reports[seed][right]["evaluation"]["test"][language]["bpb"]
                for seed in seeds
            ]
            sequence_differences = [
                losses[seed][left][language] - losses[seed][right][language]
                for seed in seeds
            ]
            for seed, reported, sequence_values in zip(
                seeds,
                reported_differences,
                sequence_differences,
                strict=True,
            ):
                reconstructed = float(sequence_values.mean()) / (
                    targets_per_sequence * math.log(2)
                )
                if not math.isclose(reported, reconstructed, abs_tol=2e-5):
                    raise ValueError(
                        f"loss/report mismatch for {name}/{language}/seed-{seed}: "
                        f"{reported} versus {reconstructed}"
                    )

            interval = paired_t_interval(reported_differences)
            bootstrap = hierarchical_paired_bootstrap(
                sequence_differences,
                targets_per_sequence=targets_per_sequence,
                repetitions=repetitions,
                seed=20_260_810 + language_index,
            )
            language_result[language] = {
                "seed_order": list(seeds),
                "paired_differences_bpb": reported_differences,
                "paired_t_95_interval": interval.to_dict(),
                "hierarchical_bootstrap_95_interval": bootstrap.to_dict(),
            }
        interactions: dict[str, Any] = {}
        for first, second in (("ko", "en"), ("ko", "zh"), ("zh", "en")):
            if first not in language_result or second not in language_result:
                continue
            differences = [
                first_value - second_value
                for first_value, second_value in zip(
                    language_result[first]["paired_differences_bpb"],
                    language_result[second]["paired_differences_bpb"],
                    strict=True,
                )
            ]
            interactions[f"{first}_minus_{second}"] = {
                "definition": (
                    f"policy contrast in {first} minus the same contrast in {second}"
                ),
                "seed_order": list(seeds),
                "paired_differences_of_differences_bpb": differences,
                "paired_t_95_interval": paired_t_interval(differences).to_dict(),
                "analysis_status": "prespecified language-by-policy interaction",
            }
        result[name] = {
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "languages": language_result,
            "language_interactions": interactions,
        }
    return result


def _training_summary(
    reports: dict[int, dict[str, dict[str, Any]]],
    routers: dict[int, dict[str, Any]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    main = {
        policy: aggregate_numeric_mappings(
            [reports[seed][policy]["training"] for seed in seeds]
        )
        for policy in policies
    }
    router_training = aggregate_numeric_mappings(
        [routers[seed]["training"] for seed in seeds]
    )
    router_test = {
        language: aggregate_numeric_mappings(
            [routers[seed]["evaluation"]["test"][language] for seed in seeds]
        )
        for language in languages
    }
    return {
        "main_by_policy": main,
        "router": {
            "parameters": numeric_summary(
                [routers[seed]["parameters"] for seed in seeds]
            ),
            "training": router_training,
            "test_by_language": router_test,
        },
    }


def _diagnostic_summary(
    diagnostics: dict[int, dict[str, Any]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for language in languages:
        result[language] = {}
        for policy in policies:
            result[language][policy] = aggregate_numeric_mappings(
                [
                    diagnostics[seed]["test"][language][policy]
                    for seed in seeds
                ]
            )
        result[language]["entropy_overlap"] = numeric_summary(
            [
                diagnostics[seed]["test"][language]["entropy_overlap"]
                for seed in seeds
            ]
        )
    return result


def _unicode_diagnostic_summary(
    artifact_root: Path,
    data_root: Path,
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    streams = {
        language: build_neural_stream(
            data_root / f"{language}.jsonl",
            language=language,
            split="test",
            byte_limit=int(manifest["limits"]["test"]),
            sequence_length=int(manifest["model_spec"]["sequence_length"]),
        )
        for language in languages
    }
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        cache = np.load(artifact_root / f"seed-{seed}" / "patches.npz")
        per_seed[seed] = {}
        for language in languages:
            per_seed[seed][language] = {
                policy: boundary_unicode_diagnostics(
                    cache[f"test__{language}__{policy}"],
                    streams[language].data,
                    streams[language].sequence_length,
                )
                for policy in policies
            }
            per_seed[seed][language]["entropy_displacement"] = (
                nearest_boundary_displacement(
                    cache[f"test__{language}__entropy_full"],
                    cache[f"test__{language}__entropy_codepoint"],
                )
            )

    return {
        language: {
            **{
                policy: aggregate_numeric_mappings(
                    [per_seed[seed][language][policy] for seed in seeds]
                )
                for policy in policies
            },
            "entropy_displacement": aggregate_numeric_mappings(
                [
                    per_seed[seed][language]["entropy_displacement"]
                    for seed in seeds
                ]
            ),
        }
        for language in languages
    }


def _alignment_stratum_summary(
    artifact_root: Path,
    data_root: Path,
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    """Post-hoc check that fixed-codepoint gains are not chunk-start artifacts."""

    target_scale = (
        (int(manifest["model_spec"]["sequence_length"]) - 1) * math.log(2)
    )
    result: dict[str, Any] = {}
    for language in languages:
        stream = build_neural_stream(
            data_root / f"{language}.jsonl",
            language=language,
            split="test",
            byte_limit=int(manifest["limits"]["test"]),
            sequence_length=int(manifest["model_spec"]["sequence_length"]),
        )
        _, masks = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        groups = {
            "starts_at_codepoint_boundary": masks[:, 0].astype(bool),
            "starts_inside_codepoint": ~masks[:, 0].astype(bool),
        }
        seed_effects: dict[str, list[float]] = {name: [] for name in groups}
        for seed in seeds:
            left = np.load(
                artifact_root
                / f"seed-{seed}"
                / "fixed_codepoint-test-nll.npz"
            )[language].astype(np.float64)
            right = np.load(
                artifact_root / f"seed-{seed}" / "fixed_byte-test-nll.npz"
            )[language].astype(np.float64)
            sequence_effects = (left - right) / target_scale
            for name, selected in groups.items():
                seed_effects[name].append(float(sequence_effects[selected].mean()))

        fixed_cache = np.load(
            artifact_root / f"seed-{seeds[0]}" / "patches.npz"
        )[f"test__{language}__fixed_byte"]
        boundaries = patch_boundaries_from_lengths(fixed_cache)
        rows = np.arange(len(boundaries))[:, None]
        boundaries_inside = masks[rows, boundaries] == 0
        result[language] = {
            "analysis_status": "post_hoc_confound_check",
            "contrast": "fixed_codepoint_minus_fixed_byte",
            "strata": {
                name: {
                    "sequences": int(selected.sum()),
                    "sequence_fraction": float(selected.mean()),
                    "paired_seed_effects_bpb": seed_effects[name],
                    "paired_t_95_interval": (
                        paired_t_interval(seed_effects[name]).to_dict()
                    ),
                    "fixed_byte_internal_boundary_rate": float(
                        boundaries_inside[selected].mean()
                    ),
                }
                for name, selected in groups.items()
            },
            "interpretation_guardrail": (
                "A similar effect in both strata argues against chunk-start "
                "misalignment as the sole mechanism; this post-hoc split is "
                "not a new confirmatory endpoint."
            ),
        }
    return result


def _gate_summary(
    contrasts: dict[str, Any],
    cost_benchmark: dict[str, Any] | None,
) -> dict[str, Any]:
    gate_a_languages = contrasts["entropy_codepoint_minus_entropy_full"]["languages"]
    gate_a_checks: dict[str, Any] = {}
    for language, values in gate_a_languages.items():
        differences = values["paired_differences_bpb"]
        mean = values["paired_t_95_interval"]["mean"]
        gate_a_checks[language] = {
            "mean_difference_at_most_0_015": mean <= 0.015,
            "supporting_seeds_at_most_0_015": sum(
                difference <= 0.015 for difference in differences
            ),
            "at_least_four_supporting_seeds": sum(
                difference <= 0.015 for difference in differences
            )
            >= 4,
            # This resolves the preregistration's otherwise redundant phrase
            # "no language has >0.03 harm" conservatively at seed × language
            # granularity rather than checking the already-constrained mean.
            "no_single_seed_over_0_03": max(differences) <= 0.03,
        }
    gate_a_pass = all(
        all(
            checks[key]
            for key in (
                "mean_difference_at_most_0_015",
                "at_least_four_supporting_seeds",
                "no_single_seed_over_0_03",
            )
        )
        for checks in gate_a_checks.values()
    )

    gate_b_languages = contrasts["fixed_codepoint_minus_entropy_full"]["languages"]
    gate_b_quality = {
        language: abs(values["paired_t_95_interval"]["mean"]) <= 0.02
        for language, values in gate_b_languages.items()
    }
    if cost_benchmark is None:
        gate_b_status = "pending_cost_benchmark"
        gate_b_cost = None
        gate_c_status = "pending_cost_benchmark"
        gate_c_pass = None
    else:
        analytical = cost_benchmark["analytical_flops"]
        fixed_reduction = analytical[
            "fixed_reduction_relative_to_entropy_end_to_end"
        ]
        measured_reduction = cost_benchmark[
            "batch1_fixed_codepoint_reduction_vs_entropy_full"
        ]
        gate_b_cost = {
            "analytical_fixed_reduction": fixed_reduction,
            "batch1_measured_fixed_codepoint_reduction": measured_reduction,
            "at_least_10_percent_analytical": fixed_reduction >= 0.10,
            "at_least_10_percent_measured_batch1": measured_reduction >= 0.10,
        }
        gate_b_pass = all(gate_b_quality.values()) and (
            fixed_reduction >= 0.10 or measured_reduction >= 0.10
        )
        gate_b_status = "pass" if gate_b_pass else "fail"
        router_share = analytical["router_share_of_entropy_end_to_end"]
        component_batch1 = cost_benchmark["measurements"]["1"][
            "component_timings"
        ]
        router_latency = component_batch1["router_only"]["median_ms"]
        main_latency = component_batch1["main_only/entropy_full"]["median_ms"]
        measured_router_share = router_latency / (router_latency + main_latency)
        gate_c_pass = router_share >= 0.10 or measured_router_share >= 0.10
        gate_c_status = "pass" if gate_c_pass else "fail"

    return {
        "gate_a_codepoint_restriction": {
            "status": "pass" if gate_a_pass else "fail",
            "pass": gate_a_pass,
            "checks_by_language": gate_a_checks,
            "operationalization_note": (
                "The preregistered 0.03 no-harm clause is checked on every "
                "seed-language cell, a stricter non-redundant interpretation."
            ),
        },
        "gate_b_parameter_free_boundary": {
            "status": gate_b_status,
            "pass": (
                None
                if cost_benchmark is None
                else gate_b_status == "pass"
            ),
            "quality_component_pass": all(gate_b_quality.values()),
            "within_0_02_by_language": gate_b_quality,
            "cost_component": gate_b_cost,
        },
        "gate_c_patcher_tax": {
            "status": gate_c_status,
            "pass": gate_c_pass,
        },
    }


def _write_observations_csv(
    path: Path,
    reports: dict[int, dict[str, dict[str, Any]]],
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    languages: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "policy",
                "language",
                "test_bpb",
                "test_nll_nats",
                "predicted_bytes",
                "training_seconds",
            ),
        )
        writer.writeheader()
        for seed in seeds:
            for policy in policies:
                for language in languages:
                    evaluation = reports[seed][policy]["evaluation"]["test"][language]
                    writer.writerow(
                        {
                            "seed": seed,
                            "policy": policy,
                            "language": language,
                            "test_bpb": f"{evaluation['bpb']:.12f}",
                            "test_nll_nats": f"{evaluation['nll_nats']:.12f}",
                            "predicted_bytes": evaluation["predicted_bytes"],
                            "training_seconds": (
                                f"{reports[seed][policy]['training']['elapsed_seconds']:.6f}"
                            ),
                        }
                    )


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("quick_smoke_only"):
        raise ValueError("refusing to promote a quick smoke run as Phase 1 results")
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    policies = tuple(manifest["policies"])
    languages = tuple(manifest["streams"]["test"])
    if len(seeds) != 5:
        raise ValueError(f"expected five preregistered seeds, found {len(seeds)}")

    reports, losses, routers, diagnostics = _load_complete_runs(
        run_root,
        artifact_root,
        seeds,
        policies,
        languages,
    )
    contrasts = _contrast_summary(
        reports,
        losses,
        seeds,
        languages,
        args.bootstrap_repetitions,
    )
    cost_path = run_root / "cost-benchmark.json"
    cost_benchmark = _read_json(cost_path) if cost_path.exists() else None
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": _sha256(manifest_path),
            "artifact_root": str(artifact_root),
            "git_commit_at_run_start": manifest.get("git_commit"),
        },
        "design": {
            "seeds": list(seeds),
            "policies": list(policies),
            "languages": list(languages),
            "test_sequences_per_language": {
                language: manifest["streams"]["test"][language]["sequence_count"]
                for language in languages
            },
            "predicted_bytes_per_sequence": 255,
        },
        "quality": _quality_summary(reports, seeds, policies, languages),
        "contrasts": contrasts,
        "training_and_router": _training_summary(
            reports,
            routers,
            seeds,
            policies,
            languages,
        ),
        "boundary_diagnostics": _diagnostic_summary(
            diagnostics,
            seeds,
            policies,
            languages,
        ),
        "unicode_boundary_diagnostics": _unicode_diagnostic_summary(
            artifact_root,
            Path(args.data_root),
            manifest,
            seeds,
            policies,
            languages,
        ),
        "chunk_start_alignment_diagnostic": _alignment_stratum_summary(
            artifact_root,
            Path(args.data_root),
            manifest,
            seeds,
            languages,
        ),
        "decision_gates": _gate_summary(contrasts, cost_benchmark),
        "cost_benchmark": cost_benchmark,
    }
    _write_json(output_root / "summary.json", summary)
    _write_observations_csv(
        output_root / "observations.csv",
        reports,
        seeds,
        policies,
        languages,
    )
    print(f"wrote validated Phase 1 aggregate results to {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase1")
    parser.add_argument("--artifact-root", default="artifacts/phase1")
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--output-root", default="results/phase1-neural")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
