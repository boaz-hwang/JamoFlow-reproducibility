#!/usr/bin/env python3
"""Validate and promote Korean normalization robustness aggregates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from jamoflow.phase1_analysis import numeric_summary, paired_t_interval


SEEDS = (1729, 2718, 31415, 57721, 65537)
CONDITIONS = ("original", "nfc", "nfd", "compatibility_jamo")
POLICIES = (
    "fixed_byte_6_rate43",
    "causal_codepoint_grid_rate43",
    "causal_whitespace_grid_rate43",
    "causal_codepoint_grid_rate28",
    "oracle_hangul_unit_grid_rate28",
)


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


def _metric_values(
    run: dict[str, Any],
    condition: str,
    policy: str,
    metric: str,
) -> list[float]:
    return [
        float(run["evaluations"][str(seed)][condition][policy][metric])
        for seed in SEEDS
    ]


def run(args: argparse.Namespace) -> int:
    run_path = Path(args.run_result)
    source = _read_json(run_path)
    if tuple(source["seeds"]) != SEEDS:
        raise ValueError("normalization seed mismatch")
    if tuple(source["conditions"]) != CONDITIONS:
        raise ValueError("normalization condition mismatch")
    if tuple(source["policies"]) != POLICIES:
        raise ValueError("normalization policy mismatch")
    if source["causality_guardrail"][
        "oracle_hangul_unit_grid_is_prefix_causal"
    ]:
        raise ValueError("oracle must not be labeled causal")

    quality: dict[str, Any] = {}
    for condition in CONDITIONS:
        quality[condition] = {}
        for policy in POLICIES:
            quality[condition][policy] = {
                metric: numeric_summary(
                    _metric_values(source, condition, policy, metric)
                )
                for metric in (
                    "bpb",
                    "bits_per_represented_source_codepoint",
                    "bits_per_represented_source_hangul_syllable",
                )
            }

    oracle_codepoint = _metric_values(
        source,
        "nfd",
        "causal_codepoint_grid_rate28",
        "bits_per_represented_source_codepoint",
    )
    oracle_unit = _metric_values(
        source,
        "nfd",
        "oracle_hangul_unit_grid_rate28",
        "bits_per_represented_source_codepoint",
    )
    oracle_differences = [
        oracle - codepoint
        for oracle, codepoint in zip(oracle_unit, oracle_codepoint, strict=True)
    ]
    oracle_relative_improvements = [
        (codepoint - oracle) / codepoint
        for oracle, codepoint in zip(oracle_unit, oracle_codepoint, strict=True)
    ]

    policy_contrasts: dict[str, Any] = {}
    for condition in CONDITIONS:
        codepoint = _metric_values(
            source,
            condition,
            "causal_codepoint_grid_rate43",
            "bits_per_represented_source_codepoint",
        )
        for label, policy in (
            ("fixed_minus_codepoint", "fixed_byte_6_rate43"),
            ("whitespace_minus_codepoint", "causal_whitespace_grid_rate43"),
        ):
            candidate = _metric_values(
                source,
                condition,
                policy,
                "bits_per_represented_source_codepoint",
            )
            differences = [
                left - right
                for left, right in zip(candidate, codepoint, strict=True)
            ]
            policy_contrasts[f"{condition}/{label}"] = {
                "seed_order": list(SEEDS),
                "paired_differences_bits_per_source_codepoint": differences,
                "paired_t_95_interval": paired_t_interval(differences).to_dict(),
            }

    degradation: dict[str, Any] = {}
    for policy in POLICIES:
        nfc = _metric_values(
            source,
            "nfc",
            policy,
            "bits_per_represented_source_codepoint",
        )
        degradation[policy] = {}
        for condition in ("nfd", "compatibility_jamo"):
            stress = _metric_values(
                source,
                condition,
                policy,
                "bits_per_represented_source_codepoint",
            )
            ratios = [
                stressed / baseline - 1
                for stressed, baseline in zip(stress, nfc, strict=True)
            ]
            degradation[policy][f"{condition}_relative_increase_vs_nfc"] = (
                numeric_summary(ratios)
            )

    diagnostics = source["condition_diagnostics"]
    matrix_identity = bool(
        diagnostics["nfc"]["oracle_equals_codepoint_matrix"]
        and diagnostics["original"]["oracle_equals_codepoint_matrix"]
    )
    exact_rate = all(
        diagnostics[condition]["policies"][policy]["minimum_data_patches"]
        == (28 if policy.endswith("rate28") else 43)
        and diagnostics[condition]["policies"][policy]["maximum_data_patches"]
        == (28 if policy.endswith("rate28") else 43)
        for condition in CONDITIONS
        for policy in POLICIES
    )
    oracle_unit_safe = (
        diagnostics["nfd"]["policies"]["oracle_hangul_unit_grid_rate28"][
            "inside_oracle_hangul_unit_boundary_rate"
        ]
        == 0.0
    )
    relative_summary = numeric_summary(oracle_relative_improvements)
    opportunity_pass = bool(
        relative_summary["mean"] >= 0.01
        and sum(value >= 0.01 for value in oracle_relative_improvements) >= 4
        and matrix_identity
        and exact_rate
        and oracle_unit_safe
    )

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": source["scope"],
        "source": {
            "run_result": str(run_path),
            "run_result_sha256": _sha256(run_path),
            "source_stream": source["source"],
        },
        "design": {
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "policies": list(POLICIES),
            "oracle_is_non_causal": True,
            "rate43_policies_are_deployable_diagnostics": True,
            "rate28_pair_is_opportunity_diagnostic_only": True,
        },
        "integrity": {
            "nfc_and_original_rate28_matrix_identity": matrix_identity,
            "all_policy_condition_rates_exact": exact_rate,
            "nfd_oracle_has_zero_unit_internal_boundaries": oracle_unit_safe,
        },
        "condition_diagnostics": diagnostics,
        "quality": quality,
        "nfd_oracle_contrast": {
            "definition": (
                "oracle_hangul_unit_grid_rate28 minus "
                "causal_codepoint_grid_rate28"
            ),
            "seed_order": list(SEEDS),
            "paired_differences_bits_per_source_codepoint": oracle_differences,
            "paired_t_95_interval": paired_t_interval(
                oracle_differences
            ).to_dict(),
            "relative_improvement_positive_favors_oracle": (
                oracle_relative_improvements
            ),
            "relative_improvement_summary": relative_summary,
        },
        "rate43_policy_contrasts": policy_contrasts,
        "normalization_degradation": degradation,
        "decision_gate_g_opportunity": {
            "status": "pass" if opportunity_pass else "fail",
            "pass": opportunity_pass,
            "mean_relative_improvement_at_least_1_percent": (
                relative_summary["mean"] >= 0.01
            ),
            "seeds_at_least_1_percent": sum(
                value >= 0.01 for value in oracle_relative_improvements
            ),
            "required_seeds": 4,
            "matrix_identity": matrix_identity,
            "exact_rate": exact_rate,
            "oracle_unit_safe": oracle_unit_safe,
            "interpretation": (
                "This gate only opens a future causal-architecture research "
                "opportunity; it never promotes the non-causal oracle itself."
            ),
        },
        "causality_guardrail": source["causality_guardrail"],
    }
    _write_json(Path(args.output), output)
    print(f"wrote validated normalization aggregates to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-result",
        default="runs/phase2-normalization/normalization-results.json",
    )
    parser.add_argument(
        "--output",
        default="results/phase2-normalization/summary.json",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
