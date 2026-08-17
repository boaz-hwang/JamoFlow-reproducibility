#!/usr/bin/env python3
"""Build a compact, content-free summary from Stage 2 aggregate reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


LANGUAGES = {
    "ko": {"name": "Korean", "script_group": "hangul_syllable"},
    "zh": {"name": "Chinese", "script_group": "cjk_ideograph"},
    "en": {"name": "English", "script_group": None},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_metric(
    report: dict[str, Any],
    comparison_group: str,
    role: str,
) -> dict[str, Any]:
    matches = [
        metric
        for metric in report["policy_metrics"]
        if metric["comparison_group"] == comparison_group
        and metric["role"] == role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one metric for {comparison_group}/{role}, found {len(matches)}"
        )
    return matches[0]


def metric_projection(metric: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "average_patch_bytes",
        "mean_boundary_entropy_bits",
        "oracle_entropy_capture_ratio",
        "top_budget_overlap",
        "top_decile_entropy_recall",
        "mean_high_entropy_patch_lag_bytes",
        "boundaries_inside_codepoint_rate",
        "boundaries_inside_hangul_syllable_rate",
        "boundaries_inside_cjk_ideograph_rate",
        "score_evaluations_per_byte",
        "bootstrap_95",
    )
    return {key: metric.get(key) for key in keys}


def build_summary(primary_root: Path, sensitivity_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary": {},
        "sensitivity": {},
        "rate_sensitivity": {},
    }
    for language, language_info in LANGUAGES.items():
        report = load_json(primary_root / f"{language}-order4" / "report.json")
        audit = report["unicode_audit"]
        primary: dict[str, Any] = {
            "language_name": language_info["name"],
            "metadata": report["metadata"],
            "corpus": {
                "records": audit["records_total"],
                "raw_bytes": audit["raw_bytes"],
                "codepoints": audit["codepoints"],
                "bytes_per_codepoint": audit["raw_bytes"] / audit["codepoints"],
                "categories": audit["categories"],
            },
            "fixed_rule": metric_projection(
                select_metric(report, "fixed_byte_6", "rule")
            ),
            "codepoint_candidate": metric_projection(
                select_metric(report, "fixed_byte_6", "candidate_entropy_matched")
            ),
            "spacebyte_rule": metric_projection(
                select_metric(report, "spacebyte_compatible", "rule")
            ),
            "eojeol_rule": metric_projection(
                select_metric(report, "eojeol_cap_24", "rule")
            ),
        }
        if language_info["script_group"] is not None:
            primary["orthographic_candidate"] = metric_projection(
                select_metric(
                    report,
                    "fixed_byte_6",
                    "orthographic_candidate_entropy_matched",
                )
            )
            primary["script_rule"] = metric_projection(
                select_metric(report, language_info["script_group"], "rule")
            )
        summary["primary"][language] = primary

        settings = {
            "order2_alpha0.1": sensitivity_root
            / f"{language}-order2"
            / "report.json",
            "order4_alpha0.01": sensitivity_root
            / f"{language}-alpha001"
            / "report.json",
            "order4_alpha0.1": primary_root
            / f"{language}-order4"
            / "report.json",
            "order4_alpha1.0": sensitivity_root
            / f"{language}-alpha1"
            / "report.json",
        }
        summary["sensitivity"][language] = {}
        for setting, report_path in settings.items():
            sensitivity = load_json(report_path)
            projected = {
                "codepoint_candidate": metric_projection(
                    select_metric(
                        sensitivity,
                        "fixed_byte_6",
                        "candidate_entropy_matched",
                    )
                )
            }
            if language_info["script_group"] is not None:
                projected["orthographic_candidate"] = metric_projection(
                    select_metric(
                        sensitivity,
                        "fixed_byte_6",
                        "orthographic_candidate_entropy_matched",
                    )
                )
            summary["sensitivity"][language][setting] = projected

        rate_settings = {
            "stride4": (
                sensitivity_root / f"{language}-rate4" / "report.json",
                "fixed_byte_4",
            ),
            "stride6": (
                primary_root / f"{language}-order4" / "report.json",
                "fixed_byte_6",
            ),
            "stride8": (
                sensitivity_root / f"{language}-rate8" / "report.json",
                "fixed_byte_8",
            ),
        }
        summary["rate_sensitivity"][language] = {}
        for setting, (report_path, comparison_group) in rate_settings.items():
            rate_report = load_json(report_path)
            summary["rate_sensitivity"][language][setting] = metric_projection(
                select_metric(
                    rate_report,
                    comparison_group,
                    "candidate_entropy_matched",
                )
            )
    return summary


def display(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 2 Aggregate Summary",
        "",
        f"> Generated: {summary['created_at']}",
        "> Aggregate only; no corpus text or record identifiers are included.",
        "",
        "## Corpora",
        "",
        "| Language | Records | Raw bytes | Codepoints | Bytes/codepoint |",
        "|---|---:|---:|---:|---:|",
    ]
    for language, item in summary["primary"].items():
        corpus = item["corpus"]
        lines.append(
            f"| {language} | {corpus['records']:,} | {corpus['raw_bytes']:,} "
            f"| {corpus['codepoints']:,} | {display(corpus['bytes_per_codepoint'])} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-rate candidate comparison",
            "",
            "| Language | Candidate | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall | Score eval/byte | Eval reduction |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for language, item in summary["primary"].items():
        candidates = [("codepoint", item["codepoint_candidate"])]
        if "orthographic_candidate" in item:
            candidates.append(("script+delimiter", item["orthographic_candidate"]))
        for label, metric in candidates:
            evaluations = metric["score_evaluations_per_byte"]
            lines.append(
                f"| {language} | {label} | {display(metric['average_patch_bytes'])} "
                f"| {display(metric['oracle_entropy_capture_ratio'])} "
                f"| {display(metric['top_budget_overlap'])} "
                f"| {display(metric['top_decile_entropy_recall'])} "
                f"| {display(evaluations)} | {display(1.0 - evaluations)} |"
            )

    lines.extend(
        [
            "",
            "## SpaceByte-compatible structural diagnostic",
            "",
            "| Language | Bytes/patch | UTF-8 internal | Hangul internal | CJK internal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for language, item in summary["primary"].items():
        metric = item["spacebyte_rule"]
        lines.append(
            f"| {language} | {display(metric['average_patch_bytes'])} "
            f"| {display(metric['boundaries_inside_codepoint_rate'])} "
            f"| {display(metric['boundaries_inside_hangul_syllable_rate'])} "
            f"| {display(metric['boundaries_inside_cjk_ideograph_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## N-gram sensitivity: fixed-rate codepoint candidate",
            "",
            "| Language | Setting | Oracle capture | Top-budget overlap | Top-decile recall |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for language, settings in summary["sensitivity"].items():
        for setting, item in settings.items():
            metric = item["codepoint_candidate"]
            lines.append(
                f"| {language} | {setting} "
                f"| {display(metric['oracle_entropy_capture_ratio'])} "
                f"| {display(metric['top_budget_overlap'])} "
                f"| {display(metric['top_decile_entropy_recall'])} |"
            )

    lines.extend(
        [
            "",
            "## Patch-rate sensitivity: codepoint candidate",
            "",
            "| Language | Setting | Bytes/patch | Oracle capture | Top-budget overlap | Top-decile recall |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for language, settings in summary["rate_sensitivity"].items():
        for setting, metric in settings.items():
            lines.append(
                f"| {language} | {setting} "
                f"| {display(metric['average_patch_bytes'])} "
                f"| {display(metric['oracle_entropy_capture_ratio'])} "
                f"| {display(metric['top_budget_overlap'])} "
                f"| {display(metric['top_decile_entropy_recall'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.primary_root, args.sensitivity_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
