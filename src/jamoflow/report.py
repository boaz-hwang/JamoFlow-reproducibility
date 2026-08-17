"""JSON and Markdown output for Phase 0 audits."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .metrics import PolicyMetrics
from .unicode_audit import UnicodeAudit


def _display(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞" if value > 0 else "−∞"
        return f"{value:.{digits}f}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def render_markdown(
    metadata: dict[str, object],
    unicode_audit: UnicodeAudit,
    policy_metrics: Sequence[PolicyMetrics],
    calibrations: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# JamoFlow Phase 0 Audit",
        "",
        f"> 생성 시각: {metadata['created_at']}",
        "> 성격: Phase 0 reference boundary audit; neural LM 결과가 아님",
        f"> 코퍼스: {metadata['corpus_label']}",
        "",
        "## 실행 정보",
        "",
        f"- Python: `{metadata['python']}`",
        f"- Platform: `{metadata['platform']}`",
        f"- Input files: {metadata['input_files']}",
        f"- Resolved files: {metadata['resolved_file_count']} "
        f"(suffixes: {metadata['included_suffixes']}; "
        f"plain record unit: {metadata['plain_record_unit']})",
        f"- Records: {metadata['records_total']} "
        f"(train {metadata['train_records']}, calibration {metadata['calibration_records']}, test {metadata['test_records']})",
        f"- Byte n-gram: order {metadata['ngram_order']}, alpha {metadata['alpha']}",
        f"- Entropy scoring: {_display(metadata['scoring_ns_per_byte'])} ns/byte",
        "",
        "## Unicode audit",
        "",
        f"- Raw bytes: {unicode_audit.raw_bytes:,}",
        f"- Unicode codepoints: {unicode_audit.codepoints:,}",
        f"- Bytes/codepoint: {_display(unicode_audit.raw_bytes / unicode_audit.codepoints if unicode_audit.codepoints else None)}",
        f"- Invalid records: {unicode_audit.invalid_records:,}",
        f"- NFC exact records: {unicode_audit.nfc_exact_records:,}/{unicode_audit.valid_unicode_records:,}",
        f"- NFD exact records: {unicode_audit.nfd_exact_records:,}/{unicode_audit.valid_unicode_records:,}",
        f"- Mixed Hangul/CJK/Latin records: {unicode_audit.mixed_script_records:,}",
        "",
        "| Character category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(
        unicode_audit.categories.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {category} | {count:,} |")

    lines.extend(
        [
            "",
            "## Matched-rate boundary results",
            "",
            "| Group | Role | Bytes/patch | Boundary H | Oracle capture | Top-budget overlap | Top-decile recall | Mean lag | UTF-8 split | Hangul split | CJK split | Score eval/byte | Policy ns/byte |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in policy_metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    metric.comparison_group,
                    metric.role,
                    _display(metric.average_patch_bytes),
                    _display(metric.mean_boundary_entropy_bits),
                    _display(metric.oracle_entropy_capture_ratio),
                    _display(metric.top_budget_overlap),
                    _display(metric.top_decile_entropy_recall),
                    _display(metric.mean_high_entropy_patch_lag_bytes),
                    _display(metric.boundaries_inside_codepoint_rate),
                    _display(metric.boundaries_inside_hangul_syllable_rate),
                    _display(metric.boundaries_inside_cjk_ideograph_rate),
                    _display(metric.score_evaluations_per_byte),
                    _display(metric.policy_runtime_median_ns_per_byte, 1),
                ]
            )
            + " |"
        )

    interval_metrics = [metric for metric in policy_metrics if metric.bootstrap_95]
    if interval_metrics:
        lines.extend(
            [
                "",
                "## Record-bootstrap 95% intervals",
                "",
                f"> Repeats: {metadata['bootstrap_repeats']}; seed: {metadata['bootstrap_seed']}",
                "",
                "| Group | Role | Bytes/patch | Boundary H | Top-decile recall | Mean lag | UTF-8 split | Score eval/byte |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for metric in interval_metrics:
            intervals = metric.bootstrap_95 or {}

            def interval(name: str) -> str:
                value = intervals.get(name)
                if value is None:
                    return "—"
                return f"[{_display(value[0])}, {_display(value[1])}]"

            lines.append(
                "| "
                + " | ".join(
                    [
                        metric.comparison_group,
                        metric.role,
                        interval("average_patch_bytes"),
                        interval("mean_boundary_entropy_bits"),
                        interval("top_decile_entropy_recall"),
                        interval("mean_high_entropy_patch_lag_bytes"),
                        interval("boundaries_inside_codepoint_rate"),
                        interval("score_evaluations_per_byte"),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Group | Rule bytes/patch | Full threshold | Full realized | Codepoint threshold | Codepoint realized | Orthographic threshold | Orthographic realized |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in calibrations:
        lines.append(
            f"| {item['group']} | {_display(item['target_average_patch_bytes'])} "
            f"| {_display(item['entropy_threshold'])} "
            f"| {_display(item['entropy_calibration_average_patch_bytes'])} "
            f"| {_display(item['candidate_entropy_threshold'])} "
            f"| {_display(item['candidate_calibration_average_patch_bytes'])} "
            f"| {_display(item.get('orthographic_candidate_entropy_threshold'))} "
            f"| {_display(item.get('orthographic_candidate_calibration_average_patch_bytes'))} |"
        )

    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- n-gram predictive entropy는 BLT entropy model의 대체물이 아니라 Phase 0 proxy다.",
            "- `entropy_matched`의 oracle capture와 top-budget overlap은 같은 n-gram entropy score로 boundary와 oracle을 정의하므로 구성상 1이다. 독립적인 성능 증거가 아니다.",
            "- policy runtime은 Python reference implementation 값이며 GPU kernel latency가 아니다.",
            "- UTF-8/Hangul 내부 경계 비율은 표현 경계 진단값이며, 그 자체가 모델 품질 저하를 입증하지 않는다.",
            f"- Corpus-specific: {metadata['interpretation_note']}",
            "- matched threshold는 calibration split에서만 정하고 test split에서 고정했다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    output_directory: str | Path,
    metadata: dict[str, object],
    unicode_audit: UnicodeAudit,
    policy_metrics: Sequence[PolicyMetrics],
    calibrations: Sequence[dict[str, object]],
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    metadata = dict(metadata)
    metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload = {
        "metadata": metadata,
        "unicode_audit": unicode_audit.to_dict(),
        "calibrations": list(calibrations),
        "policy_metrics": [metric.to_dict() for metric in policy_metrics],
    }

    json_path = output / "report.json"
    markdown_path = output / "report.md"
    json_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(metadata, unicode_audit, policy_metrics, calibrations),
        encoding="utf-8",
    )
    return json_path, markdown_path
