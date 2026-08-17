#!/usr/bin/env python3
"""Validate local HPLT3 preparation and promote aggregate-only metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from jamoflow.neural_data import build_neural_stream


SPLITS = ("train", "calibration", "test")
FORBIDDEN_KEYS = {
    "content",
    "document_id",
    "document_ids",
    "path",
    "paths",
    "prompt",
    "prompts",
    "record_id",
    "record_ids",
    "sample",
    "samples",
    "text",
    "texts",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def privacy_walk(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden promoted key: {key}")
            privacy_walk(child)
    elif isinstance(value, list):
        for child in value:
            privacy_walk(child)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def build_summary(
    integrity: dict[str, Any],
    processed_path: Path,
    *,
    sequence_length: int = 512,
) -> dict[str, Any]:
    source = integrity["source"]
    scan = integrity["scan"]
    selection = integrity["selection"]
    output = integrity["output"]
    quotas = selection["quotas"]

    if source["bytes"] <= 0 or len(source["sha256"]) != 64:
        raise ValueError("source integrity is incomplete")
    classified = (
        scan["eligible_records"]
        + scan["empty_text"]
        + scan["too_short"]
        + scan["too_long"]
        + scan["exact_duplicates"]
        + scan["invalid_utf8"]
    )
    if classified != scan["parsed_records"] - scan["missing_text"]:
        raise ValueError("scan record accounting does not close")
    if scan["source_lines"] != scan["parsed_records"] + scan["invalid_json"]:
        raise ValueError("source line accounting does not close")

    split_summaries: dict[str, Any] = {}
    all_quota_exact = True
    for split in SPLITS:
        local = output["splits"][split]
        quota = int(quotas[split])
        if local["available_stream_bytes"] < quota:
            raise ValueError(f"{split} does not satisfy its byte quota")
        stream = build_neural_stream(
            processed_path,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=quota,
            sequence_length=sequence_length,
        )
        exact = stream.selected_bytes == quota and stream.truncated_bytes == 0
        all_quota_exact &= exact
        split_summaries[split] = {
            **local,
            "hangul_syllable_codepoint_rate": _ratio(
                local["hangul_syllables"], local["codepoints"]
            ),
            "ascii_codepoint_rate": _ratio(
                local["ascii_codepoints"], local["codepoints"]
            ),
            "whitespace_codepoint_rate": _ratio(
                local["whitespace_codepoints"], local["codepoints"]
            ),
            "nfc_document_rate": _ratio(
                local["nfc_documents"], local["records"]
            ),
            "latin_mixed_document_rate": _ratio(
                local["latin_mixed_documents"], local["records"]
            ),
            "neural_stream": {
                "selected_bytes": stream.selected_bytes,
                "sequence_count": stream.sequence_count,
                "sequence_length": stream.sequence_length,
                "sequence_starts_inside_codepoint": (
                    stream.sequence_starts_inside_codepoint
                ),
                "sequence_start_inside_codepoint_rate": _ratio(
                    stream.sequence_starts_inside_codepoint,
                    stream.sequence_count,
                ),
                "quota_exact": exact,
            },
        }

    digest_sets = [
        output["splits"][split]["selected_digest_set_sha256"]
        for split in SPLITS
    ]
    if len(set(digest_sets)) != len(digest_sets):
        raise ValueError("split digest-set summaries unexpectedly collide")
    if not all_quota_exact:
        raise ValueError("one or more neural stream quotas are not exact")

    summary = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "dataset_id": integrity["dataset_id"],
        "created_at": integrity["created_at"],
        "source": source,
        "selection": selection,
        "scan": scan,
        "output": {
            "bytes": output["output_bytes"],
            "sha256": output["output_sha256"],
            "records": output["total_records"],
        },
        "splits": split_summaries,
        "integrity": {
            "all_neural_stream_quotas_exact": all_quota_exact,
            "distinct_split_digest_set_summaries": True,
            "source_line_accounting_closed": True,
            "scan_record_accounting_closed": True,
            "raw_or_processed_text_promoted": False,
        },
    }
    privacy_walk(summary)
    return summary


def run(args: argparse.Namespace) -> int:
    integrity = _read_json(args.integrity)
    summary = build_summary(
        integrity,
        args.processed,
        sequence_length=args.sequence_length,
    )
    _write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integrity",
        type=Path,
        default=Path("data/processed/hplt3-korean-phase3/integrity.json"),
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/hplt3-korean-phase3/ko.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase3-data/summary.json"),
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
