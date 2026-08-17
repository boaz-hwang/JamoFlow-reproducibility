#!/usr/bin/env python3
"""Validate and promote an aggregate-only private ecological result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TOP_LEVEL = {
    "schema_version",
    "generated_at",
    "scope",
    "source_label",
    "privacy",
    "environment",
    "design",
    "corpus_aggregate",
    "patch_integrity",
    "quality",
    "contrasts",
    "strata",
    "decision_gate_e_ecological_component",
    "claim_guardrail",
}
FORBIDDEN_KEYS = {
    "path",
    "paths",
    "filename",
    "filenames",
    "record_id",
    "record_ids",
    "sequence_nll",
    "text",
    "content",
    "sha256",
    "hash",
}


def _walk(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden private key: {key}")
            _walk(child)
    elif isinstance(value, list):
        for child in value:
            _walk(child)
    elif isinstance(value, str):
        if value.startswith(("/", "../", "~/")):
            raise ValueError("absolute or relative private path-like string found")


def run(args: argparse.Namespace) -> int:
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if set(source) != EXPECTED_TOP_LEVEL:
        missing = sorted(EXPECTED_TOP_LEVEL - set(source))
        extra = sorted(set(source) - EXPECTED_TOP_LEVEL)
        raise ValueError(f"unexpected schema; missing={missing}, extra={extra}")
    if source["schema_version"] != 1:
        raise ValueError("unsupported ecological schema")
    privacy = source["privacy"]
    required_privacy = {
        "read_only": True,
        "raw_text_serialized": False,
        "paths_or_filenames_serialized": False,
        "record_or_sequence_metrics_serialized": False,
        "private_content_hash_serialized": False,
        "primary_evidence": False,
    }
    if privacy != required_privacy:
        raise ValueError("privacy guardrail mismatch")
    if not source["patch_integrity"]["all_policies_exactly_43"]:
        raise ValueError("cannot promote a patch-rate invariant failure")
    _walk(source)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"promoted privacy-audited ecological aggregates to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/private/phase2-ecological/summary.json",
    )
    parser.add_argument(
        "--output",
        default="results/phase2-ecological/summary.json",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
