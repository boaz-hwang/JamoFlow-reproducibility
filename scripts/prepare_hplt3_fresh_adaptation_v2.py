#!/usr/bin/env python3
"""Prepare the second sealed fresh Korean vocabulary-adaptation corpus."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from jamoflow.hplt3_final_test import publish_no_clobber
from hplt3_fresh_adaptation_v2_protocol import reconstruct, serialize_seal


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/hplt3-korean-vocab-adaptation-v2.json"
ARCHIVE = ROOT / "data/raw/hplt3-korean-phase3/10_1.jsonl.zst"
PREDECESSOR_MANIFEST = ROOT / "data/manifests/hplt3-korean-phase3.json"
PREDECESSOR_SUMMARY = ROOT / "results/phase3-data/summary.json"
PREDECESSOR_INTEGRITY = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
PREDECESSOR_OUTPUT = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
FINAL_MANIFEST = ROOT / "data/manifests/hplt3-korean-final-test-v1.json"
FINAL_SEAL = ROOT / "data/seals/hplt3-korean-final-test-v1.json"
FINAL_OUTPUT = ROOT / "data/processed/hplt3-korean-final-test-v1/ko.jsonl"
FRESH_V1_MANIFEST = ROOT / "data/manifests/hplt3-korean-vocab-adaptation-v1.json"
FRESH_V1_PROTOCOL = ROOT / "scripts/hplt3_fresh_adaptation_protocol.py"
FRESH_V1_SEAL = ROOT / "data/seals/hplt3-korean-vocab-adaptation-v1.json"
FRESH_V1_OUTPUT = ROOT / "data/processed/hplt3-korean-vocab-adaptation-v1/ko.jsonl"
OUTPUT = ROOT / "data/processed/hplt3-korean-vocab-adaptation-v2/ko.jsonl"
SEAL = ROOT / "data/seals/hplt3-korean-vocab-adaptation-v2.json"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_commit() -> str:
    if Path(_git("rev-parse", "--show-toplevel")).resolve() != ROOT:
        raise RuntimeError("fresh-v2 preparation requires the repository root")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-v2 preparation requires a clean worktree")
    if _git("log", "--all", "--format=%H", "--", str(SEAL.relative_to(ROOT))):
        raise RuntimeError("fresh-v2 seal was already published in Git history")
    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("fresh-v2 preparation requires a Git commit")
    return commit


def _reconstruct(commit: str) -> tuple[bytes, dict[str, object]]:
    return reconstruct(
        manifest_path=MANIFEST,
        archive_path=ARCHIVE,
        final_manifest_path=FINAL_MANIFEST,
        final_seal_path=FINAL_SEAL,
        final_output_path=FINAL_OUTPUT,
        predecessor_manifest_path=PREDECESSOR_MANIFEST,
        predecessor_summary_path=PREDECESSOR_SUMMARY,
        predecessor_integrity_path=PREDECESSOR_INTEGRITY,
        predecessor_output_path=PREDECESSOR_OUTPUT,
        fresh_v1_manifest_path=FRESH_V1_MANIFEST,
        fresh_v1_protocol_path=FRESH_V1_PROTOCOL,
        fresh_v1_seal_path=FRESH_V1_SEAL,
        fresh_v1_output_path=FRESH_V1_OUTPUT,
        preparation_git_commit=commit,
    )


def main() -> int:
    commit = _clean_commit()
    output, envelope = _reconstruct(commit)
    seal_bytes = serialize_seal(envelope)
    publish_no_clobber(OUTPUT, output)
    publish_no_clobber(SEAL, seal_bytes)
    if _git("rev-parse", "HEAD") != commit:
        raise RuntimeError("Git HEAD changed during fresh-v2 preparation")
    print(
        json.dumps(
            {
                "dataset_id": envelope["payload"]["dataset_id"],
                "payload_sha256": envelope["payload_sha256"],
                "seal": str(SEAL.relative_to(ROOT)),
                "splits": envelope["payload"]["splits"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
