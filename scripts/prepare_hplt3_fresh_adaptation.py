#!/usr/bin/env python3
"""Prepare the sealed fresh Korean vocabulary-adaptation streams."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from jamoflow.hplt3_final_test import publish_no_clobber
from hplt3_fresh_adaptation_protocol import reconstruct, serialize_seal


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/hplt3-korean-vocab-adaptation-v1.json"
ARCHIVE = ROOT / "data/raw/hplt3-korean-phase3/10_1.jsonl.zst"
PREDECESSOR_MANIFEST = ROOT / "data/manifests/hplt3-korean-phase3.json"
PREDECESSOR_SUMMARY = ROOT / "results/phase3-data/summary.json"
PREDECESSOR_INTEGRITY = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
PREDECESSOR_OUTPUT = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
FINAL_MANIFEST = ROOT / "data/manifests/hplt3-korean-final-test-v1.json"
FINAL_SEAL = ROOT / "data/seals/hplt3-korean-final-test-v1.json"
FINAL_OUTPUT = ROOT / "data/processed/hplt3-korean-final-test-v1/ko.jsonl"
OUTPUT = ROOT / "data/processed/hplt3-korean-vocab-adaptation-v1/ko.jsonl"
SEAL = ROOT / "data/seals/hplt3-korean-vocab-adaptation-v1.json"


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
        raise RuntimeError("fresh-data preparation requires the repository root")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-data preparation requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("fresh-data preparation requires a Git commit")
    return commit


def main() -> int:
    commit = _clean_commit()
    output, envelope = reconstruct(
        manifest_path=MANIFEST,
        archive_path=ARCHIVE,
        final_manifest_path=FINAL_MANIFEST,
        final_seal_path=FINAL_SEAL,
        final_output_path=FINAL_OUTPUT,
        predecessor_manifest_path=PREDECESSOR_MANIFEST,
        predecessor_summary_path=PREDECESSOR_SUMMARY,
        predecessor_integrity_path=PREDECESSOR_INTEGRITY,
        predecessor_output_path=PREDECESSOR_OUTPUT,
        preparation_git_commit=commit,
    )
    publish_no_clobber(OUTPUT, output)
    publish_no_clobber(SEAL, serialize_seal(envelope))
    if _git("rev-parse", "HEAD") != commit:
        raise RuntimeError("Git HEAD changed during fresh-data preparation")
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
