#!/usr/bin/env python3
"""Prepare the one-shot, disjoint HPLT3 Korean final-test artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from jamoflow.hplt3_final_test import (
    publish_no_clobber,
    reconstruct_final_test,
    serialize_seal_envelope,
)


MANIFEST = Path("data/manifests/hplt3-korean-final-test-v1.json")
ARCHIVE = Path("data/raw/hplt3-korean-phase3/10_1.jsonl.zst")
PREDECESSOR_MANIFEST = Path("data/manifests/hplt3-korean-phase3.json")
PREDECESSOR_SUMMARY = Path("results/phase3-data/summary.json")
PREDECESSOR_INTEGRITY = Path(
    "data/processed/hplt3-korean-phase3/integrity.json"
)
PREDECESSOR_OUTPUT = Path("data/processed/hplt3-korean-phase3/ko.jsonl")
OUTPUT = Path("data/processed/hplt3-korean-final-test-v1/ko.jsonl")
SEAL = Path("data/seals/hplt3-korean-final-test-v1.json")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("final-test preparation requires a SHA-1 Git commit")
    return commit


def _require_clean_start() -> str:
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    ).stdout.strip()
    if Path(top_level).resolve() != Path.cwd().resolve():
        raise ValueError("run final-test preparation from the repository root")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    if status.stdout.strip():
        raise ValueError("final-test preparation requires a clean worktree")
    return _git_commit()


def main() -> int:
    preparation_commit = _require_clean_start()
    output, envelope = reconstruct_final_test(
        manifest_path=MANIFEST,
        archive_path=ARCHIVE,
        predecessor_manifest_path=PREDECESSOR_MANIFEST,
        predecessor_summary_path=PREDECESSOR_SUMMARY,
        predecessor_integrity_path=PREDECESSOR_INTEGRITY,
        predecessor_output_path=PREDECESSOR_OUTPUT,
        preparation_git_commit=preparation_commit,
    )
    publish_no_clobber(OUTPUT, output)
    seal_bytes = serialize_seal_envelope(envelope)
    publish_no_clobber(SEAL, seal_bytes)
    if _git_commit() != preparation_commit:
        raise RuntimeError("Git HEAD changed during final-test preparation")
    print(
        json.dumps(
            {
                "dataset_id": envelope["payload"]["dataset_id"],
                "evaluation_stream_bytes": envelope["payload"]["output"][
                    "evaluation_stream_bytes"
                ],
                "payload_sha256": envelope["payload_sha256"],
                "seal": str(SEAL),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
