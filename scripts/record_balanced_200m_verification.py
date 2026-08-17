#!/usr/bin/env python3
"""Run the originally sealed verifier and publish a tracked replay receipt."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from balanced_200m_failure_analysis_core import (
    PROTOCOL_ID,
    VERIFICATION_KIND,
    canonical_bytes,
    canonical_sha256,
    validate_verification_receipt,
)
from balanced_200m_trained_core import PLAN_PATH, ROOT, TRAINING_OUTPUT_PATH

from jamoflow.hplt3 import hash_file

VERIFIER_PATH = ROOT / "scripts/verify_balanced_200m_training.py"
OUTPUT_PATH = ROOT / "results/balanced-200m-trained-screen-v1/verification.json"


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _head_blob(relative: str) -> bytes:
    return subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _never_published(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    if path.exists() or _git("log", "--all", "--format=%H", "--", relative):
        raise ValueError("balanced-200M verification receipt was already published")


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("balanced-200M verification recorder requires a clean worktree")
    _never_published(OUTPUT_PATH)
    base_commit = _git("rev-parse", "HEAD")
    plan = _read(PLAN_PATH)
    summary = _read(TRAINING_OUTPUT_PATH)
    expected_verifier_sha = plan["implementation_sha256"].get(
        "scripts/verify_balanced_200m_training.py"
    )
    if hash_file(VERIFIER_PATH) != expected_verifier_sha:
        raise ValueError("sealed balanced-200M verifier implementation differs")
    summary_relative = TRAINING_OUTPUT_PATH.relative_to(ROOT).as_posix()
    summary_commit = _git("log", "-1", "--format=%H", "--", summary_relative)
    if not summary_commit or _head_blob(summary_relative) != TRAINING_OUTPUT_PATH.read_bytes():
        raise ValueError("balanced-200M training summary is not the exact HEAD blob")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:scripts"
    completed = subprocess.run(
        (sys.executable, VERIFIER_PATH.relative_to(ROOT).as_posix()),
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    transcript = completed.stdout
    expected_lines = [
        "balanced_200m_checkpoint_replay_complete=c86",
        "balanced_200m_checkpoint_replay_complete=w72",
        "balanced_200m_training_verification=pass",
        "status=balanced_200m_quality_fail",
        f"summary_sha256={summary['summary_sha256']}",
    ]
    if completed.returncode != 0 or transcript.splitlines() != expected_lines:
        raise ValueError("sealed balanced-200M verifier did not produce the exact pass transcript")
    if _git("rev-parse", "HEAD") != base_commit or _git("status", "--porcelain"):
        raise ValueError("repository changed during balanced-200M verification")

    payload = {
        "schema_version": 1,
        "kind": VERIFICATION_KIND,
        "protocol_id": PROTOCOL_ID,
        "verification_base_git_commit": base_commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "training_summary_artifact_sha256": hash_file(TRAINING_OUTPUT_PATH),
        "training_summary_sha256": summary["summary_sha256"],
        "training_summary_git_commit": summary_commit,
        "sealed_verifier_sha256": expected_verifier_sha,
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "checkpoint_replay_roles": ["c86", "w72"],
        "independent_checkpoint_replay_pass": True,
        "quality_status": summary["status"],
        "quality": summary["quality"],
        "actual_timing_authorized": False,
        "claim_boundary": {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "actual_incremental_timing_executed": False,
            "verification_replays_full_calibration_forward": True,
        },
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    validate_verification_receipt(receipt)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())
    print("balanced_200m_verification_receipt=published")
    print(f"receipt_sha256={receipt['receipt_sha256']}")


if __name__ == "__main__":
    main()
