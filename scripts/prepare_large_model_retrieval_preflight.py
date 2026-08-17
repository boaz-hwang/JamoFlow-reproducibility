#!/usr/bin/env python3
"""Download and verify only the model fixed by the committed compatibility plan."""

from __future__ import annotations

import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

from large_model_retrieval_preflight import (
    MODEL_ALLOW_PATTERNS,
    PLAN_PATH,
    PRIMARY_MODEL,
    RESULT_PATH,
    ROOT,
    hash_file,
    read_plan,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_head_blob(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)
    if committed != path.read_bytes():
        raise RuntimeError(f"artifact is not the exact HEAD blob: {relative}")


def verify_snapshot(path: Path) -> None:
    for name in PRIMARY_MODEL["expected_files"]:
        candidate = path / name
        if not candidate.is_file():
            raise FileNotFoundError(f"missing pinned model file: {name}")
    weight = path / PRIMARY_MODEL["weight_filename"]
    if (
        weight.stat().st_size != PRIMARY_MODEL["weight_bytes"]
        or hash_file(weight) != PRIMARY_MODEL["weight_sha256"]
    ):
        raise ValueError("pinned primary weight content differs")


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("model preparation requires a clean worktree")
    plan = read_plan(verify_derived=True)
    _require_head_blob(PLAN_PATH)
    if RESULT_PATH.exists():
        raise FileExistsError("compatibility result already exists")
    commit = _git("rev-parse", "HEAD")
    snapshot = Path(
        snapshot_download(
            repo_id=PRIMARY_MODEL["repo_id"],
            revision=PRIMARY_MODEL["revision"],
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
        )
    )
    verify_snapshot(snapshot)
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all")
        or plan["model_selection"]["primary"] != PRIMARY_MODEL
    ):
        raise RuntimeError("repository or model plan changed during download")
    print(f"verified_repo_id={PRIMARY_MODEL['repo_id']}")
    print(f"verified_revision={PRIMARY_MODEL['revision']}")
    print(f"verified_weight_sha256={PRIMARY_MODEL['weight_sha256']}")


if __name__ == "__main__":
    main()
