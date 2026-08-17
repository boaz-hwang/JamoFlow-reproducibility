#!/usr/bin/env python3
"""Audit or build a deterministic named release from Git-tracked files only.

The public archive is intentionally fail-closed: it requires a tracked LICENSE
and an explicit named-release flag. It is not an anonymous ARR artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATE_EPOCH = 1786924800
ARCHIVE_ROOT = "jamoflow-reproducibility-v1"
MAXIMUM_ARCHIVE_BYTES = 200_000_000

ALLOWED_EXACT = {
    ".gitignore",
    "CITATION.cff",
    "README.md",
    "pyproject.toml",
    "LICENSE",
    "LICENSE.md",
    "COPYING",
}
ALLOWED_PREFIXES = (
    "src/",
    "scripts/",
    "tests/",
    "paper/",
    "docs/",
    "data/manifests/",
    "data/seals/",
    "results/",
)
FORBIDDEN_PREFIXES = (
    "paper/private/",
    "results/private/",
    "data/raw/",
    "data/processed/",
    "data/cache/",
    "runs/",
    "artifacts/",
    "build/",
    "dist/",
)
FORBIDDEN_CONTENT_PATTERNS = {
    "absolute_user_path": re.compile(b"/" + b"Users/" + rb"[^/\s]+/"),
    "private_username": re.compile(b"hwang" + b"-gyeongchan", re.IGNORECASE),
    # `boaz-hwang` is the approved public GitHub identity. Only the local SSH
    # host alias is private machine configuration and must stay out.
    "private_github_ssh_alias": re.compile(b"github" + b"-boaz", re.IGNORECASE),
    "openai_api_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
}


@dataclass(frozen=True)
class ReleaseFile:
    path: str
    data: bytes
    executable: bool = False


def _run(command: list[str]) -> bytes:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def _clean_head() -> str:
    status = _run(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        raise ValueError("release audit requires a clean worktree")
    commit = _run(["git", "rev-parse", "HEAD"]).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("could not resolve the release commit")
    return commit


def _allowed(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        return False
    return path in ALLOWED_EXACT or path.startswith(ALLOWED_PREFIXES)


def load_head_files() -> tuple[str, dict[str, ReleaseFile]]:
    commit = _clean_head()
    archive = _run(["git", "archive", "--format=tar", "HEAD"])
    files: dict[str, ReleaseFile] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as value:
        for member in value.getmembers():
            if member.isdir() or not _allowed(member.name):
                continue
            if not member.isfile():
                raise ValueError(f"release input is not a regular file: {member.name}")
            handle = value.extractfile(member)
            if handle is None:
                raise ValueError(f"could not read release input: {member.name}")
            files[member.name] = ReleaseFile(
                path=member.name,
                data=handle.read(),
                executable=bool(member.mode & 0o111),
            )
    if not files:
        raise ValueError("release allowlist selected no files")
    return commit, files


def _license_path(files: Mapping[str, ReleaseFile]) -> str | None:
    for path in ("LICENSE", "LICENSE.md", "COPYING"):
        if path in files and len(files[path].data.strip()) >= 20:
            return path
    return None


def scan_forbidden_content(files: Mapping[str, ReleaseFile]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path, item in sorted(files.items()):
        if b"\x00" in item.data:
            continue
        for name, pattern in FORBIDDEN_CONTENT_PATTERNS.items():
            if pattern.search(item.data):
                findings.append({"path": path, "pattern": name})
    return findings


def release_manifest(commit: str, files: Mapping[str, ReleaseFile]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "jamoflow_named_reproducibility_archive_v1",
        "git_commit": commit,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "archive_root": ARCHIVE_ROOT,
        "anonymous_review_artifact": False,
        "raw_corpus_included": False,
        "model_checkpoints_included": False,
        "files": [
            {
                "path": path,
                "bytes": len(item.data),
                "sha256": hashlib.sha256(item.data).hexdigest(),
                "mode": "0755" if item.executable else "0644",
            }
            for path, item in sorted(files.items())
        ],
    }


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def audit(commit: str, files: Mapping[str, ReleaseFile]) -> dict[str, object]:
    findings = scan_forbidden_content(files)
    license_path = _license_path(files)
    return {
        "git_commit": commit,
        "selected_file_count": len(files),
        "selected_file_bytes": sum(len(item.data) for item in files.values()),
        "license_path": license_path,
        "forbidden_content_findings": findings,
        "public_release_ready": license_path is not None and not findings,
        "anonymous_arr_attachment_ready": False,
        "anonymous_arr_reason": (
            "The package and Python namespace identify the existing public project; "
            "a separately reviewed de-identified snapshot would be required."
        ),
    }


def artifact_readme(commit: str, license_path: str) -> bytes:
    return f"""# JamoFlow reproducibility archive

This named public-release archive was generated from regular Git-tracked files
at commit `{commit}`. File hashes are recorded in `ARTIFACT_MANIFEST.json`.

It includes source, tests, paper materials, protocol documents, manifests,
seals, and tracked aggregate results. It excludes raw/processed corpora,
checkpoints, machine-specific run artifacts, private vault content, raw model
outputs, and per-sequence losses. Licensing is defined by `{license_path}`.

This is not an anonymous ARR software attachment: the project/package identity
is searchable. Do not attach it to anonymous review without a separate
de-identification and license review.

Canonical regression command:

    PYTHONPATH=src .venv/bin/pytest -q tests
""".encode("utf-8")


def deterministic_tgz(files: Mapping[str, ReleaseFile]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=SOURCE_DATE_EPOCH) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path, item in sorted(files.items()):
                info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{path}")
                info.size = len(item.data)
                info.mtime = SOURCE_DATE_EPOCH
                info.mode = 0o755 if item.executable else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(item.data))
    return output.getvalue()


def build_named_public_archive(
    commit: str,
    files: Mapping[str, ReleaseFile],
) -> bytes:
    report = audit(commit, files)
    if report["forbidden_content_findings"]:
        raise ValueError("release inputs contain forbidden identity or credential patterns")
    license_path = report["license_path"]
    if not isinstance(license_path, str):
        raise ValueError("public release is blocked until a tracked LICENSE is selected")
    manifest = release_manifest(commit, files)
    expanded = dict(files)
    expanded["ARTIFACT_MANIFEST.json"] = ReleaseFile(
        path="ARTIFACT_MANIFEST.json",
        data=canonical_json(manifest),
    )
    expanded["ARTIFACT_README.md"] = ReleaseFile(
        path="ARTIFACT_README.md",
        data=artifact_readme(commit, license_path),
    )
    value = deterministic_tgz(expanded)
    if len(value) > MAXIMUM_ARCHIVE_BYTES:
        raise ValueError(f"release archive exceeds ARR's 200MB limit: {len(value)}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true", help="audit HEAD without writing an archive")
    mode.add_argument("--output", type=Path)
    parser.add_argument(
        "--named-public-release",
        action="store_true",
        help="acknowledge that the output is named/public, not anonymous review software",
    )
    args = parser.parse_args()
    if args.output is not None and not args.named_public_release:
        parser.error("--output requires --named-public-release")
    if args.output is None and args.named_public_release:
        parser.error("--named-public-release requires --output")

    commit, files = load_head_files()
    report = audit(commit, files)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output is None:
        return

    value = build_named_public_archive(commit, files)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"archive={output}")
    print(f"archive_bytes={len(value)}")
    print(f"archive_sha256={hashlib.sha256(value).hexdigest()}")


if __name__ == "__main__":
    main()
