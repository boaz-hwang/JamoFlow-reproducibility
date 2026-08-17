#!/usr/bin/env python3
"""Download and deterministically sample the pinned HPLT3 Korean shard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from jamoflow.hplt3 import (
    download_with_resume,
    hash_file,
    prepare_archive,
    remote_metadata,
    validate_source_metadata,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_id = str(manifest["dataset_id"])
    source = manifest["source"]
    selection = manifest["selection"]
    raw_path = args.data_root / "raw" / dataset_id / str(source["filename"])
    processed_root = args.data_root / "processed" / dataset_id
    output_path = processed_root / "ko.jsonl"
    integrity_path = processed_root / "integrity.json"

    metadata = remote_metadata(str(source["url"]), timeout=args.timeout)
    validate_source_metadata(metadata, source)
    download_with_resume(
        str(source["url"]),
        raw_path,
        expected_bytes=int(source["expected_bytes"]),
        timeout=args.timeout,
    )
    source_sha256 = hash_file(raw_path)

    if args.download_only:
        print(
            json.dumps(
                {
                    "archive": str(raw_path),
                    "bytes": raw_path.stat().st_size,
                    "sha256": source_sha256,
                },
                indent=2,
            )
        )
        return 0

    prepared = prepare_archive(raw_path, output_path, selection)
    integrity = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "source_manifest": str(args.manifest),
        "source": {
            "url": source["url"],
            "filename": source["filename"],
            "bytes": raw_path.stat().st_size,
            "sha256": source_sha256,
            "etag": metadata.etag,
            "last_modified": metadata.last_modified,
        },
        **prepared,
    }
    _write_json(integrity_path, integrity)
    print(json.dumps(integrity, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/hplt3-korean-phase3.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--download-only", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
