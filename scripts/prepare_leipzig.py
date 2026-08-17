#!/usr/bin/env python3
"""Download and prepare pinned Leipzig sentence corpora without redistribution."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import tarfile
from typing import BinaryIO, Iterator
from urllib.request import Request, urlopen


CHUNK_BYTES = 1024 * 1024


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "JamoFlow-research/0.1"})
    with urlopen(request, timeout=60) as response, partial.open("wb") as output:
        while chunk := response.read(CHUNK_BYTES):
            output.write(chunk)
    partial.replace(destination)


def _sentence_member(archive: tarfile.TarFile) -> tarfile.TarInfo:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile()
        and member.name.endswith(("_sentences.txt", "-sentences.txt"))
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one *[-_]sentences.txt member, "
            f"found {len(matches)}"
        )
    return matches[0]


def iter_sentences(archive_path: Path) -> Iterator[tuple[str, str]]:
    """Yield the sentence identifier and exact UTF-8 sentence field."""

    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = _sentence_member(archive)
        extracted: BinaryIO | None = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"could not read {member.name}")
        with extracted, io.TextIOWrapper(extracted, encoding="utf-8", errors="strict") as text:
            for line_number, line in enumerate(text, start=1):
                line = line.rstrip("\r\n")
                try:
                    sentence_id, sentence = line.split("\t", 1)
                except ValueError as exc:
                    raise ValueError(
                        f"{member.name}:{line_number}: expected ID<TAB>sentence"
                    ) from exc
                if sentence:
                    yield sentence_id, sentence


def prepare_source(
    source: dict[str, object],
    raw_root: Path,
    processed_root: Path,
) -> dict[str, object]:
    archive_path = raw_root / str(source["archive"])
    if not archive_path.exists():
        print(f"downloading {source['url']}")
        download(str(source["url"]), archive_path)

    actual_bytes = archive_path.stat().st_size
    expected_bytes = int(source["expected_bytes"])
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{archive_path}: expected {expected_bytes} bytes, got {actual_bytes}"
        )

    archive_sha256 = hash_file(archive_path)
    expected_sha256 = source.get("sha256")
    if expected_sha256 and archive_sha256 != expected_sha256:
        raise ValueError(
            f"{archive_path}: expected sha256 {expected_sha256}, got {archive_sha256}"
        )

    output_path = processed_root / f"{source['language']}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for sentence_id, sentence in iter_sentences(archive_path):
            payload = {
                "id": sentence_id,
                "language": source["language"],
                "text": sentence,
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
            record_count += 1

    expected_records = int(source["expected_records"])
    if record_count != expected_records:
        raise ValueError(
            f"{archive_path}: expected {expected_records} records, got {record_count}"
        )

    output_bytes = output_path.stat().st_size
    output_sha256 = hash_file(output_path)
    expected_output_bytes = source.get("expected_output_bytes")
    if expected_output_bytes is not None and output_bytes != expected_output_bytes:
        raise ValueError(
            f"{output_path}: expected {expected_output_bytes} bytes, "
            f"got {output_bytes}"
        )
    expected_output_sha256 = source.get("expected_output_sha256")
    if expected_output_sha256 and output_sha256 != expected_output_sha256:
        raise ValueError(
            f"{output_path}: expected sha256 {expected_output_sha256}, "
            f"got {output_sha256}"
        )

    return {
        "language": source["language"],
        "source_url": source["url"],
        "archive": str(archive_path),
        "archive_bytes": actual_bytes,
        "archive_sha256": archive_sha256,
        "output": str(output_path),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "records": record_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_id = manifest["dataset_id"]
    raw_root = args.data_root / "raw" / dataset_id
    processed_root = args.data_root / "processed" / dataset_id

    prepared = [
        prepare_source(source, raw_root, processed_root)
        for source in manifest["sources"]
    ]
    generated_manifest = {
        "source_manifest": str(args.manifest),
        "dataset_id": dataset_id,
        "prepared": prepared,
    }
    output_manifest = processed_root / "integrity.json"
    output_manifest.write_text(
        json.dumps(generated_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(generated_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
