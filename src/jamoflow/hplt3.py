"""Reproducible, text-safe preparation of a bounded HPLT3 sample.

The raw archive and derived JSONL are intentionally kept below ignored data
directories.  Tracked artifacts contain only source and aggregate integrity
metadata, never document text or per-document identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import heapq
import io
import json
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Literal, Mapping
import unicodedata
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .corpus import Record, SplitName, split_for_record, stable_record_id


CHUNK_BYTES = 4 * 1024 * 1024
SPLITS: tuple[SplitName, ...] = ("train", "calibration", "test")


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    content_length: int
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True, slots=True)
class Candidate:
    rank: int
    digest: bytes
    raw: bytes

    @property
    def stream_bytes(self) -> int:
        # One byte is reserved for the separator.  Final accounting corrects
        # the first-record overestimate in each split.
        return len(self.raw) + 1


@dataclass(slots=True)
class ScanStatistics:
    source_lines: int = 0
    parsed_records: int = 0
    invalid_json: int = 0
    missing_text: int = 0
    invalid_utf8: int = 0
    empty_text: int = 0
    too_short: int = 0
    too_long: int = 0
    exact_duplicates: int = 0
    eligible_records: int = 0
    eligible_text_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class BottomHashSampler:
    """Keep a bounded low-hash reservoir large enough for byte quotas."""

    def __init__(
        self,
        quotas: Mapping[SplitName, int],
        *,
        reserve_multiplier: float,
        maximum_document_bytes: int,
    ) -> None:
        if reserve_multiplier < 1.0:
            raise ValueError("reserve multiplier must be at least one")
        if maximum_document_bytes <= 0:
            raise ValueError("maximum document bytes must be positive")
        self.quotas = {split: int(quotas[split]) for split in SPLITS}
        if any(value <= 0 for value in self.quotas.values()):
            raise ValueError("all split quotas must be positive")
        self.reserve_bytes = {
            split: int(self.quotas[split] * reserve_multiplier)
            + maximum_document_bytes
            + 1
            for split in SPLITS
        }
        self.heaps: dict[SplitName, list[tuple[int, bytes, Candidate]]] = {
            split: [] for split in SPLITS
        }
        self.heap_bytes: dict[SplitName, int] = {split: 0 for split in SPLITS}

    def add(self, split: SplitName, candidate: Candidate) -> None:
        heap = self.heaps[split]
        heapq.heappush(heap, (-candidate.rank, candidate.digest, candidate))
        self.heap_bytes[split] += candidate.stream_bytes
        reserve = self.reserve_bytes[split]
        while heap:
            largest = heap[0][2]
            if self.heap_bytes[split] - largest.stream_bytes < reserve:
                break
            _, _, removed = heapq.heappop(heap)
            self.heap_bytes[split] -= removed.stream_bytes

    def finalize(self) -> dict[SplitName, list[Candidate]]:
        selected: dict[SplitName, list[Candidate]] = {}
        for split in SPLITS:
            ordered = sorted(
                (entry[2] for entry in self.heaps[split]),
                key=lambda item: (item.rank, item.digest),
            )
            chosen: list[Candidate] = []
            available = 0
            for candidate in ordered:
                available += len(candidate.raw) + (1 if chosen else 0)
                chosen.append(candidate)
                if available >= self.quotas[split]:
                    break
            if available < self.quotas[split]:
                raise ValueError(
                    f"{split} has only {available:,} sampled stream bytes; "
                    f"need {self.quotas[split]:,}"
                )
            selected[split] = chosen
        return selected


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_metadata(url: str, *, timeout: int = 60) -> SourceMetadata:
    request = Request(
        url,
        method="HEAD",
        headers={"User-Agent": "JamoFlow-research/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is None:
            raise ValueError("source response has no Content-Length")
        return SourceMetadata(
            content_length=int(length),
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


def validate_source_metadata(
    actual: SourceMetadata,
    expected: Mapping[str, object],
) -> None:
    expected_bytes = int(expected["expected_bytes"])
    if actual.content_length != expected_bytes:
        raise ValueError(
            f"source length changed: expected {expected_bytes}, "
            f"got {actual.content_length}"
        )
    for field in ("etag", "last_modified"):
        expected_value = expected.get(field)
        actual_value = getattr(actual, field)
        if expected_value is not None and actual_value != expected_value:
            raise ValueError(
                f"source {field} changed: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )


def download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    timeout: int = 120,
) -> None:
    """Download to ``.part`` and resume only after validating Range semantics."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_bytes:
            raise ValueError(
                f"existing source has {destination.stat().st_size} bytes, "
                f"expected {expected_bytes}"
            )
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    current = partial.stat().st_size if partial.exists() else 0
    if current > expected_bytes:
        raise ValueError("partial download is larger than the pinned source")
    if current == expected_bytes:
        partial.replace(destination)
        return

    headers = {"User-Agent": "JamoFlow-research/0.1"}
    if current:
        headers["Range"] = f"bytes={current}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as exc:
        if exc.code == 416 and current == expected_bytes:
            partial.replace(destination)
            return
        raise

    with response:
        status = getattr(response, "status", response.getcode())
        if current and status != 206:
            raise ValueError("server ignored Range request; refusing to append")
        if not current and status not in (200, 206):
            raise ValueError(f"unexpected download status {status}")
        mode = "ab" if current else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)

    actual = partial.stat().st_size
    if actual != expected_bytes:
        raise ValueError(
            f"partial download has {actual} bytes after transfer; "
            f"expected {expected_bytes}"
        )
    partial.replace(destination)


def iter_zstd_jsonl_lines(path: Path) -> Iterator[bytes]:
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "HPLT3 preparation requires zstandard==0.25.0"
        ) from exc

    with path.open("rb") as compressed:
        reader: BinaryIO = zstandard.ZstdDecompressor().stream_reader(compressed)
        with reader, io.BufferedReader(reader) as buffered:
            for raw_line in buffered:
                if raw_line.strip():
                    yield raw_line


def _candidate_for_text(text: str, salt: bytes) -> tuple[SplitName, Candidate]:
    raw = text.encode("utf-8", errors="strict")
    record_id = stable_record_id(raw)
    record = Record(
        record_id=record_id,
        source="hplt3",
        ordinal=0,
        raw=raw,
        text=text,
    )
    split = split_for_record(record)
    digest = bytes.fromhex(record_id)
    rank = int.from_bytes(sha256(salt + digest).digest(), "big")
    return split, Candidate(rank=rank, digest=digest, raw=raw)


def sample_hplt_lines(
    lines: Iterable[bytes],
    *,
    quotas: Mapping[SplitName, int],
    salt: str,
    minimum_document_bytes: int,
    maximum_document_bytes: int,
    reserve_multiplier: float = 2.0,
) -> tuple[dict[SplitName, list[Candidate]], ScanStatistics]:
    if minimum_document_bytes <= 0:
        raise ValueError("minimum document bytes must be positive")
    if maximum_document_bytes < minimum_document_bytes:
        raise ValueError("maximum document bytes must cover the minimum")

    sampler = BottomHashSampler(
        quotas,
        reserve_multiplier=reserve_multiplier,
        maximum_document_bytes=maximum_document_bytes,
    )
    statistics = ScanStatistics()
    seen: set[bytes] = set()
    encoded_salt = salt.encode("utf-8") + b"\0"

    for raw_line in lines:
        statistics.source_lines += 1
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            statistics.invalid_json += 1
            continue
        statistics.parsed_records += 1
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            statistics.missing_text += 1
            continue
        text = value["text"]
        try:
            raw = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            statistics.invalid_utf8 += 1
            continue
        if not raw:
            statistics.empty_text += 1
            continue
        if len(raw) < minimum_document_bytes:
            statistics.too_short += 1
            continue
        if len(raw) > maximum_document_bytes:
            statistics.too_long += 1
            continue
        digest = sha256(raw).digest()
        if digest in seen:
            statistics.exact_duplicates += 1
            continue
        seen.add(digest)
        split, candidate = _candidate_for_text(text, encoded_salt)
        if candidate.digest != digest:
            raise AssertionError("candidate and dedup digests disagree")
        sampler.add(split, candidate)
        statistics.eligible_records += 1
        statistics.eligible_text_bytes += len(raw)

    return sampler.finalize(), statistics


def _stream_bytes(candidates: list[Candidate]) -> int:
    return sum(len(candidate.raw) for candidate in candidates) + max(
        0, len(candidates) - 1
    )


def _content_statistics(candidates: list[Candidate]) -> dict[str, int]:
    text_bytes = 0
    codepoints = 0
    hangul_syllables = 0
    ascii_codepoints = 0
    whitespace_codepoints = 0
    nfc_documents = 0
    latin_mixed_documents = 0
    for candidate in candidates:
        text = candidate.raw.decode("utf-8", errors="strict")
        text_bytes += len(candidate.raw)
        codepoints += len(text)
        hangul_syllables += sum("\uac00" <= char <= "\ud7a3" for char in text)
        ascii_codepoints += sum(ord(char) < 128 for char in text)
        whitespace_codepoints += sum(char.isspace() for char in text)
        nfc_documents += unicodedata.is_normalized("NFC", text)
        has_hangul = any("\uac00" <= char <= "\ud7a3" for char in text)
        has_latin = any(
            ("A" <= char <= "Z") or ("a" <= char <= "z") for char in text
        )
        latin_mixed_documents += has_hangul and has_latin
    return {
        "records": len(candidates),
        "text_bytes": text_bytes,
        "available_stream_bytes": _stream_bytes(candidates),
        "codepoints": codepoints,
        "hangul_syllables": hangul_syllables,
        "ascii_codepoints": ascii_codepoints,
        "whitespace_codepoints": whitespace_codepoints,
        "nfc_documents": nfc_documents,
        "latin_mixed_documents": latin_mixed_documents,
    }


def write_sample(
    destination: Path,
    selected: Mapping[SplitName, list[Candidate]],
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined = [candidate for split in SPLITS for candidate in selected[split]]
    combined.sort(key=lambda item: (item.rank, item.digest))
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for candidate in combined:
            payload = {
                "language": "ko",
                "text": candidate.raw.decode("utf-8", errors="strict"),
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")

    split_integrity: dict[str, object] = {}
    for split in SPLITS:
        ordered_digests = b"".join(
            candidate.digest
            for candidate in sorted(
                selected[split], key=lambda item: (item.rank, item.digest)
            )
        )
        split_integrity[split] = {
            **_content_statistics(selected[split]),
            "selected_digest_set_sha256": sha256(ordered_digests).hexdigest(),
        }
    return {
        "output_bytes": destination.stat().st_size,
        "output_sha256": hash_file(destination),
        "total_records": len(combined),
        "splits": split_integrity,
    }


def prepare_archive(
    archive_path: Path,
    output_path: Path,
    selection: Mapping[str, object],
) -> dict[str, object]:
    quotas = {
        split: int(selection["quotas"][split])  # type: ignore[index]
        for split in SPLITS
    }
    selected, scan = sample_hplt_lines(
        iter_zstd_jsonl_lines(archive_path),
        quotas=quotas,
        salt=str(selection["salt"]),
        minimum_document_bytes=int(selection["minimum_document_bytes"]),
        maximum_document_bytes=int(selection["maximum_document_bytes"]),
        reserve_multiplier=float(selection["reserve_multiplier"]),
    )
    output = write_sample(output_path, selected)
    return {
        "selection": dict(selection),
        "scan": scan.to_dict(),
        "output": output,
    }
