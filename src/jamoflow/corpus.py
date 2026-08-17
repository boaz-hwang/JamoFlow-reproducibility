"""Streaming corpus records and deterministic data splits."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Iterator, Literal


CorpusFormat = Literal["auto", "plain", "jsonl"]
PlainRecordUnit = Literal["line", "file"]
SplitName = Literal["train", "calibration", "test"]


@dataclass(frozen=True, slots=True)
class Record:
    """One independently split corpus record.

    ``raw`` is always the exact byte sequence scored by byte models. For valid
    plain-text and JSONL records, ``text.encode('utf-8') == raw``. Invalid
    records retain their source bytes and set ``text`` to ``None``.
    """

    record_id: str
    source: str
    ordinal: int
    raw: bytes
    text: str | None
    error: str | None = None


def stable_record_id(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def split_for_record(
    record: Record,
    train_percent: int = 80,
    calibration_percent: int = 10,
) -> SplitName:
    if train_percent <= 0 or calibration_percent <= 0:
        raise ValueError("train and calibration percentages must be positive")
    if train_percent + calibration_percent >= 100:
        raise ValueError("train + calibration percentages must be below 100")

    bucket = int(record.record_id[:16], 16) % 10_000
    train_cut = train_percent * 100
    calibration_cut = train_cut + calibration_percent * 100
    if bucket < train_cut:
        return "train"
    if bucket < calibration_cut:
        return "calibration"
    return "test"


def _resolve_format(path: Path, corpus_format: CorpusFormat) -> Literal["plain", "jsonl"]:
    if corpus_format != "auto":
        return corpus_format
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "plain"


def _extract_json_field(value: object, dotted_field: str) -> object:
    current = value
    for part in dotted_field.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_field)
        current = current[part]
    return current


def _record_from_raw(path: Path, ordinal: int, raw: bytes) -> Record:
    try:
        text = raw.decode("utf-8", errors="strict")
        error = None
    except UnicodeDecodeError as exc:
        text = None
        error = f"utf8:{exc.start}:{exc.reason}"
    return Record(
        record_id=stable_record_id(raw),
        source=str(path),
        ordinal=ordinal,
        raw=raw,
        text=text,
        error=error,
    )


def _plain_records(path: Path, record_unit: PlainRecordUnit) -> Iterator[Record]:
    if record_unit == "file":
        raw = path.read_bytes()
        if raw.strip():
            yield _record_from_raw(path, 1, raw)
        return

    with path.open("rb") as handle:
        for ordinal, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            yield _record_from_raw(path, ordinal, raw_line)


def _jsonl_records(path: Path, text_field: str) -> Iterator[Record]:
    with path.open("rb") as handle:
        for ordinal, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                decoded = raw_line.decode("utf-8", errors="strict")
                value = json.loads(decoded)
                extracted = _extract_json_field(value, text_field)
                if not isinstance(extracted, str):
                    raise TypeError(f"field {text_field!r} is not a string")
                text = extracted
                raw = text.encode("utf-8")
                error = None
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                text = None
                raw = raw_line
                error = f"jsonl:{type(exc).__name__}:{exc}"
            yield Record(
                record_id=stable_record_id(raw),
                source=str(path),
                ordinal=ordinal,
                raw=raw,
                text=text,
                error=error,
            )


def _normalize_suffixes(suffixes: Iterable[str] | None) -> set[str]:
    if suffixes is None:
        return {".md", ".txt", ".text", ".jsonl", ".ndjson"}

    normalized = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in suffixes
    }
    if not normalized:
        raise ValueError("at least one input suffix is required")
    return normalized


def expand_input_paths(
    paths: Iterable[str | Path],
    include_suffixes: Iterable[str] | None = None,
) -> list[Path]:
    """Expand directories without reading or modifying their contents.

    ``include_suffixes`` filters recursively discovered files. Explicit file
    paths remain explicit inputs regardless of their suffix.
    """

    supported_suffixes = _normalize_suffixes(include_suffixes)
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in supported_suffixes
            )
        else:
            expanded.append(path)
    return expanded


def iter_records(
    paths: Iterable[str | Path],
    corpus_format: CorpusFormat = "auto",
    text_field: str = "text",
    plain_record_unit: PlainRecordUnit = "line",
    include_suffixes: Iterable[str] | None = None,
) -> Iterator[Record]:
    for path in expand_input_paths(paths, include_suffixes=include_suffixes):
        resolved = _resolve_format(path, corpus_format)
        if resolved == "jsonl":
            yield from _jsonl_records(path, text_field)
        else:
            yield from _plain_records(path, plain_record_unit)


def load_records(
    paths: Iterable[str | Path],
    corpus_format: CorpusFormat = "auto",
    text_field: str = "text",
    plain_record_unit: PlainRecordUnit = "line",
    deduplicate: bool = True,
    include_suffixes: Iterable[str] | None = None,
) -> list[Record]:
    records: list[Record] = []
    seen: set[str] = set()
    for record in iter_records(
        paths,
        corpus_format=corpus_format,
        text_field=text_field,
        plain_record_unit=plain_record_unit,
        include_suffixes=include_suffixes,
    ):
        if deduplicate and record.record_id in seen:
            continue
        seen.add(record.record_id)
        records.append(record)
    return records


def partition_records(records: Iterable[Record]) -> dict[SplitName, list[Record]]:
    partitions: dict[SplitName, list[Record]] = {
        "train": [],
        "calibration": [],
        "test": [],
    }
    for record in records:
        partitions[split_for_record(record)].append(record)
    return partitions
