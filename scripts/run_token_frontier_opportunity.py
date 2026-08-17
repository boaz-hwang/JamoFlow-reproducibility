#!/usr/bin/env python3
"""Train and audit the sealed ordinary byte-BPE vocabulary sweep."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from jamoflow.corpus import Record, load_records, partition_records
from jamoflow.neural_data import build_neural_stream
from jamoflow.publication_bpe import audit_byte_bpe_tokenizer, byte_bpe_token_bytes
from scalar_representation_core import complete_utf8_prefix, train_exact_byte_bpe
from token_frontier_protocol import (
    CALIBRATION_BYTES,
    INTEGRITY_PATH,
    OPPORTUNITY_REPORT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    SOURCE_PATH,
    TOKENIZER_PATHS,
    TOKENIZER_ENCODE_REPETITIONS,
    VOCABULARY_SIZES,
    canonical_sha256,
    hash_file,
    json_bytes,
    read_json,
    validate_plan,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_plan_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("token frontier tokenizer audit requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    last_change = _command(
        "git", "log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))
    )
    if len(commit) != 40 or last_change != commit:
        raise ValueError("token frontier plan must be committed at current HEAD")
    return commit


def _publish_exact_or_new(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"token frontier artifact differs: {path}")
        return
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _joined_rows(records: Sequence[Record]) -> Iterable[str]:
    emitted = False
    for record in records:
        if record.text is None:
            raise ValueError("token frontier source contains an invalid record")
        yield record.text if not emitted else "\n" + record.text
        emitted = True
    if not emitted:
        raise ValueError("token frontier train split is empty")


def _train_one(
    train_records: Sequence[Record],
    calibration_text: str,
    vocabulary_size: int,
) -> tuple[dict[str, Any], bytes]:
    first = train_exact_byte_bpe(
        _joined_rows(train_records), vocabulary_size=vocabulary_size, minimum_frequency=2
    )
    second = train_exact_byte_bpe(
        _joined_rows(train_records), vocabulary_size=vocabulary_size, minimum_frequency=2
    )
    first_bytes = first.to_str(pretty=False).encode("utf-8")
    if first_bytes != second.to_str(pretty=False).encode("utf-8"):
        raise ValueError("token frontier replicate tokenizer training differs")
    if first.get_vocab_size(with_added_tokens=True) != vocabulary_size:
        raise ValueError("token frontier tokenizer did not reach requested vocabulary")
    structural = audit_byte_bpe_tokenizer(
        first,
        (calibration_text, "\x00"),
        expected_vocabulary_size=vocabulary_size,
    ).to_dict()
    encoding = first.encode(calibration_text, add_special_tokens=False)
    ids = tuple(int(value) for value in encoding.ids)
    token_bytes = byte_bpe_token_bytes(first)
    rendered = b"".join(token_bytes[value] for value in ids)
    expected = calibration_text.encode("utf-8")
    if rendered != expected or first.decode(list(ids)) != calibration_text:
        raise ValueError("token frontier calibration roundtrip differs")
    lengths = Counter(len(token_bytes[value]) for value in ids)
    encode_elapsed_ms = []
    for _ in range(TOKENIZER_ENCODE_REPETITIONS):
        started = time.perf_counter_ns()
        timed = first.encode(calibration_text, add_special_tokens=False)
        finished = time.perf_counter_ns()
        if tuple(int(value) for value in timed.ids) != ids:
            raise ValueError("token frontier timed encoding differs")
        encode_elapsed_ms.append((finished - started) / 1_000_000)
    encode_median_ms = sorted(encode_elapsed_ms)[len(encode_elapsed_ms) // 2]
    if not structural["overall_pass"]:
        raise ValueError("token frontier structural audit failed")
    return (
        {
            "bytes_per_token": len(expected) / len(ids),
            "calibration_bytes": len(expected),
            "deterministic_replicate_json_identity": True,
            "diagnostic_encode_median_ms": encode_median_ms,
            "diagnostic_encode_megabytes_per_second": (
                len(expected) / 1_000_000 / (encode_median_ms / 1_000)
            ),
            "diagnostic_encode_repetitions": TOKENIZER_ENCODE_REPETITIONS,
            "raw_token_bytes_identity": True,
            "roundtrip_identity": True,
            "structural_audit": structural,
            "token_byte_length_histogram": {
                str(length): int(count) for length, count in sorted(lengths.items())
            },
            "token_count": len(ids),
            "tokenizer_json_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "tokens_per_unicode_scalar": len(ids) / len(calibration_text),
            "vocabulary_size": vocabulary_size,
        },
        first_bytes,
    )


def main() -> None:
    commit = _require_clean_plan_commit()
    if OPPORTUNITY_REPORT_PATH.exists():
        raise FileExistsError("token frontier tokenizer report already exists")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if _command("git", "rev-parse", "HEAD^") != plan["dependencies"][
        "git_commit_before_plan"
    ]:
        raise ValueError("token frontier plan parent differs")
    if hash_file(SOURCE_PATH) != plan["dependencies"]["source_sha256"]:
        raise ValueError("token frontier source changed")
    if hash_file(INTEGRITY_PATH) != plan["dependencies"]["integrity_sha256"]:
        raise ValueError("token frontier integrity changed")
    records = load_records(
        [SOURCE_PATH], corpus_format="jsonl", text_field="text", deduplicate=True
    )
    train_records = tuple(partition_records(records)["train"])
    if len(train_records) != 5_791:
        raise ValueError("token frontier train document count differs")
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=512,
    )
    if hashlib.sha256(stream.data).hexdigest() != plan["cases"]["calibration_stream_sha256"]:
        raise ValueError("token frontier calibration stream differs")
    calibration_text, trailing = complete_utf8_prefix(stream.data)
    if len(trailing) != 1:
        raise ValueError("token frontier UTF-8 calibration suffix differs")
    metrics: dict[str, Any] = {}
    payloads = {}
    for vocabulary_size in VOCABULARY_SIZES:
        row, tokenizer_bytes = _train_one(
            train_records, calibration_text, vocabulary_size
        )
        metrics[str(vocabulary_size)] = row
        payloads[TOKENIZER_PATHS[vocabulary_size]] = tokenizer_bytes
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during token frontier tokenizer audit")
    for path, payload in payloads.items():
        _publish_exact_or_new(path, payload)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "korean_bpe_systems_frontier_tokenizer_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "calibration_complete_utf8_bytes": len(calibration_text.encode("utf-8")),
        "calibration_trailing_raw_bytes": len(trailing),
        "metrics": metrics,
        "tokenizer_artifacts": {
            str(size): {
                "path": str(TOKENIZER_PATHS[size].relative_to(ROOT)),
                "sha256": hashlib.sha256(payloads[TOKENIZER_PATHS[size]]).hexdigest(),
            }
            for size in VOCABULARY_SIZES
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    _publish_exact_or_new(OPPORTUNITY_REPORT_PATH, json_bytes(report))
    print(f"wrote {OPPORTUNITY_REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
