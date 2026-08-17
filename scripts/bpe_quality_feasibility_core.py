"""Data and optimizer contracts for the BPE quality-frontier feasibility gate."""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from token_frontier_core import role_name
from tokenizers import Tokenizer

QUALITY_ROLES = (
    role_name(2_048, 8),
    role_name(4_096, 12),
    role_name(8_192, 8),
    role_name(16_000, 8),
    role_name(32_000, 8),
    role_name(64_000, 8),
)
TRAIN_BYTES = 128_000_000
CALIBRATION_BYTES = 8_000_000
SEQUENCE_LENGTH = 512
EFFECTIVE_BATCH_SIZE = 32
TRAIN_MICROBATCH_BY_VOCABULARY = {
    2_048: 32,
    4_096: 16,
    8_192: 8,
    16_000: 4,
    32_000: 2,
    64_000: 1,
}
EVALUATION_BATCH_BY_VOCABULARY = {
    2_048: 64,
    4_096: 32,
    8_192: 16,
    16_000: 8,
    32_000: 4,
    64_000: 2,
}
WARMUP_EFFECTIVE_STEPS = 1
MEASURED_EFFECTIVE_STEPS = 3
WARMUP_EVALUATION_BATCHES = 1
MEASURED_EVALUATION_BATCHES = 3
CAMPAIGN_HOUR_LIMIT = 24.0
DRIVER_MEMORY_FRACTION_LIMIT = 0.75
CANDIDATE_TRAIN_BYTE_BUDGETS = (128_000_000, 64_000_000, 32_000_000)


@dataclass(frozen=True, slots=True)
class TokenStreamInventory:
    raw_stream_bytes: int
    complete_utf8_bytes: int
    trailing_incomplete_utf8_bytes: int
    token_count: int
    full_sequence_count: int
    dropped_token_count: int
    predicted_target_raw_bytes: int
    token_ids_sha256: str
    first_batch_token_count: int
    first_batch_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _complete_utf8_prefix(raw: bytes) -> tuple[bytes, bytes]:
    try:
        raw.decode("utf-8", errors="strict")
        return raw, b""
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(raw):
            raise ValueError("BPE quality stream contains invalid interior UTF-8") from error
        if not 1 <= len(raw) - error.start <= 3:
            raise ValueError("BPE quality stream has an invalid UTF-8 suffix")
        prefix = raw[: error.start]
        prefix.decode("utf-8", errors="strict")
        return prefix, raw[error.start :]


def _canonical_pieces(raw: bytes):
    complete, _ = _complete_utf8_prefix(raw)
    for index, row in enumerate(complete.split(b"\n")):
        yield row if index == 0 else b"\n" + row


def encode_stream_to_memmap(
    raw: bytes,
    tokenizer: Tokenizer,
    token_bytes: Sequence[bytes],
    *,
    first_batch_token_count: int,
) -> tuple[TokenStreamInventory, np.memmap, str]:
    """Encode a raw stream once into a temporary int64 memmap plus inventory."""

    if first_batch_token_count <= 0 or first_batch_token_count % SEQUENCE_LENGTH:
        raise ValueError("BPE quality first batch must contain complete sequences")
    complete, trailing = _complete_utf8_prefix(raw)
    handle = tempfile.NamedTemporaryFile(prefix="jamoflow-bpe-quality-", suffix=".ids", delete=False)
    path = handle.name
    handle.close()
    token_count = 0
    predicted_target_raw_bytes = 0
    stream_digest = hashlib.sha256(b"JamoFlow/bpe-quality-token-stream/v1\0")
    rendered_digest = hashlib.sha256()
    first_ids: list[int] = []
    with open(path, "wb") as output:
        for piece in _canonical_pieces(raw):
            if not piece:
                continue
            text = piece.decode("utf-8", errors="strict")
            ids = tuple(
                int(value)
                for value in tokenizer.encode(text, add_special_tokens=False).ids
            )
            rendered = b"".join(token_bytes[value] for value in ids)
            if not ids or rendered != piece or tokenizer.decode(list(ids)) != text:
                raise ValueError("BPE quality token stream roundtrip differs")
            values = np.asarray(ids, dtype="<i8")
            output.write(values.tobytes(order="C"))
            stream_digest.update(values.astype("<u4", copy=False).tobytes(order="C"))
            rendered_digest.update(rendered)
            for value in ids:
                if token_count % SEQUENCE_LENGTH:
                    predicted_target_raw_bytes += len(token_bytes[value])
                if len(first_ids) < first_batch_token_count:
                    first_ids.append(value)
                token_count += 1
    if rendered_digest.digest() != hashlib.sha256(complete).digest():
        raise AssertionError("BPE quality reconstructed stream hash differs")
    stream_digest.update(token_count.to_bytes(8, "big"))
    full_sequences = token_count // SEQUENCE_LENGTH
    usable_tokens = full_sequences * SEQUENCE_LENGTH
    if usable_tokens < first_batch_token_count or len(first_ids) != first_batch_token_count:
        raise ValueError("BPE quality stream is too short for its fixed first batch")
    dropped = token_count - usable_tokens
    memory = np.memmap(path, dtype="<i8", mode="r", shape=(token_count,))
    if dropped > 1:
        predicted_target_raw_bytes -= sum(
            len(token_bytes[int(value)]) for value in memory[-dropped + 1 :]
        )
    first = np.asarray(first_ids, dtype=np.int64)
    first_digest = hashlib.sha256()
    first_digest.update(str(first.dtype).encode("ascii"))
    first_digest.update(np.asarray(first.shape, dtype=np.int64).tobytes())
    first_digest.update(first.tobytes(order="C"))
    inventory = TokenStreamInventory(
        raw_stream_bytes=len(raw),
        complete_utf8_bytes=len(complete),
        trailing_incomplete_utf8_bytes=len(trailing),
        token_count=token_count,
        full_sequence_count=full_sequences,
        dropped_token_count=dropped,
        predicted_target_raw_bytes=predicted_target_raw_bytes,
        token_ids_sha256=stream_digest.hexdigest(),
        first_batch_token_count=first_batch_token_count,
        first_batch_sha256=first_digest.hexdigest(),
    )
    return inventory, memory, path


def first_sequence_batch(memory: np.ndarray, batch_size: int) -> np.ndarray:
    required = batch_size * SEQUENCE_LENGTH
    if memory.ndim != 1 or len(memory) < required or batch_size <= 0:
        raise ValueError("BPE quality first sequence batch differs")
    return np.asarray(memory[:required], dtype=np.int64).reshape(batch_size, SEQUENCE_LENGTH)


def projected_optimizer_steps(full_sequence_count: int, raw_byte_budget: int) -> int:
    if full_sequence_count <= 0 or raw_byte_budget not in CANDIDATE_TRAIN_BYTE_BUDGETS:
        raise ValueError("BPE quality projection input differs")
    return math.ceil(
        full_sequence_count
        * raw_byte_budget
        / TRAIN_BYTES
        / EFFECTIVE_BATCH_SIZE
    )


def quality_role_contract(role: str, vocabulary_size: int) -> dict[str, int]:
    microbatch = TRAIN_MICROBATCH_BY_VOCABULARY[vocabulary_size]
    if EFFECTIVE_BATCH_SIZE % microbatch:
        raise AssertionError("BPE quality microbatch does not divide effective batch")
    return {
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_BY_VOCABULARY[vocabulary_size],
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // microbatch,
        "role": role,
        "train_microbatch_size": microbatch,
        "vocabulary_size": vocabulary_size,
    }


def validate_inventory(value: Mapping[str, Any]) -> None:
    expected = set(TokenStreamInventory.__dataclass_fields__)
    if set(value) != expected:
        raise ValueError("BPE quality token inventory schema differs")
    for key in expected - {"token_ids_sha256", "first_batch_sha256"}:
        if not isinstance(value[key], int) or value[key] < 0:
            raise ValueError("BPE quality token inventory count differs")
    for key in ("token_ids_sha256", "first_batch_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64:
            raise ValueError("BPE quality token inventory hash differs")
