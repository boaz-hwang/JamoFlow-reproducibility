"""Training, evaluation, and one-seed selection contracts for the BPE frontier."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from bpe_quality_feasibility_core import (
    CALIBRATION_BYTES,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_BY_VOCABULARY,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
    TRAIN_MICROBATCH_BY_VOCABULARY,
)
from token_frontier_core import parse_role
from tokenizers import Tokenizer

from jamoflow.corpus import Record, load_records, partition_records

MODEL_SEED = 20_260_817
ORDER_SEED = 20_260_818
LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE = 3e-5
WARMUP_FRACTION = 0.05
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-8
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
QUALITY_MARGIN_BPB = 0.010
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_819
DOCUMENT_PREFIX = b"\x00"


@dataclass(frozen=True, slots=True)
class DocumentTokenInventory:
    document_count: int
    raw_bytes: int
    token_count_excluding_prefix: int
    chunk_count: int
    maximum_document_tokens_including_prefix: int
    document_lengths_sha256: str
    chunk_schedule_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_order(sequence_count: int) -> np.ndarray:
    if sequence_count <= 0:
        raise ValueError("BPE quality order requires sequences")
    return (
        np.random.default_rng(ORDER_SEED)
        .permutation(sequence_count)
        .astype(np.int64, copy=False)
    )


def total_optimizer_steps(sequence_count: int) -> int:
    return math.ceil(sequence_count / EFFECTIVE_BATCH_SIZE)


def warmup_steps(sequence_count: int) -> int:
    return max(1, math.ceil(total_optimizer_steps(sequence_count) * WARMUP_FRACTION))


def cosine_learning_rate(step: int, total_steps: int, warmup: int) -> float:
    if not 0 <= step < total_steps or not 1 <= warmup <= total_steps:
        raise ValueError("BPE quality learning-rate coordinates differ")
    if step < warmup:
        return LEARNING_RATE * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MINIMUM_LEARNING_RATE + (LEARNING_RATE - MINIMUM_LEARNING_RATE) * cosine


def role_training_contract(role: str, sequence_count: int) -> dict[str, Any]:
    vocabulary, _ = parse_role(role)
    microbatch = TRAIN_MICROBATCH_BY_VOCABULARY[vocabulary]
    return {
        "beta1": BETA1,
        "beta2": BETA2,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "epsilon": EPSILON,
        "evaluation_batch_size": EVALUATION_BATCH_BY_VOCABULARY[vocabulary],
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // microbatch,
        "gradient_clip": GRADIENT_CLIP,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "model_seed": MODEL_SEED,
        "optimizer": "AdamW",
        "order_seed": ORDER_SEED,
        "sequence_count": sequence_count,
        "total_optimizer_steps": total_optimizer_steps(sequence_count),
        "train_microbatch_size": microbatch,
        "warmup_fraction": WARMUP_FRACTION,
        "warmup_steps": warmup_steps(sequence_count),
        "weight_decay_for_matrix_parameters": WEIGHT_DECAY,
        "weight_decay_for_vector_parameters": 0.0,
    }


def calibration_document_pieces(
    source_path,
) -> tuple[tuple[bytes, ...], dict[str, Any]]:
    records = load_records(
        [source_path], corpus_format="jsonl", text_field="text", deduplicate=True
    )
    calibration: Sequence[Record] = partition_records(records)["calibration"]
    pieces: list[bytes] = []
    total = 0
    identity = hashlib.sha256(b"JamoFlow/bpe-quality-documents/v1\0")
    for record in calibration:
        if record.text is None:
            continue
        piece = record.raw if not pieces else b"\n" + record.raw
        if total + len(piece) > CALIBRATION_BYTES:
            break
        pieces.append(piece)
        total += len(piece)
        identity.update(len(piece).to_bytes(8, "big"))
        identity.update(hashlib.sha256(piece).digest())
    if len(pieces) < 2 or total <= 0:
        raise ValueError("BPE quality calibration document set is empty")
    return tuple(pieces), {
        "document_count": len(pieces),
        "raw_bytes": total,
        "ordered_document_commitment_sha256": identity.hexdigest(),
        "source_limit_bytes": CALIBRATION_BYTES,
        "truncated_next_document_excluded": True,
    }


def encode_document_chunks(
    pieces: Sequence[bytes],
    tokenizer: Tokenizer,
    token_bytes: Sequence[bytes],
) -> tuple[DocumentTokenInventory, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    chunks: list[np.ndarray] = []
    chunk_documents: list[int] = []
    document_raw_bytes = np.asarray([len(piece) for piece in pieces], dtype=np.int64)
    token_count = 0
    maximum_tokens = 0
    schedule = hashlib.sha256(b"JamoFlow/bpe-quality-document-chunks/v1\0")
    null_ids = tokenizer.encode("\x00", add_special_tokens=False).ids
    if len(null_ids) != 1 or token_bytes[int(null_ids[0])] != DOCUMENT_PREFIX:
        raise ValueError("BPE quality document prefix is not one exact NUL token")
    for document_index, piece in enumerate(pieces):
        text = (DOCUMENT_PREFIX + piece).decode("utf-8", errors="strict")
        ids = tuple(
            int(value) for value in tokenizer.encode(text, add_special_tokens=False).ids
        )
        if (
            len(ids) < 2
            or token_bytes[ids[0]] != DOCUMENT_PREFIX
            or b"".join(token_bytes[value] for value in ids[1:]) != piece
            or tokenizer.decode(list(ids)) != text
        ):
            raise ValueError("BPE quality document encoding differs")
        maximum_tokens = max(maximum_tokens, len(ids))
        token_count += len(ids) - 1
        start = 0
        while start < len(ids) - 1:
            end = min(start + SEQUENCE_LENGTH, len(ids))
            chunk = np.asarray(ids[start:end], dtype=np.int64)
            if len(chunk) < 2:
                raise AssertionError(
                    "BPE quality document chunk cannot predict a token"
                )
            chunks.append(chunk)
            chunk_documents.append(document_index)
            schedule.update(document_index.to_bytes(8, "big"))
            schedule.update(len(chunk).to_bytes(8, "big"))
            schedule.update(chunk.astype("<i8", copy=False).tobytes(order="C"))
            start = end - 1
    inventory = DocumentTokenInventory(
        document_count=len(pieces),
        raw_bytes=int(document_raw_bytes.sum()),
        token_count_excluding_prefix=token_count,
        chunk_count=len(chunks),
        maximum_document_tokens_including_prefix=maximum_tokens,
        document_lengths_sha256=array_sha256(document_raw_bytes),
        chunk_schedule_sha256=schedule.hexdigest(),
    )
    return (
        inventory,
        tuple(chunks),
        np.asarray(chunk_documents, dtype=np.int64),
        document_raw_bytes,
    )


def raw_target_bytes_by_sequence(
    token_ids: np.ndarray,
    token_bytes: Sequence[bytes],
) -> np.ndarray:
    if token_ids.ndim != 2 or token_ids.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("BPE quality token sequence matrix differs")
    lengths = np.asarray([len(value) for value in token_bytes], dtype=np.int64)
    output = np.empty(len(token_ids), dtype=np.int64)
    for start in range(0, len(token_ids), 1024):
        stop = min(start + 1024, len(token_ids))
        output[start:stop] = lengths[token_ids[start:stop, 1:]].sum(axis=1)
    if np.any(output <= 0):
        raise ValueError("BPE quality raw target denominator differs")
    return output


def bpb(nll_nats: np.ndarray, raw_target_bytes: np.ndarray) -> float:
    if (
        nll_nats.ndim != 1
        or raw_target_bytes.shape != nll_nats.shape
        or not np.all(np.isfinite(nll_nats))
        or np.any(nll_nats < 0)
        or np.any(raw_target_bytes <= 0)
    ):
        raise ValueError("BPE quality BPB arrays differ")
    return float(
        math.fsum(float(value) for value in nll_nats)
        / int(raw_target_bytes.sum())
        / math.log(2.0)
    )


def document_bootstrap_upper(
    candidate_nll: np.ndarray,
    reference_nll: np.ndarray,
    raw_bytes: np.ndarray,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    if (
        candidate_nll.shape != reference_nll.shape
        or candidate_nll.shape != raw_bytes.shape
        or candidate_nll.ndim != 1
        or len(candidate_nll) < 2
        or repetitions <= 0
    ):
        raise ValueError("BPE quality document bootstrap arrays differ")
    difference = (candidate_nll - reference_nll).astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(difference), size=len(difference))
        estimates[index] = (
            float(difference[selected].sum())
            / int(raw_bytes[selected].sum())
            / math.log(2.0)
        )
    point = float(difference.sum() / int(raw_bytes.sum()) / math.log(2.0))
    return (
        point,
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def select_quality_frontier(
    contiguous_bpb_by_role: Mapping[str, float],
    document_nll_by_role: Mapping[str, np.ndarray],
    document_raw_bytes: np.ndarray,
    systems_end_to_end_ms: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the sealed quality-then-systems decision to exact role sets."""

    expected = set(QUALITY_ROLES)
    if (
        set(contiguous_bpb_by_role) != expected
        or set(document_nll_by_role) != expected
        or set(systems_end_to_end_ms) != expected
        or document_raw_bytes.ndim != 1
        or len(document_raw_bytes) < 2
        or np.any(document_raw_bytes <= 0)
    ):
        raise ValueError("BPE quality frontier selection inputs differ")
    contiguous = {role: float(contiguous_bpb_by_role[role]) for role in QUALITY_ROLES}
    if not all(math.isfinite(value) and value >= 0 for value in contiguous.values()):
        raise ValueError("BPE quality frontier contiguous BPB differs")
    document = {}
    for role in QUALITY_ROLES:
        values = document_nll_by_role[role]
        if values.shape != document_raw_bytes.shape:
            raise ValueError("BPE quality frontier document NLL shape differs")
        document[role] = bpb(values, document_raw_bytes)
    anchor = min(
        QUALITY_ROLES,
        key=lambda role: (contiguous[role], QUALITY_ROLES.index(role)),
    )
    comparisons = {}
    qualified = []
    for index, role in enumerate(QUALITY_ROLES):
        point, lower, upper = document_bootstrap_upper(
            document_nll_by_role[role],
            document_nll_by_role[anchor],
            document_raw_bytes,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED + index,
        )
        contiguous_difference = contiguous[role] - contiguous[anchor]
        document_difference = document[role] - document[anchor]
        passes = bool(
            contiguous_difference <= QUALITY_MARGIN_BPB
            and document_difference <= QUALITY_MARGIN_BPB
            and upper <= QUALITY_MARGIN_BPB
        )
        if passes:
            qualified.append(role)
        comparisons[role] = {
            "anchor": anchor,
            "contiguous_bpb_difference": contiguous_difference,
            "document_bpb_difference": document_difference,
            "document_bootstrap_difference": point,
            "document_bootstrap_95_lower": lower,
            "document_bootstrap_95_upper": upper,
            "quality_margin_bpb": QUALITY_MARGIN_BPB,
            "quality_qualified": passes,
        }
    if not qualified:
        raise AssertionError("BPE quality anchor must qualify itself")
    comparator = min(
        qualified,
        key=lambda role: (
            float(systems_end_to_end_ms[role]),
            QUALITY_ROLES.index(role),
        ),
    )
    return {
        "calibration_quality_anchor": anchor,
        "comparisons": comparisons,
        "development_bpe_comparator": comparator,
        "document_bpb_by_role": document,
        "quality_qualified_roles": qualified,
    }


def quality_role_order() -> tuple[str, ...]:
    return QUALITY_ROLES


def validate_role_training_contract(role: str, value: Any, sequence_count: int) -> None:
    if value != role_training_contract(role, sequence_count):
        raise ValueError("BPE quality role training contract differs")
