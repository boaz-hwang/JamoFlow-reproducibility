"""Raw-context-matched rolling BPB plans for publication comparisons.

The candidate and one comparator are always scored on identical raw target
spans with identical raw left context.  Minimal UTF-8-complete groups of
comparator-native units determine legal boundaries, so a BPE token is never
split or assigned a fractional loss and every model input starts on a scalar
boundary.
Only content-free offsets and cryptographic identities leave this module.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import json
import unicodedata
from typing import Sequence

from .publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_BPB_CONTEXT_BYTES,
    PUBLICATION_BPB_CONTEXT_CONTRACT,
    PUBLICATION_BPB_TARGET_BLOCK_BYTES,
    PUBLICATION_BPB_UNSCORED_PREFIX_POLICY,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
)


RAW_BYTE_TOKENIZER_SHA256 = hashlib.sha256(
    b"jamoflow-publication-raw-byte-units-v1"
).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicationBPBWindow:
    context_start_byte: int
    target_start_byte: int
    target_end_byte: int
    context_start_unit: int
    target_start_unit: int
    target_end_unit: int
    context_start_token: int
    target_start_token: int
    target_end_token: int
    source_bytes: int
    context_bytes: int
    target_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationBPBDocumentPlan:
    comparator_key: str
    document_sha256: str
    source_bytes: int
    comparator_tokens: int
    comparator_units: int
    excluded_prefix_bytes: int
    scored_bytes: int
    natural_token_ids_sha256: str
    natural_token_lengths_sha256: str
    unit_lengths_sha256: str
    unit_token_counts_sha256: str
    windows: tuple[PublicationBPBWindow, ...]
    plan_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PublicationBPBContextEvidence:
    candidate_key: str
    comparator_key: str
    tokenizer_sha256: str
    context_contract: str
    unscored_prefix_policy: str
    maximum_source_bytes: int
    maximum_target_bytes: int
    input_document_count: int
    scored_document_count: int
    unscored_document_count: int
    source_bytes: int
    scored_bytes: int
    window_count: int
    document_stream_sha256: str
    scored_document_order_sha256: str
    scored_bytes_by_document_sha256: str
    window_plan_sha256: str
    identity_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _lengths_sha256(lengths: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for length in lengths:
        digest.update(int(length).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    values = tuple(int(value) for value in token_ids)
    if not values or any(value < 0 or value >= 2**63 for value in values):
        raise ValueError("publication BPB token ids must be nonnegative int64")
    return _lengths_sha256(values)


def _raw_unit_lengths_sha256(count: int) -> str:
    """Hash ``count`` one-byte unit lengths without an O(bytes) tuple."""

    digest = hashlib.sha256()
    encoded = (1).to_bytes(8, "little", signed=False)
    units_per_chunk = 4_096
    chunk = encoded * units_per_chunk
    complete_chunks, remainder = divmod(count, units_per_chunk)
    for _ in range(complete_chunks):
        digest.update(chunk)
    digest.update(encoded * remainder)
    return digest.hexdigest()


def scored_bytes_by_document_sha256(scored_bytes: Sequence[int]) -> str:
    values = tuple(int(value) for value in scored_bytes)
    if not values or any(value <= 0 for value in values):
        raise ValueError("publication BPB scored-byte counts must be positive")
    return _lengths_sha256(values)


def _validate_document(document: bytes) -> None:
    if not document:
        raise ValueError("publication BPB documents must be nonempty")
    try:
        text = document.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("publication BPB documents must be strict UTF-8") from exc
    if unicodedata.normalize("NFC", text) != text:
        raise ValueError("publication BPB documents must already be NFC")


def _is_utf8_scalar_boundary(document: bytes, offset: int) -> bool:
    return (
        offset == 0
        or offset == len(document)
        or document[offset] & 0xC0 != 0x80
    )


def _raw_evaluation_units(
    document: bytes,
) -> tuple[tuple[int, ...], tuple[int, ...], str, str]:
    unit_lengths: list[int] = []
    unit_token_counts: list[int] = []
    unit_start = 0
    for offset in range(1, len(document) + 1):
        if _is_utf8_scalar_boundary(document, offset):
            length = offset - unit_start
            unit_lengths.append(length)
            unit_token_counts.append(length)
            unit_start = offset
    if unit_start != len(document):
        raise RuntimeError("raw UTF-8 evaluation units did not cover the document")
    return (
        tuple(unit_lengths),
        tuple(unit_token_counts),
        _token_ids_sha256(document),
        _raw_unit_lengths_sha256(len(document)),
    )


def _bpe_evaluation_units(
    document: bytes,
    comparator_key: str,
    comparator_token_ids: Sequence[int] | None,
    comparator_token_bytes: Sequence[bytes] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], str, str]:
    if comparator_key not in PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values():
        raise ValueError("unknown publication BPB comparator")
    if (
        comparator_token_ids is None
        or comparator_token_bytes is None
        or not comparator_token_ids
        or len(comparator_token_ids) != len(comparator_token_bytes)
    ):
        raise ValueError("BPE BPB plans require natural full-document token bytes")
    natural_token_ids_digest = _token_ids_sha256(comparator_token_ids)

    natural_token_lengths: list[int] = []
    evaluation_unit_lengths: list[int] = []
    evaluation_unit_token_counts: list[int] = []
    cursor = 0
    unit_start = 0
    unit_token_count = 0
    for unit in comparator_token_bytes:
        if not isinstance(unit, bytes) or not unit:
            raise ValueError("comparator units must be nonempty bytes")
        next_cursor = cursor + len(unit)
        if document[cursor:next_cursor] != unit:
            raise ValueError("comparator units do not reconstruct the document exactly")
        natural_token_lengths.append(len(unit))
        cursor = next_cursor
        unit_token_count += 1
        if _is_utf8_scalar_boundary(document, cursor):
            evaluation_unit_length = cursor - unit_start
            if evaluation_unit_length > PUBLICATION_BPB_TARGET_BLOCK_BYTES:
                raise ValueError(
                    "UTF-8-complete comparator group exceeds the sealed BPB target block"
                )
            evaluation_unit_lengths.append(evaluation_unit_length)
            evaluation_unit_token_counts.append(unit_token_count)
            unit_start = cursor
            unit_token_count = 0
    if cursor != len(document):
        raise ValueError("comparator units do not cover the complete document")
    if unit_start != len(document) or unit_token_count:
        raise RuntimeError("BPE UTF-8 evaluation groups did not close exactly")
    return (
        tuple(evaluation_unit_lengths),
        tuple(evaluation_unit_token_counts),
        natural_token_ids_digest,
        _lengths_sha256(natural_token_lengths),
    )


def _evaluation_windows(
    lengths: Sequence[int],
    token_counts: Sequence[int],
) -> tuple[PublicationBPBWindow, ...]:
    if len(lengths) != len(token_counts) or any(count <= 0 for count in token_counts):
        raise RuntimeError("publication BPB evaluation-unit geometry is invalid")
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    token_offsets = [0]
    for count in token_counts:
        token_offsets.append(token_offsets[-1] + count)

    windows: list[PublicationBPBWindow] = []
    target_start_unit = 1
    while target_start_unit < len(lengths):
        target_end_unit = target_start_unit
        while (
            target_end_unit < len(lengths)
            and offsets[target_end_unit + 1] - offsets[target_start_unit]
            <= PUBLICATION_BPB_TARGET_BLOCK_BYTES
        ):
            target_end_unit += 1
        if target_end_unit == target_start_unit:
            raise RuntimeError("sealed BPB unit-size invariant was not enforced")

        target_start_byte = offsets[target_start_unit]
        target_end_byte = offsets[target_end_unit]
        minimum_context_start = target_end_byte - PUBLICATION_BPB_CONTEXT_BYTES
        context_start_unit = bisect_left(
            offsets,
            minimum_context_start,
            0,
            target_start_unit,
        )
        if context_start_unit >= target_start_unit:
            raise ValueError("BPB target has no complete predecessor inside raw cap")
        context_start_byte = offsets[context_start_unit]
        source_bytes = target_end_byte - context_start_byte
        target_bytes = target_end_byte - target_start_byte
        context_bytes = target_start_byte - context_start_byte
        if (
            source_bytes > PUBLICATION_BPB_CONTEXT_BYTES
            or target_bytes > PUBLICATION_BPB_TARGET_BLOCK_BYTES
            or context_bytes <= 0
        ):
            raise RuntimeError("publication BPB rolling-window invariant failed")
        windows.append(
            PublicationBPBWindow(
                context_start_byte=context_start_byte,
                target_start_byte=target_start_byte,
                target_end_byte=target_end_byte,
                context_start_unit=context_start_unit,
                target_start_unit=target_start_unit,
                target_end_unit=target_end_unit,
                context_start_token=token_offsets[context_start_unit],
                target_start_token=token_offsets[target_start_unit],
                target_end_token=token_offsets[target_end_unit],
                source_bytes=source_bytes,
                context_bytes=context_bytes,
                target_bytes=target_bytes,
            )
        )
        target_start_unit = target_end_unit
    return tuple(windows)


def _plan_hash_payload(
    *,
    comparator_key: str,
    document_sha256: str,
    source_bytes: int,
    comparator_tokens: int,
    comparator_units: int,
    excluded_prefix_bytes: int,
    scored_bytes: int,
    natural_token_ids_sha256: str,
    natural_token_lengths_sha256: str,
    unit_lengths_sha256: str,
    unit_token_counts_sha256: str,
    windows: Sequence[PublicationBPBWindow],
) -> dict[str, object]:
    return {
        "comparator_key": comparator_key,
        "comparator_tokens": comparator_tokens,
        "comparator_units": comparator_units,
        "context_contract": PUBLICATION_BPB_CONTEXT_CONTRACT,
        "document_sha256": document_sha256,
        "excluded_prefix_bytes": excluded_prefix_bytes,
        "maximum_source_bytes": PUBLICATION_BPB_CONTEXT_BYTES,
        "maximum_target_bytes": PUBLICATION_BPB_TARGET_BLOCK_BYTES,
        "natural_token_ids_sha256": natural_token_ids_sha256,
        "natural_token_lengths_sha256": natural_token_lengths_sha256,
        "scored_bytes": scored_bytes,
        "source_bytes": source_bytes,
        "unit_lengths_sha256": unit_lengths_sha256,
        "unit_token_counts_sha256": unit_token_counts_sha256,
        "unscored_prefix_policy": PUBLICATION_BPB_UNSCORED_PREFIX_POLICY,
        "windows": [window.to_dict() for window in windows],
    }


def build_publication_bpb_document_plan(
    document: bytes,
    *,
    comparator_key: str,
    comparator_token_ids: Sequence[int] | None = None,
    comparator_token_bytes: Sequence[bytes] | None = None,
) -> PublicationBPBDocumentPlan:
    """Build a complete, nonoverlapping target plan with overlapping context."""

    _validate_document(document)
    if comparator_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY:
        if comparator_token_ids is not None or comparator_token_bytes is not None:
            raise ValueError("raw-byte BPB plans derive one-byte units internally")
        (
            lengths,
            token_counts,
            natural_token_ids_digest,
            natural_token_lengths_digest,
        ) = _raw_evaluation_units(document)
        comparator_token_count = len(document)
    else:
        (
            lengths,
            token_counts,
            natural_token_ids_digest,
            natural_token_lengths_digest,
        ) = _bpe_evaluation_units(
            document,
            comparator_key,
            comparator_token_ids,
            comparator_token_bytes,
        )
        assert comparator_token_ids is not None
        comparator_token_count = len(comparator_token_bytes)
    comparator_unit_count = len(lengths)
    excluded_prefix_bytes = lengths[0]
    unit_digest = _lengths_sha256(lengths)
    unit_token_counts_digest = _lengths_sha256(token_counts)
    windows = _evaluation_windows(lengths, token_counts)

    scored_bytes = len(document) - excluded_prefix_bytes
    if sum(window.target_bytes for window in windows) != scored_bytes:
        raise RuntimeError("publication BPB targets do not cover the document once")
    for previous, current in zip(windows, windows[1:]):
        if previous.target_end_byte != current.target_start_byte:
            raise RuntimeError("publication BPB target blocks are not contiguous")

    document_sha256 = hashlib.sha256(document).hexdigest()
    payload = _plan_hash_payload(
        comparator_key=comparator_key,
        document_sha256=document_sha256,
        source_bytes=len(document),
        comparator_tokens=comparator_token_count,
        comparator_units=comparator_unit_count,
        excluded_prefix_bytes=excluded_prefix_bytes,
        scored_bytes=scored_bytes,
        natural_token_ids_sha256=natural_token_ids_digest,
        natural_token_lengths_sha256=natural_token_lengths_digest,
        unit_lengths_sha256=unit_digest,
        unit_token_counts_sha256=unit_token_counts_digest,
        windows=windows,
    )
    return PublicationBPBDocumentPlan(
        comparator_key=comparator_key,
        document_sha256=document_sha256,
        source_bytes=len(document),
        comparator_tokens=comparator_token_count,
        comparator_units=comparator_unit_count,
        excluded_prefix_bytes=excluded_prefix_bytes,
        scored_bytes=scored_bytes,
        natural_token_ids_sha256=natural_token_ids_digest,
        natural_token_lengths_sha256=natural_token_lengths_digest,
        unit_lengths_sha256=unit_digest,
        unit_token_counts_sha256=unit_token_counts_digest,
        windows=windows,
        plan_sha256=_canonical_sha256(payload),
    )


def _ordered_document_digest(documents: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(len(document).to_bytes(8, "little", signed=False))
        digest.update(document)
    return digest.hexdigest()


def _ordered_plan_digest(plans: Sequence[PublicationBPBDocumentPlan]) -> str:
    digest = hashlib.sha256()
    for plan in plans:
        digest.update(bytes.fromhex(plan.plan_sha256))
    return digest.hexdigest()


def _ordered_scored_document_digest(
    plans: Sequence[PublicationBPBDocumentPlan],
) -> str:
    digest = hashlib.sha256()
    for plan in plans:
        if plan.scored_bytes:
            digest.update(bytes.fromhex(plan.document_sha256))
    return digest.hexdigest()


def _evidence_identity_payload(
    evidence: PublicationBPBContextEvidence,
) -> dict[str, str | int]:
    payload = evidence.to_dict()
    payload.pop("identity_sha256")
    return payload


def build_publication_bpb_context_evidence(
    documents: Sequence[bytes],
    *,
    candidate_key: str,
    comparator_key: str,
    tokenizer_sha256: str,
    comparator_token_ids_by_document: Sequence[Sequence[int]] | None = None,
    comparator_token_bytes_by_document: Sequence[Sequence[bytes]] | None = None,
) -> tuple[PublicationBPBContextEvidence, tuple[PublicationBPBDocumentPlan, ...]]:
    """Build content-free evidence binding a pair to one exact rolling plan."""

    if candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY or not documents:
        raise ValueError("publication BPB evidence requires the sealed candidate")
    if comparator_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY:
        if (
            comparator_token_ids_by_document is not None
            or comparator_token_bytes_by_document is not None
        ):
            raise ValueError("raw-byte BPB evidence cannot accept external units")
        if tokenizer_sha256 != RAW_BYTE_TOKENIZER_SHA256:
            raise ValueError("raw-byte BPB evidence has the wrong tokenizer identity")
        units: Sequence[Sequence[bytes] | None] = (None,) * len(documents)
        token_ids: Sequence[Sequence[int] | None] = (None,) * len(documents)
    elif comparator_key in PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values():
        if (
            comparator_token_ids_by_document is None
            or comparator_token_bytes_by_document is None
            or len(comparator_token_ids_by_document) != len(documents)
            or len(comparator_token_bytes_by_document) != len(documents)
            or not _is_sha256(tokenizer_sha256)
            or tokenizer_sha256 == RAW_BYTE_TOKENIZER_SHA256
        ):
            raise ValueError("BPE BPB evidence requires exact tokenizer-bound units")
        units = comparator_token_bytes_by_document
        token_ids = comparator_token_ids_by_document
    else:
        raise ValueError("unknown publication BPB comparator")

    plans = tuple(
        build_publication_bpb_document_plan(
            document,
            comparator_key=comparator_key,
            comparator_token_ids=document_token_ids,
            comparator_token_bytes=document_units,
        )
        for document, document_token_ids, document_units in zip(
            documents,
            token_ids,
            units,
            strict=True,
        )
    )
    scored_plans = tuple(plan for plan in plans if plan.scored_bytes > 0)
    if not scored_plans:
        raise ValueError("publication BPB evidence has no scorable documents")
    scored_bytes = tuple(plan.scored_bytes for plan in scored_plans)
    provisional = PublicationBPBContextEvidence(
        candidate_key=candidate_key,
        comparator_key=comparator_key,
        tokenizer_sha256=tokenizer_sha256,
        context_contract=PUBLICATION_BPB_CONTEXT_CONTRACT,
        unscored_prefix_policy=PUBLICATION_BPB_UNSCORED_PREFIX_POLICY,
        maximum_source_bytes=PUBLICATION_BPB_CONTEXT_BYTES,
        maximum_target_bytes=PUBLICATION_BPB_TARGET_BLOCK_BYTES,
        input_document_count=len(plans),
        scored_document_count=len(scored_plans),
        unscored_document_count=len(plans) - len(scored_plans),
        source_bytes=sum(plan.source_bytes for plan in plans),
        scored_bytes=sum(scored_bytes),
        window_count=sum(len(plan.windows) for plan in scored_plans),
        document_stream_sha256=_ordered_document_digest(documents),
        scored_document_order_sha256=_ordered_scored_document_digest(plans),
        scored_bytes_by_document_sha256=scored_bytes_by_document_sha256(
            scored_bytes
        ),
        window_plan_sha256=_ordered_plan_digest(plans),
        identity_sha256="",
    )
    evidence = PublicationBPBContextEvidence(
        **{
            **provisional.to_dict(),
            "identity_sha256": _canonical_sha256(
                _evidence_identity_payload(provisional)
            ),
        }
    )
    validate_publication_bpb_context_evidence(evidence, scored_bytes)
    return evidence, plans


def publication_bpb_scored_bytes(
    plans: Sequence[PublicationBPBDocumentPlan],
) -> tuple[int, ...]:
    values = tuple(plan.scored_bytes for plan in plans if plan.scored_bytes > 0)
    if not values:
        raise ValueError("publication BPB plans have no scored documents")
    return values


def validate_publication_bpb_context_evidence(
    evidence: PublicationBPBContextEvidence,
    scored_bytes_by_document: Sequence[int],
    *,
    candidate_key: str | None = None,
    comparator_key: str | None = None,
) -> None:
    values = tuple(int(value) for value in scored_bytes_by_document)
    expected_candidate = candidate_key or evidence.candidate_key
    expected_comparator = comparator_key or evidence.comparator_key
    if (
        evidence.candidate_key != PUBLICATION_CANDIDATE_MODEL_KEY
        or evidence.candidate_key != expected_candidate
        or evidence.comparator_key != expected_comparator
        or evidence.comparator_key
        not in {
            PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            *PUBLICATION_BPE_COMPARATOR_MODEL_KEYS.values(),
        }
        or evidence.context_contract != PUBLICATION_BPB_CONTEXT_CONTRACT
        or evidence.unscored_prefix_policy
        != PUBLICATION_BPB_UNSCORED_PREFIX_POLICY
        or evidence.maximum_source_bytes != PUBLICATION_BPB_CONTEXT_BYTES
        or evidence.maximum_target_bytes != PUBLICATION_BPB_TARGET_BLOCK_BYTES
        or evidence.input_document_count
        != evidence.scored_document_count + evidence.unscored_document_count
        or evidence.input_document_count <= 0
        or evidence.scored_document_count <= 0
        or evidence.unscored_document_count < 0
        or evidence.scored_document_count != len(values)
        or evidence.scored_bytes != sum(values)
        or evidence.source_bytes <= evidence.scored_bytes
        or evidence.scored_bytes < evidence.scored_document_count
        or evidence.window_count < evidence.scored_document_count
        or evidence.window_count > evidence.scored_bytes
        or not all(
            _is_sha256(value)
            for value in (
                evidence.tokenizer_sha256,
                evidence.document_stream_sha256,
                evidence.scored_document_order_sha256,
                evidence.scored_bytes_by_document_sha256,
                evidence.window_plan_sha256,
                evidence.identity_sha256,
            )
        )
        or evidence.scored_bytes_by_document_sha256
        != scored_bytes_by_document_sha256(values)
        or evidence.identity_sha256
        != _canonical_sha256(_evidence_identity_payload(evidence))
    ):
        raise ValueError("publication BPB context evidence is inconsistent")
    if evidence.comparator_key == PUBLICATION_RAW_COMPARATOR_MODEL_KEY:
        if (
            evidence.tokenizer_sha256 != RAW_BYTE_TOKENIZER_SHA256
            or evidence.source_bytes - evidence.scored_bytes
            < evidence.input_document_count
            or evidence.source_bytes - evidence.scored_bytes
            > 4 * evidence.input_document_count
        ):
            raise ValueError("raw-byte BPB tokenizer identity is inconsistent")
    elif (
        evidence.tokenizer_sha256 == RAW_BYTE_TOKENIZER_SHA256
        or evidence.source_bytes - evidence.scored_bytes
        < evidence.input_document_count
        or evidence.source_bytes - evidence.scored_bytes
        > evidence.maximum_target_bytes * evidence.input_document_count
    ):
        raise ValueError("BPE BPB tokenizer identity is inconsistent")
