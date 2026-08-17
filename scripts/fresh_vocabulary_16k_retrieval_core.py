"""Pure compact n-gram and prompt-lookup drafting primitives for the 16K target."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

VOCABULARY_SIZE = 16_000
MAXIMUM_CONTEXT_ORDER = 3
MAXIMUM_DRAFT_TOKENS = 3
MAXIMUM_PROMPT_MATCH = 4
MAXIMUM_TABLE_ENTRIES = 200_000
MINIMUM_CONTEXT_COUNT = 5
MINIMUM_NEXT_TOKEN_PROBABILITY = 0.8

TABLE_ORDERS = (1, 2, 3)
TABLE_ARRAY_NAMES = tuple(
    f"order_{order}_{suffix}"
    for order in TABLE_ORDERS
    for suffix in ("context", "next", "best_count", "total_count")
)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def pack_context(tokens: Sequence[int], *, vocabulary_size: int = VOCABULARY_SIZE) -> int:
    if not tokens or len(tokens) > MAXIMUM_CONTEXT_ORDER or vocabulary_size <= 1:
        raise ValueError("retrieval-draft context shape differs")
    value = 0
    for token in tokens:
        current = int(token)
        if not 0 <= current < vocabulary_size:
            raise ValueError("retrieval-draft token is outside the vocabulary")
        value = value * vocabulary_size + current
    if value >= 2**64:
        raise OverflowError("retrieval-draft packed context exceeds uint64")
    return value


@dataclass(frozen=True, slots=True)
class OrderTable:
    order: int
    contexts: np.ndarray
    next_tokens: np.ndarray
    best_counts: np.ndarray
    total_counts: np.ndarray

    def validate(self) -> None:
        length = len(self.contexts)
        if (
            self.order not in TABLE_ORDERS
            or self.contexts.dtype != np.uint64
            or self.next_tokens.dtype != np.uint16
            or self.best_counts.dtype != np.uint32
            or self.total_counts.dtype != np.uint32
            or any(
                value.ndim != 1 or len(value) != length
                for value in (
                    self.contexts,
                    self.next_tokens,
                    self.best_counts,
                    self.total_counts,
                )
            )
            or length == 0
            or np.any(self.contexts[1:] <= self.contexts[:-1])
            or np.any(self.next_tokens >= VOCABULARY_SIZE)
            or np.any(self.best_counts < MINIMUM_CONTEXT_COUNT)
            or np.any(self.total_counts < self.best_counts)
            or np.any(
                self.best_counts.astype(np.float64)
                / self.total_counts.astype(np.float64)
                < MINIMUM_NEXT_TOKEN_PROBABILITY
            )
        ):
            raise ValueError(f"retrieval-draft order-{self.order} table differs")

    def lookup(self, history: Sequence[int]) -> int | None:
        if len(history) < self.order:
            return None
        key = np.uint64(pack_context(history[-self.order :]))
        index = int(np.searchsorted(self.contexts, key))
        if index == len(self.contexts) or self.contexts[index] != key:
            return None
        return int(self.next_tokens[index])


@dataclass(frozen=True, slots=True)
class CompactBackoffTable:
    by_order: Mapping[int, OrderTable]

    def validate(self) -> None:
        if set(self.by_order) != set(TABLE_ORDERS):
            raise ValueError("retrieval-draft table orders differ")
        for order in TABLE_ORDERS:
            table = self.by_order[order]
            if table.order != order:
                raise ValueError("retrieval-draft order label differs")
            table.validate()
        if self.entry_count > MAXIMUM_TABLE_ENTRIES:
            raise ValueError("retrieval-draft table exceeds its entry budget")

    @property
    def entry_count(self) -> int:
        return sum(len(table.contexts) for table in self.by_order.values())

    def next_token(self, history: Sequence[int]) -> int | None:
        for order in reversed(TABLE_ORDERS):
            token = self.by_order[order].lookup(history)
            if token is not None:
                return token
        return None

    def propose(
        self,
        history: Sequence[int],
        *,
        maximum_tokens: int = MAXIMUM_DRAFT_TOKENS,
    ) -> tuple[int, ...]:
        if maximum_tokens <= 0 or maximum_tokens > MAXIMUM_DRAFT_TOKENS:
            raise ValueError("retrieval-draft proposal length differs")
        working = [int(value) for value in history]
        output: list[int] = []
        for _ in range(maximum_tokens):
            token = self.next_token(working)
            if token is None:
                break
            output.append(token)
            working.append(token)
        return tuple(output)

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for order in TABLE_ORDERS:
            table = self.by_order[order]
            arrays[f"order_{order}_context"] = table.contexts
            arrays[f"order_{order}_next"] = table.next_tokens
            arrays[f"order_{order}_best_count"] = table.best_counts
            arrays[f"order_{order}_total_count"] = table.total_counts
        if set(arrays) != set(TABLE_ARRAY_NAMES):
            raise AssertionError("retrieval-draft serialized array set differs")
        return arrays


def table_from_arrays(arrays: Mapping[str, np.ndarray]) -> CompactBackoffTable:
    if set(arrays) != set(TABLE_ARRAY_NAMES):
        raise ValueError("retrieval-draft artifact array set differs")
    by_order: dict[int, OrderTable] = {}
    for order in TABLE_ORDERS:
        by_order[order] = OrderTable(
            order=order,
            contexts=np.asarray(arrays[f"order_{order}_context"]),
            next_tokens=np.asarray(arrays[f"order_{order}_next"]),
            best_counts=np.asarray(arrays[f"order_{order}_best_count"]),
            total_counts=np.asarray(arrays[f"order_{order}_total_count"]),
        )
    table = CompactBackoffTable(by_order=by_order)
    table.validate()
    return table


def prompt_lookup_proposal(
    history: Sequence[int],
    *,
    maximum_match: int = MAXIMUM_PROMPT_MATCH,
    maximum_tokens: int = MAXIMUM_DRAFT_TOKENS,
) -> tuple[int, ...]:
    """Match the longest suffix, then the earliest prior continuation.

    This deliberately follows the current Transformers prompt-lookup tie rule:
    n-gram lengths descend, while equal-length matches are scanned left-to-right.
    """

    values = tuple(int(value) for value in history)
    if (
        maximum_match <= 0
        or maximum_match > MAXIMUM_PROMPT_MATCH
        or maximum_tokens <= 0
        or maximum_tokens > MAXIMUM_DRAFT_TOKENS
        or any(not 0 <= value < VOCABULARY_SIZE for value in values)
    ):
        raise ValueError("prompt-lookup input differs")
    if len(values) < 2:
        return ()
    for size in range(min(maximum_match, len(values) - 1), 0, -1):
        suffix = values[-size:]
        last_start = len(values) - size
        for start in range(last_start + 1):
            if values[start : start + size] != suffix:
                continue
            continuation_start = start + size
            continuation_end = min(
                continuation_start + maximum_tokens,
                len(values),
            )
            if continuation_start < continuation_end:
                return values[continuation_start:continuation_end]
    return ()


def hybrid_retrieval_proposal(
    table: CompactBackoffTable,
    history: Sequence[int],
) -> tuple[tuple[int, ...], str]:
    proposal = table.propose(history)
    if proposal:
        return proposal, "corpus_ngram"
    proposal = prompt_lookup_proposal(history)
    if proposal:
        return proposal, "prompt_lookup"
    return (), "none"


def _full_ngram_counts(
    token_ids: np.ndarray,
    order: int,
    *,
    vocabulary_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(token_ids)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iu"
        or order not in TABLE_ORDERS
        or len(values) <= order
        or np.any(values < 0)
        or np.any(values >= vocabulary_size)
    ):
        raise ValueError("retrieval-draft training token stream differs")
    length = len(values) - order
    contexts = np.asarray(values[:length], dtype=np.uint64).copy()
    for offset in range(1, order):
        contexts *= np.uint64(vocabulary_size)
        contexts += np.asarray(values[offset : offset + length], dtype=np.uint64)
    full = contexts * np.uint64(vocabulary_size)
    full += np.asarray(values[order : order + length], dtype=np.uint64)
    unique, counts = np.unique(full, return_counts=True)
    packed_contexts = unique // np.uint64(vocabulary_size)
    next_tokens = unique % np.uint64(vocabulary_size)
    return (
        packed_contexts.astype(np.uint64, copy=False),
        next_tokens.astype(np.uint16, copy=False),
        counts.astype(np.uint32, copy=False),
        unique,
    )


def _best_continuations(
    token_ids: np.ndarray,
    order: int,
    *,
    vocabulary_size: int,
) -> dict[str, np.ndarray]:
    contexts, next_tokens, counts, _unique = _full_ngram_counts(
        token_ids,
        order,
        vocabulary_size=vocabulary_size,
    )
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(contexts[1:] != contexts[:-1]).astype(np.int64) + 1,
        )
    )
    totals = np.add.reduceat(counts.astype(np.uint64), starts)
    maxima = np.maximum.reduceat(counts, starts)
    lengths = np.diff(np.append(starts, len(contexts)))
    repeated_maxima = np.repeat(maxima, lengths)
    maximum_indices = np.flatnonzero(counts == repeated_maxima)
    maximum_contexts = contexts[maximum_indices]
    _, first = np.unique(maximum_contexts, return_index=True)
    chosen = maximum_indices[first]
    if len(chosen) != len(starts):
        raise AssertionError("retrieval-draft best-continuation grouping differs")
    confidence = maxima.astype(np.float64) / totals.astype(np.float64)
    keep = (
        (totals >= MINIMUM_CONTEXT_COUNT)
        & (maxima >= MINIMUM_CONTEXT_COUNT)
        & (confidence >= MINIMUM_NEXT_TOKEN_PROBABILITY)
    )
    return {
        "order": np.full(int(np.count_nonzero(keep)), order, dtype=np.uint8),
        "context": contexts[chosen][keep].astype(np.uint64, copy=False),
        "next": next_tokens[chosen][keep].astype(np.uint16, copy=False),
        "best_count": maxima[keep].astype(np.uint32, copy=False),
        "total_count": totals[keep].astype(np.uint32, copy=False),
    }


def build_compact_backoff_table(
    token_ids: np.ndarray,
    *,
    vocabulary_size: int = VOCABULARY_SIZE,
    maximum_entries: int = MAXIMUM_TABLE_ENTRIES,
) -> CompactBackoffTable:
    """Build a deterministic global-budget top-continuation table."""

    if vocabulary_size != VOCABULARY_SIZE or maximum_entries != MAXIMUM_TABLE_ENTRIES:
        raise ValueError("retrieval-draft fixed table contract differs")
    candidates = [
        _best_continuations(token_ids, order, vocabulary_size=vocabulary_size)
        for order in TABLE_ORDERS
    ]
    merged = {
        name: np.concatenate([row[name] for row in candidates])
        for name in ("order", "context", "next", "best_count", "total_count")
    }
    confidence = (
        merged["best_count"].astype(np.float64)
        / merged["total_count"].astype(np.float64)
    )
    ranking = np.lexsort(
        (
            merged["next"],
            merged["context"],
            -merged["order"].astype(np.int16),
            -confidence,
            -merged["best_count"].astype(np.int64),
        )
    )
    selected = ranking[:maximum_entries]
    by_order: dict[int, OrderTable] = {}
    for order in TABLE_ORDERS:
        indices = selected[merged["order"][selected] == order]
        indices = indices[np.argsort(merged["context"][indices], kind="stable")]
        by_order[order] = OrderTable(
            order=order,
            contexts=merged["context"][indices].astype(np.uint64, copy=False),
            next_tokens=merged["next"][indices].astype(np.uint16, copy=False),
            best_counts=merged["best_count"][indices].astype(np.uint32, copy=False),
            total_counts=merged["total_count"][indices].astype(np.uint32, copy=False),
        )
    table = CompactBackoffTable(by_order=by_order)
    table.validate()
    if table.entry_count != min(maximum_entries, len(ranking)):
        raise AssertionError("retrieval-draft selected table size differs")
    return table


def table_report(table: CompactBackoffTable) -> dict[str, Any]:
    table.validate()
    orders: dict[str, Any] = {}
    for order in TABLE_ORDERS:
        row = table.by_order[order]
        confidence = row.best_counts.astype(np.float64) / row.total_counts
        orders[str(order)] = {
            "entries": len(row.contexts),
            "context_sha256": array_sha256(row.contexts),
            "next_sha256": array_sha256(row.next_tokens),
            "best_count_sha256": array_sha256(row.best_counts),
            "total_count_sha256": array_sha256(row.total_counts),
            "minimum_confidence": float(np.min(confidence)),
            "maximum_confidence": float(np.max(confidence)),
        }
    return {
        "entry_count": table.entry_count,
        "maximum_entries": MAXIMUM_TABLE_ENTRIES,
        "minimum_context_count": MINIMUM_CONTEXT_COUNT,
        "minimum_next_token_probability": MINIMUM_NEXT_TOKEN_PROBABILITY,
        "orders": orders,
    }


def proposal_acceptance(proposal: Sequence[int], expected: Sequence[int]) -> int:
    accepted = 0
    for draft, target in zip(proposal, expected, strict=False):
        if int(draft) != int(target):
            break
        accepted += 1
    return accepted


def committed_token_count(proposal_length: int, accepted: int) -> int:
    if (
        proposal_length <= 0
        or proposal_length > MAXIMUM_DRAFT_TOKENS
        or not 0 <= accepted <= proposal_length
    ):
        raise ValueError("retrieval-draft committed-token accounting differs")
    # Accepted prefix plus one target correction/bonus.
    value = accepted + 1
    if not math.isfinite(float(value)):
        raise AssertionError("retrieval-draft committed-token count is not finite")
    return value
