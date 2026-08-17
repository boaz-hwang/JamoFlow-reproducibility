"""Pure event replay for the trained 16K retrieval-draft mechanism audit."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from fresh_vocabulary_16k_retrieval_core import (
    MAXIMUM_DRAFT_TOKENS,
    MAXIMUM_PROMPT_MATCH,
    TABLE_ORDERS,
    CompactBackoffTable,
    pack_context,
)

MODES = ("controlled_replay", "free_running_utf8_greedy")
PROFILE_ROLES = (
    "prompt_lookup_block_4",
    "corpus_ngram_block_4",
    "hybrid_retrieval_block_4",
)
BOUNDARY_CLASSES = (
    "inside_utf8_scalar",
    "after_whitespace",
    "within_hangul_eojeol",
    "after_other",
)
PRIMARY_BOUNDARIES = ("within_hangul_eojeol", "after_whitespace")
PRIMARY_SOURCE = "prompt_lookup"
MINIMUM_STRATUM_CYCLES = 32
MINIMUM_PAIRED_CASES = 16
MINIMUM_ACCEPTED_TOKENS_PER_CYCLE_GAP = 0.25
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260852


def _is_hangul(value: str) -> bool:
    codepoint = ord(value)
    return (
        0xAC00 <= codepoint <= 0xD7A3
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def _boundary_class(raw: bytes, state_index: int) -> tuple[str, int]:
    text = raw.decode("utf-8", errors="ignore")
    current_eojeol = text.rsplit(maxsplit=1)[-1] if text and not text[-1].isspace() else ""
    hangul_syllables = sum(_is_hangul(value) for value in current_eojeol)
    if state_index != 0:
        return "inside_utf8_scalar", hangul_syllables
    if not text:
        return "after_other", 0
    if text[-1].isspace():
        return "after_whitespace", 0
    if _is_hangul(text[-1]):
        return "within_hangul_eojeol", hangul_syllables
    return "after_other", hangul_syllables


def _contains_whitespace(raw: bytes) -> bool:
    return any(value in b" \t\r\n" for value in raw)


def _prompt_proposal(history: Sequence[int]) -> tuple[tuple[int, ...], int]:
    values = tuple(int(value) for value in history)
    if len(values) < 2:
        return (), 0
    for size in range(min(MAXIMUM_PROMPT_MATCH, len(values) - 1), 0, -1):
        suffix = values[-size:]
        last_start = len(values) - size
        for start in range(last_start + 1):
            if values[start : start + size] != suffix:
                continue
            continuation_start = start + size
            continuation_end = min(
                continuation_start + MAXIMUM_DRAFT_TOKENS,
                len(values),
            )
            if continuation_start < continuation_end:
                return values[continuation_start:continuation_end], size
    return (), 0


def _table_next(
    table: CompactBackoffTable,
    history: Sequence[int],
) -> tuple[int, int, float] | None:
    for order in reversed(TABLE_ORDERS):
        if len(history) < order:
            continue
        row = table.by_order[order]
        key = np.uint64(pack_context(history[-order:]))
        index = int(np.searchsorted(row.contexts, key))
        if index == len(row.contexts) or row.contexts[index] != key:
            continue
        return (
            int(row.next_tokens[index]),
            order,
            float(row.best_counts[index]) / float(row.total_counts[index]),
        )
    return None


def _corpus_proposal(
    table: CompactBackoffTable,
    history: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    working = [int(value) for value in history]
    output: list[int] = []
    orders: list[int] = []
    confidence: list[float] = []
    for _ in range(MAXIMUM_DRAFT_TOKENS):
        row = _table_next(table, working)
        if row is None:
            break
        token, order, probability = row
        output.append(token)
        orders.append(order)
        confidence.append(probability)
        working.append(token)
    return tuple(output), tuple(orders), tuple(confidence)


def _proposal(
    role: str,
    table: CompactBackoffTable,
    history: Sequence[int],
) -> tuple[tuple[int, ...], str, int, tuple[int, ...], tuple[float, ...]]:
    if role == "prompt_lookup_block_4":
        values, match = _prompt_proposal(history)
        return values, "prompt_lookup" if values else "none", match, (), ()
    corpus, orders, confidence = _corpus_proposal(table, history)
    if role == "corpus_ngram_block_4":
        return corpus, "corpus_ngram" if corpus else "none", 0, orders, confidence
    if role != "hybrid_retrieval_block_4":
        raise ValueError("retrieval mechanism role differs")
    if corpus:
        return corpus, "corpus_ngram", 0, orders, confidence
    values, match = _prompt_proposal(history)
    return values, "prompt_lookup" if values else "none", match, (), ()


@dataclass(frozen=True, slots=True)
class ProposalEvent:
    case_index: int
    mode: str
    role: str
    source: str
    boundary_class: str
    eojeol_hangul_syllables: int
    proposal_tokens: int
    accepted_tokens: int
    outcome: str
    proposal_contains_whitespace: bool
    prompt_match_tokens: int
    first_table_order: int
    minimum_table_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay_proposal_events(
    *,
    case_index: int,
    mode: str,
    role: str,
    prompt_raw: bytes,
    prompt_ids: Sequence[int],
    expected_raw: bytes,
    expected_ids: Sequence[int],
    token_bytes: Sequence[bytes],
    next_state_indices: Sequence[Sequence[int]],
    table: CompactBackoffTable,
) -> list[ProposalEvent]:
    if mode not in MODES or role not in PROFILE_ROLES or case_index < 0:
        raise ValueError("retrieval mechanism replay identity differs")
    expected = tuple(int(value) for value in expected_ids)
    prompt = tuple(int(value) for value in prompt_ids)
    if not expected or not prompt:
        raise ValueError("retrieval mechanism trace is empty")
    if b"".join(token_bytes[value] for value in prompt) != prompt_raw:
        raise ValueError("retrieval mechanism prompt bytes differ")
    state_index = 0
    for token in prompt:
        state_index = int(next_state_indices[state_index][token])
        if state_index < 0:
            raise ValueError("retrieval mechanism prompt violates UTF-8")
    if state_index != 0:
        raise ValueError("retrieval mechanism prompt ends inside a scalar")

    generated_ids = [expected[0]]
    generated = bytearray(token_bytes[expected[0]])
    state_index = int(next_state_indices[0][expected[0]])
    if state_index < 0:
        raise ValueError("retrieval mechanism first token violates UTF-8")
    next_index = 1
    history = list(prompt) + generated_ids
    events: list[ProposalEvent] = []
    while next_index < len(expected):
        boundary, syllables = _boundary_class(prompt_raw + bytes(generated), state_index)
        proposal, source, prompt_match, orders, confidence = _proposal(role, table, history)
        if not proposal:
            token = expected[next_index]
            events.append(
                ProposalEvent(
                    case_index=case_index,
                    mode=mode,
                    role=role,
                    source="none",
                    boundary_class=boundary,
                    eojeol_hangul_syllables=syllables,
                    proposal_tokens=0,
                    accepted_tokens=0,
                    outcome="no_proposal",
                    proposal_contains_whitespace=False,
                    prompt_match_tokens=0,
                    first_table_order=0,
                    minimum_table_confidence=0.0,
                )
            )
            generated_ids.append(token)
            generated.extend(token_bytes[token])
            state_index = int(next_state_indices[state_index][token])
            if state_index < 0:
                raise AssertionError("retrieval mechanism AR token violates UTF-8")
            history.append(token)
            next_index += 1
            continue

        accepted = 0
        outcome = "rejection"
        for draft_token in proposal:
            target = expected[next_index]
            if int(draft_token) != target:
                generated_ids.append(target)
                generated.extend(token_bytes[target])
                state_index = int(next_state_indices[state_index][target])
                if state_index < 0:
                    raise AssertionError("retrieval mechanism correction violates UTF-8")
                history.append(target)
                next_index += 1
                break
            generated_ids.append(target)
            generated.extend(token_bytes[target])
            state_index = int(next_state_indices[state_index][target])
            if state_index < 0:
                raise AssertionError("retrieval mechanism accepted token violates UTF-8")
            history.append(target)
            accepted += 1
            next_index += 1
            if next_index == len(expected):
                outcome = "terminal_accept"
                break
        else:
            outcome = "full_accept_bonus"
            bonus = expected[next_index]
            generated_ids.append(bonus)
            generated.extend(token_bytes[bonus])
            state_index = int(next_state_indices[state_index][bonus])
            if state_index < 0:
                raise AssertionError("retrieval mechanism bonus violates UTF-8")
            history.append(bonus)
            next_index += 1

        proposal_raw = b"".join(token_bytes[value] for value in proposal)
        events.append(
            ProposalEvent(
                case_index=case_index,
                mode=mode,
                role=role,
                source=source,
                boundary_class=boundary,
                eojeol_hangul_syllables=syllables,
                proposal_tokens=len(proposal),
                accepted_tokens=accepted,
                outcome=outcome,
                proposal_contains_whitespace=_contains_whitespace(proposal_raw),
                prompt_match_tokens=prompt_match,
                first_table_order=orders[0] if orders else 0,
                minimum_table_confidence=min(confidence) if confidence else 0.0,
            )
        )

    if (
        tuple(generated_ids) != expected
        or bytes(generated) != expected_raw
        or state_index != 0
    ):
        raise AssertionError("retrieval mechanism replay output differs")
    return events


def _aggregate_rows(events: Sequence[ProposalEvent]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[ProposalEvent]] = defaultdict(list)
    for event in events:
        groups[(event.mode, event.role, event.source, event.boundary_class)].append(event)
    output: dict[str, Any] = {}
    for key in sorted(groups):
        values = groups[key]
        proposals = [value for value in values if value.proposal_tokens > 0]
        proposal_tokens = sum(value.proposal_tokens for value in proposals)
        accepted = sum(value.accepted_tokens for value in proposals)
        label = "/".join(key)
        output[label] = {
            "cycles": len(values),
            "proposal_cycles": len(proposals),
            "no_proposal_cycles": len(values) - len(proposals),
            "proposal_tokens": proposal_tokens,
            "accepted_tokens": accepted,
            "draft_token_acceptance_rate": (
                accepted / proposal_tokens if proposal_tokens else 0.0
            ),
            "accepted_tokens_per_proposal_cycle": (
                accepted / len(proposals) if proposals else 0.0
            ),
            "rejection_rate": (
                sum(value.outcome == "rejection" for value in proposals) / len(proposals)
                if proposals
                else 0.0
            ),
            "proposal_crosses_whitespace_rate": (
                sum(value.proposal_contains_whitespace for value in proposals) / len(proposals)
                if proposals
                else 0.0
            ),
        }
    return output


def _paired_bootstrap(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.ndim != 1 or len(values) < MINIMUM_PAIRED_CASES or not np.isfinite(values).all():
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sample = rng.integers(0, len(values), size=len(values))
        estimates[index] = float(np.mean(values[sample]))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def primary_hangul_boundary_contrast(events: Sequence[ProposalEvent]) -> dict[str, Any]:
    selected = [
        event
        for event in events
        if event.mode == "free_running_utf8_greedy"
        and event.role == "hybrid_retrieval_block_4"
        and event.source == PRIMARY_SOURCE
        and event.proposal_tokens > 0
        and event.boundary_class in PRIMARY_BOUNDARIES
    ]
    by_case_boundary: dict[tuple[int, str], list[int]] = defaultdict(list)
    for event in selected:
        by_case_boundary[(event.case_index, event.boundary_class)].append(
            event.accepted_tokens
        )
    paired: list[float] = []
    for case_index in sorted({event.case_index for event in selected}):
        inside = by_case_boundary.get((case_index, "within_hangul_eojeol"), [])
        after = by_case_boundary.get((case_index, "after_whitespace"), [])
        if inside and after:
            paired.append(float(np.mean(inside)) - float(np.mean(after)))
    differences = np.asarray(paired, dtype=np.float64)
    lower, upper = _paired_bootstrap(differences)
    counts = {
        boundary: sum(event.boundary_class == boundary for event in selected)
        for boundary in PRIMARY_BOUNDARIES
    }
    means = {
        boundary: (
            float(
                np.mean(
                    [
                        event.accepted_tokens
                        for event in selected
                        if event.boundary_class == boundary
                    ]
                )
            )
            if counts[boundary]
            else 0.0
        )
        for boundary in PRIMARY_BOUNDARIES
    }
    point = float(np.mean(differences)) if len(differences) else None
    gates = {
        "minimum_cycles_each": all(
            counts[boundary] >= MINIMUM_STRATUM_CYCLES for boundary in PRIMARY_BOUNDARIES
        ),
        "minimum_paired_cases": len(differences) >= MINIMUM_PAIRED_CASES,
        "minimum_effect": bool(
            point is not None and point >= MINIMUM_ACCEPTED_TOKENS_PER_CYCLE_GAP
        ),
        "bootstrap_lower_positive": bool(lower is not None and lower > 0),
    }
    gates["overall_pass"] = all(gates.values())
    return {
        "mode": "free_running_utf8_greedy",
        "role": "hybrid_retrieval_block_4",
        "source": PRIMARY_SOURCE,
        "contrast": "within_hangul_eojeol_minus_after_whitespace",
        "cycle_counts": counts,
        "accepted_tokens_per_cycle": means,
        "paired_case_count": len(differences),
        "paired_case_mean_difference": point,
        "paired_case_bootstrap_95_interval": {"lower": lower, "upper": upper},
        "minimum_cycles_each": MINIMUM_STRATUM_CYCLES,
        "minimum_paired_cases": MINIMUM_PAIRED_CASES,
        "minimum_accepted_tokens_per_cycle_gap": (
            MINIMUM_ACCEPTED_TOKENS_PER_CYCLE_GAP
        ),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate": gates,
    }


def summarize_mechanism(events: Sequence[ProposalEvent]) -> dict[str, Any]:
    if not events:
        raise ValueError("retrieval mechanism event stream is empty")
    expected_modes = {event.mode for event in events}
    expected_roles = {event.role for event in events}
    if expected_modes != set(MODES) or expected_roles != set(PROFILE_ROLES):
        raise ValueError("retrieval mechanism event coverage differs")
    contrast = primary_hangul_boundary_contrast(events)
    return {
        "event_count": len(events),
        "aggregate_rows": _aggregate_rows(events),
        "primary_hangul_boundary_contrast": contrast,
        "decision": {
            "hangul_boundary_router_hypothesis_supported": contrast["gate"][
                "overall_pass"
            ],
            "disjoint_actual_design_authorized": contrast["gate"]["overall_pass"],
            "efficiency_claim": False,
        },
    }
