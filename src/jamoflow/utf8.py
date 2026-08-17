"""Small streaming UTF-8 state machine used by causal policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable


@dataclass(frozen=True, slots=True)
class PrefixState:
    """Parser state after consuming a prefix.

    At index ``t``, this describes ``data[:t]``. ``completed_codepoint`` is set
    only when byte ``t-1`` completed a valid codepoint.
    """

    at_codepoint_boundary: bool
    completed_codepoint: int | None
    invalid_sequences: int


@dataclass(frozen=True, slots=True)
class CodepointSpan:
    start: int
    end: int
    codepoint: int | None
    valid: bool


@dataclass(frozen=True, slots=True)
class StrictUtf8State:
    """Strict RFC 3629 decoder state after a valid byte prefix.

    ``remaining`` is the number of continuation bytes required to close the
    current Unicode scalar.  ``lower`` and ``upper`` constrain the next byte;
    the restricted first-continuation ranges reject overlong sequences,
    surrogates, and values above U+10FFFF before they enter a generated stream.
    """

    remaining: int = 0
    lower: int = 0
    upper: int = 0
    valid: bool = True

    def __post_init__(self) -> None:
        if not self.valid:
            if (self.remaining, self.lower, self.upper) != (0, 0, 0):
                raise ValueError("an invalid UTF-8 state cannot remain open")
            return
        if self.remaining == 0:
            if (self.lower, self.upper) != (0, 0):
                raise ValueError("a closed UTF-8 state cannot have a byte range")
        elif not (
            1 <= self.remaining <= 3
            and 0x80 <= self.lower <= self.upper <= 0xBF
        ):
            raise ValueError("malformed strict UTF-8 continuation state")

    @property
    def at_codepoint_boundary(self) -> bool:
        return self.valid and self.remaining == 0


STRICT_UTF8_INITIAL_STATE = StrictUtf8State()
STRICT_UTF8_INVALID_STATE = StrictUtf8State(valid=False)


@dataclass(frozen=True, slots=True)
class StrictUtf8TokenTransitions:
    """Precompiled strict-DFA transitions for atomic variable-byte units."""

    states: tuple[StrictUtf8State, ...]
    token_byte_lengths: tuple[int, ...]
    next_state_indices: tuple[tuple[int, ...], ...]
    allowed_token_ids: tuple[tuple[int, ...], ...]
    token_bytes_sha256: str
    transition_table_sha256: str

    @property
    def vocabulary_size(self) -> int:
        return len(self.token_byte_lengths)

    @property
    def maximum_token_bytes(self) -> int:
        return max(self.token_byte_lengths)

    def transition(self, state_index: int, token_id: int) -> int | None:
        if not 0 <= state_index < len(self.states):
            raise ValueError("UTF-8 token state index is invalid")
        if not 0 <= token_id < self.vocabulary_size:
            raise ValueError("UTF-8 token id is invalid")
        output = self.next_state_indices[state_index][token_id]
        return None if output < 0 else output


def advance_strict_utf8(
    state: StrictUtf8State,
    value: int,
) -> StrictUtf8State:
    """Advance a strict non-recovering UTF-8 DFA by one byte."""

    if not isinstance(value, int) or not 0 <= value <= 0xFF:
        raise ValueError("UTF-8 DFA input must be one byte")
    if not state.valid:
        return state
    if state.remaining:
        if not state.lower <= value <= state.upper:
            return STRICT_UTF8_INVALID_STATE
        remaining = state.remaining - 1
        return (
            StrictUtf8State(remaining, 0x80, 0xBF)
            if remaining
            else STRICT_UTF8_INITIAL_STATE
        )
    if value <= 0x7F:
        return STRICT_UTF8_INITIAL_STATE
    if 0xC2 <= value <= 0xDF:
        return StrictUtf8State(1, 0x80, 0xBF)
    if value == 0xE0:
        return StrictUtf8State(2, 0xA0, 0xBF)
    if 0xE1 <= value <= 0xEC or 0xEE <= value <= 0xEF:
        return StrictUtf8State(2, 0x80, 0xBF)
    if value == 0xED:
        return StrictUtf8State(2, 0x80, 0x9F)
    if value == 0xF0:
        return StrictUtf8State(3, 0x90, 0xBF)
    if 0xF1 <= value <= 0xF3:
        return StrictUtf8State(3, 0x80, 0xBF)
    if value == 0xF4:
        return StrictUtf8State(3, 0x80, 0x8F)
    return STRICT_UTF8_INVALID_STATE


def strict_utf8_state(data: bytes) -> StrictUtf8State:
    """Return the strict non-recovering DFA state for ``data``."""

    state = STRICT_UTF8_INITIAL_STATE
    for value in data:
        state = advance_strict_utf8(state, value)
        if not state.valid:
            break
    return state


def strict_utf8_allowed_ranges(
    state: StrictUtf8State,
    *,
    remaining_bytes_after_choice: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return inclusive next-byte ranges preserving strict UTF-8 validity.

    When a finite remaining budget is supplied, lead bytes that could not be
    closed within that budget are excluded.  ``None`` is the streaming form:
    it constrains transition legality without distorting generation near an
    arbitrary fixed horizon.
    """

    if remaining_bytes_after_choice is not None and (
        not isinstance(remaining_bytes_after_choice, int)
        or remaining_bytes_after_choice < 0
    ):
        raise ValueError("remaining byte budget must be non-negative")
    if not state.valid:
        return ()
    if state.remaining:
        if (
            remaining_bytes_after_choice is not None
            and state.remaining - 1 > remaining_bytes_after_choice
        ):
            return ()
        return ((state.lower, state.upper),)

    ranges: list[tuple[int, int]] = [(0x00, 0x7F)]
    if remaining_bytes_after_choice is None or remaining_bytes_after_choice >= 1:
        ranges.append((0xC2, 0xDF))
    if remaining_bytes_after_choice is None or remaining_bytes_after_choice >= 2:
        ranges.append((0xE0, 0xEF))
    if remaining_bytes_after_choice is None or remaining_bytes_after_choice >= 3:
        ranges.append((0xF0, 0xF4))
    return tuple(ranges)


def strict_utf8_reachable_states() -> tuple[StrictUtf8State, ...]:
    """Return every valid state reachable by the strict DFA."""

    return (
        STRICT_UTF8_INITIAL_STATE,
        StrictUtf8State(1, 0x80, 0xBF),
        StrictUtf8State(2, 0x80, 0x9F),
        StrictUtf8State(2, 0x80, 0xBF),
        StrictUtf8State(2, 0xA0, 0xBF),
        StrictUtf8State(3, 0x80, 0x8F),
        StrictUtf8State(3, 0x80, 0xBF),
        StrictUtf8State(3, 0x90, 0xBF),
    )


def compile_strict_utf8_token_transitions(
    token_bytes: tuple[bytes, ...],
) -> StrictUtf8TokenTransitions:
    """Compile state-specific validity and next state for every model unit."""

    if not token_bytes or any(
        not isinstance(values, bytes) or not values for values in token_bytes
    ):
        raise ValueError("UTF-8 token table requires non-empty byte strings")
    states = strict_utf8_reachable_states()
    state_indices = {state: index for index, state in enumerate(states)}
    next_rows: list[tuple[int, ...]] = []
    allowed_rows: list[tuple[int, ...]] = []
    for state in states:
        next_indices: list[int] = []
        allowed: list[int] = []
        for token_id, values in enumerate(token_bytes):
            advanced = state
            for value in values:
                advanced = advance_strict_utf8(advanced, value)
                if not advanced.valid:
                    break
            next_index = state_indices.get(advanced, -1)
            next_indices.append(next_index)
            if next_index >= 0:
                allowed.append(token_id)
        if not allowed:
            raise ValueError("UTF-8 token vocabulary strands a reachable state")
        next_rows.append(tuple(next_indices))
        allowed_rows.append(tuple(allowed))
    token_digest = hashlib.sha256()
    for values in token_bytes:
        token_digest.update(len(values).to_bytes(8, "little"))
        token_digest.update(values)
    transition_digest = hashlib.sha256()
    for row in next_rows:
        for value in row:
            transition_digest.update(value.to_bytes(4, "little", signed=True))
    return StrictUtf8TokenTransitions(
        states=states,
        token_byte_lengths=tuple(len(values) for values in token_bytes),
        next_state_indices=tuple(next_rows),
        allowed_token_ids=tuple(allowed_rows),
        token_bytes_sha256=token_digest.hexdigest(),
        transition_table_sha256=transition_digest.hexdigest(),
    )


def is_continuation_byte(value: int) -> bool:
    return 0x80 <= value <= 0xBF


def _lead_spec(value: int) -> tuple[int, int, int] | None:
    if value <= 0x7F:
        return (0, value, 0)
    if 0xC2 <= value <= 0xDF:
        return (1, value & 0x1F, 0x80)
    if 0xE0 <= value <= 0xEF:
        return (2, value & 0x0F, 0x800)
    if 0xF0 <= value <= 0xF4:
        return (3, value & 0x07, 0x10000)
    return None


def _valid_codepoint(codepoint: int, minimum: int) -> bool:
    return (
        codepoint >= minimum
        and codepoint <= 0x10FFFF
        and not 0xD800 <= codepoint <= 0xDFFF
    )


def scan_prefix_states(data: bytes) -> list[PrefixState]:
    states = [PrefixState(True, None, 0)]
    remaining = 0
    accumulator = 0
    minimum = 0
    invalid_sequences = 0

    for value in data:
        completed: int | None = None
        if remaining == 0:
            spec = _lead_spec(value)
            if spec is None:
                invalid_sequences += 1
            else:
                remaining, accumulator, minimum = spec
                if remaining == 0:
                    completed = accumulator
        elif is_continuation_byte(value):
            accumulator = (accumulator << 6) | (value & 0x3F)
            remaining -= 1
            if remaining == 0:
                if _valid_codepoint(accumulator, minimum):
                    completed = accumulator
                else:
                    invalid_sequences += 1
        else:
            # The previous sequence is malformed. Reprocess this byte as a new
            # leading byte so an ASCII byte can immediately restore the parser.
            invalid_sequences += 1
            remaining = 0
            spec = _lead_spec(value)
            if spec is None:
                invalid_sequences += 1
            else:
                remaining, accumulator, minimum = spec
                if remaining == 0:
                    completed = accumulator

        states.append(
            PrefixState(
                at_codepoint_boundary=remaining == 0,
                completed_codepoint=completed,
                invalid_sequences=invalid_sequences,
            )
        )
    return states


def prefix_boundary_mask(data: bytes) -> bytearray:
    """Return one byte per prefix indicating a complete UTF-8 state.

    The result has ``len(data) + 1`` entries. Entry ``t`` describes
    ``data[:t]``, matching :func:`scan_prefix_states`, but avoids allocating a
    Python object for every byte. This representation is used for multi-million
    byte neural experiment streams.
    """

    mask = bytearray(len(data) + 1)
    mask[0] = 1
    remaining = 0
    accumulator = 0
    minimum = 0

    for index, value in enumerate(data, start=1):
        if remaining == 0:
            spec = _lead_spec(value)
            if spec is not None:
                remaining, accumulator, minimum = spec
        elif is_continuation_byte(value):
            accumulator = (accumulator << 6) | (value & 0x3F)
            remaining -= 1
            if remaining == 0 and not _valid_codepoint(accumulator, minimum):
                # Invalid completed sequences recover at this prefix just like
                # scan_prefix_states; the mask describes parser state, not
                # Unicode validity.
                remaining = 0
        else:
            remaining = 0
            spec = _lead_spec(value)
            if spec is not None:
                remaining, accumulator, minimum = spec

        mask[index] = remaining == 0

    return mask


def prefix_codepoint_predicate_mask(
    data: bytes,
    predicate: Callable[[int], bool],
) -> bytearray:
    """Mark prefixes where the just-completed codepoint matches a predicate.

    Like :func:`prefix_boundary_mask`, this uses constant parser state rather
    than allocating one object per byte. It is suitable for corpus-scale
    causal feature masks. The returned array has ``len(data) + 1`` entries;
    entry ``t`` depends only on ``data[:t]``.
    """

    mask = bytearray(len(data) + 1)
    remaining = 0
    accumulator = 0
    minimum = 0

    for index, value in enumerate(data, start=1):
        completed: int | None = None
        if remaining == 0:
            spec = _lead_spec(value)
            if spec is not None:
                remaining, accumulator, minimum = spec
                if remaining == 0:
                    completed = accumulator
        elif is_continuation_byte(value):
            accumulator = (accumulator << 6) | (value & 0x3F)
            remaining -= 1
            if remaining == 0 and _valid_codepoint(accumulator, minimum):
                completed = accumulator
        else:
            remaining = 0
            spec = _lead_spec(value)
            if spec is not None:
                remaining, accumulator, minimum = spec
                if remaining == 0:
                    completed = accumulator

        if completed is not None and predicate(completed):
            mask[index] = 1

    return mask


def codepoint_spans(data: bytes) -> list[CodepointSpan]:
    spans: list[CodepointSpan] = []
    index = 0
    while index < len(data):
        spec = _lead_spec(data[index])
        if spec is None:
            spans.append(CodepointSpan(index, index + 1, None, False))
            index += 1
            continue

        remaining, accumulator, minimum = spec
        if remaining == 0:
            spans.append(CodepointSpan(index, index + 1, accumulator, True))
            index += 1
            continue

        end = index + remaining + 1
        if end > len(data) or any(
            not is_continuation_byte(value) for value in data[index + 1 : end]
        ):
            spans.append(CodepointSpan(index, index + 1, None, False))
            index += 1
            continue

        for value in data[index + 1 : end]:
            accumulator = (accumulator << 6) | (value & 0x3F)
        valid = _valid_codepoint(accumulator, minimum)
        spans.append(
            CodepointSpan(
                start=index,
                end=end if valid else index + 1,
                codepoint=accumulator if valid else None,
                valid=valid,
            )
        )
        index = end if valid else index + 1
    return spans
