"""Prefix-causal boundary policies for Phase 0 comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import unicodedata
from typing import Iterable, Sequence

from .entropy import PositionScore
from .unicode_audit import is_cjk_ideograph, is_hangul_syllable
from .utf8 import is_continuation_byte, scan_prefix_states


def _normalized_boundaries(data: bytes, boundaries: Iterable[int]) -> tuple[int, ...]:
    if not data:
        return ()
    normalized = {0}
    normalized.update(index for index in boundaries if 0 < index < len(data))
    return tuple(sorted(normalized))


def _is_space_or_punctuation(codepoint: int | None) -> bool:
    if codepoint is None:
        return False
    character = chr(codepoint)
    return character.isspace() or unicodedata.category(character).startswith("P")


class BoundaryPolicy(ABC):
    name: str

    @abstractmethod
    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        raise NotImplementedError

    def score_evaluations(self, data: bytes) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class FixedStridePolicy(BoundaryPolicy):
    stride: int

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    @property
    def name(self) -> str:
        return f"fixed_byte_{self.stride}"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        return _normalized_boundaries(data, range(0, len(data), self.stride))


@dataclass(frozen=True, slots=True)
class CodepointAlignedStridePolicy(BoundaryPolicy):
    budget: int

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")

    @property
    def name(self) -> str:
        return f"codepoint_stride_{self.budget}"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        states = scan_prefix_states(data)
        selected = [0]
        last = 0
        for index in range(1, len(data)):
            if index - last >= self.budget and states[index].at_codepoint_boundary:
                selected.append(index)
                last = index
        return _normalized_boundaries(data, selected)


def is_spacebyte_spacelike(value: int) -> bool:
    is_ascii_letter = 0x41 <= value <= 0x5A or 0x61 <= value <= 0x7A
    is_ascii_digit = 0x30 <= value <= 0x39
    return not is_ascii_letter and not is_ascii_digit and not is_continuation_byte(value)


@dataclass(frozen=True, slots=True)
class SpaceBytePolicy(BoundaryPolicy):
    name: str = "spacebyte_compatible"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        selected = [0]
        for index in range(1, len(data)):
            previous_is_spacelike = is_spacebyte_spacelike(data[index - 1])
            before_previous_is_spacelike = (
                index >= 2 and is_spacebyte_spacelike(data[index - 2])
            )
            if previous_is_spacelike and not before_previous_is_spacelike:
                selected.append(index)
        return _normalized_boundaries(data, selected)


@dataclass(frozen=True, slots=True)
class HangulSyllablePolicy(BoundaryPolicy):
    name: str = "hangul_syllable"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        states = scan_prefix_states(data)
        selected = [0]
        for index in range(1, len(data)):
            codepoint = states[index].completed_codepoint
            if codepoint is not None and is_hangul_syllable(codepoint):
                selected.append(index)
        return _normalized_boundaries(data, selected)


@dataclass(frozen=True, slots=True)
class CJKIdeographPolicy(BoundaryPolicy):
    name: str = "cjk_ideograph"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        states = scan_prefix_states(data)
        selected = [0]
        for index in range(1, len(data)):
            codepoint = states[index].completed_codepoint
            if codepoint is not None and is_cjk_ideograph(codepoint):
                selected.append(index)
        return _normalized_boundaries(data, selected)


@dataclass(frozen=True, slots=True)
class EojeolCappedPolicy(BoundaryPolicy):
    max_patch_bytes: int

    def __post_init__(self) -> None:
        if self.max_patch_bytes <= 0:
            raise ValueError("max_patch_bytes must be positive")

    @property
    def name(self) -> str:
        return f"eojeol_cap_{self.max_patch_bytes}"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        states = scan_prefix_states(data)
        selected = [0]
        last = 0
        for index in range(1, len(data)):
            completed = states[index].completed_codepoint
            delimiter = _is_space_or_punctuation(completed)
            capped = (
                index - last >= self.max_patch_bytes
                and states[index].at_codepoint_boundary
            )
            if delimiter or capped:
                selected.append(index)
                last = index
        return _normalized_boundaries(data, selected)


@dataclass(frozen=True, slots=True)
class EntropyPolicy(BoundaryPolicy):
    threshold: float
    min_patch_bytes: int = 1
    label: str = "entropy"

    def __post_init__(self) -> None:
        if self.min_patch_bytes <= 0:
            raise ValueError("min_patch_bytes must be positive")

    @property
    def name(self) -> str:
        return f"{self.label}_t{self.threshold:.6f}"

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        if scores is None or len(scores) != len(data):
            raise ValueError("entropy policy requires one score per byte")
        selected = [0]
        last = 0
        for index in range(1, len(data)):
            if (
                index - last >= self.min_patch_bytes
                and scores[index].entropy_bits >= self.threshold
            ):
                selected.append(index)
                last = index
        return _normalized_boundaries(data, selected)

    def score_evaluations(self, data: bytes) -> int:
        return max(0, len(data) - 1)


@dataclass(frozen=True, slots=True)
class CandidateEntropyPolicy(BoundaryPolicy):
    threshold: float
    min_patch_bytes: int = 1
    max_patch_bytes: int | None = None
    label: str = "candidate_entropy"

    def __post_init__(self) -> None:
        if self.min_patch_bytes <= 0:
            raise ValueError("min_patch_bytes must be positive")
        if self.max_patch_bytes is not None and self.max_patch_bytes <= 0:
            raise ValueError("max_patch_bytes must be positive when set")

    @property
    def name(self) -> str:
        return f"{self.label}_t{self.threshold:.6f}"

    def candidate_indices(self, data: bytes) -> tuple[int, ...]:
        states = scan_prefix_states(data)
        return tuple(
            index
            for index in range(1, len(data))
            if states[index].at_codepoint_boundary
        )

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        if scores is None or len(scores) != len(data):
            raise ValueError("candidate entropy policy requires one score per byte")
        selected = [0]
        last = 0
        for index in self.candidate_indices(data):
            capped = (
                self.max_patch_bytes is not None
                and index - last >= self.max_patch_bytes
            )
            entropy_trigger = (
                index - last >= self.min_patch_bytes
                and scores[index].entropy_bits >= self.threshold
            )
            if capped or entropy_trigger:
                selected.append(index)
                last = index
        return _normalized_boundaries(data, selected)

    def score_evaluations(self, data: bytes) -> int:
        return len(self.candidate_indices(data))


@dataclass(frozen=True, slots=True)
class OrthographicCandidateEntropyPolicy(BoundaryPolicy):
    """Score entropy only after target-script characters or delimiters.

    The deterministic byte cap does not itself require an entropy score. Every
    decision uses only the already-consumed prefix.
    """

    threshold: float
    script: str
    max_patch_bytes: int = 24
    min_patch_bytes: int = 1
    label: str = "orthographic_candidate_entropy"

    def __post_init__(self) -> None:
        if self.script not in {"hangul", "cjk"}:
            raise ValueError("script must be 'hangul' or 'cjk'")
        if self.max_patch_bytes <= 0:
            raise ValueError("max_patch_bytes must be positive")
        if self.min_patch_bytes <= 0:
            raise ValueError("min_patch_bytes must be positive")

    @property
    def name(self) -> str:
        return f"{self.label}_{self.script}_t{self.threshold:.6f}"

    def _matches_script(self, codepoint: int | None) -> bool:
        if codepoint is None:
            return False
        if self.script == "hangul":
            return is_hangul_syllable(codepoint)
        return is_cjk_ideograph(codepoint)

    def candidate_indices(self, data: bytes) -> tuple[int, ...]:
        states = scan_prefix_states(data)
        return tuple(
            index
            for index in range(1, len(data))
            if states[index].at_codepoint_boundary
            and (
                self._matches_script(states[index].completed_codepoint)
                or _is_space_or_punctuation(states[index].completed_codepoint)
            )
        )

    def boundaries(
        self,
        data: bytes,
        scores: Sequence[PositionScore] | None = None,
    ) -> tuple[int, ...]:
        if scores is None or len(scores) != len(data):
            raise ValueError("orthographic entropy policy requires one score per byte")
        states = scan_prefix_states(data)
        candidates = set(self.candidate_indices(data))
        selected = [0]
        last = 0
        for index in range(1, len(data)):
            if not states[index].at_codepoint_boundary:
                continue
            capped = index - last >= self.max_patch_bytes
            entropy_trigger = (
                index in candidates
                and index - last >= self.min_patch_bytes
                and scores[index].entropy_bits >= self.threshold
            )
            if capped or entropy_trigger:
                selected.append(index)
                last = index
        return _normalized_boundaries(data, selected)

    def score_evaluations(self, data: bytes) -> int:
        return len(self.candidate_indices(data))


def calibrate_entropy_threshold(
    scored_sequences: Sequence[tuple[bytes, Sequence[PositionScore]]],
    target_average_patch_bytes: float,
    candidate_only: bool = False,
    candidate_policy: CandidateEntropyPolicy
    | OrthographicCandidateEntropyPolicy
    | None = None,
) -> float:
    if target_average_patch_bytes <= 0:
        raise ValueError("target_average_patch_bytes must be positive")

    total_bytes = sum(len(data) for data, _ in scored_sequences)
    nonempty_records = sum(1 for data, _ in scored_sequences if data)
    desired_patches = max(
        nonempty_records,
        round(total_bytes / target_average_patch_bytes),
    )
    desired_extra_boundaries = desired_patches - nonempty_records

    eligible: list[float] = []
    for data, scores in scored_sequences:
        if len(data) != len(scores):
            raise ValueError("score length mismatch")
        if candidate_only and candidate_policy is not None:
            raise ValueError("choose candidate_only or candidate_policy, not both")
        if candidate_policy is not None:
            indices = candidate_policy.candidate_indices(data)
        elif candidate_only:
            indices = CandidateEntropyPolicy(math.inf).candidate_indices(data)
        else:
            indices = range(1, len(data))
        eligible.extend(scores[index].entropy_bits for index in indices)

    if desired_extra_boundaries <= 0 or not eligible:
        return math.inf
    if desired_extra_boundaries >= len(eligible):
        return -math.inf

    eligible.sort(reverse=True)
    lower_selected = eligible[desired_extra_boundaries - 1]
    upper_unselected = eligible[desired_extra_boundaries]
    if lower_selected == upper_unselected:
        return lower_selected
    return (lower_selected + upper_unselected) / 2.0


def assert_prefix_causal(
    policy: BoundaryPolicy,
    data: bytes,
    scores: Sequence[PositionScore] | None = None,
) -> None:
    full = policy.boundaries(data, scores)
    for end in range(1, len(data) + 1):
        prefix_scores = scores[:end] if scores is not None else None
        observed = policy.boundaries(data[:end], prefix_scores)
        expected = tuple(index for index in full if index < end)
        if observed != expected:
            raise AssertionError(
                f"{policy.name} is not prefix-causal at prefix {end}: "
                f"observed={observed}, expected={expected}"
            )
