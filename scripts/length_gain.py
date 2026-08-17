"""Deterministic train-only construction for fixed-budget byte vocabularies.

The constructor starts from the complete byte alphabet.  At every round it
examines n-grams in the *current* minimum-token segmentation, ranks them by
their exact left-to-right non-overlapping token saving, adds a small batch of
direct byte pieces, and resegments the corpus.  The implementation is kept in
``scripts/`` because it is an experimental tokenizer constructor, not part of
the historical JamoFlow model package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import statistics
import time
from typing import Any, Literal, Sequence

import numpy as np

from fixed_byte_tokenizer import build_fixed_byte_tokenizer, encode_raw_bytes
from jamoflow.publication_bpe import byte_bpe_token_bytes


ScoreKind = Literal["immediate_saving", "current_token_length"]
BYTE_ALPHABET_SIZE = 256
DEFAULT_VOCABULARY_SIZE = 2_048
DEFAULT_MAXIMUM_PIECE_BYTES = 48
_HASH_MULTIPLIER_1 = np.uint64(11_400_714_819_323_198_485)
_HASH_MULTIPLIER_2 = np.uint64(14_029_467_366_897_019_727)
_HASH_DTYPE = np.dtype([("first", "<u8"), ("second", "<u8")])


@dataclass(frozen=True, slots=True)
class LengthGainCandidate:
    raw: bytes
    token_ids: tuple[int, ...]
    token_arity: int
    overlapping_occurrences: int
    nonoverlapping_occurrences: int
    score: int


@dataclass(frozen=True, slots=True)
class LengthGainRound:
    round_index: int
    vocabulary_size: int
    token_count_before: int
    token_count_after: int
    realized_token_reduction: int
    selected_score_sum: int
    selected_candidate_count: int
    best_candidate_score: int
    best_candidate_nonoverlapping_occurrences: int
    maximum_selected_piece_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LengthGainTrainingResult:
    pieces: tuple[bytes, ...]
    rounds: tuple[LengthGainRound, ...]
    initial_token_count: int
    final_token_count: int
    token_ids_sha256: str
    ordered_pieces_sha256: str


@dataclass(frozen=True, slots=True)
class LengthGainOpportunityMetrics:
    role: str
    vocabulary_size: int
    calibration_complete_utf8_bytes: int
    calibration_trailing_bytes: int
    calibration_token_count: int
    calibration_token_ids_sha256: str
    vocabulary_used: int
    maximum_used_piece_bytes: int
    encode_seconds: tuple[float, ...]
    encode_median_megabytes_per_second: float
    prompt_token_counts: tuple[int, ...]
    continuation_token_counts: tuple[int, ...]
    joint_token_counts: tuple[int, ...]
    multibyte_vocabulary_count: int
    strict_utf8_multibyte_vocabulary_count: int
    hangul_multibyte_vocabulary_count: int
    cross_eojeol_multibyte_vocabulary_count: int
    exact_roundtrip: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_pieces_sha256(pieces: Sequence[bytes]) -> str:
    digest = hashlib.sha256(b"JamoFlow/length-gain-pieces/v1\0")
    digest.update(len(pieces).to_bytes(8, "big"))
    for piece in pieces:
        digest.update(len(piece).to_bytes(8, "big"))
        digest.update(piece)
    return digest.hexdigest()


def _token_ids_sha256(token_ids: np.ndarray) -> str:
    values = np.ascontiguousarray(token_ids, dtype="<u2")
    digest = hashlib.sha256(b"JamoFlow/length-gain-token-ids/v1\0")
    digest.update(len(values).to_bytes(8, "big"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _rolling_hashes(windows: np.ndarray) -> np.ndarray:
    if windows.ndim != 2 or windows.shape[1] < 2:
        raise ValueError("length-gain windows differ")
    first = np.zeros(len(windows), dtype=np.uint64)
    second = np.zeros(len(windows), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for column in range(windows.shape[1]):
            values = windows[:, column].astype(np.uint64) + np.uint64(1)
            first = first * _HASH_MULTIPLIER_1 + values
            second = second * _HASH_MULTIPLIER_2 + values
    output = np.empty(len(windows), dtype=_HASH_DTYPE)
    output["first"] = first
    output["second"] = second
    return output


def _nonoverlapping_count(positions: np.ndarray, width: int) -> int:
    if positions.ndim != 1 or width <= 0:
        raise ValueError("length-gain occurrence coordinates differ")
    count = 0
    next_allowed = -1
    for raw_position in positions:
        position = int(raw_position)
        if position < next_allowed:
            continue
        count += 1
        next_allowed = position + width
    return count


def _score(occurrences: int, arity: int, score_kind: ScoreKind) -> int:
    if score_kind == "immediate_saving":
        return occurrences * (arity - 1)
    if score_kind == "current_token_length":
        return occurrences * arity
    raise ValueError("unknown length-gain score")


def _rank(candidate: LengthGainCandidate) -> tuple[int, int, int, bytes]:
    return (
        -candidate.score,
        -candidate.nonoverlapping_occurrences,
        -candidate.token_arity,
        candidate.raw,
    )


def _candidates_for_arity(
    token_ids: np.ndarray,
    pieces: Sequence[bytes],
    existing: set[bytes],
    *,
    arity: int,
    maximum_piece_bytes: int,
    score_kind: ScoreKind,
    group_limit: int,
) -> tuple[list[LengthGainCandidate], int]:
    """Return exact candidates plus an upper bound for skipped hash groups.

    Two independent 64-bit rolling hashes are only an indexing accelerator.
    Every examined group is checked against its exact token tuple.  A hash
    collision therefore fails closed instead of changing the vocabulary.
    """

    windows = np.lib.stride_tricks.sliding_window_view(token_ids, arity)
    hashes = _rolling_hashes(windows)
    unique, first_indices, counts = np.unique(
        hashes,
        return_index=True,
        return_counts=True,
    )
    if len(counts) == 0:
        return [], 0
    take = min(group_limit, len(counts))
    if take == len(counts):
        selected_groups = np.arange(len(counts), dtype=np.int64)
        skipped_upper_bound = 0
    else:
        cutoff = int(np.partition(counts, len(counts) - take)[len(counts) - take])
        selected_groups = np.flatnonzero(counts >= cutoff)
        skipped_counts = counts[counts < cutoff]
        skipped_upper_bound = (
            _score(int(skipped_counts.max()), arity, score_kind)
            if len(skipped_counts)
            else 0
        )

    output: list[LengthGainCandidate] = []
    for group_index in selected_groups:
        representative_index = int(first_indices[group_index])
        representative = tuple(int(value) for value in windows[representative_index])
        key = unique[group_index]
        matches = np.flatnonzero(
            (hashes["first"] == key["first"])
            & (hashes["second"] == key["second"])
        )
        if len(matches) != int(counts[group_index]):
            raise AssertionError("length-gain hash group count differs")
        if not np.all(windows[matches] == np.asarray(representative, dtype=token_ids.dtype)):
            raise RuntimeError("length-gain double-hash collision")
        raw = b"".join(pieces[token_id] for token_id in representative)
        if (
            raw in existing
            or b"\n" in raw
            or not 1 < len(raw) <= maximum_piece_bytes
        ):
            continue
        nonoverlapping = _nonoverlapping_count(matches, arity)
        output.append(
            LengthGainCandidate(
                raw=raw,
                token_ids=representative,
                token_arity=arity,
                overlapping_occurrences=int(counts[group_index]),
                nonoverlapping_occurrences=nonoverlapping,
                score=_score(nonoverlapping, arity, score_kind),
            )
        )
    output.sort(key=_rank)
    return output, skipped_upper_bound


def rank_length_gain_candidates(
    token_ids: np.ndarray,
    pieces: Sequence[bytes],
    *,
    batch_size: int,
    maximum_token_arity: int,
    maximum_piece_bytes: int,
    score_kind: ScoreKind,
    initial_group_limit: int = 64,
) -> tuple[LengthGainCandidate, ...]:
    if (
        token_ids.ndim != 1
        or token_ids.dtype != np.uint16
        or len(token_ids) < maximum_token_arity
        or not 1 <= batch_size
        or not 2 <= maximum_token_arity <= 16
        or maximum_piece_bytes <= 1
        or initial_group_limit < batch_size
    ):
        raise ValueError("length-gain candidate coordinates differ")
    existing = set(pieces)
    group_limit = initial_group_limit
    while True:
        candidates: list[LengthGainCandidate] = []
        skipped_upper_bound = 0
        for arity in range(2, maximum_token_arity + 1):
            if len(token_ids) < arity:
                continue
            rows, upper = _candidates_for_arity(
                token_ids,
                pieces,
                existing,
                arity=arity,
                maximum_piece_bytes=maximum_piece_bytes,
                score_kind=score_kind,
                group_limit=group_limit,
            )
            candidates.extend(rows)
            skipped_upper_bound = max(skipped_upper_bound, upper)
        by_raw: dict[bytes, LengthGainCandidate] = {}
        for candidate in candidates:
            prior = by_raw.get(candidate.raw)
            if prior is None or _rank(candidate) < _rank(prior):
                by_raw[candidate.raw] = candidate
        ranked = sorted(by_raw.values(), key=_rank)
        if len(ranked) < batch_size:
            if skipped_upper_bound == 0:
                raise RuntimeError("length-gain corpus has too few eligible candidates")
            group_limit *= 2
            continue
        selected = tuple(ranked[:batch_size])
        # Strict inequality also proves that an unexamined tie cannot change
        # deterministic raw-byte tie breaking.
        if selected[-1].score > skipped_upper_bound:
            return selected
        group_limit *= 2


def train_length_gain_vocabulary(
    raw: bytes,
    *,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    batch_size: int = 8,
    maximum_token_arity: int = 8,
    maximum_piece_bytes: int = DEFAULT_MAXIMUM_PIECE_BYTES,
    score_kind: ScoreKind = "immediate_saving",
) -> LengthGainTrainingResult:
    if (
        not isinstance(raw, bytes)
        or not raw
        or vocabulary_size <= BYTE_ALPHABET_SIZE
        or (vocabulary_size - BYTE_ALPHABET_SIZE) % batch_size
    ):
        raise ValueError("length-gain training coordinates differ")
    pieces: list[bytes] = [bytes((value,)) for value in range(256)]
    token_ids = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    initial_token_count = len(token_ids)
    traces: list[LengthGainRound] = []
    rounds = (vocabulary_size - BYTE_ALPHABET_SIZE) // batch_size
    for round_index in range(rounds):
        selected = rank_length_gain_candidates(
            token_ids,
            pieces,
            batch_size=batch_size,
            maximum_token_arity=maximum_token_arity,
            maximum_piece_bytes=maximum_piece_bytes,
            score_kind=score_kind,
        )
        before = len(token_ids)
        pieces.extend(candidate.raw for candidate in selected)
        tokenizer = build_fixed_byte_tokenizer(
            pieces,
            segmentation="minimum_token_dp",
            maximum_piece_bytes=maximum_piece_bytes,
        )
        token_ids = np.asarray(encode_raw_bytes(tokenizer, raw), dtype=np.uint16)
        if len(token_ids) >= before:
            raise RuntimeError("length-gain round did not reduce train tokens")
        traces.append(
            LengthGainRound(
                round_index=round_index,
                vocabulary_size=len(pieces),
                token_count_before=before,
                token_count_after=len(token_ids),
                realized_token_reduction=before - len(token_ids),
                selected_score_sum=sum(candidate.score for candidate in selected),
                selected_candidate_count=len(selected),
                best_candidate_score=selected[0].score,
                best_candidate_nonoverlapping_occurrences=(
                    selected[0].nonoverlapping_occurrences
                ),
                maximum_selected_piece_bytes=max(len(candidate.raw) for candidate in selected),
            )
        )
    values = tuple(pieces)
    return LengthGainTrainingResult(
        pieces=values,
        rounds=tuple(traces),
        initial_token_count=initial_token_count,
        final_token_count=len(token_ids),
        token_ids_sha256=_token_ids_sha256(token_ids),
        ordered_pieces_sha256=_ordered_pieces_sha256(values),
    )


def training_public_metadata(result: LengthGainTrainingResult) -> dict[str, Any]:
    return {
        "initial_token_count": result.initial_token_count,
        "final_token_count": result.final_token_count,
        "ordered_pieces_sha256": result.ordered_pieces_sha256,
        "round_count": len(result.rounds),
        "rounds": [row.to_dict() for row in result.rounds],
        "token_ids_sha256": result.token_ids_sha256,
        "vocabulary_size": len(result.pieces),
    }


def _complete_utf8_prefix(raw: bytes) -> tuple[bytes, bytes]:
    try:
        raw.decode("utf-8", errors="strict")
        return raw, b""
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(raw):
            raise ValueError("length-gain stream contains invalid interior UTF-8") from error
        prefix = raw[: error.start]
        suffix = raw[error.start :]
        prefix.decode("utf-8", errors="strict")
        if not 1 <= len(suffix) <= 3:
            raise ValueError("length-gain UTF-8 suffix differs")
        return prefix, suffix


def _canonical_text_pieces(raw: bytes) -> tuple[tuple[str, bytes], ...]:
    complete, _ = _complete_utf8_prefix(raw)
    output = []
    for index, row in enumerate(complete.split(b"\n")):
        piece = row if index == 0 else b"\n" + row
        if piece:
            output.append((piece.decode("utf-8", errors="strict"), piece))
    if not output or b"".join(piece for _, piece in output) != complete:
        raise ValueError("length-gain canonical text pieces differ")
    return tuple(output)


def _encode_text(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False).ids)


def _count_sha256(values: Sequence[int]) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    digest = hashlib.sha256(b"JamoFlow/length-gain-counts/v1\0")
    digest.update(len(array).to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _vocabulary_structure(pieces: Sequence[bytes]) -> dict[str, int]:
    multibyte = [piece for piece in pieces if len(piece) > 1]
    strict = 0
    hangul = 0
    cross = 0
    for piece in multibyte:
        try:
            text = piece.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        strict += 1
        if any("가" <= character <= "힣" for character in text):
            hangul += 1
        nonspace = [index for index, value in enumerate(text) if not value.isspace()]
        if nonspace and any(
            nonspace[0] < index < nonspace[-1] and value.isspace()
            for index, value in enumerate(text)
        ):
            cross += 1
    return {
        "multibyte_vocabulary_count": len(multibyte),
        "strict_utf8_multibyte_vocabulary_count": strict,
        "hangul_multibyte_vocabulary_count": hangul,
        "cross_eojeol_multibyte_vocabulary_count": cross,
    }


def evaluate_length_gain_opportunity(
    *,
    role: str,
    tokenizer,
    calibration_raw: bytes,
    prompts: np.ndarray,
    continuations: np.ndarray,
    encode_repetitions: int = 3,
) -> LengthGainOpportunityMetrics:
    if (
        not role
        or prompts.ndim != 2
        or prompts.shape != continuations.shape
        or prompts.dtype != np.uint8
        or continuations.dtype != np.uint8
        or encode_repetitions <= 0
    ):
        raise ValueError("length-gain opportunity coordinates differ")
    pieces = byte_bpe_token_bytes(tokenizer)
    complete, trailing = _complete_utf8_prefix(calibration_raw)
    counts: dict[int, int] = {}
    digest = hashlib.sha256(b"JamoFlow/length-gain-calibration-ids/v1\0")
    token_count = 0
    for text, raw_piece in _canonical_text_pieces(calibration_raw):
        ids = _encode_text(tokenizer, text)
        if not ids or b"".join(pieces[value] for value in ids) != raw_piece:
            raise ValueError("length-gain calibration roundtrip differs")
        for token_id in ids:
            counts[token_id] = counts.get(token_id, 0) + 1
        array = np.ascontiguousarray(ids, dtype="<u4")
        digest.update(len(array).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
        token_count += len(ids)
    digest.update(token_count.to_bytes(8, "big"))

    timings = []
    text_pieces = _canonical_text_pieces(calibration_raw)
    for _ in range(encode_repetitions):
        start = time.perf_counter()
        repeated = sum(len(_encode_text(tokenizer, text)) for text, _ in text_pieces)
        timings.append(time.perf_counter() - start)
        if repeated != token_count:
            raise RuntimeError("length-gain encoding is nondeterministic")

    prompt_counts = []
    continuation_counts = []
    joint_counts = []
    for prompt, continuation in zip(prompts, continuations, strict=True):
        prompt_raw = bytes(prompt)
        continuation_raw = bytes(continuation)
        rows = []
        for raw in (prompt_raw, continuation_raw, prompt_raw + continuation_raw):
            ids = _encode_text(tokenizer, raw.decode("utf-8", errors="strict"))
            if b"".join(pieces[value] for value in ids) != raw:
                raise ValueError("length-gain case roundtrip differs")
            rows.append(len(ids))
        prompt_counts.append(rows[0])
        continuation_counts.append(rows[1])
        joint_counts.append(rows[2])

    structure = _vocabulary_structure(pieces)
    median_seconds = statistics.median(timings)
    return LengthGainOpportunityMetrics(
        role=role,
        vocabulary_size=len(pieces),
        calibration_complete_utf8_bytes=len(complete),
        calibration_trailing_bytes=len(trailing),
        calibration_token_count=token_count,
        calibration_token_ids_sha256=digest.hexdigest(),
        vocabulary_used=len(counts),
        maximum_used_piece_bytes=max(len(pieces[value]) for value in counts),
        encode_seconds=tuple(timings),
        encode_median_megabytes_per_second=len(complete) / 1_000_000 / median_seconds,
        prompt_token_counts=tuple(prompt_counts),
        continuation_token_counts=tuple(continuation_counts),
        joint_token_counts=tuple(joint_counts),
        exact_roundtrip=True,
        **structure,
    )


def length_gain_decision(
    metrics_by_role: dict[str, dict[str, Any]],
    *,
    baseline_role: str,
    primary_order: Sequence[str],
    warmup_cases: int,
    minimum_reduction: float,
) -> dict[str, Any]:
    if set(metrics_by_role) != {baseline_role, *primary_order}:
        raise ValueError("length-gain decision role set differs")
    baseline = metrics_by_role[baseline_role]
    baseline_calibration = int(baseline["calibration_token_count"])
    baseline_continuations = np.asarray(
        baseline["continuation_token_counts"], dtype=np.int64
    )[warmup_cases:]
    comparisons: dict[str, Any] = {}
    selected = None
    for role in primary_order:
        candidate = metrics_by_role[role]
        candidate_continuations = np.asarray(
            candidate["continuation_token_counts"], dtype=np.int64
        )[warmup_cases:]
        calibration_reduction = 1.0 - int(candidate["calibration_token_count"]) / baseline_calibration
        continuation_reduction = 1.0 - float(candidate_continuations.sum()) / float(
            baseline_continuations.sum()
        )
        passed = (
            calibration_reduction >= minimum_reduction
            and continuation_reduction >= minimum_reduction
        )
        comparisons[role] = {
            "calibration_token_reduction": calibration_reduction,
            "continuation_token_reduction": continuation_reduction,
            "overall_pass": passed,
        }
        if selected is None and passed:
            selected = role
    return {
        "comparisons": comparisons,
        "minimum_reduction": minimum_reduction,
        "selected_role": selected,
        "status": (
            "generic_length_gain_passed_for_korean_complete_control"
            if selected is not None
            else "same2k_length_gain_branch_stopped"
        ),
    }
