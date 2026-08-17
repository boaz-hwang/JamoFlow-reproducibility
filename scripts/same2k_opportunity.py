"""Token-only opportunity metrics for equal-2K byte tokenizers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import statistics
import time
from typing import Any

import numpy as np

from jamoflow.publication_bpe import byte_bpe_token_bytes


BPE_ROLE = "byte_bpe_2k"
SCORED_UNIGRAM_ROLE = "byte_unigram_2k_scored"
LONGEST_MATCH_ROLE = "byte_unigram_vocab_2k_leftmost_longest"
MINIMUM_TOKEN_ROLE = "byte_unigram_vocab_2k_minimum_token_dp"
ROLES = (BPE_ROLE, SCORED_UNIGRAM_ROLE, LONGEST_MATCH_ROLE, MINIMUM_TOKEN_ROLE)
VOCABULARY_SIZE = 2_048
WARMUP_CASES = 6
MEASURED_CASES = 36
ENCODE_REPETITIONS = 3
MINIMUM_STEP_REDUCTION = 0.10


@dataclass(frozen=True, slots=True)
class TokenizerOpportunityMetrics:
    role: str
    vocabulary_size: int
    calibration_complete_utf8_bytes: int
    calibration_trailing_bytes: int
    calibration_token_count: int
    calibration_token_ids_sha256: str
    vocabulary_used: int
    vocabulary_utilization: float
    maximum_used_piece_bytes: int
    encode_seconds: tuple[float, ...]
    encode_median_megabytes_per_second: float
    prompt_token_counts: tuple[int, ...]
    continuation_token_counts: tuple[int, ...]
    joint_token_counts: tuple[int, ...]
    prompt_counts_sha256: str
    continuation_counts_sha256: str
    joint_counts_sha256: str
    multibyte_vocabulary_count: int
    strict_utf8_multibyte_vocabulary_count: int
    strict_utf8_multibyte_vocabulary_fraction: float
    hangul_multibyte_vocabulary_count: int
    cross_eojeol_multibyte_vocabulary_count: int
    boundary_complete_cross_eojeol_vocabulary_count: int
    exact_roundtrip: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _array_sha256(values: Sequence[int]) -> str:
    array = np.asarray(tuple(values), dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _complete_utf8_prefix(raw: bytes) -> tuple[bytes, bytes]:
    try:
        raw.decode("utf-8", errors="strict")
        return raw, b""
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(raw):
            raise ValueError("same-2K stream contains invalid interior UTF-8") from error
        suffix = raw[error.start :]
        if not 1 <= len(suffix) <= 3:
            raise ValueError("same-2K stream has an invalid UTF-8 suffix")
        prefix = raw[: error.start]
        prefix.decode("utf-8", errors="strict")
        return prefix, suffix


def canonical_text_pieces(raw: bytes) -> tuple[tuple[str, bytes], ...]:
    complete, _ = _complete_utf8_prefix(raw)
    output: list[tuple[str, bytes]] = []
    for index, row in enumerate(complete.split(b"\n")):
        piece = row if index == 0 else b"\n" + row
        if not piece:
            continue
        output.append((piece.decode("utf-8", errors="strict"), piece))
    if not output or b"".join(piece for _, piece in output) != complete:
        raise ValueError("same-2K canonical text pieces differ")
    return tuple(output)


def _encode_ids(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in tokenizer.encode(text, add_special_tokens=False).ids
    )


def _encode_case(raw: bytes, tokenizer, token_bytes: Sequence[bytes]) -> tuple[int, ...]:
    text = raw.decode("utf-8", errors="strict")
    ids = _encode_ids(tokenizer, text)
    if (
        not ids
        or b"".join(token_bytes[token_id] for token_id in ids) != raw
        or tokenizer.decode(list(ids), skip_special_tokens=False) != text
    ):
        raise ValueError("same-2K case roundtrip differs")
    return ids


def _vocabulary_structure(token_bytes: Sequence[bytes]) -> dict[str, int | float]:
    multibyte = [piece for piece in token_bytes if len(piece) > 1]
    strict = 0
    hangul = 0
    cross = 0
    complete_cross = 0
    for piece in multibyte:
        try:
            text = piece.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        strict += 1
        if any("가" <= character <= "힣" for character in text):
            hangul += 1
        whitespace_positions = [
            index for index, character in enumerate(text) if character.isspace()
        ]
        if whitespace_positions and any(not character.isspace() for character in text):
            first_nonspace = next(
                index for index, character in enumerate(text) if not character.isspace()
            )
            last_nonspace = max(
                index for index, character in enumerate(text) if not character.isspace()
            )
            internal = any(first_nonspace < index < last_nonspace for index in whitespace_positions)
            if internal:
                cross += 1
                if text[0].isspace() and text[-1].isspace():
                    complete_cross += 1
    count = len(multibyte)
    return {
        "multibyte_vocabulary_count": count,
        "strict_utf8_multibyte_vocabulary_count": strict,
        "strict_utf8_multibyte_vocabulary_fraction": strict / count if count else 0.0,
        "hangul_multibyte_vocabulary_count": hangul,
        "cross_eojeol_multibyte_vocabulary_count": cross,
        "boundary_complete_cross_eojeol_vocabulary_count": complete_cross,
    }


def evaluate_tokenizer_opportunity(
    *,
    role: str,
    tokenizer,
    calibration_raw: bytes,
    prompts: np.ndarray,
    continuations: np.ndarray,
    encode_repetitions: int = ENCODE_REPETITIONS,
) -> TokenizerOpportunityMetrics:
    if (
        role not in ROLES
        or prompts.shape != (WARMUP_CASES + MEASURED_CASES, 128)
        or continuations.shape != prompts.shape
        or prompts.dtype != np.uint8
        or continuations.dtype != np.uint8
        or encode_repetitions <= 0
    ):
        raise ValueError("same-2K opportunity coordinates differ")
    token_bytes = byte_bpe_token_bytes(tokenizer)
    if len(token_bytes) != VOCABULARY_SIZE:
        raise ValueError("same-2K opportunity vocabulary size differs")
    complete, trailing = _complete_utf8_prefix(calibration_raw)
    text_pieces = canonical_text_pieces(calibration_raw)

    counts: Counter[int] = Counter()
    digest = hashlib.sha256(b"JamoFlow/same2k-token-stream/v1\0")
    rendered = hashlib.sha256()
    token_count = 0
    for text, raw_piece in text_pieces:
        ids = _encode_ids(tokenizer, text)
        if not ids:
            raise ValueError("same-2K tokenizer emitted an empty piece")
        recovered = b"".join(token_bytes[token_id] for token_id in ids)
        if (
            recovered != raw_piece
            or tokenizer.decode(list(ids), skip_special_tokens=False) != text
        ):
            raise ValueError("same-2K calibration roundtrip differs")
        array = np.asarray(ids, dtype="<u4")
        digest.update(len(ids).to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
        rendered.update(recovered)
        counts.update(ids)
        token_count += len(ids)
    digest.update(token_count.to_bytes(8, "big"))
    if rendered.digest() != hashlib.sha256(complete).digest():
        raise AssertionError("same-2K rendered calibration hash differs")

    timings: list[float] = []
    for _ in range(encode_repetitions):
        start = time.perf_counter()
        repeated_count = 0
        for text, _ in text_pieces:
            repeated_count += len(_encode_ids(tokenizer, text))
        timings.append(time.perf_counter() - start)
        if repeated_count != token_count:
            raise ValueError("same-2K repeated encoding is nondeterministic")

    prompt_counts: list[int] = []
    continuation_counts: list[int] = []
    joint_counts: list[int] = []
    for prompt, continuation in zip(prompts, continuations, strict=True):
        prompt_raw = bytes(prompt)
        continuation_raw = bytes(continuation)
        prompt_counts.append(len(_encode_case(prompt_raw, tokenizer, token_bytes)))
        continuation_counts.append(
            len(_encode_case(continuation_raw, tokenizer, token_bytes))
        )
        joint_counts.append(
            len(_encode_case(prompt_raw + continuation_raw, tokenizer, token_bytes))
        )

    structure = _vocabulary_structure(token_bytes)
    median_seconds = statistics.median(timings)
    used = len(counts)
    return TokenizerOpportunityMetrics(
        role=role,
        vocabulary_size=len(token_bytes),
        calibration_complete_utf8_bytes=len(complete),
        calibration_trailing_bytes=len(trailing),
        calibration_token_count=token_count,
        calibration_token_ids_sha256=digest.hexdigest(),
        vocabulary_used=used,
        vocabulary_utilization=used / len(token_bytes),
        maximum_used_piece_bytes=max(len(token_bytes[token_id]) for token_id in counts),
        encode_seconds=tuple(timings),
        encode_median_megabytes_per_second=(len(complete) / 1_000_000 / median_seconds),
        prompt_token_counts=tuple(prompt_counts),
        continuation_token_counts=tuple(continuation_counts),
        joint_token_counts=tuple(joint_counts),
        prompt_counts_sha256=_array_sha256(prompt_counts),
        continuation_counts_sha256=_array_sha256(continuation_counts),
        joint_counts_sha256=_array_sha256(joint_counts),
        exact_roundtrip=True,
        **structure,
    )


def opportunity_decision(
    metrics_by_role: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(metrics_by_role) != set(ROLES):
        raise ValueError("same-2K opportunity role set differs")
    baseline = metrics_by_role[BPE_ROLE]
    baseline_calibration = int(baseline["calibration_token_count"])
    baseline_continuations = np.asarray(
        baseline["continuation_token_counts"], dtype=np.int64
    )[WARMUP_CASES:]
    if baseline_calibration <= 0 or np.any(baseline_continuations <= 0):
        raise ValueError("same-2K opportunity baseline counts differ")
    comparisons = {}
    eligible = []
    for role in ROLES[1:]:
        candidate = metrics_by_role[role]
        calibration = int(candidate["calibration_token_count"])
        continuations = np.asarray(
            candidate["continuation_token_counts"], dtype=np.int64
        )[WARMUP_CASES:]
        if continuations.shape != baseline_continuations.shape or np.any(continuations <= 0):
            raise ValueError("same-2K opportunity continuation counts differ")
        calibration_reduction = 1.0 - calibration / baseline_calibration
        continuation_reduction = 1.0 - float(continuations.sum()) / float(
            baseline_continuations.sum()
        )
        per_case = 1.0 - continuations / baseline_continuations
        passes = bool(
            calibration_reduction >= MINIMUM_STEP_REDUCTION
            and continuation_reduction >= MINIMUM_STEP_REDUCTION
        )
        if passes:
            eligible.append(role)
        comparisons[role] = {
            "calibration_token_reduction": calibration_reduction,
            "measured_continuation_token_reduction": continuation_reduction,
            "median_paired_continuation_token_reduction": float(np.median(per_case)),
            "cases_with_fewer_continuation_tokens": int(np.count_nonzero(per_case > 0)),
            "measured_case_count": len(per_case),
            "minimum_required_reduction": MINIMUM_STEP_REDUCTION,
            "passes_token_only_opportunity": passes,
        }
    next_action = (
        "one_seed_quality_training_for_eligible_generic_roles"
        if eligible
        else "construct_length_gain_vocabulary_before_any_model_training"
    )
    return {
        "baseline": BPE_ROLE,
        "comparisons": comparisons,
        "eligible_generic_roles": eligible,
        "next_action": next_action,
        "quality_or_model_latency_used": False,
    }


def metrics_identity_without_timing(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project exact replay fields while excluding wall-clock diagnostics."""

    output = dict(value)
    try:
        output.pop("encode_seconds")
        output.pop("encode_median_megabytes_per_second")
    except KeyError as error:
        raise ValueError("same-2K metrics timing fields differ") from error
    # JSON evidence turns dataclass tuples into arrays/lists.  Compare the
    # canonical transport representation so a scientifically identical replay
    # is not rejected solely because one side crossed the JSON boundary.
    normalized = json.loads(
        json.dumps(
            output,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    if not isinstance(normalized, dict):
        raise AssertionError("same-2K metric identity must remain a mapping")
    return normalized
