"""Model-free accounting for reversible scalar and Hangul-hybrid units.

The helpers in this module deliberately stop before model construction.  They
measure representation length, empirical Hangul dependence, reversible
ByteLevel-BPE compression, and a transparent dense-matmul opportunity model.
None of those quantities is an inference-speed or matched-quality result.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from jamoflow.cost import compact_blt_flops, linear_flops
from jamoflow.neural_model import Phase1ModelSpec


UTF8_PRIMARY_ROWS = 256
UTF8_CONTINUATION_ROWS = 64
UTF8_MAXIMUM_CONTINUATIONS = 3
HANGUL_ONSET_ROWS = 19
HANGUL_VOWEL_ROWS = 21
HANGUL_CODA_ROWS = 28
HANGUL_PRIMARY_ROWS = UTF8_PRIMARY_ROWS + HANGUL_ONSET_ROWS
HANGUL_CONDITIONAL_ROWS = HANGUL_VOWEL_ROWS + HANGUL_CODA_ROWS
GENERIC_UTF8_RESIDENT_ROWS = (
    UTF8_PRIMARY_ROWS
    + UTF8_CONTINUATION_ROWS * UTF8_MAXIMUM_CONTINUATIONS
)
HANGUL_HYBRID_RESIDENT_ROWS = HANGUL_PRIMARY_ROWS + HANGUL_CONDITIONAL_ROWS


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def complete_utf8_prefix(data: bytes) -> tuple[str, bytes]:
    """Return the strict complete-scalar prefix and a truncated suffix.

    Invalid UTF-8 in the interior is rejected.  Only an otherwise-valid final
    prefix of a multibyte scalar is accepted as a raw fallback suffix.
    """

    try:
        return data.decode("utf-8", errors="strict"), b""
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(data):
            raise ValueError("stream contains invalid interior UTF-8") from error
        prefix = data[: error.start]
        suffix = data[error.start :]
        if not 1 <= len(suffix) <= 3:
            raise ValueError("truncated UTF-8 suffix has an invalid length")
        return prefix.decode("utf-8", errors="strict"), suffix


def is_precomposed_hangul(codepoint: int) -> bool:
    return 0xAC00 <= codepoint <= 0xD7A3


def hangul_components(codepoint: int) -> tuple[int, int, int]:
    if not is_precomposed_hangul(codepoint):
        raise ValueError("codepoint is not a precomposed Hangul syllable")
    offset = codepoint - 0xAC00
    return offset // 588, (offset % 588) // 28, offset % 28


def _entropy(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count
    )


def hangul_dependence(text: str | Iterable[str]) -> dict[str, Any]:
    onset: Counter[int] = Counter()
    vowel: Counter[int] = Counter()
    coda: Counter[int] = Counter()
    onset_vowel: Counter[tuple[int, int]] = Counter()
    joint: Counter[tuple[int, int, int]] = Counter()
    rows: Iterable[str] = (text,) if isinstance(text, str) else text
    for row in rows:
        if not isinstance(row, str) or not row:
            raise ValueError("Hangul dependence rows must be non-empty strings")
        for character in row:
            codepoint = ord(character)
            if not is_precomposed_hangul(codepoint):
                continue
            l_index, v_index, t_index = hangul_components(codepoint)
            onset[l_index] += 1
            vowel[v_index] += 1
            coda[t_index] += 1
            onset_vowel[(l_index, v_index)] += 1
            joint[(l_index, v_index, t_index)] += 1
    observations = sum(joint.values())
    if observations <= 0:
        raise ValueError("Hangul dependence requires precomposed syllables")
    h_l = _entropy(onset)
    h_v = _entropy(vowel)
    h_t = _entropy(coda)
    h_lv = _entropy(onset_vowel)
    h_lvt = _entropy(joint)
    h_v_given_l = h_lv - h_l
    h_t_given_lv = h_lvt - h_lv
    total_correlation = h_l + h_v + h_t - h_lvt
    return {
        "observations": observations,
        "distinct_syllables": len(joint),
        "entropy_bits": {
            "onset": h_l,
            "vowel": h_v,
            "coda": h_t,
            "joint": h_lvt,
            "vowel_given_onset": h_v_given_l,
            "coda_given_onset_vowel": h_t_given_lv,
            "conditional_chain_total": h_l + h_v_given_l + h_t_given_lv,
            "independent_excess_total_correlation": total_correlation,
        },
        "interpretation": {
            "conditional_chain_can_represent_joint_exactly": True,
            "independent_heads_empirically_misspecified": total_correlation > 0,
            "distribution_only_not_contextual_model_loss": True,
        },
    }


def representation_counts(data: bytes) -> dict[str, Any]:
    text, trailing = complete_utf8_prefix(data)
    complete = text.encode("utf-8")
    utf8_lengths: Counter[int] = Counter()
    hangul = 0
    hangul_bytes = 0
    non_hangul_bytes = 0
    for character in text:
        encoded = character.encode("utf-8")
        utf8_lengths[len(encoded)] += 1
        if is_precomposed_hangul(ord(character)):
            hangul += 1
            hangul_bytes += len(encoded)
        else:
            non_hangul_bytes += len(encoded)
    scalar_steps = len(text) + len(trailing)
    hangul_hybrid_steps = hangul + non_hangul_bytes + len(trailing)
    byte_steps = len(data)
    if len(complete) + len(trailing) != byte_steps:
        raise AssertionError("UTF-8 representation accounting did not close")
    return {
        "input_bytes": byte_steps,
        "complete_scalar_bytes": len(complete),
        "trailing_raw_fallback_bytes": len(trailing),
        "complete_scalars": len(text),
        "precomposed_hangul_scalars": hangul,
        "precomposed_hangul_bytes": hangul_bytes,
        "non_hangul_complete_bytes": non_hangul_bytes,
        "utf8_length_counts": {
            str(length): int(utf8_lengths[length]) for length in range(1, 5)
        },
        "sequential_steps": {
            "raw_byte": byte_steps,
            "generic_unicode_scalar_with_raw_suffix_fallback": scalar_steps,
            "hangul_scalar_otherwise_raw_byte": hangul_hybrid_steps,
        },
        "reductions_relative_to_raw_byte": {
            "generic_unicode_scalar": 1.0 - scalar_steps / byte_steps,
            "hangul_hybrid": 1.0 - hangul_hybrid_steps / byte_steps,
        },
    }


def _scalar_counter(texts: str | Iterable[str]) -> Counter[int]:
    rows: Iterable[str] = (texts,) if isinstance(texts, str) else texts
    output: Counter[int] = Counter()
    seen = False
    for text in rows:
        if not isinstance(text, str) or not text:
            raise ValueError("scalar inventory rows must be non-empty strings")
        output.update(map(ord, text))
        seen = True
    if not seen:
        raise ValueError("scalar inventory needs at least one row")
    return output


def scalar_inventory(
    train_text: str | Iterable[str],
    calibration_text: str | Iterable[str],
) -> dict[str, Any]:
    train = _scalar_counter(train_text)
    calibration = _scalar_counter(calibration_text)
    train_set = set(train)
    unseen = set(calibration) - train_set
    calibration_count = sum(calibration.values())
    unseen_occurrences = sum(calibration[value] for value in unseen)
    train_hangul = {value for value in train if is_precomposed_hangul(value)}
    calibration_hangul = {
        value for value in calibration if is_precomposed_hangul(value)
    }
    unseen_hangul = calibration_hangul - train_hangul
    unseen_hangul_occurrences = sum(calibration[value] for value in unseen_hangul)
    calibration_hangul_occurrences = sum(
        count for value, count in calibration.items() if is_precomposed_hangul(value)
    )
    if calibration_count <= 0 or calibration_hangul_occurrences <= 0:
        raise ValueError("scalar inventory requires calibration observations")
    return {
        "train": {
            "scalar_occurrences": sum(train.values()),
            "unique_scalars": len(train),
            "unique_precomposed_hangul": len(train_hangul),
            "unique_non_hangul": len(train) - len(train_hangul),
        },
        "calibration": {
            "scalar_occurrences": calibration_count,
            "unique_scalars": len(calibration),
            "unique_precomposed_hangul": len(calibration_hangul),
            "unseen_scalar_types": len(unseen),
            "unseen_scalar_occurrences": unseen_occurrences,
            "unseen_scalar_occurrence_rate": unseen_occurrences / calibration_count,
            "unseen_hangul_types": len(unseen_hangul),
            "unseen_hangul_occurrences": unseen_hangul_occurrences,
            "unseen_hangul_occurrence_rate": (
                unseen_hangul_occurrences / calibration_hangul_occurrences
            ),
        },
    }


def train_exact_byte_bpe(
    texts: Iterable[str],
    *,
    vocabulary_size: int,
    minimum_frequency: int = 2,
):
    """Train a reversible ByteLevel BPE without normalizing source text."""

    if vocabulary_size < 256:
        raise ValueError("BPE vocabulary cannot be smaller than 256")
    if minimum_frequency <= 0:
        raise ValueError("BPE minimum frequency must be positive")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    def checked() -> Iterable[str]:
        count = 0
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("BPE rows must be non-empty strings")
            count += 1
            yield text
        if count == 0:
            raise ValueError("BPE training needs at least one row")

    tokenizer = Tokenizer(
        models.BPE(
            unk_token=None,
            dropout=None,
            fuse_unk=False,
            byte_fallback=False,
        )
    )
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=minimum_frequency,
        show_progress=False,
        special_tokens=[],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer.train_from_iterator(checked(), trainer=trainer)
    return tokenizer


def audit_bpe_encoding(tokenizer, text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("BPE audit text must be non-empty")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_ids = tuple(int(value) for value in encoding.ids)
    if not token_ids:
        raise ValueError("BPE audit produced no tokens")
    decoded = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    serialized = tokenizer.to_str(pretty=False).encode("utf-8")
    return {
        "token_count": len(token_ids),
        "input_bytes": len(text.encode("utf-8")),
        "bytes_per_token": len(text.encode("utf-8")) / len(token_ids),
        "tokens_per_unicode_scalar": len(token_ids) / len(text),
        "roundtrip_identity": decoded == text,
        "vocabulary_size": int(tokenizer.get_vocab_size(with_added_tokens=True)),
        "tokenizer_json_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _window_index(offset: int, sequence_bytes: int, windows: int) -> int:
    index = offset // sequence_bytes
    return min(index, windows - 1)


def scalar_blt_opportunity_flops(
    data: bytes,
    *,
    baseline_spec: Phase1ModelSpec,
    data_patches: int,
) -> dict[str, Any]:
    """Estimate dense matmuls at the same raw-byte horizon and patch count.

    Scalar/Hangul units are assigned to the 512-byte window in which their
    first byte starts.  This is an analytical opportunity model only: it omits
    scalar parsing, conditional-head dispatch, kernel launches, cache layout,
    and any quality-preserving capacity changes.
    """

    sequence_bytes = baseline_spec.sequence_length
    if len(data) % sequence_bytes:
        raise ValueError("opportunity FLOPs require complete raw-byte windows")
    windows = len(data) // sequence_bytes
    if windows <= 0:
        raise ValueError("opportunity FLOPs require at least one window")
    text, trailing = complete_utf8_prefix(data)
    scalar_steps = [0] * windows
    hybrid_steps = [0] * windows
    generic_active_rows = [0] * windows
    hybrid_active_rows = [0] * windows
    offset = 0
    for character in text:
        encoded = character.encode("utf-8")
        index = _window_index(offset, sequence_bytes, windows)
        scalar_steps[index] += 1
        generic_active_rows[index] += (
            UTF8_PRIMARY_ROWS
            + (len(encoded) - 1) * UTF8_CONTINUATION_ROWS
        )
        if is_precomposed_hangul(ord(character)):
            hybrid_steps[index] += 1
            hybrid_active_rows[index] += (
                HANGUL_PRIMARY_ROWS + HANGUL_CONDITIONAL_ROWS
            )
        else:
            for local_offset in range(len(encoded)):
                byte_index = _window_index(
                    offset + local_offset,
                    sequence_bytes,
                    windows,
                )
                hybrid_steps[byte_index] += 1
                hybrid_active_rows[byte_index] += HANGUL_PRIMARY_ROWS
        offset += len(encoded)
    for local_offset in range(len(trailing)):
        index = _window_index(offset + local_offset, sequence_bytes, windows)
        scalar_steps[index] += 1
        hybrid_steps[index] += 1
        generic_active_rows[index] += UTF8_PRIMARY_ROWS
        hybrid_active_rows[index] += HANGUL_PRIMARY_ROWS
    if offset + len(trailing) != len(data):
        raise AssertionError("opportunity window accounting did not close")

    baseline = int(
        compact_blt_flops(
            baseline_spec,
            data_patches=data_patches,
        )["forward_flops_per_sequence"]
    )

    def candidate_cost(
        steps: Sequence[int], active_rows: Sequence[int]
    ) -> dict[str, Any]:
        costs: list[int] = []
        for positions, rows in zip(steps, active_rows, strict=True):
            if positions <= 0 or rows <= 0:
                raise ValueError("scalar opportunity window is empty")
            one_row_spec = replace(
                baseline_spec,
                sequence_length=positions,
                vocab_size=1,
            )
            one_row = compact_blt_flops(
                one_row_spec,
                data_patches=data_patches,
            )
            one_row_total = int(one_row["forward_flops_per_sequence"])
            one_row_head = linear_flops(
                positions,
                baseline_spec.local_width,
                1,
            )
            head = linear_flops(
                1,
                baseline_spec.local_width,
                rows,
            )
            costs.append(one_row_total - one_row_head + head)
        mean = sum(costs) / len(costs)
        return {
            "mean_dense_matmul_flops_per_512_raw_bytes": mean,
            "reduction_relative_to_w72": 1.0 - mean / baseline,
            "step_count_per_512_raw_bytes": {
                "minimum": min(steps),
                "mean": sum(steps) / len(steps),
                "maximum": max(steps),
            },
            "active_output_rows_per_512_raw_bytes": {
                "minimum": min(active_rows),
                "mean": sum(active_rows) / len(active_rows),
                "maximum": max(active_rows),
            },
        }

    generic = candidate_cost(scalar_steps, generic_active_rows)
    hybrid = candidate_cost(hybrid_steps, hybrid_active_rows)
    return {
        "method": "dense matmul opportunity estimate; multiply-add=2",
        "raw_windows": windows,
        "raw_bytes_per_window": sequence_bytes,
        "data_patches_per_window": data_patches,
        "w72_baseline_dense_matmul_flops_per_512_bytes": baseline,
        "generic_unicode_scalar": generic,
        "hangul_scalar_otherwise_raw_byte": hybrid,
        "resident_output_rows": {
            "raw_byte": UTF8_PRIMARY_ROWS,
            "generic_unicode_scalar_conditional_utf8": (
                GENERIC_UTF8_RESIDENT_ROWS
            ),
            "hangul_scalar_otherwise_raw_byte_conditional_lvt": (
                HANGUL_HYBRID_RESIDENT_ROWS
            ),
        },
        "resident_output_projection_parameters_at_local_width": {
            "raw_byte": UTF8_PRIMARY_ROWS * baseline_spec.local_width,
            "generic_unicode_scalar_conditional_utf8": (
                GENERIC_UTF8_RESIDENT_ROWS * baseline_spec.local_width
            ),
            "hangul_scalar_otherwise_raw_byte_conditional_lvt": (
                HANGUL_HYBRID_RESIDENT_ROWS * baseline_spec.local_width
            ),
        },
        "omitted": [
            "UTF-8 or Hangul transducer and raw-fallback dispatch",
            "conditional micro-head dependencies and kernel launches",
            "embedding lookup, hashing, normalization, RoPE, softmax, and masking",
            "cache layout, memory movement, synchronization, and allocator effects",
            "parameter matching or capacity changes required for equal quality",
        ],
    }
