"""Korean-specific analysis helpers for the preregistered Phase 2 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import unicodedata

import numpy as np


@dataclass(frozen=True, slots=True)
class KoreanStratum:
    name: str
    definition: str
    selected: np.ndarray

    def metadata(self) -> dict[str, int | float | str]:
        count = int(self.selected.sum())
        return {
            "name": self.name,
            "definition": self.definition,
            "sequences": count,
            "sequence_fraction": count / len(self.selected),
        }


def _is_hanja(codepoint: int) -> bool:
    ranges = (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFAFF),
        (0x20000, 0x2EBEF),
        (0x2F800, 0x2FA1F),
        (0x30000, 0x323AF),
    )
    return any(start <= codepoint <= end for start, end in ranges)


def _is_modern_jamo(codepoint: int) -> bool:
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def korean_test_strata(
    stream_data: bytes,
    boundary_masks: np.ndarray,
    sequence_length: int = 256,
) -> tuple[dict[str, KoreanStratum], dict[str, object]]:
    """Classify fixed byte windows using deterministic Unicode rules.

    Incomplete UTF-8 fragments at arbitrary chunk edges are ignored for text
    classification, while the separately supplied global boundary mask retains
    their chunk-start status.
    """

    if len(stream_data) % sequence_length:
        raise ValueError("stream must contain complete fixed-length sequences")
    sequence_count = len(stream_data) // sequence_length
    if boundary_masks.shape != (sequence_count, sequence_length):
        raise ValueError("boundary masks have an unexpected shape")

    hangul_heavy = np.zeros(sequence_count, dtype=bool)
    latin_mixed = np.zeros(sequence_count, dtype=bool)
    digit_mixed = np.zeros(sequence_count, dtype=bool)
    hanja_mixed = np.zeros(sequence_count, dtype=bool)
    compatibility_jamo = np.zeros(sequence_count, dtype=bool)
    modern_jamo = np.zeros(sequence_count, dtype=bool)
    whitespace_density = np.zeros(sequence_count, dtype=np.float64)

    for index in range(sequence_count):
        start = index * sequence_length
        text = stream_data[start : start + sequence_length].decode(
            "utf-8",
            errors="ignore",
        )
        codepoints = [ord(character) for character in text]
        letters = [
            codepoint
            for codepoint in codepoints
            if unicodedata.category(chr(codepoint)).startswith("L")
        ]
        if letters:
            precomposed = sum(0xAC00 <= codepoint <= 0xD7A3 for codepoint in letters)
            hangul_heavy[index] = precomposed / len(letters) >= 0.8
        latin_mixed[index] = any(
            ord("A") <= codepoint <= ord("Z")
            or ord("a") <= codepoint <= ord("z")
            for codepoint in codepoints
        )
        digit_mixed[index] = any(ord("0") <= value <= ord("9") for value in codepoints)
        hanja_mixed[index] = any(_is_hanja(value) for value in codepoints)
        compatibility_jamo[index] = any(0x3130 <= value <= 0x318F for value in codepoints)
        modern_jamo[index] = any(_is_modern_jamo(value) for value in codepoints)
        whitespace_density[index] = (
            sum(chr(value).isspace() for value in codepoints) / len(codepoints)
            if codepoints
            else 0.0
        )

    # Stable rank quartiles prevent ties from creating overlapping or empty
    # bins. Density and original sequence index form the deterministic order.
    order = np.lexsort((np.arange(sequence_count), whitespace_density))
    quartile_labels = np.empty(sequence_count, dtype=np.int8)
    for quartile, indices in enumerate(np.array_split(order, 4), start=1):
        quartile_labels[indices] = quartile

    raw = {
        "hangul_heavy": (
            "letter codepoints are at least 80% precomposed Hangul",
            hangul_heavy,
        ),
        "latin_mixed": ("contains an ASCII Latin letter", latin_mixed),
        "digit_mixed": ("contains an ASCII digit", digit_mixed),
        "hanja_mixed": ("contains a CJK ideograph", hanja_mixed),
        "compatibility_jamo_present": (
            "contains U+3130–U+318F compatibility jamo",
            compatibility_jamo,
        ),
        "modern_jamo_present": (
            "contains a modern or extended Hangul Jamo codepoint",
            modern_jamo,
        ),
        "starts_at_codepoint_boundary": (
            "global UTF-8 parser is complete at byte-window start",
            boundary_masks[:, 0].astype(bool),
        ),
        "starts_inside_codepoint": (
            "byte window starts inside a global UTF-8 codepoint",
            ~boundary_masks[:, 0].astype(bool),
        ),
    }
    for quartile in range(1, 5):
        raw[f"whitespace_density_q{quartile}"] = (
            f"stable-rank whitespace-density quartile {quartile}",
            quartile_labels == quartile,
        )

    strata = {
        name: KoreanStratum(name, definition, selected)
        for name, (definition, selected) in raw.items()
    }
    metadata = {
        "sequence_count": sequence_count,
        "classification_decode_errors": "ignored only at incomplete chunk edges",
        "whitespace_density": {
            "minimum": float(whitespace_density.min()),
            "median": float(np.median(whitespace_density)),
            "maximum": float(whitespace_density.max()),
            "quartile_assignment": "stable rank by (density, sequence index)",
        },
        "strata": {
            name: stratum.metadata()
            for name, stratum in strata.items()
        },
    }
    return strata, metadata


def gate_effect_checks(
    paired_effects: list[float],
    *,
    maximum_mean: float,
    required_negative_seeds: int = 4,
    require_negative_upper: bool = True,
    interval_upper: float,
) -> dict[str, bool | float | int]:
    effects = np.asarray(paired_effects, dtype=np.float64)
    if effects.ndim != 1 or not len(effects):
        raise ValueError("gate effects must be a non-empty vector")
    checks: dict[str, bool | float | int] = {
        "mean_effect_bpb": float(effects.mean()),
        "maximum_allowed_mean_bpb": maximum_mean,
        "mean_at_or_below_threshold": float(effects.mean()) <= maximum_mean,
        "negative_seed_count": int((effects < 0).sum()),
        "required_negative_seed_count": required_negative_seeds,
        "enough_negative_seeds": int((effects < 0).sum()) >= required_negative_seeds,
        "paired_t_upper_bpb": interval_upper,
        "paired_t_upper_is_negative": interval_upper < 0,
    }
    checks["primary_effect_pass"] = bool(
        checks["mean_at_or_below_threshold"]
        and checks["enough_negative_seeds"]
        and (
            checks["paired_t_upper_is_negative"]
            if require_negative_upper
            else True
        )
    )
    return checks
