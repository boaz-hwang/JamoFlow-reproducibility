#!/usr/bin/env python3
"""Calibration-only UTF-8/Hangul opportunity analysis for block decoding."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from jamoflow.neural_data import build_neural_stream


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/hangul-block-opportunity-v1.json"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_head() -> str:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("opportunity analysis requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("invalid git commit identity")
    return commit


def _continuation(value: int) -> bool:
    return 0x80 <= value <= 0xBF


def _utf8_length_and_codepoint(data: bytes, index: int) -> tuple[int, int] | None:
    """Decode one strict scalar or return None for a valid truncated suffix."""

    first = data[index]
    available = len(data) - index
    if first <= 0x7F:
        return 1, first
    if 0xC2 <= first <= 0xDF:
        length = 2
    elif 0xE0 <= first <= 0xEF:
        length = 3
    elif 0xF0 <= first <= 0xF4:
        length = 4
    else:
        raise ValueError(f"invalid UTF-8 lead byte at {index}")
    suffix = data[index + 1 : min(index + length, len(data))]
    if any(not _continuation(value) for value in suffix):
        raise ValueError(f"invalid UTF-8 continuation byte at {index}")
    if available < length:
        if len(suffix) >= 1:
            second = suffix[0]
            if first == 0xE0 and second < 0xA0:
                raise ValueError("overlong truncated UTF-8 prefix")
            if first == 0xED and second > 0x9F:
                raise ValueError("surrogate truncated UTF-8 prefix")
            if first == 0xF0 and second < 0x90:
                raise ValueError("overlong truncated UTF-8 prefix")
            if first == 0xF4 and second > 0x8F:
                raise ValueError("out-of-range truncated UTF-8 prefix")
        return None
    second = data[index + 1]
    if first == 0xE0 and second < 0xA0:
        raise ValueError("overlong UTF-8 scalar")
    if first == 0xED and second > 0x9F:
        raise ValueError("UTF-8 surrogate scalar")
    if first == 0xF0 and second < 0x90:
        raise ValueError("overlong UTF-8 scalar")
    if first == 0xF4 and second > 0x8F:
        raise ValueError("UTF-8 scalar exceeds U+10FFFF")
    value = first & (0x7F >> length)
    for offset in range(1, length):
        value = (value << 6) | (data[index + offset] & 0x3F)
    return length, value


def iter_utf8_scalars(data: bytes) -> Iterable[tuple[int, int, bytes]]:
    index = 0
    while index < len(data):
        decoded = _utf8_length_and_codepoint(data, index)
        if decoded is None:
            break
        length, codepoint = decoded
        raw = data[index : index + length]
        yield codepoint, length, raw
        index += length


def _complete_scalar_bytes(data: bytes) -> tuple[int, int]:
    consumed = 0
    count = 0
    for _, length, _ in iter_utf8_scalars(data):
        consumed += length
        count += 1
    return consumed, count


def _entropy(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count
    )


def _conditional_entropy(joint: Counter[Any], condition: Counter[Any]) -> float:
    return _entropy(joint) - _entropy(condition)


def _mode_key(counter: Counter[int]) -> int:
    if not counter:
        raise ValueError("cannot select a mode from an empty counter")
    maximum = max(counter.values())
    return min(key for key, count in counter.items() if count == maximum)


def _hangul_indices(codepoint: int) -> tuple[int, int, int]:
    if not 0xAC00 <= codepoint <= 0xD7A3:
        raise ValueError("codepoint is not a precomposed Hangul syllable")
    offset = codepoint - 0xAC00
    return offset // 588, (offset % 588) // 28, offset % 28


def _is_jamo_block(codepoint: int) -> bool:
    return (
        0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xA960 <= codepoint <= 0xA97F
        or 0xD7B0 <= codepoint <= 0xD7FF
    )


def _theoretical_hangul_pairs() -> dict[str, int]:
    pairs: dict[int, set[tuple[int, int]]] = {}
    for codepoint in range(0xAC00, 0xD7A4):
        raw = chr(codepoint).encode("utf-8")
        pairs.setdefault(raw[0], set()).add((raw[1], raw[2]))
    return {f"0x{lead:02x}": len(values) for lead, values in sorted(pairs.items())}


def analyze_stream(data: bytes, fixed_block_sizes: Iterable[int]) -> dict[str, Any]:
    length_counts: Counter[int] = Counter()
    category_counts: Counter[str] = Counter()
    onset: Counter[int] = Counter()
    vowel: Counter[int] = Counter()
    coda: Counter[int] = Counter()
    components: Counter[tuple[int, int, int]] = Counter()
    lead: Counter[int] = Counter()
    lead_second: Counter[tuple[int, int]] = Counter()
    lead_third: Counter[tuple[int, int]] = Counter()
    lead_pair: Counter[tuple[int, int, int]] = Counter()
    empirical_pairs: dict[int, set[tuple[int, int]]] = {}

    complete_bytes = 0
    scalar_count = 0
    hangul_count = 0
    whitespace_count = 0
    for codepoint, length, raw in iter_utf8_scalars(data):
        complete_bytes += length
        scalar_count += 1
        length_counts[length] += 1
        if chr(codepoint).isspace():
            whitespace_count += 1
        if codepoint <= 0x7F:
            category_counts["ascii"] += 1
        elif 0xAC00 <= codepoint <= 0xD7A3:
            category_counts["precomposed_hangul_syllable"] += 1
            hangul_count += 1
            l_index, v_index, t_index = _hangul_indices(codepoint)
            onset[l_index] += 1
            vowel[v_index] += 1
            coda[t_index] += 1
            components[(l_index, v_index, t_index)] += 1
            first, second, third = raw
            lead[first] += 1
            lead_second[(first, second)] += 1
            lead_third[(first, third)] += 1
            lead_pair[(first, second, third)] += 1
            empirical_pairs.setdefault(first, set()).add((second, third))
        elif _is_jamo_block(codepoint):
            category_counts["jamo_blocks"] += 1
        else:
            category_counts["other"] += 1

    if scalar_count == 0 or hangul_count == 0:
        raise ValueError("stream lacks required UTF-8/Hangul observations")
    if complete_bytes + (len(data) - complete_bytes) != len(data):
        raise AssertionError("UTF-8 accounting did not close")

    joint_pair_hits = 0
    independent_pair_hits = 0
    for first in lead:
        pair_rows = Counter({
            (second, third): count
            for (current, second, third), count in lead_pair.items()
            if current == first
        })
        best_pair = min(
            key for key, count in pair_rows.items() if count == max(pair_rows.values())
        )
        joint_pair_hits += pair_rows[best_pair]
        second_rows = Counter({
            second: count
            for (current, second), count in lead_second.items()
            if current == first
        })
        third_rows = Counter({
            third: count
            for (current, third), count in lead_third.items()
            if current == first
        })
        independent_pair = (_mode_key(second_rows), _mode_key(third_rows))
        independent_pair_hits += pair_rows[independent_pair]

    l_mode = _mode_key(onset)
    v_mode = _mode_key(vowel)
    t_mode = _mode_key(coda)
    byte_savings_from_all_scalars = complete_bytes - scalar_count
    byte_savings_from_hangul = 2 * hangul_count
    hangul_calls = complete_bytes - byte_savings_from_hangul
    oracle = {
        "byte_autoregressive_calls": complete_bytes,
        "one_call_per_scalar": {
            "calls": scalar_count,
            "reduction": 1.0 - scalar_count / complete_bytes,
        },
        "hangul_only_adaptive": {
            "calls": hangul_calls,
            "reduction": 1.0 - hangul_calls / complete_bytes,
            "saved_calls": byte_savings_from_hangul,
        },
        "fixed_byte_blocks": {
            str(size): {
                "calls": math.ceil(complete_bytes / size),
                "reduction": 1.0 - math.ceil(complete_bytes / size) / complete_bytes,
            }
            for size in fixed_block_sizes
        },
        "hangul_share_of_scalar_savings": (
            byte_savings_from_hangul / byte_savings_from_all_scalars
        ),
    }

    empirical_pair_counts = {
        f"0x{first:02x}": len(values)
        for first, values in sorted(empirical_pairs.items())
    }
    return {
        "stream": {
            "input_bytes": len(data),
            "complete_scalar_bytes": complete_bytes,
            "trailing_incomplete_bytes": len(data) - complete_bytes,
            "complete_scalars": scalar_count,
        },
        "composition": {
            "utf8_length_counts": {
                str(length): int(length_counts[length]) for length in range(1, 5)
            },
            "unicode_category_counts": {
                category: int(category_counts[category])
                for category in (
                    "ascii",
                    "precomposed_hangul_syllable",
                    "jamo_blocks",
                    "other",
                )
            },
            "precomposed_hangul_scalar_rate": hangul_count / scalar_count,
            "precomposed_hangul_byte_rate": 3 * hangul_count / complete_bytes,
            "whitespace_scalar_count": whitespace_count,
        },
        "target_call_oracles": oracle,
        "hangul_distribution": {
            "observed_distinct_syllables": len(components),
            "valid_precomposed_syllables": 19 * 21 * 28,
            "component_logits": {"onset": 19, "vowel": 21, "coda": 28, "total": 68},
            "flat_syllable_logits": 19 * 21 * 28,
            "component_logit_reduction": 1.0 - 68 / (19 * 21 * 28),
            "entropy_bits": {
                "onset": _entropy(onset),
                "vowel": _entropy(vowel),
                "coda": _entropy(coda),
                "joint_syllable": _entropy(components),
                "component_total_correlation": (
                    _entropy(onset) + _entropy(vowel) + _entropy(coda) - _entropy(components)
                ),
            },
            "context_free_factorized_component_mode_exact_syllable_rate": (
                components[(l_mode, v_mode, t_mode)] / hangul_count
            ),
            "utf8_continuation_pair": {
                "conditional_joint_entropy_bits_given_lead": _conditional_entropy(
                    lead_pair, lead
                ),
                "conditional_second_entropy_bits_given_lead": _conditional_entropy(
                    lead_second, lead
                ),
                "conditional_third_entropy_bits_given_lead": _conditional_entropy(
                    lead_third, lead
                ),
                "conditional_mutual_information_bits": (
                    _conditional_entropy(lead_second, lead)
                    + _conditional_entropy(lead_third, lead)
                    - _conditional_entropy(lead_pair, lead)
                ),
                "context_free_joint_pair_mode_exact_rate": joint_pair_hits / hangul_count,
                "context_free_independent_byte_modes_exact_pair_rate": (
                    independent_pair_hits / hangul_count
                ),
                "empirical_distinct_pairs_by_lead": empirical_pair_counts,
                "theoretical_valid_pairs_by_lead": _theoretical_hangul_pairs(),
            },
        },
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "byte_limit",
        "decision_rule",
        "fixed_block_sizes",
        "input",
        "kind",
        "metrics",
        "output_path",
        "profile_context",
        "protocol_id",
        "schema_version",
        "sequence_length",
        "status",
        "threat_model",
    }
    if set(plan) != expected or plan.get("kind") != "hangul_block_opportunity_plan_v1":
        raise ValueError("invalid opportunity plan schema")
    if plan.get("schema_version") != 1 or plan.get("byte_limit") != 8_000_000:
        raise ValueError("unexpected opportunity plan version or byte limit")
    if plan.get("fixed_block_sizes") != [2, 3, 4, 8]:
        raise ValueError("fixed block controls changed")
    threat = plan.get("threat_model", {})
    if any(threat.get(key) is not False for key in (
        "final_test_read",
        "model_checkpoint_read",
        "model_metric_read",
        "raw_text_promoted",
    )):
        raise ValueError("opportunity threat-model boundary changed")


def main() -> None:
    commit = _require_clean_head()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    _validate_plan(plan)
    inputs = plan["input"]
    for path_key, hash_key in (
        ("source_path", "source_sha256"),
        ("integrity_path", "integrity_sha256"),
        ("selection_plan_path", "selection_plan_sha256"),
    ):
        path = ROOT / inputs[path_key]
        if _hash_file(path) != inputs[hash_key]:
            raise ValueError(f"input artifact changed: {inputs[path_key]}")
    profile = plan["profile_context"]
    if _hash_file(ROOT / profile["artifact_path"]) != profile["artifact_sha256"]:
        raise ValueError("component profile context changed")
    stream = build_neural_stream(
        ROOT / inputs["source_path"],
        language="ko",
        split="calibration",
        byte_limit=plan["byte_limit"],
        sequence_length=plan["sequence_length"],
    )
    stream_sha256 = hashlib.sha256(stream.data).hexdigest()
    if stream_sha256 != inputs["calibration_stream_sha256"]:
        raise ValueError("calibration stream differs from the sealed input")
    metrics = analyze_stream(stream.data, plan["fixed_block_sizes"])
    rule = plan["decision_rule"]
    hangul = metrics["target_call_oracles"]["hangul_only_adaptive"]["reduction"]
    share = metrics["target_call_oracles"]["hangul_share_of_scalar_savings"]
    gate = {
        "minimum_hangul_target_call_reduction": rule[
            "minimum_hangul_target_call_reduction"
        ],
        "minimum_hangul_share_of_scalar_savings": rule[
            "minimum_hangul_share_of_scalar_savings"
        ],
        "observed_hangul_target_call_reduction": hangul,
        "observed_hangul_share_of_scalar_savings": share,
        "pass": (
            hangul >= rule["minimum_hangul_target_call_reduction"]
            and share >= rule["minimum_hangul_share_of_scalar_savings"]
        ),
        "authorizes": rule["pass_authorizes"],
    }
    payload: dict[str, Any] = {
        "claim_scope": {
            "acceptance_or_speed_evidence": False,
            "calibration_only": True,
            "confirmatory": False,
            "perfect_oracle_is_implementation_bound": False,
            "raw_text_promoted": False,
        },
        "complete": True,
        "decision": gate,
        "git_commit": commit,
        "kind": "hangul_block_opportunity_result_v1",
        "metrics": metrics,
        "plan_artifact_sha256": _hash_file(PLAN_PATH),
        "protocol_id": plan["protocol_id"],
        "schema_version": 1,
        "source": {
            "calibration_stream_sha256": stream_sha256,
            "source_artifact_sha256": inputs["source_sha256"],
        },
    }
    payload["summary_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    output = ROOT / plan["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("opportunity result already exists with different bytes")
        return
    output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
