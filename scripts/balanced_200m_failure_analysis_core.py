"""Pure contracts and statistics for the balanced-200M failure analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PROTOCOL_ID = "jamoflow-balanced-200m-quality-failure-analysis-v1"
VERIFICATION_KIND = "balanced_200m_checkpoint_replay_receipt_v1"
ANALYSIS_KIND = "balanced_200m_quality_failure_analysis_v1"
BLOCK_SIZE = 64
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260903
CANDIDATE_PATCH_COUNTS = (72, 76, 78, 80, 82, 84, 86)
PREDICTED_BYTES_PER_SEQUENCE = 511


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def paired_bpb_effects(c86_nll: np.ndarray, w72_nll: np.ndarray) -> np.ndarray:
    left = np.asarray(c86_nll)
    right = np.asarray(w72_nll)
    if (
        left.dtype != np.float32
        or right.dtype != np.float32
        or left.ndim != 1
        or right.shape != left.shape
        or len(left) == 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or (left < 0).any()
        or (right < 0).any()
    ):
        raise ValueError("balanced-200M paired NLL arrays differ")
    return np.ascontiguousarray(
        (right.astype(np.float64) - left.astype(np.float64))
        / (PREDICTED_BYTES_PER_SEQUENCE * math.log(2.0))
    )


def contiguous_block_bootstrap(
    effects: np.ndarray,
    *,
    block_size: int = BLOCK_SIZE,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    values = np.asarray(effects, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < block_size * 2
        or block_size <= 0
        or repetitions <= 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("balanced-200M block-bootstrap input differs")
    usable = len(values) // block_size * block_size
    blocks = values[:usable].reshape(-1, block_size).mean(axis=1)
    generator = np.random.default_rng(seed)
    draws = blocks[
        generator.integers(0, len(blocks), size=(repetitions, len(blocks)))
    ].mean(axis=1)
    return {
        "block_size_sequences": block_size,
        "block_count": int(len(blocks)),
        "used_sequences": int(usable),
        "dropped_tail_sequences": int(len(values) - usable),
        "repetitions": repetitions,
        "seed": seed,
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("rank input differs")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(feature: np.ndarray, effects: np.ndarray) -> float:
    x = np.asarray(feature)
    y = np.asarray(effects, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("balanced-200M correlation input differs")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def equal_count_quintiles(feature: np.ndarray, effects: np.ndarray) -> list[dict[str, Any]]:
    x = np.asarray(feature)
    y = np.asarray(effects, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 5:
        raise ValueError("balanced-200M quintile input differs")
    order = np.lexsort((np.arange(len(x)), x))
    rows: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(order, 5), start=1):
        rows.append(
            {
                "quintile": index,
                "examples": int(len(indices)),
                "feature_minimum": float(np.min(x[indices])),
                "feature_maximum": float(np.max(x[indices])),
                "mean_effect_bpb": float(np.mean(y[indices])),
                "median_effect_bpb": float(np.median(y[indices])),
                "positive_effect_rate": float(np.mean(y[indices] > 0)),
            }
        )
    return rows


def positive_excess_concentration(effects: np.ndarray) -> dict[str, float]:
    positive = np.maximum(np.asarray(effects, dtype=np.float64), 0.0)
    total = float(positive.sum())
    if total <= 0:
        return {"top_1_percent": 0.0, "top_5_percent": 0.0, "top_10_percent": 0.0}
    ordered = np.sort(positive)[::-1]
    output: dict[str, float] = {}
    for percentage in (1, 5, 10):
        count = max(1, math.ceil(len(ordered) * percentage / 100))
        output[f"top_{percentage}_percent"] = float(ordered[:count].sum() / total)
    return output


def linear_density_heuristic(observed_w72_delta_bpb: float, patch_count: int) -> float:
    if not math.isfinite(observed_w72_delta_bpb) or patch_count not in range(72, 87):
        raise ValueError("balanced-200M density heuristic input differs")
    return observed_w72_delta_bpb * (86 - patch_count) / (86 - 72)


def validate_verification_receipt(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "kind",
        "protocol_id",
        "verification_base_git_commit",
        "plan_artifact_sha256",
        "plan_sha256",
        "training_summary_artifact_sha256",
        "training_summary_sha256",
        "training_summary_git_commit",
        "sealed_verifier_sha256",
        "transcript_sha256",
        "checkpoint_replay_roles",
        "independent_checkpoint_replay_pass",
        "quality_status",
        "quality",
        "actual_timing_authorized",
        "claim_boundary",
        "receipt_sha256",
    }
    if set(value) != expected:
        raise ValueError("balanced-200M verification receipt schema differs")
    payload = dict(value)
    claimed = payload.pop("receipt_sha256")
    quality = value.get("quality")
    if (
        value["schema_version"] != 1
        or value["kind"] != VERIFICATION_KIND
        or value["protocol_id"] != PROTOCOL_ID
        or not is_git_commit(value["verification_base_git_commit"])
        or not is_git_commit(value["training_summary_git_commit"])
        or any(
            not is_sha256(value[key])
            for key in (
                "plan_artifact_sha256",
                "plan_sha256",
                "training_summary_artifact_sha256",
                "training_summary_sha256",
                "sealed_verifier_sha256",
                "transcript_sha256",
            )
        )
        or value["checkpoint_replay_roles"] != ["c86", "w72"]
        or value["independent_checkpoint_replay_pass"] is not True
        or value["quality_status"] != "balanced_200m_quality_fail"
        or not isinstance(quality, Mapping)
        or quality.get("quality_screen_pass") is not False
        or quality.get("actual_timing_authorized") is not False
        or value["actual_timing_authorized"] is not False
        or value["claim_boundary"]
        != {
            "one_seed_mechanism_screen": True,
            "sufficiently_trained_llm_claimed": False,
            "actual_incremental_timing_executed": False,
            "verification_replays_full_calibration_forward": True,
        }
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
    ):
        raise ValueError("balanced-200M verification receipt differs")

