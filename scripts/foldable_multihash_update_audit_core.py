"""Pure metrics for the foldable multi-hash first-update audit."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

PROTOCOL_ID = "jamoflow-foldable-multihash-update-audit-v4"


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or array.shape[0] < 2
        or array.shape[1] < 2
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} update matrix differs")
    return array


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("update quantile input differs")
    return {
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "maximum": float(values.max()),
    }


def update_geometry(
    dense_update: np.ndarray,
    multihash_update: np.ndarray,
    exposure_counts: np.ndarray,
) -> dict[str, Any]:
    """Compare one effective multi-hash update with an ordinary dense update."""

    dense = _finite_matrix(dense_update, name="dense")
    candidate = _finite_matrix(multihash_update, name="multihash")
    exposure = np.asarray(exposure_counts)
    if (
        candidate.shape != dense.shape
        or exposure.dtype != np.int64
        or exposure.shape != (len(dense),)
        or np.any(exposure < 0)
    ):
        raise ValueError("update geometry inputs differ")

    dense64 = dense.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    dense_energy = float(np.square(dense64).sum())
    candidate_energy = float(np.square(candidate64).sum())
    if dense_energy <= 0.0 or candidate_energy <= 0.0:
        raise ValueError("update geometry has zero energy")
    dot = float((dense64 * candidate64).sum())
    projection = dot / dense_energy
    cosine = dot / math.sqrt(dense_energy * candidate_energy)
    residual = candidate64 - projection * dense64
    row_dense = np.linalg.norm(dense64, axis=1)
    row_candidate = np.linalg.norm(candidate64, axis=1)
    nonzero = row_dense > 0.0
    if not np.all(nonzero):
        raise ValueError("dense row update is zero")
    row_ratio = row_candidate / row_dense
    row_dot = (dense64 * candidate64).sum(axis=1)
    row_cosine = row_dot / (row_dense * row_candidate)
    if not np.isfinite(row_cosine).all():
        raise ValueError("row update cosine differs")

    order = np.argsort(exposure, kind="stable")
    strata: list[dict[str, Any]] = []
    for index, indices in enumerate(np.array_split(order, 4)):
        if len(indices) == 0:
            raise ValueError("exposure stratum is empty")
        strata.append(
            {
                "stratum": index,
                "row_count": len(indices),
                "minimum_exposure": int(exposure[indices].min()),
                "maximum_exposure": int(exposure[indices].max()),
                "median_norm_ratio": float(np.median(row_ratio[indices])),
                "median_cosine": float(np.median(row_cosine[indices])),
            }
        )

    result = {
        "candidate_update_sha256": array_sha256(candidate),
        "cosine": float(cosine),
        "dense_update_sha256": array_sha256(dense),
        "energy_ratio": float(candidate_energy / dense_energy),
        "exposure_strata": strata,
        "frobenius_norm_ratio": float(math.sqrt(candidate_energy / dense_energy)),
        "orthogonal_fraction_of_candidate": float(
            math.sqrt(float(np.square(residual).sum()) / candidate_energy)
        ),
        "projection_multiplier": float(projection),
        "row_cosine": _quantiles(row_cosine),
        "row_norm_ratio": _quantiles(row_ratio),
        "row_count": len(dense),
    }
    if any(
        not math.isfinite(float(value))
        for key, value in result.items()
        if key
        in {
            "cosine",
            "energy_ratio",
            "frobenius_norm_ratio",
            "orthogonal_fraction_of_candidate",
            "projection_multiplier",
        }
    ):
        raise AssertionError("update geometry is nonfinite")
    return result


def select_update_matched_control(
    geometry_by_matrix: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one input/output row multiplier without using a model-quality metric."""

    if set(geometry_by_matrix) != {"input", "output"}:
        raise ValueError("update-control matrix set differs")
    multipliers: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for name in ("input", "output"):
        row = geometry_by_matrix[name]
        if set(row) != {
            "candidate_update_sha256",
            "cosine",
            "dense_update_sha256",
            "energy_ratio",
            "exposure_strata",
            "frobenius_norm_ratio",
            "orthogonal_fraction_of_candidate",
            "projection_multiplier",
            "row_cosine",
            "row_norm_ratio",
            "row_count",
        }:
            raise ValueError("update-control geometry schema differs")
        multiplier = float(row["projection_multiplier"])
        if not math.isfinite(multiplier) or not 1.0 < multiplier < 16.0:
            raise ValueError(
                "update-control multiplier is outside the sealed safety range"
            )
        multipliers[name] = multiplier
        diagnostics[name] = {
            "cosine": float(row["cosine"]),
            "frobenius_norm_ratio": float(row["frobenius_norm_ratio"]),
            "orthogonal_fraction_of_candidate": float(
                row["orthogonal_fraction_of_candidate"]
            ),
        }
    return {
        "control_kind": "post_adamw_new_row_update_projection_v1",
        "input_multiplier": multipliers["input"],
        "output_multiplier": multipliers["output"],
        "quality_metric_used": False,
        "source": "fixed_first_training_batch_effective_update_projection",
        "diagnostics": diagnostics,
    }
