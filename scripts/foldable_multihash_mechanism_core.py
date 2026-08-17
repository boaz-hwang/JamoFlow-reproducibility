"""Pure controls and decisions for the foldable multi-hash mechanism screen."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from bpe_quality_frontier_core import document_bootstrap_upper
from foldable_jamo_residual_core import (
    CODEBOOK_SIZE,
    RESIDUAL_SLOT_COUNT,
    array_sha256,
)
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    TARGET_VOCABULARY_SIZE,
)

PROTOCOL_ID = "jamoflow-foldable-multihash-mechanism-v1"
NEW_ROLES = (
    "update_matched_dense",
    "stratified_generic_shuffle",
    "balanced_random_multihash",
)
REFERENCE_ROLES = ("untied_base", "untied_generic_surface")
ALL_ROLES = (*REFERENCE_ROLES, *NEW_ROLES)
INPUT_UPDATE_MULTIPLIER = 1.485414522979104
OUTPUT_UPDATE_MULTIPLIER = 2.170601418278963
STRATIFIED_SHUFFLE_SEED = 20_260_834
BALANCED_RANDOM_SEED = 20_260_835
BOOTSTRAP_SEED = 20_260_836
BOOTSTRAP_REPETITIONS = 10_000
MINIMUM_ADVANTAGE_BPB = 0.002
MAXIMUM_ANCHOR_GAP_BPB = 0.050


def _validate_assignment(values: np.ndarray) -> np.ndarray:
    assignment = np.asarray(values)
    if (
        assignment.dtype != np.int64
        or assignment.shape != (TARGET_VOCABULARY_SIZE, RESIDUAL_SLOT_COUNT)
        or np.any(assignment < 0)
        or np.any(assignment >= CODEBOOK_SIZE)
    ):
        raise ValueError("mechanism assignment differs")
    return assignment


def generic_assignment_from_code_indices(values: torch.Tensor) -> np.ndarray:
    if (
        values.dtype != torch.int64
        or tuple(values.shape) != (TARGET_VOCABULARY_SIZE, RESIDUAL_SLOT_COUNT)
    ):
        raise ValueError("generic code indices differ")
    offsets = np.arange(RESIDUAL_SLOT_COUNT, dtype=np.int64) * CODEBOOK_SIZE
    assignment = values.detach().cpu().numpy().astype(np.int64, copy=True)
    assignment -= offsets[None, :]
    return _validate_assignment(assignment)


def stratified_generic_shuffle(
    generic_assignment: np.ndarray,
    token_bytes: Sequence[bytes],
    exposure_counts: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    generic = _validate_assignment(generic_assignment)
    pieces = tuple(token_bytes)
    exposure = np.asarray(exposure_counts)
    if (
        len(pieces) != TARGET_VOCABULARY_SIZE
        or exposure.dtype != np.int64
        or exposure.shape != (TARGET_VOCABULARY_SIZE,)
        or np.any(exposure < 0)
    ):
        raise ValueError("stratified assignment inputs differ")
    lengths = np.asarray([len(piece) for piece in pieces], dtype=np.int64)
    new_rows = np.arange(BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
    source_by_target = np.arange(TARGET_VOCABULARY_SIZE)
    rng = np.random.default_rng(STRATIFIED_SHUFFLE_SEED)
    stratum_sizes: list[int] = []
    for length in np.unique(lengths[new_rows]):
        rows_at_length = new_rows[lengths[new_rows] == length]
        for count in np.unique(exposure[rows_at_length]):
            rows = rows_at_length[exposure[rows_at_length] == count]
            stratum_sizes.append(len(rows))
            if len(rows) > 1:
                ordered = rows[rng.permutation(len(rows))]
                source_by_target[ordered] = np.roll(ordered, -1)
                if np.any(source_by_target[rows] == rows):
                    raise AssertionError("stratified assignment derangement differs")
    if not stratum_sizes or sum(stratum_sizes) != len(new_rows):
        raise AssertionError("stratified assignment partition differs")
    output = generic.copy()
    output[new_rows] = generic[source_by_target[new_rows]]
    _validate_assignment(output)
    non_singleton_rows = int(sum(size for size in stratum_sizes if size > 1))
    return output, {
        "seed": STRATIFIED_SHUFFLE_SEED,
        "stratum_count": len(stratum_sizes),
        "singleton_stratum_count": sum(size == 1 for size in stratum_sizes),
        "non_singleton_row_count": non_singleton_rows,
        "minimum_stratum_size": min(stratum_sizes),
        "median_stratum_size": float(np.median(stratum_sizes)),
        "maximum_stratum_size": max(stratum_sizes),
        "source_row_mapping_sha256": array_sha256(source_by_target.astype(np.int64)),
    }


def balanced_random_assignment(
    generic_assignment: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    generic = _validate_assignment(generic_assignment)
    new_count = TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE
    if new_count % CODEBOOK_SIZE != 0:
        raise AssertionError("balanced random occupancy is not integral")
    labels = np.repeat(np.arange(CODEBOOK_SIZE, dtype=np.int64), new_count // CODEBOOK_SIZE)
    rng = np.random.default_rng(BALANCED_RANDOM_SEED)
    output = generic.copy()
    for slot in range(RESIDUAL_SLOT_COUNT):
        output[BASE_VOCABULARY_SIZE:, slot] = labels[rng.permutation(new_count)]
    _validate_assignment(output)
    occupancy = np.stack(
        [
            np.bincount(
                output[BASE_VOCABULARY_SIZE:, slot], minlength=CODEBOOK_SIZE
            )
            for slot in range(RESIDUAL_SLOT_COUNT)
        ]
    ).astype(np.int64, copy=False)
    expected = new_count // CODEBOOK_SIZE
    if not np.all(occupancy == expected):
        raise AssertionError("balanced random occupancy differs")
    return output, {
        "seed": BALANCED_RANDOM_SEED,
        "occupancy_per_bucket": expected,
        "occupancy_sha256": array_sha256(occupancy),
    }


def assignment_audit(
    assignment: np.ndarray,
    generic_assignment: np.ndarray,
    exposure_counts: np.ndarray,
    *,
    kind: str,
    construction: Mapping[str, Any],
) -> dict[str, Any]:
    values = _validate_assignment(assignment)
    generic = _validate_assignment(generic_assignment)
    exposure = np.asarray(exposure_counts)
    if exposure.dtype != np.int64 or exposure.shape != (TARGET_VOCABULARY_SIZE,):
        raise ValueError("mechanism assignment exposure differs")
    new = slice(BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
    changed = np.any(values[new] != generic[new], axis=1)
    new_exposure = exposure[new]
    total_exposure = int(new_exposure.sum())
    unique_counts: list[int] = []
    entropies: list[float] = []
    occupancy_hashes: list[str] = []
    for slot in range(RESIDUAL_SLOT_COUNT):
        counts = np.bincount(values[new, slot], minlength=CODEBOOK_SIZE).astype(
            np.int64, copy=False
        )
        probabilities = counts[counts > 0].astype(np.float64) / int(counts.sum())
        unique_counts.append(int(np.count_nonzero(counts)))
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
        occupancy_hashes.append(array_sha256(counts))
    unique_vectors = np.unique(values[new], axis=0)
    return {
        "assignment_kind": kind,
        "assignment_sha256": array_sha256(values),
        "construction": dict(construction),
        "new_row_count": TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE,
        "changed_new_row_count_vs_generic": int(changed.sum()),
        "changed_new_row_fraction_vs_generic": float(changed.mean()),
        "changed_new_token_exposure_fraction_vs_generic": (
            0.0
            if total_exposure == 0
            else float(new_exposure[changed].sum() / total_exposure)
        ),
        "slot_unique_counts": unique_counts,
        "slot_entropy_bits": entropies,
        "slot_occupancy_sha256": occupancy_hashes,
        "unique_new_code_vector_count": len(unique_vectors),
    }


def install_assignment(model: Any, assignment: np.ndarray) -> None:
    values = _validate_assignment(assignment)
    offsets = torch.arange(RESIDUAL_SLOT_COUNT, dtype=torch.long) * CODEBOOK_SIZE
    codes = torch.from_numpy(values.copy()).long() + offsets.unsqueeze(0)
    target = model.foldable_residual.code_indices
    if target.dtype != torch.int64 or tuple(target.shape) != tuple(codes.shape):
        raise ValueError("mechanism model code buffer differs")
    with torch.no_grad():
        target.copy_(codes.to(target.device))


def scale_new_row_update_(
    weight: torch.Tensor,
    before_new_rows: torch.Tensor,
    multiplier: float,
) -> None:
    if (
        weight.ndim != 2
        or before_new_rows.shape != weight[BASE_VOCABULARY_SIZE:].shape
        or before_new_rows.dtype != weight.dtype
        or not math.isfinite(multiplier)
        or multiplier <= 1.0
    ):
        raise ValueError("post-AdamW row scaling inputs differ")
    with torch.no_grad():
        rows = weight[BASE_VOCABULARY_SIZE:]
        rows.copy_(before_new_rows + multiplier * (rows - before_new_rows))


def _contrast(
    candidate: str,
    control: str,
    contiguous_bpb: Mapping[str, float],
    document_nll: Mapping[str, np.ndarray],
    document_raw_bytes: np.ndarray,
    *,
    minimum_advantage: float,
    seed: int,
) -> dict[str, Any]:
    point, lower, upper = document_bootstrap_upper(
        document_nll[candidate],
        document_nll[control],
        document_raw_bytes,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=seed,
    )
    contiguous_difference = float(contiguous_bpb[candidate]) - float(
        contiguous_bpb[control]
    )
    return {
        "candidate": candidate,
        "control": control,
        "minimum_required_advantage_bpb": minimum_advantage,
        "contiguous_bpb_difference": contiguous_difference,
        "document_bpb_difference": point,
        "bootstrap_95_lower": lower,
        "bootstrap_95_upper": upper,
        "pass": bool(
            contiguous_difference <= -minimum_advantage
            and point <= -minimum_advantage
            and upper <= 0.0
        ),
    }


def mechanism_decision(
    contiguous_bpb: Mapping[str, float],
    document_nll: Mapping[str, np.ndarray],
    document_raw_bytes: np.ndarray,
    *,
    anchor_bpb: float,
) -> dict[str, Any]:
    if (
        set(contiguous_bpb) != set(ALL_ROLES)
        or set(document_nll) != set(ALL_ROLES)
        or document_raw_bytes.ndim != 1
        or len(document_raw_bytes) < 2
        or np.any(document_raw_bytes <= 0)
        or not math.isfinite(anchor_bpb)
        or anchor_bpb <= 0.0
    ):
        raise ValueError("mechanism decision inputs differ")
    primary = _contrast(
        "untied_generic_surface",
        "update_matched_dense",
        contiguous_bpb,
        document_nll,
        document_raw_bytes,
        minimum_advantage=MINIMUM_ADVANTAGE_BPB,
        seed=BOOTSTRAP_SEED,
    )
    base = _contrast(
        "untied_generic_surface",
        "untied_base",
        contiguous_bpb,
        document_nll,
        document_raw_bytes,
        minimum_advantage=0.0,
        seed=BOOTSTRAP_SEED + 1,
    )
    surface_controls = {
        role: _contrast(
            "untied_generic_surface",
            role,
            contiguous_bpb,
            document_nll,
            document_raw_bytes,
            minimum_advantage=MINIMUM_ADVANTAGE_BPB,
            seed=BOOTSTRAP_SEED + 2 + index,
        )
        for index, role in enumerate(
            ("stratified_generic_shuffle", "balanced_random_multihash")
        )
    }
    random_vs_scale = {
        role: _contrast(
            role,
            "update_matched_dense",
            contiguous_bpb,
            document_nll,
            document_raw_bytes,
            minimum_advantage=MINIMUM_ADVANTAGE_BPB,
            seed=BOOTSTRAP_SEED + 10 + index,
        )
        for index, role in enumerate(
            ("stratified_generic_shuffle", "balanced_random_multihash")
        )
    }
    anchor_gap = float(contiguous_bpb["untied_generic_surface"]) - anchor_bpb
    anchor_pass = anchor_gap <= MAXIMUM_ANCHOR_GAP_BPB
    primary_pass = bool(primary["pass"] and base["pass"] and anchor_pass)
    surface_pass = bool(primary_pass and all(row["pass"] for row in surface_controls.values()))
    random_opportunities = [
        role for role, row in random_vs_scale.items() if row["pass"]
    ]
    return {
        "status": (
            "foldable_multihash_mechanism_pass"
            if primary_pass
            else (
                "generic_surface_stopped_random_opportunity_requires_new_protocol"
                if random_opportunities
                else "foldable_multihash_mechanism_stopped"
            )
        ),
        "primary_candidate": "untied_generic_surface",
        "primary_scale_control": primary,
        "historical_base_control": base,
        "anchor_gap_bpb": anchor_gap,
        "anchor_recovery_pass": anchor_pass,
        "surface_assignment_supported": surface_pass,
        "surface_assignment_controls": surface_controls,
        "random_shared_hash_diagnostics": random_vs_scale,
        "random_opportunities_requiring_new_protocol": random_opportunities,
        "fresh_korean_multiseed_stage_authorized": primary_pass,
        "threshold_or_role_fallback": None,
    }
