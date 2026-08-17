"""Conditional Phase 3 controls for identifying the whitespace mechanism.

The primary Phase 3 experiment must be summarized before these controls are
trained.  This module keeps the gate check and seed-independent patch
construction testable without importing an executable script.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .neural_model import Phase1ModelSpec
from .neural_patching import hf_patch_lengths
from .phase2_patching import (
    calibrate_placebo_threshold,
    causal_offset_grid_boundaries,
    causal_window_grid_trace,
    rolling_hash_event_mask,
    scheduled_targets,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from .phase3 import PHASE3_MODEL_SPEC


INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_SEEDS = (57721, 65537)
ALL_SEEDS = (*INITIAL_SEEDS, *CONFIRMATION_SEEDS)
DELAYED_POLICY = "causal_grid_delayed2"
PLACEBO_POLICY = "causal_placebo_grid"
WHITESPACE_POLICY = "causal_whitespace_grid"
MECHANISM_POLICIES = (DELAYED_POLICY, PLACEBO_POLICY)

_MANIFEST_INVARIANTS = (
    "phase",
    "quick_smoke_only",
    "language",
    "limits",
    "global_max_position_embeddings",
    "model_spec",
    "optimization_spec",
    "streams",
    "policies",
)


def array_sha256(array: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def mechanism_cache_provenance(
    inputs: Mapping[str, np.ndarray],
    boundaries: Mapping[str, np.ndarray],
    whitespace: Mapping[str, np.ndarray],
    spec: Phase1ModelSpec = PHASE3_MODEL_SPEC,
) -> dict[str, Any]:
    """Identify every input that determines the D/P patch matrices."""

    split_names = _validate_matrices(inputs, boundaries, whitespace, spec)
    return {
        "schema_version": 1,
        "kind": "phase3_mechanism_patch_cache",
        "model_spec": spec.to_dict(),
        "policies": list(MECHANISM_POLICIES),
        "splits": {
            split: {
                "inputs_sha256": array_sha256(inputs[split]),
                "boundaries_sha256": array_sha256(boundaries[split]),
                "whitespace_sha256": array_sha256(whitespace[split]),
            }
            for split in split_names
        },
    }


def merge_mechanism_manifest(
    existing: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge resumable initial/confirmation invocations without losing gates."""

    current_copy = deepcopy(dict(current))
    invocation = {
        key: deepcopy(current_copy[key])
        for key in (
            "created_at",
            "git_commit",
            "device",
            "platform",
            "versions",
            "seeds",
            "policies",
            "gate_authorization",
            "primary_summary_sha256",
            "primary_context_checks",
            "force",
            "save_checkpoints",
        )
        if key in current_copy
    }
    if existing is None:
        current_copy["invocations"] = [invocation]
        return current_copy

    merged = deepcopy(dict(existing))
    for key in _MANIFEST_INVARIANTS:
        if key not in merged or key not in current_copy:
            raise ValueError(f"mechanism manifest invariant is missing: {key}")
        if merged[key] != current_copy[key]:
            raise ValueError(f"mechanism manifest invariant changed: {key}")
    invocations = deepcopy(merged.get("invocations", []))
    if not isinstance(invocations, list):
        raise ValueError("mechanism manifest invocations must be a list")
    invocations.append(invocation)
    merged["invocations"] = invocations
    merged["seeds"] = [
        *merged.get("seeds", []),
        *[
            seed
            for seed in current_copy["seeds"]
            if seed not in merged.get("seeds", [])
        ],
    ]
    merged["updated_at"] = current_copy["created_at"]
    return merged


def validate_mechanism_execution_gate(
    primary_summary: Mapping[str, Any] | None,
    seeds: Sequence[int],
    *,
    quick: bool,
) -> dict[str, Any]:
    """Require Gate I for initial controls and Gate J for confirmations."""

    requested = tuple(int(seed) for seed in seeds)
    if not requested:
        raise ValueError("at least one mechanism seed is required")
    if len(set(requested)) != len(requested):
        raise ValueError("mechanism seeds must be unique")
    unknown = set(requested) - set(ALL_SEEDS)
    if unknown:
        raise ValueError(f"unregistered mechanism seeds: {sorted(unknown)}")
    if quick:
        return {
            "status": "quick_smoke_bypass",
            "evidence_eligible": False,
            "required_gate": None,
            "requested_seeds": list(requested),
        }
    if primary_summary is None:
        raise ValueError("a full primary summary is required before controls")

    requires_confirmation = bool(set(requested) & set(CONFIRMATION_SEEDS))
    gate_name = "gate_j" if requires_confirmation else "gate_i"
    gate = primary_summary.get(gate_name)
    if not isinstance(gate, Mapping):
        raise ValueError(f"primary summary does not contain {gate_name}")
    if gate.get("overall_pass") is not True:
        raise ValueError(
            f"refusing mechanism training because {gate_name} did not pass"
        )
    return {
        "status": "authorized_by_preregistered_gate",
        "evidence_eligible": True,
        "required_gate": gate_name,
        "required_gate_status": gate.get("status"),
        "requested_seeds": list(requested),
    }


def _validate_matrices(
    inputs: Mapping[str, np.ndarray],
    boundaries: Mapping[str, np.ndarray],
    whitespace: Mapping[str, np.ndarray],
    spec: Phase1ModelSpec,
) -> tuple[str, ...]:
    split_names = tuple(inputs)
    if not split_names or "calibration" not in inputs:
        raise ValueError("inputs must include a calibration split")
    if set(boundaries) != set(inputs) or set(whitespace) != set(inputs):
        raise ValueError("input, boundary, and whitespace splits must match")
    expected_width = spec.sequence_length
    for split in split_names:
        shape = inputs[split].shape
        if len(shape) != 2 or shape[1] != expected_width or not shape[0]:
            raise ValueError(f"{split} inputs have an unexpected shape")
        if boundaries[split].shape != shape or whitespace[split].shape != shape:
            raise ValueError(f"{split} masks have an unexpected shape")
    return split_names


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values):
        raise ValueError("diagnostic values must be a non-empty vector")
    return {
        "mean_target_displacement_bytes": float(values.mean()),
        "median_target_displacement_bytes": float(np.median(values)),
        "p05_target_displacement_bytes": float(np.percentile(values, 5)),
        "p95_target_displacement_bytes": float(np.percentile(values, 95)),
        "minimum_target_displacement_bytes": int(values.min()),
        "maximum_target_displacement_bytes": int(values.max()),
    }


def _patch_diagnostics(
    matrix: np.ndarray,
    boundary_masks: np.ndarray,
) -> dict[str, float | int | str]:
    validate_padded_patch_matrix(matrix, boundary_masks.shape[1])
    return {
        **{
            f"patch_{key}": value
            for key, value in variable_patch_diagnostics(
                matrix, boundary_masks
            ).to_dict().items()
        },
        "matrix_sha256": array_sha256(matrix),
    }


def _offset_matrix(
    boundary_masks: np.ndarray,
    spec: Phase1ModelSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = len(boundary_masks)
    matrix = np.empty((rows, spec.patch_count + 1), dtype=np.uint16)
    displacements = np.empty(
        rows * (spec.patch_count - 1), dtype=np.int16
    )
    cursor = 0
    targets = np.asarray(
        scheduled_targets(spec.sequence_length, spec.patch_count),
        dtype=np.int16,
    )
    for row_index, boundary_mask in enumerate(boundary_masks):
        boundaries = causal_offset_grid_boundaries(
            boundary_mask,
            spec.patch_count,
            offset=2,
        )
        matrix[row_index] = np.asarray(
            hf_patch_lengths(boundaries, spec.sequence_length),
            dtype=np.uint16,
        )
        local = np.asarray(boundaries[1:], dtype=np.int16) - targets
        displacements[cursor : cursor + len(local)] = local
        cursor += len(local)
    if cursor != len(displacements):
        raise AssertionError("delayed-grid displacement accounting failed")
    nonfinal = rows * (spec.patch_count - 2)
    diagnostics: dict[str, Any] = {
        "examples": rows,
        "events": 0,
        "deadlines": nonfinal,
        "final_boundaries": rows,
        "nonfinal_boundaries": nonfinal,
        "event_trigger_fraction": 0.0,
        "selected_event_whitespace_count": 0,
        "selected_event_whitespace_rate": None,
        **_distribution(displacements),
        **_patch_diagnostics(matrix, boundary_masks),
    }
    return matrix, diagnostics


def _trace_matrix(
    inputs: np.ndarray,
    boundary_masks: np.ndarray,
    whitespace_masks: np.ndarray,
    event_provider: Callable[[int, np.ndarray], np.ndarray],
    spec: Phase1ModelSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = len(inputs)
    matrix = np.empty((rows, spec.patch_count + 1), dtype=np.uint16)
    displacements = np.empty(
        rows * (spec.patch_count - 1), dtype=np.int16
    )
    cursor = 0
    events = 0
    deadlines = 0
    finals = 0
    whitespace_hits = 0
    for row_index, (row, boundary_mask) in enumerate(
        zip(inputs, boundary_masks, strict=True)
    ):
        event_mask = event_provider(row_index, row)
        if event_mask.shape != boundary_mask.shape:
            raise ValueError("event provider returned an unexpected shape")
        trace = causal_window_grid_trace(
            boundary_mask,
            event_mask,
            spec.patch_count,
        )
        matrix[row_index] = np.asarray(
            hf_patch_lengths(trace.boundaries, spec.sequence_length),
            dtype=np.uint16,
        )
        local = np.asarray(trace.target_displacements, dtype=np.int16)
        displacements[cursor : cursor + len(local)] = local
        cursor += len(local)
        for position, kind in zip(
            trace.boundaries[1:], trace.trigger_kinds, strict=True
        ):
            if kind == "event":
                events += 1
                whitespace_hits += int(bool(whitespace_masks[row_index, position]))
            elif kind == "deadline":
                deadlines += 1
            elif kind == "final":
                finals += 1
            else:  # pragma: no cover - upstream invariant
                raise AssertionError(f"unknown trigger kind: {kind}")
    if cursor != len(displacements):
        raise AssertionError("trace displacement accounting failed")
    nonfinal = events + deadlines
    diagnostics: dict[str, Any] = {
        "examples": rows,
        "events": events,
        "deadlines": deadlines,
        "final_boundaries": finals,
        "nonfinal_boundaries": nonfinal,
        "event_trigger_fraction": events / nonfinal if nonfinal else None,
        "selected_event_whitespace_count": whitespace_hits,
        "selected_event_whitespace_rate": (
            whitespace_hits / events if events else None
        ),
        **_distribution(displacements),
        **_patch_diagnostics(matrix, boundary_masks),
    }
    return matrix, diagnostics


def build_mechanism_patch_matrices(
    inputs: Mapping[str, np.ndarray],
    boundaries: Mapping[str, np.ndarray],
    whitespace: Mapping[str, np.ndarray],
    spec: Phase1ModelSpec = PHASE3_MODEL_SPEC,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Build D/P matrices and an independently reconstructed W reference."""

    split_names = _validate_matrices(inputs, boundaries, whitespace, spec)
    whitespace_reference: dict[str, dict[str, Any]] = {}
    for split in split_names:
        reference_matrix, reference = _trace_matrix(
            inputs[split],
            boundaries[split],
            whitespace[split],
            lambda index, _row, split=split: whitespace[split][index],
            spec,
        )
        whitespace_reference[split] = reference
        del reference_matrix

    target_fraction = float(
        whitespace_reference["calibration"]["event_trigger_fraction"]
    )
    placebo_calibration = calibrate_placebo_threshold(
        inputs["calibration"],
        boundaries["calibration"],
        target_fraction,
        spec.patch_count,
    )

    matrices: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "policies": list(MECHANISM_POLICIES),
        "placebo_calibration": placebo_calibration.to_dict(),
        "calibration_target": {
            "source_policy": WHITESPACE_POLICY,
            "split": "calibration",
            "quantity": "nonfinal_event_trigger_fraction",
            "value": target_fraction,
        },
        "whitespace_reference": whitespace_reference,
        "splits": {},
    }
    for split in split_names:
        delayed, delayed_diagnostics = _offset_matrix(
            boundaries[split], spec
        )
        placebo, placebo_diagnostics = _trace_matrix(
            inputs[split],
            boundaries[split],
            whitespace[split],
            lambda _index, row: rolling_hash_event_mask(
                bytes(row),
                placebo_calibration.low_bit_threshold,
                hash_bits=placebo_calibration.hash_bits,
            ),
            spec,
        )
        matrices[split] = {
            DELAYED_POLICY: delayed,
            PLACEBO_POLICY: placebo,
        }
        diagnostics["splits"][split] = {
            DELAYED_POLICY: delayed_diagnostics,
            PLACEBO_POLICY: placebo_diagnostics,
        }
    return matrices, diagnostics
