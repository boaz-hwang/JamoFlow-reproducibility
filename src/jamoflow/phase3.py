"""Pinned model, optimization, and structural patch policies for Phase 3."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .neural_model import Phase1ModelSpec
from .neural_patching import fixed_byte_boundaries, hf_patch_lengths
from .neural_training import OptimizationSpec
from .patching import is_spacebyte_spacelike
from .phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_window_grid_trace,
    padded_hf_patch_matrix,
)


PHASE3_MODEL_SPEC = Phase1ModelSpec(
    sequence_length=512,
    patch_count=86,
    patch_stride=6,
    local_width=192,
    global_width=384,
    local_heads=6,
    global_heads=8,
    encoder_layers=2,
    global_layers=8,
    decoder_layers=2,
    local_ffn=576,
    global_ffn=1152,
    cross_attention_k=2,
    hash_group_size=3,
    hash_vocabulary=8192,
    router_width=192,
    router_heads=6,
    router_layers=4,
    router_ffn=576,
)

PHASE3_OPTIMIZATION_SPEC = OptimizationSpec(
    batch_size=32,
    router_batch_size=64,
    evaluation_batch_size=64,
    learning_rate=3e-4,
    minimum_learning_rate=3e-5,
    warmup_steps=500,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    weight_decay=0.1,
    gradient_clip=1.0,
)

STRUCTURAL_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
    "spacebyte_spacelike",
)
THRESHOLD_POLICIES = (
    "entropy_threshold_full",
    "entropy_threshold_codepoint",
)
PHASE3_POLICIES = (*STRUCTURAL_POLICIES, *THRESHOLD_POLICIES)

_MANIFEST_INVARIANTS = (
    "quick_smoke_only",
    "language",
    "limits",
    "source_artifact",
    "source_integrity_artifact",
    "global_max_position_embeddings",
    "model_spec",
    "optimization_spec",
    "streams",
)
_UPGRADEABLE_MANIFEST_INVARIANTS = {
    "source_artifact",
    "source_integrity_artifact",
}

_OOD_MANIFEST_INVARIANTS = (
    "schema_version",
    "requested_byte_limit",
    "source",
    "stream",
    "global_max_position_embeddings",
    "model_spec",
)


def _manifest_invocation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract invocation-scoped provenance from a Phase 3 manifest."""

    invocation = {
        key: deepcopy(manifest[key])
        for key in (
            "created_at",
            "git_commit",
            "device",
            "platform",
            "versions",
            "seeds",
            "policies",
            "git_worktree_clean_at_start",
        )
        if key in manifest
    }
    for key in ("force", "save_checkpoints", "authorization"):
        if key in manifest:
            invocation[key] = deepcopy(manifest[key])
    return invocation


def merge_phase3_manifest(
    existing: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge one run invocation without losing earlier Phase 3 provenance.

    A run root may be populated in several invocations (primary policies,
    learned baselines, and confirmation seeds). Dataset and experiment design
    fields must remain identical; invocation-varying fields are preserved in
    an append-only list while the top-level seed/policy lists become unions.
    """

    current_copy = deepcopy(dict(current))
    current_invocation = _manifest_invocation(current_copy)
    if existing is None:
        current_copy["invocations"] = [current_invocation]
        return current_copy

    merged = deepcopy(dict(existing))
    for key in _MANIFEST_INVARIANTS:
        if key not in current_copy:
            raise ValueError(f"Phase 3 manifest invariant is missing: {key}")
        if key not in merged and key in _UPGRADEABLE_MANIFEST_INVARIANTS:
            merged[key] = deepcopy(current_copy[key])
            continue
        if key not in merged:
            raise ValueError(f"Phase 3 manifest invariant is missing: {key}")
        if merged[key] != current_copy[key]:
            raise ValueError(f"Phase 3 manifest invariant changed: {key}")

    invocations = deepcopy(merged.get("invocations"))
    if invocations is None:
        # Upgrade manifests written before invocation-level provenance existed.
        invocations = [_manifest_invocation(merged)]
    if not isinstance(invocations, list):
        raise ValueError("Phase 3 manifest invocations must be a list")
    invocations.append(current_invocation)
    merged["invocations"] = invocations

    for key in ("seeds", "policies"):
        values = list(merged.get(key, []))
        values.extend(value for value in current_copy[key] if value not in values)
        merged[key] = values
    merged["updated_at"] = current_copy["created_at"]
    return merged


def _ood_manifest_invocation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Extract invocation-scoped provenance from a Phase 3 OOD manifest."""

    return {
        key: deepcopy(manifest[key])
        for key in (
            "created_at",
            "git_commit",
            "device",
            "platform",
            "versions",
            "seeds",
            "policies",
            "force",
            "authorization",
            "git_worktree_clean_at_start",
        )
        if key in manifest
    }


def merge_phase3_ood_manifest(
    existing: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge OOD invocations while pinning corpus and model provenance.

    The initial three seeds and two confirmation seeds are evaluated in
    separate invocations.  Their manifest must retain both invocations rather
    than silently replacing the first one, while any change to the source
    artifact, selected stream, byte limit, or model design is rejected.
    """

    current_copy = deepcopy(dict(current))
    current_invocation = _ood_manifest_invocation(current_copy)
    if existing is None:
        current_copy["invocations"] = [current_invocation]
        return current_copy

    merged = deepcopy(dict(existing))
    for key in _OOD_MANIFEST_INVARIANTS:
        if key not in merged or key not in current_copy:
            raise ValueError(f"Phase 3 OOD manifest invariant is missing: {key}")
        if merged[key] != current_copy[key]:
            raise ValueError(f"Phase 3 OOD manifest invariant changed: {key}")

    invocations = deepcopy(merged.get("invocations"))
    if invocations is None:
        invocations = [_ood_manifest_invocation(merged)]
    if not isinstance(invocations, list):
        raise ValueError("Phase 3 OOD manifest invocations must be a list")
    invocations.append(current_invocation)
    merged["invocations"] = invocations

    for key in ("seeds", "policies"):
        values = list(merged.get(key, []))
        values.extend(value for value in current_copy[key] if value not in values)
        merged[key] = values
    merged["updated_at"] = current_copy["created_at"]
    return merged


def spacebyte_causal_prefix_mask(data: bytes) -> np.ndarray:
    """Mark BLT boundaries after official SpaceByte global-position bytes.

    The suppression state is scanned over the complete continuous stream before
    reshaping into windows. Entry ``t`` describes the prefix ``data[:t]`` and
    therefore remains usable before consuming byte ``t``.
    """

    output = np.zeros(len(data), dtype=np.uint8)
    previous_spacelike = False
    for index, value in enumerate(data):
        current_spacelike = is_spacebyte_spacelike(value)
        prefix_position = index + 1
        if (
            prefix_position < len(data)
            and current_spacelike
            and not previous_spacelike
        ):
            output[prefix_position] = 1
        previous_spacelike = current_spacelike
    return output


def spacebyte_boundaries(event_mask: np.ndarray) -> tuple[int, ...]:
    if event_mask.ndim != 1 or not len(event_mask):
        raise ValueError("SpaceByte event mask must be a non-empty vector")
    return (0, *map(int, np.flatnonzero(event_mask[1:] != 0) + 1))


def structural_patch_matrices(
    boundary_masks: np.ndarray,
    whitespace_masks: np.ndarray,
    spacebyte_masks: np.ndarray,
    spec: Phase1ModelSpec = PHASE3_MODEL_SPEC,
) -> dict[str, np.ndarray]:
    expected = (len(boundary_masks), spec.sequence_length)
    if boundary_masks.ndim != 2 or boundary_masks.shape != expected:
        raise ValueError("boundary masks have an unexpected shape")
    if whitespace_masks.shape != expected or spacebyte_masks.shape != expected:
        raise ValueError("all structural masks must have equal shape")

    fixed_row = np.asarray(
        hf_patch_lengths(
            fixed_byte_boundaries(spec.sequence_length, spec.patch_stride),
            spec.sequence_length,
        ),
        dtype=np.uint16,
    )
    fixed = np.broadcast_to(
        fixed_row,
        (len(boundary_masks), len(fixed_row)),
    ).copy()
    codepoint_rows: list[tuple[int, ...]] = []
    whitespace_rows: list[tuple[int, ...]] = []
    spacebyte_rows: list[tuple[int, ...]] = []
    for index, (boundary, whitespace, spacebyte) in enumerate(
        zip(
            boundary_masks,
            whitespace_masks,
            spacebyte_masks,
            strict=True,
        )
    ):
        try:
            codepoint_rows.append(
                causal_codepoint_grid_boundaries(boundary, spec.patch_count)
            )
            whitespace_rows.append(
                causal_window_grid_trace(
                    boundary,
                    whitespace,
                    spec.patch_count,
                ).boundaries
            )
            spacebyte_rows.append(spacebyte_boundaries(spacebyte))
        except ValueError as exc:
            raise ValueError(f"cannot construct Phase 3 row {index}: {exc}") from exc

    codepoint = padded_hf_patch_matrix(codepoint_rows, spec.sequence_length)
    whitespace = padded_hf_patch_matrix(whitespace_rows, spec.sequence_length)
    spacebyte = padded_hf_patch_matrix(spacebyte_rows, spec.sequence_length)
    if fixed.shape != codepoint.shape or codepoint.shape != whitespace.shape:
        raise AssertionError("F/C/W exact-rate matrices must have equal shapes")
    return {
        "fixed_byte_6": fixed,
        "causal_codepoint_grid": codepoint,
        "causal_whitespace_grid": whitespace,
        "spacebyte_spacelike": spacebyte,
    }
