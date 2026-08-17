#!/usr/bin/env python3
"""Validate and aggregate paired Phase 3 NFC/NFD stress results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np
import torch

from jamoflow.neural_data import build_neural_stream
from jamoflow.normalization import (
    count_precomposed_hangul,
    oracle_hangul_unit_boundary_mask,
    padded_normalization_stream,
    transform_text,
)
from jamoflow.phase1 import patch_boundaries_from_lengths, stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)


CONDITIONS = ("nfc", "nfd")
F = "fixed_byte_6"
C = "causal_codepoint_grid"
W = "causal_whitespace_grid"
POLICIES = (F, C, W)
KNOWN_SEEDS = (1729, 2718, 31415, 57721, 65537)
INITIAL_SEEDS = KNOWN_SEEDS[:3]
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
METRICS = (
    "bpb",
    "scored_bits_per_source_utf8_byte",
    "scored_bits_per_source_unicode_codepoint",
    "scored_bits_per_source_precomposed_hangul_syllable",
)
POLICY_CONTRASTS = {
    "whitespace_minus_codepoint": (W, C),
    "whitespace_minus_fixed": (W, F),
    "codepoint_minus_fixed": (C, F),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"unexpected checkpoint entry in {path}")
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _strict_decodable_prefix(data: bytes) -> tuple[str, int]:
    try:
        return data.decode("utf-8", errors="strict"), 0
    except UnicodeDecodeError as error:
        if error.end != len(data) or len(data) - error.start > 3:
            raise ValueError("source contains a non-terminal UTF-8 error") from error
        prefix = data[: error.start]
        return prefix.decode("utf-8", errors="strict"), len(data) - len(prefix)


def _conjoining_jamo_counts(text: str) -> dict[str, int]:
    counts = {"leading": 0, "vowel": 0, "trailing": 0}
    for character in text:
        value = ord(character)
        if 0x1100 <= value <= 0x115F or 0xA960 <= value <= 0xA97F:
            counts["leading"] += 1
        elif 0x1160 <= value <= 0x11A7 or 0xD7B0 <= value <= 0xD7C6:
            counts["vowel"] += 1
        elif 0x11A8 <= value <= 0x11FF or 0xD7CB <= value <= 0xD7FB:
            counts["trailing"] += 1
    return counts


def _inside_hangul_unit_diagnostics(
    matrix: np.ndarray,
    unit_masks: np.ndarray,
) -> dict[str, int | float]:
    boundaries = patch_boundaries_from_lengths(matrix)
    rows = np.arange(len(boundaries))[:, None]
    inside = int((unit_masks[rows, boundaries] == 0).sum())
    total = int(boundaries.size)
    return {
        "inside_descriptive_hangul_unit_boundaries": inside,
        "total_noninitial_boundaries": total,
        "inside_descriptive_hangul_unit_boundary_rate": (
            inside / total if total else math.nan
        ),
    }


def _reconstruct_conditions(
    manifest: dict[str, Any],
    data_root: Path,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, int],
]:
    if manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("normalization manifest model spec mismatch")
    if manifest.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT:
        raise ValueError("normalization global position limit mismatch")
    source_path = data_root / "ko.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_manifest = manifest.get("source", {})
    artifact = source_manifest.get("source_artifact", {})
    expected_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    if artifact != expected_artifact:
        raise ValueError("normalization source artifact mismatch")
    byte_limit = source_manifest.get("requested_byte_limit")
    if not isinstance(byte_limit, int) or byte_limit <= 0:
        raise ValueError("normalization source byte limit is invalid")
    source_stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=byte_limit,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    source_text, discarded_tail = _strict_decodable_prefix(source_stream.data)
    source_bytes = source_text.encode("utf-8")
    denominators = {
        "utf8_bytes": len(source_bytes),
        "unicode_codepoints": len(source_text),
        "precomposed_hangul_syllables": count_precomposed_hangul(source_text),
    }
    expected_source = {
        "source_artifact": expected_artifact,
        "requested_byte_limit": byte_limit,
        "primary_stream": source_stream.metadata(),
        "primary_stream_sha256": _sha256_bytes(source_stream.data),
        "strict_source_utf8_sha256": _sha256_bytes(source_bytes),
        "strict_source_bytes": len(source_bytes),
        "discarded_incomplete_terminal_bytes": discarded_tail,
        "denominators": denominators,
    }
    if source_manifest != expected_source:
        raise ValueError("normalization source stream differs from reconstruction")

    condition_inputs: dict[str, np.ndarray] = {}
    condition_masks: dict[str, np.ndarray] = {}
    condition_matrices: dict[str, dict[str, np.ndarray]] = {}
    condition_metadata: dict[str, Any] = {}
    for condition in CONDITIONS:
        padded = padded_normalization_stream(
            source_text,
            condition,
            PHASE3_MODEL_SPEC.sequence_length,
        )
        inputs, boundaries = stream_arrays(
            padded.data,
            padded.codepoint_boundaries,
            padded.sequence_length,
        )
        whitespace = compact_whitespace_mask(padded.data).reshape(inputs.shape)
        spacelike = spacebyte_causal_prefix_mask(padded.data).reshape(inputs.shape)
        structural = structural_patch_matrices(
            boundaries,
            whitespace,
            spacelike,
        )
        matrices = {policy: structural[policy] for policy in POLICIES}
        unit_masks = oracle_hangul_unit_boundary_mask(padded.data).reshape(
            inputs.shape
        )
        patch_diagnostics: dict[str, Any] = {}
        for policy, matrix in matrices.items():
            validate_padded_patch_matrix(
                matrix,
                PHASE3_MODEL_SPEC.sequence_length,
            )
            values = variable_patch_diagnostics(matrix, boundaries).to_dict()
            patch_diagnostics[policy] = {
                **values,
                **_inside_hangul_unit_diagnostics(matrix, unit_masks),
                "matrix_sha256": _array_sha256(matrix),
            }
        transformed_text = transform_text(source_text, condition)
        transformed_bytes = transformed_text.encode("utf-8")
        metadata = {
            **padded.metadata(),
            "actual_transformed_stream_sha256": _sha256_bytes(
                transformed_bytes
            ),
            "padded_stream_sha256": _sha256_bytes(padded.data),
            "target_mask_sha256": _array_sha256(padded.target_mask),
            "transformed_unicode_codepoints": len(transformed_text),
            "transformed_precomposed_hangul_syllables": (
                count_precomposed_hangul(transformed_text)
            ),
            "conjoining_jamo_codepoints": _conjoining_jamo_counts(
                transformed_text
            ),
            "equal_to_source_text": transformed_text == source_text,
            "patch_diagnostics": patch_diagnostics,
        }
        condition_inputs[condition] = inputs
        condition_masks[condition] = padded.target_mask
        condition_matrices[condition] = matrices
        condition_metadata[condition] = metadata
    if manifest.get("conditions") != condition_metadata:
        raise ValueError("normalization conditions differ from reconstruction")
    return (
        condition_inputs,
        condition_masks,
        condition_matrices,
        denominators,
    )


def paired_values(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    """Return finite paired left-minus-right effects."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired values need equal lengths of at least two")
    effects = [
        float(left_value) - float(right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    if not all(math.isfinite(value) for value in effects):
        raise ValueError("paired effects must be finite")
    return effects


def relative_increases(
    stressed: Sequence[float],
    reference: Sequence[float],
) -> list[float]:
    """Return paired stressed/reference minus one values."""

    if len(stressed) != len(reference) or len(stressed) < 2:
        raise ValueError("relative values need equal lengths of at least two")
    if any(float(value) <= 0 for value in reference):
        raise ValueError("relative reference values must be positive")
    values = [
        float(stress) / float(base) - 1.0
        for stress, base in zip(stressed, reference, strict=True)
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("relative effects must be finite")
    return values


def _effect_summary(
    values: Sequence[float],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "seed_order": list(seeds),
        "paired_values": list(values),
        "numeric_summary": numeric_summary(values),
        "paired_t_95_interval": paired_t_interval(values).to_dict(),
    }


def _validate_manifest_execution(
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("normalization manifest schema mismatch")
    if len(manifest.get("seeds", [])) != len(set(manifest.get("seeds", []))):
        raise ValueError("normalization manifest contains duplicate seeds")
    if len(manifest.get("policies", [])) != len(
        set(manifest.get("policies", []))
    ):
        raise ValueError("normalization manifest contains duplicate policies")
    if not set(seeds) <= set(manifest.get("seeds", [])):
        raise ValueError("normalization manifest does not cover requested seeds")
    if set(manifest.get("policies", [])) != set(POLICIES):
        raise ValueError("normalization manifest does not cover F/C/W")
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("normalization manifest lacks invocation provenance")
    for seed in seeds:
        for policy in POLICIES:
            if not any(
                isinstance(invocation, dict)
                and invocation.get("prepare_only") is False
                and seed in invocation.get("seeds", [])
                and policy in invocation.get("policies", [])
                for invocation in invocations
            ):
                raise ValueError(
                    f"normalization manifest lacks evaluation invocation for "
                    f"seed {seed}/{policy}"
                )


def _validate_primary_manifest(
    normalization_manifest: dict[str, Any],
    training_run_root: Path,
) -> dict[str, Any]:
    primary_path = training_run_root / "manifest.json"
    primary = _read_json(primary_path)
    if primary.get("quick_smoke_only"):
        raise ValueError("normalization cannot use smoke primary evidence")
    if primary.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("normalization primary model spec mismatch")
    if primary.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict():
        raise ValueError("normalization primary optimization spec mismatch")
    source = normalization_manifest["source"]
    if (
        primary.get("limits", {}).get("test")
        != source["requested_byte_limit"]
        or primary.get("streams", {}).get("test", {}).get(
            "selected_stream_sha256"
        )
        != source["primary_stream_sha256"]
    ):
        raise ValueError("normalization source differs from primary test stream")
    return {
        "manifest_sha256": _sha256(primary_path),
        "test_stream_matches": True,
        "model_and_optimization_match": True,
    }


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    if seeds not in (INITIAL_SEEDS, KNOWN_SEEDS):
        raise ValueError(
            "normalization summary requires the preregistered initial 3 or final 5 seeds"
        )
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    training_run_root = Path(args.training_run_root)
    checkpoint_root = Path(args.checkpoint_root)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    _validate_manifest_execution(manifest, seeds)
    expected_design = {
        "conditions": list(CONDITIONS),
        "known_seeds": list(KNOWN_SEEDS),
        "policies": list(POLICIES),
        "terminal_padding_target_masked": True,
        "row_leading_byte_scored": False,
        "oracle_policy_evaluated": False,
        "descriptive_hangul_unit_mask_prefix_causal": False,
        "natural_text_gate": False,
    }
    if manifest.get("design") != expected_design:
        raise ValueError("normalization design mismatch")
    if tuple(manifest["design"]["conditions"]) != CONDITIONS:
        raise ValueError("normalization condition design mismatch")
    if tuple(manifest["design"]["policies"]) != POLICIES:
        raise ValueError("normalization policy design mismatch")
    if manifest["design"]["oracle_policy_evaluated"]:
        raise ValueError("Phase 3 normalization must not evaluate an oracle")
    if manifest["design"]["descriptive_hangul_unit_mask_prefix_causal"]:
        raise ValueError("descriptive Hangul-unit mask must remain non-causal")
    if manifest["design"]["natural_text_gate"]:
        raise ValueError("normalization stress must not become a natural-text gate")
    condition_inputs, condition_masks, condition_matrices, denominators = (
        _reconstruct_conditions(manifest, Path(args.data_root))
    )
    primary_context = _validate_primary_manifest(
        manifest,
        training_run_root,
    )

    checkpoint_lineage: dict[str, dict[str, dict[str, str]]] = {}
    for seed in seeds:
        checkpoint_lineage[str(seed)] = {}
        for policy in POLICIES:
            training_report_path = (
                training_run_root / f"seed-{seed}" / f"{policy}.json"
            )
            checkpoint_path = checkpoint_root / f"seed-{seed}" / f"{policy}.pt"
            if not training_report_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"missing primary evidence for seed {seed}/{policy}"
                )
            training_report = _read_json(training_report_path)
            if (
                training_report.get("seed") != seed
                or training_report.get("policy") != policy
                or training_report.get("parameters") != 19_596_096
                or training_report.get("model_spec")
                != PHASE3_MODEL_SPEC.to_dict()
                or training_report.get("optimization_spec")
                != PHASE3_OPTIMIZATION_SPEC.to_dict()
            ):
                raise ValueError(
                    f"primary report mismatch for seed {seed}/{policy}"
                )
            state_hash = _checkpoint_state_sha256(checkpoint_path)
            if state_hash != training_report.get("trained_state_sha256"):
                raise ValueError(
                    f"primary checkpoint mismatch for seed {seed}/{policy}"
                )
            checkpoint_lineage[str(seed)][policy] = {
                "checkpoint_state_sha256": state_hash,
                "training_report_state_sha256": state_hash,
                "checkpoint_artifact_sha256": _sha256(checkpoint_path),
                "training_report_artifact_sha256": _sha256(
                    training_report_path
                ),
            }

    reports: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    losses: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    counts: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    artifact_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for seed in seeds:
        reports[seed] = {condition: {} for condition in CONDITIONS}
        losses[seed] = {condition: {} for condition in CONDITIONS}
        counts[seed] = {condition: {} for condition in CONDITIONS}
        artifact_hashes[str(seed)] = {condition: {} for condition in CONDITIONS}
        for condition in CONDITIONS:
            expected_condition = manifest["conditions"][condition]
            for policy in POLICIES:
                report_path = (
                    run_root
                    / f"seed-{seed}"
                    / f"{condition}-{policy}.json"
                )
                artifact_path = (
                    artifact_root
                    / f"seed-{seed}"
                    / f"{condition}-{policy}-nll.npz"
                )
                if not report_path.exists() or not artifact_path.exists():
                    raise FileNotFoundError(
                        f"missing normalization result for "
                        f"seed {seed}/{condition}/{policy}"
                    )
                report = _read_json(report_path)
                if (
                    report.get("seed") != seed
                    or report.get("condition") != condition
                    or report.get("policy") != policy
                ):
                    raise ValueError("normalization report identity mismatch")
                expected_fields: dict[str, Any] = {
                    "schema_version": 1,
                    "parameters": 19_596_096,
                    "model_spec": PHASE3_MODEL_SPEC.to_dict(),
                    "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
                    "checkpoint_state_sha256": checkpoint_lineage[str(seed)][
                        policy
                    ]["checkpoint_state_sha256"],
                    "training_report_state_sha256": checkpoint_lineage[
                        str(seed)
                    ][policy]["training_report_state_sha256"],
                    "checkpoint_artifact_sha256": checkpoint_lineage[str(seed)][
                        policy
                    ]["checkpoint_artifact_sha256"],
                    "training_report_artifact_sha256": checkpoint_lineage[
                        str(seed)
                    ][policy]["training_report_artifact_sha256"],
                }
                for key, expected in expected_fields.items():
                    if report.get(key) != expected:
                        raise ValueError(
                            f"normalization report provenance mismatch: {key}"
                        )
                if (
                    report["condition_stream_sha256"]
                    != expected_condition["padded_stream_sha256"]
                    or report["target_mask_sha256"]
                    != expected_condition["target_mask_sha256"]
                    or report["patch_matrix_sha256"]
                    != expected_condition["patch_diagnostics"][policy][
                        "matrix_sha256"
                    ]
                ):
                    raise ValueError("normalization data hash mismatch")
                if (
                    _array_sha256(condition_matrices[condition][policy])
                    != report["patch_matrix_sha256"]
                ):
                    raise ValueError(
                        "normalization matrix differs from reconstruction"
                    )
                diagnostics = report["patch_diagnostics"]
                if diagnostics != expected_condition["patch_diagnostics"][policy]:
                    raise ValueError("normalization patch diagnostics mismatch")
                if (
                    diagnostics["minimum_data_patches"]
                    != PHASE3_MODEL_SPEC.patch_count
                    or diagnostics["maximum_data_patches"]
                    != PHASE3_MODEL_SPEC.patch_count
                ):
                    raise ValueError("normalization exact-rate invariant failed")
                with np.load(artifact_path) as archive:
                    if set(archive.files) != {
                        "sequence_nll_nats",
                        "sequence_target_counts",
                    }:
                        raise ValueError("unexpected normalization loss keys")
                    local_losses = archive["sequence_nll_nats"].astype(
                        np.float64
                    )
                    local_counts = archive["sequence_target_counts"].astype(
                        np.int64
                    )
                if (
                    local_losses.ndim != 1
                    or local_counts.shape != local_losses.shape
                    or len(local_losses)
                    != expected_condition["sequence_count"]
                    or not np.isfinite(local_losses).all()
                    or np.any(local_losses < 0)
                    or np.any(local_counts < 0)
                ):
                    raise ValueError("invalid normalization sequence artifacts")
                expected_counts = condition_masks[condition].sum(axis=1).astype(
                    np.int64
                )
                if not np.array_equal(local_counts, expected_counts):
                    raise ValueError("normalization target counts differ from mask")
                evaluation = report["evaluation"]
                predicted = int(local_counts.sum())
                total_nll = float(local_losses.sum())
                total_bits = total_nll / math.log(2)
                expected_metrics = {
                    "predicted_bytes": predicted,
                    "total_nll_nats": total_nll,
                    "bpb": total_bits / predicted,
                    "scored_bits_per_source_utf8_byte": (
                        total_bits / denominators["utf8_bytes"]
                    ),
                    "scored_bits_per_source_unicode_codepoint": (
                        total_bits / denominators["unicode_codepoints"]
                    ),
                    "scored_bits_per_source_precomposed_hangul_syllable": (
                        total_bits
                        / denominators["precomposed_hangul_syllables"]
                    ),
                }
                if evaluation.get("examples") != len(condition_inputs[condition]):
                    raise ValueError("normalization report example mismatch")
                for key, expected in expected_metrics.items():
                    actual = evaluation.get(key)
                    if key == "predicted_bytes":
                        matches = actual == expected
                    else:
                        matches = isinstance(actual, (int, float)) and math.isclose(
                            float(actual), float(expected), abs_tol=1e-7
                        )
                    if not matches:
                        raise ValueError(
                            f"normalization report/loss mismatch: {key}"
                        )
                reports[seed][condition][policy] = report
                losses[seed][condition][policy] = local_losses
                counts[seed][condition][policy] = local_counts
                artifact_hashes[str(seed)][condition][policy] = _sha256(
                    artifact_path
                )
            reference_counts = counts[seed][condition][F]
            if any(
                not np.array_equal(reference_counts, counts[seed][condition][policy])
                for policy in (C, W)
            ):
                raise ValueError("policy target masks differ within a condition")

    def metric_values(condition: str, policy: str, metric: str) -> list[float]:
        return [
            float(reports[seed][condition][policy]["evaluation"][metric])
            for seed in seeds
        ]

    quality = {
        condition: {
            policy: {
                metric: numeric_summary(metric_values(condition, policy, metric))
                for metric in METRICS
            }
            for policy in POLICIES
        }
        for condition in CONDITIONS
    }

    policy_contrasts: dict[str, Any] = {}
    for condition in CONDITIONS:
        policy_contrasts[condition] = {}
        for contrast_name, (left, right) in POLICY_CONTRASTS.items():
            policy_contrasts[condition][contrast_name] = {
                "left_policy": left,
                "right_policy": right,
                "difference_direction": "left_minus_right; negative favors left",
                "metrics": {
                    metric: _effect_summary(
                        paired_values(
                            metric_values(condition, left, metric),
                            metric_values(condition, right, metric),
                        ),
                        seeds,
                    )
                    for metric in (
                        "bpb",
                        "scored_bits_per_source_unicode_codepoint",
                    )
                },
            }

    stress: dict[str, Any] = {}
    for policy in POLICIES:
        nfd_bpb = metric_values("nfd", policy, "bpb")
        nfc_bpb = metric_values("nfc", policy, "bpb")
        metric = "scored_bits_per_source_unicode_codepoint"
        nfd_source = metric_values("nfd", policy, metric)
        nfc_source = metric_values("nfc", policy, metric)
        stress[policy] = {
            "nfd_minus_nfc_bpb": _effect_summary(
                paired_values(nfd_bpb, nfc_bpb),
                seeds,
            ),
            "nfd_minus_nfc_scored_bits_per_source_codepoint": _effect_summary(
                paired_values(nfd_source, nfc_source),
                seeds,
            ),
            "nfd_relative_increase_scored_bits_per_source_codepoint": (
                _effect_summary(
                    relative_increases(nfd_source, nfc_source),
                    seeds,
                )
            ),
        }

    matrix_hashes = {
        condition: {
            policy: sorted(
                {
                    reports[seed][condition][policy]["patch_matrix_sha256"]
                    for seed in seeds
                }
            )
            for policy in POLICIES
        }
        for condition in CONDITIONS
    }
    if any(
        len(matrix_hashes[condition][policy]) != 1
        for condition in CONDITIONS
        for policy in POLICIES
    ):
        raise ValueError("normalization matrices are not seed-independent")

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "policies": list(POLICIES),
        "run_manifest": manifest,
        "primary_context": primary_context,
        "source": manifest["source"],
        "condition_geometry": manifest["conditions"],
        "quality": quality,
        "policy_contrasts": policy_contrasts,
        "normalization_stress": stress,
        "integrity": {
            "all_integrity_checks_pass": True,
            "source_and_conditions_match_independent_reconstruction": True,
            "primary_test_stream_matches_normalization_source": True,
            "same_source_denominators_for_both_conditions": True,
            "terminal_padding_targets_excluded": True,
            "all_policies_exactly_86_patches": True,
            "all_checkpoint_state_hashes_match": True,
            "all_checkpoint_artifact_hashes_match": True,
            "all_training_report_artifact_hashes_match": True,
            "policy_target_masks_identical_within_condition": True,
            "structural_matrices_seed_independent": True,
            "matrix_sha256": matrix_hashes,
            "loss_artifact_sha256": artifact_hashes,
            "primary_checkpoint_lineage": checkpoint_lineage,
            "raw_or_normalized_text_promoted": False,
        },
        "decision_gate": None,
        "interpretation_guardrail": (
            "Synthetic NFD is reported separately from natural-text gates; "
            "scored bits/source-codepoint omits each row-leading target and "
            "is not a complete lossless-compression codelength."
        ),
    }
    _write_json(Path(args.output), output)
    print(f"wrote Phase 3 normalization summary to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase3-normalization")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-normalization",
    )
    parser.add_argument(
        "--data-root", default="data/processed/hplt3-korean-phase3"
    )
    parser.add_argument("--training-run-root", default="runs/phase3")
    parser.add_argument("--checkpoint-root", default="artifacts/phase3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase3-normalization/summary.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
