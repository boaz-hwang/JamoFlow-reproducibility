#!/usr/bin/env python3
"""Evaluate Phase 3 F/C/W checkpoints on paired HPLT3 NFC/NFD streams."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import (
    evaluate_main_model_masked,
    resolve_device,
)
from jamoflow.normalization import (
    count_precomposed_hangul,
    oracle_hangul_unit_boundary_mask,
    padded_normalization_stream,
    transform_text,
)
from jamoflow.phase1 import patch_boundaries_from_lengths, stream_arrays
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
POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
KNOWN_SEEDS = (1729, 2718, 31415, 57721, 65537)
DEFAULT_SEEDS = KNOWN_SEEDS[:3]
DEFAULT_BYTE_LIMIT = 16_000_000
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
_MANIFEST_INVARIANTS = (
    "schema_version",
    "design",
    "source",
    "conditions",
    "global_max_position_embeddings",
    "model_spec",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def strict_decodable_prefix(data: bytes) -> tuple[str, int]:
    """Decode a valid stream, permitting only a truncated final codepoint."""

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


def merge_normalization_manifest(
    existing: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve invariant data provenance across seed invocations."""

    candidate = deepcopy(dict(current))
    invocation = {
        key: deepcopy(candidate[key])
        for key in (
            "created_at",
            "git_commit",
            "device",
            "platform",
            "versions",
            "seeds",
            "policies",
            "prepare_only",
        )
    }
    if existing is None:
        candidate["invocations"] = [invocation]
        return candidate
    merged = deepcopy(dict(existing))
    for key in _MANIFEST_INVARIANTS:
        if merged.get(key) != candidate.get(key):
            raise ValueError(f"normalization manifest invariant changed: {key}")
    invocations = list(merged.get("invocations", []))
    if not invocations:
        raise ValueError("normalization manifest lacks invocation provenance")
    invocations.append(invocation)
    merged["invocations"] = invocations
    for key in ("seeds", "policies"):
        values = list(merged[key])
        values.extend(value for value in candidate[key] if value not in values)
        merged[key] = values
    merged["updated_at"] = candidate["created_at"]
    return merged


def _load_verified_model(
    seed: int,
    policy: str,
    training_run_root: Path,
    checkpoint_root: Path,
) -> tuple[Any, dict[str, str]]:
    report_path = training_run_root / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = checkpoint_root / f"seed-{seed}" / f"{policy}.pt"
    if not report_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing Phase 3 checkpoint/report for seed {seed}/{policy}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("seed") != seed or report.get("policy") != policy:
        raise ValueError("Phase 3 checkpoint report identity mismatch")
    if (
        report.get("parameters") != 19_596_096
        or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
    ):
        raise ValueError("Phase 3 checkpoint model spec mismatch")
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state_hash = _state_dict_sha256(model)
    training_state_hash = report.get("trained_state_sha256")
    if (
        not isinstance(training_state_hash, str)
        or state_hash != training_state_hash
    ):
        raise ValueError("Phase 3 checkpoint state hash mismatch")
    return model, {
        "checkpoint_state_sha256": state_hash,
        "training_report_state_sha256": training_state_hash,
        "checkpoint_artifact_sha256": _sha256_file(checkpoint_path),
        "training_report_artifact_sha256": _sha256_file(report_path),
    }


def _validate_completed_result(
    report_path: Path,
    artifact_path: Path,
    *,
    seed: int,
    condition: str,
    policy: str,
    lineage: dict[str, str],
    expected_condition: dict[str, Any],
    expected_target_counts: np.ndarray,
    source_denominators: dict[str, int],
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "condition": condition,
        "policy": policy,
        "parameters": 19_596_096,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "condition_stream_sha256": expected_condition[
            "padded_stream_sha256"
        ],
        "target_mask_sha256": expected_condition["target_mask_sha256"],
        "patch_matrix_sha256": expected_condition["patch_diagnostics"][policy][
            "matrix_sha256"
        ],
        "patch_diagnostics": expected_condition["patch_diagnostics"][policy],
        **lineage,
    }
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(
                f"stale normalization result ({key}): "
                f"{seed}/{condition}/{policy}; rerun with --force"
            )
    with np.load(artifact_path) as archive:
        if set(archive.files) != {
            "sequence_nll_nats",
            "sequence_target_counts",
        }:
            raise ValueError(f"unexpected normalization loss keys: {artifact_path}")
        losses = archive["sequence_nll_nats"].astype(np.float64)
        target_counts = archive["sequence_target_counts"].astype(np.int64)
    if (
        losses.shape != expected_target_counts.shape
        or target_counts.shape != expected_target_counts.shape
        or not np.array_equal(target_counts, expected_target_counts)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError(f"invalid normalization artifact: {artifact_path}")
    evaluation = report.get("evaluation", {})
    predicted_bytes = int(expected_target_counts.sum())
    total_nll = float(losses.sum())
    total_bits = total_nll / math.log(2)
    expected_metrics = {
        "predicted_bytes": predicted_bytes,
        "total_nll_nats": total_nll,
        "bpb": total_bits / predicted_bytes,
        "scored_bits_per_source_utf8_byte": (
            total_bits / source_denominators["utf8_bytes"]
        ),
        "scored_bits_per_source_unicode_codepoint": (
            total_bits / source_denominators["unicode_codepoints"]
        ),
        "scored_bits_per_source_precomposed_hangul_syllable": (
            total_bits / source_denominators["precomposed_hangul_syllables"]
        ),
    }
    if evaluation.get("examples") != len(expected_target_counts):
        raise ValueError(f"normalization example count mismatch: {report_path}")
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
                f"normalization metric mismatch ({key}): {report_path}"
            )


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    policies = tuple(args.policies)
    if (
        len(seeds) < 1
        or len(set(seeds)) != len(seeds)
        or set(seeds) - set(KNOWN_SEEDS)
    ):
        raise ValueError("normalization needs preregistered Phase 3 seeds")
    unknown = set(policies) - set(POLICIES)
    if unknown or not policies or len(set(policies)) != len(policies):
        raise ValueError(f"unsupported normalization policies: {sorted(unknown)}")
    if args.batch_size <= 0:
        raise ValueError("normalization batch size must be positive")

    device = resolve_device(args.device)
    training_run_root = Path(args.training_run_root)
    checkpoint_root = Path(args.checkpoint_root)
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.data_root) / "ko.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=args.byte_limit,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    source_text, discarded_tail = strict_decodable_prefix(source_stream.data)
    source_bytes = source_text.encode("utf-8")
    source_denominators = {
        "utf8_bytes": len(source_bytes),
        "unicode_codepoints": len(source_text),
        "precomposed_hangul_syllables": count_precomposed_hangul(source_text),
    }
    if not all(source_denominators.values()):
        raise ValueError("normalization source denominators must be positive")

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
        diagnostics: dict[str, Any] = {}
        for policy, matrix in matrices.items():
            validate_padded_patch_matrix(
                matrix,
                PHASE3_MODEL_SPEC.sequence_length,
            )
            values = variable_patch_diagnostics(matrix, boundaries).to_dict()
            if (
                values["minimum_data_patches"] != PHASE3_MODEL_SPEC.patch_count
                or values["maximum_data_patches"]
                != PHASE3_MODEL_SPEC.patch_count
            ):
                raise AssertionError("normalization patch rate is not exact")
            diagnostics[policy] = {
                **values,
                **_inside_hangul_unit_diagnostics(matrix, unit_masks),
                "matrix_sha256": _array_sha256(matrix),
            }
        transformed_text = transform_text(source_text, condition)
        transformed_bytes = transformed_text.encode("utf-8")
        if len(transformed_bytes) != padded.actual_transformed_bytes:
            raise AssertionError("normalization transform length changed")
        condition_inputs[condition] = inputs
        condition_masks[condition] = padded.target_mask
        condition_matrices[condition] = matrices
        condition_metadata[condition] = {
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
            "patch_diagnostics": diagnostics,
        }
        print(
            f"{condition}: {len(padded.data):,} padded bytes, "
            f"{padded.sequence_count:,} rows, "
            f"{padded.scored_actual_target_bytes:,} scored targets",
            flush=True,
        )

    current_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "seeds": list(seeds),
        "policies": list(policies),
        "prepare_only": bool(args.prepare_only),
        "design": {
            "conditions": list(CONDITIONS),
            "known_seeds": list(KNOWN_SEEDS),
            "policies": list(POLICIES),
            "terminal_padding_target_masked": True,
            "row_leading_byte_scored": False,
            "oracle_policy_evaluated": False,
            "descriptive_hangul_unit_mask_prefix_causal": False,
            "natural_text_gate": False,
        },
        "source": {
            "source_artifact": {
                "filename": "ko.jsonl",
                "bytes": source_path.stat().st_size,
                "sha256": _sha256_file(source_path),
            },
            "requested_byte_limit": args.byte_limit,
            "primary_stream": source_stream.metadata(),
            "primary_stream_sha256": _sha256_bytes(source_stream.data),
            "strict_source_utf8_sha256": _sha256_bytes(source_bytes),
            "strict_source_bytes": len(source_bytes),
            "discarded_incomplete_terminal_bytes": discarded_tail,
            "denominators": source_denominators,
        },
        "conditions": condition_metadata,
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
    }
    manifest_path = run_root / "manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    manifest = merge_normalization_manifest(existing, current_manifest)
    _write_json(manifest_path, manifest)
    if args.prepare_only:
        print("normalization geometry prepared; checkpoints were not evaluated")
        return 0

    for seed in seeds:
        for policy in policies:
            model, lineage = _load_verified_model(
                seed,
                policy,
                training_run_root,
                checkpoint_root,
            )
            pending: list[str] = []
            for condition in CONDITIONS:
                report_path = (
                    run_root / f"seed-{seed}" / f"{condition}-{policy}.json"
                )
                artifact_path = (
                    artifact_root
                    / f"seed-{seed}"
                    / f"{condition}-{policy}-nll.npz"
                )
                if args.force or not report_path.exists() or not artifact_path.exists():
                    pending.append(condition)
                    continue
                expected_target_counts = condition_masks[condition].sum(
                    axis=1
                ).astype(np.uint16)
                _validate_completed_result(
                    report_path,
                    artifact_path,
                    seed=seed,
                    condition=condition,
                    policy=policy,
                    lineage=lineage,
                    expected_condition=condition_metadata[condition],
                    expected_target_counts=expected_target_counts,
                    source_denominators=source_denominators,
                )
            if not pending:
                print(f"seed {seed}/{policy}: normalization already complete")
                _release_model(model, device)
                continue
            for condition in pending:
                print(
                    f"seed {seed}/{policy}/{condition}: evaluating",
                    flush=True,
                )
                evaluation, sequence_nll = evaluate_main_model_masked(
                    model,
                    condition_inputs[condition],
                    condition_matrices[condition][policy],
                    condition_masks[condition],
                    device,
                    batch_size=args.batch_size,
                    return_sequence_nll=True,
                )
                if sequence_nll is None:
                    raise AssertionError("normalization sequence losses missing")
                target_counts = condition_masks[condition].sum(axis=1).astype(
                    np.uint16
                )
                if evaluation.predicted_bytes != int(target_counts.sum()):
                    raise AssertionError("normalization target counts changed")
                total_nll = evaluation.nll_nats * evaluation.predicted_bytes
                total_bits = total_nll / math.log(2)
                _save_npz(
                    artifact_root
                    / f"seed-{seed}"
                    / f"{condition}-{policy}-nll.npz",
                    sequence_nll_nats=sequence_nll,
                    sequence_target_counts=target_counts,
                )
                _write_json(
                    run_root
                    / f"seed-{seed}"
                    / f"{condition}-{policy}.json",
                    {
                        "schema_version": 1,
                        "seed": seed,
                        "condition": condition,
                        "policy": policy,
                        "parameters": parameter_count(model),
                        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
                        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
                        **lineage,
                        "condition_stream_sha256": condition_metadata[
                            condition
                        ]["padded_stream_sha256"],
                        "target_mask_sha256": condition_metadata[condition][
                            "target_mask_sha256"
                        ],
                        "patch_matrix_sha256": condition_metadata[condition][
                            "patch_diagnostics"
                        ][policy]["matrix_sha256"],
                        "patch_diagnostics": condition_metadata[condition][
                            "patch_diagnostics"
                        ][policy],
                        "evaluation": {
                            **evaluation.to_dict(),
                            "total_nll_nats": total_nll,
                            "scored_bits_per_source_utf8_byte": (
                                total_bits / source_denominators["utf8_bytes"]
                            ),
                            "scored_bits_per_source_unicode_codepoint": (
                                total_bits
                                / source_denominators["unicode_codepoints"]
                            ),
                            "scored_bits_per_source_precomposed_hangul_syllable": (
                                total_bits
                                / source_denominators[
                                    "precomposed_hangul_syllables"
                                ]
                            ),
                        },
                    },
                )
            _release_model(model, device)
    print(f"completed Phase 3 normalization under {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--training-run-root", default="runs/phase3")
    parser.add_argument("--checkpoint-root", default="artifacts/phase3")
    parser.add_argument("--run-root", default="runs/phase3-normalization")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-normalization",
    )
    parser.add_argument("--byte-limit", type=int, default=DEFAULT_BYTE_LIMIT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
