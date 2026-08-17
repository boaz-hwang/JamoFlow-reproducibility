#!/usr/bin/env python3
"""Evaluate Korean normalization robustness and a non-causal Hangul-unit oracle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    research_versions,
)
from jamoflow.neural_training import evaluate_main_model, resolve_device
from jamoflow.normalization import (
    CONDITIONS,
    count_precomposed_hangul,
    oracle_hangul_unit_boundary_mask,
    represented_source_prefix_length,
    transform_text,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_controls import aligned_pack_stream
from jamoflow.phase2_patching import (
    causal_codepoint_grid_boundaries,
    compact_whitespace_mask,
    padded_hf_patch_matrix,
    structural_patch_matrices,
    variable_patch_diagnostics,
)


SEEDS = (1729, 2718, 31415, 57721, 65537)
POLICIES = (
    "fixed_byte_6_rate43",
    "causal_codepoint_grid_rate43",
    "causal_whitespace_grid_rate43",
    "causal_codepoint_grid_rate28",
    "oracle_hangul_unit_grid_rate28",
)
ORACLE_PATCH_COUNT = 28
GLOBAL_POSITION_LIMIT = DEFAULT_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_state(model: Any, path: Path) -> Any:
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def _release_models(models: dict[str, Any], device: str) -> None:
    for model in models.values():
        model.to("cpu")
    models.clear()
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _inside_unit_rate(matrix: np.ndarray, unit_masks: np.ndarray) -> dict[str, Any]:
    internal = 0
    total = 0
    for row_index, row in enumerate(matrix):
        positive = row[1:][row[1:] > 0].astype(np.int64)
        boundaries = np.cumsum(positive)[:-1]
        internal += int((unit_masks[row_index, boundaries] == 0).sum())
        total += len(boundaries)
    return {
        "inside_oracle_hangul_unit_boundaries": internal,
        "total_noninitial_boundaries": total,
        "inside_oracle_hangul_unit_boundary_rate": (
            internal / total if total else math.nan
        ),
    }


def _condition_data(
    source_text: str,
    condition: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    transformed_text = transform_text(source_text, condition)
    transformed_bytes = transformed_text.encode("utf-8")
    prepack_unit_mask = oracle_hangul_unit_boundary_mask(transformed_bytes)
    pack_candidates = np.zeros(len(transformed_bytes) + 1, dtype=np.uint8)
    pack_candidates[:-1] = prepack_unit_mask
    pack_candidates[-1] = 1
    packed = aligned_pack_stream(
        transformed_bytes,
        end_boundary_mask=pack_candidates,
        maximum_padding=8,
    )
    inputs, masks = stream_arrays(
        packed.data,
        packed.codepoint_boundaries,
        packed.sequence_length,
    )
    whitespace = compact_whitespace_mask(packed.data).reshape(inputs.shape)
    structural = structural_patch_matrices(masks, whitespace)
    unit_masks = oracle_hangul_unit_boundary_mask(packed.data).reshape(inputs.shape)
    codepoint_rate28 = padded_hf_patch_matrix(
        [
            causal_codepoint_grid_boundaries(
                mask,
                ORACLE_PATCH_COUNT,
            )
            for mask in masks
        ],
        DEFAULT_MODEL_SPEC.sequence_length,
    )
    oracle_rate28 = padded_hf_patch_matrix(
        [
            causal_codepoint_grid_boundaries(
                unit_mask,
                ORACLE_PATCH_COUNT,
            )
            for unit_mask in unit_masks
        ],
        DEFAULT_MODEL_SPEC.sequence_length,
    )
    matrices = {
        "fixed_byte_6_rate43": structural["fixed_byte_6"],
        "causal_codepoint_grid_rate43": structural["causal_codepoint_grid"],
        "causal_whitespace_grid_rate43": structural["causal_eojeol_grid"],
        "causal_codepoint_grid_rate28": codepoint_rate28,
        "oracle_hangul_unit_grid_rate28": oracle_rate28,
    }
    represented_source_codepoints = represented_source_prefix_length(
        source_text,
        condition,
        packed.raw_bytes_used,
    )
    represented_source_text = source_text[:represented_source_codepoints]
    diagnostics = {
        "condition": condition,
        "source_codepoints_total": len(source_text),
        "represented_source_codepoints": represented_source_codepoints,
        "represented_precomposed_hangul_syllables": count_precomposed_hangul(
            represented_source_text
        ),
        "transformed_bytes_before_packing": len(transformed_bytes),
        "packed_stream_sha256": _sha256_bytes(packed.data),
        "packing": packed.metadata(),
        "packing_end_alignment": "oracle Hangul-unit candidates",
        "packing_maximum_newline_padding_per_row": 8,
        "nfc_equal_to_source": (
            condition == "nfc" and transformed_text == source_text
        ),
        "policies": {
            policy: {
                **variable_patch_diagnostics(matrix, masks).to_dict(),
                **_inside_unit_rate(matrix, unit_masks),
            }
            for policy, matrix in matrices.items()
        },
        "oracle_equals_codepoint_matrix": bool(
            np.array_equal(
                matrices["oracle_hangul_unit_grid_rate28"],
                matrices["causal_codepoint_grid_rate28"],
            )
        ),
    }
    for policy in (
        "causal_codepoint_grid_rate28",
        "oracle_hangul_unit_grid_rate28",
    ):
        policy_diagnostics = diagnostics["policies"][policy]
        if (
            policy_diagnostics["minimum_data_patches"] != ORACLE_PATCH_COUNT
            or policy_diagnostics["maximum_data_patches"] != ORACLE_PATCH_COUNT
        ):
            raise AssertionError(f"{policy} is not exact rate {ORACLE_PATCH_COUNT}")
    if diagnostics["policies"]["oracle_hangul_unit_grid_rate28"][
        "inside_oracle_hangul_unit_boundaries"
    ]:
        raise AssertionError("oracle policy split a Hangul unit")
    return inputs, matrices, diagnostics


def run(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    primary_artifact_root = Path(args.primary_artifact_root)
    control_artifact_root = Path(args.control_artifact_root)
    source_stream = build_neural_stream(
        Path(args.data_root) / "ko.jsonl",
        language="ko",
        split="test",
        byte_limit=1_000_000,
        sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
    )
    source_text = source_stream.data.decode("utf-8", errors="ignore")
    strict_prefix_bytes = len(source_text.encode("utf-8"))
    condition_inputs: dict[str, np.ndarray] = {}
    condition_matrices: dict[str, dict[str, np.ndarray]] = {}
    condition_diagnostics: dict[str, Any] = {}
    for condition in CONDITIONS:
        inputs, matrices, diagnostics = _condition_data(source_text, condition)
        condition_inputs[condition] = inputs
        condition_matrices[condition] = matrices
        condition_diagnostics[condition] = diagnostics
        print(
            f"condition {condition}: {len(inputs):,} sequences, "
            f"{diagnostics['packing']['inserted_fraction_of_packed_bytes']:.4%} padding",
            flush=True,
        )

    if not condition_diagnostics["nfc"]["oracle_equals_codepoint_matrix"]:
        raise AssertionError("NFC oracle and codepoint matrices must be identical")

    evaluations: dict[str, Any] = {}
    for seed in SEEDS:
        primary_seed = primary_artifact_root / f"seed-{seed}"
        control_seed = control_artifact_root / f"mechanism-seed-{seed}"
        models = {
            "fixed_byte_6": _load_state(
                build_main_model(
                    seed=seed,
                    global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
                ),
                primary_seed / "fixed_byte_6.pt",
            ).to(device).eval(),
            "causal_codepoint_grid": _load_state(
                build_main_model(
                    seed=seed,
                    global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
                ),
                primary_seed / "causal_codepoint_grid.pt",
            ).to(device).eval(),
            "causal_whitespace_grid": _load_state(
                build_main_model(
                    seed=seed,
                    global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
                ),
                control_seed / "causal_whitespace_grid.pt",
            ).to(device).eval(),
        }
        evaluations[str(seed)] = {}
        for condition in CONDITIONS:
            inputs = condition_inputs[condition]
            metadata = condition_diagnostics[condition]
            source_codepoints = metadata["represented_source_codepoints"]
            source_hangul = metadata["represented_precomposed_hangul_syllables"]
            evaluations[str(seed)][condition] = {}
            for policy in POLICIES:
                model_key = (
                    "causal_codepoint_grid"
                    if policy
                    in (
                        "causal_codepoint_grid_rate43",
                        "causal_codepoint_grid_rate28",
                        "oracle_hangul_unit_grid_rate28",
                    )
                    else "fixed_byte_6"
                    if policy == "fixed_byte_6_rate43"
                    else "causal_whitespace_grid"
                )
                evaluation, _ = evaluate_main_model(
                    models[model_key],
                    inputs,
                    condition_matrices[condition][policy],
                    device,
                    batch_size=64,
                    return_sequence_nll=False,
                )
                total_nll = evaluation.nll_nats * evaluation.predicted_bytes
                values = evaluation.to_dict()
                values.update(
                    {
                        "total_nll_nats": total_nll,
                        "bits_per_represented_source_codepoint": (
                            total_nll / math.log(2) / source_codepoints
                        ),
                        "bits_per_represented_source_hangul_syllable": (
                            total_nll / math.log(2) / source_hangul
                            if source_hangul
                            else math.nan
                        ),
                    }
                )
                evaluations[str(seed)][condition][policy] = values
            print(f"seed {seed}/{condition}: evaluated", flush=True)
        _release_models(models, device)

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "NFC-trained compact Korean BLT checkpoints evaluated without "
            "retraining; oracle Hangul-unit policy is non-causal"
        ),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "policies": list(POLICIES),
        "source": {
            "primary_selected_bytes": len(source_stream.data),
            "strict_decodable_prefix_bytes": strict_prefix_bytes,
            "discarded_incomplete_tail_bytes": (
                len(source_stream.data) - strict_prefix_bytes
            ),
            "source_text_sha256": _sha256_bytes(source_text.encode("utf-8")),
        },
        "condition_diagnostics": condition_diagnostics,
        "evaluations": evaluations,
        "causality_guardrail": {
            "oracle_hangul_unit_grid_is_prefix_causal": False,
            "reason": (
                "After L+V, optional T cannot be known without observing a "
                "future codepoint; candidate boundaries use full-sequence lookahead."
            ),
        },
    }
    _write_json(run_root / "normalization-results.json", output)
    print(f"wrote normalization run to {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--primary-artifact-root", default="artifacts/phase2")
    parser.add_argument(
        "--control-artifact-root",
        default="artifacts/phase2-controls",
    )
    parser.add_argument("--run-root", default="runs/phase2-normalization")
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
