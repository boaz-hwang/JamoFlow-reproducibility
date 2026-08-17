#!/usr/bin/env python3
"""Evaluate Phase 2 checkpoints on a read-only private Markdown test split.

The serialized result contains corpus-level aggregates only. Source paths,
filenames, content, hashes, and per-document/per-sequence values are never
written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from jamoflow.ecological import (
    build_private_markdown_test_stream,
    stratum_bpb,
    whitespace_grid_patch_matrix,
)
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import evaluate_main_model, resolve_device
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import (
    hierarchical_paired_bootstrap,
    numeric_summary,
    paired_t_interval,
)
from jamoflow.phase2_analysis import korean_test_strata
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    structural_patch_matrices,
    variable_patch_diagnostics,
)


SEEDS = (1729, 2718, 31415, 57721, 65537)
POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
CONTRASTS = {
    "whitespace_minus_codepoint": (
        "causal_whitespace_grid",
        "causal_codepoint_grid",
    ),
    "codepoint_minus_fixed": (
        "causal_codepoint_grid",
        "fixed_byte_6",
    ),
    "exploratory_whitespace_minus_fixed": (
        "causal_whitespace_grid",
        "fixed_byte_6",
    ),
}
CONTRAST_STATUS = {
    "whitespace_minus_codepoint": "preregistered_primary_ecological_contrast",
    "codepoint_minus_fixed": "preregistered_diagnostic",
    "exploratory_whitespace_minus_fixed": (
        "post_result_contextual_contrast_added_after_initial_ecological_run"
    ),
}
GLOBAL_POSITION_LIMIT = DEFAULT_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _checkpoint_path(
    seed: int,
    policy: str,
    primary_root: Path,
    control_root: Path,
) -> Path:
    if policy == "causal_whitespace_grid":
        return (
            control_root
            / f"mechanism-seed-{seed}"
            / "causal_whitespace_grid.pt"
        )
    return primary_root / f"seed-{seed}" / f"{policy}.pt"


def _contrast_summaries(
    bpb: dict[int, dict[str, float]],
    losses: dict[int, dict[str, np.ndarray]],
    repetitions: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        effects = [bpb[seed][left] - bpb[seed][right] for seed in SEEDS]
        sequence_differences = [
            losses[seed][left] - losses[seed][right] for seed in SEEDS
        ]
        reconstructed = [
            float(values.mean()) / (255 * math.log(2))
            for values in sequence_differences
        ]
        if any(
            not math.isclose(effect, rebuilt, abs_tol=2e-5)
            for effect, rebuilt in zip(effects, reconstructed, strict=True)
        ):
            raise ValueError(f"aggregate/sequence mismatch in {name}")
        output[name] = {
            "analysis_status": CONTRAST_STATUS[name],
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(SEEDS),
            "paired_differences_bpb": effects,
            "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            "hierarchical_bootstrap_95_interval": hierarchical_paired_bootstrap(
                sequence_differences,
                targets_per_sequence=255,
                repetitions=repetitions,
                seed=20_260_810 + index,
            ).to_dict(),
        }
    return output


def _stratum_summaries(
    stream_data: bytes,
    boundary_masks: np.ndarray,
    losses: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    strata, metadata = korean_test_strata(stream_data, boundary_masks)
    contrasts: dict[str, Any] = {}
    for contrast_name, (left, right) in CONTRASTS.items():
        values: dict[str, Any] = {}
        for name, stratum in strata.items():
            count = int(stratum.selected.sum())
            base = stratum.metadata()
            if count < 50:
                values[name] = {
                    **base,
                    "status": "descriptive_only_below_50_windows",
                }
                continue
            effects = [
                stratum_bpb(
                    losses[seed][left] - losses[seed][right],
                    stratum.selected,
                    targets_per_sequence=255,
                )
                for seed in SEEDS
            ]
            values[name] = {
                **base,
                "status": "estimated",
                "seed_order": list(SEEDS),
                "paired_seed_effects_bpb": effects,
                "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            }
        contrasts[contrast_name] = values
    return {
        "definitions_and_counts": metadata,
        "minimum_interpreted_windows": 50,
        "contrasts": contrasts,
    }


def run(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    stream = build_private_markdown_test_stream(
        args.vault_root,
        sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    structural = structural_patch_matrices(boundaries, whitespace)
    matrices = {
        "fixed_byte_6": structural["fixed_byte_6"],
        "causal_codepoint_grid": structural["causal_codepoint_grid"],
        "causal_whitespace_grid": whitespace_grid_patch_matrix(
            boundaries,
            whitespace,
            patch_count=DEFAULT_MODEL_SPEC.patch_count,
        ),
    }
    diagnostics = {
        policy: variable_patch_diagnostics(matrix, boundaries).to_dict()
        for policy, matrix in matrices.items()
    }
    exact_rate = all(
        values["minimum_data_patches"] == DEFAULT_MODEL_SPEC.patch_count
        and values["maximum_data_patches"] == DEFAULT_MODEL_SPEC.patch_count
        for values in diagnostics.values()
    )
    if not exact_rate:
        raise AssertionError("ecological patch-rate invariant failed")

    print(
        f"private aggregate: {stream.valid_test_records} valid test records, "
        f"{stream.sequence_count} sequences; device={device}",
        flush=True,
    )
    primary_root = Path(args.primary_artifact_root)
    control_root = Path(args.control_artifact_root)
    bpb: dict[int, dict[str, float]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    timing: dict[int, dict[str, float]] = {}
    parameters: set[int] = set()
    for seed in SEEDS:
        bpb[seed] = {}
        losses[seed] = {}
        timing[seed] = {}
        for policy in POLICIES:
            checkpoint = _checkpoint_path(seed, policy, primary_root, control_root)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = build_main_model(
                seed=seed,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            )
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            parameters.add(parameter_count(model))
            evaluation, sequence_nll = evaluate_main_model(
                model,
                inputs,
                matrices[policy],
                device,
                batch_size=args.batch_size,
                return_sequence_nll=True,
            )
            if sequence_nll is None:
                raise AssertionError("sequence NLL was not returned")
            bpb[seed][policy] = evaluation.bpb
            losses[seed][policy] = sequence_nll.astype(np.float64)
            timing[seed][policy] = evaluation.elapsed_seconds
            print(
                f"seed {seed}/{policy}: BPB={evaluation.bpb:.6f}",
                flush=True,
            )
            _release_model(model, device)
    if len(parameters) != 1:
        raise AssertionError("policy parameter counts differ")

    contrasts = _contrast_summaries(bpb, losses, args.bootstrap_repetitions)
    ecological = contrasts["whitespace_minus_codepoint"]
    effects = ecological["paired_differences_bpb"]
    mean_ok = float(np.mean(effects)) <= 0.02
    every_seed_ok = all(effect <= 0.02 for effect in effects)
    gate_pass = bool(mean_ok and every_seed_ok and exact_rate)
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 2c read-only private ecological diagnostic",
        "source_label": "user-authorized private Korean Markdown convenience sample",
        "privacy": {
            "read_only": True,
            "raw_text_serialized": False,
            "paths_or_filenames_serialized": False,
            "record_or_sequence_metrics_serialized": False,
            "private_content_hash_serialized": False,
            "primary_evidence": False,
        },
        "environment": {
            "device": device,
            "platform": platform.platform(),
            "versions": research_versions(),
        },
        "design": {
            "seeds": list(SEEDS),
            "policies": list(POLICIES),
            "model_parameters": next(iter(parameters)),
            "sequence_length": DEFAULT_MODEL_SPEC.sequence_length,
            "targets_per_sequence": 255,
            "patch_count": DEFAULT_MODEL_SPEC.patch_count,
            "test_partition": "existing deterministic content-hash test split",
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "corpus_aggregate": stream.public_metadata(),
        "patch_integrity": {
            "all_policies_exactly_43": exact_rate,
            "by_policy": diagnostics,
        },
        "quality": {
            policy: {
                "seed_order": list(SEEDS),
                "bpb": numeric_summary([bpb[seed][policy] for seed in SEEDS]),
                "evaluation_seconds": numeric_summary(
                    [timing[seed][policy] for seed in SEEDS]
                ),
            }
            for policy in POLICIES
        },
        "contrasts": contrasts,
        "strata": _stratum_summaries(stream.data, boundaries, losses),
        "decision_gate_e_ecological_component": {
            "status": "pass" if gate_pass else "fail",
            "pass": gate_pass,
            "regression_margin_bpb": 0.02,
            "mean_at_or_below_margin": mean_ok,
            "every_seed_at_or_below_margin": every_seed_ok,
            "all_policy_rates_exact": exact_rate,
            "paired_t_upper_at_or_below_margin": (
                ecological["paired_t_95_interval"]["upper"] <= 0.02
            ),
            "bootstrap_upper_at_or_below_margin": (
                ecological["hierarchical_bootstrap_95_interval"]["upper"] <= 0.02
            ),
        },
        "claim_guardrail": (
            "This one-user convenience sample is an ecological stress check, "
            "not representative Korean-corpus or independent benchmark evidence."
        ),
    }
    _write_json(Path(args.output), output)
    print(f"wrote aggregate-only ecological result; gate={output['decision_gate_e_ecological_component']['status']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-root", required=True)
    parser.add_argument(
        "--primary-artifact-root",
        default="artifacts/phase2",
    )
    parser.add_argument(
        "--control-artifact-root",
        default="artifacts/phase2-controls",
    )
    parser.add_argument(
        "--output",
        default="results/private/phase2-ecological/summary.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
