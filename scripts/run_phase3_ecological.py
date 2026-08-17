#!/usr/bin/env python3
"""Evaluate Phase 3 checkpoints on private Markdown without serializing content."""

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

from jamoflow.ecological import build_private_markdown_test_stream, stratum_bpb
from jamoflow.neural_model import build_main_model, parameter_count, research_versions
from jamoflow.neural_training import evaluate_main_model, resolve_device
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.phase3_analysis import (
    hierarchical_paired_bootstrap_estimates,
    phase3_test_strata,
)


KNOWN_SEEDS = (1729, 2718, 31415, 57721, 65537)
DEFAULT_SEEDS = KNOWN_SEEDS[:3]
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
    "whitespace_minus_fixed": (
        "causal_whitespace_grid",
        "fixed_byte_6",
    ),
    "codepoint_minus_fixed": (
        "causal_codepoint_grid",
        "fixed_byte_6",
    ),
}
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
TARGETS_PER_SEQUENCE = PHASE3_MODEL_SPEC.sequence_length - 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _load_verified_model(
    seed: int,
    policy: str,
    run_root: Path,
    artifact_root: Path,
) -> Any:
    report_path = run_root / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("seed") != seed or report.get("policy") != policy:
        raise ValueError("Phase 3 checkpoint report identity mismatch")
    if report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("Phase 3 checkpoint model spec mismatch")
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    if _state_dict_sha256(model) != report["trained_state_sha256"]:
        raise ValueError("Phase 3 checkpoint state hash mismatch")
    return model


def _bootstrap_summary(
    sequence_differences: list[np.ndarray],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    estimates = hierarchical_paired_bootstrap_estimates(
        sequence_differences,
        targets_per_sequence=TARGETS_PER_SEQUENCE,
        repetitions=repetitions,
        seed=seed,
    )
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {
        "repetitions": repetitions,
        "seed": seed,
        "resampling_design": "crossed seeds x shared test sequences",
        "mean": float(estimates.mean()),
        "median": float(median),
        "lower": float(lower),
        "upper": float(upper),
    }


def _contrast_summaries(
    seeds: tuple[int, ...],
    bpb: dict[int, dict[str, float]],
    losses: dict[int, dict[str, np.ndarray]],
    repetitions: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        effects = [bpb[seed][left] - bpb[seed][right] for seed in seeds]
        sequence_differences = [
            losses[seed][left] - losses[seed][right] for seed in seeds
        ]
        reconstructed = [
            float(values.mean()) / (TARGETS_PER_SEQUENCE * math.log(2))
            for values in sequence_differences
        ]
        if any(
            not math.isclose(effect, rebuilt, abs_tol=2e-5)
            for effect, rebuilt in zip(effects, reconstructed, strict=True)
        ):
            raise ValueError(f"aggregate/sequence mismatch in {name}")
        output[name] = {
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(seeds),
            "paired_differences_bpb": effects,
            "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            "hierarchical_bootstrap_95_interval": _bootstrap_summary(
                sequence_differences,
                repetitions=repetitions,
                seed=20_260_810 + index,
            ),
        }
    return output


def _stratum_summaries(
    stream_data: bytes,
    boundary_masks: np.ndarray,
    seeds: tuple[int, ...],
    losses: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    strata, metadata = phase3_test_strata(
        stream_data,
        boundary_masks,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    contrasts: dict[str, Any] = {}
    for contrast_name, (left, right) in CONTRASTS.items():
        values: dict[str, Any] = {}
        for name, stratum in strata.items():
            base = stratum.metadata()
            if int(stratum.selected.sum()) < 50:
                values[name] = {
                    **base,
                    "status": "descriptive_only_below_50_windows",
                }
                continue
            effects = [
                stratum_bpb(
                    losses[seed][left] - losses[seed][right],
                    stratum.selected,
                    targets_per_sequence=TARGETS_PER_SEQUENCE,
                )
                for seed in seeds
            ]
            values[name] = {
                **base,
                "status": "estimated",
                "seed_order": list(seeds),
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
    seeds = tuple(args.seeds)
    if len(seeds) < 2 or set(seeds) - set(KNOWN_SEEDS):
        raise ValueError("ecological evaluation needs >=2 preregistered seeds")
    device = resolve_device(args.device)
    stream = build_private_markdown_test_stream(
        args.vault_root,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    structural = structural_patch_matrices(boundaries, whitespace, spacelike)
    matrices = {policy: structural[policy] for policy in POLICIES}
    diagnostics = {
        policy: variable_patch_diagnostics(matrix, boundaries).to_dict()
        for policy, matrix in matrices.items()
    }
    exact_rate = all(
        values["minimum_data_patches"] == PHASE3_MODEL_SPEC.patch_count
        and values["maximum_data_patches"] == PHASE3_MODEL_SPEC.patch_count
        for values in diagnostics.values()
    )
    if not exact_rate:
        raise AssertionError("Phase 3 ecological patch-rate invariant failed")

    print(
        f"private aggregate: {stream.valid_test_records} valid test records, "
        f"{stream.sequence_count} sequences; device={device}",
        flush=True,
    )
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    bpb: dict[int, dict[str, float]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    timing: dict[int, dict[str, float]] = {}
    parameters: set[int] = set()
    for seed in seeds:
        bpb[seed] = {}
        losses[seed] = {}
        timing[seed] = {}
        for policy in POLICIES:
            model = _load_verified_model(
                seed, policy, run_root, artifact_root
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
            print(f"seed {seed}/{policy}: BPB={evaluation.bpb:.6f}", flush=True)
            _release_model(model, device)
    if len(parameters) != 1:
        raise AssertionError("Phase 3 checkpoint parameter counts differ")

    contrasts = _contrast_summaries(
        seeds, bpb, losses, args.bootstrap_repetitions
    )
    diagnostic_checks = {}
    for name in ("whitespace_minus_codepoint", "whitespace_minus_fixed"):
        mean = float(contrasts[name]["paired_t_95_interval"]["mean"])
        diagnostic_checks[name] = {
            "mean_bpb": mean,
            "maximum_regression_bpb": 0.020,
            "pass": mean <= 0.020,
        }
    diagnostic_pass = exact_rate and all(
        values["pass"] for values in diagnostic_checks.values()
    )
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 3 read-only private Korean ecological diagnostic",
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
            "seeds": list(seeds),
            "policies": list(POLICIES),
            "model_parameters": next(iter(parameters)),
            "sequence_length": PHASE3_MODEL_SPEC.sequence_length,
            "targets_per_sequence": TARGETS_PER_SEQUENCE,
            "patch_count": PHASE3_MODEL_SPEC.patch_count,
            "test_partition": "existing deterministic content-hash test split",
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "all_checkpoint_states_verified_against_public_training_reports": True,
        },
        "corpus_aggregate": stream.public_metadata(),
        "patch_integrity": {
            "all_policies_exactly_86": exact_rate,
            "by_policy": diagnostics,
        },
        "quality": {
            policy: {
                "seed_order": list(seeds),
                "bpb": numeric_summary([bpb[seed][policy] for seed in seeds]),
                "evaluation_seconds": numeric_summary(
                    [timing[seed][policy] for seed in seeds]
                ),
            }
            for policy in POLICIES
        },
        "contrasts": contrasts,
        "strata": _stratum_summaries(
            stream.data, boundaries, seeds, losses
        ),
        "diagnostic_guard": {
            "status": "pass" if diagnostic_pass else "fail",
            "pass": diagnostic_pass,
            "by_contrast": diagnostic_checks,
            "public_gate_replacement": False,
        },
        "claim_guardrail": (
            "This one-user convenience sample is an ecological stress check, "
            "not representative Korean-corpus or independent benchmark evidence."
        ),
    }
    _write_json(Path(args.output), output)
    print(f"wrote aggregate-only Phase 3 ecological result to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--run-root", default="runs/phase3")
    parser.add_argument("--artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--output", default="results/private/phase3-ecological/summary.json"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
