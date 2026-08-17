#!/usr/bin/env python3
"""Reconstruct, validate, and aggregate Phase 3 generation validity stress."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from jamoflow.generation import (
    DECODING_MODES,
    GENERATION_POLICIES,
    continuation_metrics_from_diagnostics,
    select_generation_prompts,
    utf8_failure_metrics_from_diagnostics,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase3 import PHASE3_MODEL_SPEC, PHASE3_OPTIMIZATION_SPEC


KNOWN_SEEDS = (1729, 2718, 31415, 57721, 65537)
INITIAL_SEEDS = KNOWN_SEEDS[:3]
HARD_MASK_SEED = 1729
TEST_BYTE_LIMIT = 16_000_000
PROMPT_COUNT = 256
PROMPT_BYTES = 256
CONTINUATION_BYTES = 256
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
PARAMETERS = 19_596_096
F = "fixed_byte_6"
C = "causal_codepoint_grid"
W = "causal_whitespace_grid"
CONTRASTS = {
    "whitespace_minus_codepoint": (W, C),
    "whitespace_minus_fixed": (W, F),
    "codepoint_minus_fixed": (C, F),
}
METRIC_PATHS = {
    "valid_utf8_rate": ("valid_utf8_rate",),
    "replacement_character_free_rate": (
        "replacement_character_free_rate",
    ),
    "valid_jamo_transition_rate": ("valid_jamo_transition_rate",),
    "illegal_transition_rate": (
        "utf8_failure_taxonomy",
        "illegal_transition_rate",
    ),
    "incomplete_terminal_scalar_rate": (
        "utf8_failure_taxonomy",
        "incomplete_terminal_scalar_rate",
    ),
    "mean_legal_prefix_fraction": (
        "utf8_failure_taxonomy",
        "mean_legal_prefix_fraction",
    ),
    "mean_closed_codepoint_prefix_fraction": (
        "utf8_failure_taxonomy",
        "mean_closed_codepoint_prefix_fraction",
    ),
}
DIAGNOSTIC_NAMES = (
    "structural__strict_valid",
    "structural__replacement_character_free",
    "structural__valid_jamo_transition",
    "structural__bytes_per_codepoint",
    "utf8__failure_category",
    "utf8__legal_prefix_bytes",
    "utf8__closed_codepoint_prefix_bytes",
    "utf8__first_illegal_byte_position",
)
DIAGNOSTIC_DTYPES = {
    "structural__strict_valid": np.dtype(np.uint8),
    "structural__replacement_character_free": np.dtype(np.uint8),
    "structural__valid_jamo_transition": np.dtype(np.uint8),
    "structural__bytes_per_codepoint": np.dtype(np.float64),
    "utf8__failure_category": np.dtype(np.uint8),
    "utf8__legal_prefix_bytes": np.dtype(np.int64),
    "utf8__closed_codepoint_prefix_bytes": np.dtype(np.int64),
    "utf8__first_illegal_byte_position": np.dtype(np.int64),
}
REPORT_KEYS = {
    "schema_version",
    "seed",
    "policy",
    "parameters",
    "model_spec",
    "optimization_spec",
    "global_max_position_embeddings",
    "checkpoint_state_sha256",
    "training_report_state_sha256",
    "checkpoint_artifact_sha256",
    "training_report_artifact_sha256",
    "source_stream_sha256",
    "prompt_selection",
    "diagnostic_artifact_filename",
    "diagnostic_artifact_sha256",
    "modes",
    "raw_generation_serialized",
    "prompts_or_prompt_hashes_serialized",
    "non_content_per_prompt_diagnostics_serialized",
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


def paired_effects(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("paired generation values need equal length >=2")
    values = [
        float(left_value) - float(right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired generation effects must be finite")
    return values


def failure_partition_is_valid(payload: dict[str, Any], expected: int) -> bool:
    try:
        counts = [
            int(payload[key])
            for key in (
                "strict_valid_count",
                "illegal_transition_count",
                "incomplete_terminal_scalar_count",
            )
        ]
        rates = [
            float(payload[key])
            for key in (
                "strict_valid_rate",
                "illegal_transition_rate",
                "incomplete_terminal_scalar_rate",
            )
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        payload.get("continuations") == expected
        and sum(counts) == expected
        and all(value >= 0 for value in counts)
        and all(math.isfinite(value) and 0 <= value <= 1 for value in rates)
        and math.isclose(sum(rates), 1.0, abs_tol=1e-12)
        and all(
            math.isclose(rate, count / expected, abs_tol=1e-12)
            for rate, count in zip(rates, counts, strict=True)
        )
    )


def _value(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"generation metric is not finite: {path}")
    return result


def _effect_summary(
    effects: Sequence[float],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "seed_order": list(seeds),
        "paired_differences": list(effects),
        "numeric_summary": numeric_summary(effects),
        "paired_t_95_interval": paired_t_interval(effects).to_dict(),
    }


def _validate_manifest_execution(
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("generation manifest schema mismatch")
    manifest_seeds = manifest.get("seeds", [])
    manifest_policies = manifest.get("policies", [])
    if (
        not isinstance(manifest_seeds, list)
        or len(manifest_seeds) != len(set(manifest_seeds))
        or set(manifest_seeds) - set(KNOWN_SEEDS)
        or not set(seeds) <= set(manifest_seeds)
    ):
        raise ValueError("generation manifest seed coverage is invalid")
    if (
        not isinstance(manifest_policies, list)
        or len(manifest_policies) != len(set(manifest_policies))
        or set(manifest_policies) != set(GENERATION_POLICIES)
    ):
        raise ValueError("generation manifest must cover F/C/W exactly")
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("generation manifest lacks invocation provenance")
    for invocation in invocations:
        if not isinstance(invocation, dict):
            raise ValueError("generation invocation is malformed")
        invocation_seeds = invocation.get("seeds", [])
        invocation_policies = invocation.get("policies", [])
        if (
            not isinstance(invocation_seeds, list)
            or not isinstance(invocation_policies, list)
            or len(invocation_seeds) != len(set(invocation_seeds))
            or len(invocation_policies) != len(set(invocation_policies))
            or set(invocation_seeds) - set(KNOWN_SEEDS)
            or set(invocation_policies) - set(GENERATION_POLICIES)
        ):
            raise ValueError("generation invocation coverage is malformed")
    for seed in seeds:
        for policy in GENERATION_POLICIES:
            if not any(
                seed in invocation["seeds"]
                and policy in invocation["policies"]
                for invocation in invocations
            ):
                raise ValueError(
                    f"generation manifest lacks invocation for {seed}/{policy}"
                )


def _reconstruct_source_and_prompts(
    manifest: dict[str, Any],
    data_root: Path,
    training_run_root: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    if manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("generation manifest model spec mismatch")
    if manifest.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict():
        raise ValueError("generation manifest optimization spec mismatch")
    if manifest.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT:
        raise ValueError("generation global position limit mismatch")
    source_path = data_root / "ko.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = manifest.get("source", {})
    expected_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    if source.get("source_artifact") != expected_artifact:
        raise ValueError("generation source artifact mismatch")
    if source.get("requested_byte_limit") != TEST_BYTE_LIMIT:
        raise ValueError("generation source byte limit mismatch")
    stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=TEST_BYTE_LIMIT,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    expected_source = {
        "source_artifact": expected_artifact,
        "requested_byte_limit": TEST_BYTE_LIMIT,
        "stream": stream.metadata(),
        "selected_stream_sha256": _sha256_bytes(stream.data),
    }
    if source != expected_source:
        raise ValueError("generation source differs from reconstruction")
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    selection = select_generation_prompts(
        inputs,
        boundaries,
        prompt_count=PROMPT_COUNT,
        prompt_length=PROMPT_BYTES,
    )
    prompt_metadata = selection.public_metadata()
    if manifest.get("prompt_selection") != prompt_metadata:
        raise ValueError("generation prompt selection differs from reconstruction")

    primary_path = training_run_root / "manifest.json"
    primary = _read_json(primary_path)
    expected_primary_test = {
        **stream.metadata(),
        "selected_stream_sha256": _sha256_bytes(stream.data),
    }
    if (
        primary.get("quick_smoke_only") is not False
        or primary.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or primary.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or primary.get("limits", {}).get("test") != TEST_BYTE_LIMIT
        or primary.get("streams", {}).get("test") != expected_primary_test
    ):
        raise ValueError("generation source/model differs from primary experiment")
    primary_context = {
        "manifest_sha256": _sha256(primary_path),
        "test_stream_matches": True,
        "model_and_optimization_match": True,
    }
    return prompt_metadata, primary_context


def _reconstruct_condition_metrics(
    diagnostics: Mapping[str, np.ndarray],
    mode: str,
    variant: str,
) -> dict[str, Any]:
    prefix = f"{mode}__{variant}__"
    local = {
        key[len(prefix) :]: np.asarray(value)
        for key, value in diagnostics.items()
        if key.startswith(prefix)
    }
    structural = {
        key[len("structural__") :]: value
        for key, value in local.items()
        if key.startswith("structural__")
    }
    failure = {
        key[len("utf8__") :]: value
        for key, value in local.items()
        if key.startswith("utf8__")
    }
    structural_metrics = continuation_metrics_from_diagnostics(
        structural,
        CONTINUATION_BYTES,
    ).to_dict()
    failure_metrics = utf8_failure_metrics_from_diagnostics(
        failure,
        CONTINUATION_BYTES,
    ).to_dict()
    if (
        structural_metrics["continuations"] != PROMPT_COUNT
        or structural_metrics["continuation_bytes"] != CONTINUATION_BYTES
        or structural_metrics["valid_utf8_count"]
        != failure_metrics["strict_valid_count"]
    ):
        raise ValueError("generation diagnostic geometry or taxonomy mismatch")
    return {
        **structural_metrics,
        "utf8_failure_taxonomy": failure_metrics,
    }


def _load_checkpoint_lineage(
    seed: int,
    policy: str,
    training_run_root: Path,
    checkpoint_root: Path,
) -> dict[str, str]:
    training_path = training_run_root / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = checkpoint_root / f"seed-{seed}" / f"{policy}.pt"
    if not training_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing primary evidence for generation {seed}/{policy}"
        )
    training = _read_json(training_path)
    if (
        training.get("seed") != seed
        or training.get("policy") != policy
        or training.get("parameters") != PARAMETERS
        or training.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or training.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
    ):
        raise ValueError(f"primary report mismatch for {seed}/{policy}")
    state_hash = _checkpoint_state_sha256(checkpoint_path)
    if state_hash != training.get("trained_state_sha256"):
        raise ValueError(f"primary checkpoint mismatch for {seed}/{policy}")
    return {
        "checkpoint_state_sha256": state_hash,
        "training_report_state_sha256": state_hash,
        "checkpoint_artifact_sha256": _sha256(checkpoint_path),
        "training_report_artifact_sha256": _sha256(training_path),
    }


def _validate_generation_report(
    report_path: Path,
    artifact_path: Path,
    *,
    seed: int,
    policy: str,
    lineage: Mapping[str, str],
    prompt_metadata: Mapping[str, int],
    source_stream_sha256: str,
) -> tuple[dict[str, Any], str]:
    report = _read_json(report_path)
    if set(report) != REPORT_KEYS:
        raise ValueError(f"unexpected generation report fields: {seed}/{policy}")
    artifact_hash = _sha256(artifact_path)
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "policy": policy,
        "parameters": PARAMETERS,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "source_stream_sha256": source_stream_sha256,
        "prompt_selection": dict(prompt_metadata),
        "diagnostic_artifact_filename": (
            f"seed-{seed}/{policy}-diagnostics.npz"
        ),
        "diagnostic_artifact_sha256": artifact_hash,
        "raw_generation_serialized": False,
        "prompts_or_prompt_hashes_serialized": False,
        "non_content_per_prompt_diagnostics_serialized": True,
        **lineage,
    }
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(f"generation report provenance mismatch: {key}")

    expected_variants = {"unconstrained"}
    if seed == HARD_MASK_SEED:
        expected_variants.add("hard_mask_control")
    modes = report.get("modes")
    if not isinstance(modes, dict) or set(modes) != set(DECODING_MODES):
        raise ValueError("generation report modes mismatch")
    with np.load(artifact_path, allow_pickle=False) as archive:
        diagnostics = {key: archive[key] for key in archive.files}
    expected_keys = {
        f"{mode}__{variant}__{name}"
        for mode in DECODING_MODES
        for variant in expected_variants
        for name in DIAGNOSTIC_NAMES
    }
    if set(diagnostics) != expected_keys:
        raise ValueError("generation diagnostic artifact keys mismatch")
    for key, value in diagnostics.items():
        suffix = "__".join(key.split("__")[2:])
        if value.dtype != DIAGNOSTIC_DTYPES[suffix]:
            raise ValueError("generation diagnostic artifact dtype mismatch")

    for mode in DECODING_MODES:
        variants = modes[mode]
        if not isinstance(variants, dict) or set(variants) != expected_variants:
            raise ValueError("generation report variants mismatch")
        for variant in expected_variants:
            reconstructed = _reconstruct_condition_metrics(
                diagnostics,
                mode,
                variant,
            )
            recorded = variants[variant]
            if not isinstance(recorded, dict):
                raise ValueError("generation metrics are malformed")
            elapsed = recorded.get("elapsed_seconds_diagnostic_only")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
            ):
                raise ValueError("generation elapsed diagnostic is malformed")
            comparable = dict(recorded)
            comparable.pop("elapsed_seconds_diagnostic_only", None)
            if comparable != reconstructed:
                raise ValueError("generation aggregates differ from diagnostics")
            taxonomy = reconstructed["utf8_failure_taxonomy"]
            if not failure_partition_is_valid(taxonomy, PROMPT_COUNT):
                raise ValueError("generation failure taxonomy is invalid")
            if (
                variant == "hard_mask_control"
                and (
                    reconstructed["valid_utf8_count"] != PROMPT_COUNT
                    or taxonomy["illegal_transition_count"] != 0
                    or taxonomy["incomplete_terminal_scalar_count"] != 0
                )
            ):
                raise ValueError("Phase 3 UTF-8 hard-mask invariant failed")
    return report, artifact_hash


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    if seeds not in (INITIAL_SEEDS, KNOWN_SEEDS):
        raise ValueError("summarize exactly initial 3 or preregistered 5 seeds")
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    training_run_root = Path(args.training_run_root)
    checkpoint_root = Path(args.checkpoint_root)
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    _validate_manifest_execution(manifest, seeds)
    expected_design = {
        "full_preregistered_design": True,
        "known_seeds": list(KNOWN_SEEDS),
        "policies": list(GENERATION_POLICIES),
        "decoding_modes": list(DECODING_MODES),
        "prompt_count": PROMPT_COUNT,
        "prompt_length_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "fixed_horizon_bytes": PHASE3_MODEL_SPEC.sequence_length,
        "hard_mask_seed": HARD_MASK_SEED,
        "hard_mask_all_policies_and_modes": True,
        "samples_prompts_or_prompt_hashes_serialized": False,
        "non_content_per_prompt_diagnostics_serialized": True,
        "use_cache": False,
        "elapsed_time_is_latency_evidence": False,
        "decision_gate": None,
    }
    if manifest.get("design") != expected_design:
        raise ValueError("generation design mismatch")
    prompt_metadata, primary_context = _reconstruct_source_and_prompts(
        manifest,
        Path(args.data_root),
        training_run_root,
    )

    reports: dict[int, dict[str, dict[str, Any]]] = {}
    report_hashes: dict[str, dict[str, str]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    checkpoint_lineage: dict[str, dict[str, dict[str, str]]] = {}
    source_stream_sha256 = manifest["source"]["selected_stream_sha256"]
    for seed in seeds:
        reports[seed] = {}
        report_hashes[str(seed)] = {}
        artifact_hashes[str(seed)] = {}
        checkpoint_lineage[str(seed)] = {}
        for policy in GENERATION_POLICIES:
            report_path = run_root / f"seed-{seed}" / f"{policy}.json"
            artifact_path = (
                artifact_root
                / f"seed-{seed}"
                / f"{policy}-diagnostics.npz"
            )
            if not report_path.exists() or not artifact_path.exists():
                raise FileNotFoundError(
                    f"missing Phase 3 generation result for {seed}/{policy}"
                )
            lineage = _load_checkpoint_lineage(
                seed,
                policy,
                training_run_root,
                checkpoint_root,
            )
            report, artifact_hash = _validate_generation_report(
                report_path,
                artifact_path,
                seed=seed,
                policy=policy,
                lineage=lineage,
                prompt_metadata=prompt_metadata,
                source_stream_sha256=source_stream_sha256,
            )
            reports[seed][policy] = report
            report_hashes[str(seed)][policy] = _sha256(report_path)
            artifact_hashes[str(seed)][policy] = artifact_hash
            checkpoint_lineage[str(seed)][policy] = lineage

    def conditions(policy: str, mode: str) -> list[dict[str, Any]]:
        return [
            reports[seed][policy]["modes"][mode]["unconstrained"]
            for seed in seeds
        ]

    quality: dict[str, Any] = {}
    for policy in GENERATION_POLICIES:
        quality[policy] = {}
        for mode in DECODING_MODES:
            values = conditions(policy, mode)
            summary = {
                metric: numeric_summary(
                    [_value(condition, path) for condition in values]
                )
                for metric, path in METRIC_PATHS.items()
            }
            elapsed_values = [
                float(condition["elapsed_seconds_diagnostic_only"])
                for condition in values
            ]
            if not all(math.isfinite(value) and value >= 0 for value in elapsed_values):
                raise ValueError("generation elapsed diagnostic is invalid")
            summary["elapsed_seconds_diagnostic_only"] = numeric_summary(
                elapsed_values
            )
            for metric in (
                "mean_bytes_per_codepoint_valid_utf8",
                "median_bytes_per_codepoint_valid_utf8",
            ):
                available = [
                    float(condition[metric])
                    for condition in values
                    if condition[metric] is not None
                ]
                summary[metric] = (
                    numeric_summary(available) if available else None
                )
            first_illegal = [
                float(
                    condition["utf8_failure_taxonomy"][
                        "mean_first_illegal_byte_position"
                    ]
                )
                for condition in values
                if condition["utf8_failure_taxonomy"][
                    "mean_first_illegal_byte_position"
                ]
                is not None
            ]
            summary["mean_first_illegal_byte_position"] = (
                numeric_summary(first_illegal) if first_illegal else None
            )
            quality[policy][mode] = summary

    contrasts: dict[str, Any] = {}
    for mode in DECODING_MODES:
        contrasts[mode] = {}
        for name, (left, right) in CONTRASTS.items():
            left_values = conditions(left, mode)
            right_values = conditions(right, mode)
            contrasts[mode][name] = {
                "left_policy": left,
                "right_policy": right,
                "difference_direction": (
                    "left_minus_right; sign preference depends on metric"
                ),
                "metrics": {
                    metric: _effect_summary(
                        paired_effects(
                            [_value(value, path) for value in left_values],
                            [_value(value, path) for value in right_values],
                        ),
                        seeds,
                    )
                    for metric, path in METRIC_PATHS.items()
                },
            }

    hard_mask: dict[str, Any] = {}
    for policy in GENERATION_POLICIES:
        hard_mask[policy] = {}
        for mode in DECODING_MODES:
            hard_mask[policy][mode] = reports[HARD_MASK_SEED][policy]["modes"][
                mode
            ]["hard_mask_control"]

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "analysis_scope": (
            "initial_3_seeds" if seeds == INITIAL_SEEDS else "final_5_seeds"
        ),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "run_manifest": manifest,
        "primary_context": primary_context,
        "source": manifest["source"],
        "prompt_selection": prompt_metadata,
        "design": manifest["design"],
        "seeds": list(seeds),
        "policies": list(GENERATION_POLICIES),
        "quality": quality,
        "policy_contrasts": contrasts,
        "hard_mask_control_seed1729": hard_mask,
        "integrity": {
            "all_integrity_checks_pass": True,
            "source_and_prompt_metadata_match_independent_reconstruction": True,
            "primary_test_stream_matches_generation_source": True,
            "all_checkpoint_state_hashes_match": True,
            "all_checkpoint_artifact_hashes_match": True,
            "all_training_report_artifact_hashes_match": True,
            "all_aggregate_metrics_reconstructed_from_diagnostics": True,
            "failure_taxonomy_partitions_every_condition": True,
            "all_hard_mask_continuations_strict_valid": True,
            "no_prompts_prompt_hashes_or_generations_serialized": True,
            "diagnostics_are_non_content_numeric_arrays": True,
            "elapsed_time_reconstruction_exempt_diagnostic_only": True,
            "report_sha256": report_hashes,
            "diagnostic_artifact_sha256": artifact_hashes,
            "primary_checkpoint_lineage": checkpoint_lineage,
        },
        "decision_gate": None,
        "claim_guardrail": (
            "Encoding validity is not semantic quality. Per-prompt numeric "
            "diagnostics do not prove the hidden prompt bytes, and full-prefix "
            "elapsed time is not incremental decoding latency evidence."
        ),
    }
    _write_json(Path(args.output), output)
    print(f"wrote Phase 3 generation summary to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase3-generation")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-generation",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--training-run-root", default="runs/phase3")
    parser.add_argument("--checkpoint-root", default="artifacts/phase3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase3-generation/summary.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
