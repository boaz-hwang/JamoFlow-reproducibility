#!/usr/bin/env python3
"""Validate and aggregate the full Phase 2d generation-validity run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from jamoflow.generation import DECODING_MODES, GENERATION_POLICIES
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval


SEEDS = (1729, 2718, 31415, 57721, 65537)
HARD_MASK_SEED = 1729
RATE_METRICS = (
    "valid_utf8_rate",
    "replacement_character_free_rate",
    "valid_jamo_transition_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition(
    source: dict[str, Any],
    seed: int,
    policy: str,
    mode: str,
    variant: str = "unconstrained",
) -> dict[str, Any]:
    return source["conditions"][str(seed)][policy][mode][variant]


def run(args: argparse.Namespace) -> int:
    path = Path(args.run_result)
    source = _read_json(path)
    design = source["design"]
    if source.get("quick_smoke_only"):
        raise ValueError("refusing to promote a generation smoke run")
    expected = {
        "seeds": list(SEEDS),
        "policies": list(GENERATION_POLICIES),
        "decoding_modes": list(DECODING_MODES),
        "prompt_count": 256,
        "prompt_length_bytes": 128,
        "continuation_bytes": 128,
        "fixed_horizon_bytes": 256,
        "hard_mask_seed": HARD_MASK_SEED,
        "hard_mask_all_policies_and_modes": True,
        "samples_or_prompts_serialized": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"generation design mismatch for {key}")

    quality: dict[str, Any] = {}
    for policy in GENERATION_POLICIES:
        quality[policy] = {}
        for mode in DECODING_MODES:
            conditions = [
                _condition(source, seed, policy, mode) for seed in SEEDS
            ]
            quality[policy][mode] = {
                metric: numeric_summary(
                    [float(condition[metric]) for condition in conditions]
                )
                for metric in (*RATE_METRICS, "elapsed_seconds")
            }
            codepoint_values = [
                condition["mean_bytes_per_codepoint_valid_utf8"]
                for condition in conditions
                if condition["mean_bytes_per_codepoint_valid_utf8"] is not None
            ]
            quality[policy][mode]["mean_bytes_per_codepoint_valid_utf8"] = (
                numeric_summary([float(value) for value in codepoint_values])
                if codepoint_values
                else None
            )

    contrasts: dict[str, Any] = {}
    gate_modes: dict[str, Any] = {}
    for mode in DECODING_MODES:
        mode_contrasts: dict[str, Any] = {}
        for metric in RATE_METRICS:
            effects = [
                float(
                    _condition(
                        source,
                        seed,
                        "causal_whitespace_grid",
                        mode,
                    )[metric]
                )
                - float(
                    _condition(source, seed, "fixed_byte_6", mode)[metric]
                )
                for seed in SEEDS
            ]
            mode_contrasts[f"whitespace_minus_fixed/{metric}"] = {
                "seed_order": list(SEEDS),
                "paired_differences": effects,
                "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            }
        validity = mode_contrasts[
            "whitespace_minus_fixed/valid_utf8_rate"
        ]["paired_differences"]
        mean_difference = float(np.mean(validity))
        gate_modes[mode] = {
            "mean_whitespace_minus_fixed_valid_utf8_rate": mean_difference,
            "minimum_allowed": -0.01,
            "pass": mean_difference >= -0.01,
        }
        contrasts[mode] = mode_contrasts

    hard_mask: dict[str, Any] = {}
    all_hard_valid = True
    for policy in GENERATION_POLICIES:
        hard_mask[policy] = {}
        for mode in DECODING_MODES:
            condition = _condition(
                source,
                HARD_MASK_SEED,
                policy,
                mode,
                "hard_mask_control",
            )
            hard_mask[policy][mode] = condition
            all_hard_valid &= (
                condition["valid_utf8_count"] == 256
                and condition["valid_utf8_rate"] == 1.0
            )
    if not all_hard_valid:
        raise ValueError("UTF-8 hard-mask invariant failed")

    gate_pass = all(value["pass"] for value in gate_modes.values())
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": source["scope"],
        "source": {
            "run_result": str(path),
            "run_result_sha256": _sha256(path),
            "prompt_source": source["source"],
        },
        "design": design,
        "prompt_selection": source["prompt_selection"],
        "quality": quality,
        "whitespace_minus_fixed_contrasts": contrasts,
        "hard_mask_control_seed1729": hard_mask,
        "integrity": {
            "no_samples_or_prompts_serialized": not design[
                "samples_or_prompts_serialized"
            ],
            "all_hard_mask_continuations_valid_utf8": all_hard_valid,
        },
        "decision_gate_h_validity_component": {
            "status": "pass" if gate_pass else "fail",
            "pass": gate_pass,
            "by_decoding_mode": gate_modes,
        },
        "claim_guardrail": source["claim_guardrail"],
    }
    _write_json(Path(args.output), output)
    print(f"wrote validated generation summary to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-result",
        default="runs/phase2-generation/generation-results.json",
    )
    parser.add_argument(
        "--output",
        default="results/phase2-generation/summary.json",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
