#!/usr/bin/env python3
"""Lock the strongest feasible Phase 3 inference comparator before timing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from jamoflow.compute_conversion import CONVERSION_RATES, conversion_policy
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
)


INITIAL_SEEDS = (1729, 2718, 31415)
REFERENCE_ORDER = (*PHASE3_POLICIES, "selected_same_rate_codepoint")


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


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def select_reference(
    mean_calibration_bpb: Mapping[str, float],
    ordered_candidates: tuple[str, ...],
) -> tuple[str, dict[str, Any]]:
    """Select minimum mean BPB with the preregistered order as exact-tie break."""

    if (
        not ordered_candidates
        or len(set(ordered_candidates)) != len(ordered_candidates)
        or set(mean_calibration_bpb) != set(ordered_candidates)
        or any(
            not math.isfinite(float(value))
            for value in mean_calibration_bpb.values()
        )
    ):
        raise ValueError("reference selection requires one finite value per candidate")
    selected = min(
        ordered_candidates,
        key=lambda policy: (
            float(mean_calibration_bpb[policy]),
            ordered_candidates.index(policy),
        ),
    )
    return selected, {
        "criterion": "lowest initial-three-seed mean calibration BPB",
        "tie_break": "first policy in preregistered candidate order",
        "candidate_order": list(ordered_candidates),
        "mean_calibration_bpb": {
            policy: float(mean_calibration_bpb[policy])
            for policy in ordered_candidates
        },
        "selected_policy": selected,
        "selected_mean_calibration_bpb": float(
            mean_calibration_bpb[selected]
        ),
    }


def _validate_phase3_summary(summary: dict[str, Any]) -> None:
    if (
        tuple(summary.get("seeds", [])) != INITIAL_SEEDS
        or tuple(summary.get("policies", [])) != PHASE3_POLICIES
        or summary.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or summary.get("gate_i", {}).get("overall_pass") is not True
        or summary.get("run_manifest", {}).get("model_spec")
        != PHASE3_MODEL_SPEC.to_dict()
        or summary.get("run_manifest", {}).get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
    ):
        raise ValueError("inference selection requires complete initial F/C/W/S/E/EC evidence")
    test_quality = summary.get("quality", {})
    calibration_quality = summary.get("calibration_quality", {})
    if (
        set(test_quality) != set(PHASE3_POLICIES)
        or set(calibration_quality) != set(PHASE3_POLICIES)
        or any(
            test_quality[policy].get("count") != 3
            or calibration_quality[policy].get("count") != 3
            or not math.isfinite(
                float(test_quality[policy].get("mean", math.nan))
            )
            or not math.isfinite(
                float(calibration_quality[policy].get("mean", math.nan))
            )
            for policy in PHASE3_POLICIES
        )
    ):
        raise ValueError("Phase 3 initial quality summary is incomplete")


def _validate_conversion_summary(summary: dict[str, Any]) -> int:
    rate = summary.get("calibration_rate_selection", {}).get("selected_rate")
    if (
        summary.get("stage") != "initial"
        or tuple(summary.get("seeds", [])) != INITIAL_SEEDS
        or summary.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or summary.get("initial_conversion_gate", {}).get("overall_pass") is not True
        or rate not in CONVERSION_RATES
    ):
        raise ValueError("inference selection requires a passing initial conversion")
    codepoint = conversion_policy("codepoint", int(rate))
    whitespace = conversion_policy("whitespace", int(rate))
    quality = summary.get("quality", {})
    if any(
        quality.get(policy, {}).get("test_bpb", {}).get("count") != 3
        or quality.get(policy, {}).get("calibration_bpb", {}).get("count")
        != 3
        or not math.isfinite(
            float(quality.get(policy, {}).get("test_bpb", {}).get("mean", math.nan))
        )
        or not math.isfinite(
            float(
                quality.get(policy, {})
                .get("calibration_bpb", {})
                .get("mean", math.nan)
            )
        )
        for policy in (codepoint, whitespace)
    ):
        raise ValueError("selected conversion quality is incomplete")
    return int(rate)


def _validate_shared_data(
    phase3: dict[str, Any],
    conversion: dict[str, Any],
) -> None:
    phase3_manifest = phase3["run_manifest"]
    conversion_context = conversion["integrity"]["source_context"]
    if (
        phase3_manifest.get("source_artifact")
        != conversion_context.get("source_artifact")
        or phase3_manifest.get("source_integrity_artifact")
        != conversion_context.get("source_integrity_artifact")
    ):
        raise ValueError("Phase 3 and conversion source artifacts differ")
    for split in ("train", "calibration", "test"):
        phase3_stream = phase3_manifest.get("streams", {}).get(split, {})
        converted_stream = conversion_context.get("streams", {}).get(split, {})
        if (
            phase3_stream.get("selected_stream_sha256")
            != converted_stream.get("selected_stream_sha256")
            or phase3_stream.get("sequence_count")
            != converted_stream.get("sequence_count")
        ):
            raise ValueError(f"Phase 3 and conversion stream differ: {split}")


def run(args: argparse.Namespace) -> int:
    phase3_path = Path(args.phase3_summary)
    conversion_path = Path(args.conversion_summary)
    output_path = Path(args.output)
    phase3 = _read_json(phase3_path)
    conversion = _read_json(conversion_path)
    _validate_phase3_summary(phase3)
    rate = _validate_conversion_summary(conversion)
    _validate_shared_data(phase3, conversion)

    selected_codepoint = conversion_policy("codepoint", rate)
    selected_whitespace = conversion_policy("whitespace", rate)
    actual_order = tuple(
        selected_codepoint if policy == "selected_same_rate_codepoint" else policy
        for policy in REFERENCE_ORDER
    )
    calibration_means = {
        policy: float(phase3["calibration_quality"][policy]["mean"])
        for policy in PHASE3_POLICIES
    }
    test_means = {
        policy: float(phase3["quality"][policy]["mean"])
        for policy in PHASE3_POLICIES
    }
    calibration_means[selected_codepoint] = float(
        conversion["quality"][selected_codepoint]["calibration_bpb"]["mean"]
    )
    test_means[selected_codepoint] = float(
        conversion["quality"][selected_codepoint]["test_bpb"]["mean"]
    )
    reference, selection = select_reference(calibration_means, actual_order)
    candidate_calibration_mean = float(
        conversion["quality"][selected_whitespace]["calibration_bpb"]["mean"]
    )
    candidate_test_mean = float(
        conversion["quality"][selected_whitespace]["test_bpb"]["mean"]
    )
    reference_family = (
        "compute_conversion" if reference == selected_codepoint else "phase3"
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_git_commit": _git_commit(),
        "phase3_initial_summary": {
            "path": str(phase3_path),
            "sha256": _sha256(phase3_path),
        },
        "conversion_initial_summary": {
            "path": str(conversion_path),
            "sha256": _sha256(conversion_path),
        },
        "selection_uses_latency": False,
        "seed_order": list(INITIAL_SEEDS),
        "candidate": {
            "policy": selected_whitespace,
            "runtime_policy": "causal_whitespace_grid",
            "model_family": "compute_conversion",
            "patch_count": rate,
            "initial_mean_calibration_bpb": candidate_calibration_mean,
            "initial_mean_test_bpb": candidate_test_mean,
        },
        "reference": {
            "policy": reference,
            "runtime_policy": (
                "causal_codepoint_grid"
                if reference == selected_codepoint
                else reference
            ),
            "model_family": reference_family,
            "patch_count": (
                rate if reference_family == "compute_conversion" else 86
            ),
            "initial_mean_calibration_bpb": float(
                calibration_means[reference]
            ),
            "initial_mean_test_bpb": float(test_means[reference]),
            "requires_entropy_router": reference
            in {"entropy_threshold_full", "entropy_threshold_codepoint"},
        },
        "reference_selection": selection,
        "initial_candidate_minus_reference_mean_calibration_bpb": (
            candidate_calibration_mean - float(calibration_means[reference])
        ),
        "initial_candidate_minus_reference_mean_test_bpb": (
            candidate_test_mean - float(test_means[reference])
        ),
        "status": "locked_before_latency_pending_five_seed_quality",
    }
    if output_path.exists():
        previous = _read_json(output_path)
        invariant_keys = (
            "schema_version",
            "phase3_initial_summary",
            "conversion_initial_summary",
            "selection_uses_latency",
            "seed_order",
            "candidate",
            "reference",
            "reference_selection",
            "initial_candidate_minus_reference_mean_calibration_bpb",
            "initial_candidate_minus_reference_mean_test_bpb",
            "status",
        )
        if any(previous.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError("existing inference comparator selection differs")
        payload["created_at"] = previous["created_at"]
        payload["selection_git_commit"] = previous.get("selection_git_commit")
    _write_json(output_path, payload)
    print(json.dumps(payload["reference"], indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase3-summary",
        default="results/phase3-all-initial/summary.json",
    )
    parser.add_argument(
        "--conversion-summary",
        default="results/phase3-compute-conversion/initial-summary.json",
    )
    parser.add_argument(
        "--output",
        default="results/phase3-inference-selection/selection.json",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
