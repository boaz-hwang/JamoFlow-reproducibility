#!/usr/bin/env python3
"""Evaluate the fixed conditional-local factorial on one frozen W72 checkpoint."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from conditional_local import conditional_easy_mask, install_conditional_local
from conditional_local_sensitivity_core import (
    CANDIDATE_ORDER,
    PAIR_ORDER,
    PREOUTCOME_ROUTE_GEOMETRY,
    PROTOCOL_ID,
    ROUTE_ORDER,
    candidate_definition,
    summarize_frozen_sensitivity,
)
from run_static_geometry_one_seed import (
    BASELINE_CHECKPOINT_PATH,
    GLOBAL_POSITION_LIMIT,
    PLAN_PATH as STATIC_PLAN_PATH,
    SEED,
    _read_json,
    _reconstruct_data,
    _validate_plan as _validate_static_plan,
    _verify_static_and_baseline,
)
from static_geometry_preflight_core import geometry_spec
from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import array_sha256
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_initial_model_identity_v2 import runtime_environment_v2
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import evaluate_main_model


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/conditional-local-frozen-sensitivity-v1.json"
STATIC_RESULT_PATH = ROOT / "results/static-geometry-one-seed-v1/summary.json"
ARTIFACT_ROOT = ROOT / "artifacts/conditional-local-frozen-sensitivity-v1"
OUTPUT_PATH = ROOT / "results/conditional-local-frozen-sensitivity-v1/summary.json"
IMPLEMENTATION_PATHS = (
    "docs/108-conditional-local-frozen-sensitivity-protocol.md",
    "pyproject.toml",
    "scripts/conditional_local.py",
    "scripts/conditional_local_sensitivity_core.py",
    "scripts/run_conditional_local_frozen_sensitivity.py",
    "scripts/run_static_geometry_one_seed.py",
    "scripts/static_geometry_one_seed_core.py",
    "scripts/static_geometry_preflight_core.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/utf8.py",
    "tests/test_conditional_local.py",
    "tests/test_conditional_local_sensitivity.py",
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _summary_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("summary_sha256", None)
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _nll_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, sequence_nll_nats=np.ascontiguousarray(values))
    return buffer.getvalue()


def _clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("conditional sensitivity requires a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_unchanged(commit: str) -> None:
    if _clean_commit() != commit:
        raise RuntimeError("conditional sensitivity repository changed during evaluation")


def _require_ac_power() -> str:
    output = subprocess.run(
        ["pmset", "-g", "batt"], check=True, capture_output=True, text=True
    ).stdout
    if "AC Power" not in output:
        raise RuntimeError("conditional sensitivity requires AC power")
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def _publish_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise


def _validate_plan(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "baseline",
        "candidates",
        "claim_boundary",
        "data",
        "decision_rule",
        "implementation_sha256",
        "kind",
        "output",
        "protocol_id",
        "runtime_environment",
        "schema_version",
        "static_result",
        "status",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "conditional_local_frozen_sensitivity_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_frozen_evaluation"
        or set(plan.get("implementation_sha256", {})) != set(IMPLEMENTATION_PATHS)
    ):
        raise ValueError("conditional sensitivity plan schema differs")
    for relative, expected in plan["implementation_sha256"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or hash_file(path) != expected:
            raise ValueError(f"conditional sensitivity implementation differs: {relative}")
    if plan["candidates"] != {
        "candidate_order": list(CANDIDATE_ORDER),
        "pair_order": list(PAIR_ORDER),
        "route_order": list(ROUTE_ORDER),
        "definitions": {
            name: candidate_definition(name) for name in CANDIDATE_ORDER
        },
    }:
        raise ValueError("conditional sensitivity candidate contract differs")
    if plan["runtime_environment"] != runtime_environment_v2():
        raise ValueError("conditional sensitivity runtime environment differs")
    if plan["claim_boundary"] != {
        "calibration_only": True,
        "frozen_failure_falsifies_trained_conditional_models": False,
        "frozen_checkpoint_sensitivity_not_trained_quality": True,
        "hangul_specific_effect_claimed": False,
        "publication_quality_or_efficiency_claimed": False,
        "static_failure_is_not_reclassified_as_pass": True,
        "test_final_downstream_or_latency_read": False,
        "this_stream_reused_as_confirmatory_trained_evaluation": False,
    }:
        raise ValueError("conditional sensitivity claim boundary differs")
    if plan["decision_rule"] != {
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 20261101,
        "minimum_document_coverage": 0.95,
        "minimum_easy_rate": 0.3,
        "pair_requirement": "both utf8_incomplete and hangul_prefix variants must pass",
        "risk_margin_bpb": 0.02,
        "selection": "first passing pair in fixed descending expected-savings order",
    }:
        raise ValueError("conditional sensitivity decision rule differs")
    if plan["output"] != {
        "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
        "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
    }:
        raise ValueError("conditional sensitivity output contract differs")


def _verify_authorities(plan: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if hash_file(STATIC_RESULT_PATH) != plan["static_result"]["artifact_sha256"]:
        raise ValueError("conditional sensitivity static result differs")
    static = _read_json(STATIC_RESULT_PATH)
    if (
        static.get("status") != "one_seed_static_control_stopped"
        or static.get("decision", {}).get("quality_pass") is not False
        or static.get("decision", {}).get("actual_latency_pass") is not True
    ):
        raise ValueError("conditional sensitivity pivot authority differs")
    static_plan = _read_json(STATIC_PLAN_PATH)
    _validate_static_plan(static_plan)
    baseline = _verify_static_and_baseline(static_plan)
    expected = plan["baseline"]
    if expected != static_plan["baseline"]:
        raise ValueError("conditional sensitivity baseline identity differs")
    context = _reconstruct_data(static_plan)
    if plan["data"] != {
        "calibration_document_map": static_plan["data"]["calibration_document_map"],
        "calibration_stream": static_plan["data"]["streams"]["calibration"],
        "preoutcome_route_geometry": PREOUTCOME_ROUTE_GEOMETRY,
        "source_path": static_plan["data"]["source_path"],
        "source_sha256": static_plan["data"]["source_sha256"],
    }:
        raise ValueError("conditional sensitivity data identity differs")
    return baseline, context


def main() -> None:
    commit = _clean_commit()
    if OUTPUT_PATH.exists() or ARTIFACT_ROOT.exists():
        raise FileExistsError("conditional sensitivity evidence already exists")
    plan = _read_json(PLAN_PATH)
    _validate_plan(plan)
    power_sha256 = _require_ac_power()
    baseline, context = _verify_authorities(plan)
    calibration = context["calibration"]
    inputs = calibration["inputs"]
    matrix = calibration["matrix"]
    route_rates: dict[str, float] = {}
    byte_tensor = torch.from_numpy(inputs.astype(np.int64, copy=False))
    generic = conditional_easy_mask(byte_tensor, "utf8_incomplete")
    hangul = conditional_easy_mask(byte_tensor, "hangul_prefix")
    if not bool(torch.all(~hangul | generic)):
        raise RuntimeError("Hangul route is not a subset of the generic UTF-8 route")
    observed_route_geometry = {
        "total_positions": int(generic.numel()),
        "utf8_incomplete_easy_positions": int(generic.sum()),
        "hangul_prefix_easy_positions": int(hangul.sum()),
        "hangul_is_subset_of_utf8_incomplete": True,
    }
    if observed_route_geometry != PREOUTCOME_ROUTE_GEOMETRY:
        raise RuntimeError("conditional sensitivity route geometry differs")
    for route, mask in (("utf8_incomplete", generic), ("hangul_prefix", hangul)):
        route_rates[route] = float(mask.to(dtype=torch.float64).mean())
    del byte_tensor, generic, hangul, context["train"]

    candidate_losses: dict[str, np.ndarray] = {}
    artifacts: dict[str, bytes] = {}
    checkpoint_state = torch.load(
        BASELINE_CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    with publication_mps_exclusive():
        for name in CANDIDATE_ORDER:
            definition = candidate_definition(name)
            model = build_main_model(
                geometry_spec("baseline_w72"),
                seed=SEED,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            )
            model.load_state_dict(checkpoint_state)
            if parameter_count(model) != plan["baseline"]["parameter_count"]:
                raise ValueError("conditional sensitivity parameter count differs")
            install_conditional_local(model, **definition)
            evaluation, losses = evaluate_main_model(
                model,
                inputs,
                matrix,
                "mps",
                batch_size=64,
                return_sequence_nll=True,
            )
            if losses is None:
                raise RuntimeError("conditional sensitivity NLL is missing")
            values = np.ascontiguousarray(losses)
            candidate_losses[name] = values
            artifacts[name] = _nll_bytes(values)
            print(f"{name}: frozen BPB={evaluation.bpb:.9f}", flush=True)
            model.to("cpu")
            del model
            torch.mps.empty_cache()
            torch.mps.synchronize()

    aggregate = summarize_frozen_sensitivity(
        candidate_losses_nats=candidate_losses,
        baseline_losses_nats=baseline,
        document_indices=context["document_map"].document_indices,
        route_rates=route_rates,
        eligible_sequence_fraction=float(
            context["document_map"].metadata()["eligible_sequence_fraction"]
        ),
    )
    evidence = {}
    for name in CANDIDATE_ORDER:
        path = ARTIFACT_ROOT / f"{name}-calibration-nll.npz"
        evidence[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "artifact_sha256": hashlib.sha256(artifacts[name]).hexdigest(),
            "array_sha256": array_sha256(candidate_losses[name]),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "conditional_local_frozen_sensitivity_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": aggregate["selection"]["status"],
        "aggregate": aggregate,
        "route_rates": route_rates,
        "provenance": {
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "static_result_artifact_sha256": hash_file(STATIC_RESULT_PATH),
            "baseline_checkpoint_sha256": hash_file(BASELINE_CHECKPOINT_PATH),
            "power_snapshot_sha256": power_sha256,
            "runtime_environment": runtime_environment_v2(),
        },
        "raw_evidence": evidence,
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = _summary_sha(summary)
    _require_unchanged(commit)
    for name in CANDIDATE_ORDER:
        _publish_no_clobber(
            ARTIFACT_ROOT / f"{name}-calibration-nll.npz", artifacts[name]
        )
    _publish_no_clobber(OUTPUT_PATH, _json_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
