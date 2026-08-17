#!/usr/bin/env python3
"""Read-only verification of the global-heavy bridge evidence."""

from __future__ import annotations

import gc
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from global_heavy_schedule_core import (
    EXPECTED_GLOBAL_PARAMETER_COUNT,
    EXPECTED_PARAMETER_COUNT,
    GLOBAL_HEAVY_SPEC,
    GLOBAL_POSITION_LIMIT,
    MODEL_SEED,
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    SCHEDULE_ORDER,
    SESSION_ORDER,
    build_summary,
    load_case_arrays,
    summarize,
    validate_plan,
    validate_summary,
)
from run_global_heavy_schedule_bridge import (
    _correctness,
    _load_worker,
    _require_operational_environment,
)

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def main() -> None:
    if _git("status", "--porcelain"):
        raise ValueError("global-heavy verifier requires a clean worktree")
    plan = _read_json(PLAN_PATH)
    summary = _read_json(OUTPUT_PATH)
    validate_plan(plan, current_environment=current_runtime_environment_contract())
    validate_summary(summary)
    commit = summary["summary_base_git_commit"]
    timings: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    for session in SESSION_ORDER:
        timing, report, row = _load_worker(session, plan=plan, runner_commit=commit)
        timings.append(timing)
        reports.append(report)
        evidence[session] = row
    aggregate = summarize(np.stack(timings), reports)
    rebuilt = build_summary(
        plan=plan,
        plan_artifact_sha256=hash_file(PLAN_PATH),
        summary_base_git_commit=commit,
        worker_evidence=evidence,
        aggregate=aggregate,
    )
    if rebuilt != summary:
        raise ValueError("global-heavy summary does not reconstruct")
    _require_operational_environment()
    prompts, continuations, _, _, _ = load_case_arrays()
    model = build_main_model(
        GLOBAL_HEAVY_SPEC,
        seed=MODEL_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    if (
        parameter_count(model) != EXPECTED_PARAMETER_COUNT
        or sum(
            parameter.numel()
            for parameter in model.model.global_transformer.parameters()
        )
        != EXPECTED_GLOBAL_PARAMETER_COUNT
        or state_sha256(model) != plan["model"]["model_state_sha256"]
    ):
        raise ValueError("global-heavy verifier model identity differs")
    with publication_mps_exclusive(), torch.inference_mode():
        model = model.to("mps")
        model.eval()
        replay = {
            role: _correctness(model, prompts, continuations, schedule=role)
            for role in SCHEDULE_ORDER
        }
    if replay != reports[0]["correctness"]:
        raise ValueError("global-heavy independent correctness replay differs")
    model = model.to("cpu")
    del model
    gc.collect()
    torch.mps.empty_cache()
    print("global_heavy_schedule_verification=pass")
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
