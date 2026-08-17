#!/usr/bin/env python3
"""Read-only reconstruction of the sealed scale-schedule preflight result."""

from __future__ import annotations

import gc
import json
import subprocess
from typing import Any

import numpy as np
import torch
from run_scale_schedule_preflight import (
    _correctness,
    _load_worker,
    _require_operational_environment,
)
from scale_schedule_preflight_core import (
    ACTIVE_PATH,
    GLOBAL_POSITION_LIMIT,
    MODEL_SEED,
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    SCHEDULE_ORDER,
    SESSION_ORDER,
    TARGET_ORDER,
    _correctness_pass,
    build_scale_schedule_summary,
    summarize_scale_schedule_preflight,
    validate_case_arrays,
    validate_plan,
    validate_scale_schedule_summary,
)

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.publication_scale import publication_model_spec


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _read_json(path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _history(path) -> tuple[str, ...]:
    output = _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    return tuple(line for line in output.splitlines() if line)


def _require_exact_head_blob(path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)
    if path.is_symlink() or committed != path.read_bytes():
        raise ValueError(f"scale-schedule tracked artifact differs: {relative}")


def _require_chronology() -> tuple[str, str, str]:
    if _git("status", "--porcelain"):
        raise ValueError("scale-schedule verification requires a clean worktree")
    if ACTIVE_PATH.exists():
        raise ValueError("scale-schedule execution is still active")
    plan_history = _history(PLAN_PATH)
    summary_history = _history(OUTPUT_PATH)
    if len(plan_history) != 1 or len(summary_history) != 1:
        raise ValueError("scale-schedule publication history differs")
    plan_commit = plan_history[0]
    summary_commit = summary_history[0]
    head = _git("rev-parse", "HEAD")
    if (
        plan_commit == summary_commit
        or subprocess.run(
            ("git", "merge-base", "--is-ancestor", plan_commit, summary_commit),
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
        or subprocess.run(
            ("git", "merge-base", "--is-ancestor", summary_commit, head),
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("scale-schedule Git chronology differs")
    _require_exact_head_blob(PLAN_PATH)
    _require_exact_head_blob(OUTPUT_PATH)
    return plan_commit, summary_commit, head


def _independent_correctness_replay(
    plan: dict[str, Any],
    reports_by_target: dict[int, tuple[dict[str, Any], ...]],
) -> None:
    prompts, continuations = validate_case_arrays(plan)
    _require_operational_environment()
    with publication_mps_exclusive(), torch.inference_mode():
        for target in TARGET_ORDER:
            spec = publication_model_spec(target, 86)
            model = build_main_model(
                spec,
                seed=MODEL_SEED,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            )
            expected = plan["models"][str(target)]
            if (
                parameter_count(model) != expected["expected_parameter_count"]
                or state_sha256(model) != expected["model_state_sha256"]
            ):
                raise ValueError(
                    f"scale-schedule verifier model identity differs: {target}"
                )
            model = model.to("mps")
            model.eval()
            replay = {
                schedule: _correctness(
                    model,
                    prompts,
                    continuations,
                    schedule=schedule,
                )
                for schedule in SCHEDULE_ORDER
            }
            replay_pass = {
                name: _correctness_pass(replay[name]) for name in SCHEDULE_ORDER
            }
            for report in reports_by_target[target]:
                stored_pass = {
                    name: _correctness_pass(report["correctness"][name])
                    for name in SCHEDULE_ORDER
                }
                if stored_pass != replay_pass:
                    raise ValueError(
                        f"scale-schedule correctness replay differs: {target}"
                    )
            model = model.to("cpu")
            del model
            gc.collect()
            torch.mps.empty_cache()
            torch.mps.synchronize()
    _require_operational_environment()


def main() -> None:
    plan_commit, _, head = _require_chronology()
    environment = current_runtime_environment_contract()
    plan = _read_json(PLAN_PATH)
    summary = _read_json(OUTPUT_PATH)
    validate_plan(plan, current_environment=environment)
    validate_scale_schedule_summary(summary)

    timings_by_target: dict[int, np.ndarray] = {}
    reports_by_target: dict[int, tuple[dict[str, Any], ...]] = {}
    evidence_by_target: dict[str, Any] = {}
    for target in TARGET_ORDER:
        timing_rows: list[np.ndarray] = []
        report_rows: list[dict[str, Any]] = []
        evidence_rows: dict[str, Any] = {}
        for session in SESSION_ORDER:
            timing, report, evidence = _load_worker(
                target,
                session,
                plan=plan,
                commit=plan_commit,
            )
            timing_rows.append(timing)
            report_rows.append(report)
            evidence_rows[session] = evidence
        timings_by_target[target] = np.stack(timing_rows, axis=0)
        reports_by_target[target] = tuple(report_rows)
        evidence_by_target[str(target)] = evidence_rows

    aggregate = summarize_scale_schedule_preflight(
        timings_by_target=timings_by_target,
        reports_by_target=reports_by_target,
    )
    rebuilt = build_scale_schedule_summary(
        plan_artifact_sha256=hash_file(PLAN_PATH),
        plan_sha256=plan["plan_sha256"],
        summary_base_git_commit=plan_commit,
        worker_evidence=evidence_by_target,
        aggregate=aggregate,
    )
    if summary != rebuilt:
        raise ValueError("scale-schedule summary reconstruction differs")
    _independent_correctness_replay(plan, reports_by_target)
    if _git("rev-parse", "HEAD") != head or _git("status", "--porcelain"):
        raise ValueError("scale-schedule repository changed during verification")
    print("scale_schedule_full_correctness_verification=pass")
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
