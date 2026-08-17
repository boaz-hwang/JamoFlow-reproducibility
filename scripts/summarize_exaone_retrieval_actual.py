#!/usr/bin/env python3
"""Independently replay and summarize all five sealed EXAONE actual sessions."""

from __future__ import annotations

import gc
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from exaone_actual_runtime import load_case_arrays, load_exaone_runtime
from exaone_retrieval_actual import (
    ARTIFACT_ROOT,
    BASELINE_ROLE_INDEX,
    CANDIDATE_ROLE_INDEX,
    COUNTER_NAMES,
    INNER_REPETITIONS,
    MAXIMUM_DRAFT_TOKENS,
    MEASURED_CASES,
    OUTPUT_TOKENS,
    PLAN_PATH,
    ROLES,
    SESSION_RECEIPT_ROOT,
    SESSIONS,
    SUMMARY_PATH,
    WARMUP_CASES,
    actual_mps_exclusive,
    assert_canonical_workspace_path,
    balanced_role_order,
    build_actual_summary,
    load_session_arrays,
    read_actual_summary,
    read_plan,
    read_session_receipt,
    require_distinct_git_commits,
    session_active_path,
    session_artifact_path,
    session_receipt_path,
    summarize_actual_arrays,
    warmup_case_order,
)
from exaone_retrieval_actual_runtime import (
    ActualGenerationTrial,
    run_actual_baseline_trial,
    run_actual_candidate_trial,
)
from exaone_retrieval_data import (
    ROOT,
    canonical_bytes,
    canonical_sha256,
    hash_file,
)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _history(path: Path) -> tuple[str, ...]:
    value = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    return tuple(line for line in value.splitlines() if line)


def _require_exact_head_blob(path: Path) -> None:
    payload = subprocess.check_output(
        ("git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"), cwd=ROOT
    )
    if payload != path.read_bytes():
        raise ValueError(
            f"artifact is not the exact HEAD blob: {path.relative_to(ROOT)}"
        )


def _require_ancestor(ancestor: str, descendant: str, *, label: str) -> None:
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=ROOT,
        check=False,
        capture_output=True,
    ).returncode:
        raise ValueError(f"EXAONE actual Git chronology differs: {label}")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_evidence(plan: Mapping[str, Any], *, head: str):
    plan_history = _history(PLAN_PATH)
    if len(plan_history) != 1:
        raise ValueError("EXAONE actual plan history differs")
    _require_exact_head_blob(PLAN_PATH)
    if plan["git_commit_before_plan"] == plan_history[0]:
        raise ValueError("EXAONE actual plan was not published after its base commit")
    _require_ancestor(
        plan["git_commit_before_plan"],
        plan_history[0],
        label="implementation base to plan publication",
    )
    _require_ancestor(plan_history[0], head, label="plan to summary")
    receipts = []
    arrays = []
    process_tokens: set[str] = set()
    preceding_publication_commit = plan_history[0]
    for index in range(SESSIONS):
        if session_active_path(index).exists():
            raise RuntimeError("unfinished EXAONE actual session blocks summary")
        receipt_path = session_receipt_path(index)
        history = _history(receipt_path)
        if len(history) != 1:
            raise ValueError("EXAONE actual receipt history differs")
        _require_exact_head_blob(receipt_path)
        receipt = read_session_receipt(index, plan=plan, verify_artifact=True)
        _require_ancestor(
            preceding_publication_commit,
            receipt["runner_git_commit"],
            label="preceding evidence publication to next run",
        )
        _require_ancestor(
            plan_history[0], receipt["runner_git_commit"], label="plan to run"
        )
        require_distinct_git_commits(
            receipt["runner_git_commit"], history[0], label="run to own receipt"
        )
        _require_ancestor(
            receipt["runner_git_commit"], history[0], label="run to receipt"
        )
        _require_ancestor(history[0], head, label="receipt to summary")
        token = receipt["process_start_token_sha256"]
        if token in process_tokens:
            raise ValueError("EXAONE actual fresh-process token is reused")
        process_tokens.add(token)
        receipts.append(receipt)
        arrays.append(load_session_arrays(index))
        preceding_publication_commit = history[0]
    expected_artifacts = {session_artifact_path(index) for index in range(SESSIONS)}
    actual_artifacts = {
        path for path in ARTIFACT_ROOT.rglob("*") if path.is_file() or path.is_symlink()
    }
    if actual_artifacts != expected_artifacts:
        raise ValueError("EXAONE actual artifact namespace differs")
    expected_receipts = {session_receipt_path(index) for index in range(SESSIONS)}
    actual_receipts = {
        path
        for path in SESSION_RECEIPT_ROOT.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_receipts != expected_receipts:
        raise ValueError("EXAONE actual receipt namespace differs")
    return receipts, arrays, plan_history[0]


def _trial_for_role(
    bundle, prompt: np.ndarray, role_index: int
) -> ActualGenerationTrial:
    if role_index == BASELINE_ROLE_INDEX:
        return run_actual_baseline_trial(bundle, prompt, output_tokens=OUTPUT_TOKENS)
    if role_index == CANDIDATE_ROLE_INDEX:
        return run_actual_candidate_trial(
            bundle,
            prompt,
            output_tokens=OUTPUT_TOKENS,
            maximum_draft_tokens=MAXIMUM_DRAFT_TOKENS,
        )
    raise ValueError("EXAONE actual replay role differs")


def _pair_exact(left: ActualGenerationTrial, right: ActualGenerationTrial) -> None:
    if (
        left.output_token_ids != right.output_token_ids
        or left.output_token_sha256 != right.output_token_sha256
        or left.decoded_utf8_sha256 != right.decoded_utf8_sha256
    ):
        raise ValueError("EXAONE actual replay roles differ")


def _counter_projection(trial: ActualGenerationTrial) -> dict[str, int]:
    return {name: int(getattr(trial, name)) for name in COUNTER_NAMES}


def _validate_stored_trial(
    arrays: Mapping[str, np.ndarray],
    *,
    case_index: int,
    repetition: int,
    role_index: int,
    trial: ActualGenerationTrial,
) -> None:
    index = (case_index, repetition, role_index)
    if (
        tuple(int(value) for value in arrays["output_token_ids"][index])
        != trial.output_token_ids
        or bytes(arrays["output_token_sha256"][index]).hex()
        != trial.output_token_sha256
        or bytes(arrays["decoded_utf8_sha256"][index]).hex()
        != trial.decoded_utf8_sha256
        or any(
            int(arrays[name][index]) != value
            for name, value in _counter_projection(trial).items()
        )
    ):
        raise ValueError("EXAONE actual stored trial differs from independent replay")


def _warmup_root(bundle, cases: Mapping[str, np.ndarray], session_index: int) -> str:
    rows = []
    for case_index in warmup_case_order(session_index):
        trials = {
            role: _trial_for_role(bundle, cases["prompt_token_ids"][case_index], role)
            for role in balanced_role_order(session_index, case_index, 0)
        }
        _pair_exact(trials[BASELINE_ROLE_INDEX], trials[CANDIDATE_ROLE_INDEX])
        rows.append(
            {
                "case_index": case_index,
                "decoded_utf8_sha256": trials[BASELINE_ROLE_INDEX].decoded_utf8_sha256,
                "output_token_sha256": trials[BASELINE_ROLE_INDEX].output_token_sha256,
            }
        )
    return canonical_sha256({"session_index": session_index, "warmup": rows})


def _independent_replay(
    receipts: list[Mapping[str, Any]], sessions: list[Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    cases = load_case_arrays()
    bundle = load_exaone_runtime(load_table=True)
    try:
        if (
            bundle.model_files != receipts[0]["model_identity"]["model_files"]
            or bundle.model_parameter_count
            != receipts[0]["model_identity"]["model_parameter_count"]
            or bundle.table_resident_bytes
            != receipts[0]["model_identity"]["table_resident_bytes"]
        ):
            raise ValueError("EXAONE actual replay model identity differs")
        for index, receipt in enumerate(receipts):
            if (
                _warmup_root(bundle, cases, index)
                != receipt["warmup_output_root_sha256"]
            ):
                raise ValueError("EXAONE actual warmup replay root differs")

        replay_rows = []
        stored_comparisons = 0
        for case_index in range(MEASURED_CASES):
            prompt = cases["prompt_token_ids"][WARMUP_CASES + case_index]
            baseline = _trial_for_role(bundle, prompt, BASELINE_ROLE_INDEX)
            candidate = _trial_for_role(bundle, prompt, CANDIDATE_ROLE_INDEX)
            _pair_exact(baseline, candidate)
            for arrays in sessions:
                for repetition in range(INNER_REPETITIONS):
                    _validate_stored_trial(
                        arrays,
                        case_index=case_index,
                        repetition=repetition,
                        role_index=BASELINE_ROLE_INDEX,
                        trial=baseline,
                    )
                    _validate_stored_trial(
                        arrays,
                        case_index=case_index,
                        repetition=repetition,
                        role_index=CANDIDATE_ROLE_INDEX,
                        trial=candidate,
                    )
                    stored_comparisons += len(ROLES)
            replay_rows.append(
                {
                    "case_index": case_index,
                    "candidate_counters": _counter_projection(candidate),
                    "decoded_utf8_sha256": baseline.decoded_utf8_sha256,
                    "output_token_sha256": baseline.output_token_sha256,
                }
            )
        return {
            "independent_checkpoint_forward_replay": True,
            "measured_case_count": MEASURED_CASES,
            "replay_root_sha256": canonical_sha256({"cases": replay_rows}),
            "stored_trial_comparisons": stored_comparisons,
            "warmup_session_root_comparisons": SESSIONS,
        }
    finally:
        del bundle
        gc.collect()
        mx.clear_cache()
        mx.synchronize()


def _memory_summary(
    receipts: list[Mapping[str, Any]], sessions: list[Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    peaks = np.stack([arrays["peak_active_bytes"] for arrays in sessions])
    return {
        "claim_scope": "descriptive_only_not_a_memory_improvement_gate",
        "baseline_trial_peak_active_bytes_median": float(
            np.median(peaks[..., BASELINE_ROLE_INDEX])
        ),
        "baseline_trial_peak_active_bytes_maximum": int(
            np.max(peaks[..., BASELINE_ROLE_INDEX])
        ),
        "candidate_trial_peak_active_bytes_median": float(
            np.median(peaks[..., CANDIDATE_ROLE_INDEX])
        ),
        "candidate_trial_peak_active_bytes_maximum": int(
            np.max(peaks[..., CANDIDATE_ROLE_INDEX])
        ),
        "session_working_set_fraction_maximum": max(
            receipt["memory"]["working_set_fraction"] for receipt in receipts
        ),
        "all_session_memory_safety_pass": all(
            receipt["memory"]["safety_pass"] for receipt in receipts
        ),
    }


def _session_lineage(receipts: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(receipts) != SESSIONS:
        raise ValueError("EXAONE actual lineage receipt count differs")
    return [
        {
            "artifact_sha256": receipt["artifact"]["sha256"],
            "receipt_artifact_sha256": hash_file(session_receipt_path(index)),
            "receipt_publication_git_commit": _history(session_receipt_path(index))[0],
            "receipt_sha256": receipt["receipt_sha256"],
            "runner_git_commit": receipt["runner_git_commit"],
            "session_index": index,
        }
        for index, receipt in enumerate(receipts)
    ]


def _run_locked() -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE actual summary requires a clean worktree")
    if SUMMARY_PATH.exists() or _history(SUMMARY_PATH):
        raise FileExistsError("EXAONE actual summary already exists or was deleted")
    for path in (ARTIFACT_ROOT, PLAN_PATH, SUMMARY_PATH):
        assert_canonical_workspace_path(path)
    for index in range(SESSIONS):
        assert_canonical_workspace_path(session_receipt_path(index))
    head = _git("rev-parse", "HEAD")
    plan = read_plan(verify_derived=True)
    receipts, sessions, plan_publication_commit = _load_evidence(plan, head=head)
    replay = _independent_replay(receipts, sessions)
    statistics = summarize_actual_arrays(sessions, correctness_pass=True)
    lineage = _session_lineage(receipts)
    memory = _memory_summary(receipts, sessions)
    payload = build_actual_summary(
        plan=plan,
        summary_base_git_commit=head,
        plan_publication_git_commit=plan_publication_commit,
        session_lineage=lineage,
        independent_replay=replay,
        statistics=statistics,
        memory=memory,
    )
    if read_plan(verify_derived=True) != plan:
        raise RuntimeError("EXAONE actual plan or environment changed during replay")
    if head != _git("rev-parse", "HEAD") or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during EXAONE actual summary")
    _publish(SUMMARY_PATH, canonical_bytes(payload))
    restored = read_actual_summary(
        plan=plan,
        verify_derived=True,
        expected_lineage=lineage,
        expected_replay=replay,
        expected_statistics=statistics,
        expected_memory=memory,
    )
    if restored != payload:
        raise ValueError("EXAONE actual published summary differs")
    return payload


def main() -> None:
    with actual_mps_exclusive():
        summary = _run_locked()
    primary = summary["statistics"]["primary_end_to_end"]
    print(f"status={summary['status']}")
    print(f"summary_sha256={summary['summary_sha256']}")
    print(f"median_reduction={primary['median_reduction']:.9f}")
    print(
        "bootstrap_95_interval="
        f"{primary['crossed_session_prompt_bootstrap_95_interval']}"
    )


if __name__ == "__main__":
    main()
