#!/usr/bin/env python3
"""Regenerate target traces and publish aggregate retrieval mechanism evidence."""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from benchmark_fresh_vocabulary_16k_block import load_target, prepare_payloads
from fresh_vocabulary_16k_retrieval_mechanism_core import (
    MODES,
    PROFILE_ROLES,
    ProposalEvent,
    replay_proposal_events,
    summarize_mechanism,
)
from fresh_vocabulary_16k_retrieval_mechanism_protocol import (
    ACTUAL_PLAN_PATH,
    ACTUAL_RESULT_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    ROOT,
    canonical_sha256,
    hash_file,
    json_bytes,
    read_json,
    validate_plan,
)
from fresh_vocabulary_16k_retrieval_protocol import (
    load_table,
    reconstruct_cases,
)

from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_never_published(path: Path) -> None:
    if path.exists() or _git(
        "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    ):
        raise FileExistsError(path)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _event_commitment(events: list[ProposalEvent]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(canonical_sha256(event.to_dict()).encode("ascii"))
    return digest.hexdigest()


def _validate_against_timed_aggregate(
    events: list[ProposalEvent],
    actual_result: dict[str, Any],
) -> None:
    comparisons = actual_result["actual_retrieval"]["comparisons"]
    for role in PROFILE_ROLES:
        for mode in MODES:
            selected = [
                event for event in events if event.role == role and event.mode == mode
            ]
            proposals = [event for event in selected if event.proposal_tokens > 0]
            expected = comparisons[role][mode]
            actual = {
                "proposal_attempts": len(proposals) * 5,
                "proposal_tokens": sum(event.proposal_tokens for event in proposals) * 5,
                "accepted_draft_tokens": sum(event.accepted_tokens for event in proposals) * 5,
                "corpus_ngram_proposals": (
                    sum(event.source == "corpus_ngram" for event in proposals) * 5
                ),
                "prompt_lookup_proposals": (
                    sum(event.source == "prompt_lookup" for event in proposals) * 5
                ),
            }
            if any(actual[key] != expected[key] for key in actual):
                raise AssertionError(f"retrieval mechanism/timing aggregate differs: {role}/{mode}")


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("retrieval mechanism replay requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()) != commit:
        raise RuntimeError("retrieval mechanism plan must be current HEAD")
    _require_never_published(OUTPUT_PATH)
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    actual_plan = read_json(ACTUAL_PLAN_PATH)
    actual_result = read_json(ACTUAL_RESULT_PATH)
    prompts, continuations, _cases = reconstruct_cases()
    table = load_table()
    events: list[ProposalEvent] = []
    with publication_mps_exclusive():
        bundle = load_target(actual_plan)
        payloads = prepare_payloads(bundle, prompts[-64:], continuations[-64:])
        for case_index, payload in enumerate(payloads):
            for mode in MODES:
                ids = payload[
                    "controlled_ids" if mode == "controlled_replay" else "free_ids"
                ]
                raw = payload[
                    "controlled_raw" if mode == "controlled_replay" else "free_raw"
                ]
                for role in PROFILE_ROLES:
                    events.extend(
                        replay_proposal_events(
                            case_index=case_index,
                            mode=mode,
                            role=role,
                            prompt_raw=payload["prompt_raw"],
                            prompt_ids=payload["prompt_ids"],
                            expected_raw=raw,
                            expected_ids=ids,
                            token_bytes=bundle.token_bytes,
                            next_state_indices=bundle.transitions.next_state_indices,
                            table=table,
                        )
                    )
        bundle.model.to("cpu")
        del bundle, payloads
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
    _validate_against_timed_aggregate(events, actual_result)
    mechanism = summarize_mechanism(events)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "kind": "fresh_vocabulary_16k_retrieval_mechanism_result_v2",
        "protocol_id": plan["protocol_id"],
        "status": (
            "pass_hangul_boundary_mechanism_screen"
            if mechanism["decision"]["disjoint_actual_design_authorized"]
            else "fail_hangul_boundary_mechanism_screen"
        ),
        "git_commit": commit,
        "plan": {
            "artifact_sha256": hash_file(PLAN_PATH),
            "plan_sha256": plan["plan_sha256"],
        },
        "dependencies": {
            "actual_plan_artifact_sha256": hash_file(ACTUAL_PLAN_PATH),
            "actual_result_artifact_sha256": hash_file(ACTUAL_RESULT_PATH),
            "actual_summary_sha256": actual_result["summary_sha256"],
        },
        "event_stream": {
            "event_count": len(events),
            "aggregate_replay_matches_timing_counters": True,
            "commitment_sha256": _event_commitment(events),
            "raw_text_or_token_ids_published": False,
        },
        "mechanism": mechanism,
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during retrieval mechanism replay")
    _publish(OUTPUT_PATH, json_bytes(summary))
    contrast = mechanism["primary_hangul_boundary_contrast"]
    print(f"status={summary['status']}")
    print(f"paired_case_mean_difference={contrast['paired_case_mean_difference']}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
