#!/usr/bin/env python3
"""Read-only full-forward verifier for the published EXAONE actual summary."""

from __future__ import annotations

from exaone_retrieval_actual import (
    ARTIFACT_ROOT,
    PLAN_PATH,
    SESSION_RECEIPT_ROOT,
    SESSIONS,
    SUMMARY_PATH,
    actual_mps_exclusive,
    assert_canonical_workspace_path,
    read_actual_summary,
    read_plan,
    require_distinct_git_commits,
    session_artifact_path,
    session_receipt_path,
    summarize_actual_arrays,
)
from summarize_exaone_retrieval_actual import (
    _git,
    _history,
    _independent_replay,
    _load_evidence,
    _memory_summary,
    _require_ancestor,
    _require_exact_head_blob,
    _session_lineage,
)


def _run_locked() -> dict:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("EXAONE actual verification requires a clean worktree")
    for path in (ARTIFACT_ROOT, PLAN_PATH, SESSION_RECEIPT_ROOT, SUMMARY_PATH):
        assert_canonical_workspace_path(path)
    for index in range(SESSIONS):
        assert_canonical_workspace_path(session_artifact_path(index))
        assert_canonical_workspace_path(session_receipt_path(index))
    summary_history = _history(SUMMARY_PATH)
    if len(summary_history) != 1:
        raise ValueError("EXAONE actual summary publication history differs")
    _require_exact_head_blob(SUMMARY_PATH)
    head = _git("rev-parse", "HEAD")
    _require_ancestor(summary_history[0], head, label="summary publication to verify")

    plan = read_plan(verify_derived=True)
    receipts, sessions, plan_publication_commit = _load_evidence(plan, head=head)
    replay = _independent_replay(receipts, sessions)
    statistics = summarize_actual_arrays(sessions, correctness_pass=True)
    memory = _memory_summary(receipts, sessions)
    lineage = _session_lineage(receipts)
    summary = read_actual_summary(
        plan=plan,
        verify_derived=True,
        expected_lineage=lineage,
        expected_replay=replay,
        expected_statistics=statistics,
        expected_memory=memory,
    )
    if summary["plan_publication_git_commit"] != plan_publication_commit:
        raise ValueError("EXAONE actual summary plan-publication identity differs")
    require_distinct_git_commits(
        summary["summary_base_git_commit"],
        summary_history[0],
        label="summary base to publication",
    )
    _require_ancestor(
        summary["summary_base_git_commit"],
        summary_history[0],
        label="summary base to publication",
    )
    for index in range(SESSIONS):
        _require_ancestor(
            lineage[index]["receipt_publication_git_commit"],
            summary["summary_base_git_commit"],
            label="receipt publication to summary base",
        )
    if read_plan(verify_derived=True) != plan:
        raise RuntimeError("EXAONE actual plan or environment changed during verify")
    if head != _git("rev-parse", "HEAD") or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during EXAONE actual verification")
    return summary


def main() -> None:
    with actual_mps_exclusive():
        summary = _run_locked()
    print("summary_full_forward_verification=pass")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
