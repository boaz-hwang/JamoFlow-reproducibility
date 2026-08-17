#!/usr/bin/env python3
"""Seal the result-blind inference-selection plan before conversion training."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import (
    publish_no_clobber,
    validate_seal_envelope,
)
from jamoflow.inference_selection_plan import (
    FINAL_TEST_MANIFEST_PATH,
    FINAL_TEST_SEAL_PATH,
    PHASE3_ALL_INITIAL_SUMMARY_PATH,
    PHASE3_PRIMARY_SUMMARY_PATH,
    SELECTION_PLAN_PATH,
    build_selection_plan_v2,
    validate_selection_plan_v2,
)
from jamoflow.inference_selection_v2 import CONFIRMATION_SEEDS, INITIAL_SEEDS
from jamoflow.phase3 import PHASE3_POLICIES


OUTPUT = Path(SELECTION_PLAN_PATH)
FINAL_MANIFEST = Path(FINAL_TEST_MANIFEST_PATH)
FINAL_SEAL = Path(FINAL_TEST_SEAL_PATH)
ALL_INITIAL_SUMMARY = Path(PHASE3_ALL_INITIAL_SUMMARY_PATH)
PRIMARY_SUMMARY = Path(PHASE3_PRIMARY_SUMMARY_PATH)
FORBIDDEN_PREEXISTING = (
    Path("results/phase3-inference-selection-v2/calibration-evidence.json"),
    Path("results/phase3-inference-selection-v2/selection-lock.json"),
)
FORBIDDEN_RUN_ROOTS = (
    Path("runs/phase3-compute-conversion"),
    Path("artifacts/phase3-compute-conversion"),
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_root() -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(root).resolve() != Path.cwd().resolve():
        raise ValueError("run selection-plan sealing from the repository root")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("selection-plan sealing requires a clean worktree")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("selection-plan sealing requires a SHA-1 Git commit")
    return commit


def _require_tracked_head_blob(path: Path) -> None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not path.is_file() or path.read_bytes() != result.stdout:
        raise ValueError(f"selection-plan input is not the exact HEAD blob: {path}")


def _load_summary_identity(
    path: Path,
    *,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    require_completed_confirmation: bool = False,
) -> dict:
    _require_tracked_head_blob(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        tuple(summary.get("seeds", ())) != seeds
        or tuple(summary.get("policies", ())) != policies
        or summary.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or summary.get("targets_per_sequence") != 511
    ):
        raise ValueError(f"historical screening identity is invalid: {path}")
    if require_completed_confirmation:
        authorization = summary.get("confirmation_authorization")
        ood = summary.get("ood")
        if (
            summary.get("gate_i", {}).get("overall_pass") is not True
            or summary.get("gate_j", {}).get("overall_pass") is not True
            or not isinstance(authorization, dict)
            or authorization.get("authorization_kind")
            != "phase3_corrected_gate_i_confirmation_v1"
            or not isinstance(ood, dict)
            or ood.get("gate_i_ood_guard", {}).get("pass") is not True
            or ood.get("integrity", {}).get("all_integrity_checks_pass") is not True
        ):
            raise ValueError(
                "selection planning requires completed five-seed Gate J and OOD evidence"
            )
    return summary


def main() -> int:
    plan_commit = _require_clean_root()
    if any(path.exists() for path in FORBIDDEN_PREEXISTING):
        raise ValueError("selection evidence/lock exists before the plan")
    if any(root.exists() and any(root.rglob("*")) for root in FORBIDDEN_RUN_ROOTS):
        raise ValueError("conversion artifacts exist before the selection plan")
    for path in (FINAL_MANIFEST, FINAL_SEAL):
        _require_tracked_head_blob(path)
    final_seal = json.loads(FINAL_SEAL.read_text(encoding="utf-8"))
    validate_seal_envelope(final_seal)
    preparation_commit = final_seal["payload"]["preparation_git_commit"]
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", preparation_commit, plan_commit],
        check=False,
    )
    if final_seal["payload"]["manifest"]["sha256"] != hash_file(FINAL_MANIFEST):
        raise ValueError("final-test seal and manifest identities differ")
    if ancestry.returncode != 0:
        raise ValueError("final-test preparation commit is not a plan ancestor")

    all_initial = _load_summary_identity(
        ALL_INITIAL_SUMMARY,
        seeds=INITIAL_SEEDS,
        policies=tuple(PHASE3_POLICIES),
    )
    _load_summary_identity(
        PRIMARY_SUMMARY,
        seeds=(*INITIAL_SEEDS, *CONFIRMATION_SEEDS),
        policies=(
            "fixed_byte_6",
            "causal_codepoint_grid",
            "causal_whitespace_grid",
        ),
        require_completed_confirmation=True,
    )
    run_manifest = all_initial.get("run_manifest", {})
    source = run_manifest.get("source_artifact", {})
    integrity = run_manifest.get("source_integrity_artifact", {})
    calibration = run_manifest.get("streams", {}).get("calibration", {})
    if (
        not isinstance(source.get("sha256"), str)
        or not isinstance(integrity.get("sha256"), str)
        or not isinstance(calibration.get("selected_stream_sha256"), str)
        or not isinstance(calibration.get("sequence_count"), int)
    ):
        raise ValueError("Phase 3 calibration/source identity is incomplete")

    plan = build_selection_plan_v2(
        plan_git_commit=plan_commit,
        final_test_manifest_sha256=hash_file(FINAL_MANIFEST),
        final_test_seal_sha256=hash_file(FINAL_SEAL),
        final_test_payload_sha256=final_seal["payload_sha256"],
        phase3_all_initial_summary_sha256=hash_file(ALL_INITIAL_SUMMARY),
        phase3_primary_summary_sha256=hash_file(PRIMARY_SUMMARY),
        source_artifact_sha256=source["sha256"],
        source_integrity_artifact_sha256=integrity["sha256"],
        calibration_stream_sha256=calibration["selected_stream_sha256"],
        calibration_sequence_count=calibration["sequence_count"],
    )
    validate_selection_plan_v2(plan)
    serialized = (
        json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_no_clobber(OUTPUT, serialized)
    if _git_commit() != plan_commit:
        raise RuntimeError("Git HEAD changed while sealing the selection plan")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "plan_sha256": plan["plan_sha256"],
                "status": "sealed_pending_commit",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
