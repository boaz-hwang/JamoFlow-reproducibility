#!/usr/bin/env python3
"""Create the one canonical calibration-only selection lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber, validate_seal_envelope
from jamoflow.inference_calibration_evidence import (
    validate_calibration_evidence_manifest,
)
from jamoflow.inference_calibration_replay_v2 import (
    array_sha256,
    load_calibration_context,
    model_spec,
    publication_mps_exclusive,
    reconstruct_entropy_matrices,
    replay_calibration_unit,
)
from jamoflow.inference_initial_model_identity_v2 import (
    INITIAL_MODEL_IDENTITY_LOCK_PATH,
    runtime_environment_v2,
    validate_initial_model_identity_lock_v2,
)
from jamoflow.inference_selection_plan import validate_selection_plan_v2
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
    build_independent_calibration_recomputation_v2,
    build_selection_decision_v2,
    build_selection_lock_v2,
    validate_selection_lock_v2,
)
from jamoflow.neural_training import resolve_device


PLAN = Path("data/manifests/phase3-inference-selection-plan-v2.json")
EVIDENCE = Path("results/phase3-inference-selection-v2/calibration-evidence.json")
IDENTITY = Path(INITIAL_MODEL_IDENTITY_LOCK_PATH)
FINAL_SEAL = Path("data/seals/hplt3-korean-final-test-v1.json")
OUTPUT = Path("results/phase3-inference-selection-v2/selection-lock.json")


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
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if Path(root).resolve() != Path.cwd().resolve() or status.strip():
        raise ValueError("selection lock requires a clean repository root")
    return _git_commit()


def _require_output_never_published() -> None:
    history = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", str(OUTPUT)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if OUTPUT.exists() or history:
        raise ValueError("canonical selection lock already exists or was deleted")


def _tracked_head_sha256(path: Path) -> str:
    return _tracked_head_identity(path)["sha256"]


def _tracked_head_identity(path: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not path.is_file() or path.read_bytes() != result.stdout:
        raise ValueError(f"selection-lock input is not the exact HEAD blob: {path}")
    commit = subprocess.run(
        ["git", "rev-list", "-1", "HEAD", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"selection-lock input has no tracked commit: {path}")
    return {
        "git_commit": commit,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def _git_blob_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"selection implementation is absent at {commit}: {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} is not an ancestor of the selection lock")


def _verify_identity_implementation(
    identity: dict[str, Any], *, current_commit: str
) -> None:
    implementation = identity["calibration_selection_implementation"]
    _require_ancestor(
        implementation["producer_git_commit"],
        current_commit,
        "initial model identity implementation",
    )
    if implementation["environment"] != runtime_environment_v2():
        raise ValueError("selection replay environment differs from identity lock")
    for path in implementation["file_order"]:
        if _tracked_head_sha256(Path(path)) != implementation["sha256_by_path"][path]:
            raise ValueError(f"selection replay implementation differs: {path}")


def _verify_double_replay_chronology(
    *,
    identity: dict[str, Any],
    evidence: dict[str, Any],
    verification_commit: str,
) -> None:
    identity_artifact = _tracked_head_identity(IDENTITY)
    evidence_artifact = _tracked_head_identity(EVIDENCE)
    evaluator_commit = evidence["evaluator_git_commit"]
    if evaluator_commit == evidence_artifact["git_commit"]:
        raise ValueError("calibration evidence was not committed after its evaluator")
    _require_ancestor(
        identity_artifact["git_commit"],
        evaluator_commit,
        "initial identity artifact",
    )
    _require_ancestor(
        evaluator_commit,
        evidence_artifact["git_commit"],
        "calibration evaluator",
    )
    _require_ancestor(
        evidence_artifact["git_commit"],
        verification_commit,
        "calibration evidence artifact",
    )
    implementation = identity["calibration_selection_implementation"]
    for path in implementation["file_order"]:
        expected = implementation["sha256_by_path"][path]
        if (
            _git_blob_sha256(evaluator_commit, path) != expected
            or _git_blob_sha256(verification_commit, path) != expected
        ):
            raise ValueError(f"double-replay implementation differs: {path}")


def _validate_receipt_artifacts(evidence: dict) -> None:
    for seed in evidence["seed_order"]:
        for policy in evidence["policy_order"]:
            receipt = evidence["receipts"][str(seed)][policy]
            report = Path(receipt["training_report"]["path"])
            checkpoint = Path(receipt["checkpoint"]["path"])
            if (
                hash_file(report)
                != receipt["training_report"]["artifact_sha256"]
                or hash_file(checkpoint)
                != receipt["checkpoint"]["artifact_sha256"]
            ):
                raise ValueError(f"selection receipt artifact differs: {seed}/{policy}")
            auxiliary = receipt["auxiliary"]
            if auxiliary["kind"] == "entropy_router":
                for path_key, hash_key in (
                    ("cache_path", "cache_artifact_sha256"),
                    ("diagnostics_path", "diagnostics_artifact_sha256"),
                    (
                        "router_checkpoint_path",
                        "router_checkpoint_artifact_sha256",
                    ),
                    ("router_report_path", "router_report_artifact_sha256"),
                ):
                    if hash_file(Path(auxiliary[path_key])) != auxiliary[hash_key]:
                        raise ValueError(
                            f"selection router bundle differs: {seed}/{policy}"
                        )


def _independent_replay(
    *,
    plan: dict[str, Any],
    evidence: dict[str, Any],
    identity: dict[str, Any],
    device: str,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[str, str]],
]:
    stream, inputs, boundaries, shared_matrices = load_calibration_context(plan)
    if hashlib.sha256(stream).hexdigest() != plan["calibration_evaluator"][
        "input_stream_sha256"
    ]:
        raise AssertionError("selection replay stream identity disappeared")
    bpb: dict[int, dict[str, float]] = {}
    hashes: dict[int, dict[str, str]] = {}
    for seed in INITIAL_SEEDS:
        entropy_matrices, entropy_bundles = reconstruct_entropy_matrices(
            seed=seed,
            inputs=inputs,
            boundaries=boundaries,
            device=device,
            identity_lock=identity,
        )
        matrices = {**shared_matrices, **entropy_matrices}
        if set(matrices) != set(CALIBRATION_POLICY_ORDER):
            raise AssertionError("selection replay matrix set is incomplete")
        bpb[seed] = {}
        hashes[seed] = {}
        for policy in CALIBRATION_POLICY_ORDER:
            receipt = evidence["receipts"][str(seed)][policy]
            auxiliary = entropy_bundles.get(policy, {"kind": "none"})
            replay = replay_calibration_unit(
                seed=seed,
                policy=policy,
                inputs=inputs,
                boundaries=boundaries,
                matrix=matrices[policy],
                auxiliary=auxiliary,
                plan=plan,
                identity_lock=identity,
                device=device,
            )
            calibration = receipt["calibration"]
            spec = model_spec(policy)
            if (
                replay["nll_array_sha256"]
                != calibration["nll_array_sha256"]
                or replay["bpb"] != calibration["bpb"]
                or replay["report_bpb"] != calibration["report_bpb"]
                or calibration["inputs_sha256"] != array_sha256(inputs)
                or calibration["boundaries_sha256"] != array_sha256(boundaries)
                or calibration["matrix_sha256"]
                != array_sha256(matrices[policy])
                or calibration["stream_sha256"]
                != hashlib.sha256(stream).hexdigest()
                or calibration["count"] != len(inputs)
                or calibration["predicted_bytes"] != len(inputs) * 511
                or calibration["dtype"] != "float32"
                or replay["checkpoint_artifact_sha256"]
                != receipt["checkpoint"]["artifact_sha256"]
                or replay["checkpoint_path"] != receipt["checkpoint"]["path"]
                or replay["checkpoint_state_sha256"]
                != receipt["checkpoint"]["state_sha256"]
                or replay["report_artifact_sha256"]
                != receipt["training_report"]["artifact_sha256"]
                or replay["report_path"] != receipt["training_report"]["path"]
                or replay["parameter_count"] != receipt["model"]["parameters"]
                or replay["spec_sha256"] != receipt["model"]["spec_sha256"]
                or receipt["model"]["global_max_position_embeddings"] != 1_032
                or receipt["patch_count"] != spec.patch_count
                or replay["auxiliary"] != receipt["auxiliary"]
            ):
                raise ValueError(
                    f"selection receipt fails independent causal replay: {seed}/{policy}"
                )
            bpb[seed][policy] = float(replay["bpb"])
            hashes[seed][policy] = replay["nll_array_sha256"]
    return bpb, hashes


def _main_locked() -> int:
    lock_commit = _require_clean_root()
    _require_output_never_published()
    plan_sha256 = _tracked_head_sha256(PLAN)
    evidence_sha256 = _tracked_head_sha256(EVIDENCE)
    identity_sha256 = _tracked_head_sha256(IDENTITY)
    final_seal_sha256 = _tracked_head_sha256(FINAL_SEAL)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    final_seal = json.loads(FINAL_SEAL.read_text(encoding="utf-8"))
    validate_selection_plan_v2(plan)
    validate_calibration_evidence_manifest(evidence, plan=plan)
    validate_initial_model_identity_lock_v2(identity)
    validate_seal_envelope(final_seal)
    if (
        evidence["plan_artifact_sha256"] != plan_sha256
        or evidence["initial_model_identity_lock_sha256"]
        != identity_sha256
        or identity["plan_artifact_sha256"] != plan_sha256
        or identity["plan_payload_sha256"] != plan["plan_sha256"]
        or plan["final_test"]["seal_sha256"] != final_seal_sha256
        or plan["final_test"]["seal_payload_sha256"]
        != final_seal["payload_sha256"]
    ):
        raise ValueError("selection plan/evidence/final-test lineage differs")
    _verify_identity_implementation(identity, current_commit=lock_commit)
    _verify_double_replay_chronology(
        identity=identity,
        evidence=evidence,
        verification_commit=lock_commit,
    )
    device = resolve_device("mps")
    _validate_receipt_artifacts(evidence)
    recomputed_bpb, recomputed_hashes = _independent_replay(
        plan=plan,
        evidence=evidence,
        identity=identity,
        device=device,
    )
    decision = build_selection_decision_v2(recomputed_bpb)
    verification = build_independent_calibration_recomputation_v2(
        recomputed_bpb,
        nll_array_sha256_by_seed_policy=recomputed_hashes,
        evaluator_git_commit=evidence["evaluator_git_commit"],
        verification_git_commit=lock_commit,
        environment_sha256=identity[
            "calibration_selection_implementation"
        ]["environment_sha256"],
        implementation_manifest_sha256=identity[
            "calibration_selection_implementation"
        ]["manifest_sha256"],
    )
    lock = build_selection_lock_v2(
        decision,
        plan_sha256=plan_sha256,
        calibration_evidence_manifest_sha256=evidence_sha256,
        final_test_seal_sha256=final_seal_sha256,
        initial_model_identity_lock_sha256=identity_sha256,
        independent_calibration_recomputation=verification,
    )
    validate_selection_lock_v2(lock)
    if (
        _git_commit() != lock_commit
        or subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        or _tracked_head_sha256(PLAN) != plan_sha256
        or _tracked_head_sha256(EVIDENCE) != evidence_sha256
        or _tracked_head_sha256(IDENTITY) != identity_sha256
        or _tracked_head_sha256(FINAL_SEAL) != final_seal_sha256
    ):
        raise RuntimeError("repository or selection inputs changed during replay")
    output = (
        json.dumps(lock, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_no_clobber(OUTPUT, output)
    if _git_commit() != lock_commit:
        raise RuntimeError("Git HEAD changed while sealing the selection lock")
    print(
        json.dumps(
            {
                "decision_sha256": decision["decision_sha256"],
                "lock_sha256": lock["lock_sha256"],
                "output": str(OUTPUT),
                "status": decision["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    with publication_mps_exclusive():
        return _main_locked()


if __name__ == "__main__":
    raise SystemExit(main())
