#!/usr/bin/env python3
"""Re-evaluate all initial checkpoints on calibration and seal selection evidence."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from jamoflow.compute_conversion import conversion_model_spec
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.inference_calibration_evidence import (
    EVIDENCE_PROTOCOL,
    build_calibration_evidence_manifest,
    expected_evidence_paths,
    seal_calibration_receipt,
    validate_calibration_receipt,
)
from jamoflow.inference_calibration_replay_v2 import (
    load_calibration_context as replay_load_calibration_context,
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
)
from jamoflow.neural_model import build_main_model
from jamoflow.neural_training import resolve_device
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_POLICIES,
)


PLAN_PATH = Path("data/manifests/phase3-inference-selection-plan-v2.json")
IDENTITY_PATH = Path(INITIAL_MODEL_IDENTITY_LOCK_PATH)
OUTPUT = Path("results/phase3-inference-selection-v2/calibration-evidence.json")
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _require_clean_root() -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(root).resolve() != Path.cwd().resolve():
        raise ValueError("run calibration reconstruction from the repository root")
    if _git_status().strip():
        raise ValueError("calibration reconstruction requires a clean worktree")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("calibration reconstruction requires a SHA-1 commit")
    return commit


def _tracked_head_sha256(path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not path.is_file() or path.read_bytes() != result.stdout:
        raise ValueError(f"calibration input is not the exact HEAD blob: {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} is not an ancestor of calibration replay")


def _verify_identity_implementation(
    identity_lock: dict[str, Any], *, current_commit: str
) -> None:
    implementation = identity_lock["calibration_selection_implementation"]
    _require_ancestor(
        implementation["producer_git_commit"],
        current_commit,
        "initial model identity implementation",
    )
    if implementation["environment"] != runtime_environment_v2():
        raise ValueError("calibration runtime environment differs from identity lock")
    for path in implementation["file_order"]:
        if _tracked_head_sha256(Path(path)) != implementation["sha256_by_path"][path]:
            raise ValueError(f"calibration implementation differs: {path}")


def _npz_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, sequence_nll_nats=array)
    return output.getvalue()


def _serialize_receipt(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _model_spec(policy: str):
    if policy in PHASE3_POLICIES:
        return PHASE3_MODEL_SPEC
    return conversion_model_spec(int(policy.rsplit("_", 1)[1]))


def _load_existing_receipt(
    *,
    receipt_path: Path,
    nll_path: Path,
    plan: dict[str, Any],
    plan_artifact_sha256: str,
    evaluator_git_commit: str,
    report_path: Path,
    checkpoint_path: Path,
    matrix: np.ndarray,
    auxiliary: dict[str, Any],
    inputs: np.ndarray,
    boundaries: np.ndarray,
    stream_sha256: str,
    initial_model_identity_lock_sha256: str,
) -> tuple[dict[str, Any], np.ndarray] | None:
    if receipt_path.exists() != nll_path.exists():
        raise ValueError(
            f"partial calibration evidence exists: {receipt_path.parent}"
        )
    if not receipt_path.exists():
        return None
    receipt = _read_json(receipt_path)
    validate_calibration_receipt(receipt, plan=plan)
    with np.load(nll_path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError("resumed calibration NLL has unexpected keys")
        nll = archive["sequence_nll_nats"]
    expected_count = plan["calibration_evaluator"]["sequence_count"]
    if (
        nll.dtype != np.float32
        or nll.shape != (expected_count,)
        or not np.isfinite(nll).all()
        or np.any(nll < 0)
    ):
        raise ValueError("resumed calibration NLL is invalid")
    reconstructed_bpb = math.fsum(float(value) for value in nll) / (
        expected_count * (PHASE3_MODEL_SPEC.sequence_length - 1) * math.log(2)
    )
    spec = _model_spec(receipt["policy"])
    model = build_main_model(
        spec,
        seed=receipt["seed"],
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    checkpoint_state_sha256 = _state_sha256(model)
    del model
    if (
        receipt["plan_artifact_sha256"] != plan_artifact_sha256
        or receipt["initial_model_identity_lock_sha256"]
        != initial_model_identity_lock_sha256
        or receipt["evaluator_git_commit"] != evaluator_git_commit
        or receipt["training_report"]["artifact_sha256"] != hash_file(report_path)
        or receipt["checkpoint"]["artifact_sha256"] != hash_file(checkpoint_path)
        or receipt["calibration"]["matrix_sha256"] != _array_sha256(matrix)
        or receipt["calibration"]["inputs_sha256"] != _array_sha256(inputs)
        or receipt["calibration"]["boundaries_sha256"]
        != _array_sha256(boundaries)
        or receipt["calibration"]["stream_sha256"] != stream_sha256
        or receipt["calibration"]["count"] != len(inputs)
        or receipt["calibration"]["predicted_bytes"] != len(inputs) * 511
        or receipt["calibration"]["nll_artifact_sha256"] != hash_file(nll_path)
        or receipt["calibration"]["nll_array_sha256"] != _array_sha256(nll)
        or not math.isclose(
            receipt["calibration"]["bpb"],
            reconstructed_bpb,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or receipt["checkpoint"]["state_sha256"]
        != checkpoint_state_sha256
        or receipt["auxiliary"] != auxiliary
    ):
        raise ValueError("resumed calibration receipt identity differs")
    return receipt, np.ascontiguousarray(nll)


def _evaluate_receipt(
    *,
    seed: int,
    policy: str,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    stream_sha256: str,
    matrix: np.ndarray,
    auxiliary: dict[str, Any],
    plan: dict[str, Any],
    plan_artifact_sha256: str,
    evaluator_git_commit: str,
    initial_model_identity_lock_sha256: str,
    identity_lock: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    paths = expected_evidence_paths(seed, policy)
    report_path = Path(paths["training_report"])
    checkpoint_path = Path(paths["checkpoint"])
    nll_path = Path(paths["nll"])
    receipt_path = Path(paths["receipt"])
    for path in (report_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    existing = _load_existing_receipt(
        receipt_path=receipt_path,
        nll_path=nll_path,
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        evaluator_git_commit=evaluator_git_commit,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        matrix=matrix,
        auxiliary=auxiliary,
        inputs=inputs,
        boundaries=boundaries,
        stream_sha256=stream_sha256,
        initial_model_identity_lock_sha256=(
            initial_model_identity_lock_sha256
        ),
    )
    print(f"calibration {seed}/{policy}: mandatory checkpoint replay", flush=True)
    replay = replay_calibration_unit(
        seed=seed,
        policy=policy,
        inputs=inputs,
        boundaries=boundaries,
        matrix=matrix,
        auxiliary=auxiliary,
        plan=plan,
        identity_lock=identity_lock,
        device=device,
    )
    losses = replay["losses"]
    bpb = replay["bpb"]
    state_sha256 = replay["checkpoint_state_sha256"]
    actual_parameters = replay["parameter_count"]
    report_bpb = replay["report_bpb"]
    spec_sha256 = replay["spec_sha256"]
    if existing is not None:
        existing_receipt, existing_losses = existing
        if not np.array_equal(existing_losses, losses):
            raise ValueError(
                f"resumed calibration NLL fails causal-forward replay: {seed}/{policy}"
            )
    spec = _model_spec(policy)
    nll_bytes = _npz_bytes(losses)
    nll_artifact_sha256 = (
        hash_file(nll_path)
        if existing is not None
        else hashlib.sha256(nll_bytes).hexdigest()
    )
    payload = {
        "auxiliary": auxiliary,
        "calibration": {
            "boundaries_sha256": _array_sha256(boundaries),
            "bpb": bpb,
            "count": len(losses),
            "dtype": str(losses.dtype),
            "inputs_sha256": _array_sha256(inputs),
            "matrix_sha256": _array_sha256(matrix),
            "nll_array_sha256": _array_sha256(losses),
            "nll_artifact_path": str(nll_path),
            "nll_artifact_sha256": nll_artifact_sha256,
            "predicted_bytes": len(losses)
            * (PHASE3_MODEL_SPEC.sequence_length - 1),
            "report_bpb": float(report_bpb),
            "stream_sha256": stream_sha256,
        },
        "checkpoint": {
            "artifact_sha256": hash_file(checkpoint_path),
            "path": str(checkpoint_path),
            "state_sha256": state_sha256,
        },
        "complete": True,
        "device": device,
        "evaluator_git_commit": evaluator_git_commit,
        "evaluator_protocol": EVIDENCE_PROTOCOL,
        "initial_model_identity_lock_sha256": (
            initial_model_identity_lock_sha256
        ),
        "kind": "phase3_calibration_receipt_v2",
        "model": {
            "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
            "parameters": actual_parameters,
            "spec_sha256": spec_sha256,
        },
        "model_family": (
            "phase3" if policy in PHASE3_POLICIES else "compute_conversion"
        ),
        "patch_count": spec.patch_count,
        "plan_artifact_sha256": plan_artifact_sha256,
        "policy": policy,
        "schema_version": 2,
        "seed": seed,
        "training_report": {
            "artifact_sha256": hash_file(report_path),
            "path": str(report_path),
        },
    }
    receipt = seal_calibration_receipt(payload)
    # Construct and validate every identity before either member of the
    # no-clobber pair becomes visible. A crash between the two publications is
    # deliberately forensic-fail-closed on resume.
    validate_calibration_receipt(receipt, plan=plan)
    if existing is not None:
        if existing_receipt != receipt:
            raise ValueError(
                f"resumed calibration receipt fails canonical replay: {seed}/{policy}"
            )
        print(
            f"calibration {seed}/{policy}: existing receipt replay verified",
            flush=True,
        )
        return existing_receipt
    publish_no_clobber(nll_path, nll_bytes)
    publish_no_clobber(receipt_path, _serialize_receipt(receipt))
    return receipt


def _main_locked() -> int:
    evaluator_commit = _require_clean_root()
    plan_artifact_sha256 = _tracked_head_sha256(PLAN_PATH)
    identity_artifact_sha256 = _tracked_head_sha256(IDENTITY_PATH)
    plan = _read_json(PLAN_PATH)
    identity_lock = _read_json(IDENTITY_PATH)
    validate_selection_plan_v2(plan)
    validate_initial_model_identity_lock_v2(identity_lock)
    if (
        identity_lock["plan_artifact_sha256"] != plan_artifact_sha256
        or identity_lock["plan_payload_sha256"] != plan["plan_sha256"]
    ):
        raise ValueError("initial model identity differs from the selection plan")
    _verify_identity_implementation(
        identity_lock, current_commit=evaluator_commit
    )
    if plan["calibration_evaluator"]["device"] != "mps":
        raise ValueError("selection plan does not authorize the MPS evaluator")
    output_history = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", str(OUTPUT)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if OUTPUT.exists() or output_history:
        raise ValueError("calibration evidence manifest already exists")
    device = resolve_device("mps")
    stream, inputs, boundaries, shared_matrices = replay_load_calibration_context(
        plan
    )
    stream_sha256 = hashlib.sha256(stream).hexdigest()
    receipts: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in INITIAL_SEEDS:
        entropy_matrices, entropy_bundles = reconstruct_entropy_matrices(
            seed=seed,
            inputs=inputs,
            boundaries=boundaries,
            device=device,
            identity_lock=identity_lock,
        )
        matrices = {**shared_matrices, **entropy_matrices}
        if set(matrices) != set(CALIBRATION_POLICY_ORDER):
            raise AssertionError("calibration policy matrix set is incomplete")
        receipts[seed] = {}
        for policy in CALIBRATION_POLICY_ORDER:
            receipts[seed][policy] = _evaluate_receipt(
                seed=seed,
                policy=policy,
                inputs=inputs,
                boundaries=boundaries,
                stream_sha256=stream_sha256,
                matrix=matrices[policy],
                auxiliary=entropy_bundles.get(policy, {"kind": "none"}),
                plan=plan,
                plan_artifact_sha256=plan_artifact_sha256,
                evaluator_git_commit=evaluator_commit,
                initial_model_identity_lock_sha256=(
                    identity_artifact_sha256
                ),
                identity_lock=identity_lock,
                device=device,
            )
        del matrices, entropy_matrices, entropy_bundles
        gc.collect()
        torch.mps.empty_cache()
    if _git_commit() != evaluator_commit or _git_status().strip():
        raise RuntimeError("Git HEAD/worktree changed during calibration reconstruction")
    manifest = build_calibration_evidence_manifest(
        plan=plan,
        plan_artifact_sha256=plan_artifact_sha256,
        evaluator_git_commit=evaluator_commit,
        initial_model_identity_lock_sha256=identity_artifact_sha256,
        receipts=receipts,
    )
    output = (
        json.dumps(
            manifest,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    publish_no_clobber(OUTPUT, output)
    if _git_commit() != evaluator_commit:
        raise RuntimeError("Git HEAD changed while publishing calibration evidence")
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(OUTPUT),
                "status": "complete_pending_commit",
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
