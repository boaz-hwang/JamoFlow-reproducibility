"""Canonical causal-forward replay for initial calibration selection evidence."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from .compute_conversion import (
    CONVERSION_POLICIES,
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
)
from .hplt3 import hash_file
from .inference_initial_model_identity_v2 import (
    canonical_sha256,
    model_identity,
    validate_initial_model_identity_lock_v2,
)
from .inference_selection_v2 import CALIBRATION_POLICY_ORDER
from .neural_data import build_neural_stream
from .neural_model import build_main_model, build_router, parameter_count
from .neural_training import evaluate_main_model, router_entropy_scores
from .phase1 import stream_arrays
from .phase2_patching import (
    calibrate_threshold,
    compact_whitespace_mask,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from .phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
    STRUCTURAL_POLICIES,
    THRESHOLD_POLICIES,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)


SOURCE = Path("data/processed/hplt3-korean-phase3/ko.jsonl")
INTEGRITY = Path("data/processed/hplt3-korean-phase3/integrity.json")
CALIBRATION_BYTES = 8_000_000
GLOBAL_POSITION_LIMIT = 1_032
PUBLICATION_MPS_LOCK_PATH = Path("/tmp/jamoflow-publication-mps.lock")
MPS_PROCESS_MARKERS = (
    "benchmark_inference_actual_v5.py",
    "benchmark_phase3_actual_inference.py",
    "measure_inference_memory_v5.py",
    "reconstruct_inference_calibration_v2.py",
    "reconstruct_inference_confirmation_calibration_v2.py",
    "run_inference_final_quality_v2.py",
    "run_phase1.py",
    "run_phase2.py",
    "run_phase2_controls.py",
    "run_phase2_ecological.py",
    "run_phase2_generation.py",
    "run_phase2_normalization.py",
    "run_phase3.py",
    "run_phase3_compute_conversion.py",
    "run_phase3_ecological.py",
    "run_phase3_generation.py",
    "run_phase3_mechanism.py",
    "run_phase3_normalization.py",
    "run_phase3_ood.py",
    "seal_inference_final_quality_lock_v2.py",
    "seal_inference_selection_lock_v2.py",
)


def _process_inventory() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("MPS process inventory failed closed")
    rows: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        try:
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
        except ValueError:
            continue
    if not rows or os.getpid() not in {row[0] for row in rows}:
        raise ValueError("MPS process inventory could not identify this process")
    return rows


def _ancestor_pids(rows: list[tuple[int, int, str]]) -> set[int]:
    parents = {pid: parent for pid, parent, _ in rows}
    ancestors = {os.getpid()}
    cursor = os.getpid()
    while cursor in parents and parents[cursor] > 0 and parents[cursor] not in ancestors:
        cursor = parents[cursor]
        ancestors.add(cursor)
    return ancestors


@contextmanager
def publication_mps_exclusive():
    """Hold the machine-global evidence lock and reject another neural runner."""

    PUBLICATION_MPS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        PUBLICATION_MPS_LOCK_PATH,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another publication MPS process holds the machine lock") from error
        rows = _process_inventory()
        ignored = _ancestor_pids(rows)
        conflicts = [
            {"pid": pid, "command": command}
            for pid, _, command in rows
            if pid not in ignored
            and any(marker in command for marker in MPS_PROCESS_MARKERS)
        ]
        if conflicts:
            raise ValueError(f"another JamoFlow neural process is active: {conflicts}")
        if (
            not torch.backends.mps.is_built()
            or not torch.backends.mps.is_available()
        ):
            raise ValueError("publication calibration requires available Apple MPS")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def model_spec(policy: str):
    if policy in PHASE3_POLICIES:
        return PHASE3_MODEL_SPEC
    if policy not in CONVERSION_POLICIES:
        raise ValueError("calibration replay policy is outside the sealed pool")
    return conversion_model_spec(int(policy.rsplit("_", 1)[1]))


def load_calibration_context(
    plan: Mapping[str, Any],
) -> tuple[bytes, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if (
        hash_file(SOURCE) != plan["initial_design"]["source_artifact_sha256"]
        or hash_file(INTEGRITY)
        != plan["initial_design"]["source_integrity_artifact_sha256"]
    ):
        raise ValueError("calibration source differs from the selection plan")
    stream = build_neural_stream(
        SOURCE,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data, stream.codepoint_boundaries, stream.sequence_length
    )
    if (
        hashlib.sha256(stream.data).hexdigest()
        != plan["calibration_evaluator"]["input_stream_sha256"]
        or len(inputs) != plan["calibration_evaluator"]["sequence_count"]
    ):
        raise ValueError("calibration stream differs from the selection plan")
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    matrices = dict(
        structural_patch_matrices(boundaries, whitespace, spacelike)
    )
    for rate in CONVERSION_RATES:
        matrices.update(
            conversion_patch_matrices(boundaries, whitespace, rate=rate)
        )
    if set(matrices) != set(STRUCTURAL_POLICIES) | set(CONVERSION_POLICIES):
        raise AssertionError("calibration structural matrix set is incomplete")
    return stream.data, inputs, boundaries, matrices


def reconstruct_entropy_matrices(
    *,
    seed: int,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    device: str,
    identity_lock: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    validate_initial_model_identity_lock_v2(identity_lock)
    anchor = model_identity(
        identity_lock, seed=seed, policy="entropy_threshold_full"
    )["auxiliary"]
    router_checkpoint = Path(anchor["router_checkpoint_path"])
    router_report_path = Path(anchor["router_report_path"])
    cache_path = Path(anchor["threshold_cache_path"])
    diagnostics_path = Path(anchor["threshold_diagnostics_path"])
    for path, expected in (
        (router_checkpoint, anchor["router_checkpoint_artifact_sha256"]),
        (router_report_path, anchor["router_report_artifact_sha256"]),
        (cache_path, anchor["threshold_cache_artifact_sha256"]),
        (
            diagnostics_path,
            anchor["threshold_diagnostics_artifact_sha256"],
        ),
    ):
        if path.is_symlink() or hash_file(path) != expected:
            raise ValueError(f"entropy artifact differs from identity lock: {path}")
    router_report = read_json(router_report_path)
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    router.load_state_dict(
        torch.load(router_checkpoint, map_location="cpu", weights_only=True)
    )
    router_state = state_sha256(router)
    if (
        router_state != anchor["router_state_sha256"]
        or router_report.get("seed") != seed
        or router_report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or router_report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or router_report.get("parameters") != parameter_count(router)
        or router_report.get("trained_state_sha256") != router_state
    ):
        raise ValueError(f"entropy router identity differs for seed {seed}")
    scores = router_entropy_scores(router, inputs, device)
    calibrations = {
        "entropy_threshold_full": calibrate_threshold(
            scores, 86, maximum_patch_length=24
        ),
        "entropy_threshold_codepoint": calibrate_threshold(
            scores,
            86,
            candidate_masks=boundaries,
            maximum_patch_length=24,
        ),
    }
    matrices = {
        "entropy_threshold_full": threshold_patch_matrix(
            scores,
            calibrations["entropy_threshold_full"].threshold_nats,
            maximum_patch_length=24,
        ),
        "entropy_threshold_codepoint": threshold_patch_matrix(
            scores,
            calibrations["entropy_threshold_codepoint"].threshold_nats,
            candidate_masks=boundaries,
            maximum_patch_length=24,
        ),
    }
    scores_sha256 = array_sha256(scores)
    del scores
    release_model(router, device)
    diagnostics = read_json(diagnostics_path)
    provenance = diagnostics.get("_provenance", {})
    if (
        provenance.get("kind") != "phase3_threshold_patch_cache"
        or provenance.get("seed") != seed
        or provenance.get("router_state_sha256") != router_state
        or provenance.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or provenance.get("maximum_patch_length") != 24
        or provenance.get("splits", {}).get("calibration", {}).get(
            "inputs_sha256"
        )
        != array_sha256(inputs)
        or provenance.get("splits", {}).get("calibration", {}).get(
            "boundaries_sha256"
        )
        != array_sha256(boundaries)
    ):
        raise ValueError(f"entropy calibration provenance differs: {seed}")
    with np.load(cache_path, allow_pickle=False) as cache:
        for policy, matrix in matrices.items():
            key = f"calibration__{policy}"
            if key not in cache.files or not np.array_equal(cache[key], matrix):
                raise ValueError(f"entropy cache differs: {seed}/{policy}")
    bundles = {}
    for policy in THRESHOLD_POLICIES:
        calibration = calibrations[policy]
        if diagnostics.get("calibration", {}).get(policy) != calibration.to_dict():
            raise ValueError(f"entropy threshold differs: {seed}/{policy}")
        expected = {
            **variable_patch_diagnostics(matrices[policy], boundaries).to_dict(),
            "matrix_sha256": array_sha256(matrices[policy]),
        }
        if diagnostics.get("splits", {}).get("calibration", {}).get(policy) != expected:
            raise ValueError(f"entropy diagnostics differ: {seed}/{policy}")
        policy_anchor = model_identity(identity_lock, seed=seed, policy=policy)[
            "auxiliary"
        ]
        if dict(policy_anchor) != dict(anchor):
            raise ValueError("entropy policies do not share the locked seed router")
        bundles[policy] = {
            "cache_artifact_sha256": hash_file(cache_path),
            "cache_path": str(cache_path),
            "candidate_mask": "none" if policy.endswith("full") else "codepoint",
            "diagnostics_artifact_sha256": hash_file(diagnostics_path),
            "diagnostics_path": str(diagnostics_path),
            "kind": "entropy_router",
            "maximum_patch_length": 24,
            "router_checkpoint_artifact_sha256": hash_file(router_checkpoint),
            "router_checkpoint_path": str(router_checkpoint),
            "router_report_artifact_sha256": hash_file(router_report_path),
            "router_report_path": str(router_report_path),
            "router_scores_sha256": scores_sha256,
            "router_state_sha256": router_state,
            "threshold_nats": calibration.threshold_nats,
        }
    return matrices, bundles


def replay_calibration_unit(
    *,
    seed: int,
    policy: str,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    matrix: np.ndarray,
    auxiliary: Mapping[str, Any],
    plan: Mapping[str, Any],
    identity_lock: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    if seed not in identity_lock["seed_order"] or policy not in CALIBRATION_POLICY_ORDER:
        raise ValueError("calibration replay unit is outside the locked design")
    anchor = model_identity(identity_lock, seed=seed, policy=policy)
    report_path = Path(anchor["training_report"]["path"])
    checkpoint_path = Path(anchor["checkpoint"]["path"])
    if (
        report_path.is_symlink()
        or checkpoint_path.is_symlink()
        or hash_file(report_path) != anchor["training_report"]["artifact_sha256"]
        or hash_file(checkpoint_path) != anchor["checkpoint"]["artifact_sha256"]
    ):
        raise ValueError(f"calibration model artifact differs: {seed}/{policy}")
    report = read_json(report_path)
    spec = model_spec(policy)
    model = build_main_model(
        spec, seed=seed, global_max_position_embeddings=GLOBAL_POSITION_LIMIT
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state = state_sha256(model)
    parameters = parameter_count(model)
    report_bpb = report.get("evaluation", {}).get("calibration", {}).get("bpb")
    if (
        state != anchor["checkpoint"]["state_sha256"]
        or parameters != anchor["parameter_count"]
        or canonical_sha256(spec.to_dict()) != anchor["model_spec_sha256"]
        or canonical_sha256(PHASE3_OPTIMIZATION_SPEC.to_dict())
        != anchor["optimization_spec_sha256"]
        or report.get("seed") != seed
        or report.get("policy") != policy
        or report.get("parameters") != parameters
        or report.get("model_spec") != spec.to_dict()
        or report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("trained_state_sha256") != state
        or report.get("patch_matrix_sha256", {}).get("calibration")
        != array_sha256(matrix)
        or report.get("patch_diagnostics", {}).get("calibration")
        != variable_patch_diagnostics(matrix, boundaries).to_dict()
        or not isinstance(report_bpb, (int, float))
        or not math.isfinite(float(report_bpb))
    ):
        raise ValueError(f"calibration training identity differs: {seed}/{policy}")
    if policy in THRESHOLD_POLICIES:
        locked_auxiliary = anchor["auxiliary"]
        for receipt_key, lock_key in (
            ("cache_artifact_sha256", "threshold_cache_artifact_sha256"),
            (
                "diagnostics_artifact_sha256",
                "threshold_diagnostics_artifact_sha256",
            ),
            (
                "router_checkpoint_artifact_sha256",
                "router_checkpoint_artifact_sha256",
            ),
            ("router_report_artifact_sha256", "router_report_artifact_sha256"),
            ("router_state_sha256", "router_state_sha256"),
        ):
            if auxiliary.get(receipt_key) != locked_auxiliary.get(lock_key):
                raise ValueError(f"calibration auxiliary differs: {seed}/{policy}")
    elif dict(auxiliary) != {"kind": "none"}:
        raise ValueError("structural calibration replay claims an auxiliary")
    validate_padded_patch_matrix(matrix, 512)
    summary, losses = evaluate_main_model(
        model,
        inputs,
        matrix,
        device,
        batch_size=plan["calibration_evaluator"]["batch_size"],
        return_sequence_nll=True,
    )
    if losses is None or losses.dtype != np.float32:
        release_model(model, device)
        raise ValueError("calibration evaluator did not return exact float32 losses")
    values = np.ascontiguousarray(losses)
    release_model(model, device)
    if (
        values.shape != (len(inputs),)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(f"recomputed calibration NLL is invalid: {seed}/{policy}")
    bpb = math.fsum(float(value) for value in values) / (
        len(values) * 511 * math.log(2)
    )
    if (
        summary.examples != len(values)
        or summary.predicted_bytes != len(values) * 511
        or not math.isclose(summary.bpb, bpb, rel_tol=0, abs_tol=1e-7)
        or not math.isclose(float(report_bpb), bpb, rel_tol=0, abs_tol=1e-7)
    ):
        raise ValueError(f"recomputed calibration BPB differs: {seed}/{policy}")
    return {
        "auxiliary": dict(auxiliary),
        "bpb": bpb,
        "checkpoint_artifact_sha256": hash_file(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_state_sha256": state,
        "losses": values,
        "nll_array_sha256": array_sha256(values),
        "parameter_count": parameters,
        "report_artifact_sha256": hash_file(report_path),
        "report_bpb": float(report_bpb),
        "report_path": str(report_path),
        "spec_sha256": canonical_sha256(spec.to_dict()),
    }
