#!/usr/bin/env python3
"""Run the one authorized evaluation on the sealed Korean final-test stream."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.compute_conversion import (
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
)
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import (
    publish_no_clobber,
    validate_seal_envelope,
)
from jamoflow.inference_final_authorization_v2 import (
    FINAL_ARTIFACT_ROOT,
    FINAL_AUTHORIZATION_PATH,
    FINAL_EVIDENCE_PATH,
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_ROUTER_PARAMETER_COUNT,
    FINAL_SEEDS,
    FINAL_TEST_MANIFEST_PATH,
    FINAL_TEST_OUTPUT_PATH,
    FINAL_TEST_SEAL_PATH,
    IMPLEMENTATION_FILE_ORDER,
    SELECTION_LOCK_PATH,
    build_final_model_identity,
    canonical_sha256,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_final_quality_evidence_v2 import (
    FINAL_EVALUATION_BATCH_SIZE,
    FINAL_SESSION_PATH,
    authorized_unit_order,
    build_final_quality_evidence_manifest,
    build_final_quality_receipt,
    build_final_quality_session_plan,
    expected_final_evidence_paths,
    validate_final_quality_evidence_manifest,
    validate_final_quality_receipt,
    validate_final_quality_session_plan,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    build_router,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import (
    evaluate_main_model,
    resolve_device,
    router_entropy_scores,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)


AUTHORIZATION_PATH = Path(FINAL_AUTHORIZATION_PATH)
LOCK_PATH = Path(SELECTION_LOCK_PATH)
FINAL_MANIFEST_PATH = Path(FINAL_TEST_MANIFEST_PATH)
FINAL_SEAL_PATH = Path(FINAL_TEST_SEAL_PATH)
FINAL_OUTPUT_PATH = Path(FINAL_TEST_OUTPUT_PATH)
SESSION_PLAN_PATH = Path(FINAL_SESSION_PATH)
EVIDENCE_PATH = Path(FINAL_EVIDENCE_PATH)
QUALITY_LOCK_PATH = Path(FINAL_QUALITY_LOCK_PATH)
ACTIVE_SENTINEL = Path(FINAL_ARTIFACT_ROOT) / ".active"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _npz_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, sequence_nll_nats=array)
    return output.getvalue()


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
        digest.update(array.tobytes(order="C"))
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


def _git_path_history(path: Path) -> tuple[str, ...]:
    output = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return tuple(line for line in output.splitlines() if line)


def _require_not_deleted_tracked_artifact(path: Path) -> None:
    if not path.exists() and _git_path_history(path):
        raise ValueError(f"final evaluation artifact was published then deleted: {path}")


def _require_clean_root() -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(root).resolve() != Path.cwd().resolve() or _git_status().strip():
        raise ValueError("final evaluation requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("final evaluation requires a Git commit")
    return commit


def _tracked_head_identity(path: Path) -> dict[str, str]:
    blob = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-list", "-1", "HEAD", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    commit_value = commit.stdout.strip()
    if (
        blob.returncode != 0
        or commit.returncode != 0
        or len(commit_value) != 40
        or not path.is_file()
        or path.read_bytes() != blob.stdout
    ):
        raise ValueError(f"final evaluation input is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit_value,
        "sha256": hashlib.sha256(blob.stdout).hexdigest(),
    }


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"final evaluation Git order differs: {label}")


def _runtime_identity() -> dict[str, Any]:
    versions = research_versions()
    return {
        "batch_size": FINAL_EVALUATION_BATCH_SIZE,
        "device": "mps",
        "mps_available": versions["mps_available"],
        "numpy": versions["numpy"],
        "python": platform.python_version(),
        "torch": versions["python_torch"],
        "transformers": versions["transformers"],
    }


def _verify_implementation(authorization: Mapping[str, Any]) -> None:
    for path in IMPLEMENTATION_FILE_ORDER:
        identity = _tracked_head_identity(Path(path))
        if identity["sha256"] != authorization["implementation_sha256"][path]:
            raise ValueError(f"final evaluator implementation differs: {path}")


def _load_authorized_context() -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    evaluator_commit = _require_clean_root()
    _require_not_deleted_tracked_artifact(EVIDENCE_PATH)
    _require_not_deleted_tracked_artifact(QUALITY_LOCK_PATH)
    if QUALITY_LOCK_PATH.exists():
        raise ValueError("final quality lock already exists; evaluator is immutable")
    authorization_identity = _tracked_head_identity(AUTHORIZATION_PATH)
    lock_identity = _tracked_head_identity(LOCK_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)
    selection_lock = _read_json(LOCK_PATH)
    validate_selection_lock_v2(selection_lock)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    if (
        authorization["upstream_artifacts"]["selection_lock"]["sha256"]
        != lock_identity["sha256"]
    ):
        raise ValueError("final authorization and selection-lock blobs differ")
    _require_ancestor(
        authorization["authorization_git_commit"],
        authorization_identity["git_commit"],
        "authorization base -> authorization artifact",
    )
    _require_ancestor(
        authorization_identity["git_commit"],
        evaluator_commit,
        "authorization artifact -> evaluator",
    )
    for section in (
        authorization["upstream_artifacts"].values(),
        (
            authorization["confirmation_evidence"]["artifact"],
            authorization["confirmation_evidence"][
                "historical_primary_phase3_provenance"
            ]["artifact"],
            *(
                completion["artifact"]
                for completion in authorization["confirmation_evidence"][
                    "training_completions"
                ].values()
            ),
        ),
        (
            authorization["final_test"]["manifest"],
            authorization["final_test"]["seal"],
        ),
    ):
        for artifact in section:
            _require_ancestor(
                artifact["git_commit"],
                authorization_identity["git_commit"],
                artifact["path"],
            )
            current = _tracked_head_identity(Path(artifact["path"]))
            if (
                current["sha256"] != artifact["sha256"]
                or current["git_commit"] != artifact["git_commit"]
            ):
                raise ValueError(
                    f"final authorization dependency changed: {artifact['path']}"
                )
    _verify_implementation(authorization)
    return (
        authorization,
        authorization_identity["sha256"],
        selection_lock,
        evaluator_commit,
    )


def _load_final_stream(
    authorization: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    final_seal = _read_json(FINAL_SEAL_PATH)
    validate_seal_envelope(final_seal)
    sealed_output = final_seal["payload"]["output"]
    if (
        hash_file(FINAL_MANIFEST_PATH)
        != authorization["final_test"]["manifest"]["sha256"]
        or hash_file(FINAL_SEAL_PATH)
        != authorization["final_test"]["seal"]["sha256"]
        or hash_file(FINAL_OUTPUT_PATH)
        != authorization["final_test"]["output_jsonl"]["sha256"]
        or final_seal["payload_sha256"]
        != authorization["final_test"]["seal_payload_sha256"]
        or final_seal["payload"]["manifest"]["sha256"]
        != authorization["final_test"]["manifest"]["sha256"]
        or sealed_output["full_jsonl_sha256"]
        != authorization["final_test"]["output_jsonl"]["sha256"]
        or sealed_output["evaluation_stream_sha256"]
        != authorization["final_test"]["evaluation_stream_sha256"]
        or sealed_output["evaluation_stream_bytes"] != 32_000_000
        or sealed_output["sequence_count"] != 62_500
        or sealed_output["sequence_length"] != 512
    ):
        raise ValueError("final-test manifest/seal/output differs")
    stream = build_neural_stream(
        FINAL_OUTPUT_PATH,
        language="ko",
        split="test",
        byte_limit=32_000_000,
        sequence_length=512,
    )
    if (
        len(stream.data) != 32_000_000
        or stream.sequence_count != 62_500
        or hashlib.sha256(stream.data).hexdigest()
        != authorization["final_test"]["evaluation_stream_sha256"]
    ):
        raise ValueError("final evaluation stream differs from authorization")
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    matrices = structural_patch_matrices(boundaries, whitespace, spacelike)
    required_rates = {
        int(model["descriptor"]["patch_count"])
        for model in authorization["models"]
        if model["descriptor"]["model_family"] == "compute_conversion"
    }
    if not required_rates <= set(CONVERSION_RATES):
        raise ValueError("final authorization contains an unknown conversion rate")
    for rate in sorted(required_rates):
        matrices.update(
            conversion_patch_matrices(boundaries, whitespace, rate=rate)
        )
    document_map = reconstruct_document_window_map(
        FINAL_OUTPUT_PATH,
        split="test",
        byte_limit=32_000_000,
        sequence_length=512,
        expected_stream=stream.data,
    )
    if not document_map.coverage_pass:
        raise ValueError("final document-cluster coverage is below protocol minimum")
    metadata = document_map.metadata()
    final_context = {
        "boundaries_sha256": _array_sha256(boundaries),
        "document_assignment_sha256": metadata[
            "document_assignment_sha256"
        ],
        "document_layout_sha256": document_map.layout_sha256,
        "eligible_sequence_count": document_map.eligible_sequence_count,
        "inputs_sha256": _array_sha256(inputs),
        "stream_bytes": len(stream.data),
        "stream_sha256": hashlib.sha256(stream.data).hexdigest(),
    }
    return inputs, boundaries, matrices, final_context


def _load_or_create_session_plan(
    *,
    authorization: Mapping[str, Any],
    authorization_artifact_sha256: str,
    authorization_git_commit: str,
    selection_lock: Mapping[str, Any],
    evaluator_git_commit: str,
    final_context: Mapping[str, Any],
) -> dict[str, Any]:
    if SESSION_PLAN_PATH.exists():
        existing = _read_json(SESSION_PLAN_PATH)
        validate_final_quality_session_plan(
            existing,
            authorization=authorization,
            selection_lock=selection_lock,
        )
        _require_ancestor(
            existing["evaluator_git_commit"],
            evaluator_git_commit,
            "session evaluator -> current evaluator",
        )
        if (
            existing["authorization"]["artifact_sha256"]
            != authorization_artifact_sha256
            or existing["authorization"]["git_commit"]
            != authorization_git_commit
            or existing["final_context"] != final_context
            or existing["runtime"] != _runtime_identity()
        ):
            raise ValueError("existing final session plan differs")
        return existing
    _require_not_deleted_tracked_artifact(EVIDENCE_PATH)
    if EVIDENCE_PATH.exists():
        raise ValueError("final evidence exists before its session plan")
    artifact_root = Path(FINAL_ARTIFACT_ROOT)
    if artifact_root.exists() and any(artifact_root.rglob("*")):
        raise ValueError("final artifacts exist before their session plan")
    plan = build_final_quality_session_plan(
        authorization=authorization,
        authorization_artifact_sha256=authorization_artifact_sha256,
        authorization_git_commit=authorization_git_commit,
        selection_lock=selection_lock,
        evaluator_git_commit=evaluator_git_commit,
        runtime=_runtime_identity(),
        final_context=final_context,
    )
    publish_no_clobber(SESSION_PLAN_PATH, _json_bytes(plan))
    return plan


def _model_spec(descriptor: Mapping[str, Any]):
    if descriptor["model_family"] == "phase3":
        return PHASE3_MODEL_SPEC
    return conversion_model_spec(int(descriptor["patch_count"]))


def _release(model: Any) -> None:
    model.to("cpu")
    del model
    gc.collect()
    torch.mps.empty_cache()


def _load_main_model(
    model_identity: Mapping[str, Any],
    seed: int,
) -> Any:
    build_final_model_identity(
        artifact_role=model_identity["artifact_role"],
        descriptor=model_identity["descriptor"],
        seed_evidence={
            item_seed: model_identity["seeds"][str(item_seed)]
            for item_seed in FINAL_SEEDS
        },
        parameter_count=model_identity["parameter_count"],
    )
    evidence = model_identity["seeds"][str(seed)]
    checkpoint = evidence["checkpoint"]
    if hash_file(Path(checkpoint["path"])) != checkpoint["artifact_sha256"]:
        raise ValueError(f"final checkpoint artifact differs: {seed}")
    model = build_main_model(
        _model_spec(model_identity["descriptor"]),
        seed=seed,
        global_max_position_embeddings=1_032,
    )
    model.load_state_dict(
        torch.load(Path(checkpoint["path"]), map_location="cpu", weights_only=True)
    )
    if (
        parameter_count(model) != FINAL_MAIN_PARAMETER_COUNT
        or _state_sha256(model) != checkpoint["state_sha256"]
    ):
        raise ValueError(f"final checkpoint state differs: {seed}")
    return model


def _router_matrix(
    *,
    model_identity: Mapping[str, Any],
    seed: int,
    inputs: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    auxiliary = model_identity["seeds"][str(seed)]["auxiliary"]
    for path_key, hash_key in (
        ("router_checkpoint_path", "router_checkpoint_artifact_sha256"),
        ("router_report_path", "router_report_artifact_sha256"),
        ("threshold_cache_path", "threshold_cache_artifact_sha256"),
        (
            "threshold_diagnostics_path",
            "threshold_diagnostics_artifact_sha256",
        ),
    ):
        if hash_file(Path(auxiliary[path_key])) != auxiliary[hash_key]:
            raise ValueError(f"final entropy artifact differs: {seed}/{path_key}")
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    router.load_state_dict(
        torch.load(
            Path(auxiliary["router_checkpoint_path"]),
            map_location="cpu",
            weights_only=True,
        )
    )
    if (
        parameter_count(router) != FINAL_ROUTER_PARAMETER_COUNT
        or _state_sha256(router)
        != auxiliary["router_checkpoint_state_sha256"]
    ):
        raise ValueError(f"final entropy router state differs: {seed}")
    scores = router_entropy_scores(
        router,
        inputs,
        "mps",
        batch_size=128,
    )
    _release(router)
    candidate_mask = (
        None if auxiliary["candidate_mask"] == "none" else boundaries
    )
    matrix = threshold_patch_matrix(
        scores,
        auxiliary["threshold_nats"],
        candidate_masks=candidate_mask,
        maximum_patch_length=auxiliary["maximum_patch_length"],
    )
    execution = {
        "final_matrix_sha256": _array_sha256(matrix),
        "kind": "entropy_router",
        "locked_bundle_sha256": canonical_sha256(auxiliary),
        "router_scores_sha256": _array_sha256(scores),
    }
    del scores
    return matrix, execution


def _matrix_for_unit(
    *,
    model_identity: Mapping[str, Any],
    seed: int,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    structural_matrices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    descriptor = model_identity["descriptor"]
    if descriptor["requires_entropy_router"]:
        matrix, auxiliary = _router_matrix(
            model_identity=model_identity,
            seed=seed,
            inputs=inputs,
            boundaries=boundaries,
        )
    else:
        policy = descriptor["policy"]
        if policy not in structural_matrices:
            raise ValueError(f"final structural matrix is missing: {policy}")
        matrix = structural_matrices[policy]
        auxiliary = {"kind": "none"}
    if matrix.shape != (62_500, int(descriptor["patch_count"]) + 1):
        raise ValueError("final patch matrix shape differs from model geometry")
    validate_padded_patch_matrix(matrix, 512)
    return matrix, auxiliary


def _load_nll(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError("final NLL archive keys differ")
        values = archive["sequence_nll_nats"]
    if (
        values.dtype != np.float32
        or values.shape != (62_500,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("final NLL array is malformed")
    return values


def _validate_completed_unit(
    *,
    receipt_path: Path,
    nll_path: Path,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
    matrix_sha256: str,
    auxiliary_execution: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]] | None:
    if receipt_path.exists() != nll_path.exists():
        raise ValueError(f"partial final unit exists: {receipt_path.parent}")
    for path in (
        receipt_path.with_suffix(receipt_path.suffix + ".part"),
        nll_path.with_suffix(nll_path.suffix + ".part"),
    ):
        if path.exists():
            raise ValueError(f"staged final unit requires forensic review: {path}")
    if not receipt_path.exists():
        return None
    receipt = _read_json(receipt_path)
    validate_final_quality_receipt(
        receipt,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
    )
    values = _load_nll(nll_path)
    reconstructed_bpb = math.fsum(float(value) for value in values) / (
        len(values) * 511 * math.log(2.0)
    )
    if (
        receipt["patch_matrix_sha256"] != matrix_sha256
        or receipt["auxiliary_execution"] != auxiliary_execution
        or receipt["nll"]["artifact_sha256"] != hash_file(nll_path)
        or receipt["nll"]["array_sha256"] != _array_sha256(values)
        or not math.isclose(
            receipt["nll"]["bpb"],
            reconstructed_bpb,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("completed final unit evidence differs")
    return receipt, {
        "path": receipt_path.as_posix(),
        "sha256": hash_file(receipt_path),
    }


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_unit_pair(
    *,
    nll_path: Path,
    nll_bytes: bytes,
    receipt_path: Path,
    receipt_bytes: bytes,
) -> None:
    nll_stage = nll_path.with_suffix(nll_path.suffix + ".part")
    receipt_stage = receipt_path.with_suffix(receipt_path.suffix + ".part")
    if any(path.exists() for path in (nll_path, receipt_path, nll_stage, receipt_stage)):
        raise FileExistsError("final unit publish target already exists")
    _exclusive_write(nll_stage, nll_bytes)
    _exclusive_write(receipt_stage, receipt_bytes)
    if nll_stage.read_bytes() != nll_bytes or receipt_stage.read_bytes() != receipt_bytes:
        raise IOError("staged final unit differs before publish")
    os.link(nll_stage, nll_path)
    os.link(receipt_stage, receipt_path)
    _fsync_directory(nll_path.parent)
    nll_stage.unlink()
    receipt_stage.unlink()
    _fsync_directory(nll_path.parent)


def _evaluate_unit(
    *,
    unit_index: int,
    artifact_role: str,
    seed: int,
    authorization: Mapping[str, Any],
    authorization_artifact_sha256: str,
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
    inputs: np.ndarray,
    boundaries: np.ndarray,
    structural_matrices: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, str]]:
    model_identity = next(
        model
        for model in authorization["models"]
        if model["artifact_role"] == artifact_role
    )
    model = _load_main_model(model_identity, seed)
    matrix, auxiliary_execution = _matrix_for_unit(
        model_identity=model_identity,
        seed=seed,
        inputs=inputs,
        boundaries=boundaries,
        structural_matrices=structural_matrices,
    )
    matrix_sha256 = _array_sha256(matrix)
    paths = expected_final_evidence_paths(artifact_role, seed)
    receipt_path = Path(paths["receipt"])
    nll_path = Path(paths["nll"])
    completed = _validate_completed_unit(
        receipt_path=receipt_path,
        nll_path=nll_path,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
        matrix_sha256=matrix_sha256,
        auxiliary_execution=auxiliary_execution,
    )
    if completed is not None:
        del matrix
        _release(model)
        return completed
    _, values = evaluate_main_model(
        model,
        inputs,
        matrix,
        "mps",
        batch_size=FINAL_EVALUATION_BATCH_SIZE,
        return_sequence_nll=True,
    )
    if values is None:
        raise AssertionError("final evaluation did not return sequence NLL")
    values = np.ascontiguousarray(values, dtype=np.float32)
    if (
        values.shape != (62_500,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("new final NLL array is malformed")
    nll_bytes = _npz_bytes(values)
    bpb = math.fsum(float(value) for value in values) / (
        len(values) * 511 * math.log(2.0)
    )
    receipt = build_final_quality_receipt(
        authorization=authorization,
        authorization_artifact_sha256=authorization_artifact_sha256,
        selection_lock=selection_lock,
        session_plan=session_plan,
        unit_index=unit_index,
        artifact_role=artifact_role,
        seed=seed,
        patch_matrix_sha256=matrix_sha256,
        auxiliary_execution=auxiliary_execution,
        nll={
            "array_sha256": _array_sha256(values),
            "artifact_path": nll_path.as_posix(),
            "artifact_sha256": hashlib.sha256(nll_bytes).hexdigest(),
            "bpb": bpb,
            "count": len(values),
            "dtype": "float32",
            "predicted_bytes": len(values) * 511,
        },
    )
    _publish_unit_pair(
        nll_path=nll_path,
        nll_bytes=nll_bytes,
        receipt_path=receipt_path,
        receipt_bytes=_json_bytes(receipt),
    )
    del matrix, values
    _release(model)
    validated = _validate_completed_unit(
        receipt_path=receipt_path,
        nll_path=nll_path,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
        matrix_sha256=matrix_sha256,
        auxiliary_execution=auxiliary_execution,
    )
    if validated is None:
        raise AssertionError("published final unit was not recovered")
    return validated


def _active_sentinel_bytes(session_plan: Mapping[str, Any]) -> bytes:
    return _json_bytes(
        {
            "evaluator_git_commit": session_plan["evaluator_git_commit"],
            "session_id": session_plan["session_id"],
            "session_plan_sha256": session_plan["session_plan_sha256"],
        }
    )


def _unit_artifact_paths(
    authorization: Mapping[str, Any],
) -> set[Path]:
    paths: set[Path] = set()
    for _, artifact_role, seed in authorized_unit_order(authorization):
        unit = expected_final_evidence_paths(artifact_role, seed)
        for value in unit.values():
            path = Path(value)
            paths.add(path)
            paths.add(path.with_suffix(path.suffix + ".part"))
    return paths


def _require_unsymlinked_path_within_root(path: Path, root: Path) -> None:
    if path.is_absolute() != root.is_absolute():
        raise ValueError("final artifact path/root forms differ")
    if root.is_absolute():
        root_absolute = root
        path_absolute = path
        cursor = root_absolute
        root_parts: tuple[str, ...] = ()
    else:
        root_absolute = Path.cwd() / root
        path_absolute = Path.cwd() / path
        cursor = Path.cwd()
        root_parts = root.parts
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError("final artifact path escapes its canonical root") from exc
    if cursor.is_symlink():
        raise ValueError("final artifact path contains a symlink")
    for component in (*root_parts, *relative.parts):
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("final artifact path contains a symlink")
    root_resolved = root_absolute.resolve(strict=False)
    path_resolved = path_absolute.resolve(strict=False)
    if not path_resolved.is_relative_to(root_resolved):
        raise ValueError("final artifact path resolves outside its canonical root")


@contextmanager
def _exclusive_session_process_lock():
    root = Path(FINAL_ARTIFACT_ROOT)
    _require_unsymlinked_path_within_root(SESSION_PLAN_PATH, root)
    handle = SESSION_PLAN_PATH.open("rb")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another final evaluator or verifier owns this session"
            ) from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _validate_artifact_namespace(authorization: Mapping[str, Any]) -> None:
    root = Path(FINAL_ARTIFACT_ROOT)
    if not root.exists():
        return
    allowed = {
        SESSION_PLAN_PATH,
        ACTIVE_SENTINEL,
        *_unit_artifact_paths(authorization),
    }
    entries = set(root.rglob("*"))
    actual = {path for path in entries if not path.is_dir()}
    for path in {root, *allowed, *entries}:
        _require_unsymlinked_path_within_root(path, root)
    if any(path.is_symlink() for path in entries) or not actual <= allowed:
        raise ValueError("final evaluation artifact namespace is not canonical")


def _start_or_resume_active_session(
    session_plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> None:
    expected = _active_sentinel_bytes(session_plan)
    if ACTIVE_SENTINEL.exists():
        if ACTIVE_SENTINEL.is_symlink() or ACTIVE_SENTINEL.read_bytes() != expected:
            raise ValueError("final evaluation active sentinel differs")
        return
    if any(path.exists() for path in _unit_artifact_paths(authorization)):
        raise ValueError("final unit artifacts exist without their active session")
    _exclusive_write(ACTIVE_SENTINEL, expected)


def _validate_existing_evidence(
    *,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    session_plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not EVIDENCE_PATH.exists():
        return None
    evidence = _read_json(EVIDENCE_PATH)
    validate_final_quality_evidence_manifest(
        evidence,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
    )
    for receipt, artifact in zip(
        evidence["receipts"],
        evidence["receipt_artifacts"],
        strict=True,
    ):
        receipt_path = Path(artifact["path"])
        if (
            hash_file(receipt_path) != artifact["sha256"]
            or _read_json(receipt_path) != receipt
        ):
            raise ValueError("final receipt artifact differs from evidence manifest")
        nll_path = Path(receipt["nll"]["artifact_path"])
        values = _load_nll(nll_path)
        reconstructed_bpb = math.fsum(float(value) for value in values) / (
            len(values) * 511 * math.log(2.0)
        )
        if (
            hash_file(nll_path) != receipt["nll"]["artifact_sha256"]
            or _array_sha256(values) != receipt["nll"]["array_sha256"]
            or not math.isclose(
                receipt["nll"]["bpb"],
                reconstructed_bpb,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("final NLL artifact differs from evidence manifest")
    return evidence


def _post_evaluation_status_is_clean() -> bool:
    lines = [line for line in _git_status().splitlines() if line.strip()]
    allowed = {f"?? {EVIDENCE_PATH.as_posix()}"}
    return not lines or set(lines) <= allowed


def _run_locked_session(
    *,
    authorization: Mapping[str, Any],
    authorization_artifact_sha256: str,
    selection_lock: Mapping[str, Any],
    evaluator_commit: str,
    session_plan: Mapping[str, Any],
    inputs: np.ndarray,
    boundaries: np.ndarray,
    structural_matrices: Mapping[str, np.ndarray],
) -> int:
    with _exclusive_session_process_lock():
        _validate_artifact_namespace(authorization)
        existing = _validate_existing_evidence(
            authorization=authorization,
            selection_lock=selection_lock,
            session_plan=session_plan,
        )
        if existing is None or ACTIVE_SENTINEL.exists():
            _start_or_resume_active_session(session_plan, authorization)
        receipts: list[dict[str, Any]] = []
        receipt_artifacts: list[dict[str, str]] = []
        for unit_index, artifact_role, seed in authorized_unit_order(authorization):
            receipt, artifact = _evaluate_unit(
                unit_index=unit_index,
                artifact_role=artifact_role,
                seed=seed,
                authorization=authorization,
                authorization_artifact_sha256=authorization_artifact_sha256,
                selection_lock=selection_lock,
                session_plan=session_plan,
                inputs=inputs,
                boundaries=boundaries,
                structural_matrices=structural_matrices,
            )
            receipts.append(receipt)
            receipt_artifacts.append(artifact)
            print(
                json.dumps(
                    {
                        "artifact_role": artifact_role,
                        "receipt_sha256": receipt["receipt_sha256"],
                        "seed": seed,
                        "status": "unit_complete_no_metric_printed",
                        "unit_index": unit_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if existing is not None:
            if (
                receipts != existing["receipts"]
                or receipt_artifacts != existing["receipt_artifacts"]
            ):
                raise ValueError(
                    "completed final evidence differs after reconstruction"
                )
            if (
                _git_commit() != evaluator_commit
                or not _post_evaluation_status_is_clean()
            ):
                raise RuntimeError(
                    "repository changed while validating sealed final evidence"
                )
            if ACTIVE_SENTINEL.exists():
                ACTIVE_SENTINEL.unlink()
                _fsync_directory(ACTIVE_SENTINEL.parent)
            print(
                json.dumps(
                    {
                        "manifest_sha256": existing["manifest_sha256"],
                        "status": "complete_validated_no_rerun",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        evidence = build_final_quality_evidence_manifest(
            authorization=authorization,
            authorization_artifact_sha256=authorization_artifact_sha256,
            selection_lock=selection_lock,
            session_plan=session_plan,
            receipts=receipts,
            receipt_artifacts=receipt_artifacts,
        )
        publish_no_clobber(EVIDENCE_PATH, _json_bytes(evidence))
        validate_final_quality_evidence_manifest(
            _read_json(EVIDENCE_PATH),
            authorization=authorization,
            selection_lock=selection_lock,
            session_plan=session_plan,
        )
        if _git_commit() != evaluator_commit or not _post_evaluation_status_is_clean():
            raise RuntimeError("repository changed during sealed final evaluation")
        ACTIVE_SENTINEL.unlink()
        _fsync_directory(ACTIVE_SENTINEL.parent)
        print(
            json.dumps(
                {
                    "manifest_sha256": evidence["manifest_sha256"],
                    "output": EVIDENCE_PATH.as_posix(),
                    "status": "complete_pending_immutable_quality_lock",
                    "unit_count": len(receipts),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0


def main() -> int:
    if resolve_device("auto") != "mps":
        raise RuntimeError("sealed final evaluation requires Apple MPS")
    (
        authorization,
        authorization_artifact_sha256,
        selection_lock,
        evaluator_commit,
    ) = _load_authorized_context()
    inputs, boundaries, structural_matrices, final_context = _load_final_stream(
        authorization
    )
    authorization_commit = _tracked_head_identity(AUTHORIZATION_PATH)["git_commit"]
    session_plan = _load_or_create_session_plan(
        authorization=authorization,
        authorization_artifact_sha256=authorization_artifact_sha256,
        authorization_git_commit=authorization_commit,
        selection_lock=selection_lock,
        evaluator_git_commit=evaluator_commit,
        final_context=final_context,
    )
    return _run_locked_session(
        authorization=authorization,
        authorization_artifact_sha256=authorization_artifact_sha256,
        selection_lock=selection_lock,
        evaluator_commit=evaluator_commit,
        session_plan=session_plan,
        inputs=inputs,
        boundaries=boundaries,
        structural_matrices=structural_matrices,
    )


if __name__ == "__main__":
    raise SystemExit(main())
