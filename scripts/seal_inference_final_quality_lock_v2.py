#!/usr/bin/env python3
"""Reconstruct sealed-final arrays and publish the immutable quality lock."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch

from jamoflow.compute_conversion import (
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
)
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber, validate_seal_envelope
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_ARTIFACT_ROOT,
    FINAL_EVIDENCE_PATH,
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_ROUTER_PARAMETER_COUNT,
    FINAL_TEST_MANIFEST_PATH,
    FINAL_TEST_OUTPUT_PATH,
    FINAL_TEST_SEAL_PATH,
    SELECTION_LOCK_PATH,
    canonical_sha256,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_final_quality_evidence_v2 import (
    FINAL_EVALUATION_BATCH_SIZE,
    FINAL_SESSION_PATH,
    validate_final_quality_evidence_manifest,
    validate_final_quality_receipt,
    validate_final_quality_session_plan,
)
from jamoflow.inference_final_quality_lock_v2 import (
    build_final_quality_lock_v2,
    validate_final_quality_lock_v2,
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
from jamoflow.publication_reference import entropy_policy_definition_sha256


AUTHORIZATION_PATH = Path(FINAL_AUTHORIZATION_PATH)
SELECTION_PATH = Path(SELECTION_LOCK_PATH)
SESSION_PATH = Path(FINAL_SESSION_PATH)
EVIDENCE_PATH = Path(FINAL_EVIDENCE_PATH)
OUTPUT_PATH = Path(FINAL_QUALITY_LOCK_PATH)
FINAL_OUTPUT_PATH = Path(FINAL_TEST_OUTPUT_PATH)
FINAL_SEAL_PATH = Path(FINAL_TEST_SEAL_PATH)
FINAL_MANIFEST_PATH = Path(FINAL_TEST_MANIFEST_PATH)
ACTIVE_SENTINEL = Path(FINAL_ARTIFACT_ROOT) / ".active"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _require_never_published(path: Path) -> None:
    if path.exists() or _git_path_history(path):
        raise ValueError(f"canonical final quality lock already exists or was deleted: {path}")


def _require_clean_root() -> str:
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(root).resolve() != Path.cwd().resolve() or _git_status().strip():
        raise ValueError("final quality lock requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("final quality lock requires a Git commit")
    return commit


def _tracked_identity(path: Path) -> dict[str, str]:
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
        raise ValueError(f"final quality input is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit_value,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(blob.stdout).hexdigest(),
    }


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"final quality Git order differs: {label}")


def _verify_implementation(authorization: dict[str, Any]) -> None:
    for path in authorization["implementation_sha256"]:
        identity = _tracked_identity(Path(path))
        if identity["sha256"] != authorization["implementation_sha256"][path]:
            raise ValueError(f"final quality implementation differs: {path}")


def _require_unsymlinked_path_within_root(path: Path, root: Path) -> None:
    if path.is_absolute() != root.is_absolute():
        raise ValueError("final quality artifact path/root forms differ")
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
        raise ValueError(
            "final quality artifact path escapes its canonical root"
        ) from exc
    if cursor.is_symlink():
        raise ValueError("final quality artifact path contains a symlink")
    for component in (*root_parts, *relative.parts):
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("final quality artifact path contains a symlink")
    root_resolved = root_absolute.resolve(strict=False)
    path_resolved = path_absolute.resolve(strict=False)
    if not path_resolved.is_relative_to(root_resolved):
        raise ValueError(
            "final quality artifact resolves outside its canonical root"
        )


@contextmanager
def _exclusive_session_process_lock():
    root = Path(FINAL_ARTIFACT_ROOT)
    _require_unsymlinked_path_within_root(SESSION_PATH, root)
    handle = SESSION_PATH.open("rb")
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


def _release(model: Any) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _model_spec(descriptor: dict[str, Any]):
    if descriptor["model_family"] == "phase3":
        return PHASE3_MODEL_SPEC
    return conversion_model_spec(int(descriptor["patch_count"]))


def _runtime_identity() -> dict[str, Any]:
    versions = research_versions()
    return {
        "batch_size": 64,
        "device": "mps",
        "mps_available": versions["mps_available"],
        "numpy": versions["numpy"],
        "python": platform.python_version(),
        "torch": versions["python_torch"],
        "transformers": versions["transformers"],
    }


def _require_completed_session(
    session_plan: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    if (
        ACTIVE_SENTINEL.exists()
        or session_plan["runtime"] != _runtime_identity()
        or resolve_device("auto") != "mps"
        or evidence["runtime"] != session_plan["runtime"]
        or evidence["final_context"] != session_plan["final_context"]
        or evidence["session_plan"]["sha256"]
        != session_plan["session_plan_sha256"]
    ):
        raise ValueError("final quality session is incomplete or differs")


def _require_canonical_artifact_namespace(evidence: dict[str, Any]) -> None:
    root = Path(FINAL_ARTIFACT_ROOT)
    expected = {SESSION_PATH}
    for receipt, artifact in zip(
        evidence["receipts"],
        evidence["receipt_artifacts"],
        strict=True,
    ):
        expected.add(Path(artifact["path"]))
        expected.add(Path(receipt["nll"]["artifact_path"]))
    entries = set(root.rglob("*"))
    actual = {path for path in entries if not path.is_dir()}
    for path in {root, *expected, *entries}:
        _require_unsymlinked_path_within_root(path, root)
    if (
        any(path.is_symlink() for path in entries)
        or actual != expected
        or any(path.suffix == ".part" for path in actual)
    ):
        raise ValueError("final quality artifact namespace is not complete")


def _load_nll(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError("final quality NLL archive keys differ")
        values = archive["sequence_nll_nats"]
    if (
        values.dtype != np.float32
        or values.shape != (62_500,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("final quality NLL array is malformed")
    return values


def _reconstruct_final_context(
    authorization: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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
        != evidence["final_context"]["stream_sha256"]
        or sealed_output["evaluation_stream_bytes"] != 32_000_000
        or sealed_output["sequence_count"] != 62_500
        or sealed_output["sequence_length"] != 512
    ):
        raise ValueError("final quality sealed stream identity differs")
    stream = build_neural_stream(
        FINAL_OUTPUT_PATH,
        language="ko",
        split="test",
        byte_limit=32_000_000,
        sequence_length=512,
    )
    if (
        len(stream.data) != 32_000_000
        or hashlib.sha256(stream.data).hexdigest()
        != evidence["final_context"]["stream_sha256"]
    ):
        raise ValueError("final quality stream bytes differ")
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    if (
        _array_sha256(inputs) != evidence["final_context"]["inputs_sha256"]
        or _array_sha256(boundaries)
        != evidence["final_context"]["boundaries_sha256"]
    ):
        raise ValueError("final quality stream arrays differ")
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    matrices = structural_patch_matrices(boundaries, whitespace, spacelike)
    required_rates = {
        int(model["descriptor"]["patch_count"])
        for model in authorization["models"]
        if model["descriptor"]["model_family"] == "compute_conversion"
    }
    if not required_rates <= set(CONVERSION_RATES):
        raise ValueError("final quality authorization has an unknown rate")
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
    return document_map, inputs, boundaries, matrices


def _load_verified_main_checkpoint(
    model_identity: dict[str, Any],
    seed: int,
) -> Any:
    checkpoint = model_identity["seeds"][str(seed)]["checkpoint"]
    if hash_file(Path(checkpoint["path"])) != checkpoint["artifact_sha256"]:
        raise ValueError(f"final quality checkpoint artifact differs: {seed}")
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
        or parameter_count(model) != model_identity["parameter_count"]
        or _state_sha256(model) != checkpoint["state_sha256"]
    ):
        raise ValueError(f"final quality checkpoint state differs: {seed}")
    return model


def _reconstruct_entropy_execution(
    *,
    model_identity: dict[str, Any],
    seed: int,
    inputs: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    auxiliary = model_identity["seeds"][str(seed)]["auxiliary"]
    if (
        auxiliary["router_config_sha256"]
        != canonical_sha256(PHASE3_MODEL_SPEC.to_dict())
        or auxiliary["policy_definition_sha256"]
        != entropy_policy_definition_sha256(model_identity["descriptor"]["policy"])
    ):
        raise ValueError("final quality entropy config/policy identity differs")
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
            raise ValueError(
                f"final quality entropy artifact differs: {seed}/{path_key}"
            )
    if resolve_device("auto") != "mps":
        raise RuntimeError("entropy final-quality verification requires Apple MPS")
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
        or parameter_count(router) != auxiliary["router_parameter_count"]
        or _state_sha256(router)
        != auxiliary["router_checkpoint_state_sha256"]
    ):
        raise ValueError(f"final quality entropy router state differs: {seed}")
    scores = router_entropy_scores(router, inputs, "mps", batch_size=128)
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


def _recompute_verified_nll_arrays(
    *,
    authorization: dict[str, Any],
    evidence: dict[str, Any],
    inputs: np.ndarray,
    boundaries: np.ndarray,
    structural_matrices: dict[str, np.ndarray],
    stored_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    models = {
        model["artifact_role"]: model for model in authorization["models"]
    }
    recomputed_arrays: dict[str, np.ndarray] = {}
    for receipt in evidence["receipts"]:
        model_identity = models[receipt["artifact_role"]]
        seed = int(receipt["seed"])
        model = _load_verified_main_checkpoint(model_identity, seed)
        descriptor = model_identity["descriptor"]
        try:
            if descriptor["requires_entropy_router"]:
                matrix, execution = _reconstruct_entropy_execution(
                    model_identity=model_identity,
                    seed=seed,
                    inputs=inputs,
                    boundaries=boundaries,
                )
            else:
                policy = descriptor["policy"]
                if policy not in structural_matrices:
                    raise ValueError(
                        f"final quality structural matrix is missing: {policy}"
                    )
                matrix = structural_matrices[policy]
                execution = {"kind": "none"}
            if matrix.shape != (
                62_500,
                int(descriptor["patch_count"]) + 1,
            ):
                raise ValueError("final quality patch matrix geometry differs")
            validate_padded_patch_matrix(matrix, 512)
            if (
                _array_sha256(matrix) != receipt["patch_matrix_sha256"]
                or receipt["auxiliary_execution"] != execution
            ):
                raise ValueError(
                    "final quality patch matrix/router execution differs"
                )
            _, values = evaluate_main_model(
                model,
                inputs,
                matrix,
                "mps",
                batch_size=FINAL_EVALUATION_BATCH_SIZE,
                return_sequence_nll=True,
            )
        finally:
            _release(model)
        if values is None:
            raise AssertionError("final quality verifier returned no sequence NLL")
        raw_values = np.asarray(values)
        if raw_values.dtype != np.float32:
            raise ValueError("final quality verifier NLL dtype differs")
        values = np.ascontiguousarray(raw_values)
        receipt_sha256 = receipt["receipt_sha256"]
        if (
            values.shape != (62_500,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or receipt_sha256 not in stored_arrays
            or not np.array_equal(values, stored_arrays[receipt_sha256])
            or _array_sha256(values) != receipt["nll"]["array_sha256"]
            or receipt_sha256 in recomputed_arrays
        ):
            raise ValueError(
                "final quality NLL does not exactly match independent model forward"
            )
        recomputed_arrays[receipt_sha256] = values
    if tuple(recomputed_arrays) != tuple(stored_arrays):
        raise ValueError("final quality recomputation order/set differs")
    return recomputed_arrays


def _load_receipt_arrays(
    evidence: dict[str, Any],
    *,
    authorization: dict[str, Any],
    selection_lock: dict[str, Any],
    session_plan: dict[str, Any],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for receipt, receipt_artifact in zip(
        evidence["receipts"],
        evidence["receipt_artifacts"],
        strict=True,
    ):
        receipt_path = Path(receipt_artifact["path"])
        nll_path = Path(receipt["nll"]["artifact_path"])
        if (
            hash_file(receipt_path) != receipt_artifact["sha256"]
            or _read_json(receipt_path) != receipt
            or hash_file(nll_path) != receipt["nll"]["artifact_sha256"]
        ):
            raise ValueError("final quality receipt/NLL artifact differs")
        validate_final_quality_receipt(
            receipt,
            authorization=authorization,
            selection_lock=selection_lock,
            session_plan=session_plan,
        )
        values = _load_nll(nll_path)
        bpb = math.fsum(float(value) for value in values) / (
            len(values) * 511 * math.log(2.0)
        )
        if (
            _array_sha256(values) != receipt["nll"]["array_sha256"]
            or not math.isclose(
                bpb,
                receipt["nll"]["bpb"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            or receipt["receipt_sha256"] in arrays
        ):
            raise ValueError("final quality receipt array identity differs")
        arrays[receipt["receipt_sha256"]] = values
    return arrays


def _post_publish_status_is_clean() -> bool:
    lines = [line for line in _git_status().splitlines() if line.strip()]
    allowed = {f"?? {OUTPUT_PATH.as_posix()}"}
    return not lines or set(lines) <= allowed


def _seal_locked_quality(
    *,
    base_commit: str,
    authorization_artifact: dict[str, str],
    selection_artifact: dict[str, str],
    evidence_artifact: dict[str, str],
    authorization: dict[str, Any],
    selection_lock: dict[str, Any],
    session_plan: dict[str, Any],
    evidence: dict[str, Any],
) -> int:
    with _exclusive_session_process_lock():
        _require_completed_session(session_plan, evidence)
        _require_canonical_artifact_namespace(evidence)
        if (
            authorization["upstream_artifacts"]["selection_lock"]["sha256"]
            != selection_artifact["sha256"]
            or evidence["authorization"]["artifact_sha256"]
            != authorization_artifact["sha256"]
        ):
            raise ValueError("final quality tracked dependency differs")
        _require_ancestor(
            authorization_artifact["git_commit"],
            evidence_artifact["git_commit"],
            "authorization -> final evidence",
        )
        _require_ancestor(
            evidence_artifact["git_commit"],
            base_commit,
            "final evidence -> quality lock base",
        )
        _require_ancestor(
            session_plan["evaluator_git_commit"],
            evidence_artifact["git_commit"],
            "session evaluator -> final evidence",
        )
        arrays = _load_receipt_arrays(
            evidence,
            authorization=authorization,
            selection_lock=selection_lock,
            session_plan=session_plan,
        )
        (
            document_map,
            inputs,
            boundaries,
            structural_matrices,
        ) = _reconstruct_final_context(authorization, evidence)
        arrays = _recompute_verified_nll_arrays(
            authorization=authorization,
            evidence=evidence,
            inputs=inputs,
            boundaries=boundaries,
            structural_matrices=structural_matrices,
            stored_arrays=arrays,
        )
        quality_lock = build_final_quality_lock_v2(
            authorization=authorization,
            authorization_artifact=authorization_artifact,
            selection_lock=selection_lock,
            session_plan=session_plan,
            evidence=evidence,
            evidence_artifact=evidence_artifact,
            quality_lock_base_git_commit=base_commit,
            document_window_map=document_map,
            arrays_by_receipt_sha256=arrays,
        )
        validate_final_quality_lock_v2(
            quality_lock,
            authorization=authorization,
            selection_lock=selection_lock,
            session_plan=session_plan,
            evidence=evidence,
            document_window_map=document_map,
            arrays_by_receipt_sha256=arrays,
        )
        serialized = (
            json.dumps(
                quality_lock,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        publish_no_clobber(OUTPUT_PATH, serialized)
        if _git_commit() != base_commit or not _post_publish_status_is_clean():
            raise RuntimeError("repository changed while sealing final quality")
        print(
            json.dumps(
                {
                    "output": OUTPUT_PATH.as_posix(),
                    "primary_publication_timing_authorized": quality_lock[
                        "primary_publication_timing_authorized"
                    ],
                    "quality_lock_sha256": quality_lock["quality_lock_sha256"],
                    "status": quality_lock["status"],
                    "timing_authorizations": {
                        key: value["authorized"]
                        for key, value in quality_lock[
                            "timing_authorizations"
                        ].items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0


def main() -> int:
    base_commit = _require_clean_root()
    _require_never_published(OUTPUT_PATH)
    authorization_artifact = _tracked_identity(AUTHORIZATION_PATH)
    selection_artifact = _tracked_identity(SELECTION_PATH)
    evidence_artifact = _tracked_identity(EVIDENCE_PATH)
    if len(_git_path_history(EVIDENCE_PATH)) != 1:
        raise ValueError("final evidence must have exactly one tracked publication commit")
    authorization = _read_json(AUTHORIZATION_PATH)
    selection_lock = _read_json(SELECTION_PATH)
    session_plan = _read_json(SESSION_PATH)
    evidence = _read_json(EVIDENCE_PATH)
    validate_selection_lock_v2(selection_lock)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    _verify_implementation(authorization)
    validate_final_quality_session_plan(
        session_plan,
        authorization=authorization,
        selection_lock=selection_lock,
    )
    validate_final_quality_evidence_manifest(
        evidence,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
    )
    return _seal_locked_quality(
        base_commit=base_commit,
        authorization_artifact=authorization_artifact,
        selection_artifact=selection_artifact,
        evidence_artifact=evidence_artifact,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
        evidence=evidence,
    )


if __name__ == "__main__":
    raise SystemExit(main())
