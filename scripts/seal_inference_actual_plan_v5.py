#!/usr/bin/env python3
"""Seal the fixed five-session publication actual-inference plan."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber, validate_seal_envelope
from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_CASE_PATH,
    ACTUAL_INFERENCE_V5_ERRATUM_PATH,
    ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
    ACTUAL_INFERENCE_V5_PLAN_PATH,
    ACTUAL_INFERENCE_V5_WARMUP_CASES,
    ACTUAL_INFERENCE_V5_MEASURED_CASES,
    array_sha256,
    assert_workspace_path_no_symlinks,
    build_actual_inference_plan_v5,
    current_runtime_environment_contract,
    validate_actual_inference_plan_v5,
)
from jamoflow.inference_benchmark import select_inference_cases
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_EVIDENCE_PATH,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_TEST_MANIFEST_PATH,
    FINAL_TEST_OUTPUT_PATH,
    FINAL_TEST_SEAL_PATH,
    SELECTION_LOCK_PATH,
    canonical_sha256,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_final_quality_evidence_v2 import (
    FINAL_SESSION_PATH,
    validate_final_quality_evidence_manifest,
    validate_final_quality_session_plan,
)
from jamoflow.inference_final_quality_lock_v2 import (
    validate_final_quality_lock_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays


AUTHORIZATION_PATH = Path(FINAL_AUTHORIZATION_PATH)
QUALITY_LOCK_PATH = Path(FINAL_QUALITY_LOCK_PATH)
SELECTION_LOCK_PATH_VALUE = Path(SELECTION_LOCK_PATH)
FINAL_MANIFEST_PATH = Path(FINAL_TEST_MANIFEST_PATH)
FINAL_SEAL_PATH = Path(FINAL_TEST_SEAL_PATH)
FINAL_OUTPUT_PATH = Path(FINAL_TEST_OUTPUT_PATH)
CASE_PATH = Path(ACTUAL_INFERENCE_V5_CASE_PATH)
PLAN_PATH = Path(ACTUAL_INFERENCE_V5_PLAN_PATH)
FINAL_EVIDENCE_PATH_VALUE = Path(FINAL_EVIDENCE_PATH)
FINAL_SESSION_PATH_VALUE = Path(FINAL_SESSION_PATH)
ERRATUM_PATH = Path(ACTUAL_INFERENCE_V5_ERRATUM_PATH)
PREDECESSOR_BACKEND_CORRECTNESS_ERRATUM_PATH = Path(
    "data/manifests/phase3-inference-actual-v5r2-backend-correctness-revision.json"
)
FAILED_V5R2_PLAN_PATH = Path("results/phase3-inference-actual-v5r2/plan.json")
FAILED_V5R2_PREFLIGHT_PATH = Path(
    "results/phase3-inference-actual-v5r2/failures/"
    "preflight-device-identity.json"
)
POST_FINAL_CORRECTNESS_REVISION_FILES = (
    "scripts/seal_inference_actual_plan_v5.py",
    "scripts/benchmark_inference_actual_v5.py",
    "scripts/measure_inference_memory_v5.py",
    "scripts/summarize_inference_actual_v5.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_actual_runtime_v5.py",
    "src/jamoflow/inference_quality.py",
    "docs/86-publication-actual-inference-v5-protocol.md",
)


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


def _npz_bytes(prompts: np.ndarray, continuations: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        prompts=prompts,
        replay_continuations=continuations,
    )
    return output.getvalue()


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
    if Path(root).resolve() != Path.cwd().resolve() or _git_status().strip():
        raise ValueError("actual-inference plan requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("actual-inference plan requires a Git commit")
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
        or path.is_symlink()
        or path.read_bytes() != blob.stdout
    ):
        raise ValueError(f"actual-inference input is not an exact HEAD blob: {path}")
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
        raise ValueError(f"actual-inference Git order differs: {label}")


def _git_blob_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"actual-inference historical blob is missing: {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _git_diff_sha256(commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary", commit, "--", path],
        check=True,
        capture_output=True,
    )
    if not result.stdout:
        raise ValueError(f"actual-inference revision diff is empty: {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _load_post_final_erratum(
    *,
    evaluator_commit: str,
    quality_lock: Mapping[str, Any],
) -> dict[str, Any]:
    erratum = _read_json(ERRATUM_PATH)
    expected_keys = {
        "allowed_post_evaluator_files",
        "canonical_quality_lock_hash_unchanged",
        "case_selection_changed",
        "correctness_contract",
        "correctness_contract_changed",
        "device_identity_handling_changed",
        "discovered_stage",
        "efficiency_gate_changed",
        "evaluator_git_commit",
        "failed_v5r2_plan",
        "failed_v5r2_preflight",
        "kind",
        "predecessor_backend_correctness_revision",
        "quality_lock_sha256",
        "quality_values_changed",
        "role_set_changed",
        "schema_version",
        "statistical_protocol_changed",
        "timing_workload_changed",
    }
    rows = erratum.get("allowed_post_evaluator_files")
    if (
        set(erratum) != expected_keys
        or erratum.get("kind")
        != "actual_inference_v5r3_post_final_device_identity_erratum_v1"
        or erratum.get("schema_version") != 1
        or erratum.get("discovered_stage")
        != "post-final-quality-pre-timing"
        or erratum.get("evaluator_git_commit") != evaluator_commit
        or erratum.get("quality_lock_sha256")
        != quality_lock.get("quality_lock_sha256")
        or erratum.get("canonical_quality_lock_hash_unchanged") is not True
        or erratum.get("case_selection_changed") is not False
        or erratum.get("correctness_contract_changed") is not False
        or erratum.get("device_identity_handling_changed") is not True
        or erratum.get("efficiency_gate_changed") is not False
        or erratum.get("quality_values_changed") is not False
        or erratum.get("role_set_changed") is not False
        or erratum.get("statistical_protocol_changed") is not False
        or erratum.get("timing_workload_changed") is not False
        or not isinstance(rows, Mapping)
        or set(rows) != set(POST_FINAL_CORRECTNESS_REVISION_FILES)
        or erratum.get("predecessor_backend_correctness_revision")
        != {
            "path": PREDECESSOR_BACKEND_CORRECTNESS_ERRATUM_PATH.as_posix(),
            "sha256": hash_file(PREDECESSOR_BACKEND_CORRECTNESS_ERRATUM_PATH),
        }
        or erratum.get("failed_v5r2_plan")
        != {
            "path": FAILED_V5R2_PLAN_PATH.as_posix(),
            "plan_sha256": _read_json(FAILED_V5R2_PLAN_PATH)["plan_sha256"],
            "sha256": hash_file(FAILED_V5R2_PLAN_PATH),
        }
        or erratum.get("failed_v5r2_preflight")
        != {
            "latency_metrics_inspected": False,
            "path": FAILED_V5R2_PREFLIGHT_PATH.as_posix(),
            "sha256": hash_file(FAILED_V5R2_PREFLIGHT_PATH),
        }
        or erratum.get("correctness_contract")
        != {
            "cpu_semantic": {"atol": 2e-5, "rtol": 2e-5},
            "mps_backend": {
                "atol": 1e-4,
                "maximum_probability_total_variation": 1e-5,
                "rtol": 2e-5,
            },
            "third_tolerance_relaxation_allowed": False,
        }
    ):
        raise ValueError("actual-inference correctness revision differs")
    for path in POST_FINAL_CORRECTNESS_REVISION_FILES:
        row = rows[path]
        current = _tracked_head_identity(Path(path))["sha256"]
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "current_blob_sha256",
                "diff_sha256",
                "evaluator_blob_sha256",
            }
            or row.get("current_blob_sha256") != current
            or row.get("evaluator_blob_sha256")
            != _git_blob_sha256(evaluator_commit, path)
            or row.get("diff_sha256") != _git_diff_sha256(evaluator_commit, path)
        ):
            raise ValueError(
                f"actual-inference correctness revision row differs: {path}"
            )
    return erratum


def _require_implementation_not_after_evaluator(
    evaluator_commit: str,
    *,
    erratum: Mapping[str, Any],
) -> None:
    allowed = set(erratum["allowed_post_evaluator_files"])
    for path in ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER:
        if path == ACTUAL_INFERENCE_V5_ERRATUM_PATH or path in allowed:
            continue
        implementation_commit = _tracked_head_identity(Path(path))["git_commit"]
        _require_ancestor(
            implementation_commit,
            evaluator_commit,
            f"actual-v5 implementation -> final evaluator: {path}",
        )


def _implementation_sha256() -> dict[str, str]:
    return {
        path: _tracked_head_identity(Path(path))["sha256"]
        for path in ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER
    }


def _runtime_environment_contract() -> dict[str, Any]:
    return current_runtime_environment_contract()


def _load_upstream() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, str],
]:
    authorization_identity = _tracked_head_identity(AUTHORIZATION_PATH)
    quality_identity = _tracked_head_identity(QUALITY_LOCK_PATH)
    selection_identity = _tracked_head_identity(SELECTION_LOCK_PATH_VALUE)
    authorization = _read_json(AUTHORIZATION_PATH)
    quality_lock = _read_json(QUALITY_LOCK_PATH)
    selection_lock = _read_json(SELECTION_LOCK_PATH_VALUE)
    validate_selection_lock_v2(selection_lock)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    if (
        authorization["upstream_artifacts"]["selection_lock"]["sha256"]
        != selection_identity["sha256"]
        or quality_lock.get("authorization_artifact")
        != authorization_identity
    ):
        raise ValueError("actual-inference upstream artifact stitching differs")
    _require_ancestor(
        authorization_identity["git_commit"],
        quality_identity["git_commit"],
        "final authorization -> final quality lock",
    )
    _require_ancestor(
        quality_identity["git_commit"],
        _git_commit(),
        "final quality lock -> actual plan implementation",
    )
    return (
        authorization,
        quality_lock,
        selection_lock,
        authorization_identity,
        quality_identity,
    )


def _reconstruct_cases(authorization: Mapping[str, Any]) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
    Any,
]:
    seal = _read_json(FINAL_SEAL_PATH)
    validate_seal_envelope(seal)
    sealed = authorization["final_test"]
    output = seal["payload"]["output"]
    if (
        hash_file(FINAL_MANIFEST_PATH) != sealed["manifest"]["sha256"]
        or hash_file(FINAL_SEAL_PATH) != sealed["seal"]["sha256"]
        or hash_file(FINAL_OUTPUT_PATH) != sealed["output_jsonl"]["sha256"]
        or seal["payload_sha256"] != sealed["seal_payload_sha256"]
        or output["evaluation_stream_sha256"]
        != sealed["evaluation_stream_sha256"]
        or output["evaluation_stream_bytes"] != 32_000_000
        or output["sequence_count"] != 62_500
        or output["sequence_length"] != 512
    ):
        raise ValueError("actual-inference sealed final stream differs")
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
        != sealed["evaluation_stream_sha256"]
    ):
        raise ValueError("actual-inference stream reconstruction differs")
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    documents = reconstruct_document_window_map(
        FINAL_OUTPUT_PATH,
        split="test",
        byte_limit=32_000_000,
        sequence_length=512,
        expected_stream=stream.data,
    )
    if not documents.coverage_pass:
        raise ValueError("actual-inference document coverage is insufficient")
    eligible = documents.document_indices >= 0
    cases = select_inference_cases(
        inputs[eligible],
        boundaries[eligible],
        cluster_ids=documents.document_indices[eligible],
        case_count=(
            ACTUAL_INFERENCE_V5_WARMUP_CASES
            + ACTUAL_INFERENCE_V5_MEASURED_CASES
        ),
        prompt_length=128,
        continuation_length=128,
    )
    metadata = cases.public_metadata()
    if metadata.get("selected_unique_clusters") != len(cases.prompts):
        raise ValueError("actual-inference cases are not document independent")
    selection = {
        "algorithm": "JamoFlow-actual-inference-v5-one-case-per-document",
        "continuation_array_sha256": array_sha256(cases.replay_continuations),
        "document_assignment_sha256": documents.metadata()[
            "document_assignment_sha256"
        ],
        "prompt_array_sha256": array_sha256(cases.prompts),
        "public_metadata": metadata,
        "stream_sha256": sealed["evaluation_stream_sha256"],
    }
    context = {
        "artifact_path": ACTUAL_INFERENCE_V5_CASE_PATH,
        "artifact_sha256": "0" * 64,
        "case_selection_sha256": canonical_sha256(selection),
        "continuation_array_sha256": selection[
            "continuation_array_sha256"
        ],
        "document_assignment_sha256": selection[
            "document_assignment_sha256"
        ],
        "prompt_array_sha256": selection["prompt_array_sha256"],
        "selected_unique_documents": len(cases.prompts),
        "total_cases": len(cases.prompts),
    }
    return cases.prompts, cases.replay_continuations, context, documents


def _validate_quality_lock_from_committed_evidence(
    *,
    authorization: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    quality_lock: Mapping[str, Any],
    document_window_map: Any,
) -> dict[str, Any]:
    evidence_identity = _tracked_head_identity(FINAL_EVIDENCE_PATH_VALUE)
    if evidence_identity != quality_lock["evidence_artifact"]:
        raise ValueError("actual-inference final evidence artifact differs")
    session_plan = _read_json(FINAL_SESSION_PATH_VALUE)
    evidence = _read_json(FINAL_EVIDENCE_PATH_VALUE)
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
    arrays = {}
    for receipt in evidence["receipts"]:
        path = Path(receipt["nll"]["artifact_path"])
        if hash_file(path) != receipt["nll"]["artifact_sha256"]:
            raise ValueError("actual-inference final NLL artifact differs")
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["sequence_nll_nats"]:
                raise ValueError("actual-inference final NLL schema differs")
            values = archive["sequence_nll_nats"]
        arrays[receipt["receipt_sha256"]] = values
    validate_final_quality_lock_v2(
        quality_lock,
        authorization=authorization,
        selection_lock=selection_lock,
        session_plan=session_plan,
        evidence=evidence,
        document_window_map=document_window_map,
        arrays_by_receipt_sha256=arrays,
    )
    return session_plan


def _validate_case_artifact(
    path: Path,
    *,
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> str:
    expected = _npz_bytes(prompts, continuations)
    if path.exists() and path.read_bytes() != expected:
        raise ValueError("existing actual-inference cases differ")
    return hashlib.sha256(expected).hexdigest()


def _post_publish_status_is_clean() -> bool:
    lines = {line for line in _git_status().splitlines() if line.strip()}
    return lines <= {f"?? {PLAN_PATH.as_posix()}"}


def run() -> int:
    plan_commit = _require_clean_root()
    assert_workspace_path_no_symlinks(CASE_PATH.parent)
    assert_workspace_path_no_symlinks(PLAN_PATH.parent)
    if PLAN_PATH.exists() or CASE_PATH.exists():
        raise ValueError("actual-inference plan/cases already exist; verify only")
    authorization, quality, selection, auth_identity, quality_identity = _load_upstream()
    prompts, continuations, case_context, document_window_map = _reconstruct_cases(
        authorization
    )
    session_plan = _validate_quality_lock_from_committed_evidence(
        authorization=authorization,
        selection_lock=selection,
        quality_lock=quality,
        document_window_map=document_window_map,
    )
    erratum = _load_post_final_erratum(
        evaluator_commit=session_plan["evaluator_git_commit"],
        quality_lock=quality,
    )
    _require_implementation_not_after_evaluator(
        session_plan["evaluator_git_commit"],
        erratum=erratum,
    )
    case_bytes = _npz_bytes(prompts, continuations)
    case_context["artifact_sha256"] = hashlib.sha256(case_bytes).hexdigest()
    implementation = _implementation_sha256()
    plan = build_actual_inference_plan_v5(
        quality_lock=quality,
        authorization=authorization,
        quality_lock_artifact=quality_identity,
        authorization_artifact=auth_identity,
        case_context=case_context,
        implementation_sha256=implementation,
        implementation_order=ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
        plan_base_git_commit=plan_commit,
        runtime_environment_contract=_runtime_environment_contract(),
    )
    validate_actual_inference_plan_v5(
        plan,
        quality_lock=quality,
        authorization=authorization,
    )
    publish_no_clobber(CASE_PATH, case_bytes)
    if _validate_case_artifact(
        CASE_PATH,
        prompts=prompts,
        continuations=continuations,
    ) != case_context["artifact_sha256"]:
        raise AssertionError("published actual-inference cases changed")
    publish_no_clobber(PLAN_PATH, _json_bytes(plan))
    if _git_commit() != plan_commit or not _post_publish_status_is_clean():
        raise RuntimeError("repository changed while sealing actual-inference plan")
    print(
        "sealed actual-inference v5 plan; commit the plan before any timing",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
