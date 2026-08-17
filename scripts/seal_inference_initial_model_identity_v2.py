#!/usr/bin/env python3
"""Seal outcome-insensitive physical identities for 30 selection models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.compute_conversion import (
    CONVERSION_POLICIES,
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
)
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.inference_initial_model_identity_v2 import (
    CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER,
    INITIAL_MODEL_IDENTITY_LOCK_PATH,
    PLAN_FROZEN_FULL_FILE_ORDER,
    build_implementation_manifest_v2,
    build_initial_model_identity_lock_v2,
    build_plan_frozen_selection_v2,
    canonical_sha256,
    runtime_environment_v2,
    selection_decision_contract_sha256,
)
from jamoflow.inference_selection_plan import validate_selection_plan_v2
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER,
    INITIAL_SEEDS,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import shuffled_indices
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
    THRESHOLD_POLICIES,
)


PLAN_PATH = Path("data/manifests/phase3-inference-selection-plan-v2.json")
OUTPUT = Path(INITIAL_MODEL_IDENTITY_LOCK_PATH)
SOURCE = Path("data/processed/hplt3-korean-phase3/ko.jsonl")
INTEGRITY = Path("data/processed/hplt3-korean-phase3/integrity.json")
PHASE3_RUN_ROOT = Path("runs/phase3")
PHASE3_ARTIFACT_ROOT = Path("artifacts/phase3")
CONVERSION_RUN_ROOT = Path("runs/phase3-compute-conversion")
CONVERSION_ARTIFACT_ROOT = Path("artifacts/phase3-compute-conversion")
SPLITS = ("train", "calibration", "test")
GLOBAL_POSITION_LIMIT = 1_032
CONVERSION_REPORT_KEYS = {
    "calibration_loss_artifact_sha256",
    "checkpoint_artifact_sha256",
    "evaluation",
    "evidence_binding",
    "global_max_position_embeddings",
    "initialization_sha256",
    "loss_artifact_sha256",
    "model_spec",
    "optimization_spec",
    "parameters",
    "patch_diagnostics",
    "patch_matrix_sha256",
    "policy",
    "rate",
    "schema_version",
    "seed",
    "trained_state_sha256",
    "training",
    "training_order_sha256",
}
PHASE3_REPORT_KEYS = {
    "evaluation",
    "initialization_sha256",
    "model_spec",
    "optimization_spec",
    "parameters",
    "patch_diagnostics",
    "patch_matrix_sha256",
    "policy",
    "seed",
    "trained_state_sha256",
    "training",
    "training_order_sha256",
}


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


def _require_clean_root() -> str:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if root.resolve() != Path.cwd().resolve() or _git_status().strip():
        raise ValueError("initial model identity requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("initial model identity requires a full Git commit")
    history = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", str(OUTPUT)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if OUTPUT.exists() or history:
        raise ValueError("initial model identity already exists or was deleted")
    return commit


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} is not an ancestor of the identity seal")


def _git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"missing committed implementation blob: {commit}:{path}")
    return result.stdout


def _tracked_head_sha256(path: Path) -> str:
    blob = _git_blob("HEAD", path.as_posix())
    if path.is_symlink() or not path.is_file() or path.read_bytes() != blob:
        raise ValueError(f"identity input is not the exact HEAD blob: {path}")
    return hashlib.sha256(blob).hexdigest()


def _last_change_commit(path: str) -> str:
    value = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise ValueError(f"implementation path lacks Git history: {path}")
    return value


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


def _artifact(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return {"artifact_sha256": hash_file(path), "path": str(path)}


def _source_context(primary: Mapping[str, Any]) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    run_manifest = primary.get("run_manifest")
    if not isinstance(run_manifest, Mapping):
        raise ValueError("primary summary lacks its source manifest")
    source_artifact = {
        "filename": "ko.jsonl",
        "bytes": SOURCE.stat().st_size,
        "sha256": hash_file(SOURCE),
    }
    integrity_artifact = {
        "filename": "integrity.json",
        "bytes": INTEGRITY.stat().st_size,
        "sha256": hash_file(INTEGRITY),
    }
    if (
        run_manifest.get("source_artifact") != source_artifact
        or run_manifest.get("source_integrity_artifact") != integrity_artifact
    ):
        raise ValueError("initial source artifacts differ from primary evidence")
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    streams: dict[str, Any] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            SOURCE,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=int(run_manifest["limits"][split]),
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        split_whitespace = compact_whitespace_mask(stream.data).reshape(
            split_inputs.shape
        )
        stream_sha256 = hashlib.sha256(stream.data).hexdigest()
        expected = run_manifest["streams"][split]
        if (
            stream_sha256 != expected["selected_stream_sha256"]
            or len(split_inputs) != expected["sequence_count"]
        ):
            raise ValueError(f"initial stream differs from primary evidence: {split}")
        inputs[split] = split_inputs
        boundaries[split] = split_boundaries
        whitespace[split] = split_whitespace
        streams[split] = {
            "boundaries_sha256": _array_sha256(split_boundaries),
            "inputs_sha256": _array_sha256(split_inputs),
            "selected_stream_sha256": stream_sha256,
            "sequence_count": len(split_inputs),
            "whitespace_sha256": _array_sha256(split_whitespace),
        }
    return inputs, boundaries, whitespace, {
        "source_artifact": source_artifact,
        "source_integrity_artifact": integrity_artifact,
        "streams": streams,
    }


def _conversion_matrices(
    boundaries: Mapping[str, np.ndarray],
    whitespace: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for rate in CONVERSION_RATES:
            matrices[split].update(
                conversion_patch_matrices(
                    boundaries[split], whitespace[split], rate=rate
                )
            )
    cache_path = CONVERSION_ARTIFACT_ROOT / "patches.npz"
    diagnostics_path = CONVERSION_RUN_ROOT / "patch-diagnostics.json"
    diagnostics = _read_json(diagnostics_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        expected_keys = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in CONVERSION_POLICIES
        }
        if set(archive.files) != expected_keys:
            raise ValueError("conversion patch cache keys differ")
        for split in SPLITS:
            for policy in CONVERSION_POLICIES:
                matrix = matrices[split][policy]
                if not np.array_equal(archive[f"{split}__{policy}"], matrix):
                    raise ValueError("conversion patch cache differs from reconstruction")
                expected = {
                    **variable_patch_diagnostics(
                        matrix, boundaries[split]
                    ).to_dict(),
                    "matrix_sha256": _array_sha256(matrix),
                }
                if diagnostics["splits"][split][policy] != expected:
                    raise ValueError("conversion patch diagnostics differ")
    if diagnostics.get("cache_artifact_sha256") != hash_file(cache_path):
        raise ValueError("conversion cache artifact identity differs")
    return matrices, {
        "cache": _artifact(cache_path),
        "diagnostics": _artifact(diagnostics_path),
    }


def _validate_binding(
    binding: Mapping[str, Any],
    *,
    plan_artifact_sha256: str,
    primary_summary_sha256: str,
) -> None:
    unsigned = {key: value for key, value in binding.items() if key != "identity_sha256"}
    if (
        set(binding)
        != {
            "device",
            "git_commit",
            "git_worktree_clean_at_start",
            "identity_sha256",
            "policies",
            "primary_summary_sha256",
            "schema_version",
            "seeds",
            "selection_plan_sha256",
            "selection_summary_sha256",
            "stage",
        }
        or binding.get("identity_sha256") != canonical_sha256(unsigned)
        or binding.get("stage") != "initial"
        or binding.get("device") != "mps"
        or binding.get("git_worktree_clean_at_start") is not True
        or binding.get("schema_version") != 1
        or tuple(binding.get("seeds", ())) != INITIAL_SEEDS
        or tuple(binding.get("policies", ())) != CONVERSION_POLICIES
        or binding.get("selection_plan_sha256") != plan_artifact_sha256
        or binding.get("selection_summary_sha256") is not None
        or binding.get("primary_summary_sha256") != primary_summary_sha256
        or not isinstance(binding.get("git_commit"), str)
        or len(binding["git_commit"]) != 40
    ):
        raise ValueError("initial conversion evidence binding differs")


def _conversion_identity(
    *,
    seed: int,
    policy: str,
    inputs: Mapping[str, np.ndarray],
    boundaries: Mapping[str, np.ndarray],
    matrices: Mapping[str, Mapping[str, np.ndarray]],
    binding: Mapping[str, Any],
) -> dict[str, object]:
    report_path = CONVERSION_RUN_ROOT / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = CONVERSION_ARTIFACT_ROOT / f"seed-{seed}" / f"{policy}.pt"
    report = _read_json(report_path)
    rate = int(policy.rsplit("_", 1)[1])
    spec = conversion_model_spec(rate)
    model = build_main_model(
        spec, seed=seed, global_max_position_embeddings=GLOBAL_POSITION_LIMIT
    )
    initialization = _state_sha256(model)
    parameters = parameter_count(model)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state = _state_sha256(model)
    del model
    training = report.get("training")
    if (
        set(report) != CONVERSION_REPORT_KEYS
        or report.get("schema_version") != 2
        or report.get("seed") != seed
        or report.get("policy") != policy
        or report.get("rate") != rate
        or report.get("parameters") != parameters
        or report.get("initialization_sha256") != initialization
        or report.get("trained_state_sha256") != state
        or report.get("training_order_sha256")
        != _array_sha256(shuffled_indices(len(inputs["train"]), seed))
        or report.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
        or report.get("evidence_binding") != binding
        or report.get("model_spec") != spec.to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT
        or not isinstance(training, Mapping)
        or training.get("steps")
        != math.ceil(len(inputs["train"]) / PHASE3_OPTIMIZATION_SPEC.batch_size)
        or training.get("examples") != len(inputs["train"])
        or training.get("predicted_bytes") != len(inputs["train"]) * 511
    ):
        raise ValueError(f"initial conversion model identity differs: {seed}/{policy}")
    for split in SPLITS:
        matrix = matrices[split][policy]
        validate_padded_patch_matrix(matrix, 512)
        if (
            report["patch_matrix_sha256"][split] != _array_sha256(matrix)
            or report["patch_diagnostics"][split]
            != variable_patch_diagnostics(matrix, boundaries[split]).to_dict()
        ):
            raise ValueError(f"conversion matrix identity differs: {seed}/{policy}/{split}")
    return {
        "auxiliary": {"kind": "none"},
        "checkpoint": {
            **_artifact(checkpoint_path),
            "state_sha256": state,
        },
        "initialization_sha256": initialization,
        "model_family": "compute_conversion",
        "model_spec_sha256": canonical_sha256(spec.to_dict()),
        "optimization_spec_sha256": canonical_sha256(
            PHASE3_OPTIMIZATION_SPEC.to_dict()
        ),
        "parameter_count": parameters,
        "patch_count": rate,
        "policy": policy,
        "seed": seed,
        "training_order_sha256": report["training_order_sha256"],
        "training_report": _artifact(report_path),
    }


def _phase3_identity(
    *, seed: int, policy: str, summary: Mapping[str, Any]
) -> dict[str, object]:
    evidence = summary["integrity"]["by_seed"][str(seed)]
    report_path = PHASE3_RUN_ROOT / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = PHASE3_ARTIFACT_ROOT / f"seed-{seed}" / f"{policy}.pt"
    report = _read_json(report_path)
    report_artifact = hash_file(report_path)
    checkpoint_artifact = hash_file(checkpoint_path)
    if (
        set(report) != PHASE3_REPORT_KEYS
        or report_artifact
        != evidence["training_report_artifact_sha256"][policy]
        or checkpoint_artifact
        != evidence["checkpoint_artifact_sha256"][policy]
        or report.get("seed") != seed
        or report.get("policy") != policy
        or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
    ):
        raise ValueError(f"Phase3 tracked identity differs: {seed}/{policy}")
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization = _state_sha256(model)
    parameters = parameter_count(model)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state = _state_sha256(model)
    del model
    if (
        report.get("initialization_sha256") != initialization
        or report.get("parameters") != parameters
        or report.get("trained_state_sha256") != state
        or state != evidence["checkpoint_state_sha256"][policy]
    ):
        raise ValueError(f"Phase3 checkpoint identity differs: {seed}/{policy}")
    auxiliary: dict[str, object] = {"kind": "none"}
    if policy in THRESHOLD_POLICIES:
        router = evidence.get("router_and_threshold_cache")
        if not isinstance(router, Mapping):
            raise ValueError("Phase3 entropy identity is missing")
        paths = {
            "router_checkpoint": PHASE3_ARTIFACT_ROOT / f"seed-{seed}" / "router.pt",
            "router_report": PHASE3_RUN_ROOT / f"seed-{seed}" / "router.json",
            "threshold_cache": PHASE3_ARTIFACT_ROOT / f"seed-{seed}" / "threshold-patches.npz",
            "threshold_diagnostics": PHASE3_RUN_ROOT / f"seed-{seed}" / "threshold-patch-diagnostics.json",
        }
        for label, path in paths.items():
            expected = router[f"{label}_artifact_sha256"]
            if hash_file(path) != expected:
                raise ValueError(f"Phase3 entropy artifact differs: {seed}/{label}")
        auxiliary = {
            "kind": "entropy_router",
            "router_checkpoint_artifact_sha256": router[
                "router_checkpoint_artifact_sha256"
            ],
            "router_checkpoint_path": str(paths["router_checkpoint"]),
            "router_report_artifact_sha256": router[
                "router_report_artifact_sha256"
            ],
            "router_report_path": str(paths["router_report"]),
            "router_state_sha256": router["router_checkpoint_state_sha256"],
            "threshold_cache_artifact_sha256": router[
                "threshold_cache_artifact_sha256"
            ],
            "threshold_cache_path": str(paths["threshold_cache"]),
            "threshold_diagnostics_artifact_sha256": router[
                "threshold_diagnostics_artifact_sha256"
            ],
            "threshold_diagnostics_path": str(paths["threshold_diagnostics"]),
        }
    return {
        "auxiliary": auxiliary,
        "checkpoint": {
            "artifact_sha256": checkpoint_artifact,
            "path": str(checkpoint_path),
            "state_sha256": state,
        },
        "initialization_sha256": initialization,
        "model_family": "phase3",
        "model_spec_sha256": canonical_sha256(PHASE3_MODEL_SPEC.to_dict()),
        "optimization_spec_sha256": canonical_sha256(
            PHASE3_OPTIMIZATION_SPEC.to_dict()
        ),
        "parameter_count": parameters,
        "patch_count": 86,
        "policy": policy,
        "seed": seed,
        "training_order_sha256": report["training_order_sha256"],
        "training_report": {
            "artifact_sha256": report_artifact,
            "path": str(report_path),
        },
    }


def main() -> int:
    producer_commit = _require_clean_root()
    plan_artifact_sha256 = _tracked_head_sha256(PLAN_PATH)
    plan = _read_json(PLAN_PATH)
    validate_selection_plan_v2(plan)
    if plan["plan_sha256"] != plan.get("plan_sha256"):
        raise AssertionError("selection plan identity disappeared")
    phase3_summary_path = Path(
        plan["historical_screening"]["all_initial_summary"]["path"]
    )
    primary_summary_path = Path(
        plan["historical_screening"]["primary_summary"]["path"]
    )
    phase3_summary_sha256 = _tracked_head_sha256(phase3_summary_path)
    primary_summary_sha256 = _tracked_head_sha256(primary_summary_path)
    if (
        phase3_summary_sha256
        != plan["historical_screening"]["all_initial_summary"]["sha256"]
        or primary_summary_sha256
        != plan["historical_screening"]["primary_summary"]["sha256"]
    ):
        raise ValueError("selection plan summary trust roots differ")
    phase3_summary = _read_json(phase3_summary_path)
    primary_summary = _read_json(primary_summary_path)
    if (
        phase3_summary.get("integrity", {}).get("all_integrity_checks_pass")
        is not True
        or tuple(phase3_summary.get("seeds", ())) != INITIAL_SEEDS
        or tuple(phase3_summary.get("policies", ())) != PHASE3_POLICIES
    ):
        raise ValueError("tracked Phase3 identity summary is incomplete")

    inputs, boundaries, whitespace, source_context = _source_context(
        primary_summary
    )
    if (
        source_context["source_artifact"]["sha256"]
        != plan["initial_design"]["source_artifact_sha256"]
        or source_context["source_integrity_artifact"]["sha256"]
        != plan["initial_design"]["source_integrity_artifact_sha256"]
        or source_context["streams"]["calibration"]["selected_stream_sha256"]
        != plan["calibration_evaluator"]["input_stream_sha256"]
    ):
        raise ValueError("initial model identity source differs from plan")
    matrices, conversion_patch_artifacts = _conversion_matrices(
        boundaries, whitespace
    )
    conversion_manifest_path = CONVERSION_RUN_ROOT / "manifest.json"
    conversion_manifest = _read_json(conversion_manifest_path)
    conversion_manifest_sha256 = hash_file(conversion_manifest_path)
    if (
        conversion_manifest.get("schema_version") != 2
        or conversion_manifest.get("selection_plan_sha256")
        != plan_artifact_sha256
        or conversion_manifest.get("source_context") != source_context
        or tuple(conversion_manifest.get("policies", ())) != CONVERSION_POLICIES
        or tuple(conversion_manifest.get("rates", ())) != CONVERSION_RATES
        or conversion_manifest.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or conversion_manifest.get("global_max_position_embeddings")
        != GLOBAL_POSITION_LIMIT
    ):
        raise ValueError("initial conversion manifest identity differs")

    binding: Mapping[str, Any] | None = None
    for seed in INITIAL_SEEDS:
        for policy in CONVERSION_POLICIES:
            report = _read_json(
                CONVERSION_RUN_ROOT / f"seed-{seed}" / f"{policy}.json"
            )
            candidate = report.get("evidence_binding")
            if not isinstance(candidate, Mapping):
                raise ValueError("initial conversion report lacks evidence binding")
            if binding is None:
                binding = candidate
            elif dict(binding) != dict(candidate):
                raise ValueError("initial conversion reports rotate evidence binding")
    if binding is None:
        raise AssertionError("initial conversion evidence binding is missing")
    _validate_binding(
        binding,
        plan_artifact_sha256=plan_artifact_sha256,
        primary_summary_sha256=primary_summary_sha256,
    )
    run_commit = binding["git_commit"]
    _require_ancestor(run_commit, producer_commit, "conversion run commit")
    invocations = conversion_manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("initial conversion manifest lacks invocation evidence")
    matching = []
    for invocation in invocations:
        if not isinstance(invocation, Mapping):
            raise ValueError("initial conversion invocation is malformed")
        if invocation.get("stage") == "initial":
            if (
                tuple(invocation.get("seeds", ())) != INITIAL_SEEDS
                or tuple(invocation.get("policies", ())) != CONVERSION_POLICIES
                or invocation.get("selection_summary") is not None
                or invocation.get("selection_summary_sha256") is not None
                or invocation.get("git_commit") != run_commit
                or invocation.get("evidence_binding") != binding
                or invocation.get("git_worktree_clean_at_start") is not True
            ):
                raise ValueError("conflicting initial conversion invocation exists")
            matching.append(invocation)
    if not matching:
        raise ValueError("initial conversion lacks an authorized invocation")

    current_hashes = {
        path: _tracked_head_sha256(Path(path))
        for path in CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
    }
    changes = {
        path: _last_change_commit(path)
        for path in CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
    }
    for path, change in changes.items():
        _require_ancestor(change, producer_commit, f"implementation {path}")
    environment = runtime_environment_v2()
    plan_commit = plan["plan_git_commit"]
    _require_ancestor(plan_commit, producer_commit, "selection plan commit")
    selection_path = "src/jamoflow/inference_selection_v2.py"
    planned_decision_sha256 = selection_decision_contract_sha256(
        _git_blob(plan_commit, selection_path).decode("utf-8")
    )
    current_decision_sha256 = selection_decision_contract_sha256(
        _git_blob("HEAD", selection_path).decode("utf-8")
    )
    if current_decision_sha256 != planned_decision_sha256:
        raise ValueError("selection decision contract changed after its sealed plan")
    full_file_sha256_at_plan = {
        path: hashlib.sha256(_git_blob(plan_commit, path)).hexdigest()
        for path in PLAN_FROZEN_FULL_FILE_ORDER
    }
    for path, expected_sha256 in full_file_sha256_at_plan.items():
        if current_hashes[path] != expected_sha256:
            raise ValueError(f"plan-frozen selection dependency changed: {path}")
    plan_frozen_selection = build_plan_frozen_selection_v2(
        plan_git_commit=plan_commit,
        decision_ast_sha256=planned_decision_sha256,
        full_file_sha256_at_plan=full_file_sha256_at_plan,
    )
    implementation = build_implementation_manifest_v2(
        sha256_by_path=current_hashes,
        last_change_commit_by_path=changes,
        producer_git_commit=producer_commit,
        environment=environment,
        plan_frozen_selection=plan_frozen_selection,
    )
    run_implementation = {
        path: hashlib.sha256(_git_blob(run_commit, path)).hexdigest()
        for path in CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER
    }

    models: dict[int, dict[str, dict[str, object]]] = {}
    for seed in INITIAL_SEEDS:
        models[seed] = {}
        for policy in PHASE3_POLICIES:
            models[seed][policy] = _phase3_identity(
                seed=seed, policy=policy, summary=phase3_summary
            )
        for policy in CONVERSION_POLICIES:
            models[seed][policy] = _conversion_identity(
                seed=seed,
                policy=policy,
                inputs=inputs,
                boundaries=boundaries,
                matrices=matrices,
                binding=binding,
            )
    conversion_training = {
        "authorized_invocation_count": len(matching),
        "evidence_binding": dict(binding),
        "manifest": {
            "artifact_sha256": conversion_manifest_sha256,
            "path": str(conversion_manifest_path),
        },
        "run_git_commit": run_commit,
        "run_implementation_file_order": list(
            CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER
        ),
        "run_implementation_sha256": run_implementation,
    }
    source_identities = {
        "conversion_patch_artifacts": conversion_patch_artifacts,
        "phase3_initial_summary": {
            "artifact_sha256": phase3_summary_sha256,
            "path": str(phase3_summary_path),
        },
        "primary_summary": {
            "artifact_sha256": primary_summary_sha256,
            "path": str(primary_summary_path),
        },
        **source_context,
    }
    lock = build_initial_model_identity_lock_v2(
        plan_artifact_sha256=plan_artifact_sha256,
        plan_payload_sha256=plan["plan_sha256"],
        producer_git_commit=producer_commit,
        implementation_manifest=implementation,
        source_identities=source_identities,
        conversion_training=conversion_training,
        models=models,
    )
    if _git_commit() != producer_commit or _git_status().strip():
        raise RuntimeError("Git state changed while sealing initial model identities")
    payload = (
        json.dumps(lock, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    publish_no_clobber(OUTPUT, payload)
    print(
        json.dumps(
            {
                "lock_sha256": lock["lock_sha256"],
                "output": str(OUTPUT),
                "status": "complete_pending_commit",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
