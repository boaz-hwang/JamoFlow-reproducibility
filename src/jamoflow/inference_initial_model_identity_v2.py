"""Outcome-insensitive physical-model trust root for inference selection v2."""

from __future__ import annotations

import ast
from hashlib import sha256
import importlib.metadata
import json
import platform
import subprocess
import sys
from typing import Any, Mapping

import torch

from .compute_conversion import CONVERSION_POLICIES, conversion_model_spec
from .inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER,
    INITIAL_SEEDS,
)
from .phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
    THRESHOLD_POLICIES,
)


INITIAL_MODEL_IDENTITY_KIND = "phase3_initial_model_identity_lock_v2"
INITIAL_MODEL_IDENTITY_PROTOCOL = "jamoflow-initial-model-identity-v2"
INITIAL_MODEL_IDENTITY_LOCK_PATH = (
    "results/phase3-inference-selection-v2/initial-model-identity-lock.json"
)
CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER = (
    "scripts/run_phase3_compute_conversion.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/hplt3_final_test.py",
    "src/jamoflow/inference_selection_plan.py",
    "src/jamoflow/inference_selection_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_patching.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/unicode_audit.py",
    "src/jamoflow/utf8.py",
    "pyproject.toml",
)
PLAN_FROZEN_FULL_FILE_ORDER = (
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/phase3.py",
)
PLAN_FROZEN_SELECTION_AST_NAMES = (
    "INITIAL_SEEDS",
    "CONFIRMATION_SEEDS",
    "C86",
    "PRIMARY_CONFIRMED_POLICIES",
    "CALIBRATION_POLICY_ORDER",
    "BROAD_REFERENCE_CALIBRATION_FUTILITY_MARGIN_BPB",
    "BROAD_REFERENCE_CALIBRATION_FUTILITY_MINIMUM_SEEDS",
    "_canonical_json_bytes",
    "_is_sha256",
    "_mean",
    "_descriptor",
    "_validate_calibration_matrix",
    "build_selection_decision_v2",
    "validate_selection_decision_v2",
)
PLAN_FROZEN_SELECTION_PROTOCOL = "jamoflow-selection-decision-ast-v1"
INITIAL_MAIN_PARAMETER_COUNT = 19_596_096

_TOP_KEYS = {
    "calibration_selection_implementation",
    "conversion_training",
    "kind",
    "lock_sha256",
    "models",
    "plan_artifact_sha256",
    "plan_payload_sha256",
    "policy_order",
    "producer_git_commit",
    "protocol",
    "result_inputs",
    "schema_version",
    "seed_order",
    "source_identities",
}
_MODEL_KEYS = {
    "auxiliary",
    "checkpoint",
    "initialization_sha256",
    "model_family",
    "model_spec_sha256",
    "optimization_spec_sha256",
    "parameter_count",
    "patch_count",
    "policy",
    "seed",
    "training_order_sha256",
    "training_report",
}
_ARTIFACT_KEYS = {"artifact_sha256", "path"}
_CHECKPOINT_KEYS = {"artifact_sha256", "path", "state_sha256"}
_ENTROPY_KEYS = {
    "kind",
    "router_checkpoint_artifact_sha256",
    "router_checkpoint_path",
    "router_report_artifact_sha256",
    "router_report_path",
    "router_state_sha256",
    "threshold_cache_artifact_sha256",
    "threshold_cache_path",
    "threshold_diagnostics_artifact_sha256",
    "threshold_diagnostics_path",
}
_SOURCE_KEYS = {
    "conversion_patch_artifacts",
    "phase3_initial_summary",
    "primary_summary",
    "source_artifact",
    "source_integrity_artifact",
    "streams",
}
_SOURCE_FILE_KEYS = {"bytes", "filename", "sha256"}
_STREAM_KEYS = {
    "boundaries_sha256",
    "inputs_sha256",
    "selected_stream_sha256",
    "sequence_count",
    "whitespace_sha256",
}
_CONVERSION_TRAINING_KEYS = {
    "authorized_invocation_count",
    "evidence_binding",
    "manifest",
    "run_git_commit",
    "run_implementation_file_order",
    "run_implementation_sha256",
}
_CONVERSION_BINDING_KEYS = {
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
_RESULT_INPUTS = {
    "calibration_metric_used_for_identity_seal": False,
    "calibration_metric_used_later_by_locked_selection": True,
    "final_test_metric_used_for_identity_seal": False,
    "historical_test_metric_used_for_identity_seal": False,
    "latency_used_for_identity_seal": False,
    "metric_bearing_training_artifacts_read_for_identity_only": True,
    "training_artifact_identity": True,
}
_ENVIRONMENT_KEYS = {
    "hardware",
    "machine",
    "mac_ver",
    "packages",
    "platform",
    "python_implementation",
    "python_version",
    "sys_version_info",
    "torch_runtime",
}
_HARDWARE_KEYS = {"chip", "machine_model", "memory_bytes", "os_build"}
_TORCH_RUNTIME_KEYS = {"git_version", "mps_available", "mps_built", "version"}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _required_command_value(command: tuple[str, ...], label: str) -> str:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise ValueError(f"initial identity hardware value is missing: {label}")
    return value


def runtime_environment_v2() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("selection evidence requires the sealed Apple-MPS runtime")
    memory_value = _required_command_value(
        ("sysctl", "-n", "hw.memsize"), "physical memory"
    )
    try:
        memory_bytes = int(memory_value)
    except ValueError as error:
        raise ValueError("initial identity physical memory is malformed") from error
    return {
        "hardware": {
            "chip": _required_command_value(
                ("sysctl", "-n", "machdep.cpu.brand_string"), "chip"
            ),
            "machine_model": _required_command_value(
                ("sysctl", "-n", "hw.model"), "machine model"
            ),
            "memory_bytes": memory_bytes,
            "os_build": _required_command_value(
                ("sw_vers", "-buildVersion"), "OS build"
            ),
        },
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "packages": packages,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "sys_version_info": list(sys.version_info[:3]),
        "torch_runtime": {
            "git_version": str(torch.version.git_version or "unknown"),
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
            "version": str(torch.__version__),
        },
    }


def validate_runtime_environment_v2(environment: Mapping[str, object]) -> None:
    if not isinstance(environment, Mapping) or set(environment) != _ENVIRONMENT_KEYS:
        raise ValueError("selection runtime environment schema differs")
    packages = environment.get("packages")
    hardware = environment.get("hardware")
    torch_runtime = environment.get("torch_runtime")
    sys_version = environment.get("sys_version_info")
    if (
        environment.get("machine") != "arm64"
        or not isinstance(environment.get("mac_ver"), str)
        or not environment.get("mac_ver")
        or not isinstance(environment.get("platform"), str)
        or not environment.get("platform")
        or not isinstance(environment.get("python_implementation"), str)
        or not isinstance(environment.get("python_version"), str)
        or not isinstance(sys_version, list)
        or len(sys_version) != 3
        or not all(isinstance(value, int) and value >= 0 for value in sys_version)
        or not isinstance(packages, Mapping)
        or set(packages) != {"numpy", "torch", "transformers"}
        or not all(
            isinstance(packages[name], str) and packages[name] != "missing"
            for name in packages
        )
        or not isinstance(hardware, Mapping)
        or set(hardware) != _HARDWARE_KEYS
        or not all(
            isinstance(hardware[key], str) and bool(hardware[key])
            for key in ("chip", "machine_model", "os_build")
        )
        or not isinstance(hardware.get("memory_bytes"), int)
        or isinstance(hardware.get("memory_bytes"), bool)
        or hardware.get("memory_bytes", 0) <= 0
        or not isinstance(torch_runtime, Mapping)
        or set(torch_runtime) != _TORCH_RUNTIME_KEYS
        or not isinstance(torch_runtime.get("version"), str)
        or not isinstance(torch_runtime.get("git_version"), str)
        or torch_runtime.get("mps_available") is not True
        or torch_runtime.get("mps_built") is not True
    ):
        raise ValueError("selection runtime environment is not Apple-MPS eligible")


def selection_decision_contract_sha256(source: str) -> str:
    """Hash only the preregistered decision constants/functions, not hardening code."""

    tree = ast.parse(source)
    selected: dict[str, str] = {}
    imports: list[str] = []
    wanted = set(PLAN_FROZEN_SELECTION_AST_NAMES)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(
                ast.dump(node, annotate_fields=True, include_attributes=False)
            )
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [
                target.id for target in targets if isinstance(target, ast.Name)
            ]
        for name in names:
            if name in wanted:
                selected[name] = ast.dump(
                    node,
                    annotate_fields=True,
                    include_attributes=False,
                )
    if set(selected) != wanted:
        missing = sorted(wanted - set(selected))
        raise ValueError(f"selection decision AST contract is incomplete: {missing}")
    return canonical_sha256(
        {
            "imports": imports,
            "nodes": {name: selected[name] for name in PLAN_FROZEN_SELECTION_AST_NAMES},
            "protocol": PLAN_FROZEN_SELECTION_PROTOCOL,
        }
    )


def build_plan_frozen_selection_v2(
    *,
    plan_git_commit: str,
    decision_ast_sha256: str,
    full_file_sha256_at_plan: Mapping[str, str],
) -> dict[str, object]:
    if (
        not is_git_commit(plan_git_commit)
        or not is_sha256(decision_ast_sha256)
        or set(full_file_sha256_at_plan) != set(PLAN_FROZEN_FULL_FILE_ORDER)
        or not all(
            is_sha256(full_file_sha256_at_plan[path])
            for path in PLAN_FROZEN_FULL_FILE_ORDER
        )
    ):
        raise ValueError("plan-frozen selection identity is malformed")
    unsigned: dict[str, object] = {
        "decision_ast_sha256": decision_ast_sha256,
        "full_file_order": list(PLAN_FROZEN_FULL_FILE_ORDER),
        "full_file_sha256_at_plan": {
            path: full_file_sha256_at_plan[path]
            for path in PLAN_FROZEN_FULL_FILE_ORDER
        },
        "plan_git_commit": plan_git_commit,
        "protocol": PLAN_FROZEN_SELECTION_PROTOCOL,
    }
    return {**unsigned, "identity_sha256": canonical_sha256(unsigned)}


def validate_plan_frozen_selection_v2(identity: Mapping[str, object]) -> None:
    expected_keys = {
        "decision_ast_sha256",
        "full_file_order",
        "full_file_sha256_at_plan",
        "identity_sha256",
        "plan_git_commit",
        "protocol",
    }
    if not isinstance(identity, Mapping):
        raise ValueError("plan-frozen selection identity differs")
    files = identity.get("full_file_sha256_at_plan")
    unsigned = {
        key: value for key, value in identity.items() if key != "identity_sha256"
    }
    if (
        set(identity) != expected_keys
        or identity.get("protocol") != PLAN_FROZEN_SELECTION_PROTOCOL
        or not is_git_commit(identity.get("plan_git_commit"))
        or not is_sha256(identity.get("decision_ast_sha256"))
        or tuple(identity.get("full_file_order", ()))
        != PLAN_FROZEN_FULL_FILE_ORDER
        or not isinstance(files, Mapping)
        or set(files) != set(PLAN_FROZEN_FULL_FILE_ORDER)
        or not all(is_sha256(files.get(path)) for path in PLAN_FROZEN_FULL_FILE_ORDER)
        or identity.get("identity_sha256") != canonical_sha256(unsigned)
    ):
        raise ValueError("plan-frozen selection identity differs")


def build_implementation_manifest_v2(
    *,
    sha256_by_path: Mapping[str, str],
    last_change_commit_by_path: Mapping[str, str],
    producer_git_commit: str,
    environment: Mapping[str, object],
    plan_frozen_selection: Mapping[str, object],
) -> dict[str, object]:
    order = CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
    validate_runtime_environment_v2(environment)
    validate_plan_frozen_selection_v2(plan_frozen_selection)
    if (
        set(sha256_by_path) != set(order)
        or len(sha256_by_path) != len(order)
        or set(last_change_commit_by_path) != set(order)
        or len(last_change_commit_by_path) != len(order)
        or not all(is_sha256(sha256_by_path[path]) for path in order)
        or not all(
            is_git_commit(last_change_commit_by_path[path]) for path in order
        )
        or not is_git_commit(producer_git_commit)
    ):
        raise ValueError("calibration implementation identity is malformed")
    unsigned: dict[str, object] = {
        "environment": dict(environment),
        "environment_sha256": canonical_sha256(environment),
        "file_order": list(order),
        "last_change_commit_by_path": {
            path: last_change_commit_by_path[path] for path in order
        },
        "plan_frozen_selection": dict(plan_frozen_selection),
        "producer_git_commit": producer_git_commit,
        "sha256_by_path": {path: sha256_by_path[path] for path in order},
    }
    return {
        **unsigned,
        "manifest_sha256": canonical_sha256(unsigned),
    }


def validate_implementation_manifest_v2(
    manifest: Mapping[str, object],
) -> None:
    expected_keys = {
        "environment",
        "environment_sha256",
        "file_order",
        "last_change_commit_by_path",
        "manifest_sha256",
        "plan_frozen_selection",
        "producer_git_commit",
        "sha256_by_path",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_keys:
        raise ValueError("calibration implementation manifest schema differs")
    order = CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
    hashes = manifest.get("sha256_by_path")
    changes = manifest.get("last_change_commit_by_path")
    environment = manifest.get("environment")
    frozen = manifest.get("plan_frozen_selection")
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        tuple(manifest.get("file_order", ())) != order
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(order)
        or not isinstance(changes, Mapping)
        or set(changes) != set(order)
        or not isinstance(environment, Mapping)
        or manifest.get("environment_sha256") != canonical_sha256(environment)
        or not is_git_commit(manifest.get("producer_git_commit"))
        or not all(is_sha256(hashes.get(path)) for path in order)
        or not all(is_git_commit(changes.get(path)) for path in order)
        or manifest.get("manifest_sha256") != canonical_sha256(unsigned)
    ):
        raise ValueError("calibration implementation manifest identity differs")
    validate_runtime_environment_v2(environment)
    if not isinstance(frozen, Mapping):
        raise ValueError("calibration implementation lacks plan-frozen selection")
    validate_plan_frozen_selection_v2(frozen)


def _validate_artifact(
    artifact: object, *, expected_path: str, label: str
) -> None:
    if (
        not isinstance(artifact, Mapping)
        or set(artifact) != _ARTIFACT_KEYS
        or artifact.get("path") != expected_path
        or not is_sha256(artifact.get("artifact_sha256"))
    ):
        raise ValueError(f"{label} artifact identity differs")


def _validate_source_identities(source: object) -> None:
    if not isinstance(source, Mapping) or set(source) != _SOURCE_KEYS:
        raise ValueError("initial source identity schema differs")
    patches = source.get("conversion_patch_artifacts")
    if not isinstance(patches, Mapping) or set(patches) != {"cache", "diagnostics"}:
        raise ValueError("conversion patch artifact identity schema differs")
    _validate_artifact(
        patches["cache"],
        expected_path="artifacts/phase3-compute-conversion/patches.npz",
        label="conversion patch cache",
    )
    _validate_artifact(
        patches["diagnostics"],
        expected_path="runs/phase3-compute-conversion/patch-diagnostics.json",
        label="conversion patch diagnostics",
    )
    _validate_artifact(
        source.get("phase3_initial_summary"),
        expected_path="results/phase3-all-initial/summary.json",
        label="Phase3 initial summary",
    )
    _validate_artifact(
        source.get("primary_summary"),
        expected_path="results/phase3-primary-five-seed/summary.json",
        label="Phase3 primary summary",
    )
    for key, filename in (
        ("source_artifact", "ko.jsonl"),
        ("source_integrity_artifact", "integrity.json"),
    ):
        artifact = source.get(key)
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != _SOURCE_FILE_KEYS
            or artifact.get("filename") != filename
            or not isinstance(artifact.get("bytes"), int)
            or isinstance(artifact.get("bytes"), bool)
            or artifact.get("bytes", 0) <= 0
            or not is_sha256(artifact.get("sha256"))
        ):
            raise ValueError(f"initial {key} identity differs")
    streams = source.get("streams")
    if not isinstance(streams, Mapping) or set(streams) != {
        "train",
        "calibration",
        "test",
    }:
        raise ValueError("initial stream identity set differs")
    for split in ("train", "calibration", "test"):
        stream = streams[split]
        if (
            not isinstance(stream, Mapping)
            or set(stream) != _STREAM_KEYS
            or not all(
                is_sha256(stream.get(key))
                for key in _STREAM_KEYS
                if key.endswith("sha256")
            )
            or not isinstance(stream.get("sequence_count"), int)
            or isinstance(stream.get("sequence_count"), bool)
            or stream.get("sequence_count", 0) <= 0
        ):
            raise ValueError(f"initial {split} stream identity differs")


def _validate_conversion_training(
    training: object,
    *,
    plan_artifact_sha256: str,
    source_identities: Mapping[str, object],
) -> None:
    if not isinstance(training, Mapping) or set(training) != _CONVERSION_TRAINING_KEYS:
        raise ValueError("initial conversion training identity schema differs")
    binding = training.get("evidence_binding")
    manifest = training.get("manifest")
    hashes = training.get("run_implementation_sha256")
    unsigned_binding = (
        {key: value for key, value in binding.items() if key != "identity_sha256"}
        if isinstance(binding, Mapping)
        else {}
    )
    if (
        not isinstance(training.get("authorized_invocation_count"), int)
        or isinstance(training.get("authorized_invocation_count"), bool)
        or training.get("authorized_invocation_count", 0) <= 0
        or not isinstance(binding, Mapping)
        or set(binding) != _CONVERSION_BINDING_KEYS
        or binding.get("identity_sha256") != canonical_sha256(unsigned_binding)
        or binding.get("stage") != "initial"
        or binding.get("schema_version") != 1
        or binding.get("device") != "mps"
        or binding.get("git_worktree_clean_at_start") is not True
        or tuple(binding.get("seeds", ())) != INITIAL_SEEDS
        or tuple(binding.get("policies", ())) != CONVERSION_POLICIES
        or binding.get("selection_plan_sha256") != plan_artifact_sha256
        or binding.get("selection_summary_sha256") is not None
        or binding.get("primary_summary_sha256")
        != source_identities["primary_summary"]["artifact_sha256"]  # type: ignore[index]
        or not is_git_commit(binding.get("git_commit"))
        or training.get("run_git_commit") != binding.get("git_commit")
        or not isinstance(manifest, Mapping)
        or set(manifest) != _ARTIFACT_KEYS
        or manifest.get("path") != "runs/phase3-compute-conversion/manifest.json"
        or not is_sha256(manifest.get("artifact_sha256"))
        or tuple(training.get("run_implementation_file_order", ()))
        != CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER)
        or not all(
            is_sha256(hashes.get(path))
            for path in CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER
        )
    ):
        raise ValueError("initial conversion training identity differs")


def build_initial_model_identity_lock_v2(
    *,
    plan_artifact_sha256: str,
    plan_payload_sha256: str,
    producer_git_commit: str,
    implementation_manifest: Mapping[str, object],
    source_identities: Mapping[str, object],
    conversion_training: Mapping[str, object],
    models: Mapping[int, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    validate_implementation_manifest_v2(implementation_manifest)
    if not (
        is_sha256(plan_artifact_sha256)
        and is_sha256(plan_payload_sha256)
        and is_git_commit(producer_git_commit)
    ):
        raise ValueError("initial model lock identity is malformed")
    normalized: dict[str, dict[str, object]] = {}
    if tuple(sorted(models)) != INITIAL_SEEDS:
        raise ValueError("initial model lock seed set is not exact")
    for seed in INITIAL_SEEDS:
        row = models[seed]
        if set(row) != set(CALIBRATION_POLICY_ORDER):
            raise ValueError("initial model lock policy set is not exact")
        normalized[str(seed)] = {
            policy: dict(row[policy]) for policy in CALIBRATION_POLICY_ORDER
        }
    unsigned: dict[str, object] = {
        "calibration_selection_implementation": dict(implementation_manifest),
        "conversion_training": dict(conversion_training),
        "kind": INITIAL_MODEL_IDENTITY_KIND,
        "models": normalized,
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_payload_sha256": plan_payload_sha256,
        "policy_order": list(CALIBRATION_POLICY_ORDER),
        "producer_git_commit": producer_git_commit,
        "protocol": INITIAL_MODEL_IDENTITY_PROTOCOL,
        "result_inputs": dict(_RESULT_INPUTS),
        "schema_version": 2,
        "seed_order": list(INITIAL_SEEDS),
        "source_identities": dict(source_identities),
    }
    lock = {**unsigned, "lock_sha256": canonical_sha256(unsigned)}
    validate_initial_model_identity_lock_v2(lock)
    return lock


def _validate_model_identity(
    model: Mapping[str, object], *, seed: int, policy: str
) -> None:
    if not isinstance(model, Mapping) or set(model) != _MODEL_KEYS:
        raise ValueError("initial physical model identity schema differs")
    report = model.get("training_report")
    checkpoint = model.get("checkpoint")
    auxiliary = model.get("auxiliary")
    phase3 = policy in PHASE3_POLICIES
    family_root = "phase3" if phase3 else "phase3-compute-conversion"
    patch_count = 86 if phase3 else int(policy.rsplit("_", 1)[1])
    spec = PHASE3_MODEL_SPEC if phase3 else conversion_model_spec(patch_count)
    expected_report_path = f"runs/{family_root}/seed-{seed}/{policy}.json"
    expected_checkpoint_path = f"artifacts/{family_root}/seed-{seed}/{policy}.pt"
    if (
        model.get("seed") != seed
        or model.get("policy") != policy
        or model.get("model_family")
        != ("phase3" if phase3 else "compute_conversion")
        or model.get("patch_count") != patch_count
        or model.get("parameter_count") != INITIAL_MAIN_PARAMETER_COUNT
        or model.get("model_spec_sha256")
        != canonical_sha256(spec.to_dict())
        or model.get("optimization_spec_sha256")
        != canonical_sha256(PHASE3_OPTIMIZATION_SPEC.to_dict())
        or not is_sha256(model.get("initialization_sha256"))
        or not is_sha256(model.get("training_order_sha256"))
        or not isinstance(report, Mapping)
        or set(report) != _ARTIFACT_KEYS
        or report.get("path") != expected_report_path
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint) != _CHECKPOINT_KEYS
        or checkpoint.get("path") != expected_checkpoint_path
        or not is_sha256(report.get("artifact_sha256"))
        or not all(
            is_sha256(checkpoint.get(key))
            for key in ("artifact_sha256", "state_sha256")
        )
    ):
        raise ValueError("initial physical model identity is invalid")
    if policy in THRESHOLD_POLICIES:
        if (
            not isinstance(auxiliary, Mapping)
            or set(auxiliary) != _ENTROPY_KEYS
            or auxiliary.get("kind") != "entropy_router"
            or auxiliary.get("router_checkpoint_path")
            != f"artifacts/phase3/seed-{seed}/router.pt"
            or auxiliary.get("router_report_path")
            != f"runs/phase3/seed-{seed}/router.json"
            or auxiliary.get("threshold_cache_path")
            != f"artifacts/phase3/seed-{seed}/threshold-patches.npz"
            or auxiliary.get("threshold_diagnostics_path")
            != f"runs/phase3/seed-{seed}/threshold-patch-diagnostics.json"
            or not all(
                is_sha256(auxiliary.get(key))
                for key in _ENTROPY_KEYS
                if key.endswith("sha256")
            )
        ):
            raise ValueError("initial entropy model identity is incomplete")
    elif dict(auxiliary) != {"kind": "none"}:  # type: ignore[arg-type]
        raise ValueError("initial structural model identity claims an auxiliary")


def validate_initial_model_identity_lock_v2(
    lock: Mapping[str, Any],
) -> None:
    if not isinstance(lock, Mapping) or set(lock) != _TOP_KEYS:
        raise ValueError("initial model identity lock is not the sealed schema")
    implementation = lock.get("calibration_selection_implementation")
    models = lock.get("models")
    result_inputs = lock.get("result_inputs")
    unsigned = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if (
        lock.get("kind") != INITIAL_MODEL_IDENTITY_KIND
        or lock.get("protocol") != INITIAL_MODEL_IDENTITY_PROTOCOL
        or lock.get("schema_version") != 2
        or not is_sha256(lock.get("plan_artifact_sha256"))
        or not is_sha256(lock.get("plan_payload_sha256"))
        or not is_sha256(lock.get("lock_sha256"))
        or not is_git_commit(lock.get("producer_git_commit"))
        or lock.get("lock_sha256") != canonical_sha256(unsigned)
        or tuple(lock.get("seed_order", ())) != INITIAL_SEEDS
        or tuple(lock.get("policy_order", ())) != CALIBRATION_POLICY_ORDER
        or not isinstance(implementation, Mapping)
        or not isinstance(models, Mapping)
        or result_inputs != _RESULT_INPUTS
    ):
        raise ValueError("initial model identity lock identity is invalid")
    validate_implementation_manifest_v2(implementation)
    source = lock.get("source_identities")
    training = lock.get("conversion_training")
    _validate_source_identities(source)
    if not isinstance(source, Mapping):
        raise AssertionError("validated source identity disappeared")
    _validate_conversion_training(
        training,
        plan_artifact_sha256=lock["plan_artifact_sha256"],
        source_identities=source,
    )
    if tuple(sorted(int(seed) for seed in models)) != INITIAL_SEEDS:
        raise ValueError("initial model identity lock seed map differs")
    for seed in INITIAL_SEEDS:
        row = models.get(str(seed))
        if not isinstance(row, Mapping) or set(row) != set(
            CALIBRATION_POLICY_ORDER
        ):
            raise ValueError("initial model identity lock policy map differs")
        for policy in CALIBRATION_POLICY_ORDER:
            _validate_model_identity(row[policy], seed=seed, policy=policy)


def model_identity(
    lock: Mapping[str, Any], *, seed: int, policy: str
) -> Mapping[str, Any]:
    validate_initial_model_identity_lock_v2(lock)
    return lock["models"][str(seed)][policy]


def validate_current_implementation_v2(
    lock: Mapping[str, Any],
    *,
    sha256_by_path: Mapping[str, str],
    environment: Mapping[str, object],
) -> None:
    """Bind every downstream selection/confirmation process to the sealed code."""

    validate_initial_model_identity_lock_v2(lock)
    implementation = lock["calibration_selection_implementation"]
    order = tuple(implementation["file_order"])
    if (
        order != CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
        or set(sha256_by_path) != set(order)
        or len(sha256_by_path) != len(order)
        or any(
            sha256_by_path.get(path) != implementation["sha256_by_path"][path]
            for path in order
        )
        or dict(environment) != implementation["environment"]
        or canonical_sha256(environment) != implementation["environment_sha256"]
    ):
        raise ValueError("current selection/confirmation implementation differs")


def validate_selection_lock_identity_binding_v2(
    selection_lock: Mapping[str, Any],
    identity_lock: Mapping[str, Any],
) -> None:
    """Cross-check selection replay provenance against its physical trust root."""

    validate_initial_model_identity_lock_v2(identity_lock)
    replay = selection_lock.get("independent_calibration_recomputation")
    implementation = identity_lock["calibration_selection_implementation"]
    if (
        not isinstance(replay, Mapping)
        or replay.get("environment_sha256")
        != implementation["environment_sha256"]
        or replay.get("implementation_manifest_sha256")
        != implementation["manifest_sha256"]
    ):
        raise ValueError("selection replay is not bound to its implementation lock")
