#!/usr/bin/env python3
"""Train preregistered 64/72-patch C/W compute-conversion models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
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
    conversion_policy,
)
from jamoflow.hplt3_final_test import publish_no_clobber, validate_seal_envelope
from jamoflow.inference_calibration_evidence import (
    calibration_bpb_matrix,
    validate_calibration_evidence_manifest,
)
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_confirmation_evidence_v2 import (
    COMPUTE_CONFIRMATION_COMPLETION_PATH,
    build_confirmation_training_completion,
    validate_confirmation_training_completion,
)
from jamoflow.inference_initial_model_identity_v2 import (
    INITIAL_MODEL_IDENTITY_LOCK_PATH,
    runtime_environment_v2,
    validate_current_implementation_v2,
    validate_initial_model_identity_lock_v2,
    validate_selection_lock_identity_binding_v2,
)
from jamoflow.inference_selection_plan import (
    PHASE3_PRIMARY_SUMMARY_PATH,
    validate_selection_plan_v2,
)
from jamoflow.inference_selection_v2 import (
    build_selection_decision_v2,
    validate_selection_lock_v2,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import (
    evaluate_main_model,
    resolve_device,
    shuffled_indices,
    train_main_model,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import PHASE3_MODEL_SPEC, PHASE3_OPTIMIZATION_SPEC


INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_SEEDS = (57721, 65537)
SPLITS = ("train", "calibration", "test")
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
SELECTION_PLAN_PATH = Path(
    "data/manifests/phase3-inference-selection-plan-v2.json"
)
SELECTION_LOCK_PATH = Path(
    "results/phase3-inference-selection-v2/selection-lock.json"
)
CALIBRATION_EVIDENCE_PATH = Path(
    "results/phase3-inference-selection-v2/calibration-evidence.json"
)
FINAL_TEST_SEAL_PATH = Path("data/seals/hplt3-korean-final-test-v1.json")
CONVERSION_REPORT_KEYS = {
    "schema_version",
    "seed",
    "policy",
    "rate",
    "parameters",
    "initialization_sha256",
    "trained_state_sha256",
    "training_order_sha256",
    "checkpoint_artifact_sha256",
    "calibration_loss_artifact_sha256",
    "loss_artifact_sha256",
    "evidence_binding",
    "patch_matrix_sha256",
    "patch_diagnostics",
    "training",
    "evaluation",
    "model_spec",
    "optimization_spec",
    "global_max_position_embeddings",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise ValueError(f"partial JSON artifact requires forensic recovery: {temporary}")
    with temporary.open("x", encoding="utf-8") as output:
        output.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        raise ValueError(f"partial NPZ artifact requires forensic recovery: {temporary}")
    with temporary.open("xb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _save_state(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    if temporary.exists():
        raise ValueError(
            f"partial checkpoint requires forensic recovery: {temporary}"
        )
    with temporary.open("xb") as output:
        torch.save(state, output)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_status() -> str | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _clean_git_commit() -> str:
    commit = _git_commit()
    status = _git_status()
    if not commit or status is None or status.strip():
        raise ValueError(
            "compute-conversion evidence requires a clean committed worktree"
        )
    return commit


def _require_unchanged_clean_git(expected_commit: str) -> None:
    status = _git_status()
    if _git_commit() != expected_commit or status is None or status.strip():
        raise RuntimeError(
            "git HEAD/worktree changed during compute-conversion evidence execution"
        )


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(payload: Any) -> bytes:
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


def _git_path_history(path: Path) -> str:
    return subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _start_confirmation_attempt(
    *,
    artifact_root: Path,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    run_git_commit: str,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> tuple[Path, Path]:
    completion_path = Path(COMPUTE_CONFIRMATION_COMPLETION_PATH)
    if completion_path.exists() or _git_path_history(completion_path):
        raise ValueError("compute confirmation completion was already published")
    active = artifact_root / ".publication-confirmation-active.json"
    completed = artifact_root / ".publication-confirmation-completed.json"
    payload = {
        "family": "compute_conversion",
        "policies": list(policies),
        "run_git_commit": run_git_commit,
        "seeds": list(seeds),
        "selection_lock_artifact_sha256": selection_lock_artifact_sha256,
        "selection_lock_payload_sha256": selection_lock["lock_sha256"],
    }
    expected = _json_bytes(payload)
    if completed.exists():
        raise ValueError("compute confirmation completed marker requires forensic review")
    if active.exists():
        if active.is_symlink() or active.read_bytes() != expected:
            raise ValueError("compute confirmation active attempt differs")
    else:
        target_paths = [
            path
            for seed in seeds
            for policy in policies
            for path in (
                Path("runs/phase3-compute-conversion")
                / f"seed-{seed}"
                / f"{policy}.json",
                artifact_root / f"seed-{seed}" / f"{policy}.pt",
                artifact_root / f"seed-{seed}" / f"{policy}-test-nll.npz",
            )
        ]
        if any(path.exists() for path in target_paths):
            raise ValueError(
                "compute confirmation artifacts exist without their active attempt"
            )
        publish_no_clobber(active, expected)
    return active, completed


def _complete_confirmation_attempt(
    *,
    active: Path,
    completed: Path,
    selection_lock: Mapping[str, Any],
    selection_lock_artifact_sha256: str,
    identity: Mapping[str, Any],
    run_git_commit: str,
    manifest_path: Path,
    run_root: Path,
    artifact_root: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    units: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        units[seed] = {}
        for policy in policies:
            report_path = run_root / f"seed-{seed}" / f"{policy}.json"
            checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
            report = _read_json(report_path)
            unit = {
                "auxiliary": {"kind": "none"},
                "checkpoint_artifact_sha256": _sha256(checkpoint_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_state_sha256": report["trained_state_sha256"],
                "training_report_artifact_sha256": _sha256(report_path),
                "training_report_path": str(report_path),
            }
            if report.get("checkpoint_artifact_sha256") != unit[
                "checkpoint_artifact_sha256"
            ]:
                raise ValueError("compute confirmation checkpoint receipt differs")
            units[seed][policy] = unit
    implementation = identity["calibration_selection_implementation"]
    completion = build_confirmation_training_completion(
        selection_lock=selection_lock,
        selection_lock_artifact_sha256=selection_lock_artifact_sha256,
        family="compute_conversion",
        run_git_commit=run_git_commit,
        run_manifest={
            "artifact_sha256": _sha256(manifest_path),
            "path": str(manifest_path),
        },
        implementation_manifest_sha256=implementation["manifest_sha256"],
        environment_sha256=implementation["environment_sha256"],
        units=units,
    )
    validate_confirmation_training_completion(
        completion, selection_lock=selection_lock
    )
    publish_no_clobber(Path(COMPUTE_CONFIRMATION_COMPLETION_PATH), _json_bytes(completion))
    publish_no_clobber(
        completed,
        _json_bytes(
            {
                "completion_sha256": completion["completion_sha256"],
                "family": "compute_conversion",
            }
        ),
    )
    active.unlink()
    return completion


def _tracked_head_artifact_sha256(path: Path) -> str:
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    root = Path(top_level).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("evidence artifact is outside the repository") from error
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not path.is_file() or path.read_bytes() != result.stdout:
        raise ValueError(f"evidence artifact is not the exact committed HEAD blob: {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _validate_selection_lock_artifacts(
    selection_path: Path,
    selection: dict[str, Any],
) -> None:
    if selection_path.resolve() != SELECTION_LOCK_PATH.resolve():
        raise ValueError("confirmation requires the canonical selection-lock path")
    _tracked_head_artifact_sha256(selection_path)
    expected = (
        (SELECTION_PLAN_PATH, selection["plan_sha256"]),
        (
            CALIBRATION_EVIDENCE_PATH,
            selection["calibration_evidence_manifest_sha256"],
        ),
        (FINAL_TEST_SEAL_PATH, selection["final_test_seal_sha256"]),
        (
            Path(INITIAL_MODEL_IDENTITY_LOCK_PATH),
            selection["initial_model_identity_lock_sha256"],
        ),
    )
    for path, expected_sha256 in expected:
        if _tracked_head_artifact_sha256(path) != expected_sha256:
            raise ValueError(f"selection lock dependency differs from HEAD: {path}")
    plan = _read_json(SELECTION_PLAN_PATH)
    evidence = _read_json(CALIBRATION_EVIDENCE_PATH)
    validate_selection_plan_v2(plan)
    validate_calibration_evidence_manifest(evidence, plan=plan)
    canonical_decision = build_selection_decision_v2(
        calibration_bpb_matrix(evidence, plan=plan)
    )
    if selection["decision"] != canonical_decision:
        raise ValueError("selection lock decision differs from calibration evidence")
    final_seal = _read_json(FINAL_TEST_SEAL_PATH)
    validate_seal_envelope(final_seal)
    identity = _read_json(Path(INITIAL_MODEL_IDENTITY_LOCK_PATH))
    validate_initial_model_identity_lock_v2(identity)
    validate_selection_lock_identity_binding_v2(selection, identity)
    validate_current_implementation_v2(
        identity,
        sha256_by_path={
            path: _tracked_head_artifact_sha256(Path(path))
            for path in identity["calibration_selection_implementation"][
                "file_order"
            ]
        },
        environment=runtime_environment_v2(),
    )


def _validate_selection_plan_artifacts(plan: dict[str, Any]) -> None:
    validate_selection_plan_v2(plan)
    dependencies = (
        (
            Path(plan["final_test"]["manifest_path"]),
            plan["final_test"]["manifest_sha256"],
        ),
        (
            Path(plan["final_test"]["seal_path"]),
            plan["final_test"]["seal_sha256"],
        ),
        (
            Path(plan["historical_screening"]["all_initial_summary"]["path"]),
            plan["historical_screening"]["all_initial_summary"]["sha256"],
        ),
        (
            Path(plan["historical_screening"]["primary_summary"]["path"]),
            plan["historical_screening"]["primary_summary"]["sha256"],
        ),
    )
    for path, expected_sha256 in dependencies:
        if _tracked_head_artifact_sha256(path) != expected_sha256:
            raise ValueError(f"selection plan dependency differs from HEAD: {path}")
    final_seal = _read_json(Path(plan["final_test"]["seal_path"]))
    validate_seal_envelope(final_seal)
    if final_seal["payload_sha256"] != plan["final_test"]["seal_payload_sha256"]:
        raise ValueError("selection plan final-test payload identity differs")


def _release(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _rate_for_policy(policy: str) -> int:
    try:
        rate = int(policy.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"malformed conversion policy: {policy}") from error
    if rate not in CONVERSION_RATES or policy not in CONVERSION_POLICIES:
        raise ValueError(f"unknown conversion policy: {policy}")
    return rate


def _validate_stage(
    stage: str,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    selection_summary: Path | None,
) -> dict[str, Any] | None:
    if stage == "initial":
        if seeds != INITIAL_SEEDS or policies != CONVERSION_POLICIES:
            raise ValueError("initial conversion must run all preregistered seeds/policies")
        if selection_summary is not None:
            raise ValueError("initial conversion does not accept a selection summary")
        return None
    if stage != "confirmation" or seeds != CONFIRMATION_SEEDS:
        raise ValueError("confirmation conversion needs both confirmation seeds")
    if selection_summary is None:
        raise ValueError("confirmation conversion requires the selection-v2 lock")
    selection = _read_json(selection_summary)
    validate_selection_lock_v2(selection)
    decision = selection["decision"]
    rate = decision.get("rate_selection", {}).get("selected_rate")
    if rate not in CONVERSION_RATES:
        raise ValueError("selection-v2 lock has no confirmable conversion rate")
    confirmation = decision.get("confirmation_plan", {})
    compute_confirmation = (
        confirmation.get("compute_conversion", {})
        if isinstance(confirmation, dict)
        else {}
    )
    expected = tuple(compute_confirmation.get("policies", ()))
    if (
        decision.get("status") != "locked_pending_confirmation_and_new_final_test"
        or compute_confirmation.get("authorization_kind")
        != "compute_conversion_confirmation_v2"
        or compute_confirmation.get("selected_rate") != rate
        or tuple(compute_confirmation.get("seeds", ())) != CONFIRMATION_SEEDS
        or expected
        != (
            conversion_policy("codepoint", rate),
            conversion_policy("whitespace", rate),
        )
        or policies != expected
    ):
        raise ValueError("confirmation policies differ from the sealed selection-v2 lock")
    return selection


def _load_primary_gate(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    authorization = summary.get("confirmation_authorization")
    ood = summary.get("ood")
    if (
        summary.get("gate_i", {}).get("overall_pass") is not True
        or summary.get("gate_j", {}).get("overall_pass") is not True
        or summary.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or tuple(summary.get("seeds", []))
        != (*INITIAL_SEEDS, *CONFIRMATION_SEEDS)
        or tuple(summary.get("policies", []))
        != ("fixed_byte_6", "causal_codepoint_grid", "causal_whitespace_grid")
        or summary.get("targets_per_sequence") != 511
        or not isinstance(authorization, dict)
        or authorization.get("authorization_kind")
        != "phase3_corrected_gate_i_confirmation_v1"
        or not isinstance(ood, dict)
        or ood.get("gate_i_ood_guard", {}).get("pass") is not True
        or ood.get("integrity", {}).get("all_integrity_checks_pass") is not True
    ):
        raise ValueError(
            "compute conversion is blocked unless five-seed Gate J and OOD pass"
        )
    return summary


def _reconstruct_data(
    data_root: Path,
    primary: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    manifest = primary["run_manifest"]
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
    source_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    integrity_artifact = {
        "filename": "integrity.json",
        "bytes": integrity_path.stat().st_size,
        "sha256": _sha256(integrity_path),
    }
    if (
        manifest.get("source_artifact") != source_artifact
        or manifest.get("source_integrity_artifact") != integrity_artifact
    ):
        raise ValueError("conversion source artifacts differ from primary evidence")
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    stream_context: dict[str, Any] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            source_path,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=int(manifest["limits"][split]),
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        expected = manifest["streams"][split]
        stream_sha256 = hashlib.sha256(stream.data).hexdigest()
        if (
            stream_sha256 != expected["selected_stream_sha256"]
            or stream.sequence_count != expected["sequence_count"]
        ):
            raise ValueError(f"conversion stream differs from primary: {split}")
        split_whitespace = compact_whitespace_mask(stream.data).reshape(
            split_inputs.shape
        )
        inputs[split] = split_inputs
        boundaries[split] = split_boundaries
        whitespace[split] = split_whitespace
        stream_context[split] = {
            "selected_stream_sha256": stream_sha256,
            "inputs_sha256": _array_sha256(split_inputs),
            "boundaries_sha256": _array_sha256(split_boundaries),
            "whitespace_sha256": _array_sha256(split_whitespace),
            "sequence_count": len(split_inputs),
        }
        print(f"conversion data {split}: {len(split_inputs):,} rows", flush=True)
    return inputs, boundaries, whitespace, {
        "source_artifact": source_artifact,
        "source_integrity_artifact": integrity_artifact,
        "streams": stream_context,
    }


def _conversion_matrices(
    boundaries: dict[str, np.ndarray],
    whitespace: dict[str, np.ndarray],
    artifact_root: Path,
    run_root: Path,
) -> dict[str, dict[str, np.ndarray]]:
    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for rate in CONVERSION_RATES:
            matrices[split].update(
                conversion_patch_matrices(
                    boundaries[split],
                    whitespace[split],
                    rate=rate,
                )
            )
    arrays = {
        f"{split}__{policy}": matrices[split][policy]
        for split in SPLITS
        for policy in CONVERSION_POLICIES
    }
    cache_path = artifact_root / "patches.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays) or any(
                not np.array_equal(loaded[key], value)
                for key, value in arrays.items()
            ):
                raise ValueError("conversion patch cache differs from reconstruction")
    else:
        _save_npz(cache_path, **arrays)
    diagnostics = {
        "cache_artifact_sha256": _sha256(cache_path),
        "splits": {
            split: {
                policy: {
                    **variable_patch_diagnostics(
                        matrices[split][policy], boundaries[split]
                    ).to_dict(),
                    "matrix_sha256": _array_sha256(matrices[split][policy]),
                }
                for policy in CONVERSION_POLICIES
            }
            for split in SPLITS
        },
    }
    diagnostics_path = run_root / "patch-diagnostics.json"
    if diagnostics_path.exists() and _read_json(diagnostics_path) != diagnostics:
        raise ValueError("conversion patch diagnostics differ from reconstruction")
    _write_json(diagnostics_path, diagnostics)
    return matrices


def _completed_result_valid(
    report_path: Path,
    checkpoint_path: Path,
    calibration_loss_path: Path,
    loss_path: Path,
    *,
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    evidence_binding: dict[str, Any],
) -> bool:
    partials = tuple(
        path.with_suffix(path.suffix + ".part")
        for path in (
            report_path,
            checkpoint_path,
            calibration_loss_path,
            loss_path,
        )
    )
    if any(path.exists() for path in partials):
        raise ValueError(
            f"partial conversion staging artifact exists for {seed}/{policy}"
        )
    artifact_presence = tuple(
        path.exists()
        for path in (
            report_path,
            checkpoint_path,
            calibration_loss_path,
            loss_path,
        )
    )
    if not any(artifact_presence):
        return False
    if not all(artifact_presence):
        raise ValueError(
            f"partial conversion result for {seed}/{policy}; "
            "preserve it for forensic recovery and do not overwrite"
        )
    report = _read_json(report_path)
    rate = _rate_for_policy(policy)
    spec = conversion_model_spec(rate)
    model = build_main_model(
        spec,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    expected_parameters = parameter_count(model)
    expected_init = _state_sha256(model)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state_hash = _state_sha256(model)
    del model
    with np.load(loss_path, allow_pickle=False) as loaded:
        if loaded.files != ["sequence_nll_nats"]:
            raise ValueError("conversion loss artifact has unexpected keys")
        losses = loaded["sequence_nll_nats"]
    with np.load(calibration_loss_path, allow_pickle=False) as loaded:
        if loaded.files != ["sequence_nll_nats"]:
            raise ValueError("conversion calibration-loss artifact has unexpected keys")
        calibration_losses = loaded["sequence_nll_nats"]
    if (
        losses.dtype != np.float32
        or losses.shape != (len(inputs["test"]),)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
        or calibration_losses.dtype != np.float32
        or calibration_losses.shape != (len(inputs["calibration"]),)
        or not np.isfinite(calibration_losses).all()
        or np.any(calibration_losses < 0)
    ):
        raise ValueError(f"invalid conversion loss evidence: {seed}/{policy}")
    expected_bpb = float(losses.astype(np.float64).sum()) / (
        len(losses) * (PHASE3_MODEL_SPEC.sequence_length - 1) * np.log(2)
    )
    test_evaluation = report.get("evaluation", {}).get("test", {})
    calibration_evaluation = report.get("evaluation", {}).get("calibration", {})
    predicted_bytes = len(losses) * (PHASE3_MODEL_SPEC.sequence_length - 1)
    calibration_predicted_bytes = len(calibration_losses) * (
        PHASE3_MODEL_SPEC.sequence_length - 1
    )
    calibration_bpb = float(calibration_losses.astype(np.float64).sum()) / (
        calibration_predicted_bytes * np.log(2)
    )
    valid = bool(
        set(report) == CONVERSION_REPORT_KEYS
        and report.get("schema_version") == 2
        and report.get("seed") == seed
        and report.get("policy") == policy
        and report.get("rate") == rate
        and report.get("parameters") == expected_parameters
        and report.get("model_spec") == spec.to_dict()
        and report.get("optimization_spec") == PHASE3_OPTIMIZATION_SPEC.to_dict()
        and report.get("global_max_position_embeddings") == GLOBAL_POSITION_LIMIT
        and report.get("initialization_sha256") == expected_init
        and report.get("training_order_sha256")
        == _array_sha256(shuffled_indices(len(inputs["train"]), seed))
        and report.get("trained_state_sha256") == state_hash
        and report.get("checkpoint_artifact_sha256") == _sha256(checkpoint_path)
        and report.get("calibration_loss_artifact_sha256")
        == _sha256(calibration_loss_path)
        and report.get("loss_artifact_sha256") == _sha256(loss_path)
        and report.get("evidence_binding") == evidence_binding
        and calibration_evaluation.get("examples") == len(calibration_losses)
        and calibration_evaluation.get("predicted_bytes")
        == calibration_predicted_bytes
        and isinstance(calibration_evaluation.get("bpb"), (int, float))
        and np.isclose(
            float(calibration_evaluation["bpb"]),
            calibration_bpb,
            rtol=0,
            atol=1e-7,
        )
        and test_evaluation.get("examples") == len(losses)
        and test_evaluation.get("predicted_bytes") == predicted_bytes
        and isinstance(test_evaluation.get("bpb"), (int, float))
        and np.isclose(
            float(test_evaluation["bpb"]),
            expected_bpb,
            rtol=0,
            atol=1e-7,
        )
    )
    for split in SPLITS:
        matrix = matrices[split][policy]
        valid &= (
            report.get("patch_matrix_sha256", {}).get(split)
            == _array_sha256(matrix)
            and report.get("patch_diagnostics", {}).get(split)
            == variable_patch_diagnostics(matrix, boundaries[split]).to_dict()
        )
    if not valid:
        raise ValueError(f"stale conversion result: {seed}/{policy}")
    return True


def _train_policy(
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    device: str,
    run_root: Path,
    artifact_root: Path,
    evidence_binding: dict[str, Any],
) -> None:
    run_directory = run_root / f"seed-{seed}"
    artifact_directory = artifact_root / f"seed-{seed}"
    report_path = run_directory / f"{policy}.json"
    checkpoint_path = artifact_directory / f"{policy}.pt"
    calibration_loss_path = (
        artifact_directory / f"{policy}-calibration-nll.npz"
    )
    loss_path = artifact_directory / f"{policy}-test-nll.npz"
    if _completed_result_valid(
        report_path,
        checkpoint_path,
        calibration_loss_path,
        loss_path,
        seed=seed,
        policy=policy,
        inputs=inputs,
        boundaries=boundaries,
        matrices=matrices,
        evidence_binding=evidence_binding,
    ):
        print(f"seed {seed}/{policy}: conversion already complete", flush=True)
        return

    rate = _rate_for_policy(policy)
    spec = conversion_model_spec(rate)
    for split in SPLITS:
        validate_padded_patch_matrix(
            matrices[split][policy],
            PHASE3_MODEL_SPEC.sequence_length,
        )
    order = shuffled_indices(len(inputs["train"]), seed)
    model = build_main_model(
        spec,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization = _state_sha256(model)
    print(
        f"seed {seed}/{policy}: training {parameter_count(model):,} parameters",
        flush=True,
    )
    training = train_main_model(
        model,
        inputs["train"],
        matrices["train"][policy],
        order,
        device,
        PHASE3_OPTIMIZATION_SPEC,
    )
    evaluations: dict[str, Any] = {}
    split_losses: dict[str, np.ndarray] = {}
    for split in ("calibration", "test"):
        evaluation, losses = evaluate_main_model(
            model,
            inputs[split],
            matrices[split][policy],
            device,
            batch_size=PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
            return_sequence_nll=True,
        )
        evaluations[split] = evaluation.to_dict()
        if losses is None:
            raise AssertionError(f"conversion {split} losses were not produced")
        split_losses[split] = losses.astype(np.float32, copy=False)
    _save_npz(
        calibration_loss_path,
        sequence_nll_nats=split_losses["calibration"],
    )
    _save_npz(loss_path, sequence_nll_nats=split_losses["test"])
    trained_state = _state_sha256(model)
    _save_state(checkpoint_path, model)
    checkpoint_hash = _sha256(checkpoint_path)
    _write_json(
        report_path,
        {
            "schema_version": 2,
            "seed": seed,
            "policy": policy,
            "rate": rate,
            "parameters": parameter_count(model),
            "initialization_sha256": initialization,
            "trained_state_sha256": trained_state,
            "training_order_sha256": _array_sha256(order),
            "checkpoint_artifact_sha256": checkpoint_hash,
            "calibration_loss_artifact_sha256": _sha256(
                calibration_loss_path
            ),
            "loss_artifact_sha256": _sha256(loss_path),
            "evidence_binding": evidence_binding,
            "patch_matrix_sha256": {
                split: _array_sha256(matrices[split][policy])
                for split in SPLITS
            },
            "patch_diagnostics": {
                split: variable_patch_diagnostics(
                    matrices[split][policy], boundaries[split]
                ).to_dict()
                for split in SPLITS
            },
            "training": training.to_dict(),
            "evaluation": evaluations,
            "model_spec": spec.to_dict(),
            "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
            "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        },
    )
    _release(model, device)


def _run_locked(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    policies = tuple(args.policies)
    run_git_commit = _clean_git_commit()
    device = resolve_device(args.device)
    selection = _validate_stage(
        args.stage,
        seeds,
        policies,
        args.selection_summary,
    )
    selection_plan_sha256 = _tracked_head_artifact_sha256(SELECTION_PLAN_PATH)
    selection_plan = _read_json(SELECTION_PLAN_PATH)
    _validate_selection_plan_artifacts(selection_plan)
    if selection is not None:
        if args.selection_summary is None:
            raise AssertionError("confirmation selection path disappeared")
        _validate_selection_lock_artifacts(args.selection_summary, selection)
        if selection["plan_sha256"] != selection_plan_sha256:
            raise ValueError("selection lock was not built from the committed plan")
    primary_path = Path(args.primary_summary)
    primary = _load_primary_gate(primary_path)
    planned_primary = selection_plan["historical_screening"]["primary_summary"]
    if (
        primary_path.resolve() != Path(planned_primary["path"]).resolve()
        or _sha256(primary_path) != planned_primary["sha256"]
    ):
        raise ValueError("conversion primary summary differs from the selection plan")
    evidence_binding_payload = {
        "git_commit": run_git_commit,
        "git_worktree_clean_at_start": True,
        "device": device,
        "policies": list(policies),
        "primary_summary_sha256": _sha256(primary_path),
        "schema_version": 1,
        "selection_plan_sha256": selection_plan_sha256,
        "seeds": list(seeds),
        "selection_summary_sha256": (
            _sha256(args.selection_summary)
            if args.selection_summary is not None
            else None
        ),
        "stage": args.stage,
    }
    evidence_binding = {
        **evidence_binding_payload,
        "identity_sha256": _canonical_sha256(evidence_binding_payload),
    }
    data_root = Path(args.data_root)
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    identity: dict[str, Any] | None = None
    if args.stage == "confirmation":
        if (
            device != "mps"
            or data_root != Path("data/processed/hplt3-korean-phase3")
            or run_root != Path("runs/phase3-compute-conversion")
            or artifact_root != Path("artifacts/phase3-compute-conversion")
        ):
            raise ValueError(
                "publication confirmation requires canonical roots and Apple MPS"
            )
        identity = _read_json(Path(INITIAL_MODEL_IDENTITY_LOCK_PATH))
    inputs, boundaries, whitespace, source_context = _reconstruct_data(
        data_root,
        primary,
    )
    if (
        source_context["source_artifact"]["sha256"]
        != selection_plan["initial_design"]["source_artifact_sha256"]
        or source_context["source_integrity_artifact"]["sha256"]
        != selection_plan["initial_design"]["source_integrity_artifact_sha256"]
        or source_context["streams"]["calibration"]["selected_stream_sha256"]
        != selection_plan["calibration_evaluator"]["input_stream_sha256"]
        or source_context["streams"]["calibration"]["sequence_count"]
        != selection_plan["calibration_evaluator"]["sequence_count"]
    ):
        raise ValueError("conversion source/calibration differs from the selection plan")
    matrices = _conversion_matrices(
        boundaries,
        whitespace,
        artifact_root,
        run_root,
    )
    manifest_path = run_root / "manifest.json"
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": run_git_commit,
        "git_worktree_clean_at_start": True,
        "primary_gate_summary": str(primary_path),
        "primary_gate_summary_sha256": _sha256(primary_path),
        "primary_gate_i": primary["gate_i"],
        "selection_plan": str(SELECTION_PLAN_PATH),
        "selection_plan_sha256": selection_plan_sha256,
        "source_context": source_context,
        "rates": list(CONVERSION_RATES),
        "policies": list(CONVERSION_POLICIES),
        "model_specs": {
            str(rate): conversion_model_spec(rate).to_dict()
            for rate in CONVERSION_RATES
        },
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "invocations": [
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": run_git_commit,
                "git_worktree_clean_at_start": True,
                "stage": args.stage,
                "seeds": list(seeds),
                "policies": list(policies),
                "selection_summary": (
                    str(args.selection_summary)
                    if args.selection_summary is not None
                    else None
                ),
                "selection_summary_sha256": (
                    _sha256(args.selection_summary)
                    if args.selection_summary is not None
                    else None
                ),
                "evidence_binding": evidence_binding,
            }
        ],
    }
    if manifest_path.exists():
        previous = _read_json(manifest_path)
        invariant_keys = (
            "schema_version",
            "primary_gate_summary",
            "primary_gate_summary_sha256",
            "primary_gate_i",
            "selection_plan",
            "selection_plan_sha256",
            "source_context",
            "rates",
            "policies",
            "model_specs",
            "optimization_spec",
            "global_max_position_embeddings",
            "git_worktree_clean_at_start",
        )
        if any(previous.get(key) != manifest.get(key) for key in invariant_keys):
            raise ValueError("compute-conversion manifest invariant changed")
        invocations = previous.get("invocations", [])
        for invocation in manifest["invocations"]:
            if invocation not in invocations:
                invocations.append(invocation)
        manifest["created_at"] = previous["created_at"]
        manifest["git_commit"] = previous.get("git_commit")
        manifest["invocations"] = invocations
    active: Path | None = None
    completed: Path | None = None
    if args.stage == "confirmation":
        if args.selection_summary is None or identity is None or selection is None:
            raise AssertionError("confirmation identity context disappeared")
        active, completed = _start_confirmation_attempt(
            artifact_root=artifact_root,
            selection_lock=selection,
            selection_lock_artifact_sha256=_sha256(args.selection_summary),
            run_git_commit=run_git_commit,
            seeds=seeds,
            policies=policies,
        )
    _write_json(manifest_path, manifest)

    for seed in seeds:
        for policy in policies:
            _train_policy(
                seed,
                policy,
                inputs,
                boundaries,
                matrices,
                device,
                run_root,
                artifact_root,
                evidence_binding,
            )
    _require_unchanged_clean_git(run_git_commit)
    if args.stage == "confirmation":
        if (
            active is None
            or completed is None
            or args.selection_summary is None
            or identity is None
            or selection is None
        ):
            raise AssertionError("confirmation completion context disappeared")
        completion = _complete_confirmation_attempt(
            active=active,
            completed=completed,
            selection_lock=selection,
            selection_lock_artifact_sha256=_sha256(args.selection_summary),
            identity=identity,
            run_git_commit=run_git_commit,
            manifest_path=manifest_path,
            run_root=run_root,
            artifact_root=artifact_root,
            seeds=seeds,
            policies=policies,
        )
        print(
            json.dumps(
                {
                    "completion_sha256": completion["completion_sha256"],
                    "status": "complete_pending_receipt_commit",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print("completed compute-conversion training", flush=True)
    return 0


def run(args: argparse.Namespace) -> int:
    with publication_mps_exclusive():
        return _run_locked(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("initial", "confirmation"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument(
        "--primary-summary",
        default=PHASE3_PRIMARY_SUMMARY_PATH,
    )
    parser.add_argument("--selection-summary", type=Path)
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--run-root", default="runs/phase3-compute-conversion")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-compute-conversion",
    )
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
