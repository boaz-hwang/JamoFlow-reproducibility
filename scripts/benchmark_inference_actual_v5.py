#!/usr/bin/env python3
"""Run exactly one fresh publication timing session from the sealed v5 plan."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.incremental_blt import (
    IncrementalEntropyRouter,
    structural_prefix_boundaries,
)
from jamoflow.inference_actual_runtime_v5 import (
    ACTUAL_INFERENCE_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_EQUIVALENCE_RTOL,
    ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION,
    ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL,
    ACTUAL_INFERENCE_PATCH_HORIZON,
    LoadedActualModel,
    full_entropy_boundaries,
    full_main_logits,
    full_router_trace,
    load_actual_model,
    model_spec_for_descriptor,
    release_actual_model,
)
from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_ARTIFACT_ROOT,
    ACTUAL_INFERENCE_V5_CASE_PATH,
    ACTUAL_INFERENCE_V5_COMPONENTS,
    ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
    ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
    ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES,
    ACTUAL_INFERENCE_V5_MEASURED_CASES,
    ACTUAL_INFERENCE_V5_MODES,
    ACTUAL_INFERENCE_V5_PLAN_PATH,
    ACTUAL_INFERENCE_V5_PROTOCOL_REVISION,
    ACTUAL_INFERENCE_V5_PROMPT_BYTES,
    ACTUAL_INFERENCE_V5_REPETITIONS,
    ACTUAL_INFERENCE_V5_ROLES,
    ACTUAL_INFERENCE_V5_SESSION_ROOT,
    ACTUAL_INFERENCE_V5_SESSION_RECEIPT_ROOT,
    ACTUAL_INFERENCE_V5_SESSIONS,
    ACTUAL_INFERENCE_V5_WARMUP_CASES,
    MPS_ENTRYPOINT_MARKERS,
    RUNTIME_COUNTER_NAMES,
    array_sha256,
    canonical_sha256,
    current_runtime_environment_contract,
    session_schedule,
    validate_actual_inference_plan_v5,
    validate_runtime_counter_arrays,
)
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_SEEDS,
    SELECTION_LOCK_PATH,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.phase2_patching import padded_hf_patch_matrix
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    StrictUtf8State,
    advance_strict_utf8,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
    strict_utf8_state,
)


PLAN_PATH = Path(ACTUAL_INFERENCE_V5_PLAN_PATH)
CASE_PATH = Path(ACTUAL_INFERENCE_V5_CASE_PATH)
AUTHORIZATION_PATH = Path(FINAL_AUTHORIZATION_PATH)
QUALITY_LOCK_PATH = Path(FINAL_QUALITY_LOCK_PATH)
SELECTION_PATH = Path(SELECTION_LOCK_PATH)
RUN_ROOT = Path(ACTUAL_INFERENCE_V5_SESSION_ROOT)
SESSION_RECEIPT_ROOT = Path(ACTUAL_INFERENCE_V5_SESSION_RECEIPT_ROOT)
ARTIFACT_ROOT = Path(ACTUAL_INFERENCE_V5_ARTIFACT_ROOT)
ACTIVE_SENTINEL = ARTIFACT_ROOT / ".active"
PROCESS_LOCK_PATH = ARTIFACT_ROOT / ".process.lock"
MACHINE_LOCK_PATH = Path("/tmp/jamoflow-publication-mps.lock")


@dataclass(frozen=True, slots=True)
class TimedTrial:
    ttft_ms: float
    decode_ms: float
    end_to_end_ms: float
    emitted_output_bytes: int
    emitted_global_patches: int
    runtime_observed_bytes: int
    counters: dict[str, int]
    generated: bytes | None


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


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
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
        raise ValueError("actual timing requires a clean repository root")
    commit = _git_commit()
    if len(commit) != 40:
        raise ValueError("actual timing requires a Git commit")
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
        raise ValueError(f"actual timing input is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit_value,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(blob.stdout).hexdigest(),
    }


def _tracked_history_exists(path: Path) -> bool:
    result = subprocess.run(
        ["git", "log", "--all", "-1", "--format=%H", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"actual timing receipt history check failed: {path}")
    return bool(result.stdout.strip())


def _tracked_touch_count(path: Path) -> int:
    result = subprocess.run(
        ["git", "rev-list", "--all", "--count", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        count = int(result.stdout.strip())
    except ValueError as error:
        raise ValueError("actual timing receipt history is malformed") from error
    if result.returncode != 0 or count < 0:
        raise ValueError("actual timing receipt history check failed")
    return count


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode != 0:
        raise ValueError(f"actual timing Git order differs: {label}")


def _command_snapshot(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "command": command,
            "returncode": None,
            "stderr": str(error),
            "stdout": "",
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "stdout": result.stdout.strip(),
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _command_snapshot(["pmset", "-g", "batt"]),
        "settings": _command_snapshot(["pmset", "-g", "custom"]),
        "thermal": _command_snapshot(["pmset", "-g", "therm"]),
    }


def _process_inventory() -> dict[str, Any]:
    snapshot = _command_snapshot(
        ["ps", "-axo", "pid=,ppid=,lstart=,command="]
    )
    output = str(snapshot.get("stdout", ""))
    parsed = []
    for line in output.splitlines():
        parts = line.split(maxsplit=7)
        if len(parts) != 8:
            continue
        try:
            pid, parent = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        parsed.append((pid, parent, parts[7], line.strip()))
    parents = {pid: parent for pid, parent, _, _ in parsed}
    exempt = {os.getpid()}
    cursor = os.getpid()
    while cursor in parents and parents[cursor] > 0 and parents[cursor] not in exempt:
        cursor = parents[cursor]
        exempt.add(cursor)
    conflicts = [
        raw
        for pid, _, command, raw in parsed
        if pid not in exempt
        and any(marker in command for marker in MPS_ENTRYPOINT_MARKERS)
    ]
    return {
        "conflicting_neural_processes": conflicts,
        "ps_parse_pass": bool(parsed) and os.getpid() in parents,
        "ps_returncode": snapshot["returncode"],
        "ps_stdout_nonempty": bool(output.strip()),
        "snapshot_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _environment() -> dict[str, Any]:
    return {
        "device": "mps",
        "mps_available": torch.backends.mps.is_available(),
        **current_runtime_environment_contract(),
    }


def _assert_no_symlink_namespace(root: Path) -> None:
    workspace = Path.cwd().absolute()
    target = root.absolute() if root.is_absolute() else (workspace / root)
    try:
        relative = target.relative_to(workspace)
    except ValueError as error:
        raise ValueError("actual timing namespace is outside the repository") from error
    cursor = workspace
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"actual timing namespace contains a symlink: {cursor}")
    if workspace.resolve() not in (target.resolve(), *target.resolve().parents):
        raise ValueError("actual timing namespace resolves outside the repository")


@contextmanager
def _exclusive_process_lock():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_namespace(ARTIFACT_ROOT)
    with MACHINE_LOCK_PATH.open("a+b") as machine_handle, PROCESS_LOCK_PATH.open(
        "a+b"
    ) as handle:
        try:
            fcntl.flock(
                machine_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another actual-inference process is live") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            fcntl.flock(machine_handle.fileno(), fcntl.LOCK_UN)


def _model_for_role(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    identity = plan["timing_pair"]["roles"][role]["model_identity_sha256"]
    matches = [
        model
        for model in authorization["models"]
        if model["identity_sha256"] == identity
    ]
    if len(matches) != 1:
        raise ValueError("actual timing role has no unique model identity")
    return matches[0]


def _load_context() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str
]:
    commit = _require_clean_root()
    plan_identity = _tracked_head_identity(PLAN_PATH)
    authorization_identity = _tracked_head_identity(AUTHORIZATION_PATH)
    quality_identity = _tracked_head_identity(QUALITY_LOCK_PATH)
    selection_identity = _tracked_head_identity(SELECTION_PATH)
    plan = _read_json(PLAN_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)
    quality = _read_json(QUALITY_LOCK_PATH)
    selection = _read_json(SELECTION_PATH)
    validate_selection_lock_v2(selection)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection,
    )
    validate_actual_inference_plan_v5(
        plan,
        quality_lock=quality,
        authorization=authorization,
    )
    current_environment = _environment()
    if (
        plan["authorization_artifact"] != authorization_identity
        or plan["quality_lock_artifact"] != quality_identity
        or authorization["upstream_artifacts"]["selection_lock"]["sha256"]
        != selection_identity["sha256"]
        or {
            key: current_environment[key]
            for key in (
                "hardware",
                "machine",
                "packages",
                "platform",
                "python",
                "system",
            )
        }
        != plan["runtime_environment_contract"]
        or not torch.backends.mps.is_available()
    ):
        raise ValueError("actual timing upstream/runtime identity differs")
    _require_ancestor(
        plan["plan_base_git_commit"],
        plan_identity["git_commit"],
        "plan implementation -> plan artifact",
    )
    _require_ancestor(
        plan_identity["git_commit"],
        commit,
        "plan artifact -> timing session",
    )
    for path, expected in plan["implementation_sha256"].items():
        if _tracked_head_identity(Path(path))["sha256"] != expected:
            raise ValueError(f"actual timing implementation differs: {path}")
    if hash_file(CASE_PATH) != plan["case_context"]["artifact_sha256"]:
        raise ValueError("actual timing cases artifact differs")
    return plan, authorization, quality, selection, plan_identity["sha256"]


def _load_cases(plan: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(CASE_PATH, allow_pickle=False) as archive:
        if set(archive.files) != {"prompts", "replay_continuations"}:
            raise ValueError("actual timing case artifact schema differs")
        prompts = archive["prompts"]
        continuations = archive["replay_continuations"]
    expected_shape = (
        ACTUAL_INFERENCE_V5_WARMUP_CASES
        + ACTUAL_INFERENCE_V5_MEASURED_CASES,
        ACTUAL_INFERENCE_V5_PROMPT_BYTES,
    )
    if (
        prompts.dtype != np.uint8
        or prompts.shape != expected_shape
        or continuations.dtype != np.uint8
        or continuations.shape
        != (
            expected_shape[0],
            ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
        )
        or array_sha256(prompts)
        != plan["case_context"]["prompt_array_sha256"]
        or array_sha256(continuations)
        != plan["case_context"]["continuation_array_sha256"]
    ):
        raise ValueError("actual timing cases differ from their plan")
    return prompts, continuations


def _session_paths(session_id: str) -> dict[str, Path]:
    return {
        "report": SESSION_RECEIPT_ROOT / f"{session_id}.json",
        "timings": ARTIFACT_ROOT / session_id / "timings.npz",
        "outputs": ARTIFACT_ROOT / session_id / "free-outputs.npz",
    }


def _completed_session_valid(
    session_id: str,
    *,
    plan_sha256: str,
) -> bool:
    paths = _session_paths(session_id)
    exists = {name: path.exists() for name, path in paths.items()}
    if not any(exists.values()):
        if _tracked_history_exists(paths["report"]):
            raise ValueError(
                f"deleted actual timing receipt forbids rerun: {session_id}"
            )
        return False
    if not all(exists.values()):
        raise ValueError(f"partial actual timing session requires review: {session_id}")
    _tracked_head_identity(paths["report"])
    if _tracked_touch_count(paths["report"]) != 1:
        raise ValueError(f"actual timing receipt was rewritten: {session_id}")
    report = _read_json(paths["report"])
    if (
        report.get("complete") is not True
        or report.get("session_id") != session_id
        or report.get("plan_sha256") != plan_sha256
        or report.get("timing_artifact_sha256") != hash_file(paths["timings"])
        or report.get("output_artifact_sha256") != hash_file(paths["outputs"])
    ):
        raise ValueError(f"completed actual timing session differs: {session_id}")
    return True


def _next_session(plan: Mapping[str, Any]) -> str | None:
    complete = [
        _completed_session_valid(
            session_id,
            plan_sha256=plan["plan_sha256"],
        )
        for session_id in ACTUAL_INFERENCE_V5_SESSIONS
    ]
    if any(
        not complete[index] and any(complete[index + 1 :])
        for index in range(len(complete))
    ):
        raise ValueError("actual timing sessions are not a complete prefix")
    return (
        None
        if all(complete)
        else ACTUAL_INFERENCE_V5_SESSIONS[complete.index(False)]
    )


def _runtime_main_diagnostics(runtime: Any) -> Any:
    diagnostics = runtime.diagnostics
    return diagnostics.main if hasattr(diagnostics, "main") else diagnostics


def _assert_cache(runtime: Any, expected_bytes: int) -> None:
    diagnostics = runtime.diagnostics
    main = _runtime_main_diagnostics(runtime)
    if (
        main.observed_bytes != expected_bytes
        or main.local_encoder_cached_bytes != expected_bytes
        or main.local_decoder_cached_bytes != expected_bytes
        or main.global_cached_patches != main.emitted_data_patches
    ):
        raise AssertionError("actual timing main cache invariants differ")
    if hasattr(diagnostics, "router_cached_bytes") and (
        diagnostics.router_cached_bytes != expected_bytes
    ):
        raise AssertionError("actual timing router cache invariant differs")


@dataclass(frozen=True, slots=True)
class LogitComparison:
    maximum_absolute_error: float
    maximum_normalized_tolerance_ratio: float
    maximum_nominal_normalized_tolerance_ratio: float
    maximum_probability_total_variation: float
    nominal_tolerance_violation_elements: int
    exact_argmax_count: int
    tolerance_tie_argmax_count: int


def _comparison_parameters(contract: str) -> tuple[float, float]:
    if contract == "cpu_semantic":
        return ACTUAL_INFERENCE_EQUIVALENCE_ATOL, ACTUAL_INFERENCE_EQUIVALENCE_RTOL
    if contract == "mps_backend":
        return (
            ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL,
            ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL,
        )
    raise ValueError("actual timing comparison contract differs")


def _compare_logits(
    left: Any,
    right: Any,
    *,
    contract: str = "cpu_semantic",
) -> LogitComparison:
    atol, rtol = _comparison_parameters(contract)
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise AssertionError("actual timing equivalence logits are non-finite")
    torch.testing.assert_close(
        left,
        right,
        rtol=rtol,
        atol=atol,
    )
    left_argmax = left.argmax(dim=-1)
    right_argmax = right.argmax(dim=-1)
    exact_argmax = left_argmax == right_argmax
    mismatch_rows = (~exact_argmax).nonzero(as_tuple=False).flatten()
    for row in mismatch_rows.tolist():
        left_index = int(left_argmax[row].item())
        right_index = int(right_argmax[row].item())
        left_value_at_left = left.float()[row, left_index]
        left_value_at_right = left.float()[row, right_index]
        right_value_at_left = right.float()[row, left_index]
        right_value_at_right = right.float()[row, right_index]
        left_tolerance_at_left = (
            atol + rtol * right_value_at_left.abs()
        )
        left_tolerance_at_right = (
            atol + rtol * right_value_at_right.abs()
        )
        left_preference_bound = (
            left_value_at_left
            - left_value_at_right
            - left_tolerance_at_left
            - left_tolerance_at_right
        )
        right_preference_bound = (
            right_value_at_right
            - right_value_at_left
            - left_tolerance_at_right
            - left_tolerance_at_left
        )
        if left_preference_bound > 0 or right_preference_bound > 0:
            raise AssertionError(
                "actual timing equivalence stable argmax differs"
            )
    difference = (left.float() - right.float()).abs()
    tolerance = atol + rtol * right.float().abs()
    nominal_tolerance = (
        ACTUAL_INFERENCE_EQUIVALENCE_ATOL
        + ACTUAL_INFERENCE_EQUIVALENCE_RTOL * right.float().abs()
    )
    normalized_ratio = float((difference / tolerance).max().item())
    if not math.isfinite(normalized_ratio) or normalized_ratio > 1.0:
        raise AssertionError("actual timing equivalence tolerance ratio differs")
    nominal_ratio = float((difference / nominal_tolerance).max().item())
    left_probability = torch.softmax(left.float(), dim=-1)
    right_probability = torch.softmax(right.float(), dim=-1)
    maximum_probability_total_variation = float(
        (
            0.5
            * (left_probability - right_probability).abs().sum(dim=-1)
        ).max().item()
    )
    if (
        not math.isfinite(nominal_ratio)
        or not math.isfinite(maximum_probability_total_variation)
        or maximum_probability_total_variation
        > ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
    ):
        raise AssertionError("actual timing probability-distribution evidence differs")
    nominal_violations = int((difference > nominal_tolerance).sum().item())
    if contract == "cpu_semantic" and nominal_violations != 0:
        raise AssertionError("CPU semantic oracle exceeds original tolerance")
    return LogitComparison(
        maximum_absolute_error=float(difference.max().item()),
        maximum_normalized_tolerance_ratio=normalized_ratio,
        maximum_nominal_normalized_tolerance_ratio=nominal_ratio,
        maximum_probability_total_variation=maximum_probability_total_variation,
        nominal_tolerance_violation_elements=nominal_violations,
        exact_argmax_count=int(exact_argmax.sum().item()),
        tolerance_tie_argmax_count=int((~exact_argmax).sum().item()),
    )


def _compare_entropy_values(
    left: np.ndarray,
    right: np.ndarray,
    *,
    contract: str,
) -> tuple[float, float, float, int]:
    atol, rtol = _comparison_parameters(contract)
    if (
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or not np.allclose(
            left,
            right,
            rtol=rtol,
            atol=atol,
        )
    ):
        raise AssertionError("actual timing router entropy differs")
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    tolerance = atol + rtol * np.abs(right.astype(np.float64))
    nominal_tolerance = (
        ACTUAL_INFERENCE_EQUIVALENCE_ATOL
        + ACTUAL_INFERENCE_EQUIVALENCE_RTOL * np.abs(right.astype(np.float64))
    )
    normalized_ratio = float(np.max(difference / tolerance))
    if not math.isfinite(normalized_ratio) or normalized_ratio > 1.0:
        raise AssertionError("actual timing router entropy tolerance ratio differs")
    nominal_ratio = float(np.max(difference / nominal_tolerance))
    nominal_violations = int(np.sum(difference > nominal_tolerance))
    if (
        not math.isfinite(nominal_ratio)
        or (contract == "cpu_semantic" and nominal_violations != 0)
    ):
        raise AssertionError("actual timing router nominal tolerance differs")
    return (
        float(np.max(difference)),
        normalized_ratio,
        nominal_ratio,
        nominal_violations,
    )


def _synchronize(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _boundary_trace_sha256(rows: list[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"JamoFlow/actual-v5/boundary-trace/v1\0")
    for row in rows:
        digest.update(len(row).to_bytes(8, "big"))
        for value in row:
            digest.update(int(value).to_bytes(8, "big"))
    return digest.hexdigest()


def _verify_bundle(
    bundle: LoadedActualModel,
    prompts: np.ndarray,
    continuations: np.ndarray,
    *,
    comparison_contract: str = "cpu_semantic",
) -> dict[str, Any]:
    device_matches = (
        bundle.device == "cpu"
        if comparison_contract == "cpu_semantic"
        else bundle.device.startswith("mps")
    )
    if not device_matches:
        raise ValueError("actual timing comparison device/contract differs")
    active_atol, active_rtol = _comparison_parameters(comparison_contract)
    full_main_positions = 0
    parallel_positions = 0
    router_positions = 0
    maximum_main_error = 0.0
    maximum_main_normalized_ratio = 0.0
    maximum_main_nominal_normalized_ratio = 0.0
    maximum_main_probability_total_variation = 0.0
    main_nominal_tolerance_violations = 0
    maximum_router_logit_error = 0.0
    maximum_router_logit_normalized_ratio = 0.0
    maximum_router_logit_nominal_normalized_ratio = 0.0
    maximum_router_probability_total_variation = 0.0
    router_logit_nominal_tolerance_violations = 0
    maximum_router_entropy_error = 0.0
    maximum_router_entropy_normalized_ratio = 0.0
    maximum_router_entropy_nominal_normalized_ratio = 0.0
    router_entropy_nominal_tolerance_violations = 0
    main_full_exact_argmax = 0
    main_full_tolerance_ties = 0
    main_parallel_exact_argmax = 0
    main_parallel_tolerance_ties = 0
    router_exact_argmax = 0
    router_tolerance_ties = 0
    boundary_rows: list[tuple[int, ...]] = []
    with torch.inference_mode():
        for prompt, continuation in zip(prompts, continuations, strict=True):
            raw_prompt = bytes(prompt)
            observed = raw_prompt + bytes(continuation[:-1])
            sequential = bundle.runtime()
            sequential_logits = []
            sequential_boundaries = []
            for value in observed:
                sequential_logits.append(sequential.consume(value))
                sequential_boundaries.append(
                    _runtime_main_diagnostics(sequential).boundaries
                )
            stacked = torch.cat(sequential_logits, dim=0)
            if bundle.requires_entropy_router:
                full_router_logits, full_entropies = full_router_trace(
                    bundle, observed
                )
                cached_router = IncrementalEntropyRouter(bundle.router)
                cached_logits = []
                cached_entropies = []
                for value in observed:
                    logits, entropy = cached_router.consume(value)
                    cached_logits.append(logits)
                    cached_entropies.append(entropy)
                cached = torch.cat(cached_logits, dim=0)
                router_comparison = _compare_logits(
                    cached,
                    full_router_logits,
                    contract=comparison_contract,
                )
                maximum_router_logit_error = max(
                    maximum_router_logit_error,
                    router_comparison.maximum_absolute_error,
                )
                maximum_router_logit_normalized_ratio = max(
                    maximum_router_logit_normalized_ratio,
                    router_comparison.maximum_normalized_tolerance_ratio,
                )
                maximum_router_logit_nominal_normalized_ratio = max(
                    maximum_router_logit_nominal_normalized_ratio,
                    router_comparison.maximum_nominal_normalized_tolerance_ratio,
                )
                maximum_router_probability_total_variation = max(
                    maximum_router_probability_total_variation,
                    router_comparison.maximum_probability_total_variation,
                )
                router_logit_nominal_tolerance_violations += (
                    router_comparison.nominal_tolerance_violation_elements
                )
                router_exact_argmax += router_comparison.exact_argmax_count
                router_tolerance_ties += (
                    router_comparison.tolerance_tie_argmax_count
                )
                cached_entropy_array = np.asarray(
                    cached_entropies, dtype=np.float32
                )
                (
                    router_entropy_error,
                    router_entropy_ratio,
                    router_entropy_nominal_ratio,
                    router_entropy_nominal_violations,
                ) = _compare_entropy_values(
                    cached_entropy_array,
                    full_entropies,
                    contract=comparison_contract,
                )
                maximum_router_entropy_error = max(
                    maximum_router_entropy_error,
                    router_entropy_error,
                )
                maximum_router_entropy_normalized_ratio = max(
                    maximum_router_entropy_normalized_ratio,
                    router_entropy_ratio,
                )
                maximum_router_entropy_nominal_normalized_ratio = max(
                    maximum_router_entropy_nominal_normalized_ratio,
                    router_entropy_nominal_ratio,
                )
                router_entropy_nominal_tolerance_violations += (
                    router_entropy_nominal_violations
                )
                expected_boundaries = full_entropy_boundaries(
                    bundle,
                    observed,
                    full_entropies,
                )
                router_positions += len(observed)
            else:
                expected_boundaries = structural_prefix_boundaries(
                    observed,
                    bundle.runtime_policy,
                    horizon=ACTUAL_INFERENCE_PATCH_HORIZON,
                    patch_count=bundle.patch_count,
                    fixed_stride=model_spec_for_descriptor(
                        bundle.descriptor
                    ).patch_stride,
                )
            if sequential_boundaries[-1] != expected_boundaries:
                raise AssertionError("actual timing online boundaries differ")
            for position, boundaries in enumerate(sequential_boundaries):
                expected_prefix = tuple(
                    value for value in expected_boundaries if value <= position
                )
                if boundaries != expected_prefix:
                    raise AssertionError("actual timing boundary prefix differs")
            boundary_rows.append(expected_boundaries)
            full_logits = full_main_logits(bundle, observed, expected_boundaries)
            if full_logits.shape != stacked.shape:
                raise AssertionError("actual timing full-main output shape differs")
            main_comparison = _compare_logits(
                stacked,
                full_logits,
                contract=comparison_contract,
            )
            maximum_main_error = max(
                maximum_main_error,
                main_comparison.maximum_absolute_error,
            )
            maximum_main_normalized_ratio = max(
                maximum_main_normalized_ratio,
                main_comparison.maximum_normalized_tolerance_ratio,
            )
            maximum_main_nominal_normalized_ratio = max(
                maximum_main_nominal_normalized_ratio,
                main_comparison.maximum_nominal_normalized_tolerance_ratio,
            )
            maximum_main_probability_total_variation = max(
                maximum_main_probability_total_variation,
                main_comparison.maximum_probability_total_variation,
            )
            main_nominal_tolerance_violations += (
                main_comparison.nominal_tolerance_violation_elements
            )
            main_full_exact_argmax += main_comparison.exact_argmax_count
            main_full_tolerance_ties += (
                main_comparison.tolerance_tie_argmax_count
            )
            full_main_positions += len(observed)

            parallel = bundle.runtime()
            logits = parallel.prefill_parallel(raw_prompt)
            parallel_comparison = _compare_logits(
                logits,
                stacked[127:128],
                contract=comparison_contract,
            )
            maximum_main_error = max(
                maximum_main_error,
                parallel_comparison.maximum_absolute_error,
            )
            maximum_main_normalized_ratio = max(
                maximum_main_normalized_ratio,
                parallel_comparison.maximum_normalized_tolerance_ratio,
            )
            maximum_main_nominal_normalized_ratio = max(
                maximum_main_nominal_normalized_ratio,
                parallel_comparison.maximum_nominal_normalized_tolerance_ratio,
            )
            maximum_main_probability_total_variation = max(
                maximum_main_probability_total_variation,
                parallel_comparison.maximum_probability_total_variation,
            )
            main_nominal_tolerance_violations += (
                parallel_comparison.nominal_tolerance_violation_elements
            )
            main_parallel_exact_argmax += parallel_comparison.exact_argmax_count
            main_parallel_tolerance_ties += (
                parallel_comparison.tolerance_tie_argmax_count
            )
            parallel_positions += 1
            for offset, value in enumerate(bytes(continuation[:-1]), start=128):
                logits = parallel.consume(value)
                parallel_comparison = _compare_logits(
                    logits,
                    stacked[offset : offset + 1],
                    contract=comparison_contract,
                )
                maximum_main_error = max(
                    maximum_main_error,
                    parallel_comparison.maximum_absolute_error,
                )
                maximum_main_normalized_ratio = max(
                    maximum_main_normalized_ratio,
                    parallel_comparison.maximum_normalized_tolerance_ratio,
                )
                maximum_main_nominal_normalized_ratio = max(
                    maximum_main_nominal_normalized_ratio,
                    parallel_comparison.maximum_nominal_normalized_tolerance_ratio,
                )
                maximum_main_probability_total_variation = max(
                    maximum_main_probability_total_variation,
                    parallel_comparison.maximum_probability_total_variation,
                )
                main_nominal_tolerance_violations += (
                    parallel_comparison.nominal_tolerance_violation_elements
                )
                main_parallel_exact_argmax += (
                    parallel_comparison.exact_argmax_count
                )
                main_parallel_tolerance_ties += (
                    parallel_comparison.tolerance_tie_argmax_count
                )
                parallel_positions += 1
            _assert_cache(sequential, len(observed))
            _assert_cache(parallel, len(observed))
            if sequential.diagnostics != parallel.diagnostics:
                raise AssertionError("actual timing parallel cache trace differs")
            del sequential, parallel, sequential_logits, stacked, full_logits
    _synchronize(bundle.device)
    expected_cases = len(prompts)
    if (
        full_main_positions != expected_cases * 255
        or parallel_positions != expected_cases * 128
        or router_positions
        != (expected_cases * 255 if bundle.requires_entropy_router else 0)
        or main_full_exact_argmax + main_full_tolerance_ties
        != full_main_positions
        or main_parallel_exact_argmax + main_parallel_tolerance_ties
        != parallel_positions
        or router_exact_argmax + router_tolerance_ties != router_positions
    ):
        raise AssertionError("actual timing correctness count differs")
    return {
        "comparison_contract": comparison_contract,
        "boundary_trace_sha256": _boundary_trace_sha256(boundary_rows),
        "entropy_router_argmax_exact_comparisons": router_exact_argmax,
        "entropy_router_position_comparisons": router_positions,
        "entropy_router_tolerance_tie_argmax_comparisons": router_tolerance_ties,
        "main_full_causal_argmax_exact_comparisons": main_full_exact_argmax,
        "main_full_causal_position_comparisons": full_main_positions,
        "main_full_causal_tolerance_tie_argmax_comparisons": main_full_tolerance_ties,
        "main_parallel_argmax_exact_comparisons": main_parallel_exact_argmax,
        "main_parallel_position_comparisons": parallel_positions,
        "main_parallel_tolerance_tie_argmax_comparisons": main_parallel_tolerance_ties,
        "maximum_main_absolute_logit_error": maximum_main_error,
        "maximum_main_normalized_tolerance_ratio": maximum_main_normalized_ratio,
        "maximum_main_nominal_normalized_tolerance_ratio": maximum_main_nominal_normalized_ratio,
        "maximum_main_probability_total_variation": maximum_main_probability_total_variation,
        "maximum_router_absolute_entropy_error": maximum_router_entropy_error,
        "maximum_router_absolute_logit_error": maximum_router_logit_error,
        "maximum_router_nominal_entropy_tolerance_ratio": maximum_router_entropy_nominal_normalized_ratio,
        "maximum_router_nominal_logit_tolerance_ratio": maximum_router_logit_nominal_normalized_ratio,
        "maximum_router_normalized_entropy_tolerance_ratio": maximum_router_entropy_normalized_ratio,
        "maximum_router_normalized_logit_tolerance_ratio": maximum_router_logit_normalized_ratio,
        "maximum_router_probability_total_variation": maximum_router_probability_total_variation,
        "main_nominal_tolerance_violation_elements": main_nominal_tolerance_violations,
        "router_entropy_nominal_tolerance_violation_elements": router_entropy_nominal_tolerance_violations,
        "router_logit_nominal_tolerance_violation_elements": router_logit_nominal_tolerance_violations,
        "pass": True,
        "probability_total_variation_limit": ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION,
        "rtol": active_rtol,
        "atol": active_atol,
        "nominal_rtol": ACTUAL_INFERENCE_EQUIVALENCE_RTOL,
        "nominal_atol": ACTUAL_INFERENCE_EQUIVALENCE_ATOL,
    }


def _verify_free_bundle(
    bundle: LoadedActualModel,
    prompts: np.ndarray,
    output_values: np.ndarray,
    output_lengths: np.ndarray,
    utf8_masks: Mapping[StrictUtf8State, torch.Tensor],
) -> dict[str, Any]:
    if not bundle.device.startswith("mps"):
        raise ValueError("free-path correctness requires the timed MPS bundle")
    comparison_contract = "mps_backend"
    active_atol, active_rtol = _comparison_parameters(comparison_contract)
    main_positions = 0
    parallel_positions = 0
    greedy_positions = 0
    router_positions = 0
    maximum_main_error = 0.0
    maximum_main_normalized_ratio = 0.0
    maximum_main_nominal_normalized_ratio = 0.0
    maximum_main_probability_total_variation = 0.0
    main_nominal_tolerance_violations = 0
    maximum_router_logit_error = 0.0
    maximum_router_logit_normalized_ratio = 0.0
    maximum_router_logit_nominal_normalized_ratio = 0.0
    maximum_router_probability_total_variation = 0.0
    router_logit_nominal_tolerance_violations = 0
    maximum_router_entropy_error = 0.0
    maximum_router_entropy_normalized_ratio = 0.0
    maximum_router_entropy_nominal_normalized_ratio = 0.0
    router_entropy_nominal_tolerance_violations = 0
    main_full_exact_argmax = 0
    main_full_tolerance_ties = 0
    main_parallel_exact_argmax = 0
    main_parallel_tolerance_ties = 0
    router_exact_argmax = 0
    router_tolerance_ties = 0
    boundary_rows: list[tuple[int, ...]] = []
    if (
        output_values.shape
        != (len(prompts), ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES)
        or output_lengths.shape != (len(prompts),)
    ):
        raise ValueError("free-path correctness output shape differs")
    with torch.inference_mode():
        for prompt, padded, raw_length in zip(
            prompts, output_values, output_lengths, strict=True
        ):
            length = int(raw_length)
            generated = bytes(padded[:length])
            observed = bytes(prompt) + generated[:-1]
            sequential = bundle.runtime()
            sequential_logits = []
            sequential_boundaries = []
            for value in observed:
                sequential_logits.append(sequential.consume(value))
                sequential_boundaries.append(
                    _runtime_main_diagnostics(sequential).boundaries
                )
            stacked = torch.cat(sequential_logits, dim=0)
            if bundle.requires_entropy_router:
                full_router_logits, full_entropies = full_router_trace(
                    bundle, observed
                )
                cached_router = IncrementalEntropyRouter(bundle.router)
                cached_logits = []
                cached_entropies = []
                for value in observed:
                    logits, entropy = cached_router.consume(value)
                    cached_logits.append(logits)
                    cached_entropies.append(entropy)
                cached = torch.cat(cached_logits, dim=0)
                router_comparison = _compare_logits(
                    cached,
                    full_router_logits,
                    contract=comparison_contract,
                )
                maximum_router_logit_error = max(
                    maximum_router_logit_error,
                    router_comparison.maximum_absolute_error,
                )
                maximum_router_logit_normalized_ratio = max(
                    maximum_router_logit_normalized_ratio,
                    router_comparison.maximum_normalized_tolerance_ratio,
                )
                maximum_router_logit_nominal_normalized_ratio = max(
                    maximum_router_logit_nominal_normalized_ratio,
                    router_comparison.maximum_nominal_normalized_tolerance_ratio,
                )
                maximum_router_probability_total_variation = max(
                    maximum_router_probability_total_variation,
                    router_comparison.maximum_probability_total_variation,
                )
                router_logit_nominal_tolerance_violations += (
                    router_comparison.nominal_tolerance_violation_elements
                )
                router_exact_argmax += router_comparison.exact_argmax_count
                router_tolerance_ties += (
                    router_comparison.tolerance_tie_argmax_count
                )
                cached_entropy_array = np.asarray(
                    cached_entropies, dtype=np.float32
                )
                (
                    router_entropy_error,
                    router_entropy_ratio,
                    router_entropy_nominal_ratio,
                    router_entropy_nominal_violations,
                ) = _compare_entropy_values(
                    cached_entropy_array,
                    full_entropies,
                    contract=comparison_contract,
                )
                maximum_router_entropy_error = max(
                    maximum_router_entropy_error,
                    router_entropy_error,
                )
                maximum_router_entropy_normalized_ratio = max(
                    maximum_router_entropy_normalized_ratio,
                    router_entropy_ratio,
                )
                maximum_router_entropy_nominal_normalized_ratio = max(
                    maximum_router_entropy_nominal_normalized_ratio,
                    router_entropy_nominal_ratio,
                )
                router_entropy_nominal_tolerance_violations += (
                    router_entropy_nominal_violations
                )
                boundaries = full_entropy_boundaries(
                    bundle, observed, full_entropies
                )
                router_positions += len(observed)
            else:
                boundaries = structural_prefix_boundaries(
                    observed,
                    bundle.runtime_policy,
                    horizon=ACTUAL_INFERENCE_PATCH_HORIZON,
                    patch_count=bundle.patch_count,
                    fixed_stride=model_spec_for_descriptor(
                        bundle.descriptor
                    ).patch_stride,
                )
            if sequential_boundaries[-1] != boundaries:
                raise AssertionError("free-path online boundaries differ")
            for position, prefix in enumerate(sequential_boundaries):
                expected_prefix = tuple(
                    value for value in boundaries if value <= position
                )
                if prefix != expected_prefix:
                    raise AssertionError("free-path boundary prefix differs")
            boundary_rows.append(boundaries)
            full_logits = full_main_logits(bundle, observed, boundaries)
            main_comparison = _compare_logits(
                stacked,
                full_logits,
                contract=comparison_contract,
            )
            maximum_main_error = max(
                maximum_main_error,
                main_comparison.maximum_absolute_error,
            )
            maximum_main_normalized_ratio = max(
                maximum_main_normalized_ratio,
                main_comparison.maximum_normalized_tolerance_ratio,
            )
            maximum_main_nominal_normalized_ratio = max(
                maximum_main_nominal_normalized_ratio,
                main_comparison.maximum_nominal_normalized_tolerance_ratio,
            )
            maximum_main_probability_total_variation = max(
                maximum_main_probability_total_variation,
                main_comparison.maximum_probability_total_variation,
            )
            main_nominal_tolerance_violations += (
                main_comparison.nominal_tolerance_violation_elements
            )
            main_full_exact_argmax += main_comparison.exact_argmax_count
            main_full_tolerance_ties += (
                main_comparison.tolerance_tie_argmax_count
            )
            parallel = bundle.runtime()
            logits = parallel.prefill_parallel(bytes(prompt))
            parallel_comparison = _compare_logits(
                logits,
                stacked[127:128],
                contract=comparison_contract,
            )
            maximum_main_error = max(
                maximum_main_error,
                parallel_comparison.maximum_absolute_error,
            )
            maximum_main_normalized_ratio = max(
                maximum_main_normalized_ratio,
                parallel_comparison.maximum_normalized_tolerance_ratio,
            )
            maximum_main_nominal_normalized_ratio = max(
                maximum_main_nominal_normalized_ratio,
                parallel_comparison.maximum_nominal_normalized_tolerance_ratio,
            )
            maximum_main_probability_total_variation = max(
                maximum_main_probability_total_variation,
                parallel_comparison.maximum_probability_total_variation,
            )
            main_nominal_tolerance_violations += (
                parallel_comparison.nominal_tolerance_violation_elements
            )
            main_parallel_exact_argmax += parallel_comparison.exact_argmax_count
            main_parallel_tolerance_ties += (
                parallel_comparison.tolerance_tie_argmax_count
            )
            parallel_positions += 1
            state = STRICT_UTF8_INITIAL_STATE
            for output_index, value in enumerate(generated):
                allowed = utf8_masks[state]
                expected = int(
                    logits.masked_fill(~allowed, -torch.inf)
                    .argmax(dim=-1)
                    .item()
                )
                if expected != value:
                    raise AssertionError("free-path masked greedy byte differs")
                greedy_positions += 1
                state = advance_strict_utf8(state, value)
                if not state.valid:
                    raise AssertionError("free-path output is invalid UTF-8")
                eligible_stop = (
                    output_index + 1 >= ACTUAL_INFERENCE_V5_CONTINUATION_BYTES
                    and state.at_codepoint_boundary
                )
                if eligible_stop != (output_index + 1 == length):
                    raise AssertionError("free-path stopping decision differs")
                if output_index + 1 < length:
                    logits = parallel.consume(value)
                    expected_index = 128 + output_index
                    parallel_comparison = _compare_logits(
                        logits,
                        stacked[expected_index : expected_index + 1],
                        contract=comparison_contract,
                    )
                    maximum_main_error = max(
                        maximum_main_error,
                        parallel_comparison.maximum_absolute_error,
                    )
                    maximum_main_normalized_ratio = max(
                        maximum_main_normalized_ratio,
                        parallel_comparison.maximum_normalized_tolerance_ratio,
                    )
                    maximum_main_nominal_normalized_ratio = max(
                        maximum_main_nominal_normalized_ratio,
                        parallel_comparison.maximum_nominal_normalized_tolerance_ratio,
                    )
                    maximum_main_probability_total_variation = max(
                        maximum_main_probability_total_variation,
                        parallel_comparison.maximum_probability_total_variation,
                    )
                    main_nominal_tolerance_violations += (
                        parallel_comparison.nominal_tolerance_violation_elements
                    )
                    main_parallel_exact_argmax += (
                        parallel_comparison.exact_argmax_count
                    )
                    main_parallel_tolerance_ties += (
                        parallel_comparison.tolerance_tie_argmax_count
                    )
                    parallel_positions += 1
            _assert_cache(sequential, len(observed))
            _assert_cache(parallel, len(observed))
            if sequential.diagnostics != parallel.diagnostics:
                raise AssertionError("free-path parallel cache trace differs")
            main_positions += len(observed)
            del sequential, parallel, sequential_logits, stacked, full_logits
    _synchronize(bundle.device)
    if (
        main_full_exact_argmax + main_full_tolerance_ties != main_positions
        or main_parallel_exact_argmax + main_parallel_tolerance_ties
        != parallel_positions
        or router_exact_argmax + router_tolerance_ties != router_positions
    ):
        raise AssertionError("free-path argmax comparison count differs")
    return {
        "comparison_contract": comparison_contract,
        "boundary_trace_sha256": _boundary_trace_sha256(boundary_rows),
        "entropy_router_argmax_exact_comparisons": router_exact_argmax,
        "entropy_router_position_comparisons": router_positions,
        "entropy_router_tolerance_tie_argmax_comparisons": router_tolerance_ties,
        "greedy_byte_argmax_comparisons": greedy_positions,
        "main_full_causal_argmax_exact_comparisons": main_full_exact_argmax,
        "main_full_causal_position_comparisons": main_positions,
        "main_full_causal_tolerance_tie_argmax_comparisons": main_full_tolerance_ties,
        "main_parallel_argmax_exact_comparisons": main_parallel_exact_argmax,
        "main_parallel_position_comparisons": parallel_positions,
        "main_parallel_tolerance_tie_argmax_comparisons": main_parallel_tolerance_ties,
        "maximum_main_absolute_logit_error": maximum_main_error,
        "maximum_main_normalized_tolerance_ratio": maximum_main_normalized_ratio,
        "maximum_main_nominal_normalized_tolerance_ratio": maximum_main_nominal_normalized_ratio,
        "maximum_main_probability_total_variation": maximum_main_probability_total_variation,
        "maximum_router_absolute_entropy_error": maximum_router_entropy_error,
        "maximum_router_absolute_logit_error": maximum_router_logit_error,
        "maximum_router_nominal_entropy_tolerance_ratio": maximum_router_entropy_nominal_normalized_ratio,
        "maximum_router_nominal_logit_tolerance_ratio": maximum_router_logit_nominal_normalized_ratio,
        "maximum_router_normalized_entropy_tolerance_ratio": maximum_router_entropy_normalized_ratio,
        "maximum_router_normalized_logit_tolerance_ratio": maximum_router_logit_normalized_ratio,
        "maximum_router_probability_total_variation": maximum_router_probability_total_variation,
        "main_nominal_tolerance_violation_elements": main_nominal_tolerance_violations,
        "router_entropy_nominal_tolerance_violation_elements": router_entropy_nominal_tolerance_violations,
        "router_logit_nominal_tolerance_violation_elements": router_logit_nominal_tolerance_violations,
        "pass": True,
        "probability_total_variation_limit": ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION,
        "rtol": active_rtol,
        "atol": active_atol,
        "nominal_rtol": ACTUAL_INFERENCE_EQUIVALENCE_RTOL,
        "nominal_atol": ACTUAL_INFERENCE_EQUIVALENCE_ATOL,
    }


def _utf8_mask_cache(
    device: str = "mps",
) -> dict[StrictUtf8State, torch.Tensor]:
    masks = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device=device)
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        masks[state] = mask
    _synchronize(device)
    return masks


def _run_trial(
    bundle: LoadedActualModel,
    prompt: bytes,
    continuation: bytes,
    mode: str,
    utf8_masks: Mapping[StrictUtf8State, torch.Tensor],
) -> TimedTrial:
    if mode not in ACTUAL_INFERENCE_V5_MODES:
        raise ValueError("actual timing mode differs")
    if (
        len(prompt) != ACTUAL_INFERENCE_V5_PROMPT_BYTES
        or len(continuation) != ACTUAL_INFERENCE_V5_CONTINUATION_BYTES
    ):
        raise ValueError("actual timing trial horizon differs")
    if mode == "controlled_replay" and not strict_utf8_state(
        continuation
    ).at_codepoint_boundary:
        raise ValueError("controlled replay does not end at a scalar boundary")
    _synchronize(bundle.device)
    started = time.perf_counter_ns()
    generated = bytearray()
    synchronizations_inside_timing = 0
    device_to_host_readbacks_inside_timing = 0
    argmax_calls = 0
    utf8_mask_calls = 0
    utf8_dfa_advances = 0
    stop_checks = 0
    with torch.inference_mode():
        runtime = bundle.runtime()
        logits = runtime.prefill_parallel(prompt)
        device_to_host_readbacks_inside_timing += int(
            bundle.requires_entropy_router
        )
        _synchronize(bundle.device)
        synchronizations_inside_timing += 1
        prefilled = time.perf_counter_ns()
        if mode == "controlled_replay":
            for value in continuation[:-1]:
                logits = runtime.consume(value)
                device_to_host_readbacks_inside_timing += int(
                    bundle.requires_entropy_router
                )
        else:
            state = STRICT_UTF8_INITIAL_STATE
            while True:
                allowed = utf8_masks[state]
                utf8_mask_calls += 1
                value = int(
                    logits.masked_fill(~allowed, -torch.inf)
                    .argmax(dim=-1)
                    .item()
                )
                device_to_host_readbacks_inside_timing += 1
                argmax_calls += 1
                generated.append(value)
                state = advance_strict_utf8(state, value)
                utf8_dfa_advances += 1
                if not state.valid:
                    raise AssertionError("strict UTF-8 mask admitted an invalid byte")
                stop_checks += 1
                if (
                    len(generated) >= ACTUAL_INFERENCE_V5_CONTINUATION_BYTES
                    and state.at_codepoint_boundary
                ):
                    break
                if len(generated) >= ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES:
                    raise AssertionError("free output exceeded UTF-8 overshoot bound")
                logits = runtime.consume(value)
                device_to_host_readbacks_inside_timing += int(
                    bundle.requires_entropy_router
                )
        _synchronize(bundle.device)
        synchronizations_inside_timing += 1
        finished = time.perf_counter_ns()
    output_bytes = (
        ACTUAL_INFERENCE_V5_CONTINUATION_BYTES
        if mode == "controlled_replay"
        else len(generated)
    )
    observed = ACTUAL_INFERENCE_V5_PROMPT_BYTES + output_bytes - 1
    _assert_cache(runtime, observed)
    internal = runtime.runtime_counters
    counters = {
        "parallel_prefill_calls": internal.parallel_prefill_calls,
        "main_consume_calls": internal.main_consume_calls,
        "selector_observed_bytes": internal.selector_observed_bytes,
        "router_forward_calls": internal.router_forward_calls,
        "router_scored_bytes": internal.router_scored_bytes,
        "argmax_calls": argmax_calls,
        "utf8_mask_calls": utf8_mask_calls,
        "utf8_dfa_advances": utf8_dfa_advances,
        "stop_checks": stop_checks,
        "device_to_host_readbacks_inside_timing": (
            device_to_host_readbacks_inside_timing
        ),
        "explicit_device_synchronizations_inside_timing": (
            synchronizations_inside_timing
        ),
    }
    one = {name: np.asarray([[value]], dtype=np.int64) for name, value in counters.items()}
    validate_runtime_counter_arrays(
        one,
        requires_entropy_router=bundle.requires_entropy_router,
        mode=mode,
        emitted_output_bytes=np.asarray([[output_bytes]], dtype=np.int64),
    )
    main = _runtime_main_diagnostics(runtime)
    result = TimedTrial(
        ttft_ms=(prefilled - started) / 1_000_000,
        decode_ms=(finished - prefilled) / 1_000_000,
        end_to_end_ms=(finished - started) / 1_000_000,
        emitted_output_bytes=output_bytes,
        emitted_global_patches=main.emitted_data_patches,
        runtime_observed_bytes=main.observed_bytes,
        counters=counters,
        generated=bytes(generated) if mode != "controlled_replay" else None,
    )
    del runtime, logits
    return result


def _empty_timing_arrays() -> dict[str, np.ndarray]:
    shape = (len(FINAL_SEEDS), ACTUAL_INFERENCE_V5_MEASURED_CASES, ACTUAL_INFERENCE_V5_REPETITIONS)
    arrays = {}
    for mode in ACTUAL_INFERENCE_V5_MODES:
        for role in ACTUAL_INFERENCE_V5_ROLES:
            for component in ACTUAL_INFERENCE_V5_COMPONENTS:
                arrays[f"{mode}__{component}__{role}"] = np.zeros(shape, dtype=np.float64)
            arrays[f"{mode}__emitted_output_bytes__{role}"] = np.zeros(shape, dtype=np.int64)
            arrays[f"{mode}__global_patches__{role}"] = np.zeros(shape, dtype=np.int64)
            arrays[f"{mode}__runtime_observed_bytes__{role}"] = np.zeros(shape, dtype=np.int64)
            for counter in RUNTIME_COUNTER_NAMES:
                arrays[f"{mode}__counter_{counter}__{role}"] = np.zeros(shape, dtype=np.int64)
    return arrays


def _record_trial(
    arrays: dict[str, np.ndarray],
    *,
    seed_index: int,
    mode: str,
    role: str,
    prompt_index: int,
    repetition: int,
    result: TimedTrial,
) -> None:
    index = (seed_index, prompt_index, repetition)
    for component in ACTUAL_INFERENCE_V5_COMPONENTS:
        arrays[f"{mode}__{component}__{role}"][index] = getattr(result, component)
    arrays[f"{mode}__emitted_output_bytes__{role}"][index] = result.emitted_output_bytes
    arrays[f"{mode}__global_patches__{role}"][index] = result.emitted_global_patches
    arrays[f"{mode}__runtime_observed_bytes__{role}"][index] = result.runtime_observed_bytes
    for counter, value in result.counters.items():
        arrays[f"{mode}__counter_{counter}__{role}"][index] = value


def _thermal_checkpoint(samples: list[dict[str, Any]]) -> None:
    state = _session_state()
    inventory = _process_inventory()
    sample = {
        "environment": state,
        "inventory": inventory,
        "eligible": (
            timing_environment_eligible(state)
            and inventory["ps_returncode"] == 0
            and inventory["ps_stdout_nonempty"]
            and inventory["ps_parse_pass"]
            and not inventory["conflicting_neural_processes"]
        ),
    }
    samples.append(sample)
    if not sample["eligible"]:
        raise RuntimeError("actual timing environment became ineligible")


def _run_session(
    *,
    session_id: str,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    plan_artifact_sha256: str,
    session_git_commit: str,
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    session_index = ACTUAL_INFERENCE_V5_SESSIONS.index(session_id)
    schedule = session_schedule(session_index)
    warmup_schedule = session_schedule(
        session_index,
        prompt_count=ACTUAL_INFERENCE_V5_WARMUP_CASES,
        repetitions=1,
    )
    recorded = plan["session_schedules"][session_index]
    if (
        recorded["session_id"] != session_id
        or recorded["candidate_first_sha256"]
        != array_sha256(schedule["candidate_first"])
        or recorded["warmup_candidate_first_sha256"]
        != array_sha256(warmup_schedule["candidate_first"])
        or recorded["seed_execution_order"] != schedule["seed_order"].tolist()
    ):
        raise ValueError("actual timing session schedule differs")
    timing_arrays = _empty_timing_arrays()
    output_shape = (
        len(FINAL_SEEDS),
        len(ACTUAL_INFERENCE_V5_ROLES),
        ACTUAL_INFERENCE_V5_MEASURED_CASES,
        ACTUAL_INFERENCE_V5_REPETITIONS,
    )
    output_values = np.zeros(
        (*output_shape, ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES),
        dtype=np.uint8,
    )
    output_lengths = np.zeros(output_shape, dtype=np.int64)
    masks = _utf8_mask_cache()
    thermal_samples: list[dict[str, Any]] = []
    _thermal_checkpoint(thermal_samples)
    correctness: dict[str, dict[str, Any]] = {}
    cpu_semantic_correctness: dict[str, dict[str, Any]] = {}
    free_path_correctness: dict[str, dict[str, Any]] = {}
    model_provenance: dict[str, dict[str, Any]] = {}
    for executed_seed in schedule["seed_order"].tolist():
        seed = int(executed_seed)
        seed_index = FINAL_SEEDS.index(seed)
        cpu_semantic_correctness[str(seed)] = {}
        for role in ACTUAL_INFERENCE_V5_ROLES:
            cpu_bundle = load_actual_model(
                role=role,
                identity=_model_for_role(authorization, plan, role),
                seed=seed,
                device="cpu",
            )
            try:
                cpu_semantic_correctness[str(seed)][role] = _verify_bundle(
                    cpu_bundle,
                    prompts,
                    continuations,
                    comparison_contract="cpu_semantic",
                )
            finally:
                release_actual_model(cpu_bundle)
        bundles = {
            role: load_actual_model(
                role=role,
                identity=_model_for_role(authorization, plan, role),
                seed=seed,
                device="mps",
            )
            for role in ACTUAL_INFERENCE_V5_ROLES
        }
        model_provenance[str(seed)] = {
            role: {
                "checkpoint_artifact_sha256": bundle.evidence["checkpoint"]["artifact_sha256"],
                "checkpoint_state_sha256": bundle.evidence["checkpoint"]["state_sha256"],
                "model_identity_sha256": bundle.identity["identity_sha256"],
                "requires_entropy_router": bundle.requires_entropy_router,
                "router_checkpoint_state_sha256": (
                    bundle.evidence["auxiliary"].get("router_checkpoint_state_sha256")
                    if bundle.requires_entropy_router
                    else None
                ),
            }
            for role, bundle in bundles.items()
        }
        correctness[str(seed)] = {
            role: _verify_bundle(
                bundle,
                prompts,
                continuations,
                comparison_contract="mps_backend",
            )
            for role, bundle in bundles.items()
        }
        if any(
            cpu_semantic_correctness[str(seed)][role][
                "boundary_trace_sha256"
            ]
            != correctness[str(seed)][role]["boundary_trace_sha256"]
            for role in ACTUAL_INFERENCE_V5_ROLES
        ):
            raise AssertionError("CPU/MPS semantic boundary trace differs")
        _thermal_checkpoint(thermal_samples)
        for mode_index, mode in enumerate(ACTUAL_INFERENCE_V5_MODES):
            for prompt_index in range(ACTUAL_INFERENCE_V5_WARMUP_CASES):
                roles = (
                    ACTUAL_INFERENCE_V5_ROLES
                    if warmup_schedule["candidate_first"][seed_index, mode_index, prompt_index, 0]
                    else tuple(reversed(ACTUAL_INFERENCE_V5_ROLES))
                )
                for role in roles:
                    _run_trial(
                        bundles[role],
                        bytes(prompts[prompt_index]),
                        bytes(continuations[prompt_index]),
                        mode,
                        masks,
                    )
                    gc.collect()
        measured_prompts = prompts[ACTUAL_INFERENCE_V5_WARMUP_CASES :]
        measured_continuations = continuations[ACTUAL_INFERENCE_V5_WARMUP_CASES :]
        for mode_index, mode in enumerate(ACTUAL_INFERENCE_V5_MODES):
            for prompt_index, (prompt, continuation) in enumerate(
                zip(measured_prompts, measured_continuations, strict=True)
            ):
                for repetition in range(ACTUAL_INFERENCE_V5_REPETITIONS):
                    roles = (
                        ACTUAL_INFERENCE_V5_ROLES
                        if schedule["candidate_first"][seed_index, mode_index, prompt_index, repetition]
                        else tuple(reversed(ACTUAL_INFERENCE_V5_ROLES))
                    )
                    for role in roles:
                        result = _run_trial(
                            bundles[role],
                            bytes(prompt),
                            bytes(continuation),
                            mode,
                            masks,
                        )
                        _record_trial(
                            timing_arrays,
                            seed_index=seed_index,
                            mode=mode,
                            role=role,
                            prompt_index=prompt_index,
                            repetition=repetition,
                            result=result,
                        )
                        if mode == "free_running_utf8_greedy":
                            if result.generated is None:
                                raise AssertionError("free trial did not preserve bytes")
                            role_index = ACTUAL_INFERENCE_V5_ROLES.index(role)
                            length = len(result.generated)
                            output_values[
                                seed_index,
                                role_index,
                                prompt_index,
                                repetition,
                                :length,
                            ] = np.frombuffer(result.generated, dtype=np.uint8)
                            output_lengths[
                                seed_index,
                                role_index,
                                prompt_index,
                                repetition,
                            ] = length
                        gc.collect()
                if (prompt_index + 1) % 8 == 0:
                    _thermal_checkpoint(thermal_samples)
        for role, bundle in bundles.items():
            role_index = ACTUAL_INFERENCE_V5_ROLES.index(role)
            lengths = output_lengths[seed_index, role_index]
            if not np.all(lengths == lengths[:, :1]):
                raise AssertionError(
                    "free output length changed across timing repetitions"
                )
            for prompt_index in range(ACTUAL_INFERENCE_V5_MEASURED_CASES):
                first = output_values[
                    seed_index, role_index, prompt_index, 0
                ]
                length = int(lengths[prompt_index, 0])
                if any(
                    not np.array_equal(
                        first[:length],
                        output_values[
                            seed_index,
                            role_index,
                            prompt_index,
                            repetition,
                            :length,
                        ],
                    )
                    for repetition in range(1, ACTUAL_INFERENCE_V5_REPETITIONS)
                ):
                    raise AssertionError(
                        "free output changed across timing repetitions"
                    )
            free_path_correctness.setdefault(str(seed), {})[role] = (
                _verify_free_bundle(
                    bundle,
                    measured_prompts,
                    output_values[seed_index, role_index, :, 0],
                    lengths[:, 0],
                    masks,
                )
            )
            for mode in ACTUAL_INFERENCE_V5_MODES:
                emitted = timing_arrays[f"{mode}__emitted_output_bytes__{role}"][seed_index]
                counters = {
                    name: timing_arrays[f"{mode}__counter_{name}__{role}"][seed_index]
                    for name in RUNTIME_COUNTER_NAMES
                }
                validate_runtime_counter_arrays(
                    counters,
                    requires_entropy_router=bundle.requires_entropy_router,
                    mode=mode,
                    emitted_output_bytes=emitted,
                )
            release_actual_model(bundle)
        _thermal_checkpoint(thermal_samples)
    output_arrays = {
        "free_output_bytes": output_values,
        "free_output_lengths": output_lengths,
    }
    report = {
        "complete": True,
        "correctness": correctness,
        "cpu_semantic_correctness": cpu_semantic_correctness,
        "environment": _environment(),
        "free_path_correctness": free_path_correctness,
        "kind": "phase3_inference_actual_session_v5r3",
        "model_provenance": model_provenance,
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "process": {
            "pid": os.getpid(),
            "process_start": _command_snapshot(
                ["ps", "-o", "lstart=", "-p", str(os.getpid())]
            ),
        },
        "protocol_version": 5,
        "protocol_revision": ACTUAL_INFERENCE_V5_PROTOCOL_REVISION,
        "schema_version": 6,
        "session_git_commit": session_git_commit,
        "session_id": session_id,
        "session_schedule_sha256": canonical_sha256(recorded),
        "thermal_samples": thermal_samples,
    }
    report["process"]["start_token_sha256"] = hashlib.sha256(
        json.dumps(report["process"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return timing_arrays, output_arrays, report


def _start_session_sentinel(
    *, session_id: str, plan: Mapping[str, Any], plan_artifact_sha256: str
) -> None:
    if ACTIVE_SENTINEL.exists():
        raise ValueError("unfinished actual timing session requires forensic review")
    payload = {
        "git_commit": _git_commit(),
        "pid": os.getpid(),
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan["plan_sha256"],
        "session_id": session_id,
    }
    publish_no_clobber(ACTIVE_SENTINEL, _json_bytes(payload))


def run() -> int:
    with _exclusive_process_lock():
        plan, authorization, _, _, plan_artifact_sha256 = _load_context()
        session_git_commit = _git_commit()
        _assert_no_symlink_namespace(RUN_ROOT)
        _assert_no_symlink_namespace(ARTIFACT_ROOT)
        session_id = _next_session(plan)
        if session_id is None:
            print("all five actual-inference v5 sessions are complete", flush=True)
            return 0
        prompts, continuations = _load_cases(plan)
        for path in _session_paths(session_id).values():
            _assert_no_symlink_namespace(path.parent)
        inventory = _process_inventory()
        state = _session_state()
        if (
            not timing_environment_eligible(state)
            or inventory["ps_returncode"] != 0
            or not inventory["ps_stdout_nonempty"]
            or not inventory["ps_parse_pass"]
            or inventory["conflicting_neural_processes"]
        ):
            raise RuntimeError("actual timing session starts in an ineligible state")
        _start_session_sentinel(
            session_id=session_id,
            plan=plan,
            plan_artifact_sha256=plan_artifact_sha256,
        )
        timings, outputs, report = _run_session(
            session_id=session_id,
            plan=plan,
            authorization=authorization,
            plan_artifact_sha256=plan_artifact_sha256,
            session_git_commit=session_git_commit,
            prompts=prompts,
            continuations=continuations,
        )
        timing_bytes = _npz_bytes(timings)
        output_bytes = _npz_bytes(outputs)
        paths = _session_paths(session_id)
        report["timing_artifact_sha256"] = hashlib.sha256(timing_bytes).hexdigest()
        report["timing_array_sha256"] = {
            key: array_sha256(value) for key, value in timings.items()
        }
        report["output_artifact_sha256"] = hashlib.sha256(output_bytes).hexdigest()
        report["output_array_sha256"] = {
            key: array_sha256(value) for key, value in outputs.items()
        }
        if _git_commit() != session_git_commit or _git_status().strip():
            raise ValueError("repository changed during actual timing session")
        publish_no_clobber(paths["timings"], timing_bytes)
        publish_no_clobber(paths["outputs"], output_bytes)
        publish_no_clobber(paths["report"], _json_bytes(report))
        ACTIVE_SENTINEL.unlink()
        print(
            f"completed actual-inference v5 {session_id}; commit its receipt before the next session; no metrics opened",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(run())
