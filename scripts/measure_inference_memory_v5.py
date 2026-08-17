#!/usr/bin/env python3
"""Measure one fixed role/seed memory unit in a fresh isolated process."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.inference_actual_runtime_v5 import (
    LoadedActualModel,
    load_actual_model,
    release_actual_model,
)
from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_ARTIFACT_ROOT,
    ACTUAL_INFERENCE_V5_CASE_PATH,
    ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
    ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
    ACTUAL_INFERENCE_V5_MEMORY_ROOT,
    ACTUAL_INFERENCE_V5_MEMORY_REPETITIONS,
    ACTUAL_INFERENCE_V5_MEASURED_CASES,
    ACTUAL_INFERENCE_V5_PLAN_PATH,
    ACTUAL_INFERENCE_V5_PROTOCOL_REVISION,
    ACTUAL_INFERENCE_V5_PROMPT_BYTES,
    ACTUAL_INFERENCE_V5_REPETITIONS,
    ACTUAL_INFERENCE_V5_ROLES,
    ACTUAL_INFERENCE_V5_SESSION_RECEIPT_ROOT,
    ACTUAL_INFERENCE_V5_SESSIONS,
    ACTUAL_INFERENCE_V5_WARMUP_CASES,
    MPS_ENTRYPOINT_MARKERS,
    canonical_sha256,
    assert_workspace_path_no_symlinks,
    current_runtime_environment_contract,
    validate_actual_inference_plan_v5,
    validate_isolated_memory_receipt,
)
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_QUALITY_LOCK_PATH,
    FINAL_SEEDS,
    SELECTION_LOCK_PATH,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    advance_strict_utf8,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
)


PLAN_PATH = Path(ACTUAL_INFERENCE_V5_PLAN_PATH)
CASE_PATH = Path(ACTUAL_INFERENCE_V5_CASE_PATH)
AUTHORIZATION_PATH = Path(FINAL_AUTHORIZATION_PATH)
QUALITY_LOCK_PATH = Path(FINAL_QUALITY_LOCK_PATH)
SELECTION_PATH = Path(SELECTION_LOCK_PATH)
ARTIFACT_ROOT = Path(ACTUAL_INFERENCE_V5_ARTIFACT_ROOT)
MEMORY_ROOT = Path(ACTUAL_INFERENCE_V5_MEMORY_ROOT)
SESSION_RECEIPT_ROOT = Path(ACTUAL_INFERENCE_V5_SESSION_RECEIPT_ROOT)
PROCESS_LOCK_PATH = ARTIFACT_ROOT / ".process.lock"
MACHINE_LOCK_PATH = Path("/tmp/jamoflow-publication-mps.lock")
ACTIVE_SENTINEL = ARTIFACT_ROOT / ".memory-active"


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
        raise ValueError("isolated memory requires a clean repository root")
    if platform.system() != "Darwin" or not torch.backends.mps.is_available():
        raise ValueError("isolated memory v5 requires the sealed Apple MPS platform")
    return _git_commit()


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
        raise ValueError(f"isolated memory input is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit_value,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(blob.stdout).hexdigest(),
    }


def _tracked_head_sha256(path: Path) -> str:
    return _tracked_head_identity(path)["sha256"]


def _tracked_history_exists(path: Path) -> bool:
    result = subprocess.run(
        ["git", "log", "--all", "-1", "--format=%H", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"isolated memory receipt history check failed: {path}")
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
        raise ValueError("isolated memory receipt history is malformed") from error
    if result.returncode != 0 or count < 0:
        raise ValueError("isolated memory receipt history check failed")
    return count


def _require_no_conflicting_neural_processes() -> None:
    snapshot = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    output = snapshot.stdout
    parsed = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            pid, parent = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        parsed.append((pid, parent, parts[2], line.strip()))
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
    if (
        snapshot.returncode != 0
        or not output.strip()
        or not parsed
        or os.getpid() not in parents
        or conflicts
    ):
        raise RuntimeError("isolated memory found another neural/MPS process")


@contextmanager
def _exclusive_process_lock():
    assert_workspace_path_no_symlinks(ARTIFACT_ROOT)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    assert_workspace_path_no_symlinks(ARTIFACT_ROOT)
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
        raise ValueError("isolated memory role has no unique model")
    return matches[0]


def _load_context() -> tuple[dict[str, Any], dict[str, Any], str]:
    commit = _require_clean_root()
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
    runtime_environment = current_runtime_environment_contract()
    if runtime_environment != plan["runtime_environment_contract"]:
        raise ValueError("isolated memory runtime environment differs")
    for path, expected in plan["implementation_sha256"].items():
        if _tracked_head_sha256(Path(path)) != expected:
            raise ValueError(f"isolated memory implementation differs: {path}")
    if hash_file(CASE_PATH) != plan["case_context"]["artifact_sha256"]:
        raise ValueError("isolated memory cases differ")
    return plan, authorization, commit


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode != 0:
        raise ValueError(f"isolated memory Git order differs: {label}")


def _require_timing_campaign_complete(
    *,
    plan: Mapping[str, Any],
    current_commit: str,
) -> None:
    previous_commit = _tracked_head_identity(PLAN_PATH)["git_commit"]
    seen_commits = {previous_commit}
    for session_id in ACTUAL_INFERENCE_V5_SESSIONS:
        report_path = SESSION_RECEIPT_ROOT / f"{session_id}.json"
        timing_path = ARTIFACT_ROOT / session_id / "timings.npz"
        output_path = ARTIFACT_ROOT / session_id / "free-outputs.npz"
        if any(
            not path.is_file() or path.is_symlink()
            for path in (report_path, timing_path, output_path)
        ):
            raise ValueError(
                "all five timing sessions must be committed before memory"
            )
        identity = _tracked_head_identity(report_path)
        if (
            _tracked_touch_count(report_path) != 1
            or identity["git_commit"] in seen_commits
        ):
            raise ValueError("timing session receipts were not committed separately")
        report = _read_json(report_path)
        if (
            report.get("complete") is not True
            or report.get("kind") != "phase3_inference_actual_session_v5r3"
            or report.get("schema_version") != 6
            or report.get("protocol_version") != 5
            or report.get("protocol_revision")
            != ACTUAL_INFERENCE_V5_PROTOCOL_REVISION
            or report.get("session_id") != session_id
            or report.get("plan_sha256") != plan["plan_sha256"]
            or report.get("timing_artifact_sha256") != hash_file(timing_path)
            or report.get("output_artifact_sha256") != hash_file(output_path)
        ):
            raise ValueError(f"timing session is incomplete before memory: {session_id}")
        _require_ancestor(
            previous_commit,
            identity["git_commit"],
            f"previous evidence -> {session_id}",
        )
        previous_commit = identity["git_commit"]
        seen_commits.add(previous_commit)
    _require_ancestor(
        previous_commit,
        current_commit,
        "last timing receipt -> memory measurement",
    )


def _receipt_path(role: str, seed: int) -> Path:
    return MEMORY_ROOT / role / f"seed-{seed}.json"


def _unit_order() -> tuple[tuple[str, int], ...]:
    return tuple(
        (role, seed)
        for role in ACTUAL_INFERENCE_V5_ROLES
        for seed in FINAL_SEEDS
    )


def _next_unit(
    *, plan: Mapping[str, Any], authorization: Mapping[str, Any]
) -> tuple[str, int] | None:
    missing: tuple[str, int] | None = None
    for role, seed in _unit_order():
        path = _receipt_path(role, seed)
        if not path.exists():
            if _tracked_history_exists(path):
                raise ValueError("deleted isolated memory receipt forbids rerun")
            if missing is None:
                missing = (role, seed)
            continue
        if missing is not None:
            raise ValueError("isolated memory receipts are not a complete prefix")
        _tracked_head_sha256(path)
        if _tracked_touch_count(path) != 1:
            raise ValueError("isolated memory receipt was rewritten")
        identity = _model_for_role(authorization, plan, role)
        receipt = _read_json(path)
        validate_isolated_memory_receipt(
            receipt,
            role=role,
            model_identity_sha256=identity["identity_sha256"],
            seed=seed,
            plan_sha256=plan["plan_sha256"],
            expected_checkpoint_state_sha256=identity["seeds"][str(seed)][
                "checkpoint"
            ]["state_sha256"],
            expected_router_checkpoint_state_sha256=(
                identity["seeds"][str(seed)]["auxiliary"].get(
                    "router_checkpoint_state_sha256"
                )
                if identity["descriptor"]["requires_entropy_router"]
                else None
            ),
            expected_parameter_bytes=plan["timing_pair"]["roles"][role][
                "parameter_bytes_float32"
            ],
        )
    return missing


def _mps_snapshot() -> tuple[int, int]:
    torch.mps.synchronize()
    return (
        int(torch.mps.current_allocated_memory()),
        int(torch.mps.driver_allocated_memory()),
    )


def _rss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _parameter_bytes(bundle: LoadedActualModel) -> int:
    models = [bundle.model]
    if bundle.router is not None:
        models.append(bundle.router)
    return sum(
        int(value.numel() * value.element_size())
        for model in models
        for value in model.parameters()
    )


def _mask_cache() -> dict[Any, torch.Tensor]:
    output = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device="mps")
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        output[state] = mask
    torch.mps.synchronize()
    return output


def _memory_workload(
    bundle: LoadedActualModel,
    prompts: tuple[bytes, ...],
    masks: Mapping[Any, torch.Tensor],
) -> None:
    for prompt in prompts:
        for _ in range(ACTUAL_INFERENCE_V5_MEMORY_REPETITIONS):
            runtime = bundle.runtime()
            with torch.inference_mode():
                logits = runtime.prefill_parallel(prompt)
                state = STRICT_UTF8_INITIAL_STATE
                generated = 0
                while True:
                    value = int(
                        logits.masked_fill(~masks[state], -torch.inf)
                        .argmax(dim=-1)
                        .item()
                    )
                    generated += 1
                    state = advance_strict_utf8(state, value)
                    if (
                        generated >= ACTUAL_INFERENCE_V5_CONTINUATION_BYTES
                        and state.at_codepoint_boundary
                    ):
                        break
                    if generated >= ACTUAL_INFERENCE_V5_CONTINUATION_BYTES + 3:
                        raise AssertionError("memory workload exceeded UTF-8 bound")
                    logits = runtime.consume(value)
            del runtime, logits
    torch.mps.synchronize()


def _measure_unit(
    *,
    role: str,
    seed: int,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    commit: str,
) -> dict[str, Any]:
    with np.load(CASE_PATH, allow_pickle=False) as archive:
        prompt_array = archive["prompts"][
            ACTUAL_INFERENCE_V5_WARMUP_CASES :
        ]
    if prompt_array.shape != (
        ACTUAL_INFERENCE_V5_MEASURED_CASES,
        ACTUAL_INFERENCE_V5_PROMPT_BYTES,
    ):
        raise ValueError("isolated memory prompts differ")
    prompts = tuple(bytes(row) for row in prompt_array)
    torch.mps.synchronize()
    baseline_rss = _rss()
    baseline_current, baseline_driver = _mps_snapshot()
    identity = _model_for_role(authorization, plan, role)
    bundle = load_actual_model(
        role=role,
        identity=identity,
        seed=seed,
        device="mps",
    )
    torch.mps.synchronize()
    load_rss = _rss()
    load_current, load_driver = _mps_snapshot()
    parameter_bytes = _parameter_bytes(bundle)
    masks = _mask_cache()
    _memory_workload(bundle, prompts, masks)
    inference_rss = _rss()
    inference_current, inference_driver = _mps_snapshot()
    checkpoint_state = identity["seeds"][str(seed)]["checkpoint"]["state_sha256"]
    auxiliary = identity["seeds"][str(seed)]["auxiliary"]
    del masks
    release_actual_model(bundle)
    torch.mps.empty_cache()
    release_current, release_driver = _mps_snapshot()
    payload = {
        "backend": "isolated-process-ru_maxrss-macos",
        "checkpoint_state_sha256": checkpoint_state,
        "measurement_git_commit": commit,
        "model_identity_sha256": identity["identity_sha256"],
        "mps_snapshots": {
            "after_inference_current_bytes": inference_current,
            "after_inference_driver_bytes": inference_driver,
            "after_load_current_bytes": load_current,
            "after_load_driver_bytes": load_driver,
            "after_release_current_bytes": release_current,
            "after_release_driver_bytes": release_driver,
            "baseline_current_bytes": baseline_current,
            "baseline_driver_bytes": baseline_driver,
        },
        "parameter_bytes": parameter_bytes,
        "plan_sha256": plan["plan_sha256"],
        "process_rss": {
            "after_inference_bytes": inference_rss,
            "after_model_load_bytes": load_rss,
            "baseline_bytes": baseline_rss,
            "high_water_bytes": inference_rss,
            "unit": "bytes_on_macos",
        },
        "resettable_peak_supported": False,
        "role": role,
        "router_checkpoint_state_sha256": (
            auxiliary["router_checkpoint_state_sha256"]
            if auxiliary["kind"] == "entropy_router"
            else None
        ),
        "seed": seed,
        "workload": {
            "case_artifact_sha256": plan["case_context"]["artifact_sha256"],
            "continuation_bytes": ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
            "measured_cases": ACTUAL_INFERENCE_V5_MEASURED_CASES,
            "mode": "free_running_utf8_greedy",
            "prompt_bytes": ACTUAL_INFERENCE_V5_PROMPT_BYTES,
            "prompt_array_sha256": plan["case_context"]["prompt_array_sha256"],
            "repetitions": ACTUAL_INFERENCE_V5_MEMORY_REPETITIONS,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    validate_isolated_memory_receipt(
        payload,
        role=role,
        model_identity_sha256=identity["identity_sha256"],
        seed=seed,
        plan_sha256=plan["plan_sha256"],
        expected_checkpoint_state_sha256=checkpoint_state,
        expected_router_checkpoint_state_sha256=(
            auxiliary.get("router_checkpoint_state_sha256")
            if auxiliary["kind"] == "entropy_router"
            else None
        ),
        expected_parameter_bytes=plan["timing_pair"]["roles"][role][
            "parameter_bytes_float32"
        ],
    )
    return payload


def run() -> int:
    with _exclusive_process_lock():
        plan, authorization, commit = _load_context()
        _require_no_conflicting_neural_processes()
        _require_timing_campaign_complete(
            plan=plan,
            current_commit=commit,
        )
        unit = _next_unit(plan=plan, authorization=authorization)
        if unit is None:
            print("all ten isolated memory units are complete", flush=True)
            return 0
        if ACTIVE_SENTINEL.exists():
            raise ValueError("unfinished isolated memory unit requires review")
        role, seed = unit
        assert_workspace_path_no_symlinks(_receipt_path(role, seed).parent)
        publish_no_clobber(
            ACTIVE_SENTINEL,
            _json_bytes(
                {
                    "commit": commit,
                    "pid": os.getpid(),
                    "plan_sha256": plan["plan_sha256"],
                    "role": role,
                    "seed": seed,
                }
            ),
        )
        receipt = _measure_unit(
            role=role,
            seed=seed,
            plan=plan,
            authorization=authorization,
            commit=commit,
        )
        if _git_commit() != commit or _git_status().strip():
            raise ValueError("repository changed during isolated memory measurement")
        publish_no_clobber(_receipt_path(role, seed), _json_bytes(receipt))
        ACTIVE_SENTINEL.unlink()
        print(
            f"completed isolated memory {role}/{seed}; commit its receipt before the next unit; no metric opened",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(run())
