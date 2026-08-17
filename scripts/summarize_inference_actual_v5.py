#!/usr/bin/env python3
"""Validate all v5 evidence and publish the matched-quality speed summary."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.hplt3 import hash_file
from jamoflow.hplt3_final_test import publish_no_clobber
from jamoflow.inference_actual_runtime_v5 import (
    ACTUAL_INFERENCE_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_EQUIVALENCE_RTOL,
    ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION,
    ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL,
    load_actual_model,
    release_actual_model,
)
from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_ARTIFACT_ROOT,
    ACTUAL_INFERENCE_V5_BOOTSTRAP_REPETITIONS,
    ACTUAL_INFERENCE_V5_BOOTSTRAP_SEED,
    ACTUAL_INFERENCE_V5_CASE_PATH,
    ACTUAL_INFERENCE_V5_COMPONENTS,
    ACTUAL_INFERENCE_V5_CONTINUATION_BYTES,
    ACTUAL_INFERENCE_V5_IMPLEMENTATION_FILE_ORDER,
    ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES,
    ACTUAL_INFERENCE_V5_MEMORY_ROOT,
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
    ACTUAL_INFERENCE_V5_SUMMARY_PATH,
    ACTUAL_INFERENCE_V5_WARMUP_CASES,
    RUNTIME_COUNTER_NAMES,
    actual_efficiency_component_pass,
    array_sha256,
    assert_workspace_path_no_symlinks,
    canonical_sha256,
    session_schedule,
    three_way_paired_latency,
    validate_actual_inference_plan_v5,
    validate_free_output_bytes,
    validate_isolated_memory_receipt,
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


PLAN_PATH = Path(ACTUAL_INFERENCE_V5_PLAN_PATH)
CASE_PATH = Path(ACTUAL_INFERENCE_V5_CASE_PATH)
AUTHORIZATION_PATH = Path(FINAL_AUTHORIZATION_PATH)
QUALITY_LOCK_PATH = Path(FINAL_QUALITY_LOCK_PATH)
SELECTION_PATH = Path(SELECTION_LOCK_PATH)
RUN_ROOT = Path(ACTUAL_INFERENCE_V5_SESSION_ROOT)
SESSION_RECEIPT_ROOT = Path(ACTUAL_INFERENCE_V5_SESSION_RECEIPT_ROOT)
ARTIFACT_ROOT = Path(ACTUAL_INFERENCE_V5_ARTIFACT_ROOT)
MEMORY_ROOT = Path(ACTUAL_INFERENCE_V5_MEMORY_ROOT)
OUTPUT_PATH = Path(ACTUAL_INFERENCE_V5_SUMMARY_PATH)
ACTIVE_PATHS = (
    ARTIFACT_ROOT / ".active",
    ARTIFACT_ROOT / ".memory-active",
)
MACHINE_LOCK_PATH = Path("/tmp/jamoflow-publication-mps.lock")


@contextmanager
def _exclusive_evidence_snapshot():
    with MACHINE_LOCK_PATH.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another actual-inference measurement is live"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
        raise ValueError("actual-inference summary requires a clean repository root")
    if any(path.exists() for path in ACTIVE_PATHS):
        raise ValueError("actual-inference work is still active or incomplete")
    return _git_commit()


def _post_publish_status_is_clean() -> bool:
    lines = {line for line in _git_status().splitlines() if line.strip()}
    return lines <= {f"?? {OUTPUT_PATH.as_posix()}"}


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
    ).stdout.strip()
    if (
        blob.returncode != 0
        or len(commit) != 40
        or not path.is_file()
        or path.is_symlink()
        or path.read_bytes() != blob.stdout
    ):
        raise ValueError(f"actual summary input is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit,
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
        raise ValueError("actual summary history check failed")
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
        raise ValueError("actual evidence receipt history is malformed") from error
    if result.returncode != 0 or count != 1:
        raise ValueError("actual evidence receipt was rewritten or deleted")
    return count


def _require_ancestor(ancestor: str, descendant: str, label: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    ).returncode != 0:
        raise ValueError(f"actual summary Git order differs: {label}")


def _verify_existing_summary() -> int:
    _require_clean_root()
    identity = _tracked_head_identity(OUTPUT_PATH)
    _tracked_touch_count(OUTPUT_PATH)
    payload = _read_json(OUTPUT_PATH)
    if (
        payload.get("kind") != "phase3_inference_actual_summary_v5r3"
        or payload.get("schema_version") != 6
        or payload.get("protocol_version") != 5
        or payload.get("protocol_revision")
        != ACTUAL_INFERENCE_V5_PROTOCOL_REVISION
        or payload.get("summary_path") != ACTUAL_INFERENCE_V5_SUMMARY_PATH
        or payload.get("summary_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "summary_sha256"
            }
        )
        or payload.get("plan_artifact") != _tracked_head_identity(PLAN_PATH)
        or not isinstance(payload.get("summary_base_git_commit"), str)
        or len(payload["summary_base_git_commit"]) != 40
        or payload["summary_base_git_commit"] == identity["git_commit"]
    ):
        raise ValueError("committed actual summary identity differs")
    _require_ancestor(
        payload["summary_base_git_commit"],
        identity["git_commit"],
        "summary base -> summary artifact",
    )
    print("verified immutable committed actual-inference v5 summary", flush=True)
    return 0


def _load_context() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], str
]:
    summary_commit = _require_clean_root()
    plan_identity = _tracked_head_identity(PLAN_PATH)
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
    if (
        _tracked_head_identity(AUTHORIZATION_PATH)
        != plan["authorization_artifact"]
        or _tracked_head_identity(QUALITY_LOCK_PATH)
        != plan["quality_lock_artifact"]
        or hash_file(CASE_PATH) != plan["case_context"]["artifact_sha256"]
    ):
        raise ValueError("actual summary upstream artifacts differ")
    for path, expected in plan["implementation_sha256"].items():
        if _tracked_head_identity(Path(path))["sha256"] != expected:
            raise ValueError(f"actual summary implementation differs: {path}")
    return plan, authorization, quality, plan_identity, summary_commit


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
        raise ValueError("actual summary role has no unique physical model")
    return matches[0]


def _session_paths(session_id: str) -> dict[str, Path]:
    return {
        "outputs": ARTIFACT_ROOT / session_id / "free-outputs.npz",
        "report": SESSION_RECEIPT_ROOT / f"{session_id}.json",
        "timings": ARTIFACT_ROOT / session_id / "timings.npz",
    }


def _expected_timing_keys() -> set[str]:
    output = set()
    for mode in ACTUAL_INFERENCE_V5_MODES:
        for role in ACTUAL_INFERENCE_V5_ROLES:
            output.update(
                f"{mode}__{component}__{role}"
                for component in ACTUAL_INFERENCE_V5_COMPONENTS
            )
            output.update(
                {
                    f"{mode}__emitted_output_bytes__{role}",
                    f"{mode}__global_patches__{role}",
                    f"{mode}__runtime_observed_bytes__{role}",
                }
            )
            output.update(
                f"{mode}__counter_{counter}__{role}"
                for counter in RUNTIME_COUNTER_NAMES
            )
    return output


def _valid_argmax_partition(
    values: Mapping[str, Any],
    *,
    exact_key: str,
    tie_key: str,
    position_key: str,
) -> bool:
    counts = (
        values.get(exact_key),
        values.get(tie_key),
        values.get(position_key),
    )
    return bool(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for value in counts
        )
        and counts[0] + counts[1] == counts[2]
    )


def _validate_correctness(
    correctness: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    comparison_contract: str = "mps_backend",
) -> None:
    if comparison_contract == "cpu_semantic":
        expected_atol = ACTUAL_INFERENCE_EQUIVALENCE_ATOL
        expected_rtol = ACTUAL_INFERENCE_EQUIVALENCE_RTOL
    elif comparison_contract == "mps_backend":
        expected_atol = ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL
        expected_rtol = ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL
    else:
        raise ValueError("actual correctness comparison contract differs")
    if set(correctness) != {str(seed) for seed in FINAL_SEEDS}:
        raise ValueError("actual correctness seed set differs")
    for seed in FINAL_SEEDS:
        rows = correctness[str(seed)]
        if set(rows) != set(ACTUAL_INFERENCE_V5_ROLES):
            raise ValueError("actual correctness role set differs")
        for role in ACTUAL_INFERENCE_V5_ROLES:
            values = rows[role]
            expected_keys = {
                "atol",
                "boundary_trace_sha256",
                "comparison_contract",
                "entropy_router_argmax_exact_comparisons",
                "entropy_router_position_comparisons",
                "entropy_router_tolerance_tie_argmax_comparisons",
                "main_full_causal_argmax_exact_comparisons",
                "main_full_causal_position_comparisons",
                "main_full_causal_tolerance_tie_argmax_comparisons",
                "main_parallel_argmax_exact_comparisons",
                "main_parallel_position_comparisons",
                "main_parallel_tolerance_tie_argmax_comparisons",
                "maximum_main_absolute_logit_error",
                "maximum_main_nominal_normalized_tolerance_ratio",
                "maximum_main_normalized_tolerance_ratio",
                "maximum_main_probability_total_variation",
                "maximum_router_absolute_entropy_error",
                "maximum_router_absolute_logit_error",
                "maximum_router_nominal_entropy_tolerance_ratio",
                "maximum_router_nominal_logit_tolerance_ratio",
                "maximum_router_normalized_entropy_tolerance_ratio",
                "maximum_router_normalized_logit_tolerance_ratio",
                "maximum_router_probability_total_variation",
                "main_nominal_tolerance_violation_elements",
                "nominal_atol",
                "nominal_rtol",
                "pass",
                "probability_total_variation_limit",
                "router_entropy_nominal_tolerance_violation_elements",
                "router_logit_nominal_tolerance_violation_elements",
                "rtol",
            }
            model = _model_for_role(authorization, plan, role)
            entropy = bool(model["descriptor"]["requires_entropy_router"])
            if (
                not isinstance(values, Mapping)
                or set(values) != expected_keys
                or values.get("pass") is not True
                or values.get("comparison_contract") != comparison_contract
                or values.get("rtol") != expected_rtol
                or values.get("atol") != expected_atol
                or values.get("nominal_rtol")
                != ACTUAL_INFERENCE_EQUIVALENCE_RTOL
                or values.get("nominal_atol")
                != ACTUAL_INFERENCE_EQUIVALENCE_ATOL
                or values.get("probability_total_variation_limit")
                != ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
                or values.get("main_full_causal_position_comparisons")
                != (ACTUAL_INFERENCE_V5_WARMUP_CASES + ACTUAL_INFERENCE_V5_MEASURED_CASES) * 255
                or not _valid_argmax_partition(
                    values,
                    exact_key="main_full_causal_argmax_exact_comparisons",
                    tie_key="main_full_causal_tolerance_tie_argmax_comparisons",
                    position_key="main_full_causal_position_comparisons",
                )
                or values.get("main_parallel_position_comparisons")
                != (ACTUAL_INFERENCE_V5_WARMUP_CASES + ACTUAL_INFERENCE_V5_MEASURED_CASES) * 128
                or not _valid_argmax_partition(
                    values,
                    exact_key="main_parallel_argmax_exact_comparisons",
                    tie_key="main_parallel_tolerance_tie_argmax_comparisons",
                    position_key="main_parallel_position_comparisons",
                )
                or values.get("entropy_router_position_comparisons")
                != (
                    (ACTUAL_INFERENCE_V5_WARMUP_CASES + ACTUAL_INFERENCE_V5_MEASURED_CASES) * 255
                    if entropy
                    else 0
                )
                or not _valid_argmax_partition(
                    values,
                    exact_key="entropy_router_argmax_exact_comparisons",
                    tie_key="entropy_router_tolerance_tie_argmax_comparisons",
                    position_key="entropy_router_position_comparisons",
                )
                or not isinstance(values.get("boundary_trace_sha256"), str)
                or len(values["boundary_trace_sha256"]) != 64
                or any(
                    not isinstance(values.get(key), (int, float))
                    or not math.isfinite(float(values[key]))
                    or float(values[key]) < 0
                    for key in (
                        "maximum_main_absolute_logit_error",
                        "maximum_main_nominal_normalized_tolerance_ratio",
                        "maximum_main_normalized_tolerance_ratio",
                        "maximum_main_probability_total_variation",
                        "maximum_router_absolute_entropy_error",
                        "maximum_router_absolute_logit_error",
                        "maximum_router_nominal_entropy_tolerance_ratio",
                        "maximum_router_nominal_logit_tolerance_ratio",
                        "maximum_router_normalized_entropy_tolerance_ratio",
                        "maximum_router_normalized_logit_tolerance_ratio",
                        "maximum_router_probability_total_variation",
                    )
                )
                or any(
                    float(values[key]) > 1.0
                    for key in (
                        "maximum_main_normalized_tolerance_ratio",
                        "maximum_router_normalized_entropy_tolerance_ratio",
                        "maximum_router_normalized_logit_tolerance_ratio",
                    )
                )
                or values["maximum_main_probability_total_variation"]
                > ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
                or values["maximum_router_probability_total_variation"]
                > ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
                or any(
                    not isinstance(values.get(key), int)
                    or isinstance(values[key], bool)
                    or values[key] < 0
                    for key in (
                        "main_nominal_tolerance_violation_elements",
                        "router_entropy_nominal_tolerance_violation_elements",
                        "router_logit_nominal_tolerance_violation_elements",
                    )
                )
                or (
                    comparison_contract == "cpu_semantic"
                    and (
                        values["maximum_main_nominal_normalized_tolerance_ratio"]
                        > 1.0
                        or values[
                            "maximum_router_nominal_entropy_tolerance_ratio"
                        ]
                        > 1.0
                        or values[
                            "maximum_router_nominal_logit_tolerance_ratio"
                        ]
                        > 1.0
                        or values["main_nominal_tolerance_violation_elements"]
                        != 0
                        or values[
                            "router_entropy_nominal_tolerance_violation_elements"
                        ]
                        != 0
                        or values[
                            "router_logit_nominal_tolerance_violation_elements"
                        ]
                        != 0
                    )
                )
                or (
                    not entropy
                    and (
                        values["maximum_router_absolute_entropy_error"] != 0
                        or values["maximum_router_absolute_logit_error"] != 0
                        or values["maximum_router_normalized_entropy_tolerance_ratio"] != 0
                        or values["maximum_router_normalized_logit_tolerance_ratio"] != 0
                        or values[
                            "maximum_router_nominal_entropy_tolerance_ratio"
                        ]
                        != 0
                        or values[
                            "maximum_router_nominal_logit_tolerance_ratio"
                        ]
                        != 0
                        or values["maximum_router_probability_total_variation"]
                        != 0
                        or values[
                            "router_entropy_nominal_tolerance_violation_elements"
                        ]
                        != 0
                        or values[
                            "router_logit_nominal_tolerance_violation_elements"
                        ]
                        != 0
                    )
                )
            ):
                raise ValueError(f"actual correctness evidence differs: {seed}/{role}")


def _validate_model_provenance(
    provenance: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    if set(provenance) != {str(seed) for seed in FINAL_SEEDS}:
        raise ValueError("actual model provenance seed set differs")
    for seed in FINAL_SEEDS:
        if set(provenance[str(seed)]) != set(ACTUAL_INFERENCE_V5_ROLES):
            raise ValueError("actual model provenance role set differs")
        for role in ACTUAL_INFERENCE_V5_ROLES:
            model = _model_for_role(authorization, plan, role)
            evidence = model["seeds"][str(seed)]
            auxiliary = evidence["auxiliary"]
            expected = {
                "checkpoint_artifact_sha256": evidence["checkpoint"]["artifact_sha256"],
                "checkpoint_state_sha256": evidence["checkpoint"]["state_sha256"],
                "model_identity_sha256": model["identity_sha256"],
                "requires_entropy_router": bool(model["descriptor"]["requires_entropy_router"]),
                "router_checkpoint_state_sha256": (
                    auxiliary["router_checkpoint_state_sha256"]
                    if auxiliary["kind"] == "entropy_router"
                    else None
                ),
            }
            if provenance[str(seed)][role] != expected:
                raise ValueError(f"actual model provenance differs: {seed}/{role}")


def _validate_free_path_correctness(
    correctness: Mapping[str, Any],
    *,
    output_lengths: np.ndarray,
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    if set(correctness) != {str(seed) for seed in FINAL_SEEDS}:
        raise ValueError("free-path correctness seed set differs")
    for seed_index, seed in enumerate(FINAL_SEEDS):
        if set(correctness[str(seed)]) != set(ACTUAL_INFERENCE_V5_ROLES):
            raise ValueError("free-path correctness role set differs")
        for role_index, role in enumerate(ACTUAL_INFERENCE_V5_ROLES):
            values = correctness[str(seed)][role]
            expected_keys = {
                "atol",
                "boundary_trace_sha256",
                "comparison_contract",
                "entropy_router_argmax_exact_comparisons",
                "entropy_router_position_comparisons",
                "entropy_router_tolerance_tie_argmax_comparisons",
                "greedy_byte_argmax_comparisons",
                "main_full_causal_argmax_exact_comparisons",
                "main_full_causal_position_comparisons",
                "main_full_causal_tolerance_tie_argmax_comparisons",
                "main_parallel_argmax_exact_comparisons",
                "main_parallel_position_comparisons",
                "main_parallel_tolerance_tie_argmax_comparisons",
                "maximum_main_absolute_logit_error",
                "maximum_main_nominal_normalized_tolerance_ratio",
                "maximum_main_normalized_tolerance_ratio",
                "maximum_main_probability_total_variation",
                "maximum_router_absolute_entropy_error",
                "maximum_router_absolute_logit_error",
                "maximum_router_nominal_entropy_tolerance_ratio",
                "maximum_router_nominal_logit_tolerance_ratio",
                "maximum_router_normalized_entropy_tolerance_ratio",
                "maximum_router_normalized_logit_tolerance_ratio",
                "maximum_router_probability_total_variation",
                "main_nominal_tolerance_violation_elements",
                "nominal_atol",
                "nominal_rtol",
                "pass",
                "probability_total_variation_limit",
                "router_entropy_nominal_tolerance_violation_elements",
                "router_logit_nominal_tolerance_violation_elements",
                "rtol",
            }
            lengths = output_lengths[seed_index, role_index]
            if not np.all(lengths == lengths[:, :1]):
                raise ValueError("free-path output lengths changed across repetitions")
            expected_positions = int(
                np.sum(
                    ACTUAL_INFERENCE_V5_PROMPT_BYTES
                    + lengths[:, 0].astype(np.int64)
                    - 1
                )
            )
            expected_generated = int(np.sum(lengths[:, 0]))
            model = _model_for_role(authorization, plan, role)
            entropy = bool(model["descriptor"]["requires_entropy_router"])
            if (
                not isinstance(values, Mapping)
                or set(values) != expected_keys
                or values.get("pass") is not True
                or values.get("comparison_contract") != "mps_backend"
                or values.get("rtol") != ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL
                or values.get("atol") != ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL
                or values.get("nominal_rtol")
                != ACTUAL_INFERENCE_EQUIVALENCE_RTOL
                or values.get("nominal_atol")
                != ACTUAL_INFERENCE_EQUIVALENCE_ATOL
                or values.get("probability_total_variation_limit")
                != ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
                or values.get("main_full_causal_position_comparisons")
                != expected_positions
                or not _valid_argmax_partition(
                    values,
                    exact_key="main_full_causal_argmax_exact_comparisons",
                    tie_key="main_full_causal_tolerance_tie_argmax_comparisons",
                    position_key="main_full_causal_position_comparisons",
                )
                or values.get("main_parallel_position_comparisons")
                != expected_generated
                or not _valid_argmax_partition(
                    values,
                    exact_key="main_parallel_argmax_exact_comparisons",
                    tie_key="main_parallel_tolerance_tie_argmax_comparisons",
                    position_key="main_parallel_position_comparisons",
                )
                or values.get("greedy_byte_argmax_comparisons")
                != expected_generated
                or values.get("entropy_router_position_comparisons")
                != (expected_positions if entropy else 0)
                or not _valid_argmax_partition(
                    values,
                    exact_key="entropy_router_argmax_exact_comparisons",
                    tie_key="entropy_router_tolerance_tie_argmax_comparisons",
                    position_key="entropy_router_position_comparisons",
                )
                or not isinstance(values.get("boundary_trace_sha256"), str)
                or len(values["boundary_trace_sha256"]) != 64
                or any(
                    not isinstance(values.get(key), (int, float))
                    or not math.isfinite(float(values[key]))
                    or float(values[key]) < 0
                    for key in (
                        "maximum_main_absolute_logit_error",
                        "maximum_main_nominal_normalized_tolerance_ratio",
                        "maximum_main_normalized_tolerance_ratio",
                        "maximum_main_probability_total_variation",
                        "maximum_router_absolute_entropy_error",
                        "maximum_router_absolute_logit_error",
                        "maximum_router_nominal_entropy_tolerance_ratio",
                        "maximum_router_nominal_logit_tolerance_ratio",
                        "maximum_router_normalized_entropy_tolerance_ratio",
                        "maximum_router_normalized_logit_tolerance_ratio",
                        "maximum_router_probability_total_variation",
                    )
                )
                or any(
                    float(values[key]) > 1.0
                    for key in (
                        "maximum_main_normalized_tolerance_ratio",
                        "maximum_router_normalized_entropy_tolerance_ratio",
                        "maximum_router_normalized_logit_tolerance_ratio",
                    )
                )
                or values["maximum_main_probability_total_variation"]
                > ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
                or values["maximum_router_probability_total_variation"]
                > ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
                or any(
                    not isinstance(values.get(key), int)
                    or isinstance(values[key], bool)
                    or values[key] < 0
                    for key in (
                        "main_nominal_tolerance_violation_elements",
                        "router_entropy_nominal_tolerance_violation_elements",
                        "router_logit_nominal_tolerance_violation_elements",
                    )
                )
                or (
                    not entropy
                    and (
                        values["maximum_router_absolute_entropy_error"] != 0
                        or values["maximum_router_absolute_logit_error"] != 0
                        or values["maximum_router_normalized_entropy_tolerance_ratio"] != 0
                        or values["maximum_router_normalized_logit_tolerance_ratio"] != 0
                        or values[
                            "maximum_router_nominal_entropy_tolerance_ratio"
                        ]
                        != 0
                        or values[
                            "maximum_router_nominal_logit_tolerance_ratio"
                        ]
                        != 0
                        or values["maximum_router_probability_total_variation"]
                        != 0
                        or values[
                            "router_entropy_nominal_tolerance_violation_elements"
                        ]
                        != 0
                        or values[
                            "router_logit_nominal_tolerance_violation_elements"
                        ]
                        != 0
                    )
                )
            ):
                raise ValueError(
                    f"free-path correctness evidence differs: {seed}/{role}"
                )


def _validate_boundary_trace_stability(
    reports: list[Mapping[str, Any]],
) -> None:
    if len(reports) != len(ACTUAL_INFERENCE_V5_SESSIONS):
        raise ValueError("actual boundary trace session count differs")
    for evidence_key in (
        "correctness",
        "cpu_semantic_correctness",
        "free_path_correctness",
    ):
        for seed in FINAL_SEEDS:
            for role in ACTUAL_INFERENCE_V5_ROLES:
                traces = {
                    report[evidence_key][str(seed)][role][
                        "boundary_trace_sha256"
                    ]
                    for report in reports
                }
                if len(traces) != 1:
                    raise ValueError(
                        "actual boundary trace changed across sessions: "
                        f"{evidence_key}/{seed}/{role}"
                    )


def _argmax_partition_summary(
    reports: list[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "criterion": (
            "exact argmax or differing argmax whose two candidate-logit "
            "active backend-tolerance intervals overlap; CPU logits must pass "
            "the original semantic envelope and MPS logits must pass the "
            "safety envelope plus probability-TV bound"
        )
    }
    for evidence_key, label in (
        ("correctness", "controlled_replay"),
        ("cpu_semantic_correctness", "cpu_semantic_controlled_replay"),
        ("free_path_correctness", "free_running_utf8_greedy"),
    ):
        output[label] = {}
        for prefix in (
            "main_full_causal",
            "main_parallel",
            "entropy_router",
        ):
            exact_key = f"{prefix}_argmax_exact_comparisons"
            tie_key = f"{prefix}_tolerance_tie_argmax_comparisons"
            position_key = f"{prefix}_position_comparisons"
            exact = sum(
                int(report[evidence_key][str(seed)][role][exact_key])
                for report in reports
                for seed in FINAL_SEEDS
                for role in ACTUAL_INFERENCE_V5_ROLES
            )
            ties = sum(
                int(report[evidence_key][str(seed)][role][tie_key])
                for report in reports
                for seed in FINAL_SEEDS
                for role in ACTUAL_INFERENCE_V5_ROLES
            )
            positions = sum(
                int(report[evidence_key][str(seed)][role][position_key])
                for report in reports
                for seed in FINAL_SEEDS
                for role in ACTUAL_INFERENCE_V5_ROLES
            )
            if exact + ties != positions:
                raise ValueError("actual aggregate argmax partition differs")
            output[label][prefix] = {
                "exact_argmax_comparisons": exact,
                "position_comparisons": positions,
                "tolerance_tie_argmax_comparisons": ties,
            }
    return output


def _numerical_correctness_summary(
    reports: list[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for evidence_key, label in (
        ("cpu_semantic_correctness", "cpu_semantic_controlled_replay"),
        ("correctness", "mps_controlled_replay"),
        ("free_path_correctness", "mps_free_running_utf8_greedy"),
    ):
        rows = [
            report[evidence_key][str(seed)][role]
            for report in reports
            for seed in FINAL_SEEDS
            for role in ACTUAL_INFERENCE_V5_ROLES
        ]
        contracts = {row["comparison_contract"] for row in rows}
        if len(contracts) != 1:
            raise ValueError("actual numerical correctness contract changed")
        output[label] = {
            "comparison_contract": next(iter(contracts)),
            "main_nominal_tolerance_violation_elements": sum(
                row["main_nominal_tolerance_violation_elements"] for row in rows
            ),
            "maximum_main_active_normalized_tolerance_ratio": max(
                row["maximum_main_normalized_tolerance_ratio"] for row in rows
            ),
            "maximum_main_nominal_normalized_tolerance_ratio": max(
                row["maximum_main_nominal_normalized_tolerance_ratio"]
                for row in rows
            ),
            "maximum_main_probability_total_variation": max(
                row["maximum_main_probability_total_variation"] for row in rows
            ),
            "maximum_router_active_normalized_entropy_tolerance_ratio": max(
                row["maximum_router_normalized_entropy_tolerance_ratio"]
                for row in rows
            ),
            "maximum_router_active_normalized_logit_tolerance_ratio": max(
                row["maximum_router_normalized_logit_tolerance_ratio"]
                for row in rows
            ),
            "maximum_router_probability_total_variation": max(
                row["maximum_router_probability_total_variation"] for row in rows
            ),
            "router_entropy_nominal_tolerance_violation_elements": sum(
                row["router_entropy_nominal_tolerance_violation_elements"]
                for row in rows
            ),
            "router_logit_nominal_tolerance_violation_elements": sum(
                row["router_logit_nominal_tolerance_violation_elements"]
                for row in rows
            ),
        }
    return output


def _validate_fresh_session_environments(
    reports: list[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> list[str]:
    tokens = [report["process"]["start_token_sha256"] for report in reports]
    environments = [report["environment"] for report in reports]
    if (
        len(reports) != len(ACTUAL_INFERENCE_V5_SESSIONS)
        or len(set(tokens)) != len(ACTUAL_INFERENCE_V5_SESSIONS)
        or any(environment != environments[0] for environment in environments[1:])
        or environments[0].get("device") != "mps"
        or environments[0].get("mps_available") is not True
        or {
            key: environments[0][key]
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
    ):
        raise ValueError("actual timing sessions did not use fresh processes")
    return tokens


def _load_session(
    session_id: str,
    *,
    plan: Mapping[str, Any],
    plan_artifact_sha256: str,
    authorization: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    paths = _session_paths(session_id)
    if any(not path.is_file() or path.is_symlink() for path in paths.values()):
        raise ValueError(f"actual session artifact is missing or symlinked: {session_id}")
    _tracked_head_identity(paths["report"])
    _tracked_touch_count(paths["report"])
    report = _read_json(paths["report"])
    expected_report_keys = {
        "complete",
        "correctness",
        "cpu_semantic_correctness",
        "environment",
        "free_path_correctness",
        "kind",
        "model_provenance",
        "output_array_sha256",
        "output_artifact_sha256",
        "plan_artifact_sha256",
        "plan_sha256",
        "process",
        "protocol_version",
        "protocol_revision",
        "schema_version",
        "session_git_commit",
        "session_id",
        "session_schedule_sha256",
        "thermal_samples",
        "timing_array_sha256",
        "timing_artifact_sha256",
    }
    session_index = ACTUAL_INFERENCE_V5_SESSIONS.index(session_id)
    if (
        set(report) != expected_report_keys
        or report.get("complete") is not True
        or report.get("kind") != "phase3_inference_actual_session_v5r3"
        or report.get("schema_version") != 6
        or report.get("protocol_version") != 5
        or report.get("protocol_revision")
        != ACTUAL_INFERENCE_V5_PROTOCOL_REVISION
        or report.get("session_id") != session_id
        or not isinstance(report.get("session_git_commit"), str)
        or len(report["session_git_commit"]) != 40
        or report.get("plan_sha256") != plan["plan_sha256"]
        or report.get("plan_artifact_sha256") != plan_artifact_sha256
        or report.get("session_schedule_sha256")
        != canonical_sha256(plan["session_schedules"][session_index])
        or report.get("timing_artifact_sha256") != hash_file(paths["timings"])
        or report.get("output_artifact_sha256") != hash_file(paths["outputs"])
    ):
        raise ValueError(f"actual session report differs: {session_id}")
    process = report["process"]
    if (
        set(process) != {"pid", "process_start", "start_token_sha256"}
        or not isinstance(process["pid"], int)
        or process["pid"] <= 0
        or process["process_start"].get("returncode") != 0
        or not process["process_start"].get("stdout")
        or process["start_token_sha256"]
        != hashlib.sha256(
            json.dumps(
                {key: value for key, value in process.items() if key != "start_token_sha256"},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise ValueError(f"actual session process identity differs: {session_id}")
    samples = report["thermal_samples"]
    if len(samples) != 91 or any(
        set(sample) != {"eligible", "environment", "inventory"}
        or sample["eligible"] is not True
        or not timing_environment_eligible(sample["environment"])
        or set(sample["inventory"])
        != {
            "conflicting_neural_processes",
            "ps_parse_pass",
            "ps_returncode",
            "ps_stdout_nonempty",
            "snapshot_sha256",
        }
        or sample["inventory"].get("conflicting_neural_processes") != []
        or sample["inventory"].get("ps_parse_pass") is not True
        or sample["inventory"].get("ps_returncode") != 0
        or sample["inventory"].get("ps_stdout_nonempty") is not True
        or not isinstance(sample["inventory"].get("snapshot_sha256"), str)
        or len(sample["inventory"]["snapshot_sha256"]) != 64
        for sample in samples
    ):
        raise ValueError(f"actual session thermal/process evidence differs: {session_id}")
    _validate_correctness(
        report["correctness"],
        authorization=authorization,
        plan=plan,
        comparison_contract="mps_backend",
    )
    _validate_correctness(
        report["cpu_semantic_correctness"],
        authorization=authorization,
        plan=plan,
        comparison_contract="cpu_semantic",
    )
    for seed in FINAL_SEEDS:
        for role in ACTUAL_INFERENCE_V5_ROLES:
            if (
                report["correctness"][str(seed)][role][
                    "boundary_trace_sha256"
                ]
                != report["cpu_semantic_correctness"][str(seed)][role][
                    "boundary_trace_sha256"
                ]
            ):
                raise ValueError("CPU/MPS semantic boundary trace differs")
    _validate_model_provenance(
        report["model_provenance"],
        authorization=authorization,
        plan=plan,
    )
    with np.load(paths["timings"], allow_pickle=False) as archive:
        if set(archive.files) != _expected_timing_keys():
            raise ValueError(f"actual timing array set differs: {session_id}")
        timings = {key: archive[key] for key in archive.files}
    shape = (len(FINAL_SEEDS), ACTUAL_INFERENCE_V5_MEASURED_CASES, ACTUAL_INFERENCE_V5_REPETITIONS)
    if set(report["timing_array_sha256"]) != set(timings):
        raise ValueError(f"actual timing hash set differs: {session_id}")
    for key, values in timings.items():
        is_time = any(f"__{component}__" in key for component in ACTUAL_INFERENCE_V5_COMPONENTS)
        if (
            values.shape != shape
            or (is_time and (values.dtype != np.float64 or not np.isfinite(values).all() or np.any(values <= 0)))
            or (not is_time and (not np.issubdtype(values.dtype, np.integer) or np.any(values < 0)))
            or report["timing_array_sha256"].get(key) != array_sha256(values)
        ):
            raise ValueError(f"actual timing array differs: {session_id}/{key}")
    for mode in ACTUAL_INFERENCE_V5_MODES:
        for role in ACTUAL_INFERENCE_V5_ROLES:
            ttft = timings[f"{mode}__ttft_ms__{role}"]
            decode = timings[f"{mode}__decode_ms__{role}"]
            total = timings[f"{mode}__end_to_end_ms__{role}"]
            if not np.allclose(total, ttft + decode, rtol=0, atol=1e-9):
                raise ValueError("actual timing component identity differs")
            emitted = timings[f"{mode}__emitted_output_bytes__{role}"]
            observed = timings[f"{mode}__runtime_observed_bytes__{role}"]
            patches = timings[f"{mode}__global_patches__{role}"]
            if (
                np.any(observed != ACTUAL_INFERENCE_V5_PROMPT_BYTES + emitted - 1)
                or np.any(patches <= 0)
                or np.any(patches > observed)
            ):
                raise ValueError("actual timing patch/output identity differs")
            model = _model_for_role(authorization, plan, role)
            validate_runtime_counter_arrays(
                {
                    counter: timings[f"{mode}__counter_{counter}__{role}"]
                    for counter in RUNTIME_COUNTER_NAMES
                },
                requires_entropy_router=bool(model["descriptor"]["requires_entropy_router"]),
                mode=mode,
                emitted_output_bytes=emitted,
            )
    with np.load(paths["outputs"], allow_pickle=False) as archive:
        if set(archive.files) != {"free_output_bytes", "free_output_lengths"}:
            raise ValueError(f"actual output array set differs: {session_id}")
        outputs = {key: archive[key] for key in archive.files}
    if (
        set(report["output_array_sha256"]) != set(outputs)
        or any(report["output_array_sha256"][key] != array_sha256(value) for key, value in outputs.items())
        or outputs["free_output_bytes"].dtype != np.uint8
        or outputs["free_output_bytes"].shape
        != (
            len(FINAL_SEEDS),
            len(ACTUAL_INFERENCE_V5_ROLES),
            ACTUAL_INFERENCE_V5_MEASURED_CASES,
            ACTUAL_INFERENCE_V5_REPETITIONS,
            ACTUAL_INFERENCE_V5_MAXIMUM_OUTPUT_BYTES,
        )
        or not np.issubdtype(outputs["free_output_lengths"].dtype, np.integer)
        or outputs["free_output_lengths"].shape
        != outputs["free_output_bytes"].shape[:-1]
    ):
        raise ValueError(f"actual output arrays differ: {session_id}")
    for role_index, role in enumerate(ACTUAL_INFERENCE_V5_ROLES):
        if not np.array_equal(
            outputs["free_output_lengths"][:, role_index],
            timings[f"free_running_utf8_greedy__emitted_output_bytes__{role}"],
        ):
            raise ValueError("actual stored output lengths differ from timing")
    _validate_free_path_correctness(
        report["free_path_correctness"],
        output_lengths=outputs["free_output_lengths"],
        authorization=authorization,
        plan=plan,
    )
    return timings, outputs, report


def _validate_current_models(
    *, authorization: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    for role in ACTUAL_INFERENCE_V5_ROLES:
        identity = _model_for_role(authorization, plan, role)
        for seed in FINAL_SEEDS:
            bundle = load_actual_model(
                role=role,
                identity=identity,
                seed=seed,
                device="cpu",
            )
            release_actual_model(bundle)


def _load_memory(
    *,
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_git_commit: str,
    summary_git_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts = []
    summary: dict[str, Any] = {}
    by_role_seed: dict[str, dict[int, dict[str, Any]]] = {}
    for role in ACTUAL_INFERENCE_V5_ROLES:
        identity = _model_for_role(authorization, plan, role)
        role_receipts = []
        by_role_seed[role] = {}
        for seed in FINAL_SEEDS:
            path = MEMORY_ROOT / role / f"seed-{seed}.json"
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"isolated memory receipt is missing: {role}/{seed}")
            receipt = _read_json(path)
            receipt_identity = _tracked_head_identity(path)
            _tracked_touch_count(path)
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
            if (
                receipt["workload"]["case_artifact_sha256"]
                != plan["case_context"]["artifact_sha256"]
                or receipt["workload"]["prompt_array_sha256"]
                != plan["case_context"]["prompt_array_sha256"]
            ):
                raise ValueError(
                    f"isolated memory workload differs: {role}/{seed}"
                )
            _require_ancestor(
                plan_git_commit,
                receipt["measurement_git_commit"],
                f"plan -> memory {role}/{seed}",
            )
            _require_ancestor(
                receipt["measurement_git_commit"],
                receipt_identity["git_commit"],
                f"memory measurement -> receipt {role}/{seed}",
            )
            _require_ancestor(
                receipt_identity["git_commit"],
                summary_git_commit,
                f"memory receipt {role}/{seed} -> summary",
            )
            receipts.append({"path": path.as_posix(), "sha256": hash_file(path), "receipt": receipt})
            role_receipts.append(receipt)
            by_role_seed[role][seed] = receipt
        summary[role] = {
            "maximum_mps_current_increment_bytes": max(
                row["mps_snapshots"]["after_inference_current_bytes"]
                - row["mps_snapshots"]["baseline_current_bytes"]
                for row in role_receipts
            ),
            "maximum_mps_driver_increment_bytes": max(
                row["mps_snapshots"]["after_inference_driver_bytes"]
                - row["mps_snapshots"]["baseline_driver_bytes"]
                for row in role_receipts
            ),
            "maximum_process_high_water_increment_bytes": max(
                row["process_rss"]["high_water_bytes"]
                - row["process_rss"]["baseline_bytes"]
                for row in role_receipts
            ),
            "maximum_process_high_water_bytes": max(
                row["process_rss"]["high_water_bytes"] for row in role_receipts
            ),
            "maximum_mps_after_inference_current_bytes": max(
                row["mps_snapshots"]["after_inference_current_bytes"] for row in role_receipts
            ),
            "maximum_mps_after_inference_driver_bytes": max(
                row["mps_snapshots"]["after_inference_driver_bytes"] for row in role_receipts
            ),
            "maximum_parameter_bytes": max(row["parameter_bytes"] for row in role_receipts),
            "measurement_role": "descriptive_only_not_a_publication_gate",
        }
    summary["paired_candidate_minus_reference_by_seed"] = {
        str(seed): {
            "mps_current_increment_bytes": (
                by_role_seed["candidate"][seed]["mps_snapshots"][
                    "after_inference_current_bytes"
                ]
                - by_role_seed["candidate"][seed]["mps_snapshots"][
                    "baseline_current_bytes"
                ]
                - by_role_seed["reference"][seed]["mps_snapshots"][
                    "after_inference_current_bytes"
                ]
                + by_role_seed["reference"][seed]["mps_snapshots"][
                    "baseline_current_bytes"
                ]
            ),
            "mps_driver_increment_bytes": (
                by_role_seed["candidate"][seed]["mps_snapshots"][
                    "after_inference_driver_bytes"
                ]
                - by_role_seed["candidate"][seed]["mps_snapshots"][
                    "baseline_driver_bytes"
                ]
                - by_role_seed["reference"][seed]["mps_snapshots"][
                    "after_inference_driver_bytes"
                ]
                + by_role_seed["reference"][seed]["mps_snapshots"][
                    "baseline_driver_bytes"
                ]
            ),
            "parameter_bytes": (
                by_role_seed["candidate"][seed]["parameter_bytes"]
                - by_role_seed["reference"][seed]["parameter_bytes"]
            ),
            "process_high_water_increment_bytes": (
                by_role_seed["candidate"][seed]["process_rss"][
                    "high_water_bytes"
                ]
                - by_role_seed["candidate"][seed]["process_rss"][
                    "baseline_bytes"
                ]
                - by_role_seed["reference"][seed]["process_rss"][
                    "high_water_bytes"
                ]
                + by_role_seed["reference"][seed]["process_rss"][
                    "baseline_bytes"
                ]
            ),
        }
        for seed in FINAL_SEEDS
    }
    return receipts, summary


def _latency_summaries(
    sessions: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    output = {}
    for mode in ACTUAL_INFERENCE_V5_MODES:
        output[mode] = {}
        for component in ACTUAL_INFERENCE_V5_COMPONENTS:
            candidate = np.stack(
                [row[f"{mode}__{component}__candidate"] for row in sessions]
            )
            reference = np.stack(
                [row[f"{mode}__{component}__reference"] for row in sessions]
            )
            output[mode][component] = three_way_paired_latency(
                candidate,
                reference,
                bootstrap_repetitions=ACTUAL_INFERENCE_V5_BOOTSTRAP_REPETITIONS,
                bootstrap_seed=(
                    ACTUAL_INFERENCE_V5_BOOTSTRAP_SEED
                    + ACTUAL_INFERENCE_V5_MODES.index(mode) * 10
                    + ACTUAL_INFERENCE_V5_COMPONENTS.index(component)
                ),
            ).to_dict()
    return output


def _execution_order_diagnostics(
    sessions: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Describe order effects and within-cell dispersion without gating claims."""

    output: dict[str, Any] = {}
    for mode_index, mode in enumerate(ACTUAL_INFERENCE_V5_MODES):
        output[mode] = {}
        flags = np.stack(
            [
                session_schedule(index)["candidate_first"][:, mode_index]
                for index in range(len(ACTUAL_INFERENCE_V5_SESSIONS))
            ]
        ).astype(bool)
        for component in ACTUAL_INFERENCE_V5_COMPONENTS:
            candidate = np.stack(
                [row[f"{mode}__{component}__candidate"] for row in sessions]
            )
            reference = np.stack(
                [row[f"{mode}__{component}__reference"] for row in sessions]
            )
            if candidate.shape != flags.shape or reference.shape != flags.shape:
                raise ValueError("actual order diagnostic array shape differs")
            reductions = 1 - candidate / reference
            candidate_cells = np.median(candidate, axis=-1)
            reference_cells = np.median(reference, axis=-1)
            candidate_mad = np.median(
                np.abs(candidate - candidate_cells[..., None]), axis=-1
            )
            reference_mad = np.median(
                np.abs(reference - reference_cells[..., None]), axis=-1
            )
            candidate_iqr = np.percentile(candidate, 75, axis=-1) - np.percentile(
                candidate, 25, axis=-1
            )
            reference_iqr = np.percentile(reference, 75, axis=-1) - np.percentile(
                reference, 25, axis=-1
            )
            candidate_first = float(np.median(reductions[flags]))
            reference_first = float(np.median(reductions[~flags]))
            output[mode][component] = {
                "candidate_first_median_paired_reduction": candidate_first,
                "candidate_first_trial_count": int(np.sum(flags)),
                "candidate_median_within_cell_iqr_ms": float(
                    np.median(candidate_iqr)
                ),
                "candidate_median_within_cell_mad_ms": float(
                    np.median(candidate_mad)
                ),
                "descriptive_only": True,
                "direction_reversal_by_order": bool(
                    candidate_first * reference_first < 0
                ),
                "order_reduction_difference": candidate_first - reference_first,
                "reference_first_median_paired_reduction": reference_first,
                "reference_first_trial_count": int(np.sum(~flags)),
                "reference_median_within_cell_iqr_ms": float(
                    np.median(reference_iqr)
                ),
                "reference_median_within_cell_mad_ms": float(
                    np.median(reference_mad)
                ),
            }
    return output


def _run_locked() -> int:
    plan, authorization, quality, plan_identity, summary_commit = _load_context()
    for root in (
        RUN_ROOT,
        SESSION_RECEIPT_ROOT,
        ARTIFACT_ROOT,
        MEMORY_ROOT,
        OUTPUT_PATH.parent,
    ):
        assert_workspace_path_no_symlinks(root)
    _validate_current_models(authorization=authorization, plan=plan)
    timing_sessions = []
    output_sessions = []
    reports = []
    artifacts = []
    for session_id in ACTUAL_INFERENCE_V5_SESSIONS:
        timings, outputs, report = _load_session(
            session_id,
            plan=plan,
            plan_artifact_sha256=plan_identity["sha256"],
            authorization=authorization,
        )
        timing_sessions.append(timings)
        output_sessions.append(outputs)
        reports.append(report)
        report_identity = _tracked_head_identity(
            _session_paths(session_id)["report"]
        )
        _require_ancestor(
            plan_identity["git_commit"],
            report["session_git_commit"],
            f"plan -> {session_id}",
        )
        _require_ancestor(
            report["session_git_commit"],
            report_identity["git_commit"],
            f"{session_id} timing -> receipt",
        )
        _require_ancestor(
            report_identity["git_commit"],
            summary_commit,
            f"{session_id} receipt -> summary",
        )
        artifacts.append(
            {
                name: {"path": path.as_posix(), "sha256": hash_file(path)}
                for name, path in _session_paths(session_id).items()
            }
        )
    _validate_boundary_trace_stability(reports)
    tokens = _validate_fresh_session_environments(reports, plan=plan)
    free_outputs = np.stack(
        [row["free_output_bytes"] for row in output_sessions]
    )
    free_lengths = np.stack(
        [row["free_output_lengths"] for row in output_sessions]
    )
    output_evidence = validate_free_output_bytes(free_outputs, free_lengths)
    memory_receipts, memory_summary = _load_memory(
        authorization=authorization,
        plan=plan,
        plan_git_commit=plan_identity["git_commit"],
        summary_git_commit=summary_commit,
    )
    latency = _latency_summaries(timing_sessions)
    order_diagnostics = _execution_order_diagnostics(timing_sessions)
    controlled_pass = actual_efficiency_component_pass(
        latency["controlled_replay"]["end_to_end_ms"]
    )
    free_pass = actual_efficiency_component_pass(
        latency["free_running_utf8_greedy"]["end_to_end_ms"]
    )
    quality_authorized = bool(
        quality["primary_publication_timing_authorized"]
        and plan["timing_pair"]["authorization_key"]
        == quality["primary_timing_authorization_key"]
    )
    overall = bool(quality_authorized and controlled_pass and free_pass)
    payload = {
        "claim_scope": {
            "cryptographic_one_shot_claimed": False,
            "description": (
                "pre-final outcome-sensitive timing, case-selection, and "
                "statistical logic with a tracked post-final pre-timing "
                "dual CPU-semantic and MPS distribution-aware correctness "
                "revision; timing workload, pair, cases, statistics, and "
                "efficiency gate are unchanged"
            ),
            "final_test_blind_confirmatory_claimed": False,
            "general_hardware_or_general_llm_claimed": False,
            "memory_improvement_claimed": False,
            "primary_gate_interpretation": (
                "observed median reduction at least 10%; crossed-bootstrap 95% "
                "interval excludes zero; predeclared session/seed stability gates"
            ),
            "ten_percent_confidence_lower_bound_claimed": False,
            "timing_scope": {
                "device": "Apple MPS",
                "free_output_bytes": "128-to-131 strict-UTF8-boundary bytes",
                "hardware_contract": plan["runtime_environment_contract"][
                    "hardware"
                ],
                "measured_prompts": 64,
                "model_checkpoints_per_role": 5,
                "prompt_bytes": 128,
                "replay_output_bytes": 128,
                "sessions": len(ACTUAL_INFERENCE_V5_SESSIONS),
                "ttft_and_decode": "secondary",
                "workload": "Hangul-heavy one-document-per-case byte generation",
            },
            "repetitions_treated_as_independent_samples": False,
        },
        "correctness": {
            "all_session_seed_role_checks_pass": True,
            "argmax_partition": _argmax_partition_summary(reports),
            "cpu_original_semantic_oracle_every_session": True,
            "full_causal_main_reconstruction": True,
            "full_vs_cached_entropy_router_when_required": True,
            "free_path_masked_greedy_bytes_reconstructed": True,
            "free_path_parallel_runtime_reconstructed": True,
            "mps_probability_total_variation_bound": (
                ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION
            ),
            "numerical_diagnostics": _numerical_correctness_summary(reports),
        },
        "gate": {
            "controlled_replay_end_to_end_pass": controlled_pass,
            "free_running_end_to_end_pass": free_pass,
            "matched_quality_authorization_pass": quality_authorized,
            "overall_pass": overall,
            "status": (
                "pass_matched_quality_actual_efficiency_v5r3"
                if overall
                else "fail_matched_quality_actual_efficiency_v5r3"
            ),
        },
        "kind": "phase3_inference_actual_summary_v5r3",
        "latency": latency,
        "memory": {
            "claim_role": "descriptive_only_not_a_publication_gate",
            "receipts": memory_receipts,
            "role_summary": memory_summary,
        },
        "order_and_dispersion_diagnostics": order_diagnostics,
        "output_validity": output_evidence,
        "plan_artifact": plan_identity,
        "plan_sha256": plan["plan_sha256"],
        "protocol_version": 5,
        "protocol_revision": ACTUAL_INFERENCE_V5_PROTOCOL_REVISION,
        "schema_version": 6,
        "session_artifacts": artifacts,
        "session_process_start_tokens": tokens,
        "summary_base_git_commit": summary_commit,
        "summary_path": ACTUAL_INFERENCE_V5_SUMMARY_PATH,
        "timing_pair": plan["timing_pair"],
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    publish_no_clobber(OUTPUT_PATH, _json_bytes(payload))
    if _git_commit() != summary_commit or not _post_publish_status_is_clean():
        raise RuntimeError("repository changed while sealing actual summary")
    print("sealed actual-inference v5 summary; inspect only after commit", flush=True)
    return 0


def run() -> int:
    with _exclusive_evidence_snapshot():
        if OUTPUT_PATH.exists():
            return _verify_existing_summary()
        if _tracked_history_exists(OUTPUT_PATH):
            raise ValueError("deleted actual summary forbids resealing")
        return _run_locked()


if __name__ == "__main__":
    raise SystemExit(run())
