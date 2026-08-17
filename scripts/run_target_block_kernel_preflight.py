#!/usr/bin/env python3
"""Measure the exact inference-mode perfect-draft W72 block-kernel bound."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

from scripts.incremental_block_kernel import IncrementalBlockBltDecoder
from scripts.target_block_kernel_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    MAXIMUM_CONTINUATION_BYTES,
    MICRO_CASES,
    MICRO_REPETITIONS,
    MICRO_STRATA,
    MINIMUM_CONTINUATION_BYTES,
    MODES,
    PROMPT_BYTES,
    PROTOCOL_ID,
    WHOLE_CASES,
    WHOLE_REPETITIONS,
    perfect_hangul_groups,
    summarize_block_kernel,
)
from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import (
    IncrementalBltDecoder,
    IncrementalStructuralSelector,
    structural_prefix_boundaries,
)
from jamoflow.inference_actual_runtime_v5 import (
    ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL,
    load_actual_model,
    release_actual_model,
)
from jamoflow.inference_actual_v5 import array_sha256
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    SELECTION_LOCK_PATH,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import research_versions
from jamoflow.neural_training import synchronize
from jamoflow.utf8 import prefix_boundary_mask


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/target-block-kernel-v2.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
AUTHORIZATION_PATH = ROOT / FINAL_AUTHORIZATION_PATH
SELECTION_PATH = ROOT / SELECTION_LOCK_PATH
ACCEPTANCE_PATH = ROOT / "results/hangul-draft-acceptance-v1/summary.json"
INVALIDATION_PATH = ROOT / "results/target-block-kernel-v1/invalidation.json"
CASE_PATH = ROOT / "artifacts/target-block-kernel-v2/cases.npz"
RAW_PATH = ROOT / "artifacts/target-block-kernel-v2/timings.npz"
OUTPUT_PATH = ROOT / "results/target-block-kernel-v2/summary.json"


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


def _publish_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _clean_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("block-kernel preflight requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("block-kernel preflight requires a Git commit")
    return commit


def _require_ac_power() -> str:
    state = _command("pmset", "-g", "batt")
    if "Now drawing from 'AC Power'" not in state:
        raise RuntimeError("block-kernel preflight requires AC power")
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _validate_plan(plan: Mapping[str, Any], commit: str) -> None:
    if (
        set(plan)
        != {
            "cases",
            "decision_rule",
            "fixed_independent_head",
            "implementation_sha256",
            "input",
            "kind",
            "model",
            "output",
            "protocol_id",
            "schema_version",
            "status",
            "threat_model",
            "timing",
        }
        or plan.get("schema_version") != 1
        or plan.get("kind") != "target_block_kernel_plan_v2"
        or plan.get("protocol_id") != PROTOCOL_ID
        or tuple(plan["cases"]["micro_strata"]) != MICRO_STRATA
        or tuple(plan["timing"]["modes"]) != MODES
        or int(plan["cases"]["micro_cases_per_stratum"]) != MICRO_CASES
        or int(plan["timing"]["micro_repetitions"]) != MICRO_REPETITIONS
        or int(plan["cases"]["whole_cases"]) != WHOLE_CASES
        or int(plan["timing"]["whole_repetitions"]) != WHOLE_REPETITIONS
        or int(plan["decision_rule"]["bootstrap_repetitions"])
        != BOOTSTRAP_REPETITIONS
        or int(plan["decision_rule"]["bootstrap_seed"]) != BOOTSTRAP_SEED
        or int(plan["cases"]["prompt_bytes"]) != PROMPT_BYTES
        or int(plan["cases"]["block_bytes"]) != 3
        or int(plan["cases"]["micro_follow_bytes"]) != 1
        or tuple(plan["cases"]["whole_continuation_bytes"])
        != (MINIMUM_CONTINUATION_BYTES, MAXIMUM_CONTINUATION_BYTES)
        or int(plan["model"]["patching_horizon"]) != 512
        or plan["timing"]["device"] != "mps"
        or plan["timing"]["torch_inference_mode"] is not True
        or plan["threat_model"]["torch_inference_mode_required"] is not True
        or float(plan["timing"]["correctness_atol"])
        != ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL
        or float(plan["timing"]["correctness_rtol"])
        != ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL
    ):
        raise ValueError("block-kernel plan schema differs")
    for relative, expected in plan["implementation_sha256"].items():
        if hash_file(ROOT / relative) != expected:
            raise ValueError(f"block-kernel implementation differs: {relative}")
    if (
        hash_file(SOURCE_PATH) != plan["input"]["source_sha256"]
        or hash_file(AUTHORIZATION_PATH)
        != plan["model"]["authorization_artifact_sha256"]
        or hash_file(SELECTION_PATH)
        != plan["model"]["selection_artifact_sha256"]
        or hash_file(ACCEPTANCE_PATH)
        != plan["fixed_independent_head"]["acceptance_artifact_sha256"]
        or hash_file(INVALIDATION_PATH)
        != plan["threat_model"]["invalidated_v1_artifact_sha256"]
    ):
        raise ValueError("block-kernel upstream artifact differs")
    if any(path.exists() for path in (CASE_PATH, RAW_PATH, OUTPUT_PATH)):
        raise FileExistsError("block-kernel output already exists")
    if _command("git", "rev-parse", "HEAD") != commit:
        raise RuntimeError("block-kernel plan verification changed HEAD")


def _hangul_share(raw: bytes) -> float:
    text = raw.decode("utf-8")
    return sum(0xAC00 <= ord(char) <= 0xD7A3 for char in text) / max(1, len(text))


def _rank(domain: bytes, start: int, payload: bytes) -> bytes:
    return hashlib.sha256(domain + start.to_bytes(8, "big") + payload).digest()


def _nonoverlap_select(
    candidates: list[tuple[bytes, int, bytes, bytes]],
    *,
    count: int,
    total_length: int,
    occupied: list[tuple[int, int]] | None = None,
) -> list[tuple[int, bytes, bytes]]:
    selected: list[tuple[int, bytes, bytes]] = []
    used = [] if occupied is None else list(occupied)
    for _, start, prompt, continuation in sorted(candidates):
        end = start + total_length
        if any(not (end <= left or right <= start) for left, right in used):
            continue
        selected.append((start, prompt, continuation))
        used.append((start, end))
        if len(selected) == count:
            return selected
    raise ValueError("block-kernel case selection lacks enough non-overlap rows")


def _micro_cases(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.frombuffer(prefix_boundary_mask(data), dtype=np.uint8)
    by_stratum: dict[int, list[tuple[bytes, int, bytes, bytes]]] = {0: [], 1: []}
    domain = b"JamoFlow/target-block-micro/v1\0"
    for start_value in np.flatnonzero(boundaries[: -(PROMPT_BYTES + 4)]):
        start = int(start_value)
        prompt_end = start + PROMPT_BYTES
        if not boundaries[prompt_end]:
            continue
        prompt = data[start:prompt_end]
        block_and_follow = data[prompt_end : prompt_end + 4]
        try:
            scalar = block_and_follow[:3].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if len(scalar) != 1 or not 0xAC00 <= ord(scalar) <= 0xD7A3:
            continue
        before = structural_prefix_boundaries(
            prompt,
            "causal_whitespace_grid",
            horizon=512,
            patch_count=72,
            fixed_stride=6,
        )
        after = structural_prefix_boundaries(
            prompt + block_and_follow[:3],
            "causal_whitespace_grid",
            horizon=512,
            patch_count=72,
            fixed_stride=6,
        )
        new_patches = len(after) - len(before)
        if new_patches not in (0, 1):
            continue
        payload = prompt + block_and_follow
        by_stratum[new_patches].append(
            (_rank(domain, start, payload), start, prompt, block_and_follow)
        )
    selected: list[tuple[int, bytes, bytes]] = []
    occupied: list[tuple[int, int]] = []
    labels: list[int] = []
    for stratum in range(2):
        rows = _nonoverlap_select(
            by_stratum[stratum],
            count=MICRO_CASES,
            total_length=PROMPT_BYTES + 4,
            occupied=occupied,
        )
        selected.extend(rows)
        labels.extend([stratum] * MICRO_CASES)
        occupied.extend((start, start + PROMPT_BYTES + 4) for start, _, _ in rows)
    prompts = np.asarray(
        [np.frombuffer(row[1], dtype=np.uint8) for row in selected], dtype=np.uint8
    )
    values = np.asarray(
        [np.frombuffer(row[2], dtype=np.uint8) for row in selected], dtype=np.uint8
    )
    return prompts, values, np.asarray(labels, dtype=np.int64)


def _whole_cases(
    data: bytes,
    *,
    minimum_hangul_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boundaries = np.frombuffer(prefix_boundary_mask(data), dtype=np.uint8)
    candidates: list[tuple[bytes, int, bytes, bytes]] = []
    domain = b"JamoFlow/target-block-whole/v1\0"
    for start_value in np.flatnonzero(
        boundaries[: -(PROMPT_BYTES + MAXIMUM_CONTINUATION_BYTES)]
    ):
        start = int(start_value)
        prompt_end = start + PROMPT_BYTES
        if not boundaries[prompt_end]:
            continue
        prompt = data[start:prompt_end]
        try:
            if _hangul_share(prompt) < minimum_hangul_share:
                continue
        except UnicodeDecodeError:
            continue
        end = None
        for length in range(MINIMUM_CONTINUATION_BYTES, MAXIMUM_CONTINUATION_BYTES + 1):
            if boundaries[prompt_end + length]:
                end = prompt_end + length
                break
        if end is None:
            continue
        continuation = data[prompt_end:end]
        payload = prompt + continuation
        candidates.append((_rank(domain, start, payload), start, prompt, continuation))
    rows = _nonoverlap_select(
        candidates,
        count=WHOLE_CASES,
        total_length=PROMPT_BYTES + MAXIMUM_CONTINUATION_BYTES,
    )
    prompts = np.asarray(
        [np.frombuffer(row[1], dtype=np.uint8) for row in rows], dtype=np.uint8
    )
    lengths = np.asarray([len(row[2]) for row in rows], dtype=np.int64)
    continuations = np.zeros(
        (WHOLE_CASES, MAXIMUM_CONTINUATION_BYTES), dtype=np.uint8
    )
    starts = np.asarray([row[0] for row in rows], dtype=np.int64)
    for index, (_, _, continuation) in enumerate(rows):
        continuations[index, : len(continuation)] = np.frombuffer(
            continuation, dtype=np.uint8
        )
    return prompts, continuations, lengths, starts


def _runtime(bundle: Any, *, block: bool) -> Any:
    cls = IncrementalBlockBltDecoder if block else IncrementalBltDecoder
    return cls(
        bundle.model,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )


def _compare_logits(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, int]:
    if left.shape != right.shape or not bool(torch.isfinite(left).all()) or not bool(
        torch.isfinite(right).all()
    ):
        raise AssertionError("block-kernel comparison tensor differs")
    torch.testing.assert_close(
        left,
        right,
        rtol=ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL,
        atol=ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL,
    )
    if not torch.equal(left.argmax(dim=-1), right.argmax(dim=-1)):
        raise AssertionError("block-kernel argmax differs")
    error = (left - right).abs()
    tolerance = ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL + (
        ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL * right.abs()
    )
    return (
        float(error.max().cpu()),
        float((error / tolerance).max().cpu()),
        int(left.shape[0]),
    )


def _mode_order(*indices: int) -> tuple[int, int]:
    # Every fixed case x repetition grid has an even number of cells, so parity
    # gives exact role-order balance rather than approximate random balance.
    return (0, 1) if sum(indices) % 2 == 0 else (1, 0)


def _require_inference_mode() -> None:
    if not torch.is_inference_mode_enabled():
        raise RuntimeError("target block-kernel timing requires torch.inference_mode")


def _measure_micro(
    bundle: Any,
    prompts: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require_inference_mode()
    timings = np.empty(
        (len(MICRO_STRATA), MICRO_CASES, MICRO_REPETITIONS, len(MODES)),
        dtype=np.float64,
    )
    maximum_absolute = 0.0
    maximum_normalized = 0.0
    argmax_comparisons = 0
    cache_comparisons = 0
    for stratum in range(len(MICRO_STRATA)):
        for case in range(MICRO_CASES):
            row = stratum * MICRO_CASES + case
            prompt = bytes(prompts[row])
            block_values = bytes(values[row, :3])
            follow = int(values[row, 3])
            for repetition in range(MICRO_REPETITIONS):
                runtimes: dict[int, Any] = {}
                logits: dict[int, torch.Tensor] = {}
                for mode in _mode_order(stratum, case, repetition):
                    runtime = _runtime(bundle, block=mode == 1)
                    runtime.prefill_parallel(prompt)
                    synchronize("mps")
                    started = time.perf_counter_ns()
                    if mode == 0:
                        result = torch.cat(
                            [runtime.consume(value) for value in block_values], dim=0
                        )
                    else:
                        result = runtime.consume_block(block_values)
                    synchronize("mps")
                    timings[stratum, case, repetition, mode] = (
                        time.perf_counter_ns() - started
                    ) / 1_000_000
                    runtimes[mode] = runtime
                    logits[mode] = result
                absolute, normalized, count = _compare_logits(logits[1], logits[0])
                maximum_absolute = max(maximum_absolute, absolute)
                maximum_normalized = max(maximum_normalized, normalized)
                argmax_comparisons += count
                if runtimes[0].diagnostics != runtimes[1].diagnostics:
                    raise AssertionError("block-kernel micro cache differs")
                cache_comparisons += 1
                left = runtimes[1].consume(follow)
                right = runtimes[0].consume(follow)
                absolute, normalized, count = _compare_logits(left, right)
                maximum_absolute = max(maximum_absolute, absolute)
                maximum_normalized = max(maximum_normalized, normalized)
                argmax_comparisons += count
                if runtimes[0].diagnostics != runtimes[1].diagnostics:
                    raise AssertionError("block-kernel propagated cache differs")
                cache_comparisons += 1
    return timings, {
        "argmax_comparisons": argmax_comparisons,
        "cache_comparisons": cache_comparisons,
        "maximum_absolute_logit_error": maximum_absolute,
        "maximum_normalized_tolerance_ratio": maximum_normalized,
        "pass": True,
    }


def _group_boundary_counts(prompt: bytes, groups: tuple[bytes, ...]) -> tuple[int, int]:
    selector = IncrementalStructuralSelector(
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )
    for value in prompt:
        selector.consume(value)
    hangul_blocks = 0
    boundary_blocks = 0
    for group in groups:
        before = len(selector.boundaries)
        for value in group:
            selector.consume(value)
        if len(group) == 3:
            hangul_blocks += 1
            boundary_blocks += int(len(selector.boundaries) > before)
    return hangul_blocks, boundary_blocks


def _measure_whole(
    bundle: Any,
    prompts: np.ndarray,
    continuations: np.ndarray,
    lengths: np.ndarray,
    correctness: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require_inference_mode()
    timings = np.empty((WHOLE_CASES, WHOLE_REPETITIONS, len(MODES)), dtype=np.float64)
    hangul_blocks = np.empty(WHOLE_CASES, dtype=np.int64)
    boundary_blocks = np.empty(WHOLE_CASES, dtype=np.int64)
    for case in range(WHOLE_CASES):
        prompt = bytes(prompts[case])
        # Like generation timing, the final emitted byte is not consumed.
        observed = bytes(continuations[case, : int(lengths[case]) - 1])
        groups = perfect_hangul_groups(observed)
        hangul_blocks[case], boundary_blocks[case] = _group_boundary_counts(
            prompt, groups
        )
        for repetition in range(WHOLE_REPETITIONS):
            runtimes: dict[int, Any] = {}
            logits: dict[int, torch.Tensor] = {}
            for mode in _mode_order(99, case, repetition):
                synchronize("mps")
                started = time.perf_counter_ns()
                runtime = _runtime(bundle, block=mode == 1)
                result = runtime.prefill_parallel(prompt)
                if mode == 0:
                    for value in observed:
                        result = runtime.consume(value)
                else:
                    for group in groups:
                        if len(group) == 3:
                            result = runtime.consume_block(group)[-1:, :]
                        else:
                            result = runtime.consume(group[0])
                synchronize("mps")
                timings[case, repetition, mode] = (
                    time.perf_counter_ns() - started
                ) / 1_000_000
                runtimes[mode] = runtime
                logits[mode] = result
            absolute, normalized, count = _compare_logits(logits[1], logits[0])
            correctness["maximum_absolute_logit_error"] = max(
                correctness["maximum_absolute_logit_error"], absolute
            )
            correctness["maximum_normalized_tolerance_ratio"] = max(
                correctness["maximum_normalized_tolerance_ratio"], normalized
            )
            correctness["argmax_comparisons"] += count
            if runtimes[0].diagnostics != runtimes[1].diagnostics:
                raise AssertionError("block-kernel whole cache differs")
            correctness["cache_comparisons"] += 1

        # Timing stores only the final row so that tensor collection does not
        # contaminate the hot path. This separate oracle compares every logit
        # position and the fully propagated caches over the same continuation.
        sequential = _runtime(bundle, block=False)
        blocked = _runtime(bundle, block=True)
        sequential.prefill_parallel(prompt)
        blocked.prefill_parallel(prompt)
        sequential_rows: list[torch.Tensor] = []
        blocked_rows: list[torch.Tensor] = []
        for group in groups:
            sequential_rows.extend(sequential.consume(value) for value in group)
            if len(group) == 3:
                blocked_rows.extend(blocked.consume_block(group).split(1, dim=0))
            else:
                blocked_rows.append(blocked.consume(group[0]))
        absolute, normalized, count = _compare_logits(
            torch.cat(blocked_rows, dim=0),
            torch.cat(sequential_rows, dim=0),
        )
        correctness["maximum_absolute_logit_error"] = max(
            correctness["maximum_absolute_logit_error"], absolute
        )
        correctness["maximum_normalized_tolerance_ratio"] = max(
            correctness["maximum_normalized_tolerance_ratio"], normalized
        )
        correctness["argmax_comparisons"] += count
        if sequential.diagnostics != blocked.diagnostics:
            raise AssertionError("block-kernel whole oracle cache differs")
        correctness["cache_comparisons"] += 1
    return timings, hangul_blocks, boundary_blocks


def run() -> None:
    commit = _clean_commit()
    plan = _read_json(PLAN_PATH)
    _validate_plan(plan, commit)
    selection = _read_json(SELECTION_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)
    acceptance = _read_json(ACCEPTANCE_PATH)
    validate_selection_lock_v2(selection)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection,
    )
    candidate = next(
        model for model in authorization["models"] if model["artifact_role"] == "candidate"
    )
    if (
        candidate["identity_sha256"] != plan["model"]["candidate_identity_sha256"]
        or candidate["seeds"][str(plan["model"]["seed"])]["checkpoint"]
        != plan["model"]["checkpoint"]
    ):
        raise ValueError("block-kernel target identity differs")
    fixed = plan["fixed_independent_head"]
    run_key = f"generic_independent_utf8__{fixed['head_seed']}"
    source_row = acceptance["per_training_run"][run_key]
    if (
        source_row["free_target_calibration"]["first_continuation_acceptance"]
        != fixed["first_continuation_acceptance"]
        or source_row["free_target_calibration"]["complete_pair_acceptance"]
        != fixed["complete_pair_acceptance"]
        or source_row["isolated_head_latency_ms"]["median"]
        != fixed["head_latency_ms"]
    ):
        raise ValueError("block-kernel fixed head opportunity differs")
    power_sha256 = _require_ac_power()
    stream = build_neural_stream(
        SOURCE_PATH,
        "ko",
        "calibration",
        int(plan["input"]["byte_limit"]),
        512,
    )
    if (
        len(stream.data) != plan["input"]["stream_bytes"]
        or hashlib.sha256(stream.data).hexdigest()
        != plan["input"]["stream_sha256"]
    ):
        raise ValueError("block-kernel calibration stream differs")
    micro_prompts, micro_values, micro_strata = _micro_cases(stream.data)
    whole_prompts, continuations, lengths, whole_starts = _whole_cases(
        stream.data,
        minimum_hangul_share=float(plan["cases"]["minimum_prompt_hangul_share"]),
    )
    cases = {
        "micro_prompts": micro_prompts,
        "micro_values": micro_values,
        "micro_strata": micro_strata,
        "whole_prompts": whole_prompts,
        "whole_continuations": continuations,
        "whole_lengths": lengths,
        "whole_starts": whole_starts,
    }
    started = time.time()
    with publication_mps_exclusive():
        bundle = load_actual_model(
            role="candidate",
            identity=candidate,
            seed=int(plan["model"]["seed"]),
            device="mps",
        )
        with torch.inference_mode():
            _require_inference_mode()
            micro_ms, correctness = _measure_micro(
                bundle,
                micro_prompts,
                micro_values,
            )
            whole_ms, whole_blocks, boundary_blocks = _measure_whole(
                bundle,
                whole_prompts,
                continuations,
                lengths,
                correctness,
            )
        release_actual_model(bundle)
    raw = {
        "micro_ms": micro_ms,
        "whole_ms": whole_ms,
        "whole_hangul_blocks": whole_blocks,
        "whole_boundary_blocks": boundary_blocks,
    }
    aggregate = summarize_block_kernel(
        **raw,
        correctness=correctness,
        independent_first_acceptance=float(fixed["first_continuation_acceptance"]),
        independent_pair_acceptance=float(fixed["complete_pair_acceptance"]),
        independent_head_latency_ms=float(fixed["head_latency_ms"]),
        minimum_micro_reduction=float(
            plan["decision_rule"]["minimum_micro_target_block_reduction"]
        ),
        minimum_micro_lower_bound=float(
            plan["decision_rule"]["minimum_micro_target_block_lower_bound"]
        ),
        minimum_perfect_whole_reduction=float(
            plan["decision_rule"]["minimum_perfect_hangul_whole_reduction"]
        ),
        minimum_perfect_whole_lower_bound=float(
            plan["decision_rule"]["minimum_perfect_hangul_whole_lower_bound"]
        ),
        minimum_projected_reduction=float(
            plan["decision_rule"]["minimum_independent_projected_reduction"]
        ),
        minimum_projected_lower_bound=float(
            plan["decision_rule"]["minimum_independent_projected_lower_bound"]
        ),
    )
    case_bytes = _npz_bytes(cases)
    raw_bytes = _npz_bytes(raw)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "target_block_kernel_summary_v2",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "full_speculative_runtime_authorized"
            if aggregate["gates"]["full_speculative_runtime_authorized"]
            else "multi_byte_branch_stopped"
        ),
        "provenance": {
            "git_commit": commit,
            "plan_path": PLAN_PATH.relative_to(ROOT).as_posix(),
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "authorization_artifact_sha256": hash_file(AUTHORIZATION_PATH),
            "acceptance_artifact_sha256": hash_file(ACCEPTANCE_PATH),
            "candidate_identity_sha256": candidate["identity_sha256"],
            "power_snapshot_sha256": power_sha256,
            "runtime": {
                **research_versions(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch_inference_mode": True,
            },
        },
        "case_context": {
            "artifact_path": CASE_PATH.relative_to(ROOT).as_posix(),
            "artifact_sha256": hashlib.sha256(case_bytes).hexdigest(),
            "array_sha256": {key: array_sha256(value) for key, value in cases.items()},
        },
        "raw_timing": {
            "artifact_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "artifact_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "array_sha256": {key: array_sha256(value) for key, value in raw.items()},
        },
        "aggregate": aggregate,
        "elapsed_seconds": float(time.time() - started),
        "claim_boundary": {
            "perfect_draft_upper_bound": True,
            "actual_speculative_rollback_implemented": False,
            "head_executed_inside_whole_timing": False,
            "projected_head_cost_included": True,
            "quality_or_final_test_claimed": False,
            "pass_authorizes": "calibration-only exact speculative rollback prototype",
        },
    }
    summary["summary_sha256"] = hashlib.sha256(_json_bytes(summary)).hexdigest()
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("block-kernel preflight changed tracked repository state")
    _publish_no_clobber(CASE_PATH, case_bytes)
    _publish_no_clobber(RAW_PATH, raw_bytes)
    _publish_no_clobber(OUTPUT_PATH, _json_bytes(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "micro_reduction": aggregate["weighted_micro"][
                    "target_block_reduction"
                ],
                "perfect_whole_reduction": aggregate["perfect_hangul_whole_path"][
                    "reduction"
                ],
                "projected_reduction": aggregate["fixed_independent_projection"][
                    "projected_reduction"
                ],
                "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
