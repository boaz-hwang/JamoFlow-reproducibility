#!/usr/bin/env python3
"""Measure trained fresh-v2 16K, 2K, and 8K generation on Apple MPS."""

from __future__ import annotations

import gc
import hashlib
import io
import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from benchmark_fresh_vocabulary_actual import (
    RoleBundle,
    generate_free_untimed,
    run_trial,
    verify_sequence,
)
from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_actual_core import (
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    balanced_role_order,
    correctness_pass,
)
from fresh_vocabulary_16k_actual_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_BY_ROLE,
    MAXIMUM_FREE_TOKENS,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    RUNTIME_REPORT_PATH,
    TIMING_PATH,
    VOCABULARY_BY_ROLE,
    array_sha256,
    build_role_model,
    canonical_sha256,
    encode_raw,
    hash_file,
    json_bytes,
    read_plan_json,
    reconstruct_cases,
    validate_plan,
)
from fresh_vocabulary_actual_core import WARMUP_CASES
from vocabulary_transfer_probe_core import state_mapping_sha256

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.utf8 import compile_strict_utf8_token_transitions


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _snapshot(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout": result.stdout,
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _snapshot(("pmset", "-g", "batt")),
        "settings": _snapshot(("pmset", "-g", "custom")),
        "thermal": _snapshot(("pmset", "-g", "therm")),
    }


def _publish(path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": array_sha256(value),
    }


def _require_plan_commit() -> tuple[str, dict[str, Any]]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-16k actual benchmark requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    plan_path = PLAN_PATH.relative_to(ROOT).as_posix()
    if _git("log", "-1", "--format=%H", "--", plan_path) != commit:
        raise RuntimeError("fresh-16k actual plan must be current HEAD")
    plan = read_plan_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("fresh-16k actual plan parent differs")
    return commit, plan


def _compile_masks(transitions) -> tuple[torch.Tensor, ...]:
    masks = []
    for allowed in transitions.allowed_token_ids:
        mask = torch.zeros(
            transitions.vocabulary_size,
            dtype=torch.bool,
            device="mps",
        )
        mask[torch.tensor(allowed, dtype=torch.long, device="mps")] = True
        masks.append(mask)
    torch.mps.synchronize()
    return tuple(masks)


def load_bundles(plan: Mapping[str, Any]) -> dict[str, RoleBundle]:
    loaded = load_tokenizers()
    bundles: dict[str, RoleBundle] = {}
    for role in ROLES:
        model = build_role_model(role)
        state = torch.load(
            CHECKPOINT_BY_ROLE[role], map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        identity = plan["models"][role]
        if (
            hash_file(CHECKPOINT_BY_ROLE[role])
            != identity["checkpoint_artifact_sha256"]
            or state_mapping_sha256(model.state_dict())
            != identity["checkpoint_state_sha256"]
            or sum(parameter.numel() for parameter in model.parameters())
            != identity["parameter_count"]
        ):
            raise RuntimeError("fresh-16k actual loaded model identity differs")
        size = VOCABULARY_BY_ROLE[role]
        tokenizer, token_bytes = loaded[size]
        transitions = compile_strict_utf8_token_transitions(token_bytes)
        contract = plan["tokenizer_runtime"][role]["strict_utf8_transitions"]
        if (
            transitions.token_bytes_sha256 != contract["token_bytes_sha256"]
            or transitions.transition_table_sha256
            != contract["transition_table_sha256"]
            or transitions.maximum_token_bytes != contract["maximum_token_bytes"]
        ):
            raise RuntimeError("fresh-16k actual token transition identity differs")
        model = model.eval().to("mps")
        bundles[role] = RoleBundle(
            role=role,
            model=model,
            tokenizer=tokenizer,
            token_bytes=token_bytes,
            transitions=transitions,
            masks=_compile_masks(transitions),
        )
    return bundles


def correctness_replay(
    bundles: Mapping[str, RoleBundle],
    prompts: np.ndarray,
    continuations: np.ndarray,
    *,
    expected_free_ids: Mapping[str, Sequence[Sequence[int]]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    if prompts.shape != continuations.shape or prompts.ndim != 2:
        raise ValueError("fresh-16k actual correctness case arrays differ")
    case_count = int(prompts.shape[0])
    if expected_free_ids is not None and set(expected_free_ids) != set(ROLES):
        raise ValueError("fresh-16k actual expected free role set differs")
    if expected_free_ids is not None and any(
        len(expected_free_ids[role]) != case_count for role in ROLES
    ):
        raise ValueError("fresh-16k actual expected free case count differs")
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ROLES:
        bundle = bundles[role]
        rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        for case_index, (prompt, continuation) in enumerate(
            zip(prompts, continuations, strict=True)
        ):
            prompt_ids = encode_raw(bytes(prompt), bundle.tokenizer, bundle.token_bytes)
            controlled_ids = encode_raw(
                bytes(continuation), bundle.tokenizer, bundle.token_bytes
            )
            controlled = verify_sequence(bundle, prompt_ids, controlled_ids)
            controlled["trace_contract_exact"] = True
            rows["controlled_replay"].append(controlled)

            free_ids, _ = generate_free_untimed(bundle, prompt_ids)
            if expected_free_ids is not None:
                expected = tuple(
                    int(value) for value in expected_free_ids[role][case_index]
                )
                if free_ids != expected:
                    raise AssertionError(
                        "fresh-16k actual measured free greedy trace differs"
                    )
            free = verify_sequence(bundle, prompt_ids, free_ids)
            free["trace_contract_exact"] = True
            rows["free_running_utf8_greedy"].append(free)
        output[role] = {}
        for mode in MODES:
            mode_rows = rows[mode]
            comparisons = sum(int(row["comparisons"]) for row in mode_rows)
            argmax = sum(int(row["argmax_exact"]) for row in mode_rows)
            output[role][mode] = {
                "cases": len(mode_rows),
                "comparisons": comparisons,
                "argmax_comparisons": comparisons,
                "argmax_exact": argmax,
                "trace_contract_exact": True,
                "maximum_normalized_tolerance_ratio": max(
                    float(row["maximum_normalized_tolerance_ratio"])
                    for row in mode_rows
                ),
                "pass": bool(
                    argmax == comparisons and all(row["pass"] for row in mode_rows)
                ),
            }
    if not correctness_pass(output, expected_cases=case_count):
        raise AssertionError("fresh-16k actual correctness aggregate differs")
    return output


def preflight_bundles(plan: Mapping[str, Any]) -> None:
    """Loss- and timing-silent feasibility check used before plan sealing."""

    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("fresh-16k actual preflight cases differ")
    with publication_mps_exclusive():
        bundles = load_bundles(plan)
        correctness_replay(
            bundles,
            prompts[:1],
            continuations[:1],
        )
        for bundle in bundles.values():
            bundle.model.to("cpu")
        del bundles
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()


def main() -> None:
    commit, plan = _require_plan_commit()
    if any(
        path.exists()
        for path in (ACTIVE_PATH, RUNTIME_REPORT_PATH, TIMING_PATH, OUTPUT_PATH)
    ):
        raise FileExistsError("fresh-16k actual runtime namespace is not empty")
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("fresh-16k actual cases changed after sealing")
    _publish(
        ACTIVE_PATH,
        json_bytes(
            {
                "git_commit": commit,
                "plan_artifact_sha256": hash_file(PLAN_PATH),
            }
        ),
    )
    with publication_mps_exclusive():
        start_state = _session_state()
        if not timing_environment_eligible(start_state):
            raise RuntimeError(
                "fresh-16k actual timing environment is ineligible at start"
            )
        bundles = load_bundles(plan)
        warmup_correctness = correctness_replay(
            bundles,
            prompts[:WARMUP_CASES],
            continuations[:WARMUP_CASES],
        )
        payloads: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        for role in ROLES:
            bundle = bundles[role]
            payloads[role] = [
                (
                    encode_raw(bytes(prompt), bundle.tokenizer, bundle.token_bytes),
                    encode_raw(
                        bytes(continuation), bundle.tokenizer, bundle.token_bytes
                    ),
                )
                for prompt, continuation in zip(prompts, continuations, strict=True)
            ]
        for case_index in range(WARMUP_CASES):
            for mode in MODES:
                for role in ROLES:
                    prompt_ids, continuation_ids = payloads[role][case_index]
                    run_trial(
                        bundles[role],
                        bytes(prompts[case_index]),
                        prompt_ids,
                        continuation_ids,
                        bytes(continuations[case_index]),
                        mode,
                    )

        shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
        arrays: dict[str, np.ndarray] = {
            name: np.empty(shape, dtype=np.float64) for name in TIMING_COMPONENTS
        }
        arrays["output_token_count"] = np.empty(shape, dtype=np.int16)
        arrays["output_raw_byte_count"] = np.empty(shape, dtype=np.int16)
        maximum_output_bytes = max(
            int(
                plan["tokenizer_runtime"][role]["strict_utf8_transitions"][
                    "maximum_free_output_bytes"
                ]
            )
            for role in ROLES
        )
        arrays["free_token_ids"] = np.full(
            (MEASURED_CASES, REPETITIONS, len(ROLES), MAXIMUM_FREE_TOKENS),
            -1,
            dtype=np.int32,
        )
        arrays["free_output_bytes"] = np.zeros(
            (MEASURED_CASES, REPETITIONS, len(ROLES), maximum_output_bytes),
            dtype=np.uint8,
        )
        arrays["free_output_lengths"] = np.zeros(
            (MEASURED_CASES, REPETITIONS, len(ROLES)), dtype=np.int16
        )
        for case_index in range(MEASURED_CASES):
            source_index = WARMUP_CASES + case_index
            prompt = bytes(prompts[source_index])
            continuation = bytes(continuations[source_index])
            for repetition in range(REPETITIONS):
                for mode_index, mode in enumerate(MODES):
                    for role_index in balanced_role_order(
                        case_index, repetition, mode_index
                    ):
                        role = ROLES[role_index]
                        prompt_ids, continuation_ids = payloads[role][source_index]
                        metrics, output_ids, output = run_trial(
                            bundles[role],
                            prompt,
                            prompt_ids,
                            continuation_ids,
                            continuation,
                            mode,
                        )
                        for name, value in metrics.items():
                            arrays[name][
                                mode_index, case_index, repetition, role_index
                            ] = value
                        arrays["output_token_count"][
                            mode_index, case_index, repetition, role_index
                        ] = len(output_ids)
                        arrays["output_raw_byte_count"][
                            mode_index, case_index, repetition, role_index
                        ] = len(output)
                        if mode == "free_running_utf8_greedy":
                            arrays["free_token_ids"][
                                case_index, repetition, role_index, : len(output_ids)
                            ] = np.asarray(output_ids, dtype=np.int32)
                            arrays["free_output_bytes"][
                                case_index, repetition, role_index, : len(output)
                            ] = np.frombuffer(output, dtype=np.uint8)
                            arrays["free_output_lengths"][
                                case_index, repetition, role_index
                            ] = len(output)
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError(
                "fresh-16k actual timing environment is ineligible at end"
            )
        for bundle in bundles.values():
            bundle.model.to("cpu")
        del bundles
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    timing_payload = _npz_bytes(arrays)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_actual_one_seed_runtime_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "cases": cases,
        "models": plan["models"],
        "tokenizer_runtime": plan["tokenizer_runtime"],
        "warmup_correctness": warmup_correctness,
        "arrays": {name: _array_descriptor(value) for name, value in arrays.items()},
        "timing_artifact_sha256": hashlib.sha256(timing_payload).hexdigest(),
        "session_state": {"start": start_state, "end": end_state},
        "timed_scope": plan["experiment"]["timed_scope"],
    }
    report["report_sha256"] = canonical_sha256(report)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during fresh-16k actual benchmark")
    _publish(TIMING_PATH, timing_payload)
    _publish(RUNTIME_REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed after fresh-16k actual benchmark")
    print("status=fresh_vocabulary_16k_actual_timing_complete")


if __name__ == "__main__":
    main()
