#!/usr/bin/env python3
"""Measure a perfect-draft target-block upper bound on the trained 16K model."""

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
from benchmark_fresh_vocabulary_16k_actual import _compile_masks
from benchmark_fresh_vocabulary_actual import (
    RoleBundle,
    generate_free_untimed,
    run_trial,
    verify_sequence,
)
from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_actual_protocol import encode_raw, json_bytes
from fresh_vocabulary_16k_block_core import (
    BLOCK_SIZE_BY_ROLE,
    MEASURED_CASES,
    MODES,
    REPETITIONS,
    ROLES,
    TIMING_COMPONENTS,
    balanced_role_order,
    correctness_pass,
)
from fresh_vocabulary_16k_block_protocol import (
    ACTIVE_PATH,
    ARTIFACT_ROOT,
    MAXIMUM_FREE_TOKENS,
    MPS_ATOL,
    MPS_RTOL,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    RUNTIME_REPORT_PATH,
    TARGET_CHECKPOINT_PATH,
    TARGET_VOCABULARY_SIZE,
    TIMING_PATH,
    array_sha256,
    canonical_sha256,
    hash_file,
    read_json,
    reconstruct_cases,
    validate_plan,
)
from fresh_vocabulary_16k_block_runtime import (
    run_perfect_block_trial,
    verify_block_sequence,
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
        raise RuntimeError("16K target-block benchmark requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    relative = PLAN_PATH.relative_to(ROOT).as_posix()
    if _git("log", "-1", "--format=%H", "--", relative) != commit:
        raise RuntimeError("16K target-block plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=False)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("16K target-block plan parent differs")
    return commit, plan


def load_target(plan: Mapping[str, Any]) -> RoleBundle:
    from fresh_vocabulary_16k_actual_protocol import build_role_model

    model = build_role_model("candidate_16k")
    state = torch.load(
        TARGET_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    target = plan["target"]
    if (
        hash_file(TARGET_CHECKPOINT_PATH) != target["checkpoint_artifact_sha256"]
        or state_mapping_sha256(model.state_dict()) != target["checkpoint_state_sha256"]
        or sum(parameter.numel() for parameter in model.parameters())
        != target["parameter_count"]
    ):
        raise RuntimeError("16K target-block loaded model identity differs")
    tokenizer, token_bytes = load_tokenizers()[TARGET_VOCABULARY_SIZE]
    transitions = compile_strict_utf8_token_transitions(token_bytes)
    contract = plan["tokenizer_runtime"]["strict_utf8_transitions"]
    if (
        transitions.token_bytes_sha256 != contract["token_bytes_sha256"]
        or transitions.transition_table_sha256 != contract["transition_table_sha256"]
        or transitions.maximum_token_bytes != contract["maximum_token_bytes"]
    ):
        raise RuntimeError("16K target-block token transition identity differs")
    model = model.eval().to("mps")
    return RoleBundle(
        role="candidate_16k_target",
        model=model,
        tokenizer=tokenizer,
        token_bytes=token_bytes,
        transitions=transitions,
        masks=_compile_masks(transitions),
    )


def prepare_payloads(
    bundle: RoleBundle,
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> list[dict[str, Any]]:
    if prompts.shape != continuations.shape or prompts.ndim != 2:
        raise ValueError("16K target-block case arrays differ")
    payloads: list[dict[str, Any]] = []
    for prompt, continuation in zip(prompts, continuations, strict=True):
        prompt_raw = bytes(prompt)
        controlled_raw = bytes(continuation)
        prompt_ids = encode_raw(prompt_raw, bundle.tokenizer, bundle.token_bytes)
        controlled_ids = encode_raw(
            controlled_raw,
            bundle.tokenizer,
            bundle.token_bytes,
        )
        free_ids, free_raw = generate_free_untimed(bundle, prompt_ids)
        payloads.append(
            {
                "prompt_raw": prompt_raw,
                "prompt_ids": prompt_ids,
                "controlled_raw": controlled_raw,
                "controlled_ids": controlled_ids,
                "free_raw": free_raw,
                "free_ids": free_ids,
            }
        )
    return payloads


def correctness_replay(
    bundle: RoleBundle,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not payloads:
        raise ValueError("16K target-block correctness payloads are empty")
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {mode: [] for mode in MODES} for role in ROLES
    }
    for payload in payloads:
        prompt_ids = payload["prompt_ids"]
        by_mode = {
            "controlled_replay": payload["controlled_ids"],
            "free_running_utf8_greedy": payload["free_ids"],
        }
        for role in ROLES:
            for mode, output_ids in by_mode.items():
                if role == "baseline_ar":
                    result = verify_sequence(bundle, prompt_ids, output_ids)
                    result["decode_calls"] = len(output_ids) - 1
                else:
                    result = verify_block_sequence(
                        bundle,
                        prompt_ids,
                        output_ids,
                        block_size=BLOCK_SIZE_BY_ROLE[role],
                        rtol=MPS_RTOL,
                        atol=MPS_ATOL,
                    )
                result["trace_contract_exact"] = True
                rows[role][mode].append(result)
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ROLES:
        output[role] = {}
        for mode in MODES:
            mode_rows = rows[role][mode]
            comparisons = sum(int(row["comparisons"]) for row in mode_rows)
            argmax = sum(int(row["argmax_exact"]) for row in mode_rows)
            output[role][mode] = {
                "cases": len(mode_rows),
                "comparisons": comparisons,
                "argmax_comparisons": comparisons,
                "argmax_exact": argmax,
                "decode_calls": sum(int(row["decode_calls"]) for row in mode_rows),
                "trace_contract_exact": True,
                "maximum_normalized_tolerance_ratio": max(
                    float(row["maximum_normalized_tolerance_ratio"])
                    for row in mode_rows
                ),
                "pass": bool(
                    argmax == comparisons and all(row["pass"] for row in mode_rows)
                ),
            }
    if not correctness_pass(output, expected_cases=len(payloads)):
        raise AssertionError("16K target-block correctness aggregate differs")
    return output


def preflight_target(plan: Mapping[str, Any]) -> None:
    """Run a timing- and result-silent real-checkpoint feasibility check."""

    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("16K target-block preflight cases differ")
    with publication_mps_exclusive():
        bundle = load_target(plan)
        payloads = prepare_payloads(bundle, prompts[:1], continuations[:1])
        correctness_replay(bundle, payloads)
        bundle.model.to("cpu")
        del bundle
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()


def _store_free_trace(
    arrays: Mapping[str, np.ndarray],
    *,
    case_index: int,
    repetition: int,
    role_index: int,
    output_ids: Sequence[int],
    output: bytes,
) -> None:
    arrays["free_token_ids"][
        case_index,
        repetition,
        role_index,
        : len(output_ids),
    ] = np.asarray(output_ids, dtype=np.int32)
    arrays["free_output_bytes"][
        case_index,
        repetition,
        role_index,
        : len(output),
    ] = np.frombuffer(output, dtype=np.uint8)
    arrays["free_output_lengths"][case_index, repetition, role_index] = len(output)


def main() -> None:
    commit, plan = _require_plan_commit()
    if any(
        path.exists()
        for path in (ACTIVE_PATH, RUNTIME_REPORT_PATH, TIMING_PATH, OUTPUT_PATH)
    ):
        raise FileExistsError("16K target-block runtime namespace is not empty")
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise FileExistsError("16K target-block artifact namespace is not empty")
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("16K target-block cases changed after sealing")
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
            raise RuntimeError("16K target-block timing environment is ineligible")
        bundle = load_target(plan)
        payloads = prepare_payloads(bundle, prompts, continuations)
        warmup_correctness = correctness_replay(
            bundle,
            payloads[:WARMUP_CASES],
        )
        for payload in payloads[:WARMUP_CASES]:
            for mode in MODES:
                output_ids = payload[
                    "controlled_ids" if mode == "controlled_replay" else "free_ids"
                ]
                output_raw = payload[
                    "controlled_raw" if mode == "controlled_replay" else "free_raw"
                ]
                for role in ROLES:
                    if role == "baseline_ar":
                        baseline = run_trial(
                            bundle,
                            payload["prompt_raw"],
                            payload["prompt_ids"],
                            payload["controlled_ids"],
                            payload["controlled_raw"],
                            mode,
                        )
                        if (
                            baseline[1] != tuple(output_ids)
                            or baseline[2] != output_raw
                        ):
                            raise AssertionError(
                                "16K target-block warmup baseline output differs"
                            )
                    else:
                        run_perfect_block_trial(
                            bundle,
                            payload["prompt_raw"],
                            payload["prompt_ids"],
                            output_ids,
                            output_raw,
                            mode=mode,
                            block_size=BLOCK_SIZE_BY_ROLE[role],
                            continuation_bytes=plan["experiment"]["continuation_bytes"],
                        )

        shape = (len(MODES), MEASURED_CASES, REPETITIONS, len(ROLES))
        arrays: dict[str, np.ndarray] = {
            name: np.empty(shape, dtype=np.float64) for name in TIMING_COMPONENTS
        }
        arrays["output_token_count"] = np.empty(shape, dtype=np.int16)
        arrays["output_raw_byte_count"] = np.empty(shape, dtype=np.int16)
        arrays["target_forward_calls"] = np.empty(shape, dtype=np.int16)
        maximum_output_bytes = int(
            plan["tokenizer_runtime"]["strict_utf8_transitions"][
                "maximum_free_output_bytes"
            ]
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
            (MEASURED_CASES, REPETITIONS, len(ROLES)),
            dtype=np.int16,
        )
        for case_index in range(MEASURED_CASES):
            payload = payloads[WARMUP_CASES + case_index]
            for repetition in range(REPETITIONS):
                for mode_index, mode in enumerate(MODES):
                    output_ids = payload[
                        "controlled_ids" if mode == "controlled_replay" else "free_ids"
                    ]
                    output_raw = payload[
                        "controlled_raw" if mode == "controlled_replay" else "free_raw"
                    ]
                    for role_index in balanced_role_order(
                        case_index,
                        repetition,
                        mode_index,
                    ):
                        role = ROLES[role_index]
                        if role == "baseline_ar":
                            metrics, actual_ids, actual_raw = run_trial(
                                bundle,
                                payload["prompt_raw"],
                                payload["prompt_ids"],
                                payload["controlled_ids"],
                                payload["controlled_raw"],
                                mode,
                            )
                            counters = {"target_forward_calls": len(actual_ids)}
                        else:
                            (
                                metrics,
                                actual_ids,
                                actual_raw,
                                counters,
                            ) = run_perfect_block_trial(
                                bundle,
                                payload["prompt_raw"],
                                payload["prompt_ids"],
                                output_ids,
                                output_raw,
                                mode=mode,
                                block_size=BLOCK_SIZE_BY_ROLE[role],
                                continuation_bytes=plan["experiment"][
                                    "continuation_bytes"
                                ],
                            )
                        if actual_ids != tuple(output_ids) or actual_raw != output_raw:
                            raise AssertionError(
                                "16K target-block measured output differs"
                            )
                        for name in TIMING_COMPONENTS:
                            arrays[name][
                                mode_index,
                                case_index,
                                repetition,
                                role_index,
                            ] = metrics[name]
                        arrays["output_token_count"][
                            mode_index,
                            case_index,
                            repetition,
                            role_index,
                        ] = len(actual_ids)
                        arrays["output_raw_byte_count"][
                            mode_index,
                            case_index,
                            repetition,
                            role_index,
                        ] = len(actual_raw)
                        arrays["target_forward_calls"][
                            mode_index,
                            case_index,
                            repetition,
                            role_index,
                        ] = counters["target_forward_calls"]
                        if mode == "free_running_utf8_greedy":
                            _store_free_trace(
                                arrays,
                                case_index=case_index,
                                repetition=repetition,
                                role_index=role_index,
                                output_ids=actual_ids,
                                output=actual_raw,
                            )
        end_state = _session_state()
        if not timing_environment_eligible(end_state):
            raise RuntimeError(
                "16K target-block timing environment is ineligible at end"
            )
        bundle.model.to("cpu")
        del bundle, payloads
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    timing_payload = _npz_bytes(arrays)
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_target_block_runtime_report_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "cases": cases,
        "target": plan["target"],
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
        raise RuntimeError("repository changed during 16K target-block benchmark")
    _publish(TIMING_PATH, timing_payload)
    _publish(RUNTIME_REPORT_PATH, json_bytes(report))
    ACTIVE_PATH.unlink()
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed after 16K target-block benchmark")
    print("status=fresh_vocabulary_16k_target_block_timing_complete")


if __name__ == "__main__":
    main()
