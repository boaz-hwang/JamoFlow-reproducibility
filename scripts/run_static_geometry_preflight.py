#!/usr/bin/env python3
"""Run a calibration-only random-weight BLT geometry latency preflight."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.cost import compact_blt_flops
from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import IncrementalBltDecoder
from jamoflow.inference_actual_v5 import array_sha256
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    parameter_count,
    research_versions,
)
from static_geometry_preflight_core import (
    BASELINE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CANDIDATE_ORDER,
    CONTINUATION_BYTES,
    GEOMETRY_ORDER,
    MAXIMUM_PARAMETER_RELATIVE_DIFFERENCE,
    MINIMUM_ANALYTICAL_FLOP_REDUCTION,
    MINIMUM_BOOTSTRAP_LOWER_BOUND,
    MINIMUM_POINT_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODEL_SEED,
    PROMPT_BYTES,
    PROMPT_COUNT,
    PROTOCOL_ID,
    REPETITIONS,
    WARMUP_PROMPTS,
    geometry_contract,
    geometry_spec,
    summarize_geometry_preflight,
    validate_geometry_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/static-geometry-preflight-v1.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
PROMPT_SOURCE_PATH = (
    ROOT / "artifacts/hangul-draft-acceptance-v1/free-target.npz"
)
RAW_PATH = ROOT / "artifacts/static-geometry-preflight-v1/raw.npz"
OUTPUT_PATH = ROOT / "results/static-geometry-preflight-v1/summary.json"
GLOBAL_POSITION_LIMIT = 1032
RTOL = 1e-4
ATOL = 2e-5
IMPLEMENTATION_PATHS = (
    "docs/104-static-local-global-geometry-preflight.md",
    "pyproject.toml",
    "scripts/run_static_geometry_preflight.py",
    "scripts/static_geometry_preflight_core.py",
    "src/jamoflow/cost.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/patching.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/utf8.py",
    "tests/test_static_geometry_preflight.py",
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


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


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


def _require_clean_plan_commit() -> str:
    if _command("git", "status", "--porcelain"):
        raise ValueError("static geometry preflight requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("static geometry preflight requires a Git commit")
    last_change = _command("git", "log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT)))
    if last_change != commit:
        raise ValueError("static geometry plan must be sealed at current HEAD")
    return commit


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"static geometry output already exists: {path}")
    history = _command(
        "git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT))
    )
    if history:
        raise FileExistsError(f"static geometry output has publication history: {path}")


def _require_ac_power() -> str:
    value = _command("pmset", "-g", "batt")
    if "Now drawing from 'AC Power'" not in value:
        raise RuntimeError("static geometry preflight requires AC power")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_plan(plan: Mapping[str, Any], commit: str) -> None:
    expected_keys = {
        "cases",
        "decision_rule",
        "geometry",
        "geometry_measurements",
        "implementation_sha256",
        "input",
        "kind",
        "output",
        "protocol_id",
        "schema_version",
        "status",
        "threat_model",
        "timing",
    }
    if set(plan) != expected_keys:
        raise ValueError("static geometry plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "static_geometry_preflight_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_random_weight_actual_timing"
    ):
        raise ValueError("static geometry plan identity differs")
    if plan["cases"] != {
        "continuation_bytes": CONTINUATION_BYTES,
        "continuations_array_sha256": "eb2718698f32dd4b78324b1a5bc537a38bf639f2d660c710b165d10e6956951a",
        "observed_bytes_per_trial": PROMPT_BYTES + CONTINUATION_BYTES - 1,
        "offsets_array_sha256": "4f90d68d269a0ded7da93643264c99580583ff304324534cbda54245086a25df",
        "prompt_bytes": PROMPT_BYTES,
        "prompt_count": PROMPT_COUNT,
        "prompt_selection": "first 32 prompts in the preexisting model-free domain-separated bottom-hash calibration order",
        "prompt_source_path": str(PROMPT_SOURCE_PATH.relative_to(ROOT)),
        "prompt_source_sha256": "03808c1dd66d3d9cf30e702899a61188a486a3d8ea40ae96636927923450a9f1",
        "prompts_array_sha256": "43a66d0d67aac8c37eb95794a603c1d29213552b0a804b5ab3990d8bfd3e5851",
    }:
        raise ValueError("static geometry case contract differs")
    if plan["input"] != {
        "byte_limit": 1_000_000,
        "source_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": "f789bc7e0ec0252c4c7c636e67a7c44f6d2c528a292ec47542af98488c8b36a5",
        "split": "calibration",
        "stream_bytes": 999_936,
        "stream_sha256": "69f6aa9347f7e265d6df5097e4219c944b4da7cf6d8522a831a26c670a4c39ec",
    }:
        raise ValueError("static geometry input contract differs")
    validate_geometry_contract(plan["geometry"])
    if plan["decision_rule"] != {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidate_selection": "first passing candidate in fixed order",
        "maximum_parameter_relative_difference": MAXIMUM_PARAMETER_RELATIVE_DIFFERENCE,
        "minimum_analytical_flop_reduction": MINIMUM_ANALYTICAL_FLOP_REDUCTION,
        "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
        "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "pass_authorizes": "one Korean train/calibration seed for the selected static control",
        "stop_rule": "if no candidate passes every fixed gate, terminate the static geometry branch without training",
    }:
        raise ValueError("static geometry decision rule differs")
    if plan["timing"] != {
        "ac_power_required": True,
        "device": "mps",
        "global_position_limit": GLOBAL_POSITION_LIMIT,
        "model_seed": MODEL_SEED,
        "process_exclusion": "shared publication MPS flock and known-entrypoint inventory",
        "repetitions": REPETITIONS,
        "role_order": "cyclic position-balanced within prompt x repetition",
        "scope": "fresh runtime construction, 128-byte parallel prefill, 127 controlled feedback bytes, final synchronization",
        "torch_inference_mode": True,
        "warmup_prompts": WARMUP_PROMPTS,
    }:
        raise ValueError("static geometry timing contract differs")
    if plan["output"] != {
        "raw_artifact_path": str(RAW_PATH.relative_to(ROOT)),
        "summary_path": str(OUTPUT_PATH.relative_to(ROOT)),
    }:
        raise ValueError("static geometry output contract differs")
    if set(plan["geometry_measurements"]) != set(GEOMETRY_ORDER):
        raise ValueError("static geometry measurement rows differ")
    for name in GEOMETRY_ORDER:
        row = plan["geometry_measurements"][name]
        if (
            set(row)
            != {"analytical_dense_matmul_flops", "parameter_count"}
            or not isinstance(row["parameter_count"], int)
            or not isinstance(row["analytical_dense_matmul_flops"], int)
            or row["parameter_count"] <= 0
            or row["analytical_dense_matmul_flops"] <= 0
        ):
            raise ValueError(f"static geometry measurement schema differs: {name}")
    if plan["threat_model"] != {
        "calibration_only": True,
        "case_selected_by_model_output": False,
        "final_test_or_final_timing_read": False,
        "quality_evidence_from_random_weights": False,
        "static_geometry_novelty_claimed": False,
        "timing_used_only_as_one_seed_training_feasibility": True,
    }:
        raise ValueError("static geometry threat model differs")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("static geometry implementation file set differs")
    for relative, expected in plan["implementation_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or hash_file(path) != expected:
            raise ValueError(f"static geometry implementation differs: {relative}")


def _load_cases(plan: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    input_contract = plan["input"]
    if hash_file(SOURCE_PATH) != input_contract["source_sha256"]:
        raise ValueError("static geometry source differs")
    stream = build_neural_stream(
        SOURCE_PATH,
        "ko",
        "calibration",
        int(input_contract["byte_limit"]),
        512,
    )
    if (
        len(stream.data) != input_contract["stream_bytes"]
        or hashlib.sha256(stream.data).hexdigest()
        != input_contract["stream_sha256"]
    ):
        raise ValueError("static geometry calibration stream differs")
    if hash_file(PROMPT_SOURCE_PATH) != plan["cases"]["prompt_source_sha256"]:
        raise ValueError("static geometry prompt source differs")
    with np.load(PROMPT_SOURCE_PATH, allow_pickle=False) as source:
        prompts = np.ascontiguousarray(source["prompts"][:PROMPT_COUNT])
        offsets = np.ascontiguousarray(source["prompt_offsets"][:PROMPT_COUNT])
    if prompts.dtype != np.uint8 or prompts.shape != (PROMPT_COUNT, PROMPT_BYTES):
        raise ValueError("static geometry prompt array differs")
    continuations = np.stack(
        [
            np.frombuffer(
                stream.data[int(offset) + PROMPT_BYTES : int(offset) + PROMPT_BYTES + CONTINUATION_BYTES],
                dtype=np.uint8,
            )
            for offset in offsets
        ]
    )
    if continuations.shape != (PROMPT_COUNT, CONTINUATION_BYTES):
        raise ValueError("static geometry continuation array differs")
    if (
        array_sha256(prompts) != plan["cases"]["prompts_array_sha256"]
        or array_sha256(offsets) != plan["cases"]["offsets_array_sha256"]
        or array_sha256(continuations)
        != plan["cases"]["continuations_array_sha256"]
    ):
        raise ValueError("static geometry case identity differs")
    return prompts, continuations


def _runtime(model: Any) -> IncrementalBltDecoder:
    return IncrementalBltDecoder(
        model,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )


def _normalized_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = ATOL + RTOL * torch.abs(right)
    return float(torch.max(torch.abs(left - right) / denominator).item())


def _correctness(model: Any, prompt: bytes, continuation: bytes) -> dict[str, Any]:
    sequential = _runtime(model)
    parallel = _runtime(model)
    with torch.inference_mode():
        left = sequential.prefill(prompt)
        right = parallel.prefill_parallel(prompt)
        maximum = _normalized_error(left, right)
        comparisons = 1
        argmax_exact = int(left.argmax().item() == right.argmax().item())
        for value in continuation[:-1]:
            left = sequential.consume(value)
            right = parallel.consume(value)
            maximum = max(maximum, _normalized_error(left, right))
            comparisons += 1
            argmax_exact += int(left.argmax().item() == right.argmax().item())
    seq_diag = sequential.diagnostics
    par_diag = parallel.diagnostics
    return {
        "argmax_comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "boundary_trace_exact": seq_diag.boundaries == par_diag.boundaries,
        "cache_diagnostics_exact": seq_diag == par_diag,
        "maximum_normalized_logit_error": maximum,
    }


def _trial(model: Any, prompt: bytes, continuation: bytes) -> float:
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        runtime = _runtime(model)
        runtime.prefill_parallel(prompt)
        for value in continuation[:-1]:
            runtime.consume(value)
        torch.mps.synchronize()
    finished = time.perf_counter_ns()
    diagnostics = runtime.diagnostics
    expected = len(prompt) + len(continuation) - 1
    if (
        diagnostics.observed_bytes != expected
        or diagnostics.local_encoder_cached_bytes != expected
        or diagnostics.local_decoder_cached_bytes != expected
        or diagnostics.global_cached_patches != diagnostics.emitted_data_patches
    ):
        raise AssertionError("static geometry runtime cache invariant differs")
    return (finished - started) / 1_000_000


def _order(case: int, repetition: int) -> tuple[int, ...]:
    start = (case + repetition) % len(GEOMETRY_ORDER)
    return tuple((start + offset) % len(GEOMETRY_ORDER) for offset in range(len(GEOMETRY_ORDER)))


def _summary_sha(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("summary_sha256", None)
    return hashlib.sha256(_json_bytes(copy)).hexdigest()


def main() -> None:
    commit = _require_clean_plan_commit()
    _require_never_published(OUTPUT_PATH)
    if RAW_PATH.exists():
        raise FileExistsError(f"static geometry raw artifact already exists: {RAW_PATH}")
    plan = _read_json(PLAN_PATH)
    _validate_plan(plan, commit)
    prompts, continuations = _load_cases(plan)
    power_sha256 = _require_ac_power()
    parameter_counts: dict[str, int] = {}
    analytical_flops: dict[str, int] = {}
    correctness: dict[str, dict[str, Any]] = {}
    timings = np.empty(
        (PROMPT_COUNT, REPETITIONS, len(GEOMETRY_ORDER)), dtype=np.float64
    )
    models: list[Any] = []
    started = time.time()
    with publication_mps_exclusive(), torch.inference_mode():
        for name in GEOMETRY_ORDER:
            spec = geometry_spec(name)
            model = build_main_model(
                spec,
                seed=MODEL_SEED,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            ).to("mps")
            model.eval()
            parameters = parameter_count(model)
            flops = int(
                compact_blt_flops(spec, data_patches=72)[
                    "forward_flops_per_sequence"
                ]
            )
            expected = plan["geometry_measurements"][name]
            if (
                parameters != expected["parameter_count"]
                or flops != expected["analytical_dense_matmul_flops"]
            ):
                raise ValueError(f"static geometry analytical identity differs: {name}")
            parameter_counts[name] = parameters
            analytical_flops[name] = flops
            models.append(model)

        for index, name in enumerate(GEOMETRY_ORDER):
            correctness[name] = _correctness(
                models[index], bytes(prompts[0]), bytes(continuations[0])
            )

        for case in range(WARMUP_PROMPTS):
            for model in models:
                _trial(model, bytes(prompts[case]), bytes(continuations[case]))

        for case in range(PROMPT_COUNT):
            prompt = bytes(prompts[case])
            continuation = bytes(continuations[case])
            for repetition in range(REPETITIONS):
                for index in _order(case, repetition):
                    timings[case, repetition, index] = _trial(
                        models[index], prompt, continuation
                    )

        for model in models:
            model.to("cpu")
        models.clear()
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()

    aggregate = summarize_geometry_preflight(
        timings_ms=timings,
        parameter_counts=parameter_counts,
        analytical_flops=analytical_flops,
        correctness=correctness,
    )
    raw_bytes = _npz_bytes({"timings_ms": timings})
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "static_geometry_preflight_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": aggregate["status"],
        "aggregate": aggregate,
        "elapsed_seconds": time.time() - started,
        "provenance": {
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "power_snapshot_sha256": power_sha256,
            "runtime": research_versions(),
            "torch_inference_mode": True,
        },
        "raw_evidence": {
            "artifact_path": str(RAW_PATH.relative_to(ROOT)),
            "artifact_sha256": raw_sha256,
            "array_sha256": {"timings_ms": array_sha256(timings)},
        },
        "claim_boundary": {
            "calibration_only": True,
            "random_weight_latency_only": True,
            "quality_or_publication_efficiency_claimed": False,
            "static_geometry_is_novelty_claimed": False,
            "pass_authorizes_only_one_seed_training": True,
        },
    }
    summary["summary_sha256"] = _summary_sha(summary)
    summary_bytes = _json_bytes(summary)
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain"
    ):
        raise ValueError("repository changed during static geometry preflight")
    _publish_no_clobber(RAW_PATH, raw_bytes)
    _publish_no_clobber(OUTPUT_PATH, summary_bytes)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
