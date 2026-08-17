#!/usr/bin/env python3
"""Train and time the one authorized static-geometry seed on calibration only."""

from __future__ import annotations

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

from jamoflow.compute_conversion import conversion_patch_matrices
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import IncrementalBltDecoder
from jamoflow.inference_actual_v5 import array_sha256
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_initial_model_identity_v2 import runtime_environment_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import (
    evaluate_main_model,
    shuffled_indices,
    train_main_model,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.phase3 import PHASE3_OPTIMIZATION_SPEC
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    StrictUtf8State,
    advance_strict_utf8,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
)
from static_geometry_one_seed_core import (
    CONTINUATION_BYTES,
    MODES,
    PROMPT_BYTES,
    PROMPT_COUNT,
    PROTOCOL_ID,
    REPETITIONS,
    ROLES,
    SEED,
    WARMUP_PROMPTS,
    one_seed_decision,
    summarize_one_seed_quality,
    summarize_one_seed_timing,
)
from static_geometry_preflight_core import geometry_spec


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/static-geometry-one-seed-v1.json"
STATIC_RESULT_PATH = ROOT / "results/static-geometry-preflight-v1/summary.json"
PRIMARY_SUMMARY_PATH = ROOT / "results/phase3-primary-five-seed/summary.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
PROMPT_SOURCE_PATH = ROOT / "artifacts/hangul-draft-acceptance-v1/free-target.npz"
BASELINE_REPORT_PATH = ROOT / "runs/phase3-compute-conversion/seed-1729/causal_whitespace_grid_72.json"
BASELINE_CHECKPOINT_PATH = ROOT / "artifacts/phase3-compute-conversion/seed-1729/causal_whitespace_grid_72.pt"
BASELINE_NLL_PATH = ROOT / "artifacts/phase3-compute-conversion/seed-1729/causal_whitespace_grid_72-calibration-nll.npz"
ARTIFACT_ROOT = ROOT / "artifacts/static-geometry-one-seed-v1"
CANDIDATE_CHECKPOINT_PATH = ARTIFACT_ROOT / "candidate.pt"
CANDIDATE_NLL_PATH = ARTIFACT_ROOT / "candidate-calibration-nll.npz"
TRAINING_RECEIPT_PATH = ARTIFACT_ROOT / "training-receipt.json"
TIMING_PATH = ARTIFACT_ROOT / "timing.npz"
OUTPUT_PATH = ROOT / "results/static-geometry-one-seed-v1/summary.json"
GLOBAL_POSITION_LIMIT = 1032
ATOL = 2e-5
RTOL = 1e-4
MAXIMUM_FREE_OUTPUT_BYTES = CONTINUATION_BYTES + 3

IMPLEMENTATION_PATHS = (
    "docs/106-static-geometry-one-seed-protocol.md",
    "pyproject.toml",
    "scripts/run_static_geometry_one_seed.py",
    "scripts/static_geometry_one_seed_core.py",
    "scripts/static_geometry_preflight_core.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/inference_initial_model_identity_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/neural_training.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/utf8.py",
    "tests/test_static_geometry_one_seed.py",
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


def _clean_plan_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("one-seed geometry screen requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    if _command(
        "git", "log", "-1", "--format=%H", "--", PLAN_PATH.relative_to(ROOT).as_posix()
    ) != commit:
        raise ValueError("one-seed geometry plan must be sealed at current HEAD")
    return commit


def _require_unchanged(commit: str) -> None:
    if (
        _command("git", "rev-parse", "HEAD") != commit
        or _command("git", "status", "--porcelain", "--untracked-files=all")
    ):
        raise RuntimeError("repository changed during one-seed geometry screen")


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"one-seed geometry output already exists: {path}")
    history = _command(
        "git", "log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix()
    )
    if history:
        raise FileExistsError(f"one-seed geometry output has history: {path}")


def _require_ac_power() -> str:
    value = _command("pmset", "-g", "batt")
    if "Now drawing from 'AC Power'" not in value:
        raise RuntimeError("one-seed geometry screen requires AC power")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _strict_npz_vector(path: Path, expected_sha256: str) -> np.ndarray:
    if path.is_symlink() or hash_file(path) != expected_sha256:
        raise ValueError(f"one-seed NLL artifact differs: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError("one-seed NLL key differs")
        values = np.ascontiguousarray(archive["sequence_nll_nats"])
    if (
        values.dtype != np.float32
        or values.shape != (15_625,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError("one-seed NLL vector differs")
    return values


def _validate_plan(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "artifacts",
        "baseline",
        "candidate",
        "cases",
        "claim_boundary",
        "data",
        "decision_rule",
        "implementation_sha256",
        "kind",
        "output",
        "protocol_id",
        "runtime_environment",
        "schema_version",
        "static_preflight",
        "status",
        "training",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "static_geometry_one_seed_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_before_candidate_training"
        or set(plan.get("implementation_sha256", {})) != set(IMPLEMENTATION_PATHS)
    ):
        raise ValueError("one-seed geometry plan schema differs")
    for relative, expected in plan["implementation_sha256"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or hash_file(path) != expected:
            raise ValueError(f"one-seed implementation differs: {relative}")
    if plan["static_preflight"] != {
        "artifact_path": STATIC_RESULT_PATH.relative_to(ROOT).as_posix(),
        "artifact_sha256": "0735291749e1305835a9dd09a4a22293e240cf9cc9f3e656fe1a69a001b0c352",
        "selected_candidate": "thin160_e1_d1_g384x9",
        "status": "one_seed_static_control_authorized",
    }:
        raise ValueError("one-seed static authorization differs")
    if plan["training"] != {
        "device": "mps",
        "global_position_limit": GLOBAL_POSITION_LIMIT,
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "patch_policy": "causal_whitespace_grid_72",
        "seed": SEED,
        "test_split_evaluated": False,
        "training_order_sha256": "2691425529b67f9a17ca69c7a7516cc7e1b853731caa53871781e2f89a35aff1",
    }:
        raise ValueError("one-seed training contract differs")
    if plan["candidate"] != {
        "geometry": "thin160_e1_d1_g384x9",
        "initial_state_sha256": "3d52df3b248ef97803cd3d235f8b171a685b447c55efce106ed5aa3e2c3c3af7",
        "model_spec": geometry_spec("thin160_e1_d1_g384x9").to_dict(),
        "parameter_count": 19_571_872,
    }:
        raise ValueError("one-seed candidate contract differs")
    if plan["cases"] != {
        "continuation_bytes": CONTINUATION_BYTES,
        "continuations_array_sha256": "b9938bb6995caaa71768d553d67a8505e2a431b7c7af171e928528268416b06b",
        "free_maximum_output_bytes": MAXIMUM_FREE_OUTPUT_BYTES,
        "offsets_array_sha256": "3782da3c84838e8766303d6c72e42511b8d836f54037f40b526176fc590aeef9",
        "prompt_bytes": PROMPT_BYTES,
        "prompt_count": PROMPT_COUNT,
        "prompt_selection": "first 64 prompts in the preexisting model-free domain-separated bottom-hash calibration order",
        "prompt_source_path": PROMPT_SOURCE_PATH.relative_to(ROOT).as_posix(),
        "prompt_source_sha256": "03808c1dd66d3d9cf30e702899a61188a486a3d8ea40ae96636927923450a9f1",
        "prompts_array_sha256": "73c7ee6be116a78240d63600e466b4b025e0779e967b7e5125bed216d61ee4c3",
    }:
        raise ValueError("one-seed case contract differs")
    if plan["claim_boundary"] != {
        "calibration_only": True,
        "one_model_seed_screen": True,
        "publication_quality_or_efficiency_claimed": False,
        "static_geometry_novelty_claimed": False,
        "test_or_final_evidence_read": False,
    }:
        raise ValueError("one-seed claim boundary differs")
    if plan["decision_rule"] != {
        "overall_pass": "quality and controlled-replay latency and free-running latency must all pass",
        "quality": {
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 20_261_001,
            "margin_bpb": 0.01,
            "minimum_document_coverage": 0.95,
            "requirements": "full-stream mean and document-bootstrap one-sided 95% upper must both be <= margin",
        },
        "timing": {
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 20_261_002,
            "minimum_bootstrap_lower_bound": 0.10,
            "minimum_point_reduction": 0.15,
            "minimum_positive_prompts": 48,
            "modes": list(MODES),
            "repetitions": REPETITIONS,
            "warmup_prompts": WARMUP_PROMPTS,
        },
    }:
        raise ValueError("one-seed decision rule differs")
    data = plan["data"]
    if (
        not isinstance(data, Mapping)
        or set(data)
        != {
            "calibration_document_map",
            "primary_summary_path",
            "primary_summary_sha256",
            "source_bytes",
            "source_path",
            "source_sha256",
            "streams",
        }
        or data["primary_summary_path"]
        != PRIMARY_SUMMARY_PATH.relative_to(ROOT).as_posix()
        or data["source_path"] != SOURCE_PATH.relative_to(ROOT).as_posix()
        or data["source_bytes"] != 152_461_842
        or set(data["streams"]) != {"train", "calibration"}
    ):
        raise ValueError("one-seed data contract differs")
    if set(plan["baseline"]) != {
        "calibration_nll_array_sha256",
        "calibration_nll_path",
        "calibration_nll_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "parameter_count",
        "report_path",
        "report_sha256",
        "state_sha256",
    }:
        raise ValueError("one-seed baseline schema differs")
    if plan["output"] != {
        "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "timing_artifact_path": TIMING_PATH.relative_to(ROOT).as_posix(),
    }:
        raise ValueError("one-seed output contract differs")
    if plan["artifacts"] != {
        "candidate_checkpoint_path": CANDIDATE_CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        "candidate_nll_path": CANDIDATE_NLL_PATH.relative_to(ROOT).as_posix(),
        "training_receipt_path": TRAINING_RECEIPT_PATH.relative_to(ROOT).as_posix(),
    }:
        raise ValueError("one-seed artifact contract differs")
    if plan["runtime_environment"] != runtime_environment_v2():
        raise ValueError("one-seed runtime environment differs")


def _verify_static_and_baseline(plan: Mapping[str, Any]) -> np.ndarray:
    if hash_file(STATIC_RESULT_PATH) != plan["static_preflight"]["artifact_sha256"]:
        raise ValueError("one-seed static result differs")
    static = _read_json(STATIC_RESULT_PATH)
    if (
        static.get("status") != "one_seed_static_control_authorized"
        or static.get("aggregate", {}).get("selection", {}).get("selected_candidate")
        != "thin160_e1_d1_g384x9"
    ):
        raise ValueError("one-seed static result does not authorize training")
    baseline = plan["baseline"]
    expected_paths = {
        "report_path": BASELINE_REPORT_PATH,
        "checkpoint_path": BASELINE_CHECKPOINT_PATH,
        "calibration_nll_path": BASELINE_NLL_PATH,
    }
    for key, path in expected_paths.items():
        if baseline[key] != path.relative_to(ROOT).as_posix():
            raise ValueError("one-seed baseline path differs")
    if (
        hash_file(BASELINE_REPORT_PATH) != baseline["report_sha256"]
        or hash_file(BASELINE_CHECKPOINT_PATH) != baseline["checkpoint_sha256"]
    ):
        raise ValueError("one-seed baseline artifact differs")
    report = _read_json(BASELINE_REPORT_PATH)
    if (
        report.get("seed") != SEED
        or report.get("policy") != "causal_whitespace_grid_72"
        or report.get("parameters") != baseline["parameter_count"]
        or report.get("model_spec") != geometry_spec("baseline_w72").to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT
        or report.get("training_order_sha256")
        != plan["training"]["training_order_sha256"]
        or report.get("checkpoint_artifact_sha256") != baseline["checkpoint_sha256"]
        or report.get("calibration_loss_artifact_sha256")
        != baseline["calibration_nll_sha256"]
    ):
        raise ValueError("one-seed baseline training identity differs")
    model = build_main_model(
        geometry_spec("baseline_w72"),
        seed=SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(BASELINE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    if (
        parameter_count(model) != baseline["parameter_count"]
        or _state_sha256(model) != baseline["state_sha256"]
    ):
        raise ValueError("one-seed baseline state differs")
    del model
    values = _strict_npz_vector(BASELINE_NLL_PATH, baseline["calibration_nll_sha256"])
    if array_sha256(values) != baseline["calibration_nll_array_sha256"]:
        raise ValueError("one-seed baseline NLL array differs")
    return values


def _reconstruct_data(plan: Mapping[str, Any]) -> dict[str, Any]:
    if (
        hash_file(PRIMARY_SUMMARY_PATH) != plan["data"]["primary_summary_sha256"]
        or hash_file(SOURCE_PATH) != plan["data"]["source_sha256"]
    ):
        raise ValueError("one-seed data authority differs")
    primary = _read_json(PRIMARY_SUMMARY_PATH)
    manifest = primary["run_manifest"]
    context: dict[str, Any] = {}
    for split in ("train", "calibration"):
        expected = plan["data"]["streams"][split]
        stream = build_neural_stream(
            SOURCE_PATH,
            "ko",
            split,
            int(expected["byte_limit"]),
            512,
        )
        inputs, boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            512,
        )
        whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
        matrix = conversion_patch_matrices(
            boundaries,
            whitespace,
            rate=72,
        )["causal_whitespace_grid_72"]
        actual = {
            "boundaries_sha256": array_sha256(boundaries),
            "byte_limit": int(expected["byte_limit"]),
            "inputs_sha256": array_sha256(inputs),
            "matrix_sha256": array_sha256(matrix),
            "sequence_count": len(inputs),
            "stream_sha256": hashlib.sha256(stream.data).hexdigest(),
            "whitespace_sha256": array_sha256(whitespace),
        }
        if actual != expected or manifest["streams"][split]["sequence_count"] != len(inputs):
            raise ValueError(f"one-seed {split} stream differs")
        context[split] = {
            "stream": stream,
            "inputs": inputs,
            "matrix": matrix,
        }
    document_map = reconstruct_document_window_map(
        SOURCE_PATH,
        split="calibration",
        byte_limit=int(plan["data"]["streams"]["calibration"]["byte_limit"]),
        sequence_length=512,
        expected_stream=context["calibration"]["stream"].data,
    )
    if document_map.metadata() != plan["data"]["calibration_document_map"]:
        raise ValueError("one-seed calibration document map differs")
    context["document_map"] = document_map
    return context


def _save_state_bytes(model: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        buffer,
    )
    return buffer.getvalue()


def _training_receipt_payload(
    *,
    commit: str,
    model: Any,
    checkpoint_sha256: str,
    nll_sha256: str,
    nll_array_sha256: str,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "static_geometry_one_seed_training_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "git_commit": commit,
        "seed": SEED,
        "geometry": "thin160_e1_d1_g384x9",
        "parameter_count": parameter_count(model),
        "initial_state_sha256": "3d52df3b248ef97803cd3d235f8b171a685b447c55efce106ed5aa3e2c3c3af7",
        "trained_state_sha256": _state_sha256(model),
        "training_order_sha256": "2691425529b67f9a17ca69c7a7516cc7e1b853731caa53871781e2f89a35aff1",
        "checkpoint_artifact_sha256": checkpoint_sha256,
        "calibration_nll_artifact_sha256": nll_sha256,
        "calibration_nll_array_sha256": nll_array_sha256,
        "training": dict(training),
        "calibration_evaluation": dict(evaluation),
        "test_split_evaluated": False,
    }


def _load_or_train_candidate(
    context: Mapping[str, Any],
    *,
    commit: str,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    presence = tuple(
        path.exists()
        for path in (
            CANDIDATE_CHECKPOINT_PATH,
            CANDIDATE_NLL_PATH,
            TRAINING_RECEIPT_PATH,
        )
    )
    if any(presence) and not all(presence):
        raise ValueError("partial one-seed training evidence requires forensic recovery")
    spec = geometry_spec("thin160_e1_d1_g384x9")
    if all(presence):
        receipt = _read_json(TRAINING_RECEIPT_PATH)
        model = build_main_model(
            spec,
            seed=SEED,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        model.load_state_dict(
            torch.load(CANDIDATE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        )
        nll = _strict_npz_vector(
            CANDIDATE_NLL_PATH,
            receipt["calibration_nll_artifact_sha256"],
        )
        replay_evaluation, replay_losses = evaluate_main_model(
            model,
            context["calibration"]["inputs"],
            context["calibration"]["matrix"],
            "mps",
            batch_size=PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
            return_sequence_nll=True,
        )
        if replay_losses is None or not np.array_equal(
            nll,
            replay_losses.astype(np.float32, copy=False),
        ):
            raise ValueError("one-seed resumed NLL differs from checkpoint replay")
        replay_metrics = replay_evaluation.to_dict()
        stored_metrics = receipt["calibration_evaluation"]
        if any(
            not np.isclose(
                float(replay_metrics[key]),
                float(stored_metrics[key]),
                rtol=0,
                atol=1e-7,
            )
            for key in ("examples", "predicted_bytes", "nll_nats", "bpb")
        ):
            raise ValueError("one-seed resumed evaluation differs")
        rebuilt = _training_receipt_payload(
            commit=commit,
            model=model,
            checkpoint_sha256=hash_file(CANDIDATE_CHECKPOINT_PATH),
            nll_sha256=hash_file(CANDIDATE_NLL_PATH),
            nll_array_sha256=array_sha256(nll),
            training=receipt["training"],
            evaluation=receipt["calibration_evaluation"],
        )
        if receipt != rebuilt:
            raise ValueError("one-seed training receipt differs")
        print("one-seed candidate training already complete", flush=True)
        return model, nll, receipt

    model = build_main_model(
        spec,
        seed=SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    if (
        parameter_count(model) != 19_571_872
        or _state_sha256(model)
        != "3d52df3b248ef97803cd3d235f8b171a685b447c55efce106ed5aa3e2c3c3af7"
    ):
        raise ValueError("one-seed candidate initialization differs")
    order = shuffled_indices(len(context["train"]["inputs"]), SEED)
    if array_sha256(order) != "2691425529b67f9a17ca69c7a7516cc7e1b853731caa53871781e2f89a35aff1":
        raise ValueError("one-seed training order differs")
    print("one-seed candidate: training 19,571,872 parameters", flush=True)
    training = train_main_model(
        model,
        context["train"]["inputs"],
        context["train"]["matrix"],
        order,
        "mps",
        PHASE3_OPTIMIZATION_SPEC,
    )
    evaluation, losses = evaluate_main_model(
        model,
        context["calibration"]["inputs"],
        context["calibration"]["matrix"],
        "mps",
        batch_size=PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
        return_sequence_nll=True,
    )
    if losses is None:
        raise AssertionError("one-seed calibration losses were not produced")
    nll = np.ascontiguousarray(losses.astype(np.float32, copy=False))
    checkpoint_bytes = _save_state_bytes(model)
    nll_bytes = _npz_bytes({"sequence_nll_nats": nll})
    receipt = _training_receipt_payload(
        commit=commit,
        model=model,
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        nll_sha256=hashlib.sha256(nll_bytes).hexdigest(),
        nll_array_sha256=array_sha256(nll),
        training=training.to_dict(),
        evaluation=evaluation.to_dict(),
    )
    _publish_no_clobber(CANDIDATE_CHECKPOINT_PATH, checkpoint_bytes)
    _publish_no_clobber(CANDIDATE_NLL_PATH, nll_bytes)
    _publish_no_clobber(TRAINING_RECEIPT_PATH, _json_bytes(receipt))
    return model, nll, receipt


def _load_cases(plan: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if hash_file(PROMPT_SOURCE_PATH) != plan["cases"]["prompt_source_sha256"]:
        raise ValueError("one-seed prompt source differs")
    with np.load(PROMPT_SOURCE_PATH, allow_pickle=False) as archive:
        prompts = np.ascontiguousarray(archive["prompts"][:PROMPT_COUNT])
        offsets = np.ascontiguousarray(archive["prompt_offsets"][:PROMPT_COUNT])
    stream = build_neural_stream(SOURCE_PATH, "ko", "calibration", 1_000_000, 512)
    continuations = np.stack(
        [
            np.frombuffer(
                stream.data[int(offset) + PROMPT_BYTES : int(offset) + PROMPT_BYTES + CONTINUATION_BYTES],
                dtype=np.uint8,
            )
            for offset in offsets
        ]
    )
    if (
        prompts.dtype != np.uint8
        or prompts.shape != (PROMPT_COUNT, PROMPT_BYTES)
        or continuations.shape != (PROMPT_COUNT, CONTINUATION_BYTES)
        or array_sha256(prompts) != plan["cases"]["prompts_array_sha256"]
        or array_sha256(offsets) != plan["cases"]["offsets_array_sha256"]
        or array_sha256(continuations) != plan["cases"]["continuations_array_sha256"]
    ):
        raise ValueError("one-seed timing cases differ")
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


def _correctness(
    model: Any,
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> dict[str, Any]:
    maximum = 0.0
    comparisons = 0
    argmax_exact = 0
    boundaries_exact = True
    caches_exact = True
    with torch.inference_mode():
        for prompt, continuation in zip(
            prompts[:WARMUP_PROMPTS],
            continuations[:WARMUP_PROMPTS],
            strict=True,
        ):
            sequential = _runtime(model)
            parallel = _runtime(model)
            left = sequential.prefill(bytes(prompt))
            right = parallel.prefill_parallel(bytes(prompt))
            maximum = max(maximum, _normalized_error(left, right))
            comparisons += 1
            argmax_exact += int(left.argmax().item() == right.argmax().item())
            for value in bytes(continuation[:-1]):
                left = sequential.consume(value)
                right = parallel.consume(value)
                maximum = max(maximum, _normalized_error(left, right))
                comparisons += 1
                argmax_exact += int(left.argmax().item() == right.argmax().item())
            boundaries_exact &= sequential.diagnostics.boundaries == parallel.diagnostics.boundaries
            caches_exact &= sequential.diagnostics == parallel.diagnostics
    return {
        "argmax_comparisons": comparisons,
        "argmax_exact": argmax_exact,
        "boundary_trace_exact": bool(boundaries_exact),
        "cache_diagnostics_exact": bool(caches_exact),
        "maximum_normalized_logit_error": maximum,
        "strict_free_outputs": 0,
    }


def _utf8_masks() -> dict[StrictUtf8State, torch.Tensor]:
    masks: dict[StrictUtf8State, torch.Tensor] = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device="mps")
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        masks[state] = mask
    torch.mps.synchronize()
    return masks


def _trial(
    model: Any,
    prompt: bytes,
    continuation: bytes,
    mode: str,
    masks: Mapping[StrictUtf8State, torch.Tensor],
) -> tuple[float, float, float, bytes]:
    torch.mps.synchronize()
    started = time.perf_counter_ns()
    generated = bytearray()
    with torch.inference_mode():
        runtime = _runtime(model)
        logits = runtime.prefill_parallel(prompt)
        torch.mps.synchronize()
        prefilled = time.perf_counter_ns()
        if mode == "controlled_replay":
            for value in continuation[:-1]:
                logits = runtime.consume(value)
            output = continuation
        elif mode == "free_running_utf8_greedy":
            state = STRICT_UTF8_INITIAL_STATE
            while True:
                value = int(
                    logits.masked_fill(~masks[state], -torch.inf)
                    .argmax(dim=-1)
                    .item()
                )
                generated.append(value)
                state = advance_strict_utf8(state, value)
                if not state.valid:
                    raise AssertionError("strict mask admitted invalid UTF-8")
                if len(generated) >= CONTINUATION_BYTES and state.at_codepoint_boundary:
                    break
                if len(generated) >= MAXIMUM_FREE_OUTPUT_BYTES:
                    raise AssertionError("free output exceeded UTF-8 bound")
                logits = runtime.consume(value)
            output = bytes(generated)
            output.decode("utf-8", errors="strict")
        else:
            raise ValueError("one-seed timing mode differs")
        torch.mps.synchronize()
        finished = time.perf_counter_ns()
    expected = len(prompt) + len(output) - 1
    diagnostics = runtime.diagnostics
    if (
        diagnostics.observed_bytes != expected
        or diagnostics.local_encoder_cached_bytes != expected
        or diagnostics.local_decoder_cached_bytes != expected
        or diagnostics.global_cached_patches != diagnostics.emitted_data_patches
    ):
        raise AssertionError("one-seed timing cache invariant differs")
    return (
        (prefilled - started) / 1_000_000,
        (finished - prefilled) / 1_000_000,
        (finished - started) / 1_000_000,
        output,
    )


def _role_order(prompt: int, repetition: int, mode: int) -> tuple[int, int]:
    first = (prompt + repetition + mode) % 2
    return (first, 1 - first)


def _measure(
    candidate_model: Any,
    baseline_model: Any,
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    models = [candidate_model.to("mps").eval(), baseline_model.to("mps").eval()]
    masks = _utf8_masks()
    correctness = {
        role: _correctness(models[index], prompts, continuations)
        for index, role in enumerate(ROLES)
    }
    for prompt in range(WARMUP_PROMPTS):
        for mode in MODES:
            for model in models:
                _trial(
                    model,
                    bytes(prompts[prompt]),
                    bytes(continuations[prompt]),
                    mode,
                    masks,
                )
    shape = (len(MODES), PROMPT_COUNT, REPETITIONS, len(ROLES))
    ttft = np.empty(shape, dtype=np.float64)
    decode = np.empty(shape, dtype=np.float64)
    end_to_end = np.empty(shape, dtype=np.float64)
    emitted = np.empty((len(MODES), PROMPT_COUNT, len(ROLES)), dtype=np.int16)
    free_outputs = np.zeros(
        (PROMPT_COUNT, len(ROLES), MAXIMUM_FREE_OUTPUT_BYTES),
        dtype=np.uint8,
    )
    free_lengths = np.zeros((PROMPT_COUNT, len(ROLES)), dtype=np.int16)
    for prompt in range(PROMPT_COUNT):
        raw_prompt = bytes(prompts[prompt])
        continuation = bytes(continuations[prompt])
        for repetition in range(REPETITIONS):
            for mode_index, mode in enumerate(MODES):
                for role_index in _role_order(prompt, repetition, mode_index):
                    values = _trial(
                        models[role_index],
                        raw_prompt,
                        continuation,
                        mode,
                        masks,
                    )
                    ttft[mode_index, prompt, repetition, role_index] = values[0]
                    decode[mode_index, prompt, repetition, role_index] = values[1]
                    end_to_end[mode_index, prompt, repetition, role_index] = values[2]
                    output = values[3]
                    if repetition == 0:
                        emitted[mode_index, prompt, role_index] = len(output)
                        if mode == "free_running_utf8_greedy":
                            free_lengths[prompt, role_index] = len(output)
                            free_outputs[prompt, role_index, : len(output)] = np.frombuffer(
                                output,
                                dtype=np.uint8,
                            )
                    else:
                        if emitted[mode_index, prompt, role_index] != len(output):
                            raise AssertionError("deterministic output length changed")
                        if mode == "free_running_utf8_greedy" and not np.array_equal(
                            free_outputs[prompt, role_index, : len(output)],
                            np.frombuffer(output, dtype=np.uint8),
                        ):
                            raise AssertionError("deterministic free output changed")
    for role in ROLES:
        correctness[role]["strict_free_outputs"] = PROMPT_COUNT
    arrays = {
        "ttft_ms": ttft,
        "decode_ms": decode,
        "end_to_end_ms": end_to_end,
        "emitted_output_bytes": emitted,
        "free_output_bytes": free_outputs,
        "free_output_lengths": free_lengths,
    }
    for model in models:
        model.to("cpu")
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    return arrays, correctness


def _summary_sha(payload: Mapping[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("summary_sha256", None)
    return hashlib.sha256(_json_bytes(copy)).hexdigest()


def main() -> None:
    commit = _clean_plan_commit()
    _require_never_published(OUTPUT_PATH)
    if TIMING_PATH.exists():
        raise FileExistsError("one-seed timing artifact already exists")
    plan = _read_json(PLAN_PATH)
    _validate_plan(plan)
    power_sha256 = _require_ac_power()
    baseline_nll = _verify_static_and_baseline(plan)
    context = _reconstruct_data(plan)
    started = time.time()
    with publication_mps_exclusive():
        candidate_model, candidate_nll, training_receipt = _load_or_train_candidate(
            context,
            commit=commit,
        )
        quality = summarize_one_seed_quality(
            candidate_losses_nats=candidate_nll,
            baseline_losses_nats=baseline_nll,
            document_indices=context["document_map"].document_indices,
            document_metadata=context["document_map"].metadata(),
        )
        del context["train"]
        gc.collect()
        baseline_model = build_main_model(
            geometry_spec("baseline_w72"),
            seed=SEED,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        baseline_model.load_state_dict(
            torch.load(BASELINE_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        )
        prompts, continuations = _load_cases(plan)
        arrays, correctness = _measure(
            candidate_model,
            baseline_model,
            prompts,
            continuations,
        )
    timing = summarize_one_seed_timing(
        end_to_end_ms=arrays["end_to_end_ms"],
        ttft_ms=arrays["ttft_ms"],
        decode_ms=arrays["decode_ms"],
        emitted_output_bytes=arrays["emitted_output_bytes"],
        correctness=correctness,
    )
    decision = one_seed_decision(quality, timing)
    timing_bytes = _npz_bytes(arrays)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "static_geometry_one_seed_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": decision["status"],
        "quality": quality,
        "actual_timing": timing,
        "decision": decision,
        "training_receipt": training_receipt,
        "provenance": {
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "static_result_artifact_sha256": hash_file(STATIC_RESULT_PATH),
            "power_snapshot_sha256": power_sha256,
            "runtime_environment": runtime_environment_v2(),
        },
        "raw_evidence": {
            "timing_artifact_path": TIMING_PATH.relative_to(ROOT).as_posix(),
            "timing_artifact_sha256": hashlib.sha256(timing_bytes).hexdigest(),
            "arrays_sha256": {
                key: array_sha256(value) for key, value in arrays.items()
            },
        },
        "claim_boundary": {
            "calibration_only": True,
            "one_model_seed_screen": True,
            "test_or_final_evidence_read": False,
            "publication_quality_or_efficiency_claimed": False,
            "static_geometry_novelty_claimed": False,
            "pass_authorizes_multi_seed_replication_and_conditional_method_work": True,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    summary["summary_sha256"] = _summary_sha(summary)
    _require_unchanged(commit)
    _publish_no_clobber(TIMING_PATH, timing_bytes)
    _publish_no_clobber(OUTPUT_PATH, _json_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
