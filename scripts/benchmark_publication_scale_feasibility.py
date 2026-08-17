#!/usr/bin/env python3
"""Run candidate-only Mac preflight without authorizing publication training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from typing import Any

import numpy as np
import torch

from jamoflow.compute_conversion import (
    conversion_patch_matrices,
    conversion_policy,
)
from jamoflow.incremental_blt import IncrementalBltDecoder
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import build_main_model, parameter_count
from jamoflow.neural_training import synchronize
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.publication_scale import (
    PUBLICATION_BATCH_SIZE,
    PUBLICATION_EVALUATION_BATCH_SIZE,
    PUBLICATION_EXPECTED_PARAMETERS,
    PUBLICATION_PROJECTED_TRAIN_STEPS,
    PUBLICATION_SCALE_ORDER,
    PUBLICATION_SEQUENCE_LENGTH,
    PUBLICATION_TRAIN_BYTES,
    ScaleFeasibility,
    publication_model_spec,
    select_largest_feasible_scale,
)


GLOBAL_POSITION_LIMIT = PUBLICATION_SEQUENCE_LENGTH * 2 + 8
TRAIN_WARMUP_STEPS = 1
TRAIN_MEASUREMENT_STEPS = 3
FEASIBILITY_SEED = 1729


def _read_json(path: Path) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
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


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def _artifact(path: Path, filename: str) -> dict[str, int | str]:
    return {
        "filename": filename,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_authorization(
    actual_summary_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    actual = _read_json(actual_summary_path)
    selection = _read_json(selection_path)
    candidate = selection.get("candidate", {})
    rate = candidate.get("patch_count")
    if (
        actual.get("schema_version") != 2
        or actual.get("integrity", {}).get("all_integrity_checks_pass")
        is not True
        or actual.get("compact_actual_inference_gate", {}).get("overall_pass")
        is not True
        or actual.get("selection", {}).get("sha256") != _sha256(selection_path)
        or actual.get("candidate") != candidate
        or not isinstance(rate, int)
        or candidate.get("policy") != conversion_policy("whitespace", rate)
    ):
        raise ValueError("publication feasibility is blocked by the compact gate")
    quality_item = actual.get("quality_summary", {})
    quality_path = Path(str(quality_item.get("path", "")))
    if not quality_path.is_file() or quality_item.get("sha256") != _sha256(
        quality_path
    ):
        raise ValueError("compact quality-summary lineage changed")
    quality = _read_json(quality_path)
    if (
        quality.get("selection", {}).get("sha256") != _sha256(selection_path)
        or quality.get("candidate") != candidate
        or quality.get("quality_noninferiority", {}).get("overall_pass")
        is not True
        or quality.get("integrity", {}).get("all_integrity_checks_pass")
        is not True
    ):
        raise ValueError("compact quality authorization does not reconstruct")
    return actual, selection, rate


def _validate_source_context(
    actual: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
    expected = actual.get("integrity", {}).get("case_context", {})
    source_artifact = _artifact(source_path, "ko.jsonl")
    integrity_artifact = _artifact(integrity_path, "integrity.json")
    if (
        expected.get("source_artifact") != source_artifact
        or expected.get("source_integrity_artifact") != integrity_artifact
    ):
        raise ValueError("publication feasibility source artifacts changed")
    integrity = _read_json(integrity_path)
    if (
        integrity.get("dataset_id") != "hplt3-korean-phase3"
        or integrity.get("output", {}).get("output_bytes")
        != source_artifact["bytes"]
        or integrity.get("output", {}).get("output_sha256")
        != source_artifact["sha256"]
    ):
        raise ValueError("publication feasibility source integrity failed")
    return {
        "source_artifact": source_artifact,
        "source_integrity_artifact": integrity_artifact,
        "processed_integrity_dataset_id": integrity["dataset_id"],
    }


def _worker_data(
    data_root: Path,
    rate: int,
    actual: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_context = _validate_source_context(actual, data_root)
    source_path = data_root / "ko.jsonl"
    byte_limit = PUBLICATION_EVALUATION_BATCH_SIZE * PUBLICATION_SEQUENCE_LENGTH
    stream = build_neural_stream(
        source_path,
        language="ko",
        split="train",
        byte_limit=byte_limit,
        sequence_length=PUBLICATION_SEQUENCE_LENGTH,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    matrices = conversion_patch_matrices(
        boundaries,
        whitespace,
        rate=rate,
    )
    policy = conversion_policy("whitespace", rate)
    patches = matrices[policy]
    expected_input_shape = (
        PUBLICATION_EVALUATION_BATCH_SIZE,
        PUBLICATION_SEQUENCE_LENGTH,
    )
    if inputs.shape != expected_input_shape or patches.shape != (
        PUBLICATION_EVALUATION_BATCH_SIZE,
        rate,
    ):
        raise ValueError("publication feasibility batch is incomplete")
    return inputs, patches, {
        **source_context,
        "selected_stream_sha256": hashlib.sha256(stream.data).hexdigest(),
        "stream_metadata": stream.metadata(),
        "inputs_sha256": _array_sha256(inputs),
        "patch_matrix_sha256": _array_sha256(patches),
        "examples": len(inputs),
        "policy": policy,
        "rate": rate,
    }


def _memory_snapshot() -> dict[str, int]:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        "recommended_max_memory_bytes": int(torch.mps.recommended_max_memory()),
    }


def _train_step(
    model: Any,
    optimizer: Any,
    batch: Any,
    patches: Any,
) -> tuple[float, bool]:
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    output = model(
        input_ids=batch,
        patch_lengths=patches,
        labels=batch,
        use_cache=False,
    )
    loss = output.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    synchronize("mps")
    return time.perf_counter() - started, bool(torch.isfinite(loss).item())


def _run_worker(args: argparse.Namespace) -> int:
    if not torch.backends.mps.is_available():
        raise RuntimeError("publication feasibility worker requires Apple MPS")
    if not _git_is_clean():
        raise RuntimeError("publication feasibility requires a clean source tree")
    actual_path = Path(args.actual_summary)
    selection_path = Path(args.selection)
    actual, _, rate = _validate_authorization(actual_path, selection_path)
    source_commit = _git_commit()
    if not source_commit:
        raise RuntimeError("publication feasibility requires a Git commit")
    target = int(args.worker_target)
    spec = publication_model_spec(target, rate)
    inputs, patch_lengths, data_context = _worker_data(
        Path(args.data_root),
        rate,
        actual,
    )
    model = build_main_model(
        spec,
        seed=FEASIBILITY_SEED,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    ).to("mps")
    count = parameter_count(model)
    if count != PUBLICATION_EXPECTED_PARAMETERS[target]:
        raise ValueError("publication candidate parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    batch = torch.from_numpy(
        inputs[:PUBLICATION_BATCH_SIZE].astype(np.int64, copy=False)
    ).to("mps")
    train_patches = torch.from_numpy(
        patch_lengths[:PUBLICATION_BATCH_SIZE].astype(np.int64, copy=False)
    ).to("mps")
    memory = [_memory_snapshot()]
    training_finite = True
    model.train()
    for _ in range(TRAIN_WARMUP_STEPS):
        elapsed, step_finite = _train_step(model, optimizer, batch, train_patches)
        training_finite &= (
            step_finite and math.isfinite(elapsed) and elapsed > 0
        )
        memory.append(_memory_snapshot())
    train_seconds = []
    for _ in range(TRAIN_MEASUREMENT_STEPS):
        elapsed, step_finite = _train_step(model, optimizer, batch, train_patches)
        train_seconds.append(elapsed)
        training_finite &= (
            step_finite and math.isfinite(elapsed) and elapsed > 0
        )
        memory.append(_memory_snapshot())

    model.eval()
    evaluation_batch = torch.from_numpy(
        inputs[:PUBLICATION_EVALUATION_BATCH_SIZE].astype(np.int64, copy=False)
    ).to("mps")
    evaluation_patches = torch.from_numpy(
        patch_lengths[:PUBLICATION_EVALUATION_BATCH_SIZE].astype(
            np.int64,
            copy=False,
        )
    ).to("mps")
    synchronize("mps")
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        evaluation = model(
            input_ids=evaluation_batch,
            patch_lengths=evaluation_patches,
            labels=evaluation_batch,
            use_cache=False,
        )
    synchronize("mps")
    evaluation_seconds = time.perf_counter() - evaluation_started
    evaluation_finite = bool(
        torch.isfinite(evaluation.loss).item()
        and math.isfinite(evaluation_seconds)
        and evaluation_seconds > 0
    )
    memory.append(_memory_snapshot())

    del optimizer, evaluation, evaluation_batch, evaluation_patches
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.mps.empty_cache()
    prompt = bytes(inputs[0, :128])
    runtime = IncrementalBltDecoder(
        model,
        "causal_whitespace_grid",
        horizon=PUBLICATION_SEQUENCE_LENGTH,
        patch_count=rate,
        fixed_stride=spec.patch_stride,
    )
    synchronize("mps")
    incremental_started = time.perf_counter()
    with torch.inference_mode():
        logits = runtime.prefill_parallel(prompt)
        logits = runtime.consume(int(inputs[0, 128]))
    synchronize("mps")
    incremental_seconds = time.perf_counter() - incremental_started
    incremental_finite = bool(
        torch.isfinite(logits).all().item()
        and math.isfinite(incremental_seconds)
        and incremental_seconds > 0
    )
    memory.append(_memory_snapshot())

    maximum_driver = max(value["driver_allocated_bytes"] for value in memory)
    recommended = min(value["recommended_max_memory_bytes"] for value in memory)
    median_step = float(np.median(train_seconds))
    projected_steps = math.ceil(
        PUBLICATION_TRAIN_BYTES
        / (PUBLICATION_SEQUENCE_LENGTH * PUBLICATION_BATCH_SIZE)
    )
    if projected_steps != PUBLICATION_PROJECTED_TRAIN_STEPS:
        raise ValueError("publication projected-step constant changed")
    projected_hours = median_step * projected_steps / 3600
    all_finite = training_finite and evaluation_finite and incremental_finite
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": source_commit,
        "source_tree_clean": True,
        "authorization_summary_sha256": _sha256(actual_path),
        "selection_sha256": _sha256(selection_path),
        "target_millions": target,
        "model_spec": spec.to_dict(),
        "parameter_count": count,
        "completed": True,
        "finite_steps": all_finite,
        "training": {
            "batch_size": PUBLICATION_BATCH_SIZE,
            "finite": training_finite,
            "warmup_steps": TRAIN_WARMUP_STEPS,
            "measurement_steps": TRAIN_MEASUREMENT_STEPS,
            "measurement_seconds": train_seconds,
            "median_step_seconds": median_step,
            "projected_steps_for_256m_bytes": projected_steps,
            "projected_hours_per_model": projected_hours,
        },
        "evaluation": {
            "batch_size": PUBLICATION_EVALUATION_BATCH_SIZE,
            "elapsed_seconds": evaluation_seconds,
            "finite": evaluation_finite,
        },
        "incremental": {
            "prompt_bytes": 128,
            "decode_bytes": 1,
            "elapsed_seconds": incremental_seconds,
            "finite": incremental_finite,
        },
        "memory_snapshots": memory,
        "maximum_driver_allocated_bytes": maximum_driver,
        "recommended_max_memory_bytes": recommended,
        "data_context": data_context,
        "quality_used_for_selection": False,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": "mps",
        },
    }
    _write_json(Path(args.worker_output), payload)
    return 0


def _worker_command(
    args: argparse.Namespace,
    target: int,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-target",
        str(target),
        "--worker-output",
        str(output),
        "--actual-summary",
        str(args.actual_summary),
        "--selection",
        str(args.selection),
        "--data-root",
        str(args.data_root),
    ]


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _validate_worker_report(
    report: dict[str, Any],
    *,
    target: int,
    actual_sha256: str,
    selection_sha256: str,
    rate: int,
    data_context: dict[str, Any],
    git_commit: str,
) -> None:
    expected_spec = publication_model_spec(target, rate).to_dict()
    common_valid = bool(
        report.get("schema_version") == 1
        and report.get("authorization_summary_sha256") == actual_sha256
        and report.get("selection_sha256") == selection_sha256
        and report.get("target_millions") == target
        and report.get("model_spec") == expected_spec
        and report.get("parameter_count")
        == PUBLICATION_EXPECTED_PARAMETERS[target]
        and report.get("git_commit") == git_commit
        and report.get("source_tree_clean") is True
        and report.get("data_context") == data_context
        and report.get("quality_used_for_selection") is False
    )
    if not common_valid:
        raise ValueError(f"publication worker provenance differs: {target}")
    completed = report.get("completed")
    if completed is False:
        if (
            report.get("finite_steps") is not False
            or report.get("training", {}).get("projected_hours_per_model")
            is not None
            or report.get("maximum_driver_allocated_bytes") != 0
            or report.get("recommended_max_memory_bytes") != 0
            or not isinstance(report.get("failure"), dict)
        ):
            raise ValueError(f"publication worker failure record differs: {target}")
        return
    if completed is not True:
        raise ValueError(f"publication worker completion is invalid: {target}")

    training = report.get("training", {})
    measurement_seconds = training.get("measurement_seconds")
    if (
        training.get("batch_size") != PUBLICATION_BATCH_SIZE
        or training.get("warmup_steps") != TRAIN_WARMUP_STEPS
        or training.get("measurement_steps") != TRAIN_MEASUREMENT_STEPS
        or not isinstance(measurement_seconds, list)
        or len(measurement_seconds) != TRAIN_MEASUREMENT_STEPS
        or not all(_positive_finite(value) for value in measurement_seconds)
        or training.get("projected_steps_for_256m_bytes")
        != PUBLICATION_PROJECTED_TRAIN_STEPS
        or not isinstance(training.get("finite"), bool)
        or not _positive_finite(training.get("median_step_seconds"))
        or not _positive_finite(training.get("projected_hours_per_model"))
    ):
        raise ValueError(f"publication worker training record differs: {target}")
    median_step = float(np.median(measurement_seconds))
    projected_hours = (
        median_step * PUBLICATION_PROJECTED_TRAIN_STEPS / 3600
    )
    if (
        not math.isclose(
            float(training.get("median_step_seconds", math.nan)),
            median_step,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(training.get("projected_hours_per_model", math.nan)),
            projected_hours,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"publication worker projection differs: {target}")

    evaluation = report.get("evaluation", {})
    incremental = report.get("incremental", {})
    expected_finite = bool(
        training["finite"]
        and evaluation.get("finite")
        and incremental.get("finite")
    )
    if (
        evaluation.get("batch_size") != PUBLICATION_EVALUATION_BATCH_SIZE
        or not isinstance(evaluation.get("finite"), bool)
        or not _positive_finite(evaluation.get("elapsed_seconds"))
        or incremental.get("prompt_bytes") != 128
        or incremental.get("decode_bytes") != 1
        or not isinstance(incremental.get("finite"), bool)
        or not _positive_finite(incremental.get("elapsed_seconds"))
        or not isinstance(report.get("finite_steps"), bool)
        or report.get("finite_steps") != expected_finite
    ):
        raise ValueError(f"publication worker runtime record differs: {target}")

    memory = report.get("memory_snapshots")
    expected_snapshots = (
        1 + TRAIN_WARMUP_STEPS + TRAIN_MEASUREMENT_STEPS + 1 + 1
    )
    if not isinstance(memory, list) or len(memory) != expected_snapshots:
        raise ValueError(f"publication worker memory record differs: {target}")
    memory_keys = {
        "current_allocated_bytes",
        "driver_allocated_bytes",
        "recommended_max_memory_bytes",
    }
    for snapshot in memory:
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != memory_keys
            or any(
                not isinstance(snapshot[key], int) or snapshot[key] < 0
                for key in memory_keys
            )
            or snapshot["recommended_max_memory_bytes"] <= 0
        ):
            raise ValueError(f"publication worker memory values differ: {target}")
    maximum_driver = max(item["driver_allocated_bytes"] for item in memory)
    recommended = min(item["recommended_max_memory_bytes"] for item in memory)
    if (
        report.get("maximum_driver_allocated_bytes") != maximum_driver
        or report.get("recommended_max_memory_bytes") != recommended
        or report.get("environment", {}).get("device") != "mps"
    ):
        raise ValueError(f"publication worker MPS evidence differs: {target}")


def _failed_worker_report(
    *,
    target: int,
    actual_sha256: str,
    selection_sha256: str,
    rate: int,
    data_context: dict[str, Any],
    git_commit: str,
    returncode: int,
    stdout_tail: str,
    stderr_tail: str,
    validation_error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "source_tree_clean": True,
        "authorization_summary_sha256": actual_sha256,
        "selection_sha256": selection_sha256,
        "target_millions": target,
        "model_spec": publication_model_spec(target, rate).to_dict(),
        "parameter_count": PUBLICATION_EXPECTED_PARAMETERS[target],
        "completed": False,
        "finite_steps": False,
        "training": {"projected_hours_per_model": None},
        "maximum_driver_allocated_bytes": 0,
        "recommended_max_memory_bytes": 0,
        "data_context": data_context,
        "quality_used_for_selection": False,
        "failure": {
            "worker_returncode": returncode,
            "worker_stdout_tail": stdout_tail,
            "worker_stderr_tail": stderr_tail,
            "worker_validation_error": validation_error,
        },
    }


def _result_from_report(target: int, report: dict[str, Any]) -> ScaleFeasibility:
    projected = report.get("training", {}).get("projected_hours_per_model")
    return ScaleFeasibility(
        target_millions=target,
        completed=report.get("completed") is True,
        finite_steps=report.get("finite_steps") is True,
        parameter_count=int(report.get("parameter_count", 0)),
        maximum_driver_allocated_bytes=int(
            report.get("maximum_driver_allocated_bytes", 0)
        ),
        recommended_max_memory_bytes=int(
            report.get("recommended_max_memory_bytes", 0)
        ),
        projected_hours_per_model=(
            math.inf if projected is None else float(projected)
        ),
    )


def run(args: argparse.Namespace) -> int:
    if not _git_is_clean():
        raise RuntimeError("publication feasibility requires a clean source tree")
    actual_path = Path(args.actual_summary)
    selection_path = Path(args.selection)
    actual, _, rate = _validate_authorization(actual_path, selection_path)
    git_commit = _git_commit()
    if not git_commit:
        raise RuntimeError("publication feasibility requires a Git commit")
    actual_sha256 = _sha256(actual_path)
    selection_sha256 = _sha256(selection_path)
    _, _, expected_data_context = _worker_data(
        Path(args.data_root),
        rate,
        actual,
    )
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    reports: dict[int, dict[str, Any]] = {}
    results: dict[int, ScaleFeasibility] = {}
    for target in PUBLICATION_SCALE_ORDER:
        output = run_root / f"target-{target}.json"
        if output.exists() and not args.force:
            report = _read_json(output)
            _validate_worker_report(
                report,
                target=target,
                actual_sha256=actual_sha256,
                selection_sha256=selection_sha256,
                rate=rate,
                data_context=expected_data_context,
                git_commit=git_commit,
            )
        else:
            worker_output = output.with_name(
                f".{output.name}.{uuid.uuid4().hex}.worker"
            )
            completed = subprocess.run(
                _worker_command(args, target, worker_output),
                check=False,
                capture_output=True,
                text=True,
            )
            report = None
            validation_error = (
                None
                if worker_output.is_file()
                else "worker output was not created"
            )
            if completed.returncode == 0 and worker_output.is_file():
                try:
                    candidate_report = _read_json(worker_output)
                    _validate_worker_report(
                        candidate_report,
                        target=target,
                        actual_sha256=actual_sha256,
                        selection_sha256=selection_sha256,
                        rate=rate,
                        data_context=expected_data_context,
                        git_commit=git_commit,
                    )
                    report = candidate_report
                except (OSError, TypeError, ValueError) as exc:
                    validation_error = f"{type(exc).__name__}: {exc}"
            if report is None:
                report = _failed_worker_report(
                    target=target,
                    actual_sha256=actual_sha256,
                    selection_sha256=selection_sha256,
                    rate=rate,
                    data_context=expected_data_context,
                    git_commit=git_commit,
                    returncode=completed.returncode,
                    stdout_tail=completed.stdout[-4000:],
                    stderr_tail=completed.stderr[-4000:],
                    validation_error=validation_error,
                )
                _write_json(output, report)
            else:
                worker_output.replace(output)
            if worker_output.exists():
                worker_output.unlink()
            worker_partial = worker_output.with_suffix(
                worker_output.suffix + ".part"
            )
            if worker_partial.exists():
                worker_partial.unlink()
            _validate_worker_report(
                report,
                target=target,
                actual_sha256=actual_sha256,
                selection_sha256=selection_sha256,
                rate=rate,
                data_context=expected_data_context,
                git_commit=git_commit,
            )
        reports[target] = report
        results[target] = _result_from_report(target, report)
        print(
            f"publication feasibility {target}M: "
            f"{'pass' if results[target].passes else 'fail'}",
            flush=True,
        )
    provisional = select_largest_feasible_scale(results)
    summary = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "source_tree_clean": True,
        "actual_inference_summary": {
            "path": str(actual_path),
            "sha256": _sha256(actual_path),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
        },
        "candidate_order": list(PUBLICATION_SCALE_ORDER),
        "by_target": {
            str(target): {
                **results[target].to_dict(),
                "worker_report_sha256": _sha256(
                    run_root / f"target-{target}.json"
                ),
            }
            for target in PUBLICATION_SCALE_ORDER
        },
        "provisional_selected_target_millions": provisional,
        "overall_preflight_pass": provisional is not None,
        "family_aware_campaign_lock": False,
        "publication_training_authorized": False,
        "status": (
            "provisional_candidate_preflight_pass"
            if provisional is not None
            else "fail_no_candidate_preflight_scale"
        ),
        "quality_used_for_selection": False,
        "interpretation_guardrail": (
            "Candidate-only preflight cannot select the publication scale. "
            "All four runtime families require a separate campaign lock."
        ),
    }
    _write_json(Path(args.output), summary)
    print(
        json.dumps(
            {"provisional_selected_target_millions": provisional},
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actual-summary",
        default="results/phase3-actual-inference/summary.json",
    )
    parser.add_argument(
        "--selection",
        default="results/phase3-inference-selection/selection.json",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument(
        "--run-root",
        default="runs/publication-scale-feasibility",
    )
    parser.add_argument(
        "--output",
        default="results/publication-scale-feasibility/summary.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-target", type=int, choices=PUBLICATION_SCALE_ORDER)
    parser.add_argument("--worker-output")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.worker_target is not None:
        if parsed.worker_output is None:
            raise SystemExit("--worker-output is required with --worker-target")
        raise SystemExit(_run_worker(parsed))
    raise SystemExit(run(parsed))
