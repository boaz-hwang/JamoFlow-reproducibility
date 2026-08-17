#!/usr/bin/env python3
"""Run preregistered Phase 3 delayed-grid and causal-placebo controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch

from jamoflow.neural_data import NeuralStream, build_neural_stream
from jamoflow.neural_model import build_main_model, parameter_count, research_versions
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
from jamoflow.phase3_mechanism import (
    INITIAL_SEEDS,
    MECHANISM_POLICIES,
    WHITESPACE_POLICY,
    array_sha256,
    build_mechanism_patch_matrices,
    mechanism_cache_provenance,
    merge_mechanism_manifest,
    validate_mechanism_execution_gate,
)


SPLITS = ("train", "calibration", "test")
FULL_LIMITS = {
    "train": 128_000_000,
    "calibration": 8_000_000,
    "test": 16_000_000,
}
QUICK_LIMITS = {
    "train": 131_072,
    "calibration": 32_768,
    "test": 32_768,
}
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_mapping_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    if not state:
        raise ValueError("checkpoint state must not be empty")
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint state contains an unexpected entry")
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    return _state_mapping_sha256(model.state_dict())


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state mapping: {path}")
    return _state_mapping_sha256(state)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _save_torch_state(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    state = {
        name: value.detach().to("cpu")
        for name, value in model.state_dict().items()
    }
    torch.save(state, temporary)
    temporary.replace(path)


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _load_streams(
    data_root: Path,
    limits: dict[str, int],
) -> tuple[
    dict[str, NeuralStream],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    source = data_root / "ko.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)
    streams: dict[str, NeuralStream] = {}
    inputs: dict[str, np.ndarray] = {}
    boundaries: dict[str, np.ndarray] = {}
    whitespace: dict[str, np.ndarray] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            source,
            language="ko",
            split=split,  # type: ignore[arg-type]
            byte_limit=limits[split],
            sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_boundaries = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        streams[split] = stream
        inputs[split] = split_inputs
        boundaries[split] = split_boundaries
        whitespace[split] = compact_whitespace_mask(stream.data).reshape(
            split_inputs.shape
        )
        print(
            f"data ko/{split}: {stream.selected_bytes:,} bytes, "
            f"{stream.sequence_count:,} sequences",
            flush=True,
        )
    return streams, inputs, boundaries, whitespace


def _save_matrix_cache(
    path: Path,
    matrices: dict[str, dict[str, np.ndarray]],
) -> None:
    _save_npz(
        path,
        **{
            f"{split}__{policy}": matrices[split][policy]
            for split in SPLITS
            for policy in MECHANISM_POLICIES
        },
    )


def _load_matrix_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    matrices = {split: {} for split in SPLITS}
    with np.load(path) as archive:
        expected = {
            f"{split}__{policy}"
            for split in SPLITS
            for policy in MECHANISM_POLICIES
        }
        if set(archive.files) != expected:
            raise ValueError("mechanism matrix cache has unexpected keys")
        for key in archive.files:
            split, policy = key.split("__", 1)
            matrices[split][policy] = archive[key]
    return matrices


def _matrices(
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    whitespace: dict[str, np.ndarray],
    artifact_root: Path,
    run_root: Path,
    *,
    force: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    cache_path = artifact_root / "mechanism-patches.npz"
    diagnostics_path = run_root / "mechanism-patch-diagnostics.json"
    provenance = mechanism_cache_provenance(inputs, boundaries, whitespace)
    if (
        cache_path.exists()
        and diagnostics_path.exists()
        and not force
        and _read_json(diagnostics_path).get("_provenance") == provenance
    ):
        matrices = _load_matrix_cache(cache_path)
        diagnostics = _read_json(diagnostics_path)
        for split in SPLITS:
            for policy in MECHANISM_POLICIES:
                matrix = matrices[split][policy]
                validate_padded_patch_matrix(
                    matrix, PHASE3_MODEL_SPEC.sequence_length
                )
                expected_hash = diagnostics["splits"][split][policy][
                    "matrix_sha256"
                ]
                if array_sha256(matrix) != expected_hash:
                    raise ValueError(
                        f"cached mechanism matrix hash mismatch: {split}/{policy}"
                    )
        print("loaded Phase 3 mechanism patch cache", flush=True)
        return matrices, diagnostics
    if cache_path.exists() and diagnostics_path.exists() and not force:
        print("ignoring stale Phase 3 mechanism patch cache", flush=True)

    print("constructing Phase 3 mechanism patch matrices", flush=True)
    matrices, diagnostics = build_mechanism_patch_matrices(
        inputs, boundaries, whitespace
    )
    diagnostics["_provenance"] = provenance
    _save_matrix_cache(cache_path, matrices)
    _write_json(diagnostics_path, diagnostics)
    return matrices, diagnostics


def _validate_primary_context(
    primary_summary: dict[str, Any],
    streams: dict[str, NeuralStream],
    seeds: tuple[int, ...],
    primary_run_root: Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    manifest = primary_summary.get("run_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("primary summary lacks its run manifest")
    if manifest.get("quick_smoke_only"):
        raise ValueError("quick primary evidence cannot authorize full controls")
    if manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("primary mechanism model specification mismatch")
    if manifest.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict():
        raise ValueError("primary mechanism optimization mismatch")
    if manifest.get("limits") != FULL_LIMITS:
        raise ValueError("primary mechanism byte limits mismatch")
    primary_integrity = primary_summary.get("integrity", {})
    if primary_integrity.get("all_integrity_checks_pass") is not True:
        raise ValueError("primary mechanism integrity is incomplete")
    primary_seeds = set(primary_summary.get("seeds", []))
    if not set(seeds) <= primary_seeds:
        raise ValueError("requested controls lack paired primary W seeds")

    stream_checks: dict[str, Any] = {}
    for split in SPLITS:
        actual = _sha256_bytes(streams[split].data)
        expected = manifest["streams"][split]["selected_stream_sha256"]
        if actual != expected:
            raise ValueError(f"primary stream mismatch for {split}")
        stream_checks[split] = {
            "selected_stream_sha256": actual,
            "matches_primary": True,
        }

    report_checks: dict[str, Any] = {}
    for seed in seeds:
        path = primary_run_root / f"seed-{seed}" / f"{WHITESPACE_POLICY}.json"
        report = _read_json(path)
        if (
            report.get("seed") != seed
            or report.get("policy") != WHITESPACE_POLICY
            or report.get("parameters") != 19_596_096
            or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
            or report.get("optimization_spec")
            != PHASE3_OPTIMIZATION_SPEC.to_dict()
        ):
            raise ValueError(f"primary W identity mismatch for seed {seed}")
        primary_seed_integrity = primary_integrity["by_seed"][str(seed)]
        expected_state = primary_seed_integrity["checkpoint_state_sha256"][
            WHITESPACE_POLICY
        ]
        if report.get("trained_state_sha256") != expected_state:
            raise ValueError(f"primary W state mismatch for seed {seed}")
        split_checks = {}
        for split in SPLITS:
            actual = report["patch_matrix_sha256"][split]
            expected = diagnostics["whitespace_reference"][split][
                "matrix_sha256"
            ]
            if actual != expected:
                raise ValueError(
                    f"rebuilt W matrix differs from primary: seed {seed}/{split}"
                )
            split_checks[split] = True
        report_checks[str(seed)] = {
            "identity_matches": True,
            "checkpoint_state_matches_primary_summary": True,
            "whitespace_reference_matches_all_splits": all(
                split_checks.values()
            ),
        }
    return {"streams": stream_checks, "primary_w_reports": report_checks}


def _completed_control_is_valid(
    report_path: Path,
    checkpoint_path: Path,
    loss_path: Path,
    *,
    seed: int,
    policy: str,
    save_checkpoints: bool,
    expected_patch_hashes: dict[str, str],
    expected_stream_hashes: dict[str, str],
    expected_examples: dict[str, int],
    expected_order_sha256: str,
) -> bool:
    if not report_path.exists() or not loss_path.exists():
        return False
    if save_checkpoints and not checkpoint_path.exists():
        return False
    report = _read_json(report_path)
    expected_fields = {
        "seed": seed,
        "policy": policy,
        "parameters": 19_596_096,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "training_order_sha256": expected_order_sha256,
        "patch_matrix_sha256": expected_patch_hashes,
        "stream_selected_sha256": expected_stream_hashes,
    }
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(
                f"existing control provenance mismatch ({key}): {report_path}"
            )
    if save_checkpoints:
        if _sha256_file(checkpoint_path) != report.get(
            "checkpoint_artifact_sha256"
        ):
            raise ValueError(f"existing control checkpoint artifact mismatch: {policy}")
        if _checkpoint_state_sha256(checkpoint_path) != report.get(
            "trained_state_sha256"
        ):
            raise ValueError(f"existing control checkpoint hash mismatch: {policy}")
    if _sha256_file(loss_path) != report.get("test_loss_file_sha256"):
        raise ValueError(f"existing control loss hash mismatch: {policy}")
    with np.load(loss_path) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError(f"existing control loss keys mismatch: {policy}")
        losses = archive["sequence_nll_nats"].astype(np.float64)
    if (
        losses.shape != (expected_examples["test"],)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError(f"existing control loss vector is invalid: {policy}")
    for split in ("calibration", "test"):
        evaluation = report.get("evaluation", {}).get(split, {})
        if (
            evaluation.get("examples") != expected_examples[split]
            or evaluation.get("predicted_bytes")
            != expected_examples[split] * (PHASE3_MODEL_SPEC.sequence_length - 1)
        ):
            raise ValueError(f"existing control evaluation counts mismatch: {policy}")
    predicted_bytes = int(report["evaluation"]["test"]["predicted_bytes"])
    reconstructed_bpb = float(losses.sum()) / (
        predicted_bytes * np.log(2.0)
    )
    if not np.isclose(
        reconstructed_bpb,
        float(report["evaluation"]["test"]["bpb"]),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(f"existing control loss/report mismatch: {policy}")
    return True


def _train_control(
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    boundaries: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    stream_hashes: dict[str, str],
    device: str,
    run_directory: Path,
    artifact_directory: Path,
    *,
    force: bool,
    save_checkpoints: bool,
) -> None:
    report_path = run_directory / f"{policy}.json"
    checkpoint_path = artifact_directory / f"{policy}.pt"
    loss_path = artifact_directory / f"{policy}-test-nll.npz"
    order = shuffled_indices(len(inputs["train"]), seed)
    order_sha256 = array_sha256(order)
    patch_hashes = {
        split: array_sha256(matrices[split][policy]) for split in SPLITS
    }
    expected_examples = {split: len(inputs[split]) for split in SPLITS}
    if not force and _completed_control_is_valid(
        report_path,
        checkpoint_path,
        loss_path,
        seed=seed,
        policy=policy,
        save_checkpoints=save_checkpoints,
        expected_patch_hashes=patch_hashes,
        expected_stream_hashes=stream_hashes,
        expected_examples=expected_examples,
        expected_order_sha256=order_sha256,
    ):
        print(f"seed {seed}/{policy}: already complete", flush=True)
        return

    for split in SPLITS:
        validate_padded_patch_matrix(
            matrices[split][policy], PHASE3_MODEL_SPEC.sequence_length
        )
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization_sha256 = _state_dict_sha256(model)
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
    test_losses: np.ndarray | None = None
    for split in ("calibration", "test"):
        evaluation, sequence_nll = evaluate_main_model(
            model,
            inputs[split],
            matrices[split][policy],
            device,
            batch_size=PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
            return_sequence_nll=split == "test",
        )
        evaluations[split] = evaluation.to_dict()
        if sequence_nll is not None:
            test_losses = sequence_nll
    if test_losses is None:
        raise AssertionError("control test losses were not produced")

    _save_npz(loss_path, sequence_nll_nats=test_losses)
    trained_state_sha256 = _state_dict_sha256(model)
    checkpoint_artifact_sha256 = None
    if save_checkpoints:
        _save_torch_state(checkpoint_path, model)
        checkpoint_artifact_sha256 = _sha256_file(checkpoint_path)
    report = {
        "seed": seed,
        "policy": policy,
        "language": "ko",
        "parameters": parameter_count(model),
        "initialization_sha256": initialization_sha256,
        "trained_state_sha256": trained_state_sha256,
        "training_order_sha256": order_sha256,
        "stream_selected_sha256": stream_hashes,
        "patch_matrix_sha256": patch_hashes,
        "patch_diagnostics": {
            split: variable_patch_diagnostics(
                matrices[split][policy], boundaries[split]
            ).to_dict()
            for split in SPLITS
        },
        "training": training.to_dict(),
        "evaluation": evaluations,
        "test_loss_file_sha256": _sha256_file(loss_path),
        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
    }
    _write_json(report_path, report)
    print(
        f"seed {seed}/{policy}: test BPB={evaluations['test']['bpb']:.6f}",
        flush=True,
    )
    _release_model(model, device)


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    if args.quick and seeds == INITIAL_SEEDS:
        seeds = (INITIAL_SEEDS[0],)
    primary_summary_path = Path(args.primary_summary)
    primary_summary = (
        _read_json(primary_summary_path)
        if primary_summary_path.exists()
        else None
    )
    authorization = validate_mechanism_execution_gate(
        primary_summary, seeds, quick=args.quick
    )
    limits = dict(QUICK_LIMITS if args.quick else FULL_LIMITS)
    device = resolve_device(args.device)
    run_root = Path(
        args.run_root
        or (
            "runs/phase3-mechanism-smoke"
            if args.quick
            else "runs/phase3-mechanism"
        )
    )
    artifact_root = Path(
        args.artifact_root
        or (
            "artifacts/phase3-mechanism-smoke"
            if args.quick
            else "artifacts/phase3-mechanism"
        )
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    print(
        f"device: {device}; seeds: {seeds}; gate: {authorization['status']}",
        flush=True,
    )

    streams, inputs, boundaries, whitespace = _load_streams(
        Path(args.data_root), limits
    )
    matrices, diagnostics = _matrices(
        inputs,
        boundaries,
        whitespace,
        artifact_root,
        run_root,
        force=args.force,
    )
    primary_checks = None
    if not args.quick:
        if primary_summary is None:  # pragma: no cover - gate already rejects
            raise AssertionError("missing primary summary")
        primary_checks = _validate_primary_context(
            primary_summary,
            streams,
            seeds,
            Path(args.primary_run_root),
            diagnostics,
        )

    stream_hashes = {
        split: _sha256_bytes(streams[split].data) for split in SPLITS
    }
    current_manifest = {
        "phase": "phase3_mechanism",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick_smoke_only": bool(args.quick),
        "git_commit": _git_commit(),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "language": "ko",
        "seeds": list(seeds),
        "policies": list(MECHANISM_POLICIES),
        "limits": limits,
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "gate_authorization": authorization,
        "primary_summary_sha256": (
            _sha256_file(primary_summary_path)
            if primary_summary_path.exists()
            else None
        ),
        "primary_context_checks": primary_checks,
        "force": bool(args.force),
        "save_checkpoints": not args.no_checkpoints,
        "streams": {
            split: {
                **streams[split].metadata(),
                "selected_stream_sha256": stream_hashes[split],
            }
            for split in SPLITS
        },
    }
    manifest_path = run_root / "manifest.json"
    existing = _read_json(manifest_path) if manifest_path.exists() else None
    manifest = merge_mechanism_manifest(existing, current_manifest)
    _write_json(manifest_path, manifest)

    for seed in seeds:
        seed_run = run_root / f"seed-{seed}"
        seed_artifact = artifact_root / f"seed-{seed}"
        seed_run.mkdir(parents=True, exist_ok=True)
        seed_artifact.mkdir(parents=True, exist_ok=True)
        for policy in MECHANISM_POLICIES:
            _train_control(
                seed,
                policy,
                inputs,
                boundaries,
                matrices,
                stream_hashes,
                device,
                seed_run,
                seed_artifact,
                force=args.force,
                save_checkpoints=not args.no_checkpoints,
            )
    print(f"completed Phase 3 mechanism runs under {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/processed/hplt3-korean-phase3"
    )
    parser.add_argument("--run-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--primary-run-root", default="runs/phase3")
    parser.add_argument(
        "--primary-summary",
        default="results/phase3-primary-clustered/summary.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(INITIAL_SEEDS)
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
