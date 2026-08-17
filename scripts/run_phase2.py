#!/usr/bin/env python3
"""Run the preregistered Korean-only compact-BLT Phase 2 experiment.

Raw corpora, checkpoints, patch matrices, entropy scores, and per-sequence
losses remain in ignored directories. Only aggregate results are promoted by
the separate summarization step.
"""

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
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    build_router,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import (
    DEFAULT_OPTIMIZATION_SPEC,
    evaluate_main_model,
    evaluate_router,
    resolve_device,
    router_entropy_scores,
    shuffled_indices,
    train_main_model,
    train_router,
)
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    PHASE2_POLICIES,
    STRUCTURAL_POLICIES,
    THRESHOLD_POLICIES,
    calibrate_threshold,
    compact_delimiter_mask,
    structural_patch_matrices,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)


SPLITS = ("train", "calibration", "test")
DEFAULT_SEEDS = (1729, 2718, 31415, 57721, 65537)
FULL_LIMITS = {
    "train": 11_000_000,
    "calibration": 1_000_000,
    "test": 1_000_000,
}
QUICK_LIMITS = {
    "train": 65_536,
    "calibration": 32_768,
    "test": 32_768,
}
GLOBAL_POSITION_LIMIT = DEFAULT_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cpu_state_dict(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to("cpu")
        for name, value in model.state_dict().items()
    }


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
    path = data_root / "ko.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    streams: dict[str, NeuralStream] = {}
    inputs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    delimiter_masks: dict[str, np.ndarray] = {}
    for split in SPLITS:
        stream = build_neural_stream(
            path,
            language="ko",
            split=split,
            byte_limit=limits[split],
            sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
        )
        split_inputs, split_masks = stream_arrays(
            stream.data,
            stream.codepoint_boundaries,
            stream.sequence_length,
        )
        # Scan the continuous stream before reshaping. This preserves causal
        # parser state at arbitrary 256-byte chunk starts.
        split_delimiters = compact_delimiter_mask(stream.data).reshape(
            -1,
            stream.sequence_length,
        )
        streams[split] = stream
        inputs[split] = split_inputs
        masks[split] = split_masks
        delimiter_masks[split] = split_delimiters
        print(
            f"data ko/{split}: {stream.selected_bytes:,} bytes, "
            f"{stream.sequence_count:,} sequences",
            flush=True,
        )
    return streams, inputs, masks, delimiter_masks


def _build_structural(
    masks: dict[str, np.ndarray],
    delimiter_masks: dict[str, np.ndarray],
    run_root: Path,
) -> dict[str, dict[str, np.ndarray]]:
    matrices = {
        split: structural_patch_matrices(masks[split], delimiter_masks[split])
        for split in SPLITS
    }
    diagnostics = {
        split: {
            policy: {
                **variable_patch_diagnostics(
                    matrices[split][policy],
                    masks[split],
                ).to_dict(),
                "matrix_sha256": _array_sha256(matrices[split][policy]),
            }
            for policy in STRUCTURAL_POLICIES
        }
        for split in SPLITS
    }
    _write_json(run_root / "structural-patch-diagnostics.json", diagnostics)
    return matrices


def _ensure_router(
    seed: int,
    inputs: dict[str, np.ndarray],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    force: bool,
) -> Any:
    checkpoint = artifact_directory / "router.pt"
    report_path = run_directory / "router.json"
    router = build_router(seed=seed)
    initialization_sha256 = _state_dict_sha256(router)
    if checkpoint.exists() and report_path.exists() and not force:
        router.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        print(f"seed {seed}: loaded Korean router checkpoint", flush=True)
        return router

    order = shuffled_indices(len(inputs["train"]), seed)
    print(
        f"seed {seed}: training Korean router on "
        f"{inputs['train'].nbytes:,} bytes",
        flush=True,
    )
    training = train_router(
        router,
        inputs["train"],
        order,
        device,
        DEFAULT_OPTIMIZATION_SPEC,
    )
    evaluations = {
        split: evaluate_router(router, inputs[split], device).to_dict()
        for split in ("calibration", "test")
    }
    torch.save(_cpu_state_dict(router), checkpoint)
    _write_json(
        report_path,
        {
            "seed": seed,
            "language": "ko",
            "parameters": parameter_count(router),
            "initialization_sha256": initialization_sha256,
            "training_order_sha256": _array_sha256(order),
            "training": training.to_dict(),
            "evaluation": evaluations,
        },
    )
    return router


def _threshold_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    loaded = np.load(path)
    matrices = {split: {} for split in SPLITS}
    for key in loaded.files:
        split, policy = key.split("__")
        matrices[split][policy] = loaded[key]
    return matrices


def _ensure_threshold_matrices(
    seed: int,
    router: Any,
    inputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    force: bool,
) -> dict[str, dict[str, np.ndarray]]:
    cache_path = artifact_directory / "threshold-patches.npz"
    diagnostics_path = run_directory / "threshold-patch-diagnostics.json"
    if cache_path.exists() and diagnostics_path.exists() and not force:
        print(f"seed {seed}: loaded threshold patch cache", flush=True)
        return _threshold_cache(cache_path)

    print(f"seed {seed}: scoring Korean calibration entropy", flush=True)
    calibration_scores = router_entropy_scores(
        router,
        inputs["calibration"],
        device,
    )
    full_calibration = calibrate_threshold(
        calibration_scores,
        DEFAULT_MODEL_SPEC.patch_count,
        maximum_patch_length=24,
    )
    codepoint_calibration = calibrate_threshold(
        calibration_scores,
        DEFAULT_MODEL_SPEC.patch_count,
        candidate_masks=masks["calibration"],
        maximum_patch_length=24,
    )
    calibrations = {
        "entropy_threshold_full": full_calibration,
        "entropy_threshold_codepoint": codepoint_calibration,
    }
    matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    diagnostics: dict[str, Any] = {
        "seed": seed,
        "calibration": {
            policy: calibration.to_dict()
            for policy, calibration in calibrations.items()
        },
        "splits": {},
    }
    for split in SPLITS:
        scores = (
            calibration_scores
            if split == "calibration"
            else router_entropy_scores(router, inputs[split], device)
        )
        matrices[split]["entropy_threshold_full"] = threshold_patch_matrix(
            scores,
            full_calibration.threshold_nats,
            maximum_patch_length=24,
        )
        matrices[split]["entropy_threshold_codepoint"] = threshold_patch_matrix(
            scores,
            codepoint_calibration.threshold_nats,
            candidate_masks=masks[split],
            maximum_patch_length=24,
        )
        diagnostics["splits"][split] = {
            policy: {
                **variable_patch_diagnostics(
                    matrices[split][policy],
                    masks[split],
                ).to_dict(),
                "matrix_sha256": _array_sha256(matrices[split][policy]),
            }
            for policy in THRESHOLD_POLICIES
        }
        del scores

    np.savez_compressed(
        cache_path,
        **{
            f"{split}__{policy}": matrices[split][policy]
            for split in SPLITS
            for policy in THRESHOLD_POLICIES
        },
    )
    _write_json(diagnostics_path, diagnostics)
    return matrices


def _policy_complete(
    run_path: Path,
    checkpoint: Path,
    save_checkpoints: bool,
) -> bool:
    return run_path.exists() and (checkpoint.exists() or not save_checkpoints)


def _train_policy(
    seed: int,
    policy: str,
    inputs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    matrices: dict[str, dict[str, np.ndarray]],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    force: bool,
    save_checkpoints: bool,
) -> None:
    run_path = run_directory / f"{policy}.json"
    checkpoint = artifact_directory / f"{policy}.pt"
    losses_path = artifact_directory / f"{policy}-test-nll.npz"
    if _policy_complete(run_path, checkpoint, save_checkpoints) and not force:
        print(f"seed {seed}/{policy}: already complete", flush=True)
        return

    for split in SPLITS:
        validate_padded_patch_matrix(
            matrices[split][policy],
            DEFAULT_MODEL_SPEC.sequence_length,
        )
    order = shuffled_indices(len(inputs["train"]), seed)
    model = build_main_model(
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    initialization_sha256 = _state_dict_sha256(model)
    print(
        f"seed {seed}/{policy}: training {parameter_count(model):,} parameters; "
        f"matrix width={matrices['train'][policy].shape[1]}",
        flush=True,
    )
    training = train_main_model(
        model,
        inputs["train"],
        matrices["train"][policy],
        order,
        device,
        DEFAULT_OPTIMIZATION_SPEC,
    )

    evaluations: dict[str, Any] = {}
    test_losses: np.ndarray | None = None
    for split in ("calibration", "test"):
        summary, sequence_nll = evaluate_main_model(
            model,
            inputs[split],
            matrices[split][policy],
            device,
            batch_size=DEFAULT_OPTIMIZATION_SPEC.evaluation_batch_size,
            return_sequence_nll=split == "test",
        )
        evaluations[split] = summary.to_dict()
        if sequence_nll is not None:
            test_losses = sequence_nll

    if save_checkpoints:
        torch.save(_cpu_state_dict(model), checkpoint)
    if test_losses is None:
        raise AssertionError("test evaluation did not return per-sequence losses")
    np.savez_compressed(losses_path, ko=test_losses)
    _write_json(
        run_path,
        {
            "seed": seed,
            "policy": policy,
            "language": "ko",
            "parameters": parameter_count(model),
            "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
            "initialization_sha256": initialization_sha256,
            "training_order_sha256": _array_sha256(order),
            "patch_matrix_sha256": {
                split: _array_sha256(matrices[split][policy])
                for split in SPLITS
            },
            "patch_diagnostics": {
                split: variable_patch_diagnostics(
                    matrices[split][policy],
                    masks[split],
                ).to_dict()
                for split in SPLITS
            },
            "model_spec": DEFAULT_MODEL_SPEC.to_dict(),
            "optimization_spec": DEFAULT_OPTIMIZATION_SPEC.to_dict(),
            "training": training.to_dict(),
            "evaluation": evaluations,
        },
    )
    print(
        f"seed {seed}/{policy}: test BPB={evaluations['test']['bpb']:.6f}",
        flush=True,
    )
    _release_model(model, device)


def run(args: argparse.Namespace) -> int:
    limits = QUICK_LIMITS if args.quick else FULL_LIMITS
    seeds = tuple(args.seed or ([DEFAULT_SEEDS[0]] if args.quick else DEFAULT_SEEDS))
    policies = tuple(args.policy or PHASE2_POLICIES)
    unknown = set(policies) - set(PHASE2_POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    device = resolve_device(args.device)
    run_root = Path(args.run_root or ("runs/phase2-smoke" if args.quick else "runs/phase2"))
    artifact_root = Path(
        args.artifact_root
        or ("artifacts/phase2-smoke" if args.quick else "artifacts/phase2")
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}; seeds: {seeds}; policies: {policies}", flush=True)
    streams, inputs, masks, delimiter_masks = _load_streams(
        Path(args.data_root),
        limits,
    )
    structural = _build_structural(masks, delimiter_masks, run_root)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick_smoke_only": bool(args.quick),
        "git_commit": _git_commit(),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "language": "ko",
        "seeds": list(seeds),
        "policies": list(policies),
        "limits": limits,
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "model_spec": DEFAULT_MODEL_SPEC.to_dict(),
        "optimization_spec": DEFAULT_OPTIMIZATION_SPEC.to_dict(),
        "streams": {
            split: {
                **streams[split].metadata(),
                "selected_stream_sha256": _sha256_bytes(streams[split].data),
            }
            for split in SPLITS
        },
    }
    _write_json(run_root / "manifest.json", manifest)

    needs_thresholds = bool(set(policies) & set(THRESHOLD_POLICIES))
    for seed in seeds:
        seed_run = run_root / f"seed-{seed}"
        seed_artifact = artifact_root / f"seed-{seed}"
        seed_run.mkdir(parents=True, exist_ok=True)
        seed_artifact.mkdir(parents=True, exist_ok=True)
        matrices = {
            split: dict(structural[split])
            for split in SPLITS
        }
        if needs_thresholds:
            router = _ensure_router(
                seed,
                inputs,
                device,
                seed_artifact,
                seed_run,
                args.force,
            )
            threshold = _ensure_threshold_matrices(
                seed,
                router,
                inputs,
                masks,
                device,
                seed_artifact,
                seed_run,
                args.force,
            )
            for split in SPLITS:
                matrices[split].update(threshold[split])
            _release_model(router, device)

        for policy in policies:
            _train_policy(
                seed,
                policy,
                inputs,
                masks,
                matrices,
                device,
                seed_artifact,
                seed_run,
                args.force,
                not args.no_checkpoints,
            )

    print(f"completed Phase 2 primary runs under {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--run-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--policy", action="append", choices=PHASE2_POLICIES)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
