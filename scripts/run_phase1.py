#!/usr/bin/env python3
"""Run the preregistered compact-BLT Phase 1 experiment.

Generated corpora, checkpoints, patch caches, and per-sequence losses live in
ignored directories. A separate summarizer promotes only aggregate results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
from jamoflow.phase1 import (
    POLICIES,
    boundary_overlap,
    entropy_patch_matrices,
    fixed_patch_matrices,
    patch_diagnostics,
    selected_boundary_entropy,
    stream_arrays,
)


LANGUAGES = ("ko", "zh", "en")
SPLITS = ("train", "calibration", "test")
DEFAULT_SEEDS = (1729, 2718, 31415, 57721, 65537)
FULL_LIMITS = {
    "train": 6_000_000,
    "calibration": 500_000,
    "test": 1_000_000,
}
QUICK_LIMITS = {
    "train": 65_536,
    "calibration": 32_768,
    "test": 32_768,
}


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


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to("cpu")
        for name, value in model.state_dict().items()
    }


def _release_model(model, device: str) -> None:
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
    dict[str, dict[str, NeuralStream]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    streams: dict[str, dict[str, NeuralStream]] = {split: {} for split in SPLITS}
    inputs: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    masks: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
    for language in LANGUAGES:
        path = data_root / f"{language}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for split in SPLITS:
            stream = build_neural_stream(
                path,
                language=language,
                split=split,
                byte_limit=limits[split],
                sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
            )
            split_inputs, split_masks = stream_arrays(
                stream.data,
                stream.codepoint_boundaries,
                stream.sequence_length,
            )
            streams[split][language] = stream
            inputs[split][language] = split_inputs
            masks[split][language] = split_masks
            print(
                f"data {language}/{split}: {stream.selected_bytes:,} bytes, "
                f"{stream.sequence_count:,} sequences",
                flush=True,
            )
    return streams, inputs, masks


def _fixed_matrices(
    masks: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    return {
        split: {
            language: fixed_patch_matrices(masks[split][language])
            for language in LANGUAGES
        }
        for split in SPLITS
    }


def _cache_keys(
    matrices: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    return {
        f"{split}__{language}__{policy}": matrix
        for split in SPLITS
        for language in LANGUAGES
        for policy, matrix in matrices[split][language].items()
    }


def _load_patch_cache(path: Path):
    loaded = np.load(path)
    matrices = {
        split: {language: {} for language in LANGUAGES}
        for split in SPLITS
    }
    for key in loaded.files:
        split, language, policy = key.split("__")
        matrices[split][language][policy] = loaded[key]
    return matrices


def _router_and_patches(
    seed: int,
    inputs: dict[str, dict[str, np.ndarray]],
    masks: dict[str, dict[str, np.ndarray]],
    fixed: dict[str, dict[str, dict[str, np.ndarray]]],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    force: bool,
) -> tuple[Any, dict[str, dict[str, dict[str, np.ndarray]]]]:
    router_checkpoint = artifact_directory / "router.pt"
    router_report = run_directory / "router.json"
    patch_cache = artifact_directory / "patches.npz"
    router = build_router(seed=seed)

    if router_checkpoint.exists() and router_report.exists() and not force:
        router.load_state_dict(
            torch.load(router_checkpoint, map_location="cpu", weights_only=True)
        )
        print(f"seed {seed}: loaded router checkpoint", flush=True)
    else:
        train_inputs = np.concatenate(
            [inputs["train"][language] for language in LANGUAGES], axis=0
        )
        order = shuffled_indices(len(train_inputs), seed)
        print(
            f"seed {seed}: training router on {train_inputs.nbytes:,} bytes",
            flush=True,
        )
        training = train_router(
            router,
            train_inputs,
            order,
            device,
            DEFAULT_OPTIMIZATION_SPEC,
        )
        evaluations = {
            split: {
                language: evaluate_router(
                    router,
                    inputs[split][language],
                    device,
                ).to_dict()
                for language in LANGUAGES
            }
            for split in ("calibration", "test")
        }
        torch.save(_cpu_state_dict(router), router_checkpoint)
        _write_json(
            router_report,
            {
                "seed": seed,
                "parameters": parameter_count(router),
                "training": training.to_dict(),
                "evaluation": evaluations,
            },
        )

    if patch_cache.exists() and not force:
        matrices = _load_patch_cache(patch_cache)
        print(f"seed {seed}: loaded patch cache", flush=True)
        return router, matrices

    matrices = {
        split: {
            language: dict(fixed[split][language])
            for language in LANGUAGES
        }
        for split in SPLITS
    }
    diagnostics: dict[str, Any] = {split: {} for split in SPLITS}
    print(f"seed {seed}: scoring router and constructing patches", flush=True)
    for split in SPLITS:
        for language in LANGUAGES:
            scores = router_entropy_scores(
                router,
                inputs[split][language],
                device,
            )
            entropy_matrices = entropy_patch_matrices(
                scores,
                masks[split][language],
            )
            matrices[split][language].update(entropy_matrices)
            diagnostics[split][language] = {
                policy: {
                    **patch_diagnostics(
                        matrix,
                        masks[split][language],
                    ).to_dict(),
                    "mean_selected_router_entropy_nats": (
                        selected_boundary_entropy(matrix, scores)
                    ),
                }
                for policy, matrix in matrices[split][language].items()
            }
            diagnostics[split][language]["entropy_overlap"] = boundary_overlap(
                entropy_matrices["entropy_full"],
                entropy_matrices["entropy_codepoint"],
            )
            del scores

    np.savez_compressed(patch_cache, **_cache_keys(matrices))
    _write_json(run_directory / "patch-diagnostics.json", diagnostics)
    return router, matrices


def _policy_is_complete(run_path: Path, checkpoint: Path, save_checkpoints: bool):
    return run_path.exists() and (checkpoint.exists() or not save_checkpoints)


def _train_policy(
    seed: int,
    policy: str,
    inputs: dict[str, dict[str, np.ndarray]],
    matrices: dict[str, dict[str, dict[str, np.ndarray]]],
    device: str,
    artifact_directory: Path,
    run_directory: Path,
    force: bool,
    save_checkpoints: bool,
) -> None:
    run_path = run_directory / f"{policy}.json"
    checkpoint = artifact_directory / f"{policy}.pt"
    losses_path = artifact_directory / f"{policy}-test-nll.npz"
    if _policy_is_complete(run_path, checkpoint, save_checkpoints) and not force:
        print(f"seed {seed}/{policy}: already complete", flush=True)
        return

    train_inputs = np.concatenate(
        [inputs["train"][language] for language in LANGUAGES], axis=0
    )
    train_patches = np.concatenate(
        [matrices["train"][language][policy] for language in LANGUAGES],
        axis=0,
    )
    order = shuffled_indices(len(train_inputs), seed)
    model = build_main_model(seed=seed)
    print(
        f"seed {seed}/{policy}: training {parameter_count(model):,} parameters",
        flush=True,
    )
    training = train_main_model(
        model,
        train_inputs,
        train_patches,
        order,
        device,
        DEFAULT_OPTIMIZATION_SPEC,
    )

    evaluations: dict[str, dict[str, Any]] = {}
    test_losses: dict[str, np.ndarray] = {}
    for split in ("calibration", "test"):
        evaluations[split] = {}
        for language in LANGUAGES:
            summary, sequence_nll = evaluate_main_model(
                model,
                inputs[split][language],
                matrices[split][language][policy],
                device,
                batch_size=DEFAULT_OPTIMIZATION_SPEC.evaluation_batch_size,
                return_sequence_nll=split == "test",
            )
            evaluations[split][language] = summary.to_dict()
            if sequence_nll is not None:
                test_losses[language] = sequence_nll

    if save_checkpoints:
        torch.save(_cpu_state_dict(model), checkpoint)
    np.savez_compressed(losses_path, **test_losses)
    _write_json(
        run_path,
        {
            "seed": seed,
            "policy": policy,
            "parameters": parameter_count(model),
            "model_spec": DEFAULT_MODEL_SPEC.to_dict(),
            "optimization_spec": DEFAULT_OPTIMIZATION_SPEC.to_dict(),
            "training": training.to_dict(),
            "evaluation": evaluations,
        },
    )
    print(
        f"seed {seed}/{policy}: test BPB "
        + ", ".join(
            f"{language}={evaluations['test'][language]['bpb']:.4f}"
            for language in LANGUAGES
        ),
        flush=True,
    )
    _release_model(model, device)


def run(args: argparse.Namespace) -> int:
    limits = QUICK_LIMITS if args.quick else FULL_LIMITS
    seeds = tuple(args.seed or ([DEFAULT_SEEDS[0]] if args.quick else DEFAULT_SEEDS))
    policies = tuple(args.policy or POLICIES)
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    device = resolve_device(args.device)
    run_root = Path(args.run_root or ("runs/phase1-smoke" if args.quick else "runs/phase1"))
    artifact_root = Path(
        args.artifact_root
        or ("artifacts/phase1-smoke" if args.quick else "artifacts/phase1")
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}; seeds: {seeds}; policies: {policies}", flush=True)
    streams, inputs, masks = _load_streams(Path(args.data_root), limits)
    fixed = _fixed_matrices(masks)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quick_smoke_only": bool(args.quick),
        "git_commit": _git_commit(),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "seeds": list(seeds),
        "policies": list(policies),
        "limits": limits,
        "model_spec": DEFAULT_MODEL_SPEC.to_dict(),
        "optimization_spec": DEFAULT_OPTIMIZATION_SPEC.to_dict(),
        "streams": {
            split: {
                language: streams[split][language].metadata()
                for language in LANGUAGES
            }
            for split in SPLITS
        },
    }
    _write_json(run_root / "manifest.json", metadata)

    for seed in seeds:
        seed_run = run_root / f"seed-{seed}"
        seed_artifact = artifact_root / f"seed-{seed}"
        seed_run.mkdir(parents=True, exist_ok=True)
        seed_artifact.mkdir(parents=True, exist_ok=True)
        router, matrices = _router_and_patches(
            seed,
            inputs,
            masks,
            fixed,
            device,
            seed_artifact,
            seed_run,
            args.force,
        )
        _release_model(router, device)
        for policy in policies:
            _train_policy(
                seed,
                policy,
                inputs,
                matrices,
                device,
                seed_artifact,
                seed_run,
                args.force,
                not args.no_checkpoints,
            )

    print(f"completed Phase 1 runs under {run_root}", flush=True)
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
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
