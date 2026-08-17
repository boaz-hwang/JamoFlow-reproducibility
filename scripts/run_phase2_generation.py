#!/usr/bin/env python3
"""Run the preregistered compact-BLT autoregressive validity evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from jamoflow.generation import (
    DECODING_MODES,
    GENERATION_POLICIES,
    continuation_metrics,
    generation_patch_matrix,
    greedy_byte,
    sampling_generators,
    select_generation_prompts,
    top_p_sample,
    utf8_allowed_next_bytes,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    DEFAULT_MODEL_SPEC,
    build_main_model,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import resolve_device, synchronize
from jamoflow.phase1 import stream_arrays


SEEDS = (1729, 2718, 31415, 57721, 65537)
HARD_MASK_SEED = 1729
GLOBAL_POSITION_LIMIT = DEFAULT_MODEL_SPEC.sequence_length * 2 + 8
TEST_BYTE_LIMIT = 1_000_000


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _checkpoint_path(
    seed: int,
    policy: str,
    primary_root: Path,
    control_root: Path,
) -> Path:
    if policy == "causal_whitespace_grid":
        return (
            control_root
            / f"mechanism-seed-{seed}"
            / "causal_whitespace_grid.pt"
        )
    return primary_root / f"seed-{seed}" / f"{policy}.pt"


def _generate(
    model: Any,
    prompts: np.ndarray,
    policy: str,
    mode: str,
    seed: int,
    device: str,
    *,
    continuation_bytes: int,
    batch_size: int,
    hard_mask: bool,
) -> tuple[list[bytes], float]:
    if mode not in DECODING_MODES:
        raise ValueError(f"unknown decoding mode: {mode}")
    if prompts.shape[1] + continuation_bytes > DEFAULT_MODEL_SPEC.sequence_length:
        raise ValueError("prompt and continuation exceed the 256-byte horizon")
    sequences = prompts.copy()
    generators = sampling_generators(seed, len(prompts))
    model.to(device)
    model.eval()
    synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for step in range(continuation_bytes):
            selected = np.empty(len(sequences), dtype=np.uint8)
            remaining_after = continuation_bytes - step - 1
            for start in range(0, len(sequences), batch_size):
                end = min(start + batch_size, len(sequences))
                batch = sequences[start:end]
                patches = generation_patch_matrix(
                    batch,
                    policy,
                    horizon=DEFAULT_MODEL_SPEC.sequence_length,
                    patch_count=DEFAULT_MODEL_SPEC.patch_count,
                    fixed_stride=DEFAULT_MODEL_SPEC.patch_stride,
                )
                batch_tensor = torch.from_numpy(
                    batch.astype(np.int64, copy=False)
                ).to(device)
                patch_tensor = torch.from_numpy(
                    patches.astype(np.int64, copy=False)
                ).to(device)
                logits = model(
                    input_ids=batch_tensor,
                    patch_lengths=patch_tensor,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                logits_cpu = logits.float().cpu().numpy().astype(
                    np.float64,
                    copy=False,
                )
                for local, values in enumerate(logits_cpu):
                    prompt_index = start + local
                    allowed = (
                        utf8_allowed_next_bytes(
                            bytes(batch[local]),
                            remaining_bytes_after_choice=remaining_after,
                        )
                        if hard_mask
                        else None
                    )
                    if mode == "greedy":
                        byte = greedy_byte(values, allowed)
                    else:
                        byte = top_p_sample(
                            values,
                            generators[prompt_index],
                            temperature=0.8,
                            top_p=0.95,
                            allowed=allowed,
                        )
                    selected[prompt_index] = byte
            sequences = np.concatenate([sequences, selected[:, None]], axis=1)
            if step == 0 or (step + 1) % 32 == 0 or step + 1 == continuation_bytes:
                print(
                    f"  {policy}/{mode}/hard={hard_mask}: "
                    f"{step + 1}/{continuation_bytes} bytes",
                    flush=True,
                )

    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = sequences[:, prompts.shape[1] :]
    return [bytes(row) for row in generated], elapsed


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seed or SEEDS)
    policies = tuple(args.policy or GENERATION_POLICIES)
    if set(seeds) - set(SEEDS):
        raise ValueError("generation seed is outside the preregistered set")
    if set(policies) - set(GENERATION_POLICIES):
        raise ValueError("unknown generation policy")
    if args.prompt_length + args.continuation_bytes > DEFAULT_MODEL_SPEC.sequence_length:
        raise ValueError("prompt plus continuation exceeds model horizon")

    stream = build_neural_stream(
        Path(args.data_root) / "ko.jsonl",
        language="ko",
        split="test",
        byte_limit=TEST_BYTE_LIMIT,
        sequence_length=DEFAULT_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    selection = select_generation_prompts(
        inputs,
        boundaries,
        prompt_count=args.prompt_count,
        prompt_length=args.prompt_length,
    )
    device = resolve_device(args.device)
    primary_root = Path(args.primary_artifact_root)
    control_root = Path(args.control_artifact_root)
    full_protocol = bool(
        seeds == SEEDS
        and policies == GENERATION_POLICIES
        and args.prompt_count == 256
        and args.prompt_length == 128
        and args.continuation_bytes == 128
        and not args.skip_hard_mask_control
    )
    print(
        f"device={device}; prompts={len(selection.prompts)}; "
        f"seeds={seeds}; policies={policies}; full={full_protocol}",
        flush=True,
    )

    conditions: dict[str, Any] = {}
    parameters: set[int] = set()
    for seed in seeds:
        seed_results: dict[str, Any] = {}
        for policy in policies:
            checkpoint = _checkpoint_path(seed, policy, primary_root, control_root)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = build_main_model(
                seed=seed,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            )
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            parameters.add(parameter_count(model))
            policy_results: dict[str, Any] = {}
            for mode in DECODING_MODES:
                print(f"seed {seed}/{policy}/{mode}", flush=True)
                generated, elapsed = _generate(
                    model,
                    selection.prompts,
                    policy,
                    mode,
                    seed,
                    device,
                    continuation_bytes=args.continuation_bytes,
                    batch_size=args.batch_size,
                    hard_mask=False,
                )
                result = continuation_metrics(generated).to_dict()
                result["elapsed_seconds"] = elapsed
                mode_results: dict[str, Any] = {"unconstrained": result}

                if seed == HARD_MASK_SEED and not args.skip_hard_mask_control:
                    print(f"seed {seed}/{policy}/{mode}/hard-mask", flush=True)
                    constrained, hard_elapsed = _generate(
                        model,
                        selection.prompts,
                        policy,
                        mode,
                        seed,
                        device,
                        continuation_bytes=args.continuation_bytes,
                        batch_size=args.batch_size,
                        hard_mask=True,
                    )
                    hard_result = continuation_metrics(constrained).to_dict()
                    hard_result["elapsed_seconds"] = hard_elapsed
                    if hard_result["valid_utf8_count"] != len(selection.prompts):
                        raise AssertionError("UTF-8 DFA hard mask failed")
                    mode_results["hard_mask_control"] = hard_result
                policy_results[mode] = mode_results
            seed_results[policy] = policy_results
            _release_model(model, device)
        conditions[str(seed)] = seed_results
    if len(parameters) != 1:
        raise AssertionError("checkpoint parameter counts differ")

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 2d compact-BLT autoregressive structural validity",
        "quick_smoke_only": not full_protocol,
        "environment": {
            "device": device,
            "platform": platform.platform(),
            "versions": research_versions(),
        },
        "design": {
            "seeds": list(seeds),
            "policies": list(policies),
            "decoding_modes": list(DECODING_MODES),
            "prompt_count": args.prompt_count,
            "prompt_length_bytes": args.prompt_length,
            "continuation_bytes": args.continuation_bytes,
            "fixed_horizon_bytes": DEFAULT_MODEL_SPEC.sequence_length,
            "model_parameters": next(iter(parameters)),
            "hard_mask_seed": HARD_MASK_SEED,
            "hard_mask_all_policies_and_modes": not args.skip_hard_mask_control,
            "samples_or_prompts_serialized": False,
        },
        "prompt_selection": selection.public_metadata(),
        "source": {
            "label": "held-out Korean Wikipedia content-hash test stream",
            "selected_test_bytes": stream.selected_bytes,
            "test_sequences": stream.sequence_count,
        },
        "conditions": conditions,
        "claim_guardrail": (
            "Full-prefix recomputation is a validity reference path; elapsed "
            "times are not incremental-generation latency evidence. No generated "
            "sample or prompt is serialized."
        ),
    }
    _write_json(Path(args.output), output)
    print(f"wrote generation aggregates to {args.output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument(
        "--primary-artifact-root",
        default="artifacts/phase2",
    )
    parser.add_argument(
        "--control-artifact-root",
        default="artifacts/phase2-controls",
    )
    parser.add_argument(
        "--output",
        default="runs/phase2-generation/generation-results.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--policy", action="append", choices=GENERATION_POLICIES)
    parser.add_argument("--prompt-count", type=int, default=256)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--continuation-bytes", type=int, default=128)
    parser.add_argument("--skip-hard-mask-control", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
