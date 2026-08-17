#!/usr/bin/env python3
"""Run Phase 3 full-prefix autoregressive encoding-validity stress."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.generation import (
    DECODING_MODES,
    GENERATION_POLICIES,
    continuation_diagnostic_arrays,
    continuation_metrics,
    continuation_metrics_from_diagnostics,
    generation_patch_matrix,
    greedy_byte,
    sampling_generators,
    select_generation_prompts,
    top_p_sample,
    utf8_allowed_next_bytes,
    utf8_failure_diagnostic_arrays,
    utf8_failure_metrics,
    utf8_failure_metrics_from_diagnostics,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import resolve_device, synchronize
from jamoflow.phase1 import stream_arrays
from jamoflow.phase3 import PHASE3_MODEL_SPEC, PHASE3_OPTIMIZATION_SPEC


KNOWN_SEEDS = (1729, 2718, 31415, 57721, 65537)
DEFAULT_SEEDS = KNOWN_SEEDS[:3]
HARD_MASK_SEED = 1729
TEST_BYTE_LIMIT = 16_000_000
PROMPT_COUNT = 256
PROMPT_BYTES = 256
CONTINUATION_BYTES = 256
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
_MANIFEST_INVARIANTS = (
    "schema_version",
    "design",
    "source",
    "prompt_selection",
    "global_max_position_embeddings",
    "model_spec",
    "optimization_spec",
)
_DIAGNOSTIC_DTYPES = {
    "structural__strict_valid": np.dtype(np.uint8),
    "structural__replacement_character_free": np.dtype(np.uint8),
    "structural__valid_jamo_transition": np.dtype(np.uint8),
    "structural__bytes_per_codepoint": np.dtype(np.float64),
    "utf8__failure_category": np.dtype(np.uint8),
    "utf8__legal_prefix_bytes": np.dtype(np.int64),
    "utf8__closed_codepoint_prefix_bytes": np.dtype(np.int64),
    "utf8__first_illegal_byte_position": np.dtype(np.int64),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def merge_generation_manifest(
    existing: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge resumable generation invocations without changing the design."""

    candidate = deepcopy(dict(current))
    invocation = {
        key: deepcopy(candidate[key])
        for key in (
            "created_at",
            "git_commit",
            "device",
            "platform",
            "versions",
            "seeds",
            "policies",
            "force",
        )
    }
    if existing is None:
        candidate["invocations"] = [invocation]
        return candidate
    merged = deepcopy(dict(existing))
    for key in _MANIFEST_INVARIANTS:
        if merged.get(key) != candidate.get(key):
            raise ValueError(f"generation manifest invariant changed: {key}")
    invocations = list(merged.get("invocations", []))
    if not invocations:
        raise ValueError("generation manifest lacks invocation provenance")
    invocations.append(invocation)
    merged["invocations"] = invocations
    for key in ("seeds", "policies"):
        values = list(merged[key])
        values.extend(value for value in candidate[key] if value not in values)
        merged[key] = values
    merged["updated_at"] = candidate["created_at"]
    return merged


def _load_verified_model(
    seed: int,
    policy: str,
    training_run_root: Path,
    checkpoint_root: Path,
) -> tuple[Any, dict[str, str]]:
    report_path = training_run_root / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = checkpoint_root / f"seed-{seed}" / f"{policy}.pt"
    if not report_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing Phase 3 checkpoint/report for seed {seed}/{policy}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("seed") != seed or report.get("policy") != policy:
        raise ValueError("Phase 3 generation checkpoint identity mismatch")
    if (
        report.get("parameters") != 19_596_096
        or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
    ):
        raise ValueError("Phase 3 generation checkpoint model spec mismatch")
    model = build_main_model(
        PHASE3_MODEL_SPEC,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    checkpoint_hash = _state_dict_sha256(model)
    training_hash = report.get("trained_state_sha256")
    if not isinstance(training_hash, str) or checkpoint_hash != training_hash:
        raise ValueError("Phase 3 generation checkpoint state hash mismatch")
    return model, {
        "checkpoint_state_sha256": checkpoint_hash,
        "training_report_state_sha256": training_hash,
        "checkpoint_artifact_sha256": _sha256_file(checkpoint_path),
        "training_report_artifact_sha256": _sha256_file(report_path),
    }


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
    horizon = PHASE3_MODEL_SPEC.sequence_length
    if prompts.shape[1] + continuation_bytes > horizon:
        raise ValueError("prompt and continuation exceed Phase 3 horizon")
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
                    horizon=horizon,
                    patch_count=PHASE3_MODEL_SPEC.patch_count,
                    fixed_stride=PHASE3_MODEL_SPEC.patch_stride,
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
                    selected[prompt_index] = (
                        greedy_byte(values, allowed)
                        if mode == "greedy"
                        else top_p_sample(
                            values,
                            generators[prompt_index],
                            temperature=0.8,
                            top_p=0.95,
                            allowed=allowed,
                        )
                    )
            sequences = np.concatenate([sequences, selected[:, None]], axis=1)
            if (
                step == 0
                or (step + 1) % 32 == 0
                or step + 1 == continuation_bytes
            ):
                print(
                    f"  {policy}/{mode}/hard={hard_mask}: "
                    f"{step + 1}/{continuation_bytes} bytes",
                    flush=True,
                )

    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = sequences[:, prompts.shape[1] :]
    return [bytes(row) for row in generated], elapsed


def _aggregate_generation_metrics(
    prompts: np.ndarray,
    generated: list[bytes],
    elapsed: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    structural_arrays, structural_bytes = continuation_diagnostic_arrays(
        generated
    )
    failure_arrays, failure_bytes = utf8_failure_diagnostic_arrays(
        (bytes(row) for row in prompts),
        generated,
    )
    if structural_bytes != failure_bytes:
        raise AssertionError("generation diagnostic byte lengths disagree")
    structural = continuation_metrics_from_diagnostics(
        structural_arrays,
        structural_bytes,
    ).to_dict()
    failure = utf8_failure_metrics_from_diagnostics(
        failure_arrays,
        failure_bytes,
    ).to_dict()
    if structural != continuation_metrics(generated).to_dict():
        raise AssertionError("structural diagnostic reconstruction disagrees")
    if failure != utf8_failure_metrics(
        (bytes(row) for row in prompts),
        generated,
    ).to_dict():
        raise AssertionError("failure diagnostic reconstruction disagrees")
    if structural["valid_utf8_count"] != failure["strict_valid_count"]:
        raise AssertionError("strict decoder and UTF-8 failure taxonomy disagree")
    return (
        {
            **structural,
            "utf8_failure_taxonomy": failure,
            "elapsed_seconds_diagnostic_only": elapsed,
        },
        {
            **{
                f"structural__{key}": value
                for key, value in structural_arrays.items()
            },
            **{f"utf8__{key}": value for key, value in failure_arrays.items()},
        },
    )


def _tag_diagnostics(
    mode: str,
    variant: str,
    diagnostics: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if mode not in DECODING_MODES or variant not in {
        "unconstrained",
        "hard_mask_control",
    }:
        raise ValueError("invalid generation diagnostic condition")
    return {
        f"{mode}__{variant}__{key}": np.asarray(value)
        for key, value in diagnostics.items()
    }


def _metrics_from_tagged_diagnostics(
    diagnostics: Mapping[str, np.ndarray],
    mode: str,
    variant: str,
    continuation_bytes: int,
) -> dict[str, Any]:
    prefix = f"{mode}__{variant}__"
    local = {
        key[len(prefix) :]: np.asarray(value)
        for key, value in diagnostics.items()
        if key.startswith(prefix)
    }
    structural = {
        key[len("structural__") :]: value
        for key, value in local.items()
        if key.startswith("structural__")
    }
    failure = {
        key[len("utf8__") :]: value
        for key, value in local.items()
        if key.startswith("utf8__")
    }
    structural_metrics = continuation_metrics_from_diagnostics(
        structural,
        continuation_bytes,
    ).to_dict()
    failure_metrics = utf8_failure_metrics_from_diagnostics(
        failure,
        continuation_bytes,
    ).to_dict()
    if structural_metrics["valid_utf8_count"] != failure_metrics[
        "strict_valid_count"
    ]:
        raise ValueError("generation diagnostic families disagree")
    return {
        **structural_metrics,
        "utf8_failure_taxonomy": failure_metrics,
    }


def _validate_completed_result(
    report_path: Path,
    artifact_path: Path,
    *,
    seed: int,
    policy: str,
    lineage: Mapping[str, str],
    prompt_selection: Mapping[str, int],
    source_stream_sha256: str,
    continuation_bytes: int,
    hard_mask_control: bool,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "policy": policy,
        "parameters": 19_596_096,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "source_stream_sha256": source_stream_sha256,
        "prompt_selection": dict(prompt_selection),
        "diagnostic_artifact_filename": (
            f"seed-{seed}/{policy}-diagnostics.npz"
        ),
        "diagnostic_artifact_sha256": _sha256_file(artifact_path),
        "raw_generation_serialized": False,
        "prompts_or_prompt_hashes_serialized": False,
        "non_content_per_prompt_diagnostics_serialized": True,
        **lineage,
    }
    if set(report) != set(expected_fields) | {"modes"}:
        raise ValueError(
            f"unexpected generation report fields: {seed}/{policy}; "
            "rerun with --force"
        )
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(
                f"stale generation result ({key}): {seed}/{policy}; "
                "rerun with --force"
            )

    expected_variants = {"unconstrained"}
    if hard_mask_control:
        expected_variants.add("hard_mask_control")
    modes = report.get("modes")
    if not isinstance(modes, dict) or set(modes) != set(DECODING_MODES):
        raise ValueError(
            f"stale generation modes: {seed}/{policy}; rerun with --force"
        )
    with np.load(artifact_path, allow_pickle=False) as archive:
        diagnostics = {key: archive[key] for key in archive.files}
    expected_keys: set[str] = set()
    for mode in DECODING_MODES:
        variants = modes[mode]
        if not isinstance(variants, dict) or set(variants) != expected_variants:
            raise ValueError(
                f"stale generation variants: {seed}/{policy}/{mode}; "
                "rerun with --force"
            )
        for variant in sorted(expected_variants):
            prefix = f"{mode}__{variant}__"
            expected_keys.update(
                prefix + key
                for key in (
                    "structural__strict_valid",
                    "structural__replacement_character_free",
                    "structural__valid_jamo_transition",
                    "structural__bytes_per_codepoint",
                    "utf8__failure_category",
                    "utf8__legal_prefix_bytes",
                    "utf8__closed_codepoint_prefix_bytes",
                    "utf8__first_illegal_byte_position",
                )
            )
            reconstructed = _metrics_from_tagged_diagnostics(
                diagnostics,
                mode,
                variant,
                continuation_bytes,
            )
            if (
                reconstructed["continuations"]
                != prompt_selection["selected_prompts"]
                or reconstructed["continuation_bytes"] != continuation_bytes
            ):
                raise ValueError("generation diagnostic geometry changed")
            recorded = variants[variant]
            if not isinstance(recorded, dict):
                raise ValueError("generation metric payload is malformed")
            elapsed = recorded.get("elapsed_seconds_diagnostic_only")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0
            ):
                raise ValueError("generation elapsed diagnostic is malformed")
            comparable = dict(recorded)
            comparable.pop("elapsed_seconds_diagnostic_only", None)
            if comparable != reconstructed:
                raise ValueError(
                    f"stale generation aggregates: {seed}/{policy}/{mode}/"
                    f"{variant}; rerun with --force"
                )
            if (
                variant == "hard_mask_control"
                and reconstructed["valid_utf8_count"]
                != prompt_selection["selected_prompts"]
            ):
                raise ValueError("UTF-8 hard-mask invariant failed")
    if set(diagnostics) != expected_keys:
        raise ValueError(
            f"unexpected generation diagnostic keys: {seed}/{policy}; "
            "rerun with --force"
        )
    for key, value in diagnostics.items():
        suffix = "__".join(key.split("__")[2:])
        if value.dtype != _DIAGNOSTIC_DTYPES[suffix]:
            raise ValueError(
                f"unexpected generation diagnostic dtype: {seed}/{policy}; "
                "rerun with --force"
            )


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    policies = tuple(args.policies)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or set(seeds) - set(KNOWN_SEEDS)
    ):
        raise ValueError("generation needs preregistered Phase 3 seeds")
    unknown = set(policies) - set(GENERATION_POLICIES)
    if unknown or not policies or len(set(policies)) != len(policies):
        raise ValueError(f"unsupported generation policies: {sorted(unknown)}")
    if args.prompt_length + args.continuation_bytes > PHASE3_MODEL_SPEC.sequence_length:
        raise ValueError("prompt and continuation exceed Phase 3 horizon")
    if (
        args.prompt_count <= 0
        or args.prompt_length <= 0
        or args.continuation_bytes <= 0
        or args.batch_size <= 0
    ):
        raise ValueError("generation counts and lengths must be positive")

    source_path = Path(args.data_root) / "ko.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=args.byte_limit,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
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
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    training_run_root = Path(args.training_run_root)
    checkpoint_root = Path(args.checkpoint_root)
    full_design = bool(
        args.byte_limit == TEST_BYTE_LIMIT
        and args.prompt_count == PROMPT_COUNT
        and args.prompt_length == PROMPT_BYTES
        and args.continuation_bytes == CONTINUATION_BYTES
        and not args.skip_hard_mask_control
    )
    current_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "seeds": list(seeds),
        "policies": list(policies),
        "force": bool(args.force),
        "design": {
            "full_preregistered_design": full_design,
            "known_seeds": list(KNOWN_SEEDS),
            "policies": list(GENERATION_POLICIES),
            "decoding_modes": list(DECODING_MODES),
            "prompt_count": args.prompt_count,
            "prompt_length_bytes": args.prompt_length,
            "continuation_bytes": args.continuation_bytes,
            "fixed_horizon_bytes": PHASE3_MODEL_SPEC.sequence_length,
            "hard_mask_seed": HARD_MASK_SEED,
            "hard_mask_all_policies_and_modes": (
                not args.skip_hard_mask_control
            ),
            "samples_prompts_or_prompt_hashes_serialized": False,
            "non_content_per_prompt_diagnostics_serialized": True,
            "use_cache": False,
            "elapsed_time_is_latency_evidence": False,
            "decision_gate": None,
        },
        "source": {
            "source_artifact": {
                "filename": "ko.jsonl",
                "bytes": source_path.stat().st_size,
                "sha256": _sha256_file(source_path),
            },
            "requested_byte_limit": args.byte_limit,
            "stream": stream.metadata(),
            "selected_stream_sha256": _sha256_bytes(stream.data),
        },
        "prompt_selection": selection.public_metadata(),
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
    }
    manifest_path = run_root / "manifest.json"
    existing = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    manifest = merge_generation_manifest(existing, current_manifest)
    _write_json(manifest_path, manifest)
    print(
        f"device={device}; prompts={len(selection.prompts)}; "
        f"seeds={seeds}; policies={policies}; full={full_design}",
        flush=True,
    )

    for seed in seeds:
        for policy in policies:
            output_path = run_root / f"seed-{seed}" / f"{policy}.json"
            artifact_path = (
                artifact_root
                / f"seed-{seed}"
                / f"{policy}-diagnostics.npz"
            )
            model, lineage = _load_verified_model(
                seed,
                policy,
                training_run_root,
                checkpoint_root,
            )
            hard_mask_control = bool(
                seed == HARD_MASK_SEED and not args.skip_hard_mask_control
            )
            if output_path.exists() and artifact_path.exists() and not args.force:
                _validate_completed_result(
                    output_path,
                    artifact_path,
                    seed=seed,
                    policy=policy,
                    lineage=lineage,
                    prompt_selection=selection.public_metadata(),
                    source_stream_sha256=_sha256_bytes(stream.data),
                    continuation_bytes=args.continuation_bytes,
                    hard_mask_control=hard_mask_control,
                )
                print(f"seed {seed}/{policy}: generation already complete")
                _release_model(model, device)
                continue
            if (
                output_path.exists() != artifact_path.exists()
                and not args.force
            ):
                _release_model(model, device)
                raise ValueError(
                    f"incomplete generation result for {seed}/{policy}; "
                    "rerun with --force"
                )
            mode_results: dict[str, Any] = {}
            diagnostic_arrays: dict[str, np.ndarray] = {}
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
                metrics, diagnostics = _aggregate_generation_metrics(
                    selection.prompts,
                    generated,
                    elapsed,
                )
                variants: dict[str, Any] = {"unconstrained": metrics}
                diagnostic_arrays.update(
                    _tag_diagnostics(mode, "unconstrained", diagnostics)
                )
                if hard_mask_control:
                    print(
                        f"seed {seed}/{policy}/{mode}/hard-mask",
                        flush=True,
                    )
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
                    hard, hard_diagnostics = _aggregate_generation_metrics(
                        selection.prompts,
                        constrained,
                        hard_elapsed,
                    )
                    if hard["valid_utf8_count"] != len(selection.prompts):
                        raise AssertionError("UTF-8 hard-mask invariant failed")
                    variants["hard_mask_control"] = hard
                    diagnostic_arrays.update(
                        _tag_diagnostics(
                            mode,
                            "hard_mask_control",
                            hard_diagnostics,
                        )
                    )
                mode_results[mode] = variants
            _save_npz(artifact_path, diagnostic_arrays)
            _write_json(
                output_path,
                {
                    "schema_version": 1,
                    "seed": seed,
                    "policy": policy,
                    "parameters": parameter_count(model),
                    "model_spec": PHASE3_MODEL_SPEC.to_dict(),
                    "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
                    "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
                    **lineage,
                    "source_stream_sha256": _sha256_bytes(stream.data),
                    "prompt_selection": selection.public_metadata(),
                    "diagnostic_artifact_filename": (
                        f"seed-{seed}/{policy}-diagnostics.npz"
                    ),
                    "diagnostic_artifact_sha256": _sha256_file(artifact_path),
                    "modes": mode_results,
                    "raw_generation_serialized": False,
                    "prompts_or_prompt_hashes_serialized": False,
                    "non_content_per_prompt_diagnostics_serialized": True,
                },
            )
            _release_model(model, device)
    print(f"completed Phase 3 generation under {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--training-run-root", default="runs/phase3")
    parser.add_argument("--checkpoint-root", default="artifacts/phase3")
    parser.add_argument("--run-root", default="runs/phase3-generation")
    parser.add_argument(
        "--artifact-root",
        default="artifacts/phase3-generation",
    )
    parser.add_argument("--byte-limit", type=int, default=TEST_BYTE_LIMIT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(GENERATION_POLICIES),
    )
    parser.add_argument("--prompt-count", type=int, default=PROMPT_COUNT)
    parser.add_argument("--prompt-length", type=int, default=PROMPT_BYTES)
    parser.add_argument(
        "--continuation-bytes",
        type=int,
        default=CONTINUATION_BYTES,
    )
    parser.add_argument("--skip-hard-mask-control", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
