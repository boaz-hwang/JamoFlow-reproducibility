#!/usr/bin/env python3
"""Run gated five-seed incremental autoregressive inference measurements."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from jamoflow.actual_inference_protocol import (
    ACTUAL_INFERENCE_PROTOCOL_VERSION,
    ACTUAL_INFERENCE_SELECTION_ALGORITHM,
    COMPONENTS,
    CONTINUATION_BYTES,
    CORRECTNESS_CONTINUATION_BYTES,
    FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES,
    FREE_RUNNING_UTF8_CONSTRAINT,
    MEASURED_CASES,
    MODES,
    OUTPUT_DIAGNOSTICS,
    PROMPT_BYTES,
    REPETITIONS,
    ROLES,
    SEED_EXECUTION_ORDER_SEED,
    SEEDS,
    TIME_TO_OUTPUT_SEMANTICS,
    TIMING_ORDER_SEED,
    WARMUP_CASES,
    decode_forward_steps,
    free_running_maximum_output_bytes,
    reconstruct_valid_completion_metrics,
    runtime_observed_bytes,
    timing_environment_eligible,
    valid_output_overshoot,
    validate_output_diagnostic_arrays,
)
from jamoflow.compute_conversion import conversion_model_spec
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.generation import (
    valid_completion_metrics,
    valid_conjoining_jamo_transitions,
)
from jamoflow.incremental_blt import (
    INCREMENTAL_ENTROPY_POLICIES,
    IncrementalBltDecoder,
    IncrementalEntropyBltDecoder,
    structural_prefix_boundaries,
)
from jamoflow.inference_benchmark import (
    select_inference_cases,
    timing_order_schedule,
    verification_prefix_lengths,
)
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import build_main_model, build_router, parameter_count
from jamoflow.neural_training import synchronize
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import padded_hf_patch_matrix
from jamoflow.phase3 import PHASE3_MODEL_SPEC, PHASE3_OPTIMIZATION_SPEC
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    StrictUtf8State,
    advance_strict_utf8,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
    strict_utf8_state,
)


GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8


@dataclass(slots=True)
class LoadedPolicy:
    role: str
    policy: str
    runtime_policy: str
    family: str
    patch_count: int
    model: Any
    router: Any | None
    threshold_nats: float | None
    maximum_patch_length: int | None
    provenance: dict[str, Any]

    def runtime(self) -> Any:
        if self.runtime_policy in INCREMENTAL_ENTROPY_POLICIES:
            if (
                self.router is None
                or self.threshold_nats is None
                or self.maximum_patch_length is None
            ):
                raise RuntimeError("entropy runtime lacks router calibration")
            return IncrementalEntropyBltDecoder(
                self.model,
                self.router,
                self.runtime_policy,
                threshold_nats=self.threshold_nats,
                maximum_patch_length=self.maximum_patch_length,
                horizon=PHASE3_MODEL_SPEC.sequence_length,
                patch_count=self.patch_count,
                fixed_stride=PHASE3_MODEL_SPEC.patch_stride,
            )
        return IncrementalBltDecoder(
            self.model,
            self.runtime_policy,
            horizon=PHASE3_MODEL_SPEC.sequence_length,
            patch_count=self.patch_count,
            fixed_stride=PHASE3_MODEL_SPEC.patch_stride,
        )


@dataclass(frozen=True, slots=True)
class TrialResult:
    ttft_ms: float
    decode_ms: float
    end_to_end_ms: float
    emitted_global_patches: int
    observed_bytes: int
    emitted_output_bytes: int
    decode_forward_steps: int
    overshoot_bytes: int
    valid_output_stop: bool
    replacement_character_free: bool
    valid_jamo_transition: bool
    output_codepoints: int
    mps_current_allocated_bytes: int | None
    mps_driver_allocated_bytes: int | None
    generated: bytes | None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
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


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _command_snapshot(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _environment(device: str) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": packages,
        "device": device,
        "mps_available": torch.backends.mps.is_available(),
    }


def _session_state() -> dict[str, Any]:
    return {
        "power": _command_snapshot(["pmset", "-g", "batt"]),
        "thermal": _command_snapshot(["pmset", "-g", "therm"]),
        "settings": _command_snapshot(["pmset", "-g", "custom"]),
    }


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    return requested


def _validate_quality_gate(
    quality_path: Path,
    selection_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    quality = _read_json(quality_path)
    selection = _read_json(selection_path)
    if (
        quality.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or quality.get("quality_noninferiority", {}).get("overall_pass") is not True
        or quality.get("selection", {}).get("sha256") != _sha256(selection_path)
        or quality.get("candidate") != selection.get("candidate")
        or quality.get("reference") != selection.get("reference")
    ):
        raise ValueError("actual timing is blocked by quality or lineage failure")
    phase3_item = quality.get("phase3_confirmation_summary", {})
    phase3_path = Path(phase3_item["path"])
    if _sha256(phase3_path) != phase3_item.get("sha256"):
        raise ValueError("Phase 3 confirmation summary changed after quality gate")
    phase3 = _read_json(phase3_path)
    if (
        tuple(phase3.get("seeds", [])) != SEEDS
        or phase3.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or phase3.get("gate_j", {}).get("overall_pass") is not True
    ):
        raise ValueError("actual timing requires passing five-seed Phase 3 evidence")
    return quality, selection, phase3


def _reconstruct_cases(
    phase3: dict[str, Any],
    data_root: Path,
    *,
    total_cases: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = phase3["run_manifest"]
    source_path = data_root / "ko.jsonl"
    integrity_path = data_root / "integrity.json"
    source_artifact = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    integrity_artifact = {
        "filename": "integrity.json",
        "bytes": integrity_path.stat().st_size,
        "sha256": _sha256(integrity_path),
    }
    if (
        manifest.get("source_artifact") != source_artifact
        or manifest.get("source_integrity_artifact") != integrity_artifact
    ):
        raise ValueError("actual-inference source artifacts differ from Phase 3")
    stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=int(manifest["limits"]["test"]),
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    recorded_stream = manifest["streams"]["test"]
    if (
        hashlib.sha256(stream.data).hexdigest()
        != recorded_stream["selected_stream_sha256"]
        or len(inputs) != recorded_stream["sequence_count"]
    ):
        raise ValueError("actual-inference test stream differs from Phase 3")
    document_window_map = reconstruct_document_window_map(
        source_path,
        split="test",
        byte_limit=int(manifest["limits"]["test"]),
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        expected_stream=stream.data,
    )
    document_contained = document_window_map.document_indices >= 0
    cases = select_inference_cases(
        inputs[document_contained],
        boundaries[document_contained],
        cluster_ids=document_window_map.document_indices[document_contained],
        case_count=total_cases,
        prompt_length=PROMPT_BYTES,
        continuation_length=CONTINUATION_BYTES,
    )
    return cases.prompts, cases.replay_continuations, {
        "source_artifact": source_artifact,
        "source_integrity_artifact": integrity_artifact,
        "test_stream_sha256": recorded_stream["selected_stream_sha256"],
        "test_sequence_count": len(inputs),
        "selection_algorithm": ACTUAL_INFERENCE_SELECTION_ALGORITHM,
        "document_window_map": document_window_map.metadata(),
        **cases.public_metadata(),
    }


def _policy_paths(
    family: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    if family == "compute_conversion":
        return Path(args.conversion_run_root), Path(args.conversion_artifact_root)
    if family == "phase3":
        return Path(args.phase3_run_root), Path(args.phase3_artifact_root)
    raise ValueError(f"unknown model family: {family}")


def _load_entropy_router(
    seed: int,
    policy: str,
    phase3: dict[str, Any],
    run_root: Path,
    artifact_root: Path,
    device: str,
) -> tuple[Any, float, int, dict[str, Any]]:
    run_directory = run_root / f"seed-{seed}"
    artifact_directory = artifact_root / f"seed-{seed}"
    report_path = run_directory / "router.json"
    checkpoint_path = artifact_directory / "router.pt"
    diagnostics_path = run_directory / "threshold-patch-diagnostics.json"
    cache_path = artifact_directory / "threshold-patches.npz"
    recorded = phase3["integrity"]["by_seed"][str(seed)][
        "router_and_threshold_cache"
    ]
    actual = {
        "router_checkpoint_artifact_sha256": _sha256(checkpoint_path),
        "router_report_artifact_sha256": _sha256(report_path),
        "threshold_cache_artifact_sha256": _sha256(cache_path),
        "threshold_diagnostics_artifact_sha256": _sha256(diagnostics_path),
    }
    for key, value in actual.items():
        if recorded.get(key) != value:
            raise ValueError(f"router evidence mismatch: {seed}/{key}")
    report = _read_json(report_path)
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    router.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state_hash = _state_dict_sha256(router)
    if (
        report.get("seed") != seed
        or report.get("parameters") != parameter_count(router)
        or report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("trained_state_sha256") != state_hash
        or recorded.get("router_checkpoint_state_sha256") != state_hash
    ):
        raise ValueError(f"router checkpoint identity mismatch: {seed}")
    diagnostics = _read_json(diagnostics_path)
    calibration = diagnostics.get("calibration", {}).get(policy, {})
    threshold = calibration.get("threshold_nats")
    maximum = diagnostics.get("_provenance", {}).get("maximum_patch_length")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not isinstance(maximum, int)
        or maximum <= 0
    ):
        raise ValueError(f"router calibration mismatch: {seed}/{policy}")
    router.to(device).eval()
    return router, float(threshold), maximum, {
        **actual,
        "router_checkpoint_state_sha256": state_hash,
        "threshold_nats": float(threshold),
        "maximum_patch_length": maximum,
    }


def _load_policy(
    role: str,
    descriptor: dict[str, Any],
    seed: int,
    quality: dict[str, Any],
    phase3: dict[str, Any],
    args: argparse.Namespace,
    device: str,
) -> LoadedPolicy:
    policy = descriptor["policy"]
    runtime_policy = descriptor["runtime_policy"]
    family = descriptor["model_family"]
    patch_count = int(descriptor["patch_count"])
    run_root, artifact_root = _policy_paths(family, args)
    report_path = run_root / f"seed-{seed}" / f"{policy}.json"
    checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
    evidence = quality["integrity"][f"{role}_evidence"][str(seed)]
    if (
        _sha256(report_path) != evidence["training_report_artifact_sha256"]
        or _sha256(checkpoint_path) != evidence["checkpoint_artifact_sha256"]
    ):
        raise ValueError(f"timing checkpoint artifacts changed: {seed}/{role}")
    spec = (
        conversion_model_spec(patch_count)
        if family == "compute_conversion"
        else PHASE3_MODEL_SPEC
    )
    report = _read_json(report_path)
    model = build_main_model(
        spec,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state_hash = _state_dict_sha256(model)
    if (
        report.get("seed") != seed
        or report.get("policy") != policy
        or report.get("parameters") != parameter_count(model)
        or report.get("model_spec") != spec.to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("trained_state_sha256") != state_hash
        or evidence["checkpoint_state_sha256"] != state_hash
    ):
        raise ValueError(f"timing checkpoint identity mismatch: {seed}/{role}")
    model.to(device).eval()
    router = None
    threshold = None
    maximum = None
    router_provenance = None
    if runtime_policy in INCREMENTAL_ENTROPY_POLICIES:
        if family != "phase3":
            raise ValueError("entropy reference must use the Phase 3 family")
        router, threshold, maximum, router_provenance = _load_entropy_router(
            seed,
            policy,
            phase3,
            run_root,
            artifact_root,
            device,
        )
    return LoadedPolicy(
        role=role,
        policy=policy,
        runtime_policy=runtime_policy,
        family=family,
        patch_count=patch_count,
        model=model,
        router=router,
        threshold_nats=threshold,
        maximum_patch_length=maximum,
        provenance={
            "training_report_artifact_sha256": _sha256(report_path),
            "checkpoint_artifact_sha256": _sha256(checkpoint_path),
            "checkpoint_state_sha256": state_hash,
            "model_spec": spec.to_dict(),
            "router": router_provenance,
        },
    )


def _main_diagnostics(runtime: Any) -> Any:
    diagnostics = runtime.diagnostics
    return diagnostics.main if hasattr(diagnostics, "main") else diagnostics


def _assert_cache_invariants(runtime: Any, expected_bytes: int) -> None:
    diagnostics = runtime.diagnostics
    main = _main_diagnostics(runtime)
    if (
        main.observed_bytes != expected_bytes
        or main.local_encoder_cached_bytes != expected_bytes
        or main.local_decoder_cached_bytes != expected_bytes
        or main.global_cached_patches != main.emitted_data_patches
    ):
        raise AssertionError("incremental main cache lengths are inconsistent")
    if hasattr(diagnostics, "router_cached_bytes") and (
        diagnostics.router_cached_bytes != expected_bytes
    ):
        raise AssertionError("incremental router cache length is inconsistent")


def _compare_logits(left: Any, right: Any) -> float:
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
    if int(left.argmax(dim=-1).item()) != int(right.argmax(dim=-1).item()):
        raise AssertionError("incremental and reference argmax differ")
    return float((left.float() - right.float()).abs().max().item())


def _full_prefix_logits(
    bundle: LoadedPolicy,
    prefix: bytes,
    boundaries: tuple[int, ...],
) -> Any:
    device = next(bundle.model.parameters()).device
    patches = padded_hf_patch_matrix([boundaries], len(prefix))
    return bundle.model(
        input_ids=torch.tensor(
            [list(prefix)],
            dtype=torch.long,
            device=device,
        ),
        patch_lengths=torch.from_numpy(
            patches.astype(np.int64, copy=False)
        ).to(device),
        use_cache=False,
        logits_to_keep=1,
    ).logits[:, -1, :].float()


def _verify_bundle(
    bundle: LoadedPolicy,
    seed: int,
    warmup_prompts: np.ndarray,
    measured_prompts: np.ndarray,
    measured_continuations: np.ndarray,
    *,
    device: str,
    correctness_continuation_bytes: int,
    minimum_positions: int,
) -> dict[str, Any]:
    maximum_error = 0.0
    full_prefix_comparisons = 0
    parallel_comparisons = 0
    with torch.inference_mode():
        for prompt_index, row in enumerate(warmup_prompts):
            prompt = bytes(row)
            runtime = bundle.runtime()
            logits_by_length: list[Any] = []
            boundaries_by_length: list[tuple[int, ...]] = []
            for length, value in enumerate(prompt, start=1):
                logits = runtime.consume(value)
                logits_by_length.append(logits.detach().cpu())
                boundaries = _main_diagnostics(runtime).boundaries
                boundaries_by_length.append(boundaries)
                if bundle.runtime_policy not in INCREMENTAL_ENTROPY_POLICIES:
                    expected = structural_prefix_boundaries(
                        prompt[:length],
                        bundle.runtime_policy,
                        horizon=PHASE3_MODEL_SPEC.sequence_length,
                        patch_count=bundle.patch_count,
                        fixed_stride=PHASE3_MODEL_SPEC.patch_stride,
                    )
                    if boundaries != expected:
                        raise AssertionError("online structural selector differs")
            _assert_cache_invariants(runtime, len(prompt))
            positions = verification_prefix_lengths(
                boundaries_by_length[-1],
                len(prompt),
                minimum_positions=minimum_positions,
                selection_seed=(
                    20_260_811
                    + seed
                    + prompt_index * 17
                    + (0 if bundle.role == "candidate" else 1)
                ),
            )
            for prefix_length in positions:
                full = _full_prefix_logits(
                    bundle,
                    prompt[:prefix_length],
                    boundaries_by_length[prefix_length - 1],
                )
                maximum_error = max(
                    maximum_error,
                    _compare_logits(
                        logits_by_length[prefix_length - 1],
                        full.detach().cpu(),
                    ),
                )
                full_prefix_comparisons += 1
            del runtime, logits_by_length, boundaries_by_length

        for prompt, continuation in zip(
            measured_prompts,
            measured_continuations,
            strict=True,
        ):
            raw_prompt = bytes(prompt)
            sequential = bundle.runtime()
            parallel = bundle.runtime()
            sequential_logits = sequential.prefill(raw_prompt)
            parallel_logits = parallel.prefill_parallel(raw_prompt)
            maximum_error = max(
                maximum_error,
                _compare_logits(sequential_logits, parallel_logits),
            )
            _assert_cache_invariants(sequential, len(raw_prompt))
            _assert_cache_invariants(parallel, len(raw_prompt))
            if sequential.diagnostics != parallel.diagnostics:
                raise AssertionError("parallel and sequential prompt caches differ")
            parallel_comparisons += 1
            for value in bytes(continuation)[:correctness_continuation_bytes]:
                sequential_logits = sequential.consume(value)
                parallel_logits = parallel.consume(value)
                maximum_error = max(
                    maximum_error,
                    _compare_logits(sequential_logits, parallel_logits),
                )
                if sequential.diagnostics != parallel.diagnostics:
                    raise AssertionError("parallel continuation caches differ")
                parallel_comparisons += 1
            expected_bytes = len(raw_prompt) + correctness_continuation_bytes
            _assert_cache_invariants(sequential, expected_bytes)
            _assert_cache_invariants(parallel, expected_bytes)
            del sequential, parallel
    synchronize(device)
    return {
        "pass": True,
        "rtol": 2e-5,
        "atol": 2e-5,
        "argmax_match_rate": 1.0,
        "maximum_absolute_logit_error": maximum_error,
        "full_prefix_boundary_position_comparisons": full_prefix_comparisons,
        "parallel_prefill_and_continuation_comparisons": parallel_comparisons,
        "warmup_prompts": len(warmup_prompts),
        "measured_prompts": len(measured_prompts),
        "continuation_bytes_per_measured_prompt": correctness_continuation_bytes,
    }


def _mps_memory(device: str) -> tuple[int | None, int | None]:
    if device != "mps":
        return None, None
    return (
        int(torch.mps.current_allocated_memory()),
        int(torch.mps.driver_allocated_memory()),
    )


def _utf8_mask_cache(device: str) -> dict[StrictUtf8State, torch.Tensor]:
    """Compile the shared strict-DFA masks outside measured trials."""

    masks: dict[StrictUtf8State, torch.Tensor] = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device=device)
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        if not bool(mask.any().item()):
            raise AssertionError("a reachable strict UTF-8 state has no output")
        masks[state] = mask
    synchronize(device)
    return masks


def _run_trial(
    bundle: LoadedPolicy,
    prompt: bytes,
    continuation: bytes,
    mode: str,
    device: str,
    utf8_masks: Mapping[StrictUtf8State, torch.Tensor] | None = None,
) -> TrialResult:
    if mode not in MODES:
        raise ValueError(f"unknown inference timing mode: {mode}")
    if not continuation:
        raise ValueError("inference timing needs at least one output byte")
    if mode == "controlled_replay":
        continuation_state = strict_utf8_state(continuation)
        if not continuation_state.at_codepoint_boundary:
            raise ValueError("controlled replay must end at a strict UTF-8 boundary")
    elif utf8_masks is None or set(utf8_masks) != set(
        strict_utf8_reachable_states()
    ):
        raise ValueError("free-running timing requires the compiled UTF-8 DFA")
    synchronize(device)
    started = time.perf_counter_ns()
    generated = bytearray()
    with torch.inference_mode():
        runtime = bundle.runtime()
        logits = runtime.prefill_parallel(prompt)
        synchronize(device)
        prefilled = time.perf_counter_ns()
        if mode == "controlled_replay":
            for value in continuation[:-1]:
                logits = runtime.consume(value)
        else:
            state = STRICT_UTF8_INITIAL_STATE
            maximum_output_bytes = free_running_maximum_output_bytes(
                len(continuation)
            )
            while True:
                if logits.shape[-1] != 256:
                    raise AssertionError("byte runtime logits changed vocabulary")
                allowed = utf8_masks[state]
                value = int(
                    logits.masked_fill(~allowed, -torch.inf)
                    .argmax(dim=-1)
                    .item()
                )
                generated.append(value)
                state = advance_strict_utf8(state, value)
                if not state.valid:
                    raise AssertionError("the strict UTF-8 mask admitted an error")
                if (
                    len(generated) >= len(continuation)
                    and state.at_codepoint_boundary
                ):
                    break
                if len(generated) >= maximum_output_bytes:
                    raise AssertionError("valid-output generation exceeded UTF-8 bound")
                logits = runtime.consume(value)
        synchronize(device)
        finished = time.perf_counter_ns()
    diagnostics = _main_diagnostics(runtime)
    emitted_output_bytes = (
        len(continuation)
        if mode == "controlled_replay"
        else len(generated)
    )
    expected_observed = runtime_observed_bytes(
        len(prompt),
        emitted_output_bytes,
    )
    _assert_cache_invariants(runtime, expected_observed)
    current_memory, driver_memory = _mps_memory(device)
    output = continuation if mode == "controlled_replay" else bytes(generated)
    output_text = output.decode("utf-8", errors="strict")
    result = TrialResult(
        ttft_ms=(prefilled - started) / 1_000_000,
        decode_ms=(finished - prefilled) / 1_000_000,
        end_to_end_ms=(finished - started) / 1_000_000,
        emitted_global_patches=diagnostics.emitted_data_patches,
        observed_bytes=diagnostics.observed_bytes,
        emitted_output_bytes=emitted_output_bytes,
        decode_forward_steps=decode_forward_steps(emitted_output_bytes),
        overshoot_bytes=(
            0
            if mode == "controlled_replay"
            else valid_output_overshoot(
                emitted_output_bytes,
                len(continuation),
            )
        ),
        valid_output_stop=(
            True
            if mode == "controlled_replay"
            else state.at_codepoint_boundary
        ),
        replacement_character_free="\ufffd" not in output_text,
        valid_jamo_transition=valid_conjoining_jamo_transitions(output_text),
        output_codepoints=len(output_text),
        mps_current_allocated_bytes=current_memory,
        mps_driver_allocated_bytes=driver_memory,
        generated=(
            bytes(generated)
            if mode == "free_running_utf8_greedy"
            else None
        ),
    )
    del runtime, logits
    return result


def _timing_arrays(
    prompt_count: int,
    repetitions: int,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for mode in MODES:
        for role in ROLES:
            for component in COMPONENTS:
                arrays[f"{mode}__{component}__{role}"] = np.zeros(
                    (prompt_count, repetitions),
                    dtype=np.float64,
                )
            arrays[f"{mode}__global_patches__{role}"] = np.zeros(
                (prompt_count, repetitions),
                dtype=np.int32,
            )
            for diagnostic in OUTPUT_DIAGNOSTICS:
                binary = diagnostic in {
                    "valid_output_stop",
                    "replacement_character_free",
                    "valid_jamo_transition",
                }
                arrays[f"{mode}__{diagnostic}__{role}"] = np.zeros(
                    (prompt_count, repetitions),
                    dtype=(np.uint8 if binary else np.int32),
                )
            arrays[f"{mode}__mps_current_bytes__{role}"] = np.full(
                (prompt_count, repetitions),
                -1,
                dtype=np.int64,
            )
            arrays[f"{mode}__mps_driver_bytes__{role}"] = np.full(
                (prompt_count, repetitions),
                -1,
                dtype=np.int64,
            )
    return arrays


def _record_trial(
    arrays: dict[str, np.ndarray],
    mode: str,
    role: str,
    prompt_index: int,
    repetition: int,
    result: TrialResult,
) -> None:
    for component in COMPONENTS:
        arrays[f"{mode}__{component}__{role}"][prompt_index, repetition] = (
            getattr(result, component)
        )
    arrays[f"{mode}__global_patches__{role}"][prompt_index, repetition] = (
        result.emitted_global_patches
    )
    for diagnostic, attribute in (
        ("emitted_output_bytes", "emitted_output_bytes"),
        ("decode_forward_steps", "decode_forward_steps"),
        ("runtime_observed_bytes", "observed_bytes"),
        ("overshoot_bytes", "overshoot_bytes"),
        ("valid_output_stop", "valid_output_stop"),
        ("replacement_character_free", "replacement_character_free"),
        ("valid_jamo_transition", "valid_jamo_transition"),
        ("output_codepoints", "output_codepoints"),
    ):
        arrays[f"{mode}__{diagnostic}__{role}"][prompt_index, repetition] = int(
            getattr(result, attribute)
        )
    if result.mps_current_allocated_bytes is not None:
        arrays[f"{mode}__mps_current_bytes__{role}"][
            prompt_index,
            repetition,
        ] = result.mps_current_allocated_bytes
    if result.mps_driver_allocated_bytes is not None:
        arrays[f"{mode}__mps_driver_bytes__{role}"][
            prompt_index,
            repetition,
        ] = result.mps_driver_allocated_bytes


def _release_bundle(bundle: LoadedPolicy, device: str) -> None:
    bundle.model.to("cpu")
    if bundle.router is not None:
        bundle.router.to("cpu")
    del bundle.model
    if bundle.router is not None:
        del bundle.router
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _run_seed(
    seed: int,
    seed_index: int,
    bundles: dict[str, LoadedPolicy],
    warmup_prompts: np.ndarray,
    measured_prompts: np.ndarray,
    warmup_continuations: np.ndarray,
    measured_continuations: np.ndarray,
    order: np.ndarray,
    warmup_order: np.ndarray,
    device: str,
    *,
    repetitions: int,
    continuation_bytes: int,
    correctness_continuation_bytes: int,
    minimum_verification_positions: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    utf8_masks = _utf8_mask_cache(device)
    correctness = {
        role: _verify_bundle(
            bundle,
            seed,
            warmup_prompts,
            measured_prompts,
            measured_continuations,
            device=device,
            correctness_continuation_bytes=correctness_continuation_bytes,
            minimum_positions=minimum_verification_positions,
        )
        for role, bundle in bundles.items()
    }
    for mode_index, mode in enumerate(MODES):
        for prompt_index, (prompt, continuation) in enumerate(
            zip(warmup_prompts, warmup_continuations, strict=True)
        ):
            roles = (
                ROLES
                if warmup_order[seed_index, mode_index, prompt_index, 0]
                else tuple(reversed(ROLES))
            )
            for role in roles:
                _run_trial(
                    bundles[role],
                    bytes(prompt),
                    bytes(continuation)[:continuation_bytes],
                    mode,
                    device,
                    utf8_masks,
                )
                gc.collect()

    arrays = _timing_arrays(len(measured_prompts), repetitions)
    free_outputs: dict[str, list[bytes | None]] = {
        role: [None] * len(measured_prompts) for role in ROLES
    }
    started = time.perf_counter()
    for mode_index, mode in enumerate(MODES):
        for prompt_index, (prompt, continuation) in enumerate(
            zip(measured_prompts, measured_continuations, strict=True)
        ):
            for repetition in range(repetitions):
                roles = (
                    ROLES
                    if order[seed_index, mode_index, prompt_index, repetition]
                    else tuple(reversed(ROLES))
                )
                for role in roles:
                    result = _run_trial(
                        bundles[role],
                        bytes(prompt),
                        bytes(continuation)[:continuation_bytes],
                        mode,
                        device,
                        utf8_masks,
                    )
                    exact_relationships = bool(
                        result.decode_forward_steps
                        == decode_forward_steps(result.emitted_output_bytes)
                        and result.observed_bytes
                        == runtime_observed_bytes(
                            len(prompt),
                            result.emitted_output_bytes,
                        )
                        and result.valid_output_stop
                    )
                    if mode == "controlled_replay":
                        mode_contract = bool(
                            result.emitted_output_bytes == continuation_bytes
                            and result.overshoot_bytes == 0
                        )
                    else:
                        mode_contract = bool(
                            continuation_bytes
                            <= result.emitted_output_bytes
                            <= free_running_maximum_output_bytes(
                                continuation_bytes
                            )
                            and result.overshoot_bytes
                            == valid_output_overshoot(
                                result.emitted_output_bytes,
                                continuation_bytes,
                            )
                        )
                    if not exact_relationships or not mode_contract:
                        raise AssertionError(
                            "trial does not implement time-to-valid-output"
                        )
                    _record_trial(
                        arrays,
                        mode,
                        role,
                        prompt_index,
                        repetition,
                        result,
                    )
                    if mode == "free_running_utf8_greedy":
                        if result.generated is None:
                            raise AssertionError("free-running trial lacks output")
                        previous = free_outputs[role][prompt_index]
                        if previous is None:
                            free_outputs[role][prompt_index] = result.generated
                        elif previous != result.generated:
                            raise AssertionError("greedy output changed across repetitions")
                    gc.collect()
            if (prompt_index + 1) % 8 == 0 or prompt_index + 1 == len(
                measured_prompts
            ):
                print(
                    f"seed {seed}/{mode}: {prompt_index + 1}/"
                    f"{len(measured_prompts)} prompts",
                    flush=True,
                )
    validate_output_diagnostic_arrays(
        arrays,
        expected_shape=(len(measured_prompts), repetitions),
        minimum_output_bytes=continuation_bytes,
    )
    generation: dict[str, Any] = {}
    for role in ROLES:
        outputs = free_outputs[role]
        if any(value is None for value in outputs):
            raise AssertionError("free-running output set is incomplete")
        generated = [value for value in outputs if value is not None]
        metrics = valid_completion_metrics(
            generated,
            minimum_completion_bytes=continuation_bytes,
        )
        if (
            metrics.valid_utf8_count != len(generated)
            or metrics.maximum_overshoot_bytes
            > FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES
        ):
            raise AssertionError("shared UTF-8 decoder failed its output contract")
        reconstructed = reconstruct_valid_completion_metrics(
            arrays,
            role,
            minimum_output_bytes=continuation_bytes,
        )
        if reconstructed != metrics.to_dict():
            raise AssertionError("free-running aggregate does not reconstruct")
        generation[role] = {
            **reconstructed,
            "utf8_constraint": FREE_RUNNING_UTF8_CONSTRAINT,
            "all_stops_at_strict_utf8_boundary": True,
            "greedy_outputs_identical_across_repetitions": True,
        }
    report = {
        "seed": seed,
        "correctness": correctness,
        "generation": generation,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_provenance": {
            role: bundle.provenance for role, bundle in bundles.items()
        },
        "maximum_process_rss_raw": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "maximum_process_rss_unit": (
            "bytes_on_macos; kibibytes_on_linux"
        ),
    }
    return arrays, report


def _expected_array_keys() -> set[str]:
    return {
        f"{mode}__{metric}__{role}"
        for mode in MODES
        for role in ROLES
        for metric in (
            *COMPONENTS,
            "global_patches",
            *OUTPUT_DIAGNOSTICS,
            "mps_current_bytes",
            "mps_driver_bytes",
        )
    }


def _completed_checkpoint_artifacts_current(
    report: dict[str, Any],
    selection: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> bool:
    recorded_by_role = report.get("checkpoint_provenance", {})
    if set(recorded_by_role) != set(ROLES):
        return False
    for role in ROLES:
        descriptor = selection[role]
        run_root, artifact_root = _policy_paths(descriptor["model_family"], args)
        policy = descriptor["policy"]
        report_path = run_root / f"seed-{seed}" / f"{policy}.json"
        checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
        recorded = recorded_by_role[role]
        if (
            _sha256(report_path)
            != recorded.get("training_report_artifact_sha256")
            or _sha256(checkpoint_path)
            != recorded.get("checkpoint_artifact_sha256")
        ):
            return False
        router = recorded.get("router")
        if descriptor["runtime_policy"] in INCREMENTAL_ENTROPY_POLICIES:
            if not isinstance(router, dict):
                return False
            router_paths = {
                "router_checkpoint_artifact_sha256": (
                    artifact_root / f"seed-{seed}" / "router.pt"
                ),
                "router_report_artifact_sha256": (
                    run_root / f"seed-{seed}" / "router.json"
                ),
                "threshold_cache_artifact_sha256": (
                    artifact_root / f"seed-{seed}" / "threshold-patches.npz"
                ),
                "threshold_diagnostics_artifact_sha256": (
                    run_root
                    / f"seed-{seed}"
                    / "threshold-patch-diagnostics.json"
                ),
            }
            if any(
                _sha256(path) != router.get(key)
                for key, path in router_paths.items()
            ):
                return False
        elif router is not None:
            return False
    return True


def _completed_seed_valid(
    report_path: Path,
    artifact_path: Path,
    *,
    seed: int,
    expected_shape: tuple[int, int],
    maximum_emitted_global_patches: int,
    selection_hash: str,
    quality_hash: str,
    schedule_hash: str,
    selection: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    presence = (report_path.exists(), artifact_path.exists())
    if not any(presence):
        return False
    if not all(presence):
        raise ValueError(f"partial actual-inference result for seed {seed}")
    report = _read_json(report_path)
    if (
        report.get("seed") != seed
        or report.get("selection_sha256") != selection_hash
        or report.get("quality_summary_sha256") != quality_hash
        or report.get("timing_schedule_sha256") != schedule_hash
        or report.get("timing_artifact_sha256") != _sha256(artifact_path)
        or report.get("session_start_timing_environment_eligible") is not True
        or not timing_environment_eligible(report.get("session_start", {}))
        or report.get("session_end_timing_environment_eligible") is not True
        or not timing_environment_eligible(report.get("session_end", {}))
        or not _completed_checkpoint_artifacts_current(
            report,
            selection,
            seed,
            args,
        )
        or set(report.get("correctness", {})) != set(ROLES)
        or any(
            values.get("pass") is not True
            for values in report.get("correctness", {}).values()
        )
    ):
        raise ValueError(f"stale actual-inference report for seed {seed}")
    loaded_arrays: dict[str, np.ndarray] = {}
    with np.load(artifact_path, allow_pickle=False) as archive:
        if set(archive.files) != _expected_array_keys():
            raise ValueError(f"actual-inference timing keys differ for seed {seed}")
        for key in archive.files:
            values = archive[key]
            if report.get("timing_array_sha256", {}).get(key) != _array_sha256(
                values
            ):
                raise ValueError(f"actual-inference array hash differs: {key}")
            if values.shape != expected_shape:
                raise ValueError(f"actual-inference timing shape differs: {key}")
            if "_ms__" in key and (
                values.dtype != np.float64
                or not np.isfinite(values).all()
                or np.any(values <= 0)
            ):
                raise ValueError(f"actual-inference latency values invalid: {key}")
            if "global_patches" in key and (
                not np.issubdtype(values.dtype, np.integer)
                or np.any(values <= 0)
                or np.any(values > maximum_emitted_global_patches)
            ):
                raise ValueError(f"actual-inference patch values invalid: {key}")
            loaded_arrays[key] = values.copy()
    validate_output_diagnostic_arrays(
        loaded_arrays,
        expected_shape=expected_shape,
        minimum_output_bytes=CONTINUATION_BYTES,
    )
    generation = report.get("generation", {})
    if set(generation) != set(ROLES):
        raise ValueError(f"actual-inference generation roles differ: seed {seed}")
    for role in ROLES:
        expected_generation = reconstruct_valid_completion_metrics(
            loaded_arrays,
            role,
        )
        recorded = generation[role]
        if (
            any(
                recorded.get(key) != value
                for key, value in expected_generation.items()
            )
            or recorded.get("utf8_constraint") != FREE_RUNNING_UTF8_CONSTRAINT
            or recorded.get("all_stops_at_strict_utf8_boundary") is not True
            or recorded.get("greedy_outputs_identical_across_repetitions")
            is not True
        ):
            raise ValueError(
                f"actual-inference generation summary differs: seed {seed}/{role}"
            )
    return True


def run(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    quality_path = Path(args.quality_summary)
    selection_path = Path(args.selection)
    quality, selection, phase3 = _validate_quality_gate(
        quality_path,
        selection_path,
    )
    if args.quick:
        seeds = (SEEDS[0],)
        warmup_count = 1
        measured_count = 2
        repetitions = 1
        continuation_bytes = CONTINUATION_BYTES
        correctness_continuation_bytes = 2
        minimum_verification_positions = 4
    else:
        seeds = SEEDS
        warmup_count = WARMUP_CASES
        measured_count = MEASURED_CASES
        repetitions = REPETITIONS
        continuation_bytes = CONTINUATION_BYTES
        correctness_continuation_bytes = CORRECTNESS_CONTINUATION_BYTES
        minimum_verification_positions = 16
    prompts, continuations, case_context = _reconstruct_cases(
        phase3,
        Path(args.data_root),
        total_cases=warmup_count + measured_count,
    )
    warmup_prompts = prompts[:warmup_count]
    measured_prompts = prompts[warmup_count:]
    warmup_continuations = continuations[:warmup_count]
    measured_continuations = continuations[warmup_count:]
    order = timing_order_schedule(
        seeds,
        mode_count=len(MODES),
        prompt_count=measured_count,
        repetitions=repetitions,
        random_seed=TIMING_ORDER_SEED,
    )
    warmup_order = timing_order_schedule(
        seeds,
        mode_count=len(MODES),
        prompt_count=warmup_count,
        repetitions=1,
        random_seed=TIMING_ORDER_SEED + 1,
    )
    seed_execution_order = tuple(
        seeds[index]
        for index in np.random.default_rng(SEED_EXECUTION_ORDER_SEED).permutation(
            len(seeds)
        )
    )
    run_root = Path(
        args.run_root
        or (
            "runs/phase3-actual-inference-smoke"
            if args.quick
            else "runs/phase3-actual-inference"
        )
    )
    artifact_root = Path(
        args.artifact_root
        or (
            "artifacts/phase3-actual-inference-smoke"
            if args.quick
            else "artifacts/phase3-actual-inference"
        )
    )
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    environment = _environment(device)
    session_start = _session_state()
    session_start_eligible = timing_environment_eligible(session_start)
    if device == "mps" and not args.quick and not session_start_eligible:
        raise RuntimeError(
            "evidentiary MPS timing requires AC power and no thermal warning"
        )
    manifest_path = run_root / "manifest.json"
    manifest = {
        "schema_version": ACTUAL_INFERENCE_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "quick_smoke_only": bool(args.quick),
        "evidence_eligible": not args.quick,
        "selection": {"path": str(selection_path), "sha256": _sha256(selection_path)},
        "quality_summary": {"path": str(quality_path), "sha256": _sha256(quality_path)},
        "seeds": list(seeds),
        "seed_execution_order": list(seed_execution_order),
        "roles": {
            role: selection[role] for role in ROLES
        },
        "case_context": case_context,
        "protocol": {
            "prompt_bytes": PROMPT_BYTES,
            "continuation_bytes": continuation_bytes,
            "warmup_cases": warmup_count,
            "measured_cases": measured_count,
            "repetitions_per_prompt": repetitions,
            "modes": list(MODES),
            "components": list(COMPONENTS),
            "correctness_continuation_bytes": correctness_continuation_bytes,
            "minimum_verification_positions": minimum_verification_positions,
            "timing_order_seed": TIMING_ORDER_SEED,
            "seed_execution_order_seed": SEED_EXECUTION_ORDER_SEED,
            "timing_schedule_sha256": _array_sha256(order),
            "warmup_schedule_sha256": _array_sha256(warmup_order),
            "parallel_prefill_only_in_timing": True,
            "selector_router_cache_and_synchronization_inside_timing": True,
            "utf8_dfa_mask_compilation_outside_timing": True,
            "utf8_mask_argmax_state_and_stop_checks_inside_timing": True,
            "free_running_utf8_constraint": FREE_RUNNING_UTF8_CONSTRAINT,
            "time_to_output_semantics": TIME_TO_OUTPUT_SEMANTICS,
            "controlled_replay_decode_forward_steps": decode_forward_steps(
                continuation_bytes
            ),
            "controlled_replay_emitted_output_bytes": continuation_bytes,
            "controlled_replay_runtime_observed_bytes": runtime_observed_bytes(
                PROMPT_BYTES,
                continuation_bytes,
            ),
            "free_running_minimum_output_bytes": continuation_bytes,
            "free_running_maximum_output_bytes": (
                free_running_maximum_output_bytes(continuation_bytes)
            ),
            "free_running_maximum_overshoot_bytes": (
                FREE_RUNNING_MAXIMUM_OVERSHOOT_BYTES
            ),
            "session_start_timing_environment_eligible": (
                session_start_eligible
            ),
        },
        "environment": environment,
        "session_start": session_start,
    }
    invariant_keys = tuple(
        key for key in manifest if key not in {"created_at", "session_start"}
    )
    if manifest_path.exists() and not args.force:
        previous = _read_json(manifest_path)
        if any(previous.get(key) != manifest.get(key) for key in invariant_keys):
            raise ValueError("actual-inference manifest invariant changed")
        manifest = previous
    else:
        _write_json(manifest_path, manifest)

    for seed in seed_execution_order:
        seed_index = seeds.index(seed)
        report_path = run_root / f"seed-{seed}.json"
        artifact_path = artifact_root / f"seed-{seed}-timings.npz"
        in_progress_path = run_root / f"seed-{seed}.in-progress.json"
        schedule_hash = _array_sha256(order[seed_index])
        if in_progress_path.exists() and not args.force:
            raise ValueError(
                f"unfinished actual-inference attempt for seed {seed}; "
                "use a new output root or explicitly rerun with --force"
            )
        if not args.force and _completed_seed_valid(
            report_path,
            artifact_path,
            seed=seed,
            expected_shape=(measured_count, repetitions),
            maximum_emitted_global_patches=runtime_observed_bytes(
                PROMPT_BYTES,
                free_running_maximum_output_bytes(continuation_bytes),
            ),
            selection_hash=_sha256(selection_path),
            quality_hash=_sha256(quality_path),
            schedule_hash=schedule_hash,
            selection=selection,
            args=args,
        ):
            print(f"seed {seed}: actual inference already complete", flush=True)
            continue
        _write_json(
            in_progress_path,
            {
                "schema_version": ACTUAL_INFERENCE_PROTOCOL_VERSION,
                "seed": seed,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
            },
        )
        bundles = {
            role: _load_policy(
                role,
                selection[role],
                seed,
                quality,
                phase3,
                args,
                device,
            )
            for role in ROLES
        }
        seed_session_start = _session_state()
        seed_session_start_eligible = timing_environment_eligible(
            seed_session_start
        )
        if device == "mps" and not args.quick and not seed_session_start_eligible:
            raise RuntimeError(
                f"seed {seed} timing began under ineligible power/thermal state"
            )
        arrays, report = _run_seed(
            seed,
            seed_index,
            bundles,
            warmup_prompts,
            measured_prompts,
            warmup_continuations,
            measured_continuations,
            order,
            warmup_order,
            device,
            repetitions=repetitions,
            continuation_bytes=continuation_bytes,
            correctness_continuation_bytes=correctness_continuation_bytes,
            minimum_verification_positions=minimum_verification_positions,
        )
        session_end = _session_state()
        session_end_eligible = timing_environment_eligible(session_end)
        if device == "mps" and not args.quick and not session_end_eligible:
            raise RuntimeError(
                f"seed {seed} timing ended under ineligible power/thermal state"
            )
        _save_npz(artifact_path, arrays)
        report.update(
            {
                "selection_sha256": _sha256(selection_path),
                "quality_summary_sha256": _sha256(quality_path),
                "timing_schedule_sha256": schedule_hash,
                "timing_artifact_sha256": _sha256(artifact_path),
                "timing_array_sha256": {
                    key: _array_sha256(value) for key, value in arrays.items()
                },
                "session_start": seed_session_start,
                "session_start_timing_environment_eligible": (
                    seed_session_start_eligible
                ),
                "session_end": session_end,
                "session_end_timing_environment_eligible": (
                    session_end_eligible
                ),
            }
        )
        _write_json(report_path, report)
        for bundle in bundles.values():
            _release_bundle(bundle, device)
        in_progress_path.unlink()
        print(f"seed {seed}: actual inference complete", flush=True)
    print("completed actual-inference benchmark", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-summary",
        default="results/phase3-inference-quality/summary.json",
    )
    parser.add_argument(
        "--selection",
        default="results/phase3-inference-selection/selection.json",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--phase3-run-root", default="runs/phase3")
    parser.add_argument("--phase3-artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--conversion-run-root",
        default="runs/phase3-compute-conversion",
    )
    parser.add_argument(
        "--conversion-artifact-root",
        default="artifacts/phase3-compute-conversion",
    )
    parser.add_argument("--run-root")
    parser.add_argument("--artifact-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
