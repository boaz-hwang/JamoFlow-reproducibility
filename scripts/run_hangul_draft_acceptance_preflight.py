#!/usr/bin/env python3
"""Train and evaluate fixed-budget Hangul/UTF-8 draft heads on W72.

The target model is frozen.  Training uses only the historical train split;
architecture comparison and target-generated acceptance use only calibration.
No sealed-final input, model metric, or v5 latency result is consumed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

try:
    from scripts.hangul_draft_acceptance_core import (
        ARCHITECTURES,
        HEAD_TRAINING_SEEDS,
        HIDDEN_WIDTH,
        PRIMARY_GENERIC_CONTROL,
        PRIMARY_HANGUL_DRAFT,
        PROTOCOL_ID,
        DeviceHangulTables,
        array_sha256,
        build_head,
        evaluate_gates,
        hangul_components,
        paired_prompt_bootstrap,
        proposal_metrics,
        propose_pairs,
        trainable_parameter_count,
        training_loss,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from hangul_draft_acceptance_core import (
        ARCHITECTURES,
        HEAD_TRAINING_SEEDS,
        HIDDEN_WIDTH,
        PRIMARY_GENERIC_CONTROL,
        PRIMARY_HANGUL_DRAFT,
        PROTOCOL_ID,
        DeviceHangulTables,
        array_sha256,
        build_head,
        evaluate_gates,
        hangul_components,
        paired_prompt_bootstrap,
        proposal_metrics,
        propose_pairs,
        trainable_parameter_count,
        training_loss,
    )
from jamoflow.compute_conversion import conversion_patch_matrices
from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_runtime_v5 import (
    load_actual_model,
    release_actual_model,
)
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    SELECTION_LOCK_PATH,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import research_versions
from jamoflow.neural_training import synchronize
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import compact_whitespace_mask
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    advance_strict_utf8,
    prefix_boundary_mask,
    strict_utf8_allowed_ranges,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/hangul-draft-acceptance-v1.json"
AUTHORIZATION_PATH = ROOT / FINAL_AUTHORIZATION_PATH
SELECTION_PATH = ROOT / SELECTION_LOCK_PATH
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
RAW_ROOT = ROOT / "artifacts/hangul-draft-acceptance-v1"
TRAIN_CACHE_PATH = RAW_ROOT / "train-hidden.npz"
CALIBRATION_CACHE_PATH = RAW_ROOT / "calibration-hidden.npz"
FREE_CACHE_PATH = RAW_ROOT / "free-target.npz"
HEAD_ROOT = RAW_ROOT / "heads"
PREDICTION_PATH = RAW_ROOT / "predictions.npz"
OUTPUT_PATH = ROOT / "results/hangul-draft-acceptance-v1/summary.json"


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


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _clean_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("draft preflight requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("draft preflight requires a Git commit")
    return commit


def _require_ac_power() -> str:
    state = _command("pmset", "-g", "batt")
    if "Now drawing from 'AC Power'" not in state:
        raise RuntimeError("draft preflight requires AC power")
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


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


def _save_npz_no_clobber(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _save_head_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    import io

    output = io.BytesIO()
    torch.save(dict(payload), output)
    _publish_no_clobber(path, output.getvalue())


def _strict_plan(plan: Mapping[str, Any], commit: str) -> None:
    expected_keys = {
        "architecture_contract",
        "calibration_free_generation",
        "decision_rule",
        "implementation_sha256",
        "input",
        "kind",
        "model",
        "optimization",
        "output",
        "protocol_id",
        "schema_version",
        "status",
        "threat_model",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "hangul_draft_acceptance_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "post_v5r3_exploratory_calibration_only"
        or tuple(plan["architecture_contract"]["order"]) != ARCHITECTURES
        or tuple(plan["optimization"]["training_seeds"]) != HEAD_TRAINING_SEEDS
    ):
        raise ValueError("draft preflight plan schema differs")
    for relative, expected in plan["implementation_sha256"].items():
        path = ROOT / relative
        if hash_file(path) != expected:
            raise ValueError(f"draft implementation differs: {relative}")
    if plan["input"]["source_sha256"] != hash_file(SOURCE_PATH):
        raise ValueError("draft source artifact differs")
    if plan["model"]["authorization_artifact_sha256"] != hash_file(
        AUTHORIZATION_PATH
    ):
        raise ValueError("draft authorization artifact differs")
    if plan["threat_model"] != {
        "final_test_read": False,
        "historical_test_metric_read": False,
        "target_weights_frozen": True,
        "training_split_used_for_head_fit": True,
        "calibration_split_used_for_head_comparison": True,
        "v5_timing_result_numeric_input": False,
    }:
        raise ValueError("draft threat model differs")
    if any(path.exists() for path in (TRAIN_CACHE_PATH, CALIBRATION_CACHE_PATH, FREE_CACHE_PATH, PREDICTION_PATH, OUTPUT_PATH)):
        raise FileExistsError("draft preflight output already exists")
    if HEAD_ROOT.exists() and any(HEAD_ROOT.iterdir()):
        raise FileExistsError("draft head output already exists")
    if _command("git", "rev-parse", "HEAD") != commit:
        raise RuntimeError("draft plan verification changed HEAD")


def _stream_context(plan: Mapping[str, Any], split: str):
    byte_limit = int(plan["input"]["byte_limit"])
    sequence_length = int(plan["input"]["sequence_length"])
    stream = build_neural_stream(
        SOURCE_PATH,
        "ko",
        split,
        byte_limit,
        sequence_length,
    )
    expected = plan["input"]["streams"][split]
    if (
        len(stream.data) != expected["bytes"]
        or hashlib.sha256(stream.data).hexdigest() != expected["stream_sha256"]
        or hashlib.sha256(stream.codepoint_boundaries).hexdigest()
        != expected["boundary_sha256"]
    ):
        raise ValueError(f"draft {split} stream differs")
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    matrix = conversion_patch_matrices(
        boundaries,
        whitespace,
        rate=72,
    )["causal_whitespace_grid_72"]
    if array_sha256(matrix) != expected["patch_matrix_sha256"]:
        raise ValueError(f"draft {split} W72 matrix differs")
    return stream, inputs, boundaries, matrix


def _hangul_locations(inputs: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for sequence_index, (values, boundary) in enumerate(
        zip(inputs, boundaries, strict=True)
    ):
        for start in np.flatnonzero(boundary[1:-2]) + 1:
            raw = bytes(values[start : start + 3])
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(text) == 1 and 0xAC00 <= ord(text) <= 0xD7A3:
                rows.append((sequence_index, int(start)))
    output = np.asarray(rows, dtype=np.int64)
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError("draft Hangul location extraction differs")
    return output


def _sample_locations(
    locations: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    if len(locations) < count:
        raise ValueError("draft stream has too few Hangul contexts")
    indices = np.random.default_rng(seed).choice(len(locations), count, replace=False)
    selected = locations[np.sort(indices)]
    if len(np.unique(selected, axis=0)) != count:
        raise AssertionError("draft context sampling duplicated a location")
    return selected


def _extract_hidden_cache(
    model: Any,
    inputs: np.ndarray,
    matrix: np.ndarray,
    locations: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    hidden = np.empty((len(locations), HIDDEN_WIDTH), dtype=np.float32)
    lead = np.empty(len(locations), dtype=np.uint8)
    second = np.empty(len(locations), dtype=np.uint8)
    third = np.empty(len(locations), dtype=np.uint8)
    onset = np.empty(len(locations), dtype=np.uint8)
    vowel = np.empty(len(locations), dtype=np.uint8)
    coda = np.empty(len(locations), dtype=np.uint8)
    by_sequence: dict[int, list[tuple[int, int]]] = {}
    for output_index, (sequence, start) in enumerate(locations):
        by_sequence.setdefault(int(sequence), []).append((output_index, int(start)))
    sequence_order = np.asarray(sorted(by_sequence), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(sequence_order), batch_size):
            sequence_indices = sequence_order[batch_start : batch_start + batch_size]
            batch = torch.from_numpy(inputs[sequence_indices].astype(np.int64, copy=False)).to(device)
            selected_matrix = matrix[sequence_indices]
            used = np.flatnonzero(np.any(selected_matrix != 0, axis=0))
            patches = torch.from_numpy(
                selected_matrix[:, : int(used[-1]) + 1].astype(np.int64, copy=False)
            ).to(device)
            decoder = model.model(
                input_ids=batch,
                patch_lengths=patches,
                use_cache=False,
            ).last_hidden_state
            decoder_cpu = decoder.float().cpu().numpy()
            for local_row, sequence in enumerate(sequence_indices.tolist()):
                values = inputs[sequence]
                for output_index, start in by_sequence[sequence]:
                    hidden[output_index] = decoder_cpu[local_row, start - 1]
                    raw = bytes(values[start : start + 3])
                    codepoint = ord(raw.decode("utf-8"))
                    l_index, v_index, t_index = hangul_components(codepoint)
                    lead[output_index] = raw[0] - 0xEA
                    second[output_index] = raw[1] - 0x80
                    third[output_index] = raw[2] - 0x80
                    onset[output_index] = l_index
                    vowel[output_index] = v_index
                    coda[output_index] = t_index
            print(
                f"hidden extraction {batch_start + len(sequence_indices)}/{len(sequence_order)} sequences",
                flush=True,
            )
    synchronize(device)
    arrays = {
        "hidden": hidden,
        "lead": lead,
        "second": second,
        "third": third,
        "onset": onset,
        "vowel": vowel,
        "coda": coda,
        "locations": locations.astype(np.int64, copy=False),
    }
    if not np.all(np.isfinite(hidden)):
        raise ValueError("draft hidden cache is nonfinite")
    return arrays


def _hangul_share(text: str) -> float:
    return sum(0xAC00 <= ord(char) <= 0xD7A3 for char in text) / max(1, len(text))


def _select_prompts(
    data: bytes,
    *,
    prompt_bytes: int,
    count: int,
    minimum_hangul_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.frombuffer(prefix_boundary_mask(data), dtype=np.uint8)
    candidates: list[tuple[bytes, int]] = []
    domain = b"JamoFlow/hangul-draft-calibration-prompts/v1\0"
    for start in np.flatnonzero(boundaries[:-prompt_bytes]):
        end = int(start) + prompt_bytes
        if end >= len(boundaries) or not boundaries[end]:
            continue
        raw = data[int(start) : end]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _hangul_share(text) < minimum_hangul_share:
            continue
        key = hashlib.sha256(
            domain + int(start).to_bytes(8, "big") + raw
        ).digest()
        candidates.append((key, int(start)))
    candidates.sort()
    selected: list[int] = []
    for _, start in candidates:
        if any(
            not (start + prompt_bytes <= previous or previous + prompt_bytes <= start)
            for previous in selected
        ):
            continue
        selected.append(start)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("draft prompt selection lacks enough independent cases")
    prompts = np.asarray(
        [np.frombuffer(data[start : start + prompt_bytes], dtype=np.uint8) for start in selected],
        dtype=np.uint8,
    )
    return prompts, np.asarray(selected, dtype=np.int64)


def _utf8_masks(device: str) -> dict[Any, torch.Tensor]:
    from jamoflow.utf8 import strict_utf8_reachable_states

    output = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device=device)
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        output[state] = mask
    synchronize(device)
    return output


def _extract_free_target_cache(
    bundle: Any,
    prompts: np.ndarray,
    *,
    continuation_bytes: int,
    maximum_output_bytes: int,
) -> dict[str, np.ndarray]:
    device = bundle.device
    masks = _utf8_masks(device)
    hidden_rows: list[np.ndarray] = []
    lead_rows: list[int] = []
    second_rows: list[int] = []
    third_rows: list[int] = []
    target_hangul: list[bool] = []
    prompt_rows: list[int] = []
    output_offsets: list[int] = []
    generated_hashes: list[bytes] = []
    latest: list[torch.Tensor | None] = [None]

    def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        latest[0] = output[:, -1, :].detach()

    handle = bundle.model.model.local_decoder.norm.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for prompt_index, prompt_values in enumerate(prompts):
                prompt = bytes(prompt_values)
                runtime = bundle.runtime()
                logits = runtime.prefill_parallel(prompt)
                state = STRICT_UTF8_INITIAL_STATE
                generated = bytearray()
                pending: dict[str, Any] | None = None
                while True:
                    if latest[0] is None:
                        raise AssertionError("draft runtime hidden hook did not fire")
                    value = int(
                        logits.masked_fill(~masks[state], -torch.inf)
                        .argmax(dim=-1)
                        .item()
                    )
                    if state.at_codepoint_boundary and 0xEA <= value <= 0xED:
                        pending = {
                            "hidden": latest[0][0].float().cpu().numpy().copy(),
                            "lead": value - 0xEA,
                            "offset": len(generated),
                            "bytes": bytearray(),
                        }
                    generated.append(value)
                    if pending is not None:
                        pending["bytes"].append(value)
                    state = advance_strict_utf8(state, value)
                    if not state.valid:
                        raise AssertionError("draft free target emitted invalid UTF-8")
                    if pending is not None and state.at_codepoint_boundary:
                        raw = bytes(pending["bytes"])
                        if len(raw) != 3:
                            raise AssertionError("EA-ED target scalar length differs")
                        scalar = ord(raw.decode("utf-8"))
                        hidden_rows.append(pending["hidden"])
                        lead_rows.append(pending["lead"])
                        second_rows.append(raw[1] - 0x80)
                        third_rows.append(raw[2] - 0x80)
                        target_hangul.append(0xAC00 <= scalar <= 0xD7A3)
                        prompt_rows.append(prompt_index)
                        output_offsets.append(pending["offset"])
                        pending = None
                    if len(generated) >= continuation_bytes and state.at_codepoint_boundary:
                        break
                    if len(generated) >= maximum_output_bytes:
                        raise RuntimeError("draft free target exceeded maximum output bytes")
                    logits = runtime.consume(value)
                generated_hashes.append(hashlib.sha256(bytes(generated)).digest())
                print(f"free target {prompt_index + 1}/{len(prompts)} prompts", flush=True)
    finally:
        handle.remove()
    if not hidden_rows:
        raise ValueError("draft free target produced no EA-ED scalar attempts")
    output = {
        "hidden": np.asarray(hidden_rows, dtype=np.float32),
        "lead": np.asarray(lead_rows, dtype=np.uint8),
        "second": np.asarray(second_rows, dtype=np.uint8),
        "third": np.asarray(third_rows, dtype=np.uint8),
        "target_is_hangul": np.asarray(target_hangul, dtype=np.bool_),
        "prompt_index": np.asarray(prompt_rows, dtype=np.int64),
        "output_offset": np.asarray(output_offsets, dtype=np.int64),
        "generated_sha256": np.frombuffer(b"".join(generated_hashes), dtype=np.uint8).reshape(
            len(generated_hashes), 32
        ),
    }
    if output["hidden"].shape != (len(lead_rows), HIDDEN_WIDTH):
        raise AssertionError("draft free target hidden shape differs")
    return output


def _batch_tensors(cache: Mapping[str, np.ndarray], indices: np.ndarray, device: str):
    return tuple(
        torch.from_numpy(cache[key][indices].astype(dtype, copy=False)).to(device)
        for key, dtype in (
            ("hidden", np.float32),
            ("lead", np.int64),
            ("second", np.int64),
            ("third", np.int64),
            ("onset", np.int64),
            ("vowel", np.int64),
            ("coda", np.int64),
        )
    )


def _train_head(
    architecture: str,
    seed: int,
    cache: Mapping[str, np.ndarray],
    optimization: Mapping[str, Any],
    *,
    device: str,
) -> tuple[Any, list[float]]:
    torch.manual_seed(seed)
    model = build_head(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    batch_size = int(optimization["batch_size"])
    history: list[float] = []
    for epoch in range(int(optimization["epochs"])):
        order = np.random.default_rng(seed + epoch).permutation(len(cache["hidden"]))
        loss_sum = 0.0
        examples = 0
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            tensors = _batch_tensors(cache, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, *tensors)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimization["gradient_clip"]),
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            examples += len(indices)
        synchronize(device)
        history.append(loss_sum / examples)
        print(
            f"head {architecture} seed={seed} epoch={epoch + 1}/{optimization['epochs']} loss={history[-1]:.6f}",
            flush=True,
        )
    return model.eval(), history


def _predict(
    model: Any,
    cache: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    output = np.empty(len(cache["hidden"]), dtype=np.int64)
    tables = DeviceHangulTables.build(device)
    with torch.inference_mode():
        for start in range(0, len(output), batch_size):
            end = min(len(output), start + batch_size)
            hidden = torch.from_numpy(cache["hidden"][start:end]).to(device)
            lead = torch.from_numpy(cache["lead"][start:end].astype(np.int64)).to(device)
            output[start:end] = propose_pairs(model, hidden, lead, tables).cpu().numpy()
    return output


def _head_latency(
    model: Any,
    cache: Mapping[str, np.ndarray],
    *,
    warmups: int,
    repetitions: int,
    device: str,
) -> np.ndarray:
    if len(cache["hidden"]) < 1:
        raise ValueError("draft latency cache is empty")
    tables = DeviceHangulTables.build(device)
    hidden = torch.from_numpy(cache["hidden"]).to(device)
    lead = torch.from_numpy(cache["lead"].astype(np.int64)).to(device)
    with torch.inference_mode():
        for index in range(warmups):
            row = index % len(hidden)
            value = propose_pairs(model, hidden[row : row + 1], lead[row : row + 1], tables)
            int(value.item())
        synchronize(device)
        timings = np.empty(repetitions, dtype=np.float64)
        for index in range(repetitions):
            row = index % len(hidden)
            synchronize(device)
            started = time.perf_counter_ns()
            value = propose_pairs(model, hidden[row : row + 1], lead[row : row + 1], tables)
            int(value.item())
            synchronize(device)
            timings[index] = (time.perf_counter_ns() - started) / 1_000_000
    return timings


def _checkpoint_payload(
    model: Any,
    *,
    architecture: str,
    seed: int,
    plan_sha256: str,
    history: list[float],
) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "head_seed": seed,
        "parameter_count": trainable_parameter_count(model),
        "plan_artifact_sha256": plan_sha256,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training_loss_by_epoch": history,
    }


def run() -> None:
    commit = _clean_commit()
    plan = _read_json(PLAN_PATH)
    _strict_plan(plan, commit)
    plan_sha256 = hash_file(PLAN_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)
    selection_lock = _read_json(SELECTION_PATH)
    validate_selection_lock_v2(selection_lock)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection_lock,
    )
    candidate = next(
        model for model in authorization["models"] if model["artifact_role"] == "candidate"
    )
    model_plan = plan["model"]
    if (
        candidate["identity_sha256"] != model_plan["candidate_identity_sha256"]
        or candidate["descriptor"] != model_plan["candidate_descriptor"]
        or candidate["seeds"][str(model_plan["seed"])]["checkpoint"]
        != model_plan["checkpoint"]
    ):
        raise ValueError("draft target identity differs")
    power_sha256 = _require_ac_power()
    started = time.time()
    predictions: dict[str, np.ndarray] = {}
    per_run: dict[str, Any] = {}
    with publication_mps_exclusive():
        train_stream, train_inputs, train_boundaries, train_matrix = _stream_context(
            plan, "train"
        )
        calibration_stream, calibration_inputs, calibration_boundaries, calibration_matrix = (
            _stream_context(plan, "calibration")
        )
        target_seed = int(model_plan["seed"])
        bundle = load_actual_model(
            role="candidate",
            identity=candidate,
            seed=target_seed,
            device="mps",
        )
        sample_count = int(plan["input"]["sample_count_per_split"])
        sample_seed = int(plan["input"]["context_sampling_seed"])
        train_locations = _sample_locations(
            _hangul_locations(train_inputs, train_boundaries),
            count=sample_count,
            seed=sample_seed,
        )
        calibration_locations = _sample_locations(
            _hangul_locations(calibration_inputs, calibration_boundaries),
            count=sample_count,
            seed=sample_seed,
        )
        teacher_batch = int(plan["input"]["target_batch_size"])
        train_cache = _extract_hidden_cache(
            bundle.model,
            train_inputs,
            train_matrix,
            train_locations,
            batch_size=teacher_batch,
            device="mps",
        )
        calibration_cache = _extract_hidden_cache(
            bundle.model,
            calibration_inputs,
            calibration_matrix,
            calibration_locations,
            batch_size=teacher_batch,
            device="mps",
        )
        free_plan = plan["calibration_free_generation"]
        prompts, prompt_offsets = _select_prompts(
            calibration_stream.data,
            prompt_bytes=int(free_plan["prompt_bytes"]),
            count=int(free_plan["prompt_count"]),
            minimum_hangul_share=float(free_plan["minimum_prompt_hangul_share"]),
        )
        free_cache = _extract_free_target_cache(
            bundle,
            prompts,
            continuation_bytes=int(free_plan["continuation_bytes"]),
            maximum_output_bytes=int(free_plan["maximum_output_bytes"]),
        )
        release_actual_model(bundle)
        gc.collect()
        torch.mps.empty_cache()

        train_cache_with_context = {
            **train_cache,
            "stream_sha256": np.frombuffer(
                bytes.fromhex(plan["input"]["streams"]["train"]["stream_sha256"]),
                dtype=np.uint8,
            ),
        }
        calibration_cache_with_context = {
            **calibration_cache,
            "stream_sha256": np.frombuffer(
                bytes.fromhex(
                    plan["input"]["streams"]["calibration"]["stream_sha256"]
                ),
                dtype=np.uint8,
            ),
        }
        free_cache_with_context = {
            **free_cache,
            "prompts": prompts,
            "prompt_offsets": prompt_offsets,
        }
        _save_npz_no_clobber(TRAIN_CACHE_PATH, train_cache_with_context)
        _save_npz_no_clobber(CALIBRATION_CACHE_PATH, calibration_cache_with_context)
        _save_npz_no_clobber(FREE_CACHE_PATH, free_cache_with_context)

        optimization = plan["optimization"]
        prediction_batch = int(optimization["prediction_batch_size"])
        latency_plan = optimization["isolated_latency"]
        latency_cache = {
            key: value[: min(128, len(value))]
            for key, value in free_cache.items()
            if key in {"hidden", "lead"}
        }
        for architecture in ARCHITECTURES:
            for head_seed in HEAD_TRAINING_SEEDS:
                run_key = f"{architecture}__{head_seed}"
                model, history = _train_head(
                    architecture,
                    head_seed,
                    train_cache,
                    optimization,
                    device="mps",
                )
                teacher_prediction = _predict(
                    model,
                    calibration_cache,
                    batch_size=prediction_batch,
                    device="mps",
                )
                free_prediction = _predict(
                    model,
                    free_cache,
                    batch_size=prediction_batch,
                    device="mps",
                )
                latency = _head_latency(
                    model,
                    latency_cache,
                    warmups=int(latency_plan["warmups"]),
                    repetitions=int(latency_plan["repetitions"]),
                    device="mps",
                )
                checkpoint_path = HEAD_ROOT / f"{architecture}-seed-{head_seed}.pt"
                _save_head_no_clobber(
                    checkpoint_path,
                    _checkpoint_payload(
                        model,
                        architecture=architecture,
                        seed=head_seed,
                        plan_sha256=plan_sha256,
                        history=history,
                    ),
                )
                predictions[f"teacher__{run_key}"] = teacher_prediction
                predictions[f"free__{run_key}"] = free_prediction
                predictions[f"latency_ms__{run_key}"] = latency
                teacher_metrics = proposal_metrics(
                    teacher_prediction,
                    calibration_cache["second"],
                    calibration_cache["third"],
                    np.ones(len(teacher_prediction), dtype=np.bool_),
                )
                free_metrics = proposal_metrics(
                    free_prediction,
                    free_cache["second"],
                    free_cache["third"],
                    free_cache["target_is_hangul"],
                )
                per_run[run_key] = {
                    "architecture": architecture,
                    "head_seed": head_seed,
                    "parameter_count": trainable_parameter_count(model),
                    "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
                    "checkpoint_artifact_sha256": hash_file(checkpoint_path),
                    "training_loss_by_epoch": history,
                    "teacher_forced_calibration": teacher_metrics,
                    "free_target_calibration": free_metrics,
                    "isolated_head_latency_ms": {
                        "median": float(np.median(latency)),
                        "p95": float(np.quantile(latency, 0.95)),
                        "minimum": float(latency.min()),
                        "maximum": float(latency.max()),
                    },
                    "prediction_sha256": {
                        "teacher": array_sha256(teacher_prediction),
                        "free": array_sha256(free_prediction),
                        "latency": array_sha256(latency),
                    },
                }
                model.to("cpu")
                del model
                gc.collect()
                torch.mps.empty_cache()
        _save_npz_no_clobber(PREDICTION_PATH, predictions)

    architecture_summary: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        rows = [per_run[f"{architecture}__{seed}"] for seed in HEAD_TRAINING_SEEDS]
        free_acceptance = {
            str(seed): float(row["free_target_calibration"]["complete_pair_acceptance"])
            for seed, row in zip(HEAD_TRAINING_SEEDS, rows, strict=True)
        }
        suffix = [
            float(row["free_target_calibration"]["mean_accepted_suffix_bytes"])
            for row in rows
        ]
        latency = [float(row["isolated_head_latency_ms"]["median"]) for row in rows]
        architecture_summary[architecture] = {
            "parameter_count": int(rows[0]["parameter_count"]),
            "free_attempt_count": int(rows[0]["free_target_calibration"]["attempt_count"]),
            "per_seed_free_complete_pair_acceptance": free_acceptance,
            "median_free_complete_pair_acceptance": float(
                np.median(list(free_acceptance.values()))
            ),
            "median_free_mean_accepted_suffix_bytes": float(np.median(suffix)),
            "median_head_latency_ms": float(np.median(latency)),
            "maximum_head_latency_p95_ms": float(
                max(row["isolated_head_latency_ms"]["p95"] for row in rows)
            ),
            "median_teacher_forced_complete_pair_accuracy": float(
                np.median(
                    [
                        row["teacher_forced_calibration"]["complete_pair_acceptance"]
                        for row in rows
                    ]
                )
            ),
        }

    left_exact = []
    right_exact = []
    prompt_indices = []
    target_pair = (
        free_cache["second"].astype(np.int64) * 64
        + free_cache["third"].astype(np.int64)
    )
    for head_seed in HEAD_TRAINING_SEEDS:
        left = predictions[f"free__{PRIMARY_HANGUL_DRAFT}__{head_seed}"]
        right = predictions[f"free__{PRIMARY_GENERIC_CONTROL}__{head_seed}"]
        left_exact.append(left == target_pair)
        right_exact.append(right == target_pair)
        prompt_indices.append(free_cache["prompt_index"])
    specificity = paired_prompt_bootstrap(
        np.concatenate(left_exact).astype(np.bool_),
        np.concatenate(right_exact).astype(np.bool_),
        np.concatenate(prompt_indices).astype(np.int64),
        repetitions=int(plan["decision_rule"]["bootstrap_repetitions"]),
        seed=int(plan["decision_rule"]["bootstrap_seed"]),
    )
    gate_plan = plan["decision_rule"]
    gates = evaluate_gates(
        architecture_summary,
        specificity,
        minimum_attempts=int(gate_plan["minimum_free_attempts"]),
        minimum_complete_pair_acceptance=float(
            gate_plan["minimum_complete_pair_acceptance"]
        ),
        minimum_mean_accepted_suffix_bytes=float(
            gate_plan["minimum_mean_accepted_suffix_bytes"]
        ),
        minimum_per_seed_complete_pair_acceptance=float(
            gate_plan["minimum_per_seed_complete_pair_acceptance"]
        ),
        maximum_median_head_latency_ms=float(
            gate_plan["maximum_median_head_latency_ms"]
        ),
        minimum_specificity_acceptance_gain=float(
            gate_plan["minimum_specificity_acceptance_gain"]
        ),
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "hangul_draft_acceptance_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "hangul_prototype_authorized"
            if gates["overall_hangul_prototype_authorized"]
            else "hangul_prototype_not_authorized"
        ),
        "provenance": {
            "git_commit": commit,
            "plan_path": PLAN_PATH.relative_to(ROOT).as_posix(),
            "plan_artifact_sha256": plan_sha256,
            "authorization_artifact_sha256": hash_file(AUTHORIZATION_PATH),
            "candidate_identity_sha256": candidate["identity_sha256"],
            "target_seed": int(plan["model"]["seed"]),
            "power_snapshot_sha256": power_sha256,
            "runtime": {
                **research_versions(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
        },
        "data": {
            "train_cache_examples": len(train_cache["hidden"]),
            "calibration_cache_examples": len(calibration_cache["hidden"]),
            "free_prompt_count": len(prompts),
            "free_attempt_count": len(free_cache["hidden"]),
            "free_target_hangul_count": int(free_cache["target_is_hangul"].sum()),
            "free_target_hangul_rate": float(free_cache["target_is_hangul"].mean()),
            "cache_artifact_sha256": {
                "train": hash_file(TRAIN_CACHE_PATH),
                "calibration": hash_file(CALIBRATION_CACHE_PATH),
                "free_target": hash_file(FREE_CACHE_PATH),
                "predictions": hash_file(PREDICTION_PATH),
            },
            "cache_array_sha256": {
                "train_hidden": array_sha256(train_cache["hidden"]),
                "calibration_hidden": array_sha256(calibration_cache["hidden"]),
                "free_hidden": array_sha256(free_cache["hidden"]),
                "prompts": array_sha256(prompts),
                "prompt_offsets": array_sha256(prompt_offsets),
            },
        },
        "architecture_summary": architecture_summary,
        "per_training_run": per_run,
        "gates": gates,
        "claim_boundary": {
            "calibration_only": True,
            "frozen_single_target_seed": True,
            "actual_block_verifier_measured": False,
            "actual_end_to_end_speed_claimed": False,
            "quality_claimed": False,
            "generic_joint_prior_displaced": False,
            "pass_authorizes": "exact target block verifier prototype only",
        },
        "elapsed_seconds": float(time.time() - started),
    }
    summary["summary_sha256"] = hashlib.sha256(_json_bytes(summary)).hexdigest()
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("draft preflight changed tracked repository state")
    _publish_no_clobber(OUTPUT_PATH, _json_bytes(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "free_attempt_count": summary["data"]["free_attempt_count"],
                "primary_specificity": gates["primary_korean_specificity"],
                "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
