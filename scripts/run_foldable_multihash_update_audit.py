#!/usr/bin/env python3
"""Run the sealed first-update geometry audit for the foldable multi-hash role."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from compositional_head_preflight_protocol import load_tokenizers
from foldable_jamo_residual_core import (
    CODEBOOK_SIZE,
    RESIDUAL_SLOT_COUNT,
    build_foldable_model,
    build_folded_dense_model,
    folded_dense_state,
)
from foldable_jamo_residual_protocol import (
    IMPLEMENTATION_PATHS as PARENT_IMPLEMENTATION_PATHS,
)
from foldable_jamo_residual_protocol import (
    OUTPUT_PATH as PARENT_RESULT_PATH,
)
from foldable_jamo_residual_protocol import (
    PLAN_PATH as PARENT_PLAN_PATH,
)
from foldable_jamo_residual_protocol import (
    ROOT,
    base_checkpoint_state,
    current_environment,
    hash_file,
    read_json,
    target_order,
    training_contract,
)
from foldable_jamo_residual_protocol import (
    canonical_sha256 as parent_canonical_sha256,
)
from foldable_jamo_residual_protocol import (
    dependency_identity as parent_dependency_identity,
)
from foldable_multihash_update_audit_core import (
    PROTOCOL_ID,
    array_sha256,
    select_update_matched_control,
    update_geometry,
)
from run_foldable_jamo_residual import (
    _cleanup_data,
    _role_data,
    _scheduled_exposure_counts,
)
from run_foldable_jamo_residual import (
    _optimizer as residual_optimizer,
)
from run_vocabulary_transfer_baseline import _all_parameter_optimizer as dense_optimizer
from seal_foldable_multihash_update_audit_plan import (
    IMPLEMENTATION_PATHS,
    PLAN_PATH,
    RESULT_PATH,
    ROLE,
    WORKER_PATH,
)
from vocabulary_transfer_baseline_core import state_mapping_sha256
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    EFFECTIVE_BATCH_SIZE,
    GRADIENT_CLIP,
    HEAD_PEAK_LEARNING_RATE,
    TARGET_VOCABULARY_SIZE,
    TRAIN_MICROBATCH_SIZE,
    build_canonical_bpe_decomposition_table,
    probe_learning_rate,
)

from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True
    ).stdout


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def _history(path: Path) -> tuple[str, ...]:
    output = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    return tuple(line for line in output.splitlines() if line)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_plan(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "batch",
        "claim_boundary",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "metric",
        "optimizer",
        "output_path",
        "parent",
        "plan_sha256",
        "protocol_id",
        "role",
        "schema_version",
        "status",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != "foldable_multihash_update_audit_plan_v4"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("role") != ROLE
        or plan.get("output_path") != str(RESULT_PATH.relative_to(ROOT))
    ):
        raise ValueError("update-audit plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if _canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("update-audit plan hash differs")
    if plan["implementation_sha256"] != {
        path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS
    }:
        raise ValueError("update-audit implementation differs")
    if (
        plan["environment"] != current_environment()
        or plan["optimizer"] != training_contract()
    ):
        raise ValueError("update-audit runtime contract differs")
    parent_plan = read_json(PARENT_PLAN_PATH)
    parent_result = read_json(PARENT_RESULT_PATH)
    worker = read_json(WORKER_PATH)
    checkpoint = worker["checkpoints"]["0"]
    expected_parent = {
        "plan_path": str(PARENT_PLAN_PATH.relative_to(ROOT)),
        "plan_artifact_sha256": hash_file(PARENT_PLAN_PATH),
        "plan_payload_sha256": parent_plan["plan_sha256"],
        "result_path": str(PARENT_RESULT_PATH.relative_to(ROOT)),
        "result_artifact_sha256": hash_file(PARENT_RESULT_PATH),
        "result_payload_sha256": parent_result["summary_sha256"],
        "worker_path": str(WORKER_PATH.relative_to(ROOT)),
        "worker_artifact_sha256": hash_file(WORKER_PATH),
        "worker_payload_sha256": worker["worker_sha256"],
        "step_zero_checkpoint_path": checkpoint["checkpoint_path"],
        "step_zero_checkpoint_artifact_sha256": checkpoint[
            "checkpoint_artifact_sha256"
        ],
        "step_zero_checkpoint_state_sha256": checkpoint["checkpoint_state_sha256"],
    }
    if plan["parent"] != expected_parent:
        raise ValueError("update-audit parent identity differs")
    order = target_order(int(training_contract()["sequence_count"]))
    if plan["batch"] != {
        "sequence_count": int(training_contract()["sequence_count"]),
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "first_batch_indices_sha256": array_sha256(order[:EFFECTIVE_BATCH_SIZE]),
        "training_order_prefix_sha256": training_contract()[
            "training_order_prefix_sha256"
        ],
    }:
        raise ValueError("update-audit batch identity differs")


def _validate_parent_plan_for_historical_replay(plan: Mapping[str, Any]) -> None:
    """Verify the old plan without requiring later result-doc amendments to revert."""

    unsigned = dict(plan)
    plan_sha256 = unsigned.pop("plan_sha256", None)
    if (
        plan.get("kind") != "foldable_jamo_residual_plan_v1"
        or parent_canonical_sha256(unsigned) != plan_sha256
        or plan.get("dependencies") != parent_dependency_identity()
        or plan.get("training") != training_contract()
        or not isinstance(plan.get("implementation_sha256"), Mapping)
        or len(plan["implementation_sha256"]) != len(PARENT_IMPLEMENTATION_PATHS)
        or set(plan["implementation_sha256"]) != set(PARENT_IMPLEMENTATION_PATHS)
    ):
        raise ValueError("historical foldable parent plan differs")
    base_commit = plan.get("git_commit_before_plan")
    if not isinstance(base_commit, str) or len(base_commit) != 40:
        raise ValueError("historical foldable parent commit differs")
    allowed_current_drift = {
        "docs/139-strong-vocabulary-transfer-baseline-result-and-foldable-jamo-decision.md",
        "docs/140-foldable-jamo-residual-protocol.md",
    }
    observed_current_drift: set[str] = set()
    for path, expected in plan["implementation_sha256"].items():
        historical = hashlib.sha256(_git_bytes("show", f"{base_commit}:{path}"))
        if historical.hexdigest() != expected:
            raise ValueError("historical foldable parent blob differs")
        if hash_file(ROOT / path) != expected:
            observed_current_drift.add(path)
    if observed_current_drift != allowed_current_drift:
        raise ValueError("historical foldable parent runtime drift differs")


def _set_step_zero_learning_rates(optimizer: torch.optim.Optimizer) -> None:
    head = probe_learning_rate(0, peak=HEAD_PEAK_LEARNING_RATE, minimum=3e-5)
    for group in optimizer.param_groups:
        group["lr"] = 3e-5 if group["schedule_kind"] == "body" else head


def _sequence_view(values: np.ndarray, sequence_count: int) -> np.ndarray:
    if sequence_count <= 0 or values.ndim != 1 or len(values) < sequence_count * 512:
        raise ValueError("update-audit token memmap differs")
    output = values[: sequence_count * 512].reshape(sequence_count, 512)
    if output.dtype != np.int64:
        raise ValueError("update-audit sequence dtype differs")
    return output


def _backward_and_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    batch: np.ndarray,
    *,
    lexical: Callable[[Any], tuple[torch.Tensor, torch.Tensor]],
    residual_tables: Callable[[Any], tuple[torch.Tensor, torch.Tensor] | None],
) -> dict[str, Any]:
    model.train()
    _set_step_zero_learning_rates(optimizer)
    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    for start in range(0, len(batch), TRAIN_MICROBATCH_SIZE):
        values = torch.tensor(
            batch[start : start + TRAIN_MICROBATCH_SIZE],
            dtype=torch.long,
            device="mps",
        )
        output = model(input_ids=values, labels=values, use_cache=False)
        loss = output.loss * (TRAIN_MICROBATCH_SIZE / len(batch))
        if not torch.isfinite(loss):
            raise RuntimeError("update-audit loss is nonfinite")
        micro_losses.append(float(loss.detach().cpu()))
        loss.backward()
        del values, output, loss
    input_weight, output_weight = lexical(model)
    if input_weight.grad is None or output_weight.grad is None:
        raise RuntimeError("update-audit lexical gradient is absent")
    input_gradient = input_weight.grad.detach().cpu().float().contiguous().numpy()
    output_gradient = output_weight.grad.detach().cpu().float().contiguous().numpy()
    table_values = residual_tables(model)
    table_gradients = None
    if table_values is not None:
        input_table, output_table = table_values
        if input_table.grad is None or output_table.grad is None:
            raise RuntimeError("update-audit residual gradient is absent")
        table_gradients = (
            input_table.grad.detach().cpu().float().contiguous().numpy(),
            output_table.grad.detach().cpu().float().contiguous().numpy(),
        )
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    total_norm_value = float(total_norm.detach().cpu())
    if not math.isfinite(total_norm_value) or total_norm_value <= 0.0:
        raise RuntimeError("update-audit gradient norm differs")
    optimizer.step()
    torch.mps.synchronize()
    return {
        "micro_losses": np.asarray(micro_losses, dtype=np.float32),
        "input_gradient": input_gradient,
        "output_gradient": output_gradient,
        "table_gradients": table_gradients,
        "preclip_total_norm": total_norm_value,
        "clip_coefficient": min(1.0, GRADIENT_CLIP / total_norm_value),
    }


def _bucket_alignment(
    row_gradient: np.ndarray,
    table_gradient: np.ndarray,
    assignment: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray(row_gradient, dtype=np.float32)
    tables = np.asarray(table_gradient, dtype=np.float32)
    codes = np.asarray(assignment, dtype=np.int64)
    if (
        rows.ndim != 2
        or len(rows) < 2
        or tables.ndim != 3
        or tables.shape[0] != RESIDUAL_SLOT_COUNT
        or tables.shape[2] != rows.shape[1]
        or codes.shape != (len(rows), RESIDUAL_SLOT_COUNT)
        or np.any(codes < 0)
        or np.any(codes >= tables.shape[1])
        or not np.isfinite(rows).all()
        or not np.isfinite(tables).all()
    ):
        raise ValueError("bucket-alignment inputs differ")
    rows64 = rows.astype(np.float64, copy=False)
    row_norm = np.linalg.norm(rows64, axis=1)
    nonzero_rows = row_norm > 0.0
    nonzero_row_count = int(nonzero_rows.sum())
    if nonzero_row_count == 0:
        raise ValueError("bucket-alignment has no nonzero row gradient")
    per_slot: list[dict[str, Any]] = []
    all_cosines: list[np.ndarray] = []
    all_sign: list[float] = []
    for slot in range(RESIDUAL_SLOT_COUNT):
        selected = tables[slot, codes[:, slot]].astype(np.float64, copy=False)
        selected_norm = np.linalg.norm(selected, axis=1)
        valid = nonzero_rows & (selected_norm > 0.0)
        aligned_row_count = int(valid.sum())
        if aligned_row_count == 0:
            raise ValueError("bucket-alignment has no aligned row")
        cosine = (rows64[valid] * selected[valid]).sum(axis=1) / (
            row_norm[valid] * selected_norm[valid]
        )
        if not np.isfinite(cosine).all():
            raise ValueError("bucket-alignment cosine is nonfinite")
        sign = float(
            (np.sign(rows64[valid]) == np.sign(selected[valid])).mean()
        )
        all_cosines.append(cosine)
        all_sign.append(sign)
        per_slot.append(
            {
                "slot": slot,
                "aligned_row_count": aligned_row_count,
                "zero_selected_bucket_gradient_count_among_nonzero_rows": int(
                    (nonzero_rows & (selected_norm == 0.0)).sum()
                ),
                "mean_cosine": float(cosine.mean()),
                "median_cosine": float(np.median(cosine)),
                "coordinate_sign_agreement": sign,
            }
        )
    concatenated = np.concatenate(all_cosines)
    return {
        "total_row_count": len(rows),
        "nonzero_row_gradient_count": nonzero_row_count,
        "zero_row_gradient_count": len(rows) - nonzero_row_count,
        "aligned_pair_count": int(sum(len(values) for values in all_cosines)),
        "mean_cosine": float(concatenated.mean()),
        "median_cosine": float(np.median(concatenated)),
        "p10_cosine": float(np.quantile(concatenated, 0.10)),
        "p90_cosine": float(np.quantile(concatenated, 0.90)),
        "mean_coordinate_sign_agreement": float(np.mean(all_sign)),
        "per_slot": per_slot,
    }


def _run_locked(plan: Mapping[str, Any]) -> dict[str, Any]:
    parent_plan = read_json(PARENT_PLAN_PATH)
    _validate_parent_plan_for_historical_replay(parent_plan)
    worker = read_json(WORKER_PATH)
    checkpoint_row = worker["checkpoints"]["0"]
    checkpoint_path = ROOT / checkpoint_row["checkpoint_path"]
    if hash_file(checkpoint_path) != checkpoint_row["checkpoint_artifact_sha256"]:
        raise RuntimeError("update-audit step-zero checkpoint file differs")
    checkpoint_state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if (
        state_mapping_sha256(checkpoint_state)
        != checkpoint_row["checkpoint_state_sha256"]
    ):
        raise RuntimeError("update-audit step-zero checkpoint state differs")

    data = _role_data(parent_plan)
    try:
        train_count = int(data["train_inventory"].full_sequence_count)
        train_sequences = _sequence_view(data["train_memory"], train_count)
        order = target_order(train_count)
        selected = order[:EFFECTIVE_BATCH_SIZE]
        batch = np.asarray(train_sequences[selected], dtype=np.int64)
        if batch.shape != (EFFECTIVE_BATCH_SIZE, 512):
            raise RuntimeError("update-audit first batch differs")
        batch_sha256 = array_sha256(batch)
        exposure_counts = _scheduled_exposure_counts(train_sequences, order)

        tokenizers = load_tokenizers()
        base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
        target_tokenizer, target_pieces = tokenizers[TARGET_VOCABULARY_SIZE]
        decompositions = build_canonical_bpe_decomposition_table(
            base_tokenizer,
            target_tokenizer,
            base_pieces,
            target_pieces,
        )
        multihash, _, assignment_audit = build_foldable_model(
            ROLE,
            base_state=base_checkpoint_state(),
            base_tokenizer=base_tokenizer,
            base_pieces=base_pieces,
            target_pieces=target_pieces,
            decompositions=decompositions,
            exposure_counts=exposure_counts,
        )
        multihash.load_state_dict(checkpoint_state, strict=True)
        if (
            state_mapping_sha256(multihash.state_dict())
            != checkpoint_row["checkpoint_state_sha256"]
        ):
            raise RuntimeError("update-audit loaded multihash state differs")
        dense = build_folded_dense_model(multihash, ROLE)
        if (
            state_mapping_sha256(dense.state_dict())
            != worker["initialization_identity"]["folded_dense_state_sha256"]
        ):
            raise RuntimeError("update-audit dense folded state differs")
        if state_mapping_sha256(
            folded_dense_state(multihash, ROLE)
        ) != state_mapping_sha256(dense.state_dict()):
            raise RuntimeError("update-audit initial effective state differs")

        dense_input_before = dense.model.embed_tokens.weight.detach().cpu().clone()
        dense_output_before = dense.lm_head.weight.detach().cpu().clone()
        multihash_input_before = (
            multihash.foldable_residual.effective_input_weight().detach().cpu().clone()
        )
        multihash_output_before = (
            multihash.foldable_residual.effective_output_weight().detach().cpu().clone()
        )
        if not torch.equal(
            dense_input_before, multihash_input_before
        ) or not torch.equal(dense_output_before, multihash_output_before):
            raise RuntimeError("update-audit step-zero effective weights differ")

        dense = dense.to("mps")
        multihash = multihash.to("mps")
        torch.manual_seed(20260831)
        dense_evidence = _backward_and_step(
            dense,
            dense_optimizer(dense),
            batch,
            lexical=lambda model: (
                model.model.embed_tokens.weight,
                model.lm_head.weight,
            ),
            residual_tables=lambda model: None,
        )
        torch.manual_seed(20260831)
        multihash_evidence = _backward_and_step(
            multihash,
            residual_optimizer(multihash),
            batch,
            lexical=lambda model: (
                model.foldable_residual.base_input_weight,
                model.foldable_residual.base_output_weight,
            ),
            residual_tables=lambda model: (
                model.foldable_residual.input_residual,
                model.foldable_residual.output_residual,
            ),
        )
        loss_difference = float(
            np.max(
                np.abs(
                    dense_evidence["micro_losses"].astype(np.float64)
                    - multihash_evidence["micro_losses"].astype(np.float64)
                )
            )
        )
        if loss_difference > 1e-6:
            raise RuntimeError("update-audit step-zero loss differs")

        dense_input_after = dense.model.embed_tokens.weight.detach().cpu()
        dense_output_after = dense.lm_head.weight.detach().cpu()
        multihash_input_after = (
            multihash.foldable_residual.effective_input_weight().detach().cpu()
        )
        multihash_output_after = (
            multihash.foldable_residual.effective_output_weight().detach().cpu()
        )
        new = slice(BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
        dense_input_update = (
            (dense_input_after[new] - dense_input_before[new])
            .float()
            .contiguous()
            .numpy()
        )
        dense_output_update = (
            (dense_output_after[new] - dense_output_before[new])
            .float()
            .contiguous()
            .numpy()
        )
        multihash_input_update = (
            (multihash_input_after[new] - multihash_input_before[new])
            .float()
            .contiguous()
            .numpy()
        )
        multihash_output_update = (
            (multihash_output_after[new] - multihash_output_before[new])
            .float()
            .contiguous()
            .numpy()
        )
        geometry = {
            "input": update_geometry(
                dense_input_update,
                multihash_input_update,
                exposure_counts[new],
            ),
            "output": update_geometry(
                dense_output_update,
                multihash_output_update,
                exposure_counts[new],
            ),
        }
        assignment = (
            multihash.foldable_residual.code_indices.detach().cpu().numpy().copy()
        )
        assignment -= (
            np.arange(RESIDUAL_SLOT_COUNT, dtype=np.int64)[None, :] * CODEBOOK_SIZE
        )
        table_gradients = multihash_evidence["table_gradients"]
        if table_gradients is None:
            raise AssertionError("update-audit table gradients are absent")
        collision_alignment = {
            "input": _bucket_alignment(
                multihash_evidence["input_gradient"][new],
                table_gradients[0],
                assignment[new],
            ),
            "output": _bucket_alignment(
                multihash_evidence["output_gradient"][new],
                table_gradients[1],
                assignment[new],
            ),
        }
        dense.to("cpu")
        multihash.to("cpu")
        del dense, multihash, checkpoint_state
        gc.collect()
        torch.mps.empty_cache()
        return {
            "batch": {
                "first_batch_indices_sha256": array_sha256(selected),
                "token_ids_sha256": batch_sha256,
                "shape": list(batch.shape),
            },
            "initial_equivalence": {
                "effective_input_bitwise_equal": True,
                "effective_output_bitwise_equal": True,
                "maximum_micro_loss_absolute_difference": loss_difference,
            },
            "gradient_clipping": {
                "dense_preclip_total_norm": dense_evidence["preclip_total_norm"],
                "dense_clip_coefficient": dense_evidence["clip_coefficient"],
                "multihash_preclip_total_norm": multihash_evidence[
                    "preclip_total_norm"
                ],
                "multihash_clip_coefficient": multihash_evidence["clip_coefficient"],
            },
            "geometry": geometry,
            "collision_alignment": collision_alignment,
            "assignment": assignment_audit.to_dict(),
            "selected_control": select_update_matched_control(geometry),
        }
    finally:
        _cleanup_data(data)


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("update-audit runner requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("log", "-1", "--format=%H", "--", str(PLAN_PATH.relative_to(ROOT))) != head:
        raise RuntimeError("update-audit plan must be current HEAD")
    plan = read_json(PLAN_PATH)
    _validate_plan(plan)
    if _git("rev-parse", "HEAD^") != plan["git_commit_before_plan"]:
        raise RuntimeError("update-audit plan chronology differs")
    if RESULT_PATH.exists() or _history(RESULT_PATH):
        raise RuntimeError("update-audit result was already published")
    started = time.perf_counter()
    with publication_mps_exclusive():
        evidence = _run_locked(plan)
    elapsed = time.perf_counter() - started
    if _git("rev-parse", "HEAD") != head or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during update audit")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "foldable_multihash_update_audit_summary_v4",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": head,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_payload_sha256": plan["plan_sha256"],
        "parent": plan["parent"],
        "environment": current_environment(),
        "evidence": evidence,
        "elapsed_seconds": elapsed,
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    _publish(RESULT_PATH, _json_bytes(summary))
    print(f"status={summary['kind']}")
    print(f"summary_path={RESULT_PATH.relative_to(ROOT)}")
    print(f"summary_sha256={summary['summary_sha256']}")


if __name__ == "__main__":
    main()
