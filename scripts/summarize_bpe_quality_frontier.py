#!/usr/bin/env python3
"""Validate, compare, and summarize the one-seed BPE quality frontier."""

from __future__ import annotations

import gc
import math
import os
import subprocess
from typing import Any

import numpy as np
import torch
from bpe_quality_feasibility_core import (
    CALIBRATION_BYTES,
    EVALUATION_BATCH_BY_VOCABULARY,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
    encode_stream_to_memmap,
)
from bpe_quality_feasibility_protocol import PLAN_PATH as FEASIBILITY_PLAN_PATH
from bpe_quality_frontier_core import (
    array_sha256,
    bpb,
    calibration_document_pieces,
    encode_document_chunks,
    raw_target_bytes_by_sequence,
    select_quality_frontier,
)
from bpe_quality_frontier_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_ROOT,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    ROOT,
    SOURCE_PATH,
    WORKER_ROOT,
    canonical_sha256,
    hash_file,
    json_bytes,
    read_json,
    validate_plan,
)
from run_bpe_quality_frontier import _evaluate_contiguous, _evaluate_documents
from token_frontier_core import FRONTIER_SPECS, build_frontier_model, parse_role
from token_frontier_protocol import load_tokenizers

from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import (
    publication_mps_exclusive,
    state_sha256,
)
from jamoflow.neural_data import build_neural_stream


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command(
        "git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT))
    )
    if history:
        raise FileExistsError(f"BPE quality frontier result has Git history: {path}")


def _load_role(
    role: str,
    descriptor: dict[str, Any],
    plan: dict[str, Any],
    commit: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report_path = ROOT / descriptor["path"]
    expected_report_path = WORKER_ROOT / f"{role}.json"
    if (
        report_path != expected_report_path
        or set(descriptor) != {"path", "sha256"}
        or descriptor["sha256"] != hash_file(report_path)
    ):
        raise ValueError("BPE quality frontier worker descriptor differs")
    report = read_json(report_path)
    unsigned = dict(report)
    expected_hash = unsigned.pop("worker_sha256")
    checkpoint_path = CHECKPOINT_ROOT / f"{role}.pt"
    nll_path = NLL_ROOT / f"{role}.npz"
    if (
        canonical_sha256(unsigned) != expected_hash
        or report.get("schema_version") != 1
        or report.get("kind") != "bpe_quality_frontier_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("role") != role
        or report.get("parameter_count") != FRONTIER_SPECS[role].expected_parameters
        or report.get("training_contract") != plan["training"][role]
        or report.get("initial_state_sha256") != plan["initial_state_sha256"][role]
        or report.get("checkpoint_path") != str(checkpoint_path.relative_to(ROOT))
        or report.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
        or report.get("nll_path") != str(nll_path.relative_to(ROOT))
        or report.get("nll_artifact_sha256") != hash_file(nll_path)
        or report.get("environment") != plan["environment"]
    ):
        raise ValueError(f"BPE quality frontier worker identity differs: {role}")
    training = report.get("training")
    if (
        not isinstance(training, dict)
        or training.get("completed") is not True
        or training.get("finite_optimizer_steps")
        != plan["training"][role]["total_optimizer_steps"]
        or training.get("total_optimizer_steps")
        != plan["training"][role]["total_optimizer_steps"]
        or training.get("sequence_examples") != plan["training"][role]["sequence_count"]
        or not math.isfinite(training.get("elapsed_seconds", math.nan))
        or training["elapsed_seconds"] <= 0
    ):
        raise ValueError("BPE quality frontier training completion differs")
    if not all(
        timing_environment_eligible(report["session_state"][key])
        for key in ("start", "end")
    ):
        raise ValueError("BPE quality frontier training environment is ineligible")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_frontier_model(role, seed=plan["training"][role]["model_seed"])
    model.load_state_dict(state, strict=True)
    if state_sha256(model) != report["trained_state_sha256"]:
        raise ValueError("BPE quality frontier checkpoint state differs")
    del model, state
    expected_keys = {
        "contiguous_nll_nats",
        "contiguous_raw_target_bytes",
        "document_nll_nats",
        "document_raw_bytes",
    }
    with np.load(nll_path, allow_pickle=False) as archive:
        if (
            set(archive.files) != expected_keys
            or set(report.get("arrays", {})) != expected_keys
        ):
            raise ValueError("BPE quality frontier NLL key set differs")
        arrays = {name: archive[name] for name in archive.files}
    feasibility = read_json(FEASIBILITY_PLAN_PATH)["inventories"][role]["calibration"]
    document_count = plan["document_evaluation"]["common"]["document_count"]
    expected_shapes = {
        "contiguous_nll_nats": (feasibility["full_sequence_count"],),
        "contiguous_raw_target_bytes": (feasibility["full_sequence_count"],),
        "document_nll_nats": (document_count,),
        "document_raw_bytes": (document_count,),
    }
    for name, values in arrays.items():
        descriptor_row = report["arrays"][name]
        expected_dtype = (
            "float32"
            if name == "contiguous_nll_nats"
            else "float64"
            if name == "document_nll_nats"
            else "int64"
        )
        if (
            str(values.dtype) != expected_dtype
            or values.shape != expected_shapes[name]
            or descriptor_row
            != {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(f"BPE quality frontier NLL array differs: {role}/{name}")
    if (
        int(arrays["contiguous_raw_target_bytes"].sum())
        != feasibility["predicted_target_raw_bytes"]
        or int(arrays["document_raw_bytes"].sum())
        != plan["document_evaluation"]["common"]["raw_bytes"]
    ):
        raise ValueError("BPE quality frontier raw-byte denominator differs")
    contiguous = bpb(
        arrays["contiguous_nll_nats"], arrays["contiguous_raw_target_bytes"]
    )
    document = bpb(arrays["document_nll_nats"], arrays["document_raw_bytes"])
    if not math.isclose(
        report["metrics"]["contiguous_bpb"], contiguous, abs_tol=1e-12
    ) or not math.isclose(report["metrics"]["document_bpb"], document, abs_tol=1e-12):
        raise ValueError("BPE quality frontier stored BPB differs")
    return report, arrays


def _independent_nll_replay(
    loaded: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    feasibility_plan = read_json(FEASIBILITY_PLAN_PATH)
    calibration_stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    pieces, document_common = calibration_document_pieces(SOURCE_PATH)
    if document_common != {
        key: plan["document_evaluation"]["common"][key] for key in document_common
    }:
        raise ValueError("BPE quality replay document set differs")
    tokenizers = load_tokenizers()
    replay_hashes = {}
    with publication_mps_exclusive():
        for role in QUALITY_ROLES:
            vocabulary, _ = parse_role(role)
            tokenizer, token_bytes = tokenizers[vocabulary]
            evaluation_batch = EVALUATION_BATCH_BY_VOCABULARY[vocabulary]
            memory = None
            memory_path = None
            model = None
            try:
                inventory, memory, memory_path = encode_stream_to_memmap(
                    calibration_stream.data,
                    tokenizer,
                    token_bytes,
                    first_batch_token_count=evaluation_batch * SEQUENCE_LENGTH,
                )
                if (
                    inventory.to_dict()
                    != feasibility_plan["inventories"][role]["calibration"]
                ):
                    raise ValueError("BPE quality replay token inventory differs")
                sequences = memory[
                    : inventory.full_sequence_count * SEQUENCE_LENGTH
                ].reshape(inventory.full_sequence_count, SEQUENCE_LENGTH)
                raw_target_bytes = raw_target_bytes_by_sequence(sequences, token_bytes)
                document_inventory, chunks, chunk_documents, document_raw_bytes = (
                    encode_document_chunks(pieces, tokenizer, token_bytes)
                )
                if (
                    document_inventory.to_dict()
                    != plan["document_evaluation"]["by_role"][role]
                ):
                    raise ValueError("BPE quality replay document inventory differs")
                checkpoint = torch.load(
                    CHECKPOINT_ROOT / f"{role}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model = build_frontier_model(
                    role,
                    seed=plan["training"][role]["model_seed"],
                )
                model.load_state_dict(checkpoint, strict=True)
                if state_sha256(model) != loaded[role][0]["trained_state_sha256"]:
                    raise ValueError("BPE quality replay checkpoint state differs")
                model.to("mps").eval()
                replayed = {
                    "contiguous_nll_nats": _evaluate_contiguous(
                        model,
                        sequences,
                        evaluation_batch,
                    ),
                    "contiguous_raw_target_bytes": raw_target_bytes,
                    "document_nll_nats": _evaluate_documents(
                        model,
                        chunks,
                        chunk_documents,
                        len(document_raw_bytes),
                        evaluation_batch,
                    ),
                    "document_raw_bytes": document_raw_bytes,
                }
                for name, values in replayed.items():
                    if not np.array_equal(values, loaded[role][1][name]):
                        raise ValueError(
                            f"BPE quality independent replay differs: {role}/{name}"
                        )
                replay_hashes[role] = {
                    name: array_sha256(values) for name, values in replayed.items()
                }
                del checkpoint
            finally:
                if model is not None:
                    model.to("cpu")
                    del model
                if memory is not None:
                    del memory
                if memory_path is not None and os.path.exists(memory_path):
                    os.unlink(memory_path)
                gc.collect()
                torch.mps.empty_cache()
    return {
        "array_comparison": "bitwise_equal",
        "array_sha256_by_role": replay_hashes,
        "pass": True,
        "role_count": len(replay_hashes),
    }


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("BPE quality frontier summary requires a clean root")
    commit = _command("git", "rev-parse", "HEAD")
    _never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists():
        raise ValueError("BPE quality frontier remains active")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    campaign = read_json(REPORT_PATH)
    unsigned = dict(campaign)
    expected_hash = unsigned.pop("report_sha256", None)
    if (
        canonical_sha256(unsigned) != expected_hash
        or campaign.get("schema_version") != 1
        or campaign.get("kind") != "bpe_quality_frontier_report_v1"
        or campaign.get("protocol_id") != PROTOCOL_ID
        or campaign.get("complete") is not True
        or campaign.get("git_commit") != commit
        or campaign.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or set(campaign.get("workers", {})) != set(QUALITY_ROLES)
    ):
        raise ValueError("BPE quality frontier campaign report differs")
    loaded = {
        role: _load_role(role, campaign["workers"][role], plan, commit)
        for role in QUALITY_ROLES
    }
    common_document_bytes = loaded[QUALITY_ROLES[0]][1]["document_raw_bytes"]
    for role in QUALITY_ROLES[1:]:
        if not np.array_equal(
            loaded[role][1]["document_raw_bytes"], common_document_bytes
        ):
            raise ValueError("BPE quality document denominators differ across roles")
    replay = _independent_nll_replay(loaded, plan)
    metrics = {
        role: {
            "contiguous_bpb": bpb(
                arrays["contiguous_nll_nats"],
                arrays["contiguous_raw_target_bytes"],
            ),
            "document_bpb": bpb(
                arrays["document_nll_nats"], arrays["document_raw_bytes"]
            ),
            "training_elapsed_seconds": report["training"]["elapsed_seconds"],
            "optimizer_steps": report["training"]["total_optimizer_steps"],
            "raw_train_bytes": 128_000_000,
            "parameters": report["parameter_count"],
            "vocabulary_size": parse_role(role)[0],
            "hidden_size": FRONTIER_SPECS[role].hidden_size,
            "layers": FRONTIER_SPECS[role].layers,
            "presealed_systems_end_to_end_ms": plan["systems_end_to_end_ms"][role],
        }
        for role, (report, arrays) in loaded.items()
    }
    decision = select_quality_frontier(
        {role: metrics[role]["contiguous_bpb"] for role in QUALITY_ROLES},
        {role: loaded[role][1]["document_nll_nats"] for role in QUALITY_ROLES},
        common_document_bytes,
        plan["systems_end_to_end_ms"],
    )
    total_training_seconds = math.fsum(
        float(metrics[role]["training_elapsed_seconds"]) for role in QUALITY_ROLES
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_one_seed_result_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "campaign_report_artifact_sha256": hash_file(REPORT_PATH),
        "metrics": metrics,
        "quality_comparisons": decision["comparisons"],
        "independent_nll_recomputation": replay,
        "decision": {
            "calibration_quality_anchor": decision["calibration_quality_anchor"],
            "quality_qualified_roles": decision["quality_qualified_roles"],
            "development_bpe_comparator": decision["development_bpe_comparator"],
            "status": "development_bpe_comparator_selected",
            "total_training_elapsed_hours": total_training_seconds / 3600,
            "publication_comparator_selected": False,
            "next_stage": (
                "compare exact BPE, generic long-token, and Korean-aware tokenizers "
                "against this development systems-quality frontier"
            ),
        },
        "artifact_lineage": {
            role: {
                "worker_report_sha256": campaign["workers"][role]["sha256"],
                "checkpoint_sha256": loaded[role][0]["checkpoint_artifact_sha256"],
                "checkpoint_state_sha256": loaded[role][0]["trained_state_sha256"],
                "nll_sha256": loaded[role][0]["nll_artifact_sha256"],
            }
            for role in QUALITY_ROLES
        },
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError("repository changed during BPE quality frontier summary")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(summary))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
