#!/usr/bin/env python3
"""Independently replay and summarize the one-seed compositional quality grid."""

from __future__ import annotations

import gc
import math
import os
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from bpe_quality_feasibility_core import encode_stream_to_memmap
from bpe_quality_frontier_core import (
    array_sha256,
    bpb,
    calibration_document_pieces,
    document_bootstrap_upper,
    encode_document_chunks,
    raw_target_bytes_by_sequence,
)
from compositional_quality_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CALIBRATION_BYTES,
    EVALUATION_BATCH,
    MODEL_SEED,
    QUALITY_ROLES,
    QUALITY_SPECS,
    SEQUENCE_LENGTH,
    TRAIN_BYTES,
    build_quality_model,
    quality_decision,
    state_subset_sha256,
)
from compositional_quality_protocol import (
    ACTIVE_PATH,
    CHECKPOINT_ROOT,
    NLL_ROOT,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    REPORT_PATH,
    RESOURCE_REPORT_PATH,
    ROOT,
    SOURCE_PATH,
    WORKER_ROOT,
    canonical_sha256,
    hash_file,
    json_bytes,
    load_tokenizers,
    read_json,
    validate_plan,
)
from jamoflow.actual_inference_protocol import timing_environment_eligible
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.neural_data import build_neural_stream
from run_compositional_quality import _evaluate_contiguous, _evaluate_documents
from scalar_runtime_core import model_parameter_count


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _never_published(path) -> None:
    history = _git("log", "--all", "--format=%H", "--", str(path.relative_to(ROOT)))
    if path.exists() or history:
        raise RuntimeError(f"compositional quality result already exists or has history: {path}")


def _load_role(
    role: str,
    descriptor: Mapping[str, Any],
    plan: Mapping[str, Any],
    commit: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_report_path = WORKER_ROOT / f"{role}.json"
    if set(descriptor) != {"path", "sha256"}:
        raise RuntimeError("compositional quality worker descriptor differs")
    report_path = ROOT / descriptor["path"]
    if (
        report_path != expected_report_path
        or descriptor["sha256"] != hash_file(report_path)
    ):
        raise RuntimeError("compositional quality worker artifact differs")
    report = read_json(report_path)
    unsigned = dict(report)
    receipt = unsigned.pop("worker_sha256", None)
    checkpoint_path = CHECKPOINT_ROOT / f"{role}.pt"
    nll_path = NLL_ROOT / f"{role}.npz"
    if (
        canonical_sha256(unsigned) != receipt
        or report.get("schema_version") != 1
        or report.get("kind") != "compositional_quality_worker_v1"
        or report.get("protocol_id") != PROTOCOL_ID
        or report.get("complete") is not True
        or report.get("git_commit") != commit
        or report.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or report.get("resource_report_artifact_sha256")
        != hash_file(RESOURCE_REPORT_PATH)
        or report.get("role") != role
        or report.get("parameter_count") != QUALITY_SPECS[role].expected_parameters
        or report.get("training_contract") != plan["training"][role]
        or report.get("initial_state_sha256") != plan["initial_state_sha256"][role]
        or report.get("checkpoint_path") != str(checkpoint_path.relative_to(ROOT))
        or report.get("checkpoint_artifact_sha256") != hash_file(checkpoint_path)
        or report.get("nll_path") != str(nll_path.relative_to(ROOT))
        or report.get("nll_artifact_sha256") != hash_file(nll_path)
        or report.get("environment") != plan["environment"]
    ):
        raise RuntimeError(f"compositional quality worker identity differs: {role}")
    training = report.get("training")
    expected_steps = plan["training"][role]["total_optimizer_steps"]
    if (
        not isinstance(training, Mapping)
        or training.get("completed") is not True
        or training.get("finite_optimizer_steps") != expected_steps
        or training.get("total_optimizer_steps") != expected_steps
        or training.get("sequence_examples") != plan["training"][role]["sequence_count"]
        or not math.isfinite(float(training.get("elapsed_seconds", math.nan)))
        or float(training["elapsed_seconds"]) <= 0
    ):
        raise RuntimeError("compositional quality training completion differs")
    if not all(
        timing_environment_eligible(report["session_state"][key])
        for key in ("start", "end")
    ):
        raise RuntimeError("compositional quality training environment differs")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = QUALITY_SPECS[role]
    table = load_tokenizers()[spec.vocabulary_size][1]
    model = build_quality_model(
        role,
        token_bytes=table if "code" in spec.head_kind else None,
        seed=MODEL_SEED,
    )
    model.load_state_dict(state, strict=True)
    if (
        model_parameter_count(model) != spec.expected_parameters
        or state_subset_sha256(model, transformer_body_only=False)
        != report["trained_state_sha256"]
    ):
        raise RuntimeError("compositional quality checkpoint state differs")
    del state, model
    expected_names = {
        "contiguous_nll_nats",
        "contiguous_raw_target_bytes",
        "document_nll_nats",
        "document_raw_bytes",
    }
    with np.load(nll_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names or set(report.get("arrays", {})) != expected_names:
            raise RuntimeError("compositional quality NLL key set differs")
        arrays = {name: archive[name] for name in archive.files}
    spec_inventory = plan["inventories"][str(spec.vocabulary_size)]
    expected_shapes = {
        "contiguous_nll_nats": (spec_inventory["calibration"]["full_sequence_count"],),
        "contiguous_raw_target_bytes": (
            spec_inventory["calibration"]["full_sequence_count"],
        ),
        "document_nll_nats": (plan["document_common"]["document_count"],),
        "document_raw_bytes": (plan["document_common"]["document_count"],),
    }
    for name, values in arrays.items():
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
            or report["arrays"][name]
            != {
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "sha256": array_sha256(values),
            }
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise RuntimeError(f"compositional quality NLL array differs: {role}/{name}")
    if (
        int(arrays["contiguous_raw_target_bytes"].sum())
        != spec_inventory["calibration"]["predicted_target_raw_bytes"]
        or int(arrays["document_raw_bytes"].sum()) != plan["document_common"]["raw_bytes"]
    ):
        raise RuntimeError("compositional quality raw-byte denominator differs")
    contiguous = bpb(arrays["contiguous_nll_nats"], arrays["contiguous_raw_target_bytes"])
    document = bpb(arrays["document_nll_nats"], arrays["document_raw_bytes"])
    if (
        not math.isclose(report["metrics"]["contiguous_bpb"], contiguous, abs_tol=1e-12)
        or not math.isclose(report["metrics"]["document_bpb"], document, abs_tol=1e-12)
    ):
        raise RuntimeError("compositional quality stored BPB differs")
    return report, arrays


def _replay_inputs(plan: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    pieces, common = calibration_document_pieces(SOURCE_PATH)
    if common != {key: plan["document_common"][key] for key in common}:
        raise RuntimeError("compositional quality replay document set differs")
    tokenizers = load_tokenizers()
    output = {}
    for vocabulary_size in (2_048, 8_192):
        role = "dense_v2048" if vocabulary_size == 2_048 else "dense_v8192"
        tokenizer, token_bytes = tokenizers[vocabulary_size]
        evaluation_batch = EVALUATION_BATCH[role]
        inventory, memory, memory_path = encode_stream_to_memmap(
            stream.data,
            tokenizer,
            token_bytes,
            first_batch_token_count=evaluation_batch * SEQUENCE_LENGTH,
        )
        if inventory.to_dict() != plan["inventories"][str(vocabulary_size)]["calibration"]:
            raise RuntimeError("compositional quality replay inventory differs")
        sequences = memory[: inventory.full_sequence_count * SEQUENCE_LENGTH].reshape(
            inventory.full_sequence_count, SEQUENCE_LENGTH
        )
        raw_bytes = raw_target_bytes_by_sequence(sequences, token_bytes)
        document_inventory, chunks, chunk_documents, document_raw_bytes = encode_document_chunks(
            pieces, tokenizer, token_bytes
        )
        if document_inventory.to_dict() != plan["inventories"][str(vocabulary_size)][
            "documents"
        ]:
            raise RuntimeError("compositional quality replay document inventory differs")
        output[vocabulary_size] = {
            "token_bytes": token_bytes,
            "memory": memory,
            "memory_path": memory_path,
            "sequences": sequences,
            "raw_target_bytes": raw_bytes,
            "chunks": chunks,
            "chunk_documents": chunk_documents,
            "document_raw_bytes": document_raw_bytes,
        }
    return output


def _independent_replay(
    loaded: Mapping[str, tuple[dict[str, Any], dict[str, np.ndarray]]],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    inputs = _replay_inputs(plan)
    replayed_by_role = {}
    hashes = {}
    try:
        with publication_mps_exclusive():
            for role in QUALITY_ROLES:
                spec = QUALITY_SPECS[role]
                context = inputs[spec.vocabulary_size]
                checkpoint = torch.load(
                    CHECKPOINT_ROOT / f"{role}.pt", map_location="cpu", weights_only=True
                )
                model = build_quality_model(
                    role,
                    token_bytes=context["token_bytes"] if "code" in spec.head_kind else None,
                    seed=MODEL_SEED,
                )
                model.load_state_dict(checkpoint, strict=True)
                if state_subset_sha256(model, transformer_body_only=False) != loaded[role][0][
                    "trained_state_sha256"
                ]:
                    raise RuntimeError("compositional quality replay checkpoint differs")
                model.to("mps").eval()
                replayed = {
                    "contiguous_nll_nats": _evaluate_contiguous(
                        model, context["sequences"], EVALUATION_BATCH[role]
                    ),
                    "contiguous_raw_target_bytes": context["raw_target_bytes"],
                    "document_nll_nats": _evaluate_documents(
                        model,
                        context["chunks"],
                        context["chunk_documents"],
                        len(context["document_raw_bytes"]),
                        EVALUATION_BATCH[role],
                    ),
                    "document_raw_bytes": context["document_raw_bytes"],
                }
                for name, values in replayed.items():
                    if not np.array_equal(values, loaded[role][1][name]):
                        raise RuntimeError(
                            f"compositional quality independent replay differs: {role}/{name}"
                        )
                replayed_by_role[role] = replayed
                hashes[role] = {name: array_sha256(value) for name, value in replayed.items()}
                model.to("cpu")
                del model, checkpoint
                gc.collect()
                torch.mps.empty_cache()
    finally:
        for context in inputs.values():
            del context["sequences"], context["memory"]
            if os.path.exists(context["memory_path"]):
                os.unlink(context["memory_path"])
    return (
        {
            "pass": True,
            "role_count": len(hashes),
            "array_comparison": "bitwise_equal",
            "array_sha256_by_role": hashes,
        },
        replayed_by_role,
    )


def _contrast(
    candidate: str,
    reference: str,
    metrics: Mapping[str, Mapping[str, float]],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed_offset: int,
) -> dict[str, float]:
    raw_bytes = arrays[candidate]["document_raw_bytes"]
    if not np.array_equal(raw_bytes, arrays[reference]["document_raw_bytes"]):
        raise RuntimeError("compositional quality paired document bytes differ")
    point, _lower, upper = document_bootstrap_upper(
        arrays[candidate]["document_nll_nats"],
        arrays[reference]["document_nll_nats"],
        raw_bytes,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    return {
        "contiguous_bpb_difference": (
            metrics[candidate]["contiguous_bpb"] - metrics[reference]["contiguous_bpb"]
        ),
        "document_bpb_difference": point,
        "bootstrap_95_upper": upper,
    }


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("compositional quality summary requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    _never_published(OUTPUT_PATH)
    if ACTIVE_PATH.exists():
        raise RuntimeError("compositional quality campaign remains active")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    resource = read_json(RESOURCE_REPORT_PATH)
    resource_unsigned = dict(resource)
    resource_hash = resource_unsigned.pop("report_sha256", None)
    if (
        canonical_sha256(resource_unsigned) != resource_hash
        or resource.get("complete") is not True
        or resource.get("git_commit") != commit
        or resource.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or resource.get("projection", {}).get("passes") is not True
    ):
        raise RuntimeError("compositional quality resource authorization differs")
    campaign = read_json(REPORT_PATH)
    campaign_unsigned = dict(campaign)
    campaign_hash = campaign_unsigned.pop("report_sha256", None)
    if (
        canonical_sha256(campaign_unsigned) != campaign_hash
        or campaign.get("schema_version") != 1
        or campaign.get("kind") != "compositional_quality_report_v1"
        or campaign.get("protocol_id") != PROTOCOL_ID
        or campaign.get("complete") is not True
        or campaign.get("git_commit") != commit
        or campaign.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or campaign.get("resource_report_artifact_sha256")
        != hash_file(RESOURCE_REPORT_PATH)
        or set(campaign.get("workers", {})) != set(QUALITY_ROLES)
    ):
        raise RuntimeError("compositional quality campaign report differs")
    loaded = {
        role: _load_role(role, campaign["workers"][role], plan, commit)
        for role in QUALITY_ROLES
    }
    reference_bytes = loaded[QUALITY_ROLES[0]][1]["document_raw_bytes"]
    if any(
        not np.array_equal(loaded[role][1]["document_raw_bytes"], reference_bytes)
        for role in QUALITY_ROLES[1:]
    ):
        raise RuntimeError("compositional quality document denominators differ")
    replay, replayed = _independent_replay(loaded, plan)
    metrics = {
        role: {
            "contiguous_bpb": bpb(
                arrays["contiguous_nll_nats"], arrays["contiguous_raw_target_bytes"]
            ),
            "document_bpb": bpb(
                arrays["document_nll_nats"], arrays["document_raw_bytes"]
            ),
            "parameters": QUALITY_SPECS[role].expected_parameters,
            "vocabulary_size": QUALITY_SPECS[role].vocabulary_size,
            "head_kind": QUALITY_SPECS[role].head_kind,
            "optimizer_steps": loaded[role][0]["training"]["total_optimizer_steps"],
            "training_elapsed_seconds": loaded[role][0]["training"]["elapsed_seconds"],
            "raw_train_bytes": TRAIN_BYTES,
        }
        for role, arrays in replayed.items()
    }
    contrast_specs = {
        "hangul_vs_dense_2k": ("hangul_code_v8192", "dense_v2048"),
        "hangul_vs_generic": ("hangul_code_v8192", "generic_code_v8192"),
        "hangul_vs_shuffled": (
            "hangul_code_v8192",
            "shuffled_hangul_code_v8192",
        ),
        "hangul_vs_low_rank": ("hangul_code_v8192", "low_rank_v8192"),
        "generic_vs_dense_2k": ("generic_code_v8192", "dense_v2048"),
    }
    contrasts = {
        name: _contrast(candidate, reference, metrics, replayed, seed_offset=index)
        for index, (name, (candidate, reference)) in enumerate(contrast_specs.items())
    }
    decision = quality_decision(contrasts)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "compositional_head_quality_one_seed_result_v1",
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "resource_report_artifact_sha256": hash_file(RESOURCE_REPORT_PATH),
        "campaign_report_artifact_sha256": hash_file(REPORT_PATH),
        "metrics": metrics,
        "quality_contrasts": contrasts,
        "decision": decision,
        "independent_nll_recomputation": replay,
        "artifact_lineage": {
            role: {
                "worker_report_sha256": campaign["workers"][role]["sha256"],
                "checkpoint_sha256": loaded[role][0]["checkpoint_artifact_sha256"],
                "checkpoint_state_sha256": loaded[role][0]["trained_state_sha256"],
                "nll_sha256": loaded[role][0]["nll_artifact_sha256"],
            }
            for role in QUALITY_ROLES
        },
        "resource_projection": resource["projection"],
        "claim_boundary": plan["claim_boundary"],
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    if _git("rev-parse", "HEAD") != commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during compositional quality summary")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(OUTPUT_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(summary))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote={OUTPUT_PATH.relative_to(ROOT)}")
    print(f"status={decision['status']}")


if __name__ == "__main__":
    main()
