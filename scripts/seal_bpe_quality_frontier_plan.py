#!/usr/bin/env python3
"""Seal the one-seed BPE quality frontier before training any role."""

from __future__ import annotations

import os
import subprocess

from bpe_quality_feasibility_core import QUALITY_ROLES
from bpe_quality_feasibility_protocol import (
    OUTPUT_PATH as FEASIBILITY_RESULT_PATH,
)
from bpe_quality_feasibility_protocol import PLAN_PATH as FEASIBILITY_PLAN_PATH
from bpe_quality_frontier_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    DOCUMENT_PREFIX,
    QUALITY_MARGIN_BPB,
    array_sha256,
    calibration_document_pieces,
    deterministic_order,
    encode_document_chunks,
    role_training_contract,
)
from bpe_quality_frontier_protocol import (
    IMPLEMENTATION_PATHS,
    INTEGRITY_PATH,
    OUTPUT_PATH,
    PLAN_PATH,
    PROTOCOL_ID,
    ROOT,
    SOURCE_PATH,
    SYSTEMS_RESULT_PATH,
    TOKENIZER_PATHS,
    canonical_sha256,
    current_frontier_environment,
    hash_file,
    json_bytes,
    read_json,
)
from scalar_runtime_core import model_parameter_count
from token_frontier_core import FRONTIER_SPECS, build_frontier_model, parse_role
from token_frontier_protocol import load_tokenizers

from jamoflow.inference_calibration_replay_v2 import state_sha256


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _never_published(path) -> None:
    if path.exists():
        raise FileExistsError(path)
    history = _command(
        "git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT))
    )
    if history:
        raise FileExistsError(f"BPE quality frontier path has Git history: {path}")


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("BPE quality frontier plan requires a clean root")
    _never_published(PLAN_PATH)
    _never_published(OUTPUT_PATH)
    commit = _command("git", "rev-parse", "HEAD")
    feasibility_plan = read_json(FEASIBILITY_PLAN_PATH)
    feasibility_result = read_json(FEASIBILITY_RESULT_PATH)
    if (
        feasibility_result["decision"]["selected_train_raw_bytes"] != 128_000_000
        or feasibility_result["decision"]["quality_or_loss_used"] is not False
    ):
        raise ValueError("BPE quality frontier feasibility did not authorize 128M")
    systems_result = read_json(SYSTEMS_RESULT_PATH)
    tokenizers = load_tokenizers()
    pieces, document_metadata = calibration_document_pieces(SOURCE_PATH)
    document_by_role = {}
    initial_states = {}
    training = {}
    for role in QUALITY_ROLES:
        vocabulary, _ = parse_role(role)
        tokenizer, token_bytes = tokenizers[vocabulary]
        inventory, _, _, _ = encode_document_chunks(pieces, tokenizer, token_bytes)
        document_by_role[role] = inventory.to_dict()
        sequence_count = feasibility_plan["inventories"][role]["train"][
            "full_sequence_count"
        ]
        contract = role_training_contract(role, sequence_count)
        order = deterministic_order(sequence_count)
        contract["training_order_sha256"] = array_sha256(order)
        training[role] = contract
        model = build_frontier_model(role, seed=contract["model_seed"])
        if model_parameter_count(model) != FRONTIER_SPECS[role].expected_parameters:
            raise ValueError("BPE quality frontier initial parameter count differs")
        initial_states[role] = state_sha256(model)
        del model
    dependencies = {
        "git_commit_before_plan": commit,
        "feasibility_plan": {
            "path": str(FEASIBILITY_PLAN_PATH.relative_to(ROOT)),
            "sha256": hash_file(FEASIBILITY_PLAN_PATH),
        },
        "feasibility_result": {
            "path": str(FEASIBILITY_RESULT_PATH.relative_to(ROOT)),
            "sha256": hash_file(FEASIBILITY_RESULT_PATH),
        },
        "integrity": {
            "path": str(INTEGRITY_PATH.relative_to(ROOT)),
            "sha256": hash_file(INTEGRITY_PATH),
        },
        "source": {
            "path": str(SOURCE_PATH.relative_to(ROOT)),
            "sha256": hash_file(SOURCE_PATH),
        },
        "systems_result": {
            "path": str(SYSTEMS_RESULT_PATH.relative_to(ROOT)),
            "sha256": hash_file(SYSTEMS_RESULT_PATH),
        },
        "tokenizers": {
            str(size): {
                "path": str(TOKENIZER_PATHS[size].relative_to(ROOT)),
                "sha256": hash_file(TOKENIZER_PATHS[size]),
            }
            for size, _ in map(parse_role, QUALITY_ROLES)
        },
    }
    payload = {
        "schema_version": 1,
        "kind": "bpe_quality_frontier_one_seed_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_model_training_or_quality",
        "dependencies": dependencies,
        "environment": current_frontier_environment(),
        "roles": list(QUALITY_ROLES),
        "model_specs": {role: FRONTIER_SPECS[role].to_dict() for role in QUALITY_ROLES},
        "initial_state_sha256": initial_states,
        "training": training,
        "document_evaluation": {
            "common": {
                **document_metadata,
                "context_prefix_hex": DOCUMENT_PREFIX.hex(),
            },
            "by_role": document_by_role,
        },
        "systems_end_to_end_ms": {
            role: systems_result["runtime_metrics"][role]["end_to_end_median_ms"]
            for role in QUALITY_ROLES
        },
        "selection_rule": {
            "anchor": "minimum contiguous calibration aggregate BPB",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "comparator": (
                "lowest presealed systems-frontier E2E among quality-qualified roles"
            ),
            "exact_tie_order": list(QUALITY_ROLES),
            "quality_margin_bpb": QUALITY_MARGIN_BPB,
            "qualification": [
                "contiguous aggregate candidate-anchor BPB <= +0.010",
                "document aggregate candidate-anchor BPB <= +0.010",
                "paired document bootstrap 95% upper <= +0.010",
            ],
        },
        "implementation_sha256": {
            relative: hash_file(ROOT / relative) for relative in IMPLEMENTATION_PATHS
        },
        "claim_boundary": {
            "calibration_development_only": True,
            "document_paired_quality_diagnostic": True,
            "matched_quality_multi_seed": False,
            "one_model_seed": True,
            "publication_comparator_selected": False,
            "raw_byte_bpb": True,
            "same_128m_raw_training_stream": True,
        },
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PLAN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    print(f"wrote {PLAN_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
