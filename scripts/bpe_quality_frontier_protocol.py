"""Sealed artifacts and validation for the one-seed BPE quality frontier."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    role_training_contract,
)
from token_frontier_core import FRONTIER_SPECS, parse_role
from token_frontier_protocol import (
    INTEGRITY_PATH,
    ROOT,
    SOURCE_PATH,
    TOKENIZER_PATHS,
    current_frontier_environment,
)
from token_frontier_protocol import OUTPUT_PATH as SYSTEMS_RESULT_PATH

PROTOCOL_ID = "jamoflow-bpe-quality-frontier-one-seed-v1"
PLAN_PATH = ROOT / "data/manifests/bpe-quality-frontier-one-seed-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/bpe-quality-frontier-one-seed-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
WORKER_ROOT = ARTIFACT_ROOT / "workers"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
NLL_ROOT = ARTIFACT_ROOT / "nll"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/bpe-quality-frontier-one-seed-v1/summary.json"

IMPLEMENTATION_PATHS = (
    "docs/116-korean-bpe-systems-frontier-result.md",
    "docs/118-bpe-quality-frontier-feasibility-result.md",
    "docs/119-bpe-quality-frontier-one-seed-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/bpe_quality_frontier_protocol.py",
    "scripts/run_bpe_quality_frontier.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_bpe_quality_frontier_plan.py",
    "scripts/summarize_bpe_quality_frontier.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/inference_actual_runtime_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_bpe_quality_frontier.py",
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("BPE quality frontier JSON root differs")
    return value


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "claim_boundary",
        "dependencies",
        "document_evaluation",
        "environment",
        "implementation_sha256",
        "initial_state_sha256",
        "kind",
        "model_specs",
        "plan_sha256",
        "protocol_id",
        "roles",
        "schema_version",
        "selection_rule",
        "status",
        "systems_end_to_end_ms",
        "training",
    }
    if set(plan) != expected:
        raise ValueError("BPE quality frontier plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "bpe_quality_frontier_one_seed_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_model_training_or_quality"
    ):
        raise ValueError("BPE quality frontier plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("BPE quality frontier plan hash differs")
    if plan["environment"] != current_frontier_environment():
        raise ValueError("BPE quality frontier environment differs")
    if plan["roles"] != list(QUALITY_ROLES):
        raise ValueError("BPE quality frontier role order differs")
    if plan["model_specs"] != {
        role: FRONTIER_SPECS[role].to_dict() for role in QUALITY_ROLES
    }:
        raise ValueError("BPE quality frontier model specs differ")
    systems_result = read_json(SYSTEMS_RESULT_PATH)
    expected_systems_e2e = {
        role: systems_result["runtime_metrics"][role]["end_to_end_median_ms"]
        for role in QUALITY_ROLES
    }
    if plan["systems_end_to_end_ms"] != expected_systems_e2e:
        raise ValueError("BPE quality frontier presealed systems timing differs")
    feasibility_plan = read_json(FEASIBILITY_PLAN_PATH)
    if set(plan["training"]) != set(QUALITY_ROLES):
        raise ValueError("BPE quality frontier training role set differs")
    for role in QUALITY_ROLES:
        sequence_count = feasibility_plan["inventories"][role]["train"][
            "full_sequence_count"
        ]
        expected_training = role_training_contract(role, sequence_count)
        expected_training["training_order_sha256"] = array_sha256(
            deterministic_order(sequence_count)
        )
        if plan["training"][role] != expected_training:
            raise ValueError("BPE quality frontier training contract differs")
        state = plan["initial_state_sha256"].get(role)
        if not isinstance(state, str) or len(state) != 64:
            raise ValueError("BPE quality frontier initial state identity differs")
    pieces, document_metadata = calibration_document_pieces(SOURCE_PATH)
    if plan["document_evaluation"].get("common") != {
        **document_metadata,
        "context_prefix_hex": DOCUMENT_PREFIX.hex(),
    }:
        raise ValueError("BPE quality frontier document set differs")
    if set(plan["document_evaluation"].get("by_role", {})) != set(QUALITY_ROLES):
        raise ValueError("BPE quality frontier document inventory roles differ")
    document_keys = {
        "chunk_count",
        "chunk_schedule_sha256",
        "document_count",
        "document_lengths_sha256",
        "maximum_document_tokens_including_prefix",
        "raw_bytes",
        "token_count_excluding_prefix",
    }
    for row in plan["document_evaluation"]["by_role"].values():
        if set(row) != document_keys:
            raise ValueError("BPE quality frontier document inventory schema differs")
        for key in document_keys - {
            "chunk_schedule_sha256",
            "document_lengths_sha256",
        }:
            if not isinstance(row[key], int) or row[key] <= 0:
                raise ValueError(
                    "BPE quality frontier document inventory count differs"
                )
        for key in ("chunk_schedule_sha256", "document_lengths_sha256"):
            if not isinstance(row[key], str) or len(row[key]) != 64:
                raise ValueError("BPE quality frontier document inventory hash differs")
    if not pieces:
        raise AssertionError("BPE quality frontier document reconstruction failed")
    if plan["selection_rule"] != {
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
    }:
        raise ValueError("BPE quality frontier selection rule differs")
    dependencies = plan["dependencies"]
    expected_paths = {
        "feasibility_plan": FEASIBILITY_PLAN_PATH,
        "feasibility_result": FEASIBILITY_RESULT_PATH,
        "integrity": INTEGRITY_PATH,
        "source": SOURCE_PATH,
        "systems_result": SYSTEMS_RESULT_PATH,
    }
    if set(dependencies) != {
        "feasibility_plan",
        "feasibility_result",
        "git_commit_before_plan",
        "integrity",
        "source",
        "systems_result",
        "tokenizers",
    }:
        raise ValueError("BPE quality frontier dependencies differ")
    if (
        not isinstance(dependencies["git_commit_before_plan"], str)
        or len(dependencies["git_commit_before_plan"]) != 40
    ):
        raise ValueError("BPE quality frontier base commit differs")
    for key, path in expected_paths.items():
        if dependencies[key] != {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        }:
            raise ValueError(f"BPE quality frontier dependency changed: {key}")
    if set(dependencies["tokenizers"]) != {
        str(parse_role(role)[0]) for role in QUALITY_ROLES
    }:
        raise ValueError("BPE quality frontier tokenizer set differs")
    for key, row in dependencies["tokenizers"].items():
        path = TOKENIZER_PATHS[int(key)]
        if row != {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}:
            raise ValueError("BPE quality frontier tokenizer dependency changed")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("BPE quality frontier implementation set differs")
    for relative in IMPLEMENTATION_PATHS:
        if hash_file(ROOT / relative) != plan["implementation_sha256"][relative]:
            raise ValueError(f"BPE quality frontier implementation changed: {relative}")
    if plan["claim_boundary"] != {
        "calibration_development_only": True,
        "document_paired_quality_diagnostic": True,
        "matched_quality_multi_seed": False,
        "one_model_seed": True,
        "publication_comparator_selected": False,
        "raw_byte_bpb": True,
        "same_128m_raw_training_stream": True,
    }:
        raise ValueError("BPE quality frontier claim boundary differs")
