"""Sealed paths and validation for BPE quality-frontier feasibility v1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bpe_quality_feasibility_core import (
    CALIBRATION_BYTES,
    CAMPAIGN_HOUR_LIMIT,
    CANDIDATE_TRAIN_BYTE_BUDGETS,
    DRIVER_MEMORY_FRACTION_LIMIT,
    EFFECTIVE_BATCH_SIZE,
    EVALUATION_BATCH_BY_VOCABULARY,
    MEASURED_EFFECTIVE_STEPS,
    MEASURED_EVALUATION_BATCHES,
    QUALITY_ROLES,
    SEQUENCE_LENGTH,
    TRAIN_BYTES,
    TRAIN_MICROBATCH_BY_VOCABULARY,
    WARMUP_EFFECTIVE_STEPS,
    WARMUP_EVALUATION_BATCHES,
    validate_inventory,
)
from token_frontier_core import FRONTIER_SPECS, parse_role
from token_frontier_protocol import (
    INTEGRITY_PATH,
    ROOT,
    SOURCE_PATH,
    TOKENIZER_PATHS,
    current_frontier_environment,
)
from token_frontier_protocol import (
    OUTPUT_PATH as SYSTEMS_RESULT_PATH,
)

PROTOCOL_ID = "jamoflow-bpe-quality-frontier-feasibility-v1"
PLAN_PATH = ROOT / "data/manifests/bpe-quality-frontier-feasibility-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/bpe-quality-frontier-feasibility-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
REPORT_PATH = ARTIFACT_ROOT / "report.json"
OUTPUT_PATH = ROOT / "results/bpe-quality-frontier-feasibility-v1/summary.json"

IMPLEMENTATION_PATHS = (
    "docs/116-korean-bpe-systems-frontier-result.md",
    "docs/117-bpe-quality-frontier-feasibility-protocol.md",
    "pyproject.toml",
    "scripts/benchmark_bpe_quality_feasibility.py",
    "scripts/bpe_quality_feasibility_core.py",
    "scripts/bpe_quality_feasibility_protocol.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_bpe_quality_feasibility_plan.py",
    "scripts/summarize_bpe_quality_feasibility.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_bpe_quality_feasibility.py",
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
        raise ValueError("BPE quality feasibility JSON root differs")
    return value


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "claim_boundary",
        "dependencies",
        "environment",
        "feasibility",
        "implementation_sha256",
        "inventories",
        "kind",
        "model_specs",
        "plan_sha256",
        "protocol_id",
        "roles",
        "schema_version",
        "status",
    }
    if set(plan) != expected:
        raise ValueError("BPE quality feasibility plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "bpe_quality_frontier_feasibility_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_training_step_timing"
    ):
        raise ValueError("BPE quality feasibility plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("BPE quality feasibility plan hash differs")
    if plan["environment"] != current_frontier_environment():
        raise ValueError("BPE quality feasibility environment differs")
    if plan["roles"] != list(QUALITY_ROLES):
        raise ValueError("BPE quality feasibility roles differ")
    if plan["model_specs"] != {
        role: FRONTIER_SPECS[role].to_dict() for role in QUALITY_ROLES
    }:
        raise ValueError("BPE quality feasibility model specs differ")
    expected_feasibility = {
        "calibration_bytes": CALIBRATION_BYTES,
        "campaign_hour_limit": CAMPAIGN_HOUR_LIMIT,
        "candidate_train_byte_budgets": list(CANDIDATE_TRAIN_BYTE_BUDGETS),
        "driver_memory_fraction_limit": DRIVER_MEMORY_FRACTION_LIMIT,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "evaluation_batch_by_vocabulary": {
            str(key): value for key, value in EVALUATION_BATCH_BY_VOCABULARY.items()
        },
        "measured_effective_steps": MEASURED_EFFECTIVE_STEPS,
        "measured_evaluation_batches": MEASURED_EVALUATION_BATCHES,
        "sequence_length": SEQUENCE_LENGTH,
        "train_bytes": TRAIN_BYTES,
        "train_microbatch_by_vocabulary": {
            str(key): value for key, value in TRAIN_MICROBATCH_BY_VOCABULARY.items()
        },
        "warmup_effective_steps": WARMUP_EFFECTIVE_STEPS,
        "warmup_evaluation_batches": WARMUP_EVALUATION_BATCHES,
    }
    if plan["feasibility"] != expected_feasibility:
        raise ValueError("BPE quality feasibility timing contract differs")
    if set(plan["inventories"]) != set(QUALITY_ROLES):
        raise ValueError("BPE quality feasibility inventory roles differ")
    for role, row in plan["inventories"].items():
        if set(row) != {"calibration", "train"}:
            raise ValueError("BPE quality feasibility inventory split differs")
        validate_inventory(row["train"])
        validate_inventory(row["calibration"])
        vocabulary, _ = parse_role(role)
        if (
            row["train"]["raw_stream_bytes"] != TRAIN_BYTES
            or row["calibration"]["raw_stream_bytes"] != CALIBRATION_BYTES
            or row["train"]["first_batch_token_count"]
            != EFFECTIVE_BATCH_SIZE * SEQUENCE_LENGTH
            or row["calibration"]["first_batch_token_count"]
            != EVALUATION_BATCH_BY_VOCABULARY[vocabulary] * SEQUENCE_LENGTH
        ):
            raise ValueError("BPE quality feasibility inventory size differs")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("BPE quality feasibility implementation set differs")
    for relative in IMPLEMENTATION_PATHS:
        if hash_file(ROOT / relative) != plan["implementation_sha256"][relative]:
            raise ValueError(f"BPE quality feasibility implementation changed: {relative}")
    if plan["claim_boundary"] != {
        "actual_mps_training_and_evaluation_steps": True,
        "model_quality_measured": False,
        "projection_not_full_training": True,
        "random_weights": True,
        "selected_budget_uses_only_time_and_memory": True,
    }:
        raise ValueError("BPE quality feasibility claim boundary differs")
    dependencies = plan["dependencies"]
    expected_dependency_paths = {
        "integrity": INTEGRITY_PATH,
        "source": SOURCE_PATH,
        "systems_result": SYSTEMS_RESULT_PATH,
    }
    if set(dependencies) != {
        "git_commit_before_plan",
        "integrity",
        "source",
        "systems_result",
        "tokenizers",
    }:
        raise ValueError("BPE quality feasibility dependencies differ")
    if (
        not isinstance(dependencies["git_commit_before_plan"], str)
        or len(dependencies["git_commit_before_plan"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in dependencies["git_commit_before_plan"]
        )
    ):
        raise ValueError("BPE quality feasibility base commit differs")
    for key, path in expected_dependency_paths.items():
        if dependencies[key] != {
            "path": str(path.relative_to(ROOT)),
            "sha256": hash_file(path),
        }:
            raise ValueError(f"BPE quality feasibility dependency changed: {key}")
    if set(dependencies["tokenizers"]) != {
        str(parse_role(role)[0]) for role in QUALITY_ROLES
    }:
        raise ValueError("BPE quality feasibility tokenizer dependency set differs")
    for key, row in dependencies["tokenizers"].items():
        path = TOKENIZER_PATHS[int(key)]
        if row != {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}:
            raise ValueError("BPE quality feasibility tokenizer dependency changed")
