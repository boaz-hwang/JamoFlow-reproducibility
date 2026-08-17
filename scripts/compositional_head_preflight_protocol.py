"""Sealed inputs and identities for compositional-head systems preflight v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from tokenizers import Tokenizer

from compositional_head_core import (
    BASE_ROLE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    MINIMUM_END_TO_END_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MINIMUM_STEP_REDUCTION,
    MODEL_SEED,
    ROLE_ORDER,
    ROLE_SPECS,
    VOCABULARY_SIZES,
    analytical_head_multiply_adds_per_position,
    assignment_audit_for_role,
    parse_role,
)
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.publication_bpe import byte_bpe_token_bytes
from token_frontier_protocol import (
    CONTINUATION_BYTES,
    INTEGRITY_PATH,
    MEASURED_CASES,
    PROMPT_BYTES,
    REPETITIONS,
    ROOT,
    SOURCE_PATH,
    TOKENIZER_PATHS,
    WARMUP_CASES,
    array_sha256,
    reconstruct_cases,
)


PROTOCOL_ID = "jamoflow-compositional-head-systems-preflight-v2"
PLAN_PATH = ROOT / "data/manifests/compositional-head-systems-preflight-v2.json"
ARTIFACT_ROOT = ROOT / "artifacts/compositional-head-systems-preflight-v2"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
RESULT_ROOT = ROOT / "results/compositional-head-systems-preflight-v2"
EVIDENCE_ROOT = RESULT_ROOT / "evidence"
TIMING_PATH = EVIDENCE_ROOT / "timings.npz"
REPORT_PATH = EVIDENCE_ROOT / "report.json"
RESULT_PATH = RESULT_ROOT / "summary.json"

LENGTH_GAIN_RESULT_PATH = ROOT / "results/length-gain-opportunity-v1/summary.json"
BPE_QUALITY_RESULT_PATH = ROOT / "results/bpe-quality-frontier-one-seed-v1/summary.json"
TOKEN_FRONTIER_PLAN_PATH = ROOT / "data/manifests/korean-bpe-systems-frontier-v1.json"
TOKEN_FRONTIER_RESULT_PATH = ROOT / "results/korean-bpe-systems-frontier-v1/summary.json"
TOKENIZER_SIZES = (2_048,) + VOCABULARY_SIZES
MPS_ATOL = 1e-4
MPS_RTOL = 2e-5

IMPLEMENTATION_PATHS = (
    "docs/120-bpe-quality-frontier-one-seed-result.md",
    "docs/131-length-gain-result-and-compositional-head-pivot.md",
    "docs/132-compositional-head-systems-preflight-protocol.md",
    "pyproject.toml",
    "scripts/benchmark_compositional_head_preflight.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_compositional_head_preflight_plan.py",
    "scripts/summarize_compositional_head_preflight.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "scripts/compositional_token_head.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_compositional_head_core.py",
    "tests/test_compositional_head_preflight_protocol.py",
    "tests/test_compositional_token_head.py",
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
        raise ValueError(f"compositional-head JSON root differs: {path}")
    return value


def current_environment() -> dict[str, Any]:
    import tokenizers
    import transformers

    return {
        **current_runtime_environment_contract(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "bpe_quality_result": BPE_QUALITY_RESULT_PATH,
        "integrity": INTEGRITY_PATH,
        "length_gain_result": LENGTH_GAIN_RESULT_PATH,
        "source": SOURCE_PATH,
        "token_frontier_plan": TOKEN_FRONTIER_PLAN_PATH,
        "token_frontier_result": TOKEN_FRONTIER_RESULT_PATH,
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in paths.items()
    }


def implementation_identity() -> dict[str, str]:
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def load_tokenizers() -> dict[int, tuple[Tokenizer, tuple[bytes, ...]]]:
    output = {}
    for vocabulary_size in TOKENIZER_SIZES:
        path = TOKENIZER_PATHS[vocabulary_size]
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size(with_added_tokens=True) != vocabulary_size:
            raise ValueError("compositional-head tokenizer vocabulary differs")
        output[vocabulary_size] = (tokenizer, byte_bpe_token_bytes(tokenizer))
    return output


def tokenizer_identity() -> dict[str, dict[str, Any]]:
    tokenizers = load_tokenizers()
    return {
        str(vocabulary_size): {
            "path": str(TOKENIZER_PATHS[vocabulary_size].relative_to(ROOT)),
            "sha256": hash_file(TOKENIZER_PATHS[vocabulary_size]),
            "vocabulary_size": vocabulary_size,
            "ordered_token_bytes_sha256": _ordered_token_bytes_sha256(table),
        }
        for vocabulary_size, (_, table) in tokenizers.items()
    }


def _ordered_token_bytes_sha256(table: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256(b"JamoFlow/compositional-head-token-bytes/v1\0")
    digest.update(len(table).to_bytes(8, "big"))
    for piece in table:
        digest.update(len(piece).to_bytes(8, "big"))
        digest.update(piece)
    return digest.hexdigest()


def assignment_audits() -> dict[str, dict[str, Any]]:
    tokenizers = load_tokenizers()
    output = {}
    for role in ROLE_ORDER:
        kind, vocabulary_size = parse_role(role)
        if kind not in ("generic_code", "hangul_code"):
            continue
        row = assignment_audit_for_role(role, tokenizers[vocabulary_size][1])
        if row is None:
            raise AssertionError("compositional assignment audit is missing")
        output[role] = row
    return output


def case_identity() -> dict[str, Any]:
    prompts, continuations, metadata = reconstruct_cases()
    return {
        **metadata,
        "prompt_shape": list(prompts.shape),
        "continuation_shape": list(continuations.shape),
        "prompt_array_sha256": array_sha256(prompts),
        "continuation_array_sha256": array_sha256(continuations),
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "continuation_bytes": CONTINUATION_BYTES,
        "measured_cases": MEASURED_CASES,
        "minimum_end_to_end_reduction": MINIMUM_END_TO_END_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "minimum_step_reduction": MINIMUM_STEP_REDUCTION,
        "model_seed": MODEL_SEED,
        "mps_atol": MPS_ATOL,
        "mps_rtol": MPS_RTOL,
        "prompt_bytes": PROMPT_BYTES,
        "repetitions": REPETITIONS,
        "role_order": list(ROLE_ORDER),
        "selection_rule": (
            "smallest vocabulary where both generic and Hangul codebook roles "
            "pass; no result-dependent fallback"
        ),
        "tokenizer_inside_model_timer": False,
        "vocabulary_order": list(VOCABULARY_SIZES),
        "warmup_cases": WARMUP_CASES,
    }


def model_contract() -> dict[str, Any]:
    return {
        role: {
            **ROLE_SPECS[role].to_dict(),
            "analytical_head_multiply_adds_per_position": (
                analytical_head_multiply_adds_per_position(role)
            ),
        }
        for role in ROLE_ORDER
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "assignment_audits",
        "cases",
        "claim_boundary",
        "dependencies",
        "environment",
        "experiment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "known_engineering_smoke",
        "model_contract",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "status",
        "tokenizers",
    }
    if set(plan) != expected:
        raise ValueError("compositional-head plan schema differs")
    if (
        plan["schema_version"] != 2
        or plan["kind"] != "compositional_head_systems_preflight_plan_v2"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_full_grid_timing"
    ):
        raise ValueError("compositional-head plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("compositional-head plan hash differs")
    if plan["dependencies"] != dependency_identity():
        raise ValueError("compositional-head dependencies differ")
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("compositional-head implementation differs")
    if plan["environment"] != current_environment():
        raise ValueError("compositional-head environment differs")
    if plan["tokenizers"] != tokenizer_identity():
        raise ValueError("compositional-head tokenizer identity differs")
    if plan["assignment_audits"] != assignment_audits():
        raise ValueError("compositional-head assignments differ")
    if plan["cases"] != case_identity():
        raise ValueError("compositional-head cases differ")
    if plan["experiment"] != experiment_contract():
        raise ValueError("compositional-head experiment differs")
    if plan["model_contract"] != model_contract():
        raise ValueError("compositional-head model contract differs")
