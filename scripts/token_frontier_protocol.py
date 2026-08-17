"""Sealed inputs, cases, artifacts, and validation for token frontier v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tokenizers import Tokenizer

from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.inference_benchmark import select_inference_cases
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.publication_bpe import PINNED_TOKENIZERS_VERSION, byte_bpe_token_bytes
from token_frontier_core import (
    DEPTHS,
    FRONTIER_SPECS,
    PARAMETER_RELATIVE_TOLERANCE,
    PARAMETER_TARGET,
    RUNTIME_ROLES,
    VOCABULARY_SIZES,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "jamoflow-korean-bpe-systems-frontier-v1"
PLAN_PATH = ROOT / "data/manifests/korean-bpe-systems-frontier-v1.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
INTEGRITY_PATH = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
PRIOR_RESULT_PATH = ROOT / "results/scalar-runtime-preflight-v1/summary.json"
PRIOR_OPPORTUNITY_PATH = (
    ROOT / "results/scalar-representation-opportunity-v1/summary.json"
)
ARTIFACT_ROOT = ROOT / "artifacts/korean-bpe-systems-frontier-v1"
OPPORTUNITY_REPORT_PATH = ARTIFACT_ROOT / "tokenizer-report.json"
RUNTIME_REPORT_PATH = ARTIFACT_ROOT / "runtime-report.json"
TIMING_PATH = ARTIFACT_ROOT / "timings.npz"
RUNTIME_ACTIVE_PATH = ARTIFACT_ROOT / ".runtime-active"
OUTPUT_PATH = ROOT / "results/korean-bpe-systems-frontier-v1/summary.json"
TOKENIZER_PATHS = {
    size: ARTIFACT_ROOT / f"byte-bpe-{size}.json" for size in VOCABULARY_SIZES
}

CALIBRATION_BYTES = 8_000_000
SEQUENCE_LENGTH = 512
PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
WARMUP_CASES = 6
MEASURED_CASES = 36
REPETITIONS = 3
MODEL_SEED = 20_260_814
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_815
TOKENIZER_ENCODE_REPETITIONS = 5
MPS_ATOL = 1e-4
MPS_RTOL = 2e-5

KNOWN_PRESEAL_ENGINEERING_ANCHORS = {
    "interpretation": (
        "disclosed prior engineering observations; used to motivate the grid, "
        "not to select a quality comparator"
    ),
    "new_outcomes_unobserved_at_seal": [
        "2K/4K/8K/64K calibration token counts",
        "all 18 parameter-matched graph runtimes",
    ],
    "prior_byte_bpe_16000": {
        "calibration_token_count": 1_533_938,
        "random_weight_end_to_end_median_ms": 67.6335205,
    },
    "prior_byte_bpe_32000": {
        "calibration_token_count": 1_388_745,
        "random_weight_end_to_end_median_ms": 92.8866045,
    },
}

IMPLEMENTATION_PATHS = (
    "docs/114-latest-tokenizer-frontier-reassessment.md",
    "docs/115-korean-bpe-systems-frontier-protocol.md",
    "pyproject.toml",
    "scripts/benchmark_token_frontier_runtime.py",
    "scripts/run_token_frontier_opportunity.py",
    "scripts/scalar_representation_core.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_token_frontier_plan.py",
    "scripts/summarize_token_frontier.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_token_frontier_core.py",
    "tests/test_token_frontier_protocol.py",
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def current_frontier_environment() -> dict[str, Any]:
    return {
        "device": "mps",
        "mps_available": bool(torch.backends.mps.is_available()),
        **current_runtime_environment_contract(),
    }


def reconstruct_cases() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        SEQUENCE_LENGTH,
    )
    documents = reconstruct_document_window_map(
        SOURCE_PATH,
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=SEQUENCE_LENGTH,
        expected_stream=stream.data,
    )
    eligible = documents.document_indices >= 0
    cases = select_inference_cases(
        inputs[eligible],
        boundaries[eligible],
        cluster_ids=documents.document_indices[eligible],
        case_count=WARMUP_CASES + MEASURED_CASES,
        prompt_length=PROMPT_BYTES,
        continuation_length=CONTINUATION_BYTES,
    )
    prompts = cases.prompts.astype(np.uint8, copy=False)
    continuations = cases.replay_continuations.astype(np.uint8, copy=False)
    for row in np.concatenate((prompts, continuations), axis=0):
        bytes(row).decode("utf-8", errors="strict")
    metadata = {
        "algorithm": "outcome-independent one-case-per-document Hangul-heavy selector",
        "calibration_stream_sha256": hashlib.sha256(stream.data).hexdigest(),
        "candidate_document_windows": int(np.count_nonzero(eligible)),
        "continuation_array_sha256": array_sha256(continuations),
        "document_assignment_sha256": documents.metadata()["document_assignment_sha256"],
        "measured_cases": MEASURED_CASES,
        "prompt_array_sha256": array_sha256(prompts),
        "warmup_cases": WARMUP_CASES,
        **cases.public_metadata(),
    }
    return prompts, continuations, metadata


def load_tokenizers() -> dict[int, tuple[Tokenizer, tuple[bytes, ...]]]:
    output = {}
    for size, path in TOKENIZER_PATHS.items():
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size(with_added_tokens=True) != size:
            raise ValueError("frontier tokenizer vocabulary differs")
        output[size] = (tokenizer, byte_bpe_token_bytes(tokenizer))
    return output


def encode_case(
    raw: bytes,
    tokenizer: Tokenizer,
    token_bytes: Sequence[bytes],
) -> tuple[int, ...]:
    text = raw.decode("utf-8", errors="strict")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = tuple(int(value) for value in encoding.ids)
    if not ids or tokenizer.decode(list(ids)) != text:
        raise ValueError("frontier BPE text roundtrip differs")
    if b"".join(token_bytes[value] for value in ids) != raw:
        raise ValueError("frontier BPE raw-byte roundtrip differs")
    return ids


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "cases",
        "claim_boundary",
        "dependencies",
        "environment",
        "experiment",
        "implementation_sha256",
        "kind",
        "known_preseal_engineering_anchors",
        "model_specs",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "status",
        "tokenizer",
    }
    if set(plan) != expected:
        raise ValueError("token frontier plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "korean_bpe_systems_frontier_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"]
        != "sealed_after_known_16k_32k_anchors_before_new_grid_counts_and_runtime"
    ):
        raise ValueError("token frontier plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("token frontier plan hash differs")
    expected_dependencies = {
        "git_commit_before_plan": plan["dependencies"].get("git_commit_before_plan"),
        "integrity_path": str(INTEGRITY_PATH.relative_to(ROOT)),
        "integrity_sha256": hash_file(INTEGRITY_PATH),
        "prior_scalar_opportunity_path": str(PRIOR_OPPORTUNITY_PATH.relative_to(ROOT)),
        "prior_scalar_opportunity_sha256": hash_file(PRIOR_OPPORTUNITY_PATH),
        "prior_scalar_runtime_path": str(PRIOR_RESULT_PATH.relative_to(ROOT)),
        "prior_scalar_runtime_sha256": hash_file(PRIOR_RESULT_PATH),
        "source_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": hash_file(SOURCE_PATH),
    }
    if (
        not isinstance(plan["dependencies"].get("git_commit_before_plan"), str)
        or len(plan["dependencies"]["git_commit_before_plan"]) != 40
        or plan["dependencies"] != expected_dependencies
    ):
        raise ValueError("token frontier dependency identity differs")
    if plan["environment"] != current_frontier_environment():
        raise ValueError("token frontier runtime environment changed")
    if plan["known_preseal_engineering_anchors"] != KNOWN_PRESEAL_ENGINEERING_ANCHORS:
        raise ValueError("token frontier disclosed anchors differ")
    if plan["tokenizer"] != {
        "add_prefix_space": False,
        "byte_fallback": False,
        "dropout": None,
        "initial_alphabet": "complete ByteLevel alphabet",
        "minimum_frequency": 2,
        "normalizer": None,
        "replicate_training_for_exact_json_determinism": True,
        "diagnostic_calibration_encode_repetitions": TOKENIZER_ENCODE_REPETITIONS,
        "tokenizers_version": PINNED_TOKENIZERS_VERSION,
        "train_split_only": True,
        "use_regex": True,
        "vocabulary_sizes": list(VOCABULARY_SIZES),
    }:
        raise ValueError("token frontier tokenizer contract differs")
    if plan["experiment"] != {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "depths": list(DEPTHS),
        "measured_cases": MEASURED_CASES,
        "model_seed": MODEL_SEED,
        "mps_atol": MPS_ATOL,
        "mps_rtol": MPS_RTOL,
        "parameter_relative_tolerance": PARAMETER_RELATIVE_TOLERANCE,
        "parameter_target": PARAMETER_TARGET,
        "prompt_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "repetitions": REPETITIONS,
        "roles": list(RUNTIME_ROLES),
        "vocabulary_sizes": list(VOCABULARY_SIZES),
        "warmup_cases": WARMUP_CASES,
    }:
        raise ValueError("token frontier experiment differs")
    expected_case_keys = {
        "algorithm",
        "calibration_stream_sha256",
        "candidate_document_windows",
        "candidate_rows",
        "continuation_array_sha256",
        "continuation_length_bytes",
        "document_assignment_sha256",
        "measured_cases",
        "prompt_array_sha256",
        "prompt_length_bytes",
        "selected_cases",
        "selected_unique_clusters",
        "unique_candidate_clusters",
        "unique_candidate_prompts",
        "warmup_cases",
    }
    if (
        set(plan["cases"]) != expected_case_keys
        or plan["cases"]["selected_cases"] != WARMUP_CASES + MEASURED_CASES
        or plan["cases"]["selected_unique_clusters"] != WARMUP_CASES + MEASURED_CASES
        or plan["cases"]["prompt_length_bytes"] != PROMPT_BYTES
        or plan["cases"]["continuation_length_bytes"] != CONTINUATION_BYTES
        or plan["cases"]["warmup_cases"] != WARMUP_CASES
        or plan["cases"]["measured_cases"] != MEASURED_CASES
    ):
        raise ValueError("token frontier case contract differs")
    if plan["model_specs"] != {
        role: FRONTIER_SPECS[role].to_dict() for role in RUNTIME_ROLES
    }:
        raise ValueError("token frontier model specs differ")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("token frontier implementation set differs")
    for relative in IMPLEMENTATION_PATHS:
        if hash_file(ROOT / relative) != plan["implementation_sha256"][relative]:
            raise ValueError(f"token frontier implementation changed: {relative}")
    if plan["claim_boundary"] != {
        "actual_model_graph_timing": True,
        "calibration_development_only": True,
        "free_running_generation": False,
        "matched_quality": False,
        "new_tokenizer_method": False,
        "random_weights_only": True,
        "tokenization_inside_model_timer": False,
    }:
        raise ValueError("token frontier claim boundary differs")
