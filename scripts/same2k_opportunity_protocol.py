"""Paths, frozen inputs, and validation for the same-2K opportunity gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tokenizers import Tokenizer

from byte_unigram import PINNED_SENTENCEPIECE_VERSION
from fixed_byte_tokenizer import (
    build_fixed_byte_tokenizer,
    build_scored_byte_unigram_tokenizer,
)
from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.neural_data import build_neural_stream
from jamoflow.publication_bpe import PINNED_TOKENIZERS_VERSION, byte_bpe_token_bytes
from same2k_opportunity import (
    BPE_ROLE,
    ENCODE_REPETITIONS,
    LONGEST_MATCH_ROLE,
    MEASURED_CASES,
    MINIMUM_STEP_REDUCTION,
    MINIMUM_TOKEN_ROLE,
    ROLES,
    SCORED_UNIGRAM_ROLE,
    VOCABULARY_SIZE,
    WARMUP_CASES,
)
from token_frontier_protocol import (
    CALIBRATION_BYTES,
    CONTINUATION_BYTES,
    PROMPT_BYTES,
    array_sha256,
    reconstruct_cases,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "jamoflow-same2k-generic-opportunity-v6"
PLAN_PATH = ROOT / "data/manifests/same2k-generic-opportunity-v6.json"
RESULT_PATH = ROOT / "results/same2k-generic-opportunity-v6/summary.json"
ARTIFACT_ROOT = ROOT / "artifacts/same2k-generic-opportunity-v6"
TRAINED_TOKENIZER_PATH = ARTIFACT_ROOT / "byte-unigram-2048.json"
SENTENCEPIECE_MODEL_PATH = ARTIFACT_ROOT / "byte-unigram-2048.model"
PIECES_PATH = ARTIFACT_ROOT / "byte-unigram-2048-pieces.npz"
WORKER_PATH = ARTIFACT_ROOT / "worker.json"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"

SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
INTEGRITY_PATH = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
SYSTEMS_PLAN_PATH = ROOT / "data/manifests/korean-bpe-systems-frontier-v1.json"
SYSTEMS_RESULT_PATH = ROOT / "results/korean-bpe-systems-frontier-v1/summary.json"
BPE_TOKENIZER_PATH = (
    ROOT / "artifacts/korean-bpe-systems-frontier-v1/byte-bpe-2048.json"
)

EXPECTED_DEPENDENCY_SHA256 = {
    "bpe_tokenizer": "7db98d328ba1b7e7300c040a40fcdae6901d77db34cd52ea07f52bdc9fe15054",
    "integrity": "472cc5da045909109718be71168e516be19043cb2a08363d573ed77650038181",
    "source": "f789bc7e0ec0252c4c7c636e67a7c44f6d2c528a292ec47542af98488c8b36a5",
    "systems_plan": "3851b989ef36a841c72a9f2de35568a5d8188eea06b1a1cbb691e5d61b67d8ee",
    "systems_result": "01fba62f80ddf22e0c7aeb2cf7e40b36a6c36d31eebce8863f62a9381e971b8a",
}

IMPLEMENTATION_PATHS = (
    "docs/121-length-max-thunder-tok-reproduction-audit.md",
    "docs/122-byte-unigram-exploration-and-protocol-decision.md",
    "docs/123-same2k-generic-opportunity-protocol.md",
    "docs/124-same2k-v1-nfc-assumption-invalidation.md",
    "docs/125-same2k-v2-byte-fallback-invalidation.md",
    "docs/126-same2k-v3-wordpiece-runtime-invalidation.md",
    "docs/127-same2k-v4-tokenizer-json-verifier-invalidation.md",
    "docs/128-same2k-v5-deployable-runtime-invalidation.md",
    "pyproject.toml",
    "scripts/run_same2k_opportunity.py",
    "scripts/same2k_opportunity_protocol.py",
    "scripts/seal_same2k_opportunity_plan.py",
    "scripts/summarize_same2k_opportunity.py",
    "scripts/token_frontier_protocol.py",
    "requirements/same2k-opportunity-v1.txt",
    "scripts/byte_unigram.py",
    "scripts/fixed_byte_tokenizer.py",
    "scripts/same2k_opportunity.py",
    "tests/test_byte_unigram.py",
    "tests/test_fixed_byte_tokenizer.py",
    "tests/test_same2k_opportunity.py",
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
        raise ValueError(f"same-2K JSON root differs: {path}")
    return value


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "bpe_tokenizer": BPE_TOKENIZER_PATH,
        "integrity": INTEGRITY_PATH,
        "source": SOURCE_PATH,
        "systems_plan": SYSTEMS_PLAN_PATH,
        "systems_result": SYSTEMS_RESULT_PATH,
    }
    output = {}
    for name, path in paths.items():
        actual = hash_file(path)
        if actual != EXPECTED_DEPENDENCY_SHA256[name]:
            raise ValueError(f"same-2K dependency changed: {name}")
        output[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
        }
    return output


def current_environment() -> dict[str, Any]:
    import sentencepiece
    import tokenizers

    if (
        tokenizers.__version__ != PINNED_TOKENIZERS_VERSION
        or sentencepiece.__version__ != PINNED_SENTENCEPIECE_VERSION
    ):
        raise RuntimeError("same-2K tokenizer package version differs")
    return {
        **current_runtime_environment_contract(),
        "sentencepiece": sentencepiece.__version__,
        "tokenizers": tokenizers.__version__,
    }


def reconstruct_shared_inputs() -> tuple[bytes, np.ndarray, np.ndarray, dict[str, Any]]:
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=512,
    )
    prompts, continuations, case_metadata = reconstruct_cases()
    if (
        len(stream.data) != CALIBRATION_BYTES
        or prompts.shape != (WARMUP_CASES + MEASURED_CASES, PROMPT_BYTES)
        or continuations.shape != prompts.shape
    ):
        raise ValueError("same-2K shared input shape differs")
    metadata = {
        "calibration_bytes": len(stream.data),
        "calibration_stream_sha256": hashlib.sha256(stream.data).hexdigest(),
        "case_metadata": case_metadata,
        "continuation_array_sha256": array_sha256(continuations),
        "prompt_array_sha256": array_sha256(prompts),
    }
    return stream.data, prompts, continuations, metadata


def load_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(BPE_TOKENIZER_PATH))
    if (
        tokenizer.get_vocab_size(with_added_tokens=True) != VOCABULARY_SIZE
        or len(byte_bpe_token_bytes(tokenizer)) != VOCABULARY_SIZE
    ):
        raise ValueError("same-2K BPE tokenizer differs")
    return tokenizer


def tokenizer_semantic_sha256(tokenizer) -> str:
    """Hash canonical tokenizer JSON independent of serializer key order."""

    payload = json.loads(tokenizer.to_str(pretty=False))
    if not isinstance(payload, dict):
        raise ValueError("same-2K tokenizer JSON root differs")
    return canonical_sha256(payload)


def tokenizer_payload_semantic_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the serialized artifact payload before loader float rewriting."""

    if not isinstance(payload, Mapping):
        raise ValueError("same-2K tokenizer payload differs")
    return canonical_sha256(dict(payload))


def load_learned_tokenizers():
    tokenizer_payload = read_json(TRAINED_TOKENIZER_PATH)
    tokenizer = Tokenizer.from_file(str(TRAINED_TOKENIZER_PATH))
    with np.load(PIECES_PATH, allow_pickle=False) as artifact:
        if set(artifact.files) != {"lengths", "offsets", "raw", "scores"}:
            raise ValueError("same-2K piece artifact schema differs")
        raw = np.asarray(artifact["raw"], dtype=np.uint8)
        offsets = np.asarray(artifact["offsets"], dtype=np.int64)
        lengths = np.asarray(artifact["lengths"], dtype=np.int64)
        scores = np.asarray(artifact["scores"], dtype=np.float64)
    if (
        offsets.shape != (VOCABULARY_SIZE,)
        or lengths.shape != (VOCABULARY_SIZE,)
        or scores.shape != (VOCABULARY_SIZE,)
        or np.any(offsets < 0)
        or np.any(lengths <= 0)
        or np.any(offsets + lengths > len(raw))
    ):
        raise ValueError("same-2K piece arrays differ")
    pieces = tuple(
        bytes(raw[offset : offset + length])
        for offset, length in zip(offsets, lengths, strict=True)
    )
    if byte_bpe_token_bytes(tokenizer) != pieces:
        raise ValueError("same-2K learned tokenizer piece identity differs")
    scored = build_scored_byte_unigram_tokenizer(pieces, tuple(scores))
    if tokenizer_payload_semantic_sha256(tokenizer_payload) != tokenizer_semantic_sha256(
        scored
    ):
        raise ValueError("same-2K learned tokenizer reconstruction differs")
    return {
        # Evaluate the deployable from-file runtime.  Rust may rewrite a small
        # subset of score decimals by one ULP while loading; the worker uses the
        # same round-trip, and exact token-stream replay detects any consequence.
        SCORED_UNIGRAM_ROLE: tokenizer,
        LONGEST_MATCH_ROLE: build_fixed_byte_tokenizer(
            pieces, segmentation="leftmost_longest"
        ),
        MINIMUM_TOKEN_ROLE: build_fixed_byte_tokenizer(
            pieces, segmentation="minimum_token_dp"
        ),
    }, pieces, tuple(float(value) for value in scores)


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "claim_boundary",
        "dependencies",
        "environment",
        "experiment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "known_exploratory_anchors",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "shared_inputs",
        "status",
    }
    if set(plan) != expected_keys:
        raise ValueError("same-2K plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "same2k_generic_opportunity_plan_v6"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_deterministic_full_corpus_training"
    ):
        raise ValueError("same-2K plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("same-2K plan hash differs")
    if plan["dependencies"] != dependency_identity():
        raise ValueError("same-2K plan dependency identity differs")
    if plan["environment"] != current_environment():
        raise ValueError("same-2K plan environment differs")
    if plan["experiment"] != {
        "calibration_bytes": CALIBRATION_BYTES,
        "encode_repetitions": ENCODE_REPETITIONS,
        "measured_cases": MEASURED_CASES,
        "minimum_step_reduction": MINIMUM_STEP_REDUCTION,
        "roles": list(ROLES),
        "tokenizer_training_maximum_piece_bytes": 48,
        "vocabulary_size": VOCABULARY_SIZE,
        "warmup_cases": WARMUP_CASES,
    }:
        raise ValueError("same-2K experiment contract differs")
    _, _, _, shared = reconstruct_shared_inputs()
    if plan["shared_inputs"] != shared:
        raise ValueError("same-2K shared input identity differs")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("same-2K implementation set differs")
    for relative in IMPLEMENTATION_PATHS:
        if hash_file(ROOT / relative) != plan["implementation_sha256"][relative]:
            raise ValueError(f"same-2K implementation changed: {relative}")
    if plan["known_exploratory_anchors"] != {
        "hf_unigram_no_regex_calibration_token_count": 2_173_590,
        "hf_unigram_no_regex_reduction_vs_bpe": 0.03971148799457114,
        "hf_unigram_regex_calibration_token_count": 2_328_984,
        "hf_unigram_regex_reduction_vs_bpe": -0.02894132740970079,
        "interpretation": "unsealed nondeterministic API exploration; disclosed, not selection evidence",
    }:
        raise ValueError("same-2K disclosed exploratory anchors differ")
    if plan["claim_boundary"] != {
        "actual_model_latency": False,
        "calibration_development_only": True,
        "korean_aware_method_evaluated": False,
        "model_quality_used": False,
        "publication_evidence": False,
        "token_only_generic_upper_bound": True,
    }:
        raise ValueError("same-2K claim boundary differs")
