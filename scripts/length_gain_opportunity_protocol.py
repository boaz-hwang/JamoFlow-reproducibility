"""Sealed coordinates for the train-only Length-Gain opportunity gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from jamoflow.inference_actual_v5 import current_runtime_environment_contract
from jamoflow.neural_data import build_neural_stream
from length_gain import DEFAULT_MAXIMUM_PIECE_BYTES, DEFAULT_VOCABULARY_SIZE
from same2k_opportunity_protocol import reconstruct_shared_inputs


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "jamoflow-length-gain-opportunity-v1"
PLAN_PATH = ROOT / "data/manifests/length-gain-opportunity-v1.json"
RESULT_PATH = ROOT / "results/length-gain-opportunity-v1/summary.json"
ARTIFACT_ROOT = ROOT / "artifacts/length-gain-opportunity-v1"
PIECES_PATH = ARTIFACT_ROOT / "pieces.npz"
WORKER_PATH = ARTIFACT_ROOT / "worker.json"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
SUMMARY_ACTIVE_PATH = ARTIFACT_ROOT / ".summary-active"

SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
INTEGRITY_PATH = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
BPE_TOKENIZER_PATH = ROOT / "artifacts/korean-bpe-systems-frontier-v1/byte-bpe-2048.json"
SAME2K_PLAN_PATH = ROOT / "data/manifests/same2k-generic-opportunity-v6.json"
SAME2K_RESULT_PATH = ROOT / "results/same2k-generic-opportunity-v6/summary.json"

EXPECTED_DEPENDENCY_SHA256 = {
    "bpe_tokenizer": "7db98d328ba1b7e7300c040a40fcdae6901d77db34cd52ea07f52bdc9fe15054",
    "integrity": "472cc5da045909109718be71168e516be19043cb2a08363d573ed77650038181",
    "same2k_plan": "154d2249694a152b17680cfb431350aca1082c222dd17232835d016659aef37d",
    "same2k_result": "1e9e57e76998c8729c68cee4f5a3f8dd8eb4528eda933597bf6eb4060ab09126",
    "source": "f789bc7e0ec0252c4c7c636e67a7c44f6d2c528a292ec47542af98488c8b36a5",
}

TRAIN_BYTES = 8_000_000
TRAIN_SEQUENCE_LENGTH = 512
VOCABULARY_SIZE = DEFAULT_VOCABULARY_SIZE
BATCH_SIZE = 8
MAXIMUM_TOKEN_ARITY = 8
MAXIMUM_PIECE_BYTES = DEFAULT_MAXIMUM_PIECE_BYTES
SCORE_KIND = "immediate_saving"
MINIMUM_REDUCTION = 0.10
WARMUP_CASES = 6
MEASURED_CASES = 36
ENCODE_REPETITIONS = 3

BPE_ROLE = "byte_bpe_2k"
LONGEST_ROLE = "byte_length_gain_2k_leftmost_longest"
MINIMUM_ROLE = "byte_length_gain_2k_minimum_token_dp"
ROLE_ORDER = (BPE_ROLE, LONGEST_ROLE, MINIMUM_ROLE)
PRIMARY_ORDER = (LONGEST_ROLE, MINIMUM_ROLE)

IMPLEMENTATION_PATHS = (
    "docs/121-length-max-thunder-tok-reproduction-audit.md",
    "docs/129-same2k-generic-opportunity-result-and-length-gain-pivot.md",
    "docs/130-length-gain-opportunity-protocol.md",
    "pyproject.toml",
    "scripts/fixed_byte_tokenizer.py",
    "scripts/length_gain.py",
    "scripts/length_gain_opportunity_protocol.py",
    "scripts/run_length_gain_opportunity.py",
    "scripts/seal_length_gain_opportunity_plan.py",
    "scripts/summarize_length_gain_opportunity.py",
    "scripts/same2k_opportunity_protocol.py",
    "tests/test_length_gain.py",
    "tests/test_length_gain_opportunity_protocol.py",
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
        raise ValueError(f"length-gain JSON root differs: {path}")
    return value


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "bpe_tokenizer": BPE_TOKENIZER_PATH,
        "integrity": INTEGRITY_PATH,
        "same2k_plan": SAME2K_PLAN_PATH,
        "same2k_result": SAME2K_RESULT_PATH,
        "source": SOURCE_PATH,
    }
    output = {}
    for name, path in paths.items():
        actual = hash_file(path)
        if actual != EXPECTED_DEPENDENCY_SHA256[name]:
            raise ValueError(f"length-gain dependency changed: {name}")
        output[name] = {"path": str(path.relative_to(ROOT)), "sha256": actual}
    return output


def implementation_identity() -> dict[str, str]:
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def current_environment() -> dict[str, Any]:
    import tokenizers

    return {
        **current_runtime_environment_contract(),
        "tokenizers": tokenizers.__version__,
    }


def reconstruct_train_stream() -> tuple[bytes, dict[str, Any]]:
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="train",
        byte_limit=TRAIN_BYTES,
        sequence_length=TRAIN_SEQUENCE_LENGTH,
    )
    if len(stream.data) != TRAIN_BYTES:
        raise ValueError("length-gain train stream length differs")
    return stream.data, {
        **stream.metadata(),
        "sha256": hashlib.sha256(stream.data).hexdigest(),
    }


def reconstruct_all_inputs():
    train_raw, train_metadata = reconstruct_train_stream()
    calibration_raw, prompts, continuations, shared = reconstruct_shared_inputs()
    return train_raw, calibration_raw, prompts, continuations, {
        "train": train_metadata,
        "calibration_and_cases": shared,
    }


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "claim_boundary",
        "dependencies",
        "environment",
        "experiment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "known_train_only_preflight",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "shared_inputs",
        "status",
    }
    if set(plan) != expected:
        raise ValueError("length-gain plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "length_gain_opportunity_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_first_length_gain_calibration_evaluation"
    ):
        raise ValueError("length-gain plan identity differs")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != plan["plan_sha256"]:
        raise ValueError("length-gain plan hash differs")
    experiment = plan["experiment"]
    expected_experiment = {
        "batch_size": BATCH_SIZE,
        "encode_repetitions": ENCODE_REPETITIONS,
        "maximum_piece_bytes": MAXIMUM_PIECE_BYTES,
        "maximum_token_arity": MAXIMUM_TOKEN_ARITY,
        "measured_cases": MEASURED_CASES,
        "minimum_reduction": MINIMUM_REDUCTION,
        "primary_order": list(PRIMARY_ORDER),
        "role_order": list(ROLE_ORDER),
        "score_kind": SCORE_KIND,
        "train_bytes": TRAIN_BYTES,
        "vocabulary_size": VOCABULARY_SIZE,
        "warmup_cases": WARMUP_CASES,
    }
    if experiment != expected_experiment:
        raise ValueError("length-gain experiment differs")
    if plan["dependencies"] != dependency_identity():
        raise ValueError("length-gain dependency identity differs")
    if plan["implementation_sha256"] != implementation_identity():
        raise ValueError("length-gain implementation identity differs")
    if plan["environment"] != current_environment():
        raise ValueError("length-gain environment differs")


def serialize_pieces(path: Path, pieces: tuple[bytes, ...]) -> None:
    lengths = np.asarray([len(piece) for piece in pieces], dtype="<u2")
    offsets = np.zeros(len(pieces) + 1, dtype="<u8")
    offsets[1:] = np.cumsum(lengths, dtype=np.uint64)
    raw = np.frombuffer(b"".join(pieces), dtype=np.uint8).copy()
    with path.open("wb") as handle:
        np.savez(handle, lengths=lengths, offsets=offsets, raw=raw)


def load_pieces(path: Path) -> tuple[bytes, ...]:
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"lengths", "offsets", "raw"}:
            raise ValueError("length-gain pieces artifact schema differs")
        lengths = artifact["lengths"]
        offsets = artifact["offsets"]
        raw = artifact["raw"]
    if (
        lengths.dtype != np.dtype("<u2")
        or offsets.dtype != np.dtype("<u8")
        or raw.dtype != np.uint8
        or lengths.shape != (VOCABULARY_SIZE,)
        or offsets.shape != (VOCABULARY_SIZE + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(raw)
        or not np.array_equal(np.diff(offsets), lengths.astype(np.uint64))
    ):
        raise ValueError("length-gain pieces artifact arrays differ")
    blob = bytes(raw)
    return tuple(blob[int(offsets[i]) : int(offsets[i + 1])] for i in range(len(lengths)))
