"""Sealed inputs, cases, schedules, and statistics for scalar runtime v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer

from jamoflow.corpus import load_records, partition_records
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.inference_benchmark import paired_prompt_latency, select_inference_cases
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.publication_bpe import byte_bpe_token_bytes
from scalar_runtime_core import BPE_PRIMARY_SPEC, BPE_SECONDARY_SPEC, RUNTIME_ROLES


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "jamoflow-scalar-runtime-preflight-v1"
PLAN_PATH = ROOT / "data/manifests/scalar-runtime-preflight-v1.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
INTEGRITY_PATH = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
OPPORTUNITY_PATH = ROOT / "results/scalar-representation-opportunity-v1/summary.json"
TOKENIZER_PATHS = {
    16_000: ROOT / "artifacts/scalar-representation-opportunity-v1/byte-bpe-16000.json",
    32_000: ROOT / "artifacts/scalar-representation-opportunity-v1/byte-bpe-32000.json",
}
ARTIFACT_ROOT = ROOT / "artifacts/scalar-runtime-preflight-v1"
REPORT_PATH = ARTIFACT_ROOT / "runtime-report.json"
TIMING_PATH = ARTIFACT_ROOT / "timings.npz"
OUTPUT_PATH = ROOT / "results/scalar-runtime-preflight-v1/summary.json"

CALIBRATION_BYTES = 8_000_000
SEQUENCE_LENGTH = 512
PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
WARMUP_CASES = 8
MEASURED_CASES = 32
REPETITIONS = 3
CASE_POOL_SIZE = 330
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_814
PARAMETER_TARGET = 19_596_096
PARAMETER_RELATIVE_TOLERANCE = 0.0025
MPS_ATOL = 1e-4
MPS_RTOL = 2e-5

IMPLEMENTATION_PATHS = (
    "README.md",
    "docs/89-fable5-interim-review-and-research-direction.md",
    "docs/111-scalar-representation-opportunity-result.md",
    "docs/112-scalar-runtime-preflight-protocol.md",
    "pyproject.toml",
    "scripts/analyze_scalar_representation_opportunity.py",
    "scripts/benchmark_scalar_runtime_preflight.py",
    "scripts/scalar_representation_core.py",
    "scripts/scalar_runtime_core.py",
    "scripts/scalar_runtime_protocol.py",
    "scripts/seal_scalar_runtime_preflight_plan.py",
    "scripts/summarize_scalar_runtime_preflight.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/publication_bpe.py",
    "tests/test_scalar_runtime.py",
    "tests/test_scalar_runtime_protocol.py",
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_git_blob(commit: str, relative: str) -> str:
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("scalar runtime Git commit is malformed")
    payload = subprocess.check_output(
        ("git", "show", f"{commit}:{relative}"), cwd=ROOT
    )
    return hashlib.sha256(payload).hexdigest()


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
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _bpe_ids_and_boundary(
    tokenizer: Tokenizer,
    token_bytes: Sequence[bytes],
    raw: bytes,
) -> tuple[tuple[int, ...], int] | None:
    text = raw.decode("utf-8", errors="strict")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = tuple(int(value) for value in encoding.ids)
    if (
        not ids
        or tokenizer.decode(list(ids)) != text
        or b"".join(token_bytes[value] for value in ids) != raw
    ):
        raise ValueError("runtime BPE encoding is not reversible")
    offset = 0
    split = None
    for index, token_id in enumerate(ids, start=1):
        offset += len(token_bytes[token_id])
        if offset == PROMPT_BYTES:
            split = index
        if offset > PROMPT_BYTES and split is None:
            return None
    if offset != PROMPT_BYTES + CONTINUATION_BYTES or split is None:
        return None
    return ids, split


def load_tokenizers() -> dict[int, tuple[Tokenizer, tuple[bytes, ...]]]:
    output = {}
    for vocabulary_size, path in TOKENIZER_PATHS.items():
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size(with_added_tokens=True) != vocabulary_size:
            raise ValueError("runtime BPE vocabulary differs")
        output[vocabulary_size] = (tokenizer, byte_bpe_token_bytes(tokenizer))
    return output


def reconstruct_cases() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select fixed document-independent cases with boundaries for both BPEs."""

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
    pool = select_inference_cases(
        inputs[eligible],
        boundaries[eligible],
        cluster_ids=documents.document_indices[eligible],
        case_count=CASE_POOL_SIZE,
        prompt_length=PROMPT_BYTES,
        continuation_length=CONTINUATION_BYTES,
    )
    tokenizers = load_tokenizers()
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    token_counts: dict[int, list[tuple[int, int]]] = {
        value: [] for value in tokenizers
    }
    boundary_eligible = 0
    for prompt, continuation in zip(
        pool.prompts,
        pool.replay_continuations,
        strict=True,
    ):
        raw = bytes(prompt) + bytes(continuation)
        encodings = {
            vocabulary_size: _bpe_ids_and_boundary(tokenizer, table, raw)
            for vocabulary_size, (tokenizer, table) in tokenizers.items()
        }
        if any(value is None for value in encodings.values()):
            continue
        boundary_eligible += 1
        if len(selected) >= WARMUP_CASES + MEASURED_CASES:
            continue
        selected.append((prompt, continuation))
        for vocabulary_size, encoded in encodings.items():
            if encoded is None:
                raise AssertionError("BPE eligibility disappeared")
            ids, split = encoded
            token_counts[vocabulary_size].append((split, len(ids) - split))
    if len(selected) != WARMUP_CASES + MEASURED_CASES:
        raise ValueError("runtime preflight lacks BPE-aligned cases")
    prompts = np.stack([value[0] for value in selected]).astype(np.uint8)
    continuations = np.stack([value[1] for value in selected]).astype(np.uint8)
    if any(
        bytes(row).decode("utf-8", errors="strict") is None
        for row in np.concatenate((prompts, continuations), axis=0)
    ):
        raise AssertionError("strict runtime case validation failed")
    metadata = {
        "algorithm": (
            "existing deterministic Hangul-heavy one-case-per-document order, "
            "then exact 128-byte token-boundary filter for both sealed BPEs"
        ),
        "bpe_boundary_eligible_cases": boundary_eligible,
        "candidate_document_cases": CASE_POOL_SIZE,
        "continuation_array_sha256": array_sha256(continuations),
        "document_assignment_sha256": documents.metadata()[
            "document_assignment_sha256"
        ],
        "measured_cases": MEASURED_CASES,
        "prompt_array_sha256": array_sha256(prompts),
        "selected_cases": len(prompts),
        "stream_sha256": hashlib.sha256(stream.data).hexdigest(),
        "token_counts": {
            str(vocabulary_size): {
                "continuation_maximum": max(value[1] for value in rows),
                "continuation_median": float(np.median([value[1] for value in rows])),
                "continuation_minimum": min(value[1] for value in rows),
                "prompt_maximum": max(value[0] for value in rows),
                "prompt_median": float(np.median([value[0] for value in rows])),
                "prompt_minimum": min(value[0] for value in rows),
            }
            for vocabulary_size, rows in token_counts.items()
        },
        "warmup_cases": WARMUP_CASES,
    }
    return prompts, continuations, metadata


def encode_bpe_case(
    raw_prompt: bytes,
    raw_continuation: bytes,
    tokenizer: Tokenizer,
    token_bytes: Sequence[bytes],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    result = _bpe_ids_and_boundary(
        tokenizer,
        token_bytes,
        raw_prompt + raw_continuation,
    )
    if result is None:
        raise ValueError("sealed runtime case lost its BPE boundary")
    ids, split = result
    return ids[:split], ids[split:]


def role_schedule(prompt_index: int, repetition: int) -> tuple[str, ...]:
    if prompt_index < 0 or repetition < 0:
        raise ValueError("runtime schedule indices must be nonnegative")
    roles = list(RUNTIME_ROLES)
    offset = (prompt_index * REPETITIONS + repetition) % len(roles)
    return tuple(roles[offset:] + roles[:offset])


def schedule_sha256() -> str:
    schedule = [
        role_schedule(prompt, repetition)
        for prompt in range(MEASURED_CASES)
        for repetition in range(REPETITIONS)
    ]
    return canonical_sha256({"schedule": schedule})


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "benchmark",
        "cases",
        "claim_boundary",
        "decision_rule",
        "dependencies",
        "graphs",
        "implementation_sha256",
        "kind",
        "protocol_id",
        "schema_version",
        "status",
    }
    if set(plan) != expected:
        raise ValueError("scalar runtime plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "scalar_runtime_preflight_plan_v1"
        or plan["protocol_id"] != PROTOCOL_ID
        or plan["status"] != "sealed_before_runtime_measurement"
    ):
        raise ValueError("scalar runtime plan identity differs")
    if tuple(plan["benchmark"]["roles"]) != RUNTIME_ROLES:
        raise ValueError("scalar runtime role order differs")
    if plan["benchmark"] != {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "continuation_raw_bytes": CONTINUATION_BYTES,
        "correctness_cases": WARMUP_CASES,
        "correctness_tolerance": {
            "atol": MPS_ATOL,
            "normalized_worst_error_maximum": 1.0,
            "rtol": MPS_RTOL,
        },
        "device": "mps",
        "measured_cases": MEASURED_CASES,
        "mode": "controlled_fixed_route_sampling",
        "model_seed": 20_260_814,
        "prompt_raw_bytes": PROMPT_BYTES,
        "repetitions": REPETITIONS,
        "roles": list(RUNTIME_ROLES),
        "schedule_sha256": schedule_sha256(),
        "warmup_cases": WARMUP_CASES,
    }:
        raise ValueError("scalar runtime benchmark contract differs")
    if plan["cases"]["prompt_array_sha256"] != reconstruct_cases()[2]["prompt_array_sha256"]:
        raise ValueError("scalar runtime prompt cases differ")
    if plan["cases"] != reconstruct_cases()[2]:
        raise ValueError("scalar runtime case metadata differs")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("scalar runtime implementation set differs")
    historical = OUTPUT_PATH.exists()
    base_commit = plan["dependencies"]["plan_base_git_commit"]
    for relative in IMPLEMENTATION_PATHS:
        observed = (
            hash_git_blob(base_commit, relative)
            if historical
            else hash_file(ROOT / relative)
        )
        if observed != plan["implementation_sha256"][relative]:
            raise ValueError(f"scalar runtime implementation changed: {relative}")
    expected_graphs = {
        "byte_w72": (PARAMETER_TARGET, "sealed W72 BLT, 72 causal-whitespace-grid patches"),
        "generic_unicode_scalar": (
            19_632_960,
            "W72 BLT backbone plus 3x64 conditional UTF-8 heads",
        ),
        "hangul_hybrid": (
            19_609_152,
            "W72 BLT backbone plus 19/21/28 conditional LVT heads",
        ),
        "byte_bpe_32000": (19_593_984, dict(BPE_PRIMARY_SPEC)),
        "byte_bpe_16000": (19_595_200, dict(BPE_SECONDARY_SPEC)),
    }
    if set(plan["graphs"]) != set(expected_graphs):
        raise ValueError("scalar runtime graph role set differs")
    for role, (count, specification) in expected_graphs.items():
        graph = plan["graphs"][role]
        if (
            set(graph)
            != {
                "parameter_count",
                "relative_parameter_difference",
                "specification",
            }
            or graph["parameter_count"] != count
            or graph["specification"] != specification
            or not np.isclose(
                graph["relative_parameter_difference"],
                count / PARAMETER_TARGET - 1.0,
                rtol=0,
                atol=1e-15,
            )
            or abs(graph["relative_parameter_difference"])
            > PARAMETER_RELATIVE_TOLERANCE
        ):
            raise ValueError(f"scalar runtime graph differs: {role}")
    if plan["decision_rule"] != {
        "bpe_competitive_minimum_bootstrap_lower_reduction": -0.10,
        "byte_minimum_bootstrap_lower_reduction": 0.0,
        "byte_minimum_median_reduction": 0.10,
        "byte_minimum_positive_prompts": 28,
        "hangul_specific_maximum_lower_bound_slowdown_vs_generic": 0.05,
        "maximum_relative_parameter_difference": PARAMETER_RELATIVE_TOLERANCE,
        "requires_all_correctness_checks": True,
        "training_authorized_if_any_scalar_candidate_passes": True,
    }:
        raise ValueError("scalar runtime decision rule differs")
    if plan["claim_boundary"] != {
        "actual_mps_wall_time": True,
        "calibration_development_cases": True,
        "controlled_target_route_not_free_generation": True,
        "matched_quality_evidence": False,
        "random_weights_only": True,
        "training_or_model_loss_read": False,
        "publication_speed_claim": False,
        "tokenization_and_unit_encoding_outside_timing": True,
    }:
        raise ValueError("scalar runtime claim boundary differs")
    dependencies = plan["dependencies"]
    if set(dependencies) != {
        "integrity_artifact_sha256",
        "opportunity_artifact_sha256",
        "opportunity_summary_sha256",
        "plan_base_git_commit",
        "plan_payload_sha256",
        "source_artifact_sha256",
        "tokenizer_artifact_sha256",
    }:
        raise ValueError("scalar runtime dependency schema differs")
    if (
        hash_file(SOURCE_PATH) != dependencies["source_artifact_sha256"]
        or hash_file(INTEGRITY_PATH) != dependencies["integrity_artifact_sha256"]
        or hash_file(OPPORTUNITY_PATH) != dependencies["opportunity_artifact_sha256"]
        or {
            str(size): hash_file(path) for size, path in TOKENIZER_PATHS.items()
        }
        != dependencies["tokenizer_artifact_sha256"]
    ):
        raise ValueError("scalar runtime dependency artifact differs")
    opportunity = read_json(OPPORTUNITY_PATH)
    if opportunity.get("summary_sha256") != dependencies["opportunity_summary_sha256"]:
        raise ValueError("scalar runtime opportunity identity differs")
    if (
        not isinstance(dependencies["plan_base_git_commit"], str)
        or len(dependencies["plan_base_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in dependencies["plan_base_git_commit"]
        )
    ):
        raise ValueError("scalar runtime plan base commit differs")
    unsigned = dict(plan)
    unsigned_dependencies = dict(dependencies)
    unsigned_dependencies.pop("plan_payload_sha256")
    unsigned["dependencies"] = unsigned_dependencies
    if canonical_sha256(unsigned) != dependencies["plan_payload_sha256"]:
        raise ValueError("scalar runtime plan payload hash differs")


def comparison_summary(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    result = paired_prompt_latency(
        candidate,
        reference,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    ).to_dict()
    candidate_by_prompt = np.median(candidate, axis=1)
    reference_by_prompt = np.median(reference, axis=1)
    result["positive_prompt_count"] = int(
        np.sum(candidate_by_prompt < reference_by_prompt)
    )
    return result
