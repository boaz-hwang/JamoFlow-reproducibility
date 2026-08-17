"""Pure contracts and statistics for the post-100M schedule extrapolation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jamoflow.corpus import load_records, partition_records
from jamoflow.hplt3 import hash_file
from jamoflow.incremental_blt import structural_prefix_boundaries
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import Phase1ModelSpec

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "jamoflow-scale-schedule-extrapolation-v1"
PLAN_PATH = ROOT / "data/manifests/scale-schedule-extrapolation-v1.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
PROMPT_SOURCE_PATH = ROOT / "artifacts/hangul-draft-acceptance-v1/free-target.npz"
ARTIFACT_ROOT = ROOT / "artifacts/scale-schedule-extrapolation-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
OUTPUT_PATH = ROOT / "results/scale-schedule-extrapolation-v1/summary.json"

TARGET_ORDER = (200, 400, 800, 1600)
SESSION_ORDER = ("session-0", "session-1", "session-2")
SCHEDULE_ORDER = ("c86", "w72")
REFERENCE_INDEX = SCHEDULE_ORDER.index("c86")
CANDIDATE_INDEX = SCHEDULE_ORDER.index("w72")
MODEL_SEED = 20260816
PROMPT_BYTES = 128
CONTINUATION_BYTES = 128
WARMUP_PROMPTS = 4
MEASURED_PROMPTS = 16
INNER_REPETITIONS = 3
CORRECTNESS_PROMPTS = 4
GLOBAL_POSITION_LIMIT = 1032
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20260817
PRIMARY_TARGET = 1600
MINIMUM_POINT_REDUCTION = 0.10
MINIMUM_BOOTSTRAP_LOWER_BOUND = 0.08
MINIMUM_POSITIVE_PROMPTS = 15
MINIMUM_POSITIVE_SESSIONS = len(SESSION_ORDER)
MINIMUM_SESSIONS_AT_POINT_TARGET = 2
RTOL = 1e-4
ATOL = 2e-5
MAXIMUM_RECOMMENDED_MEMORY_FRACTION = 0.75

LARGE_SCALE_SPECS = {
    200: Phase1ModelSpec(
        sequence_length=512,
        patch_count=86,
        patch_stride=6,
        local_width=448,
        global_width=896,
        local_heads=14,
        global_heads=14,
        encoder_layers=2,
        global_layers=16,
        decoder_layers=2,
        local_ffn=1344,
        global_ffn=2688,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=448,
        router_heads=14,
        router_layers=4,
        router_ffn=1344,
    ),
    400: Phase1ModelSpec(
        sequence_length=512,
        patch_count=86,
        patch_stride=6,
        local_width=576,
        global_width=1152,
        local_heads=18,
        global_heads=18,
        encoder_layers=2,
        global_layers=20,
        decoder_layers=2,
        local_ffn=1728,
        global_ffn=3456,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=576,
        router_heads=18,
        router_layers=4,
        router_ffn=1728,
    ),
    800: Phase1ModelSpec(
        sequence_length=512,
        patch_count=86,
        patch_stride=6,
        local_width=768,
        global_width=1536,
        local_heads=24,
        global_heads=24,
        encoder_layers=2,
        global_layers=24,
        decoder_layers=2,
        local_ffn=2304,
        global_ffn=4608,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=768,
        router_heads=24,
        router_layers=4,
        router_ffn=2304,
    ),
    1600: Phase1ModelSpec(
        sequence_length=512,
        patch_count=86,
        patch_stride=6,
        local_width=1024,
        global_width=2048,
        local_heads=32,
        global_heads=32,
        encoder_layers=2,
        global_layers=28,
        decoder_layers=2,
        local_ffn=3072,
        global_ffn=6144,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=1024,
        router_heads=32,
        router_layers=4,
        router_ffn=3072,
    ),
}
EXPECTED_PARAMETERS = {
    200: 188_639_808,
    400: 378_058_176,
    800: 790_449_408,
    1600: 1_617_558_528,
}

IMPLEMENTATION_PATHS = (
    "docs/187-scale-schedule-preflight-result-and-terminal-research-decision.md",
    "docs/189-scale-schedule-extrapolation-protocol.md",
    "pyproject.toml",
    "scripts/run_scale_schedule_extrapolation.py",
    "scripts/scale_schedule_extrapolation_core.py",
    "scripts/seal_scale_schedule_extrapolation_plan.py",
    "scripts/verify_scale_schedule_extrapolation.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/patching.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/publication_protocol.py",
    "src/jamoflow/publication_reference.py",
    "src/jamoflow/utf8.py",
    "tests/test_scale_schedule_extrapolation.py",
)


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def is_git_commit(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def schedule_contract() -> dict[str, Any]:
    return {
        "order": list(SCHEDULE_ORDER),
        "horizon": 512,
        "fixed_stride": 6,
        "c86": {
            "patch_count": 86,
            "policy": "causal_codepoint_grid",
        },
        "w72": {
            "patch_count": 72,
            "policy": "causal_whitespace_grid",
        },
    }


def model_contract() -> dict[str, Any]:
    return {
        str(target): {
            "expected_parameter_count": EXPECTED_PARAMETERS[target],
            "spec": large_scale_model_spec(target, 86).to_dict(),
        }
        for target in TARGET_ORDER
    }


def large_scale_model_spec(target_millions: int, patch_count: int) -> Phase1ModelSpec:
    if target_millions not in LARGE_SCALE_SPECS:
        raise ValueError("unknown large scale target")
    if not 1 < patch_count <= 512:
        raise ValueError("large scale patch count is invalid")
    return replace(LARGE_SCALE_SPECS[target_millions], patch_count=patch_count)


def role_order(
    target_index: int,
    session_index: int,
    prompt_index: int,
    repetition: int,
) -> tuple[int, int]:
    if (
        not 0 <= target_index < len(TARGET_ORDER)
        or not 0 <= session_index < len(SESSION_ORDER)
        or not 0 <= prompt_index < MEASURED_PROMPTS
        or not 0 <= repetition < INNER_REPETITIONS
    ):
        raise ValueError("scale-schedule role-order coordinate differs")
    first = (target_index + session_index + prompt_index + repetition) % 2
    return (first, 1 - first)


def _worker_identity(target: int, session: str) -> tuple[int, int]:
    if target not in TARGET_ORDER or session not in SESSION_ORDER:
        raise ValueError("scale-schedule worker identity differs")
    return TARGET_ORDER.index(target), SESSION_ORDER.index(session)


def worker_timing_path(target: int, session: str) -> Path:
    _worker_identity(target, session)
    return ARTIFACT_ROOT / f"target-{target}-{session}-timings.npz"


def worker_report_path(target: int, session: str) -> Path:
    _worker_identity(target, session)
    return ARTIFACT_ROOT / f"target-{target}-{session}-report.json"


def _document_spans(expected_stream: bytes) -> tuple[tuple[int, int, int], ...]:
    records = partition_records(
        load_records(
            [SOURCE_PATH],
            corpus_format="jsonl",
            text_field="text",
            deduplicate=True,
        )
    )["calibration"]
    buffer = bytearray()
    spans: list[tuple[int, int, int]] = []
    first = True
    for document_index, record in enumerate(records):
        if record.text is None:
            continue
        separator = b"" if first else b"\n"
        first = False
        if len(buffer) >= 1_000_000:
            break
        remaining = 1_000_000 - len(buffer)
        chunk_start = len(buffer)
        selected = (separator + record.raw)[:remaining]
        buffer.extend(selected)
        separator_bytes = min(len(separator), len(selected))
        raw_bytes = len(selected) - separator_bytes
        if raw_bytes > 0:
            spans.append(
                (
                    chunk_start + separator_bytes,
                    chunk_start + separator_bytes + raw_bytes,
                    document_index,
                )
            )
    usable = len(buffer) - len(buffer) % 512
    if bytes(buffer[:usable]) != expected_stream:
        raise ValueError("scale-schedule document layout differs from the stream")
    return tuple(
        (start, min(end, usable), document)
        for start, end, document in spans
        if start < min(end, usable)
    )


def _select_independent_case_indices(
    offsets: np.ndarray,
    spans: Sequence[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[int] = []
    selected_documents: list[int] = []
    observed_spans: list[tuple[int, int]] = []
    required = WARMUP_PROMPTS + MEASURED_PROMPTS
    for pool_index, raw_offset in enumerate(offsets):
        start = int(raw_offset)
        end = start + PROMPT_BYTES + CONTINUATION_BYTES - 1
        documents = [
            document
            for span_start, span_end, document in spans
            if span_start <= start and end <= span_end
        ]
        if len(documents) != 1 or documents[0] in selected_documents:
            continue
        if any(
            not (end <= previous_start or previous_end <= start)
            for previous_start, previous_end in observed_spans
        ):
            continue
        selected.append(pool_index)
        selected_documents.append(documents[0])
        observed_spans.append((start, end))
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError("scale-schedule lacks enough document-independent cases")
    return (
        np.asarray(selected, dtype=np.int64),
        np.asarray(selected_documents, dtype=np.int64),
    )


def load_case_arrays() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Reconstruct the fixed warmup+measured controlled byte cases."""

    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=1_000_000,
        sequence_length=512,
    )
    total = WARMUP_PROMPTS + MEASURED_PROMPTS
    with np.load(PROMPT_SOURCE_PATH, allow_pickle=False) as source:
        if set(source.files) != {
            "generated_sha256",
            "hidden",
            "lead",
            "output_offset",
            "prompt_index",
            "prompt_offsets",
            "prompts",
            "second",
            "target_is_hangul",
            "third",
        }:
            raise ValueError("scale-schedule prompt source schema differs")
        pool_prompts = np.ascontiguousarray(source["prompts"])
        pool_offsets = np.ascontiguousarray(source["prompt_offsets"])
    if pool_prompts.dtype != np.uint8 or pool_prompts.shape != (128, PROMPT_BYTES):
        raise ValueError("scale-schedule prompt pool differs")
    if pool_offsets.dtype != np.int64 or pool_offsets.shape != (128,):
        raise ValueError("scale-schedule offset pool differs")
    if len(np.unique(pool_offsets)) != len(pool_offsets):
        raise ValueError("scale-schedule prompt offsets are not unique")
    for prompt, raw_offset in zip(pool_prompts, pool_offsets, strict=True):
        offset = int(raw_offset)
        if not 0 <= offset <= len(stream.data) - PROMPT_BYTES or not np.array_equal(
            prompt,
            np.frombuffer(stream.data[offset : offset + PROMPT_BYTES], dtype=np.uint8),
        ):
            raise ValueError("scale-schedule prompt is not its source prefix")
        text = bytes(prompt).decode("utf-8", errors="strict")
        hangul_share = sum(0xAC00 <= ord(char) <= 0xD7A3 for char in text) / max(
            1, len(text)
        )
        if hangul_share < 0.79:
            raise ValueError("scale-schedule prompt is not Hangul-heavy")
    spans = _document_spans(stream.data)
    pool_indices, document_indices = _select_independent_case_indices(
        pool_offsets, spans
    )
    prompts = np.ascontiguousarray(pool_prompts[pool_indices])
    offsets = np.ascontiguousarray(pool_offsets[pool_indices])
    continuations = np.stack(
        [
            np.frombuffer(
                stream.data[
                    int(offset) + PROMPT_BYTES : int(offset)
                    + PROMPT_BYTES
                    + CONTINUATION_BYTES
                ],
                dtype=np.uint8,
            )
            for offset in offsets
        ]
    )
    continuations = np.ascontiguousarray(continuations)
    if continuations.shape != (total, CONTINUATION_BYTES):
        raise ValueError("scale-schedule continuation array differs")
    return prompts, continuations, offsets, pool_indices, document_indices


def _boundary_digest(boundaries: Sequence[int]) -> bytes:
    values = np.asarray(tuple(boundaries), dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(b"JamoFlow/scale-schedule-boundaries/v1\0")
    digest.update(np.asarray([len(values)], dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.digest()


def mechanism_arrays(
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return schedule patch counts and boundary commitments for fixed cases."""

    prompt_array = np.asarray(prompts)
    continuation_array = np.asarray(continuations)
    if (
        prompt_array.dtype != np.uint8
        or continuation_array.dtype != np.uint8
        or prompt_array.ndim != 2
        or continuation_array.ndim != 2
        or prompt_array.shape[0] != continuation_array.shape[0]
        or prompt_array.shape[1] != PROMPT_BYTES
        or continuation_array.shape[1] != CONTINUATION_BYTES
    ):
        raise ValueError("scale-schedule mechanism cases differ")
    rows = prompt_array.shape[0]
    counts = np.empty((rows, len(SCHEDULE_ORDER)), dtype=np.int64)
    hashes = np.empty((rows, len(SCHEDULE_ORDER), 32), dtype=np.uint8)
    contract = schedule_contract()
    for case in range(rows):
        observed = bytes(prompt_array[case]) + bytes(continuation_array[case][:-1])
        for schedule_index, schedule in enumerate(SCHEDULE_ORDER):
            row = contract[schedule]
            boundaries = structural_prefix_boundaries(
                observed,
                row["policy"],
                horizon=contract["horizon"],
                patch_count=row["patch_count"],
                fixed_stride=contract["fixed_stride"],
            )
            counts[case, schedule_index] = len(boundaries)
            hashes[case, schedule_index] = np.frombuffer(
                _boundary_digest(boundaries), dtype=np.uint8
            )
    return counts, hashes


def case_contract() -> dict[str, Any]:
    prompts, continuations, offsets, pool_indices, document_indices = load_case_arrays()
    patch_counts, boundary_hashes = mechanism_arrays(prompts, continuations)
    return {
        "source_path": SOURCE_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": hash_file(SOURCE_PATH),
        "prompt_source_path": PROMPT_SOURCE_PATH.relative_to(ROOT).as_posix(),
        "prompt_source_sha256": hash_file(PROMPT_SOURCE_PATH),
        "selection": "preexisting pool order; first 20 full 255-byte windows from distinct documents",
        "pool_prompt_count": 128,
        "warmup_prompts": WARMUP_PROMPTS,
        "measured_prompts": MEASURED_PROMPTS,
        "prompt_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "prompts_array_sha256": array_sha256(prompts),
        "continuations_array_sha256": array_sha256(continuations),
        "offsets_array_sha256": array_sha256(offsets),
        "selected_pool_indices": pool_indices.tolist(),
        "document_indices_array_sha256": array_sha256(document_indices),
        "distinct_document_count": len(np.unique(document_indices)),
        "minimum_prompt_hangul_share": 0.79,
        "patch_counts_array_sha256": array_sha256(patch_counts),
        "boundary_hashes_array_sha256": array_sha256(boundary_hashes),
    }


def validate_case_arrays(plan: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if plan.get("cases") != case_contract():
        raise ValueError("scale-schedule case contract differs")
    prompts, continuations, _, _, _ = load_case_arrays()
    return prompts, continuations


def build_scale_schedule_plan(
    *,
    git_commit_before_plan: str,
    models: Mapping[str, Any],
    environment: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": "scale_schedule_extrapolation_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_first_scale_schedule_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "cases": case_contract(),
        "schedules": schedule_contract(),
        "models": dict(models),
        "environment": dict(environment),
        "timing": {
            "atol": ATOL,
            "controlled_observed_bytes": PROMPT_BYTES + CONTINUATION_BYTES - 1,
            "correctness_prompts": CORRECTNESS_PROMPTS,
            "device": "mps",
            "global_position_limit": GLOBAL_POSITION_LIMIT,
            "inner_repetitions": INNER_REPETITIONS,
            "model_seed": MODEL_SEED,
            "role_order": "(target_index + session_index + prompt_index + repetition) mod 2",
            "rtol": RTOL,
            "session_order": list(SESSION_ORDER),
            "scope": "fresh runtime, parallel prefill, 127 controlled consume calls, final MPS synchronize",
            "target_order": list(TARGET_ORDER),
        },
        "gate": {
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed_base": BOOTSTRAP_SEED,
            "maximum_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
            "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
            "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "minimum_positive_sessions": MINIMUM_POSITIVE_SESSIONS,
            "minimum_sessions_at_point_target": MINIMUM_SESSIONS_AT_POINT_TARGET,
            "primary_target_millions": PRIMARY_TARGET,
            "stop_rule": "1600M primary failure rejects a 10-percent scale-crossing claim; no target fallback",
        },
        "implementation_sha256": dict(implementation_sha256),
        "outputs": {
            "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
            "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
            "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        },
        "threat_model": {
            "case_pool_selected_after_compact_w72_result": True,
            "case_pool_preexisting_before_exaone_actual_result": True,
            "case_subset_selected_after_exaone_actual_result": True,
            "case_subset_selection_uses_model_output_or_scale_timing": False,
            "confirmatory_or_final_claimed": False,
            "document_independence_filter_added_before_scale_timing": True,
            "quality_evidence_from_random_weights": False,
            "retrieval_failure_rescued": False,
            "prior_50m_to_100m_scale_timing_known_before_plan": True,
            "new_200m_to_1600m_timing_observed_before_plan": False,
            "timing_used_for_trained_quality_claim": False,
            "larger_scale_training_directly_authorized": False,
        },
    }
    plan = {**payload, "plan_sha256": canonical_sha256(payload)}
    validate_plan(
        plan,
        current_environment=environment,
        verify_implementation=False,
    )
    return plan


def validate_plan(
    value: Mapping[str, Any],
    *,
    current_environment: Mapping[str, Any] | None = None,
    verify_implementation: bool = True,
) -> None:
    expected_keys = {
        "cases",
        "environment",
        "gate",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "models",
        "outputs",
        "plan_sha256",
        "protocol_id",
        "schedules",
        "schema_version",
        "status",
        "threat_model",
        "timing",
    }
    if set(value) != expected_keys:
        raise ValueError("scale-schedule plan schema differs")
    claimed = value["plan_sha256"]
    payload = dict(value)
    payload.pop("plan_sha256")
    if (
        value["schema_version"] != 1
        or value["kind"] != "scale_schedule_extrapolation_plan_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or value["status"] != "sealed_before_first_scale_schedule_timing"
        or not is_git_commit(value["git_commit_before_plan"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
    ):
        raise ValueError("scale-schedule plan identity differs")
    if value["cases"] != case_contract():
        raise ValueError("scale-schedule plan cases differ")
    if value["schedules"] != schedule_contract():
        raise ValueError("scale-schedule plan schedules differ")
    models = value["models"]
    if not isinstance(models, Mapping) or set(models) != {
        str(target) for target in TARGET_ORDER
    }:
        raise ValueError("scale-schedule plan model targets differ")
    expected_models = model_contract()
    for target in TARGET_ORDER:
        row = models[str(target)]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"expected_parameter_count", "model_state_sha256", "spec"}
            or row["expected_parameter_count"]
            != expected_models[str(target)]["expected_parameter_count"]
            or row["spec"] != expected_models[str(target)]["spec"]
            or not is_sha256(row["model_state_sha256"])
        ):
            raise ValueError(f"scale-schedule model contract differs: {target}")
    if current_environment is not None and value["environment"] != dict(
        current_environment
    ):
        raise ValueError("scale-schedule runtime environment differs")
    if value["timing"] != {
        "atol": ATOL,
        "controlled_observed_bytes": PROMPT_BYTES + CONTINUATION_BYTES - 1,
        "correctness_prompts": CORRECTNESS_PROMPTS,
        "device": "mps",
        "global_position_limit": GLOBAL_POSITION_LIMIT,
        "inner_repetitions": INNER_REPETITIONS,
        "model_seed": MODEL_SEED,
        "role_order": "(target_index + session_index + prompt_index + repetition) mod 2",
        "rtol": RTOL,
        "session_order": list(SESSION_ORDER),
        "scope": "fresh runtime, parallel prefill, 127 controlled consume calls, final MPS synchronize",
        "target_order": list(TARGET_ORDER),
    }:
        raise ValueError("scale-schedule timing contract differs")
    if value["gate"] != {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "maximum_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
        "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
        "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
        "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "minimum_positive_sessions": MINIMUM_POSITIVE_SESSIONS,
        "minimum_sessions_at_point_target": MINIMUM_SESSIONS_AT_POINT_TARGET,
        "primary_target_millions": PRIMARY_TARGET,
        "stop_rule": "1600M primary failure rejects a 10-percent scale-crossing claim; no target fallback",
    }:
        raise ValueError("scale-schedule gate contract differs")
    if value["outputs"] != {
        "active_path": ACTIVE_PATH.relative_to(ROOT).as_posix(),
        "artifact_root": ARTIFACT_ROOT.relative_to(ROOT).as_posix(),
        "summary_path": OUTPUT_PATH.relative_to(ROOT).as_posix(),
    }:
        raise ValueError("scale-schedule output contract differs")
    if value["threat_model"] != {
        "case_pool_selected_after_compact_w72_result": True,
        "case_pool_preexisting_before_exaone_actual_result": True,
        "case_subset_selected_after_exaone_actual_result": True,
        "case_subset_selection_uses_model_output_or_scale_timing": False,
        "confirmatory_or_final_claimed": False,
        "document_independence_filter_added_before_scale_timing": True,
        "quality_evidence_from_random_weights": False,
        "retrieval_failure_rescued": False,
        "prior_50m_to_100m_scale_timing_known_before_plan": True,
        "new_200m_to_1600m_timing_observed_before_plan": False,
        "timing_used_for_trained_quality_claim": False,
        "larger_scale_training_directly_authorized": False,
    }:
        raise ValueError("scale-schedule threat model differs")
    implementation = value["implementation_sha256"]
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or any(not is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
    ):
        raise ValueError("scale-schedule implementation file set differs")
    if verify_implementation:
        for relative in IMPLEMENTATION_PATHS:
            path = ROOT / relative
            if (
                not path.is_file()
                or path.is_symlink()
                or hash_file(path) != implementation[relative]
            ):
                raise ValueError(f"scale-schedule implementation differs: {relative}")


def _correctness_pass(value: Mapping[str, Any]) -> bool:
    expected_keys = {
        "argmax_comparisons",
        "argmax_exact",
        "boundary_prefix_comparisons",
        "boundary_trace_exact",
        "cache_diagnostics_exact",
        "maximum_normalized_logit_error",
        "offline_boundary_prefix_exact",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        return False
    comparisons = CORRECTNESS_PROMPTS * CONTINUATION_BYTES
    maximum = value["maximum_normalized_logit_error"]
    return bool(
        value["argmax_comparisons"] == comparisons
        and value["argmax_exact"] == comparisons
        and value["boundary_prefix_comparisons"] == comparisons
        and value["boundary_trace_exact"] is True
        and value["cache_diagnostics_exact"] is True
        and value["offline_boundary_prefix_exact"] is True
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and math.isfinite(float(maximum))
        and 0 <= float(maximum) <= 1
    )


def _validate_timing_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64 or array.shape != (
        len(SESSION_ORDER),
        MEASURED_PROMPTS,
        INNER_REPETITIONS,
        len(SCHEDULE_ORDER),
    ):
        raise ValueError("scale-schedule timing array shape/dtype differs")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("scale-schedule timings must be finite and positive")
    return array


def _reduction(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.median(reference))
    if denominator <= 0:
        raise ValueError("scale-schedule reference timing is nonpositive")
    return 1.0 - float(np.median(candidate)) / denominator


def _bootstrap_interval(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    target: int,
) -> tuple[float, float]:
    if candidate.shape != (len(SESSION_ORDER), MEASURED_PROMPTS) or (
        reference.shape != candidate.shape
    ):
        raise ValueError("scale-schedule bootstrap cells differ")
    rng = np.random.default_rng(BOOTSTRAP_SEED + target)
    values = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sessions = rng.integers(0, len(SESSION_ORDER), size=len(SESSION_ORDER))
        prompts = rng.integers(0, MEASURED_PROMPTS, size=MEASURED_PROMPTS)
        cells = np.ix_(sessions, prompts)
        values[index] = _reduction(candidate[cells], reference[cells])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_scale_schedule_extrapolation(
    *,
    timings_by_target: Mapping[int, np.ndarray],
    reports_by_target: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize all fixed targets without selecting a favorable scale."""

    if set(timings_by_target) != set(TARGET_ORDER):
        raise ValueError("scale-schedule timing targets differ")
    if set(reports_by_target) != set(TARGET_ORDER):
        raise ValueError("scale-schedule report targets differ")
    rows: dict[str, Any] = {}
    evidence_valid_by_target: dict[int, bool] = {}

    for target in TARGET_ORDER:
        timings = _validate_timing_array(timings_by_target[target])
        reports = tuple(reports_by_target[target])
        if len(reports) != len(SESSION_ORDER):
            raise ValueError(f"scale-schedule report session count differs: {target}")
        expected_report_keys = {
            "correctness",
            "environment_end",
            "environment_start",
            "maximum_driver_allocated_bytes",
            "model_state_sha256",
            "parameter_count",
            "patch_count_summary",
            "recommended_max_memory_bytes",
            "same_model_object_for_both_schedules",
            "session_id",
            "target_millions",
        }
        correctness_rows: dict[str, Any] = {}
        memory_fractions: list[float | None] = []
        state_hashes: list[str] = []
        mechanism_rows: list[Mapping[str, Any]] = []
        environment_starts: list[Mapping[str, Any]] = []
        session_validity: dict[str, bool] = {}
        for session, report in zip(SESSION_ORDER, reports, strict=True):
            if set(report) != expected_report_keys:
                raise ValueError(
                    f"scale-schedule report schema differs: {target}/{session}"
                )
            correctness = report["correctness"]
            if not isinstance(correctness, Mapping) or set(correctness) != set(
                SCHEDULE_ORDER
            ):
                raise ValueError(
                    f"scale-schedule correctness roles differ: {target}/{session}"
                )
            correctness_pass = all(
                isinstance(correctness[name], Mapping)
                and _correctness_pass(correctness[name])
                for name in SCHEDULE_ORDER
            )
            parameter_pass = bool(
                report["target_millions"] == target
                and report["session_id"] == session
                and report["parameter_count"] == EXPECTED_PARAMETERS[target]
                and is_sha256(report["model_state_sha256"])
                and report["same_model_object_for_both_schedules"] is True
            )
            patch_summary = report["patch_count_summary"]
            mechanism_pass = bool(
                isinstance(patch_summary, Mapping)
                and set(patch_summary) == set(SCHEDULE_ORDER)
                and all(
                    isinstance(patch_summary[name], Mapping)
                    and set(patch_summary[name])
                    == {"maximum", "median", "minimum", "sum"}
                    and type(patch_summary[name]["maximum"]) is int
                    and type(patch_summary[name]["minimum"]) is int
                    and type(patch_summary[name]["sum"]) is int
                    and isinstance(patch_summary[name]["median"], (int, float))
                    and not isinstance(patch_summary[name]["median"], bool)
                    and 0
                    < patch_summary[name]["minimum"]
                    <= patch_summary[name]["median"]
                    <= patch_summary[name]["maximum"]
                    and patch_summary[name]["sum"] > 0
                    for name in SCHEDULE_ORDER
                )
                and patch_summary["w72"]["sum"] < patch_summary["c86"]["sum"]
            )
            maximum_driver = report["maximum_driver_allocated_bytes"]
            recommended = report["recommended_max_memory_bytes"]
            memory_pass = bool(
                type(maximum_driver) is int
                and type(recommended) is int
                and 0
                < maximum_driver
                <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION * recommended
            )
            environment_pass = bool(
                isinstance(report["environment_start"], Mapping)
                and report["environment_start"] == report["environment_end"]
            )
            session_validity[session] = bool(
                correctness_pass
                and parameter_pass
                and mechanism_pass
                and memory_pass
                and environment_pass
            )
            correctness_rows[session] = {
                name: dict(correctness[name]) for name in SCHEDULE_ORDER
            }
            memory_fractions.append(
                float(maximum_driver / recommended) if memory_pass else None
            )
            state_hashes.append(report["model_state_sha256"])
            mechanism_rows.append(patch_summary)
            environment_starts.append(report["environment_start"])
        identity_consistent = bool(
            len(set(state_hashes)) == 1
            and all(row == mechanism_rows[0] for row in mechanism_rows[1:])
            and all(
                environment == environment_starts[0]
                for environment in environment_starts[1:]
            )
        )
        evidence_valid = bool(all(session_validity.values()) and identity_consistent)
        evidence_valid_by_target[target] = evidence_valid

        prompt_points = np.median(timings, axis=2)
        reference = prompt_points[:, :, REFERENCE_INDEX]
        candidate = prompt_points[:, :, CANDIDATE_INDEX]
        reduction = _reduction(candidate, reference)
        lower, upper = _bootstrap_interval(candidate, reference, target=target)
        prompt_effects = 1.0 - np.median(candidate, axis=0) / np.median(
            reference, axis=0
        )
        positive = int(np.sum(prompt_effects > 0))
        session_reductions = np.asarray(
            [
                _reduction(candidate[index], reference[index])
                for index in range(len(SESSION_ORDER))
            ],
            dtype=np.float64,
        )
        positive_sessions = int(np.sum(session_reductions > 0))
        sessions_at_point_target = int(
            np.sum(session_reductions >= MINIMUM_POINT_REDUCTION)
        )
        median_session_reduction = float(np.median(session_reductions))
        patch_event_reduction = float(
            1 - mechanism_rows[0]["w72"]["sum"] / mechanism_rows[0]["c86"]["sum"]
        )
        gates = {
            "evidence_valid": evidence_valid,
            "point_reduction_at_least_10_percent": reduction >= MINIMUM_POINT_REDUCTION,
            "bootstrap_lower_at_least_8_percent": lower
            >= MINIMUM_BOOTSTRAP_LOWER_BOUND,
            "positive_prompts_at_least_15": positive >= MINIMUM_POSITIVE_PROMPTS,
            "all_three_sessions_positive": positive_sessions
            >= MINIMUM_POSITIVE_SESSIONS,
            "at_least_two_sessions_reach_10_percent": sessions_at_point_target
            >= MINIMUM_SESSIONS_AT_POINT_TARGET,
        }
        rows[str(target)] = {
            "c86_median_ms": float(np.median(reference)),
            "w72_median_ms": float(np.median(candidate)),
            "median_reduction": float(reduction),
            "prompt_bootstrap_95_interval": {
                "lower": lower,
                "upper": upper,
            },
            "positive_prompt_count": positive,
            "session_reductions": {
                session: float(session_reductions[index])
                for index, session in enumerate(SESSION_ORDER)
            },
            "positive_session_count": positive_sessions,
            "sessions_at_least_10_percent": sessions_at_point_target,
            "median_session_reduction": median_session_reduction,
            "maximum_memory_fraction": (
                max(value for value in memory_fractions if value is not None)
                if all(value is not None for value in memory_fractions)
                else None
            ),
            "patch_count_summary": dict(mechanism_rows[0]),
            "patch_event_reduction": patch_event_reduction,
            "affected_time_share_proxy": float(reduction / patch_event_reduction),
            "affected_time_share_proxy_interpretation": (
                "descriptive Amdahl ratio only: E2E reduction divided by fixed "
                "patch-event reduction; not a measured component-time share"
            ),
            "correctness_by_session": correctness_rows,
            "session_evidence_validity": session_validity,
            "identity_consistent_across_sessions": identity_consistent,
            "gates": gates,
            "overall_threshold_pass": bool(all(gates.values())),
        }

    all_evidence_valid = all(evidence_valid_by_target.values())
    primary_gates = rows[str(PRIMARY_TARGET)]["gates"]
    primary_pass = bool(all_evidence_valid and all(primary_gates.values()))
    ordered_reductions = [rows[str(target)]["median_reduction"] for target in TARGET_ORDER]
    return {
        "protocol_id": PROTOCOL_ID,
        "target_order": list(TARGET_ORDER),
        "schedule_order": list(SCHEDULE_ORDER),
        "bootstrap": {
            "unit": "crossed fresh-process session and prompt after within-cell repetition median",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed_rule": "BOOTSTRAP_SEED + target_millions",
            "base_seed": BOOTSTRAP_SEED,
        },
        "thresholds": {
            "primary_target_millions": PRIMARY_TARGET,
            "minimum_point_reduction": MINIMUM_POINT_REDUCTION,
            "minimum_bootstrap_lower_bound": MINIMUM_BOOTSTRAP_LOWER_BOUND,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "minimum_positive_sessions": MINIMUM_POSITIVE_SESSIONS,
            "minimum_sessions_at_point_target": MINIMUM_SESSIONS_AT_POINT_TARGET,
            "maximum_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
        },
        "rows": rows,
        "descriptive_scaling": {
            "median_reductions_in_target_order": ordered_reductions,
            "strictly_increasing": all(
                right > left
                for left, right in zip(
                    ordered_reductions, ordered_reductions[1:]
                )
            ),
            "scaling_law_fitted": False,
            "out_of_range_extrapolation_claimed": False,
        },
        "all_target_evidence_valid": all_evidence_valid,
        "primary_1600m_pass": primary_pass,
        "ten_percent_large_scale_headroom_detected": primary_pass,
        "larger_scale_training_directly_authorized": False,
        "status": (
            "large_scale_10_percent_headroom_detected"
            if primary_pass
            else "large_scale_10_percent_headroom_not_detected"
        ),
    }


def build_scale_schedule_summary(
    *,
    plan_artifact_sha256: str,
    plan_sha256: str,
    summary_base_git_commit: str,
    worker_evidence: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not is_sha256(plan_artifact_sha256)
        or not is_sha256(plan_sha256)
        or not is_git_commit(summary_base_git_commit)
        or aggregate.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("scale-schedule summary dependency differs")
    _validate_worker_evidence(worker_evidence)
    detected = aggregate.get("ten_percent_large_scale_headroom_detected")
    if not isinstance(detected, bool):
        raise TypeError("scale-schedule summary headroom decision differs")
    payload = {
        "schema_version": 1,
        "kind": "scale_schedule_extrapolation_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": aggregate["status"],
        "plan_artifact_sha256": plan_artifact_sha256,
        "plan_sha256": plan_sha256,
        "summary_base_git_commit": summary_base_git_commit,
        "worker_evidence": dict(worker_evidence),
        "aggregate": dict(aggregate),
        "claim_boundary": {
            "confirmatory_or_final_claimed": False,
            "quality_or_matched_quality_claimed": False,
            "random_weight_runtime_preflight": True,
            "retrieval_result_changed": False,
            "ten_percent_large_scale_headroom_detected": detected,
            "larger_scale_training_directly_authorized": False,
        },
    }
    return {**payload, "summary_sha256": canonical_sha256(payload)}


def validate_scale_schedule_summary(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "aggregate",
        "claim_boundary",
        "kind",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "schema_version",
        "status",
        "summary_base_git_commit",
        "summary_sha256",
        "worker_evidence",
    }
    if set(value) != expected_keys:
        raise ValueError("scale-schedule summary schema differs")
    claimed = value["summary_sha256"]
    payload = dict(value)
    payload.pop("summary_sha256")
    aggregate = value["aggregate"]
    if (
        value["schema_version"] != 1
        or value["kind"] != "scale_schedule_extrapolation_summary_v1"
        or value["protocol_id"] != PROTOCOL_ID
        or not is_sha256(value["plan_artifact_sha256"])
        or not is_sha256(value["plan_sha256"])
        or not is_git_commit(value["summary_base_git_commit"])
        or not is_sha256(claimed)
        or canonical_sha256(payload) != claimed
        or not isinstance(aggregate, Mapping)
        or aggregate.get("protocol_id") != PROTOCOL_ID
        or value["status"] != aggregate.get("status")
        or value["claim_boundary"]
        != {
            "confirmatory_or_final_claimed": False,
            "quality_or_matched_quality_claimed": False,
            "random_weight_runtime_preflight": True,
            "retrieval_result_changed": False,
            "ten_percent_large_scale_headroom_detected": aggregate.get(
                "ten_percent_large_scale_headroom_detected"
            ),
            "larger_scale_training_directly_authorized": False,
        }
    ):
        raise ValueError("scale-schedule summary identity differs")
    evidence = value["worker_evidence"]
    _validate_worker_evidence(evidence)


def _validate_worker_evidence(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        str(target) for target in TARGET_ORDER
    }:
        raise ValueError("scale-schedule summary worker evidence differs")
    for target in TARGET_ORDER:
        sessions = value[str(target)]
        if not isinstance(sessions, Mapping) or set(sessions) != set(SESSION_ORDER):
            raise ValueError("scale-schedule summary session evidence differs")
        for session in SESSION_ORDER:
            row = sessions[session]
            if (
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "report_path",
                    "report_sha256",
                    "timing_path",
                    "timing_sha256",
                }
                or row["report_path"]
                != worker_report_path(target, session).relative_to(ROOT).as_posix()
                or row["timing_path"]
                != worker_timing_path(target, session).relative_to(ROOT).as_posix()
                or not is_sha256(row["report_sha256"])
                or not is_sha256(row["timing_sha256"])
            ):
                raise ValueError("scale-schedule summary evidence identity differs")
