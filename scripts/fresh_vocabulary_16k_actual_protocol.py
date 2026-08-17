"""Sealed inputs for the fresh-v2 trained 16K actual-inference preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from compositional_head_core import build_model
from compositional_head_preflight_protocol import (
    ROOT,
    current_environment,
    hash_file,
    load_tokenizers,
    tokenizer_identity,
)
from fresh_vocabulary_16k_actual_core import (
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_END_TO_END_POINT_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    PAIR_ORDER,
    PAIR_ROLES,
    PROTOCOL_ID,
    REPETITIONS,
    ROLES,
    SECONDARY_MINIMUM_POSITIVE_PROMPTS,
)
from fresh_vocabulary_16k_core import (
    BASE_VOCABULARY_SIZE,
    REPLICATION_VOCABULARY_SIZE,
    TARGET_VOCABULARY_SIZE,
    build_canonical_decomposition_table,
    build_transferred_model,
    expected_parameter_count,
)
from fresh_vocabulary_16k_protocol import (
    CALIBRATION_BYTES,
    FRESH_SEAL_PATH,
    FRESH_SOURCE_PATH,
    canonical_sha256,
    read_json,
    verified_fresh_streams,
)
from fresh_vocabulary_16k_protocol import (
    CHECKPOINT_ROOT as QUALITY_CHECKPOINT_ROOT,
)
from fresh_vocabulary_16k_protocol import (
    OUTPUT_PATH as QUALITY_RESULT_PATH,
)
from fresh_vocabulary_16k_protocol import (
    PLAN_PATH as QUALITY_PLAN_PATH,
)
from fresh_vocabulary_16k_protocol import (
    json_bytes as _quality_json_bytes,
)
from fresh_vocabulary_actual_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    PROMPT_BYTES,
    WARMUP_CASES,
)
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import state_mapping_sha256
from vocabulary_transfer_probe_protocol import base_checkpoint_state

from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.inference_benchmark import select_inference_cases
from jamoflow.phase1 import stream_arrays
from jamoflow.utf8 import compile_strict_utf8_token_transitions

PLAN_PATH = ROOT / "data/manifests/fresh-vocabulary-16k-actual-one-seed-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/fresh-vocabulary-16k-actual-one-seed-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
RUNTIME_REPORT_PATH = ARTIFACT_ROOT / "runtime-report.json"
TIMING_PATH = ARTIFACT_ROOT / "timing.npz"
OUTPUT_PATH = ROOT / "results/fresh-vocabulary-16k-actual-one-seed-v1/summary.json"

QUALITY_ROLE_BY_ACTUAL_ROLE = {
    "candidate_16k": "dense16k_update_geometry",
    "baseline_2k": "dense2k_joint_v2",
    "frontier_8k": "dense8k_update_geometry_v2",
}
VOCABULARY_BY_ROLE = {
    "candidate_16k": TARGET_VOCABULARY_SIZE,
    "baseline_2k": BASE_VOCABULARY_SIZE,
    "frontier_8k": REPLICATION_VOCABULARY_SIZE,
}
CHECKPOINT_BY_ROLE = {
    role: QUALITY_CHECKPOINT_ROOT / f"{quality_role}.pt"
    for role, quality_role in QUALITY_ROLE_BY_ACTUAL_ROLE.items()
}

MPS_ATOL = 1e-4
MPS_RTOL = 2e-5
MAXIMUM_FREE_TOKENS = CONTINUATION_BYTES + 3
TIMED_SCOPE = (
    "raw-prompt UTF-8 decode and tokenizer encode; runtime/cache construction; "
    "parallel prefill; every autoregressive argmax and device-host token readback; "
    "incremental KV-cache decode; token-byte reconstruction; strict UTF-8 stop/decode"
)


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return _quality_json_bytes(value)


def experiment_contract() -> dict[str, Any]:
    return {
        "roles": list(ROLES),
        "modes": list(MODES),
        "prompt_bytes": PROMPT_BYTES,
        "continuation_bytes": CONTINUATION_BYTES,
        "warmup_cases": WARMUP_CASES,
        "measured_cases": MEASURED_CASES,
        "repetitions": REPETITIONS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_minimum_end_to_end_point_reduction": MINIMUM_END_TO_END_POINT_REDUCTION,
        "primary_minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
        "primary_minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
        "secondary_minimum_positive_prompts": SECONDARY_MINIMUM_POSITIVE_PROMPTS,
        "mps_atol": MPS_ATOL,
        "mps_rtol": MPS_RTOL,
        "maximum_free_tokens": MAXIMUM_FREE_TOKENS,
        "timed_scope": TIMED_SCOPE,
        "checkpoint_loading_inside_timing": False,
        "all_three_models_resident_during_timing": True,
        "measured_free_greedy_traces_independently_regenerated": True,
    }


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "development_one_seed_one_session": True,
        "trained_models": True,
        "actual_end_to_end_measured": True,
        "primary_requires_both_modes_at_least_10pct": True,
        "secondary_frontier_is_mandatory_diagnostic": True,
        "parameter_and_checkpoint_cost_reported": True,
        "memory_improvement_claimed": False,
        "publication_claim": False,
        "multiseed_requires_primary_gate": True,
    }


IMPLEMENTATION_PATHS = (
    "docs/155-fresh-vocabulary-trained-actual-preflight-protocol.md",
    "docs/156-fresh-vocabulary-actual-result-and-16k-pivot.md",
    "docs/159-fresh-vocabulary-16k-quality-protocol.md",
    "docs/160-fresh-v2-16k-quality-result.md",
    "docs/161-fresh-vocabulary-16k-trained-actual-protocol.md",
    "pyproject.toml",
    "scripts/bpe_quality_frontier_core.py",
    "scripts/benchmark_fresh_vocabulary_16k_actual.py",
    "scripts/benchmark_fresh_vocabulary_actual.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/compositional_quality_core.py",
    "scripts/compositional_quality_protocol.py",
    "scripts/compositional_token_head.py",
    "scripts/fresh_vocabulary_16k_actual_core.py",
    "scripts/fresh_vocabulary_16k_actual_protocol.py",
    "scripts/fresh_vocabulary_16k_core.py",
    "scripts/fresh_vocabulary_16k_protocol.py",
    "scripts/fresh_vocabulary_actual_core.py",
    "scripts/fresh_vocabulary_actual_protocol.py",
    "scripts/hplt3_fresh_adaptation_v2_protocol.py",
    "scripts/hplt3_fresh_adaptation_protocol.py",
    "scripts/preflight_fresh_vocabulary_16k_actual.py",
    "scripts/scalar_runtime_core.py",
    "scripts/scalar_representation_core.py",
    "scripts/seal_fresh_vocabulary_16k_actual_plan.py",
    "scripts/summarize_fresh_vocabulary_16k_actual.py",
    "scripts/token_frontier_core.py",
    "scripts/token_frontier_protocol.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "scripts/vocabulary_transfer_probe_protocol.py",
    "src/jamoflow/actual_inference_protocol.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/hplt3.py",
    "src/jamoflow/hplt3_final_test.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/incremental_blt.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/phase2_patching.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/publication_bpe.py",
    "src/jamoflow/utf8.py",
    "tests/test_fresh_vocabulary_16k_actual_core.py",
    "tests/test_fresh_vocabulary_16k_actual_protocol.py",
)


def read_plan_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"fresh-16k actual JSON root differs: {path}")
    return value


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("fresh-16k actual implementation list is duplicated")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def quality_result() -> dict[str, Any]:
    result = read_json(QUALITY_RESULT_PATH)
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256", None)
    decision = result.get("decision", {})
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "fresh_vocabulary_16k_quality_one_seed_result_v1"
        or result.get("status") != "pass_16k_quality_for_actual_preflight"
        or decision.get("actual_inference_preflight_authorized") is not True
        or decision.get("authorized_actual_role")
        != QUALITY_ROLE_BY_ACTUAL_ROLE["candidate_16k"]
        or decision.get("stronger_observed_anchor")
        != QUALITY_ROLE_BY_ACTUAL_ROLE["frontier_8k"]
        or result.get("independent_nll_recomputation", {}).get("pass") is not True
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("fresh-16k actual quality result differs")
    return result


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "fresh_v2_seal": FRESH_SEAL_PATH,
        "fresh_v2_source": FRESH_SOURCE_PATH,
        "quality_plan": QUALITY_PLAN_PATH,
        "quality_result": QUALITY_RESULT_PATH,
        **{
            f"{role}_checkpoint": checkpoint
            for role, checkpoint in CHECKPOINT_BY_ROLE.items()
        },
    }
    return {
        name: {"path": str(path.relative_to(ROOT)), "sha256": hash_file(path)}
        for name, path in paths.items()
    }


def reconstruct_cases() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stream = verified_fresh_streams()["calibration"]
    inputs, boundaries = stream_arrays(stream.data, stream.codepoint_boundaries, 512)
    documents = reconstruct_document_window_map(
        FRESH_SOURCE_PATH,
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=512,
        expected_stream=stream.data,
    )
    eligible = documents.document_indices >= 0
    selected = select_inference_cases(
        inputs[eligible],
        boundaries[eligible],
        cluster_ids=documents.document_indices[eligible],
        case_count=WARMUP_CASES + MEASURED_CASES,
        prompt_length=PROMPT_BYTES,
        continuation_length=CONTINUATION_BYTES,
    )
    prompts = selected.prompts.astype(np.uint8, copy=False)
    continuations = selected.replay_continuations.astype(np.uint8, copy=False)
    for row in np.concatenate((prompts, continuations), axis=0):
        bytes(row).decode("utf-8", errors="strict")
    metadata = {
        "algorithm": "outcome-independent one-case-per-document Hangul-heavy selector",
        "calibration_stream_sha256": hashlib.sha256(stream.data).hexdigest(),
        "document_assignment_sha256": documents.metadata()[
            "document_assignment_sha256"
        ],
        "prompt_array_sha256": array_sha256(prompts),
        "continuation_array_sha256": array_sha256(continuations),
        "warmup_cases": WARMUP_CASES,
        "measured_cases": MEASURED_CASES,
        **selected.public_metadata(),
    }
    return prompts, continuations, metadata


def encode_raw(
    raw: bytes,
    tokenizer: Any,
    token_bytes: Sequence[bytes],
) -> tuple[int, ...]:
    text = raw.decode("utf-8", errors="strict")
    ids = tuple(
        int(value) for value in tokenizer.encode(text, add_special_tokens=False).ids
    )
    if (
        not ids
        or tokenizer.decode(list(ids)) != text
        or b"".join(token_bytes[value] for value in ids) != raw
    ):
        raise ValueError("fresh-16k actual tokenizer roundtrip differs")
    return ids


def tokenizer_runtime_contract(
    prompts: np.ndarray,
    continuations: np.ndarray,
) -> dict[str, Any]:
    loaded = load_tokenizers()
    output: dict[str, Any] = {}
    for role in ROLES:
        size = VOCABULARY_BY_ROLE[role]
        tokenizer, token_bytes = loaded[size]
        prompt_lengths = np.asarray(
            [len(encode_raw(bytes(row), tokenizer, token_bytes)) for row in prompts],
            dtype=np.int16,
        )
        continuation_lengths = np.asarray(
            [
                len(encode_raw(bytes(row), tokenizer, token_bytes))
                for row in continuations
            ],
            dtype=np.int16,
        )
        transitions = compile_strict_utf8_token_transitions(token_bytes)
        output[role] = {
            "vocabulary_size": size,
            "prompt_token_count": {
                "minimum": int(prompt_lengths.min()),
                "median": float(np.median(prompt_lengths)),
                "maximum": int(prompt_lengths.max()),
                "array_sha256": array_sha256(prompt_lengths),
            },
            "continuation_token_count": {
                "minimum": int(continuation_lengths.min()),
                "median": float(np.median(continuation_lengths)),
                "maximum": int(continuation_lengths.max()),
                "array_sha256": array_sha256(continuation_lengths),
            },
            "strict_utf8_transitions": {
                "state_count": len(transitions.states),
                "maximum_token_bytes": transitions.maximum_token_bytes,
                "maximum_free_output_bytes": CONTINUATION_BYTES
                + transitions.maximum_token_bytes
                - 1,
                "token_bytes_sha256": transitions.token_bytes_sha256,
                "transition_table_sha256": transitions.transition_table_sha256,
            },
        }
    return output


def build_role_model(role: str) -> Any:
    size = VOCABULARY_BY_ROLE.get(role)
    if size is None:
        raise ValueError("fresh-16k actual model role differs")
    if size == BASE_VOCABULARY_SIZE:
        return build_model("dense_v2048")
    tokenizers = load_tokenizers()
    base_tokenizer, base_pieces = tokenizers[BASE_VOCABULARY_SIZE]
    target_tokenizer, target_pieces = tokenizers[size]
    decompositions = build_canonical_decomposition_table(
        base_tokenizer,
        target_tokenizer,
        base_pieces,
        target_pieces,
    )
    model, _ = build_transferred_model(
        size,
        base_state=base_checkpoint_state(),
        base_pieces=base_pieces,
        target_pieces=target_pieces,
        decompositions=decompositions,
    )
    return model


def model_identity() -> dict[str, Any]:
    result = quality_result()
    output: dict[str, Any] = {}
    for role in ROLES:
        quality_role = QUALITY_ROLE_BY_ACTUAL_ROLE[role]
        checkpoint = CHECKPOINT_BY_ROLE[role]
        lineage = result["artifact_lineage"][quality_role]
        replay = result["independent_nll_recomputation"]["by_role"][quality_role]
        expected_artifact = lineage["checkpoint_sha256"]
        if (
            hash_file(checkpoint) != expected_artifact
            or replay["checkpoint_artifact_sha256"] != expected_artifact
        ):
            raise ValueError("fresh-16k actual checkpoint artifact differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_role_model(role)
        model.load_state_dict(state, strict=True)
        state_sha = state_mapping_sha256(model.state_dict())
        parameters = model_parameter_count(model)
        expected_metrics = result["metrics"][quality_role]
        size = VOCABULARY_BY_ROLE[role]
        if (
            state_sha != replay["checkpoint_state_sha256"]
            or parameters != expected_metrics["parameter_count"]
            or parameters != expected_parameter_count(size)
            or checkpoint.stat().st_size != expected_metrics["checkpoint_bytes"]
        ):
            raise ValueError("fresh-16k actual checkpoint state differs")
        output[role] = {
            "quality_role": quality_role,
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_artifact_sha256": expected_artifact,
            "checkpoint_state_sha256": state_sha,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "parameter_count": parameters,
            "vocabulary_size": size,
            "document_bpb": expected_metrics["document_bpb"],
        }
        del model, state
    return output


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if not isinstance(git_commit_before_plan, str) or len(git_commit_before_plan) != 40:
        raise ValueError("fresh-16k actual pre-plan commit differs")
    result = quality_result()
    prompts, continuations, cases = reconstruct_cases()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_actual_one_seed_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_after_16k_quality_before_actual_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "selection": {
            "quality_result_sha256": result["summary_sha256"],
            "candidate": QUALITY_ROLE_BY_ACTUAL_ROLE["candidate_16k"],
            "primary_reference": QUALITY_ROLE_BY_ACTUAL_ROLE["baseline_2k"],
            "secondary_reference": QUALITY_ROLE_BY_ACTUAL_ROLE["frontier_8k"],
            "selection_metric": "fresh_v2_calibration_document_bpb_only",
            "actual_latency_used_for_selection": False,
            "no_result_dependent_fallback": True,
        },
        "pairs": {
            "order": list(PAIR_ORDER),
            "roles": {key: list(PAIR_ROLES[key]) for key in PAIR_ORDER},
            "primary_gate_pair": "candidate_vs_2k",
            "secondary_frontier_pair": "candidate_vs_8k",
            "secondary_does_not_replace_primary": True,
        },
        "models": model_identity(),
        "tokenizers": {
            role: tokenizer_identity()[str(VOCABULARY_BY_ROLE[role])] for role in ROLES
        },
        "cases": cases,
        "tokenizer_runtime": tokenizer_runtime_contract(prompts, continuations),
        "experiment": experiment_contract(),
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "claim_boundary": claim_boundary_contract(),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected = {
        "schema_version",
        "kind",
        "protocol_id",
        "status",
        "git_commit_before_plan",
        "dependencies",
        "implementation_sha256",
        "environment",
        "selection",
        "pairs",
        "models",
        "tokenizers",
        "cases",
        "tokenizer_runtime",
        "experiment",
        "output_path",
        "claim_boundary",
        "plan_sha256",
    }
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("kind") != "fresh_vocabulary_16k_actual_one_seed_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_after_16k_quality_before_actual_timing"
        or canonical_sha256(unsigned) != recorded
        or plan.get("dependencies") != dependency_identity()
        or plan.get("implementation_sha256") != implementation_identity()
        or plan.get("environment") != current_environment()
        or plan.get("output_path") != str(OUTPUT_PATH.relative_to(ROOT))
    ):
        raise ValueError("fresh-16k actual plan identity differs")
    result = quality_result()
    if plan.get("selection") != {
        "quality_result_sha256": result["summary_sha256"],
        "candidate": QUALITY_ROLE_BY_ACTUAL_ROLE["candidate_16k"],
        "primary_reference": QUALITY_ROLE_BY_ACTUAL_ROLE["baseline_2k"],
        "secondary_reference": QUALITY_ROLE_BY_ACTUAL_ROLE["frontier_8k"],
        "selection_metric": "fresh_v2_calibration_document_bpb_only",
        "actual_latency_used_for_selection": False,
        "no_result_dependent_fallback": True,
    }:
        raise ValueError("fresh-16k actual selection differs")
    if plan.get("pairs") != {
        "order": list(PAIR_ORDER),
        "roles": {key: list(PAIR_ROLES[key]) for key in PAIR_ORDER},
        "primary_gate_pair": "candidate_vs_2k",
        "secondary_frontier_pair": "candidate_vs_8k",
        "secondary_does_not_replace_primary": True,
    }:
        raise ValueError("fresh-16k actual pair contract differs")
    expected_tokenizers = tokenizer_identity()
    if plan.get("tokenizers") != {
        role: expected_tokenizers[str(VOCABULARY_BY_ROLE[role])] for role in ROLES
    }:
        raise ValueError("fresh-16k actual tokenizer identity differs")
    if plan.get("experiment") != experiment_contract():
        raise ValueError("fresh-16k actual experiment differs")
    if plan.get("claim_boundary") != claim_boundary_contract():
        raise ValueError("fresh-16k actual claim boundary differs")
    if verify_derived:
        prompts, continuations, cases = reconstruct_cases()
        if (
            plan.get("cases") != cases
            or plan.get("models") != model_identity()
            or plan.get("tokenizer_runtime")
            != tokenizer_runtime_contract(prompts, continuations)
        ):
            raise ValueError("fresh-16k actual derived plan fields differ")
