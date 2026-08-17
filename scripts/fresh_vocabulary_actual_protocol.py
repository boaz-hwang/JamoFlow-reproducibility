"""Sealed inputs and reconstruction for trained vocabulary actual inference."""

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
from fresh_vocabulary_actual_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    CONTINUATION_BYTES,
    MEASURED_CASES,
    MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
    MINIMUM_END_TO_END_POINT_REDUCTION,
    MINIMUM_POSITIVE_PROMPTS,
    MODES,
    PROMPT_BYTES,
    PROTOCOL_ID,
    REPETITIONS,
    ROLES,
    WARMUP_CASES,
)
from fresh_vocabulary_adaptation_protocol import (
    CALIBRATION_BYTES,
    CHECKPOINT_ROOT,
    FRESH_SEAL_PATH,
    FRESH_SOURCE_PATH,
    verified_fresh_streams,
)
from fresh_vocabulary_adaptation_protocol import (
    PLAN_PATH as ADAPTATION_PLAN_PATH,
)
from scalar_runtime_core import model_parameter_count
from vocabulary_transfer_probe_core import (
    build_target_graph,
    state_mapping_sha256,
)

from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.inference_benchmark import select_inference_cases
from jamoflow.phase1 import stream_arrays
from jamoflow.utf8 import compile_strict_utf8_token_transitions

PLAN_PATH = ROOT / "data/manifests/fresh-vocabulary-actual-one-seed-v1.json"
ADAPTATION_RESULT_PATH = (
    ROOT / "results/fresh-vocabulary-adaptation-one-seed-v1/summary.json"
)
ARTIFACT_ROOT = ROOT / "artifacts/fresh-vocabulary-actual-one-seed-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
RUNTIME_REPORT_PATH = ARTIFACT_ROOT / "runtime-report.json"
TIMING_PATH = ARTIFACT_ROOT / "timing.npz"
OUTPUT_PATH = ROOT / "results/fresh-vocabulary-actual-one-seed-v1/summary.json"

CANDIDATE_ADAPTATION_ROLE = "dense8k_update_geometry"
REFERENCE_ADAPTATION_ROLE = "dense2k_joint"
VOCABULARY_BY_ROLE = {"candidate": 8_192, "reference": 2_048}
CHECKPOINT_BY_ROLE = {
    "candidate": CHECKPOINT_ROOT / f"{CANDIDATE_ADAPTATION_ROLE}.pt",
    "reference": CHECKPOINT_ROOT / f"{REFERENCE_ADAPTATION_ROLE}.pt",
}

MPS_ATOL = 1e-4
MPS_RTOL = 2e-5
# A vocabulary may choose one raw-byte token per step and open a four-byte
# scalar at byte 128, requiring at most three continuation-token steps.
MAXIMUM_FREE_TOKENS = CONTINUATION_BYTES + 3

IMPLEMENTATION_PATHS = (
    "docs/154-fresh-vocabulary-adaptation-one-seed-result.md",
    "docs/155-fresh-vocabulary-trained-actual-preflight-protocol.md",
    "pyproject.toml",
    "scripts/benchmark_fresh_vocabulary_actual.py",
    "scripts/compositional_head_core.py",
    "scripts/compositional_head_preflight_protocol.py",
    "scripts/fresh_vocabulary_actual_core.py",
    "scripts/fresh_vocabulary_actual_protocol.py",
    "scripts/fresh_vocabulary_adaptation_core.py",
    "scripts/fresh_vocabulary_adaptation_protocol.py",
    "scripts/scalar_runtime_core.py",
    "scripts/seal_fresh_vocabulary_actual_plan.py",
    "scripts/summarize_fresh_vocabulary_actual.py",
    "scripts/vocabulary_transfer_probe_core.py",
    "src/jamoflow/document_inference.py",
    "src/jamoflow/inference_actual_v5.py",
    "src/jamoflow/inference_benchmark.py",
    "src/jamoflow/inference_calibration_replay_v2.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/phase1.py",
    "src/jamoflow/publication_bpe.py",
    "src/jamoflow/utf8.py",
    "tests/test_fresh_vocabulary_actual_core.py",
    "tests/test_fresh_vocabulary_actual_protocol.py",
)


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
        raise TypeError(f"fresh actual JSON root differs: {path}")
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
        raise AssertionError("fresh actual implementation list is duplicated")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def _adaptation_result() -> dict[str, Any]:
    result = read_json(ADAPTATION_RESULT_PATH)
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256", None)
    if (
        result.get("schema_version") != 1
        or result.get("kind") != "fresh_vocabulary_adaptation_one_seed_result_v1"
        or result.get("status") != "optimizer_geometry_and_deployment_opportunity"
        or result.get("decision", {}).get("actual_inference_preflight_authorized") is not True
        or result.get("decision", {}).get("selected_dense8k_role_for_actual_preflight")
        != CANDIDATE_ADAPTATION_ROLE
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("fresh actual adaptation result differs")
    return result


def dependency_identity() -> dict[str, dict[str, str]]:
    paths = {
        "adaptation_plan": ADAPTATION_PLAN_PATH,
        "adaptation_result": ADAPTATION_RESULT_PATH,
        "fresh_seal": FRESH_SEAL_PATH,
        "fresh_source": FRESH_SOURCE_PATH,
        "candidate_checkpoint": CHECKPOINT_BY_ROLE["candidate"],
        "reference_checkpoint": CHECKPOINT_BY_ROLE["reference"],
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
        "document_assignment_sha256": documents.metadata()["document_assignment_sha256"],
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
    ids = tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False).ids)
    if (
        not ids
        or tokenizer.decode(list(ids)) != text
        or b"".join(token_bytes[value] for value in ids) != raw
    ):
        raise ValueError("fresh actual tokenizer roundtrip differs")
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
    if role == "reference":
        return build_model("dense_v2048")
    if role == "candidate":
        return build_target_graph("untied_uniform_in_byte_weighted_out")
    raise ValueError("fresh actual model role differs")


def model_identity() -> dict[str, Any]:
    result = _adaptation_result()
    output: dict[str, Any] = {}
    for role in ROLES:
        adaptation_role = (
            CANDIDATE_ADAPTATION_ROLE if role == "candidate" else REFERENCE_ADAPTATION_ROLE
        )
        checkpoint = CHECKPOINT_BY_ROLE[role]
        expected_artifact = result["artifact_lineage"][adaptation_role][
            "checkpoint_sha256"
        ]
        recomputation = result["independent_nll_recomputation"]["by_role"][
            adaptation_role
        ]
        if (
            hash_file(checkpoint) != expected_artifact
            or recomputation["checkpoint_artifact_sha256"] != expected_artifact
        ):
            raise ValueError("fresh actual checkpoint artifact differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_role_model(role)
        model.load_state_dict(state, strict=True)
        state_sha = state_mapping_sha256(model.state_dict())
        parameters = model_parameter_count(model)
        expected_metrics = result["metrics"][adaptation_role]
        if (
            state_sha != recomputation["checkpoint_state_sha256"]
            or parameters != expected_metrics["parameter_count"]
            or checkpoint.stat().st_size != expected_metrics["checkpoint_bytes"]
        ):
            raise ValueError("fresh actual checkpoint state differs")
        output[role] = {
            "adaptation_role": adaptation_role,
            "checkpoint_path": str(checkpoint.relative_to(ROOT)),
            "checkpoint_artifact_sha256": expected_artifact,
            "checkpoint_state_sha256": state_sha,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "parameter_count": parameters,
            "vocabulary_size": VOCABULARY_BY_ROLE[role],
        }
        del model, state
    return output


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if not isinstance(git_commit_before_plan, str) or len(git_commit_before_plan) != 40:
        raise ValueError("fresh actual pre-plan commit differs")
    result = _adaptation_result()
    prompts, continuations, cases = reconstruct_cases()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_actual_one_seed_plan_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_after_trained_quality_before_actual_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependency_identity(),
        "implementation_sha256": implementation_identity(),
        "environment": current_environment(),
        "selection": {
            "adaptation_result_sha256": result["summary_sha256"],
            "candidate": CANDIDATE_ADAPTATION_ROLE,
            "reference": REFERENCE_ADAPTATION_ROLE,
            "selection_metric": "fresh_calibration_document_bpb",
            "actual_latency_used_for_selection": False,
        },
        "models": model_identity(),
        "tokenizers": {
            key: tokenizer_identity()[str(VOCABULARY_BY_ROLE[key])] for key in ROLES
        },
        "cases": cases,
        "tokenizer_runtime": tokenizer_runtime_contract(prompts, continuations),
        "experiment": {
            "roles": list(ROLES),
            "modes": list(MODES),
            "prompt_bytes": PROMPT_BYTES,
            "continuation_bytes": CONTINUATION_BYTES,
            "warmup_cases": WARMUP_CASES,
            "measured_cases": MEASURED_CASES,
            "repetitions": REPETITIONS,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_end_to_end_point_reduction": MINIMUM_END_TO_END_POINT_REDUCTION,
            "minimum_bootstrap_lower_reduction": MINIMUM_BOOTSTRAP_LOWER_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "mps_atol": MPS_ATOL,
            "mps_rtol": MPS_RTOL,
            "maximum_free_tokens": MAXIMUM_FREE_TOKENS,
            "timed_scope": (
                "raw-prompt UTF-8 decode and tokenizer encode; runtime/cache construction; "
                "parallel prefill; every autoregressive argmax and device-host token readback; "
                "incremental KV-cache decode; token-byte reconstruction; strict UTF-8 stop/decode"
            ),
            "checkpoint_loading_inside_timing": False,
        },
        "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
        "claim_boundary": {
            "development_one_seed_one_session": True,
            "trained_models": True,
            "actual_end_to_end_measured": True,
            "requires_both_modes_at_least_10pct": True,
            "parameter_and_checkpoint_cost_reported": True,
            "memory_improvement_claimed": False,
            "publication_claim": False,
            "multiseed_requires_this_gate": True,
        },
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
        or plan.get("kind") != "fresh_vocabulary_actual_one_seed_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status") != "sealed_after_trained_quality_before_actual_timing"
        or canonical_sha256(unsigned) != recorded
        or plan.get("dependencies") != dependency_identity()
        or plan.get("implementation_sha256") != implementation_identity()
        or plan.get("environment") != current_environment()
        or plan.get("output_path") != str(OUTPUT_PATH.relative_to(ROOT))
    ):
        raise ValueError("fresh actual plan identity differs")
    result = _adaptation_result()
    if plan.get("selection") != {
        "adaptation_result_sha256": result["summary_sha256"],
        "candidate": CANDIDATE_ADAPTATION_ROLE,
        "reference": REFERENCE_ADAPTATION_ROLE,
        "selection_metric": "fresh_calibration_document_bpb",
        "actual_latency_used_for_selection": False,
    }:
        raise ValueError("fresh actual selection differs")
    expected_tokenizers = tokenizer_identity()
    if plan.get("tokenizers") != {
        role: expected_tokenizers[str(VOCABULARY_BY_ROLE[role])] for role in ROLES
    }:
        raise ValueError("fresh actual tokenizer identity differs")
    experiment = plan.get("experiment", {})
    if (
        experiment.get("roles") != list(ROLES)
        or experiment.get("modes") != list(MODES)
        or experiment.get("prompt_bytes") != PROMPT_BYTES
        or experiment.get("continuation_bytes") != CONTINUATION_BYTES
        or experiment.get("warmup_cases") != WARMUP_CASES
        or experiment.get("measured_cases") != MEASURED_CASES
        or experiment.get("repetitions") != REPETITIONS
        or experiment.get("bootstrap_repetitions") != BOOTSTRAP_REPETITIONS
        or experiment.get("bootstrap_seed") != BOOTSTRAP_SEED
        or experiment.get("minimum_end_to_end_point_reduction")
        != MINIMUM_END_TO_END_POINT_REDUCTION
        or experiment.get("minimum_bootstrap_lower_reduction")
        != MINIMUM_BOOTSTRAP_LOWER_REDUCTION
        or experiment.get("minimum_positive_prompts") != MINIMUM_POSITIVE_PROMPTS
        or experiment.get("mps_atol") != MPS_ATOL
        or experiment.get("mps_rtol") != MPS_RTOL
        or experiment.get("maximum_free_tokens") != MAXIMUM_FREE_TOKENS
    ):
        raise ValueError("fresh actual experiment differs")
    if verify_derived:
        prompts, continuations, cases = reconstruct_cases()
        if (
            plan.get("cases") != cases
            or plan.get("models") != model_identity()
            or plan.get("tokenizer_runtime")
            != tokenizer_runtime_contract(prompts, continuations)
        ):
            raise ValueError("fresh actual derived plan fields differ")
