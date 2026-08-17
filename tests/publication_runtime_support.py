"""Synthetic, content-free publication runtime evidence for unit tests."""

from __future__ import annotations

import hashlib

import numpy as np

from jamoflow.inference_benchmark import timing_order_schedule
from jamoflow.publication_bpb import RAW_BYTE_TOKENIZER_SHA256
from jamoflow.publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_CANDIDATE_MODEL_KEY,
    PUBLICATION_PRETRAIN_SEEDS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
    PUBLICATION_RUNTIME_EQUIVALENCE_PATHS,
    PUBLICATION_RUNTIME_MEASURED_CASES,
    PUBLICATION_RUNTIME_MODES,
    PUBLICATION_RUNTIME_REPETITIONS,
    PUBLICATION_RUNTIME_ROLES,
    PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED,
    PUBLICATION_RUNTIME_TIMING_ORDER_SEED,
    PUBLICATION_RUNTIME_WARMUP_CASES,
)
from jamoflow.publication_reference import (
    PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
)
from jamoflow.publication_runtime import (
    PublicationRuntimeLineage,
    build_publication_runtime_equivalence,
    build_publication_runtime_evidence,
    build_publication_runtime_lineage,
    build_publication_timing_evidence,
    build_publication_valid_output_evidence,
    publication_output_diagnostic_keys,
    publication_timing_array_keys,
)
from tests.publication_reference_support import (
    make_reference_descriptor,
    make_router_bundles,
)


def content_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


TRIAL_ARTIFACT_SHA256 = content_hash("synthetic-trial-artifact")
OUTPUT_TRACE_ARTIFACT_SHA256 = content_hash("synthetic-output-trace-artifact")
OUTPUT_TRACE_AUDIT_SHA256 = content_hash("synthetic-output-trace-audit")


def output_evidence_hashes(
    *,
    trial_artifact_sha256: str = TRIAL_ARTIFACT_SHA256,
) -> dict[str, str]:
    return {
        "trial_artifact_sha256": trial_artifact_sha256,
        "output_trace_artifact_sha256": OUTPUT_TRACE_ARTIFACT_SHA256,
        "output_trace_audit_sha256": OUTPUT_TRACE_AUDIT_SHA256,
    }


def comparator_identity(
    family: str,
    comparator_key: str | None = None,
) -> str:
    if family == "raw_byte":
        return comparator_key or PUBLICATION_RAW_COMPARATOR_MODEL_KEY
    if family == "standard_bpe":
        return comparator_key or PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]
    raise ValueError("unknown synthetic comparator family")


def make_lineage(
    family: str = "raw_byte",
    *,
    comparator_key: str | None = None,
    candidate_variant: str = "shared",
    raw_reference_policy: str = "entropy_threshold_full",
) -> PublicationRuntimeLineage:
    resolved = comparator_identity(family, comparator_key)

    def seeded(label: str, *, identity: str) -> dict[int, str]:
        return {
            seed: content_hash(f"{label}:{identity}:{seed}")
            for seed in PUBLICATION_PRETRAIN_SEEDS
        }

    def constant(label: str, *, identity: str) -> dict[int, str]:
        value = content_hash(f"{label}:{identity}")
        return {seed: value for seed in PUBLICATION_PRETRAIN_SEEDS}

    descriptor = (
        make_reference_descriptor(raw_reference_policy)
        if family == "raw_byte"
        else None
    )
    auxiliary_bundles = (
        make_router_bundles(descriptor)
        if descriptor is not None and descriptor.requires_entropy_router
        else None
    )
    return build_publication_runtime_lineage(
        candidate_key=PUBLICATION_CANDIDATE_MODEL_KEY,
        comparator_key=resolved,
        comparator_family=family,
        candidate_checkpoint_sha256=seeded(
            "candidate-checkpoint",
            identity=candidate_variant,
        ),
        comparator_checkpoint_sha256=seeded(
            "comparator-checkpoint",
            identity=resolved,
        ),
        candidate_model_config_sha256=constant(
            "candidate-config",
            identity=candidate_variant,
        ),
        comparator_model_config_sha256=constant(
            "comparator-config",
            identity=resolved,
        ),
        raw_reference_descriptor=descriptor,
        comparator_auxiliary_bundles=auxiliary_bundles,
        candidate_tokenizer_sha256=RAW_BYTE_TOKENIZER_SHA256,
        comparator_tokenizer_sha256=(
            RAW_BYTE_TOKENIZER_SHA256
            if family == "raw_byte"
            else content_hash(f"comparator-tokenizer:{resolved}")
        ),
        candidate_utf8_transition_sha256=content_hash("raw-byte-utf8"),
        comparator_utf8_transition_sha256=(
            content_hash("raw-byte-utf8")
            if family == "raw_byte"
            else content_hash(f"comparator-utf8:{resolved}")
        ),
        runtime_source_sha256=content_hash("publication-runtime-source"),
        timing_scope_audit_sha256=content_hash("timing-scope-audit"),
        case_manifest_sha256=content_hash("case-manifest"),
        raw_prompt_array_sha256=content_hash("raw-prompts"),
        raw_replay_continuation_array_sha256=content_hash("raw-continuations"),
        candidate_prompt_unit_array_sha256=content_hash(
            f"candidate-prompt-units:{candidate_variant}"
        ),
        comparator_prompt_unit_array_sha256=content_hash(
            f"comparator-prompt-units:{resolved}"
        ),
        candidate_replay_unit_array_sha256=content_hash(
            f"candidate-replay-units:{candidate_variant}"
        ),
        comparator_replay_unit_array_sha256=content_hash(
            f"comparator-replay-units:{resolved}"
        ),
        unitization_audit_sha256=content_hash(
            f"unitization-audit:{resolved}:{candidate_variant}"
        ),
    )


def make_equivalence_comparisons(
    *,
    pass_equivalence: bool = True,
) -> dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]]:
    comparisons = {}
    for index, key in enumerate(
        (
            (seed, role, path)
            for seed in PUBLICATION_PRETRAIN_SEEDS
            for role in PUBLICATION_RUNTIME_ROLES
            for path in PUBLICATION_RUNTIME_EQUIVALENCE_PATHS
        )
    ):
        full = np.linspace(-2.0, 2.0, 16 * 7, dtype=np.float32).reshape(16, 7)
        full = full + np.float32(index / 100.0)
        incremental = full.copy()
        if not pass_equivalence and index == 0:
            incremental = incremental + np.float32(1.0)
        comparisons[key] = (full, incremental)
    return comparisons


def eligible_environment() -> dict[str, object]:
    return {
        "power": {
            "returncode": 0,
            "stdout": "Now drawing from 'AC Power'",
        },
        "thermal": {
            "returncode": 0,
            "stdout": (
                "No thermal warning level has been recorded\n"
                "No performance warning level has been recorded"
            ),
        },
        "settings": {
            "returncode": 0,
            "stdout": "AC Power:\n lowpowermode 0\n powermode 0",
        },
    }


def make_timing_inputs(
    *,
    candidate_decode_ms: float = 8.0,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    dict[int, dict[str, object]],
    str,
]:
    shape = (
        len(PUBLICATION_PRETRAIN_SEEDS),
        PUBLICATION_RUNTIME_MEASURED_CASES,
        PUBLICATION_RUNTIME_REPETITIONS,
    )
    arrays: dict[str, np.ndarray] = {}
    for mode in PUBLICATION_RUNTIME_MODES:
        for role in PUBLICATION_RUNTIME_ROLES:
            ttft = 1.0 if role == "candidate" else 1.0
            decode = candidate_decode_ms if role == "candidate" else 10.0
            arrays[f"{mode}__ttft_ms__{role}"] = np.full(
                shape,
                ttft,
                dtype=np.float64,
            )
            arrays[f"{mode}__decode_ms__{role}"] = np.full(
                shape,
                decode,
                dtype=np.float64,
            )
            arrays[f"{mode}__end_to_end_ms__{role}"] = np.full(
                shape,
                ttft + decode,
                dtype=np.float64,
            )
    if set(arrays) != set(publication_timing_array_keys()):
        raise AssertionError("synthetic timing fixture drifted")
    schedule = timing_order_schedule(
        PUBLICATION_PRETRAIN_SEEDS,
        mode_count=len(PUBLICATION_RUNTIME_MODES),
        prompt_count=PUBLICATION_RUNTIME_MEASURED_CASES,
        repetitions=PUBLICATION_RUNTIME_REPETITIONS,
        random_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED,
    )
    warmup_schedule = timing_order_schedule(
        PUBLICATION_PRETRAIN_SEEDS,
        mode_count=len(PUBLICATION_RUNTIME_MODES),
        prompt_count=PUBLICATION_RUNTIME_WARMUP_CASES,
        repetitions=1,
        random_seed=PUBLICATION_RUNTIME_TIMING_ORDER_SEED + 1,
    )
    warmup_completion = np.ones(
        (
            len(PUBLICATION_PRETRAIN_SEEDS),
            len(PUBLICATION_RUNTIME_MODES),
            PUBLICATION_RUNTIME_WARMUP_CASES,
            len(PUBLICATION_RUNTIME_ROLES),
        ),
        dtype=np.uint8,
    )
    seed_execution_order = tuple(
        PUBLICATION_PRETRAIN_SEEDS[index]
        for index in np.random.default_rng(
            PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED
        ).permutation(len(PUBLICATION_PRETRAIN_SEEDS))
    )
    environments = {
        seed: {
            "start": eligible_environment(),
            "end": eligible_environment(),
        }
        for seed in PUBLICATION_PRETRAIN_SEEDS
    }
    return (
        arrays,
        schedule,
        warmup_schedule,
        warmup_completion,
        seed_execution_order,
        environments,
        TRIAL_ARTIFACT_SHA256,
    )


def make_diagnostic_inputs(
    family: str,
    *,
    valid_output_rate: float = 1.0,
    replacement_free_rate: float = 1.0,
    bpe_reference_overshoot_bytes: int = 0,
    comparator_maximum_unit_bytes: int | None = None,
    entropy_reference: bool | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    if not 0 <= valid_output_rate <= 1 or not 0 <= replacement_free_rate <= 1:
        raise ValueError("synthetic rates must be fractions")
    bound = comparator_maximum_unit_bytes or (
        1 if family == "raw_byte" else max(2, bpe_reference_overshoot_bytes + 1)
    )
    if entropy_reference is None:
        entropy_reference = family == "raw_byte"
    shape = (
        len(PUBLICATION_PRETRAIN_SEEDS),
        PUBLICATION_RUNTIME_MEASURED_CASES,
        PUBLICATION_RUNTIME_REPETITIONS,
    )
    arrays = {
        key: np.zeros(shape, dtype=np.int64)
        for key in publication_output_diagnostic_keys()
    }
    complete_prompts = round(PUBLICATION_RUNTIME_MEASURED_CASES * valid_output_rate)
    replacement_prompts = round(
        PUBLICATION_RUNTIME_MEASURED_CASES * replacement_free_rate
    )
    for mode in PUBLICATION_RUNTIME_MODES:
        for role in PUBLICATION_RUNTIME_ROLES:
            exact_byte_units = role == "candidate" or family == "raw_byte"
            emitted_bytes = 128
            if (
                mode == "free_running_utf8_greedy"
                and role == "reference"
                and family == "standard_bpe"
            ):
                emitted_bytes += bpe_reference_overshoot_bytes
            emitted_units = (
                emitted_bytes
                if exact_byte_units
                else (emitted_bytes + bound - 1) // bound
            )
            prompt_units = 128 if exact_byte_units else 64
            values = {
                "prompt_model_units": prompt_units,
                "emitted_output_bytes": emitted_bytes,
                "emitted_model_units": emitted_units,
                "decode_forward_steps": emitted_units - 1,
                "runtime_observed_model_units": prompt_units + emitted_units - 1,
                "overshoot_bytes": (
                    0
                    if mode == "controlled_replay"
                    else emitted_bytes - 128
                ),
                "valid_output_stop": 1,
                "final_utf8_accept": 1,
                "transition_trace_valid": 1,
                "replacement_character_free": 1,
                "output_codepoints": max(1, emitted_bytes // 3),
                "router_observed_model_units": (
                    prompt_units + emitted_units - 1
                    if role == "reference" and entropy_reference
                    else 0
                ),
                "router_cached_model_units": (
                    prompt_units + emitted_units - 1
                    if role == "reference" and entropy_reference
                    else 0
                ),
                "router_scored_model_units": (
                    prompt_units + emitted_units - 1
                    if role == "reference" and entropy_reference
                    else 0
                ),
                "router_forward_calls": (
                    emitted_units
                    if role == "reference" and entropy_reference
                    else 0
                ),
            }
            for name, value in values.items():
                arrays[f"{mode}__{name}__{role}"].fill(value)
            if mode == "free_running_utf8_greedy":
                stopped = arrays[f"{mode}__valid_output_stop__{role}"]
                stopped[:, complete_prompts:, :] = 0
                replacement = arrays[
                    f"{mode}__replacement_character_free__{role}"
                ]
                replacement[:, replacement_prompts:, :] = 0
    return arrays, bound


def make_runtime_evidence(
    family: str = "raw_byte",
    *,
    comparator_key: str | None = None,
    candidate_variant: str = "shared",
    candidate_decode_ms: float = 8.0,
    pass_equivalence: bool = True,
    valid_output_rate: float = 1.0,
    replacement_free_rate: float = 1.0,
    bpe_reference_overshoot_bytes: int = 0,
    raw_reference_policy: str = "entropy_threshold_full",
):
    lineage = make_lineage(
        family,
        comparator_key=comparator_key,
        candidate_variant=candidate_variant,
        raw_reference_policy=raw_reference_policy,
    )
    equivalence = build_publication_runtime_equivalence(
        lineage,
        make_equivalence_comparisons(pass_equivalence=pass_equivalence),
    )
    timing_inputs = make_timing_inputs(candidate_decode_ms=candidate_decode_ms)
    timing = build_publication_timing_evidence(lineage, *timing_inputs)
    diagnostics, comparator_bound = make_diagnostic_inputs(
        family,
        valid_output_rate=valid_output_rate,
        replacement_free_rate=replacement_free_rate,
        bpe_reference_overshoot_bytes=bpe_reference_overshoot_bytes,
        entropy_reference=(
            lineage.comparator_auxiliary_kind
            == PUBLICATION_AUXILIARY_ENTROPY_ROUTER
        ),
    )
    valid_output = build_publication_valid_output_evidence(
        lineage,
        diagnostics,
        comparator_maximum_unit_bytes=comparator_bound,
        **output_evidence_hashes(),
    )
    return build_publication_runtime_evidence(
        lineage,
        equivalence,
        timing,
        valid_output,
    )
