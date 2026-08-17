"""Shared exact MLX runtime for EXAONE baseline and retrieval generation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import mlx.core as mx
import numpy as np
from exaone_retrieval_data import (
    CASE_ARRAY_NAMES,
    CASES_PATH,
    PRIMARY_MODEL,
    TABLE_ARRAY_NAMES,
    TABLE_PATH,
    CompactBackoffTable,
    _load_npz,
    hash_file,
    hybrid_retrieval_proposal,
    read_validated_compatibility_result,
    read_verification,
    table_from_arrays,
)
from huggingface_hub import snapshot_download
from large_model_retrieval_preflight import MODEL_ALLOW_PATTERNS, token_sequence_sha256
from mlx_lm import load
from mlx_lm.utils import get_total_parameters
from mlx_retrieval_runtime import forced_speculative_generate, greedy_generate


@dataclass(frozen=True, slots=True)
class ExaoneRuntimeBundle:
    model: Any
    tokenizer: Any
    table: CompactBackoffTable | None
    model_parameter_count: int
    model_files: dict[str, dict[str, Any]]
    table_resident_bytes: int


@dataclass(frozen=True, slots=True)
class GenerationTrial:
    decoded_utf8_sha256: str
    elapsed_ns: int
    output_token_ids: tuple[int, ...]
    output_token_sha256: str
    prompt_token_count: int
    target_generation_forward_calls: int
    target_prefill_forward_calls: int
    corpus_proposal_calls: int
    prompt_proposal_calls: int
    no_proposal_calls: int
    proposed_tokens: int
    full_accept_cycles: int
    immediate_reject_cycles: int
    partial_accept_cycles: int


def _model_file_manifest(snapshot: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in PRIMARY_MODEL["expected_files"]:
        path = snapshot / name
        if not path.is_file() or path.is_symlink() and not path.resolve().is_file():
            raise ValueError(f"EXAONE model file is missing: {name}")
        output[name] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    return output


def _config_projection(config: dict[str, Any]) -> dict[str, Any]:
    projection = {key: config.get(key) for key in PRIMARY_MODEL["config_projection"]}
    if projection != PRIMARY_MODEL["config_projection"]:
        raise ValueError("EXAONE loaded config differs")
    return projection


def load_exaone_runtime(*, load_table: bool) -> ExaoneRuntimeBundle:
    read_verification()
    compatibility = read_validated_compatibility_result()
    snapshot = Path(
        snapshot_download(
            repo_id=PRIMARY_MODEL["repo_id"],
            revision=PRIMARY_MODEL["revision"],
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
            local_files_only=True,
        )
    )
    model_files = _model_file_manifest(snapshot)
    if model_files != compatibility["model_files"]:
        raise ValueError("EXAONE runtime files differ from compatibility result")
    mx.clear_cache()
    model, tokenizer, config = load(
        str(snapshot),
        lazy=False,
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _config_projection(config)
    parameter_count = int(get_total_parameters(model))
    expected_parameters = int(compatibility["memory"]["model_parameters"])
    if parameter_count != expected_parameters:
        raise ValueError("EXAONE runtime parameter count differs")
    if int(tokenizer.vocab_size) != PRIMARY_MODEL["config_projection"]["vocab_size"]:
        raise ValueError("EXAONE runtime tokenizer vocabulary differs")
    table = None
    table_bytes = 0
    if load_table:
        table_arrays = _load_npz(TABLE_PATH, TABLE_ARRAY_NAMES)
        table = table_from_arrays(table_arrays)
        table_bytes = sum(int(value.nbytes) for value in table_arrays.values())
    return ExaoneRuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        table=table,
        model_parameter_count=parameter_count,
        model_files=model_files,
        table_resident_bytes=table_bytes,
    )


def load_case_arrays() -> dict[str, np.ndarray]:
    read_verification()
    return _load_npz(CASES_PATH, CASE_ARRAY_NAMES)


def _prompt_text(tokenizer: Any, prompt_ids: Sequence[int]) -> str:
    expected = tuple(int(value) for value in prompt_ids)
    text = tokenizer.decode(
        expected,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if tuple(tokenizer.encode(text, add_special_tokens=False)) != expected:
        raise ValueError("EXAONE timed prompt round trip differs")
    return text


def _validate_decoded_sequence(
    tokenizer: Any, prompt_ids: Sequence[int], output_ids: Sequence[int], decoded: str
) -> str:
    expected = tuple(int(value) for value in prompt_ids) + tuple(
        int(value) for value in output_ids
    )
    replay = tokenizer.decode(
        expected,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if replay != decoded:
        raise ValueError("EXAONE generated detokenization is not deterministic")
    try:
        encoded = decoded.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("EXAONE generated text is not strict UTF-8") from error
    return hashlib.sha256(encoded).hexdigest()


def run_baseline_trial(
    bundle: ExaoneRuntimeBundle,
    prompt_ids: Sequence[int],
    *,
    output_tokens: int,
) -> GenerationTrial:
    if output_tokens <= 0:
        raise ValueError("EXAONE output token count differs")
    prompt_text = _prompt_text(bundle.tokenizer, prompt_ids)
    mx.synchronize()
    started = perf_counter_ns()
    encoded_prompt = tuple(
        int(value)
        for value in bundle.tokenizer.encode(prompt_text, add_special_tokens=False)
    )
    output = greedy_generate(bundle.model, encoded_prompt, maximum_tokens=output_tokens)
    decoded = bundle.tokenizer.decode(
        encoded_prompt + output,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    mx.synchronize()
    stopped = perf_counter_ns()
    decoded_sha256 = _validate_decoded_sequence(
        bundle.tokenizer, encoded_prompt, output, decoded
    )
    return GenerationTrial(
        decoded_utf8_sha256=decoded_sha256,
        elapsed_ns=stopped - started,
        output_token_ids=output,
        output_token_sha256=token_sequence_sha256(output),
        prompt_token_count=len(encoded_prompt),
        target_generation_forward_calls=output_tokens,
        target_prefill_forward_calls=1 if len(encoded_prompt) > 1 else 0,
        corpus_proposal_calls=0,
        prompt_proposal_calls=0,
        no_proposal_calls=0,
        proposed_tokens=0,
        full_accept_cycles=0,
        immediate_reject_cycles=0,
        partial_accept_cycles=0,
    )


def run_candidate_trial(
    bundle: ExaoneRuntimeBundle,
    prompt_ids: Sequence[int],
    *,
    output_tokens: int,
    maximum_draft_tokens: int,
) -> GenerationTrial:
    if bundle.table is None:
        raise ValueError("EXAONE candidate requires the sealed retrieval table")
    if output_tokens <= 0 or maximum_draft_tokens <= 0:
        raise ValueError("EXAONE candidate token limits differ")
    prompt_text = _prompt_text(bundle.tokenizer, prompt_ids)
    proposal_calls = {"corpus_ngram": 0, "prompt_lookup": 0, "none": 0}
    proposed_tokens = 0

    def provider(
        history: tuple[int, ...], remaining: int, emitted: int
    ) -> tuple[int, ...]:
        del emitted
        nonlocal proposed_tokens
        proposal, source = hybrid_retrieval_proposal(bundle.table, history)
        proposal = proposal[: min(maximum_draft_tokens, remaining)]
        proposal_calls[source] += 1
        proposed_tokens += len(proposal)
        return proposal

    mx.synchronize()
    started = perf_counter_ns()
    encoded_prompt = tuple(
        int(value)
        for value in bundle.tokenizer.encode(prompt_text, add_special_tokens=False)
    )
    trace = forced_speculative_generate(
        bundle.model,
        encoded_prompt,
        maximum_tokens=output_tokens,
        maximum_draft_tokens=maximum_draft_tokens,
        proposal_provider=provider,
    )
    decoded = bundle.tokenizer.decode(
        encoded_prompt + trace.token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    mx.synchronize()
    stopped = perf_counter_ns()
    decoded_sha256 = _validate_decoded_sequence(
        bundle.tokenizer, encoded_prompt, trace.token_ids, decoded
    )
    return GenerationTrial(
        decoded_utf8_sha256=decoded_sha256,
        elapsed_ns=stopped - started,
        output_token_ids=trace.token_ids,
        output_token_sha256=token_sequence_sha256(trace.token_ids),
        prompt_token_count=len(encoded_prompt),
        target_generation_forward_calls=trace.target_forward_calls,
        target_prefill_forward_calls=1 if len(encoded_prompt) > 1 else 0,
        corpus_proposal_calls=proposal_calls["corpus_ngram"],
        prompt_proposal_calls=proposal_calls["prompt_lookup"],
        no_proposal_calls=proposal_calls["none"],
        proposed_tokens=proposed_tokens,
        full_accept_cycles=trace.full_accept_cycles,
        immediate_reject_cycles=trace.immediate_reject_cycles,
        partial_accept_cycles=trace.partial_accept_cycles,
    )
