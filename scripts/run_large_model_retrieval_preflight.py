#!/usr/bin/env python3
"""Run compatibility checks without measuring candidate or baseline latency."""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.utils import get_total_parameters

from large_model_retrieval_preflight import (
    CHAT_SYSTEM_TEXT,
    CHAT_USER_TEXT,
    DIRECT_ROUNDTRIP_TEXTS,
    FULL_CACHE_ATOL,
    FULL_CACHE_RTOL,
    MAXIMUM_DRAFT_TOKENS,
    MAXIMUM_GENERATED_TOKENS,
    MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
    MODEL_ALLOW_PATTERNS,
    MODEL_CHECK_TEXT,
    PLAN_PATH,
    PRIMARY_MODEL,
    RESULT_PATH,
    ROLLBACK_SUFFIX_TEXT,
    ROOT,
    build_pass_result,
    canonical_bytes,
    hash_file,
    read_plan,
    token_sequence_sha256,
    validate_pass_result,
)
from mlx_retrieval_runtime import (
    forced_speculative_generate,
    greedy_generate,
    prefill_decode_equivalence,
    rollback_equivalence,
)


LOCK_PATH = ROOT / "artifacts/large-model-retrieval-preflight-v4/process.lock"


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _require_head_blob(path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(("git", "show", f"HEAD:{relative}"), cwd=ROOT)
    if committed != path.read_bytes():
        raise RuntimeError(f"artifact is not the exact HEAD blob: {relative}")


def _require_never_published(path: Path) -> None:
    history = _git("log", "--all", "--format=%H", "--", path.relative_to(ROOT).as_posix())
    if history:
        raise FileExistsError(f"artifact was already published: {path.relative_to(ROOT)}")


def _model_file_manifest(snapshot: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in PRIMARY_MODEL["expected_files"]:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned model file: {name}")
        result[name] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    weight = result[PRIMARY_MODEL["weight_filename"]]
    if (
        weight["bytes"] != PRIMARY_MODEL["weight_bytes"]
        or weight["sha256"] != PRIMARY_MODEL["weight_sha256"]
    ):
        raise ValueError("pinned primary model weight differs")
    return result


def _config_projection(config: dict[str, Any]) -> dict[str, Any]:
    expected = PRIMARY_MODEL["config_projection"]
    projection = {key: config.get(key) for key in expected}
    if projection != expected:
        raise ValueError("loaded EXAONE config projection differs")
    return projection


def _tokenizer_checks(tokenizer, config: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, ...]]:
    roundtrips = []
    for text in DIRECT_ROUNDTRIP_TEXTS:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        roundtrips.append(tokenizer.decode(token_ids) == text)
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_TEXT},
        {"role": "user", "content": CHAT_USER_TEXT},
    ]
    chat_a = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    chat_b = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if chat_a != chat_b or len(chat_a) <= 1:
        raise ValueError("EXAONE chat template is not deterministic")
    prompt_ids = tuple(int(value) for value in tokenizer.encode(MODEL_CHECK_TEXT))
    if len(prompt_ids) <= 1:
        raise ValueError("EXAONE compatibility prompt is too short")
    vocab_size = int(config["vocab_size"])
    observed_vocab = int(getattr(tokenizer, "vocab_size", vocab_size))
    if observed_vocab != vocab_size:
        raise ValueError("EXAONE tokenizer vocabulary differs")
    result = {
        "chat_template_deterministic": True,
        "direct_roundtrip_count": len(roundtrips),
        "direct_roundtrip_exact": bool(all(roundtrips)),
        "prompt_token_count": len(prompt_ids),
        "vocab_size": vocab_size,
    }
    if result["direct_roundtrip_exact"] is not True:
        raise ValueError("EXAONE direct tokenizer round trip differs")
    return result, prompt_ids


def _forced_checks(model, prompt_ids: tuple[int, ...], baseline: tuple[int, ...], vocab_size: int):
    def full_provider(_context, remaining, output_index):
        return baseline[output_index : output_index + min(MAXIMUM_DRAFT_TOKENS, remaining)]

    def reject_provider(_context, remaining, output_index):
        width = min(MAXIMUM_DRAFT_TOKENS, remaining)
        correct = baseline[output_index]
        wrong = (correct + 1) % vocab_size
        return tuple(wrong for _ in range(width))

    def partial_provider(_context, remaining, output_index):
        if remaining < 2:
            return ()
        width = min(MAXIMUM_DRAFT_TOKENS, remaining)
        values = [baseline[output_index]]
        wrong = (baseline[output_index + 1] + 1) % vocab_size
        values.extend(wrong for _ in range(width - 1))
        return tuple(values)

    traces = {
        "full_accept": forced_speculative_generate(
            model,
            prompt_ids,
            maximum_tokens=MAXIMUM_GENERATED_TOKENS,
            maximum_draft_tokens=MAXIMUM_DRAFT_TOKENS,
            proposal_provider=full_provider,
        ),
        "immediate_reject": forced_speculative_generate(
            model,
            prompt_ids,
            maximum_tokens=MAXIMUM_GENERATED_TOKENS,
            maximum_draft_tokens=MAXIMUM_DRAFT_TOKENS,
            proposal_provider=reject_provider,
        ),
        "partial_accept": forced_speculative_generate(
            model,
            prompt_ids,
            maximum_tokens=MAXIMUM_GENERATED_TOKENS,
            maximum_draft_tokens=MAXIMUM_DRAFT_TOKENS,
            proposal_provider=partial_provider,
        ),
    }
    counters = {
        "full_accept": traces["full_accept"].full_accept_cycles,
        "immediate_reject": traces["immediate_reject"].immediate_reject_cycles,
        "partial_accept": traces["partial_accept"].partial_accept_cycles,
    }
    baseline_hash = token_sequence_sha256(baseline)
    paths = {}
    for name, trace in traces.items():
        passed = trace.token_ids == baseline and counters[name] > 0
        if not passed:
            raise ValueError(f"forced speculative path differs: {name}")
        paths[name] = {
            "counter": counters[name],
            "output_token_sequence_sha256": token_sequence_sha256(trace.token_ids),
            "pass": True,
        }
    return {
        "baseline_token_sequence_sha256": baseline_hash,
        "maximum_draft_tokens": MAXIMUM_DRAFT_TOKENS,
        "paths": paths,
        "pass": True,
    }


def _run_locked() -> dict[str, Any]:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("large-model compatibility requires a clean worktree")
    if RESULT_PATH.exists():
        raise FileExistsError("large-model compatibility result already exists")
    _require_never_published(RESULT_PATH)
    plan = read_plan(verify_derived=True)
    _require_head_blob(PLAN_PATH)
    commit = _git("rev-parse", "HEAD")
    snapshot = Path(
        snapshot_download(
            repo_id=PRIMARY_MODEL["repo_id"],
            revision=PRIMARY_MODEL["revision"],
            allow_patterns=list(MODEL_ALLOW_PATTERNS),
            local_files_only=True,
        )
    )
    model_files = _model_file_manifest(snapshot)
    config_on_disk = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    _config_projection(config_on_disk)

    mx.clear_cache()
    mx.reset_peak_memory()
    model, tokenizer, loaded_config = load(
        str(snapshot),
        lazy=False,
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
    )
    _config_projection(loaded_config)
    tokenizer_evidence, prompt_ids = _tokenizer_checks(tokenizer, loaded_config)

    full_cache, oracle_baseline = prefill_decode_equivalence(
        model,
        prompt_ids,
        maximum_tokens=MAXIMUM_GENERATED_TOKENS,
        atol=FULL_CACHE_ATOL,
        rtol=FULL_CACHE_RTOL,
    )
    if full_cache["decision_equivalence_pass"] is not True:
        raise ValueError("EXAONE full/cache greedy decision equivalence failed")

    suffix_ids = tuple(
        int(value)
        for value in tokenizer.encode(ROLLBACK_SUFFIX_TEXT, add_special_tokens=False)
    )
    if len(suffix_ids) < 4:
        raise ValueError("rollback suffix tokenization is too short")
    rollback = rollback_equivalence(
        model,
        prompt_ids,
        suffix_ids[:3],
        keep_speculative_tokens=1,
        correction_token_id=suffix_ids[3],
        atol=FULL_CACHE_ATOL,
        rtol=FULL_CACHE_RTOL,
    )
    if rollback["decision_equivalence_pass"] is not True:
        raise ValueError("EXAONE rollback greedy decision equivalence failed")

    baseline_a = greedy_generate(
        model, prompt_ids, maximum_tokens=MAXIMUM_GENERATED_TOKENS
    )
    baseline_b = greedy_generate(
        model, prompt_ids, maximum_tokens=MAXIMUM_GENERATED_TOKENS
    )
    if (
        baseline_a != baseline_b
        or baseline_a != oracle_baseline
        or len(baseline_a) != MAXIMUM_GENERATED_TOKENS
    ):
        raise ValueError("EXAONE deterministic greedy replay differs")
    deterministic = {
        "generated_tokens": len(baseline_a),
        "pass": True,
        "repetitions": 2,
        "token_sequence_sha256": token_sequence_sha256(baseline_a),
    }
    forced = _forced_checks(
        model, prompt_ids, baseline_a, int(loaded_config["vocab_size"])
    )

    mx.synchronize()
    peak_bytes = int(mx.get_peak_memory())
    working_set = int(plan["environment"]["mlx"]["max_recommended_working_set_size"])
    maximum_allowed = math.floor(MAXIMUM_RECOMMENDED_MEMORY_FRACTION * working_set)
    peak_fraction = peak_bytes / working_set
    memory = {
        "maximum_allowed_bytes": maximum_allowed,
        "maximum_recommended_working_set_size": working_set,
        "model_parameters": int(get_total_parameters(model)),
        "peak_bytes": peak_bytes,
        "peak_fraction": peak_fraction,
        "safety_pass": bool(0 < peak_bytes <= maximum_allowed),
    }
    if memory["safety_pass"] is not True:
        raise ValueError("EXAONE compatibility memory safety failed")

    result = build_pass_result(
        plan=plan,
        runner_git_commit=commit,
        model_files=model_files,
        tokenizer=tokenizer_evidence,
        full_cache_equivalence=full_cache,
        rollback_equivalence=rollback,
        deterministic_greedy=deterministic,
        forced_speculative=forced,
        memory=memory,
    )
    validate_pass_result(result, plan=plan)
    if (
        _git("rev-parse", "HEAD") != commit
        or _git("status", "--porcelain", "--untracked-files=all")
    ):
        raise RuntimeError("repository changed during large-model compatibility")
    return result


def main() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError("another large-model compatibility process is active") from error
    try:
        result = _run_locked()
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        output = os.open(RESULT_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(output, "wb") as handle:
            handle.write(canonical_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        print(f"status={result['status']}")
        print(f"summary_sha256={result['summary_sha256']}")
        print("candidate-vs-baseline timing was not executed")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


if __name__ == "__main__":
    main()
