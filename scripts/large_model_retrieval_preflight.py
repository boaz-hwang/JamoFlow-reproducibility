"""Pure contracts for the timing-silent public 7.8B MLX compatibility preflight."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVALIDATED_PLAN_PATH = ROOT / "data/manifests/large-model-retrieval-preflight-v1.json"
INVALIDATED_V2_PLAN_PATH = ROOT / "data/manifests/large-model-retrieval-preflight-v2.json"
INVALIDATED_V3_PLAN_PATH = ROOT / "data/manifests/large-model-retrieval-preflight-v3.json"
PLAN_PATH = ROOT / "data/manifests/large-model-retrieval-preflight-v4.json"
RESULT_PATH = ROOT / "results/large-model-retrieval-preflight-v4/summary.json"

PROTOCOL_ID = "jamoflow-large-model-retrieval-preflight-v4"
PLAN_KIND = "large_model_retrieval_preflight_plan_v4"
RESULT_KIND = "large_model_retrieval_preflight_result_v4"

PRIMARY_MODEL: dict[str, Any] = {
    "role": "primary",
    "repo_id": "mlx-community/EXAONE-3.5-7.8B-Instruct-4bit",
    "revision": "6f8fba5756a6e2987aecacd8d7e8bb9410ef1a53",
    "model_type": "exaone",
    "architecture": "ExaoneForCausalLM",
    "weight_filename": "model.safetensors",
    "weight_bytes": 4_398_345_620,
    "weight_sha256": "d9796bd9c23f506751f618fc08780b197106c50adbf317e4fa518a3c8a40974c",
    "expected_files": [
        "config.json",
        "configuration_exaone.py",
        "merges.txt",
        "model.safetensors",
        "model.safetensors.index.json",
        "modeling_exaone.py",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    "config_projection": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "model_type": "exaone",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_layers": 32,
        "quantization": {"bits": 4, "group_size": 64},
        "tie_word_embeddings": False,
        "vocab_size": 102400,
    },
}

TECHNICAL_FALLBACK_MODEL: dict[str, Any] = {
    "role": "technical_fallback_only",
    "repo_id": "mlx-community/Qwen3-8B-4bit",
    "revision": "545dc4251c05440727734bcd94334791f6ab0192",
    "model_type": "qwen3",
    "architecture": "Qwen3ForCausalLM",
    "weight_filename": "model.safetensors",
    "weight_bytes": 4_607_835_174,
    "weight_sha256": "f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8",
}

MODEL_ALLOW_PATTERNS = (
    "*.json",
    "*.py",
    "*.txt",
    "model*.safetensors",
)

IMPLEMENTATION_PATHS = (
    "docs/171-retrieval-novelty-closure-and-large-model-replication-direction.md",
    "docs/172-large-model-retrieval-compatibility-preflight-protocol.md",
    "docs/173-large-model-preflight-v1-invalidation-and-v2-correction.md",
    "docs/174-large-model-preflight-v2-invalidation-and-v3-oracle-correction.md",
    "docs/175-large-model-preflight-v3-invalidation-and-v4-decision-contract.md",
    "requirements/apple-retrieval-v1.txt",
    "scripts/prepare_large_model_retrieval_preflight.py",
    "scripts/run_large_model_retrieval_preflight.py",
    "scripts/seal_large_model_retrieval_preflight_plan.py",
    "scripts/large_model_retrieval_preflight.py",
    "scripts/mlx_retrieval_runtime.py",
    "tests/test_large_model_retrieval_preflight.py",
    "tests/test_mlx_retrieval_runtime.py",
)

DIRECT_ROUNDTRIP_TEXTS = (
    "한국어 형태와 띄어쓰기의 관계를 설명한다.",
    "JamoFlow는 실제 추론 시간을 우선한다.",
    "한글, ASCII, 123, code-mixing을 함께 확인한다.",
)
CHAT_USER_TEXT = "한국어 추론 효율을 한 문장으로 설명해 줘."
CHAT_SYSTEM_TEXT = "You are EXAONE model from LG AI Research, a helpful assistant."
MODEL_CHECK_TEXT = "한국어 추론에서 캐시와 검증 경로가 정확히 일치해야 한다."
ROLLBACK_SUFFIX_TEXT = " 가나다라마 바사아"

FULL_CACHE_ATOL = 0.05
FULL_CACHE_RTOL = 0.01
MAXIMUM_GENERATED_TOKENS = 16
MAXIMUM_DRAFT_TOKENS = 3
MAXIMUM_RECOMMENDED_MEMORY_FRACTION = 0.75

TECHNICAL_FALLBACK_REASONS = (
    "built_in_loader_failure",
    "tokenizer_or_chat_template_failure",
    "full_cache_greedy_decision_failure",
    "cache_trim_or_rollback_failure",
    "deterministic_greedy_failure",
    "forced_speculative_exactness_failure",
    "memory_safety_failure",
    "runtime_crash",
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def environment_identity() -> dict[str, Any]:
    import mlx.core as mx

    info = mx.device_info()
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "macos_version": platform.mac_ver()[0],
        "packages": {
            "huggingface-hub": package_version("huggingface-hub"),
            "mlx": package_version("mlx"),
            "mlx-lm": package_version("mlx-lm"),
            "numpy": package_version("numpy"),
            "tokenizers": package_version("tokenizers"),
            "transformers": package_version("transformers"),
        },
        "mlx": {
            "default_device": str(mx.default_device()),
            "metal_available": bool(mx.metal.is_available()),
            "device_name": info["device_name"],
            "architecture": info["architecture"],
            "memory_size": int(info["memory_size"]),
            "max_recommended_working_set_size": int(
                info["max_recommended_working_set_size"]
            ),
        },
    }


def validate_environment(environment: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "python",
        "platform_system",
        "platform_machine",
        "macos_version",
        "packages",
        "mlx",
    }
    expected_packages = {
        "huggingface-hub",
        "mlx",
        "mlx-lm",
        "numpy",
        "tokenizers",
        "transformers",
    }
    expected_mlx = {
        "default_device",
        "metal_available",
        "device_name",
        "architecture",
        "memory_size",
        "max_recommended_working_set_size",
    }
    packages = environment.get("packages")
    mlx = environment.get("mlx")
    if (
        set(environment) != expected_top
        or environment.get("schema_version") != 1
        or environment.get("platform_system") != "Darwin"
        or environment.get("platform_machine") != "arm64"
        or not isinstance(environment.get("python"), str)
        or not isinstance(environment.get("macos_version"), str)
        or not isinstance(packages, Mapping)
        or set(packages) != expected_packages
        or packages.get("mlx") != "0.31.2"
        or packages.get("mlx-lm") != "0.31.3"
        or not all(isinstance(packages[key], str) for key in expected_packages)
        or not isinstance(mlx, Mapping)
        or set(mlx) != expected_mlx
        or mlx.get("default_device") != "Device(gpu, 0)"
        or mlx.get("metal_available") is not True
        or not isinstance(mlx.get("device_name"), str)
        or not isinstance(mlx.get("architecture"), str)
        or not isinstance(mlx.get("memory_size"), int)
        or int(mlx.get("memory_size", 0)) <= 0
        or not isinstance(mlx.get("max_recommended_working_set_size"), int)
        or int(mlx.get("max_recommended_working_set_size", 0)) <= 0
    ):
        raise ValueError("large-model MLX environment differs")


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("large-model preflight implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def model_selection_contract() -> dict[str, Any]:
    return {
        "primary": PRIMARY_MODEL,
        "technical_fallback": TECHNICAL_FALLBACK_MODEL,
        "fallback_reasons": list(TECHNICAL_FALLBACK_REASONS),
        "fallback_may_use_latency_acceptance_or_quality": False,
        "model_shopping_for_fastest_result": False,
        "primary_pass_permanently_selects_primary": True,
        "primary_latency_failure_may_open_fallback": False,
    }


def preflight_contract() -> dict[str, Any]:
    return {
        "timing_observed": False,
        "candidate_vs_baseline_executed": False,
        "acceptance_rate_observed": False,
        "throughput_observed": False,
        "direct_roundtrip_text_sha256": canonical_sha256(
            list(DIRECT_ROUNDTRIP_TEXTS)
        ),
        "chat_user_text_sha256": hashlib.sha256(
            CHAT_USER_TEXT.encode("utf-8")
        ).hexdigest(),
        "chat_system_text_sha256": hashlib.sha256(
            CHAT_SYSTEM_TEXT.encode("utf-8")
        ).hexdigest(),
        "model_check_text_sha256": hashlib.sha256(
            MODEL_CHECK_TEXT.encode("utf-8")
        ).hexdigest(),
        "rollback_suffix_text_sha256": hashlib.sha256(
            ROLLBACK_SUFFIX_TEXT.encode("utf-8")
        ).hexdigest(),
        "all_vocabulary_numeric_tolerance_is_a_pass_gate": False,
        "cache_oracle": (
            "parallel prefill plus cached decode compared at every step against "
            "the corresponding monolithic full prefix"
        ),
        "finite_logits_required": True,
        "greedy_argmax_exact_required": True,
        "logit_diagnostic_atol": FULL_CACHE_ATOL,
        "logit_diagnostic_rtol": FULL_CACHE_RTOL,
        "maximum_logit_error_is_descriptive_only": True,
        "tokenwise_single-token_replay_of_prompt_required": False,
        "rollback_fresh_cache_argmax_exact_required": True,
        "maximum_generated_tokens": MAXIMUM_GENERATED_TOKENS,
        "maximum_draft_tokens": MAXIMUM_DRAFT_TOKENS,
        "forced_paths": ["full_accept", "immediate_reject", "partial_accept"],
        "all_forced_outputs_equal_baseline_required": True,
        "maximum_recommended_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
        "raw_generated_text_published": False,
        "generated_token_ids_published": False,
        "generated_token_hashes_published": True,
    }


def prior_invalidated_attempts_contract() -> list[dict[str, Any]]:
    common = {
        "result_published": False,
        "candidate_vs_baseline_executed": False,
        "timing_observed": False,
        "fallback_authorized": False,
    }
    return [
        {
            **common,
            "plan_path": INVALIDATED_PLAN_PATH.relative_to(ROOT).as_posix(),
            "plan_artifact_sha256": hash_file(INVALIDATED_PLAN_PATH),
            "plan_sha256": "a1964a1b623a7e8c7904ec2cfce811e122802aa6d8df3c99733dfaaf3cac325d",
            "failure_stage": "tokenizer_load_before_load_return_or_model_forward",
            "failure_reason": "runner_omitted_pinned_trust_remote_code",
            "model_forward_executed": False,
            "generated_token_observed": False,
        },
        {
            **common,
            "plan_path": INVALIDATED_V2_PLAN_PATH.relative_to(ROOT).as_posix(),
            "plan_artifact_sha256": hash_file(INVALIDATED_V2_PLAN_PATH),
            "plan_sha256": "81ed3957443c3fc935391886c37f3549f8f5f3b70dcd53194c4546ac57c52a09",
            "failure_stage": "tokenwise_prompt_cache_oracle_before_generation",
            "failure_reason": "oracle_did_not_match_parallel_prefill_runtime",
            "model_forward_executed": True,
            "generated_token_observed": False,
        },
        {
            "plan_path": INVALIDATED_V3_PLAN_PATH.relative_to(ROOT).as_posix(),
            "plan_artifact_sha256": hash_file(INVALIDATED_V3_PLAN_PATH),
            "plan_sha256": "f8b4b32afbd8883ce427a9fc71ca8e28cfc4386c3ecad955cade0dd0d2ff62ad",
            "failure_stage": "parallel_prefill_decode_all_logit_numeric_bound",
            "failure_reason": (
                "one_decode_step_exceeded_an_all_vocabulary_numeric_bound_while_"
                "all_greedy_argmax_decisions_remained_exact"
            ),
            "result_published": False,
            "model_forward_executed": True,
            "generated_token_observed": True,
            "candidate_vs_baseline_executed": False,
            "timing_observed": False,
            "fallback_authorized": True,
            "timing_silent_diagnostic": {
                "argmax_mismatch_count": 0,
                "comparison_positions": 28,
                "failed_numeric_steps": 1,
                "forced_full_accept_cycles": 4,
                "forced_immediate_reject_cycles": 16,
                "forced_partial_accept_cycles": 8,
                "forced_paths_exact": True,
                "maximum_absolute_error": 0.080078125,
                "maximum_normalized_error": 1.3894497156143188,
                "rollback_argmax_exact": True,
                "rollback_maximum_normalized_error": 0.8595988750457764,
            },
        },
    ]


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "model_compatibility_only": True,
        "actual_efficiency_tested": False,
        "korean_specific_method_tested": False,
        "generic_retrieval_method_novelty_claimed": False,
        "publication_efficiency_claim": False,
        "fallback_selected": False,
    }


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if not (
        isinstance(git_commit_before_plan, str)
        and len(git_commit_before_plan) == 40
        and all(c in "0123456789abcdef" for c in git_commit_before_plan)
    ):
        raise ValueError("large-model pre-plan commit differs")
    environment = environment_identity()
    validate_environment(environment)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_after_v3_numeric_bound_failure_before_any_retrieval_or_timing",
        "git_commit_before_plan": git_commit_before_plan,
        "implementation_sha256": implementation_identity(),
        "environment": environment,
        "prior_invalidated_attempts": prior_invalidated_attempts_contract(),
        "model_selection": model_selection_contract(),
        "model_download": {
            "allow_patterns": list(MODEL_ALLOW_PATTERNS),
            "local_files_only_during_forward": True,
            "repository_revision_is_authority": True,
            "tokenizer_trust_remote_code": True,
            "remote_code_pinned_by_revision_and_file_hash": True,
            "weight_content_sha256_and_bytes_required": True,
        },
        "preflight": preflight_contract(),
        "result_path": RESULT_PATH.relative_to(ROOT).as_posix(),
        "claim_boundary": claim_boundary_contract(),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "status",
        "git_commit_before_plan",
        "implementation_sha256",
        "environment",
        "prior_invalidated_attempts",
        "model_selection",
        "model_download",
        "preflight",
        "result_path",
        "claim_boundary",
        "plan_sha256",
    }
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    environment = plan.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("large-model plan environment differs")
    validate_environment(environment)
    if (
        set(plan) != expected_keys
        or plan.get("schema_version") != 1
        or plan.get("kind") != PLAN_KIND
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status")
        != "sealed_after_v3_numeric_bound_failure_before_any_retrieval_or_timing"
        or not isinstance(recorded, str)
        or canonical_sha256(unsigned) != recorded
        or plan.get("prior_invalidated_attempts")
        != prior_invalidated_attempts_contract()
        or plan.get("model_selection") != model_selection_contract()
        or plan.get("model_download")
        != {
            "allow_patterns": list(MODEL_ALLOW_PATTERNS),
            "local_files_only_during_forward": True,
            "repository_revision_is_authority": True,
            "tokenizer_trust_remote_code": True,
            "remote_code_pinned_by_revision_and_file_hash": True,
            "weight_content_sha256_and_bytes_required": True,
        }
        or plan.get("preflight") != preflight_contract()
        or plan.get("result_path") != RESULT_PATH.relative_to(ROOT).as_posix()
        or plan.get("claim_boundary") != claim_boundary_contract()
    ):
        raise ValueError("large-model preflight plan identity differs")
    implementation = plan.get("implementation_sha256")
    if (
        not isinstance(implementation, Mapping)
        or len(implementation) != len(IMPLEMENTATION_PATHS)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or not all(is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
    ):
        raise ValueError("large-model implementation manifest differs")
    if verify_derived:
        if dict(implementation) != implementation_identity():
            raise ValueError("large-model implementation files differ")
        if dict(environment) != environment_identity():
            raise ValueError("large-model runtime environment differs")


def read_plan(*, verify_derived: bool) -> dict[str, Any]:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    validate_plan(payload, verify_derived=verify_derived)
    return payload


def token_sequence_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256(b"JamoFlow/MLX-token-sequence/v1\0")
    digest.update(len(token_ids).to_bytes(8, "big"))
    for token_id in token_ids:
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("token ids must be nonnegative integers")
        digest.update(token_id.to_bytes(8, "big"))
    return digest.hexdigest()


def _validate_file_manifest(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(PRIMARY_MODEL["expected_files"]):
        raise ValueError("large-model file manifest differs")
    for path, identity in value.items():
        if (
            not isinstance(path, str)
            or not isinstance(identity, Mapping)
            or set(identity) != {"bytes", "sha256"}
            or not isinstance(identity.get("bytes"), int)
            or int(identity.get("bytes", 0)) <= 0
            or not is_sha256(identity.get("sha256"))
        ):
            raise ValueError("large-model file identity differs")
    weight = value[PRIMARY_MODEL["weight_filename"]]
    if (
        weight["bytes"] != PRIMARY_MODEL["weight_bytes"]
        or weight["sha256"] != PRIMARY_MODEL["weight_sha256"]
    ):
        raise ValueError("large-model weight identity differs")


def validate_pass_result(result: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "status",
        "plan_artifact_sha256",
        "plan_sha256",
        "runner_git_commit",
        "environment",
        "model",
        "model_files",
        "config_projection",
        "tokenizer",
        "full_cache_greedy_decision_equivalence",
        "rollback_greedy_decision_equivalence",
        "deterministic_greedy",
        "forced_speculative",
        "memory",
        "observed_scope",
        "decision",
        "claim_boundary",
        "summary_sha256",
    }
    unsigned = dict(result)
    recorded = unsigned.pop("summary_sha256", None)
    if (
        set(result) != expected_keys
        or result.get("schema_version") != 1
        or result.get("kind") != RESULT_KIND
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("status") != "pass_primary_greedy_transaction_compatibility"
        or result.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or result.get("plan_sha256") != plan.get("plan_sha256")
        or not isinstance(result.get("runner_git_commit"), str)
        or len(result["runner_git_commit"]) != 40
        or result.get("environment") != plan.get("environment")
        or result.get("model") != PRIMARY_MODEL
        or result.get("config_projection") != PRIMARY_MODEL["config_projection"]
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("large-model pass result identity differs")
    _validate_file_manifest(result.get("model_files"))
    tokenizer = result.get("tokenizer")
    full_cache = result.get("full_cache_greedy_decision_equivalence")
    rollback = result.get("rollback_greedy_decision_equivalence")
    deterministic = result.get("deterministic_greedy")
    forced = result.get("forced_speculative")
    memory = result.get("memory")
    observed = result.get("observed_scope")
    decision = result.get("decision")
    claim = result.get("claim_boundary")
    if (
        not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "chat_template_deterministic",
            "direct_roundtrip_count",
            "direct_roundtrip_exact",
            "prompt_token_count",
            "vocab_size",
        }
        or tokenizer.get("chat_template_deterministic") is not True
        or tokenizer.get("direct_roundtrip_count") != len(DIRECT_ROUNDTRIP_TEXTS)
        or tokenizer.get("direct_roundtrip_exact") is not True
        or not isinstance(tokenizer.get("prompt_token_count"), int)
        or int(tokenizer.get("prompt_token_count", 0)) <= 1
        or tokenizer.get("vocab_size") != PRIMARY_MODEL["config_projection"]["vocab_size"]
    ):
        raise ValueError("large-model tokenizer result differs")
    comparison_keys = {
        "argmax_exact",
        "comparison_positions",
        "decision_equivalence_pass",
        "finite",
        "maximum_absolute_error",
        "maximum_normalized_error",
        "numeric_tolerance_pass",
    }
    for name, value in (("full_cache", full_cache), ("rollback", rollback)):
        if (
            not isinstance(value, Mapping)
            or set(value) != comparison_keys
            or value.get("argmax_exact") is not True
            or value.get("decision_equivalence_pass") is not True
            or value.get("finite") is not True
            or not isinstance(value.get("numeric_tolerance_pass"), bool)
            or not isinstance(value.get("comparison_positions"), int)
            or int(value.get("comparison_positions", 0)) <= 0
            or not isinstance(value.get("maximum_absolute_error"), (int, float))
            or float(value.get("maximum_absolute_error", -1)) < 0
            or not isinstance(value.get("maximum_normalized_error"), (int, float))
            or not math.isfinite(float(value.get("maximum_normalized_error", -1)))
            or float(value.get("maximum_normalized_error", -1)) < 0
        ):
            raise ValueError(f"large-model {name} equivalence differs")
    if (
        not isinstance(deterministic, Mapping)
        or set(deterministic)
        != {"generated_tokens", "pass", "repetitions", "token_sequence_sha256"}
        or deterministic.get("generated_tokens") != MAXIMUM_GENERATED_TOKENS
        or deterministic.get("pass") is not True
        or deterministic.get("repetitions") != 2
        or not is_sha256(deterministic.get("token_sequence_sha256"))
    ):
        raise ValueError("large-model deterministic greedy result differs")
    if (
        not isinstance(forced, Mapping)
        or set(forced)
        != {
            "baseline_token_sequence_sha256",
            "maximum_draft_tokens",
            "paths",
            "pass",
        }
        or forced.get("baseline_token_sequence_sha256")
        != deterministic.get("token_sequence_sha256")
        or forced.get("maximum_draft_tokens") != MAXIMUM_DRAFT_TOKENS
        or forced.get("pass") is not True
        or not isinstance(forced.get("paths"), Mapping)
        or set(forced["paths"]) != {"full_accept", "immediate_reject", "partial_accept"}
    ):
        raise ValueError("large-model forced speculative result differs")
    for path, evidence in forced["paths"].items():
        if (
            not isinstance(evidence, Mapping)
            or set(evidence)
            != {"counter", "output_token_sequence_sha256", "pass"}
            or evidence.get("output_token_sequence_sha256")
            != deterministic.get("token_sequence_sha256")
            or evidence.get("pass") is not True
            or not isinstance(evidence.get("counter"), int)
            or int(evidence.get("counter", 0)) <= 0
        ):
            raise ValueError(f"large-model forced path differs: {path}")
    if (
        not isinstance(memory, Mapping)
        or set(memory)
        != {
            "maximum_allowed_bytes",
            "maximum_recommended_working_set_size",
            "model_parameters",
            "peak_bytes",
            "peak_fraction",
            "safety_pass",
        }
        or memory.get("safety_pass") is not True
        or not isinstance(memory.get("model_parameters"), int)
        or int(memory.get("model_parameters", 0)) <= 0
        or not isinstance(memory.get("peak_bytes"), int)
        or not isinstance(memory.get("maximum_allowed_bytes"), int)
        or not isinstance(memory.get("maximum_recommended_working_set_size"), int)
        or not isinstance(memory.get("peak_fraction"), (int, float))
        or not 0 < float(memory.get("peak_fraction", 0)) <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION
        or int(memory.get("peak_bytes", 0)) > int(memory.get("maximum_allowed_bytes", -1))
    ):
        raise ValueError("large-model memory result differs")
    if observed != {
        "acceptance_rate_observed": False,
        "candidate_vs_baseline_executed": False,
        "generated_token_ids_published": False,
        "raw_generated_text_published": False,
        "throughput_observed": False,
        "timing_observed": False,
    }:
        raise ValueError("large-model observed scope differs")
    if decision != {
        "fallback_authorized": False,
        "primary_greedy_transaction_compatibility_pass": True,
        "selected_model_repo_id": PRIMARY_MODEL["repo_id"],
        "selected_model_revision": PRIMARY_MODEL["revision"],
    }:
        raise ValueError("large-model decision differs")
    expected_claim = {**claim_boundary_contract(), "fallback_selected": False}
    if claim != expected_claim:
        raise ValueError("large-model result claim boundary differs")


def build_pass_result(
    *,
    plan: Mapping[str, Any],
    runner_git_commit: str,
    model_files: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    full_cache_equivalence: Mapping[str, Any],
    rollback_equivalence: Mapping[str, Any],
    deterministic_greedy: Mapping[str, Any],
    forced_speculative: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "pass_primary_greedy_transaction_compatibility",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "runner_git_commit": runner_git_commit,
        "environment": plan["environment"],
        "model": PRIMARY_MODEL,
        "model_files": dict(model_files),
        "config_projection": PRIMARY_MODEL["config_projection"],
        "tokenizer": dict(tokenizer),
        "full_cache_greedy_decision_equivalence": dict(full_cache_equivalence),
        "rollback_greedy_decision_equivalence": dict(rollback_equivalence),
        "deterministic_greedy": dict(deterministic_greedy),
        "forced_speculative": dict(forced_speculative),
        "memory": dict(memory),
        "observed_scope": {
            "acceptance_rate_observed": False,
            "candidate_vs_baseline_executed": False,
            "generated_token_ids_published": False,
            "raw_generated_text_published": False,
            "throughput_observed": False,
            "timing_observed": False,
        },
        "decision": {
            "fallback_authorized": False,
            "primary_greedy_transaction_compatibility_pass": True,
            "selected_model_repo_id": PRIMARY_MODEL["repo_id"],
            "selected_model_revision": PRIMARY_MODEL["revision"],
        },
        "claim_boundary": {**claim_boundary_contract(), "fallback_selected": False},
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    validate_pass_result(payload, plan=plan)
    return payload
