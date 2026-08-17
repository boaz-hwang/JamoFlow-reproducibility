from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

import large_model_retrieval_preflight as protocol


def _comparison() -> dict[str, object]:
    return {
        "argmax_exact": True,
        "comparison_positions": 4,
        "decision_equivalence_pass": True,
        "finite": True,
        "maximum_absolute_error": 0.0,
        "maximum_normalized_error": 0.0,
        "numeric_tolerance_pass": True,
    }


def _file_manifest() -> dict[str, dict[str, object]]:
    result = {
        name: {"bytes": 1, "sha256": "a" * 64}
        for name in protocol.PRIMARY_MODEL["expected_files"]
    }
    result[protocol.PRIMARY_MODEL["weight_filename"]] = {
        "bytes": protocol.PRIMARY_MODEL["weight_bytes"],
        "sha256": protocol.PRIMARY_MODEL["weight_sha256"],
    }
    return result


def _pass_result(plan: dict[str, object]) -> dict[str, object]:
    token_hash = protocol.token_sequence_sha256(tuple(range(16)))
    return protocol.build_pass_result(
        plan=plan,
        runner_git_commit="b" * 40,
        model_files=_file_manifest(),
        tokenizer={
            "chat_template_deterministic": True,
            "direct_roundtrip_count": len(protocol.DIRECT_ROUNDTRIP_TEXTS),
            "direct_roundtrip_exact": True,
            "prompt_token_count": 12,
            "vocab_size": protocol.PRIMARY_MODEL["config_projection"]["vocab_size"],
        },
        full_cache_equivalence=_comparison(),
        rollback_equivalence=_comparison(),
        deterministic_greedy={
            "generated_tokens": protocol.MAXIMUM_GENERATED_TOKENS,
            "pass": True,
            "repetitions": 2,
            "token_sequence_sha256": token_hash,
        },
        forced_speculative={
            "baseline_token_sequence_sha256": token_hash,
            "maximum_draft_tokens": protocol.MAXIMUM_DRAFT_TOKENS,
            "paths": {
                name: {
                    "counter": 1,
                    "output_token_sequence_sha256": token_hash,
                    "pass": True,
                }
                for name in ("full_accept", "immediate_reject", "partial_accept")
            },
            "pass": True,
        },
        memory={
            "maximum_allowed_bytes": 30_000,
            "maximum_recommended_working_set_size": 40_000,
            "model_parameters": 7_800_000_000,
            "peak_bytes": 20_000,
            "peak_fraction": 0.5,
            "safety_pass": True,
        },
    )


def test_environment_and_implementation_manifest_are_exact() -> None:
    environment = protocol.environment_identity()
    protocol.validate_environment(environment)
    assert environment["packages"]["mlx"] == "0.31.2"
    assert environment["packages"]["mlx-lm"] == "0.31.3"
    assert environment["mlx"]["metal_available"] is True
    assert len(protocol.IMPLEMENTATION_PATHS) == len(set(protocol.IMPLEMENTATION_PATHS))
    assert all((protocol.ROOT / path).is_file() for path in protocol.IMPLEMENTATION_PATHS)


def test_plan_round_trip_and_model_choice_are_fixed() -> None:
    plan = protocol.build_plan(git_commit_before_plan="a" * 40)
    protocol.validate_plan(plan, verify_derived=True)
    restored = json.loads(protocol.canonical_bytes(plan))
    protocol.validate_plan(restored, verify_derived=True)
    assert restored["model_selection"]["primary"] == protocol.PRIMARY_MODEL
    assert (
        restored["model_selection"]["technical_fallback"]
        == protocol.TECHNICAL_FALLBACK_MODEL
    )
    assert restored["preflight"]["timing_observed"] is False
    assert restored["preflight"]["candidate_vs_baseline_executed"] is False


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    (
        ("model_selection", "fallback_may_use_latency_acceptance_or_quality", True),
        ("preflight", "timing_observed", True),
        ("preflight", "all_vocabulary_numeric_tolerance_is_a_pass_gate", True),
        ("claim_boundary", "publication_efficiency_claim", True),
    ),
)
def test_resealed_plan_tamper_is_rejected(section, key, replacement) -> None:
    plan = protocol.build_plan(git_commit_before_plan="a" * 40)
    plan[section][key] = replacement
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    plan["plan_sha256"] = protocol.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="plan identity"):
        protocol.validate_plan(plan, verify_derived=False)


def test_pass_result_round_trip_and_tamper_rejection(tmp_path, monkeypatch) -> None:
    plan = protocol.build_plan(git_commit_before_plan="a" * 40)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(protocol.canonical_bytes(plan))
    monkeypatch.setattr(protocol, "PLAN_PATH", plan_path)
    result = _pass_result(plan)
    protocol.validate_pass_result(result, plan=plan)
    restored = json.loads(protocol.canonical_bytes(result))
    protocol.validate_pass_result(restored, plan=plan)

    for mutation in ("timing", "output", "memory"):
        tampered = deepcopy(result)
        if mutation == "timing":
            tampered["observed_scope"]["timing_observed"] = True
        elif mutation == "output":
            tampered["forced_speculative"]["paths"]["partial_accept"][
                "output_token_sequence_sha256"
            ] = "f" * 64
        else:
            tampered["memory"]["peak_fraction"] = 0.9
        unsigned = dict(tampered)
        unsigned.pop("summary_sha256")
        tampered["summary_sha256"] = protocol.canonical_sha256(unsigned)
        with pytest.raises(ValueError):
            protocol.validate_pass_result(tampered, plan=plan)


def test_numeric_diagnostic_failure_does_not_override_exact_greedy_decisions(
    tmp_path, monkeypatch
) -> None:
    plan = protocol.build_plan(git_commit_before_plan="a" * 40)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(protocol.canonical_bytes(plan))
    monkeypatch.setattr(protocol, "PLAN_PATH", plan_path)
    result = _pass_result(plan)
    comparison = result["full_cache_greedy_decision_equivalence"]
    comparison["maximum_normalized_error"] = 1.3894497156143188
    comparison["numeric_tolerance_pass"] = False
    unsigned = dict(result)
    unsigned.pop("summary_sha256")
    result["summary_sha256"] = protocol.canonical_sha256(unsigned)
    protocol.validate_pass_result(result, plan=plan)

    comparison["argmax_exact"] = False
    comparison["decision_equivalence_pass"] = False
    unsigned = dict(result)
    unsigned.pop("summary_sha256")
    result["summary_sha256"] = protocol.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="full_cache equivalence"):
        protocol.validate_pass_result(result, plan=plan)


def test_runner_has_no_timing_api_or_latency_output() -> None:
    path = protocol.ROOT / "scripts/run_large_model_retrieval_preflight.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "time" not in imported
    assert "timeit" not in imported
    assert "perf_counter" not in source
    assert "tokens_per_second" not in source
    assert "acceptance_rate" not in source


def test_model_weight_and_fallback_reasons_are_not_mutable_choices() -> None:
    assert protocol.PRIMARY_MODEL["weight_bytes"] == 4_398_345_620
    assert protocol.PRIMARY_MODEL["weight_sha256"] == (
        "d9796bd9c23f506751f618fc08780b197106c50adbf317e4fa518a3c8a40974c"
    )
    assert "runtime_crash" in protocol.TECHNICAL_FALLBACK_REASONS
    assert all(
        word not in " ".join(protocol.TECHNICAL_FALLBACK_REASONS)
        for word in ("latency", "acceptance", "quality", "speed")
    )


def test_no_implementation_path_escapes_repository() -> None:
    root = protocol.ROOT.resolve()
    for relative in protocol.IMPLEMENTATION_PATHS:
        assert root in (protocol.ROOT / Path(relative)).resolve().parents
