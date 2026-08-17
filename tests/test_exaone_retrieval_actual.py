from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import exaone_retrieval_actual as actual
import exaone_retrieval_actual_runtime as runtime
import numpy as np
import pytest
from exaone_retrieval_data import canonical_bytes, npz_bytes
from large_model_retrieval_preflight import token_sequence_sha256
from mlx_retrieval_runtime import ForcedSpeculativeTrace


@pytest.fixture
def isolated_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(actual, "ROOT", tmp_path)
    monkeypatch.setattr(actual, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(actual, "ARTIFACT_ROOT", tmp_path / "ignored")
    monkeypatch.setattr(actual, "SESSION_ARTIFACT_ROOT", tmp_path / "ignored/sessions")
    monkeypatch.setattr(actual, "SESSION_RECEIPT_ROOT", tmp_path / "receipts")
    monkeypatch.setattr(actual, "SUMMARY_PATH", tmp_path / "summary.json")
    dependency_paths = {
        "case_artifact": tmp_path / "source/cases.npz",
        "data_plan": tmp_path / "source/data-plan.json",
        "data_seal": tmp_path / "source/data-seal.json",
        "data_verification": tmp_path / "source/data-verification.json",
        "resource_plan": tmp_path / "source/resource-plan.json",
        "resource_result": tmp_path / "source/resource-result.json",
        "table_artifact": tmp_path / "source/table.npz",
    }
    for name, path in dependency_paths.items():
        monkeypatch.setattr(
            actual,
            {
                "case_artifact": "CASES_PATH",
                "data_plan": "DATA_PLAN_PATH",
                "data_seal": "DATA_SEAL_PATH",
                "data_verification": "DATA_VERIFICATION_PATH",
                "resource_plan": "RESOURCE_PLAN_PATH",
                "resource_result": "RESOURCE_RESULT_PATH",
                "table_artifact": "TABLE_PATH",
            }[name],
            path,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))

    environment = {
        "mlx": {"max_recommended_working_set_size": 1_000},
        "fixture": True,
    }
    model_identity = {
        "model_files": {"model.safetensors": {"bytes": 1, "sha256": "a" * 64}},
        "model_parameter_count": 123,
    }
    resource = {
        "status": "pass_baseline_resource_feasibility",
        "actual_schedule_decision": {
            "status": "feasible",
            "selected": {
                "inner_repetitions": actual.INNER_REPETITIONS,
                "projected_campaign_hours": 2.5,
                "sessions": actual.SESSIONS,
            },
        },
        "environment": environment,
        "memory": {"safety_pass": True},
        "model_identity": model_identity,
    }
    table_identity = {
        "arrays": {
            "fixture": {
                "dtype": "uint32",
                "sha256": "9" * 64,
                "shape": [1_000_000],
            }
        },
        "bytes": 10,
        "path": "ignored/table.npz",
        "sha256": "b" * 64,
    }
    case_identity = {
        "arrays": {},
        "bytes": 10,
        "path": "ignored/cases.npz",
        "sha256": "c" * 64,
    }
    data_seal = {"table_artifact": table_identity, "case_artifact": case_identity}
    actual.RESOURCE_RESULT_PATH.write_text(json.dumps(resource), encoding="utf-8")
    actual.DATA_SEAL_PATH.write_text(json.dumps(data_seal), encoding="utf-8")
    dependencies = {
        name: {
            "bytes": path.stat().st_size,
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": f"{index + 1:064x}",
        }
        for index, (name, path) in enumerate(dependency_paths.items())
    }
    implementation = {name: "d" * 64 for name in actual.IMPLEMENTATION_PATHS}
    monkeypatch.setattr(actual, "dependency_identity", lambda: dependencies)
    monkeypatch.setattr(actual, "implementation_identity", lambda: implementation)
    monkeypatch.setattr(actual, "environment_identity", lambda: deepcopy(environment))
    monkeypatch.setattr(actual, "validate_environment", lambda value: None)
    monkeypatch.setattr(
        actual,
        "read_resource_result",
        lambda verify_artifact: deepcopy(resource),
    )
    monkeypatch.setattr(
        actual,
        "read_data_seal",
        lambda verify_artifacts: deepcopy(data_seal),
    )
    monkeypatch.setattr(actual, "read_verification", dict)
    return environment, model_identity, resource


def _arrays(
    session_index: int,
    *,
    baseline_ns: int = 1_000_000_000,
    candidate_ns: int = 800_000_000,
) -> dict[str, np.ndarray]:
    shape = (actual.MEASURED_CASES, actual.INNER_REPETITIONS, len(actual.ROLES))
    output = np.zeros(shape + (actual.OUTPUT_TOKENS,), dtype=np.uint32)
    output_hash = np.asarray(
        list(bytes.fromhex(token_sequence_sha256((0,) * actual.OUTPUT_TOKENS))),
        dtype=np.uint8,
    )
    arrays = {
        "case_order": np.asarray(
            actual.measured_case_order(session_index), dtype=np.uint16
        ),
        "decoded_utf8_sha256": np.ones(shape + (32,), dtype=np.uint8),
        "first_role": np.zeros(
            (actual.MEASURED_CASES, actual.INNER_REPETITIONS), dtype=np.uint8
        ),
        "output_token_ids": output,
        "output_token_sha256": np.broadcast_to(output_hash, shape + (32,)).copy(),
        "peak_active_bytes": np.full(shape, 500, dtype=np.uint64),
    }
    for case_index in actual.measured_case_order(session_index):
        for repetition in range(actual.INNER_REPETITIONS):
            arrays["first_role"][case_index, repetition] = actual.balanced_role_order(
                session_index, case_index, repetition
            )[0]
    arrays["tokenization_ns"] = np.empty(shape, dtype=np.int64)
    arrays["generation_ns"] = np.empty(shape, dtype=np.int64)
    arrays["detokenization_ns"] = np.full(shape, 10_000_000, dtype=np.int64)
    arrays["elapsed_ns"] = np.empty(shape, dtype=np.int64)
    for role, elapsed in enumerate((baseline_ns, candidate_ns)):
        arrays["tokenization_ns"][..., role] = 10_000_000
        arrays["generation_ns"][..., role] = elapsed - 20_000_000
        arrays["elapsed_ns"][..., role] = elapsed
    arrays.update(
        {name: np.zeros(shape, dtype=np.uint16) for name in actual.COUNTER_NAMES}
    )
    baseline = (..., actual.BASELINE_ROLE_INDEX)
    arrays["prompt_token_count"][baseline] = 128
    arrays["target_prefill_forward_calls"][baseline] = 1
    arrays["target_generation_forward_calls"][baseline] = 128
    arrays["no_proposal_calls"][baseline] = 128
    arrays["final_cache_offset"][baseline] = 255
    candidate = (..., actual.CANDIDATE_ROLE_INDEX)
    arrays["prompt_token_count"][candidate] = 128
    arrays["target_prefill_forward_calls"][candidate] = 1
    arrays["target_generation_forward_calls"][candidate] = 32
    arrays["proposal_attempts"][candidate] = 32
    arrays["corpus_proposal_calls"][candidate] = 32
    arrays["corpus_proposed_tokens"][candidate] = 96
    arrays["corpus_accepted_draft_tokens"][candidate] = 96
    arrays["full_accept_cycles"][candidate] = 32
    arrays["proposed_tokens"][candidate] = 96
    arrays["accepted_draft_tokens"][candidate] = 96
    arrays["bonus_tokens"][candidate] = 32
    arrays["final_cache_offset"][candidate] = 255
    return arrays


def _operational() -> dict:
    return {
        "ac_power": True,
        "battery_sha256": "1" * 64,
        "conflicting_process_count": 0,
        "process_inventory_pass": True,
        "process_inventory_sha256": "2" * 64,
        "thermal_pass": True,
        "thermal_sha256": "3" * 64,
    }


def _memory() -> dict:
    return {
        "active_after_bytes": 400,
        "active_before_bytes": 400,
        "cache_after_bytes": 20,
        "cache_before_bytes": 20,
        "maximum_allowed_bytes": 750,
        "maximum_observed_working_set_bytes": 500,
        "maximum_recommended_working_set_size": 1_000,
        "peak_active_bytes": 500,
        "process_peak_rss_bytes": 300,
        "safety_pass": True,
        "working_set_fraction": 0.5,
    }


def test_plan_roundtrip_and_result_boundary(isolated_contract) -> None:
    plan = actual.build_plan(git_commit_before_plan="a" * 40)
    actual.PLAN_PATH.write_bytes(canonical_bytes(plan))
    restored = json.loads(canonical_bytes(plan))
    actual.validate_plan(restored, verify_derived=True)
    assert (
        plan["claim_boundary"]["actual_retrieval_latency_observed_before_plan"] is False
    )
    assert plan["claim_boundary"]["baseline_resource_latency_observed_before_plan"]
    assert plan["claim_boundary"]["generic_retrieval_is_novel"] is False
    assert plan["claim_boundary"]["confirmatory_or_final_blind"] is False
    assert plan["claim_boundary"][
        "case_rank_seed_includes_compatibility_model_output_hash"
    ]
    assert plan["claim_boundary"]["case_selection_model_output_blind"] is False
    assert plan["actual_contract"]["sessions"] == 5


def test_plan_rejects_failed_or_memory_unsafe_resource_result(
    isolated_contract,
) -> None:
    _, _, resource = isolated_contract
    resource["status"] = "stop_resource_infeasible"
    with pytest.raises(ValueError, match="resource schedule"):
        actual.build_plan(git_commit_before_plan="a" * 40)

    resource["status"] = "pass_baseline_resource_feasibility"
    resource["actual_schedule_decision"]["status"] = "infeasible"
    with pytest.raises(ValueError, match="resource schedule"):
        actual.build_plan(git_commit_before_plan="a" * 40)

    resource["actual_schedule_decision"]["status"] = "feasible"
    resource["memory"]["safety_pass"] = False
    with pytest.raises(ValueError, match="resource schedule"):
        actual.build_plan(git_commit_before_plan="a" * 40)

    resource["memory"]["safety_pass"] = True
    plan = actual.build_plan(git_commit_before_plan="a" * 40)
    resource_on_disk = json.loads(
        actual.RESOURCE_RESULT_PATH.read_text(encoding="utf-8")
    )
    resource_on_disk["memory"]["safety_pass"] = False
    actual.RESOURCE_RESULT_PATH.write_text(
        json.dumps(resource_on_disk), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resource schedule"):
        actual.validate_plan(plan, verify_derived=False)


def test_plan_rejects_execution_environment_rotation(
    isolated_contract, monkeypatch
) -> None:
    plan = actual.build_plan(git_commit_before_plan="a" * 40)
    monkeypatch.setattr(actual, "environment_identity", lambda: {"rotated": True})
    with pytest.raises(ValueError, match="execution environment"):
        actual.validate_plan(plan, verify_derived=True)


def test_schedule_is_rotated_and_exactly_role_balanced() -> None:
    assert len({actual.measured_case_order(index) for index in range(5)}) == 5
    assert all(
        sorted(actual.measured_case_order(index)) == list(range(actual.MEASURED_CASES))
        for index in range(5)
    )
    for session in range(5):
        first = [
            actual.balanced_role_order(session, case_index, repetition)[0]
            for case_index in range(actual.MEASURED_CASES)
            for repetition in range(actual.INNER_REPETITIONS)
        ]
        assert first.count(actual.BASELINE_ROLE_INDEX) == len(first) // 2
        assert first.count(actual.CANDIDATE_ROLE_INDEX) == len(first) // 2
    for case_index in range(actual.MEASURED_CASES):
        for repetition in range(actual.INNER_REPETITIONS):
            across_sessions = [
                actual.balanced_role_order(session, case_index, repetition)[0]
                for session in range(actual.SESSIONS)
            ]
            assert set(across_sessions) == {
                actual.BASELINE_ROLE_INDEX,
                actual.CANDIDATE_ROLE_INDEX,
            }
            assert (
                abs(
                    across_sessions.count(actual.BASELINE_ROLE_INDEX)
                    - across_sessions.count(actual.CANDIDATE_ROLE_INDEX)
                )
                == 1
            )
    for temporal_position in range(actual.MEASURED_CASES):
        for repetition in range(actual.INNER_REPETITIONS):
            across_sessions = []
            for session in range(actual.SESSIONS):
                case_index = actual.measured_case_order(session)[temporal_position]
                across_sessions.append(
                    actual.balanced_role_order(session, case_index, repetition)[0]
                )
            assert set(across_sessions) == {
                actual.BASELINE_ROLE_INDEX,
                actual.CANDIDATE_ROLE_INDEX,
            }
            assert (
                abs(
                    across_sessions.count(actual.BASELINE_ROLE_INDEX)
                    - across_sessions.count(actual.CANDIDATE_ROLE_INDEX)
                )
                == 1
            )


def test_session_arrays_reject_output_counter_and_schedule_tamper() -> None:
    arrays = _arrays(0)
    actual.validate_session_arrays(arrays, session_index=0)

    output = {name: value.copy() for name, value in arrays.items()}
    output["output_token_ids"][0, 0, 1, 0] = 1
    with pytest.raises(ValueError, match="token identity"):
        actual.validate_session_arrays(output, session_index=0)

    counter = {name: value.copy() for name, value in arrays.items()}
    counter["bonus_tokens"][0, 0, 1] += 1
    with pytest.raises(ValueError, match="candidate counter"):
        actual.validate_session_arrays(counter, session_index=0)

    source_counter = {name: value.copy() for name, value in arrays.items()}
    source_counter["corpus_accepted_draft_tokens"][0, 0, 1] -= 1
    with pytest.raises(ValueError, match="candidate counter"):
        actual.validate_session_arrays(source_counter, session_index=0)

    empty_source = {name: value.copy() for name, value in arrays.items()}
    index = (0, 0, actual.CANDIDATE_ROLE_INDEX)
    empty_source["target_generation_forward_calls"][index] = 66
    empty_source["no_proposal_calls"][index] = 34
    empty_source["corpus_proposal_calls"][index] = 1
    empty_source["corpus_proposed_tokens"][index] = 0
    empty_source["corpus_accepted_draft_tokens"][index] = 0
    empty_source["prompt_proposal_calls"][index] = 31
    empty_source["prompt_proposed_tokens"][index] = 62
    empty_source["prompt_accepted_draft_tokens"][index] = 62
    empty_source["proposal_attempts"][index] = 32
    empty_source["proposed_tokens"][index] = 62
    empty_source["accepted_draft_tokens"][index] = 62
    with pytest.raises(ValueError, match="candidate counter"):
        actual.validate_session_arrays(empty_source, session_index=0)

    schedule = {name: value.copy() for name, value in arrays.items()}
    schedule["first_role"][0, 0] = 1 - schedule["first_role"][0, 0]
    with pytest.raises(ValueError, match="first-role"):
        actual.validate_session_arrays(schedule, session_index=0)


def test_primary_gate_passes_and_each_predeclared_clause_can_fail(monkeypatch) -> None:
    passing = [_arrays(index) for index in range(actual.SESSIONS)]
    summary = actual.summarize_actual_arrays(passing, correctness_pass=True)
    assert summary["primary_gate"]["overall_pass"] is True
    assert summary["primary_end_to_end"]["median_reduction"] == pytest.approx(0.2)
    assert summary["role_order_diagnostic"]["baseline_first"]["paired_trial_count"] == (
        actual.SESSIONS * actual.MEASURED_CASES * actual.INNER_REPETITIONS // 2
    )
    assert summary["role_order_diagnostic"]["candidate_first"][
        "median_reduction"
    ] == pytest.approx(0.2)

    session_failure = [_arrays(index) for index in range(actual.SESSIONS)]
    session_failure[-1] = _arrays(
        actual.SESSIONS - 1, baseline_ns=1_000_000_000, candidate_ns=1_100_000_000
    )
    failed = actual.summarize_actual_arrays(session_failure, correctness_pass=True)
    assert failed["primary_gate"]["all_five_sessions_positive"] is False
    assert failed["primary_gate"]["overall_pass"] is False

    below_target = [
        _arrays(index, baseline_ns=1_000_000_000, candidate_ns=950_000_000)
        for index in range(actual.SESSIONS)
    ]
    failed = actual.summarize_actual_arrays(below_target, correctness_pass=True)
    assert failed["primary_gate"]["point_reduction_at_least_10_percent"] is False
    assert failed["primary_gate"]["overall_pass"] is False

    monkeypatch.setattr(actual, "_crossed_bootstrap", lambda *args: (-0.01, 0.30))
    failed = actual.summarize_actual_arrays(passing, correctness_pass=True)
    assert failed["primary_gate"]["bootstrap_lower_strictly_positive"] is False
    assert failed["primary_gate"]["overall_pass"] is False

    monkeypatch.setattr(actual, "_crossed_bootstrap", lambda *args: (0.01, 0.30))
    prompt_failure = [
        _arrays(index, baseline_ns=1_000_000_000, candidate_ns=800_000_000)
        for index in range(actual.SESSIONS)
    ]
    for arrays in prompt_failure:
        arrays["generation_ns"][47:, :, actual.CANDIDATE_ROLE_INDEX] = 990_000_000
        arrays["elapsed_ns"][47:, :, actual.CANDIDATE_ROLE_INDEX] = 1_010_000_000
    failed = actual.summarize_actual_arrays(prompt_failure, correctness_pass=True)
    assert failed["primary_gate"]["positive_prompts_at_least_48"] is False
    assert failed["primary_end_to_end"]["positive_prompt_count"] == 47
    assert failed["primary_gate"]["overall_pass"] is False

    incorrect = actual.summarize_actual_arrays(passing, correctness_pass=False)
    assert incorrect["primary_gate"]["correctness"] is False
    assert incorrect["primary_gate"]["overall_pass"] is False


def test_crossed_session_prompt_bootstrap_has_a_fixed_golden_result() -> None:
    session = np.arange(actual.SESSIONS, dtype=np.float64)[:, None]
    prompt = np.arange(actual.MEASURED_CASES, dtype=np.float64)[None, :]
    baseline = 1.0 + session * 0.01 + prompt * 0.001
    candidate = baseline * (0.8 + ((prompt % 7) - 3) * 0.01 + (session - 2) * 0.005)
    assert actual._crossed_bootstrap(candidate, baseline) == pytest.approx(
        (0.19122043519394505, 0.21008651680182486), abs=1e-15
    )


def test_receipt_roundtrip_and_memory_tamper(isolated_contract) -> None:
    environment, model_identity, _ = isolated_contract
    plan = actual.build_plan(git_commit_before_plan="a" * 40)
    actual.PLAN_PATH.write_bytes(canonical_bytes(plan))
    arrays = _arrays(0)
    artifact = npz_bytes(arrays)
    model = {
        **model_identity,
        "retrieval_table_loaded": True,
        "table_resident_bytes": 4_000_000,
    }
    receipt = actual.build_session_receipt(
        plan=plan,
        session_index=0,
        runner_git_commit="b" * 40,
        process_start_token_sha256="c" * 64,
        arrays=arrays,
        artifact_bytes=artifact,
        model_identity=model,
        operational_start=_operational(),
        operational_end=_operational(),
        operational_checkpoints=[_operational() for _ in range(7)],
        memory=_memory(),
        warmup_output_root_sha256="e" * 64,
    )
    actual.session_artifact_path(0).parent.mkdir(parents=True, exist_ok=True)
    actual.session_artifact_path(0).write_bytes(artifact)
    actual.validate_session_receipt(
        json.loads(canonical_bytes(receipt)),
        plan=plan,
        session_index=0,
        verify_artifact=True,
    )
    table_tamper = deepcopy(receipt)
    table_tamper["model_identity"]["table_resident_bytes"] += 4
    unsigned = dict(table_tamper)
    unsigned.pop("receipt_sha256")
    table_tamper["receipt_sha256"] = actual.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="session receipt"):
        actual.validate_session_receipt(
            table_tamper, plan=plan, session_index=0, verify_artifact=False
        )

    tampered = deepcopy(receipt)
    tampered["memory"]["working_set_fraction"] = 0.4
    unsigned = dict(tampered)
    unsigned.pop("receipt_sha256")
    tampered["receipt_sha256"] = actual.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="memory identity"):
        actual.validate_session_receipt(
            tampered, plan=plan, session_index=0, verify_artifact=False
        )

    unsafe = deepcopy(receipt)
    unsafe["memory"].update(
        {
            "active_after_bytes": 780,
            "maximum_observed_working_set_bytes": 800,
            "peak_active_bytes": 800,
            "process_peak_rss_bytes": 800,
            "safety_pass": False,
            "working_set_fraction": 0.8,
        }
    )
    unsigned = dict(unsafe)
    unsigned.pop("receipt_sha256")
    unsafe["receipt_sha256"] = actual.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="memory identity"):
        actual.validate_session_receipt(
            unsafe, plan=plan, session_index=0, verify_artifact=False
        )
    assert environment == plan["environment"]


def test_summary_roundtrip_and_schema_tamper(isolated_contract) -> None:
    plan = actual.build_plan(git_commit_before_plan="a" * 40)
    actual.PLAN_PATH.write_bytes(canonical_bytes(plan))
    statistics = actual.summarize_actual_arrays(
        [_arrays(index) for index in range(actual.SESSIONS)],
        correctness_pass=True,
    )
    lineage = [
        {
            "artifact_sha256": f"{index + 1:064x}",
            "receipt_artifact_sha256": f"{index + 11:064x}",
            "receipt_publication_git_commit": f"{index + 21:040x}",
            "receipt_sha256": f"{index + 31:064x}",
            "runner_git_commit": f"{index + 41:040x}",
            "session_index": index,
        }
        for index in range(actual.SESSIONS)
    ]
    replay = {
        "independent_checkpoint_forward_replay": True,
        "measured_case_count": actual.MEASURED_CASES,
        "replay_root_sha256": "e" * 64,
        "stored_trial_comparisons": (
            actual.SESSIONS
            * actual.MEASURED_CASES
            * actual.INNER_REPETITIONS
            * len(actual.ROLES)
        ),
        "warmup_session_root_comparisons": actual.SESSIONS,
    }
    memory = {
        "all_session_memory_safety_pass": True,
        "baseline_trial_peak_active_bytes_maximum": 500,
        "baseline_trial_peak_active_bytes_median": 450.0,
        "candidate_trial_peak_active_bytes_maximum": 510,
        "candidate_trial_peak_active_bytes_median": 460.0,
        "claim_scope": "descriptive_only_not_a_memory_improvement_gate",
        "session_working_set_fraction_maximum": 0.5,
    }
    summary = actual.build_actual_summary(
        plan=plan,
        summary_base_git_commit="f" * 40,
        plan_publication_git_commit="e" * 40,
        session_lineage=lineage,
        independent_replay=replay,
        statistics=statistics,
        memory=memory,
    )
    actual.validate_actual_summary(summary, plan=plan)

    actual.SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    actual.SUMMARY_PATH.write_bytes(canonical_bytes(summary))
    with pytest.raises(ValueError, match="derived summary evidence"):
        actual.read_actual_summary(plan=plan, verify_derived=True)
    assert (
        actual.read_actual_summary(
            plan=plan,
            verify_derived=True,
            expected_lineage=lineage,
            expected_replay=replay,
            expected_statistics=statistics,
            expected_memory=memory,
        )
        == summary
    )

    nested_tamper = deepcopy(summary)
    nested_tamper["statistics"]["primary_end_to_end"][
        "baseline_cell_median_seconds"
    ] += 1.0
    unsigned = dict(nested_tamper)
    unsigned.pop("summary_sha256")
    nested_tamper["summary_sha256"] = actual.canonical_sha256(unsigned)
    actual.SUMMARY_PATH.write_bytes(canonical_bytes(nested_tamper))
    with pytest.raises(ValueError, match="statistics reconstruction"):
        actual.read_actual_summary(
            plan=plan,
            verify_derived=True,
            expected_lineage=lineage,
            expected_replay=replay,
            expected_statistics=statistics,
            expected_memory=memory,
        )
    actual.SUMMARY_PATH.write_bytes(canonical_bytes(summary))

    tampered = deepcopy(summary)
    tampered["protocol_id"] = "rotated"
    unsigned = dict(tampered)
    unsigned.pop("summary_sha256")
    tampered["summary_sha256"] = actual.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="summary identity"):
        actual.validate_actual_summary(tampered, plan=plan)

    require_distinct = actual.require_distinct_git_commits
    with pytest.raises(ValueError, match="strict Git chronology"):
        require_distinct("a" * 40, "a" * 40, label="test")


def test_namespace_symlink_is_rejected(isolated_contract, tmp_path) -> None:
    target = tmp_path / "source"
    link = tmp_path / "linked-artifacts"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        actual.assert_canonical_workspace_path(link / "session-0.npz")


def test_candidate_runtime_derives_exact_acceptance_and_timer_counters(
    monkeypatch,
) -> None:
    bundle = SimpleNamespace(
        model=object(), table=object(), tokenizer=SimpleNamespace()
    )
    monkeypatch.setattr(runtime, "_prompt_text", lambda tokenizer, ids: "prompt")
    monkeypatch.setattr(
        bundle.tokenizer,
        "encode",
        lambda text, add_special_tokens=False: [10, 11],
        raising=False,
    )
    monkeypatch.setattr(
        bundle.tokenizer,
        "decode",
        lambda *args, **kwargs: "decoded",
        raising=False,
    )
    monkeypatch.setattr(runtime, "_validate_decoded_sequence", lambda *args: "f" * 64)
    monkeypatch.setattr(runtime.mx, "synchronize", lambda: None)
    clock = iter((10, 20, 30, 40))
    monkeypatch.setattr(runtime, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(
        runtime,
        "hybrid_retrieval_proposal",
        lambda table, history: ((1, 2, 3), "corpus_ngram"),
    )

    def fake_forced(model, prompt, **kwargs):
        assert kwargs["proposal_provider"](tuple(prompt), 4, 0) == (1, 2, 3)
        return ForcedSpeculativeTrace(
            token_ids=(1, 2, 3, 4),
            target_forward_calls=1,
            full_accept_cycles=1,
            immediate_reject_cycles=0,
            partial_accept_cycles=0,
            final_cache_offset=5,
        )

    monkeypatch.setattr(runtime, "forced_speculative_generate", fake_forced)
    result = runtime.run_actual_candidate_trial(
        bundle, (10, 11), output_tokens=4, maximum_draft_tokens=3
    )
    assert result.accepted_draft_tokens == 3
    assert result.corpus_accepted_draft_tokens == 3
    assert result.corpus_proposed_tokens == 3
    assert result.bonus_tokens == 1
    assert result.proposal_attempts == 1
    assert result.corpus_proposal_calls == 1
    assert result.target_generation_forward_calls == 1
    assert result.elapsed_ns == 30
    assert (
        result.no_proposal_calls
        + result.accepted_draft_tokens
        + result.correction_tokens
        + result.bonus_tokens
        == 4
    )


def test_actual_evidence_entrypoints_have_no_result_selecting_cli() -> None:
    for path in (
        "scripts/seal_exaone_retrieval_actual_plan.py",
        "scripts/run_exaone_retrieval_actual_session.py",
        "scripts/summarize_exaone_retrieval_actual.py",
        "scripts/verify_exaone_retrieval_actual_summary.py",
    ):
        source = actual.ROOT.joinpath(path).read_text(encoding="utf-8")
        assert "argparse" not in source
        assert "--force" not in source
        assert "--output" not in source
    assert len(actual.IMPLEMENTATION_PATHS) == len(set(actual.IMPLEMENTATION_PATHS))
