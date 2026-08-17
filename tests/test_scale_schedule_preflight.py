from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

import scripts.scale_schedule_preflight_core as core
from jamoflow.publication_scale import PUBLICATION_EXPECTED_PARAMETERS
from scripts.run_scale_schedule_preflight import _normalized_error


def _correctness(*, valid: bool = True) -> dict[str, object]:
    comparisons = core.CORRECTNESS_PROMPTS * core.CONTINUATION_BYTES
    return {
        "argmax_comparisons": comparisons,
        "argmax_exact": comparisons if valid else comparisons - 1,
        "boundary_prefix_comparisons": comparisons,
        "boundary_trace_exact": True,
        "cache_diagnostics_exact": True,
        "maximum_normalized_logit_error": 0.5,
        "offline_boundary_prefix_exact": True,
    }


def _patch_summary() -> dict[str, object]:
    return {
        "c86": {"maximum": 43, "median": 43.0, "minimum": 42, "sum": 686},
        "w72": {"maximum": 36, "median": 36.0, "minimum": 35, "sum": 574},
    }


def _report(target: int, session: str, *, valid: bool = True) -> dict[str, object]:
    correctness = {
        schedule: _correctness(valid=valid) for schedule in core.SCHEDULE_ORDER
    }
    return {
        "correctness": correctness,
        "environment_end": {"fixture": True},
        "environment_start": {"fixture": True},
        "maximum_driver_allocated_bytes": 500,
        "model_state_sha256": f"{target:064x}",
        "parameter_count": PUBLICATION_EXPECTED_PARAMETERS[target],
        "patch_count_summary": _patch_summary(),
        "recommended_max_memory_bytes": 1_000,
        "same_model_object_for_both_schedules": True,
        "session_id": session,
        "target_millions": target,
    }


def _reports(*, valid: bool = True):
    return {
        target: tuple(
            _report(target, session, valid=valid) for session in core.SESSION_ORDER
        )
        for target in core.TARGET_ORDER
    }


def _timings(candidate: float = 85.0) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    shape = (
        len(core.SESSION_ORDER),
        core.MEASURED_PROMPTS,
        core.INNER_REPETITIONS,
        len(core.SCHEDULE_ORDER),
    )
    for target in core.TARGET_ORDER:
        values = np.empty(shape, dtype=np.float64)
        values[..., core.REFERENCE_INDEX] = 100.0
        values[..., core.CANDIDATE_INDEX] = candidate
        rows[target] = values
    return rows


def test_schedule_and_model_contracts_are_exact() -> None:
    assert core.TARGET_ORDER == (50, 75, 100)
    assert core.SESSION_ORDER == ("session-0", "session-1", "session-2")
    assert core.schedule_contract() == {
        "order": ["c86", "w72"],
        "horizon": 512,
        "fixed_stride": 6,
        "c86": {"patch_count": 86, "policy": "causal_codepoint_grid"},
        "w72": {"patch_count": 72, "policy": "causal_whitespace_grid"},
    }
    assert {
        int(target): row["expected_parameter_count"]
        for target, row in core.model_contract().items()
    } == PUBLICATION_EXPECTED_PARAMETERS
    assert core.is_git_commit("a" * 40)
    assert not core.is_git_commit("a" * 64)


def test_plan_builder_round_trip_rejects_a_sha256_as_git_commit(monkeypatch) -> None:
    cases = {"fixture": True}
    monkeypatch.setattr(core, "case_contract", lambda: cases)
    models = {
        target: {**row, "model_state_sha256": "a" * 64}
        for target, row in core.model_contract().items()
    }
    implementation = {relative: "b" * 64 for relative in core.IMPLEMENTATION_PATHS}
    plan = core.build_scale_schedule_plan(
        git_commit_before_plan="c" * 40,
        models=models,
        environment={"fixture": True},
        implementation_sha256=implementation,
    )
    core.validate_plan(
        plan,
        current_environment={"fixture": True},
        verify_implementation=False,
    )
    with pytest.raises(ValueError, match="plan identity differs"):
        core.build_scale_schedule_plan(
            git_commit_before_plan="c" * 64,
            models=models,
            environment={"fixture": True},
            implementation_sha256=implementation,
        )


def test_role_order_is_balanced_across_targets_sessions_and_prompts() -> None:
    totals = [0, 0]
    for target_index in range(len(core.TARGET_ORDER)):
        for session_index in range(len(core.SESSION_ORDER)):
            session_counts = [0, 0]
            for prompt in range(core.MEASURED_PROMPTS):
                for repetition in range(core.INNER_REPETITIONS):
                    first = core.role_order(
                        target_index, session_index, prompt, repetition
                    )[0]
                    totals[first] += 1
                    session_counts[first] += 1
            assert session_counts == [24, 24]
        for prompt in range(core.MEASURED_PROMPTS):
            for repetition in range(core.INNER_REPETITIONS):
                across_sessions = [
                    core.role_order(target_index, session_index, prompt, repetition)[0]
                    for session_index in range(len(core.SESSION_ORDER))
                ]
                assert sum(across_sessions) in {1, 2}
    assert totals[0] == totals[1]


def test_mechanism_arrays_are_deterministic_and_w72_reduces_patches() -> None:
    prompts = np.zeros((2, core.PROMPT_BYTES), dtype=np.uint8)
    continuations = np.zeros((2, core.CONTINUATION_BYTES), dtype=np.uint8)
    counts, hashes = core.mechanism_arrays(prompts, continuations)
    repeated_counts, repeated_hashes = core.mechanism_arrays(prompts, continuations)
    assert counts.shape == (2, 2)
    assert hashes.shape == (2, 2, 32)
    assert np.array_equal(counts, repeated_counts)
    assert np.array_equal(hashes, repeated_hashes)
    assert np.all(counts[:, core.CANDIDATE_INDEX] < counts[:, core.REFERENCE_INDEX])
    assert not np.array_equal(
        hashes[:, core.CANDIDATE_INDEX], hashes[:, core.REFERENCE_INDEX]
    )


def test_case_filter_selects_distinct_nonoverlapping_documents_in_pool_order() -> None:
    offsets = np.asarray([0, 100, *range(300, 6_301, 300)], dtype=np.int64)
    spans = tuple(
        (start, start + 255, index) for index, start in enumerate(range(0, 6_301, 300))
    )
    selected, documents = core._select_independent_case_indices(offsets, spans)
    assert len(selected) == core.WARMUP_PROMPTS + core.MEASURED_PROMPTS
    assert selected[:3].tolist() == [0, 2, 3]
    assert len(np.unique(documents)) == len(documents)
    selected_offsets = offsets[selected]
    for index, left in enumerate(selected_offsets):
        for right in selected_offsets[index + 1 :]:
            assert left + 255 <= right or right + 255 <= left


def test_correctness_comparison_rejects_nonfinite_or_broadcast_shapes() -> None:
    assert _normalized_error(torch.zeros(2), torch.zeros(2)) == 0
    with pytest.raises(ValueError, match="structurally"):
        _normalized_error(torch.zeros(2), torch.zeros(1))
    with pytest.raises(ValueError, match="structurally"):
        _normalized_error(torch.tensor([float("nan")]), torch.zeros(1))


def test_primary_pass_requires_all_fixed_clauses() -> None:
    summary = core.summarize_scale_schedule_preflight(
        timings_by_target=_timings(),
        reports_by_target=_reports(),
    )
    primary = summary["rows"]["100"]
    assert summary["status"] == "one_seed_100m_training_authorized"
    assert summary["one_seed_100m_training_authorized"] is True
    assert all(primary["gates"].values())
    assert primary["positive_prompt_count"] == core.MEASURED_PROMPTS
    assert primary["positive_session_count"] == len(core.SESSION_ORDER)
    assert primary["sessions_at_least_10_percent"] == len(core.SESSION_ORDER)
    assert primary["patch_event_reduction"] > 0


def test_point_and_bootstrap_and_prompt_and_session_gates_can_fail() -> None:
    point = core.summarize_scale_schedule_preflight(
        timings_by_target=_timings(candidate=91.0),
        reports_by_target=_reports(),
    )
    assert not point["rows"]["100"]["gates"]["point_reduction_at_least_10_percent"]

    with patch.object(core, "_bootstrap_interval", return_value=(0.079, 0.20)):
        bootstrap = core.summarize_scale_schedule_preflight(
            timings_by_target=_timings(),
            reports_by_target=_reports(),
        )
    assert not bootstrap["rows"]["100"]["gates"]["bootstrap_lower_at_least_8_percent"]

    prompt_timings = _timings()
    prompt_timings[100][:, :2, :, core.CANDIDATE_INDEX] = 105.0
    with patch.object(core, "_bootstrap_interval", return_value=(0.10, 0.20)):
        prompts = core.summarize_scale_schedule_preflight(
            timings_by_target=prompt_timings,
            reports_by_target=_reports(),
        )
    assert not prompts["rows"]["100"]["gates"]["positive_prompts_at_least_15"]

    session_timings = _timings()
    session_timings[100][0, ..., core.CANDIDATE_INDEX] = 105.0
    with patch.object(core, "_bootstrap_interval", return_value=(0.10, 0.20)):
        sessions = core.summarize_scale_schedule_preflight(
            timings_by_target=session_timings,
            reports_by_target=_reports(),
        )
    assert not sessions["rows"]["100"]["gates"]["all_three_sessions_positive"]

    stability_timings = _timings(candidate=91.0)
    stability_timings[100][0, ..., core.CANDIDATE_INDEX] = 85.0
    with patch.object(core, "_bootstrap_interval", return_value=(0.10, 0.20)):
        stability = core.summarize_scale_schedule_preflight(
            timings_by_target=stability_timings,
            reports_by_target=_reports(),
        )
    assert stability["rows"]["100"]["gates"]["all_three_sessions_positive"]
    assert not stability["rows"]["100"]["gates"][
        "at_least_two_sessions_reach_10_percent"
    ]


def test_invalid_evidence_and_timing_arrays_stop_or_raise() -> None:
    summary = core.summarize_scale_schedule_preflight(
        timings_by_target=_timings(),
        reports_by_target=_reports(valid=False),
    )
    assert summary["status"] == "publication_scale_training_stopped"
    assert summary["all_target_evidence_valid"] is False

    malformed = _timings()
    malformed[100] = malformed[100].astype(np.float32)
    with pytest.raises(ValueError, match="shape/dtype differs"):
        core.summarize_scale_schedule_preflight(
            timings_by_target=malformed,
            reports_by_target=_reports(),
        )

    nonfinite = _timings()
    nonfinite[100][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        core.summarize_scale_schedule_preflight(
            timings_by_target=nonfinite,
            reports_by_target=_reports(),
        )


def test_summary_builder_is_canonical_and_tamper_evident() -> None:
    aggregate = core.summarize_scale_schedule_preflight(
        timings_by_target=_timings(),
        reports_by_target=_reports(),
    )
    evidence = {
        str(target): {
            session: {
                "report_path": core.worker_report_path(target, session)
                .relative_to(core.ROOT)
                .as_posix(),
                "report_sha256": "a" * 64,
                "timing_path": core.worker_timing_path(target, session)
                .relative_to(core.ROOT)
                .as_posix(),
                "timing_sha256": "b" * 64,
            }
            for session in core.SESSION_ORDER
        }
        for target in core.TARGET_ORDER
    }
    summary = core.build_scale_schedule_summary(
        plan_artifact_sha256="c" * 64,
        plan_sha256="d" * 64,
        summary_base_git_commit="e" * 40,
        worker_evidence=evidence,
        aggregate=aggregate,
    )
    core.validate_scale_schedule_summary(summary)
    tampered = deepcopy(summary)
    tampered["aggregate"]["primary_100m_pass"] = False
    with pytest.raises(ValueError, match="identity differs"):
        core.validate_scale_schedule_summary(tampered)
    rotated = deepcopy(summary)
    rotated["worker_evidence"]["100"]["session-2"]["timing_path"] = "other.npz"
    rotated_payload = dict(rotated)
    rotated_payload.pop("summary_sha256")
    rotated["summary_sha256"] = core.canonical_sha256(rotated_payload)
    with pytest.raises(ValueError, match="evidence identity differs"):
        core.validate_scale_schedule_summary(rotated)


def test_implementation_manifest_is_complete_and_unique() -> None:
    assert len(core.IMPLEMENTATION_PATHS) == len(set(core.IMPLEMENTATION_PATHS))
    assert all(
        (core.ROOT / relative).is_file() for relative in core.IMPLEMENTATION_PATHS
    )
    assert "scripts/verify_scale_schedule_preflight.py" in core.IMPLEMENTATION_PATHS
    assert "tests/test_scale_schedule_preflight.py" in core.IMPLEMENTATION_PATHS


def test_worker_paths_require_canonical_target_and_session() -> None:
    assert core.worker_report_path(100, "session-2").name == (
        "target-100-session-2-report.json"
    )
    with pytest.raises(ValueError, match="identity differs"):
        core.worker_timing_path(101, "session-0")
    with pytest.raises(ValueError, match="identity differs"):
        core.worker_timing_path(100, "session-9")
