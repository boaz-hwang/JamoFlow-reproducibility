from __future__ import annotations

import json
import runpy
from copy import deepcopy

import exaone_actual_runtime as actual_runtime
import exaone_resource_calibration as resource
import numpy as np
import pytest
from exaone_retrieval_data import canonical_bytes, canonical_sha256, npz_bytes
from large_model_retrieval_preflight import token_sequence_sha256


def _environment() -> dict:
    return {
        "schema_version": 1,
        "python": "fixture",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "macos_version": "fixture",
        "packages": {
            "huggingface-hub": "fixture",
            "mlx": "0.31.2",
            "mlx-lm": "0.31.3",
            "numpy": "fixture",
            "tokenizers": "fixture",
            "transformers": "fixture",
        },
        "mlx": {
            "default_device": "Device(gpu, 0)",
            "metal_available": True,
            "device_name": "fixture",
            "architecture": "fixture",
            "memory_size": 2_000,
            "max_recommended_working_set_size": 1_000,
        },
    }


@pytest.fixture
def isolated_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "ROOT", tmp_path)
    monkeypatch.setattr(resource, "CASES_PATH", tmp_path / "ignored/cases.npz")
    monkeypatch.setattr(resource, "DATA_PLAN_PATH", tmp_path / "data-plan.json")
    monkeypatch.setattr(resource, "DATA_SEAL_PATH", tmp_path / "data-seal.json")
    monkeypatch.setattr(
        resource, "DATA_VERIFICATION_PATH", tmp_path / "data-verification.json"
    )
    monkeypatch.setattr(resource, "INVALIDATED_V1_PLAN_PATH", tmp_path / "v1-plan.json")
    monkeypatch.setattr(
        resource, "INVALIDATED_V1_RESULT_PATH", tmp_path / "v1-invalidation.json"
    )
    monkeypatch.setattr(resource, "INVALIDATED_V2_PLAN_PATH", tmp_path / "v2-plan.json")
    monkeypatch.setattr(
        resource, "INVALIDATED_V2_RESULT_PATH", tmp_path / "v2-invalidation.json"
    )
    monkeypatch.setattr(resource, "PLAN_PATH", tmp_path / "resource-plan.json")
    monkeypatch.setattr(
        resource, "BASELINE_ARTIFACT_PATH", tmp_path / "ignored/baseline.npz"
    )
    monkeypatch.setattr(resource, "RESULT_PATH", tmp_path / "summary.json")

    dependency_paths = {
        "case_artifact": resource.CASES_PATH,
        "data_plan": resource.DATA_PLAN_PATH,
        "data_seal": resource.DATA_SEAL_PATH,
        "data_verification": resource.DATA_VERIFICATION_PATH,
        "invalidated_v1_plan": resource.INVALIDATED_V1_PLAN_PATH,
        "invalidated_v1_result": resource.INVALIDATED_V1_RESULT_PATH,
        "invalidated_v2_plan": resource.INVALIDATED_V2_PLAN_PATH,
        "invalidated_v2_result": resource.INVALIDATED_V2_RESULT_PATH,
    }
    dependencies = {
        name: {
            "bytes": index + 1,
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": f"{index + 1:064x}",
        }
        for index, (name, path) in enumerate(dependency_paths.items())
    }
    implementation = {path: "b" * 64 for path in resource.IMPLEMENTATION_PATHS}
    environment = _environment()
    model_files = {
        name: {"bytes": index + 1, "sha256": f"{index + 10:064x}"}
        for index, name in enumerate(resource.PRIMARY_MODEL["expected_files"])
    }
    compatibility = {
        "model_files": model_files,
        "memory": {"model_parameters": 123},
    }
    monkeypatch.setattr(resource, "dependency_identity", lambda: dependencies)
    monkeypatch.setattr(resource, "implementation_identity", lambda: implementation)
    monkeypatch.setattr(resource, "environment_identity", lambda: environment)
    monkeypatch.setattr(resource, "validate_environment", lambda value: None)
    monkeypatch.setattr(
        resource, "read_validated_compatibility_result", lambda: compatibility
    )
    return dependencies, environment, model_files


def _arrays(*, elapsed_ns: int = 1_000_000_000) -> dict[str, np.ndarray]:
    output_ids = np.zeros(
        (resource.TOTAL_CASES, resource.OUTPUT_TOKENS), dtype=np.uint32
    )
    output_hash = np.asarray(
        [
            list(
                bytes.fromhex(token_sequence_sha256(tuple(int(token) for token in row)))
            )
            for row in output_ids
        ],
        dtype=np.uint8,
    )
    return {
        "decoded_utf8_sha256": np.ones((resource.TOTAL_CASES, 32), dtype=np.uint8),
        "elapsed_ns": np.full(resource.TOTAL_CASES, elapsed_ns, dtype=np.int64),
        "output_token_ids": output_ids,
        "output_token_sha256": output_hash,
        "prompt_token_count": np.full(resource.TOTAL_CASES, 128, dtype=np.uint16),
        "target_generation_forward_calls": np.full(
            resource.TOTAL_CASES, resource.OUTPUT_TOKENS, dtype=np.uint16
        ),
        "target_prefill_forward_calls": np.ones(resource.TOTAL_CASES, dtype=np.uint8),
    }


def _memory() -> dict:
    return {
        "active_after_bytes": 450,
        "active_before_bytes": 400,
        "cache_after_bytes": 30,
        "cache_before_bytes": 20,
        "maximum_allowed_bytes": 750,
        "maximum_observed_working_set_bytes": 500,
        "maximum_recommended_working_set_size": 1_000,
        "peak_active_bytes": 500,
        "process_peak_rss_bytes": 300,
        "safety_pass": True,
        "working_set_fraction": 0.5,
    }


def test_plan_roundtrip_and_candidate_input_tamper(isolated_contract) -> None:
    plan = resource.build_plan(git_commit_before_plan="a" * 40)
    restored = json.loads(canonical_bytes(plan))
    resource.validate_plan(restored, verify_derived=True)
    assert plan["resource_contract"]["candidate_executed"] is False
    assert plan["result_inputs"]["retrieval_table_loaded"] is False

    tampered = deepcopy(plan)
    tampered["result_inputs"]["candidate_latency"] = True
    unsigned = dict(tampered)
    unsigned.pop("plan_sha256")
    tampered["plan_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="plan identity"):
        resource.validate_plan(tampered, verify_derived=False)


def test_runtime_uses_the_exact_mlx_exaone_config_schema() -> None:
    expected = deepcopy(resource.PRIMARY_MODEL["config_projection"])
    assert actual_runtime._config_projection(expected) == expected
    wrong_alias = deepcopy(expected)
    wrong_alias["num_hidden_layers"] = wrong_alias.pop("num_layers")
    with pytest.raises(ValueError, match="loaded config"):
        actual_runtime._config_projection(wrong_alias)


def test_v1_invalidation_is_publicly_reconstructable() -> None:
    invalidation = resource.read_invalidated_v1()
    assert invalidation["baseline_trial_count"] == 0
    assert invalidation["candidate_executed"] is False
    assert (
        invalidation["active_marker_payload"]["runner_git_commit"]
        == (resource.INVALIDATED_V1_IDENTITY["runner_git_commit"])
    )


def test_v2_invalidation_is_publicly_reconstructable() -> None:
    invalidation = resource.read_invalidated_v2()
    assert invalidation["baseline_generation_entered"] is True
    assert invalidation["baseline_numeric_latency_exposed"] is False
    assert invalidation["candidate_executed"] is False
    assert (
        invalidation["active_marker_payload"]["runner_git_commit"]
        == resource.INVALIDATED_V2_IDENTITY["runner_git_commit"]
    )


def test_schedule_uses_first_feasible_predeclared_choice() -> None:
    first = resource.select_actual_schedule(
        model_load_seconds=5.0,
        warmup_total_seconds=10.0,
        measured_total_seconds=100.0,
    )
    assert first["selected"] == first["projections"][0]
    assert first["selected"]["sessions"] == 5
    assert first["selected"]["inner_repetitions"] == 3

    reduced = resource.select_actual_schedule(
        model_load_seconds=10.0,
        warmup_total_seconds=100.0,
        measured_total_seconds=2_000.0,
    )
    assert reduced["selected"] == reduced["projections"][3]
    assert reduced["selected"]["sessions"] == 3
    assert reduced["selected"]["inner_repetitions"] == 1

    infeasible = resource.select_actual_schedule(
        model_load_seconds=10.0,
        warmup_total_seconds=100.0,
        measured_total_seconds=4_000.0,
    )
    assert infeasible["status"] == "infeasible"
    assert infeasible["selected"] is None


def test_baseline_array_hash_and_counter_tamper_are_rejected() -> None:
    arrays = _arrays()
    resource.validate_baseline_arrays(arrays)

    token_tamper = {name: value.copy() for name, value in arrays.items()}
    token_tamper["output_token_ids"][0, 0] = 1
    with pytest.raises(ValueError, match="output-token hashes"):
        resource.validate_baseline_arrays(token_tamper)

    counter_tamper = {name: value.copy() for name, value in arrays.items()}
    counter_tamper["target_generation_forward_calls"][0] -= 1
    with pytest.raises(ValueError, match="counters"):
        resource.validate_baseline_arrays(counter_tamper)


def test_result_reconstructs_artifact_timing_and_model_identity(
    isolated_contract,
) -> None:
    _, _, model_files = isolated_contract
    plan = resource.build_plan(git_commit_before_plan="a" * 40)
    resource.PLAN_PATH.write_bytes(canonical_bytes(plan))
    arrays = _arrays(elapsed_ns=2_000_000_000)
    artifact_bytes = npz_bytes(arrays)
    model_identity = {
        "model_files": model_files,
        "model_parameter_count": 123,
        "retrieval_table_loaded": False,
        "table_resident_bytes": 0,
    }
    result = resource.build_result(
        plan=plan,
        runner_git_commit="c" * 40,
        arrays=arrays,
        baseline_artifact_bytes=artifact_bytes,
        model_load_seconds=4.0,
        memory=_memory(),
        model_identity=model_identity,
    )
    resource.BASELINE_ARTIFACT_PATH.parent.mkdir(parents=True)
    resource.BASELINE_ARTIFACT_PATH.write_bytes(artifact_bytes)
    resource.validate_result(result, plan=plan, verify_artifact=True)

    timing_tamper = deepcopy(result)
    timing_tamper["timing"]["measured_case_median_seconds"] += 0.25
    unsigned = dict(timing_tamper)
    unsigned.pop("summary_sha256")
    timing_tamper["summary_sha256"] = canonical_sha256(unsigned)
    resource.validate_result(timing_tamper, plan=plan, verify_artifact=False)
    with pytest.raises(ValueError, match="timing reconstruction"):
        resource.validate_result(timing_tamper, plan=plan, verify_artifact=True)

    model_tamper = deepcopy(result)
    first = next(iter(model_tamper["model_identity"]["model_files"].values()))
    first["sha256"] = "f" * 64
    unsigned = dict(model_tamper)
    unsigned.pop("summary_sha256")
    model_tamper["summary_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="model identity"):
        resource.validate_result(model_tamper, plan=plan, verify_artifact=False)


def test_runner_is_structurally_baseline_only() -> None:
    source = (resource.ROOT / "scripts/run_exaone_resource_calibration.py").read_text(
        encoding="utf-8"
    )
    assert "run_candidate_trial" not in source
    assert "load_table=False" in source
    assert "load_table=True" not in source
    assert "candidate timing and acceptance were not executed" in source


def test_plan_sealer_module_imports_without_execution() -> None:
    namespace = runpy.run_path(
        resource.ROOT / "scripts/seal_exaone_resource_calibration_plan.py",
        run_name="exaone_resource_sealer_import_test",
    )
    assert callable(namespace["main"])


def test_baseline_timer_contains_encode_generate_decode_and_final_sync(
    monkeypatch,
) -> None:
    events: list[str] = []

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            events.append(f"encode:{text}")
            return [int(value) for value in text.split(",")]

        def decode(
            self,
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            events.append(f"decode:{len(token_ids)}")
            return ",".join(str(int(value)) for value in token_ids)

    def synchronize() -> None:
        events.append("sync")

    clock = iter((100, 250))

    def now() -> int:
        value = next(clock)
        events.append(f"clock:{value}")
        return value

    def greedy(model, prompt_ids, *, maximum_tokens):
        assert model == "model"
        assert tuple(prompt_ids) == (1, 2, 3)
        assert maximum_tokens == 2
        events.append("greedy")
        return (4, 5)

    monkeypatch.setattr(actual_runtime.mx, "synchronize", synchronize)
    monkeypatch.setattr(actual_runtime, "perf_counter_ns", now)
    monkeypatch.setattr(actual_runtime, "greedy_generate", greedy)
    bundle = actual_runtime.ExaoneRuntimeBundle(
        model="model",
        tokenizer=Tokenizer(),
        table=None,
        model_parameter_count=1,
        model_files={},
        table_resident_bytes=0,
    )
    trial = actual_runtime.run_baseline_trial(bundle, (1, 2, 3), output_tokens=2)
    assert trial.elapsed_ns == 150
    assert trial.output_token_ids == (4, 5)
    assert events == [
        "decode:3",
        "encode:1,2,3",
        "sync",
        "clock:100",
        "encode:1,2,3",
        "greedy",
        "decode:5",
        "sync",
        "clock:250",
        "decode:5",
    ]


def test_generated_decode_does_not_require_canonical_bpe_resegmentation() -> None:
    class NonCanonicalTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [99]

        def decode(
            self,
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            assert tuple(token_ids) == (1, 2, 3)
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "같은 문자열"

    digest = actual_runtime._validate_decoded_sequence(
        NonCanonicalTokenizer(), (1,), (2, 3), "같은 문자열"
    )
    assert len(digest) == 64


def test_generated_decode_rejects_nondeterminism_and_surrogates() -> None:
    class Tokenizer:
        def __init__(self, replay):
            self.replay = replay

        def decode(
            self,
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            assert tuple(token_ids) == (1, 2)
            return self.replay

    with pytest.raises(ValueError, match="not deterministic"):
        actual_runtime._validate_decoded_sequence(Tokenizer("나"), (1,), (2,), "가")
    with pytest.raises(ValueError, match="strict UTF-8"):
        actual_runtime._validate_decoded_sequence(
            Tokenizer("\ud800"), (1,), (2,), "\ud800"
        )
