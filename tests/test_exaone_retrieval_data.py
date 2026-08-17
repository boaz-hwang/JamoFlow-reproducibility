from __future__ import annotations

import ast
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import exaone_retrieval_data as data
import numpy as np
import pytest


class _CharacterTokenizer:
    vocab_size = data.VOCABULARY_SIZE
    all_special_ids: tuple[int, ...] = ()

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(int(value)) for value in token_ids)


class _PromptBoundaryUnsafeTokenizer(_CharacterTokenizer):
    def decode(
        self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        decoded = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        if len(token_ids) == data.PROMPT_TOKENS and int(token_ids[0]) == ord("나"):
            return decoded + "\ufffd"
        return decoded


def _records(count: int = 80):
    return [
        SimpleNamespace(
            text=("가" * 300) + f"문서{i}",
            raw=(("가" * 300) + f"문서{i}").encode("utf-8"),
        )
        for i in range(count)
    ]


@pytest.fixture
def stub_plan_inputs(monkeypatch):
    dependency_paths = {
        "compatibility_plan": data.COMPATIBILITY_PLAN_PATH,
        "compatibility_result": data.COMPATIBILITY_RESULT_PATH,
        "evaluation_seal": data.EVALUATION_SEAL_PATH,
        "evaluation_source": data.EVALUATION_SOURCE_PATH,
        "train_seal": data.TRAIN_SEAL_PATH,
        "train_source": data.TRAIN_SOURCE_PATH,
    }
    dependencies = {
        name: {
            "bytes": index + 1,
            "path": path.relative_to(data.ROOT).as_posix(),
            "sha256": f"{index + 1:064x}",
        }
        for index, (name, path) in enumerate(dependency_paths.items())
    }
    model_files = {
        name: {"bytes": index + 1, "sha256": f"{index + 10:064x}"}
        for index, name in enumerate(data.TOKENIZER_RUNTIME_FILES)
    }
    compatibility = {
        "model_files": model_files,
        "plan_sha256": "a" * 64,
        "result_summary_sha256": "b" * 64,
        "status": "pass_primary_greedy_transaction_compatibility",
        "tokenizer": {
            "chat_template_deterministic": True,
            "direct_roundtrip_count": 3,
            "direct_roundtrip_exact": True,
            "prompt_token_count": 10,
            "vocab_size": data.VOCABULARY_SIZE,
        },
    }
    environment = {
        "machine": "arm64",
        "numpy": "fixture",
        "packages": {
            "mlx": "fixture",
            "mlx-lm": "fixture",
            "transformers": "fixture",
        },
        "platform": "fixture",
        "python": "fixture",
        "python_executable_name": "python",
    }
    implementation = {path: "c" * 64 for path in data.IMPLEMENTATION_PATHS}
    monkeypatch.setattr(data, "dependency_identity", lambda: dependencies)
    monkeypatch.setattr(data, "compatibility_projection", lambda: compatibility)
    monkeypatch.setattr(data, "environment_identity", lambda: environment)
    monkeypatch.setattr(data, "implementation_identity", lambda: implementation)
    return dependencies, compatibility


def test_large_vocabulary_context_does_not_use_overflowing_v4_pack() -> None:
    value = data.pack_context((102_399, 102_399, 102_399))
    assert value == data.VOCABULARY_SIZE**3 - 1
    assert value < 2**64
    assert data.VOCABULARY_SIZE**4 > 2**64
    with pytest.raises(ValueError, match="context length"):
        data.pack_context((1, 2, 3, 4))


def test_uint32_next_token_table_build_and_lookup_are_exact() -> None:
    pattern = np.asarray([70_000, 80_000, 90_000, 100_000], dtype=np.uint32)
    values = np.tile(pattern, 40)
    table = data.build_compact_backoff_table(values)
    table.validate()
    assert all(
        table.by_order[order].next_tokens.dtype == np.uint32
        for order in data.TABLE_ORDERS
    )
    assert table.next_token((70_000, 80_000)) == 90_000
    assert table.propose((70_000, 80_000)) == (90_000, 100_000, 70_000)
    restored = data.table_from_arrays(table.to_arrays())
    assert data.table_report(restored) == data.table_report(table)


def _context_row(values: list[int], context: int) -> tuple[int, int] | None:
    rows = data.best_continuations(np.asarray(values, dtype=np.uint32), 1)
    matches = np.flatnonzero(rows["context"] == context)
    if len(matches) == 0:
        return None
    index = int(matches[0])
    return int(rows["next"][index]), int(rows["best_count"][index])


def _pair_stream(counts: tuple[tuple[int, int], ...]) -> list[int]:
    values: list[int] = []
    filler = 1_000
    for next_token, count in counts:
        for _ in range(count):
            values.extend((10, next_token, filler))
            filler += 1
    return values


def test_winning_count_boundary_and_smallest_token_tie_are_exact(monkeypatch) -> None:
    assert _context_row(_pair_stream(((20, 4), (21, 1))), 10) is None
    assert _context_row(_pair_stream(((20, 5), (21, 1))), 10) == (20, 5)
    monkeypatch.setattr(data, "MINIMUM_NEXT_TOKEN_PROBABILITY", 0.5)
    assert _context_row(_pair_stream(((21, 5), (20, 5))), 10) == (20, 5)


def test_hybrid_lookup_uses_corpus_then_longest_earliest_prompt_then_none() -> None:
    pattern = np.asarray([70_000, 80_000, 90_000, 100_000], dtype=np.uint32)
    table = data.build_compact_backoff_table(np.tile(pattern, 40))
    assert data.hybrid_retrieval_proposal(table, (70_000, 80_000)) == (
        (90_000, 100_000, 70_000),
        "corpus_ngram",
    )
    assert data.hybrid_retrieval_proposal(table, (1, 2, 9, 1, 2)) == (
        (9, 1, 2),
        "prompt_lookup",
    )
    assert data.hybrid_retrieval_proposal(table, (1, 2, 3)) == ((), "none")


def test_uint32_guards_reject_before_cast() -> None:
    assert data.token_ids_uint32([0, data.VOCABULARY_SIZE - 1], label="test").dtype == np.uint32
    with pytest.raises(ValueError, match="token id"):
        data.token_ids_uint32([2**32 + 1], label="test")
    assert int(data.checked_uint32_count(2**32 - 1, label="test")) == 2**32 - 1
    with pytest.raises(OverflowError, match="uint32"):
        data.checked_uint32_count(2**32, label="test")


def test_case_selection_is_ranked_unique_roundtrip_and_hangul_heavy() -> None:
    rank_key = b"k" * 32
    arrays, report = data.build_case_arrays(
        _records(), _CharacterTokenizer(), rank_key=rank_key
    )
    data.validate_case_arrays(arrays, rank_key=rank_key)
    assert report["eligible_document_count"] == 80
    assert report["total_case_count"] == data.TOTAL_CASES
    assert report["ordered_case_commitment_sha256"] == data.ordered_case_commitment(
        arrays, rank_key=rank_key
    )
    assert len({bytes(row) for row in arrays["case_digest"]}) == data.TOTAL_CASES
    assert np.all(arrays["prompt_token_ids"] == ord("가"))


def test_case_selection_rejects_prompt_boundary_that_only_full_prefix_hides() -> None:
    records = _records()
    records[0] = SimpleNamespace(
        text="나" + ("가" * 299) + "경계",
        raw=("나" + ("가" * 299) + "경계").encode("utf-8"),
    )
    arrays, report = data.build_case_arrays(
        records, _PromptBoundaryUnsafeTokenizer(), rank_key=b"k" * 32
    )
    assert report["eligible_document_count"] == len(records) - 1
    rejected = hashlib.sha256(records[0].raw).digest()
    assert rejected not in {bytes(row) for row in arrays["case_digest"]}


def test_plan_roundtrip_and_metric_tamper_rejection(stub_plan_inputs) -> None:
    plan = data.build_plan(git_commit_before_plan="a" * 40)
    data.validate_plan(plan, verify_derived=True)
    restored = json.loads(data.canonical_bytes(plan))
    data.validate_plan(restored, verify_derived=True)
    tampered = deepcopy(plan)
    tampered["result_inputs"]["candidate_or_baseline_latency"] = True
    unsigned = dict(tampered)
    unsigned.pop("plan_sha256")
    tampered["plan_sha256"] = data.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="plan identity"):
        data.validate_plan(tampered, verify_derived=False)


def test_data_seal_roundtrip_binds_table_and_case_artifacts(
    tmp_path, monkeypatch, stub_plan_inputs
) -> None:
    plan = data.build_plan(git_commit_before_plan="a" * 40)
    plan_path = tmp_path / "plan.json"
    seal_path = tmp_path / "seal.json"
    table_path = tmp_path / "table.npz"
    cases_path = tmp_path / "cases.npz"
    plan_path.write_bytes(data.canonical_bytes(plan))
    pattern = np.asarray([70_000, 80_000, 90_000, 100_000], dtype=np.uint32)
    table = data.build_compact_backoff_table(np.tile(pattern, 40))
    table_arrays = table.to_arrays()
    rank_key = bytes.fromhex(
        plan["data_contract"]["case_selection"]["case_rank_key_hex"]
    )
    case_arrays, case_report = data.build_case_arrays(
        _records(), _CharacterTokenizer(), rank_key=rank_key
    )
    table_path.write_bytes(data.npz_bytes(table_arrays))
    cases_path.write_bytes(data.npz_bytes(case_arrays))
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(data, "PLAN_PATH", plan_path)
    monkeypatch.setattr(data, "SEAL_PATH", seal_path)
    monkeypatch.setattr(data, "TABLE_PATH", table_path)
    monkeypatch.setattr(data, "CASES_PATH", cases_path)
    inventory = {
        "available_train_document_count": data.EXPECTED_AVAILABLE_TRAIN_DOCUMENTS,
        "document_count": data.EXPECTED_AVAILABLE_TRAIN_DOCUMENTS - 1,
        "evaluation_document_count": data.EXPECTED_EVALUATION_DOCUMENTS,
        "full_document_serialized_bytes": data.TRAIN_DOCUMENT_BYTE_BUDGET - 1,
        "newline_token_count": 1,
        "token_count": 160,
        "token_ids_sha256": data.array_sha256(np.tile(pattern, 40)),
        "train_document_byte_budget": data.TRAIN_DOCUMENT_BYTE_BUDGET,
        "train_evaluation_exact_intersection_count": 0,
        "train_evaluation_normalized_intersection_count": 0,
        "selected_train_exact_commitment_sha256": "1" * 64,
        "selected_train_normalized_commitment_sha256": "2" * 64,
        "evaluation_exact_commitment_sha256": "3" * 64,
        "evaluation_normalized_commitment_sha256": "4" * 64,
        "normalized_document_algorithm": (
            "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
        ),
    }
    tokenizer_runtime = {
        "compatibility_result_summary_sha256": plan["compatibility"][
            "result_summary_sha256"
        ],
        "files": plan["compatibility"]["model_files"],
        "repo_id": data.PRIMARY_MODEL["repo_id"],
        "revision": data.PRIMARY_MODEL["revision"],
        "tokenizer_class": "FixtureTokenizer",
        "vocabulary_size": data.VOCABULARY_SIZE,
    }
    seal = data.build_seal(
        plan=plan,
        builder_git_commit="b" * 40,
        table=table,
        table_arrays=table_arrays,
        case_arrays=case_arrays,
        case_report=case_report,
        tokenizer_runtime=tokenizer_runtime,
        training_inventory=inventory,
    )
    data.validate_seal(seal, plan=plan, verify_artifacts=True)
    seal_path.write_bytes(data.canonical_bytes(seal))
    verification = data.build_verification(
        plan=plan,
        seal=seal,
        verifier_git_commit="d" * 40,
        reconstructed_table_arrays=table_arrays,
        reconstructed_case_arrays=case_arrays,
        reconstructed_training_inventory=inventory,
        tokenizer_runtime=tokenizer_runtime,
    )
    data.validate_verification(verification, plan=plan, seal=seal)
    disconnected = deepcopy(verification)
    disconnected["reconstruction"]["table_array_sha256"]["order_1_next"] = (
        "e" * 64
    )
    unsigned_verification = dict(disconnected)
    unsigned_verification.pop("verification_sha256")
    disconnected["verification_sha256"] = data.canonical_sha256(
        unsigned_verification
    )
    with pytest.raises(ValueError, match="verification differs"):
        data.validate_verification(disconnected, plan=plan, seal=seal)
    tampered = deepcopy(seal)
    tampered["case_artifact"]["sha256"] = "f" * 64
    unsigned = dict(tampered)
    unsigned.pop("seal_sha256")
    tampered["seal_sha256"] = data.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="artifact differs"):
        data.validate_seal(tampered, plan=plan, verify_artifacts=True)


def test_npz_serialization_is_deterministic() -> None:
    arrays = {
        "b": np.asarray([3, 4], dtype=np.uint32),
        "a": np.asarray([1, 2], dtype=np.uint64),
    }
    assert data.npz_bytes(arrays) == data.npz_bytes(dict(reversed(list(arrays.items()))))


def test_data_builder_has_no_model_forward_or_timing_api() -> None:
    for relative in (
        "scripts/prepare_exaone_retrieval_data.py",
        "scripts/verify_exaone_retrieval_data.py",
    ):
        source = (data.ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "time" not in imported
        assert "mlx.core" not in imported
        assert "mlx_lm.load" not in source
        assert "perf_counter" not in source
        assert "candidate_latency" not in source
