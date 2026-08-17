"""Pure contracts and deterministic data structures for EXAONE retrieval timing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from large_model_retrieval_preflight import (
    PLAN_PATH as COMPATIBILITY_PLAN_PATH,
)
from large_model_retrieval_preflight import (
    PRIMARY_MODEL,
    ROOT,
    canonical_sha256,
    hash_file,
    is_sha256,
    validate_pass_result,
)
from large_model_retrieval_preflight import (
    RESULT_PATH as COMPATIBILITY_RESULT_PATH,
)
from large_model_retrieval_preflight import canonical_bytes as _canonical_bytes
from large_model_retrieval_preflight import (
    read_plan as read_compatibility_plan,
)

PLAN_PATH = ROOT / "data/manifests/exaone-retrieval-data-v1.json"
SEAL_PATH = ROOT / "data/seals/exaone-retrieval-data-v1.json"
VERIFICATION_PATH = ROOT / "data/seals/exaone-retrieval-data-verification-v1.json"
ARTIFACT_ROOT = ROOT / "artifacts/exaone-retrieval-data-v1"
ACTIVE_PATH = ARTIFACT_ROOT / ".active"
VERIFICATION_ACTIVE_PATH = ARTIFACT_ROOT / ".verify-active"
TABLE_PATH = ARTIFACT_ROOT / "compact-token-ngram.npz"
CASES_PATH = ARTIFACT_ROOT / "cases.npz"

TRAIN_SOURCE_PATH = ROOT / "data/processed/hplt3-korean-vocab-adaptation-v2/ko.jsonl"
TRAIN_SEAL_PATH = ROOT / "data/seals/hplt3-korean-vocab-adaptation-v2.json"
EVALUATION_SOURCE_PATH = ROOT / "data/processed/hplt3-korean-final-test-v1/ko.jsonl"
EVALUATION_SEAL_PATH = ROOT / "data/seals/hplt3-korean-final-test-v1.json"

PROTOCOL_ID = "jamoflow-exaone-retrieval-data-v1"
PLAN_KIND = "exaone_retrieval_data_plan_v1"
SEAL_KIND = "exaone_retrieval_data_seal_v1"
VERIFICATION_KIND = "exaone_retrieval_data_verification_v1"

VOCABULARY_SIZE = 102_400
TABLE_ORDERS = (1, 2, 3)
MAXIMUM_CONTEXT_ORDER = 3
MAXIMUM_DRAFT_TOKENS = 3
MAXIMUM_PROMPT_MATCH = 4
MAXIMUM_TABLE_ENTRIES = 200_000
MINIMUM_CONTEXT_COUNT = 5
MINIMUM_WINNING_NEXT_COUNT = 5
MINIMUM_NEXT_TOKEN_PROBABILITY = 0.8
TRAIN_DOCUMENT_BYTE_BUDGET = 128_000_000
PROMPT_TOKENS = 128
CONTROLLED_CONTINUATION_TOKENS = 128
WARMUP_CASES = 8
MEASURED_CASES = 64
TOTAL_CASES = WARMUP_CASES + MEASURED_CASES
EXPECTED_AVAILABLE_TRAIN_DOCUMENTS = 5_637
EXPECTED_EVALUATION_DOCUMENTS = 1_482
MINIMUM_HANGUL_LETTER_FRACTION = 0.8
MINIMUM_HANGUL_CODEPOINTS = 32
MINIMUM_HANGUL_VISIBLE_FRACTION = 0.5
CASE_RANK_DOMAIN = b"JamoFlow/EXAONE-retrieval-case/v1\0"
CASE_RANK_KEY_DOMAIN = b"JamoFlow/EXAONE-retrieval-case-key/v1\0"
TRAIN_EXACT_SET_DOMAIN = b"JamoFlow/EXAONE-retrieval-train-exact/v1\0"
TRAIN_NORMALIZED_SET_DOMAIN = b"JamoFlow/EXAONE-retrieval-train-normalized/v1\0"
EVALUATION_EXACT_SET_DOMAIN = b"JamoFlow/EXAONE-retrieval-eval-exact/v1\0"
EVALUATION_NORMALIZED_SET_DOMAIN = b"JamoFlow/EXAONE-retrieval-eval-normalized/v1\0"

TOKENIZER_RUNTIME_FILES = (
    "config.json",
    "configuration_exaone.py",
    "merges.txt",
    "modeling_exaone.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

TABLE_ARRAY_NAMES = tuple(
    f"order_{order}_{suffix}"
    for order in TABLE_ORDERS
    for suffix in ("context", "next", "best_count", "total_count")
)
CASE_ARRAY_NAMES = (
    "case_digest",
    "continuation_token_ids",
    "prompt_token_ids",
    "rank_digest",
)

IMPLEMENTATION_PATHS = (
    "docs/171-retrieval-novelty-closure-and-large-model-replication-direction.md",
    "docs/176-exaone-8b-compatibility-result-and-actual-stage-decision.md",
    "docs/177-exaone-retrieval-data-and-case-protocol.md",
    "requirements/apple-retrieval-v1.txt",
    "scripts/exaone_retrieval_data.py",
    "scripts/prepare_exaone_retrieval_data.py",
    "scripts/seal_exaone_retrieval_data_plan.py",
    "scripts/verify_exaone_retrieval_data.py",
    "scripts/hplt3_fresh_adaptation_protocol.py",
    "scripts/hplt3_fresh_adaptation_v2_protocol.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/hplt3_final_test.py",
    "tests/test_exaone_retrieval_data.py",
)


def canonical_bytes(value: object) -> bytes:
    return _canonical_bytes(value)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(b"JamoFlow/array/v1\0")
    dtype = str(array.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(8, "big"))
    digest.update(dtype)
    digest.update(array.ndim.to_bytes(8, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **{key: arrays[key] for key in sorted(arrays)})
    return output.getvalue()


def digest_set_commitment(digests: Sequence[bytes], *, domain: bytes) -> str:
    ordered = sorted(digests)
    if len(ordered) != len(set(ordered)) or any(len(value) != 32 for value in ordered):
        raise ValueError("EXAONE digest set differs")
    digest = hashlib.sha256(domain)
    digest.update(len(ordered).to_bytes(8, "big"))
    for value in ordered:
        digest.update(value)
    return digest.hexdigest()


def implementation_identity() -> dict[str, str]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise AssertionError("EXAONE retrieval data implementation paths duplicate")
    return {path: hash_file(ROOT / path) for path in IMPLEMENTATION_PATHS}


def environment_identity() -> dict[str, Any]:
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers")
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
    }


def validate_environment(value: object) -> None:
    if value != environment_identity():
        raise ValueError("EXAONE retrieval data environment differs")


def dependency_identity() -> dict[str, dict[str, Any]]:
    paths = {
        "compatibility_plan": COMPATIBILITY_PLAN_PATH,
        "compatibility_result": COMPATIBILITY_RESULT_PATH,
        "evaluation_seal": EVALUATION_SEAL_PATH,
        "evaluation_source": EVALUATION_SOURCE_PATH,
        "train_seal": TRAIN_SEAL_PATH,
        "train_source": TRAIN_SOURCE_PATH,
    }
    return {
        name: {
            "bytes": path.stat().st_size,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hash_file(path),
        }
        for name, path in paths.items()
    }


def read_validated_compatibility_result() -> dict[str, Any]:
    compatibility_plan = read_compatibility_plan(verify_derived=False)
    result = json.loads(COMPATIBILITY_RESULT_PATH.read_text(encoding="utf-8"))
    validate_pass_result(result, plan=compatibility_plan)
    return result


def compatibility_projection() -> dict[str, Any]:
    result = read_validated_compatibility_result()
    files = result["model_files"]
    return {
        "model_files": {
            name: dict(files[name]) for name in TOKENIZER_RUNTIME_FILES
        },
        "plan_sha256": result["plan_sha256"],
        "result_summary_sha256": result["summary_sha256"],
        "status": result["status"],
        "tokenizer": dict(result["tokenizer"]),
    }


def _validate_dependency_identity(value: object) -> None:
    expected = {
        "compatibility_plan": COMPATIBILITY_PLAN_PATH,
        "compatibility_result": COMPATIBILITY_RESULT_PATH,
        "evaluation_seal": EVALUATION_SEAL_PATH,
        "evaluation_source": EVALUATION_SOURCE_PATH,
        "train_seal": TRAIN_SEAL_PATH,
        "train_source": TRAIN_SOURCE_PATH,
    }
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("EXAONE retrieval dependency identity differs")
    for name, row in value.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "path", "sha256"}
            or not isinstance(row.get("bytes"), int)
            or int(row.get("bytes", 0)) <= 0
            or not isinstance(row.get("path"), str)
            or row["path"] != expected[name].relative_to(ROOT).as_posix()
            or not is_sha256(row.get("sha256"))
        ):
            raise ValueError(f"EXAONE retrieval dependency differs: {name}")


def _validate_compatibility_projection(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "model_files",
        "plan_sha256",
        "result_summary_sha256",
        "status",
        "tokenizer",
    }:
        raise ValueError("EXAONE compatibility projection differs")
    files = value["model_files"]
    tokenizer = value["tokenizer"]
    if (
        value["status"] != "pass_primary_greedy_transaction_compatibility"
        or not is_sha256(value["plan_sha256"])
        or not is_sha256(value["result_summary_sha256"])
        or not isinstance(files, Mapping)
        or set(files) != set(TOKENIZER_RUNTIME_FILES)
        or not isinstance(tokenizer, Mapping)
        or set(tokenizer)
        != {
            "chat_template_deterministic",
            "direct_roundtrip_count",
            "direct_roundtrip_exact",
            "prompt_token_count",
            "vocab_size",
        }
        or tokenizer.get("chat_template_deterministic") is not True
        or tokenizer.get("direct_roundtrip_exact") is not True
        or tokenizer.get("vocab_size") != VOCABULARY_SIZE
    ):
        raise ValueError("EXAONE compatibility result did not pass")
    for name in TOKENIZER_RUNTIME_FILES:
        row = files[name]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "sha256"}
            or not isinstance(row.get("bytes"), int)
            or int(row.get("bytes", 0)) <= 0
            or not is_sha256(row.get("sha256"))
        ):
            raise ValueError(f"EXAONE compatibility file differs: {name}")


def validate_tokenizer_runtime_identity(
    value: object, *, compatibility: Mapping[str, Any]
) -> None:
    _validate_compatibility_projection(compatibility)
    if not isinstance(value, Mapping) or set(value) != {
        "compatibility_result_summary_sha256",
        "files",
        "repo_id",
        "revision",
        "tokenizer_class",
        "vocabulary_size",
    }:
        raise ValueError("EXAONE tokenizer runtime identity differs")
    if (
        value["compatibility_result_summary_sha256"]
        != compatibility["result_summary_sha256"]
        or value["files"] != compatibility["model_files"]
        or value["repo_id"] != PRIMARY_MODEL["repo_id"]
        or value["revision"] != PRIMARY_MODEL["revision"]
        or not isinstance(value["tokenizer_class"], str)
        or not value["tokenizer_class"]
        or value["vocabulary_size"] != VOCABULARY_SIZE
    ):
        raise ValueError("EXAONE tokenizer runtime provenance differs")


def _case_selection_core_contract() -> dict[str, Any]:
    return {
        "controlled_continuation_tokens": CONTROLLED_CONTINUATION_TOKENS,
        "continuation_roundtrip_token_ids_required": True,
        "decode_compositionality_required": True,
        "document_digest": "sha256_exact_utf8",
        "eligible_documents_ranked_by": ["rank_digest", "document_digest"],
        "evaluation_source_document_count": EXPECTED_EVALUATION_DOCUMENTS,
        "evaluation_source_role": "previously_used_sealed_final_test_documents",
        "maximum_special_token_occurrences": 0,
        "measured_cases": MEASURED_CASES,
        "minimum_hangul_codepoints": MINIMUM_HANGUL_CODEPOINTS,
        "minimum_hangul_letter_fraction": MINIMUM_HANGUL_LETTER_FRACTION,
        "minimum_hangul_visible_fraction": MINIMUM_HANGUL_VISIBLE_FRACTION,
        "one_document_per_case": True,
        "prompt_roundtrip_token_ids_required": True,
        "prompt_tokens": PROMPT_TOKENS,
        "raw_completion_workload": True,
        "warmup_cases": WARMUP_CASES,
    }


def derive_case_rank_key(
    *, dependencies: Mapping[str, Any], compatibility: Mapping[str, Any]
) -> bytes:
    _validate_dependency_identity(dependencies)
    _validate_compatibility_projection(compatibility)
    digest = hashlib.sha256(CASE_RANK_KEY_DOMAIN)
    digest.update(bytes.fromhex(dependencies["evaluation_seal"]["sha256"]))
    digest.update(bytes.fromhex(dependencies["compatibility_result"]["sha256"]))
    digest.update(bytes.fromhex(compatibility["result_summary_sha256"]))
    digest.update(
        bytes.fromhex(canonical_sha256(_case_selection_core_contract()))
    )
    return digest.digest()


def data_contract(
    *, dependencies: Mapping[str, Any], compatibility: Mapping[str, Any]
) -> dict[str, Any]:
    rank_key = derive_case_rank_key(
        dependencies=dependencies, compatibility=compatibility
    )
    return {
        "case_selection": {
            "case_rank_domain_hex": CASE_RANK_DOMAIN.hex(),
            "case_rank_key_derivation": (
                "evaluation_seal_artifact+compatibility_result_artifact+"
                "validated_result_payload+case_contract"
            ),
            "case_rank_key_hex": rank_key.hex(),
            **_case_selection_core_contract(),
        },
        "table": {
            "cross_document_separator": "tokenizer_encode_newline",
            "maximum_context_order": MAXIMUM_CONTEXT_ORDER,
            "maximum_draft_tokens": MAXIMUM_DRAFT_TOKENS,
            "maximum_entries": MAXIMUM_TABLE_ENTRIES,
            "maximum_prompt_match": MAXIMUM_PROMPT_MATCH,
            "minimum_context_count": MINIMUM_CONTEXT_COUNT,
            "minimum_next_token_probability": MINIMUM_NEXT_TOKEN_PROBABILITY,
            "minimum_winning_next_count": MINIMUM_WINNING_NEXT_COUNT,
            "orders": list(TABLE_ORDERS),
            "pair_count_algorithm": "uint64_context_uint32_next_lexsort_v1",
            "train_document_byte_budget": TRAIN_DOCUMENT_BYTE_BUDGET,
            "train_source_document_count": EXPECTED_AVAILABLE_TRAIN_DOCUMENTS,
            "train_source_role": "fresh_v2_stable_train_full_document_prefix",
            "winner_tie_break": "smallest_token_id",
        },
        "tokenizer": {
            "add_special_tokens": False,
            "clean_up_tokenization_spaces": False,
            "repo_id": PRIMARY_MODEL["repo_id"],
            "revision": PRIMARY_MODEL["revision"],
            "trust_remote_code": True,
            "runtime_files": list(TOKENIZER_RUNTIME_FILES),
            "vocabulary_size": VOCABULARY_SIZE,
        },
    }


def result_input_contract() -> dict[str, bool]:
    return {
        "candidate_acceptance": False,
        "candidate_or_baseline_latency": False,
        "evaluation_document_text": True,
        "historical_model_output": False,
        "model_forward_or_logits": False,
        "train_document_text": True,
    }


def claim_boundary_contract() -> dict[str, bool]:
    return {
        "actual_efficiency_tested": False,
        "case_selection_is_metric_free": True,
        "chat_template_workload_tested": False,
        "confirmatory_or_final_blind": False,
        "evaluation_pool_previously_used": True,
        "generic_retrieval_novelty_claimed": False,
        "korean_specific_method_tested": False,
        "model_compatibility_already_passed": True,
        "publication_efficiency_claim": False,
        "raw_completion_workload_only": True,
        "table_uses_train_only": True,
    }


def build_plan(*, git_commit_before_plan: str) -> dict[str, Any]:
    if not (
        isinstance(git_commit_before_plan, str)
        and len(git_commit_before_plan) == 40
        and all(character in "0123456789abcdef" for character in git_commit_before_plan)
    ):
        raise ValueError("EXAONE retrieval data pre-plan commit differs")
    dependencies = dependency_identity()
    compatibility = compatibility_projection()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_before_exaone_tokenization_table_or_case_selection",
        "git_commit_before_plan": git_commit_before_plan,
        "dependencies": dependencies,
        "compatibility": compatibility,
        "implementation_sha256": implementation_identity(),
        "environment": environment_identity(),
        "model": PRIMARY_MODEL,
        "data_contract": data_contract(
            dependencies=dependencies, compatibility=compatibility
        ),
        "result_inputs": result_input_contract(),
        "outputs": {
            "case_artifact_path": CASES_PATH.relative_to(ROOT).as_posix(),
            "seal_path": SEAL_PATH.relative_to(ROOT).as_posix(),
            "table_artifact_path": TABLE_PATH.relative_to(ROOT).as_posix(),
            "verification_path": VERIFICATION_PATH.relative_to(ROOT).as_posix(),
        },
        "claim_boundary": claim_boundary_contract(),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    validate_plan(payload, verify_derived=True)
    return payload


def validate_plan(plan: Mapping[str, Any], *, verify_derived: bool) -> None:
    expected = {
        "claim_boundary",
        "compatibility",
        "data_contract",
        "dependencies",
        "environment",
        "git_commit_before_plan",
        "implementation_sha256",
        "kind",
        "model",
        "outputs",
        "plan_sha256",
        "protocol_id",
        "result_inputs",
        "schema_version",
        "status",
    }
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    dependencies = plan.get("dependencies")
    compatibility = plan.get("compatibility")
    _validate_dependency_identity(dependencies)
    _validate_compatibility_projection(compatibility)
    implementation = plan.get("implementation_sha256")
    environment = plan.get("environment")
    if (
        not isinstance(implementation, Mapping)
        or len(implementation) != len(IMPLEMENTATION_PATHS)
        or set(implementation) != set(IMPLEMENTATION_PATHS)
        or not all(is_sha256(implementation[path]) for path in IMPLEMENTATION_PATHS)
        or not isinstance(environment, Mapping)
        or set(environment)
        != {
            "machine",
            "numpy",
            "packages",
            "platform",
            "python",
            "python_executable_name",
        }
        or not isinstance(environment.get("packages"), Mapping)
        or set(environment["packages"]) != {"mlx", "mlx-lm", "transformers"}
    ):
        raise ValueError("EXAONE retrieval implementation/environment differs")
    if (
        set(plan) != expected
        or plan.get("schema_version") != 1
        or plan.get("kind") != PLAN_KIND
        or plan.get("protocol_id") != PROTOCOL_ID
        or plan.get("status")
        != "sealed_before_exaone_tokenization_table_or_case_selection"
        or plan.get("model") != PRIMARY_MODEL
        or plan.get("data_contract")
        != data_contract(
            dependencies=dependencies,
            compatibility=compatibility,
        )
        or plan.get("result_inputs") != result_input_contract()
        or plan.get("claim_boundary") != claim_boundary_contract()
        or plan.get("outputs")
        != {
            "case_artifact_path": CASES_PATH.relative_to(ROOT).as_posix(),
            "seal_path": SEAL_PATH.relative_to(ROOT).as_posix(),
            "table_artifact_path": TABLE_PATH.relative_to(ROOT).as_posix(),
            "verification_path": VERIFICATION_PATH.relative_to(ROOT).as_posix(),
        }
        or not isinstance(plan.get("git_commit_before_plan"), str)
        or len(plan["git_commit_before_plan"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in plan["git_commit_before_plan"]
        )
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE retrieval data plan identity differs")
    if verify_derived:
        if plan.get("dependencies") != dependency_identity():
            raise ValueError("EXAONE retrieval data dependencies differ")
        if plan.get("compatibility") != compatibility_projection():
            raise ValueError("EXAONE retrieval compatibility differs")
        if plan.get("implementation_sha256") != implementation_identity():
            raise ValueError("EXAONE retrieval data implementation differs")
        validate_environment(plan.get("environment"))


def read_plan(*, verify_derived: bool) -> dict[str, Any]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    validate_plan(plan, verify_derived=verify_derived)
    return plan


def pack_context(tokens: Sequence[int]) -> int:
    if not 1 <= len(tokens) <= MAXIMUM_CONTEXT_ORDER:
        raise ValueError("EXAONE retrieval context length differs")
    value = 0
    for token in tokens:
        current = int(token)
        if not 0 <= current < VOCABULARY_SIZE:
            raise ValueError("EXAONE retrieval token is outside vocabulary")
        value = value * VOCABULARY_SIZE + current
    if value >= 2**64:
        raise OverflowError("EXAONE retrieval context exceeds uint64")
    return value


def token_ids_uint32(values: Sequence[int], *, label: str) -> np.ndarray:
    checked: list[int] = []
    for value in values:
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            or not 0 <= int(value) < VOCABULARY_SIZE
            or int(value) >= 2**32
        ):
            raise ValueError(f"EXAONE {label} token id differs")
        checked.append(int(value))
    if not checked:
        raise ValueError(f"EXAONE {label} token sequence is empty")
    return np.asarray(checked, dtype=np.uint32)


def checked_uint32_count(value: int, *, label: str) -> np.uint32:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or not 0 <= int(value) < 2**32
    ):
        raise OverflowError(f"EXAONE {label} exceeds uint32")
    return np.uint32(int(value))


@dataclass(frozen=True, slots=True)
class OrderTable:
    order: int
    contexts: np.ndarray
    next_tokens: np.ndarray
    best_counts: np.ndarray
    total_counts: np.ndarray

    def validate(self) -> None:
        length = len(self.contexts)
        if (
            self.order not in TABLE_ORDERS
            or self.contexts.dtype != np.uint64
            or self.next_tokens.dtype != np.uint32
            or self.best_counts.dtype != np.uint32
            or self.total_counts.dtype != np.uint32
            or length <= 0
            or any(
                array.ndim != 1 or len(array) != length
                for array in (
                    self.contexts,
                    self.next_tokens,
                    self.best_counts,
                    self.total_counts,
                )
            )
            or np.any(self.contexts[1:] <= self.contexts[:-1])
            or np.any(self.next_tokens >= VOCABULARY_SIZE)
            or np.any(self.best_counts < MINIMUM_WINNING_NEXT_COUNT)
            or np.any(self.total_counts < MINIMUM_CONTEXT_COUNT)
            or np.any(self.total_counts < self.best_counts)
            or np.any(
                self.best_counts.astype(np.float64)
                / self.total_counts.astype(np.float64)
                < MINIMUM_NEXT_TOKEN_PROBABILITY
            )
        ):
            raise ValueError(f"EXAONE retrieval order-{self.order} table differs")

    def lookup(self, history: Sequence[int]) -> int | None:
        if len(history) < self.order:
            return None
        key = np.uint64(pack_context(history[-self.order :]))
        index = int(np.searchsorted(self.contexts, key))
        if index == len(self.contexts) or self.contexts[index] != key:
            return None
        return int(self.next_tokens[index])


@dataclass(frozen=True, slots=True)
class CompactBackoffTable:
    by_order: Mapping[int, OrderTable]

    def validate(self) -> None:
        if set(self.by_order) != set(TABLE_ORDERS):
            raise ValueError("EXAONE retrieval table orders differ")
        for order in TABLE_ORDERS:
            if self.by_order[order].order != order:
                raise ValueError("EXAONE retrieval order label differs")
            self.by_order[order].validate()
        if self.entry_count > MAXIMUM_TABLE_ENTRIES:
            raise ValueError("EXAONE retrieval table exceeds entry budget")

    @property
    def entry_count(self) -> int:
        return sum(len(self.by_order[order].contexts) for order in TABLE_ORDERS)

    def next_token(self, history: Sequence[int]) -> int | None:
        for order in reversed(TABLE_ORDERS):
            token = self.by_order[order].lookup(history)
            if token is not None:
                return token
        return None

    def propose(self, history: Sequence[int]) -> tuple[int, ...]:
        working = [int(value) for value in history]
        output: list[int] = []
        for _ in range(MAXIMUM_DRAFT_TOKENS):
            token = self.next_token(working)
            if token is None:
                break
            output.append(token)
            working.append(token)
        return tuple(output)

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for order in TABLE_ORDERS:
            row = self.by_order[order]
            arrays[f"order_{order}_context"] = row.contexts
            arrays[f"order_{order}_next"] = row.next_tokens
            arrays[f"order_{order}_best_count"] = row.best_counts
            arrays[f"order_{order}_total_count"] = row.total_counts
        return arrays


def table_from_arrays(arrays: Mapping[str, np.ndarray]) -> CompactBackoffTable:
    if set(arrays) != set(TABLE_ARRAY_NAMES):
        raise ValueError("EXAONE retrieval table artifact keys differ")
    table = CompactBackoffTable(
        by_order={
            order: OrderTable(
                order=order,
                contexts=np.asarray(arrays[f"order_{order}_context"]),
                next_tokens=np.asarray(arrays[f"order_{order}_next"]),
                best_counts=np.asarray(arrays[f"order_{order}_best_count"]),
                total_counts=np.asarray(arrays[f"order_{order}_total_count"]),
            )
            for order in TABLE_ORDERS
        }
    )
    table.validate()
    return table


def _packed_contexts(token_ids: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(token_ids)
    if (
        values.ndim != 1
        or values.dtype != np.uint32
        or order not in TABLE_ORDERS
        or len(values) <= order
        or np.any(values >= VOCABULARY_SIZE)
    ):
        raise ValueError("EXAONE retrieval training token stream differs")
    length = len(values) - order
    contexts = values[:length].astype(np.uint64, copy=True)
    for offset in range(1, order):
        contexts *= np.uint64(VOCABULARY_SIZE)
        contexts += values[offset : offset + length].astype(np.uint64, copy=False)
    return contexts, values[order:].astype(np.uint32, copy=False)


def best_continuations(token_ids: np.ndarray, order: int) -> dict[str, np.ndarray]:
    """Count (uint64 context, uint32 next) pairs without V**4 overflow."""

    contexts, next_tokens = _packed_contexts(token_ids, order)
    if len(contexts) >= 2**32:
        raise OverflowError("EXAONE retrieval pair count exceeds uint32")
    permutation = np.lexsort((next_tokens, contexts))
    contexts = contexts[permutation]
    next_tokens = next_tokens[permutation]
    pair_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(
                (contexts[1:] != contexts[:-1])
                | (next_tokens[1:] != next_tokens[:-1])
            ).astype(np.int64)
            + 1,
        )
    )
    pair_contexts = contexts[pair_starts]
    pair_next = next_tokens[pair_starts]
    pair_counts_u64 = np.diff(np.append(pair_starts, len(contexts))).astype(
        np.uint64
    )
    if len(pair_counts_u64) and int(np.max(pair_counts_u64)) >= 2**32:
        raise OverflowError("EXAONE retrieval repeated pair count exceeds uint32")
    pair_counts = pair_counts_u64.astype(np.uint32)
    context_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(pair_contexts[1:] != pair_contexts[:-1]).astype(np.int64)
            + 1,
        )
    )
    totals = np.add.reduceat(pair_counts.astype(np.uint64), context_starts)
    if len(totals) and int(np.max(totals)) >= 2**32:
        raise OverflowError("EXAONE retrieval context total exceeds uint32")
    maxima = np.maximum.reduceat(pair_counts, context_starts)
    lengths = np.diff(np.append(context_starts, len(pair_contexts)))
    maximum_indices = np.flatnonzero(pair_counts == np.repeat(maxima, lengths))
    maximum_contexts = pair_contexts[maximum_indices]
    _, first = np.unique(maximum_contexts, return_index=True)
    chosen = maximum_indices[first]
    if len(chosen) != len(context_starts):
        raise AssertionError("EXAONE retrieval best-continuation grouping differs")
    confidence = maxima.astype(np.float64) / totals.astype(np.float64)
    keep = (
        (totals >= MINIMUM_CONTEXT_COUNT)
        & (maxima >= MINIMUM_WINNING_NEXT_COUNT)
        & (confidence >= MINIMUM_NEXT_TOKEN_PROBABILITY)
    )
    return {
        "order": np.full(int(np.count_nonzero(keep)), order, dtype=np.uint8),
        "context": pair_contexts[chosen][keep].astype(np.uint64, copy=False),
        "next": pair_next[chosen][keep].astype(np.uint32, copy=False),
        "best_count": maxima[keep].astype(np.uint32, copy=False),
        "total_count": totals[keep].astype(np.uint32, copy=False),
    }


def build_compact_backoff_table(token_ids: np.ndarray) -> CompactBackoffTable:
    candidates = [best_continuations(token_ids, order) for order in TABLE_ORDERS]
    merged = {
        name: np.concatenate([row[name] for row in candidates])
        for name in ("order", "context", "next", "best_count", "total_count")
    }
    confidence = merged["best_count"].astype(np.float64) / merged["total_count"]
    ranking = np.lexsort(
        (
            merged["next"],
            merged["context"],
            -merged["order"].astype(np.int16),
            -confidence,
            -merged["best_count"].astype(np.int64),
        )
    )
    selected = ranking[:MAXIMUM_TABLE_ENTRIES]
    by_order: dict[int, OrderTable] = {}
    for order in TABLE_ORDERS:
        indices = selected[merged["order"][selected] == order]
        indices = indices[np.argsort(merged["context"][indices], kind="stable")]
        by_order[order] = OrderTable(
            order=order,
            contexts=merged["context"][indices].astype(np.uint64, copy=False),
            next_tokens=merged["next"][indices].astype(np.uint32, copy=False),
            best_counts=merged["best_count"][indices].astype(np.uint32, copy=False),
            total_counts=merged["total_count"][indices].astype(np.uint32, copy=False),
        )
    table = CompactBackoffTable(by_order=by_order)
    table.validate()
    if table.entry_count != min(MAXIMUM_TABLE_ENTRIES, len(ranking)):
        raise AssertionError("EXAONE retrieval selected table count differs")
    return table


def prompt_lookup_proposal(history: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in history)
    if any(not 0 <= value < VOCABULARY_SIZE for value in values):
        raise ValueError("EXAONE prompt lookup token differs")
    for size in range(min(MAXIMUM_PROMPT_MATCH, len(values) - 1), 0, -1):
        suffix = values[-size:]
        last_start = len(values) - size
        for start in range(last_start + 1):
            if values[start : start + size] != suffix:
                continue
            continuation_start = start + size
            continuation_end = min(
                continuation_start + MAXIMUM_DRAFT_TOKENS, len(values)
            )
            if continuation_start < continuation_end:
                return values[continuation_start:continuation_end]
    return ()


def hybrid_retrieval_proposal(
    table: CompactBackoffTable, history: Sequence[int]
) -> tuple[tuple[int, ...], str]:
    proposal = table.propose(history)
    if proposal:
        return proposal, "corpus_ngram"
    proposal = prompt_lookup_proposal(history)
    if proposal:
        return proposal, "prompt_lookup"
    return (), "none"


def hangul_prompt_profile(text: str) -> dict[str, float | int]:
    letters = [character for character in text if character.isalpha()]
    visible = [
        character
        for character in text
        if character.isprintable() and not character.isspace()
    ]
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in text)
    return {
        "hangul_codepoints": hangul,
        "hangul_letter_fraction": hangul / len(letters) if letters else 0.0,
        "hangul_visible_fraction": hangul / len(visible) if visible else 0.0,
        "letter_codepoints": len(letters),
        "visible_codepoints": len(visible),
    }


def case_rank(document_digest: bytes, *, rank_key: bytes) -> bytes:
    if len(document_digest) != 32 or len(rank_key) != 32:
        raise ValueError("EXAONE case document digest differs")
    return hashlib.sha256(CASE_RANK_DOMAIN + rank_key + document_digest).digest()


def _decode_exact_span(tokenizer: Any, token_ids: Sequence[int]) -> str | None:
    expected = tuple(int(value) for value in token_ids)
    decoded = tokenizer.decode(
        expected,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = tuple(tokenizer.encode(decoded, add_special_tokens=False))
    return decoded if encoded == expected else None


def build_case_arrays(
    records: Sequence[Any], tokenizer: Any, *, rank_key: bytes
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    special_ids = {
        int(value)
        for value in getattr(tokenizer, "all_special_ids", ())
        if value is not None
    }
    eligible: list[tuple[bytes, bytes, tuple[int, ...], tuple[int, ...]]] = []
    required = PROMPT_TOKENS + CONTROLLED_CONTINUATION_TOKENS
    seen: set[bytes] = set()
    for record in records:
        if record.text is None:
            continue
        digest = hashlib.sha256(record.raw).digest()
        if digest in seen:
            continue
        seen.add(digest)
        token_ids = tuple(
            int(value)
            for value in token_ids_uint32(
                tokenizer.encode(record.text, add_special_tokens=False),
                label="case document",
            )
        )
        if len(token_ids) < required:
            continue
        prefix = token_ids[:required]
        if any(value in special_ids for value in prefix):
            continue
        decoded = _decode_exact_span(tokenizer, prefix)
        prompt = prefix[:PROMPT_TOKENS]
        continuation = prefix[PROMPT_TOKENS:]
        prompt_text = _decode_exact_span(tokenizer, prompt)
        continuation_text = _decode_exact_span(tokenizer, continuation)
        if (
            decoded is None
            or prompt_text is None
            or continuation_text is None
            or prompt_text + continuation_text != decoded
        ):
            continue
        profile = hangul_prompt_profile(prompt_text)
        if (
            profile["hangul_codepoints"] < MINIMUM_HANGUL_CODEPOINTS
            or profile["hangul_letter_fraction"]
            < MINIMUM_HANGUL_LETTER_FRACTION
            or profile["hangul_visible_fraction"]
            < MINIMUM_HANGUL_VISIBLE_FRACTION
        ):
            continue
        eligible.append(
            (
                case_rank(digest, rank_key=rank_key),
                digest,
                prompt,
                continuation,
            )
        )
    eligible.sort(key=lambda row: (row[0], row[1]))
    if len(eligible) < TOTAL_CASES:
        raise ValueError("EXAONE retrieval eligible case count is too small")
    selected = eligible[:TOTAL_CASES]
    arrays = {
        "case_digest": np.asarray(
            [list(row[1]) for row in selected], dtype=np.uint8
        ),
        "continuation_token_ids": np.asarray(
            [row[3] for row in selected], dtype=np.uint32
        ),
        "prompt_token_ids": np.asarray(
            [row[2] for row in selected], dtype=np.uint32
        ),
        "rank_digest": np.asarray(
            [list(row[0]) for row in selected], dtype=np.uint8
        ),
    }
    validate_case_arrays(arrays, rank_key=rank_key)
    report = {
        "eligible_document_count": len(eligible),
        "measured_case_count": MEASURED_CASES,
        "ordered_case_commitment_sha256": ordered_case_commitment(
            arrays, rank_key=rank_key
        ),
        "total_case_count": TOTAL_CASES,
        "warmup_case_count": WARMUP_CASES,
    }
    return arrays, report


def validate_case_arrays(
    arrays: Mapping[str, np.ndarray], *, rank_key: bytes
) -> None:
    if set(arrays) != set(CASE_ARRAY_NAMES):
        raise ValueError("EXAONE retrieval case artifact keys differ")
    expected = {
        "case_digest": (np.dtype("uint8"), (TOTAL_CASES, 32)),
        "continuation_token_ids": (
            np.dtype("uint32"),
            (TOTAL_CASES, CONTROLLED_CONTINUATION_TOKENS),
        ),
        "prompt_token_ids": (np.dtype("uint32"), (TOTAL_CASES, PROMPT_TOKENS)),
        "rank_digest": (np.dtype("uint8"), (TOTAL_CASES, 32)),
    }
    for name, (dtype, shape) in expected.items():
        value = np.asarray(arrays[name])
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"EXAONE retrieval case array differs: {name}")
    if (
        np.any(arrays["prompt_token_ids"] >= VOCABULARY_SIZE)
        or np.any(arrays["continuation_token_ids"] >= VOCABULARY_SIZE)
        or len({bytes(row) for row in arrays["case_digest"]}) != TOTAL_CASES
        or any(
            bytes(rank) != case_rank(bytes(digest), rank_key=rank_key)
            for rank, digest in zip(arrays["rank_digest"], arrays["case_digest"])
        )
    ):
        raise ValueError("EXAONE retrieval case content differs")
    order = [
        (bytes(rank), bytes(digest))
        for rank, digest in zip(arrays["rank_digest"], arrays["case_digest"])
    ]
    if order != sorted(order):
        raise ValueError("EXAONE retrieval case order differs")


def ordered_case_commitment(
    arrays: Mapping[str, np.ndarray], *, rank_key: bytes
) -> str:
    validate_case_arrays(arrays, rank_key=rank_key)
    digest = hashlib.sha256(b"JamoFlow/EXAONE-retrieval-cases/v1\0")
    digest.update(TOTAL_CASES.to_bytes(8, "big"))
    for index in range(TOTAL_CASES):
        digest.update(bytes(arrays["rank_digest"][index]))
        digest.update(bytes(arrays["case_digest"][index]))
        digest.update(
            np.ascontiguousarray(arrays["prompt_token_ids"][index]).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                arrays["continuation_token_ids"][index]
            ).tobytes()
        )
    return digest.hexdigest()


def table_report(table: CompactBackoffTable) -> dict[str, Any]:
    table.validate()
    orders: dict[str, Any] = {}
    for order in TABLE_ORDERS:
        row = table.by_order[order]
        confidence = row.best_counts.astype(np.float64) / row.total_counts
        orders[str(order)] = {
            "entries": len(row.contexts),
            "maximum_confidence": float(np.max(confidence)),
            "minimum_confidence": float(np.min(confidence)),
            "arrays": {
                "best_count": array_sha256(row.best_counts),
                "context": array_sha256(row.contexts),
                "next": array_sha256(row.next_tokens),
                "total_count": array_sha256(row.total_counts),
            },
        }
    return {
        "entry_count": table.entry_count,
        "maximum_entries": MAXIMUM_TABLE_ENTRIES,
        "minimum_context_count": MINIMUM_CONTEXT_COUNT,
        "minimum_next_token_probability": MINIMUM_NEXT_TOKEN_PROBABILITY,
        "minimum_winning_next_count": MINIMUM_WINNING_NEXT_COUNT,
        "orders": orders,
        "winner_tie_break": "smallest_token_id",
    }


def artifact_descriptor(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "arrays": {
            name: {
                "dtype": str(np.asarray(values).dtype),
                "sha256": array_sha256(np.asarray(values)),
                "shape": list(np.asarray(values).shape),
            }
            for name, values in sorted(arrays.items())
        },
        "bytes": path.stat().st_size,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hash_file(path),
    }


def build_seal(
    *,
    plan: Mapping[str, Any],
    builder_git_commit: str,
    table: CompactBackoffTable,
    table_arrays: Mapping[str, np.ndarray],
    case_arrays: Mapping[str, np.ndarray],
    case_report: Mapping[str, Any],
    tokenizer_runtime: Mapping[str, Any],
    training_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": SEAL_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "complete_metric_free_table_and_cases",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "builder_git_commit": builder_git_commit,
        "dependencies": plan["dependencies"],
        "compatibility": plan["compatibility"],
        "environment": plan["environment"],
        "model": PRIMARY_MODEL,
        "data_contract": plan["data_contract"],
        "training_inventory": dict(training_inventory),
        "tokenizer_runtime": dict(tokenizer_runtime),
        "table_contract": table_report(table),
        "table_artifact": artifact_descriptor(TABLE_PATH, table_arrays),
        "case_contract": dict(case_report),
        "case_artifact": artifact_descriptor(CASES_PATH, case_arrays),
        "result_inputs": plan["result_inputs"],
        "claim_boundary": plan["claim_boundary"],
    }
    payload["seal_sha256"] = canonical_sha256(payload)
    validate_seal(payload, plan=plan, verify_artifacts=True)
    return payload


def _validate_training_inventory(value: object) -> None:
    expected = {
        "available_train_document_count",
        "document_count",
        "evaluation_document_count",
        "full_document_serialized_bytes",
        "newline_token_count",
        "token_count",
        "token_ids_sha256",
        "train_document_byte_budget",
        "train_evaluation_exact_intersection_count",
        "train_evaluation_normalized_intersection_count",
        "selected_train_exact_commitment_sha256",
        "selected_train_normalized_commitment_sha256",
        "evaluation_exact_commitment_sha256",
        "evaluation_normalized_commitment_sha256",
        "normalized_document_algorithm",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("EXAONE retrieval training inventory schema differs")
    if (
        value.get("available_train_document_count")
        != EXPECTED_AVAILABLE_TRAIN_DOCUMENTS
        or value.get("evaluation_document_count") != EXPECTED_EVALUATION_DOCUMENTS
        or not isinstance(value.get("document_count"), int)
        or not 0 < int(value["document_count"]) <= EXPECTED_AVAILABLE_TRAIN_DOCUMENTS
        or value.get("train_document_byte_budget") != TRAIN_DOCUMENT_BYTE_BUDGET
        or not isinstance(value.get("full_document_serialized_bytes"), int)
        or not 0 < int(value["full_document_serialized_bytes"])
        <= TRAIN_DOCUMENT_BYTE_BUDGET
        or not isinstance(value.get("newline_token_count"), int)
        or int(value["newline_token_count"]) <= 0
        or not isinstance(value.get("token_count"), int)
        or int(value["token_count"]) <= 0
        or int(value["token_count"]) >= 2**32
        or not is_sha256(value.get("token_ids_sha256"))
        or value.get("train_evaluation_exact_intersection_count") != 0
        or value.get("train_evaluation_normalized_intersection_count") != 0
        or value.get("normalized_document_algorithm")
        != "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
        or not all(
            is_sha256(value.get(name))
            for name in (
                "selected_train_exact_commitment_sha256",
                "selected_train_normalized_commitment_sha256",
                "evaluation_exact_commitment_sha256",
                "evaluation_normalized_commitment_sha256",
            )
        )
    ):
        raise ValueError("EXAONE retrieval training inventory differs")


def _validate_case_contract(
    value: object, arrays: Mapping[str, np.ndarray], *, rank_key: bytes
) -> None:
    expected = {
        "eligible_document_count",
        "measured_case_count",
        "ordered_case_commitment_sha256",
        "total_case_count",
        "warmup_case_count",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not isinstance(value.get("eligible_document_count"), int)
        or int(value["eligible_document_count"]) < TOTAL_CASES
        or value.get("measured_case_count") != MEASURED_CASES
        or value.get("total_case_count") != TOTAL_CASES
        or value.get("warmup_case_count") != WARMUP_CASES
        or value.get("ordered_case_commitment_sha256")
        != ordered_case_commitment(arrays, rank_key=rank_key)
    ):
        raise ValueError("EXAONE retrieval case contract differs")


def _validate_table_contract(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "entry_count",
        "maximum_entries",
        "minimum_context_count",
        "minimum_next_token_probability",
        "minimum_winning_next_count",
        "orders",
        "winner_tie_break",
    }:
        raise ValueError("EXAONE retrieval table contract schema differs")
    orders = value["orders"]
    if (
        not isinstance(value["entry_count"], int)
        or not 0 < value["entry_count"] <= MAXIMUM_TABLE_ENTRIES
        or value["maximum_entries"] != MAXIMUM_TABLE_ENTRIES
        or value["minimum_context_count"] != MINIMUM_CONTEXT_COUNT
        or value["minimum_next_token_probability"]
        != MINIMUM_NEXT_TOKEN_PROBABILITY
        or value["minimum_winning_next_count"] != MINIMUM_WINNING_NEXT_COUNT
        or value["winner_tie_break"] != "smallest_token_id"
        or not isinstance(orders, Mapping)
        or set(orders) != {str(order) for order in TABLE_ORDERS}
    ):
        raise ValueError("EXAONE retrieval table contract differs")
    total = 0
    for order in TABLE_ORDERS:
        row = orders[str(order)]
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "arrays",
                "entries",
                "maximum_confidence",
                "minimum_confidence",
            }
            or not isinstance(row["entries"], int)
            or row["entries"] <= 0
            or not isinstance(row["arrays"], Mapping)
            or set(row["arrays"])
            != {"best_count", "context", "next", "total_count"}
            or not all(is_sha256(item) for item in row["arrays"].values())
            or not 0.0
            <= float(row["minimum_confidence"])
            <= float(row["maximum_confidence"])
            <= 1.0
        ):
            raise ValueError(f"EXAONE retrieval order-{order} contract differs")
        total += row["entries"]
    if total != value["entry_count"]:
        raise ValueError("EXAONE retrieval table entry total differs")


def _validate_case_contract_schema(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "eligible_document_count",
            "measured_case_count",
            "ordered_case_commitment_sha256",
            "total_case_count",
            "warmup_case_count",
        }
        or not isinstance(value["eligible_document_count"], int)
        or value["eligible_document_count"] < TOTAL_CASES
        or value["measured_case_count"] != MEASURED_CASES
        or value["total_case_count"] != TOTAL_CASES
        or value["warmup_case_count"] != WARMUP_CASES
        or not is_sha256(value["ordered_case_commitment_sha256"])
    ):
        raise ValueError("EXAONE retrieval case contract schema differs")


def _load_npz(path: Path, expected_keys: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_keys):
            raise ValueError(f"EXAONE retrieval artifact key set differs: {path.name}")
        return {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }


def validate_seal(
    seal: Mapping[str, Any], *, plan: Mapping[str, Any], verify_artifacts: bool
) -> None:
    expected = {
        "builder_git_commit",
        "case_artifact",
        "case_contract",
        "claim_boundary",
        "compatibility",
        "data_contract",
        "dependencies",
        "environment",
        "kind",
        "model",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "result_inputs",
        "schema_version",
        "seal_sha256",
        "status",
        "table_artifact",
        "table_contract",
        "tokenizer_runtime",
        "training_inventory",
    }
    unsigned = dict(seal)
    recorded = unsigned.pop("seal_sha256", None)
    if (
        set(seal) != expected
        or seal.get("schema_version") != 1
        or seal.get("kind") != SEAL_KIND
        or seal.get("protocol_id") != PROTOCOL_ID
        or seal.get("status") != "complete_metric_free_table_and_cases"
        or seal.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or seal.get("plan_sha256") != plan.get("plan_sha256")
        or seal.get("dependencies") != plan.get("dependencies")
        or seal.get("compatibility") != plan.get("compatibility")
        or seal.get("environment") != plan.get("environment")
        or seal.get("model") != PRIMARY_MODEL
        or seal.get("data_contract") != plan.get("data_contract")
        or seal.get("result_inputs") != result_input_contract()
        or seal.get("claim_boundary") != claim_boundary_contract()
        or not isinstance(seal.get("builder_git_commit"), str)
        or len(seal["builder_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in seal["builder_git_commit"]
        )
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE retrieval data seal identity differs")
    validate_tokenizer_runtime_identity(
        seal.get("tokenizer_runtime"), compatibility=plan["compatibility"]
    )
    _validate_training_inventory(seal.get("training_inventory"))
    _validate_table_contract(seal.get("table_contract"))
    _validate_case_contract_schema(seal.get("case_contract"))
    if not verify_artifacts:
        return
    table_arrays = _load_npz(TABLE_PATH, TABLE_ARRAY_NAMES)
    case_arrays = _load_npz(CASES_PATH, CASE_ARRAY_NAMES)
    table = table_from_arrays(table_arrays)
    rank_key = bytes.fromhex(
        plan["data_contract"]["case_selection"]["case_rank_key_hex"]
    )
    validate_case_arrays(case_arrays, rank_key=rank_key)
    _validate_case_contract(
        seal.get("case_contract"), case_arrays, rank_key=rank_key
    )
    if (
        seal.get("table_artifact") != artifact_descriptor(TABLE_PATH, table_arrays)
        or seal.get("table_contract") != table_report(table)
        or seal.get("case_artifact") != artifact_descriptor(CASES_PATH, case_arrays)
    ):
        raise ValueError("EXAONE retrieval data artifact differs")


def read_seal(*, verify_artifacts: bool) -> dict[str, Any]:
    plan = read_plan(verify_derived=True)
    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    validate_seal(seal, plan=plan, verify_artifacts=verify_artifacts)
    return seal


def _array_identity(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {
        name: array_sha256(np.asarray(arrays[name])) for name in sorted(arrays)
    }


def build_verification(
    *,
    plan: Mapping[str, Any],
    seal: Mapping[str, Any],
    verifier_git_commit: str,
    reconstructed_table_arrays: Mapping[str, np.ndarray],
    reconstructed_case_arrays: Mapping[str, np.ndarray],
    reconstructed_training_inventory: Mapping[str, Any],
    tokenizer_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": VERIFICATION_KIND,
        "protocol_id": PROTOCOL_ID,
        "status": "pass_independent_source_tokenizer_table_case_reconstruction",
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "seal_artifact_sha256": hash_file(SEAL_PATH),
        "seal_sha256": seal["seal_sha256"],
        "verifier_git_commit": verifier_git_commit,
        "tokenizer_runtime": dict(tokenizer_runtime),
        "reconstruction": {
            "case_array_sha256": _array_identity(reconstructed_case_arrays),
            "case_bitwise_equal": True,
            "source_and_tokenizer_rebuilt": True,
            "table_array_sha256": _array_identity(reconstructed_table_arrays),
            "table_bitwise_equal": True,
            "training_inventory": dict(reconstructed_training_inventory),
        },
        "result_inputs": result_input_contract(),
        "claim_boundary": {
            **claim_boundary_contract(),
            "independent_data_reconstruction_passed": True,
        },
    }
    payload["verification_sha256"] = canonical_sha256(payload)
    validate_verification(payload, plan=plan, seal=seal)
    return payload


def validate_verification(
    verification: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> None:
    expected = {
        "claim_boundary",
        "kind",
        "plan_artifact_sha256",
        "plan_sha256",
        "protocol_id",
        "reconstruction",
        "result_inputs",
        "schema_version",
        "seal_artifact_sha256",
        "seal_sha256",
        "status",
        "tokenizer_runtime",
        "verification_sha256",
        "verifier_git_commit",
    }
    unsigned = dict(verification)
    recorded = unsigned.pop("verification_sha256", None)
    reconstruction = verification.get("reconstruction")
    expected_reconstruction = {
        "case_array_sha256": seal["case_artifact"]["arrays"],
        "table_array_sha256": seal["table_artifact"]["arrays"],
    }
    observed_case = None
    observed_table = None
    if isinstance(reconstruction, Mapping):
        observed_case = reconstruction.get("case_array_sha256")
        observed_table = reconstruction.get("table_array_sha256")
    expected_case = {
        name: row["sha256"]
        for name, row in expected_reconstruction["case_array_sha256"].items()
    }
    expected_table = {
        name: row["sha256"]
        for name, row in expected_reconstruction["table_array_sha256"].items()
    }
    if (
        set(verification) != expected
        or verification.get("schema_version") != 1
        or verification.get("kind") != VERIFICATION_KIND
        or verification.get("protocol_id") != PROTOCOL_ID
        or verification.get("status")
        != "pass_independent_source_tokenizer_table_case_reconstruction"
        or verification.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or verification.get("plan_sha256") != plan.get("plan_sha256")
        or verification.get("seal_artifact_sha256") != hash_file(SEAL_PATH)
        or verification.get("seal_sha256") != seal.get("seal_sha256")
        or not isinstance(verification.get("verifier_git_commit"), str)
        or len(verification["verifier_git_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in verification["verifier_git_commit"]
        )
        or not isinstance(reconstruction, Mapping)
        or set(reconstruction)
        != {
            "case_array_sha256",
            "case_bitwise_equal",
            "source_and_tokenizer_rebuilt",
            "table_array_sha256",
            "table_bitwise_equal",
            "training_inventory",
        }
        or reconstruction.get("case_bitwise_equal") is not True
        or reconstruction.get("table_bitwise_equal") is not True
        or reconstruction.get("source_and_tokenizer_rebuilt") is not True
        or observed_case != expected_case
        or observed_table != expected_table
        or reconstruction.get("training_inventory")
        != seal.get("training_inventory")
        or verification.get("result_inputs") != result_input_contract()
        or verification.get("claim_boundary")
        != {
            **claim_boundary_contract(),
            "independent_data_reconstruction_passed": True,
        }
        or not is_sha256(recorded)
        or canonical_sha256(unsigned) != recorded
    ):
        raise ValueError("EXAONE retrieval data verification differs")
    validate_tokenizer_runtime_identity(
        verification.get("tokenizer_runtime"),
        compatibility=plan["compatibility"],
    )


def read_verification() -> dict[str, Any]:
    plan = read_plan(verify_derived=True)
    seal = read_seal(verify_artifacts=True)
    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    validate_verification(verification, plan=plan, seal=seal)
    return verification
