#!/usr/bin/env python3
"""Run the sealed scalar/Hangul-hybrid/BPE opportunity audit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from jamoflow.compute_conversion import conversion_model_spec
from jamoflow.corpus import Record, load_records, partition_records
from jamoflow.neural_data import build_neural_stream
from jamoflow.publication_bpe import (
    PINNED_TOKENIZERS_VERSION,
    audit_byte_bpe_tokenizer,
    byte_bpe_token_bytes,
)
from scalar_representation_core import (
    GENERIC_UTF8_RESIDENT_ROWS,
    HANGUL_HYBRID_RESIDENT_ROWS,
    audit_bpe_encoding,
    canonical_json_sha256,
    complete_utf8_prefix,
    hangul_dependence,
    representation_counts,
    scalar_blt_opportunity_flops,
    scalar_inventory,
    train_exact_byte_bpe,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/scalar-representation-opportunity-v1.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
INTEGRITY_PATH = ROOT / "data/processed/hplt3-korean-phase3/integrity.json"
ARTIFACT_ROOT = ROOT / "artifacts/scalar-representation-opportunity-v1"
OUTPUT_PATH = ROOT / "results/scalar-representation-opportunity-v1/summary.json"
IMPLEMENTATION_PATHS = (
    "docs/110-scalar-representation-and-bpe-opportunity-protocol.md",
    "pyproject.toml",
    "scripts/analyze_scalar_representation_opportunity.py",
    "src/jamoflow/compute_conversion.py",
    "src/jamoflow/corpus.py",
    "src/jamoflow/cost.py",
    "src/jamoflow/neural_data.py",
    "src/jamoflow/neural_model.py",
    "src/jamoflow/phase3.py",
    "src/jamoflow/publication_bpe.py",
    "scripts/scalar_representation_core.py",
    "tests/test_scalar_representation.py",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _require_clean_plan_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("scalar opportunity audit requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    last_change = _command(
        "git",
        "log",
        "-1",
        "--format=%H",
        "--",
        str(PLAN_PATH.relative_to(ROOT)),
    )
    if len(commit) != 40 or last_change != commit:
        raise RuntimeError("scalar opportunity plan must be sealed at current HEAD")
    return commit


def _require_never_published(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"scalar opportunity output already exists: {path}")
    history = _command(
        "git", "log", "--all", "--format=%H", "--", str(path.relative_to(ROOT))
    )
    if history:
        raise FileExistsError(f"scalar opportunity output has Git history: {path}")


def _publish_exact_or_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"scalar opportunity artifact differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _joined_rows(records: Sequence[Record]) -> Iterable[str]:
    emitted = False
    for record in records:
        if record.text is None:
            raise ValueError("scalar opportunity source contains an invalid record")
        yield record.text if not emitted else "\n" + record.text
        emitted = True
    if not emitted:
        raise ValueError("scalar opportunity split is empty")


def _joined_stream_identity(records: Sequence[Record]) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    nfc_count = 0
    for row in _joined_rows(records):
        encoded = row.encode("utf-8")
        digest.update(encoded)
        byte_count += len(encoded)
        row_count += 1
        original = row if row_count == 1 else row[1:]
        nfc_count += unicodedata.normalize("NFC", original) == original
    return {
        "records": row_count,
        "joined_bytes": byte_count,
        "joined_sha256": digest.hexdigest(),
        "nfc_documents": nfc_count,
        "non_nfc_documents": row_count - nfc_count,
    }


def _validate_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "bpe",
        "claim_boundary",
        "decision_rule",
        "implementation_sha256",
        "input",
        "kind",
        "known_preseal_engineering_anchors",
        "output",
        "protocol_id",
        "representation",
        "schema_version",
        "status",
    }
    if set(plan) != expected:
        raise ValueError("scalar opportunity plan schema differs")
    if (
        plan["schema_version"] != 1
        or plan["kind"] != "scalar_representation_opportunity_plan_v1"
        or plan["protocol_id"] != "jamoflow-scalar-representation-opportunity-v1"
        or plan["status"] != "sealed_before_bpe_token_counts"
    ):
        raise ValueError("scalar opportunity plan identity differs")
    if plan["bpe"] != {
        "add_prefix_space": False,
        "byte_fallback": False,
        "dropout": None,
        "initial_alphabet": "complete ByteLevel alphabet",
        "minimum_frequency": 2,
        "normalizer": None,
        "replicate_training_for_exact_json_determinism": True,
        "tokenizers_version": PINNED_TOKENIZERS_VERSION,
        "train_split_only": True,
        "use_regex": True,
        "vocabulary_sizes": [16_000, 32_000],
    }:
        raise ValueError("scalar opportunity BPE contract differs")
    if plan["representation"] != {
        "generic_unicode_scalar": (
            "one main step per strict Unicode scalar; conditional UTF-8 "
            "micro-heads; raw-byte fallback for invalid or truncated bytes"
        ),
        "hangul_hybrid": (
            "one conditional L-V-T main step for canonical precomposed Hangul; "
            "one main step per raw byte otherwise"
        ),
        "hangul_hybrid_resident_output_rows": HANGUL_HYBRID_RESIDENT_ROWS,
        "generic_utf8_resident_output_rows": GENERIC_UTF8_RESIDENT_ROWS,
        "canonicalization_requirement": (
            "valid precomposed Hangul has exactly one Hangul-unit encoding; "
            "raw fallback must not create an alias"
        ),
    }:
        raise ValueError("scalar opportunity representation contract differs")
    if plan["claim_boundary"] != {
        "actual_latency_or_memory_evidence": False,
        "bpe_counts_observed_before_plan": False,
        "calibration_only": True,
        "conditional_three_hot_claimed_novel": False,
        "final_or_historical_test_read": False,
        "matched_quality_evidence": False,
        "model_checkpoint_or_loss_read": False,
        "opportunity_audit_is_confirmatory": False,
        "preseal_engineering_anchors_disclosed": True,
    }:
        raise ValueError("scalar opportunity claim boundary differs")
    if set(plan["implementation_sha256"]) != set(IMPLEMENTATION_PATHS):
        raise ValueError("scalar opportunity implementation set differs")
    for relative in IMPLEMENTATION_PATHS:
        if _hash_file(ROOT / relative) != plan["implementation_sha256"][relative]:
            raise ValueError(f"scalar opportunity implementation changed: {relative}")


def _bpe_result(
    train_records: Sequence[Record],
    calibration_text: str,
    vocabulary_size: int,
) -> tuple[dict[str, Any], bytes]:
    first = train_exact_byte_bpe(
        _joined_rows(train_records),
        vocabulary_size=vocabulary_size,
        minimum_frequency=2,
    )
    replicate = train_exact_byte_bpe(
        _joined_rows(train_records),
        vocabulary_size=vocabulary_size,
        minimum_frequency=2,
    )
    first_bytes = first.to_str(pretty=False).encode("utf-8")
    replicate_bytes = replicate.to_str(pretty=False).encode("utf-8")
    if first_bytes != replicate_bytes:
        raise ValueError("BPE replicate training produced different tokenizer JSON")
    structural = audit_byte_bpe_tokenizer(
        first,
        (calibration_text, "\x00"),
        expected_vocabulary_size=vocabulary_size,
    ).to_dict()
    encoding = first.encode(calibration_text, add_special_tokens=False)
    token_ids = tuple(int(value) for value in encoding.ids)
    token_bytes = byte_bpe_token_bytes(first)
    lengths = Counter(len(token_bytes[token_id]) for token_id in token_ids)
    rendered = b"".join(token_bytes[token_id] for token_id in token_ids)
    expected = calibration_text.encode("utf-8")
    if rendered != expected:
        raise ValueError("BPE token bytes do not reconstruct calibration bytes")
    metrics = audit_bpe_encoding(first, calibration_text)
    metrics.update(
        {
            "deterministic_replicate_json_identity": True,
            "raw_token_bytes_identity": True,
            "structural_audit": structural,
            "token_byte_length_histogram": {
                str(length): int(count) for length, count in sorted(lengths.items())
            },
        }
    )
    if not structural["overall_pass"] or not metrics["roundtrip_identity"]:
        raise ValueError("BPE reversibility audit failed")
    return metrics, first_bytes


def main() -> None:
    commit = _require_clean_plan_commit()
    _require_never_published(OUTPUT_PATH)
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    _validate_plan(plan)
    for relative, expected in (
        (plan["input"]["source_path"], plan["input"]["source_sha256"]),
        (plan["input"]["integrity_path"], plan["input"]["integrity_sha256"]),
        (
            plan["input"]["prior_hangul_opportunity_path"],
            plan["input"]["prior_hangul_opportunity_sha256"],
        ),
        (
            plan["input"]["conditional_failure_path"],
            plan["input"]["conditional_failure_sha256"],
        ),
    ):
        if _hash_file(ROOT / relative) != expected:
            raise ValueError(f"scalar opportunity input changed: {relative}")

    records = load_records(
        [SOURCE_PATH],
        corpus_format="jsonl",
        text_field="text",
        deduplicate=True,
    )
    splits = partition_records(records)
    train_records = tuple(splits["train"])
    calibration_records = tuple(splits["calibration"])
    if len(train_records) != 5_791 or len(calibration_records) != 386:
        raise ValueError("scalar opportunity split counts differ")
    train_identity = _joined_stream_identity(train_records)
    calibration_identity = _joined_stream_identity(calibration_records)
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=8_000_000,
        sequence_length=512,
    )
    stream_sha256 = hashlib.sha256(stream.data).hexdigest()
    if stream_sha256 != plan["input"]["calibration_stream_sha256"]:
        raise ValueError("scalar opportunity calibration stream differs")
    calibration_text, trailing = complete_utf8_prefix(stream.data)
    if len(trailing) != 1:
        raise ValueError("scalar opportunity truncated suffix differs")

    representations = representation_counts(stream.data)
    inventory = scalar_inventory(_joined_rows(train_records), calibration_text)
    dependence = {
        "train": hangul_dependence(_joined_rows(train_records)),
        "calibration": hangul_dependence(calibration_text),
    }
    opportunity = scalar_blt_opportunity_flops(
        stream.data,
        baseline_spec=conversion_model_spec(72),
        data_patches=72,
    )

    bpe_metrics: dict[str, Any] = {}
    tokenizer_payloads: dict[Path, bytes] = {}
    for vocabulary_size in plan["bpe"]["vocabulary_sizes"]:
        metrics, tokenizer_bytes = _bpe_result(
            train_records,
            calibration_text,
            vocabulary_size,
        )
        bpe_metrics[str(vocabulary_size)] = metrics
        tokenizer_payloads[
            ARTIFACT_ROOT / f"byte-bpe-{vocabulary_size}.json"
        ] = tokenizer_bytes

    local_width = conversion_model_spec(72).local_width
    train_inventory = inventory["train"]
    projection_rows = {
        "raw_byte": 256,
        "generic_conditional_utf8": GENERIC_UTF8_RESIDENT_ROWS,
        "hangul_hybrid_conditional_lvt": HANGUL_HYBRID_RESIDENT_ROWS,
        "train_scalar_vocabulary_plus_raw_fallback": (
            train_inventory["unique_scalars"] + 256
        ),
        "train_non_hangul_plus_factorized_hangul_plus_raw_fallback": (
            train_inventory["unique_non_hangul"] + 68 + 256
        ),
        "full_flat_hangul_plus_train_non_hangul_plus_raw_fallback": (
            11_172 + train_inventory["unique_non_hangul"] + 256
        ),
        "byte_bpe_16000": 16_000,
        "byte_bpe_32000": 32_000,
    }
    projections = {
        name: {
            "rows": rows,
            "single_projection_parameters_at_local_width_192": rows * local_width,
        }
        for name, rows in projection_rows.items()
    }

    rule = plan["decision_rule"]
    checks = {
        "all_representations_reversible_on_audit_stream": (
            representations["complete_scalar_bytes"] + len(trailing)
            == len(stream.data)
            and all(row["roundtrip_identity"] for row in bpe_metrics.values())
            and all(
                row["deterministic_replicate_json_identity"]
                for row in bpe_metrics.values()
            )
        ),
        "minimum_generic_scalar_step_reduction": (
            representations["reductions_relative_to_raw_byte"][
                "generic_unicode_scalar"
            ]
            >= rule["minimum_generic_scalar_step_reduction"]
        ),
        "minimum_hangul_hybrid_step_reduction": (
            representations["reductions_relative_to_raw_byte"]["hangul_hybrid"]
            >= rule["minimum_hangul_hybrid_step_reduction"]
        ),
        "minimum_hangul_hybrid_dense_flop_reduction": (
            opportunity["hangul_scalar_otherwise_raw_byte"][
                "reduction_relative_to_w72"
            ]
            >= rule["minimum_hangul_hybrid_dense_flop_reduction"]
        ),
        "maximum_train_scalar_vocabulary": (
            train_inventory["unique_scalars"]
            <= rule["maximum_train_scalar_vocabulary"]
        ),
        "maximum_calibration_unseen_scalar_occurrence_rate": (
            inventory["calibration"]["unseen_scalar_occurrence_rate"]
            <= rule["maximum_calibration_unseen_scalar_occurrence_rate"]
        ),
    }
    passed = all(checks.values())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "scalar_representation_opportunity_result_v1",
        "protocol_id": plan["protocol_id"],
        "complete": True,
        "git_commit": commit,
        "plan_artifact_sha256": _hash_file(PLAN_PATH),
        "source": {
            "source_artifact_sha256": plan["input"]["source_sha256"],
            "calibration_stream_sha256": stream_sha256,
            "complete_calibration_prefix_sha256": hashlib.sha256(
                calibration_text.encode("utf-8")
            ).hexdigest(),
            "train": train_identity,
            "calibration_full_documents": calibration_identity,
        },
        "metrics": {
            "representation_counts": representations,
            "scalar_inventory": inventory,
            "hangul_dependence": dependence,
            "bpe": bpe_metrics,
            "dense_matmul_opportunity": opportunity,
            "output_projection_diagnostics": projections,
            "sequential_step_comparison": {
                "raw_byte": representations["sequential_steps"]["raw_byte"],
                "generic_unicode_scalar": representations["sequential_steps"][
                    "generic_unicode_scalar_with_raw_suffix_fallback"
                ],
                "hangul_hybrid": representations["sequential_steps"][
                    "hangul_scalar_otherwise_raw_byte"
                ],
                "byte_bpe_16000": bpe_metrics["16000"]["token_count"],
                "byte_bpe_32000": bpe_metrics["32000"]["token_count"],
            },
        },
        "decision": {
            "checks": checks,
            "pass": passed,
            "status": (
                "random_weight_representation_construction_authorized"
                if passed
                else "scalar_representation_branch_stopped"
            ),
            "authorizes": rule["pass_authorizes"] if passed else rule["failure_action"],
        },
        "claim_boundary": plan["claim_boundary"],
        "interpretation": {
            "conditional_three_hot_precedent": "Cognetta et al., EACL 2023",
            "bpe_is_mandatory_in_later_actual_runtime_and_quality_studies": True,
            "hangul_advantage_over_generic_scalar_not_yet_established": True,
            "analytical_flops_are_not_latency": True,
            "next_stage_uses_random_weights_only": True,
        },
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    for path, payload in tokenizer_payloads.items():
        _publish_exact_or_new(path, payload)
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("repository changed during scalar opportunity audit")
    _publish_exact_or_new(OUTPUT_PATH, _json_bytes(summary))


if __name__ == "__main__":
    main()
