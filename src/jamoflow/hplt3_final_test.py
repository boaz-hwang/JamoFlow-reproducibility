"""Fail-closed construction of the disjoint HPLT3 Korean final test.

The historical Phase 3 ``test`` split has been used for development decisions.
This module deterministically constructs a second evaluation stream from the
same pinned raw shard while excluding every document in the historical sample.
Only aggregate commitments are intended to be tracked; document text and
individual document digests remain in ignored local data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256, sha512
import heapq
import json
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from .corpus import Record, split_for_record
from .hplt3 import BottomHashSampler, Candidate, hash_file, iter_zstd_jsonl_lines


FINAL_TEST_PROTOCOL_VERSION = 1
FINAL_TEST_DATASET_ID = "hplt3-korean-final-test-v1"
FINAL_TEST_STREAM_BYTES = 32_000_000
FINAL_TEST_SEQUENCE_LENGTH = 512
FINAL_TEST_SEQUENCE_COUNT = FINAL_TEST_STREAM_BYTES // FINAL_TEST_SEQUENCE_LENGTH

_KEY_DOMAIN = b"JamoFlow/final-test-key/v1\0"
_RANK_DOMAIN = b"JamoFlow/final-test-rank/v1\0"
_EXCLUSION_DOMAIN = b"JamoFlow/final-test-exclusion-set/v1\0"
_NORMALIZED_EXCLUSION_DOMAIN = (
    b"JamoFlow/final-test-normalized-exclusion-set/v1\0"
)
_SELECTED_DOMAIN = b"JamoFlow/final-test-selected-set/v1\0"
_NORMALIZED_SELECTED_DOMAIN = b"JamoFlow/final-test-normalized-selected-set/v1\0"
_ORDERED_DOMAIN = b"JamoFlow/final-test-ordered-selection/v1\0"
_INTERSECTION_DOMAIN = b"JamoFlow/final-test-intersection/v1\0"
_OVERLAP_AUDIT_DOMAIN = b"JamoFlow/final-test-overlap-audit/v1\0"
_NORMALIZED_INTERSECTION_DOMAIN = (
    b"JamoFlow/final-test-normalized-intersection/v1\0"
)
_NORMALIZED_OVERLAP_AUDIT_DOMAIN = (
    b"JamoFlow/final-test-normalized-overlap-audit/v1\0"
)

_SEALED_PREDECESSOR_MANIFEST_PATH = "data/manifests/hplt3-korean-phase3.json"
_SEALED_PREDECESSOR_SUMMARY_PATH = "results/phase3-data/summary.json"

_TOP_LEVEL_KEYS = {
    "dataset_id",
    "predecessor",
    "privacy",
    "protocol_version",
    "purpose",
    "schema_version",
    "selection",
    "source",
}
_SOURCE_KEYS = {
    "etag",
    "expected_bytes",
    "expected_sha256",
    "filename",
    "last_modified",
    "url",
}
_PREDECESSOR_KEYS = {
    "data_summary_path",
    "data_summary_sha256",
    "dataset_id",
    "integrity_cache_sha256",
    "legacy_quotas",
    "legacy_rank_order_commitments",
    "legacy_salt",
    "manifest_path",
    "manifest_sha256",
    "processed_output_bytes",
    "processed_output_sha256",
    "split_counts",
    "unique_records",
}
_SELECTION_KEYS = {
    "expected_rank_key_hex",
    "maximum_document_bytes",
    "minimum_document_bytes",
    "normalized_exclusion_algorithm",
    "order",
    "quota_stream_bytes",
    "rank_algorithm",
    "record_digest",
    "required_split",
    "reserve_stream_bytes",
    "separator_hex",
    "sequence_length",
    "split_algorithm",
    "unicode_database_version",
}
_PRIVACY_KEYS = {
    "tracked_individual_digests",
    "tracked_model_metrics",
    "tracked_text",
}
_SPLITS = ("train", "calibration", "test")
_LEGACY_SCAN_KEYS = {
    "eligible_records",
    "eligible_text_bytes",
    "empty_text",
    "exact_duplicates",
    "invalid_json",
    "invalid_utf8",
    "missing_text",
    "parsed_records",
    "source_lines",
    "too_long",
    "too_short",
}

_SEAL_PAYLOAD_KEYS = {
    "dataset_id",
    "manifest",
    "output",
    "predecessor",
    "preparation_git_commit",
    "privacy",
    "protocol_version",
    "scan",
    "selection",
    "source",
}
_SEAL_MANIFEST_KEYS = {"schema_version", "sha256"}
_SEAL_OUTPUT_KEYS = {
    "evaluation_stream_bytes",
    "evaluation_stream_sha256",
    "full_jsonl_bytes",
    "full_jsonl_sha256",
    "sequence_count",
    "sequence_length",
}
_SEAL_PREDECESSOR_KEYS = {
    "document_count",
    "exclusion_commitment_sha256",
    "legacy_reconstruction_verified",
    "normalized_exclusion_commitment_sha256",
    "normalized_unique_document_count",
}
_SEAL_PRIVACY_KEYS = {
    "individual_document_digests_tracked",
    "model_metrics_tracked",
    "raw_text_tracked",
}
_SEAL_SELECTION_KEYS = {
    "all_records_stable_test",
    "intersection_commitment_sha256",
    "intersection_count",
    "normalized_intersection_commitment_sha256",
    "normalized_intersection_count",
    "normalized_overlap_audit_sha256",
    "normalized_selected_set_commitment_sha256",
    "normalized_selected_unique_document_count",
    "ordered_selection_sha256",
    "overlap_audit_sha256",
    "overshoot_stream_bytes",
    "selected_document_count",
    "selected_document_raw_bytes",
    "selected_set_commitment_sha256",
}
_SEAL_SOURCE_KEYS = {"bytes", "sha256"}


def _u64(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("commitment integer must be a nonnegative int")
    return struct.pack(">Q", value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_nonnegative_int(value) and value > 0


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys are not the sealed schema")


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(payload: object) -> str:
    return sha256(canonical_json_bytes(payload)).hexdigest()


def derive_rank_key(
    source_sha256: str,
    predecessor_output_sha256: str,
    quota_stream_bytes: int,
    protocol_version: int = FINAL_TEST_PROTOCOL_VERSION,
) -> bytes:
    if not _is_sha256(source_sha256) or not _is_sha256(
        predecessor_output_sha256
    ):
        raise ValueError("rank-key inputs require SHA-256 hex digests")
    return sha256(
        _KEY_DOMAIN
        + bytes.fromhex(source_sha256)
        + bytes.fromhex(predecessor_output_sha256)
        + _u64(quota_stream_bytes)
        + _u64(protocol_version)
    ).digest()


def final_test_rank_digest(rank_key: bytes, document_digest: bytes) -> bytes:
    if len(rank_key) != 32 or len(document_digest) != 32:
        raise ValueError("final-test rank inputs must be 32 bytes")
    return sha256(_RANK_DOMAIN + rank_key + document_digest).digest()


def digest_set_commitment(
    digests: Iterable[bytes], *, domain: bytes
) -> str:
    ordered = sorted(digests)
    if any(len(digest) != 32 for digest in ordered):
        raise ValueError("document commitments require 32-byte digests")
    return sha256(domain + _u64(len(ordered)) + b"".join(ordered)).hexdigest()


@dataclass(frozen=True, slots=True)
class FinalTestCandidate:
    rank: bytes
    digest: bytes
    raw: bytes

    @property
    def stream_bytes(self) -> int:
        return len(self.raw) + 1


def ordered_selection_commitment(
    candidates: Sequence[FinalTestCandidate],
) -> str:
    payload = bytearray(_ORDERED_DOMAIN)
    payload.extend(_u64(len(candidates)))
    for candidate in candidates:
        if len(candidate.rank) != 32 or len(candidate.digest) != 32:
            raise ValueError("ordered selection has malformed rank/digest")
        payload.extend(candidate.rank)
        payload.extend(candidate.digest)
        payload.extend(_u64(len(candidate.raw)))
    return sha256(payload).hexdigest()


def overlap_audit_commitment(
    exclusion_commitment: str,
    selected_commitment: str,
    intersection_digests: Iterable[bytes],
    *,
    intersection_domain: bytes = _INTERSECTION_DOMAIN,
    audit_domain: bytes = _OVERLAP_AUDIT_DOMAIN,
) -> dict[str, object]:
    intersection = tuple(sorted(intersection_digests))
    intersection_commitment = digest_set_commitment(
        intersection,
        domain=intersection_domain,
    )
    if not _is_sha256(exclusion_commitment) or not _is_sha256(
        selected_commitment
    ):
        raise ValueError("overlap audit requires sealed set commitments")
    audit = sha256(
        audit_domain
        + bytes.fromhex(exclusion_commitment)
        + bytes.fromhex(selected_commitment)
        + _u64(len(intersection))
        + bytes.fromhex(intersection_commitment)
    ).hexdigest()
    return {
        "intersection_count": len(intersection),
        "intersection_commitment_sha256": intersection_commitment,
        "overlap_audit_sha256": audit,
    }


@dataclass(slots=True)
class FinalTestScanStatistics:
    source_lines: int = 0
    parsed_records: int = 0
    invalid_json: int = 0
    missing_text: int = 0
    invalid_utf8: int = 0
    empty_text: int = 0
    too_short: int = 0
    too_long: int = 0
    exact_duplicates: int = 0
    eligible_records: int = 0
    eligible_text_bytes: int = 0
    predecessor_records_found: int = 0
    normalized_predecessor_exclusions: int = 0
    normalized_source_duplicates: int = 0
    post_exclusion_test_records: int = 0
    post_exclusion_test_text_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class FinalTestBottomHashSampler:
    """Keep the lowest derived ranks needed to satisfy one stream quota."""

    def __init__(self, quota_stream_bytes: int, reserve_stream_bytes: int) -> None:
        if quota_stream_bytes <= 0 or reserve_stream_bytes <= quota_stream_bytes:
            raise ValueError("final-test sampler reserve must exceed its quota")
        self.quota_stream_bytes = quota_stream_bytes
        self.reserve_stream_bytes = reserve_stream_bytes
        self.heap: list[tuple[int, int, FinalTestCandidate]] = []
        self.heap_stream_bytes = 0
        self.seen_keys: set[tuple[bytes, bytes]] = set()

    def add(self, candidate: FinalTestCandidate) -> None:
        key = (candidate.rank, candidate.digest)
        if (
            len(candidate.rank) != 32
            or len(candidate.digest) != 32
            or not candidate.raw
            or key in self.seen_keys
        ):
            raise ValueError("final-test sampler candidate is malformed or duplicated")
        self.seen_keys.add(key)
        rank_integer = int.from_bytes(candidate.rank, "big")
        heapq.heappush(
            self.heap,
            (
                -rank_integer,
                -int.from_bytes(candidate.digest, "big"),
                candidate,
            ),
        )
        self.heap_stream_bytes += candidate.stream_bytes
        while self.heap:
            largest = self.heap[0][2]
            if (
                self.heap_stream_bytes - largest.stream_bytes
                < self.reserve_stream_bytes
            ):
                break
            _, _, removed = heapq.heappop(self.heap)
            self.heap_stream_bytes -= removed.stream_bytes

    def finalize(self) -> list[FinalTestCandidate]:
        ordered = sorted(
            (entry[2] for entry in self.heap),
            key=lambda candidate: (candidate.rank, candidate.digest),
        )
        selected: list[FinalTestCandidate] = []
        available = 0
        for candidate in ordered:
            available += len(candidate.raw) + (1 if selected else 0)
            selected.append(candidate)
            if available >= self.quota_stream_bytes:
                break
        if available < self.quota_stream_bytes:
            raise ValueError(
                "stable-test candidates do not satisfy the sealed byte quota"
            )
        return selected


def select_final_test_prefix_exhaustive(
    candidates: Iterable[FinalTestCandidate],
    quota_stream_bytes: int,
) -> list[FinalTestCandidate]:
    """Select the exact global rank prefix after all filtering/deduplication."""

    ordered = sorted(
        candidates,
        key=lambda candidate: (candidate.rank, candidate.digest),
    )
    selected: list[FinalTestCandidate] = []
    available = 0
    seen: set[tuple[bytes, bytes]] = set()
    for candidate in ordered:
        key = (candidate.rank, candidate.digest)
        if (
            len(candidate.rank) != 32
            or len(candidate.digest) != 32
            or not candidate.raw
            or key in seen
        ):
            raise ValueError("exhaustive final-test candidate is malformed or duplicated")
        seen.add(key)
        available += len(candidate.raw) + (1 if selected else 0)
        selected.append(candidate)
        if available >= quota_stream_bytes:
            return selected
    raise ValueError("stable-test candidates do not satisfy the sealed byte quota")


@dataclass(frozen=True, slots=True)
class PredecessorIndex:
    digests_by_split: dict[str, frozenset[bytes]]
    all_digests: frozenset[bytes]
    exclusion_commitment_sha256: str
    normalized_digests: frozenset[bytes]
    normalized_exclusion_commitment_sha256: str
    source_scan: dict[str, int]


def _record_split(document_digest: bytes, raw: bytes, text: str) -> str:
    record = Record(
        record_id=document_digest.hex(),
        source="hplt3",
        ordinal=0,
        raw=raw,
        text=text,
    )
    return split_for_record(record)


def normalized_record_digest(text: str) -> bytes:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    collapsed = " ".join(normalized.split())
    return sha256(collapsed.encode("utf-8", errors="strict")).digest()


def _legacy_rank(legacy_salt: str, digest: bytes) -> int:
    return int.from_bytes(
        sha256(legacy_salt.encode("utf-8") + b"\0" + digest).digest(),
        "big",
    )


def _legacy_commitment(candidates: Sequence[Candidate]) -> str:
    ordered = sorted(candidates, key=lambda item: (item.rank, item.digest))
    return sha256(b"".join(candidate.digest for candidate in ordered)).hexdigest()


def validate_final_test_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("final-test manifest must be an object")
    _require_exact_keys(manifest, _TOP_LEVEL_KEYS, "final-test manifest")
    source = manifest["source"]
    predecessor = manifest["predecessor"]
    selection = manifest["selection"]
    privacy = manifest["privacy"]
    if not all(
        isinstance(value, Mapping)
        for value in (source, predecessor, selection, privacy)
    ):
        raise ValueError("final-test manifest sections must be objects")
    _require_exact_keys(source, _SOURCE_KEYS, "final-test source")
    _require_exact_keys(
        predecessor, _PREDECESSOR_KEYS, "final-test predecessor"
    )
    _require_exact_keys(selection, _SELECTION_KEYS, "final-test selection")
    _require_exact_keys(privacy, _PRIVACY_KEYS, "final-test privacy")
    split_counts = predecessor["split_counts"]
    legacy_quotas = predecessor["legacy_quotas"]
    legacy_commitments = predecessor["legacy_rank_order_commitments"]
    split_algorithm = selection["split_algorithm"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            split_counts,
            legacy_quotas,
            legacy_commitments,
            split_algorithm,
        )
    ):
        raise ValueError("final-test manifest nested sections must be objects")
    _require_exact_keys(split_counts, set(_SPLITS), "predecessor split counts")
    _require_exact_keys(legacy_quotas, set(_SPLITS), "predecessor quotas")
    _require_exact_keys(
        legacy_commitments,
        set(_SPLITS),
        "predecessor rank commitments",
    )
    _require_exact_keys(
        split_algorithm,
        {
            "calibration_cut",
            "name",
            "required_bucket_max",
            "required_bucket_min",
            "train_cut",
        },
        "final-test split algorithm",
    )
    integer_fields = (
        manifest["schema_version"],
        manifest["protocol_version"],
        source["expected_bytes"],
        predecessor["unique_records"],
        predecessor["processed_output_bytes"],
        selection["minimum_document_bytes"],
        selection["maximum_document_bytes"],
        selection["quota_stream_bytes"],
        selection["reserve_stream_bytes"],
        selection["sequence_length"],
        *split_counts.values(),
        *legacy_quotas.values(),
    )
    if not all(_is_positive_int(value) for value in integer_fields):
        raise ValueError("final-test manifest integer fields must be positive ints")
    if (
        manifest["schema_version"] != 1
        or manifest["protocol_version"] != FINAL_TEST_PROTOCOL_VERSION
        or manifest["dataset_id"] != FINAL_TEST_DATASET_ID
        or not isinstance(manifest["purpose"], str)
        or not manifest["purpose"]
        or not _is_sha256(source["expected_sha256"])
        or not isinstance(source["filename"], str)
        or source["filename"] != "10_1.jsonl.zst"
        or not isinstance(source["url"], str)
        or not source["url"]
        or not isinstance(source["etag"], str)
        or not source["etag"]
        or not isinstance(source["last_modified"], str)
        or not source["last_modified"]
        or predecessor["dataset_id"] != "hplt3-korean-phase3"
        or predecessor["manifest_path"]
        != _SEALED_PREDECESSOR_MANIFEST_PATH
        or predecessor["data_summary_path"]
        != _SEALED_PREDECESSOR_SUMMARY_PATH
        or not all(
            _is_sha256(predecessor[field])
            for field in (
                "manifest_sha256",
                "data_summary_sha256",
                "integrity_cache_sha256",
                "processed_output_sha256",
            )
        )
        or sum(split_counts.values())
        != predecessor["unique_records"]
        or not all(
            _is_sha256(value)
            for value in legacy_commitments.values()
        )
        or predecessor["legacy_salt"] != "JamoFlow-Phase3-v1"
        or selection["record_digest"] != "sha256-exact-utf8-v1"
        or selection["required_split"] != "test"
        or selection["rank_algorithm"]
        != "sha256-domain-separated-derived-key-v1"
        or selection["separator_hex"] != "0a"
        or selection["sequence_length"] != FINAL_TEST_SEQUENCE_LENGTH
        or selection["quota_stream_bytes"] != FINAL_TEST_STREAM_BYTES
        or selection["quota_stream_bytes"]
        % selection["sequence_length"]
        != 0
        or selection["minimum_document_bytes"] != 256
        or selection["maximum_document_bytes"] != 262_144
        or selection["normalized_exclusion_algorithm"]
        != "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
        or selection["unicode_database_version"] != "15.1.0"
        or unicodedata.unidata_version != selection["unicode_database_version"]
        or selection["reserve_stream_bytes"]
        != 2 * selection["quota_stream_bytes"]
        + selection["maximum_document_bytes"]
        + 1
        or selection["order"] != ["rank_digest", "document_digest"]
        or split_algorithm
        != {
            "name": "sha256-prefix-mod-10000-v1",
            "train_cut": 8000,
            "calibration_cut": 9000,
            "required_bucket_min": 9000,
            "required_bucket_max": 9999,
        }
        or not _is_sha256(selection["expected_rank_key_hex"])
        or privacy
        != {
            "tracked_text": False,
            "tracked_individual_digests": False,
            "tracked_model_metrics": False,
        }
    ):
        raise ValueError("final-test manifest violates the sealed protocol")
    expected_key = derive_rank_key(
        source["expected_sha256"],
        predecessor["processed_output_sha256"],
        selection["quota_stream_bytes"],
        manifest["protocol_version"],
    ).hex()
    if selection["expected_rank_key_hex"] != expected_key:
        raise ValueError("final-test manifest rank key is not uniquely derived")


def load_final_test_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_final_test_manifest(manifest)
    return manifest


def load_verified_predecessor(
    manifest: Mapping[str, Any],
    *,
    predecessor_manifest_path: Path,
    predecessor_summary_path: Path,
    predecessor_integrity_path: Path,
    predecessor_output_path: Path,
) -> PredecessorIndex:
    validate_final_test_manifest(manifest)
    sealed = manifest["predecessor"]
    files = (
        (predecessor_manifest_path, "manifest_sha256"),
        (predecessor_summary_path, "data_summary_sha256"),
        (predecessor_integrity_path, "integrity_cache_sha256"),
        (predecessor_output_path, "processed_output_sha256"),
    )
    for path, field in files:
        if not path.is_file() or hash_file(path) != sealed[field]:
            raise ValueError(f"predecessor artifact mismatch: {field}")
    if predecessor_output_path.stat().st_size != sealed["processed_output_bytes"]:
        raise ValueError("predecessor processed-output size mismatch")
    legacy_manifest = json.loads(predecessor_manifest_path.read_text(encoding="utf-8"))
    integrity = json.loads(predecessor_integrity_path.read_text(encoding="utf-8"))
    summary = json.loads(predecessor_summary_path.read_text(encoding="utf-8"))
    if (
        not isinstance(summary.get("scan"), Mapping)
        or set(summary["scan"]) != _LEGACY_SCAN_KEYS
        or not all(_is_nonnegative_int(value) for value in summary["scan"].values())
    ):
        raise ValueError("predecessor source-scan summary is malformed")
    expected_legacy_selection = {
        "maximum_document_bytes": manifest["selection"]["maximum_document_bytes"],
        "minimum_document_bytes": manifest["selection"]["minimum_document_bytes"],
        "quotas": dict(sealed["legacy_quotas"]),
        "reserve_multiplier": 2.0,
        "salt": sealed["legacy_salt"],
    }
    expected_source_without_hash = {
        "etag": manifest["source"]["etag"],
        "expected_bytes": manifest["source"]["expected_bytes"],
        "filename": manifest["source"]["filename"],
        "last_modified": manifest["source"]["last_modified"],
        "url": manifest["source"]["url"],
    }
    expected_source_with_hash = {
        "bytes": manifest["source"]["expected_bytes"],
        "etag": manifest["source"]["etag"],
        "filename": manifest["source"]["filename"],
        "last_modified": manifest["source"]["last_modified"],
        "sha256": manifest["source"]["expected_sha256"],
        "url": manifest["source"]["url"],
    }
    if (
        legacy_manifest.get("dataset_id") != sealed["dataset_id"]
        or legacy_manifest.get("source") != expected_source_without_hash
        or legacy_manifest.get("selection") != expected_legacy_selection
        or integrity.get("dataset_id") != sealed["dataset_id"]
        or integrity.get("source") != expected_source_with_hash
        or integrity.get("selection") != expected_legacy_selection
        or integrity.get("output", {}).get("output_sha256")
        != sealed["processed_output_sha256"]
        or integrity.get("output", {}).get("output_bytes")
        != sealed["processed_output_bytes"]
        or summary.get("dataset_id") != sealed["dataset_id"]
        or summary.get("source") != expected_source_with_hash
        or summary.get("selection") != expected_legacy_selection
        or summary.get("scan") != integrity.get("scan")
        or summary.get("output", {}).get("sha256")
        != sealed["processed_output_sha256"]
        or summary.get("output", {}).get("records")
        != sealed["unique_records"]
    ):
        raise ValueError("predecessor aggregate metadata mismatch")
    if summary.get("integrity") != {
        "all_neural_stream_quotas_exact": True,
        "distinct_split_digest_set_summaries": True,
        "raw_or_processed_text_promoted": False,
        "scan_record_accounting_closed": True,
        "source_line_accounting_closed": True,
    }:
        raise ValueError("predecessor publication-integrity flags are not sealed")
    for split in _SPLITS:
        summary_split = summary.get("splits", {}).get(split, {})
        integrity_split = (
            integrity.get("output", {}).get("splits", {}).get(split, {})
        )
        if (
            summary_split.get("records") != sealed["split_counts"][split]
            or integrity_split.get("records") != sealed["split_counts"][split]
            or summary_split.get("selected_digest_set_sha256")
            != sealed["legacy_rank_order_commitments"][split]
            or integrity_split.get("selected_digest_set_sha256")
            != sealed["legacy_rank_order_commitments"][split]
        ):
            raise ValueError(f"predecessor aggregate split mismatch: {split}")

    by_split: dict[str, list[Candidate]] = {split: [] for split in _SPLITS}
    seen: set[bytes] = set()
    normalized_seen: set[bytes] = set()
    minimum = manifest["selection"]["minimum_document_bytes"]
    maximum = manifest["selection"]["maximum_document_bytes"]
    with predecessor_output_path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("predecessor JSONL is malformed") from exc
            if (
                not isinstance(value, dict)
                or set(value) != {"language", "text"}
                or value.get("language") != "ko"
                or not isinstance(value.get("text"), str)
            ):
                raise ValueError("predecessor JSONL schema is malformed")
            try:
                raw = value["text"].encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError("predecessor text is not strict UTF-8") from exc
            if not minimum <= len(raw) <= maximum:
                raise ValueError("predecessor document length is out of bounds")
            digest = sha256(raw).digest()
            if digest in seen:
                raise ValueError("predecessor JSONL contains a duplicate document")
            seen.add(digest)
            normalized_seen.add(normalized_record_digest(value["text"]))
            split = _record_split(digest, raw, value["text"])
            by_split[split].append(
                Candidate(
                    rank=_legacy_rank(sealed["legacy_salt"], digest),
                    digest=digest,
                    raw=raw,
                )
            )
    if len(seen) != sealed["unique_records"]:
        raise ValueError("predecessor document count mismatch")
    for split in _SPLITS:
        if (
            len(by_split[split]) != sealed["split_counts"][split]
            or _legacy_commitment(by_split[split])
            != sealed["legacy_rank_order_commitments"][split]
        ):
            raise ValueError(f"predecessor split commitment mismatch: {split}")
    frozen = {
        split: frozenset(candidate.digest for candidate in by_split[split])
        for split in _SPLITS
    }
    return PredecessorIndex(
        digests_by_split=frozen,
        all_digests=frozenset(seen),
        exclusion_commitment_sha256=digest_set_commitment(
            seen,
            domain=_EXCLUSION_DOMAIN,
        ),
        normalized_digests=frozenset(normalized_seen),
        normalized_exclusion_commitment_sha256=digest_set_commitment(
            normalized_seen,
            domain=_NORMALIZED_EXCLUSION_DOMAIN,
        ),
        source_scan={key: int(value) for key, value in summary["scan"].items()},
    )


def scan_final_test_lines(
    lines: Iterable[bytes],
    manifest: Mapping[str, Any],
    predecessor: PredecessorIndex,
) -> tuple[list[FinalTestCandidate], FinalTestScanStatistics]:
    validate_final_test_manifest(manifest)
    selection = manifest["selection"]
    predecessor_sealed = manifest["predecessor"]
    rank_key = derive_rank_key(
        manifest["source"]["expected_sha256"],
        predecessor_sealed["processed_output_sha256"],
        selection["quota_stream_bytes"],
        manifest["protocol_version"],
    )
    legacy_sampler = BottomHashSampler(
        {
            split: int(predecessor_sealed["legacy_quotas"][split])
            for split in _SPLITS
        },
        reserve_multiplier=2.0,
        maximum_document_bytes=selection["maximum_document_bytes"],
    )
    statistics = FinalTestScanStatistics()
    seen: dict[bytes, tuple[int, bytes]] = {}
    found_predecessor: set[bytes] = set()
    candidate_by_normalized_digest: dict[bytes, FinalTestCandidate] = {}

    for raw_line in lines:
        statistics.source_lines += 1
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            statistics.invalid_json += 1
            continue
        statistics.parsed_records += 1
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            statistics.missing_text += 1
            continue
        text = value["text"]
        try:
            raw = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            statistics.invalid_utf8 += 1
            continue
        if not raw:
            statistics.empty_text += 1
            continue
        if len(raw) < selection["minimum_document_bytes"]:
            statistics.too_short += 1
            continue
        if len(raw) > selection["maximum_document_bytes"]:
            statistics.too_long += 1
            continue
        digest = sha256(raw).digest()
        collision_guard = (len(raw), sha512(raw).digest())
        if digest in seen:
            if seen[digest] != collision_guard:
                raise ValueError("SHA-256 collision detected in HPLT source")
            statistics.exact_duplicates += 1
            continue
        seen[digest] = collision_guard
        split = _record_split(digest, raw, text)
        legacy_candidate = Candidate(
            rank=_legacy_rank(predecessor_sealed["legacy_salt"], digest),
            digest=digest,
            raw=raw,
        )
        legacy_sampler.add(split, legacy_candidate)  # type: ignore[arg-type]
        statistics.eligible_records += 1
        statistics.eligible_text_bytes += len(raw)
        if digest in predecessor.all_digests:
            found_predecessor.add(digest)
            statistics.predecessor_records_found += 1
            continue
        if split != "test":
            continue
        normalized_digest = normalized_record_digest(text)
        if normalized_digest in predecessor.normalized_digests:
            statistics.normalized_predecessor_exclusions += 1
            continue
        statistics.post_exclusion_test_records += 1
        statistics.post_exclusion_test_text_bytes += len(raw)
        candidate = FinalTestCandidate(
            rank=final_test_rank_digest(rank_key, digest),
            digest=digest,
            raw=raw,
        )
        previous = candidate_by_normalized_digest.get(normalized_digest)
        if previous is not None:
            statistics.normalized_source_duplicates += 1
            if (candidate.rank, candidate.digest) >= (
                previous.rank,
                previous.digest,
            ):
                continue
        candidate_by_normalized_digest[normalized_digest] = candidate

    if found_predecessor != set(predecessor.all_digests):
        raise ValueError("raw shard does not contain every predecessor document")
    rebuilt_scan = {
        key: int(getattr(statistics, key))
        for key in _LEGACY_SCAN_KEYS
    }
    if rebuilt_scan != predecessor.source_scan:
        raise ValueError("raw source-scan accounting differs from predecessor seal")
    rebuilt = legacy_sampler.finalize()
    for split in _SPLITS:
        rebuilt_digests = frozenset(
            candidate.digest for candidate in rebuilt[split]  # type: ignore[index]
        )
        if rebuilt_digests != predecessor.digests_by_split[split]:
            raise ValueError(f"raw predecessor reconstruction mismatch: {split}")
    selected = select_final_test_prefix_exhaustive(
        candidate_by_normalized_digest.values(),
        selection["quota_stream_bytes"],
    )
    if any(candidate.digest in predecessor.all_digests for candidate in selected):
        raise ValueError("final-test selection intersects predecessor documents")
    return selected, statistics


def _stream_bytes(candidates: Sequence[FinalTestCandidate]) -> bytes:
    return b"\n".join(candidate.raw for candidate in candidates)


def serialize_final_test_jsonl(
    candidates: Sequence[FinalTestCandidate],
) -> bytes:
    output = bytearray()
    for candidate in candidates:
        payload = {
            "language": "ko",
            "text": candidate.raw.decode("utf-8", errors="strict"),
        }
        output.extend(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        output.extend(b"\n")
    return bytes(output)


def _validate_selected_candidates(
    manifest: Mapping[str, Any],
    predecessor: PredecessorIndex,
    candidates: Sequence[FinalTestCandidate],
    output_bytes: bytes,
) -> bytes:
    if not candidates:
        raise ValueError("final-test selection is empty")
    selection = manifest["selection"]
    rank_key = derive_rank_key(
        manifest["source"]["expected_sha256"],
        manifest["predecessor"]["processed_output_sha256"],
        selection["quota_stream_bytes"],
        manifest["protocol_version"],
    )
    keys = [(candidate.rank, candidate.digest) for candidate in candidates]
    if keys != sorted(keys) or len({digest for _, digest in keys}) != len(keys):
        raise ValueError("final-test candidates are not a unique ordered prefix")
    normalized_keys: set[bytes] = set()
    for candidate in candidates:
        if len(candidate.rank) != 32 or len(candidate.digest) != 32:
            raise ValueError("final-test candidate rank/digest is malformed")
        if not (
            selection["minimum_document_bytes"]
            <= len(candidate.raw)
            <= selection["maximum_document_bytes"]
        ):
            raise ValueError("final-test candidate length is out of bounds")
        try:
            text = candidate.raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("final-test candidate is not strict UTF-8") from exc
        digest = sha256(candidate.raw).digest()
        if candidate.digest != digest:
            raise ValueError("final-test candidate digest does not match its text")
        if candidate.rank != final_test_rank_digest(rank_key, digest):
            raise ValueError("final-test candidate rank is not protocol-derived")
        if _record_split(digest, candidate.raw, text) != "test":
            raise ValueError("final-test candidate is outside the stable test split")
        if digest in predecessor.all_digests:
            raise ValueError("final-test candidate intersects predecessor data")
        normalized_digest = normalized_record_digest(text)
        if normalized_digest in normalized_keys:
            raise ValueError("final-test candidates contain a normalized duplicate")
        normalized_keys.add(normalized_digest)
        if normalized_digest in predecessor.normalized_digests:
            raise ValueError("final-test candidate matches normalized predecessor data")
    joined = _stream_bytes(candidates)
    quota = selection["quota_stream_bytes"]
    previous_length = len(joined) - len(candidates[-1].raw)
    if len(candidates) > 1:
        previous_length -= 1
    if previous_length >= quota or len(joined) < quota:
        raise ValueError("final-test candidates are not the minimal quota prefix")
    if output_bytes != serialize_final_test_jsonl(candidates):
        raise ValueError("final-test JSONL does not match the selected prefix")
    return joined


def build_final_test_seal_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    preparation_git_commit: str,
    source_bytes: int,
    source_sha256: str,
    predecessor: PredecessorIndex,
    statistics: FinalTestScanStatistics,
    candidates: Sequence[FinalTestCandidate],
    output_bytes: bytes,
) -> dict[str, Any]:
    validate_final_test_manifest(manifest)
    if (
        not _is_sha256(manifest_sha256)
        or not _is_sha256(source_sha256)
        or not isinstance(preparation_git_commit, str)
        or len(preparation_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in preparation_git_commit)
        or source_bytes != manifest["source"]["expected_bytes"]
        or source_sha256 != manifest["source"]["expected_sha256"]
    ):
        raise ValueError("final-test seal source/implementation identity mismatch")
    joined = _validate_selected_candidates(
        manifest,
        predecessor,
        candidates,
        output_bytes,
    )
    quota = manifest["selection"]["quota_stream_bytes"]
    evaluation_stream = joined[:quota]
    selected_digests = frozenset(candidate.digest for candidate in candidates)
    selected_commitment = digest_set_commitment(
        selected_digests,
        domain=_SELECTED_DOMAIN,
    )
    intersection = predecessor.all_digests & selected_digests
    overlap = overlap_audit_commitment(
        predecessor.exclusion_commitment_sha256,
        selected_commitment,
        intersection,
    )
    if overlap["intersection_count"] != 0:
        raise ValueError("final-test overlap audit failed")
    selected_normalized_digests = frozenset(
        normalized_record_digest(candidate.raw.decode("utf-8", errors="strict"))
        for candidate in candidates
    )
    normalized_selected_commitment = digest_set_commitment(
        selected_normalized_digests,
        domain=_NORMALIZED_SELECTED_DOMAIN,
    )
    normalized_overlap = overlap_audit_commitment(
        predecessor.normalized_exclusion_commitment_sha256,
        normalized_selected_commitment,
        predecessor.normalized_digests & selected_normalized_digests,
        intersection_domain=_NORMALIZED_INTERSECTION_DOMAIN,
        audit_domain=_NORMALIZED_OVERLAP_AUDIT_DOMAIN,
    )
    if normalized_overlap["intersection_count"] != 0:
        raise ValueError("final-test normalized-overlap audit failed")
    if (
        statistics.source_lines
        != statistics.parsed_records + statistics.invalid_json
        or statistics.parsed_records
        != statistics.missing_text
        + statistics.invalid_utf8
        + statistics.empty_text
        + statistics.too_short
        + statistics.too_long
        + statistics.eligible_records
        or statistics.predecessor_records_found != len(predecessor.all_digests)
        or statistics.post_exclusion_test_records < len(candidates)
        or statistics.post_exclusion_test_records
        - statistics.normalized_source_duplicates
        < len(candidates)
        or statistics.post_exclusion_test_text_bytes
        < sum(len(candidate.raw) for candidate in candidates)
    ):
        raise ValueError("final-test scan accounting is not closed")
    payload = {
        "dataset_id": FINAL_TEST_DATASET_ID,
        "manifest": {
            "sha256": manifest_sha256,
            "schema_version": manifest["schema_version"],
        },
        "output": {
            "evaluation_stream_bytes": len(evaluation_stream),
            "evaluation_stream_sha256": sha256(evaluation_stream).hexdigest(),
            "full_jsonl_bytes": len(output_bytes),
            "full_jsonl_sha256": sha256(output_bytes).hexdigest(),
            "sequence_count": (
                len(evaluation_stream)
                // manifest["selection"]["sequence_length"]
            ),
            "sequence_length": manifest["selection"]["sequence_length"],
        },
        "predecessor": {
            "document_count": len(predecessor.all_digests),
            "exclusion_commitment_sha256": (
                predecessor.exclusion_commitment_sha256
            ),
            "legacy_reconstruction_verified": True,
            "normalized_exclusion_commitment_sha256": (
                predecessor.normalized_exclusion_commitment_sha256
            ),
            "normalized_unique_document_count": len(
                predecessor.normalized_digests
            ),
        },
        "preparation_git_commit": preparation_git_commit,
        "privacy": {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        },
        "protocol_version": FINAL_TEST_PROTOCOL_VERSION,
        "scan": statistics.to_dict(),
        "selection": {
            "all_records_stable_test": True,
            "ordered_selection_sha256": ordered_selection_commitment(candidates),
            "normalized_intersection_commitment_sha256": normalized_overlap[
                "intersection_commitment_sha256"
            ],
            "normalized_intersection_count": normalized_overlap[
                "intersection_count"
            ],
            "normalized_overlap_audit_sha256": normalized_overlap[
                "overlap_audit_sha256"
            ],
            "normalized_selected_set_commitment_sha256": (
                normalized_selected_commitment
            ),
            "normalized_selected_unique_document_count": len(
                selected_normalized_digests
            ),
            "overshoot_stream_bytes": len(joined) - quota,
            "selected_document_count": len(candidates),
            "selected_document_raw_bytes": sum(
                len(candidate.raw) for candidate in candidates
            ),
            "selected_set_commitment_sha256": selected_commitment,
            **overlap,
        },
        "source": {
            "bytes": source_bytes,
            "sha256": source_sha256,
        },
    }
    if (
        payload["output"]["evaluation_stream_bytes"]
        != FINAL_TEST_STREAM_BYTES
        or payload["output"]["sequence_count"] != FINAL_TEST_SEQUENCE_COUNT
        or payload["selection"]["overshoot_stream_bytes"]
        > manifest["selection"]["maximum_document_bytes"]
    ):
        raise ValueError("final-test seal quota accounting failed")
    validate_final_test_seal_payload(payload)
    return payload


def seal_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_final_test_seal_payload(payload)
    return {
        "payload": dict(payload),
        "payload_sha256": canonical_payload_sha256(payload),
    }


def serialize_seal_envelope(envelope: Mapping[str, Any]) -> bytes:
    validate_seal_envelope(envelope)
    return (
        json.dumps(
            envelope,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_final_test_seal_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("final-test seal payload must be an object")
    _require_exact_keys(payload, _SEAL_PAYLOAD_KEYS, "final-test seal payload")
    sections = (
        payload["manifest"],
        payload["output"],
        payload["predecessor"],
        payload["privacy"],
        payload["scan"],
        payload["selection"],
        payload["source"],
    )
    if not all(isinstance(value, Mapping) for value in sections):
        raise ValueError("final-test seal sections must be objects")
    manifest = payload["manifest"]
    output = payload["output"]
    predecessor = payload["predecessor"]
    privacy = payload["privacy"]
    scan = payload["scan"]
    selection = payload["selection"]
    source = payload["source"]
    _require_exact_keys(manifest, _SEAL_MANIFEST_KEYS, "sealed manifest")
    _require_exact_keys(output, _SEAL_OUTPUT_KEYS, "sealed output")
    _require_exact_keys(
        predecessor,
        _SEAL_PREDECESSOR_KEYS,
        "sealed predecessor",
    )
    _require_exact_keys(privacy, _SEAL_PRIVACY_KEYS, "sealed privacy")
    _require_exact_keys(
        scan,
        set(FinalTestScanStatistics.__dataclass_fields__),
        "sealed scan",
    )
    _require_exact_keys(selection, _SEAL_SELECTION_KEYS, "sealed selection")
    _require_exact_keys(source, _SEAL_SOURCE_KEYS, "sealed source")
    commit = payload["preparation_git_commit"]
    hashes = (
        manifest["sha256"],
        output["evaluation_stream_sha256"],
        output["full_jsonl_sha256"],
        predecessor["exclusion_commitment_sha256"],
        predecessor["normalized_exclusion_commitment_sha256"],
        selection["intersection_commitment_sha256"],
        selection["normalized_intersection_commitment_sha256"],
        selection["normalized_overlap_audit_sha256"],
        selection["normalized_selected_set_commitment_sha256"],
        selection["ordered_selection_sha256"],
        selection["overlap_audit_sha256"],
        selection["selected_set_commitment_sha256"],
        source["sha256"],
    )
    positive_integers = (
        output["evaluation_stream_bytes"],
        output["full_jsonl_bytes"],
        output["sequence_count"],
        output["sequence_length"],
        predecessor["document_count"],
        predecessor["normalized_unique_document_count"],
        selection["selected_document_count"],
        selection["selected_document_raw_bytes"],
        selection["normalized_selected_unique_document_count"],
        source["bytes"],
    )
    if (
        payload["dataset_id"] != FINAL_TEST_DATASET_ID
        or not _is_positive_int(payload["protocol_version"])
        or payload["protocol_version"] != FINAL_TEST_PROTOCOL_VERSION
        or not _is_positive_int(manifest["schema_version"])
        or manifest["schema_version"] != 1
        or not all(_is_sha256(value) for value in hashes)
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not all(_is_positive_int(value) for value in positive_integers)
        or not all(_is_nonnegative_int(value) for value in scan.values())
        or not _is_nonnegative_int(selection["overshoot_stream_bytes"])
        or not _is_nonnegative_int(selection["intersection_count"])
        or not _is_nonnegative_int(selection["normalized_intersection_count"])
        or output["evaluation_stream_bytes"] != FINAL_TEST_STREAM_BYTES
        or output["sequence_length"] != FINAL_TEST_SEQUENCE_LENGTH
        or output["sequence_count"] != FINAL_TEST_SEQUENCE_COUNT
        or predecessor["legacy_reconstruction_verified"] is not True
        or predecessor["normalized_unique_document_count"]
        > predecessor["document_count"]
        or selection["all_records_stable_test"] is not True
        or selection["intersection_count"] != 0
        or selection["normalized_intersection_count"] != 0
        or selection["normalized_selected_unique_document_count"]
        != selection["selected_document_count"]
        or privacy
        != {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        }
        or scan["source_lines"]
        != scan["parsed_records"] + scan["invalid_json"]
        or scan["parsed_records"]
        != scan["missing_text"]
        + scan["invalid_utf8"]
        + scan["empty_text"]
        + scan["too_short"]
        + scan["too_long"]
        + scan["eligible_records"]
        or scan["predecessor_records_found"] != predecessor["document_count"]
        or scan["post_exclusion_test_records"]
        < selection["selected_document_count"]
        or scan["normalized_predecessor_exclusions"]
        > scan["eligible_records"] - scan["predecessor_records_found"]
        or scan["normalized_source_duplicates"]
        > scan["post_exclusion_test_records"]
        or scan["post_exclusion_test_records"]
        - scan["normalized_source_duplicates"]
        < selection["selected_document_count"]
        or scan["post_exclusion_test_text_bytes"]
        < selection["selected_document_raw_bytes"]
        or selection["selected_document_raw_bytes"]
        + selection["selected_document_count"]
        - 1
        != FINAL_TEST_STREAM_BYTES + selection["overshoot_stream_bytes"]
        or selection["overshoot_stream_bytes"] > 262_144
        or output["full_jsonl_bytes"]
        <= selection["selected_document_raw_bytes"]
    ):
        raise ValueError("final-test seal payload violates the sealed schema")


def validate_seal_envelope(envelope: Mapping[str, Any]) -> None:
    if set(envelope) != {"payload", "payload_sha256"} or not isinstance(
        envelope["payload"], Mapping
    ):
        raise ValueError("final-test seal envelope is malformed")
    validate_final_test_seal_payload(envelope["payload"])
    if envelope["payload_sha256"] != canonical_payload_sha256(
        envelope["payload"]
    ):
        raise ValueError("final-test seal payload hash mismatch")


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def publish_no_clobber(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing sealed artifact differs: {path}")
        return
    stage = path.with_suffix(path.suffix + ".preparing")
    if stage.exists():
        if stage.read_bytes() != data:
            raise ValueError(f"staged sealed artifact differs: {stage}")
    else:
        _exclusive_write(stage, data)
    try:
        os.link(stage, path)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError(f"sealed artifact was concurrently replaced: {path}")
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    stage.unlink(missing_ok=True)


def reconstruct_final_test(
    *,
    manifest_path: Path,
    archive_path: Path,
    predecessor_manifest_path: Path,
    predecessor_summary_path: Path,
    predecessor_integrity_path: Path,
    predecessor_output_path: Path,
    preparation_git_commit: str,
) -> tuple[bytes, dict[str, Any]]:
    manifest = load_final_test_manifest(manifest_path)
    if (
        archive_path.stat().st_size != manifest["source"]["expected_bytes"]
        or hash_file(archive_path) != manifest["source"]["expected_sha256"]
    ):
        raise ValueError("pinned HPLT source archive does not match final-test manifest")
    predecessor = load_verified_predecessor(
        manifest,
        predecessor_manifest_path=predecessor_manifest_path,
        predecessor_summary_path=predecessor_summary_path,
        predecessor_integrity_path=predecessor_integrity_path,
        predecessor_output_path=predecessor_output_path,
    )
    candidates, statistics = scan_final_test_lines(
        iter_zstd_jsonl_lines(archive_path),
        manifest,
        predecessor,
    )
    output = serialize_final_test_jsonl(candidates)
    payload = build_final_test_seal_payload(
        manifest=manifest,
        manifest_sha256=hash_file(manifest_path),
        preparation_git_commit=preparation_git_commit,
        source_bytes=archive_path.stat().st_size,
        source_sha256=manifest["source"]["expected_sha256"],
        predecessor=predecessor,
        statistics=statistics,
        candidates=candidates,
        output_bytes=output,
    )
    return output, seal_envelope(payload)
