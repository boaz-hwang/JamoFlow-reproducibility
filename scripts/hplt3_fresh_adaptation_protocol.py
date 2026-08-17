"""Deterministic fresh Korean train/calibration data for vocabulary adaptation.

The new streams come from the pinned HPLT3 shard but exclude every document in
the historical Phase-3 sample and the sealed final test, both exactly and under
the repository's normalized-deduplication rule.  Selection is model-free and
uses rank keys uniquely derived from already sealed artifact hashes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256, sha512
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

from jamoflow.corpus import Record, split_for_record
from jamoflow.hplt3 import hash_file, iter_zstd_jsonl_lines
from jamoflow.hplt3_final_test import (
    FinalTestBottomHashSampler,
    FinalTestCandidate,
    canonical_payload_sha256,
    digest_set_commitment,
    load_final_test_manifest,
    load_verified_predecessor,
    normalized_record_digest,
    validate_seal_envelope as validate_final_test_seal,
)


PROTOCOL_VERSION = 1
DATASET_ID = "hplt3-korean-vocab-adaptation-v1"
SEQUENCE_LENGTH = 512
SPLIT_QUOTAS = {"train": 128_000_000, "calibration": 8_000_000}
SPLIT_ORDER = ("train", "calibration")

_SOURCE_SHA256 = "de4dfa43fd9f6c62cc81781e09c1f401cc77e7a956e07ecc80ac13477e699ca4"
_PREDECESSOR_MANIFEST_SHA256 = (
    "873a303e9762906e0046cdf59f850b8d6dbeb4483cfff8ad5a6f124e00002d9e"
)
_PREDECESSOR_SUMMARY_SHA256 = (
    "5d2890a0154c0e98fe27c4d8801d5ffacea9217e25440d2631f7354e2c37f9ba"
)
_PREDECESSOR_INTEGRITY_SHA256 = (
    "472cc5da045909109718be71168e516be19043cb2a08363d573ed77650038181"
)
_PREDECESSOR_OUTPUT_SHA256 = (
    "f789bc7e0ec0252c4c7c636e67a7c44f6d2c528a292ec47542af98488c8b36a5"
)
_FINAL_MANIFEST_SHA256 = (
    "ea36dbbac63be8b9370cfee029758030cdd138aa1510bc02aeb8cbe4b95590ff"
)
_FINAL_SEAL_SHA256 = (
    "ce42e8a0b2d8161cc59e0b30d5d121b547e22d28709fe48284aa777df4a2290b"
)
_FINAL_PAYLOAD_SHA256 = (
    "97cf90d1e6e7191e7f8336647f278ae6c0e82d70540bf0f5c43f9cb426e75dc8"
)
_FINAL_OUTPUT_SHA256 = (
    "098ae8b833a1498689dae1d60341aa870fce51c7f9dde6d961c867f751ee3dc2"
)
_FRESH_MANIFEST_SHA256 = (
    "7325c2562e4e02c3b0f71fde5ac3ab7f3bb383ce17bb0756728c6cb59506859f"
)

_KEY_DOMAIN = b"JamoFlow/fresh-adaptation-key/v1\0"
_RANK_DOMAIN = b"JamoFlow/fresh-adaptation-rank/v1\0"
_SELECTED_DOMAINS = {
    "train": b"JamoFlow/fresh-adaptation-selected-train/v1\0",
    "calibration": b"JamoFlow/fresh-adaptation-selected-calibration/v1\0",
}
_NORMALIZED_SELECTED_DOMAINS = {
    "train": b"JamoFlow/fresh-adaptation-normalized-train/v1\0",
    "calibration": b"JamoFlow/fresh-adaptation-normalized-calibration/v1\0",
}

_TOP_LEVEL_KEYS = {
    "dataset_id",
    "final_test",
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
    "integrity_path",
    "integrity_sha256",
    "manifest_path",
    "manifest_sha256",
    "output_bytes",
    "output_path",
    "output_sha256",
    "summary_path",
    "summary_sha256",
    "unique_records",
}
_FINAL_TEST_KEYS = {
    "manifest_path",
    "manifest_sha256",
    "output_bytes",
    "output_path",
    "output_sha256",
    "payload_sha256",
    "seal_path",
    "seal_sha256",
    "selected_document_count",
}
_SELECTION_KEYS = {
    "expected_rank_key_hex",
    "maximum_document_bytes",
    "minimum_document_bytes",
    "normalized_exclusion_algorithm",
    "order",
    "quotas",
    "rank_algorithm",
    "record_digest",
    "reserve_stream_bytes",
    "sequence_length",
    "split_algorithm",
}
_PRIVACY_KEYS = {
    "tracked_individual_digests",
    "tracked_model_metrics",
    "tracked_text",
}

_SEAL_PAYLOAD_KEYS = {
    "dataset_id",
    "exclusions",
    "manifest_sha256",
    "output",
    "preparation_git_commit",
    "privacy",
    "protocol_version",
    "scan",
    "source",
    "splits",
}
_SEAL_EXCLUSION_KEYS = {
    "exact_commitment_sha256",
    "exact_count",
    "final_exact_count",
    "normalized_commitment_sha256",
    "normalized_count",
    "predecessor_exact_count",
}
_SEAL_OUTPUT_KEYS = {"bytes", "sha256"}
_SEAL_PRIVACY_KEYS = {
    "individual_document_digests_tracked",
    "model_metrics_tracked",
    "raw_text_tracked",
}
_SEAL_SOURCE_KEYS = {"bytes", "sha256"}
_SEALED_SPLIT_KEYS = {
    "available_stream_bytes",
    "normalized_selected_set_commitment_sha256",
    "overshoot_stream_bytes",
    "selected_document_count",
    "selected_document_raw_bytes",
    "selected_set_commitment_sha256",
    "sequence_count",
    "stream_bytes",
    "stream_sha256",
}


def _u64(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("fresh-data commitment integer differs")
    return struct.pack(">Q", value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_nonnegative_int(value) and value > 0


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def derive_split_rank_key(
    source_sha256: str,
    predecessor_output_sha256: str,
    final_output_sha256: str,
    split: str,
    quota: int,
    protocol_version: int = PROTOCOL_VERSION,
) -> bytes:
    if (
        not all(
            _is_sha256(value)
            for value in (
                source_sha256,
                predecessor_output_sha256,
                final_output_sha256,
            )
        )
        or split not in SPLIT_ORDER
        or quota != SPLIT_QUOTAS[split]
        or protocol_version != PROTOCOL_VERSION
    ):
        raise ValueError("fresh-data rank-key inputs differ")
    return sha256(
        _KEY_DOMAIN
        + bytes.fromhex(source_sha256)
        + bytes.fromhex(predecessor_output_sha256)
        + bytes.fromhex(final_output_sha256)
        + split.encode("ascii")
        + b"\0"
        + _u64(quota)
        + _u64(protocol_version)
    ).digest()


def rank_digest(rank_key: bytes, document_digest: bytes) -> bytes:
    if len(rank_key) != 32 or len(document_digest) != 32:
        raise ValueError("fresh-data rank inputs differ")
    return sha256(_RANK_DOMAIN + rank_key + document_digest).digest()


def _record_split(digest: bytes, raw: bytes, text: str) -> str:
    return split_for_record(
        Record(
            record_id=digest.hex(),
            source="hplt3",
            ordinal=0,
            raw=raw,
            text=text,
        )
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("fresh-data manifest must be an object")
    _exact_keys(manifest, _TOP_LEVEL_KEYS, "fresh-data manifest")
    source = manifest["source"]
    predecessor = manifest["predecessor"]
    final_test = manifest["final_test"]
    selection = manifest["selection"]
    privacy = manifest["privacy"]
    if not all(
        isinstance(value, Mapping)
        for value in (source, predecessor, final_test, selection, privacy)
    ):
        raise ValueError("fresh-data manifest sections differ")
    _exact_keys(source, _SOURCE_KEYS, "fresh-data source")
    _exact_keys(predecessor, _PREDECESSOR_KEYS, "fresh-data predecessor")
    _exact_keys(final_test, _FINAL_TEST_KEYS, "fresh-data final test")
    _exact_keys(selection, _SELECTION_KEYS, "fresh-data selection")
    _exact_keys(privacy, _PRIVACY_KEYS, "fresh-data privacy")
    rank_keys = selection["expected_rank_key_hex"]
    quotas = selection["quotas"]
    reserves = selection["reserve_stream_bytes"]
    if not all(isinstance(value, Mapping) for value in (rank_keys, quotas, reserves)):
        raise ValueError("fresh-data split maps differ")
    for value in (rank_keys, quotas, reserves):
        _exact_keys(value, set(SPLIT_ORDER), "fresh-data split map")
    if (
        manifest["schema_version"] != 1
        or manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["dataset_id"] != DATASET_ID
        or not isinstance(manifest["purpose"], str)
        or not manifest["purpose"]
        or source["filename"] != "10_1.jsonl.zst"
        or source["url"]
        != "https://data.hplt-project.org/three/sorted/kor_Hang/10_1.jsonl.zst"
        or source["expected_bytes"] != 1_862_302_013
        or source["expected_sha256"] != _SOURCE_SHA256
        or source["etag"] != '"6f00793d-63be01a95c540"'
        or source["last_modified"] != "Fri, 08 Aug 2025 20:06:05 GMT"
        or predecessor["unique_records"] != 6_911
        or predecessor["manifest_path"] != "data/manifests/hplt3-korean-phase3.json"
        or predecessor["summary_path"] != "results/phase3-data/summary.json"
        or predecessor["integrity_path"]
        != "data/processed/hplt3-korean-phase3/integrity.json"
        or predecessor["output_path"]
        != "data/processed/hplt3-korean-phase3/ko.jsonl"
        or predecessor["output_bytes"] != 152_461_842
        or predecessor["manifest_sha256"] != _PREDECESSOR_MANIFEST_SHA256
        or predecessor["summary_sha256"] != _PREDECESSOR_SUMMARY_SHA256
        or predecessor["integrity_sha256"] != _PREDECESSOR_INTEGRITY_SHA256
        or predecessor["output_sha256"] != _PREDECESSOR_OUTPUT_SHA256
        or final_test["output_bytes"] != 32_103_596
        or final_test["manifest_path"]
        != "data/manifests/hplt3-korean-final-test-v1.json"
        or final_test["seal_path"] != "data/seals/hplt3-korean-final-test-v1.json"
        or final_test["output_path"]
        != "data/processed/hplt3-korean-final-test-v1/ko.jsonl"
        or final_test["manifest_sha256"] != _FINAL_MANIFEST_SHA256
        or final_test["seal_sha256"] != _FINAL_SEAL_SHA256
        or final_test["payload_sha256"] != _FINAL_PAYLOAD_SHA256
        or final_test["output_sha256"] != _FINAL_OUTPUT_SHA256
        or final_test["selected_document_count"] != 1_482
        or selection["minimum_document_bytes"] != 256
        or selection["maximum_document_bytes"] != 262_144
        or selection["sequence_length"] != SEQUENCE_LENGTH
        or selection["record_digest"] != "sha256-exact-utf8-v1"
        or selection["rank_algorithm"] != "sha256-domain-separated-derived-key-v1"
        or selection["normalized_exclusion_algorithm"]
        != "unicode-nfkc-casefold-whitespace-collapse-sha256-v1"
        or selection["order"] != ["rank_digest", "document_digest"]
        or selection["split_algorithm"]
        != {
            "name": "sha256-prefix-mod-10000-v1",
            "train_cut": 8000,
            "calibration_cut": 9000,
            "required_splits": list(SPLIT_ORDER),
        }
        or dict(quotas) != SPLIT_QUOTAS
        or any(
            reserves[split]
            != quotas[split] + selection["maximum_document_bytes"] + 1
            for split in SPLIT_ORDER
        )
        or privacy
        != {
            "tracked_individual_digests": False,
            "tracked_model_metrics": False,
            "tracked_text": False,
        }
    ):
        raise ValueError("fresh-data manifest violates the protocol")
    for split in SPLIT_ORDER:
        expected = derive_split_rank_key(
            source["expected_sha256"],
            predecessor["output_sha256"],
            final_test["output_sha256"],
            split,
            quotas[split],
        ).hex()
        if rank_keys[split] != expected:
            raise ValueError("fresh-data rank key is not uniquely derived")


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    return value


@dataclass(frozen=True, slots=True)
class ExclusionIndex:
    exact: frozenset[bytes]
    normalized: frozenset[bytes]
    predecessor_exact_count: int
    final_exact_count: int


def load_exclusions(
    manifest: Mapping[str, Any],
    *,
    final_manifest_path: Path,
    final_seal_path: Path,
    final_output_path: Path,
    predecessor_manifest_path: Path,
    predecessor_summary_path: Path,
    predecessor_integrity_path: Path,
    predecessor_output_path: Path,
) -> ExclusionIndex:
    validate_manifest(manifest)
    predecessor_contract = manifest["predecessor"]
    final_contract = manifest["final_test"]
    if (
        hash_file(final_manifest_path) != final_contract["manifest_sha256"]
        or hash_file(final_seal_path) != final_contract["seal_sha256"]
        or hash_file(predecessor_manifest_path)
        != predecessor_contract["manifest_sha256"]
        or hash_file(predecessor_summary_path)
        != predecessor_contract["summary_sha256"]
        or hash_file(predecessor_integrity_path)
        != predecessor_contract["integrity_sha256"]
        or hash_file(predecessor_output_path)
        != predecessor_contract["output_sha256"]
        or predecessor_output_path.stat().st_size
        != predecessor_contract["output_bytes"]
    ):
        raise ValueError("fresh-data sealed dependency differs")
    sealed_final_manifest = load_final_test_manifest(final_manifest_path)
    sealed_predecessor = sealed_final_manifest["predecessor"]
    if (
        sealed_predecessor["manifest_sha256"]
        != predecessor_contract["manifest_sha256"]
        or sealed_predecessor["data_summary_sha256"]
        != predecessor_contract["summary_sha256"]
        or sealed_predecessor["integrity_cache_sha256"]
        != predecessor_contract["integrity_sha256"]
        or sealed_predecessor["processed_output_sha256"]
        != predecessor_contract["output_sha256"]
        or sealed_predecessor["processed_output_bytes"]
        != predecessor_contract["output_bytes"]
        or sealed_predecessor["unique_records"]
        != predecessor_contract["unique_records"]
    ):
        raise ValueError("fresh-data predecessor lineage differs")
    predecessor = load_verified_predecessor(
        sealed_final_manifest,
        predecessor_manifest_path=predecessor_manifest_path,
        predecessor_summary_path=predecessor_summary_path,
        predecessor_integrity_path=predecessor_integrity_path,
        predecessor_output_path=predecessor_output_path,
    )
    final_seal = json.loads(final_seal_path.read_text(encoding="utf-8"))
    validate_final_test_seal(final_seal)
    if (
        final_seal["payload_sha256"] != final_contract["payload_sha256"]
        or final_seal["payload"]["manifest"]["sha256"]
        != final_contract["manifest_sha256"]
        or hash_file(final_output_path) != final_contract["output_sha256"]
        or final_output_path.stat().st_size != final_contract["output_bytes"]
        or final_seal["payload"]["output"]["full_jsonl_sha256"]
        != final_contract["output_sha256"]
    ):
        raise ValueError("fresh-data final-test dependency differs")
    final_exact: set[bytes] = set()
    final_normalized: set[bytes] = set()
    with final_output_path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("final-test JSONL differs") from exc
            if (
                not isinstance(value, dict)
                or set(value) != {"language", "text"}
                or value.get("language") != "ko"
                or not isinstance(value.get("text"), str)
            ):
                raise ValueError("final-test JSONL schema differs")
            raw = value["text"].encode("utf-8", errors="strict")
            digest = sha256(raw).digest()
            if digest in final_exact or _record_split(digest, raw, value["text"]) != "test":
                raise ValueError("final-test document identity differs")
            final_exact.add(digest)
            final_normalized.add(normalized_record_digest(value["text"]))
    if (
        len(final_exact) != final_contract["selected_document_count"]
        or len(final_normalized) != len(final_exact)
        or predecessor.all_digests & final_exact
        or predecessor.normalized_digests & final_normalized
    ):
        raise ValueError("fresh-data exclusion sets overlap or differ")
    return ExclusionIndex(
        exact=frozenset(set(predecessor.all_digests) | final_exact),
        normalized=frozenset(set(predecessor.normalized_digests) | final_normalized),
        predecessor_exact_count=len(predecessor.all_digests),
        final_exact_count=len(final_exact),
    )


@dataclass(slots=True)
class FreshScanStatistics:
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
    excluded_exact_records: int = 0
    excluded_normalized_records: int = 0
    normalized_source_duplicates: int = 0
    stable_test_records_ignored: int = 0
    candidate_train_records: int = 0
    candidate_calibration_records: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def scan_lines(
    lines: Iterable[bytes],
    manifest: Mapping[str, Any],
    exclusions: ExclusionIndex,
) -> tuple[dict[str, list[FinalTestCandidate]], FreshScanStatistics]:
    validate_manifest(manifest)
    selection = manifest["selection"]
    samplers = {
        split: FinalTestBottomHashSampler(
            selection["quotas"][split],
            selection["reserve_stream_bytes"][split],
        )
        for split in SPLIT_ORDER
    }
    rank_keys = {
        split: bytes.fromhex(selection["expected_rank_key_hex"][split])
        for split in SPLIT_ORDER
    }
    statistics = FreshScanStatistics()
    seen_exact: dict[bytes, tuple[int, bytes]] = {}
    seen_normalized: set[bytes] = set()
    found_exclusions: set[bytes] = set()
    for line in lines:
        statistics.source_lines += 1
        try:
            value = json.loads(line)
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
        if digest in seen_exact:
            if seen_exact[digest] != collision_guard:
                raise ValueError("SHA-256 collision in fresh-data source")
            statistics.exact_duplicates += 1
            continue
        seen_exact[digest] = collision_guard
        statistics.eligible_records += 1
        if digest in exclusions.exact:
            found_exclusions.add(digest)
            statistics.excluded_exact_records += 1
            continue
        split = _record_split(digest, raw, text)
        if split not in samplers:
            statistics.stable_test_records_ignored += 1
            continue
        normalized = normalized_record_digest(text)
        if normalized in exclusions.normalized:
            statistics.excluded_normalized_records += 1
            continue
        if normalized in seen_normalized:
            statistics.normalized_source_duplicates += 1
            continue
        seen_normalized.add(normalized)
        setattr(
            statistics,
            f"candidate_{split}_records",
            getattr(statistics, f"candidate_{split}_records") + 1,
        )
        samplers[split].add(
            FinalTestCandidate(
                rank=rank_digest(rank_keys[split], digest),
                digest=digest,
                raw=raw,
            )
        )
    if found_exclusions != set(exclusions.exact):
        raise ValueError("raw shard does not contain every sealed exclusion")
    selected = {split: samplers[split].finalize() for split in SPLIT_ORDER}
    return selected, statistics


def stream_bytes(candidates: Sequence[FinalTestCandidate]) -> bytes:
    return b"\n".join(candidate.raw for candidate in candidates)


def serialize_jsonl(selected: Mapping[str, Sequence[FinalTestCandidate]]) -> bytes:
    if set(selected) != set(SPLIT_ORDER):
        raise ValueError("fresh-data selected split set differs")
    output = bytearray()
    for split in SPLIT_ORDER:
        for candidate in selected[split]:
            value = {
                "language": "ko",
                "text": candidate.raw.decode("utf-8", errors="strict"),
            }
            output.extend(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            output.extend(b"\n")
    return bytes(output)


def build_seal_payload(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    preparation_git_commit: str,
    exclusions: ExclusionIndex,
    selected: Mapping[str, Sequence[FinalTestCandidate]],
    statistics: FreshScanStatistics,
    output: bytes,
) -> dict[str, Any]:
    validate_manifest(manifest)
    if not _is_sha256(manifest_sha256) or not _is_git_commit(
        preparation_git_commit
    ):
        raise ValueError("fresh-data seal lineage differs")
    if output != serialize_jsonl(selected):
        raise ValueError("fresh-data serialized output differs")
    split_rows: dict[str, Any] = {}
    all_selected: set[bytes] = set()
    all_normalized: set[bytes] = set()
    for split in SPLIT_ORDER:
        candidates = tuple(selected[split])
        keys = [(candidate.rank, candidate.digest) for candidate in candidates]
        if not candidates or keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("fresh-data ordered split differs")
        raw_stream = stream_bytes(candidates)
        quota = manifest["selection"]["quotas"][split]
        previous = len(raw_stream) - len(candidates[-1].raw)
        if len(candidates) > 1:
            previous -= 1
        if previous >= quota or len(raw_stream) < quota:
            raise ValueError("fresh-data split is not its minimal quota prefix")
        normalized = {
            normalized_record_digest(candidate.raw.decode("utf-8", errors="strict"))
            for candidate in candidates
        }
        digests = {candidate.digest for candidate in candidates}
        if (
            len(digests) != len(candidates)
            or len(normalized) != len(candidates)
            or digests & exclusions.exact
            or normalized & exclusions.normalized
            or all_selected & digests
            or all_normalized & normalized
            or any(
                _record_split(
                    candidate.digest,
                    candidate.raw,
                    candidate.raw.decode("utf-8", errors="strict"),
                )
                != split
                for candidate in candidates
            )
        ):
            raise ValueError("fresh-data selected overlap or split differs")
        all_selected.update(digests)
        all_normalized.update(normalized)
        split_rows[split] = {
            "available_stream_bytes": len(raw_stream),
            "overshoot_stream_bytes": len(raw_stream) - quota,
            "selected_document_count": len(candidates),
            "selected_document_raw_bytes": sum(len(candidate.raw) for candidate in candidates),
            "selected_set_commitment_sha256": digest_set_commitment(
                digests, domain=_SELECTED_DOMAINS[split]
            ),
            "normalized_selected_set_commitment_sha256": digest_set_commitment(
                normalized, domain=_NORMALIZED_SELECTED_DOMAINS[split]
            ),
            "stream_bytes": quota,
            "stream_sha256": sha256(raw_stream[:quota]).hexdigest(),
            "sequence_count": quota // SEQUENCE_LENGTH,
        }
    payload = {
        "dataset_id": DATASET_ID,
        "exclusions": {
            "exact_count": len(exclusions.exact),
            "normalized_count": len(exclusions.normalized),
            "predecessor_exact_count": exclusions.predecessor_exact_count,
            "final_exact_count": exclusions.final_exact_count,
            "exact_commitment_sha256": digest_set_commitment(
                exclusions.exact,
                domain=b"JamoFlow/fresh-adaptation-exclusions/v1\0",
            ),
            "normalized_commitment_sha256": digest_set_commitment(
                exclusions.normalized,
                domain=b"JamoFlow/fresh-adaptation-normalized-exclusions/v1\0",
            ),
        },
        "manifest_sha256": manifest_sha256,
        "output": {
            "bytes": len(output),
            "sha256": sha256(output).hexdigest(),
        },
        "preparation_git_commit": preparation_git_commit,
        "privacy": {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        },
        "protocol_version": PROTOCOL_VERSION,
        "scan": statistics.to_dict(),
        "source": {
            "bytes": manifest["source"]["expected_bytes"],
            "sha256": manifest["source"]["expected_sha256"],
        },
        "splits": split_rows,
    }
    validate_seal_payload(payload)
    return payload


def validate_seal_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("fresh-data seal payload must be an object")
    _exact_keys(payload, _SEAL_PAYLOAD_KEYS, "fresh-data seal payload")
    sections = (
        payload["exclusions"],
        payload["output"],
        payload["privacy"],
        payload["scan"],
        payload["source"],
        payload["splits"],
    )
    if not all(isinstance(value, Mapping) for value in sections):
        raise ValueError("fresh-data seal sections differ")
    exclusions = payload["exclusions"]
    output = payload["output"]
    privacy = payload["privacy"]
    scan = payload["scan"]
    source = payload["source"]
    splits = payload["splits"]
    _exact_keys(exclusions, _SEAL_EXCLUSION_KEYS, "fresh-data sealed exclusions")
    _exact_keys(output, _SEAL_OUTPUT_KEYS, "fresh-data sealed output")
    _exact_keys(privacy, _SEAL_PRIVACY_KEYS, "fresh-data sealed privacy")
    _exact_keys(
        scan,
        set(FreshScanStatistics.__dataclass_fields__),
        "fresh-data sealed scan",
    )
    _exact_keys(source, _SEAL_SOURCE_KEYS, "fresh-data sealed source")
    _exact_keys(splits, set(SPLIT_ORDER), "fresh-data sealed splits")
    for split in SPLIT_ORDER:
        if not isinstance(splits[split], Mapping):
            raise ValueError("fresh-data sealed split must be an object")
        _exact_keys(
            splits[split],
            _SEALED_SPLIT_KEYS,
            f"fresh-data sealed {split} split",
        )
    if set(payload) != _SEAL_PAYLOAD_KEYS:
        raise ValueError("fresh-data seal payload keys differ")
    if (
        payload["dataset_id"] != DATASET_ID
        or payload["protocol_version"] != PROTOCOL_VERSION
        or payload["manifest_sha256"] != _FRESH_MANIFEST_SHA256
        or not _is_git_commit(payload["preparation_git_commit"])
        or privacy
        != {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        }
        or source["bytes"] != 1_862_302_013
        or source["sha256"] != _SOURCE_SHA256
        or not _is_positive_int(output["bytes"])
        or not _is_sha256(output["sha256"])
        or not all(_is_nonnegative_int(value) for value in scan.values())
        or exclusions["predecessor_exact_count"] != 6_911
        or exclusions["final_exact_count"] != 1_482
        or exclusions["exact_count"] != 8_393
        or exclusions["normalized_count"] != exclusions["exact_count"]
        or scan["source_lines"] != scan["parsed_records"] + scan["invalid_json"]
        or scan["parsed_records"]
        != scan["missing_text"]
        + scan["invalid_utf8"]
        + scan["empty_text"]
        + scan["too_short"]
        + scan["too_long"]
        + scan["exact_duplicates"]
        + scan["eligible_records"]
        or scan["eligible_records"]
        != scan["excluded_exact_records"]
        + scan["excluded_normalized_records"]
        + scan["normalized_source_duplicates"]
        + scan["stable_test_records_ignored"]
        + scan["candidate_train_records"]
        + scan["candidate_calibration_records"]
        or scan["excluded_exact_records"] != exclusions["exact_count"]
        or output["bytes"]
        <= sum(
            splits[split]["selected_document_raw_bytes"]
            for split in SPLIT_ORDER
        )
    ):
        raise ValueError("fresh-data seal payload differs")
    hashes = [
        output["sha256"],
        source["sha256"],
        exclusions["exact_commitment_sha256"],
        exclusions["normalized_commitment_sha256"],
    ]
    for split in SPLIT_ORDER:
        row = splits[split]
        hashes.extend(
            [
                row["selected_set_commitment_sha256"],
                row["normalized_selected_set_commitment_sha256"],
                row["stream_sha256"],
            ]
        )
        if (
            row["stream_bytes"] != SPLIT_QUOTAS[split]
            or row["sequence_count"] != SPLIT_QUOTAS[split] // SEQUENCE_LENGTH
            or row["available_stream_bytes"] < row["stream_bytes"]
            or not 0 <= row["overshoot_stream_bytes"] <= 262_144
            or row["available_stream_bytes"] - row["stream_bytes"]
            != row["overshoot_stream_bytes"]
            or not _is_positive_int(row["selected_document_count"])
            or not _is_positive_int(row["selected_document_raw_bytes"])
            or row["selected_document_raw_bytes"]
            + row["selected_document_count"]
            - 1
            != row["available_stream_bytes"]
            or row["selected_document_raw_bytes"]
            < row["selected_document_count"] * 256
            or row["selected_document_raw_bytes"]
            > row["selected_document_count"] * 262_144
            or scan[f"candidate_{split}_records"]
            < row["selected_document_count"]
        ):
            raise ValueError("fresh-data sealed split differs")
    if not all(_is_sha256(value) for value in hashes):
        raise ValueError("fresh-data seal hash differs")


def seal_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_seal_payload(payload)
    return {"payload": dict(payload), "payload_sha256": canonical_payload_sha256(payload)}


def validate_seal_envelope(envelope: Mapping[str, Any]) -> None:
    if set(envelope) != {"payload", "payload_sha256"} or not isinstance(
        envelope["payload"], Mapping
    ):
        raise ValueError("fresh-data seal envelope differs")
    validate_seal_payload(envelope["payload"])
    if envelope["payload_sha256"] != canonical_payload_sha256(envelope["payload"]):
        raise ValueError("fresh-data seal canonical hash differs")


def serialize_seal(envelope: Mapping[str, Any]) -> bytes:
    validate_seal_envelope(envelope)
    return (
        json.dumps(envelope, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def reconstruct(
    *,
    manifest_path: Path,
    archive_path: Path,
    final_manifest_path: Path,
    final_seal_path: Path,
    final_output_path: Path,
    predecessor_manifest_path: Path,
    predecessor_summary_path: Path,
    predecessor_integrity_path: Path,
    predecessor_output_path: Path,
    preparation_git_commit: str,
) -> tuple[bytes, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    if (
        archive_path.stat().st_size != manifest["source"]["expected_bytes"]
        or hash_file(archive_path) != manifest["source"]["expected_sha256"]
    ):
        raise ValueError("fresh-data source archive differs")
    exclusions = load_exclusions(
        manifest,
        final_manifest_path=final_manifest_path,
        final_seal_path=final_seal_path,
        final_output_path=final_output_path,
        predecessor_manifest_path=predecessor_manifest_path,
        predecessor_summary_path=predecessor_summary_path,
        predecessor_integrity_path=predecessor_integrity_path,
        predecessor_output_path=predecessor_output_path,
    )
    selected, statistics = scan_lines(
        iter_zstd_jsonl_lines(archive_path), manifest, exclusions
    )
    output = serialize_jsonl(selected)
    payload = build_seal_payload(
        manifest=manifest,
        manifest_sha256=hash_file(manifest_path),
        preparation_git_commit=preparation_git_commit,
        exclusions=exclusions,
        selected=selected,
        statistics=statistics,
        output=output,
    )
    return output, seal_envelope(payload)
