"""Deterministic second fresh Korean train/calibration data protocol.

The v2 streams exclude every document used by the historical Phase-3 corpus,
the sealed final test, and the first fresh vocabulary-adaptation corpus.  The
selection is model-free and its rank keys are uniquely derived from the pinned
data identities; no checkpoint, loss, BPB, latency, or model output is read.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    normalized_record_digest,
)
import hplt3_fresh_adaptation_protocol as v1
from hplt3_fresh_adaptation_protocol import (
    FreshScanStatistics,
    serialize_jsonl,
    stream_bytes,
)


PROTOCOL_VERSION = 2
DATASET_ID = "hplt3-korean-vocab-adaptation-v2"
SEQUENCE_LENGTH = 512
SPLIT_ORDER = ("train", "calibration")
SPLIT_QUOTAS = {"train": 128_000_000, "calibration": 8_000_000}

SOURCE_SHA256 = "de4dfa43fd9f6c62cc81781e09c1f401cc77e7a956e07ecc80ac13477e699ca4"
PREDECESSOR_MANIFEST_SHA256 = (
    "873a303e9762906e0046cdf59f850b8d6dbeb4483cfff8ad5a6f124e00002d9e"
)
PREDECESSOR_SUMMARY_SHA256 = (
    "5d2890a0154c0e98fe27c4d8801d5ffacea9217e25440d2631f7354e2c37f9ba"
)
PREDECESSOR_INTEGRITY_SHA256 = (
    "472cc5da045909109718be71168e516be19043cb2a08363d573ed77650038181"
)
PREDECESSOR_OUTPUT_SHA256 = (
    "f789bc7e0ec0252c4c7c636e67a7c44f6d2c528a292ec47542af98488c8b36a5"
)
FINAL_MANIFEST_SHA256 = (
    "ea36dbbac63be8b9370cfee029758030cdd138aa1510bc02aeb8cbe4b95590ff"
)
FINAL_SEAL_SHA256 = (
    "ce42e8a0b2d8161cc59e0b30d5d121b547e22d28709fe48284aa777df4a2290b"
)
FINAL_PAYLOAD_SHA256 = (
    "97cf90d1e6e7191e7f8336647f278ae6c0e82d70540bf0f5c43f9cb426e75dc8"
)
FINAL_OUTPUT_SHA256 = (
    "098ae8b833a1498689dae1d60341aa870fce51c7f9dde6d961c867f751ee3dc2"
)
FRESH_V1_MANIFEST_SHA256 = (
    "7325c2562e4e02c3b0f71fde5ac3ab7f3bb383ce17bb0756728c6cb59506859f"
)
FRESH_V1_PROTOCOL_SHA256 = (
    "356772ae7197c256f748da13780142529416c21e4ab60cb17f11ce605877723e"
)
FRESH_V1_SEAL_SHA256 = (
    "2a1457b0b1cd1ffcaabde7997056b0283d5adebe925a7b228a046d0cdbe6f916"
)
FRESH_V1_PAYLOAD_SHA256 = (
    "351a1ae05e35198e53f7576eace5e3426fe88150392e1f8f661e388e8599657a"
)
FRESH_V1_OUTPUT_SHA256 = (
    "7817d3be0d67099735e6a26c741314ba27fe12b9cb7dd7d3b7022af40ea3b2c5"
)

PREDECESSOR_EXACT_COUNT = 6_911
FINAL_EXACT_COUNT = 1_482
FRESH_V1_EXACT_COUNT = 6_076
HISTORICAL_EXACT_COUNT = PREDECESSOR_EXACT_COUNT + FINAL_EXACT_COUNT
TOTAL_EXACT_COUNT = HISTORICAL_EXACT_COUNT + FRESH_V1_EXACT_COUNT

_KEY_DOMAIN = b"JamoFlow/fresh-adaptation-key/v2\0"
_RANK_DOMAIN = b"JamoFlow/fresh-adaptation-rank/v2\0"
_SELECTED_DOMAINS = {
    "train": b"JamoFlow/fresh-adaptation-selected-train/v2\0",
    "calibration": b"JamoFlow/fresh-adaptation-selected-calibration/v2\0",
}
_NORMALIZED_SELECTED_DOMAINS = {
    "train": b"JamoFlow/fresh-adaptation-normalized-train/v2\0",
    "calibration": b"JamoFlow/fresh-adaptation-normalized-calibration/v2\0",
}

_SOURCE = {
    "etag": '"6f00793d-63be01a95c540"',
    "expected_bytes": 1_862_302_013,
    "expected_sha256": SOURCE_SHA256,
    "filename": "10_1.jsonl.zst",
    "last_modified": "Fri, 08 Aug 2025 20:06:05 GMT",
    "url": "https://data.hplt-project.org/three/sorted/kor_Hang/10_1.jsonl.zst",
}
_PREDECESSOR = {
    "integrity_path": "data/processed/hplt3-korean-phase3/integrity.json",
    "integrity_sha256": PREDECESSOR_INTEGRITY_SHA256,
    "manifest_path": "data/manifests/hplt3-korean-phase3.json",
    "manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
    "output_bytes": 152_461_842,
    "output_path": "data/processed/hplt3-korean-phase3/ko.jsonl",
    "output_sha256": PREDECESSOR_OUTPUT_SHA256,
    "summary_path": "results/phase3-data/summary.json",
    "summary_sha256": PREDECESSOR_SUMMARY_SHA256,
    "unique_records": PREDECESSOR_EXACT_COUNT,
}
_FINAL_TEST = {
    "manifest_path": "data/manifests/hplt3-korean-final-test-v1.json",
    "manifest_sha256": FINAL_MANIFEST_SHA256,
    "output_bytes": 32_103_596,
    "output_path": "data/processed/hplt3-korean-final-test-v1/ko.jsonl",
    "output_sha256": FINAL_OUTPUT_SHA256,
    "payload_sha256": FINAL_PAYLOAD_SHA256,
    "seal_path": "data/seals/hplt3-korean-final-test-v1.json",
    "seal_sha256": FINAL_SEAL_SHA256,
    "selected_document_count": FINAL_EXACT_COUNT,
}
_FRESH_V1 = {
    "calibration_document_count": 384,
    "dataset_id": "hplt3-korean-vocab-adaptation-v1",
    "manifest_path": "data/manifests/hplt3-korean-vocab-adaptation-v1.json",
    "manifest_sha256": FRESH_V1_MANIFEST_SHA256,
    "normalized_document_count": FRESH_V1_EXACT_COUNT,
    "output_bytes": 136_470_910,
    "output_path": "data/processed/hplt3-korean-vocab-adaptation-v1/ko.jsonl",
    "output_sha256": FRESH_V1_OUTPUT_SHA256,
    "payload_sha256": FRESH_V1_PAYLOAD_SHA256,
    "protocol_path": "scripts/hplt3_fresh_adaptation_protocol.py",
    "protocol_sha256": FRESH_V1_PROTOCOL_SHA256,
    "seal_path": "data/seals/hplt3-korean-vocab-adaptation-v1.json",
    "seal_sha256": FRESH_V1_SEAL_SHA256,
    "selected_document_count": FRESH_V1_EXACT_COUNT,
    "train_document_count": 5_692,
}
_PRIVACY = {
    "tracked_individual_digests": False,
    "tracked_model_metrics": False,
    "tracked_text": False,
}

_TOP_LEVEL_KEYS = {
    "dataset_id",
    "final_test",
    "fresh_v1",
    "predecessor",
    "privacy",
    "protocol_version",
    "purpose",
    "schema_version",
    "selection",
    "source",
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
_SEAL_PAYLOAD_KEYS = {
    "dataset_id",
    "exclusions",
    "fresh_v1_dependency",
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
    "fresh_v1_exact_count",
    "historical_exact_count",
    "normalized_commitment_sha256",
    "normalized_count",
    "predecessor_exact_count",
}
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
        raise ValueError("fresh-v2 commitment integer differs")
    return struct.pack(">Q", value)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_nonnegative_int(value) and value > 0


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} keys differ")


def derive_split_rank_key(
    source_sha256: str,
    predecessor_output_sha256: str,
    final_output_sha256: str,
    fresh_v1_output_sha256: str,
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
                fresh_v1_output_sha256,
            )
        )
        or split not in SPLIT_ORDER
        or quota != SPLIT_QUOTAS[split]
        or protocol_version != PROTOCOL_VERSION
    ):
        raise ValueError("fresh-v2 rank-key inputs differ")
    return sha256(
        _KEY_DOMAIN
        + bytes.fromhex(source_sha256)
        + bytes.fromhex(predecessor_output_sha256)
        + bytes.fromhex(final_output_sha256)
        + bytes.fromhex(fresh_v1_output_sha256)
        + split.encode("ascii")
        + b"\0"
        + _u64(quota)
        + _u64(protocol_version)
    ).digest()


def rank_digest(rank_key: bytes, document_digest: bytes) -> bytes:
    if len(rank_key) != 32 or len(document_digest) != 32:
        raise ValueError("fresh-v2 rank inputs differ")
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
        raise ValueError("fresh-v2 manifest must be an object")
    _exact_keys(manifest, _TOP_LEVEL_KEYS, "fresh-v2 manifest")
    for key, expected in (
        ("source", _SOURCE),
        ("predecessor", _PREDECESSOR),
        ("final_test", _FINAL_TEST),
        ("fresh_v1", _FRESH_V1),
        ("privacy", _PRIVACY),
    ):
        value = manifest[key]
        if not isinstance(value, Mapping) or dict(value) != expected:
            raise ValueError(f"fresh-v2 {key} contract differs")
    selection = manifest["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("fresh-v2 selection must be an object")
    _exact_keys(selection, _SELECTION_KEYS, "fresh-v2 selection")
    rank_keys = selection["expected_rank_key_hex"]
    quotas = selection["quotas"]
    reserves = selection["reserve_stream_bytes"]
    if not all(isinstance(value, Mapping) for value in (rank_keys, quotas, reserves)):
        raise ValueError("fresh-v2 split maps differ")
    for value in (rank_keys, quotas, reserves):
        _exact_keys(value, set(SPLIT_ORDER), "fresh-v2 split map")
    if (
        manifest["schema_version"] != 1
        or manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["dataset_id"] != DATASET_ID
        or not isinstance(manifest["purpose"], str)
        or not manifest["purpose"]
        or selection["minimum_document_bytes"] != 256
        or selection["maximum_document_bytes"] != 262_144
        or selection["sequence_length"] != SEQUENCE_LENGTH
        or selection["record_digest"] != "sha256-exact-utf8-v1"
        or selection["rank_algorithm"]
        != "sha256-domain-separated-derived-key-v2"
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
    ):
        raise ValueError("fresh-v2 manifest violates the protocol")
    for split in SPLIT_ORDER:
        expected = derive_split_rank_key(
            manifest["source"]["expected_sha256"],
            manifest["predecessor"]["output_sha256"],
            manifest["final_test"]["output_sha256"],
            manifest["fresh_v1"]["output_sha256"],
            split,
            quotas[split],
        ).hex()
        if rank_keys[split] != expected:
            raise ValueError("fresh-v2 rank key is not uniquely derived")


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
    fresh_v1_exact_count: int


def _read_fresh_v1_documents(
    *,
    manifest: Mapping[str, Any],
    fresh_v1_manifest_path: Path,
    fresh_v1_protocol_path: Path,
    fresh_v1_seal_path: Path,
    fresh_v1_output_path: Path,
) -> tuple[frozenset[bytes], frozenset[bytes]]:
    contract = manifest["fresh_v1"]
    if (
        hash_file(fresh_v1_manifest_path) != contract["manifest_sha256"]
        or hash_file(fresh_v1_protocol_path) != contract["protocol_sha256"]
        or hash_file(fresh_v1_seal_path) != contract["seal_sha256"]
        or hash_file(fresh_v1_output_path) != contract["output_sha256"]
        or fresh_v1_output_path.stat().st_size != contract["output_bytes"]
    ):
        raise ValueError("fresh-v2 v1 dependency differs")
    v1_manifest = v1.load_manifest(fresh_v1_manifest_path)
    if (
        v1_manifest["source"] != manifest["source"]
        or v1_manifest["predecessor"] != manifest["predecessor"]
        or v1_manifest["final_test"] != manifest["final_test"]
    ):
        raise ValueError("fresh-v2 v1 dependency lineage differs")
    envelope = json.loads(fresh_v1_seal_path.read_text(encoding="utf-8"))
    v1.validate_seal_envelope(envelope)
    payload = envelope["payload"]
    if (
        envelope["payload_sha256"] != contract["payload_sha256"]
        or payload["dataset_id"] != contract["dataset_id"]
        or payload["manifest_sha256"] != contract["manifest_sha256"]
        or payload["output"]
        != {
            "bytes": contract["output_bytes"],
            "sha256": contract["output_sha256"],
        }
        or payload["splits"]["train"]["selected_document_count"]
        != contract["train_document_count"]
        or payload["splits"]["calibration"]["selected_document_count"]
        != contract["calibration_document_count"]
        or payload["exclusions"]["exact_count"] != HISTORICAL_EXACT_COUNT
    ):
        raise ValueError("fresh-v2 v1 seal lineage differs")
    exact: set[bytes] = set()
    normalized: set[bytes] = set()
    split_counts = {split: 0 for split in SPLIT_ORDER}
    split_raws = {split: [] for split in SPLIT_ORDER}
    observed_order: list[str] = []
    with fresh_v1_output_path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("fresh-v2 v1 JSONL differs") from exc
            if (
                not isinstance(value, dict)
                or set(value) != {"language", "text"}
                or value.get("language") != "ko"
                or not isinstance(value.get("text"), str)
            ):
                raise ValueError("fresh-v2 v1 JSONL schema differs")
            text = value["text"]
            raw = text.encode("utf-8", errors="strict")
            digest = sha256(raw).digest()
            normalized_digest = normalized_record_digest(text)
            split = _record_split(digest, raw, text)
            if (
                split not in SPLIT_ORDER
                or digest in exact
                or normalized_digest in normalized
            ):
                raise ValueError("fresh-v2 v1 document identity differs")
            exact.add(digest)
            normalized.add(normalized_digest)
            split_counts[split] += 1
            split_raws[split].append(raw)
            observed_order.append(split)
    expected_order = ["train"] * contract["train_document_count"] + [
        "calibration"
    ] * contract["calibration_document_count"]
    if (
        len(exact) != contract["selected_document_count"]
        or len(normalized) != contract["normalized_document_count"]
        or split_counts
        != {
            "train": contract["train_document_count"],
            "calibration": contract["calibration_document_count"],
        }
        or observed_order != expected_order
        or any(
            sha256(b"\n".join(split_raws[split])[: SPLIT_QUOTAS[split]]).hexdigest()
            != payload["splits"][split]["stream_sha256"]
            for split in SPLIT_ORDER
        )
    ):
        raise ValueError("fresh-v2 v1 document set differs")
    return frozenset(exact), frozenset(normalized)


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
    fresh_v1_manifest_path: Path,
    fresh_v1_protocol_path: Path,
    fresh_v1_seal_path: Path,
    fresh_v1_output_path: Path,
) -> ExclusionIndex:
    validate_manifest(manifest)
    v1_manifest = v1.load_manifest(fresh_v1_manifest_path)
    base = v1.load_exclusions(
        v1_manifest,
        final_manifest_path=final_manifest_path,
        final_seal_path=final_seal_path,
        final_output_path=final_output_path,
        predecessor_manifest_path=predecessor_manifest_path,
        predecessor_summary_path=predecessor_summary_path,
        predecessor_integrity_path=predecessor_integrity_path,
        predecessor_output_path=predecessor_output_path,
    )
    fresh_exact, fresh_normalized = _read_fresh_v1_documents(
        manifest=manifest,
        fresh_v1_manifest_path=fresh_v1_manifest_path,
        fresh_v1_protocol_path=fresh_v1_protocol_path,
        fresh_v1_seal_path=fresh_v1_seal_path,
        fresh_v1_output_path=fresh_v1_output_path,
    )
    if (
        len(base.exact) != HISTORICAL_EXACT_COUNT
        or len(base.normalized) != HISTORICAL_EXACT_COUNT
        or len(fresh_exact) != FRESH_V1_EXACT_COUNT
        or len(fresh_normalized) != FRESH_V1_EXACT_COUNT
        or base.exact & fresh_exact
        or base.normalized & fresh_normalized
    ):
        raise ValueError("fresh-v2 exclusion sets overlap or differ")
    return ExclusionIndex(
        exact=frozenset(set(base.exact) | set(fresh_exact)),
        normalized=frozenset(set(base.normalized) | set(fresh_normalized)),
        predecessor_exact_count=base.predecessor_exact_count,
        final_exact_count=base.final_exact_count,
        fresh_v1_exact_count=len(fresh_exact),
    )


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
                raise ValueError("SHA-256 collision in fresh-v2 source")
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
        raise ValueError("raw shard does not contain every sealed v2 exclusion")
    return (
        {split: samplers[split].finalize() for split in SPLIT_ORDER},
        statistics,
    )


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
    if not _is_sha256(manifest_sha256) or not _is_git_commit(preparation_git_commit):
        raise ValueError("fresh-v2 seal lineage differs")
    if output != serialize_jsonl(selected):
        raise ValueError("fresh-v2 serialized output differs")
    split_rows: dict[str, Any] = {}
    all_selected: set[bytes] = set()
    all_normalized: set[bytes] = set()
    for split in SPLIT_ORDER:
        candidates = tuple(selected[split])
        keys = [(candidate.rank, candidate.digest) for candidate in candidates]
        if not candidates or keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("fresh-v2 ordered split differs")
        raw_stream = stream_bytes(candidates)
        quota = manifest["selection"]["quotas"][split]
        previous = len(raw_stream) - len(candidates[-1].raw)
        if len(candidates) > 1:
            previous -= 1
        if previous >= quota or len(raw_stream) < quota:
            raise ValueError("fresh-v2 split is not its minimal quota prefix")
        digests = {candidate.digest for candidate in candidates}
        normalized = {
            normalized_record_digest(candidate.raw.decode("utf-8", errors="strict"))
            for candidate in candidates
        }
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
            raise ValueError("fresh-v2 selected overlap or split differs")
        all_selected.update(digests)
        all_normalized.update(normalized)
        split_rows[split] = {
            "available_stream_bytes": len(raw_stream),
            "normalized_selected_set_commitment_sha256": digest_set_commitment(
                normalized, domain=_NORMALIZED_SELECTED_DOMAINS[split]
            ),
            "overshoot_stream_bytes": len(raw_stream) - quota,
            "selected_document_count": len(candidates),
            "selected_document_raw_bytes": sum(
                len(candidate.raw) for candidate in candidates
            ),
            "selected_set_commitment_sha256": digest_set_commitment(
                digests, domain=_SELECTED_DOMAINS[split]
            ),
            "sequence_count": quota // SEQUENCE_LENGTH,
            "stream_bytes": quota,
            "stream_sha256": sha256(raw_stream[:quota]).hexdigest(),
        }
    payload = {
        "dataset_id": DATASET_ID,
        "exclusions": {
            "exact_commitment_sha256": digest_set_commitment(
                exclusions.exact,
                domain=b"JamoFlow/fresh-adaptation-exclusions/v2\0",
            ),
            "exact_count": len(exclusions.exact),
            "final_exact_count": exclusions.final_exact_count,
            "fresh_v1_exact_count": exclusions.fresh_v1_exact_count,
            "historical_exact_count": (
                exclusions.predecessor_exact_count + exclusions.final_exact_count
            ),
            "normalized_commitment_sha256": digest_set_commitment(
                exclusions.normalized,
                domain=b"JamoFlow/fresh-adaptation-normalized-exclusions/v2\0",
            ),
            "normalized_count": len(exclusions.normalized),
            "predecessor_exact_count": exclusions.predecessor_exact_count,
        },
        "fresh_v1_dependency": {
            "output_sha256": manifest["fresh_v1"]["output_sha256"],
            "payload_sha256": manifest["fresh_v1"]["payload_sha256"],
            "selected_document_count": manifest["fresh_v1"][
                "selected_document_count"
            ],
        },
        "manifest_sha256": manifest_sha256,
        "output": {"bytes": len(output), "sha256": sha256(output).hexdigest()},
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
        raise ValueError("fresh-v2 seal payload must be an object")
    _exact_keys(payload, _SEAL_PAYLOAD_KEYS, "fresh-v2 seal payload")
    exclusions = payload["exclusions"]
    fresh_v1_dependency = payload["fresh_v1_dependency"]
    output = payload["output"]
    privacy = payload["privacy"]
    scan = payload["scan"]
    source = payload["source"]
    splits = payload["splits"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            exclusions,
            fresh_v1_dependency,
            output,
            privacy,
            scan,
            source,
            splits,
        )
    ):
        raise ValueError("fresh-v2 seal sections differ")
    _exact_keys(exclusions, _SEAL_EXCLUSION_KEYS, "fresh-v2 exclusions")
    _exact_keys(
        fresh_v1_dependency,
        {"output_sha256", "payload_sha256", "selected_document_count"},
        "fresh-v2 v1 dependency",
    )
    _exact_keys(output, {"bytes", "sha256"}, "fresh-v2 output")
    _exact_keys(
        privacy,
        {
            "individual_document_digests_tracked",
            "model_metrics_tracked",
            "raw_text_tracked",
        },
        "fresh-v2 privacy",
    )
    _exact_keys(scan, set(FreshScanStatistics.__dataclass_fields__), "fresh-v2 scan")
    _exact_keys(source, {"bytes", "sha256"}, "fresh-v2 source")
    _exact_keys(splits, set(SPLIT_ORDER), "fresh-v2 splits")
    for split in SPLIT_ORDER:
        if not isinstance(splits[split], Mapping):
            raise ValueError("fresh-v2 split must be an object")
        _exact_keys(splits[split], _SEALED_SPLIT_KEYS, f"fresh-v2 {split}")
    if (
        payload["dataset_id"] != DATASET_ID
        or payload["protocol_version"] != PROTOCOL_VERSION
        or not _is_sha256(payload["manifest_sha256"])
        or not _is_git_commit(payload["preparation_git_commit"])
        or privacy
        != {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        }
        or source != {"bytes": 1_862_302_013, "sha256": SOURCE_SHA256}
        or fresh_v1_dependency
        != {
            "output_sha256": FRESH_V1_OUTPUT_SHA256,
            "payload_sha256": FRESH_V1_PAYLOAD_SHA256,
            "selected_document_count": FRESH_V1_EXACT_COUNT,
        }
        or exclusions["predecessor_exact_count"] != PREDECESSOR_EXACT_COUNT
        or exclusions["final_exact_count"] != FINAL_EXACT_COUNT
        or exclusions["historical_exact_count"] != HISTORICAL_EXACT_COUNT
        or exclusions["fresh_v1_exact_count"] != FRESH_V1_EXACT_COUNT
        or exclusions["exact_count"] != TOTAL_EXACT_COUNT
        or exclusions["normalized_count"] != TOTAL_EXACT_COUNT
        or not _is_positive_int(output["bytes"])
        or not _is_sha256(output["sha256"])
        or not all(_is_nonnegative_int(value) for value in scan.values())
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
        or scan["excluded_exact_records"] != TOTAL_EXACT_COUNT
        or output["bytes"]
        <= sum(
            splits[split]["selected_document_raw_bytes"] for split in SPLIT_ORDER
        )
    ):
        raise ValueError("fresh-v2 seal payload differs")
    hashes = [
        exclusions["exact_commitment_sha256"],
        exclusions["normalized_commitment_sha256"],
        output["sha256"],
        source["sha256"],
    ]
    for split in SPLIT_ORDER:
        row = splits[split]
        hashes.extend(
            [
                row["normalized_selected_set_commitment_sha256"],
                row["selected_set_commitment_sha256"],
                row["stream_sha256"],
            ]
        )
        if (
            row["stream_bytes"] != SPLIT_QUOTAS[split]
            or row["sequence_count"] != SPLIT_QUOTAS[split] // SEQUENCE_LENGTH
            or row["available_stream_bytes"] < row["stream_bytes"]
            or row["available_stream_bytes"] - row["stream_bytes"]
            != row["overshoot_stream_bytes"]
            or not 0 <= row["overshoot_stream_bytes"] <= 262_144
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
            or scan[f"candidate_{split}_records"] < row["selected_document_count"]
        ):
            raise ValueError("fresh-v2 sealed split differs")
    if not all(_is_sha256(value) for value in hashes):
        raise ValueError("fresh-v2 seal hash differs")


def seal_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_seal_payload(payload)
    return {"payload": dict(payload), "payload_sha256": canonical_payload_sha256(payload)}


def validate_seal_envelope(envelope: Mapping[str, Any]) -> None:
    if set(envelope) != {"payload", "payload_sha256"} or not isinstance(
        envelope["payload"], Mapping
    ):
        raise ValueError("fresh-v2 seal envelope differs")
    validate_seal_payload(envelope["payload"])
    if envelope["payload_sha256"] != canonical_payload_sha256(envelope["payload"]):
        raise ValueError("fresh-v2 seal canonical hash differs")


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
    fresh_v1_manifest_path: Path,
    fresh_v1_protocol_path: Path,
    fresh_v1_seal_path: Path,
    fresh_v1_output_path: Path,
    preparation_git_commit: str,
) -> tuple[bytes, dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    if (
        archive_path.stat().st_size != manifest["source"]["expected_bytes"]
        or hash_file(archive_path) != manifest["source"]["expected_sha256"]
    ):
        raise ValueError("fresh-v2 source archive differs")
    exclusions = load_exclusions(
        manifest,
        final_manifest_path=final_manifest_path,
        final_seal_path=final_seal_path,
        final_output_path=final_output_path,
        predecessor_manifest_path=predecessor_manifest_path,
        predecessor_summary_path=predecessor_summary_path,
        predecessor_integrity_path=predecessor_integrity_path,
        predecessor_output_path=predecessor_output_path,
        fresh_v1_manifest_path=fresh_v1_manifest_path,
        fresh_v1_protocol_path=fresh_v1_protocol_path,
        fresh_v1_seal_path=fresh_v1_seal_path,
        fresh_v1_output_path=fresh_v1_output_path,
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
