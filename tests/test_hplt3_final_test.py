from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import jamoflow.hplt3_final_test as module
from jamoflow.corpus import Record, split_for_record
from jamoflow.hplt3_final_test import (
    FinalTestBottomHashSampler,
    FinalTestCandidate,
    FinalTestScanStatistics,
    PredecessorIndex,
    build_final_test_seal_payload,
    canonical_payload_sha256,
    derive_rank_key,
    digest_set_commitment,
    final_test_rank_digest,
    load_final_test_manifest,
    ordered_selection_commitment,
    overlap_audit_commitment,
    normalized_record_digest,
    publish_no_clobber,
    seal_envelope,
    select_final_test_prefix_exhaustive,
    serialize_final_test_jsonl,
    serialize_seal_envelope,
    validate_final_test_manifest,
    validate_seal_envelope,
)


MANIFEST = Path("data/manifests/hplt3-korean-final-test-v1.json")
EXCLUSION_DOMAIN = b"JamoFlow/final-test-exclusion-set/v1\0"
SELECTED_DOMAIN = b"JamoFlow/final-test-selected-set/v1\0"


def _candidate(rank: int, digest: int, size: int = 100) -> FinalTestCandidate:
    return FinalTestCandidate(
        rank=rank.to_bytes(32, "big"),
        digest=digest.to_bytes(32, "big"),
        raw=bytes([65 + digest % 26]) * size,
    )


def _stable_test_documents(count: int, size: int = 300) -> list[bytes]:
    selected: list[bytes] = []
    for index in range(100_000):
        prefix = f"final-{index:06d}-".encode("ascii")
        raw = prefix + b"x" * (size - len(prefix))
        digest = sha256(raw).digest()
        record = Record(
            record_id=digest.hex(),
            source="synthetic",
            ordinal=index,
            raw=raw,
            text=raw.decode("ascii"),
        )
        if split_for_record(record) == "test":
            selected.append(raw)
            if len(selected) == count:
                return selected
    raise AssertionError("could not construct deterministic stable-test documents")


class FinalTestCommitmentTests(unittest.TestCase):
    def test_format_normalized_duplicate_digest_is_explicitly_bounded(self) -> None:
        self.assertEqual(
            normalized_record_digest("ＡＢＣ\t한글\n자료"),
            normalized_record_digest("abc 한글 자료"),
        )
        self.assertNotEqual(
            normalized_record_digest("abc 한글 자료"),
            normalized_record_digest("abc 한글 다른 자료"),
        )

    def test_rank_derivation_has_golden_vectors(self) -> None:
        key = derive_rank_key("00" * 32, "11" * 32, 32_000_000, 1)
        self.assertEqual(
            key.hex(),
            "927f7096db046f93ced35ac85149f2ef665063b40ad6dfe2937dbfea97050277",
        )
        self.assertEqual(
            final_test_rank_digest(key, bytes.fromhex("22" * 32)).hex(),
            "f892f9cd92c64f000bb137c577e4539d2c5e0b332cd91d81dc622ed198c8d2a1",
        )

    def test_commitments_have_domain_separated_golden_vectors(self) -> None:
        digests = [bytes.fromhex("33" * 32), bytes.fromhex("44" * 32)]
        exclusion = digest_set_commitment(digests, domain=EXCLUSION_DOMAIN)
        selected = digest_set_commitment(digests, domain=SELECTED_DOMAIN)
        self.assertEqual(
            exclusion,
            "c876733d15a90ab5cd400dd40e419750a5050dd9a2809db411036908f3e9cf73",
        )
        candidates = [
            FinalTestCandidate(
                rank=bytes.fromhex("10" * 32),
                digest=bytes.fromhex("20" * 32),
                raw=b"abc",
            ),
            FinalTestCandidate(
                rank=bytes.fromhex("11" * 32),
                digest=bytes.fromhex("21" * 32),
                raw=b"defgh",
            ),
        ]
        self.assertEqual(
            ordered_selection_commitment(candidates),
            "1d6516cd6e7179c6f7699b21848acb44bee11f0e2270b1755891e1df96525d6d",
        )
        overlap = overlap_audit_commitment(exclusion, selected, [])
        self.assertEqual(
            overlap,
            {
                "intersection_count": 0,
                "intersection_commitment_sha256": (
                    "ffbdd555a26004d989293ae2960e8052233aba2bc357f7fd8118f5f60934141e"
                ),
                "overlap_audit_sha256": (
                    "52d2d055e6d551aea932cf56a6e097fab77ef825397ea8296f84585ea8f216f6"
                ),
            },
        )


class FinalTestSamplerTests(unittest.TestCase):
    def test_bounded_sampler_matches_exhaustive_order_and_input_reversal(self) -> None:
        candidates = [_candidate(rank, rank) for rank in range(10, 0, -1)]

        def select(values: list[FinalTestCandidate]) -> list[tuple[bytes, bytes]]:
            sampler = FinalTestBottomHashSampler(
                quota_stream_bytes=250,
                reserve_stream_bytes=450,
            )
            for value in values:
                sampler.add(value)
            return [(value.rank, value.digest) for value in sampler.finalize()]

        expected = [
            (value.rank, value.digest)
            for value in sorted(candidates, key=lambda item: (item.rank, item.digest))[:3]
        ]
        self.assertEqual(select(candidates), expected)
        self.assertEqual(select(list(reversed(candidates))), expected)

    def test_equal_rank_uses_document_digest_as_ascending_tiebreak(self) -> None:
        values = [_candidate(7, digest) for digest in range(6, 0, -1)]
        sampler = FinalTestBottomHashSampler(150, 303)
        for value in values:
            sampler.add(value)
        self.assertEqual(
            [int.from_bytes(value.digest, "big") for value in sampler.finalize()],
            [1, 2],
        )

    def test_variable_size_reservoir_matches_full_sort_prefix(self) -> None:
        values = [
            _candidate(rank, rank, size=50 + (rank * 37) % 151)
            for rank in range(1, 31)
        ]
        sampler = FinalTestBottomHashSampler(600, 1_500)
        for index in tuple(range(0, 30, 2)) + tuple(range(29, 0, -2)):
            sampler.add(values[index])
        expected: list[FinalTestCandidate] = []
        available = 0
        for value in sorted(values, key=lambda item: (item.rank, item.digest)):
            available += len(value.raw) + (1 if expected else 0)
            expected.append(value)
            if available >= 600:
                break
        self.assertEqual(
            [(value.rank, value.digest) for value in sampler.finalize()],
            [(value.rank, value.digest) for value in expected],
        )
        self.assertEqual(
            [
                (value.rank, value.digest)
                for value in select_final_test_prefix_exhaustive(values, 600)
            ],
            [(value.rank, value.digest) for value in expected],
        )

    def test_exhaustive_selector_is_input_order_invariant_and_minimal(self) -> None:
        values = [
            _candidate(rank, rank, size=80 + (rank * 19) % 83)
            for rank in range(25, 0, -1)
        ]
        first = select_final_test_prefix_exhaustive(values, 700)
        second = select_final_test_prefix_exhaustive(list(reversed(values)), 700)
        self.assertEqual(first, second)
        joined = b"\n".join(value.raw for value in first)
        previous = b"\n".join(value.raw for value in first[:-1])
        self.assertGreaterEqual(len(joined), 700)
        self.assertLess(len(previous), 700)

    def test_insufficient_candidates_never_reduce_the_quota(self) -> None:
        sampler = FinalTestBottomHashSampler(1_000, 1_100)
        sampler.add(_candidate(1, 1))
        with self.assertRaisesRegex(ValueError, "do not satisfy"):
            sampler.finalize()

    def test_sampler_rejects_duplicate_rank_digest_identity(self) -> None:
        sampler = FinalTestBottomHashSampler(100, 200)
        value = _candidate(1, 1)
        sampler.add(value)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            sampler.add(value)


class FinalTestSealTests(unittest.TestCase):
    def test_manifest_rejects_unknown_keys_and_rank_key_shopping(self) -> None:
        manifest = load_final_test_manifest(MANIFEST)
        validate_final_test_manifest(manifest)
        unknown = deepcopy(manifest)
        unknown["selection"]["model_loss"] = 1.0
        with self.assertRaisesRegex(ValueError, "sealed schema"):
            validate_final_test_manifest(unknown)
        changed = deepcopy(manifest)
        changed["selection"]["expected_rank_key_hex"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "uniquely derived"):
            validate_final_test_manifest(changed)
        wrong_type = deepcopy(manifest)
        wrong_type["selection"]["quota_stream_bytes"] = True
        with self.assertRaisesRegex(ValueError, "positive ints"):
            validate_final_test_manifest(wrong_type)

    def test_small_patched_protocol_rechecks_every_selected_document(self) -> None:
        with mock.patch.object(module, "FINAL_TEST_STREAM_BYTES", 1_024), mock.patch.object(
            module,
            "FINAL_TEST_SEQUENCE_COUNT",
            2,
        ):
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            manifest["selection"]["quota_stream_bytes"] = 1_024
            manifest["selection"]["reserve_stream_bytes"] = 264_193
            rank_key = derive_rank_key(
                manifest["source"]["expected_sha256"],
                manifest["predecessor"]["processed_output_sha256"],
                1_024,
                1,
            )
            manifest["selection"]["expected_rank_key_hex"] = rank_key.hex()
            validate_final_test_manifest(manifest)

            predecessor_digest = sha256(b"historical predecessor").digest()
            predecessor = PredecessorIndex(
                digests_by_split={
                    "train": frozenset({predecessor_digest}),
                    "calibration": frozenset(),
                    "test": frozenset(),
                },
                all_digests=frozenset({predecessor_digest}),
                exclusion_commitment_sha256=digest_set_commitment(
                    [predecessor_digest],
                    domain=EXCLUSION_DOMAIN,
                ),
                normalized_digests=frozenset(
                    {normalized_record_digest("historical predecessor")}
                ),
                normalized_exclusion_commitment_sha256=digest_set_commitment(
                    [normalized_record_digest("historical predecessor")],
                    domain=(
                        b"JamoFlow/final-test-normalized-exclusion-set/v1\0"
                    ),
                ),
                source_scan={},
            )
            candidates = []
            for raw in _stable_test_documents(4):
                digest = sha256(raw).digest()
                candidates.append(
                    FinalTestCandidate(
                        rank=final_test_rank_digest(rank_key, digest),
                        digest=digest,
                        raw=raw,
                    )
                )
            candidates.sort(key=lambda item: (item.rank, item.digest))
            output = serialize_final_test_jsonl(candidates)
            stats = FinalTestScanStatistics(
                source_lines=5,
                parsed_records=5,
                eligible_records=5,
                eligible_text_bytes=sum(len(item.raw) for item in candidates) + 300,
                predecessor_records_found=1,
                post_exclusion_test_records=4,
                post_exclusion_test_text_bytes=sum(
                    len(item.raw) for item in candidates
                ),
            )
            payload = build_final_test_seal_payload(
                manifest=manifest,
                manifest_sha256="1" * 64,
                preparation_git_commit="2" * 40,
                source_bytes=manifest["source"]["expected_bytes"],
                source_sha256=manifest["source"]["expected_sha256"],
                predecessor=predecessor,
                statistics=stats,
                candidates=candidates,
                output_bytes=output,
            )
            self.assertEqual(payload["output"]["evaluation_stream_bytes"], 1_024)
            self.assertEqual(payload["output"]["sequence_count"], 2)
            self.assertEqual(payload["selection"]["intersection_count"], 0)

            envelope = seal_envelope(payload)
            serialized = serialize_seal_envelope(envelope)
            self.assertEqual(json.loads(serialized), envelope)
            validate_seal_envelope(envelope)

            forged = deepcopy(envelope)
            forged["payload"]["model_loss"] = 1.0
            forged["payload_sha256"] = canonical_payload_sha256(forged["payload"])
            with self.assertRaisesRegex(ValueError, "sealed schema"):
                validate_seal_envelope(forged)

            wrong_rank = list(candidates)
            wrong_rank[0] = FinalTestCandidate(
                rank=b"\x00" * 32,
                digest=wrong_rank[0].digest,
                raw=wrong_rank[0].raw,
            )
            wrong_rank.sort(key=lambda item: (item.rank, item.digest))
            with self.assertRaisesRegex(ValueError, "rank is not protocol-derived"):
                build_final_test_seal_payload(
                    manifest=manifest,
                    manifest_sha256="1" * 64,
                    preparation_git_commit="2" * 40,
                    source_bytes=manifest["source"]["expected_bytes"],
                    source_sha256=manifest["source"]["expected_sha256"],
                    predecessor=predecessor,
                    statistics=stats,
                    candidates=wrong_rank,
                    output_bytes=serialize_final_test_jsonl(wrong_rank),
                )

    def test_no_clobber_publish_is_idempotent_but_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "sealed.json"
            publish_no_clobber(target, b"first")
            publish_no_clobber(target, b"first")
            self.assertEqual(target.read_bytes(), b"first")
            with self.assertRaisesRegex(ValueError, "differs"):
                publish_no_clobber(target, b"second")
            staged_target = Path(temporary) / "staged.json"
            staged_target.with_suffix(".json.preparing").write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "staged sealed artifact differs"):
                publish_no_clobber(staged_target, b"complete")

    def test_preparation_code_cannot_import_model_or_evaluation_stacks(self) -> None:
        forbidden_roots = {
            "numpy",
            "torch",
            "tokenizers",
            "transformers",
        }
        paths = [
            Path("src/jamoflow/hplt3_final_test.py"),
            Path("scripts/prepare_hplt3_final_test.py"),
            Path("scripts/verify_hplt3_final_test.py"),
        ]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            forbidden_paths: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value
                    if value.startswith(("runs/", "artifacts/")) or (
                        value.startswith("results/")
                        and value != "results/phase3-data/summary.json"
                    ):
                        forbidden_paths.append(value)
            self.assertFalse(
                imported & forbidden_roots,
                f"{path} imports model/evaluation code: {imported & forbidden_roots}",
            )
            self.assertFalse(
                forbidden_paths,
                f"{path} can access model/result paths: {forbidden_paths}",
            )


if __name__ == "__main__":
    unittest.main()
