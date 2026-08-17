from __future__ import annotations

import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest import mock

import hplt3_fresh_adaptation_protocol as module
from jamoflow.corpus import Record, split_for_record
from jamoflow.hplt3_final_test import FinalTestCandidate
from hplt3_fresh_adaptation_protocol import (
    ExclusionIndex,
    SPLIT_QUOTAS,
    derive_split_rank_key,
    normalized_record_digest,
    rank_digest,
    scan_lines,
    serialize_jsonl,
    validate_manifest,
    validate_seal_payload,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/hplt3-korean-vocab-adaptation-v1.json"


def _split(text: str) -> str:
    raw = text.encode("utf-8")
    digest = sha256(raw).digest()
    return split_for_record(
        Record(
            record_id=digest.hex(),
            source="synthetic",
            ordinal=0,
            raw=raw,
            text=text,
        )
    )


def _text_for_split(label: str, target: str) -> str:
    for index in range(10_000):
        text = f"{label}-{index:05d}-" + "한글 자료 " * 80
        if _split(text) == target:
            return text
    raise AssertionError(f"could not construct a {target} record")


def _normalized_variants_by_split() -> dict[str, str]:
    found: dict[str, str] = {}
    for width in range(1, 2_000):
        text = "가" * 150 + " " * width + "나" * 150
        found.setdefault(_split(text), text)
        if set(found) == {"train", "calibration", "test"}:
            values = list(found.values())
            if len({normalized_record_digest(value) for value in values}) != 1:
                raise AssertionError("synthetic variants are not normalized duplicates")
            return found
    raise AssertionError("could not construct normalized variants in every split")


def _small_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with mock.patch.object(
        module,
        "SPLIT_QUOTAS",
        {"train": 200, "calibration": 200},
    ):
        for split in ("train", "calibration"):
            manifest["selection"]["quotas"][split] = 200
            manifest["selection"]["reserve_stream_bytes"][split] = 262_345
            manifest["selection"]["expected_rank_key_hex"][split] = (
                derive_split_rank_key(
                    manifest["source"]["expected_sha256"],
                    manifest["predecessor"]["output_sha256"],
                    manifest["final_test"]["output_sha256"],
                    split,
                    200,
                ).hex()
            )
    return manifest


def _valid_seal_payload() -> dict[str, object]:
    zero = "0" * 64
    commit = "1" * 40
    counts = {"train": 500, "calibration": 32}
    split_rows = {}
    for split in ("train", "calibration"):
        quota = SPLIT_QUOTAS[split]
        count = counts[split]
        split_rows[split] = {
            "available_stream_bytes": quota,
            "normalized_selected_set_commitment_sha256": zero,
            "overshoot_stream_bytes": 0,
            "selected_document_count": count,
            "selected_document_raw_bytes": quota - count + 1,
            "selected_set_commitment_sha256": zero,
            "sequence_count": quota // 512,
            "stream_bytes": quota,
            "stream_sha256": zero,
        }
    scan = {
        field: 0 for field in module.FreshScanStatistics.__dataclass_fields__
    }
    scan.update(
        {
            "source_lines": 8_925,
            "parsed_records": 8_925,
            "eligible_records": 8_925,
            "excluded_exact_records": 8_393,
            "candidate_train_records": 500,
            "candidate_calibration_records": 32,
        }
    )
    return {
        "dataset_id": module.DATASET_ID,
        "exclusions": {
            "exact_commitment_sha256": zero,
            "exact_count": 8_393,
            "final_exact_count": 1_482,
            "normalized_commitment_sha256": zero,
            "normalized_count": 8_393,
            "predecessor_exact_count": 6_911,
        },
        "manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "output": {"bytes": 140_000_000, "sha256": zero},
        "preparation_git_commit": commit,
        "privacy": {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        },
        "protocol_version": module.PROTOCOL_VERSION,
        "scan": scan,
        "source": {
            "bytes": 1_862_302_013,
            "sha256": module._SOURCE_SHA256,
        },
        "splits": split_rows,
    }


class FreshAdaptationDataTests(unittest.TestCase):
    def test_tracked_manifest_and_derived_rank_keys_are_exact(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        expected_keys = {
            "train": "260a778765602de9fd0c621e632b34eea2aabc883c03ef2c9c85cdbd550624ef",
            "calibration": "ad43eca68c804fc4ad61649c20c704d05f62ef490bd5de17104a83310c291517",
        }
        for split in ("train", "calibration"):
            actual = derive_split_rank_key(
                manifest["source"]["expected_sha256"],
                manifest["predecessor"]["output_sha256"],
                manifest["final_test"]["output_sha256"],
                split,
                SPLIT_QUOTAS[split],
            ).hex()
            self.assertEqual(actual, expected_keys[split])
            self.assertEqual(actual, manifest["selection"]["expected_rank_key_hex"][split])

    def test_manifest_rejects_rank_or_quota_rotation(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for mutate in ("rank", "quota"):
            changed = copy.deepcopy(manifest)
            if mutate == "rank":
                changed["selection"]["expected_rank_key_hex"]["train"] = "0" * 64
            else:
                changed["selection"]["quotas"]["train"] += 512
            with self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_serialization_has_fixed_split_order_and_no_split_label(self) -> None:
        train_raw = "훈련 문서".encode()
        calibration_raw = "검증 문서".encode()
        selected = {
            "train": [FinalTestCandidate(b"a" * 32, b"b" * 32, train_raw)],
            "calibration": [
                FinalTestCandidate(b"c" * 32, b"d" * 32, calibration_raw)
            ],
        }
        rows = [json.loads(line) for line in serialize_jsonl(selected).splitlines()]
        self.assertEqual([row["text"] for row in rows], ["훈련 문서", "검증 문서"])
        self.assertTrue(all(set(row) == {"language", "text"} for row in rows))

    def test_rank_domain_separates_splits(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        digest = bytes.fromhex("12" * 32)
        ranks = {
            split: rank_digest(
                derive_split_rank_key(
                    manifest["source"]["expected_sha256"],
                    manifest["predecessor"]["output_sha256"],
                    manifest["final_test"]["output_sha256"],
                    split,
                    SPLIT_QUOTAS[split],
                ),
                digest,
            )
            for split in ("train", "calibration")
        }
        self.assertNotEqual(ranks["train"], ranks["calibration"])

    def test_stable_test_normalized_variant_cannot_suppress_train_candidate(self) -> None:
        variants = _normalized_variants_by_split()
        calibration = _text_for_split("independent-calibration", "calibration")
        lines = [
            json.dumps({"text": variants["test"]}, ensure_ascii=False).encode(),
            json.dumps({"text": variants["train"]}, ensure_ascii=False).encode(),
            json.dumps({"text": calibration}, ensure_ascii=False).encode(),
        ]
        manifest = _small_manifest()
        exclusions = ExclusionIndex(frozenset(), frozenset(), 0, 0)
        with mock.patch.object(
            module,
            "SPLIT_QUOTAS",
            {"train": 200, "calibration": 200},
        ):
            selected, statistics = scan_lines(lines, manifest, exclusions)
        self.assertEqual(statistics.stable_test_records_ignored, 1)
        self.assertEqual(statistics.normalized_source_duplicates, 0)
        self.assertEqual(
            selected["train"][0].digest,
            sha256(variants["train"].encode()).digest(),
        )

    def test_exact_and_normalized_historical_variants_are_both_excluded(self) -> None:
        variants = _normalized_variants_by_split()
        historical = variants["test"]
        historical_raw = historical.encode()
        historical_digest = sha256(historical_raw).digest()
        fallback_train = _text_for_split("fallback-train", "train")
        calibration = _text_for_split("fallback-calibration", "calibration")
        lines = [
            json.dumps({"text": historical}, ensure_ascii=False).encode(),
            json.dumps({"text": variants["train"]}, ensure_ascii=False).encode(),
            json.dumps({"text": fallback_train}, ensure_ascii=False).encode(),
            json.dumps({"text": calibration}, ensure_ascii=False).encode(),
        ]
        manifest = _small_manifest()
        exclusions = ExclusionIndex(
            frozenset({historical_digest}),
            frozenset({normalized_record_digest(historical)}),
            1,
            0,
        )
        with mock.patch.object(
            module,
            "SPLIT_QUOTAS",
            {"train": 200, "calibration": 200},
        ):
            selected, statistics = scan_lines(lines, manifest, exclusions)
        self.assertEqual(statistics.excluded_exact_records, 1)
        self.assertEqual(statistics.excluded_normalized_records, 1)
        selected_digests = {
            candidate.digest
            for candidates in selected.values()
            for candidate in candidates
        }
        self.assertNotIn(sha256(variants["train"].encode()).digest(), selected_digests)

    def test_seal_validator_rejects_nested_schema_and_accounting_rotation(self) -> None:
        payload = _valid_seal_payload()
        validate_seal_payload(payload)
        unknown = copy.deepcopy(payload)
        unknown["splits"]["train"]["model_loss"] = 1.0
        with self.assertRaises(ValueError):
            validate_seal_payload(unknown)
        inconsistent = copy.deepcopy(payload)
        inconsistent["scan"]["candidate_train_records"] -= 1
        with self.assertRaises(ValueError):
            validate_seal_payload(inconsistent)

    def test_manifest_directly_pins_every_preexisting_dependency(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        dependencies = {
            "predecessor.manifest": (
                ROOT / manifest["predecessor"]["manifest_path"],
                manifest["predecessor"]["manifest_sha256"],
            ),
            "predecessor.summary": (
                ROOT / manifest["predecessor"]["summary_path"],
                manifest["predecessor"]["summary_sha256"],
            ),
            "predecessor.integrity": (
                ROOT / manifest["predecessor"]["integrity_path"],
                manifest["predecessor"]["integrity_sha256"],
            ),
            "predecessor.output": (
                ROOT / manifest["predecessor"]["output_path"],
                manifest["predecessor"]["output_sha256"],
            ),
            "final.manifest": (
                ROOT / manifest["final_test"]["manifest_path"],
                manifest["final_test"]["manifest_sha256"],
            ),
            "final.seal": (
                ROOT / manifest["final_test"]["seal_path"],
                manifest["final_test"]["seal_sha256"],
            ),
            "final.output": (
                ROOT / manifest["final_test"]["output_path"],
                manifest["final_test"]["output_sha256"],
            ),
        }
        for label, (path, expected) in dependencies.items():
            with self.subTest(label=label):
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), expected)

    def test_preparation_code_cannot_read_model_or_result_artifacts(self) -> None:
        forbidden_roots = {"numpy", "tokenizers", "torch", "transformers"}
        paths = [
            ROOT / "scripts/hplt3_fresh_adaptation_protocol.py",
            ROOT / "scripts/prepare_hplt3_fresh_adaptation.py",
            ROOT / "scripts/verify_hplt3_fresh_adaptation.py",
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
                    if value.startswith(("artifacts/", "runs/")) or (
                        value.startswith("results/")
                        and value != "results/phase3-data/summary.json"
                    ):
                        forbidden_paths.append(value)
            with self.subTest(path=path):
                self.assertFalse(imported & forbidden_roots)
                self.assertFalse(forbidden_paths)


if __name__ == "__main__":
    unittest.main()
