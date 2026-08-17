from __future__ import annotations

import ast
import copy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest import mock

from jamoflow.corpus import Record, split_for_record
from jamoflow.hplt3_final_test import FinalTestCandidate, normalized_record_digest
import hplt3_fresh_adaptation_v2_protocol as module
from hplt3_fresh_adaptation_v2_protocol import (
    ExclusionIndex,
    SPLIT_QUOTAS,
    derive_split_rank_key,
    rank_digest,
    scan_lines,
    validate_manifest,
    validate_seal_payload,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/hplt3-korean-vocab-adaptation-v2.json"


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
        text = f"{label}-{index:05d}-" + "한글 연구 자료 " * 80
        if _split(text) == target:
            return text
    raise AssertionError(f"could not construct a {target} record")


def _normalized_variants_by_split() -> dict[str, str]:
    found: dict[str, str] = {}
    for width in range(1, 2_000):
        text = "가" * 150 + " " * width + "나" * 150
        found.setdefault(_split(text), text)
        if set(found) == {"train", "calibration", "test"}:
            if len({normalized_record_digest(value) for value in found.values()}) != 1:
                raise AssertionError("synthetic variants are not normalized duplicates")
            return found
    raise AssertionError("could not construct normalized variants")


def _small_manifest() -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with mock.patch.object(
        module,
        "SPLIT_QUOTAS",
        {"train": 200, "calibration": 200},
    ):
        for split in module.SPLIT_ORDER:
            manifest["selection"]["quotas"][split] = 200
            manifest["selection"]["reserve_stream_bytes"][split] = 262_345
            manifest["selection"]["expected_rank_key_hex"][split] = (
                derive_split_rank_key(
                    manifest["source"]["expected_sha256"],
                    manifest["predecessor"]["output_sha256"],
                    manifest["final_test"]["output_sha256"],
                    manifest["fresh_v1"]["output_sha256"],
                    split,
                    200,
                ).hex()
            )
    return manifest


def _valid_seal_payload() -> dict[str, object]:
    zero = "0" * 64
    counts = {"train": 500, "calibration": 32}
    splits = {}
    for split in module.SPLIT_ORDER:
        quota = SPLIT_QUOTAS[split]
        count = counts[split]
        splits[split] = {
            "available_stream_bytes": quota,
            "normalized_selected_set_commitment_sha256": zero,
            "overshoot_stream_bytes": 0,
            "selected_document_count": count,
            "selected_document_raw_bytes": quota - count + 1,
            "selected_set_commitment_sha256": zero,
            "sequence_count": quota // module.SEQUENCE_LENGTH,
            "stream_bytes": quota,
            "stream_sha256": zero,
        }
    scan = {
        field: 0 for field in module.FreshScanStatistics.__dataclass_fields__
    }
    scan.update(
        {
            "source_lines": module.TOTAL_EXACT_COUNT + sum(counts.values()),
            "parsed_records": module.TOTAL_EXACT_COUNT + sum(counts.values()),
            "eligible_records": module.TOTAL_EXACT_COUNT + sum(counts.values()),
            "excluded_exact_records": module.TOTAL_EXACT_COUNT,
            "candidate_train_records": counts["train"],
            "candidate_calibration_records": counts["calibration"],
        }
    )
    return {
        "dataset_id": module.DATASET_ID,
        "exclusions": {
            "exact_commitment_sha256": zero,
            "exact_count": module.TOTAL_EXACT_COUNT,
            "final_exact_count": module.FINAL_EXACT_COUNT,
            "fresh_v1_exact_count": module.FRESH_V1_EXACT_COUNT,
            "historical_exact_count": module.HISTORICAL_EXACT_COUNT,
            "normalized_commitment_sha256": zero,
            "normalized_count": module.TOTAL_EXACT_COUNT,
            "predecessor_exact_count": module.PREDECESSOR_EXACT_COUNT,
        },
        "fresh_v1_dependency": {
            "output_sha256": module.FRESH_V1_OUTPUT_SHA256,
            "payload_sha256": module.FRESH_V1_PAYLOAD_SHA256,
            "selected_document_count": module.FRESH_V1_EXACT_COUNT,
        },
        "manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "output": {"bytes": 140_000_000, "sha256": zero},
        "preparation_git_commit": "1" * 40,
        "privacy": {
            "individual_document_digests_tracked": False,
            "model_metrics_tracked": False,
            "raw_text_tracked": False,
        },
        "protocol_version": module.PROTOCOL_VERSION,
        "scan": scan,
        "source": {"bytes": 1_862_302_013, "sha256": module.SOURCE_SHA256},
        "splits": splits,
    }


class FreshAdaptationV2Tests(unittest.TestCase):
    def test_manifest_and_v1_bound_rank_keys_are_exact(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        expected = {
            "train": "eff3b43937f982d12e6bcc304a742ed53dd2f227fb27e75ec6e3990baaa8d418",
            "calibration": "a09677ea7f9c271b014a3246285188dba389a0e01082e34da0ce41613b3c9d65",
        }
        for split in module.SPLIT_ORDER:
            actual = derive_split_rank_key(
                manifest["source"]["expected_sha256"],
                manifest["predecessor"]["output_sha256"],
                manifest["final_test"]["output_sha256"],
                manifest["fresh_v1"]["output_sha256"],
                split,
                SPLIT_QUOTAS[split],
            ).hex()
            self.assertEqual(actual, expected[split])
            self.assertEqual(actual, manifest["selection"]["expected_rank_key_hex"][split])

    def test_rank_key_changes_when_v1_identity_changes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        actual = derive_split_rank_key(
            manifest["source"]["expected_sha256"],
            manifest["predecessor"]["output_sha256"],
            manifest["final_test"]["output_sha256"],
            manifest["fresh_v1"]["output_sha256"],
            "train",
            SPLIT_QUOTAS["train"],
        )
        rotated = derive_split_rank_key(
            manifest["source"]["expected_sha256"],
            manifest["predecessor"]["output_sha256"],
            manifest["final_test"]["output_sha256"],
            "0" * 64,
            "train",
            SPLIT_QUOTAS["train"],
        )
        self.assertNotEqual(actual, rotated)

    def test_manifest_rejects_v1_or_quota_rotation(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for label in ("v1", "quota", "unknown"):
            changed = copy.deepcopy(manifest)
            if label == "v1":
                changed["fresh_v1"]["output_sha256"] = "0" * 64
            elif label == "quota":
                changed["selection"]["quotas"]["train"] += 512
            else:
                changed["model_metric"] = 1.0
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_exact_and_normalized_v1_documents_are_excluded(self) -> None:
        variants = _normalized_variants_by_split()
        exact_text = _text_for_split("fresh-v1-exact", "train")
        exact_raw = exact_text.encode()
        fallback_train = _text_for_split("fresh-v2-train", "train")
        fallback_calibration = _text_for_split("fresh-v2-calibration", "calibration")
        lines = [
            json.dumps({"text": exact_text}, ensure_ascii=False).encode(),
            json.dumps({"text": variants["train"]}, ensure_ascii=False).encode(),
            json.dumps({"text": fallback_train}, ensure_ascii=False).encode(),
            json.dumps({"text": fallback_calibration}, ensure_ascii=False).encode(),
        ]
        exclusions = ExclusionIndex(
            exact=frozenset({sha256(exact_raw).digest()}),
            normalized=frozenset({normalized_record_digest(variants["test"])}),
            predecessor_exact_count=0,
            final_exact_count=0,
            fresh_v1_exact_count=1,
        )
        manifest = _small_manifest()
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
        self.assertNotIn(sha256(exact_raw).digest(), selected_digests)
        self.assertNotIn(sha256(variants["train"].encode()).digest(), selected_digests)

    def test_stable_test_variant_cannot_suppress_train_candidate(self) -> None:
        variants = _normalized_variants_by_split()
        calibration = _text_for_split("fresh-v2-independent-cal", "calibration")
        lines = [
            json.dumps({"text": variants["test"]}, ensure_ascii=False).encode(),
            json.dumps({"text": variants["train"]}, ensure_ascii=False).encode(),
            json.dumps({"text": calibration}, ensure_ascii=False).encode(),
        ]
        manifest = _small_manifest()
        exclusions = ExclusionIndex(frozenset(), frozenset(), 0, 0, 0)
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

    def test_seal_rejects_schema_count_and_v1_rotation(self) -> None:
        payload = _valid_seal_payload()
        validate_seal_payload(payload)
        for label in ("schema", "count", "v1"):
            changed = copy.deepcopy(payload)
            if label == "schema":
                changed["splits"]["train"]["loss"] = 1.0
            elif label == "count":
                changed["scan"]["excluded_exact_records"] -= 1
            else:
                changed["fresh_v1_dependency"]["output_sha256"] = "0" * 64
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_seal_payload(changed)

    def test_manifest_pins_all_data_dependencies(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        dependencies = []
        for section, pairs in (
            (
                "predecessor",
                (("manifest_path", "manifest_sha256"), ("summary_path", "summary_sha256"),
                 ("integrity_path", "integrity_sha256"), ("output_path", "output_sha256")),
            ),
            (
                "final_test",
                (("manifest_path", "manifest_sha256"), ("seal_path", "seal_sha256"),
                 ("output_path", "output_sha256")),
            ),
            (
                "fresh_v1",
                (("manifest_path", "manifest_sha256"), ("protocol_path", "protocol_sha256"),
                 ("seal_path", "seal_sha256"), ("output_path", "output_sha256")),
            ),
        ):
            for path_key, hash_key in pairs:
                dependencies.append(
                    (f"{section}.{path_key}", ROOT / manifest[section][path_key], manifest[section][hash_key])
                )
        for label, path, expected in dependencies:
            with self.subTest(label=label):
                self.assertEqual(sha256(path.read_bytes()).hexdigest(), expected)

    def test_preparation_code_has_no_model_or_result_input(self) -> None:
        forbidden_roots = {"numpy", "tokenizers", "torch", "transformers"}
        paths = [
            ROOT / "scripts/hplt3_fresh_adaptation_v2_protocol.py",
            ROOT / "scripts/prepare_hplt3_fresh_adaptation_v2.py",
            ROOT / "scripts/verify_hplt3_fresh_adaptation_v2.py",
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
