from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


RUNNER = _load_script("run_phase3_normalization")
SUMMARY = _load_script("summarize_phase3_normalization")


class Phase3NormalizationTests(unittest.TestCase):
    def test_strict_prefix_allows_only_truncated_terminal_codepoint(self) -> None:
        data = "가A나".encode("utf-8")[:-1]
        text, discarded = RUNNER.strict_decodable_prefix(data)
        self.assertEqual(text, "가A")
        self.assertEqual(discarded, 2)
        with self.assertRaises(ValueError):
            RUNNER.strict_decodable_prefix(b"A\xffB")

    def test_manifest_merge_preserves_invariants_and_unions_seeds(self) -> None:
        current = {
            "schema_version": 1,
            "created_at": "first",
            "git_commit": "a",
            "device": "cpu",
            "platform": "test",
            "versions": {},
            "seeds": [1729],
            "policies": ["fixed_byte_6"],
            "prepare_only": True,
            "design": {"fixed": True},
            "source": {"sha256": "source"},
            "conditions": {"nfc": {}, "nfd": {}},
            "model_spec": {"width": 1},
        }
        first = RUNNER.merge_normalization_manifest(None, current)
        later = dict(current)
        later.update(
            {
                "created_at": "second",
                "git_commit": "b",
                "seeds": [2718],
                "policies": ["causal_codepoint_grid"],
                "prepare_only": False,
            }
        )
        merged = RUNNER.merge_normalization_manifest(first, later)
        self.assertEqual(merged["seeds"], [1729, 2718])
        self.assertEqual(
            merged["policies"],
            ["fixed_byte_6", "causal_codepoint_grid"],
        )
        self.assertEqual(len(merged["invocations"]), 2)
        self.assertEqual(merged["updated_at"], "second")

    def test_manifest_merge_rejects_source_change(self) -> None:
        base = {
            "schema_version": 1,
            "created_at": "first",
            "git_commit": "a",
            "device": "cpu",
            "platform": "test",
            "versions": {},
            "seeds": [1729],
            "policies": ["fixed_byte_6"],
            "prepare_only": True,
            "design": {},
            "source": {"sha256": "a"},
            "conditions": {},
            "model_spec": {},
        }
        existing = RUNNER.merge_normalization_manifest(None, base)
        changed = dict(base)
        changed["source"] = {"sha256": "b"}
        with self.assertRaises(ValueError):
            RUNNER.merge_normalization_manifest(existing, changed)

    def test_paired_and_relative_effects(self) -> None:
        self.assertEqual(
            SUMMARY.paired_values([2.0, 4.0], [1.0, 1.5]),
            [1.0, 2.5],
        )
        self.assertEqual(
            SUMMARY.relative_increases([2.0, 3.0], [1.0, 2.0]),
            [1.0, 0.5],
        )
        with self.assertRaises(ValueError):
            SUMMARY.relative_increases([1.0, 2.0], [0.0, 1.0])

    def test_summary_rejects_non_preregistered_seed_subset(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial 3 or final 5"):
            SUMMARY.run(SimpleNamespace(seeds=[1729, 2718]))

    def test_summary_manifest_requires_evaluation_invocations(self) -> None:
        manifest = {
            "schema_version": 1,
            "seeds": list(SUMMARY.INITIAL_SEEDS),
            "policies": list(SUMMARY.POLICIES),
            "invocations": [
                {
                    "seeds": list(SUMMARY.INITIAL_SEEDS),
                    "policies": list(SUMMARY.POLICIES),
                    "prepare_only": False,
                }
            ],
        }
        SUMMARY._validate_manifest_execution(manifest, SUMMARY.INITIAL_SEEDS)
        manifest["invocations"][0]["prepare_only"] = True
        with self.assertRaisesRegex(ValueError, "evaluation invocation"):
            SUMMARY._validate_manifest_execution(
                manifest,
                SUMMARY.INITIAL_SEEDS,
            )

    def test_runner_completed_result_is_bound_to_lineage_and_arithmetic(
        self,
    ) -> None:
        counts = np.asarray([2, 3], dtype=np.uint16)
        losses = np.asarray([1.0, 2.0], dtype=np.float64)
        denominators = {
            "utf8_bytes": 10,
            "unicode_codepoints": 5,
            "precomposed_hangul_syllables": 2,
        }
        lineage = {
            "checkpoint_state_sha256": "state",
            "training_report_state_sha256": "state",
            "checkpoint_artifact_sha256": "checkpoint",
            "training_report_artifact_sha256": "report",
        }
        patch = {"matrix_sha256": "matrix"}
        condition = {
            "padded_stream_sha256": "stream",
            "target_mask_sha256": "mask",
            "patch_diagnostics": {SUMMARY.F: patch},
        }
        total_bits = float(losses.sum()) / math.log(2)
        report = {
            "schema_version": 1,
            "seed": 1729,
            "condition": "nfc",
            "policy": SUMMARY.F,
            "parameters": 19_596_096,
            "model_spec": SUMMARY.PHASE3_MODEL_SPEC.to_dict(),
            "global_max_position_embeddings": SUMMARY.GLOBAL_POSITION_LIMIT,
            **lineage,
            "condition_stream_sha256": "stream",
            "target_mask_sha256": "mask",
            "patch_matrix_sha256": "matrix",
            "patch_diagnostics": patch,
            "evaluation": {
                "examples": 2,
                "predicted_bytes": 5,
                "total_nll_nats": 3.0,
                "bpb": total_bits / 5,
                "scored_bits_per_source_utf8_byte": total_bits / 10,
                "scored_bits_per_source_unicode_codepoint": total_bits / 5,
                "scored_bits_per_source_precomposed_hangul_syllable": (
                    total_bits / 2
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            artifact_path = Path(directory) / "loss.npz"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            np.savez_compressed(
                artifact_path,
                sequence_nll_nats=losses,
                sequence_target_counts=counts,
            )
            RUNNER._validate_completed_result(
                report_path,
                artifact_path,
                seed=1729,
                condition="nfc",
                policy=SUMMARY.F,
                lineage=lineage,
                expected_condition=condition,
                expected_target_counts=counts,
                source_denominators=denominators,
            )
            changed = dict(lineage)
            changed["checkpoint_state_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "stale normalization"):
                RUNNER._validate_completed_result(
                    report_path,
                    artifact_path,
                    seed=1729,
                    condition="nfc",
                    policy=SUMMARY.F,
                    lineage=changed,
                    expected_condition=condition,
                    expected_target_counts=counts,
                    source_denominators=denominators,
                )


if __name__ == "__main__":
    unittest.main()
