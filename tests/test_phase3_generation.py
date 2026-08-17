from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
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


RUNNER = _load_script("run_phase3_generation")
SUMMARY = _load_script("summarize_phase3_generation")


class Phase3GenerationTests(unittest.TestCase):
    def test_manifest_merge_preserves_design_and_unions_work(self) -> None:
        base = {
            "schema_version": 1,
            "created_at": "first",
            "git_commit": "a",
            "device": "cpu",
            "platform": "test",
            "versions": {},
            "seeds": [1729],
            "policies": ["fixed_byte_6"],
            "force": False,
            "design": {"horizon": 512},
            "source": {"sha256": "source"},
            "prompt_selection": {"selected_prompts": 256},
            "global_max_position_embeddings": 1032,
            "model_spec": {"width": 384},
            "optimization_spec": {"learning_rate": 0.0003},
        }
        first = RUNNER.merge_generation_manifest(None, base)
        later = dict(base)
        later.update(
            {
                "created_at": "second",
                "git_commit": "b",
                "seeds": [2718],
                "policies": ["causal_codepoint_grid"],
            }
        )
        merged = RUNNER.merge_generation_manifest(first, later)
        self.assertEqual(merged["seeds"], [1729, 2718])
        self.assertEqual(
            merged["policies"],
            ["fixed_byte_6", "causal_codepoint_grid"],
        )
        self.assertEqual(len(merged["invocations"]), 2)

    def test_manifest_merge_rejects_prompt_change(self) -> None:
        base = {
            "schema_version": 1,
            "created_at": "first",
            "git_commit": "a",
            "device": "cpu",
            "platform": "test",
            "versions": {},
            "seeds": [1729],
            "policies": ["fixed_byte_6"],
            "force": False,
            "design": {},
            "source": {},
            "prompt_selection": {"selected_prompts": 256},
            "global_max_position_embeddings": 1032,
            "model_spec": {},
            "optimization_spec": {},
        }
        existing = RUNNER.merge_generation_manifest(None, base)
        changed = dict(base)
        changed["prompt_selection"] = {"selected_prompts": 255}
        with self.assertRaises(ValueError):
            RUNNER.merge_generation_manifest(existing, changed)

    def test_failure_partition_validation(self) -> None:
        valid = {
            "continuations": 10,
            "strict_valid_count": 3,
            "illegal_transition_count": 5,
            "incomplete_terminal_scalar_count": 2,
            "strict_valid_rate": 0.3,
            "illegal_transition_rate": 0.5,
            "incomplete_terminal_scalar_rate": 0.2,
        }
        self.assertTrue(SUMMARY.failure_partition_is_valid(valid, 10))
        invalid = dict(valid)
        invalid["incomplete_terminal_scalar_count"] = 3
        self.assertFalse(SUMMARY.failure_partition_is_valid(invalid, 10))

    def test_paired_effects_are_left_minus_right(self) -> None:
        self.assertEqual(
            SUMMARY.paired_effects([0.4, 0.5], [0.3, 0.2]),
            [0.10000000000000003, 0.3],
        )
        with self.assertRaises(ValueError):
            SUMMARY.paired_effects([0.4], [0.3])

    def test_summary_manifest_requires_every_seed_policy_invocation(self) -> None:
        manifest = {
            "schema_version": 1,
            "seeds": list(SUMMARY.INITIAL_SEEDS),
            "policies": list(SUMMARY.GENERATION_POLICIES),
            "invocations": [
                {
                    "seeds": [SUMMARY.INITIAL_SEEDS[0]],
                    "policies": list(SUMMARY.GENERATION_POLICIES),
                }
            ],
        }
        with self.assertRaises(ValueError):
            SUMMARY._validate_manifest_execution(
                manifest,
                SUMMARY.INITIAL_SEEDS,
            )
        manifest["invocations"] = [
            {
                "seeds": list(SUMMARY.INITIAL_SEEDS),
                "policies": list(SUMMARY.GENERATION_POLICIES),
            }
        ]
        SUMMARY._validate_manifest_execution(manifest, SUMMARY.INITIAL_SEEDS)

    def test_completed_result_is_reconstructed_and_tampering_is_rejected(
        self,
    ) -> None:
        seed = 1729
        policy = "fixed_byte_6"
        prompts = np.asarray([[65], [66]], dtype=np.uint8)
        generated = [b"xy", b"zz"]
        mode_results = {}
        artifact_arrays = {}
        for mode in RUNNER.DECODING_MODES:
            metrics, diagnostics = RUNNER._aggregate_generation_metrics(
                prompts,
                generated,
                1.0,
            )
            mode_results[mode] = {"unconstrained": metrics}
            artifact_arrays.update(
                RUNNER._tag_diagnostics(mode, "unconstrained", diagnostics)
            )
        lineage = {
            "checkpoint_state_sha256": "state",
            "training_report_state_sha256": "state",
            "checkpoint_artifact_sha256": "checkpoint",
            "training_report_artifact_sha256": "report",
        }
        prompt_selection = {
            "selected_prompts": 2,
            "candidate_prompts": 2,
            "unique_candidate_prompts": 2,
            "prompt_length_bytes": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / f"seed-{seed}" / f"{policy}-diagnostics.npz"
            report_path = root / "result.json"
            RUNNER._save_npz(artifact_path, artifact_arrays)
            report = {
                "schema_version": 1,
                "seed": seed,
                "policy": policy,
                "parameters": 19_596_096,
                "model_spec": RUNNER.PHASE3_MODEL_SPEC.to_dict(),
                "optimization_spec": RUNNER.PHASE3_OPTIMIZATION_SPEC.to_dict(),
                "global_max_position_embeddings": RUNNER.GLOBAL_POSITION_LIMIT,
                **lineage,
                "source_stream_sha256": "source",
                "prompt_selection": prompt_selection,
                "diagnostic_artifact_filename": (
                    f"seed-{seed}/{policy}-diagnostics.npz"
                ),
                "diagnostic_artifact_sha256": RUNNER._sha256_file(artifact_path),
                "modes": mode_results,
                "raw_generation_serialized": False,
                "prompts_or_prompt_hashes_serialized": False,
                "non_content_per_prompt_diagnostics_serialized": True,
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            RUNNER._validate_completed_result(
                report_path,
                artifact_path,
                seed=seed,
                policy=policy,
                lineage=lineage,
                prompt_selection=prompt_selection,
                source_stream_sha256="source",
                continuation_bytes=2,
                hard_mask_control=False,
            )

            tampered = {key: value.copy() for key, value in artifact_arrays.items()}
            tampered[
                "greedy__unconstrained__structural__strict_valid"
            ][0] = 0
            RUNNER._save_npz(artifact_path, tampered)
            report["diagnostic_artifact_sha256"] = RUNNER._sha256_file(
                artifact_path
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                RUNNER._validate_completed_result(
                    report_path,
                    artifact_path,
                    seed=seed,
                    policy=policy,
                    lineage=lineage,
                    prompt_selection=prompt_selection,
                    source_stream_sha256="source",
                    continuation_bytes=2,
                    hard_mask_control=False,
                )


if __name__ == "__main__":
    unittest.main()
