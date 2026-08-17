from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from jamoflow.inference_initial_model_identity_v2 import (
    CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER,
    PLAN_FROZEN_FULL_FILE_ORDER,
    build_implementation_manifest_v2,
    build_initial_model_identity_lock_v2,
    build_plan_frozen_selection_v2,
    canonical_sha256,
    validate_initial_model_identity_lock_v2,
)
from jamoflow.compute_conversion import CONVERSION_POLICIES, conversion_model_spec
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER,
    INITIAL_SEEDS,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    PHASE3_POLICIES,
    THRESHOLD_POLICIES,
)


def digest(character: str) -> str:
    return character * 64


def model_fixture(seed: int, policy: str) -> dict:
    phase3 = policy in PHASE3_POLICIES
    patch_count = 86 if phase3 else int(policy.rsplit("_", 1)[1])
    root = "phase3" if phase3 else "phase3-compute-conversion"
    auxiliary = {"kind": "none"}
    if policy in THRESHOLD_POLICIES:
        auxiliary = {
            "kind": "entropy_router",
            "router_checkpoint_artifact_sha256": digest("1"),
            "router_checkpoint_path": f"artifacts/phase3/seed-{seed}/router.pt",
            "router_report_artifact_sha256": digest("2"),
            "router_report_path": f"runs/phase3/seed-{seed}/router.json",
            "router_state_sha256": digest("3"),
            "threshold_cache_artifact_sha256": digest("4"),
            "threshold_cache_path": (
                f"artifacts/phase3/seed-{seed}/threshold-patches.npz"
            ),
            "threshold_diagnostics_artifact_sha256": digest("5"),
            "threshold_diagnostics_path": (
                f"runs/phase3/seed-{seed}/threshold-patch-diagnostics.json"
            ),
        }
    return {
        "auxiliary": auxiliary,
        "checkpoint": {
            "artifact_sha256": digest("6"),
            "path": f"artifacts/{root}/seed-{seed}/{policy}.pt",
            "state_sha256": digest("7"),
        },
        "initialization_sha256": digest("8"),
        "model_family": "phase3" if phase3 else "compute_conversion",
        "model_spec_sha256": canonical_sha256(
            (PHASE3_MODEL_SPEC if phase3 else conversion_model_spec(patch_count)).to_dict()
        ),
        "optimization_spec_sha256": canonical_sha256(
            PHASE3_OPTIMIZATION_SPEC.to_dict()
        ),
        "parameter_count": 19_596_096,
        "patch_count": patch_count,
        "policy": policy,
        "seed": seed,
        "training_order_sha256": digest("b"),
        "training_report": {
            "artifact_sha256": digest("c"),
            "path": f"runs/{root}/seed-{seed}/{policy}.json",
        },
    }


class InferenceInitialModelIdentityV2Tests(unittest.TestCase):
    def _environment(self) -> dict:
        return {
            "hardware": {
                "chip": "Apple M3 Max",
                "machine_model": "Mac15,9",
                "memory_bytes": 64 * 1024**3,
                "os_build": "25A1",
            },
            "machine": "arm64",
            "mac_ver": "15.0",
            "packages": {
                "numpy": "2.0",
                "torch": "2.7",
                "transformers": "4.0",
            },
            "platform": "macOS-arm64",
            "python_implementation": "CPython",
            "python_version": "3.12.0",
            "sys_version_info": [3, 12, 0],
            "torch_runtime": {
                "git_version": "abc",
                "mps_available": True,
                "mps_built": True,
                "version": "2.7",
            },
        }

    def _source_identities(self) -> dict:
        stream = {
            "boundaries_sha256": digest("1"),
            "inputs_sha256": digest("2"),
            "selected_stream_sha256": digest("3"),
            "sequence_count": 10,
            "whitespace_sha256": digest("4"),
        }
        return {
            "conversion_patch_artifacts": {
                "cache": {
                    "artifact_sha256": digest("5"),
                    "path": "artifacts/phase3-compute-conversion/patches.npz",
                },
                "diagnostics": {
                    "artifact_sha256": digest("6"),
                    "path": "runs/phase3-compute-conversion/patch-diagnostics.json",
                },
            },
            "phase3_initial_summary": {
                "artifact_sha256": digest("7"),
                "path": "results/phase3-all-initial/summary.json",
            },
            "primary_summary": {
                "artifact_sha256": digest("8"),
                "path": "results/phase3-primary-five-seed/summary.json",
            },
            "source_artifact": {
                "bytes": 100,
                "filename": "ko.jsonl",
                "sha256": digest("9"),
            },
            "source_integrity_artifact": {
                "bytes": 200,
                "filename": "integrity.json",
                "sha256": digest("a"),
            },
            "streams": {
                split: dict(stream) for split in ("train", "calibration", "test")
            },
        }

    def _conversion_training(self, plan_artifact_sha256: str, source: dict) -> dict:
        unsigned = {
            "device": "mps",
            "git_commit": "4" * 40,
            "git_worktree_clean_at_start": True,
            "policies": list(CONVERSION_POLICIES),
            "primary_summary_sha256": source["primary_summary"]["artifact_sha256"],
            "schema_version": 1,
            "seeds": list(INITIAL_SEEDS),
            "selection_plan_sha256": plan_artifact_sha256,
            "selection_summary_sha256": None,
            "stage": "initial",
        }
        binding = {**unsigned, "identity_sha256": canonical_sha256(unsigned)}
        return {
            "authorized_invocation_count": 1,
            "evidence_binding": binding,
            "manifest": {
                "artifact_sha256": digest("b"),
                "path": "runs/phase3-compute-conversion/manifest.json",
            },
            "run_git_commit": "4" * 40,
            "run_implementation_file_order": list(
                CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER
            ),
            "run_implementation_sha256": {
                path: digest("c")
                for path in CONVERSION_TRAINING_IMPLEMENTATION_FILE_ORDER
            },
        }

    def _lock(self) -> dict:
        order = CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
        frozen = build_plan_frozen_selection_v2(
            plan_git_commit="a" * 40,
            decision_ast_sha256=digest("b"),
            full_file_sha256_at_plan={
                path: digest("c") for path in PLAN_FROZEN_FULL_FILE_ORDER
            },
        )
        implementation = build_implementation_manifest_v2(
            sha256_by_path={path: digest("d") for path in order},
            last_change_commit_by_path={path: "e" * 40 for path in order},
            producer_git_commit="f" * 40,
            environment=self._environment(),
            plan_frozen_selection=frozen,
        )
        models = {
            seed: {
                policy: model_fixture(seed, policy)
                for policy in CALIBRATION_POLICY_ORDER
            }
            for seed in INITIAL_SEEDS
        }
        plan_artifact_sha256 = digest("1")
        source = self._source_identities()
        return build_initial_model_identity_lock_v2(
            plan_artifact_sha256=plan_artifact_sha256,
            plan_payload_sha256=digest("2"),
            producer_git_commit="f" * 40,
            implementation_manifest=implementation,
            source_identities=source,
            conversion_training=self._conversion_training(
                plan_artifact_sha256, source
            ),
            models=models,
        )

    def test_exact_thirty_model_lock_round_trip(self) -> None:
        lock = self._lock()
        validate_initial_model_identity_lock_v2(lock)
        self.assertEqual(
            sum(len(row) for row in lock["models"].values()),
            30,
        )
        self.assertEqual(
            lock["result_inputs"],
            {
                "calibration_metric_used_for_identity_seal": False,
                "calibration_metric_used_later_by_locked_selection": True,
                "final_test_metric_used_for_identity_seal": False,
                "historical_test_metric_used_for_identity_seal": False,
                "latency_used_for_identity_seal": False,
                "metric_bearing_training_artifacts_read_for_identity_only": True,
                "training_artifact_identity": True,
            },
        )

    def test_checkpoint_router_and_implementation_rotation_fail(self) -> None:
        lock = self._lock()
        for mutate in (
            lambda value: value["models"]["1729"]["fixed_byte_6"][
                "checkpoint"
            ].__setitem__("artifact_sha256", digest("0")),
            lambda value: value["models"]["1729"][
                "entropy_threshold_full"
            ]["auxiliary"].__setitem__("router_state_sha256", digest("0")),
            lambda value: value["calibration_selection_implementation"][
                "sha256_by_path"
            ].__setitem__(
                CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER[0], digest("0")
            ),
        ):
            tampered = deepcopy(lock)
            mutate(tampered)
            with self.assertRaises(ValueError):
                validate_initial_model_identity_lock_v2(tampered)

    def test_resealed_nested_provenance_and_paths_fail(self) -> None:
        for mutate in (
            lambda value: value.__setitem__("source_identities", {}),
            lambda value: value.__setitem__("conversion_training", {}),
            lambda value: value["models"]["1729"]["fixed_byte_6"][
                "checkpoint"
            ].__setitem__("path", "artifacts/rotated.pt"),
        ):
            tampered = deepcopy(self._lock())
            mutate(tampered)
            unsigned = {
                key: value for key, value in tampered.items()
                if key != "lock_sha256"
            }
            tampered["lock_sha256"] = canonical_sha256(unsigned)
            with self.assertRaises(ValueError):
                validate_initial_model_identity_lock_v2(tampered)

    def test_mps_environment_and_plan_frozen_contract_are_required(self) -> None:
        lock = self._lock()
        for mutate in (
            lambda value: value["calibration_selection_implementation"][
                "environment"
            ]["torch_runtime"].__setitem__("mps_available", False),
            lambda value: value["calibration_selection_implementation"][
                "plan_frozen_selection"
            ].__setitem__("decision_ast_sha256", digest("0")),
        ):
            tampered = deepcopy(lock)
            mutate(tampered)
            with self.assertRaises(ValueError):
                validate_initial_model_identity_lock_v2(tampered)

    def test_implementation_manifest_has_exact_unique_existing_paths(self) -> None:
        order = CALIBRATION_SELECTION_IMPLEMENTATION_FILE_ORDER
        self.assertEqual(len(order), len(set(order)))
        self.assertTrue(all(Path(path).is_file() for path in order))
        self.assertIn(
            "scripts/seal_inference_initial_model_identity_v2.py", order
        )
        self.assertIn(
            "src/jamoflow/inference_calibration_replay_v2.py", order
        )
        expected_package = {
            path.as_posix() for path in Path("src/jamoflow").glob("*.py")
        } - {
            "src/jamoflow/inference_actual_runtime_v5.py",
            "src/jamoflow/inference_actual_v5.py",
        }
        self.assertEqual(
            {path for path in order if path.startswith("src/jamoflow/")},
            expected_package,
        )


if __name__ == "__main__":
    unittest.main()
