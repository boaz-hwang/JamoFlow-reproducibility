from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from jamoflow.inference_final_quality_v2 import resolve_final_evaluation_roles
from jamoflow.compute_conversion import conversion_model_spec
from jamoflow.inference_final_authorization_v2 import canonical_sha256
from jamoflow.phase3 import PHASE3_OPTIMIZATION_SPEC
from tests.test_inference_final_authorization_v2 import (
    model_fixture,
    selection_lock_fixture,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "seal_inference_post_confirmation_authorization_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "seal_inference_post_confirmation_authorization_v2",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SealInferencePostConfirmationAuthorizationV2Tests(unittest.TestCase):
    def test_confirmation_evidence_requires_strict_later_commit(self) -> None:
        commit = "a" * 40
        with mock.patch.object(MODULE, "_require_ancestor") as ancestor:
            with self.assertRaisesRegex(ValueError, "not strict"):
                MODULE._require_strict_ancestor(commit, commit, "evaluator -> evidence")
        ancestor.assert_called_once_with(commit, commit, "evaluator -> evidence")

    def test_confirmation_artifact_history_requires_one_commit(self) -> None:
        path = Path("results/confirmation.json")
        commit = "a" * 40
        with mock.patch.object(MODULE, "_git_path_history", return_value=(commit,)):
            MODULE._require_single_publication_history(path, commit)
        with (
            mock.patch.object(
                MODULE, "_git_path_history", return_value=(commit, "b" * 40)
            ),
            self.assertRaisesRegex(ValueError, "exactly once"),
        ):
            MODULE._require_single_publication_history(path, commit)

    def test_builder_has_one_fixed_output_and_no_metric_input(self) -> None:
        self.assertEqual(
            MODULE.OUTPUT_PATH.as_posix(),
            "results/phase3-inference-final-v2/post-confirmation-authorization.json",
        )
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        strings = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        forbidden = (
            "-test-nll",
            "historical_test_pass",
            "latency_ms",
            "timing.json",
            "final_bpb",
        )
        for value in strings:
            self.assertFalse(any(token in value for token in forbidden))

    def test_model_assembly_requires_all_locked_roles_and_five_seeds(self) -> None:
        lock = selection_lock_fixture(broad_futile=False)
        roles = resolve_final_evaluation_roles(lock)
        identities = {
            row["policy"]: model_fixture(row) for row in roles["unique_models"]
        }
        initial = {"receipts": {}}
        for seed in (1729, 2718, 31415):
            initial["receipts"][str(seed)] = {
                row["policy"]: {
                    "model_family": row["model_family"],
                    "patch_count": row["patch_count"],
                    "policy": row["policy"],
                }
                for row in roles["unique_models"]
            }
        confirmation = {"receipts": {}}
        for seed in (57721, 65537):
            confirmation["receipts"][str(seed)] = {
                row["artifact_role"]: {
                    "descriptor": {
                        key: row[key]
                        for key in (
                            "model_family",
                            "patch_count",
                            "policy",
                            "requires_entropy_router",
                            "runtime_policy",
                        )
                    }
                }
                for row in roles["unique_models"]
            }

        def seed_evidence(
            *,
            receipt,
            descriptor,
            seed,
            initial,
            selection_lock,
            selection_lock_artifact_sha256,
            expected_selected_reference_authorization,
            selection_lock_git_commit,
            confirmation_evaluator_git_commit,
            prospective_implementation,
            historical_phase3_anchor,
        ):
            del (
                receipt,
                initial,
                selection_lock,
                selection_lock_artifact_sha256,
                expected_selected_reference_authorization,
                selection_lock_git_commit,
                confirmation_evaluator_git_commit,
                prospective_implementation,
                historical_phase3_anchor,
            )
            return identities[descriptor["policy"]]["seeds"][str(seed)]

        with mock.patch.object(
            MODULE,
            "_seed_evidence",
            side_effect=seed_evidence,
        ):
            models = MODULE._build_models(
                selection_lock=lock,
                initial_evidence=initial,
                confirmation_evidence=confirmation,
                selection_lock_artifact_sha256="a" * 64,
                expected_selected_reference_authorization=None,
                selection_lock_git_commit="1" * 40,
                confirmation_evaluator_git_commit="2" * 40,
                prospective_implementation={},
                historical_phase3_anchor={},
            )
        self.assertEqual(
            [model["artifact_role"] for model in models],
            [model["artifact_role"] for model in roles["unique_models"]],
        )
        self.assertTrue(
            all(
                tuple(model["seed_order"])
                == (1729, 2718, 31415, 57721, 65537)
                for model in models
            )
        )

        del confirmation["receipts"]["57721"][
            roles["unique_models"][-1]["artifact_role"]
        ]
        with mock.patch.object(
            MODULE,
            "_seed_evidence",
            side_effect=seed_evidence,
        ):
            with self.assertRaises(KeyError):
                MODULE._build_models(
                    selection_lock=lock,
                    initial_evidence=initial,
                    confirmation_evidence=confirmation,
                    selection_lock_artifact_sha256="a" * 64,
                    expected_selected_reference_authorization=None,
                    selection_lock_git_commit="1" * 40,
                    confirmation_evaluator_git_commit="2" * 40,
                    prospective_implementation={},
                    historical_phase3_anchor={},
                )

    def test_broad_futility_keeps_narrow_authorization_path_open(self) -> None:
        futile = selection_lock_fixture(
            broad_futile=True,
            broad_policy="spacebyte_spacelike",
        )
        self.assertEqual(
            futile["decision"]["reference"]["policy"],
            "spacebyte_spacelike",
        )
        self.assertEqual(
            futile["decision"]["broad_reference_evaluation_status"],
            "not_authorized_calibration_futility",
        )
        with mock.patch.object(
            MODULE, "selected_reference_authorization_record_v3"
        ) as builder:
            result = MODULE._selected_reference_authorization_if_required(
                selection_lock=futile,
                plan={},
                identities={},
            )
        self.assertIsNone(result)
        builder.assert_not_called()

        eligible = selection_lock_fixture(
            broad_futile=False,
            broad_policy="spacebyte_spacelike",
        )
        identities = {
            name: {"sha256": character * 64}
            for name, character in (
                ("selection_lock", "1"),
                ("selection_plan", "2"),
                ("calibration_evidence", "3"),
                ("final_seal", "4"),
            )
        }
        with mock.patch.object(
            MODULE,
            "selected_reference_authorization_record_v3",
            return_value={"authorized": True},
        ) as builder:
            result = MODULE._selected_reference_authorization_if_required(
                selection_lock=eligible,
                plan={"kind": "plan"},
                identities=identities,
            )
        self.assertEqual(result, {"authorized": True})
        builder.assert_called_once()

    def test_post_authorization_replays_every_confirmation_seed_and_role(self) -> None:
        lock = selection_lock_fixture(broad_futile=True)
        models = tuple(
            {
                "artifact_role": row["artifact_role"],
                "descriptor": {
                    key: row[key]
                    for key in (
                        "model_family",
                        "patch_count",
                        "policy",
                        "requires_entropy_router",
                        "runtime_policy",
                    )
                },
            }
            for row in resolve_final_evaluation_roles(lock)["unique_models"]
        )
        inputs = np.zeros((2, 512), dtype=np.uint8)
        boundaries = np.zeros((2, 512), dtype=bool)
        matrices = {
            model["descriptor"]["policy"]: np.zeros((2, 2), dtype=np.int16)
            for model in models
        }
        evidence = {"receipts": {}}
        for seed in (57721, 65537):
            evidence["receipts"][str(seed)] = {}
            for model in models:
                role = model["artifact_role"]
                evidence["receipts"][str(seed)][role] = {
                    "calibration": {"matrix_sha256": "1" * 64},
                    "receipt_sha256": "2" * 64,
                }

        def replay(*, descriptor, seed, **kwargs):
            del descriptor, kwargs
            return {
                "checkpoint_state_sha256": f"{seed:064x}"[-64:],
                "nll_array_sha256": "3" * 64,
            }

        with (
            mock.patch.object(MODULE, "resolve_device", return_value="mps"),
            mock.patch.object(
                MODULE,
                "load_confirmation_calibration_context",
                return_value=(b"stream", inputs, boundaries, matrices),
            ),
            mock.patch.object(
                MODULE, "required_confirmation_models", return_value=models
            ),
            mock.patch.object(
                MODULE,
                "confirmation_entropy_matrices_and_auxiliary",
                return_value=({}, {}),
            ),
            mock.patch.object(
                MODULE, "replay_confirmation_unit", side_effect=replay
            ) as forward,
            mock.patch.object(
                MODULE, "validate_confirmation_replay_against_receipt"
            ) as validate,
            mock.patch.object(
                MODULE, "_validate_stored_confirmation_nll"
            ) as validate_stored,
            mock.patch.object(MODULE.torch.mps, "empty_cache"),
        ):
            result = MODULE._independent_confirmation_recomputation(
                plan={},
                selection_lock=lock,
                confirmation_evidence=evidence,
                verification_git_commit="a" * 40,
            )
        expected = 2 * len(models)
        self.assertEqual(forward.call_count, expected)
        self.assertEqual(validate.call_count, expected)
        self.assertEqual(validate_stored.call_count, expected)
        self.assertEqual(result["receipt_count"], expected)

    def test_post_authorization_rejects_missing_or_rotated_stored_nll(self) -> None:
        losses = np.asarray([1.0, 2.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confirmation-nll.npz"
            with path.open("xb") as output:
                np.savez_compressed(output, sequence_nll_nats=losses)
            receipt = {
                "calibration": {
                    "nll_array_sha256": MODULE._array_sha256(losses),
                    "nll_artifact_path": str(path),
                    "nll_artifact_sha256": MODULE.hash_file(path),
                }
            }
            MODULE._validate_stored_confirmation_nll(
                receipt=receipt,
                replay={"losses": losses.copy()},
                seed=57721,
                role="candidate",
            )
            rotated = losses.copy()
            rotated[0] = np.float32(3.0)
            with self.assertRaisesRegex(ValueError, "independent replay"):
                MODULE._validate_stored_confirmation_nll(
                    receipt=receipt,
                    replay={"losses": rotated},
                    seed=57721,
                    role="candidate",
                )
            path.unlink()
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                MODULE._validate_stored_confirmation_nll(
                    receipt=receipt,
                    replay={"losses": losses},
                    seed=57721,
                    role="candidate",
                )

    def test_conversion_confirmation_training_must_bind_exact_selection_lock(self) -> None:
        lock = selection_lock_fixture()
        descriptor = resolve_final_evaluation_roles(lock)["logical_roles"][
            "candidate"
        ]
        binding_payload = {
            "device": "mps",
            "git_commit": "a" * 40,
            "git_worktree_clean_at_start": True,
            "policies": lock["decision"]["confirmation_plan"][
                "compute_conversion"
            ]["policies"],
            "primary_summary_sha256": "b" * 64,
            "schema_version": 1,
            "seeds": [57721, 65537],
            "selection_plan_sha256": lock["plan_sha256"],
            "selection_summary_sha256": "c" * 64,
            "stage": "confirmation",
        }
        binding = {
            **binding_payload,
            "identity_sha256": canonical_sha256(binding_payload),
        }
        report = {
            "evidence_binding": binding,
            "initialization_sha256": "d" * 64,
            "model_spec": conversion_model_spec(64).to_dict(),
            "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
            "parameters": 19_596_096,
            "patch_matrix_sha256": {"train": "e" * 64},
            "policy": descriptor["policy"],
            "seed": 57721,
            "trained_state_sha256": "f" * 64,
            "training": {
                "examples": 250_000,
                "predicted_bytes": 127_750_000,
                "steps": 7_813,
            },
            "training_order_sha256": "0" * 64,
        }
        manifest = {
            "global_max_position_embeddings": 1_032,
            "invocations": [
                {
                    "evidence_binding": binding,
                    "git_commit": "a" * 40,
                    "policies": binding["policies"],
                    "seeds": binding["seeds"],
                    "stage": "confirmation",
                }
            ],
            "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        }
        source = {
            "source_artifact_sha256": "1" * 64,
            "source_integrity_artifact_sha256": "2" * 64,
            "train_stream_sha256": "3" * 64,
        }
        with (
            mock.patch.object(MODULE, "_read_json", return_value=report),
            mock.patch.object(
                MODULE,
                "_run_manifest_context",
                return_value=(Path("runs/manifest.json"), manifest, source),
            ),
            mock.patch.object(MODULE, "hash_file", return_value="4" * 64),
            mock.patch.object(MODULE, "_verify_prospective_execution_commit"),
        ):
            identity = MODULE._training_identity(
                descriptor=descriptor,
                seed=57721,
                report_path=Path("report.json"),
                checkpoint_state_sha256="f" * 64,
                selection_lock=lock,
                selection_lock_artifact_sha256="c" * 64,
                expected_selected_reference_authorization=None,
                selection_lock_git_commit="1" * 40,
                confirmation_evaluator_git_commit="2" * 40,
                prospective_implementation={},
                historical_phase3_anchor={},
            )
            self.assertEqual(identity["train_stream_sha256"], "3" * 64)

            binding["selection_summary_sha256"] = "9" * 64
            with self.assertRaisesRegex(ValueError, "authorization differs"):
                MODULE._training_identity(
                    descriptor=descriptor,
                    seed=57721,
                    report_path=Path("report.json"),
                    checkpoint_state_sha256="f" * 64,
                    selection_lock=lock,
                    selection_lock_artifact_sha256="c" * 64,
                    expected_selected_reference_authorization=None,
                    selection_lock_git_commit="1" * 40,
                    confirmation_evaluator_git_commit="2" * 40,
                    prospective_implementation={},
                    historical_phase3_anchor={},
                )

    def test_any_conflicting_confirmation_invocation_fails(self) -> None:
        lock = selection_lock_fixture()
        descriptor = resolve_final_evaluation_roles(lock)["logical_roles"][
            "candidate"
        ]
        binding_payload = {
            "device": "mps",
            "git_commit": "a" * 40,
            "git_worktree_clean_at_start": True,
            "policies": lock["decision"]["confirmation_plan"][
                "compute_conversion"
            ]["policies"],
            "primary_summary_sha256": "b" * 64,
            "schema_version": 1,
            "seeds": [57721, 65537],
            "selection_plan_sha256": lock["plan_sha256"],
            "selection_summary_sha256": "c" * 64,
            "stage": "confirmation",
        }
        binding = {
            **binding_payload,
            "identity_sha256": canonical_sha256(binding_payload),
        }
        report = {
            "evidence_binding": binding,
            "initialization_sha256": "d" * 64,
            "model_spec": conversion_model_spec(64).to_dict(),
            "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
            "parameters": 19_596_096,
            "patch_matrix_sha256": {"train": "e" * 64},
            "policy": descriptor["policy"],
            "seed": 57721,
            "trained_state_sha256": "f" * 64,
            "training": {
                "examples": 250_000,
                "predicted_bytes": 127_750_000,
                "steps": 7_813,
            },
            "training_order_sha256": "0" * 64,
        }
        authorized = {
            "evidence_binding": binding,
            "git_commit": "a" * 40,
            "policies": binding["policies"],
            "seeds": binding["seeds"],
            "stage": "confirmation",
        }
        conflicting = {
            **authorized,
            "policies": ["causal_codepoint_grid_72"],
        }
        manifest = {
            "global_max_position_embeddings": 1_032,
            "invocations": [authorized, conflicting],
            "optimization_spec": PHASE3_OPTIMIZATION_SPEC.to_dict(),
        }
        source = {
            "source_artifact_sha256": "1" * 64,
            "source_integrity_artifact_sha256": "2" * 64,
            "train_stream_sha256": "3" * 64,
        }
        with (
            mock.patch.object(MODULE, "_read_json", return_value=report),
            mock.patch.object(
                MODULE,
                "_run_manifest_context",
                return_value=(Path("runs/manifest.json"), manifest, source),
            ),
            mock.patch.object(MODULE, "hash_file", return_value="4" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "non-conflicting invocation"):
                MODULE._training_identity(
                    descriptor=descriptor,
                    seed=57721,
                    report_path=Path("report.json"),
                    checkpoint_state_sha256="f" * 64,
                    selection_lock=lock,
                    selection_lock_artifact_sha256="c" * 64,
                    expected_selected_reference_authorization=None,
                    selection_lock_git_commit="1" * 40,
                    confirmation_evaluator_git_commit="2" * 40,
                    prospective_implementation={},
                    historical_phase3_anchor={},
                )


if __name__ == "__main__":
    unittest.main()
