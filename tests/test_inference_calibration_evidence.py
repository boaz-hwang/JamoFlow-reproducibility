from __future__ import annotations

from copy import deepcopy
import unittest

from jamoflow.inference_calibration_evidence import (
    EVIDENCE_PROTOCOL,
    build_calibration_evidence_manifest,
    calibration_bpb_matrix,
    canonical_sha256,
    expected_evidence_paths,
    seal_calibration_receipt,
    validate_calibration_evidence_manifest,
    validate_calibration_receipt,
)
from jamoflow.inference_selection_plan import build_selection_plan_v2
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    INITIAL_SEEDS,
)
from jamoflow.phase3 import PHASE3_MODEL_SPEC


def plan_fixture() -> dict:
    return build_selection_plan_v2(
        plan_git_commit="a" * 40,
        final_test_manifest_sha256="1" * 64,
        final_test_seal_sha256="2" * 64,
        final_test_payload_sha256="3" * 64,
        phase3_all_initial_summary_sha256="4" * 64,
        phase3_primary_summary_sha256="5" * 64,
        source_artifact_sha256="6" * 64,
        source_integrity_artifact_sha256="7" * 64,
        calibration_stream_sha256="8" * 64,
        calibration_sequence_count=8,
    )


def receipt_fixture(seed: int, policy: str, plan: dict) -> dict:
    paths = expected_evidence_paths(seed, policy)
    is_entropy = policy in {
        "entropy_threshold_full",
        "entropy_threshold_codepoint",
    }
    auxiliary = {"kind": "none"}
    if is_entropy:
        auxiliary = {
            "cache_artifact_sha256": "9" * 64,
            "cache_path": f"artifacts/phase3/seed-{seed}/threshold-patches.npz",
            "candidate_mask": (
                "none" if policy == "entropy_threshold_full" else "codepoint"
            ),
            "diagnostics_artifact_sha256": "a" * 64,
            "diagnostics_path": (
                f"runs/phase3/seed-{seed}/threshold-patch-diagnostics.json"
            ),
            "kind": "entropy_router",
            "maximum_patch_length": 24,
            "router_checkpoint_artifact_sha256": "b" * 64,
            "router_checkpoint_path": f"artifacts/phase3/seed-{seed}/router.pt",
            "router_report_artifact_sha256": "c" * 64,
            "router_report_path": f"runs/phase3/seed-{seed}/router.json",
            "router_scores_sha256": "e" * 64,
            "router_state_sha256": "d" * 64,
            "threshold_nats": 1.25,
        }
    patch_count = 86 if policy in CALIBRATION_POLICY_ORDER[:6] else int(
        policy.rsplit("_", 1)[1]
    )
    spec = PHASE3_MODEL_SPEC
    if patch_count != 86:
        from jamoflow.compute_conversion import conversion_model_spec

        spec = conversion_model_spec(patch_count)
    payload = {
        "auxiliary": auxiliary,
        "calibration": {
            "boundaries_sha256": "e" * 64,
            "bpb": 1.5,
            "count": 8,
            "dtype": "float32",
            "inputs_sha256": "f" * 64,
            "matrix_sha256": "0" * 64,
            "nll_array_sha256": "1" * 64,
            "nll_artifact_path": paths["nll"],
            "nll_artifact_sha256": "2" * 64,
            "predicted_bytes": 8 * 511,
            "report_bpb": 1.5,
            "stream_sha256": "8" * 64,
        },
        "checkpoint": {
            "artifact_sha256": "3" * 64,
            "path": paths["checkpoint"],
            "state_sha256": "4" * 64,
        },
        "complete": True,
        "device": "mps",
        "evaluator_git_commit": "b" * 40,
        "evaluator_protocol": EVIDENCE_PROTOCOL,
        "initial_model_identity_lock_sha256": "f" * 64,
        "kind": "phase3_calibration_receipt_v2",
        "model": {
            "global_max_position_embeddings": 1_032,
            "parameters": 19_596_096,
            "spec_sha256": canonical_sha256(spec.to_dict()),
        },
        "model_family": (
            "phase3" if policy in CALIBRATION_POLICY_ORDER[:6] else "compute_conversion"
        ),
        "patch_count": patch_count,
        "plan_artifact_sha256": "c" * 64,
        "policy": policy,
        "schema_version": 2,
        "seed": seed,
        "training_report": {
            "artifact_sha256": "5" * 64,
            "path": paths["training_report"],
        },
    }
    return seal_calibration_receipt(payload)


class InferenceCalibrationEvidenceTests(unittest.TestCase):
    def test_receipt_and_complete_manifest_round_trip(self) -> None:
        plan = plan_fixture()
        receipts = {
            seed: {
                policy: receipt_fixture(seed, policy, plan)
                for policy in CALIBRATION_POLICY_ORDER
            }
            for seed in INITIAL_SEEDS
        }
        for row in receipts.values():
            for receipt in row.values():
                validate_calibration_receipt(receipt, plan=plan)
        manifest = build_calibration_evidence_manifest(
            plan=plan,
            plan_artifact_sha256="c" * 64,
            evaluator_git_commit="b" * 40,
            initial_model_identity_lock_sha256="f" * 64,
            receipts=receipts,
        )
        validate_calibration_evidence_manifest(manifest, plan=plan)
        matrix = calibration_bpb_matrix(manifest, plan=plan)
        self.assertEqual(matrix[1729]["fixed_byte_6"], 1.5)

    def test_receipt_rejects_scalar_auxiliary_and_path_tampering(self) -> None:
        plan = plan_fixture()
        receipt = receipt_fixture(1729, "entropy_threshold_full", plan)
        for mutate in (
            lambda value: value["calibration"].__setitem__("bpb", 1.6),
            lambda value: value["auxiliary"].__setitem__("kind", "none"),
            lambda value: value["checkpoint"].__setitem__("path", "other.pt"),
        ):
            tampered = deepcopy(receipt)
            mutate(tampered)
            tampered["receipt_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in tampered.items()
                    if key != "receipt_sha256"
                }
            )
            with self.assertRaises(ValueError):
                validate_calibration_receipt(tampered, plan=plan)

    def test_manifest_rejects_missing_policy_and_receipt_rotation(self) -> None:
        plan = plan_fixture()
        receipts = {
            seed: {
                policy: receipt_fixture(seed, policy, plan)
                for policy in CALIBRATION_POLICY_ORDER
            }
            for seed in INITIAL_SEEDS
        }
        del receipts[1729][CALIBRATION_POLICY_ORDER[-1]]
        with self.assertRaisesRegex(ValueError, "policy set"):
            build_calibration_evidence_manifest(
                plan=plan,
                plan_artifact_sha256="c" * 64,
                evaluator_git_commit="b" * 40,
                initial_model_identity_lock_sha256="f" * 64,
                receipts=receipts,
            )


if __name__ == "__main__":
    unittest.main()
