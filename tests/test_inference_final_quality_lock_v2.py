from __future__ import annotations

import hashlib
import math
import unittest
from unittest import mock

import numpy as np

from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    FINAL_EVIDENCE_PATH,
)
from jamoflow.inference_final_quality_evidence_v2 import (
    build_final_quality_evidence_manifest,
    build_final_quality_receipt,
    expected_final_evidence_paths,
)
from jamoflow.inference_final_quality_lock_v2 import (
    build_final_quality_lock_v2,
    validate_final_quality_lock_v2,
)
from tests.test_inference_final_authorization_v2 import digest
from tests.test_inference_final_quality_evidence_v2 import (
    InferenceFinalQualityEvidenceV2Tests,
)


def array_sha256(array: np.ndarray) -> str:
    checksum = hashlib.sha256()
    checksum.update(str(array.dtype).encode("ascii"))
    checksum.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    checksum.update(array.tobytes(order="C"))
    return checksum.hexdigest()


class InferenceFinalQualityLockV2Tests(unittest.TestCase):
    def _fixture(self):
        (
            selection_lock,
            authorization,
            authorization_artifact_sha256,
            _,
            session_plan,
            old_receipts,
            _,
        ) = InferenceFinalQualityEvidenceV2Tests()._evidence()
        arrays = {}
        receipts = []
        receipt_artifacts = []
        for index, old in enumerate(old_receipts):
            values = np.full(62_500, 100.0 + index / 100, dtype=np.float32)
            paths = expected_final_evidence_paths(
                old["artifact_role"],
                old["seed"],
            )
            receipt = build_final_quality_receipt(
                authorization=authorization,
                authorization_artifact_sha256=authorization_artifact_sha256,
                selection_lock=selection_lock,
                session_plan=session_plan,
                unit_index=old["unit_index"],
                artifact_role=old["artifact_role"],
                seed=old["seed"],
                patch_matrix_sha256=old["patch_matrix_sha256"],
                auxiliary_execution=old["auxiliary_execution"],
                nll={
                    "array_sha256": array_sha256(values),
                    "artifact_path": paths["nll"],
                    "artifact_sha256": digest(
                        f"nll/{old['artifact_role']}/{old['seed']}"
                    ),
                    "bpb": math.fsum(float(value) for value in values)
                    / (len(values) * 511 * math.log(2.0)),
                    "count": 62_500,
                    "dtype": "float32",
                    "predicted_bytes": 62_500 * 511,
                },
            )
            receipts.append(receipt)
            receipt_artifacts.append(
                {
                    "path": paths["receipt"],
                    "sha256": digest(
                        f"receipt/{old['artifact_role']}/{old['seed']}"
                    ),
                }
            )
            arrays[receipt["receipt_sha256"]] = values
        evidence = build_final_quality_evidence_manifest(
            authorization=authorization,
            authorization_artifact_sha256=authorization_artifact_sha256,
            selection_lock=selection_lock,
            session_plan=session_plan,
            receipts=receipts,
            receipt_artifacts=receipt_artifacts,
        )
        return (
            selection_lock,
            authorization,
            authorization_artifact_sha256,
            session_plan,
            evidence,
            object(),
            arrays,
        )

    def test_lock_binds_exact_bundles_and_pair_authorizations(self) -> None:
        (
            selection_lock,
            authorization,
            authorization_artifact_sha256,
            session_plan,
            evidence,
            document_map,
            arrays,
        ) = self._fixture()
        gate = {
            "actual_timing_authorized": True,
            "broad_candidate_vs_strongest_reference": None,
            "candidate_vs_matched_efficiency_baseline": {"overall_pass": True},
            "mechanism_candidate_vs_same_rate_codepoint": {"overall_pass": False},
            "overall_pass": False,
            "status": "fail_mechanism_only",
        }
        authorization_artifact = {
            "git_commit": "d" * 40,
            "path": FINAL_AUTHORIZATION_PATH,
            "sha256": authorization_artifact_sha256,
        }
        evidence_artifact = {
            "git_commit": "e" * 40,
            "path": FINAL_EVIDENCE_PATH,
            "sha256": digest("evidence-artifact"),
        }
        with (
            mock.patch(
                "jamoflow.inference_final_quality_lock_v2.final_quality_gate_v2",
                return_value=gate,
            ),
            mock.patch(
                "jamoflow.inference_final_quality_lock_v2._validate_document_map",
                return_value={"coverage": "sealed"},
            ),
        ):
            quality_lock = build_final_quality_lock_v2(
                authorization=authorization,
                authorization_artifact=authorization_artifact,
                selection_lock=selection_lock,
                session_plan=session_plan,
                evidence=evidence,
                evidence_artifact=evidence_artifact,
                quality_lock_base_git_commit="f" * 40,
                document_window_map=document_map,
                arrays_by_receipt_sha256=arrays,
            )
            validate_final_quality_lock_v2(
                quality_lock,
                authorization=authorization,
                selection_lock=selection_lock,
                session_plan=session_plan,
                evidence=evidence,
                document_window_map=document_map,
                arrays_by_receipt_sha256=arrays,
            )
        self.assertTrue(quality_lock["primary_publication_timing_authorized"])
        self.assertEqual(
            quality_lock["primary_timing_authorization_key"],
            "candidate_vs_matched_efficiency_baseline",
        )
        self.assertEqual(quality_lock["status"], "pass_matched_quality_only")
        self.assertTrue(
            quality_lock["independent_nll_recomputation"]["pass"]
        )
        self.assertEqual(
            quality_lock["independent_nll_recomputation"]["comparison"],
            "bitwise_equal_float32_array_sha256",
        )
        self.assertTrue(
            quality_lock["independent_nll_recomputation"][
                "was_predeclared_before_first_final_loss"
            ]
        )
        matched = quality_lock["timing_authorizations"][
            "candidate_vs_matched_efficiency_baseline"
        ]
        self.assertTrue(matched["authorized"])
        self.assertFalse(
            quality_lock["timing_authorizations"][
                "candidate_vs_same_rate_codepoint_control"
            ]["authorized"]
        )
        self.assertNotEqual(
            matched["left_model_identity_sha256"],
            matched["right_model_identity_sha256"],
        )

    def test_wrong_dtype_or_receipt_order_fails_before_gate(self) -> None:
        (
            selection_lock,
            authorization,
            authorization_artifact_sha256,
            session_plan,
            evidence,
            document_map,
            arrays,
        ) = self._fixture()
        first = next(iter(arrays))
        arrays[first] = arrays[first].astype(np.float64)
        with mock.patch(
            "jamoflow.inference_final_quality_lock_v2._validate_document_map",
            return_value={"coverage": "sealed"},
        ):
            with self.assertRaisesRegex(ValueError, "NLL array"):
                build_final_quality_lock_v2(
                    authorization=authorization,
                    authorization_artifact={
                        "git_commit": "d" * 40,
                        "path": FINAL_AUTHORIZATION_PATH,
                        "sha256": authorization_artifact_sha256,
                    },
                    selection_lock=selection_lock,
                    session_plan=session_plan,
                    evidence=evidence,
                    evidence_artifact={
                        "git_commit": "e" * 40,
                        "path": FINAL_EVIDENCE_PATH,
                        "sha256": digest("evidence-artifact"),
                    },
                    quality_lock_base_git_commit="f" * 40,
                    document_window_map=document_map,
                    arrays_by_receipt_sha256=arrays,
                )


if __name__ == "__main__":
    unittest.main()
