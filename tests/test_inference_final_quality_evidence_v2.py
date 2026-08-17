from __future__ import annotations

import copy
import unittest

from jamoflow.inference_final_quality_evidence_v2 import (
    FINAL_EVALUATION_BATCH_SIZE,
    authorized_unit_order,
    build_final_quality_evidence_manifest,
    build_final_quality_receipt,
    build_final_quality_session_plan,
    expected_final_evidence_paths,
    validate_final_quality_evidence_manifest,
    validate_final_quality_receipt,
    validate_final_quality_session_plan,
)
from tests.test_inference_final_authorization_v2 import (
    InferenceFinalAuthorizationV2Tests,
    digest,
)


class InferenceFinalQualityEvidenceV2Tests(unittest.TestCase):
    def _evidence(
        self,
    ) -> tuple[dict, dict, str, dict, dict, list[dict], list[dict]]:
        lock, authorization = InferenceFinalAuthorizationV2Tests()._authorization()
        authorization_artifact = digest("authorization-artifact")
        final_context = {
            "boundaries_sha256": digest("final-boundaries"),
            "document_assignment_sha256": digest("document-assignments"),
            "document_layout_sha256": digest("document-layout"),
            "eligible_sequence_count": 61_000,
            "inputs_sha256": digest("final-inputs"),
            "stream_bytes": 32_000_000,
            "stream_sha256": authorization["final_test"][
                "evaluation_stream_sha256"
            ],
        }
        session_plan = build_final_quality_session_plan(
            authorization=authorization,
            authorization_artifact_sha256=authorization_artifact,
            authorization_git_commit="b" * 40,
            selection_lock=lock,
            evaluator_git_commit="c" * 40,
            runtime={
                "batch_size": 64,
                "device": "mps",
                "mps_available": True,
                "numpy": "2.5.2",
                "python": "3.13.11",
                "torch": "2.13.0",
                "transformers": "5.14.1",
            },
            final_context=final_context,
        )
        receipts = []
        receipt_artifacts = []
        for index, role, seed in authorized_unit_order(authorization):
            paths = expected_final_evidence_paths(role, seed)
            receipt = build_final_quality_receipt(
                authorization=authorization,
                authorization_artifact_sha256=authorization_artifact,
                selection_lock=lock,
                session_plan=session_plan,
                unit_index=index,
                artifact_role=role,
                seed=seed,
                patch_matrix_sha256=digest(f"final-matrix/{role}/{seed}"),
                auxiliary_execution={"kind": "none"},
                nll={
                    "array_sha256": digest(f"final-array/{role}/{seed}"),
                    "artifact_path": paths["nll"],
                    "artifact_sha256": digest(f"final-nll/{role}/{seed}"),
                    "bpb": 1.5 + index / 1000,
                    "count": 62_500,
                    "dtype": "float32",
                    "predicted_bytes": 62_500 * 511,
                },
            )
            receipts.append(receipt)
            receipt_artifacts.append(
                {
                    "path": paths["receipt"],
                    "sha256": digest(f"final-receipt/{role}/{seed}"),
                }
            )
        return (
            lock,
            authorization,
            authorization_artifact,
            final_context,
            session_plan,
            receipts,
            receipt_artifacts,
        )

    def test_evidence_round_trip_binds_exact_authorized_units(self) -> None:
        (
            lock,
            authorization,
            artifact,
            _,
            session_plan,
            receipts,
            receipt_artifacts,
        ) = self._evidence()
        validate_final_quality_session_plan(
            session_plan,
            authorization=authorization,
            selection_lock=lock,
        )
        manifest = build_final_quality_evidence_manifest(
            authorization=authorization,
            authorization_artifact_sha256=artifact,
            selection_lock=lock,
            session_plan=session_plan,
            receipts=receipts,
            receipt_artifacts=receipt_artifacts,
        )
        validate_final_quality_evidence_manifest(
            manifest,
            authorization=authorization,
            selection_lock=lock,
            session_plan=session_plan,
        )
        self.assertEqual(len(manifest["receipts"]), 15)
        self.assertEqual(FINAL_EVALUATION_BATCH_SIZE, 64)

    def test_receipt_rejects_wrong_session_stream_and_structural_router(self) -> None:
        (
            lock,
            authorization,
            artifact,
            context,
            session_plan,
            receipts,
            _,
        ) = self._evidence()
        receipt = receipts[0]
        validate_final_quality_receipt(
            receipt,
            authorization=authorization,
            selection_lock=lock,
            session_plan=session_plan,
        )
        with self.assertRaisesRegex(ValueError, "unit/session"):
            build_final_quality_receipt(
                authorization=authorization,
                authorization_artifact_sha256=digest("alternate-authorization"),
                selection_lock=lock,
                session_plan=session_plan,
                unit_index=receipt["unit_index"],
                artifact_role=receipt["artifact_role"],
                seed=receipt["seed"],
                patch_matrix_sha256=receipt["patch_matrix_sha256"],
                auxiliary_execution=receipt["auxiliary_execution"],
                nll=receipt["nll"],
            )
        wrong_context = dict(context)
        wrong_context["stream_sha256"] = digest("historical-screening-stream")
        with self.assertRaisesRegex(ValueError, "stream/document"):
            build_final_quality_session_plan(
                authorization=authorization,
                authorization_artifact_sha256=artifact,
                authorization_git_commit="b" * 40,
                selection_lock=lock,
                evaluator_git_commit="c" * 40,
                runtime=session_plan["runtime"],
                final_context=wrong_context,
            )
        with self.assertRaisesRegex(ValueError, "cannot execute a router"):
            build_final_quality_receipt(
                authorization=authorization,
                authorization_artifact_sha256=artifact,
                selection_lock=lock,
                session_plan=session_plan,
                unit_index=receipt["unit_index"],
                artifact_role=receipt["artifact_role"],
                seed=receipt["seed"],
                patch_matrix_sha256=receipt["patch_matrix_sha256"],
                auxiliary_execution={"kind": "entropy_router"},
                nll=receipt["nll"],
            )

    def test_manifest_rejects_missing_reordered_or_reused_receipts(self) -> None:
        (
            lock,
            authorization,
            artifact,
            _,
            session_plan,
            receipts,
            receipt_artifacts,
        ) = self._evidence()
        for malformed in (
            receipts[:-1],
            [receipts[1], receipts[0], *receipts[2:]],
        ):
            with self.assertRaises(ValueError):
                build_final_quality_evidence_manifest(
                    authorization=authorization,
                    authorization_artifact_sha256=artifact,
                    selection_lock=lock,
                    session_plan=session_plan,
                    receipts=malformed,
                    receipt_artifacts=receipt_artifacts[: len(malformed)],
                )
        reused = copy.deepcopy(receipts)
        reused[1]["nll"]["artifact_sha256"] = reused[0]["nll"][
            "artifact_sha256"
        ]
        reused[1]["receipt_sha256"] = digest("locally-rehashed")
        with self.assertRaises(ValueError):
            build_final_quality_evidence_manifest(
                authorization=authorization,
                authorization_artifact_sha256=artifact,
                selection_lock=lock,
                session_plan=session_plan,
                receipts=reused,
                receipt_artifacts=receipt_artifacts,
            )


if __name__ == "__main__":
    unittest.main()
