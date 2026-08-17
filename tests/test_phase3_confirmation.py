from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jamoflow.compute_conversion import CONVERSION_POLICIES
from jamoflow.inference_selection_plan import build_selection_plan_v2
from jamoflow.inference_selection_v2 import (
    CALIBRATION_POLICY_ORDER,
    build_independent_calibration_recomputation_v2,
    build_selection_decision_v2,
    build_selection_lock_v2,
)

from jamoflow.phase3_confirmation import (
    AUTHORIZATION_KIND,
    CONFIRMATION_ONLY_SEEDS,
    INITIAL_SEEDS,
    PRIMARY_POLICIES,
    SELECTED_REFERENCE_AUTHORIZATION_KIND,
    SELECTED_REFERENCE_AUTHORIZATION_KIND_V3,
    confirmation_authorization_record,
    file_sha256,
    load_confirmation_authorization,
    load_run_confirmation_authorization,
    selected_reference_authorization_record,
    selected_reference_authorization_record_v3,
    validate_confirmation_invocations,
    validate_confirmation_request,
    validate_selected_reference_invocation,
    validate_selected_reference_request,
    validate_selected_reference_request_v3,
)


class Phase3ConfirmationAuthorizationTests(unittest.TestCase):
    def _summary(self, source_manifest_sha256: str = "a" * 64) -> dict[str, object]:
        return {
            "summary_git_commit": "commit",
            "source_manifest": {"sha256": source_manifest_sha256},
            "seeds": list(INITIAL_SEEDS),
            "policies": list(PRIMARY_POLICIES),
            "gate_i": {"overall_pass": True},
            "integrity": {"all_integrity_checks_pass": True},
        }

    def test_record_requires_passing_exact_initial_design(self) -> None:
        record = confirmation_authorization_record(
            self._summary(), summary_artifact_sha256="b" * 64
        )
        self.assertEqual(record["authorization_kind"], AUTHORIZATION_KIND)
        self.assertEqual(record["summary_seeds"], list(INITIAL_SEEDS))

        failed = self._summary()
        failed["gate_i"] = {"overall_pass": False}
        with self.assertRaisesRegex(ValueError, "Gate I pass"):
            confirmation_authorization_record(
                failed, summary_artifact_sha256="b" * 64
            )

        wrong_policies = self._summary()
        wrong_policies["policies"] = [*PRIMARY_POLICIES, "spacebyte_spacelike"]
        with self.assertRaisesRegex(ValueError, "exactly F/C/W"):
            confirmation_authorization_record(
                wrong_policies, summary_artifact_sha256="b" * 64
            )

    def test_load_binds_summary_to_preconfirmation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text('{"stage":"initial"}\n', encoding="utf-8")
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(self._summary(file_sha256(manifest_path))),
                encoding="utf-8",
            )
            record = load_confirmation_authorization(
                summary_path,
                expected_source_manifest_path=manifest_path,
            )
            self.assertEqual(
                record["source_manifest_sha256"], file_sha256(manifest_path)
            )
            manifest_path.write_text('{"stage":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pre-confirmation manifest"):
                load_confirmation_authorization(
                    summary_path,
                    expected_source_manifest_path=manifest_path,
                )

    def test_run_authorization_allows_only_exact_recorded_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text('{"stage":"initial"}\n', encoding="utf-8")
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(self._summary(file_sha256(manifest_path))),
                encoding="utf-8",
            )
            authorization = load_run_confirmation_authorization(
                summary_path,
                manifest_path,
                seeds=CONFIRMATION_ONLY_SEEDS,
                policies=PRIMARY_POLICIES,
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "invocations": [
                            {
                                "seeds": list(CONFIRMATION_ONLY_SEEDS),
                                "policies": list(PRIMARY_POLICIES),
                                "authorization": authorization,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_run_confirmation_authorization(
                    summary_path,
                    manifest_path,
                    seeds=CONFIRMATION_ONLY_SEEDS,
                    policies=PRIMARY_POLICIES,
                ),
                authorization,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["invocations"][0]["policies"] = list(PRIMARY_POLICIES[:2])
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no exact authorized resume"):
                load_run_confirmation_authorization(
                    summary_path,
                    manifest_path,
                    seeds=CONFIRMATION_ONLY_SEEDS,
                    policies=PRIMARY_POLICIES,
                )

    def test_request_requires_both_seeds_and_exact_primary_policies(self) -> None:
        validate_confirmation_request(CONFIRMATION_ONLY_SEEDS, PRIMARY_POLICIES)
        with self.assertRaisesRegex(ValueError, "both confirmation seeds"):
            validate_confirmation_request((57721,), PRIMARY_POLICIES)
        with self.assertRaisesRegex(ValueError, "exactly F/C/W"):
            validate_confirmation_request(
                CONFIRMATION_ONLY_SEEDS,
                PRIMARY_POLICIES[:2],
            )

    def test_invocations_reject_unbound_or_rotated_authorization(self) -> None:
        authorization = confirmation_authorization_record(
            self._summary(), summary_artifact_sha256="b" * 64
        )
        manifest = {
            "invocations": [
                {
                    "seeds": list(CONFIRMATION_ONLY_SEEDS),
                    "policies": list(PRIMARY_POLICIES),
                    "authorization": authorization,
                }
            ]
        }
        validate_confirmation_invocations(manifest, authorization)

        manifest["invocations"][0]["authorization"] = {
            **authorization,
            "summary_artifact_sha256": "c" * 64,
        }
        with self.assertRaisesRegex(ValueError, "not bound"):
            validate_confirmation_invocations(manifest, authorization)

    def _selected_reference_inputs(self) -> tuple[dict, dict, dict]:
        values = {
            seed: {policy: 1.5 for policy in CALIBRATION_POLICY_ORDER}
            for seed in INITIAL_SEEDS
        }
        for seed in INITIAL_SEEDS:
            values[seed]["causal_codepoint_grid"] = 1.40
            values[seed]["causal_whitespace_grid_64"] = 1.405
            values[seed]["spacebyte_spacelike"] = 1.399
            for policy in CONVERSION_POLICIES:
                if policy not in values[seed]:
                    values[seed][policy] = 1.5
        decision = build_selection_decision_v2(values)
        plan_artifact_sha256 = "b" * 64
        primary_summary_sha256 = "c" * 64
        plan = build_selection_plan_v2(
            plan_git_commit="a" * 40,
            final_test_manifest_sha256="1" * 64,
            final_test_seal_sha256="2" * 64,
            final_test_payload_sha256="3" * 64,
            phase3_all_initial_summary_sha256="4" * 64,
            phase3_primary_summary_sha256=primary_summary_sha256,
            source_artifact_sha256="6" * 64,
            source_integrity_artifact_sha256="7" * 64,
            calibration_stream_sha256="8" * 64,
            calibration_sequence_count=15_625,
        )
        lock = build_selection_lock_v2(
            decision,
            plan_sha256=plan_artifact_sha256,
            calibration_evidence_manifest_sha256="d" * 64,
            final_test_seal_sha256="e" * 64,
            initial_model_identity_lock_sha256="f" * 64,
            independent_calibration_recomputation=(
                build_independent_calibration_recomputation_v2(
                    values,
                    nll_array_sha256_by_seed_policy={
                        seed: {
                            policy: "1" * 64
                            for policy in CALIBRATION_POLICY_ORDER
                        }
                        for seed in INITIAL_SEEDS
                    },
                    evaluator_git_commit="a" * 40,
                    verification_git_commit="b" * 40,
                    environment_sha256="c" * 64,
                    implementation_manifest_sha256="d" * 64,
                )
            ),
        )
        primary = {
            **self._summary(),
            "seeds": [*INITIAL_SEEDS, *CONFIRMATION_ONLY_SEEDS],
            "confirmation_authorization": {
                "authorization_kind": AUTHORIZATION_KIND
            },
            "gate_j": {"overall_pass": True},
            "ood": {
                "gate_i_ood_guard": {"pass": True},
                "integrity": {"all_integrity_checks_pass": True},
            },
        }
        return lock, plan, primary

    def test_selected_reference_authorization_is_exact_and_non_substitutable(self) -> None:
        lock, plan, primary = self._selected_reference_inputs()
        authorization = selected_reference_authorization_record(
            lock,
            plan,
            primary,
            selection_lock_artifact_sha256="f" * 64,
            selection_plan_artifact_sha256="b" * 64,
            primary_summary_artifact_sha256="c" * 64,
        )
        self.assertEqual(
            authorization["authorization_kind"],
            SELECTED_REFERENCE_AUTHORIZATION_KIND,
        )
        self.assertEqual(authorization["policies"], ["spacebyte_spacelike"])
        self.assertEqual(authorization["required_auxiliary"], "none")
        validate_selected_reference_request(
            CONFIRMATION_ONLY_SEEDS,
            ("spacebyte_spacelike",),
            authorization,
        )
        with self.assertRaisesRegex(ValueError, "differs from the selection lock"):
            validate_selected_reference_request(
                CONFIRMATION_ONLY_SEEDS,
                ("entropy_threshold_full",),
                authorization,
            )

        manifest = {
            "invocations": [
                {
                    "seeds": list(CONFIRMATION_ONLY_SEEDS),
                    "policies": ["spacebyte_spacelike"],
                    "authorization": authorization,
                }
            ]
        }
        validate_selected_reference_invocation(manifest, authorization)
        manifest["invocations"].append(dict(manifest["invocations"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_selected_reference_invocation(manifest, authorization)

    def test_selected_reference_authorization_requires_gate_j_and_plan_lineage(self) -> None:
        lock, plan, primary = self._selected_reference_inputs()
        primary["gate_j"] = {"overall_pass": False}
        with self.assertRaisesRegex(ValueError, "Gate J/OOD"):
            selected_reference_authorization_record(
                lock,
                plan,
                primary,
                selection_lock_artifact_sha256="f" * 64,
                selection_plan_artifact_sha256="b" * 64,
                primary_summary_artifact_sha256="c" * 64,
            )

        lock, plan, primary = self._selected_reference_inputs()
        with self.assertRaisesRegex(ValueError, "lineage differ"):
            selected_reference_authorization_record(
                lock,
                plan,
                primary,
                selection_lock_artifact_sha256="f" * 64,
                selection_plan_artifact_sha256="9" * 64,
                primary_summary_artifact_sha256="c" * 64,
            )

    def test_v3_selected_reference_uses_only_calibration_lock_lineage(self) -> None:
        lock, plan, primary = self._selected_reference_inputs()
        del primary
        authorization = selected_reference_authorization_record_v3(
            lock,
            plan,
            selection_lock_artifact_sha256="f" * 64,
            selection_plan_artifact_sha256="b" * 64,
            calibration_evidence_artifact_sha256="d" * 64,
            final_test_seal_artifact_sha256="e" * 64,
        )
        self.assertEqual(
            authorization["authorization_kind"],
            SELECTED_REFERENCE_AUTHORIZATION_KIND_V3,
        )
        self.assertEqual(
            authorization["result_inputs"],
            {
                "calibration_selection": True,
                "final_test": False,
                "historical_screening_test": False,
                "latency": False,
            },
        )
        validate_selected_reference_request_v3(
            CONFIRMATION_ONLY_SEEDS,
            ("spacebyte_spacelike",),
            authorization,
        )
        with self.assertRaisesRegex(ValueError, "v3 lock"):
            validate_selected_reference_request_v3(
                CONFIRMATION_ONLY_SEEDS,
                ("entropy_threshold_full",),
                authorization,
            )

        authorization["result_inputs"]["historical_screening_test"] = True
        with self.assertRaisesRegex(ValueError, "v3 lock"):
            validate_selected_reference_request_v3(
                CONFIRMATION_ONLY_SEEDS,
                ("spacebyte_spacelike",),
                authorization,
            )

if __name__ == "__main__":
    unittest.main()
