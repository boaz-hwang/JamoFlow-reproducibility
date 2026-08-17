import gc
from dataclasses import replace
import importlib.util
import unittest

from jamoflow.neural_model import build_main_model, build_router, parameter_count
from jamoflow.publication_reference import PUBLICATION_AUXILIARY_ENTROPY_ROUTER
from jamoflow.publication_scale import (
    CORE_MODEL_FAMILIES,
    CORE_MODEL_RUNS,
    CampaignScaleFeasibility,
    FamilyScaleFeasibility,
    PUBLICATION_EXPECTED_PARAMETERS,
    PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS,
    PUBLICATION_FAMILY_EXPECTED_PARAMETERS,
    PUBLICATION_PROJECTED_TRAIN_STEPS,
    PUBLICATION_ROUTER_EXPECTED_PARAMETERS,
    PUBLICATION_SEQUENCE_LENGTH,
    PUBLICATION_BATCH_SIZE,
    PUBLICATION_SCALE_ORDER,
    PUBLICATION_SCALE_SPECS,
    PUBLICATION_TRAIN_BYTES,
    ScaleFeasibility,
    publication_model_spec,
    select_largest_campaign_feasible_scale,
    select_largest_feasible_scale,
)
from tests.publication_reference_support import (
    content_hash,
    make_reference_descriptor,
)


HAS_RESEARCH_DEPS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_RESEARCH_DEPS, "optional neural research dependencies")
class PublicationScaleModelTests(unittest.TestCase):
    def test_candidate_graph_parameter_counts_are_exact(self) -> None:
        for target in PUBLICATION_SCALE_ORDER:
            with self.subTest(target=target):
                model = build_main_model(
                    PUBLICATION_SCALE_SPECS[target],
                    seed=1729,
                    global_max_position_embeddings=1032,
                )
                self.assertEqual(
                    parameter_count(model),
                    PUBLICATION_EXPECTED_PARAMETERS[target],
                )
                del model
                gc.collect()

    def test_entropy_router_parameter_counts_are_exact(self) -> None:
        for target in PUBLICATION_SCALE_ORDER:
            with self.subTest(target=target):
                router = build_router(PUBLICATION_SCALE_SPECS[target], seed=1729)
                self.assertEqual(
                    parameter_count(router),
                    PUBLICATION_ROUTER_EXPECTED_PARAMETERS[target],
                )
                del router
                gc.collect()

    def test_selected_patch_rate_changes_no_other_geometry(self) -> None:
        for target in PUBLICATION_SCALE_ORDER:
            baseline = PUBLICATION_SCALE_SPECS[target].to_dict()
            selected = publication_model_spec(target, 64).to_dict()
            selected["patch_count"] = baseline["patch_count"]
            self.assertEqual(selected, baseline)


class PublicationScaleSelectionTests(unittest.TestCase):
    def _result(
        self,
        target: int,
        *,
        memory_fraction: float = 0.5,
        projected_hours: float = 5.0,
    ) -> ScaleFeasibility:
        recommended = 40_000_000_000
        return ScaleFeasibility(
            target_millions=target,
            completed=True,
            finite_steps=True,
            parameter_count=PUBLICATION_EXPECTED_PARAMETERS[target],
            maximum_driver_allocated_bytes=int(recommended * memory_fraction),
            recommended_max_memory_bytes=recommended,
            projected_hours_per_model=projected_hours,
        )

    def test_selection_uses_largest_passing_candidate(self) -> None:
        self.assertEqual(CORE_MODEL_RUNS, 12)
        self.assertEqual(
            CORE_MODEL_FAMILIES,
            (
                "candidate",
                "raw_byte_reference",
                "byte_bpe_16000_body_matched",
                "byte_bpe_32000",
            ),
        )
        results = {
            50: self._result(50),
            75: self._result(75),
            100: self._result(100, memory_fraction=0.8),
        }
        self.assertEqual(select_largest_feasible_scale(results), 75)
        self.assertFalse(results[75].to_dict()["family_aware_campaign_lock"])

    def test_selection_rejects_impractical_campaign_time(self) -> None:
        results = {
            target: self._result(target, projected_hours=12.0)
            for target in PUBLICATION_SCALE_ORDER
        }
        self.assertIsNone(select_largest_feasible_scale(results))


class PublicationCampaignScaleSelectionTests(unittest.TestCase):
    def _campaign_result(
        self,
        target: int,
        *,
        failed_family: str | None = None,
        projected_hours: float = 2.0,
    ) -> CampaignScaleFeasibility:
        expected_parameters = PUBLICATION_FAMILY_EXPECTED_PARAMETERS[target]
        median_step_seconds = (
            projected_hours * 3_600 / PUBLICATION_PROJECTED_TRAIN_STEPS
        )
        descriptor = make_reference_descriptor("fixed_byte_6")
        families = tuple(
            FamilyScaleFeasibility(
                target_millions=target,
                family=family,
                completed=family != failed_family,
                finite_steps=family != failed_family,
                parameter_count=expected_parameters[family],
                expected_parameter_count=expected_parameters[family],
                maximum_driver_allocated_bytes=20_000_000_000,
                recommended_max_memory_bytes=40_000_000_000,
                median_train_step_seconds=median_step_seconds,
                raw_source_bytes_per_step=(
                    PUBLICATION_SEQUENCE_LENGTH * PUBLICATION_BATCH_SIZE
                ),
                raw_reference_descriptor_identity_sha256=(
                    descriptor.identity_sha256
                    if family == "raw_byte_reference"
                    else ""
                ),
                raw_reference_policy=(
                    descriptor.policy
                    if family == "raw_byte_reference"
                    else ""
                ),
            )
            for family in CORE_MODEL_FAMILIES
        )
        return CampaignScaleFeasibility(
            target_millions=target,
            raw_reference_descriptor=descriptor,
            family_results=families,
        )

    def test_final_scale_requires_every_runtime_family(self) -> None:
        results = {
            50: self._campaign_result(50),
            75: self._campaign_result(75),
            100: self._campaign_result(
                100,
                failed_family="byte_bpe_16000_body_matched",
            ),
        }
        self.assertEqual(select_largest_campaign_feasible_scale(results), 75)
        self.assertFalse(results[100].passes)
        self.assertTrue(results[75].passes)
        self.assertTrue(
            results[75].to_dict()["family_aware_campaign_lock"]
        )

    def test_family_specific_times_are_summed_not_candidate_multiplied(self) -> None:
        result = self._campaign_result(100, projected_hours=2.0)
        self.assertAlmostEqual(result.projected_campaign_hours, 28.8)
        self.assertTrue(result.passes)

    def test_campaign_time_cap_blocks_scale(self) -> None:
        results = {
            target: self._campaign_result(target, projected_hours=9.0)
            for target in PUBLICATION_SCALE_ORDER
        }
        self.assertIsNone(select_largest_campaign_feasible_scale(results))

    def test_candidate_parameter_identity_cannot_be_self_attested(self) -> None:
        result = self._campaign_result(50)
        candidate = result.family_results[0]
        altered_candidate = replace(
            candidate,
            parameter_count=candidate.parameter_count - 1,
            expected_parameter_count=candidate.expected_parameter_count - 1,
        )
        altered = replace(
            result,
            family_results=(altered_candidate, *result.family_results[1:]),
        )
        self.assertFalse(altered_candidate.passes)
        self.assertFalse(altered.passes)

    def test_bpe_parameter_identity_cannot_be_self_attested(self) -> None:
        result = self._campaign_result(50)
        bpe = result.family_results[2]
        altered_bpe = replace(
            bpe,
            parameter_count=bpe.parameter_count - 1,
            expected_parameter_count=bpe.expected_parameter_count - 1,
        )
        altered = replace(
            result,
            family_results=(
                *result.family_results[:2],
                altered_bpe,
                result.family_results[3],
            ),
        )
        self.assertFalse(altered_bpe.passes)
        self.assertFalse(altered.passes)

    def test_family_time_projection_is_derived_from_raw_bytes(self) -> None:
        result = self._campaign_result(50, projected_hours=2.0)
        family = result.family_results[0]
        self.assertEqual(
            family.projected_train_steps,
            PUBLICATION_PROJECTED_TRAIN_STEPS,
        )
        self.assertAlmostEqual(family.projected_hours_per_model, 2.0)
        half_raw_bytes = replace(
            family,
            raw_source_bytes_per_step=family.raw_source_bytes_per_step // 2,
        )
        self.assertAlmostEqual(half_raw_bytes.projected_hours_per_model, 4.0)

        invalid = replace(family, raw_source_bytes_per_step=0)
        self.assertFalse(invalid.passes)
        self.assertIsNone(invalid.to_dict()["projected_hours_per_model"])

    def test_entropy_router_training_scoring_and_parameters_are_not_omitted(self) -> None:
        result = self._campaign_result(50, projected_hours=2.0)
        raw = result.family_results[1]
        descriptor = make_reference_descriptor("entropy_threshold_full")
        entropy = replace(
            raw,
            raw_reference_descriptor_identity_sha256=descriptor.identity_sha256,
            raw_reference_policy=descriptor.policy,
            auxiliary_kind=PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
            auxiliary_parameter_count=PUBLICATION_ROUTER_EXPECTED_PARAMETERS[50],
            expected_auxiliary_parameter_count=(
                PUBLICATION_ROUTER_EXPECTED_PARAMETERS[50]
            ),
            auxiliary_config_sha256=content_hash("router-config"),
            auxiliary_train_completed=True,
            auxiliary_train_finite_steps=True,
            auxiliary_train_measurement_count=(
                PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS
            ),
            auxiliary_train_workload_sha256=content_hash("router-train-workload"),
            median_auxiliary_train_step_seconds=raw.median_train_step_seconds,
            auxiliary_train_raw_source_bytes_per_step=raw.raw_source_bytes_per_step,
            auxiliary_train_total_raw_source_bytes=PUBLICATION_TRAIN_BYTES,
            auxiliary_train_maximum_driver_allocated_bytes=18_000_000_000,
            auxiliary_score_completed=True,
            auxiliary_score_finite_steps=True,
            auxiliary_score_measurement_count=(
                PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS
            ),
            auxiliary_score_workload_sha256=content_hash("router-score-workload"),
            median_auxiliary_score_step_seconds=raw.median_train_step_seconds,
            auxiliary_score_raw_source_bytes_per_step=raw.raw_source_bytes_per_step,
            auxiliary_score_total_raw_source_bytes=PUBLICATION_TRAIN_BYTES,
            auxiliary_score_maximum_driver_allocated_bytes=17_000_000_000,
            auxiliary_runtime_completed=True,
            auxiliary_runtime_finite=True,
            auxiliary_runtime_measurement_count=(
                PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS
            ),
            auxiliary_runtime_observed_bytes=640,
            auxiliary_runtime_expected_bytes=640,
            auxiliary_runtime_router_forward_calls=129,
            auxiliary_runtime_workload_sha256=content_hash(
                "router-runtime-workload"
            ),
            auxiliary_runtime_maximum_driver_allocated_bytes=22_000_000_000,
        )
        self.assertTrue(entropy.auxiliary_contract_pass)
        self.assertAlmostEqual(entropy.projected_hours_per_model, 6.0)
        self.assertEqual(
            entropy.total_runtime_parameter_count,
            raw.parameter_count + PUBLICATION_ROUTER_EXPECTED_PARAMETERS[50],
        )

        omitted = replace(
            entropy,
            median_auxiliary_score_step_seconds=0.0,
            auxiliary_score_raw_source_bytes_per_step=0,
        )
        self.assertFalse(omitted.auxiliary_contract_pass)
        self.assertFalse(omitted.passes)

        disguised = replace(
            entropy,
            auxiliary_kind="none",
            auxiliary_parameter_count=0,
            expected_auxiliary_parameter_count=0,
        )
        self.assertFalse(disguised.auxiliary_contract_pass)

        main_only_memory = replace(
            entropy,
            auxiliary_runtime_maximum_driver_allocated_bytes=0,
        )
        self.assertFalse(main_only_memory.auxiliary_contract_pass)

        entropy_campaign = replace(
            result,
            raw_reference_descriptor=descriptor,
            family_results=(
                result.family_results[0],
                entropy,
                *result.family_results[2:],
            ),
        )
        self.assertTrue(entropy_campaign.passes)
        stale_descriptor = replace(
            entropy_campaign,
            raw_reference_descriptor=make_reference_descriptor("fixed_byte_6"),
        )
        self.assertFalse(stale_descriptor.passes)


if __name__ == "__main__":
    unittest.main()
