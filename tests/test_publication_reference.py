import unittest
from dataclasses import replace

from jamoflow.publication_reference import (
    PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
    PUBLICATION_AUXILIARY_NONE,
    build_publication_raw_reference_descriptor,
    validate_publication_entropy_router_bundle,
    validate_publication_raw_reference_descriptor,
)
from tests.publication_reference_support import (
    content_hash,
    make_reference_descriptor,
    make_router_bundles,
)


class PublicationReferenceTests(unittest.TestCase):
    def test_auxiliary_kind_is_derived_from_the_selected_policy(self) -> None:
        entropy = make_reference_descriptor("entropy_threshold_codepoint")
        structural = make_reference_descriptor("fixed_byte_6")
        converted = make_reference_descriptor("causal_codepoint_grid_64")
        self.assertEqual(
            entropy.auxiliary_kind,
            PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
        )
        self.assertEqual(structural.auxiliary_kind, PUBLICATION_AUXILIARY_NONE)
        self.assertEqual(converted.model_family, "compute_conversion")
        self.assertEqual(converted.patch_count, 64)
        for descriptor in (entropy, structural, converted):
            validate_publication_raw_reference_descriptor(descriptor)

    def test_selection_cannot_disguise_entropy_as_structural(self) -> None:
        selection = {
            "schema_version": 1,
            "selection_uses_latency": False,
            "seed_order": [1729, 2718, 31415],
            "phase3_initial_summary": {"sha256": content_hash("phase3")},
            "conversion_initial_summary": {
                "sha256": content_hash("conversion")
            },
            "reference": {
                "policy": "entropy_threshold_full",
                "runtime_policy": "entropy_threshold_full",
                "model_family": "phase3",
                "patch_count": 86,
                "requires_entropy_router": False,
            },
            "reference_selection": {
                "selected_policy": "entropy_threshold_full"
            },
        }
        with self.assertRaisesRegex(ValueError, "contradicts"):
            build_publication_raw_reference_descriptor(
                selection,
                selection_sha256=content_hash("selection"),
            )

    def test_router_bundle_binds_threshold_policy_streams_and_state(self) -> None:
        descriptor = make_reference_descriptor("entropy_threshold_full")
        bundle = make_router_bundles(descriptor)[1729]
        validate_publication_entropy_router_bundle(bundle, descriptor)
        for tampered in (
            replace(bundle, threshold_nats=bundle.threshold_nats + 0.1),
            replace(bundle, maximum_patch_length=23),
            replace(bundle, calibration_stream_sha256=content_hash("other-cal")),
            replace(bundle, router_checkpoint_state_sha256=content_hash("other-state")),
        ):
            with self.subTest(field=tampered):
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    validate_publication_entropy_router_bundle(
                        tampered,
                        descriptor,
                    )


if __name__ == "__main__":
    unittest.main()
