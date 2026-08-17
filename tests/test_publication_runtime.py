import unittest
from dataclasses import replace

import numpy as np

from jamoflow.publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_PRETRAIN_SEEDS,
)
from jamoflow.publication_runtime import (
    build_publication_runtime_equivalence,
    build_publication_runtime_evidence,
    build_publication_timing_evidence,
    build_publication_valid_output_evidence,
    validate_publication_runtime_lineage,
    validate_publication_runtime_evidence,
)
from tests.publication_runtime_support import (
    make_diagnostic_inputs,
    make_equivalence_comparisons,
    make_lineage,
    make_runtime_evidence,
    make_timing_inputs,
    output_evidence_hashes,
)
from tests.publication_reference_support import make_reference_descriptor


class PublicationRuntimeEvidenceTests(unittest.TestCase):
    def test_complete_runtime_evidence_binds_one_exact_pair(self) -> None:
        evidence = make_runtime_evidence("raw_byte")
        validate_publication_runtime_evidence(evidence)
        self.assertTrue(evidence.overall_integrity_pass)
        self.assertEqual(evidence.seed_order, PUBLICATION_PRETRAIN_SEEDS)
        self.assertEqual(
            evidence.lineage.identity_sha256,
            evidence.equivalence.lineage_identity_sha256,
        )
        self.assertEqual(
            evidence.lineage.identity_sha256,
            evidence.timing.lineage_identity_sha256,
        )
        self.assertEqual(
            evidence.lineage.identity_sha256,
            evidence.valid_output.lineage_identity_sha256,
        )

    def test_raw_entropy_router_checkpoint_is_part_of_runtime_identity(self) -> None:
        lineage = make_lineage("raw_byte")
        self.assertEqual(
            len(lineage.comparator_auxiliary_checkpoint_sha256),
            len(PUBLICATION_PRETRAIN_SEEDS),
        )
        tampered = replace(
            lineage,
            comparator_auxiliary_checkpoint_sha256=(
                "f" * 64,
                *lineage.comparator_auxiliary_checkpoint_sha256[1:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_publication_runtime_lineage(tampered)

        descriptor_tampered = replace(
            lineage,
            raw_reference_descriptor=make_reference_descriptor("fixed_byte_6"),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_publication_runtime_lineage(descriptor_tampered)

        bundle = lineage.comparator_auxiliary_bundles[0]
        bundle_tampered = replace(bundle, threshold_nats=bundle.threshold_nats + 0.1)
        bundles = (bundle_tampered, *lineage.comparator_auxiliary_bundles[1:])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_publication_runtime_lineage(
                replace(lineage, comparator_auxiliary_bundles=bundles)
            )

    def test_entropy_runtime_requires_router_execution_counters(self) -> None:
        lineage = make_lineage("raw_byte")
        diagnostics, bound = make_diagnostic_inputs("raw_byte")
        diagnostics[
            "controlled_replay__router_cached_model_units__reference"
        ].fill(0)
        evidence = build_publication_valid_output_evidence(
            lineage,
            diagnostics,
            comparator_maximum_unit_bytes=bound,
            **output_evidence_hashes(),
        )
        self.assertFalse(evidence.router_execution_pass)
        self.assertFalse(evidence.overall_pass)

        structural = make_runtime_evidence(
            "raw_byte",
            raw_reference_policy="fixed_byte_6",
        )
        self.assertTrue(structural.valid_output.router_execution_pass)
        self.assertTrue(structural.overall_integrity_pass)

    def test_equivalence_requires_allclose_even_when_argmax_is_identical(self) -> None:
        lineage = make_lineage()
        evidence = build_publication_runtime_equivalence(
            lineage,
            make_equivalence_comparisons(pass_equivalence=False),
        )
        self.assertEqual(evidence.argmax_match_rate, 1.0)
        self.assertFalse(evidence.allclose_pass)
        self.assertFalse(evidence.overall_pass)

    def test_equivalence_requires_sixteen_vectors_for_every_path(self) -> None:
        lineage = make_lineage()
        comparisons = make_equivalence_comparisons()
        first_key = next(iter(comparisons))
        comparisons[first_key] = (
            comparisons[first_key][0][:15],
            comparisons[first_key][1][:15],
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_publication_runtime_equivalence(lineage, comparisons)

    def test_runtime_rejects_nested_evidence_from_another_comparator(self) -> None:
        lineage_16k = make_lineage(
            "standard_bpe",
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
        )
        lineage_32k = make_lineage(
            "standard_bpe",
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )
        equivalence_16k = build_publication_runtime_equivalence(
            lineage_16k,
            make_equivalence_comparisons(),
        )
        timing_32k = build_publication_timing_evidence(
            lineage_32k,
            *make_timing_inputs(),
        )
        diagnostics, bound = make_diagnostic_inputs("standard_bpe")
        valid_32k = build_publication_valid_output_evidence(
            lineage_32k,
            diagnostics,
            comparator_maximum_unit_bytes=bound,
            **output_evidence_hashes(),
        )
        with self.assertRaisesRegex(ValueError, "equivalence"):
            build_publication_runtime_evidence(
                lineage_32k,
                equivalence_16k,
                timing_32k,
                valid_32k,
            )

    def test_timing_rejects_schedule_warmup_and_seed_order_tampering(self) -> None:
        lineage = make_lineage()
        inputs = list(make_timing_inputs())
        schedule = inputs[1].copy()
        schedule[0, 0, 0, 0] ^= np.uint8(1)
        inputs[1] = schedule
        with self.assertRaisesRegex(ValueError, "timing evidence"):
            build_publication_timing_evidence(lineage, *inputs)

        inputs = list(make_timing_inputs())
        warmup_completion = inputs[3].copy()
        warmup_completion[0, 0, 0, 0] = 0
        inputs[3] = warmup_completion
        with self.assertRaisesRegex(ValueError, "timing evidence"):
            build_publication_timing_evidence(lineage, *inputs)

        inputs = list(make_timing_inputs())
        inputs[4] = tuple(reversed(inputs[4]))
        with self.assertRaisesRegex(ValueError, "timing evidence"):
            build_publication_timing_evidence(lineage, *inputs)

    def test_timing_and_output_must_come_from_one_trial_artifact(self) -> None:
        lineage = make_lineage()
        equivalence = build_publication_runtime_equivalence(
            lineage,
            make_equivalence_comparisons(),
        )
        timing = build_publication_timing_evidence(
            lineage,
            *make_timing_inputs(),
        )
        diagnostics, bound = make_diagnostic_inputs("raw_byte")
        output = build_publication_valid_output_evidence(
            lineage,
            diagnostics,
            comparator_maximum_unit_bytes=bound,
            **output_evidence_hashes(trial_artifact_sha256="f" * 64),
        )
        with self.assertRaisesRegex(ValueError, "trial artifacts"):
            build_publication_runtime_evidence(
                lineage,
                equivalence,
                timing,
                output,
            )

    def test_timing_component_or_environment_failure_cannot_pass(self) -> None:
        lineage = make_lineage()
        inputs = list(make_timing_inputs())
        arrays = dict(inputs[0])
        arrays["controlled_replay__end_to_end_ms__candidate"] = (
            arrays["controlled_replay__end_to_end_ms__candidate"] + 0.1
        )
        inputs[0] = arrays
        component_failure = build_publication_timing_evidence(lineage, *inputs)
        self.assertFalse(component_failure.component_identity_pass)
        self.assertFalse(component_failure.overall_pass)

        inputs = list(make_timing_inputs())
        environments = dict(inputs[5])
        bad_seed = PUBLICATION_PRETRAIN_SEEDS[0]
        environments[bad_seed] = {
            "start": environments[bad_seed]["start"],
            "end": {
                "power": {"returncode": 0, "stdout": "Battery Power"},
                "thermal": {"returncode": 1, "stdout": "warning"},
                "settings": {"returncode": 0, "stdout": "Battery Power:"},
            },
        }
        inputs[5] = environments
        environment_failure = build_publication_timing_evidence(lineage, *inputs)
        self.assertFalse(environment_failure.environment_pass)
        self.assertFalse(environment_failure.overall_pass)

    def test_bpe_overshoot_is_token_bounded_not_byte_bounded(self) -> None:
        bpe = make_runtime_evidence(
            "standard_bpe",
            bpe_reference_overshoot_bytes=40,
        )
        self.assertTrue(bpe.valid_output.overall_pass)

        lineage = make_lineage("raw_byte")
        diagnostics, bound = make_diagnostic_inputs("raw_byte")
        prefix = "free_running_utf8_greedy"
        role = "reference"
        emitted = diagnostics[f"{prefix}__emitted_output_bytes__{role}"]
        units = diagnostics[f"{prefix}__emitted_model_units__{role}"]
        steps = diagnostics[f"{prefix}__decode_forward_steps__{role}"]
        observed = diagnostics[f"{prefix}__runtime_observed_model_units__{role}"]
        overshoot = diagnostics[f"{prefix}__overshoot_bytes__{role}"]
        codepoints = diagnostics[f"{prefix}__output_codepoints__{role}"]
        emitted.fill(132)
        units.fill(132)
        steps.fill(131)
        observed.fill(128 + 131)
        overshoot.fill(4)
        codepoints.fill(44)
        diagnostics[f"{prefix}__router_observed_model_units__{role}"].fill(259)
        diagnostics[f"{prefix}__router_cached_model_units__{role}"].fill(259)
        diagnostics[f"{prefix}__router_scored_model_units__{role}"].fill(259)
        diagnostics[f"{prefix}__router_forward_calls__{role}"].fill(132)
        raw = build_publication_valid_output_evidence(
            lineage,
            diagnostics,
            comparator_maximum_unit_bytes=bound,
            **output_evidence_hashes(),
        )
        self.assertFalse(raw.free_running_contract_pass)
        self.assertFalse(raw.overall_pass)

    def test_nondeterministic_or_algebraically_invalid_output_cannot_pass(self) -> None:
        lineage = make_lineage()
        diagnostics, bound = make_diagnostic_inputs("raw_byte")
        diagnostics[
            "free_running_utf8_greedy__replacement_character_free__candidate"
        ][0, 0, 1] = 0
        nondeterministic = build_publication_valid_output_evidence(
            lineage,
            diagnostics,
            comparator_maximum_unit_bytes=bound,
            **output_evidence_hashes(),
        )
        self.assertFalse(nondeterministic.deterministic_diagnostics_pass)
        self.assertFalse(nondeterministic.overall_pass)

        diagnostics, bound = make_diagnostic_inputs("raw_byte")
        diagnostics[
            "controlled_replay__decode_forward_steps__candidate"
        ].fill(128)
        invalid = build_publication_valid_output_evidence(
            lineage,
            diagnostics,
            comparator_maximum_unit_bytes=bound,
            **output_evidence_hashes(),
        )
        self.assertFalse(invalid.controlled_contract_pass)
        self.assertFalse(invalid.overall_pass)

    def test_stale_nested_identity_is_rejected(self) -> None:
        evidence = make_runtime_evidence()
        tampered_timing = replace(
            evidence.timing,
            environment_pass=False,
        )
        tampered = replace(evidence, timing=tampered_timing)
        with self.assertRaisesRegex(ValueError, "timing evidence"):
            validate_publication_runtime_evidence(tampered)


if __name__ == "__main__":
    unittest.main()
