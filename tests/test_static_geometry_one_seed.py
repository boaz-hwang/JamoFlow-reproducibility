import math
import unittest

import numpy as np

from scripts.static_geometry_one_seed_core import (
    CONTINUATION_BYTES,
    MODES,
    PROMPT_COUNT,
    REPETITIONS,
    ROLES,
    WARMUP_PROMPTS,
    one_seed_decision,
    summarize_one_seed_quality,
    summarize_one_seed_timing,
)


def _document_metadata(count):
    return {
        "sequence_count": count,
        "sequence_length": 512,
        "eligible_sequence_fraction_pass": True,
    }


def _correctness():
    comparisons = WARMUP_PROMPTS * CONTINUATION_BYTES
    return {
        role: {
            "argmax_comparisons": comparisons,
            "argmax_exact": comparisons,
            "boundary_trace_exact": True,
            "cache_diagnostics_exact": True,
            "maximum_normalized_logit_error": 0.5,
            "strict_free_outputs": PROMPT_COUNT,
        }
        for role in ROLES
    }


def _timings(candidate=8.0, baseline=10.0):
    shape = (len(MODES), PROMPT_COUNT, REPETITIONS, len(ROLES))
    values = np.empty(shape, dtype=np.float64)
    values[:, :, :, ROLES.index("candidate")] = candidate
    values[:, :, :, ROLES.index("baseline")] = baseline
    return values


class StaticGeometryOneSeedTest(unittest.TestCase):
    def test_quality_screen_uses_paired_document_noninferiority(self):
        count = 96
        baseline = np.full(count, 600.0, dtype=np.float32)
        difference_nats = 0.005 * 511 * math.log(2.0)
        candidate = baseline + np.float32(difference_nats)
        documents = np.repeat(np.arange(24, dtype=np.int32), 4)
        summary = summarize_one_seed_quality(
            candidate_losses_nats=candidate,
            baseline_losses_nats=baseline,
            document_indices=documents,
            document_metadata=_document_metadata(count),
        )
        self.assertTrue(summary["overall_pass"])
        self.assertAlmostEqual(summary["mean_difference_bpb"], 0.005, places=5)
        self.assertLessEqual(
            summary["document_bootstrap"]["one_sided_95_upper_bpb"],
            0.010,
        )

        failed = summarize_one_seed_quality(
            candidate_losses_nats=baseline
            + np.float32(0.02 * 511 * math.log(2.0)),
            baseline_losses_nats=baseline,
            document_indices=documents,
            document_metadata=_document_metadata(count),
        )
        self.assertFalse(failed["overall_pass"])

    def test_both_actual_modes_must_pass_fixed_latency_gates(self):
        timings = _timings()
        emitted = np.full(
            (len(MODES), PROMPT_COUNT, len(ROLES)),
            CONTINUATION_BYTES,
            dtype=np.int16,
        )
        timing = summarize_one_seed_timing(
            end_to_end_ms=timings,
            ttft_ms=timings * 0.2,
            decode_ms=timings * 0.8,
            emitted_output_bytes=emitted,
            correctness=_correctness(),
        )
        self.assertTrue(timing["overall_pass"])
        self.assertTrue(all(row["overall_pass"] for row in timing["modes"].values()))
        decision = one_seed_decision({"overall_pass": True}, timing)
        self.assertTrue(decision["multi_seed_static_control_authorized"])

        failed = summarize_one_seed_timing(
            end_to_end_ms=_timings(candidate=9.0),
            ttft_ms=timings * 0.2,
            decode_ms=timings * 0.8,
            emitted_output_bytes=emitted,
            correctness=_correctness(),
        )
        self.assertFalse(failed["overall_pass"])
        self.assertFalse(
            one_seed_decision({"overall_pass": True}, failed)
            ["conditional_local_compute_research_authorized"]
        )

    def test_correctness_schema_and_output_horizon_are_fail_closed(self):
        timings = _timings()
        emitted = np.full(
            (len(MODES), PROMPT_COUNT, len(ROLES)),
            CONTINUATION_BYTES,
            dtype=np.int16,
        )
        evidence = _correctness()
        evidence["candidate"]["argmax_exact"] -= 1
        summary = summarize_one_seed_timing(
            end_to_end_ms=timings,
            ttft_ms=timings,
            decode_ms=timings,
            emitted_output_bytes=emitted,
            correctness=evidence,
        )
        self.assertFalse(summary["overall_pass"])

        emitted[1, 0, 0] = CONTINUATION_BYTES + 4
        with self.assertRaisesRegex(ValueError, "emitted-byte evidence differs"):
            summarize_one_seed_timing(
                end_to_end_ms=timings,
                ttft_ms=timings,
                decode_ms=timings,
                emitted_output_bytes=emitted,
                correctness=_correctness(),
            )


if __name__ == "__main__":
    unittest.main()
