import gc
import json
from pathlib import Path
import unittest

import numpy as np

from jamoflow.cost import compact_blt_flops
from jamoflow.neural_model import build_main_model, parameter_count
from scripts.static_geometry_preflight_core import (
    BASELINE,
    CANDIDATE_ORDER,
    CONTINUATION_BYTES,
    GEOMETRY_ORDER,
    PROMPT_COUNT,
    REPETITIONS,
    geometry_contract,
    geometry_spec,
    summarize_geometry_preflight,
    validate_geometry_contract,
)
from scripts.run_static_geometry_preflight import _validate_plan


EXPECTED = {
    "baseline_w72": (19_596_096, 5_640_155_136),
    "thin128_e1_d2_g384x9": (19_605_888, 3_984_926_208),
    "thin160_e1_d1_g384x9": (19_571_872, 3_889_040_896),
    "thin128_e1_d1_g384x9": (19_575_680, 3_587_538_432),
}


def _correctness():
    return {
        name: {
            "argmax_comparisons": CONTINUATION_BYTES,
            "argmax_exact": CONTINUATION_BYTES,
            "boundary_trace_exact": True,
            "cache_diagnostics_exact": True,
            "maximum_normalized_logit_error": 0.5,
        }
        for name in GEOMETRY_ORDER
    }


class StaticGeometryPreflightTest(unittest.TestCase):
    def test_contract_is_exact_and_parameter_matched(self):
        contract = geometry_contract()
        validate_geometry_contract(contract)
        self.assertEqual(contract["baseline"], BASELINE)
        self.assertEqual(tuple(contract["candidate_order"]), CANDIDATE_ORDER)
        baseline_parameters = EXPECTED[BASELINE][0]
        for name in GEOMETRY_ORDER:
            spec = geometry_spec(name)
            model = build_main_model(
                spec, seed=1, global_max_position_embeddings=1032
            )
            parameters = parameter_count(model)
            flops = int(
                compact_blt_flops(spec, data_patches=72)[
                    "forward_flops_per_sequence"
                ]
            )
            self.assertEqual((parameters, flops), EXPECTED[name])
            self.assertLessEqual(abs(parameters / baseline_parameters - 1), 0.0025)
            del model
            gc.collect()

    def test_first_quality_conservative_passing_candidate_is_selected(self):
        timings = np.empty(
            (PROMPT_COUNT, REPETITIONS, len(GEOMETRY_ORDER)), dtype=np.float64
        )
        timings[:, :, 0] = 10.0
        timings[:, :, 1] = 7.0
        timings[:, :, 2] = 8.5
        timings[:, :, 3] = 6.0
        summary = summarize_geometry_preflight(
            timings_ms=timings,
            parameter_counts={name: EXPECTED[name][0] for name in GEOMETRY_ORDER},
            analytical_flops={name: EXPECTED[name][1] for name in GEOMETRY_ORDER},
            correctness=_correctness(),
        )
        self.assertEqual(summary["status"], "one_seed_static_control_authorized")
        self.assertEqual(
            summary["selection"]["selected_candidate"], CANDIDATE_ORDER[0]
        )
        self.assertTrue(summary["rows"][CANDIDATE_ORDER[0]]["overall_pass"])
        self.assertFalse(summary["rows"][CANDIDATE_ORDER[1]]["overall_pass"])
        self.assertTrue(summary["rows"][CANDIDATE_ORDER[2]]["overall_pass"])

    def test_branch_stops_without_a_candidate_above_the_fixed_gate(self):
        timings = np.empty(
            (PROMPT_COUNT, REPETITIONS, len(GEOMETRY_ORDER)), dtype=np.float64
        )
        timings[:, :, 0] = 10.0
        timings[:, :, 1:] = 8.5
        summary = summarize_geometry_preflight(
            timings_ms=timings,
            parameter_counts={name: EXPECTED[name][0] for name in GEOMETRY_ORDER},
            analytical_flops={name: EXPECTED[name][1] for name in GEOMETRY_ORDER},
            correctness=_correctness(),
        )
        self.assertEqual(summary["status"], "static_geometry_branch_stopped")
        self.assertIsNone(summary["selection"]["selected_candidate"])
        self.assertFalse(summary["selection"]["one_seed_training_authorized"])

    def test_nonexact_geometry_and_evidence_schemas_are_rejected(self):
        contract = geometry_contract()
        contract["candidate_order"] = list(reversed(contract["candidate_order"]))
        with self.assertRaisesRegex(ValueError, "contract differs"):
            validate_geometry_contract(contract)

        timings = np.full(
            (PROMPT_COUNT, REPETITIONS, len(GEOMETRY_ORDER)), 10.0,
            dtype=np.float64,
        )
        evidence = _correctness()
        evidence[BASELINE]["extra"] = True
        with self.assertRaisesRegex(ValueError, "correctness schema differs"):
            summarize_geometry_preflight(
                timings_ms=timings,
                parameter_counts={
                    name: EXPECTED[name][0] for name in GEOMETRY_ORDER
                },
                analytical_flops={name: EXPECTED[name][1] for name in GEOMETRY_ORDER},
                correctness=evidence,
            )

    def test_plan_cannot_omit_an_implementation_dependency(self):
        plan = json.loads(
            Path("data/manifests/static-geometry-preflight-v1.json").read_text(
                encoding="utf-8"
            )
        )
        plan["implementation_sha256"].pop("src/jamoflow/incremental_blt.py")
        with self.assertRaisesRegex(ValueError, "implementation file set differs"):
            _validate_plan(plan, "0" * 40)


if __name__ == "__main__":
    unittest.main()
