import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_phase3_cost.py"
SPEC = importlib.util.spec_from_file_location("summarize_phase3_cost", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


POLICIES = (
    MODULE.F,
    MODULE.C,
    MODULE.W,
    MODULE.S,
    MODULE.E,
    MODULE.EC,
)


def quality(means: dict[str, float] | None = None) -> dict:
    values = {
        MODULE.F: 2.01,
        MODULE.C: 2.00,
        MODULE.W: 1.989,
        MODULE.S: 2.02,
        MODULE.E: 1.985,
        MODULE.EC: 1.98,
    }
    values.update(means or {})
    return {
        "quality": {
            policy: {"count": 3, "mean": value}
            for policy, value in values.items()
        },
        "calibration_quality": {
            policy: {"count": 3, "mean": value}
            for policy, value in values.items()
        },
    }


def benchmark() -> dict:
    main = 100.0
    learned = 140.0
    analytical = {
        policy: {
            "ideal_unpadded_mean_flops_per_sequence": (
                learned if policy in (MODULE.E, MODULE.EC) else main
            )
        }
        for policy in POLICIES
    }
    analytical[MODULE.S]["ideal_unpadded_mean_flops_per_sequence"] = 180.0
    measurements = {}
    for batch in ("1", "8"):
        measurements[batch] = {
            "direct_pipeline_timings": {
                f"end_to_end/{policy}": {
                    "median_ms": (
                        10.0 if policy not in (MODULE.E, MODULE.EC) else 12.0
                    ),
                    "measurements_ms": [
                        10.0 if policy not in (MODULE.E, MODULE.EC) else 12.0
                        for _ in range(8)
                    ],
                    "measurement_input_batch_ids": list(range(8)),
                }
                for policy in POLICIES
            }
        }
    comparisons = {
        learned_policy: {
            MODULE.W: {
                "ideal_unpadded_flop_reduction": 1 - main / learned,
                "direct_latency_reduction": {"1": 1 / 6, "8": 1 / 6},
            }
        }
        for learned_policy in (MODULE.E, MODULE.EC)
    }
    return {
        "analytical_flops": {"1": analytical},
        "measurements": measurements,
        "comparisons_vs_learned_router": comparisons,
    }


class Phase3CostSummaryTests(unittest.TestCase):
    def test_learned_policy_is_selected_by_quality_not_cost(self) -> None:
        selected = MODULE.select_learned_policy(
            quality()["calibration_quality"]
        )
        self.assertEqual(selected, MODULE.EC)

    def test_gate_k_selection_uses_calibration_not_test(self) -> None:
        evidence = quality()
        evidence["quality"][MODULE.E]["mean"] = 1.90
        evidence["quality"][MODULE.EC]["mean"] = 2.10
        evidence["calibration_quality"][MODULE.E]["mean"] = 2.10
        evidence["calibration_quality"][MODULE.EC]["mean"] = 1.90
        result = MODULE.gate_k_summary(
            benchmark(), evidence, gate_j_pass=True
        )
        self.assertEqual(result["selected_learned_policy"], MODULE.EC)

    def test_gate_k_passes_quality_cost_latency_and_pareto(self) -> None:
        result = MODULE.gate_k_summary(
            benchmark(), quality(), gate_j_pass=True
        )
        self.assertTrue(result["overall_pass"])
        self.assertEqual(result["selected_learned_policy"], MODULE.EC)
        self.assertTrue(result["h2_quality_pass"])
        self.assertTrue(
            result[
                "batch1_or_batch8_latency_reduction_at_least_10_percent_with_positive_paired_bootstrap_lower_bound"
            ]
        )
        self.assertEqual(result["qualifying_latency_batches"], ["1", "8"])
        self.assertTrue(result["whitespace_nondominated_with_spacebyte_included"])

    def test_gate_k_rejects_unstable_input_batch_speedup(self) -> None:
        source = benchmark()
        for batch in ("1", "8"):
            timing = source["measurements"][batch]["direct_pipeline_timings"]
            timing[f"end_to_end/{MODULE.W}"]["measurements_ms"] = [
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
                30.0,
                30.0,
                30.0,
            ]
        result = MODULE.gate_k_summary(source, quality(), gate_j_pass=True)
        self.assertFalse(result["overall_pass"])
        self.assertEqual(result["qualifying_latency_batches"], [])

    def test_gate_k_fails_if_whitespace_is_dominated(self) -> None:
        result = MODULE.gate_k_summary(
            benchmark(),
            quality({MODULE.C: 1.98}),
            gate_j_pass=True,
        )
        self.assertFalse(result["overall_pass"])
        self.assertIn(
            MODULE.C,
            result["pareto"]["by_policy"][MODULE.W]["dominators"],
        )

    def test_gate_j_failure_prevents_scale_even_if_cost_passes(self) -> None:
        result = MODULE.gate_k_summary(
            benchmark(), quality(), gate_j_pass=False
        )
        self.assertFalse(result["overall_pass"])
        self.assertEqual(result["status"], "fail_gate_j")


if __name__ == "__main__":
    unittest.main()
