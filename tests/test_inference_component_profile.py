import unittest

import numpy as np

from scripts.inference_component_profile_core import (
    COMPONENTS,
    PROFILE_CHECKPOINT_ROLES,
    PROFILE_COMPONENT_CASES,
    PROFILE_DECODE_BYTES,
    PROFILE_SCHEDULES,
    PROFILE_SEEDS,
    PROFILE_WHOLE_CASES,
    PROFILE_WHOLE_REPETITIONS,
    WHOLE_METRICS,
    summarize_profile_arrays,
)


class ComponentProfileTest(unittest.TestCase):
    def fixture(self):
        prefix = (len(PROFILE_SEEDS), len(PROFILE_CHECKPOINT_ROLES), len(PROFILE_SCHEDULES))
        whole = np.ones(
            (*prefix, PROFILE_WHOLE_CASES, PROFILE_WHOLE_REPETITIONS, len(WHOLE_METRICS)),
            dtype=np.float64,
        )
        whole[:, :, 0] *= 90.0
        whole[:, :, 1] *= 100.0
        steps = np.ones(
            (*prefix, PROFILE_COMPONENT_CASES, PROFILE_DECODE_BYTES),
            dtype=np.float64,
        )
        boundaries = np.zeros(steps.shape, dtype=np.bool_)
        boundaries[..., ::6] = True
        steps[boundaries] = 3.0
        totals = np.ones((*prefix, PROFILE_COMPONENT_CASES, len(COMPONENTS)), dtype=np.float64)
        calls = np.ones(totals.shape, dtype=np.int64)
        prompts = np.full(
            (*prefix, PROFILE_WHOLE_CASES, PROFILE_WHOLE_REPETITIONS),
            18,
            dtype=np.int64,
        )
        finals = prompts + 18
        return whole, steps, boundaries, totals, calls, prompts, finals

    def test_summary_preserves_two_by_two_schedule_effect(self):
        values = self.fixture()
        summary = summarize_profile_arrays(
            whole_ms=values[0],
            step_ms=values[1],
            step_boundary=values[2],
            component_total_ms=values[3],
            component_calls=values[4],
            prompt_patches=values[5],
            final_patches=values[6],
        )
        effect = summary["whole_trial"]["decode_ms"][
            "same_checkpoint_W72_vs_C86_schedule"
        ]["candidate"]
        self.assertAlmostEqual(effect["median_reduction"], 0.1)
        self.assertEqual(effect["positive_seed_count"], 5)
        step = summary["step_synchronized_diagnostic"]["candidate"]["W72"]
        self.assertEqual(step["median_boundary_increment_ms"], 2.0)

    def test_summary_rejects_nonfinite_arrays(self):
        values = list(self.fixture())
        values[0][0, 0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid values"):
            summarize_profile_arrays(
                whole_ms=values[0],
                step_ms=values[1],
                step_boundary=values[2],
                component_total_ms=values[3],
                component_calls=values[4],
                prompt_patches=values[5],
                final_patches=values[6],
            )


if __name__ == "__main__":
    unittest.main()
