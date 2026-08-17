import unittest

from jamoflow.cost import (
    compact_blt_flops,
    compact_router_flops,
    cross_attention_flops,
    end_to_end_flop_summary,
    transformer_block_flops,
    variable_patch_flop_summary,
)

import numpy as np


class CostTests(unittest.TestCase):
    def test_transformer_formula_counts_dense_matmuls(self) -> None:
        self.assertEqual(
            transformer_block_flops(2, 3, 5),
            8 * 2 * 3**2 + 6 * 2 * 3 * 5 + 4 * 2**2 * 3,
        )

    def test_cross_attention_formula_distinguishes_query_and_key_lengths(self) -> None:
        self.assertEqual(
            cross_attention_flops(2, 5, 3),
            4 * 2 * 3**2 + 4 * 5 * 3**2 + 4 * 2 * 5 * 3,
        )

    def test_compact_cost_totals_are_stable_and_router_is_material(self) -> None:
        main = compact_blt_flops()
        router = compact_router_flops()
        total = end_to_end_flop_summary()

        self.assertEqual(main["forward_flops_per_sequence"], 257_261_568)
        self.assertEqual(router["forward_flops_per_sequence"], 96_468_992)
        self.assertGreater(total["router_share_of_entropy_end_to_end"], 0.10)
        self.assertLess(total["router_share_of_entropy_end_to_end"], 0.50)

    def test_variable_cost_separates_ideal_and_batch_max_padding(self) -> None:
        fixed = variable_patch_flop_summary(
            np.asarray([43, 43, 43, 43]),
            batch_size=2,
            include_router=False,
        )
        variable = variable_patch_flop_summary(
            np.asarray([20, 66, 43, 43]),
            batch_size=2,
            include_router=True,
        )
        self.assertEqual(
            fixed["ideal_unpadded_mean_flops_per_sequence"],
            compact_blt_flops()["forward_flops_per_sequence"],
        )
        self.assertEqual(fixed["patch_slot_padding_rate"], 0.0)
        self.assertGreater(
            variable["implemented_batch_max_mean_flops_per_sequence"],
            variable["ideal_unpadded_mean_flops_per_sequence"],
        )
        self.assertGreater(variable["patch_slot_padding_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
