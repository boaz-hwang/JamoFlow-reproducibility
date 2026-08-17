import unittest

import numpy as np
import torch

from jamoflow.incremental_blt import IncrementalBltDecoder
from jamoflow.neural_model import Phase1ModelSpec, build_main_model
from scripts.incremental_block_kernel import IncrementalBlockBltDecoder
from scripts.target_block_kernel_core import (
    MICRO_CASES,
    MICRO_REPETITIONS,
    WHOLE_CASES,
    WHOLE_REPETITIONS,
    perfect_hangul_groups,
    summarize_block_kernel,
)
from scripts.run_target_block_kernel_preflight import _require_inference_mode


class TargetBlockKernelTest(unittest.TestCase):
    def test_timing_requires_inference_mode(self):
        with self.assertRaisesRegex(RuntimeError, "inference_mode"):
            _require_inference_mode()
        with torch.inference_mode():
            _require_inference_mode()

    def test_perfect_grouping_reconstructs_and_only_groups_hangul(self):
        data = "가Aé각😀나".encode("utf-8")
        groups = perfect_hangul_groups(data)
        self.assertEqual(b"".join(groups), data)
        self.assertEqual([len(group) for group in groups], [3, 1, 1, 1, 3, 1, 1, 1, 1, 3])
        for group in groups:
            if len(group) == 3:
                self.assertTrue(0xAC00 <= ord(group.decode("utf-8")) <= 0xD7A3)

    def test_block_kernel_matches_sequential_tiny_cpu(self):
        spec = Phase1ModelSpec(
            sequence_length=32,
            vocab_size=256,
            patch_stride=4,
            patch_count=8,
            local_width=16,
            global_width=32,
            local_heads=4,
            global_heads=4,
            encoder_layers=1,
            global_layers=1,
            decoder_layers=1,
            local_ffn=32,
            global_ffn=64,
            router_width=16,
            router_heads=4,
            router_layers=1,
            router_ffn=32,
        )
        model = build_main_model(spec, seed=5, global_max_position_embeddings=72)
        kwargs = {
            "model": model,
            "policy": "causal_whitespace_grid",
            "horizon": 32,
            "patch_count": 8,
            "fixed_stride": 4,
        }
        prompt = "가나다".encode("utf-8")
        continuation = "라마바".encode("utf-8")
        sequential = IncrementalBltDecoder(**kwargs)
        block = IncrementalBlockBltDecoder(**kwargs)
        with torch.inference_mode():
            sequential.prefill_parallel(prompt)
            block.prefill_parallel(prompt)
            expected = torch.cat(
                [sequential.consume(value) for value in continuation[:3]], dim=0
            )
            actual = block.consume_block(continuation[:3])
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            self.assertTrue(torch.equal(actual.argmax(-1), expected.argmax(-1)))
            self.assertEqual(block.diagnostics, sequential.diagnostics)
            for value in continuation[3:]:
                torch.testing.assert_close(
                    block.consume(value), sequential.consume(value), rtol=1e-5, atol=1e-5
                )
        self.assertEqual(block.diagnostics, sequential.diagnostics)

    @staticmethod
    def _summary(micro_block: float = 3.0, whole_block: float = 70.0):
        micro = np.empty((2, MICRO_CASES, MICRO_REPETITIONS, 2), dtype=np.float64)
        micro[..., 0] = 7.2
        micro[..., 1] = micro_block
        whole = np.empty((WHOLE_CASES, WHOLE_REPETITIONS, 2), dtype=np.float64)
        whole[..., 0] = 100.0
        whole[..., 1] = whole_block
        return summarize_block_kernel(
            micro_ms=micro,
            whole_ms=whole,
            whole_hangul_blocks=np.full(WHOLE_CASES, 50, dtype=np.int64),
            whole_boundary_blocks=np.full(WHOLE_CASES, 10, dtype=np.int64),
            correctness={
                "argmax_comparisons": 100,
                "cache_comparisons": 10,
                "maximum_absolute_logit_error": 1e-6,
                "maximum_normalized_tolerance_ratio": 0.2,
                "pass": True,
            },
            independent_first_acceptance=0.42,
            independent_pair_acceptance=0.24,
            independent_head_latency_ms=1.0,
            minimum_micro_reduction=0.30,
            minimum_micro_lower_bound=0.20,
            minimum_perfect_whole_reduction=0.20,
            minimum_perfect_whole_lower_bound=0.10,
            minimum_projected_reduction=0.20,
            minimum_projected_lower_bound=0.10,
        )

    def test_summary_authorizes_only_when_every_system_gate_passes(self):
        passed = self._summary()
        self.assertTrue(passed["gates"]["full_speculative_runtime_authorized"])
        self.assertAlmostEqual(
            passed["fixed_independent_projection"][
                "expected_committed_bytes_per_verification"
            ],
            2.66,
        )
        failed = self._summary(micro_block=6.5)
        self.assertFalse(failed["gates"]["full_speculative_runtime_authorized"])
        self.assertFalse(failed["gates"]["passes"]["micro_target_block"])

    def test_block_kernel_matches_sequential_across_an_internal_patch_boundary(self):
        spec = Phase1ModelSpec(
            sequence_length=32,
            vocab_size=256,
            patch_stride=4,
            patch_count=8,
            local_width=16,
            global_width=32,
            local_heads=4,
            global_heads=4,
            encoder_layers=1,
            global_layers=1,
            decoder_layers=1,
            local_ffn=32,
            global_ffn=64,
            router_width=16,
            router_heads=4,
            router_layers=1,
            router_ffn=32,
        )
        model = build_main_model(spec, seed=11, global_max_position_embeddings=72)
        kwargs = {
            "model": model,
            "policy": "causal_whitespace_grid",
            "horizon": 32,
            "patch_count": 8,
            "fixed_stride": 4,
        }
        prompt = "가나".encode("utf-8")
        block_value = "다".encode("utf-8")
        sequential = IncrementalBltDecoder(**kwargs)
        blocked = IncrementalBlockBltDecoder(**kwargs)
        with torch.inference_mode():
            sequential.prefill_parallel(prompt)
            blocked.prefill_parallel(prompt)
            before = sequential.diagnostics.emitted_data_patches
            expected = torch.cat(
                [sequential.consume(value) for value in block_value], dim=0
            )
            actual = blocked.consume_block(block_value)
        self.assertGreater(sequential.diagnostics.emitted_data_patches, before)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(blocked.diagnostics, sequential.diagnostics)

    def test_block_transaction_rolls_every_prefix_back_to_sequential_state(self):
        spec = Phase1ModelSpec(
            sequence_length=32,
            vocab_size=256,
            patch_stride=4,
            patch_count=8,
            local_width=16,
            global_width=32,
            local_heads=4,
            global_heads=4,
            encoder_layers=1,
            global_layers=1,
            decoder_layers=1,
            local_ffn=32,
            global_ffn=64,
            router_width=16,
            router_heads=4,
            router_layers=1,
            router_ffn=32,
        )
        model = build_main_model(spec, seed=19, global_max_position_embeddings=72)
        kwargs = {
            "model": model,
            "policy": "causal_whitespace_grid",
            "horizon": 32,
            "patch_count": 8,
            "fixed_stride": 4,
        }
        prompt = "가나".encode("utf-8")
        proposed = "다".encode("utf-8")
        follow = "라".encode("utf-8")
        with torch.inference_mode():
            for keep in range(4):
                reference = IncrementalBltDecoder(**kwargs)
                speculative = IncrementalBlockBltDecoder(**kwargs)
                reference.prefill_parallel(prompt)
                speculative.prefill_parallel(prompt)
                expected_rows = [reference.consume(value) for value in proposed[:keep]]
                transaction = speculative.consume_block_transaction(proposed)
                if keep:
                    torch.testing.assert_close(
                        transaction.logits[:keep],
                        torch.cat(expected_rows, dim=0),
                        rtol=1e-5,
                        atol=1e-5,
                    )
                transaction.finish(keep)
                self.assertEqual(speculative.diagnostics, reference.diagnostics)
                for value in follow:
                    torch.testing.assert_close(
                        speculative.consume(value),
                        reference.consume(value),
                        rtol=1e-5,
                        atol=1e-5,
                    )
                self.assertEqual(speculative.diagnostics, reference.diagnostics)
                with self.assertRaisesRegex(RuntimeError, "already closed"):
                    transaction.finish(keep)


if __name__ == "__main__":
    unittest.main()
