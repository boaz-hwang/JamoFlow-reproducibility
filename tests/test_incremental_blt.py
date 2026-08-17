from __future__ import annotations

import importlib.util
import unittest

import numpy as np


HAS_RESEARCH_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_RESEARCH_DEPS, "optional neural research dependencies")
class IncrementalBltTests(unittest.TestCase):
    def _spec(self):
        from jamoflow.neural_model import Phase1ModelSpec

        return Phase1ModelSpec(
            sequence_length=48,
            patch_count=8,
            patch_stride=6,
            local_width=32,
            global_width=64,
            local_heads=4,
            global_heads=4,
            encoder_layers=1,
            global_layers=2,
            decoder_layers=2,
            local_ffn=96,
            global_ffn=192,
            cross_attention_k=2,
            hash_group_size=3,
            hash_vocabulary=256,
            router_width=32,
            router_heads=4,
            router_layers=1,
            router_ffn=96,
        )

    def test_prefix_boundary_helper_matches_generation_matrix(self) -> None:
        from jamoflow.generation import generation_patch_matrix
        from jamoflow.incremental_blt import structural_prefix_boundaries

        raw = "한국어 incremental test 문장 ".encode("utf-8")
        for policy in (
            "fixed_byte_6",
            "causal_codepoint_grid",
            "causal_whitespace_grid",
        ):
            for length in range(1, min(len(raw), 40) + 1):
                prefix = raw[:length]
                matrix = generation_patch_matrix(
                    np.asarray([list(prefix)], dtype=np.uint8),
                    policy,
                    horizon=48,
                    patch_count=8,
                    fixed_stride=6,
                )
                lengths = matrix[0, 1:]
                lengths = lengths[lengths > 0].astype(np.int64)
                expected = tuple(
                    np.concatenate(
                        [
                            np.asarray([0], dtype=np.int64),
                            np.cumsum(lengths)[:-1],
                        ]
                    ).tolist()
                )
                actual = structural_prefix_boundaries(
                    prefix,
                    policy,
                    horizon=48,
                    patch_count=8,
                    fixed_stride=6,
                )
                self.assertEqual(actual, expected)

    def test_constant_time_selector_matches_prefix_reconstruction(self) -> None:
        from jamoflow.incremental_blt import (
            IncrementalStructuralSelector,
            structural_prefix_boundaries,
        )

        valid = "한국어 selector test 와 공백 ".encode("utf-8")
        noisy = np.random.default_rng(1729).integers(
            0,
            256,
            size=64,
            dtype=np.uint8,
        ).tobytes()
        for raw in (valid[:64], noisy):
            for policy in (
                "fixed_byte_6",
                "causal_codepoint_grid",
                "causal_whitespace_grid",
                "spacebyte_spacelike",
            ):
                selector = IncrementalStructuralSelector(
                    policy,
                    horizon=64,
                    patch_count=10,
                    fixed_stride=6,
                )
                for length, value in enumerate(raw, start=1):
                    actual = selector.consume(value)
                    expected = structural_prefix_boundaries(
                        raw[:length],
                        policy,
                        horizon=64,
                        patch_count=10,
                        fixed_stride=6,
                    )
                    self.assertEqual(actual, expected)

    def test_spacebyte_prefix_helper_matches_offline_causal_mask(self) -> None:
        from jamoflow.incremental_blt import structural_prefix_boundaries
        from jamoflow.phase3 import (
            spacebyte_boundaries,
            spacebyte_causal_prefix_mask,
        )

        raw = "한국어 SpaceByte, ASCII 123!".encode("utf-8")
        for length in range(1, len(raw) + 1):
            prefix = raw[:length]
            expected = spacebyte_boundaries(
                spacebyte_causal_prefix_mask(prefix)
            )
            actual = structural_prefix_boundaries(
                prefix,
                "spacebyte_spacelike",
                horizon=len(raw),
                patch_count=8,
                fixed_stride=6,
            )
            self.assertEqual(actual, expected)

    def test_spacebyte_incremental_logits_match_full_prefix(self) -> None:
        import torch

        from jamoflow.incremental_blt import (
            IncrementalBltDecoder,
            structural_prefix_boundaries,
        )
        from jamoflow.neural_model import build_main_model
        from jamoflow.phase2_patching import padded_hf_patch_matrix

        spec = self._spec()
        raw = "한국어 S test!".encode("utf-8")[:24]
        model = build_main_model(spec, seed=1729).eval()
        decoder = IncrementalBltDecoder(
            model,
            "spacebyte_spacelike",
            horizon=spec.sequence_length,
            patch_count=spec.patch_count,
            fixed_stride=spec.patch_stride,
        )
        with torch.inference_mode():
            for length, value in enumerate(raw, start=1):
                incremental = decoder.consume(value)
                prefix = raw[:length]
                boundaries = structural_prefix_boundaries(
                    prefix,
                    "spacebyte_spacelike",
                    horizon=spec.sequence_length,
                    patch_count=spec.patch_count,
                    fixed_stride=spec.patch_stride,
                )
                patches = padded_hf_patch_matrix([boundaries], length)
                full = model(
                    input_ids=torch.tensor([list(prefix)]),
                    patch_lengths=torch.from_numpy(
                        patches.astype(np.int64, copy=False)
                    ),
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                torch.testing.assert_close(
                    incremental,
                    full,
                    rtol=2e-5,
                    atol=2e-5,
                )

    def test_incremental_logits_match_full_prefix_at_every_byte(self) -> None:
        import torch

        from jamoflow.generation import generation_patch_matrix
        from jamoflow.incremental_blt import IncrementalBltDecoder
        from jamoflow.neural_model import build_main_model

        spec = self._spec()
        raw = "한국어 test 문장 and spaces ".encode("utf-8")[:36]
        for policy in (
            "fixed_byte_6",
            "causal_codepoint_grid",
            "causal_whitespace_grid",
        ):
            model = build_main_model(spec, seed=1729)
            model.eval()
            decoder = IncrementalBltDecoder(
                model,
                policy,
                horizon=spec.sequence_length,
                patch_count=spec.patch_count,
                fixed_stride=spec.patch_stride,
            )
            with torch.inference_mode():
                for length, value in enumerate(raw, start=1):
                    incremental = decoder.consume(value)
                    prefix = raw[:length]
                    patches = generation_patch_matrix(
                        np.asarray([list(prefix)], dtype=np.uint8),
                        policy,
                        horizon=spec.sequence_length,
                        patch_count=spec.patch_count,
                        fixed_stride=spec.patch_stride,
                    )
                    full = model(
                        input_ids=torch.tensor([list(prefix)]),
                        patch_lengths=torch.from_numpy(
                            patches.astype(np.int64, copy=False)
                        ),
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits[:, -1, :]
                    torch.testing.assert_close(
                        incremental,
                        full,
                        rtol=2e-5,
                        atol=2e-5,
                    )
            diagnostics = decoder.diagnostics
            self.assertEqual(diagnostics.observed_bytes, len(raw))
            self.assertEqual(diagnostics.local_encoder_cached_bytes, len(raw))
            self.assertEqual(diagnostics.local_decoder_cached_bytes, len(raw))
            self.assertEqual(
                diagnostics.global_cached_patches,
                diagnostics.emitted_data_patches,
            )

    def test_parallel_prefill_matches_sequential_cache_and_continuation(self) -> None:
        import torch

        from jamoflow.incremental_blt import IncrementalBltDecoder
        from jamoflow.neural_model import build_main_model

        spec = self._spec()
        prompt = "한국어 prefill 문장 ".encode("utf-8")[:24]
        continuation = "후속 bytes ".encode("utf-8")[:12]
        for policy in (
            "fixed_byte_6",
            "causal_codepoint_grid",
            "causal_whitespace_grid",
            "spacebyte_spacelike",
        ):
            sequential_model = build_main_model(spec, seed=2718)
            parallel_model = build_main_model(spec, seed=2718)
            sequential = IncrementalBltDecoder(
                sequential_model,
                policy,
                horizon=spec.sequence_length,
                patch_count=spec.patch_count,
                fixed_stride=spec.patch_stride,
            )
            parallel = IncrementalBltDecoder(
                parallel_model,
                policy,
                horizon=spec.sequence_length,
                patch_count=spec.patch_count,
                fixed_stride=spec.patch_stride,
            )
            with torch.inference_mode():
                sequential_logits = sequential.prefill(prompt)
                parallel_logits = parallel.prefill_parallel(prompt)
                torch.testing.assert_close(
                    parallel_logits,
                    sequential_logits,
                    rtol=2e-5,
                    atol=2e-5,
                )
                self.assertEqual(parallel.diagnostics, sequential.diagnostics)
                for value in continuation:
                    sequential_logits = sequential.consume(value)
                    parallel_logits = parallel.consume(value)
                    torch.testing.assert_close(
                        parallel_logits,
                        sequential_logits,
                        rtol=2e-5,
                        atol=2e-5,
                    )
                    self.assertEqual(
                        parallel.diagnostics,
                        sequential.diagnostics,
                    )
            counters = parallel.runtime_counters
            self.assertEqual(counters.parallel_prefill_calls, 1)
            self.assertEqual(counters.main_consume_calls, len(continuation))
            self.assertEqual(
                counters.selector_observed_bytes,
                len(prompt) + len(continuation),
            )
            self.assertEqual(counters.router_forward_calls, 0)
            self.assertEqual(counters.router_scored_bytes, 0)

    def test_incremental_entropy_router_matches_full_prefix(self) -> None:
        import torch

        from jamoflow.incremental_blt import IncrementalEntropyRouter
        from jamoflow.neural_model import build_router

        spec = self._spec()
        full_router = build_router(spec, seed=31415).eval()
        cached_router = build_router(spec, seed=31415).eval()
        incremental = IncrementalEntropyRouter(cached_router)
        raw = "한국어 router test ".encode("utf-8")[:30]
        with torch.inference_mode():
            for length, value in enumerate(raw, start=1):
                cached_logits, cached_entropy = incremental.consume(value)
                entropies, _, full_logits = full_router(
                    torch.tensor([list(raw[:length])]),
                    patch_size=None,
                    use_cache=False,
                )
                torch.testing.assert_close(
                    cached_logits,
                    full_logits[:, -1, :],
                    rtol=2e-5,
                    atol=2e-5,
                )
                self.assertAlmostEqual(
                    cached_entropy,
                    float(entropies[0, -1]),
                    places=5,
                )
        self.assertEqual(incremental.cache.get_seq_length(), len(raw))
        self.assertEqual(incremental.forward_calls, len(raw))
        self.assertEqual(incremental.scored_bytes, len(raw))

    def test_entropy_decoder_matches_offline_selector_and_full_model(self) -> None:
        import torch

        from jamoflow.incremental_blt import IncrementalEntropyBltDecoder
        from jamoflow.neural_model import build_main_model, build_router
        from jamoflow.phase2_patching import (
            entropy_threshold_boundaries,
            padded_hf_patch_matrix,
        )
        from jamoflow.utf8 import prefix_boundary_mask

        spec = self._spec()
        raw = "한국어 entropy 경계 test ".encode("utf-8")[:34]
        threshold = 5.45
        for policy in (
            "entropy_threshold_full",
            "entropy_threshold_codepoint",
        ):
            model = build_main_model(spec, seed=1729).eval()
            router = build_router(spec, seed=2718).eval()
            offline_router = build_router(spec, seed=2718).eval()
            decoder = IncrementalEntropyBltDecoder(
                model,
                router,
                policy,
                threshold_nats=threshold,
                maximum_patch_length=6,
                horizon=spec.sequence_length,
                patch_count=spec.patch_count,
                fixed_stride=spec.patch_stride,
            )
            with torch.inference_mode():
                for length, value in enumerate(raw, start=1):
                    incremental = decoder.consume(value)
                    prefix = raw[:length]
                    entropies, _, _ = offline_router(
                        torch.tensor([list(prefix)]),
                        patch_size=None,
                        use_cache=False,
                    )
                    aligned = np.zeros(length, dtype=np.float32)
                    if length > 1:
                        aligned[1:] = entropies[0, :-1].cpu().numpy()
                    candidates = None
                    if policy == "entropy_threshold_codepoint":
                        candidates = np.frombuffer(
                            prefix_boundary_mask(prefix)[:-1],
                            dtype=np.uint8,
                        )
                    boundaries = entropy_threshold_boundaries(
                        aligned,
                        threshold,
                        candidate_mask=candidates,
                        maximum_patch_length=6,
                    )
                    self.assertEqual(
                        decoder.diagnostics.main.boundaries,
                        boundaries,
                    )
                    patches = padded_hf_patch_matrix([boundaries], length)
                    full = model(
                        input_ids=torch.tensor([list(prefix)]),
                        patch_lengths=torch.from_numpy(
                            patches.astype(np.int64, copy=False)
                        ),
                        use_cache=False,
                        logits_to_keep=1,
                    ).logits[:, -1, :]
                    torch.testing.assert_close(
                        incremental,
                        full,
                        rtol=2e-5,
                        atol=2e-5,
                    )
            self.assertEqual(
                decoder.diagnostics.router_cached_bytes,
                len(raw),
            )

    def test_entropy_parallel_prefill_matches_sequential_continuation(self) -> None:
        import torch

        from jamoflow.incremental_blt import IncrementalEntropyBltDecoder
        from jamoflow.neural_model import build_main_model, build_router

        spec = self._spec()
        prompt = "한국어 learned prefill ".encode("utf-8")[:24]
        continuation = "후속값 ".encode("utf-8")[:10]
        for policy in (
            "entropy_threshold_full",
            "entropy_threshold_codepoint",
        ):
            sequential = IncrementalEntropyBltDecoder(
                build_main_model(spec, seed=1729),
                build_router(spec, seed=2718),
                policy,
                threshold_nats=5.45,
                maximum_patch_length=6,
                horizon=spec.sequence_length,
                patch_count=spec.patch_count,
                fixed_stride=spec.patch_stride,
            )
            parallel = IncrementalEntropyBltDecoder(
                build_main_model(spec, seed=1729),
                build_router(spec, seed=2718),
                policy,
                threshold_nats=5.45,
                maximum_patch_length=6,
                horizon=spec.sequence_length,
                patch_count=spec.patch_count,
                fixed_stride=spec.patch_stride,
            )
            with torch.inference_mode():
                sequential_logits = sequential.prefill(prompt)
                parallel_logits = parallel.prefill_parallel(prompt)
                torch.testing.assert_close(
                    parallel_logits,
                    sequential_logits,
                    rtol=2e-5,
                    atol=2e-5,
                )
                self.assertEqual(parallel.diagnostics, sequential.diagnostics)
                for value in continuation:
                    sequential_logits = sequential.consume(value)
                    parallel_logits = parallel.consume(value)
                    torch.testing.assert_close(
                        parallel_logits,
                        sequential_logits,
                        rtol=2e-5,
                        atol=2e-5,
                    )
                    self.assertEqual(
                        parallel.diagnostics,
                        sequential.diagnostics,
                    )
            counters = parallel.runtime_counters
            self.assertEqual(counters.parallel_prefill_calls, 1)
            self.assertEqual(counters.main_consume_calls, len(continuation))
            self.assertEqual(
                counters.selector_observed_bytes,
                len(prompt) + len(continuation),
            )
            self.assertEqual(
                counters.router_forward_calls,
                1 + len(continuation),
            )
            self.assertEqual(
                counters.router_scored_bytes,
                len(prompt) + len(continuation),
            )


if __name__ == "__main__":
    unittest.main()
