from __future__ import annotations

import importlib.util
import unittest

import numpy as np


HAS_RESEARCH_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_RESEARCH_DEPS, "optional neural research dependencies")
class NeuralCausalityTests(unittest.TestCase):
    def test_hf_dummy_patch_shifts_encoder_grouping_by_one_byte(self) -> None:
        import torch

        from jamoflow.neural_model import build_main_model
        from jamoflow.neural_patching import hf_patch_lengths

        boundaries = (0, 6, 12, 18)
        lengths = torch.tensor(
            [hf_patch_lengths(boundaries, 24)], dtype=torch.long
        )
        model = build_main_model(seed=7)
        encoder_ids = model.model._patch_ids_from_lengths(lengths, 24)
        decoder_ids = model.model._patch_ids_from_lengths(lengths[:, 1:], 24)

        self.assertEqual(encoder_ids[0, :8].tolist(), [0, 1, 1, 1, 1, 1, 1, 2])
        self.assertEqual(decoder_ids[0, :8].tolist(), [0, 0, 0, 0, 0, 0, 1, 1])
        for boundary in boundaries[1:]:
            self.assertNotEqual(
                int(decoder_ids[0, boundary - 1]),
                int(decoder_ids[0, boundary]),
            )
            self.assertNotEqual(
                int(encoder_ids[0, boundary]),
                int(encoder_ids[0, boundary + 1]),
            )

    def test_future_bytes_and_boundaries_do_not_change_prefix_logits(self) -> None:
        import torch

        from jamoflow.neural_model import DEFAULT_MODEL_SPEC, build_main_model
        from jamoflow.phase2_patching import (
            causal_window_grid_trace,
            compact_whitespace_mask,
            padded_hf_patch_matrix,
        )
        from jamoflow.utf8 import prefix_boundary_mask

        prefix_length = 160
        prefix = (b"korean byte model boundary test " * 8)[:prefix_length]
        first = prefix + (b"suffix with many spaces " * 8)
        second = prefix + (b"XXXXXXXXXXXXXXXXXXXXXX" * 8)
        first = first[: DEFAULT_MODEL_SPEC.sequence_length]
        second = second[: DEFAULT_MODEL_SPEC.sequence_length]
        self.assertEqual(len(first), DEFAULT_MODEL_SPEC.sequence_length)
        self.assertEqual(len(second), DEFAULT_MODEL_SPEC.sequence_length)

        def patches(data: bytes) -> tuple[np.ndarray, tuple[int, ...]]:
            boundary_mask = np.frombuffer(
                prefix_boundary_mask(data)[:-1], dtype=np.uint8
            )
            whitespace = compact_whitespace_mask(data)
            trace = causal_window_grid_trace(
                boundary_mask,
                whitespace,
                DEFAULT_MODEL_SPEC.patch_count,
            )
            return padded_hf_patch_matrix(
                [trace.boundaries], DEFAULT_MODEL_SPEC.sequence_length
            ), trace.boundaries

        first_patches, first_boundaries = patches(first)
        second_patches, second_boundaries = patches(second)
        self.assertEqual(
            tuple(value for value in first_boundaries if value < prefix_length),
            tuple(value for value in second_boundaries if value < prefix_length),
        )

        model = build_main_model(seed=19)
        model.eval()
        with torch.inference_mode():
            first_logits = model(
                input_ids=torch.tensor([list(first)]),
                patch_lengths=torch.from_numpy(first_patches.astype(np.int64)),
                use_cache=False,
            ).logits
            second_logits = model(
                input_ids=torch.tensor([list(second)]),
                patch_lengths=torch.from_numpy(second_patches.astype(np.int64)),
                use_cache=False,
            ).logits
        torch.testing.assert_close(
            first_logits[:, :prefix_length, :],
            second_logits[:, :prefix_length, :],
            rtol=1e-5,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
