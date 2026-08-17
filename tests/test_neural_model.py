import importlib.util
import unittest


HAS_RESEARCH_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_RESEARCH_DEPS, "optional neural research dependencies")
class NeuralModelTests(unittest.TestCase):
    def test_compact_blt_forward_and_parameter_count(self) -> None:
        import torch

        from jamoflow.neural_model import (
            DEFAULT_MODEL_SPEC,
            build_main_model,
            build_router,
            parameter_count,
        )
        from jamoflow.neural_patching import (
            fixed_byte_boundaries,
            hf_patch_lengths,
        )

        model = build_main_model(seed=7)
        router = build_router(seed=7)
        inputs = torch.randint(0, 256, (2, 256))
        lengths = hf_patch_lengths(fixed_byte_boundaries(), 256)
        patch_lengths = torch.tensor([lengths, lengths])

        output = model(
            input_ids=inputs,
            patch_lengths=patch_lengths,
            labels=inputs,
            use_cache=False,
        )
        manual_loss = torch.nn.functional.cross_entropy(
            output.logits[:, :-1, :].reshape(-1, 256),
            inputs[:, 1:].reshape(-1),
        )

        self.assertEqual(output.logits.shape, (2, 256, 256))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertTrue(torch.allclose(output.loss, manual_loss))
        self.assertEqual(parameter_count(model), 1_251_136)
        self.assertLess(parameter_count(router), parameter_count(model))
        self.assertEqual(DEFAULT_MODEL_SPEC.patch_count, 43)

    def test_global_position_override_does_not_change_parameters(self) -> None:
        from jamoflow.neural_model import build_main_model, parameter_count

        default = build_main_model(seed=7)
        expanded = build_main_model(
            seed=7,
            global_max_position_embeddings=520,
        )
        self.assertEqual(parameter_count(default), parameter_count(expanded))
        self.assertEqual(
            expanded.config.global_config.max_position_embeddings,
            520,
        )


if __name__ == "__main__":
    unittest.main()
