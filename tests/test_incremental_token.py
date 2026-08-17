import importlib.util
import unittest


HAS_RESEARCH_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@unittest.skipUnless(HAS_RESEARCH_DEPS, "optional neural research dependencies")
class IncrementalTokenTests(unittest.TestCase):
    def _model(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(1729)
        return LlamaForCausalLM(
            LlamaConfig(
                vocab_size=300,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=32,
                attention_bias=False,
                mlp_bias=False,
                tie_word_embeddings=True,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=0,
            )
        ).eval()

    def test_cached_logits_match_every_full_prefix_and_parallel_prefill(self) -> None:
        from jamoflow.incremental_token import verify_token_incremental_equivalence

        result = verify_token_incremental_equivalence(
            self._model(),
            (7, 19, 41, 3, 299, 17),
        )
        self.assertTrue(result["overall_pass"])
        self.assertTrue(result["all_argmax_equal"])
        self.assertEqual(result["cached_tokens"], 6)
        self.assertEqual(result["parallel_cached_tokens"], 6)
        self.assertLess(result["maximum_absolute_error"], 2e-5)

    def test_runtime_rejects_unprefilled_invalid_or_overlong_inputs(self) -> None:
        from jamoflow.incremental_token import IncrementalTokenDecoder

        runtime = IncrementalTokenDecoder(self._model())
        with self.assertRaisesRegex(RuntimeError, "prefilled"):
            runtime.consume(1)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            runtime.prefill_parallel((300,))
        with self.assertRaisesRegex(ValueError, "context"):
            runtime.prefill_parallel(tuple(range(33)))

    def test_reset_removes_cache_and_observed_count(self) -> None:
        from jamoflow.incremental_token import IncrementalTokenDecoder

        runtime = IncrementalTokenDecoder(self._model())
        runtime.prefill_parallel((1, 2, 3))
        self.assertEqual(runtime.diagnostics.cached_tokens, 3)
        runtime.reset()
        self.assertEqual(runtime.diagnostics.cached_tokens, 0)
        self.assertEqual(runtime.diagnostics.observed_tokens, 0)


if __name__ == "__main__":
    unittest.main()
