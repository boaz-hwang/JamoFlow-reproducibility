import unittest

import torch

from scripts.conditional_local import (
    conditional_easy_mask,
    hangul_prefix_mask,
    install_conditional_local,
    utf8_incomplete_mask,
)
from jamoflow.neural_model import Phase1ModelSpec, build_main_model, parameter_count
from jamoflow.neural_patching import hf_patch_lengths


class ConditionalLocalRouteTests(unittest.TestCase):
    def test_utf8_route_uses_only_consumed_incomplete_scalar_state(self):
        text = "A한é😀B".encode("utf-8")
        values = torch.tensor([list(text)], dtype=torch.long)
        actual = utf8_incomplete_mask(values)[0].tolist()
        # ASCII is hard.  A 3-byte Hangul scalar has two easy prefix positions,
        # a 2-byte scalar has one, and a 4-byte scalar has three.
        expected = [False, True, True, False, True, False, True, True, True, False, False]
        self.assertEqual(actual, expected)

    def test_hangul_route_is_a_causal_subset_of_utf8_route(self):
        text = "A한힣界é".encode("utf-8")
        values = torch.tensor([list(text)], dtype=torch.long)
        hangul = hangul_prefix_mask(values)
        generic = conditional_easy_mask(values, "utf8_incomplete")
        self.assertTrue(torch.all(~hangul | generic))
        # Both known Hangul syllables expose two prefix-only easy positions.
        self.assertEqual(int(hangul.sum()), 4)

    def test_invalid_or_unknown_policy_is_fail_closed(self):
        invalid = torch.tensor([[0xE1, 0x41]], dtype=torch.long)
        # The lead is a valid open prefix, but the invalid continuation is hard.
        self.assertEqual(utf8_incomplete_mask(invalid).tolist(), [[True, False]])
        with self.assertRaisesRegex(ValueError, "unsupported conditional route"):
            conditional_easy_mask(invalid, "future_byte_oracle")


class ConditionalLocalModelTests(unittest.TestCase):
    @staticmethod
    def _spec():
        return Phase1ModelSpec(
            sequence_length=16,
            patch_count=4,
            patch_stride=4,
            local_width=16,
            global_width=32,
            local_heads=4,
            global_heads=4,
            encoder_layers=2,
            global_layers=2,
            decoder_layers=2,
            local_ffn=32,
            global_ffn=64,
            cross_attention_k=1,
            hash_group_size=2,
            hash_vocabulary=64,
        )

    def test_install_is_parameter_neutral_and_rejects_double_install(self):
        model = build_main_model(self._spec(), seed=17)
        before = parameter_count(model)
        install_conditional_local(
            model,
            route_policy="utf8_incomplete",
            operator="second_mlp",
            components="encoder_decoder",
        )
        self.assertEqual(parameter_count(model), before)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_conditional_local(
                model,
                route_policy="utf8_incomplete",
                operator="second_mlp",
                components="encoder_decoder",
            )

    def test_ascii_all_hard_graph_matches_the_original_logits(self):
        spec = self._spec()
        baseline = build_main_model(spec, seed=23).eval()
        candidate = build_main_model(spec, seed=23).eval()
        install_conditional_local(
            candidate,
            route_policy="utf8_incomplete",
            operator="second_mlp",
            components="encoder_decoder",
        )
        values = torch.tensor([list(b"abcdefghijklmnop")], dtype=torch.long)
        patches = torch.tensor(
            [hf_patch_lengths((0, 4, 8, 12), 16)], dtype=torch.long
        )
        with torch.inference_mode():
            expected = baseline(
                input_ids=values, patch_lengths=patches, use_cache=False
            ).logits
            actual = candidate(
                input_ids=values, patch_lengths=patches, use_cache=False
            ).logits
        self.assertTrue(torch.equal(actual, expected))

    def test_easy_positions_change_the_conditional_graph(self):
        spec = self._spec()
        baseline = build_main_model(spec, seed=29).eval()
        candidate = build_main_model(spec, seed=29).eval()
        install_conditional_local(
            candidate,
            route_policy="utf8_incomplete",
            operator="second_layer_kv",
            components="encoder_decoder",
        )
        data = "한글ABCD".encode("utf-8")
        data = (data + b"0123456789abcdef")[:16]
        values = torch.tensor([list(data)], dtype=torch.long)
        patches = torch.tensor(
            [hf_patch_lengths((0, 4, 8, 12), 16)], dtype=torch.long
        )
        with torch.inference_mode():
            expected = baseline(
                input_ids=values, patch_lengths=patches, use_cache=False
            ).logits
            actual = candidate(
                input_ids=values, patch_lengths=patches, use_cache=False
            ).logits
        self.assertEqual(actual.shape, expected.shape)
        self.assertFalse(torch.equal(actual, expected))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
