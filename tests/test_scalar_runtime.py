import importlib.util
import unittest

import torch

from scripts.scalar_runtime_core import (
    BPE_PRIMARY_SPEC,
    BPE_SECONDARY_SPEC,
    FactorizedUnitBlt,
    IncrementalBpeDecoder,
    IncrementalUnitBltDecoder,
    REPRESENTATIONS,
    build_bpe_model,
    decode_units,
    encode_units,
    maximum_normalized_error,
    model_parameter_count,
    unit_symbol_id,
    w72_unit_boundaries,
)


HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TRANSFORMERS, "optional neural research dependency")
class ScalarRuntimeTest(unittest.TestCase):
    def test_encodings_are_canonical_reversible_and_namespaced(self):
        raw = "가Aé😀각".encode("utf-8")
        generic = encode_units(raw, "generic_unicode_scalar")
        hybrid = encode_units(raw, "hangul_hybrid")
        self.assertEqual(decode_units(generic), raw)
        self.assertEqual(decode_units(hybrid), raw)
        self.assertEqual(len(generic), 5)
        self.assertEqual(len(hybrid), 9)
        self.assertEqual([unit.kind for unit in hybrid].count("hangul"), 2)
        self.assertEqual(generic[0].primary_target, "가".encode()[0])
        self.assertEqual(hybrid[0].primary_target, 256)
        self.assertEqual(hybrid[-1].conditional_targets, (0, 1))
        self.assertGreater(
            unit_symbol_id(generic[0], "generic_unicode_scalar"),
            255,
        )
        self.assertEqual(
            unit_symbol_id(hybrid[1], "hangul_hybrid"),
            ord("A"),
        )
        with self.assertRaisesRegex(ValueError, "strict complete UTF-8"):
            encode_units(b"\xed\x95", "generic_unicode_scalar")

    def test_fixed_route_sampling_executes_every_conditional_dependency(self):
        generic = FactorizedUnitBlt("generic_unicode_scalar").eval()
        hybrid = FactorizedUnitBlt("hangul_hybrid").eval()
        hidden = torch.zeros((1, 1, 192), dtype=torch.float32)
        generic_units = encode_units("A가😀".encode(), "generic_unicode_scalar")
        hybrid_units = encode_units("A가".encode(), "hangul_hybrid")
        self.assertEqual(
            [len(generic.sample_fixed_target_route(hidden, unit)) for unit in generic_units],
            [1, 3, 4],
        )
        self.assertEqual(
            [len(hybrid.sample_fixed_target_route(hidden, unit)) for unit in hybrid_units],
            [1, 3],
        )

    def test_parameter_matched_graphs_are_within_quarter_percent(self):
        target = 19_596_096
        counts = {
            "generic": model_parameter_count(
                FactorizedUnitBlt("generic_unicode_scalar")
            ),
            "hybrid": model_parameter_count(FactorizedUnitBlt("hangul_hybrid")),
            "bpe32": model_parameter_count(build_bpe_model(BPE_PRIMARY_SPEC)),
            "bpe16": model_parameter_count(build_bpe_model(BPE_SECONDARY_SPEC)),
        }
        self.assertEqual(counts["generic"], 19_632_960)
        self.assertEqual(counts["hybrid"], 19_609_152)
        self.assertEqual(counts["bpe32"], 19_593_984)
        self.assertEqual(counts["bpe16"], 19_595_200)
        for count in counts.values():
            self.assertLessEqual(abs(count / target - 1.0), 0.0025)

    def test_parallel_and_incremental_unit_graphs_match_full_graph(self):
        raw = "한국어 A와 B를 검증한다.".encode("utf-8")
        for representation in REPRESENTATIONS:
            with self.subTest(representation=representation):
                units = encode_units(raw, representation)
                split = len(units) - 3
                model = FactorizedUnitBlt(representation).eval()
                prompt = units[:split]
                prompt_boundaries = w72_unit_boundaries(prompt)
                with torch.inference_mode():
                    runtime = IncrementalUnitBltDecoder(model)
                    actual = runtime.prefill_parallel(prompt, prompt_boundaries)
                    expected = model.full_hidden(prompt, prompt_boundaries)[:, -1:]
                    self.assertLessEqual(
                        maximum_normalized_error(
                            actual,
                            expected,
                            rtol=2e-5,
                            atol=2e-5,
                        ),
                        1.0,
                    )
                    observed = list(prompt)
                    for unit in units[split:]:
                        position = len(observed)
                        prospective = tuple(observed) + (unit,)
                        boundaries = w72_unit_boundaries(prospective)
                        actual = runtime.consume(
                            unit,
                            boundary=position in boundaries,
                        )
                        observed.append(unit)
                        expected = model.full_hidden(observed, boundaries)[:, -1:]
                        self.assertLessEqual(
                            maximum_normalized_error(
                                actual,
                                expected,
                                rtol=2e-5,
                                atol=2e-5,
                            ),
                            1.0,
                        )

    def test_bpe_cache_helper_matches_full_tiny_llama(self):
        spec = {
            "vocabulary_size": 280,
            "hidden_size": 64,
            "intermediate_size": 96,
            "layers": 2,
            "attention_heads": 2,
            "key_value_heads": 2,
            "maximum_positions": 64,
        }
        model = build_bpe_model(spec).eval()
        prompt = [1, 4, 7, 9]
        continuation = [3, 8, 2]
        with torch.inference_mode():
            full = model(
                input_ids=torch.tensor([prompt + continuation]),
                use_cache=False,
            ).logits.float()
            runtime = IncrementalBpeDecoder(model)
            actual = runtime.prefill_parallel(prompt)
            torch.testing.assert_close(actual, full[:, len(prompt) - 1], rtol=2e-5, atol=2e-5)
            for offset, token in enumerate(continuation[:-1]):
                actual = runtime.consume(token)
                torch.testing.assert_close(
                    actual,
                    full[:, len(prompt) + offset],
                    rtol=2e-5,
                    atol=2e-5,
                )


if __name__ == "__main__":
    unittest.main()
