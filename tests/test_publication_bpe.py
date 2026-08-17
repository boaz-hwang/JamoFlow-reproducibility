import importlib.util
import unittest

from jamoflow.publication_bpe import (
    PINNED_TOKENIZERS_VERSION,
    PUBLICATION_BPE_MODEL_SPECS,
    PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY,
    PUBLICATION_BPE_STRESS_MODEL_SPECS,
    PUBLICATION_PARAMETER_TOLERANCE,
    analytical_token_transformer_parameters,
    audit_byte_bpe_tokenizer,
    byte_bpe_token_bytes,
    body_matched_bpe_spec,
    derive_parameter_matched_bpe_spec,
    compile_byte_bpe_utf8_transitions,
    parameter_match_fraction,
    prepare_byte_bpe_replay,
    publication_bpe_spec,
    train_byte_bpe_tokenizer,
    validate_publication_bpe_specs,
)
from jamoflow.publication_bpb import build_publication_bpb_document_plan
from jamoflow.publication_protocol import (
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_BPE_VOCABULARY_CANDIDATES,
)
from jamoflow.publication_scale import PUBLICATION_FAMILY_EXPECTED_PARAMETERS


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _BoundaryMergingTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return _Encoding({"a": [0], "b": [1], "ab": [2]}[text])

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return {(0,): "a", (1,): "b", (2,): "ab", (0, 1): "ab"}[
            tuple(ids)
        ]


class _StableBoundaryTokenizer(_BoundaryMergingTokenizer):
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return _Encoding({"a": [0], "b": [1], "ab": [0, 1]}[text])


class _BrokenRoundtripTokenizer(_BoundaryMergingTokenizer):
    def decode(self, ids, skip_special_tokens=False):
        del ids, skip_special_tokens
        return "broken"


class ByteBPEPromptBoundaryTests(unittest.TestCase):
    def test_api_replay_preserves_raw_text_when_joint_bpe_crosses_boundary(self) -> None:
        plan = prepare_byte_bpe_replay(
            _BoundaryMergingTokenizer(),
            "a",
            "b",
        )
        self.assertEqual(plan.prompt_token_ids, (0,))
        self.assertEqual(plan.continuation_token_ids, (1,))
        self.assertTrue(plan.audit.overall_pass)
        self.assertFalse(plan.audit.joint_has_exact_prompt_token_boundary)
        self.assertFalse(plan.audit.separate_and_joint_token_ids_identical)

    def test_stable_boundary_is_identified(self) -> None:
        plan = prepare_byte_bpe_replay(
            _StableBoundaryTokenizer(),
            "a",
            "b",
        )
        self.assertTrue(plan.audit.overall_pass)
        self.assertTrue(plan.audit.joint_has_exact_prompt_token_boundary)
        self.assertTrue(plan.audit.separate_and_joint_token_ids_identical)

    def test_roundtrip_failure_blocks_replay_audit(self) -> None:
        plan = prepare_byte_bpe_replay(
            _BrokenRoundtripTokenizer(),
            "a",
            "b",
        )
        self.assertFalse(plan.audit.overall_pass)


HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_TOKENIZERS, "optional tokenizer research dependency")
class PublicationByteBPETests(unittest.TestCase):
    corpus = (
        "한국어 문장과 공백을 그대로 보존한다.\n",
        "ASCII 0123456789, code(x), URL-like/path?q=1\n",
        "혼합 문서🙂와 한글, English를 함께 둔다.\n",
    ) * 20

    def test_synthetic_tokenizer_is_reversible_and_structurally_exact(self) -> None:
        tokenizer = train_byte_bpe_tokenizer(
            self.corpus,
            vocabulary_size=300,
            minimum_frequency=1,
        )
        audit = audit_byte_bpe_tokenizer(
            tokenizer,
            self.corpus[:3] + ("\x00",),
            expected_vocabulary_size=300,
        )
        self.assertEqual(audit.tokenizers_version, PINNED_TOKENIZERS_VERSION)
        self.assertTrue(audit.overall_pass)
        self.assertTrue(audit.full_byte_alphabet)
        self.assertTrue(audit.digit_labels_single_token)
        self.assertTrue(audit.roundtrip_identity)

    def test_training_is_deterministic_for_identical_stream(self) -> None:
        first = train_byte_bpe_tokenizer(
            self.corpus,
            vocabulary_size=280,
            minimum_frequency=1,
        )
        second = train_byte_bpe_tokenizer(
            self.corpus,
            vocabulary_size=280,
            minimum_frequency=1,
        )
        self.assertEqual(first.to_str(pretty=False), second.to_str(pretty=False))

    def test_non_nfc_training_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NFC"):
            train_byte_bpe_tokenizer(
                ("가", "가"),
                vocabulary_size=256,
                minimum_frequency=1,
            )

    def test_token_raw_bytes_and_utf8_table_cover_the_real_vocabulary(self) -> None:
        tokenizer = train_byte_bpe_tokenizer(
            self.corpus,
            vocabulary_size=300,
            minimum_frequency=1,
        )
        token_bytes = byte_bpe_token_bytes(tokenizer)
        table = compile_byte_bpe_utf8_transitions(tokenizer)
        self.assertEqual(len(token_bytes), 300)
        self.assertEqual(table.vocabulary_size, 300)
        self.assertEqual(
            {values for values in token_bytes if len(values) == 1},
            {bytes((value,)) for value in range(256)},
        )
        self.assertTrue(all(table.allowed_token_ids))
        self.assertGreaterEqual(table.maximum_token_bytes, 1)
        self.assertEqual(len(table.token_bytes_sha256), 64)
        self.assertEqual(len(table.transition_table_sha256), 64)

    def test_real_bytelevel_bpe_units_form_a_raw_capped_rolling_plan(self) -> None:
        tokenizer = train_byte_bpe_tokenizer(
            self.corpus,
            vocabulary_size=300,
            minimum_frequency=1,
        )
        document = ("한국어 문장과 ASCII 123을 함께 평가한다. " * 30).encode(
            "utf-8"
        )
        token_bytes = byte_bpe_token_bytes(tokenizer)
        token_ids = tokenizer.encode(
            document.decode("utf-8"),
            add_special_tokens=False,
        ).ids
        units = tuple(token_bytes[token_id] for token_id in token_ids)
        self.assertEqual(b"".join(units), document)
        plan = build_publication_bpb_document_plan(
            document,
            comparator_key=PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
            comparator_token_ids=token_ids,
            comparator_token_bytes=units,
        )
        self.assertGreaterEqual(plan.excluded_prefix_bytes, len(units[0]))
        self.assertEqual(
            plan.scored_bytes,
            len(document) - plan.excluded_prefix_bytes,
        )
        self.assertEqual(plan.comparator_tokens, len(token_ids))
        self.assertTrue(all(window.source_bytes <= 512 for window in plan.windows))
        self.assertTrue(
            all(
                offset == len(document)
                or document[offset] & 0xC0 != 0x80
                for window in plan.windows
                for offset in (
                    window.context_start_byte,
                    window.target_start_byte,
                    window.target_end_byte,
                )
            )
        )
        self.assertTrue(
            all(
                0 <= window.context_start_token
                < window.target_start_token
                < window.target_end_token
                <= len(token_ids)
                for window in plan.windows
            )
        )


class PublicationBPEModelSpecTests(unittest.TestCase):
    def test_32k_is_parameter_matched_and_16k_is_body_matched(self) -> None:
        validate_publication_bpe_specs()
        for vocabulary_size, specs in (
            PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY.items()
        ):
            for target, spec in specs.items():
                if vocabulary_size == 32_000:
                    self.assertLessEqual(
                        parameter_match_fraction(
                            target,
                            vocabulary_size=vocabulary_size,
                        ),
                        PUBLICATION_PARAMETER_TOLERANCE,
                    )
                    self.assertEqual(
                        spec,
                        derive_parameter_matched_bpe_spec(
                            target,
                            vocabulary_size,
                        ),
                    )
                else:
                    self.assertEqual(
                        spec,
                        body_matched_bpe_spec(
                            PUBLICATION_BPE_MODEL_SPECS[target],
                            vocabulary_size,
                        ),
                    )
                self.assertEqual(
                    spec.expected_parameters,
                    analytical_token_transformer_parameters(
                        vocabulary_size=vocabulary_size,
                        hidden_size=spec.hidden_size,
                        intermediate_size=spec.intermediate_size,
                        layers=spec.layers,
                    ),
                )

    def test_result_blind_rules_freeze_both_vocabulary_geometries(self) -> None:
        expected = {
            16_000: {
                50: (448, 1_600, 7, 42_617_792),
                75: (608, 1_792, 8, 66_710_368),
                100: (704, 2_048, 11, 86_975_680),
            },
            32_000: {
                50: (448, 1_600, 7, 49_785_792),
                75: (608, 1_792, 8, 76_438_368),
                100: (704, 2_048, 11, 98_239_680),
            },
        }
        self.assertEqual(
            set(PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY),
            set(PUBLICATION_BPE_VOCABULARY_CANDIDATES),
        )
        for vocabulary_size, targets in expected.items():
            for target, geometry in targets.items():
                spec = publication_bpe_spec(
                    target,
                    vocabulary_size=vocabulary_size,
                )
                self.assertEqual(
                    (
                        spec.hidden_size,
                        spec.intermediate_size,
                        spec.attention_heads,
                        spec.expected_parameters,
                    ),
                    geometry,
                )
                family = (
                    "byte_bpe_16000_body_matched"
                    if vocabulary_size == 16_000
                    else "byte_bpe_32000"
                )
                self.assertEqual(
                    geometry[-1],
                    PUBLICATION_FAMILY_EXPECTED_PARAMETERS[target][family],
                )
        self.assertIs(
            PUBLICATION_BPE_MODEL_SPECS,
            PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY[32_000],
        )
        self.assertIs(
            PUBLICATION_BPE_STRESS_MODEL_SPECS,
            PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY[16_000],
        )

    def test_unknown_scale_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            publication_bpe_spec(60)
        with self.assertRaisesRegex(ValueError, "unknown"):
            publication_bpe_spec(50, vocabulary_size=8_000)
        with self.assertRaisesRegex(ValueError, "only the primary 32K"):
            derive_parameter_matched_bpe_spec(50, 16_000)

    @unittest.skipUnless(HAS_TRANSFORMERS, "optional neural research dependency")
    def test_exact_parameter_counts_match_transformers_graph(self) -> None:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        for specs in PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY.values():
            for spec in specs.values():
                with torch.device("meta"):
                    model = LlamaForCausalLM(
                        LlamaConfig(
                            vocab_size=spec.vocabulary_size,
                            hidden_size=spec.hidden_size,
                            intermediate_size=spec.intermediate_size,
                            num_hidden_layers=spec.layers,
                            num_attention_heads=spec.attention_heads,
                            num_key_value_heads=spec.key_value_heads,
                            max_position_embeddings=spec.maximum_positions,
                            attention_bias=False,
                            mlp_bias=False,
                            tie_word_embeddings=True,
                        )
                    )
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    spec.expected_parameters,
                )


if __name__ == "__main__":
    unittest.main()
