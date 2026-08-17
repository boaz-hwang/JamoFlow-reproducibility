import json
import unittest

import numpy as np

from compositional_token_head import (
    CODEBOOK_COUNT,
    CODEBOOK_SIZE,
    CompositionalVocabulary,
    LowRankVocabulary,
    audit_token_code_assignment,
    build_token_code_assignment,
    install_factorized_vocabulary,
    low_rank_for_budget,
)


class AssignmentTests(unittest.TestCase):
    def test_hangul_components_and_identity_codes_are_unique(self) -> None:
        pieces = (b"a", "한".encode(), "한국".encode(), b"\xed\x95", b" ")
        generic = build_token_code_assignment(pieces, kind="generic_unicode")
        hangul = build_token_code_assignment(pieces, kind="hangul")
        self.assertEqual(generic.shape, (len(pieces), CODEBOOK_COUNT))
        self.assertTrue(np.all(generic >= 0))
        self.assertTrue(np.all(generic < CODEBOOK_SIZE))
        self.assertEqual(len(np.unique(hangul, axis=0)), len(pieces))
        # "한": onset 18, vowel 0, coda 4.
        self.assertEqual(tuple(hangul[1, 6:9]), (18, 0, 4))
        self.assertEqual(tuple(hangul[1, 9:12]), (18, 0, 4))
        self.assertFalse(np.array_equal(generic[1, 6:12], hangul[1, 6:12]))

    def test_shuffled_control_preserves_auxiliary_histograms_by_length(self) -> None:
        pieces = tuple(
            value.encode("utf-8")
            for value in ("가", "나", "다", "라", "한국", "한글", "언어", "모델")
        )
        hangul = build_token_code_assignment(pieces, kind="hangul")
        shuffled = build_token_code_assignment(pieces, kind="shuffled_hangul")
        lengths = np.asarray([len(piece) for piece in pieces])
        for length in np.unique(lengths):
            rows = lengths == length
            for slot in range(6, 12):
                self.assertEqual(
                    sorted(hangul[rows, slot].tolist()),
                    sorted(shuffled[rows, slot].tolist()),
                )
        self.assertTrue(np.array_equal(hangul[:, :6], shuffled[:, :6]))
        self.assertTrue(np.array_equal(hangul[:, 12:], shuffled[:, 12:]))
        self.assertEqual(len(np.unique(shuffled, axis=0)), len(pieces))

    def test_assignment_audit_reconstructs_exact_definition(self) -> None:
        pieces = (b"a", b"b", "가".encode(), "나다".encode())
        assignment = build_token_code_assignment(pieces, kind="hangul")
        audit = audit_token_code_assignment(pieces, assignment, kind="hangul")
        self.assertEqual(audit.vocabulary_size, len(pieces))
        self.assertEqual(audit.unique_code_tuples, len(pieces))
        self.assertEqual(audit.hangul_surface_token_count, 2)
        self.assertEqual(audit.to_dict(), json.loads(json.dumps(audit.to_dict())))
        changed = assignment.copy()
        changed[0, 0] = (changed[0, 0] + 1) % CODEBOOK_SIZE
        with self.assertRaisesRegex(ValueError, "differs"):
            audit_token_code_assignment(pieces, changed, kind="hangul")


class FactorizedVocabularyTests(unittest.TestCase):
    def test_compositional_embedding_and_logits_equal_dense_weight(self) -> None:
        import torch
        import torch.nn.functional as functional

        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(CODEBOOK_COUNT, CODEBOOK_SIZE, 5, generator=generator)
        pieces = tuple(bytes((value,)) for value in range(17))
        assignment_np = build_token_code_assignment(pieces, kind="generic_unicode")
        vocabulary = CompositionalVocabulary.build(
            weight, torch.tensor(assignment_np, dtype=torch.long)
        )
        token_ids = torch.tensor([[0, 3, 16]], dtype=torch.long)
        hidden = torch.randn(1, 3, 5, generator=generator)
        dense = vocabulary.dense_weight()
        self.assertTrue(
            torch.allclose(vocabulary.embed(token_ids), functional.embedding(token_ids, dense))
        )
        self.assertTrue(
            torch.allclose(
                vocabulary.project(hidden),
                functional.linear(hidden, dense),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_compositional_long_sequence_path_equals_dense_weight(self) -> None:
        import torch
        import torch.nn.functional as functional

        generator = torch.Generator().manual_seed(9)
        weight = torch.randn(CODEBOOK_COUNT, CODEBOOK_SIZE, 7, generator=generator)
        pieces = tuple(bytes((value,)) for value in range(23))
        assignment = torch.tensor(
            build_token_code_assignment(pieces, kind="hangul"), dtype=torch.long
        )
        vocabulary = CompositionalVocabulary.build(weight, assignment)
        hidden = torch.randn(2, 5, 7, generator=generator)
        self.assertTrue(
            torch.allclose(
                vocabulary.project(hidden),
                functional.linear(hidden, vocabulary.dense_weight()),
                rtol=1e-5,
                atol=1e-5,
            )
        )

    def test_low_rank_embedding_and_logits_equal_dense_weight(self) -> None:
        import torch
        import torch.nn.functional as functional

        generator = torch.Generator().manual_seed(11)
        factors = torch.randn(19, 3, generator=generator)
        projection = torch.randn(3, 5, generator=generator)
        vocabulary = LowRankVocabulary.build(factors, projection)
        ids = torch.tensor([[1, 7, 18]])
        hidden = torch.randn(1, 3, 5, generator=generator)
        dense = vocabulary.dense_weight()
        self.assertTrue(torch.allclose(vocabulary.embed(ids), functional.embedding(ids, dense)))
        self.assertTrue(torch.allclose(vocabulary.project(hidden), functional.linear(hidden, dense)))

    def test_low_rank_budget_uses_nearest_integer_rank(self) -> None:
        budget = 2_048 * 384
        self.assertEqual(low_rank_for_budget(8_192, 384, budget), 92)
        self.assertEqual(low_rank_for_budget(16_000, 384, budget), 48)
        self.assertEqual(low_rank_for_budget(32_000, 384, budget), 24)

    def test_tiny_llama_forward_cache_and_loss_after_installation(self) -> None:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(13)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=2_048,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                max_position_embeddings=32,
                tie_word_embeddings=True,
            )
        )
        initial = model.get_input_embeddings().weight.detach().reshape(
            CODEBOOK_COUNT, CODEBOOK_SIZE, 32
        )
        pieces = tuple(bytes((value % 256,)) for value in range(257))
        # Make raw pieces unique even though this synthetic vocabulary exceeds bytes.
        pieces = tuple(bytes((value % 256, value // 256)) for value in range(257))
        assignment = torch.tensor(
            build_token_code_assignment(pieces, kind="hangul"), dtype=torch.long
        )
        vocabulary = CompositionalVocabulary.build(initial, assignment)
        install_factorized_vocabulary(model, vocabulary)
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        output = model(input_ids=ids, labels=ids, use_cache=True)
        self.assertEqual(tuple(output.logits.shape), (1, 4, 257))
        self.assertTrue(torch.isfinite(output.loss))
        next_output = model(
            input_ids=torch.tensor([[5]]),
            past_key_values=output.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        self.assertEqual(tuple(next_output.logits.shape), (1, 1, 257))
        parameter_ids = [id(parameter) for parameter in model.parameters()]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_tiny_llama_low_rank_installation(self) -> None:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(17)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
                tie_word_embeddings=True,
            )
        )
        factors = torch.randn(41, 5)
        projection = torch.randn(5, 16)
        install_factorized_vocabulary(
            model, LowRankVocabulary.build(factors, projection)
        )
        values = torch.tensor([[1, 2, 3]], dtype=torch.long)
        output = model(input_ids=values, labels=values, use_cache=False)
        self.assertEqual(tuple(output.logits.shape), (1, 3, 41))
        self.assertTrue(torch.isfinite(output.loss))


if __name__ == "__main__":
    unittest.main()
