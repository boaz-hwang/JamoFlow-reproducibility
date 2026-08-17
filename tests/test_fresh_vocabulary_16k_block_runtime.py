from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from fresh_vocabulary_16k_block_runtime import (
    IncrementalBpeBlockDecoder,
    verify_block_sequence,
)
from transformers import LlamaConfig, LlamaForCausalLM


def _tiny_bundle() -> SimpleNamespace:
    torch.manual_seed(20260815)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            attention_dropout=0.0,
        )
    ).eval()
    return SimpleNamespace(model=model)


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_block_cache_matches_full_forward_at_every_position(block_size: int) -> None:
    bundle = _tiny_bundle()
    result = verify_block_sequence(
        bundle,
        (1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10, 11, 12, 13, 14),
        block_size=block_size,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result["pass"] is True
    assert result["comparisons"] == 9
    assert result["decode_calls"] == (8 + block_size - 1) // block_size


def test_block_decoder_rejects_invalid_state_transitions() -> None:
    runtime = IncrementalBpeBlockDecoder(_tiny_bundle().model)
    with pytest.raises(RuntimeError, match="prefixed"):
        runtime.consume_block((1, 2))
    runtime.prefill_parallel((1, 2, 3))
    with pytest.raises(RuntimeError, match="prefixed"):
        runtime.consume_block(())
