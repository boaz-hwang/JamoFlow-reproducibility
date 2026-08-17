"""Pinned compact BLT and entropy-router configurations for Phase 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any


@dataclass(frozen=True, slots=True)
class Phase1ModelSpec:
    vocab_size: int = 256
    sequence_length: int = 256
    patch_count: int = 43
    patch_stride: int = 6
    local_width: int = 64
    global_width: int = 128
    local_heads: int = 4
    global_heads: int = 4
    encoder_layers: int = 1
    global_layers: int = 4
    decoder_layers: int = 2
    local_ffn: int = 192
    global_ffn: int = 384
    cross_attention_k: int = 2
    hash_group_size: int = 3
    hash_vocabulary: int = 2048
    router_width: int = 64
    router_heads: int = 4
    router_layers: int = 2
    router_ffn: int = 192

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_MODEL_SPEC = Phase1ModelSpec()


def require_research_dependencies() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "Phase 1 requires the 'research' optional dependencies. "
            "Use Python 3.13 and install with `uv pip install -e '.[research]'`."
        ) from exc
    return torch, transformers


def set_research_seed(seed: int) -> None:
    torch, _ = require_research_dependencies()
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)


def build_blt_config(
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
    *,
    global_max_position_embeddings: int | None = None,
):
    _, transformers = require_research_dependencies()
    global_positions = (
        spec.patch_count * 2 + 8
        if global_max_position_embeddings is None
        else global_max_position_embeddings
    )
    if global_positions <= 0:
        raise ValueError("global position limit must be positive")
    return transformers.BltConfig(
        vocab_size=spec.vocab_size,
        max_position_embeddings=spec.sequence_length * 2,
        patch_in_forward=False,
        patch_size=spec.patch_stride,
        patching_mode="entropy",
        cross_attn_k=spec.cross_attention_k,
        encoder_hash_byte_group_size=[spec.hash_group_size],
        encoder_hash_byte_group_vocab=spec.hash_vocabulary,
        encoder_hash_byte_group_nb_functions=1,
        encoder_config={
            "vocab_size": spec.vocab_size,
            "hidden_size": spec.local_width,
            "hidden_size_global": spec.global_width,
            "num_attention_heads": spec.local_heads,
            "num_key_value_heads": spec.local_heads,
            "num_hidden_layers": spec.encoder_layers,
            "intermediate_size": spec.local_ffn,
            "max_position_embeddings": spec.sequence_length * 2,
            "cross_attn_k": spec.cross_attention_k,
            "cross_attn_all_layers": False,
            "dropout": 0.0,
        },
        decoder_config={
            "vocab_size": spec.vocab_size,
            "hidden_size": spec.local_width,
            "hidden_size_global": spec.global_width,
            "num_attention_heads": spec.local_heads,
            "num_key_value_heads": spec.local_heads,
            "num_hidden_layers": spec.decoder_layers,
            "intermediate_size": spec.local_ffn,
            "max_position_embeddings": spec.sequence_length * 2,
            "cross_attn_k": spec.cross_attention_k,
            "cross_attn_all_layers": True,
            "dropout": 0.0,
        },
        global_config={
            "hidden_size": spec.global_width,
            "num_attention_heads": spec.global_heads,
            "num_key_value_heads": spec.global_heads,
            "num_hidden_layers": spec.global_layers,
            "intermediate_size": spec.global_ffn,
            "max_position_embeddings": global_positions,
            "dropout": 0.0,
        },
    )


def build_main_model(
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
    seed: int | None = None,
    *,
    global_max_position_embeddings: int | None = None,
):
    torch, transformers = require_research_dependencies()
    if seed is not None:
        set_research_seed(seed)
    model = transformers.BltForCausalLM(
        build_blt_config(
            spec,
            global_max_position_embeddings=global_max_position_embeddings,
        )
    )
    return model.to(dtype=torch.float32)


def build_router_config(spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC):
    _, transformers = require_research_dependencies()
    return transformers.BltPatcherConfig(
        vocab_size=spec.vocab_size,
        hidden_size=spec.router_width,
        num_hidden_layers=spec.router_layers,
        num_attention_heads=spec.router_heads,
        num_key_value_heads=spec.router_heads,
        intermediate_size=spec.router_ffn,
        max_position_embeddings=spec.sequence_length * 2,
        dropout=0.0,
    )


def build_router(
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
    seed: int | None = None,
):
    torch, transformers = require_research_dependencies()
    if seed is not None:
        set_research_seed(seed)
    model = transformers.BltPatcher(build_router_config(spec))
    return model.to(dtype=torch.float32)


def parameter_count(model: Any, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def research_versions() -> dict[str, str | bool]:
    torch, transformers = require_research_dependencies()
    import numpy as np

    return {
        "python_torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }
