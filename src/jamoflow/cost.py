"""Transparent analytical FLOP accounting for the compact BLT experiment.

The counts cover dense matrix multiplications in the implemented Hugging Face
forward path. One multiply-add is two FLOPs. Embedding lookup, normalization,
RoPE, activations, reductions, softmax, masking, hashing, and Python dispatch are
reported as omitted operations rather than hidden inside a fitted constant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .neural_model import DEFAULT_MODEL_SPEC, Phase1ModelSpec


@dataclass(frozen=True, slots=True)
class FlopComponent:
    name: str
    forward_flops_per_sequence: int

    def to_dict(self, sequence_bytes: int) -> dict[str, int | float | str]:
        return {
            **asdict(self),
            "forward_flops_per_byte": self.forward_flops_per_sequence
            / sequence_bytes,
        }


def transformer_block_flops(
    sequence_length: int,
    hidden_size: int,
    intermediate_size: int,
    layers: int = 1,
) -> int:
    """Dense QKVO + SwiGLU + dense self-attention matmul FLOPs."""

    projection = 8 * sequence_length * hidden_size**2
    swiglu = 6 * sequence_length * hidden_size * intermediate_size
    attention = 4 * sequence_length**2 * hidden_size
    return layers * (projection + swiglu + attention)


def cross_attention_flops(
    query_length: int,
    key_value_length: int,
    hidden_size: int,
    layers: int = 1,
) -> int:
    """Dense QKVO projections and QK/AV products for cross-attention."""

    projections = (
        4 * query_length * hidden_size**2
        + 4 * key_value_length * hidden_size**2
    )
    attention = 4 * query_length * key_value_length * hidden_size
    return layers * (projections + attention)


def linear_flops(tokens: int, input_size: int, output_size: int) -> int:
    return 2 * tokens * input_size * output_size


def compact_blt_flops(
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
    *,
    data_patches: int | None = None,
) -> dict[str, object]:
    byte_length = spec.sequence_length
    # HF receives one initial dummy patch in addition to the 43 data patches.
    realized_data_patches = spec.patch_count if data_patches is None else data_patches
    if realized_data_patches <= 0:
        raise ValueError("data patch count must be positive")
    global_length = realized_data_patches + 1
    expanded_patch_length = global_length * spec.cross_attention_k

    components = [
        FlopComponent(
            "local_encoder_transformer",
            transformer_block_flops(
                byte_length,
                spec.local_width,
                spec.local_ffn,
                spec.encoder_layers,
            ),
        ),
        FlopComponent(
            "local_encoder_patch_projection",
            linear_flops(
                global_length,
                spec.local_width,
                spec.local_width * spec.cross_attention_k,
            ),
        ),
        FlopComponent(
            "local_encoder_cross_attention",
            cross_attention_flops(
                expanded_patch_length,
                byte_length,
                spec.local_width,
            ),
        ),
        FlopComponent(
            "encoder_to_global_projection",
            linear_flops(
                global_length,
                spec.local_width * spec.cross_attention_k,
                spec.global_width,
            ),
        ),
        FlopComponent(
            "global_transformer",
            transformer_block_flops(
                global_length,
                spec.global_width,
                spec.global_ffn,
                spec.global_layers,
            ),
        ),
        FlopComponent(
            "global_to_decoder_projection",
            linear_flops(
                global_length,
                spec.global_width,
                spec.local_width * spec.cross_attention_k,
            ),
        ),
        FlopComponent(
            "local_decoder_cross_attention",
            cross_attention_flops(
                byte_length,
                expanded_patch_length,
                spec.local_width,
                spec.decoder_layers,
            ),
        ),
        FlopComponent(
            "local_decoder_transformer",
            transformer_block_flops(
                byte_length,
                spec.local_width,
                spec.local_ffn,
                spec.decoder_layers,
            ),
        ),
        FlopComponent(
            "byte_lm_head",
            linear_flops(byte_length, spec.local_width, spec.vocab_size),
        ),
    ]
    total = sum(component.forward_flops_per_sequence for component in components)
    return {
        "method": "dense matmul forward FLOPs; multiply-add=2",
        "sequence_bytes": byte_length,
        "data_patches": realized_data_patches,
        "hf_global_positions_including_dummy": global_length,
        "components": [
            component.to_dict(byte_length) for component in components
        ],
        "forward_flops_per_sequence": total,
        "forward_flops_per_byte": total / byte_length,
        "omitted_operations": [
            "embedding and hash-table lookup",
            "rolling-hash integer arithmetic",
            "RMSNorm and RoPE",
            "activation, elementwise, residual, and reduction operations",
            "softmax, entropy, masking, and boundary construction",
            "framework dispatch and memory movement",
        ],
    }


def variable_patch_flop_summary(
    patch_counts: np.ndarray,
    *,
    batch_size: int,
    include_router: bool,
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, object]:
    """Report ideal per-row and implemented batch-max dense FLOPs."""

    counts = np.asarray(patch_counts, dtype=np.int64)
    if counts.ndim != 1 or not len(counts) or np.any(counts <= 0):
        raise ValueError("patch counts must be a non-empty positive vector")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    unique, frequencies = np.unique(counts, return_counts=True)
    main_by_count = {
        int(count): int(
            compact_blt_flops(spec, data_patches=int(count))[
                "forward_flops_per_sequence"
            ]
        )
        for count in unique
    }
    router = int(compact_router_flops(spec)["forward_flops_per_sequence"])
    router_cost = router if include_router else 0
    ideal_total = sum(
        int(frequency) * (main_by_count[int(count)] + router_cost)
        for count, frequency in zip(unique, frequencies, strict=True)
    )

    batch_maxima: list[int] = []
    implemented_total = 0
    for start in range(0, len(counts), batch_size):
        local = counts[start : start + batch_size]
        maximum = int(local.max())
        batch_maxima.append(maximum)
        implemented_total += len(local) * (main_by_count[maximum] + router_cost)

    unpadded_slots = int(counts.sum())
    padded_slots = sum(
        len(counts[start : start + batch_size]) * maximum
        for start, maximum in zip(
            range(0, len(counts), batch_size),
            batch_maxima,
            strict=True,
        )
    )
    return {
        "method": "dense matmul forward FLOPs; multiply-add=2",
        "examples": len(counts),
        "batch_size": batch_size,
        "include_router": include_router,
        "router_flops_per_sequence": router_cost,
        "mean_data_patches": float(counts.mean()),
        "minimum_data_patches": int(counts.min()),
        "maximum_data_patches": int(counts.max()),
        "ideal_unpadded_mean_flops_per_sequence": ideal_total / len(counts),
        "implemented_batch_max_mean_flops_per_sequence": (
            implemented_total / len(counts)
        ),
        "batch_padding_flop_overhead_relative_to_ideal": (
            implemented_total / ideal_total - 1
        ),
        "mean_batch_max_data_patches": float(np.mean(batch_maxima)),
        "patch_slot_padding_rate": (
            1 - unpadded_slots / padded_slots if padded_slots else 0.0
        ),
        "patch_count_histogram": {
            str(int(count)): int(frequency)
            for count, frequency in zip(unique, frequencies, strict=True)
        },
    }


def compact_router_flops(
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, object]:
    byte_length = spec.sequence_length
    transformer = transformer_block_flops(
        byte_length,
        spec.router_width,
        spec.router_ffn,
        spec.router_layers,
    )
    head = linear_flops(byte_length, spec.router_width, spec.vocab_size)
    components = [
        FlopComponent("router_transformer", transformer),
        FlopComponent("router_byte_lm_head", head),
    ]
    total = transformer + head
    return {
        "method": "dense matmul forward FLOPs; multiply-add=2",
        "sequence_bytes": byte_length,
        "components": [
            component.to_dict(byte_length) for component in components
        ],
        "forward_flops_per_sequence": total,
        "forward_flops_per_byte": total / byte_length,
        "omitted_operations": [
            "embedding lookup",
            "RMSNorm and RoPE",
            "activation, elementwise, residual, and reduction operations",
            "softmax and entropy",
            "framework dispatch and memory movement",
        ],
    }


def end_to_end_flop_summary(
    spec: Phase1ModelSpec = DEFAULT_MODEL_SPEC,
) -> dict[str, object]:
    main = compact_blt_flops(spec)
    router = compact_router_flops(spec)
    main_total = int(main["forward_flops_per_sequence"])
    router_total = int(router["forward_flops_per_sequence"])
    entropy_total = main_total + router_total
    return {
        "main_blt": main,
        "auxiliary_entropy_router": router,
        "fixed_policy_forward_flops_per_sequence": main_total,
        "entropy_policy_forward_flops_per_sequence": entropy_total,
        "router_overhead_relative_to_main": router_total / main_total,
        "router_share_of_entropy_end_to_end": router_total / entropy_total,
        "fixed_reduction_relative_to_entropy_end_to_end": router_total
        / entropy_total,
    }
