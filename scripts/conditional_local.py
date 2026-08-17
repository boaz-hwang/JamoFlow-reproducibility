"""Deterministic orthographic routing for prospective conditional BLT local layers.

The route is computed from bytes that have already been consumed.  It never
uses the next-byte target, model logits, or a learned router.  Training and
parallel prefill retain dense kernels and mask residual updates; the matching
incremental runtime can later replace skipped positions with cheaper cache
updates without changing the conditional model's definition.
"""

from __future__ import annotations

from types import MethodType
from typing import Any


ROUTE_POLICIES = (
    "utf8_incomplete",
    "hangul_prefix",
)
CONDITIONAL_OPERATORS = (
    "second_mlp",
    "second_layer_kv",
)
CONDITIONAL_COMPONENTS = (
    "decoder",
    "encoder_decoder",
)


def _require_byte_matrix(input_ids: Any) -> Any:
    import torch

    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.dtype != torch.long
        or input_ids.numel() == 0
        or bool(torch.any(input_ids < 0))
        or bool(torch.any(input_ids > 255))
    ):
        raise ValueError("conditional local routing requires a nonempty byte matrix")
    return input_ids


def utf8_incomplete_mask(input_ids: Any) -> Any:
    """Return positions whose consumed prefix ends inside a UTF-8 scalar.

    The corpus contract is strict UTF-8.  Invalid local byte patterns are
    conservatively routed to the full path rather than treated as easy.
    """

    values = _require_byte_matrix(input_ids)
    lead2 = (0xC2 <= values) & (values <= 0xDF)
    lead3 = (0xE0 <= values) & (values <= 0xEF)
    lead4 = (0xF0 <= values) & (values <= 0xF4)
    continuation = (0x80 <= values) & (values <= 0xBF)
    easy = lead2 | lead3 | lead4
    if values.shape[1] >= 2:
        easy[:, 1:] |= continuation[:, 1:] & (lead3[:, :-1] | lead4[:, :-1])
    if values.shape[1] >= 3:
        easy[:, 2:] |= (
            continuation[:, 2:]
            & continuation[:, 1:-1]
            & lead4[:, :-2]
        )
    return easy


def hangul_prefix_mask(input_ids: Any) -> Any:
    """Return causal byte prefixes that can still form U+AC00--U+D7A3.

    A lead byte in EA--ED is a possible precomposed-Hangul prefix.  After the
    second byte, the exact prefix range is narrowed without reading the final
    continuation byte.  The route therefore remains prefix-only.
    """

    values = _require_byte_matrix(input_ids)
    easy = (0xEA <= values) & (values <= 0xED)
    if values.shape[1] < 2:
        return easy
    lead = values[:, :-1]
    second = values[:, 1:]
    compatible = (
        ((lead == 0xEA) & (0xB0 <= second) & (second <= 0xBF))
        | (((lead == 0xEB) | (lead == 0xEC)) & (0x80 <= second) & (second <= 0xBF))
        | ((lead == 0xED) & (0x80 <= second) & (second <= 0x9E))
    )
    easy[:, 1:] |= compatible
    return easy


def conditional_easy_mask(input_ids: Any, route_policy: str) -> Any:
    if route_policy == "utf8_incomplete":
        return utf8_incomplete_mask(input_ids)
    if route_policy == "hangul_prefix":
        return hangul_prefix_mask(input_ids)
    raise ValueError(f"unsupported conditional route policy: {route_policy}")


def _masked_second_layer(
    layer: Any,
    hidden_states: Any,
    easy_mask: Any,
    *,
    operator: str,
    position_embeddings: Any,
    attention_mask: Any,
    past_key_values: Any,
    kwargs: dict[str, Any],
) -> Any:
    """Apply one conditional layer with dense training/prefill semantics."""

    import torch

    expanded = easy_mask.unsqueeze(-1)
    if operator == "second_layer_kv":
        full = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        return torch.where(expanded, hidden_states, full)
    if operator != "second_mlp":
        raise ValueError(f"unsupported conditional operator: {operator}")

    residual = hidden_states
    normalized = layer.input_layernorm(hidden_states)
    attention_output, _ = layer.self_attn(
        hidden_states=normalized,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    attention_residual = residual + attention_output
    mlp_output = layer.mlp(layer.post_attention_layernorm(attention_residual))
    full = attention_residual + mlp_output
    return torch.where(expanded, attention_residual, full)


def _conditional_encoder_forward(
    self: Any,
    input_ids: Any = None,
    inputs_embeds: Any = None,
    patch_embeds: Any = None,
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_values: Any = None,
    encoder_attention_mask: Any = None,
    num_patches: int | None = None,
    patch_ids: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    import torch.nn.functional as F

    if input_ids is None:
        raise ValueError("conditional local encoder requires input_ids")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch_size = inputs_embeds.shape[0]
    hidden_states = F.dropout(inputs_embeds, p=self.config.dropout, training=self.training)
    if position_ids is None:
        import torch

        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
    easy = conditional_easy_mask(input_ids, self._jamoflow_route_policy)

    for index, layer in enumerate(self.layers):
        if index == 1 and self._jamoflow_components == "encoder_decoder":
            hidden_states = _masked_second_layer(
                layer,
                hidden_states,
                easy,
                operator=self._jamoflow_operator,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                kwargs=kwargs,
            )
        else:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        if index == len(self.layers) - 1 or self.config.cross_attn_all_layers:
            patch_embeds = self.patch_reduce(hidden_states, num_patches, patch_ids)
            patch_embeds = self.patch_embedding_projection(patch_embeds)
            patch_embeds = patch_embeds.reshape(
                batch_size,
                patch_embeds.shape[1] * self.config.cross_attn_k,
                self.config.hidden_size,
            )
            layer_index = index if self.config.cross_attn_all_layers else 0
            cross_output, _ = self.cross_attn_layers[layer_index](
                hidden_states=patch_embeds,
                cross_attention_states=hidden_states,
                attention_mask=encoder_attention_mask,
                **kwargs,
            )
            patch_embeds = patch_embeds + cross_output.to(patch_embeds.device)
    return hidden_states, patch_embeds


def _conditional_decoder_forward(
    self: Any,
    input_ids: Any = None,
    inputs_embeds: Any = None,
    patch_embeds: Any = None,
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_values: Any = None,
    encoder_attention_mask: Any = None,
    **kwargs: Any,
) -> Any:
    import torch
    import torch.nn.functional as F

    if input_ids is None or inputs_embeds is None or patch_embeds is None:
        raise ValueError("conditional local decoder inputs are incomplete")
    batch_size = inputs_embeds.shape[0]
    hidden_states = inputs_embeds
    projected_patches = self.patch_embedding_projection(patch_embeds).reshape(
        batch_size,
        patch_embeds.shape[1] * self.config.cross_attn_k,
        self.config.hidden_size,
    )
    if position_ids is None:
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0).expand(batch_size, -1)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
    easy = conditional_easy_mask(input_ids, self._jamoflow_route_policy)

    for index, layer in enumerate(self.layers):
        stage_input = hidden_states
        cross_output, _ = self.cross_attn_layers[index](
            hidden_states=hidden_states,
            cross_attention_states=projected_patches,
            attention_mask=encoder_attention_mask,
            **kwargs,
        )
        crossed = hidden_states + cross_output
        if index != 1:
            hidden_states = layer(
                crossed,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            continue
        if self._jamoflow_operator == "second_layer_kv":
            layer_input = torch.where(easy.unsqueeze(-1), stage_input, crossed)
            full = layer(
                layer_input,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = torch.where(easy.unsqueeze(-1), stage_input, full)
        else:
            hidden_states = _masked_second_layer(
                layer,
                crossed,
                easy,
                operator=self._jamoflow_operator,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                kwargs=kwargs,
            )
    return self.norm(hidden_states)


def install_conditional_local(
    model: Any,
    *,
    route_policy: str,
    operator: str,
    components: str,
) -> Any:
    """Install a parameter-neutral conditional local graph on a BLT model."""

    if route_policy not in ROUTE_POLICIES:
        raise ValueError(f"unsupported conditional route policy: {route_policy}")
    if operator not in CONDITIONAL_OPERATORS:
        raise ValueError(f"unsupported conditional operator: {operator}")
    if components not in CONDITIONAL_COMPONENTS:
        raise ValueError(f"unsupported conditional components: {components}")
    base = model.model
    encoder = base.local_encoder
    decoder = base.local_decoder
    if len(encoder.layers) != 2 or len(decoder.layers) != 2:
        raise ValueError("conditional local graph requires exactly two local layers")
    if hasattr(encoder, "_jamoflow_original_forward") or hasattr(
        decoder, "_jamoflow_original_forward"
    ):
        raise ValueError("conditional local graph is already installed")

    for module in (encoder, decoder):
        module._jamoflow_route_policy = route_policy
        module._jamoflow_operator = operator
        module._jamoflow_components = components
        module._jamoflow_original_forward = module.forward
    encoder.forward = MethodType(_conditional_encoder_forward, encoder)
    decoder.forward = MethodType(_conditional_decoder_forward, decoder)
    model._jamoflow_conditional_local = {
        "route_policy": route_policy,
        "operator": operator,
        "components": components,
    }
    return model
