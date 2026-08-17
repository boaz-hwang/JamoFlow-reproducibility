"""Logit-equivalent batch-1 incremental decoding for the pinned HF BLT graph.

The Transformers BLT forward path exposes cache arguments, but its global trunk
is not incrementally cached and hash embeddings cannot be reconstructed by
feeding an isolated final byte.  This module therefore advances the pinned
local-encoder/global/local-decoder graph explicitly while leaving all trained
weights and arithmetic definitions unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .phase2_patching import (
    causal_codepoint_grid_boundaries,
    causal_window_grid_trace,
    compact_whitespace_mask,
    entropy_threshold_boundaries,
    padded_hf_patch_matrix,
    scheduled_targets,
)
from .patching import is_spacebyte_spacelike
from .phase3 import spacebyte_boundaries, spacebyte_causal_prefix_mask
from .utf8 import (
    _lead_spec,
    _valid_codepoint,
    is_continuation_byte,
    prefix_boundary_mask,
)


INCREMENTAL_STRUCTURAL_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
    "spacebyte_spacelike",
)
INCREMENTAL_ENTROPY_POLICIES = (
    "entropy_threshold_full",
    "entropy_threshold_codepoint",
)
INCREMENTAL_POLICIES = (
    *INCREMENTAL_STRUCTURAL_POLICIES,
    *INCREMENTAL_ENTROPY_POLICIES,
)


class _StreamingUtf8PrefixState:
    """Constant-time state equivalent to the corpus prefix mask helpers."""

    def __init__(self) -> None:
        self.remaining = 0
        self.accumulator = 0
        self.minimum = 0
        self.at_codepoint_boundary = True
        self.completed_codepoint: int | None = None

    def consume(self, value: int) -> None:
        completed: int | None = None
        if self.remaining == 0:
            spec = _lead_spec(value)
            if spec is not None:
                self.remaining, self.accumulator, self.minimum = spec
                if self.remaining == 0:
                    completed = self.accumulator
        elif is_continuation_byte(value):
            self.accumulator = (self.accumulator << 6) | (value & 0x3F)
            self.remaining -= 1
            if self.remaining == 0 and _valid_codepoint(
                self.accumulator,
                self.minimum,
            ):
                completed = self.accumulator
        else:
            self.remaining = 0
            spec = _lead_spec(value)
            if spec is not None:
                self.remaining, self.accumulator, self.minimum = spec
                if self.remaining == 0:
                    completed = self.accumulator
        self.at_codepoint_boundary = self.remaining == 0
        self.completed_codepoint = completed


class IncrementalStructuralSelector:
    """Emit F/C/W boundaries in constant time per observed byte."""

    def __init__(
        self,
        policy: str,
        *,
        horizon: int,
        patch_count: int,
        fixed_stride: int,
    ) -> None:
        if policy not in INCREMENTAL_STRUCTURAL_POLICIES:
            raise ValueError(f"unsupported structural selector: {policy}")
        if horizon <= 1 or not 1 < patch_count <= horizon or fixed_stride <= 0:
            raise ValueError("invalid structural selector geometry")
        self.policy = policy
        self.horizon = horizon
        self.fixed_stride = fixed_stride
        self.targets = scheduled_targets(horizon, patch_count)
        self.next_target = 0
        self.observed_bytes = 0
        self.boundaries: tuple[int, ...] = ()
        self.utf8 = _StreamingUtf8PrefixState()
        self._previous_spacelike = False
        self._spacebyte_boundary_pending = False

    def consume(self, value: int) -> tuple[int, ...]:
        if not 0 <= int(value) <= 255:
            raise ValueError("selector input byte must be in [0, 255]")
        if self.observed_bytes >= self.horizon:
            raise ValueError("structural selector horizon exhausted")
        position = self.observed_bytes
        emit = position == 0
        if position > 0 and self.policy == "fixed_byte_6":
            emit = position % self.fixed_stride == 0
        elif position > 0 and self.policy == "spacebyte_spacelike":
            emit = self._spacebyte_boundary_pending
        elif (
            position > 0
            and self.next_target < len(self.targets)
            and self.utf8.at_codepoint_boundary
        ):
            target = self.targets[self.next_target]
            if self.policy == "causal_codepoint_grid":
                emit = position >= target
            else:
                final_target = self.next_target == len(self.targets) - 1
                completed = self.utf8.completed_codepoint
                whitespace = completed is not None and chr(completed).isspace()
                event = (
                    not final_target
                    and whitespace
                    and position >= target - 2
                    and position - self.boundaries[-1] >= 2
                )
                deadline = not final_target and position >= target + 2
                final = final_target and position >= target
                emit = event or deadline or final
        if emit:
            self.boundaries = (*self.boundaries, position)
            if position > 0 and self.policy in {
                "causal_codepoint_grid",
                "causal_whitespace_grid",
            }:
                self.next_target += 1
        current_spacelike = is_spacebyte_spacelike(int(value))
        self._spacebyte_boundary_pending = (
            current_spacelike and not self._previous_spacelike
        )
        self._previous_spacelike = current_spacelike
        self.utf8.consume(int(value))
        self.observed_bytes += 1
        return self.boundaries


def structural_prefix_boundaries(
    data: bytes,
    policy: str,
    *,
    horizon: int,
    patch_count: int,
    fixed_stride: int,
) -> tuple[int, ...]:
    """Return every structural boundary decidable from the observed prefix."""

    if policy not in INCREMENTAL_STRUCTURAL_POLICIES:
        raise ValueError(f"unsupported incremental structural policy: {policy}")
    if not 0 < len(data) <= horizon:
        raise ValueError("observed prefix must be inside the decoding horizon")
    if policy == "fixed_byte_6":
        return tuple(range(0, len(data), fixed_stride))
    if policy == "spacebyte_spacelike":
        return spacebyte_boundaries(spacebyte_causal_prefix_mask(data))
    codepoints = np.frombuffer(prefix_boundary_mask(data)[:-1], dtype=np.uint8)
    if policy == "causal_codepoint_grid":
        return causal_codepoint_grid_boundaries(
            codepoints,
            patch_count,
            sequence_length=horizon,
            require_complete=False,
        )
    whitespace = compact_whitespace_mask(data)
    return causal_window_grid_trace(
        codepoints,
        whitespace,
        patch_count,
        sequence_length=horizon,
        require_complete=False,
    ).boundaries


@dataclass(frozen=True, slots=True)
class IncrementalBltDiagnostics:
    observed_bytes: int
    emitted_data_patches: int
    local_encoder_cached_bytes: int
    global_cached_patches: int
    local_decoder_cached_bytes: int
    boundaries: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IncrementalBltRuntimeCounters:
    """High-level calls whose exact identities are checked after timing."""

    parallel_prefill_calls: int
    main_consume_calls: int
    selector_observed_bytes: int
    router_forward_calls: int
    router_scored_bytes: int


class IncrementalBltDecoder:
    """Advance one byte at a time with exact batch-1 BLT patch lag semantics."""

    def __init__(
        self,
        model: Any,
        policy: str,
        *,
        horizon: int,
        patch_count: int,
        fixed_stride: int,
    ) -> None:
        if policy not in INCREMENTAL_POLICIES:
            raise ValueError(f"unsupported incremental policy: {policy}")
        if horizon <= 1 or not 1 < patch_count <= horizon or fixed_stride <= 0:
            raise ValueError("invalid incremental decoding geometry")
        try:
            import torch
            from transformers import DynamicCache
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "incremental BLT requires research dependencies"
            ) from error

        self._torch = torch
        self.model = model
        self.policy = policy
        self.horizon = horizon
        self.patch_count = patch_count
        self.fixed_stride = fixed_stride
        self.model.eval()
        self.base = model.model
        if (
            self.base.local_encoder.config.cross_attn_all_layers
            or not self.base.local_decoder.config.cross_attn_all_layers
            or len(self.base.local_encoder.cross_attn_layers) != 1
            or len(self.base.local_decoder.cross_attn_layers)
            != len(self.base.local_decoder.layers)
        ):
            raise ValueError(
                "incremental runtime requires the pinned BLT cross-attention graph"
            )
        self.device = next(model.parameters()).device
        self.encoder_cache = DynamicCache(config=self.base.local_encoder.config)
        self.global_cache = DynamicCache(config=self.base.global_transformer.config)
        self.decoder_cache = DynamicCache(config=self.base.local_decoder.config)
        self._data = bytearray()
        self._boundaries: tuple[int, ...] = ()
        self._pending_encoder_states: list[Any] = []
        self._current_global_state: Any | None = None
        self._parallel_prefill_calls = 0
        self._main_consume_calls = 0
        self._selector_observed_bytes = 0
        self._structural_selector = (
            IncrementalStructuralSelector(
                policy,
                horizon=horizon,
                patch_count=patch_count,
                fixed_stride=fixed_stride,
            )
            if policy in INCREMENTAL_STRUCTURAL_POLICIES
            else None
        )

    @property
    def data(self) -> bytes:
        return bytes(self._data)

    @property
    def diagnostics(self) -> IncrementalBltDiagnostics:
        return IncrementalBltDiagnostics(
            observed_bytes=len(self._data),
            emitted_data_patches=len(self._boundaries),
            local_encoder_cached_bytes=int(self.encoder_cache.get_seq_length()),
            global_cached_patches=int(self.global_cache.get_seq_length()),
            local_decoder_cached_bytes=int(self.decoder_cache.get_seq_length()),
            boundaries=self._boundaries,
        )

    @property
    def runtime_counters(self) -> IncrementalBltRuntimeCounters:
        return IncrementalBltRuntimeCounters(
            parallel_prefill_calls=self._parallel_prefill_calls,
            main_consume_calls=self._main_consume_calls,
            selector_observed_bytes=self._selector_observed_bytes,
            router_forward_calls=0,
            router_scored_bytes=0,
        )

    def _current_hash_embedding(self) -> Any:
        from transformers.models.blt.modeling_blt import compute_hash_embeddings

        group_sizes = self.base.config.encoder_hash_byte_group_size
        tail_length = min(len(self._data), max(group_sizes))
        tail = list(self._data[-tail_length:])
        tokens = self._torch.tensor(
            [tail],
            dtype=self._torch.long,
            device=self.device,
        )
        embeddings = compute_hash_embeddings(
            tokens,
            self.base.local_encoder,
            self.base.encoder_hash_tok_embedding,
            self.base.config.encoder_hash_byte_group_nb_functions,
            group_sizes,
            self.base.config.encoder_hash_byte_group_vocab,
        )
        return embeddings[:, -1:, :]

    def _advance_local_encoder(self, position: int) -> Any:
        encoder = self.base.local_encoder
        hidden = self._current_hash_embedding()
        position_ids = self._torch.tensor(
            [[position]],
            dtype=self._torch.long,
            device=self.device,
        )
        position_embeddings = encoder.rotary_emb(hidden, position_ids)
        for layer in encoder.layers:
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=self.encoder_cache,
                use_cache=True,
            )
        return hidden

    def _finalize_encoder_patch(self, patch_index: int) -> Any:
        if not self._pending_encoder_states:
            raise RuntimeError("cannot finalize an empty encoder patch")
        encoder = self.base.local_encoder
        states = self._torch.cat(self._pending_encoder_states, dim=1)
        reduced = states.amax(dim=1, keepdim=True)
        patch_queries = encoder.patch_embedding_projection(reduced)
        patch_queries = patch_queries.reshape(
            1,
            encoder.config.cross_attn_k,
            encoder.config.hidden_size,
        )
        cross_output, _ = encoder.cross_attn_layers[0](
            hidden_states=patch_queries,
            cross_attention_states=states,
            attention_mask=None,
        )
        encoder_cross_state = patch_queries + cross_output
        encoder_cross_state = encoder_cross_state.reshape(1, 1, -1)
        global_position = self._torch.tensor(
            [[patch_index]],
            dtype=self._torch.long,
            device=self.device,
        )
        global_state = self.base.global_transformer(
            inputs_embeds=encoder_cross_state,
            attention_mask=None,
            position_ids=global_position,
            past_key_values=self.global_cache,
        )
        self._pending_encoder_states.clear()
        return global_state

    def _advance_local_decoder(self, encoder_state: Any, position: int) -> Any:
        if self._current_global_state is None:
            raise RuntimeError("the first global patch must exist before decoding")
        decoder = self.base.local_decoder
        patch_embeds = decoder.patch_embedding_projection(
            self._current_global_state
        ).reshape(
            1,
            decoder.config.cross_attn_k,
            decoder.config.hidden_size,
        )
        hidden = encoder_state
        position_ids = self._torch.tensor(
            [[position]],
            dtype=self._torch.long,
            device=self.device,
        )
        position_embeddings = decoder.rotary_emb(hidden, position_ids)
        for layer_index, layer in enumerate(decoder.layers):
            cross_output, _ = decoder.cross_attn_layers[layer_index](
                hidden_states=hidden,
                cross_attention_states=patch_embeds,
                attention_mask=None,
            )
            hidden = hidden + cross_output
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=self.decoder_cache,
                use_cache=True,
            )
        return decoder.norm(hidden)

    def _consume_appended(self, boundaries: tuple[int, ...]) -> Any:
        position = len(self._data) - 1
        if boundaries[: len(self._boundaries)] != self._boundaries:
            raise RuntimeError("patch boundary decisions are not prefix invariant")
        if len(boundaries) - len(self._boundaries) not in (0, 1):
            raise RuntimeError("more than one patch boundary appeared per byte")
        boundary_emitted = len(boundaries) == len(self._boundaries) + 1

        encoder_state = self._advance_local_encoder(position)
        self._pending_encoder_states.append(encoder_state)
        if boundary_emitted:
            if boundaries[-1] != position:
                raise RuntimeError("new boundary does not start at the current byte")
            self._current_global_state = self._finalize_encoder_patch(
                len(boundaries) - 1
            )
        self._boundaries = boundaries
        decoder_state = self._advance_local_decoder(encoder_state, position)
        return self.model.lm_head(decoder_state).float()[:, -1, :]

    def consume(self, value: int) -> Any:
        """Consume one observed byte and return its next-byte logits."""

        if self.policy not in INCREMENTAL_STRUCTURAL_POLICIES:
            raise RuntimeError("learned policies require IncrementalEntropyBltDecoder")
        if not 0 <= int(value) <= 255:
            raise ValueError("BLT input byte must be in [0, 255]")
        if len(self._data) >= self.horizon:
            raise ValueError("incremental decoding horizon exhausted")
        if self._structural_selector is None:
            raise RuntimeError("structural selector state is unavailable")
        self._data.append(int(value))
        boundaries = self._structural_selector.consume(int(value))
        self._main_consume_calls += 1
        self._selector_observed_bytes += 1
        return self._consume_appended(boundaries)

    def prefill(self, data: bytes) -> Any:
        """Sequential correctness prefill; not the evidentiary TTFT path."""

        if not data:
            raise ValueError("incremental prefill requires at least one byte")
        logits = None
        with self._torch.inference_mode():
            for value in data:
                logits = self.consume(value)
        return logits

    def prefill_parallel(self, data: bytes) -> Any:
        """Build exact prompt caches in parallel and return final-byte logits."""

        if self.policy not in INCREMENTAL_STRUCTURAL_POLICIES:
            raise RuntimeError("learned policies require IncrementalEntropyBltDecoder")
        selector = IncrementalStructuralSelector(
            self.policy,
            horizon=self.horizon,
            patch_count=self.patch_count,
            fixed_stride=self.fixed_stride,
        )
        boundaries: tuple[int, ...] = ()
        for value in data:
            boundaries = selector.consume(value)
        self._structural_selector = selector
        self._parallel_prefill_calls += 1
        self._selector_observed_bytes += len(data)
        return self._prefill_parallel_selected(data, boundaries)

    def _prefill_parallel_selected(
        self,
        data: bytes,
        boundaries: tuple[int, ...],
    ) -> Any:
        """Populate main-model prompt caches for an already selected schedule."""

        from transformers.masking_utils import create_causal_mask
        from transformers.models.blt.modeling_blt import (
            _prepare_patch_cross_attention_mask,
            compute_hash_embeddings,
        )

        if not data:
            raise ValueError("parallel prefill requires at least one byte")
        if self._data:
            raise RuntimeError("parallel prefill requires a fresh runtime")
        if len(data) > self.horizon:
            raise ValueError("parallel prefill exceeds the decoding horizon")
        if not boundaries or boundaries[0] != 0 or boundaries[-1] >= len(data):
            raise ValueError("parallel prefill boundaries are malformed")

        torch = self._torch
        self._data.extend(data)
        self._boundaries = boundaries
        closed_patches = len(self._boundaries)
        input_ids = torch.tensor(
            [list(data)],
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(
            len(data),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        encoder = self.base.local_encoder
        encoder_hidden = compute_hash_embeddings(
            input_ids,
            encoder,
            self.base.encoder_hash_tok_embedding,
            self.base.config.encoder_hash_byte_group_nb_functions,
            self.base.config.encoder_hash_byte_group_size,
            self.base.config.encoder_hash_byte_group_vocab,
        )
        local_causal_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=encoder_hidden,
            attention_mask=None,
            past_key_values=self.encoder_cache,
            position_ids=position_ids,
        )
        encoder_positions = encoder.rotary_emb(encoder_hidden, position_ids)
        for layer in encoder.layers:
            encoder_hidden = layer(
                encoder_hidden,
                position_embeddings=encoder_positions,
                attention_mask=local_causal_mask,
                past_key_values=self.encoder_cache,
                use_cache=True,
            )

        patch_matrix = padded_hf_patch_matrix(
            [self._boundaries],
            len(data),
        )
        patch_lengths = torch.from_numpy(
            patch_matrix.astype(np.int64, copy=False)
        ).to(self.device)
        encoder_patch_ids = self.base._patch_ids_from_lengths(
            patch_lengths,
            len(data),
        )
        all_patch_queries = encoder.patch_reduce(
            encoder_hidden,
            patch_lengths.shape[1],
            encoder_patch_ids,
        )
        all_patch_queries = encoder.patch_embedding_projection(
            all_patch_queries
        ).reshape(
            1,
            patch_lengths.shape[1] * encoder.config.cross_attn_k,
            encoder.config.hidden_size,
        )
        patch_queries = all_patch_queries[
            :, : closed_patches * encoder.config.cross_attn_k, :
        ]
        encoder_cross_mask = _prepare_patch_cross_attention_mask(
            patch_ids=encoder_patch_ids,
            num_patches=patch_lengths.shape[1],
            sequence_length=len(data),
            patches_as_queries=True,
            cross_attn_k=encoder.config.cross_attn_k,
            dtype=encoder_hidden.dtype,
        )[:, :, : closed_patches * encoder.config.cross_attn_k, :]
        cross_output, _ = encoder.cross_attn_layers[0](
            hidden_states=patch_queries,
            cross_attention_states=encoder_hidden,
            attention_mask=encoder_cross_mask,
        )
        encoder_cross_states = (patch_queries + cross_output).reshape(
            1,
            closed_patches,
            -1,
        )

        global_positions = torch.arange(
            closed_patches,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        projected_global = self.base.global_transformer.token_embedding_projection(
            encoder_cross_states
        )
        global_causal_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=projected_global,
            attention_mask=None,
            past_key_values=self.global_cache,
            position_ids=global_positions,
        )
        global_hidden = self.base.global_transformer(
            inputs_embeds=encoder_cross_states,
            attention_mask=global_causal_mask,
            position_ids=global_positions,
            past_key_values=self.global_cache,
        )
        self._current_global_state = global_hidden[:, -1:, :]

        decoder = self.base.local_decoder
        decoder_patch_ids = self.base._patch_ids_from_lengths(
            patch_lengths[:, 1:],
            len(data),
        )
        decoder_cross_mask = _prepare_patch_cross_attention_mask(
            patch_ids=decoder_patch_ids,
            num_patches=closed_patches,
            sequence_length=len(data),
            patches_as_queries=False,
            cross_attn_k=decoder.config.cross_attn_k,
            dtype=encoder_hidden.dtype,
        )
        decoder_hidden = decoder(
            input_ids=input_ids,
            inputs_embeds=encoder_hidden,
            patch_embeds=global_hidden,
            attention_mask=local_causal_mask,
            position_ids=position_ids,
            past_key_values=self.decoder_cache,
            encoder_attention_mask=decoder_cross_mask,
        )

        pending_start = self._boundaries[-1] + 1
        if pending_start < len(data):
            self._pending_encoder_states = [
                encoder_hidden[:, pending_start:, :]
            ]
        return self.model.lm_head(decoder_hidden[:, -1:, :]).float()[:, -1, :]


class IncrementalEntropyRouter:
    """Logit-equivalent incremental runtime for the pinned BLT patcher."""

    def __init__(self, router: Any) -> None:
        try:
            import torch
            from transformers import DynamicCache
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "incremental entropy routing requires research dependencies"
            ) from error
        self._torch = torch
        self.router = router.eval()
        self.device = next(router.parameters()).device
        self.cache = DynamicCache(config=router.config)
        self.observed_bytes = 0
        self.forward_calls = 0
        self.scored_bytes = 0

    def _outputs(self, hidden: Any) -> tuple[Any, Any]:
        logits = self.router.lm_head(self.router.norm(hidden))
        entropies = self._torch.distributions.Categorical(
            logits=logits
        ).entropy()
        return logits, entropies

    def consume(self, value: int) -> tuple[Any, float]:
        if not 0 <= int(value) <= 255:
            raise ValueError("router input byte must be in [0, 255]")
        torch = self._torch
        input_ids = torch.tensor(
            [[int(value)]],
            dtype=torch.long,
            device=self.device,
        )
        hidden = self.router.embed_tokens(input_ids)
        position_ids = torch.tensor(
            [[self.observed_bytes]],
            dtype=torch.long,
            device=self.device,
        )
        positions = self.router.rotary_emb(hidden, position_ids)
        for layer in self.router.layers:
            hidden = layer(
                hidden,
                position_embeddings=positions,
                attention_mask=None,
                past_key_values=self.cache,
                use_cache=True,
            )
        logits, entropies = self._outputs(hidden)
        self.observed_bytes += 1
        self.forward_calls += 1
        self.scored_bytes += 1
        return logits[:, -1, :].float(), float(entropies[0, -1])

    def prefill_parallel(self, data: bytes) -> tuple[Any, np.ndarray]:
        from transformers.masking_utils import create_causal_mask

        if not data:
            raise ValueError("router prefill requires at least one byte")
        if self.observed_bytes:
            raise RuntimeError("router parallel prefill requires a fresh runtime")
        torch = self._torch
        input_ids = torch.tensor(
            [list(data)],
            dtype=torch.long,
            device=self.device,
        )
        hidden = self.router.embed_tokens(input_ids)
        position_ids = torch.arange(
            len(data),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.router.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=self.cache,
            position_ids=position_ids,
        )
        positions = self.router.rotary_emb(hidden, position_ids)
        for layer in self.router.layers:
            hidden = layer(
                hidden,
                position_embeddings=positions,
                attention_mask=causal_mask,
                past_key_values=self.cache,
                use_cache=True,
            )
        logits, entropies = self._outputs(hidden)
        self.observed_bytes = len(data)
        self.forward_calls += 1
        self.scored_bytes += len(data)
        return (
            logits[:, -1, :].float(),
            entropies[0].float().cpu().numpy(),
        )


@dataclass(frozen=True, slots=True)
class IncrementalEntropyBltDiagnostics:
    main: IncrementalBltDiagnostics
    router_cached_bytes: int
    threshold_nats: float
    maximum_patch_length: int


class IncrementalEntropyBltDecoder(IncrementalBltDecoder):
    """Incremental BLT decoding with a causal learned entropy selector."""

    def __init__(
        self,
        model: Any,
        router: Any,
        policy: str,
        *,
        threshold_nats: float,
        maximum_patch_length: int,
        horizon: int,
        patch_count: int,
        fixed_stride: int,
    ) -> None:
        if policy not in INCREMENTAL_ENTROPY_POLICIES:
            raise ValueError(f"unsupported entropy policy: {policy}")
        if not np.isfinite(threshold_nats) or maximum_patch_length <= 0:
            raise ValueError("invalid entropy selector configuration")
        super().__init__(
            model,
            policy,
            horizon=horizon,
            patch_count=patch_count,
            fixed_stride=fixed_stride,
        )
        self.threshold_nats = float(threshold_nats)
        self.maximum_patch_length = int(maximum_patch_length)
        self.router_runtime = IncrementalEntropyRouter(router)
        if self.router_runtime.device != self.device:
            raise ValueError("main model and entropy router must share a device")
        self._next_entropy: float | None = None
        self._utf8_state = _StreamingUtf8PrefixState()

    @property
    def diagnostics(self) -> IncrementalEntropyBltDiagnostics:
        return IncrementalEntropyBltDiagnostics(
            main=super().diagnostics,
            router_cached_bytes=int(self.router_runtime.cache.get_seq_length()),
            threshold_nats=self.threshold_nats,
            maximum_patch_length=self.maximum_patch_length,
        )

    @property
    def runtime_counters(self) -> IncrementalBltRuntimeCounters:
        return IncrementalBltRuntimeCounters(
            parallel_prefill_calls=self._parallel_prefill_calls,
            main_consume_calls=self._main_consume_calls,
            selector_observed_bytes=self._selector_observed_bytes,
            router_forward_calls=self.router_runtime.forward_calls,
            router_scored_bytes=self.router_runtime.scored_bytes,
        )

    def _candidate_allowed(self) -> bool:
        if self.policy == "entropy_threshold_full":
            return True
        return self._utf8_state.at_codepoint_boundary

    def consume(self, value: int) -> Any:
        if not 0 <= int(value) <= 255:
            raise ValueError("BLT input byte must be in [0, 255]")
        if len(self._data) >= self.horizon:
            raise ValueError("incremental decoding horizon exhausted")
        position = len(self._data)
        boundaries = self._boundaries
        if position == 0:
            boundaries = (0,)
        else:
            if self._next_entropy is None or not boundaries:
                raise RuntimeError("entropy selector state is incomplete")
            patch_length = position - boundaries[-1]
            if self._candidate_allowed() and (
                patch_length >= self.maximum_patch_length
                or self._next_entropy >= self.threshold_nats
            ):
                boundaries = (*boundaries, position)
        self._data.append(int(value))
        _, self._next_entropy = self.router_runtime.consume(value)
        self._utf8_state.consume(int(value))
        self._main_consume_calls += 1
        self._selector_observed_bytes += 1
        return self._consume_appended(boundaries)

    def prefill(self, data: bytes) -> Any:
        if not data:
            raise ValueError("incremental prefill requires at least one byte")
        logits = None
        with self._torch.inference_mode():
            for value in data:
                logits = self.consume(value)
        return logits

    def prefill_parallel(self, data: bytes) -> Any:
        if self._data:
            raise RuntimeError("parallel prefill requires a fresh runtime")
        _, entropies = self.router_runtime.prefill_parallel(data)
        aligned = np.zeros(len(data), dtype=np.float32)
        aligned[1:] = entropies[:-1]
        candidate_mask = None
        if self.policy == "entropy_threshold_codepoint":
            candidate_mask = np.frombuffer(
                prefix_boundary_mask(data)[:-1],
                dtype=np.uint8,
            )
        boundaries = entropy_threshold_boundaries(
            aligned,
            self.threshold_nats,
            candidate_mask=candidate_mask,
            maximum_patch_length=self.maximum_patch_length,
        )
        self._next_entropy = float(entropies[-1])
        for value in data:
            self._utf8_state.consume(value)
        self._parallel_prefill_calls += 1
        self._selector_observed_bytes += len(data)
        return self._prefill_parallel_selected(data, boundaries)
