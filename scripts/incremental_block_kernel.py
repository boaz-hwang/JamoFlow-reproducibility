"""Exploratory exact block-consume kernel for the structural BLT runtime.

This is intentionally outside ``src/jamoflow`` so that development of a new
kernel cannot mutate the historical publication implementation closure.  It
advances a *known-correct* byte block without rollback; speculative acceptance
is a later layer and must not be inferred from this upper-bound primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

from jamoflow.incremental_blt import (
    INCREMENTAL_STRUCTURAL_POLICIES,
    IncrementalBltDecoder,
)


@dataclass(slots=True)
class BlockTransaction:
    """A completed target block that may be cropped to a verified prefix."""

    runtime: "IncrementalBlockBltDecoder"
    start: int
    values: bytes
    logits: Any
    hidden: Any
    encoder_states: Any
    pre_pending: tuple[Any, ...]
    pre_global_state: Any
    pre_boundaries: tuple[int, ...]
    boundaries_after: tuple[tuple[int, ...], ...]
    global_states_after: tuple[Any, ...]
    boundary_emitted: tuple[bool, ...]
    selector_before: Any
    selectors_after: tuple[Any, ...]
    encoder_cache_start: int
    decoder_cache_start: int
    global_cache_start: int
    _closed: bool = False

    def finish(self, keep: int) -> tuple[Any, Any]:
        """Keep ``keep`` bytes, crop all caches, and return its final row."""

        if self._closed:
            raise RuntimeError("block transaction is already closed")
        if not 0 <= int(keep) <= len(self.values):
            raise ValueError("block transaction keep count differs")
        self.runtime._finish_block_transaction(self, int(keep))
        self._closed = True
        if keep == 0:
            return self.logits[:0], self.hidden[:0]
        return self.logits[keep - 1 : keep], self.hidden[keep - 1 : keep]


class IncrementalBlockBltDecoder(IncrementalBltDecoder):
    """Advance multiple observed bytes with block local-transformer calls."""

    @staticmethod
    def _copy_selector(selector: Any) -> Any:
        cloned = copy.copy(selector)
        cloned.utf8 = copy.copy(selector.utf8)
        return cloned

    def _block_hash_embeddings(self, values: bytes) -> Any:
        from transformers.models.blt.modeling_blt import compute_hash_embeddings

        encoder = self.base.local_encoder
        maximum_group = max(self.base.config.encoder_hash_byte_group_size)
        previous = bytes(self._data[: -len(values)])
        context = previous[-(maximum_group - 1) :] if maximum_group > 1 else b""
        tokens = self._torch.tensor(
            [list(context + values)],
            dtype=self._torch.long,
            device=self.device,
        )
        embeddings = compute_hash_embeddings(
            tokens,
            encoder,
            self.base.encoder_hash_tok_embedding,
            self.base.config.encoder_hash_byte_group_nb_functions,
            self.base.config.encoder_hash_byte_group_size,
            self.base.config.encoder_hash_byte_group_vocab,
        )
        return embeddings[:, -len(values) :, :]

    def _advance_local_encoder_block(self, values: bytes, start: int) -> Any:
        from transformers.masking_utils import create_causal_mask

        encoder = self.base.local_encoder
        hidden = self._block_hash_embeddings(values)
        positions = self._torch.arange(
            start,
            start + len(values),
            dtype=self._torch.long,
            device=self.device,
        ).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=self.encoder_cache,
            position_ids=positions,
        )
        position_embeddings = encoder.rotary_emb(hidden, positions)
        for layer in encoder.layers:
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=self.encoder_cache,
                use_cache=True,
            )
        return hidden

    def _advance_local_decoder_block(self, encoder_states: Any, start: int) -> Any:
        from transformers.masking_utils import create_causal_mask

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
        positions = self._torch.arange(
            start,
            start + encoder_states.shape[1],
            dtype=self._torch.long,
            device=self.device,
        ).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=encoder_states,
            attention_mask=None,
            past_key_values=self.decoder_cache,
            position_ids=positions,
        )
        position_embeddings = decoder.rotary_emb(encoder_states, positions)
        hidden = encoder_states
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
                attention_mask=attention_mask,
                past_key_values=self.decoder_cache,
                use_cache=True,
            )
        return decoder.norm(hidden)

    def consume_block_transaction(self, values: bytes) -> BlockTransaction:
        """Speculatively consume a block and retain exact prefix rollback state."""

        if self.policy not in INCREMENTAL_STRUCTURAL_POLICIES:
            raise RuntimeError("block kernel supports structural policies only")
        if not isinstance(values, bytes) or not 1 <= len(values) <= 4:
            raise ValueError("block kernel requires one to four bytes")
        if len(self._data) + len(values) > self.horizon:
            raise ValueError("block kernel exceeds the decoding horizon")
        if self._structural_selector is None:
            raise RuntimeError("structural selector state is unavailable")

        start = len(self._data)
        pre_pending = tuple(self._pending_encoder_states)
        pre_global_state = self._current_global_state
        previous_boundaries = self._boundaries
        selector_before = self._copy_selector(self._structural_selector)
        encoder_cache_start = int(self.encoder_cache.get_seq_length())
        decoder_cache_start = int(self.decoder_cache.get_seq_length())
        global_cache_start = int(self.global_cache.get_seq_length())
        boundary_emitted: list[bool] = []
        boundaries_after: list[tuple[int, ...]] = []
        selectors_after: list[Any] = []
        for value in values:
            if not 0 <= int(value) <= 255:
                raise ValueError("BLT block input byte must be in [0, 255]")
            self._data.append(int(value))
            before = len(self._structural_selector.boundaries)
            current = self._structural_selector.consume(int(value))
            if current[: len(previous_boundaries)] != previous_boundaries:
                raise RuntimeError("block boundary decisions are not prefix invariant")
            emitted = len(current) == before + 1
            if len(current) - before not in (0, 1):
                raise RuntimeError("block selector emitted multiple boundaries per byte")
            boundary_emitted.append(emitted)
            boundaries_after.append(current)
            selectors_after.append(self._copy_selector(self._structural_selector))

        encoder_states = self._advance_local_encoder_block(values, start)
        output_groups: list[Any] = []
        hidden_groups: list[Any] = []
        global_states_after: list[Any] = []
        segment_start = 0
        for offset, emitted in enumerate(boundary_emitted):
            self._pending_encoder_states.append(
                encoder_states[:, offset : offset + 1, :]
            )
            if not emitted:
                continue
            if offset > segment_start:
                decoder = self._advance_local_decoder_block(
                    encoder_states[:, segment_start:offset, :],
                    start + segment_start,
                )
                hidden_groups.append(decoder)
                output_groups.append(self.model.lm_head(decoder).float())
            current_boundaries = self._structural_selector.boundaries
            if current_boundaries[-1] != start + offset:
                raise RuntimeError("block boundary position differs")
            self._current_global_state = self._finalize_encoder_patch(
                len(current_boundaries) - 1
            )
            segment_start = offset
            global_states_after.append(self._current_global_state)
            continue
        # Populate the global state after every byte, including non-boundaries.
        last_global = pre_global_state
        boundary_global_iter = iter(global_states_after)
        global_states_by_offset: list[Any] = []
        for emitted in boundary_emitted:
            if emitted:
                last_global = next(boundary_global_iter)
            global_states_by_offset.append(last_global)

        if segment_start < len(values):
            decoder = self._advance_local_decoder_block(
                encoder_states[:, segment_start:, :],
                start + segment_start,
            )
            hidden_groups.append(decoder)
            output_groups.append(self.model.lm_head(decoder).float())
        self._boundaries = self._structural_selector.boundaries
        self._main_consume_calls += len(values)
        self._selector_observed_bytes += len(values)
        logits = self._torch.cat(output_groups, dim=1)[0]
        hidden = self._torch.cat(hidden_groups, dim=1)[0]
        if logits.shape != (len(values), self.model.config.vocab_size):
            raise AssertionError("block kernel logit shape differs")
        if hidden.shape != (len(values), self.base.local_decoder.config.hidden_size):
            raise AssertionError("block kernel hidden shape differs")
        return BlockTransaction(
            runtime=self,
            start=start,
            values=values,
            logits=logits,
            hidden=hidden,
            encoder_states=encoder_states,
            pre_pending=pre_pending,
            pre_global_state=pre_global_state,
            pre_boundaries=previous_boundaries,
            boundaries_after=tuple(boundaries_after),
            global_states_after=tuple(global_states_by_offset),
            boundary_emitted=tuple(boundary_emitted),
            selector_before=selector_before,
            selectors_after=tuple(selectors_after),
            encoder_cache_start=encoder_cache_start,
            decoder_cache_start=decoder_cache_start,
            global_cache_start=global_cache_start,
        )

    def _finish_block_transaction(
        self,
        transaction: BlockTransaction,
        keep: int,
    ) -> None:
        if transaction.runtime is not self or len(self._data) != (
            transaction.start + len(transaction.values)
        ):
            raise RuntimeError("block transaction no longer matches runtime state")
        if keep == len(transaction.values):
            return

        self.encoder_cache.crop(transaction.encoder_cache_start + keep)
        self.decoder_cache.crop(transaction.decoder_cache_start + keep)
        emitted_kept = sum(transaction.boundary_emitted[:keep])
        self.global_cache.crop(transaction.global_cache_start + emitted_kept)
        del self._data[transaction.start + keep :]
        boundaries = (
            transaction.pre_boundaries
            if keep == 0
            else transaction.boundaries_after[keep - 1]
        )
        self._boundaries = boundaries
        self._current_global_state = (
            transaction.pre_global_state
            if keep == 0
            else transaction.global_states_after[keep - 1]
        )

        emitted_offsets = [
            index
            for index, emitted in enumerate(transaction.boundary_emitted[:keep])
            if emitted
        ]
        if emitted_offsets:
            pending_start = emitted_offsets[-1] + 1
            self._pending_encoder_states = (
                []
                if pending_start == keep
                else [transaction.encoder_states[:, pending_start:keep, :]]
            )
        else:
            self._pending_encoder_states = list(transaction.pre_pending)
            if keep:
                self._pending_encoder_states.append(
                    transaction.encoder_states[:, :keep, :]
                )

        selector = self._copy_selector(
            transaction.selector_before
            if keep == 0
            else transaction.selectors_after[keep - 1]
        )
        if selector.boundaries != boundaries or selector.observed_bytes != len(self._data):
            raise AssertionError("rolled-back selector boundaries differ")
        self._structural_selector = selector
        diagnostics = self.diagnostics
        if (
            diagnostics.local_encoder_cached_bytes != len(self._data)
            or diagnostics.local_decoder_cached_bytes != len(self._data)
            or diagnostics.global_cached_patches != len(boundaries)
        ):
            raise AssertionError("rolled-back cache lengths differ")

    def consume_block(self, values: bytes) -> Any:
        """Consume a perfect known block and return one next-logit row per byte."""

        transaction = self.consume_block_transaction(values)
        transaction.finish(len(values))
        return transaction.logits
