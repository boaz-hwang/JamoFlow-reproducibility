"""Random-weight runtimes for scalar/hybrid latent and BPE controls.

This is an exploratory systems preflight.  The unit BLT preserves the Phase-3
W72 local/global graph and resident hash table, but replaces byte positions by
reversible units and adds small conditional output heads.  Random weights say
nothing about language-model quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from jamoflow.compute_conversion import conversion_model_spec
from jamoflow.incremental_blt import (
    IncrementalStructuralSelector,
    structural_prefix_boundaries,
)
from jamoflow.neural_model import build_main_model
from jamoflow.phase2_patching import padded_hf_patch_matrix
from scalar_representation_core import hangul_components, is_precomposed_hangul


PROTOCOL_ID = "jamoflow-scalar-runtime-preflight-v1"
MODEL_SEED = 20_260_814
W72_SPEC = conversion_model_spec(72)
GLOBAL_POSITION_LIMIT = 1_032
REPRESENTATIONS = ("generic_unicode_scalar", "hangul_hybrid")
RUNTIME_ROLES = (
    "byte_w72",
    "generic_unicode_scalar",
    "hangul_hybrid",
    "byte_bpe_32000",
    "byte_bpe_16000",
)
BPE_PRIMARY_SPEC = {
    "vocabulary_size": 32_000,
    "hidden_size": 256,
    "intermediate_size": 800,
    "layers": 13,
    "attention_heads": 4,
    "key_value_heads": 4,
    "maximum_positions": 1_024,
}
BPE_SECONDARY_SPEC = {
    "vocabulary_size": 16_000,
    "hidden_size": 320,
    "intermediate_size": 1_248,
    "layers": 9,
    "attention_heads": 5,
    "key_value_heads": 5,
    "maximum_positions": 1_024,
}


@dataclass(frozen=True, slots=True)
class EncodedUnit:
    raw: bytes
    kind: str
    primary_target: int
    conditional_targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.raw or not all(0 <= value <= 255 for value in self.raw):
            raise ValueError("encoded unit needs non-empty raw bytes")
        if self.kind not in {"generic_scalar", "raw_byte", "hangul"}:
            raise ValueError("encoded unit kind differs")


def encode_units(data: bytes, representation: str) -> tuple[EncodedUnit, ...]:
    if representation not in REPRESENTATIONS:
        raise ValueError("unknown scalar runtime representation")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("runtime cases must contain strict complete UTF-8") from error
    output: list[EncodedUnit] = []
    for character in text:
        raw = character.encode("utf-8")
        if representation == "generic_unicode_scalar":
            output.append(
                EncodedUnit(
                    raw=raw,
                    kind="generic_scalar",
                    primary_target=raw[0],
                    conditional_targets=tuple(value - 0x80 for value in raw[1:]),
                )
            )
        elif is_precomposed_hangul(ord(character)):
            onset, vowel, coda = hangul_components(ord(character))
            output.append(
                EncodedUnit(
                    raw=raw,
                    kind="hangul",
                    primary_target=256 + onset,
                    conditional_targets=(vowel, coda),
                )
            )
        else:
            output.extend(
                EncodedUnit(
                    raw=bytes((value,)),
                    kind="raw_byte",
                    primary_target=value,
                    conditional_targets=(),
                )
                for value in raw
            )
    if not output or b"".join(unit.raw for unit in output) != data:
        raise AssertionError("runtime unit encoding is not reversible")
    return tuple(output)


def decode_units(units: Iterable[EncodedUnit]) -> bytes:
    return b"".join(unit.raw for unit in units)


def unit_raw_starts(units: Sequence[EncodedUnit]) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    for unit in units:
        starts.append(offset)
        offset += len(unit.raw)
    return tuple(starts)


def w72_unit_boundaries(
    units: Sequence[EncodedUnit],
    *,
    horizon: int = 512,
) -> tuple[int, ...]:
    raw = decode_units(units)
    raw_boundaries = structural_prefix_boundaries(
        raw,
        "causal_whitespace_grid",
        horizon=horizon,
        patch_count=72,
        fixed_stride=6,
    )
    by_start = {offset: index for index, offset in enumerate(unit_raw_starts(units))}
    missing = tuple(value for value in raw_boundaries if value not in by_start)
    if missing:
        raise ValueError(f"W72 boundary does not align to a unit: {missing[:4]}")
    output = tuple(by_start[value] for value in raw_boundaries)
    if not output or output[0] != 0 or tuple(sorted(set(output))) != output:
        raise ValueError("unit W72 boundaries are malformed")
    return output


_COMPONENT_ROWS = 128


def unit_symbol_id(unit: EncodedUnit, representation: str) -> int:
    """Return the canonical integer alphabet used by the unit n-gram hash.

    BLT's byte model hashes byte trigrams into its resident 8,192-row table.
    The unit controls retain that parameter budget and hash *unit* trigrams
    instead.  Raw bytes keep 0..255; scalar and Hangul units occupy disjoint
    logical namespaces before hashing.
    """

    if representation == "generic_unicode_scalar":
        if unit.kind != "generic_scalar":
            raise ValueError("generic unit kind differs")
        return 256 + ord(unit.raw.decode("utf-8", errors="strict"))
    if representation != "hangul_hybrid":
        raise ValueError("unknown scalar runtime representation")
    if unit.kind == "raw_byte":
        return unit.raw[0]
    if unit.kind != "hangul":
        raise ValueError("hybrid unit kind differs")
    return 256 + ord(unit.raw.decode("utf-8", errors="strict")) - 0xAC00


def _unit_trigram_hash_ids(
    symbol_ids: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    from transformers.models.blt.modeling_blt import byte_group_hash_function

    if not symbol_ids:
        raise ValueError("unit hash requires a non-empty sequence")
    values = torch.tensor([list(symbol_ids)], dtype=torch.long, device=device)
    available = W72_SPEC.hash_vocabulary - _COMPONENT_ROWS
    return (
        byte_group_hash_function(values, 3, 1_000_000_007, available)
        + _COMPONENT_ROWS
    )


class FactorizedUnitBlt(nn.Module):
    """W72 backbone with reversible unit inputs and conditional micro-heads."""

    def __init__(self, representation: str, *, seed: int = MODEL_SEED) -> None:
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError("unknown factorized unit representation")
        self.representation = representation
        self.main = build_main_model(
            W72_SPEC,
            seed=seed,
            global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
        )
        width = W72_SPEC.local_width
        if representation == "generic_unicode_scalar":
            self.continuation_heads = nn.ModuleList(
                nn.Linear(width, 64, bias=False) for _ in range(3)
            )
            self.onset_head = None
            self.vowel_head = None
            self.coda_head = None
        else:
            self.continuation_heads = nn.ModuleList()
            self.onset_head = nn.Linear(width, 19, bias=False)
            self.vowel_head = nn.Linear(width, 21, bias=False)
            self.coda_head = nn.Linear(width, 28, bias=False)

    @property
    def base(self):
        return self.main.model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _hash_embedding(self, ids: Sequence[int]) -> torch.Tensor:
        values = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        return self.base.encoder_hash_tok_embedding(values)

    def embed_units(self, units: Sequence[EncodedUnit]) -> torch.Tensor:
        if not units:
            raise ValueError("unit embedding requires a non-empty sequence")
        first = torch.tensor(
            [[unit.raw[0] for unit in units]],
            dtype=torch.long,
            device=self.device,
        )
        embeddings = self.base.local_encoder.embed_tokens(first)
        symbol_ids = [unit_symbol_id(unit, self.representation) for unit in units]
        trigram_ids = _unit_trigram_hash_ids(symbol_ids, device=self.device)
        embeddings = embeddings + self.base.encoder_hash_tok_embedding(trigram_ids)
        if self.representation == "hangul_hybrid":
            for index, unit in enumerate(units):
                if unit.kind != "hangul":
                    continue
                onset = unit.primary_target - 256
                vowel, coda = unit.conditional_targets
                component_ids = (onset, 32 + vowel, 64 + coda)
                component = self._hash_embedding(component_ids).sum(dim=1)
                embeddings[:, index, :] += component
        return embeddings

    def embed_incremental_unit(
        self,
        unit: EncodedUnit,
        prior_units: Sequence[EncodedUnit],
    ) -> torch.Tensor:
        """Embed one appended unit with the same causal trigram as full input."""

        context = tuple(prior_units[-2:]) + (unit,)
        ids = [unit_symbol_id(value, self.representation) for value in context]
        first = torch.tensor(
            [[unit.raw[0]]], dtype=torch.long, device=self.device
        )
        embedding = self.base.local_encoder.embed_tokens(first)
        hash_id = _unit_trigram_hash_ids(ids, device=self.device)[:, -1:]
        embedding = embedding + self.base.encoder_hash_tok_embedding(hash_id)
        if self.representation == "hangul_hybrid" and unit.kind == "hangul":
            onset = unit.primary_target - 256
            vowel, coda = unit.conditional_targets
            component_ids = (onset, 32 + vowel, 64 + coda)
            embedding = embedding + self._hash_embedding(component_ids).sum(
                dim=1, keepdim=True
            )
        return embedding

    def output_logits(
        self,
        hidden: torch.Tensor,
        target: EncodedUnit,
    ) -> tuple[torch.Tensor, ...]:
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != 1:
            raise ValueError("incremental unit hidden state must have shape (1,1,d)")
        primary = self.main.lm_head(hidden).float()
        if self.representation == "generic_unicode_scalar":
            if target.kind != "generic_scalar":
                raise ValueError("generic target unit differs")
            outputs: list[torch.Tensor] = [primary]
            condition = hidden
            previous = target.raw[0]
            for index, component in enumerate(target.conditional_targets):
                previous_tensor = torch.tensor(
                    [[previous]], dtype=torch.long, device=self.device
                )
                condition = condition + self.base.local_encoder.embed_tokens(
                    previous_tensor
                )
                outputs.append(self.continuation_heads[index](condition).float())
                previous = component + 0x80
            return tuple(outputs)
        if target.kind == "hangul":
            if self.onset_head is None or self.vowel_head is None or self.coda_head is None:
                raise RuntimeError("Hangul conditional heads are unavailable")
            primary = torch.cat((primary, self.onset_head(hidden).float()), dim=-1)
            onset = target.primary_target - 256
            vowel, _ = target.conditional_targets
            onset_embedding = self._hash_embedding((onset,))
            vowel_logits = self.vowel_head(hidden + onset_embedding).float()
            vowel_embedding = self._hash_embedding((32 + vowel,))
            coda_logits = self.coda_head(
                hidden + onset_embedding + vowel_embedding
            ).float()
            return primary, vowel_logits, coda_logits
        if target.kind != "raw_byte":
            raise ValueError("hybrid raw target unit differs")
        return (torch.cat((primary, self.onset_head(hidden).float()), dim=-1),)

    def sample_fixed_target_route(
        self,
        hidden: torch.Tensor,
        target: EncodedUnit,
    ) -> tuple[torch.Tensor, ...]:
        """Execute the sampling dependencies for a predeclared target route.

        Random weights cannot generate a meaningful Korean continuation.  The
        preflight therefore fixes whether the next unit is a scalar, Hangul,
        or raw byte from the controlled continuation, while every component
        value used by the following micro-head is the device-side argmax.  This
        measures the conditional kernel dependency without pretending that a
        random model produced the target text.
        """

        if hidden.ndim != 3 or hidden.shape[:2] != (1, 1):
            raise ValueError("fixed-route sampling requires one hidden state")
        primary = self.main.lm_head(hidden).float()
        choices: list[torch.Tensor] = []
        if self.representation == "generic_unicode_scalar":
            if target.kind != "generic_scalar":
                raise ValueError("generic target unit differs")
            lead_ranges = {
                1: (0x00, 0x80),
                2: (0xC2, 0xE0),
                3: (0xE0, 0xF0),
                4: (0xF0, 0xF5),
            }
            start, end = lead_ranges[len(target.raw)]
            choice = primary[..., start:end].argmax(dim=-1) + start
            choices.append(choice)
            condition = hidden
            for index in range(len(target.conditional_targets)):
                if index == 0:
                    embedding_id = choice
                else:
                    embedding_id = choice + 0x80
                condition = condition + self.base.local_encoder.embed_tokens(
                    embedding_id
                )
                logits = self.continuation_heads[index](condition).float()
                choice = logits.argmax(dim=-1)
                choices.append(choice)
            return tuple(choices)
        if self.onset_head is None or self.vowel_head is None or self.coda_head is None:
            raise RuntimeError("Hangul conditional heads are unavailable")
        onset_logits = self.onset_head(hidden).float()
        combined = torch.cat((primary, onset_logits), dim=-1)
        # The route kind is the fixed controlled target; sample only within it.
        # ``combined`` is still materialized so the joint primary head cost is
        # present in the timed graph.
        first_choice = (
            primary.argmax(dim=-1)
            if target.kind == "raw_byte"
            else onset_logits.argmax(dim=-1)
        )
        choices.append(first_choice)
        if target.kind == "raw_byte":
            return tuple(choices)
        if target.kind != "hangul":
            raise ValueError("hybrid target unit differs")
        onset_choice = first_choice
        onset_embedding = self.base.encoder_hash_tok_embedding(onset_choice)
        vowel_logits = self.vowel_head(hidden + onset_embedding).float()
        vowel_choice = vowel_logits.argmax(dim=-1)
        choices.append(vowel_choice)
        vowel_embedding = self.base.encoder_hash_tok_embedding(32 + vowel_choice)
        coda_logits = self.coda_head(
            hidden + onset_embedding + vowel_embedding
        ).float()
        choices.append(coda_logits.argmax(dim=-1))
        return tuple(choices)

    def full_hidden(
        self,
        units: Sequence[EncodedUnit],
        boundaries: tuple[int, ...],
    ) -> torch.Tensor:
        matrix = padded_hf_patch_matrix([boundaries], len(units))
        patch_lengths = torch.from_numpy(matrix.astype(np.int64, copy=False)).to(
            self.device
        )
        return self.main.model(
            inputs_embeds=self.embed_units(units),
            patch_lengths=patch_lengths,
            use_cache=False,
        ).last_hidden_state


class IncrementalUnitBltDecoder:
    """Batch-1 incremental cache for one precomputed causal unit schedule."""

    def __init__(self, model: FactorizedUnitBlt) -> None:
        from transformers import DynamicCache

        self.model = model.eval()
        self.base = model.base
        self.device = model.device
        self.encoder_cache = DynamicCache(config=self.base.local_encoder.config)
        self.global_cache = DynamicCache(config=self.base.global_transformer.config)
        self.decoder_cache = DynamicCache(config=self.base.local_decoder.config)
        self.units: list[EncodedUnit] = []
        self.boundaries: tuple[int, ...] = ()
        self.pending_encoder_states: list[torch.Tensor] = []
        self.current_global_state: torch.Tensor | None = None
        self.selector = IncrementalStructuralSelector(
            "causal_whitespace_grid",
            horizon=512,
            patch_count=72,
            fixed_stride=6,
        )
        self.observed_raw_bytes = 0

    def _select_unit_boundary(self, unit: EncodedUnit) -> bool:
        before = len(self.selector.boundaries)
        unit_start = self.observed_raw_bytes
        for value in unit.raw:
            boundaries = self.selector.consume(value)
        self.observed_raw_bytes += len(unit.raw)
        appeared = boundaries[before:]
        if len(appeared) > 1 or (appeared and appeared[0] != unit_start):
            raise RuntimeError("raw W72 boundary does not align to the current unit")
        return bool(appeared)

    def _advance_local_encoder(
        self, unit: EncodedUnit, position: int
    ) -> torch.Tensor:
        encoder = self.base.local_encoder
        hidden = self.model.embed_incremental_unit(unit, self.units)
        position_ids = torch.tensor(
            [[position]], dtype=torch.long, device=self.device
        )
        positions = encoder.rotary_emb(hidden, position_ids)
        for layer in encoder.layers:
            hidden = layer(
                hidden,
                position_embeddings=positions,
                attention_mask=None,
                past_key_values=self.encoder_cache,
                use_cache=True,
            )
        return hidden

    def _finalize_patch(self, patch_index: int) -> torch.Tensor:
        if not self.pending_encoder_states:
            raise RuntimeError("cannot finalize an empty unit patch")
        encoder = self.base.local_encoder
        states = torch.cat(self.pending_encoder_states, dim=1)
        reduced = states.amax(dim=1, keepdim=True)
        queries = encoder.patch_embedding_projection(reduced).reshape(
            1, encoder.config.cross_attn_k, encoder.config.hidden_size
        )
        cross, _ = encoder.cross_attn_layers[0](
            hidden_states=queries,
            cross_attention_states=states,
            attention_mask=None,
        )
        encoded = (queries + cross).reshape(1, 1, -1)
        position = torch.tensor(
            [[patch_index]], dtype=torch.long, device=self.device
        )
        output = self.base.global_transformer(
            inputs_embeds=encoded,
            attention_mask=None,
            position_ids=position,
            past_key_values=self.global_cache,
        )
        self.pending_encoder_states.clear()
        return output

    def _advance_decoder(
        self, encoder_state: torch.Tensor, position: int
    ) -> torch.Tensor:
        if self.current_global_state is None:
            raise RuntimeError("unit decoder lacks a global state")
        decoder = self.base.local_decoder
        patch = decoder.patch_embedding_projection(
            self.current_global_state
        ).reshape(1, decoder.config.cross_attn_k, decoder.config.hidden_size)
        hidden = encoder_state
        position_ids = torch.tensor(
            [[position]], dtype=torch.long, device=self.device
        )
        positions = decoder.rotary_emb(hidden, position_ids)
        for index, layer in enumerate(decoder.layers):
            cross, _ = decoder.cross_attn_layers[index](
                hidden_states=hidden,
                cross_attention_states=patch,
                attention_mask=None,
            )
            hidden = layer(
                hidden + cross,
                position_embeddings=positions,
                attention_mask=None,
                past_key_values=self.decoder_cache,
                use_cache=True,
            )
        return decoder.norm(hidden)

    def consume(
        self,
        unit: EncodedUnit,
        *,
        boundary: bool | None = None,
    ) -> torch.Tensor:
        position = len(self.units)
        selected = self._select_unit_boundary(unit)
        if boundary is not None and boundary != selected:
            raise ValueError("provided unit boundary differs from online selector")
        boundary = selected
        encoder_state = self._advance_local_encoder(unit, position)
        self.pending_encoder_states.append(encoder_state)
        if boundary:
            self.current_global_state = self._finalize_patch(len(self.boundaries))
            self.boundaries = (*self.boundaries, position)
        self.units.append(unit)
        return self._advance_decoder(encoder_state, position)

    def prefill_parallel(
        self,
        units: Sequence[EncodedUnit],
        boundaries: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        from transformers.masking_utils import create_causal_mask
        from transformers.models.blt.modeling_blt import (
            _prepare_patch_cross_attention_mask,
        )

        selected = tuple(
            index
            for index, unit in enumerate(units)
            if self._select_unit_boundary(unit)
        )
        if boundaries is not None and boundaries != selected:
            raise ValueError("provided unit prefill boundaries differ from selector")
        boundaries = selected
        if self.units or not units or not boundaries or boundaries[0] != 0:
            raise ValueError("unit parallel prefill state or schedule differs")
        sequence_length = len(units)
        embeddings = self.model.embed_units(units)
        position_ids = torch.arange(
            sequence_length, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        local_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=embeddings,
            attention_mask=None,
            past_key_values=self.encoder_cache,
            position_ids=position_ids,
        )
        encoder = self.base.local_encoder
        encoder_hidden = embeddings
        positions = encoder.rotary_emb(encoder_hidden, position_ids)
        for layer in encoder.layers:
            encoder_hidden = layer(
                encoder_hidden,
                position_embeddings=positions,
                attention_mask=local_mask,
                past_key_values=self.encoder_cache,
                use_cache=True,
            )

        matrix = padded_hf_patch_matrix([boundaries], sequence_length)
        patch_lengths = torch.from_numpy(
            matrix.astype(np.int64, copy=False)
        ).to(self.device)
        encoder_patch_ids = self.base._patch_ids_from_lengths(
            patch_lengths, sequence_length
        )
        all_queries = encoder.patch_embedding_projection(
            encoder.patch_reduce(
                encoder_hidden,
                patch_lengths.shape[1],
                encoder_patch_ids,
            )
        ).reshape(
            1,
            patch_lengths.shape[1] * encoder.config.cross_attn_k,
            encoder.config.hidden_size,
        )
        closed = len(boundaries)
        queries = all_queries[:, : closed * encoder.config.cross_attn_k, :]
        encoder_mask = _prepare_patch_cross_attention_mask(
            patch_ids=encoder_patch_ids,
            num_patches=patch_lengths.shape[1],
            sequence_length=sequence_length,
            patches_as_queries=True,
            cross_attn_k=encoder.config.cross_attn_k,
            dtype=encoder_hidden.dtype,
        )[:, :, : closed * encoder.config.cross_attn_k, :]
        cross, _ = encoder.cross_attn_layers[0](
            hidden_states=queries,
            cross_attention_states=encoder_hidden,
            attention_mask=encoder_mask,
        )
        encoded = (queries + cross).reshape(1, closed, -1)

        global_positions = torch.arange(
            closed, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        projected = self.base.global_transformer.token_embedding_projection(encoded)
        global_mask = create_causal_mask(
            config=self.base.config,
            inputs_embeds=projected,
            attention_mask=None,
            past_key_values=self.global_cache,
            position_ids=global_positions,
        )
        global_hidden = self.base.global_transformer(
            inputs_embeds=encoded,
            attention_mask=global_mask,
            position_ids=global_positions,
            past_key_values=self.global_cache,
        )
        self.current_global_state = global_hidden[:, -1:, :]

        decoder = self.base.local_decoder
        decoder_patch_ids = self.base._patch_ids_from_lengths(
            patch_lengths[:, 1:], sequence_length
        )
        decoder_mask = _prepare_patch_cross_attention_mask(
            patch_ids=decoder_patch_ids,
            num_patches=closed,
            sequence_length=sequence_length,
            patches_as_queries=False,
            cross_attn_k=decoder.config.cross_attn_k,
            dtype=encoder_hidden.dtype,
        )
        decoder_hidden = decoder(
            input_ids=None,
            inputs_embeds=encoder_hidden,
            patch_embeds=global_hidden,
            attention_mask=local_mask,
            position_ids=position_ids,
            past_key_values=self.decoder_cache,
            encoder_attention_mask=decoder_mask,
        )
        pending_start = boundaries[-1] + 1
        if pending_start < sequence_length:
            self.pending_encoder_states = [encoder_hidden[:, pending_start:, :]]
        self.units.extend(units)
        self.boundaries = boundaries
        return decoder_hidden[:, -1:, :]

    def diagnostics(self) -> dict[str, int]:
        return {
            "cached_units_encoder": int(self.encoder_cache.get_seq_length()),
            "cached_units_decoder": int(self.decoder_cache.get_seq_length()),
            "cached_global_patches": int(self.global_cache.get_seq_length()),
            "observed_units": len(self.units),
            "observed_raw_bytes": len(decode_units(self.units)),
            "emitted_data_patches": len(self.boundaries),
        }


def build_bpe_model(spec: dict[str, int], *, seed: int = MODEL_SEED):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=spec["vocabulary_size"],
        hidden_size=spec["hidden_size"],
        intermediate_size=spec["intermediate_size"],
        num_hidden_layers=spec["layers"],
        num_attention_heads=spec["attention_heads"],
        num_key_value_heads=spec["key_value_heads"],
        max_position_embeddings=spec["maximum_positions"],
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
    )
    return LlamaForCausalLM(config).to(dtype=torch.float32)


class IncrementalBpeDecoder:
    def __init__(self, model: Any) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.cache = None
        self.observed_tokens = 0

    def prefill_parallel(self, token_ids: Sequence[int]) -> torch.Tensor:
        if self.observed_tokens or not token_ids:
            raise ValueError("BPE parallel prefill state differs")
        values = torch.tensor(
            [list(token_ids)], dtype=torch.long, device=self.device
        )
        output = self.model(
            input_ids=values,
            use_cache=True,
            logits_to_keep=1,
        )
        self.cache = output.past_key_values
        self.observed_tokens = len(token_ids)
        return output.logits[:, -1, :].float()

    def consume(self, token_id: int) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("BPE decoder must be prefixed first")
        value = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        output = self.model(
            input_ids=value,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.cache = output.past_key_values
        self.observed_tokens += 1
        return output.logits[:, -1, :].float()


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def maximum_normalized_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> float:
    if actual.shape != expected.shape:
        raise ValueError("correctness tensors have different shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("correctness tensors must be finite")
    denominator = atol + rtol * expected.abs()
    ratio = (actual - expected).abs() / denominator
    return float(ratio.max().item())
