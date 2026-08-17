"""Core contracts for the 2K-to-8K vocabulary-transfer development probe."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from compositional_head_core import ROLE_SPECS, build_model
from torch import nn

TRANSFER_ROLES = (
    "tied_random_norm",
    "tied_uniform_norm",
    "tied_byte_weighted_norm",
    "tied_last_subpiece",
    "untied_random_norm",
    "untied_uniform_in_uniform_out",
    "untied_uniform_in_byte_weighted_out",
)
COMPOSED_ROLES = (
    "tied_uniform_norm",
    "tied_byte_weighted_norm",
    "tied_last_subpiece",
    "untied_uniform_in_uniform_out",
    "untied_uniform_in_byte_weighted_out",
)
RANDOM_CONTROL_BY_ROLE = {
    "tied_uniform_norm": "tied_random_norm",
    "tied_byte_weighted_norm": "tied_random_norm",
    "tied_last_subpiece": "tied_random_norm",
    "untied_uniform_in_uniform_out": "untied_random_norm",
    "untied_uniform_in_byte_weighted_out": "untied_random_norm",
}
BASE_VOCABULARY_SIZE = 2_048
TARGET_VOCABULARY_SIZE = 8_192
PROBE_STEPS = (0, 32, 128, 512)
FINAL_PROBE_STEP = PROBE_STEPS[-1]
MODEL_SEED = 20_260_824
UNTIED_OUTPUT_SEED_DOMAIN = 80_000
ORDER_SEED = 20_260_827
EFFECTIVE_BATCH_SIZE = 32
TRAIN_MICROBATCH_SIZE = 8
EVALUATION_BATCH_SIZE = 16
BODY_LEARNING_RATE = 3e-5
HEAD_PEAK_LEARNING_RATE = 3e-4
HEAD_MINIMUM_LEARNING_RATE = 3e-5
WARMUP_STEPS = 26
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
MINIMUM_INITIALIZATION_ADVANTAGE_BPB = 0.010
MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB = 0.050

_HEAD_KEYS = frozenset(("model.embed_tokens.weight", "lm_head.weight"))


@dataclass(frozen=True, slots=True)
class TransferInitializationAudit:
    role: str
    tied_input_output: bool
    input_strategy: str
    output_strategy: str
    input_norm_calibrated: bool
    output_norm_calibrated: bool
    base_vocabulary_size: int
    target_vocabulary_size: int
    shared_token_count: int
    new_token_count: int
    maximum_constituent_count: int
    mean_constituent_count: float
    decomposition_kind: str
    decomposition_sha256: str
    initialized_input_weight_sha256: str
    initialized_output_weight_sha256: str
    base_mean_row_l2: float
    initialized_input_new_mean_row_l2: float
    initialized_output_new_mean_row_l2: float
    exact_reconstruction: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_piece_table(pieces: Sequence[bytes], expected_size: int) -> tuple[bytes, ...]:
    output = tuple(pieces)
    if (
        len(output) != expected_size
        or any(not isinstance(piece, bytes) or not piece for piece in output)
        or len(set(output)) != len(output)
    ):
        raise ValueError("vocabulary-transfer piece table differs")
    return output


def build_canonical_bpe_decomposition_table(
    base_tokenizer: Any,
    target_tokenizer: Any,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
) -> tuple[tuple[int, ...], ...]:
    """Cut the target BPE merge tree at the exact source-vocabulary frontier."""

    base = _require_piece_table(base_pieces, BASE_VOCABULARY_SIZE)
    target = _require_piece_table(target_pieces, TARGET_VOCABULARY_SIZE)
    try:
        base_payload = json.loads(base_tokenizer.to_str())
        target_payload = json.loads(target_tokenizer.to_str())
        base_model = base_payload["model"]
        target_model = target_payload["model"]
        base_vocab = base_model["vocab"]
        target_vocab = target_model["vocab"]
        base_merges = tuple(tuple(row) for row in base_model["merges"])
        target_merges = tuple(tuple(row) for row in target_model["merges"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("vocabulary-transfer tokenizer merge identity differs") from error
    if (
        base_model.get("type") != "BPE"
        or target_model.get("type") != "BPE"
        or len(base_vocab) != BASE_VOCABULARY_SIZE
        or len(target_vocab) != TARGET_VOCABULARY_SIZE
        or len(base_merges) != BASE_VOCABULARY_SIZE - 256
        or len(target_merges) != TARGET_VOCABULARY_SIZE - 256
        or base_merges != target_merges[: len(base_merges)]
        or any(target_vocab.get(symbol) != token_id for symbol, token_id in base_vocab.items())
    ):
        raise ValueError("vocabulary-transfer target is not a canonical BPE extension")
    target_symbols: list[str | None] = [None] * TARGET_VOCABULARY_SIZE
    for symbol, token_id in target_vocab.items():
        if not isinstance(symbol, str) or not isinstance(token_id, int) or not 0 <= token_id < len(target_symbols):
            raise ValueError("vocabulary-transfer target vocabulary differs")
        if target_symbols[token_id] is not None:
            raise ValueError("vocabulary-transfer target token ID is duplicated")
        target_symbols[token_id] = symbol
    creation: dict[str, tuple[str, str]] = {}
    for row in target_merges:
        if len(row) != 2 or not all(isinstance(value, str) and value for value in row):
            raise ValueError("vocabulary-transfer merge row differs")
        merged = row[0] + row[1]
        if merged in creation and creation[merged] != row:
            raise ValueError("vocabulary-transfer merge genealogy is ambiguous")
        creation[merged] = row
    cache: dict[str, tuple[int, ...]] = {}
    visiting: set[str] = set()

    def expand(symbol: str) -> tuple[int, ...]:
        if symbol in cache:
            return cache[symbol]
        if symbol in visiting:
            raise ValueError("vocabulary-transfer merge genealogy contains a cycle")
        visiting.add(symbol)
        if symbol in base_vocab:
            output = (int(base_vocab[symbol]),)
        elif symbol in creation:
            left, right = creation[symbol]
            output = (*expand(left), *expand(right))
        else:
            raise ValueError("target token has no source-BPE merge genealogy")
        visiting.remove(symbol)
        cache[symbol] = output
        return output

    if any(symbol is None for symbol in target_symbols):
        raise ValueError("vocabulary-transfer target token IDs are incomplete")
    output = tuple(expand(str(symbol)) for symbol in target_symbols)
    if any(
        b"".join(base[token_id] for token_id in row) != piece
        for row, piece in zip(output, target)
    ):
        raise ValueError("vocabulary-transfer canonical BPE decomposition is not byte exact")
    return output


def decomposition_sha256(decompositions: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256(b"JamoFlow/vocabulary-transfer-decomposition/v1\0")
    for row in decompositions:
        values = tuple(int(value) for value in row)
        if not values or any(value < 0 or value >= BASE_VOCABULARY_SIZE for value in values):
            raise ValueError("vocabulary-transfer decomposition row differs")
        digest.update(len(values).to_bytes(8, "big"))
        digest.update(np.asarray(values, dtype=">u4").tobytes(order="C"))
    digest.update(len(tuple(decompositions)).to_bytes(8, "big"))
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(b"JamoFlow/vocabulary-transfer-tensor/v1\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes(order="C"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"JamoFlow/compositional-quality-state/v1\0")
    selected = 0
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("vocabulary-transfer state mapping differs")
        value = tensor.detach().cpu().contiguous()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
        selected += 1
    if selected <= 0:
        raise ValueError("vocabulary-transfer state mapping is empty")
    digest.update(selected.to_bytes(8, "big"))
    return digest.hexdigest()


def _rescale_rows(values: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    norms = values.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or torch.any(norms <= 0):
        raise ValueError("vocabulary-transfer row norm differs")
    return values * (target_norm / norms)


def role_definition(role: str) -> dict[str, Any]:
    definitions = {
        "tied_random_norm": {
            "tied": True,
            "input_strategy": "random",
            "output_strategy": "random",
            "input_norm": True,
            "output_norm": True,
        },
        "tied_uniform_norm": {
            "tied": True,
            "input_strategy": "uniform",
            "output_strategy": "uniform",
            "input_norm": True,
            "output_norm": True,
        },
        "tied_byte_weighted_norm": {
            "tied": True,
            "input_strategy": "byte_weighted",
            "output_strategy": "byte_weighted",
            "input_norm": True,
            "output_norm": True,
        },
        "tied_last_subpiece": {
            "tied": True,
            "input_strategy": "last",
            "output_strategy": "last",
            "input_norm": False,
            "output_norm": False,
        },
        "untied_random_norm": {
            "tied": False,
            "input_strategy": "random",
            "output_strategy": "random",
            "input_norm": True,
            "output_norm": True,
        },
        "untied_uniform_in_uniform_out": {
            "tied": False,
            "input_strategy": "uniform",
            "output_strategy": "uniform",
            "input_norm": True,
            "output_norm": False,
        },
        "untied_uniform_in_byte_weighted_out": {
            "tied": False,
            "input_strategy": "uniform",
            "output_strategy": "byte_weighted",
            "input_norm": True,
            "output_norm": False,
        },
    }
    if role not in definitions:
        raise ValueError("vocabulary-transfer role differs")
    return dict(definitions[role])


def expected_parameter_count(role: str) -> int:
    definition = role_definition(role)
    spec = ROLE_SPECS["dense_v8192"]
    return (
        spec.expected_parameters
        if definition["tied"]
        else spec.expected_parameters + TARGET_VOCABULARY_SIZE * spec.hidden_size
    )


def build_target_graph(role: str, *, seed: int = MODEL_SEED) -> Any:
    definition = role_definition(role)
    model = build_model("dense_v8192", seed=seed)
    if not definition["tied"]:
        generator = torch.Generator(device="cpu").manual_seed(
            seed + UNTIED_OUTPUT_SEED_DOMAIN
        )
        weight = torch.empty(
            TARGET_VOCABULARY_SIZE,
            int(model.config.hidden_size),
            dtype=torch.float32,
        )
        weight.normal_(
            mean=0.0,
            std=float(model.config.initializer_range),
            generator=generator,
        )
        output = nn.Linear(int(model.config.hidden_size), TARGET_VOCABULARY_SIZE, bias=False)
        output.weight = nn.Parameter(weight)
        model.lm_head = output
        model.config.tie_word_embeddings = False
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameter_count(role):
        raise AssertionError("vocabulary-transfer target parameter count differs")
    tied_actual = (
        model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    )
    if tied_actual is not definition["tied"]:
        raise AssertionError("vocabulary-transfer target tying differs")
    return model


def _initialize_weight(
    strategy: str,
    *,
    norm_calibrated: bool,
    base_weight: torch.Tensor,
    random_weight: torch.Tensor,
    base: Sequence[bytes],
    target: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, tuple[int, ...]]:
    base_by_piece = {piece: token_id for token_id, piece in enumerate(base)}
    target_norm = base_weight.norm(dim=1).mean()
    output = torch.empty_like(random_weight)
    new_rows: list[int] = []
    for target_id, (piece, constituent_ids) in enumerate(zip(target, decompositions)):
        if piece in base_by_piece:
            output[target_id] = base_weight[base_by_piece[piece]]
            continue
        ids = torch.tensor(tuple(constituent_ids), dtype=torch.long)
        if strategy == "random":
            row = random_weight[target_id : target_id + 1]
        elif strategy == "uniform":
            row = base_weight.index_select(0, ids).mean(dim=0, keepdim=True)
        elif strategy == "byte_weighted":
            lengths = torch.tensor(
                [len(base[token_id]) for token_id in constituent_ids], dtype=torch.float32
            )
            weights = lengths / lengths.sum()
            row = (base_weight.index_select(0, ids) * weights[:, None]).sum(
                dim=0, keepdim=True
            )
        elif strategy == "last":
            row = base_weight[constituent_ids[-1] : constituent_ids[-1] + 1]
        else:
            raise ValueError("vocabulary-transfer initialization strategy differs")
        if norm_calibrated:
            row = _rescale_rows(row, target_norm)
        output[target_id] = row[0]
        new_rows.append(target_id)
    return output, tuple(new_rows)


def initialize_target_weights(
    role: str,
    *,
    base_weight: torch.Tensor,
    target_input_random_weight: torch.Tensor,
    target_output_random_weight: torch.Tensor,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, TransferInitializationAudit]:
    definition = role_definition(role)
    base = _require_piece_table(base_pieces, BASE_VOCABULARY_SIZE)
    target = _require_piece_table(target_pieces, TARGET_VOCABULARY_SIZE)
    if (
        base_weight.ndim != 2
        or target_input_random_weight.ndim != 2
        or target_output_random_weight.ndim != 2
        or base_weight.shape[0] != BASE_VOCABULARY_SIZE
        or target_input_random_weight.shape[0] != TARGET_VOCABULARY_SIZE
        or target_output_random_weight.shape != target_input_random_weight.shape
        or base_weight.shape[1] != target_input_random_weight.shape[1]
        or base_weight.dtype != torch.float32
        or target_input_random_weight.dtype != torch.float32
        or target_output_random_weight.dtype != torch.float32
        or not torch.isfinite(base_weight).all()
        or not torch.isfinite(target_input_random_weight).all()
        or not torch.isfinite(target_output_random_weight).all()
    ):
        raise ValueError("vocabulary-transfer weight coordinates differ")
    decompositions = tuple(tuple(int(value) for value in row) for row in decompositions)
    if (
        len(decompositions) != TARGET_VOCABULARY_SIZE
        or any(not row for row in decompositions)
        or any(
            b"".join(base[token_id] for token_id in row) != piece
            for row, piece in zip(decompositions, target)
        )
    ):
        raise ValueError("vocabulary-transfer supplied decomposition differs")
    target_norm = base_weight.norm(dim=1).mean()
    if not torch.isfinite(target_norm) or float(target_norm) <= 0:
        raise ValueError("vocabulary-transfer base norm differs")
    input_weight, input_new_rows = _initialize_weight(
        definition["input_strategy"],
        norm_calibrated=definition["input_norm"],
        base_weight=base_weight,
        random_weight=target_input_random_weight,
        base=base,
        target=target,
        decompositions=decompositions,
    )
    if definition["tied"]:
        output_weight = input_weight
        output_new_rows = input_new_rows
    else:
        output_weight, output_new_rows = _initialize_weight(
            definition["output_strategy"],
            norm_calibrated=definition["output_norm"],
            base_weight=base_weight,
            random_weight=target_output_random_weight,
            base=base,
            target=target,
            decompositions=decompositions,
        )
    if (
        len(input_new_rows) != TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE
        or output_new_rows != input_new_rows
    ):
        raise AssertionError("vocabulary-transfer shared token inventory differs")
    new_index = torch.tensor(input_new_rows, dtype=torch.long)
    audit = TransferInitializationAudit(
        role=role,
        tied_input_output=definition["tied"],
        input_strategy=definition["input_strategy"],
        output_strategy=definition["output_strategy"],
        input_norm_calibrated=definition["input_norm"],
        output_norm_calibrated=definition["output_norm"],
        base_vocabulary_size=BASE_VOCABULARY_SIZE,
        target_vocabulary_size=TARGET_VOCABULARY_SIZE,
        shared_token_count=BASE_VOCABULARY_SIZE,
        new_token_count=len(input_new_rows),
        maximum_constituent_count=max(len(row) for row in decompositions),
        mean_constituent_count=float(np.mean([len(row) for row in decompositions])),
        decomposition_kind="canonical_target_merge_tree_cut_at_source_vocab",
        decomposition_sha256=decomposition_sha256(decompositions),
        initialized_input_weight_sha256=tensor_sha256(input_weight),
        initialized_output_weight_sha256=tensor_sha256(output_weight),
        base_mean_row_l2=float(target_norm),
        initialized_input_new_mean_row_l2=float(
            input_weight.index_select(0, new_index).norm(dim=1).mean()
        ),
        initialized_output_new_mean_row_l2=float(
            output_weight.index_select(0, new_index).norm(dim=1).mean()
        ),
        exact_reconstruction=True,
    )
    return input_weight, output_weight, audit


def build_transferred_model(
    role: str,
    *,
    base_state: Mapping[str, torch.Tensor],
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
) -> tuple[Any, TransferInitializationAudit]:
    if role not in TRANSFER_ROLES or set(_HEAD_KEYS) - set(base_state):
        raise ValueError("vocabulary-transfer base checkpoint differs")
    definition = role_definition(role)
    model = build_target_graph(role, seed=MODEL_SEED)
    target_state = model.state_dict()
    if set(base_state) != set(target_state):
        raise ValueError("vocabulary-transfer model state key set differs")
    copied: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        if name in _HEAD_KEYS:
            copied[name] = target_value
            continue
        source = base_state[name]
        if source.shape != target_value.shape or source.dtype != target_value.dtype:
            raise ValueError("vocabulary-transfer body state differs")
        copied[name] = source.detach().cpu().contiguous()
    base_weight = base_state["model.embed_tokens.weight"].detach().cpu().contiguous()
    if not torch.equal(base_weight, base_state["lm_head.weight"].detach().cpu()):
        raise ValueError("vocabulary-transfer source tied head differs")
    input_initialized, output_initialized, audit = initialize_target_weights(
        role,
        base_weight=base_weight,
        target_input_random_weight=target_state["model.embed_tokens.weight"]
        .detach()
        .cpu()
        .contiguous(),
        target_output_random_weight=target_state["lm_head.weight"]
        .detach()
        .cpu()
        .contiguous(),
        base_pieces=base_pieces,
        target_pieces=target_pieces,
        decompositions=decompositions,
    )
    copied["model.embed_tokens.weight"] = input_initialized
    copied["lm_head.weight"] = output_initialized
    model.load_state_dict(copied, strict=True)
    tied_actual = model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    if tied_actual is not definition["tied"]:
        raise AssertionError("vocabulary-transfer target embedding tying differs")
    return model, audit


def probe_learning_rate(step: int, *, peak: float, minimum: float) -> float:
    if not 0 <= step < FINAL_PROBE_STEP or not 0 < minimum <= peak:
        raise ValueError("vocabulary-transfer learning-rate coordinates differ")
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, FINAL_PROBE_STEP - WARMUP_STEPS - 1)
    return minimum + (peak - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def transfer_probe_decision(
    final_bpb_by_role: Mapping[str, float], *, anchor_bpb: float
) -> dict[str, Any]:
    if (
        set(final_bpb_by_role) != set(TRANSFER_ROLES)
        or not math.isfinite(anchor_bpb)
        or anchor_bpb <= 0
        or any(not math.isfinite(float(value)) or float(value) <= 0 for value in final_bpb_by_role.values())
    ):
        raise ValueError("vocabulary-transfer decision inputs differ")
    diagnostics: dict[str, dict[str, Any]] = {}
    qualified: list[str] = []
    for role in COMPOSED_ROLES:
        control = RANDOM_CONTROL_BY_ROLE[role]
        value = float(final_bpb_by_role[role])
        advantage = float(final_bpb_by_role[control]) - value
        anchor_gap = value - anchor_bpb
        row = {
            "architecture": "tied" if role.startswith("tied_") else "untied",
            "random_control": control,
            "final_bpb": value,
            "random_control_final_bpb": float(final_bpb_by_role[control]),
            "composed_advantage_bpb": advantage,
            "anchor_gap_bpb": anchor_gap,
            "initialization_advantage_pass": advantage
            >= MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
            "recovery_progress_pass": anchor_gap
            <= MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
        }
        row["joint_pass"] = (
            row["initialization_advantage_pass"] and row["recovery_progress_pass"]
        )
        diagnostics[role] = row
        if row["joint_pass"]:
            qualified.append(role)
    lowest_observed = min(
        COMPOSED_ROLES,
        key=lambda role: (float(final_bpb_by_role[role]), COMPOSED_ROLES.index(role)),
    )
    selected = (
        min(qualified, key=lambda role: (float(final_bpb_by_role[role]), COMPOSED_ROLES.index(role)))
        if qualified
        else None
    )
    selected_row = diagnostics[selected] if selected is not None else None
    authorized = selected is not None
    return {
        "selected_composed_initializer": selected,
        "lowest_observed_composed_initializer": lowest_observed,
        "lowest_observed_composed_bpb": float(final_bpb_by_role[lowest_observed]),
        "selected_architecture": selected_row["architecture"] if selected_row else None,
        "selected_final_bpb": selected_row["final_bpb"] if selected_row else None,
        "selected_random_control": selected_row["random_control"] if selected_row else None,
        "selected_random_control_final_bpb": (
            selected_row["random_control_final_bpb"] if selected_row else None
        ),
        "selected_composed_advantage_bpb": (
            selected_row["composed_advantage_bpb"] if selected_row else None
        ),
        "dense_2k_anchor_bpb": anchor_bpb,
        "selected_anchor_gap_bpb": selected_row["anchor_gap_bpb"] if selected_row else None,
        "minimum_initialization_advantage_bpb": MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
        "maximum_anchor_gap_for_full_cpt_bpb": MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
        "candidate_diagnostics": diagnostics,
        "qualified_candidates": qualified,
        "full_cpt_authorized": authorized,
        "status": "vocabulary_transfer_probe_pass" if authorized else "vocabulary_transfer_probe_stopped",
        "no_korean_specific_role_evaluated": True,
    }
