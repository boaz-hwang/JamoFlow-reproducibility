"""Strong generic baselines for the 2K-to-8K vocabulary-transfer branch."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
    MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
    MODEL_SEED,
    TARGET_VOCABULARY_SIZE,
    decomposition_sha256,
    state_mapping_sha256,
    tensor_sha256,
)
from vocabulary_transfer_probe_core import (
    build_target_graph as build_probe_target_graph,
)

BASELINE_ROLES = (
    "untied_random_hangul_median_input_native_output",
    "untied_bil_hangul_median_char_out",
    "untied_bil_global_median_char_out",
    "untied_bil_hangul_median_uniform_out",
    "untied_eeve_uniform_in_first_out",
    "tied_random_native_all",
    "tied_uniform_no_norm_all",
    "tied_random_native_two_stage",
    "tied_uniform_no_norm_two_stage",
)
COMPOSED_BASELINE_ROLES = (
    "untied_bil_hangul_median_char_out",
    "untied_bil_global_median_char_out",
    "untied_bil_hangul_median_uniform_out",
    "untied_eeve_uniform_in_first_out",
    "tied_uniform_no_norm_all",
    "tied_uniform_no_norm_two_stage",
)
RANDOM_CONTROL_BY_ROLE = {
    "untied_bil_hangul_median_char_out": (
        "untied_random_hangul_median_input_native_output"
    ),
    "untied_bil_global_median_char_out": (
        "untied_random_hangul_median_input_native_output"
    ),
    "untied_bil_hangul_median_uniform_out": (
        "untied_random_hangul_median_input_native_output"
    ),
    "untied_eeve_uniform_in_first_out": (
        "untied_random_hangul_median_input_native_output"
    ),
    "tied_uniform_no_norm_all": "tied_random_native_all",
    "tied_uniform_no_norm_two_stage": "tied_random_native_two_stage",
}
PROBE_STEPS = (0, 32, 50, 128, 307, 512)
FINAL_PROBE_STEP = PROBE_STEPS[-1]
TWO_STAGE_BOUNDARY = 307
TWO_STAGE_FULL_STEPS = FINAL_PROBE_STEP - TWO_STAGE_BOUNDARY

_HEAD_KEYS = frozenset(("model.embed_tokens.weight", "lm_head.weight"))


@dataclass(frozen=True, slots=True)
class SourceTokenMetadata:
    decoded_character_lengths: tuple[int, ...]
    decoded_character_lengths_sha256: str
    hangul_token_mask: tuple[bool, ...]
    hangul_token_mask_sha256: str
    hangul_token_count: int
    replacement_character_token_count: int
    length_definition: str
    script_definition: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decoded_character_lengths"] = list(self.decoded_character_lengths)
        value["hangul_token_mask"] = list(self.hangul_token_mask)
        return value


@dataclass(frozen=True, slots=True)
class BaselineInitializationAudit:
    role: str
    tied_input_output: bool
    training_schedule: str
    input_strategy: str
    input_norm_scope: str
    input_norm_statistic: str
    input_target_norm: float | None
    output_strategy: str
    output_norm_scope: str
    base_vocabulary_size: int
    target_vocabulary_size: int
    shared_token_count: int
    new_token_count: int
    decomposition_kind: str
    decomposition_sha256: str
    maximum_constituent_count: int
    mean_constituent_count: float
    source_metadata_sha256: str
    initialized_input_weight_sha256: str
    initialized_output_weight_sha256: str
    initialized_input_new_mean_row_l2: float
    initialized_output_new_mean_row_l2: float
    copied_rows_exact: bool
    target_pieces_reconstruct_exactly: bool
    baseline_interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_hangul_character(character: str) -> bool:
    value = ord(character)
    return (
        0x1100 <= value <= 0x11FF
        or 0x3130 <= value <= 0x318F
        or 0xA960 <= value <= 0xA97F
        or 0xAC00 <= value <= 0xD7A3
        or 0xD7B0 <= value <= 0xD7FF
    )


def _metadata_hash(lengths: Sequence[int], mask: Sequence[bool]) -> str:
    lengths_array = np.asarray(tuple(lengths), dtype=">u4")
    mask_array = np.asarray(tuple(mask), dtype=np.uint8)
    digest = hashlib.sha256(b"JamoFlow/vocabulary-transfer-source-metadata/v1\0")
    digest.update(len(lengths_array).to_bytes(8, "big"))
    digest.update(lengths_array.tobytes(order="C"))
    digest.update(mask_array.tobytes(order="C"))
    return digest.hexdigest()


def source_token_metadata(base_tokenizer: Any) -> SourceTokenMetadata:
    decoded: list[str] = []
    for token_id in range(BASE_VOCABULARY_SIZE):
        value = base_tokenizer.decode([token_id], skip_special_tokens=False)
        if not isinstance(value, str):
            raise TypeError("baseline source-token decode differs")
        decoded.append(value)
    lengths = tuple(max(len(value), 1) for value in decoded)
    mask = tuple(any(_is_hangul_character(character) for character in value) for value in decoded)
    if len(lengths) != BASE_VOCABULARY_SIZE or not any(mask):
        raise ValueError("baseline source-token metadata differs")
    lengths_sha = hashlib.sha256(
        b"JamoFlow/vocabulary-transfer-decoded-lengths/v1\0"
        + np.asarray(lengths, dtype=">u4").tobytes(order="C")
    ).hexdigest()
    mask_sha = hashlib.sha256(
        b"JamoFlow/vocabulary-transfer-hangul-mask/v1\0"
        + np.asarray(mask, dtype=np.uint8).tobytes(order="C")
    ).hexdigest()
    return SourceTokenMetadata(
        decoded_character_lengths=lengths,
        decoded_character_lengths_sha256=lengths_sha,
        hangul_token_mask=mask,
        hangul_token_mask_sha256=mask_sha,
        hangul_token_count=sum(mask),
        replacement_character_token_count=sum("\ufffd" in value for value in decoded),
        length_definition=(
            "max(len(tokenizer.decode([source_id], skip_special_tokens=False)),1); "
            "Python len counts Unicode code points; no additional normalization"
        ),
        script_definition=(
            "contains U+1100-11FF, U+3130-318F, U+A960-A97F, "
            "U+AC00-D7A3, or U+D7B0-D7FF"
        ),
    )


def role_definition(role: str) -> dict[str, Any]:
    definitions = {
        "untied_random_hangul_median_input_native_output": {
            "tied": False,
            "input_strategy": "random",
            "input_norm_scope": "hangul_source_tokens",
            "output_strategy": "random",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "architecture_matched_random_control",
        },
        "untied_bil_hangul_median_char_out": {
            "tied": False,
            "input_strategy": "uniform",
            "input_norm_scope": "hangul_source_tokens",
            "output_strategy": "decoded_character_weighted",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "beyond_initialization_loss_method_exact",
        },
        "untied_bil_global_median_char_out": {
            "tied": False,
            "input_strategy": "uniform",
            "input_norm_scope": "all_source_tokens",
            "output_strategy": "decoded_character_weighted",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "bil_global_norm_ablation",
        },
        "untied_bil_hangul_median_uniform_out": {
            "tied": False,
            "input_strategy": "uniform",
            "input_norm_scope": "hangul_source_tokens",
            "output_strategy": "uniform",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "bil_output_weighting_ablation",
        },
        "untied_eeve_uniform_in_first_out": {
            "tied": False,
            "input_strategy": "uniform",
            "input_norm_scope": "none",
            "output_strategy": "first",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "eeve_initializer_only_analogue",
        },
        "tied_random_native_all": {
            "tied": True,
            "input_strategy": "random",
            "input_norm_scope": "none",
            "output_strategy": "shared_with_input",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "tied_random_control",
        },
        "tied_uniform_no_norm_all": {
            "tied": True,
            "input_strategy": "uniform",
            "input_norm_scope": "none",
            "output_strategy": "shared_with_input",
            "output_norm_scope": "none",
            "training_schedule": "all_parameters",
            "baseline_interpretation": "continued_bpe_mean_initializer",
        },
        "tied_random_native_two_stage": {
            "tied": True,
            "input_strategy": "random",
            "input_norm_scope": "none",
            "output_strategy": "shared_with_input",
            "output_norm_scope": "none",
            "training_schedule": "new_rows_then_all_307_205",
            "baseline_interpretation": "two_stage_random_control",
        },
        "tied_uniform_no_norm_two_stage": {
            "tied": True,
            "input_strategy": "uniform",
            "input_norm_scope": "none",
            "output_strategy": "shared_with_input",
            "output_norm_scope": "none",
            "training_schedule": "new_rows_then_all_307_205",
            "baseline_interpretation": "in_place_two_stage_ratio_analogue",
        },
    }
    if role not in definitions:
        raise ValueError("baseline role differs")
    return dict(definitions[role])


def expected_parameter_count(role: str) -> int:
    definition = role_definition(role)
    representative = "tied_random_norm" if definition["tied"] else "untied_random_norm"
    model = build_probe_target_graph(representative, seed=MODEL_SEED)
    return sum(parameter.numel() for parameter in model.parameters())


def build_target_graph(role: str, *, seed: int = MODEL_SEED) -> Any:
    definition = role_definition(role)
    representative = "tied_random_norm" if definition["tied"] else "untied_random_norm"
    model = build_probe_target_graph(representative, seed=seed)
    if sum(parameter.numel() for parameter in model.parameters()) != expected_parameter_count(role):
        raise AssertionError("baseline target parameter count differs")
    return model


def _midpoint_median(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or len(values) <= 0 or not torch.isfinite(values).all():
        raise ValueError("baseline norm population differs")
    sorted_values = torch.sort(values).values
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) * 0.5


def _target_norm(
    scope: str, base_weight: torch.Tensor, metadata: SourceTokenMetadata
) -> torch.Tensor | None:
    if scope == "none":
        return None
    norms = base_weight.norm(dim=1)
    if scope == "all_source_tokens":
        selected = norms
    elif scope == "hangul_source_tokens":
        mask = torch.tensor(metadata.hangul_token_mask, dtype=torch.bool)
        selected = norms[mask]
    else:
        raise ValueError("baseline norm scope differs")
    value = _midpoint_median(selected)
    if not torch.isfinite(value) or float(value) <= 0:
        raise ValueError("baseline target norm differs")
    return value


def _rescale(row: torch.Tensor, target_norm: torch.Tensor | None) -> torch.Tensor:
    if target_norm is None:
        return row
    norm = row.norm(dim=1, keepdim=True)
    if not torch.isfinite(norm).all() or torch.any(norm <= 0):
        raise ValueError("baseline initialized row norm differs")
    return row * (target_norm / norm)


def _initialize_matrix(
    strategy: str,
    *,
    norm_scope: str,
    base_weight: torch.Tensor,
    random_weight: torch.Tensor,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
    metadata: SourceTokenMetadata,
) -> tuple[torch.Tensor, tuple[int, ...], float | None]:
    base_by_piece = {piece: token_id for token_id, piece in enumerate(base_pieces)}
    target_norm = _target_norm(norm_scope, base_weight, metadata)
    output = torch.empty_like(random_weight)
    new_rows: list[int] = []
    lengths = torch.tensor(metadata.decoded_character_lengths, dtype=torch.float32)
    for target_id, (piece, constituents) in enumerate(zip(target_pieces, decompositions)):
        if piece in base_by_piece:
            output[target_id] = base_weight[base_by_piece[piece]]
            continue
        ids = torch.tensor(tuple(constituents), dtype=torch.long)
        source = base_weight.index_select(0, ids)
        if strategy == "random":
            row = random_weight[target_id : target_id + 1]
        elif strategy == "uniform":
            row = source.mean(dim=0, keepdim=True)
        elif strategy == "decoded_character_weighted":
            selected_lengths = lengths.index_select(0, ids)
            weights = selected_lengths / selected_lengths.sum()
            row = (source * weights[:, None]).sum(dim=0, keepdim=True)
        elif strategy == "first":
            row = source[:1]
        else:
            raise ValueError("baseline initialization strategy differs")
        output[target_id] = _rescale(row, target_norm)[0]
        new_rows.append(target_id)
    return output, tuple(new_rows), None if target_norm is None else float(target_norm)


def initialize_target_weights(
    role: str,
    *,
    base_weight: torch.Tensor,
    target_input_random_weight: torch.Tensor,
    target_output_random_weight: torch.Tensor,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
    metadata: SourceTokenMetadata,
) -> tuple[torch.Tensor, torch.Tensor, BaselineInitializationAudit]:
    definition = role_definition(role)
    base = tuple(base_pieces)
    target = tuple(target_pieces)
    decompositions = tuple(tuple(int(value) for value in row) for row in decompositions)
    if (
        len(base) != BASE_VOCABULARY_SIZE
        or len(target) != TARGET_VOCABULARY_SIZE
        or len(set(base)) != len(base)
        or len(set(target)) != len(target)
        or len(decompositions) != TARGET_VOCABULARY_SIZE
        or base_weight.shape[0] != BASE_VOCABULARY_SIZE
        or target_input_random_weight.shape[0] != TARGET_VOCABULARY_SIZE
        or target_output_random_weight.shape != target_input_random_weight.shape
        or base_weight.shape[1] != target_input_random_weight.shape[1]
        or any(not row for row in decompositions)
        or any(b"".join(base[index] for index in row) != piece for row, piece in zip(decompositions, target))
    ):
        raise ValueError("baseline initialization coordinates differ")
    if any(
        value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in (base_weight, target_input_random_weight, target_output_random_weight)
    ):
        raise ValueError("baseline initialization tensor differs")
    input_weight, new_rows, target_norm = _initialize_matrix(
        definition["input_strategy"],
        norm_scope=definition["input_norm_scope"],
        base_weight=base_weight,
        random_weight=target_input_random_weight,
        base_pieces=base,
        target_pieces=target,
        decompositions=decompositions,
        metadata=metadata,
    )
    if definition["tied"]:
        output_weight = input_weight
        output_new_rows = new_rows
    else:
        output_weight, output_new_rows, output_norm = _initialize_matrix(
            definition["output_strategy"],
            norm_scope=definition["output_norm_scope"],
            base_weight=base_weight,
            random_weight=target_output_random_weight,
            base_pieces=base,
            target_pieces=target,
            decompositions=decompositions,
            metadata=metadata,
        )
        if output_norm is not None:
            raise AssertionError("baseline output norm must be absent")
    if new_rows != output_new_rows or len(new_rows) != TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE:
        raise AssertionError("baseline new-row inventory differs")
    base_by_piece = {piece: token_id for token_id, piece in enumerate(base)}
    copied_exact = all(
        torch.equal(input_weight[target_id], base_weight[base_by_piece[piece]])
        and torch.equal(output_weight[target_id], base_weight[base_by_piece[piece]])
        for target_id, piece in enumerate(target)
        if piece in base_by_piece
    )
    new_index = torch.tensor(new_rows, dtype=torch.long)
    metadata_sha = _metadata_hash(
        metadata.decoded_character_lengths, metadata.hangul_token_mask
    )
    audit = BaselineInitializationAudit(
        role=role,
        tied_input_output=definition["tied"],
        training_schedule=definition["training_schedule"],
        input_strategy=definition["input_strategy"],
        input_norm_scope=definition["input_norm_scope"],
        input_norm_statistic="midpoint_median_l2" if target_norm is not None else "none",
        input_target_norm=target_norm,
        output_strategy=definition["output_strategy"],
        output_norm_scope=definition["output_norm_scope"],
        base_vocabulary_size=BASE_VOCABULARY_SIZE,
        target_vocabulary_size=TARGET_VOCABULARY_SIZE,
        shared_token_count=BASE_VOCABULARY_SIZE,
        new_token_count=len(new_rows),
        decomposition_kind="canonical_target_merge_tree_cut_at_source_vocab",
        decomposition_sha256=decomposition_sha256(decompositions),
        maximum_constituent_count=max(len(row) for row in decompositions),
        mean_constituent_count=float(np.mean([len(row) for row in decompositions])),
        source_metadata_sha256=metadata_sha,
        initialized_input_weight_sha256=tensor_sha256(input_weight),
        initialized_output_weight_sha256=tensor_sha256(output_weight),
        initialized_input_new_mean_row_l2=float(
            input_weight.index_select(0, new_index).norm(dim=1).mean()
        ),
        initialized_output_new_mean_row_l2=float(
            output_weight.index_select(0, new_index).norm(dim=1).mean()
        ),
        copied_rows_exact=copied_exact,
        target_pieces_reconstruct_exactly=True,
        baseline_interpretation=definition["baseline_interpretation"],
    )
    return input_weight, output_weight, audit


def build_transferred_model(
    role: str,
    *,
    base_state: Mapping[str, torch.Tensor],
    base_tokenizer: Any,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
) -> tuple[Any, BaselineInitializationAudit, SourceTokenMetadata]:
    if set(_HEAD_KEYS) - set(base_state):
        raise ValueError("baseline source checkpoint differs")
    model = build_target_graph(role)
    target_state = model.state_dict()
    if set(base_state) != set(target_state):
        raise ValueError("baseline model state key set differs")
    copied: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        if name in _HEAD_KEYS:
            copied[name] = target_value.detach().cpu().contiguous()
        else:
            source = base_state[name]
            if source.shape != target_value.shape or source.dtype != target_value.dtype:
                raise ValueError("baseline body state differs")
            copied[name] = source.detach().cpu().contiguous()
    base_weight = base_state["model.embed_tokens.weight"].detach().cpu().contiguous()
    if not torch.equal(base_weight, base_state["lm_head.weight"].detach().cpu()):
        raise ValueError("baseline source tied head differs")
    metadata = source_token_metadata(base_tokenizer)
    input_weight, output_weight, audit = initialize_target_weights(
        role,
        base_weight=base_weight,
        target_input_random_weight=target_state["model.embed_tokens.weight"].detach().cpu().contiguous(),
        target_output_random_weight=target_state["lm_head.weight"].detach().cpu().contiguous(),
        base_pieces=base_pieces,
        target_pieces=target_pieces,
        decompositions=decompositions,
        metadata=metadata,
    )
    copied["model.embed_tokens.weight"] = input_weight
    copied["lm_head.weight"] = output_weight
    model.load_state_dict(copied, strict=True)
    if state_mapping_sha256(model.state_dict()) == state_mapping_sha256(base_state):
        raise AssertionError("baseline target state unexpectedly equals source")
    return model, audit, metadata


def mask_old_row_gradient(weight: torch.Tensor, *, base_size: int = BASE_VOCABULARY_SIZE) -> None:
    if weight.grad is None or weight.grad.shape != weight.shape or not 0 < base_size < len(weight):
        raise ValueError("baseline stage-one gradient differs")
    weight.grad[:base_size].zero_()


def restore_old_rows(
    weight: torch.Tensor,
    copied_rows: torch.Tensor,
    *,
    base_size: int = BASE_VOCABULARY_SIZE,
) -> None:
    if copied_rows.shape != weight[:base_size].shape or copied_rows.dtype != weight.dtype:
        raise ValueError("baseline copied-row restore differs")
    with torch.no_grad():
        weight[:base_size].copy_(copied_rows)
    if not torch.equal(weight[:base_size].detach(), copied_rows.detach()):
        raise RuntimeError("baseline copied rows drifted")


def baseline_closure_decision(
    final_bpb_by_role: Mapping[str, float], *, anchor_bpb: float
) -> dict[str, Any]:
    if (
        set(final_bpb_by_role) != set(BASELINE_ROLES)
        or not math.isfinite(anchor_bpb)
        or anchor_bpb <= 0
        or any(not math.isfinite(float(value)) or float(value) <= 0 for value in final_bpb_by_role.values())
    ):
        raise ValueError("baseline decision inputs differ")
    diagnostics: dict[str, dict[str, Any]] = {}
    qualified: list[str] = []
    for role in COMPOSED_BASELINE_ROLES:
        control = RANDOM_CONTROL_BY_ROLE[role]
        value = float(final_bpb_by_role[role])
        advantage = float(final_bpb_by_role[control]) - value
        anchor_gap = value - anchor_bpb
        row = {
            "architecture": "tied" if role.startswith("tied_") else "untied",
            "training_schedule": role_definition(role)["training_schedule"],
            "random_control": control,
            "final_bpb": value,
            "random_control_final_bpb": float(final_bpb_by_role[control]),
            "composed_advantage_bpb": advantage,
            "anchor_gap_bpb": anchor_gap,
            "initialization_advantage_pass": advantage >= MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
            "recovery_progress_pass": anchor_gap <= MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
        }
        row["joint_pass"] = row["initialization_advantage_pass"] and row["recovery_progress_pass"]
        diagnostics[role] = row
        if row["joint_pass"]:
            qualified.append(role)

    def best(values: Sequence[str]) -> str | None:
        return (
            min(values, key=lambda role: (float(final_bpb_by_role[role]), BASELINE_ROLES.index(role)))
            if values
            else None
        )

    best_tied = best([role for role in qualified if role.startswith("tied_")])
    best_untied = best([role for role in qualified if role.startswith("untied_")])
    return {
        "status": "strong_generic_baseline_pass" if qualified else "strong_generic_baseline_stopped",
        "korean_stage_authorized": bool(qualified),
        "qualified_roles": qualified,
        "best_tied_pareto_role": best_tied,
        "best_untied_pareto_role": best_untied,
        "best_tied_bpb": None if best_tied is None else float(final_bpb_by_role[best_tied]),
        "best_untied_bpb": None if best_untied is None else float(final_bpb_by_role[best_untied]),
        "dense_2k_anchor_bpb": anchor_bpb,
        "minimum_initialization_advantage_bpb": MINIMUM_INITIALIZATION_ADVANTAGE_BPB,
        "maximum_anchor_gap_for_full_cpt_bpb": MAXIMUM_ANCHOR_GAP_FOR_FULL_CPT_BPB,
        "candidate_diagnostics": diagnostics,
        "selection_uses_step_zero_or_step_fifty": False,
        "no_korean_specific_initializer_evaluated": True,
    }
