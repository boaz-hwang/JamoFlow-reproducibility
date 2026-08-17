"""Core contracts for the fresh-v2 16K vocabulary-expansion fail-fast."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from bpe_quality_frontier_core import bpb, document_bootstrap_upper
from compositional_head_core import MODEL_SEED, ROLE_SPECS, build_model
from vocabulary_transfer_probe_core import state_mapping_sha256, tensor_sha256


ROLES = (
    "dense2k_joint_v2",
    "dense8k_update_geometry_v2",
    "dense16k_standard_joint",
    "dense16k_inplace_two_stage",
    "dense16k_update_geometry",
)
ANCHOR_ROLES = ("dense2k_joint_v2", "dense8k_update_geometry_v2")
SIXTEEN_K_CONTROL_ROLES = (
    "dense16k_standard_joint",
    "dense16k_inplace_two_stage",
)
CANDIDATE_ROLE = "dense16k_update_geometry"

BASE_VOCABULARY_SIZE = 2_048
REPLICATION_VOCABULARY_SIZE = 8_192
TARGET_VOCABULARY_SIZE = 16_000
VOCABULARY_SIZES = (
    BASE_VOCABULARY_SIZE,
    REPLICATION_VOCABULARY_SIZE,
    TARGET_VOCABULARY_SIZE,
)
SEQUENCE_LENGTH = 512
EFFECTIVE_BATCH_SIZE = 32
TRAIN_MICROBATCH_BY_VOCABULARY = {2_048: 32, 8_192: 8, 16_000: 4}
EVALUATION_BATCH_BY_VOCABULARY = {2_048: 64, 8_192: 16, 16_000: 8}

BODY_LEARNING_RATE = 3e-5
HEAD_PEAK_LEARNING_RATE = 3e-4
HEAD_MINIMUM_LEARNING_RATE = 3e-5
WARMUP_RAW_FRACTION = 0.05
INPLACE_STAGE_ONE_RAW_FRACTION = 0.60
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0

# Fixed before this experiment by the model-loss-free first-AdamW-update audit
# and then used without tuning in the fresh-v1 8K experiment.
INPUT_UPDATE_MULTIPLIER = 1.485414522979104
OUTPUT_UPDATE_MULTIPLIER = 2.170601418278963

QUALITY_NONINFERIORITY_MARGIN_BPB = 0.010
METHOD_MINIMUM_ADVANTAGE_BPB = 0.002
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_841
UNTIED_OUTPUT_SEED_DOMAIN = 80_000

_HEAD_KEYS = frozenset(("model.embed_tokens.weight", "lm_head.weight"))


@dataclass(frozen=True, slots=True)
class TransferInitializationAudit:
    target_vocabulary_size: int
    tied_input_output: bool
    input_strategy: str
    output_strategy: str
    input_norm_calibrated: bool
    output_norm_calibrated: bool
    base_vocabulary_size: int
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


def role_definition(role: str) -> dict[str, Any]:
    definitions = {
        "dense2k_joint_v2": {
            "vocabulary_size": BASE_VOCABULARY_SIZE,
            "initialization": "exact_source_checkpoint",
            "schedule": "all_parameter_joint_body_constant_head_raw_progress_cosine",
            "post_adamw_new_row_scaling": None,
        },
        "dense8k_update_geometry_v2": {
            "vocabulary_size": REPLICATION_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "all_parameter_joint_body_constant_head_raw_progress_cosine",
            "post_adamw_new_row_scaling": {
                "input_multiplier": INPUT_UPDATE_MULTIPLIER,
                "output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
                "source": "fresh_v1_presealed_update_geometry",
                "validation_metric_used": False,
            },
        },
        "dense16k_standard_joint": {
            "vocabulary_size": TARGET_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "all_parameter_joint_body_constant_head_raw_progress_cosine",
            "post_adamw_new_row_scaling": None,
        },
        "dense16k_inplace_two_stage": {
            "vocabulary_size": TARGET_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "new_rows_only_60pct_then_all_40pct",
            "post_adamw_new_row_scaling": None,
        },
        "dense16k_update_geometry": {
            "vocabulary_size": TARGET_VOCABULARY_SIZE,
            "initialization": "untied_uniform_input_byte_weighted_output",
            "schedule": "all_parameter_joint_body_constant_head_raw_progress_cosine",
            "post_adamw_new_row_scaling": {
                "input_multiplier": INPUT_UPDATE_MULTIPLIER,
                "output_multiplier": OUTPUT_UPDATE_MULTIPLIER,
                "source": "fresh_v1_presealed_update_geometry_no_retuning",
                "validation_metric_used": False,
            },
        },
    }
    if role not in definitions:
        raise ValueError("fresh-16k role differs")
    return dict(definitions[role])


def is_staged_role(role: str) -> bool:
    if role not in ROLES:
        raise ValueError("fresh-16k staged role differs")
    return role == "dense16k_inplace_two_stage"


def is_geometry_role(role: str) -> bool:
    if role not in ROLES:
        raise ValueError("fresh-16k geometry role differs")
    return role in ("dense8k_update_geometry_v2", CANDIDATE_ROLE)


def expected_parameter_count(vocabulary_size: int) -> int:
    if vocabulary_size == BASE_VOCABULARY_SIZE:
        return ROLE_SPECS["dense_v2048"].expected_parameters
    if vocabulary_size not in (
        REPLICATION_VOCABULARY_SIZE,
        TARGET_VOCABULARY_SIZE,
    ):
        raise ValueError("fresh-16k parameter vocabulary differs")
    spec = ROLE_SPECS[f"dense_v{vocabulary_size}"]
    return spec.expected_parameters + vocabulary_size * spec.hidden_size


def _require_piece_table(
    pieces: Sequence[bytes], expected_size: int
) -> tuple[bytes, ...]:
    output = tuple(pieces)
    if (
        len(output) != expected_size
        or any(not isinstance(piece, bytes) or not piece for piece in output)
        or len(set(output)) != len(output)
    ):
        raise ValueError("fresh-16k piece table differs")
    return output


def build_canonical_decomposition_table(
    base_tokenizer: Any,
    target_tokenizer: Any,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
) -> tuple[tuple[int, ...], ...]:
    """Cut a canonical target BPE merge tree at the exact 2K frontier."""

    base = _require_piece_table(base_pieces, BASE_VOCABULARY_SIZE)
    target_values = tuple(target_pieces)
    target = _require_piece_table(target_values, len(target_values))
    target_size = len(target)
    if target_size not in (REPLICATION_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE):
        raise ValueError("fresh-16k target vocabulary differs")
    try:
        base_model = json.loads(base_tokenizer.to_str())["model"]
        target_model = json.loads(target_tokenizer.to_str())["model"]
        base_vocab = base_model["vocab"]
        target_vocab = target_model["vocab"]
        base_merges = tuple(tuple(row) for row in base_model["merges"])
        target_merges = tuple(tuple(row) for row in target_model["merges"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("fresh-16k tokenizer merge identity differs") from error
    if (
        base_model.get("type") != "BPE"
        or target_model.get("type") != "BPE"
        or len(base_vocab) != BASE_VOCABULARY_SIZE
        or len(target_vocab) != target_size
        or len(base_merges) != BASE_VOCABULARY_SIZE - 256
        or len(target_merges) != target_size - 256
        or base_merges != target_merges[: len(base_merges)]
        or any(
            target_vocab.get(symbol) != token_id
            for symbol, token_id in base_vocab.items()
        )
    ):
        raise ValueError("fresh-16k target is not a canonical BPE extension")
    target_symbols: list[str | None] = [None] * target_size
    for symbol, token_id in target_vocab.items():
        if (
            not isinstance(symbol, str)
            or not isinstance(token_id, int)
            or not 0 <= token_id < target_size
            or target_symbols[token_id] is not None
        ):
            raise ValueError("fresh-16k target token IDs differ")
        target_symbols[token_id] = symbol
    creation: dict[str, tuple[str, str]] = {}
    for row in target_merges:
        if len(row) != 2 or not all(
            isinstance(value, str) and value for value in row
        ):
            raise ValueError("fresh-16k merge row differs")
        merged = row[0] + row[1]
        if merged in creation and creation[merged] != row:
            raise ValueError("fresh-16k merge genealogy is ambiguous")
        creation[merged] = row
    cache: dict[str, tuple[int, ...]] = {}
    visiting: set[str] = set()

    def expand(symbol: str) -> tuple[int, ...]:
        if symbol in cache:
            return cache[symbol]
        if symbol in visiting:
            raise ValueError("fresh-16k merge genealogy has a cycle")
        visiting.add(symbol)
        if symbol in base_vocab:
            output = (int(base_vocab[symbol]),)
        elif symbol in creation:
            left, right = creation[symbol]
            output = (*expand(left), *expand(right))
        else:
            raise ValueError("fresh-16k target token has no 2K genealogy")
        visiting.remove(symbol)
        cache[symbol] = output
        return output

    if any(symbol is None for symbol in target_symbols):
        raise ValueError("fresh-16k target token IDs are incomplete")
    output = tuple(expand(str(symbol)) for symbol in target_symbols)
    if any(
        b"".join(base[token_id] for token_id in row) != piece
        for row, piece in zip(output, target)
    ):
        raise ValueError("fresh-16k decomposition is not byte exact")
    return output


def decomposition_sha256(
    decompositions: Sequence[Sequence[int]], target_size: int
) -> str:
    rows = tuple(tuple(int(value) for value in row) for row in decompositions)
    if (
        target_size not in (REPLICATION_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
        or len(rows) != target_size
        or any(not row for row in rows)
        or any(
            value < 0 or value >= BASE_VOCABULARY_SIZE
            for row in rows
            for value in row
        )
    ):
        raise ValueError("fresh-16k decomposition rows differ")
    digest = hashlib.sha256(b"JamoFlow/vocabulary-transfer-decomposition/v2\0")
    digest.update(BASE_VOCABULARY_SIZE.to_bytes(8, "big"))
    digest.update(target_size.to_bytes(8, "big"))
    for row in rows:
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(np.asarray(row, dtype=">u4").tobytes(order="C"))
    return digest.hexdigest()


def _rescale_rows(values: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    norms = values.norm(dim=1, keepdim=True)
    if not torch.isfinite(norms).all() or torch.any(norms <= 0):
        raise ValueError("fresh-16k row norm differs")
    return values * (target_norm / norms)


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
        if strategy == "uniform":
            row = base_weight.index_select(0, ids).mean(dim=0, keepdim=True)
        elif strategy == "byte_weighted":
            lengths = torch.tensor(
                [len(base[token_id]) for token_id in constituent_ids],
                dtype=torch.float32,
            )
            weights = lengths / lengths.sum()
            row = (base_weight.index_select(0, ids) * weights[:, None]).sum(
                dim=0, keepdim=True
            )
        else:
            raise ValueError("fresh-16k initialization strategy differs")
        if norm_calibrated:
            row = _rescale_rows(row, target_norm)
        output[target_id] = row[0]
        new_rows.append(target_id)
    return output, tuple(new_rows)


def build_transferred_model(
    target_size: int,
    *,
    base_state: Mapping[str, torch.Tensor],
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
) -> tuple[Any, TransferInitializationAudit]:
    """Build the fixed untied uniform-input/byte-weighted-output graph."""

    if target_size not in (REPLICATION_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE):
        raise ValueError("fresh-16k transfer size differs")
    if set(_HEAD_KEYS) - set(base_state):
        raise ValueError("fresh-16k base checkpoint differs")
    base = _require_piece_table(base_pieces, BASE_VOCABULARY_SIZE)
    target = _require_piece_table(target_pieces, target_size)
    model = build_model(f"dense_v{target_size}", seed=MODEL_SEED)
    generator = torch.Generator(device="cpu").manual_seed(
        MODEL_SEED + UNTIED_OUTPUT_SEED_DOMAIN
    )
    random_output = torch.empty(
        target_size, int(model.config.hidden_size), dtype=torch.float32
    )
    random_output.normal_(
        mean=0.0,
        std=float(model.config.initializer_range),
        generator=generator,
    )
    output_layer = nn.Linear(
        int(model.config.hidden_size), target_size, bias=False
    )
    output_layer.weight = nn.Parameter(random_output)
    model.lm_head = output_layer
    model.config.tie_word_embeddings = False
    target_state = model.state_dict()
    if set(base_state) != set(target_state):
        raise ValueError("fresh-16k model state key set differs")
    copied: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        if name in _HEAD_KEYS:
            copied[name] = target_value
            continue
        source = base_state[name]
        if source.shape != target_value.shape or source.dtype != target_value.dtype:
            raise ValueError("fresh-16k body state differs")
        copied[name] = source.detach().cpu().contiguous()
    base_weight = base_state["model.embed_tokens.weight"].detach().cpu().contiguous()
    if not torch.equal(base_weight, base_state["lm_head.weight"].detach().cpu()):
        raise ValueError("fresh-16k source tied head differs")
    rows = tuple(tuple(int(value) for value in row) for row in decompositions)
    if (
        len(rows) != target_size
        or any(not row for row in rows)
        or any(
            b"".join(base[token_id] for token_id in row) != piece
            for row, piece in zip(rows, target)
        )
    ):
        raise ValueError("fresh-16k supplied decomposition differs")
    input_weight, input_new_rows = _initialize_weight(
        "uniform",
        norm_calibrated=True,
        base_weight=base_weight,
        random_weight=target_state["model.embed_tokens.weight"],
        base=base,
        target=target,
        decompositions=rows,
    )
    output_weight, output_new_rows = _initialize_weight(
        "byte_weighted",
        norm_calibrated=False,
        base_weight=base_weight,
        random_weight=target_state["lm_head.weight"],
        base=base,
        target=target,
        decompositions=rows,
    )
    if (
        len(input_new_rows) != target_size - BASE_VOCABULARY_SIZE
        or output_new_rows != input_new_rows
    ):
        raise AssertionError("fresh-16k shared token inventory differs")
    copied["model.embed_tokens.weight"] = input_weight
    copied["lm_head.weight"] = output_weight
    model.load_state_dict(copied, strict=True)
    if (
        model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
        or sum(parameter.numel() for parameter in model.parameters())
        != expected_parameter_count(target_size)
    ):
        raise AssertionError("fresh-16k target graph differs")
    new_index = torch.tensor(input_new_rows, dtype=torch.long)
    target_norm = base_weight.norm(dim=1).mean()
    audit = TransferInitializationAudit(
        target_vocabulary_size=target_size,
        tied_input_output=False,
        input_strategy="uniform",
        output_strategy="byte_weighted",
        input_norm_calibrated=True,
        output_norm_calibrated=False,
        base_vocabulary_size=BASE_VOCABULARY_SIZE,
        shared_token_count=BASE_VOCABULARY_SIZE,
        new_token_count=len(input_new_rows),
        maximum_constituent_count=max(len(row) for row in rows),
        mean_constituent_count=float(np.mean([len(row) for row in rows])),
        decomposition_kind="canonical_target_merge_tree_cut_at_source_vocab",
        decomposition_sha256=decomposition_sha256(rows, target_size),
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
    if state_mapping_sha256(model.state_dict()) == state_mapping_sha256(base_state):
        raise AssertionError("fresh-16k transferred state unexpectedly equals its source")
    return model, audit


def total_optimizer_steps(sequence_count: int) -> int:
    if sequence_count <= 0:
        raise ValueError("fresh-16k sequence count differs")
    return math.ceil(sequence_count / EFFECTIVE_BATCH_SIZE)


def batch_raw_target_bytes(raw_target_bytes: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_target_bytes)
    if (
        values.ndim != 1
        or len(values) <= 0
        or not np.issubdtype(values.dtype, np.integer)
        or np.any(values <= 0)
    ):
        raise ValueError("fresh-16k raw-target array differs")
    return np.asarray(
        [
            int(values[start : start + EFFECTIVE_BATCH_SIZE].sum())
            for start in range(0, len(values), EFFECTIVE_BATCH_SIZE)
        ],
        dtype=np.int64,
    )


def inplace_stage_contract(raw_target_bytes: np.ndarray) -> dict[str, Any]:
    batches = batch_raw_target_bytes(raw_target_bytes)
    cumulative = np.cumsum(batches, dtype=np.int64)
    total = int(cumulative[-1])
    target = total * INPLACE_STAGE_ONE_RAW_FRACTION
    stage_one_steps = int(np.searchsorted(cumulative, target, side="left")) + 1
    stage_one_bytes = int(cumulative[stage_one_steps - 1])
    if not 0 < stage_one_steps < len(batches) or not 0 < stage_one_bytes < total:
        raise ValueError("fresh-16k two-stage boundary differs")
    return {
        "boundary_rule": "first_complete_effective_batch_reaching_60pct_raw_target_bytes",
        "requested_stage_one_raw_fraction": INPLACE_STAGE_ONE_RAW_FRACTION,
        "stage_one_optimizer_steps": stage_one_steps,
        "stage_one_raw_target_bytes": stage_one_bytes,
        "stage_one_realized_raw_fraction": stage_one_bytes / total,
        "stage_two_optimizer_steps": len(batches) - stage_one_steps,
        "stage_two_raw_target_bytes": total - stage_one_bytes,
        "total_optimizer_steps": len(batches),
        "total_raw_target_bytes": total,
    }


def _cosine_from_progress(progress: float) -> float:
    if not 0.0 < progress <= 1.0:
        raise ValueError("fresh-16k learning-rate progress differs")
    if progress <= WARMUP_RAW_FRACTION:
        return HEAD_PEAK_LEARNING_RATE * progress / WARMUP_RAW_FRACTION
    decay = (progress - WARMUP_RAW_FRACTION) / (1.0 - WARMUP_RAW_FRACTION)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay))
    return HEAD_MINIMUM_LEARNING_RATE + (
        HEAD_PEAK_LEARNING_RATE - HEAD_MINIMUM_LEARNING_RATE
    ) * cosine


def head_learning_rate(
    role: str,
    *,
    cumulative_raw_target_bytes: int,
    total_raw_target_bytes: int,
    stage_one_raw_target_bytes: int | None,
) -> float:
    if (
        role not in ROLES
        or total_raw_target_bytes <= 0
        or not 0 < cumulative_raw_target_bytes <= total_raw_target_bytes
    ):
        raise ValueError("fresh-16k learning-rate coordinates differ")
    if not is_staged_role(role):
        if stage_one_raw_target_bytes is not None:
            raise ValueError("fresh-16k non-staged role has a boundary")
        return _cosine_from_progress(
            cumulative_raw_target_bytes / total_raw_target_bytes
        )
    if (
        stage_one_raw_target_bytes is None
        or not 0 < stage_one_raw_target_bytes < total_raw_target_bytes
    ):
        raise ValueError("fresh-16k staged boundary differs")
    if cumulative_raw_target_bytes <= stage_one_raw_target_bytes:
        progress = cumulative_raw_target_bytes / stage_one_raw_target_bytes
        return HEAD_PEAK_LEARNING_RATE * min(1.0, progress / WARMUP_RAW_FRACTION)
    progress = (cumulative_raw_target_bytes - stage_one_raw_target_bytes) / (
        total_raw_target_bytes - stage_one_raw_target_bytes
    )
    return _cosine_from_progress(progress)


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    raw_bytes: np.ndarray,
) -> dict[str, float]:
    point, lower, upper = document_bootstrap_upper(
        candidate,
        reference,
        raw_bytes,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    return {"point_bpb": point, "lower_95_bpb": lower, "upper_95_bpb": upper}


def quality_decision(
    document_nll_by_role: Mapping[str, np.ndarray],
    document_raw_bytes: np.ndarray,
) -> dict[str, Any]:
    if set(document_nll_by_role) != set(ROLES):
        raise ValueError("fresh-16k decision role set differs")
    raw = np.asarray(document_raw_bytes)
    arrays = {role: np.asarray(document_nll_by_role[role]) for role in ROLES}
    if (
        raw.ndim != 1
        or len(raw) < 2
        or not np.issubdtype(raw.dtype, np.integer)
        or np.any(raw <= 0)
        or any(
            values.shape != raw.shape
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            for values in arrays.values()
        )
    ):
        raise ValueError("fresh-16k decision arrays differ")
    role_bpb = {role: bpb(arrays[role], raw) for role in ROLES}
    anchor_comparisons: dict[str, dict[str, Any]] = {}
    for anchor in ANCHOR_ROLES:
        row: dict[str, Any] = _comparison(arrays[CANDIDATE_ROLE], arrays[anchor], raw)
        row["margin_bpb"] = QUALITY_NONINFERIORITY_MARGIN_BPB
        row["pass"] = (
            row["point_bpb"] <= QUALITY_NONINFERIORITY_MARGIN_BPB
            and row["upper_95_bpb"] <= QUALITY_NONINFERIORITY_MARGIN_BPB
        )
        anchor_comparisons[anchor] = row
    control_comparisons: dict[str, dict[str, Any]] = {}
    for control in SIXTEEN_K_CONTROL_ROLES:
        row = _comparison(arrays[CANDIDATE_ROLE], arrays[control], raw)
        row["minimum_advantage_bpb"] = METHOD_MINIMUM_ADVANTAGE_BPB
        row["pass"] = (
            row["point_bpb"] <= -METHOD_MINIMUM_ADVANTAGE_BPB
            and row["upper_95_bpb"] <= 0.0
        )
        control_comparisons[control] = row
    replication = _comparison(
        arrays["dense8k_update_geometry_v2"],
        arrays["dense2k_joint_v2"],
        raw,
    )
    replication["margin_bpb"] = QUALITY_NONINFERIORITY_MARGIN_BPB
    replication["pass"] = (
        replication["point_bpb"] <= QUALITY_NONINFERIORITY_MARGIN_BPB
        and replication["upper_95_bpb"] <= QUALITY_NONINFERIORITY_MARGIN_BPB
    )
    anchor_pass = all(row["pass"] for row in anchor_comparisons.values())
    control_pass = all(row["pass"] for row in control_comparisons.values())
    actual_authorized = anchor_pass and control_pass
    if not anchor_pass:
        status = "fail_16k_anchor_noninferiority"
    elif not control_pass:
        status = "fail_16k_method_controls"
    else:
        status = "pass_16k_quality_for_actual_preflight"
    stronger_anchor = min(ANCHOR_ROLES, key=lambda role: (role_bpb[role], role))
    return {
        "status": status,
        "document_bpb_by_role": role_bpb,
        "candidate_role": CANDIDATE_ROLE,
        "candidate_noninferiority_vs_each_anchor": anchor_comparisons,
        "candidate_beats_each_16k_control": control_comparisons,
        "stronger_observed_anchor": stronger_anchor,
        "all_anchor_noninferiority_pass": anchor_pass,
        "all_16k_method_controls_pass": control_pass,
        "eightk_cross_data_replication": replication,
        "cross_vocabulary_geometry_supported": actual_authorized
        and replication["pass"],
        "actual_inference_preflight_authorized": actual_authorized,
        "authorized_actual_role": CANDIDATE_ROLE if actual_authorized else None,
        "multiseed_authorized": False,
        "publication_claim_authorized": False,
    }
