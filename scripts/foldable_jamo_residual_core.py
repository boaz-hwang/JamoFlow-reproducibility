"""Training-only orthographic residuals that fold into ordinary dense BPE rows.

The deployed model is deliberately unmodified.  During continued pretraining,
new vocabulary rows receive an additive shared residual derived from a fixed
13-slot surface assignment.  The residual tables start at exact zero and are
materialized into the dense rows after training.  Generic, shuffled-Jamo, and
true-Jamo roles use the same table shapes and the same number of lookups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal
import weakref

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe_quality_frontier_core import document_bootstrap_upper
from compositional_token_head import (
    CODEBOOK_SIZE,
    SURFACE_AUXILIARY_SLOTS,
    array_sha256,
    build_token_code_assignment,
)
from vocabulary_transfer_baseline_core import (
    build_target_graph,
    build_transferred_model,
    state_mapping_sha256,
)
from vocabulary_transfer_probe_core import (
    BASE_VOCABULARY_SIZE,
    MODEL_SEED,
    TARGET_VOCABULARY_SIZE,
)


AssignmentKind = Literal["generic_surface", "shuffled_jamo", "jamo"]
ARCHITECTURES = ("untied", "tied")
ASSIGNMENT_KINDS: tuple[AssignmentKind, ...] = (
    "generic_surface",
    "shuffled_jamo",
    "jamo",
)
RESIDUAL_ROLES = tuple(
    f"{architecture}_{assignment}"
    for architecture in ARCHITECTURES
    for assignment in ASSIGNMENT_KINDS
)
BASELINE_ROLE_BY_ARCHITECTURE = {
    "untied": "untied_eeve_uniform_in_first_out",
    "tied": "tied_uniform_no_norm_all",
}
RESIDUAL_SLOT_COUNT = 13
RESIDUAL_SCALE = RESIDUAL_SLOT_COUNT**-0.5
RESIDUAL_SHUFFLE_SEED = 20_260_829
PROBE_STEPS = (0, 32, 128, 512)
FINAL_PROBE_STEP = PROBE_STEPS[-1]
MINIMUM_JAMO_ADVANTAGE_BPB = 0.002
MAXIMUM_ANCHOR_GAP_BPB = 0.050
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 20_260_830


@dataclass(frozen=True, slots=True)
class ResidualAssignmentAudit:
    assignment_kind: str
    assignment_sha256: str
    vocabulary_size: int
    new_row_count: int
    slot_count: int
    codebook_size: int
    residual_scale: float
    exposure_counts_sha256: str
    exposure_stratum_definition: str
    shuffle_seed: int | None
    stratum_count: int | None
    singleton_stratum_count: int | None
    median_stratum_size: float | None
    maximum_stratum_size: int | None
    changed_new_row_count_vs_true_jamo: int
    changed_new_row_fraction_vs_true_jamo: float
    changed_new_token_exposure_fraction_vs_true_jamo: float
    slot_unique_counts: tuple[int, ...]
    slot_entropy_bits: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["slot_unique_counts"] = list(self.slot_unique_counts)
        value["slot_entropy_bits"] = list(self.slot_entropy_bits)
        return value


def role_definition(role: str) -> dict[str, Any]:
    if role not in RESIDUAL_ROLES:
        raise ValueError("foldable residual role differs")
    architecture, assignment = role.split("_", 1)
    return {
        "architecture": architecture,
        "assignment_kind": assignment,
        "base_initializer_role": BASELINE_ROLE_BY_ARCHITECTURE[architecture],
        "deployed_graph": "ordinary_dense_bpe_8192",
        "residual_applies_to": "new_target_rows_only",
        "residual_initialization": "exact_zero",
        "training_schedule": "all_parameters_512_steps",
    }


def exposure_counts_sha256(exposure_counts: np.ndarray) -> str:
    values = np.asarray(exposure_counts)
    if (
        values.dtype.kind not in "iu"
        or values.shape != (TARGET_VOCABULARY_SIZE,)
        or np.any(values < 0)
    ):
        raise ValueError("foldable residual exposure counts differ")
    digest = hashlib.sha256(b"JamoFlow/foldable-residual-exposure/v1\0")
    digest.update(values.astype(">u8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _shuffle_true_jamo(
    assignment: np.ndarray,
    token_bytes: Sequence[bytes],
    exposure_counts: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    output = assignment.copy()
    pieces = tuple(token_bytes)
    if len(pieces) != TARGET_VOCABULARY_SIZE:
        raise ValueError("foldable residual token table differs")
    lengths = np.asarray([len(piece) for piece in pieces], dtype=np.int64)
    counts = np.asarray(exposure_counts, dtype=np.int64)
    new_rows = np.arange(BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
    permutation = np.arange(TARGET_VOCABULARY_SIZE)
    rng = np.random.default_rng(RESIDUAL_SHUFFLE_SEED)
    stratum_sizes: list[int] = []
    for length in np.unique(lengths[new_rows]):
        length_rows = new_rows[lengths[new_rows] == length]
        for exposure_count in np.unique(counts[length_rows]):
            rows = length_rows[counts[length_rows] == exposure_count]
            stratum_sizes.append(len(rows))
            if len(rows) > 1:
                ordered = rows[rng.permutation(len(rows))]
                permutation[ordered] = np.roll(ordered, -1)
    slots = np.asarray(SURFACE_AUXILIARY_SLOTS, dtype=np.int64)
    output[new_rows[:, None], slots] = assignment[
        permutation[new_rows, None], slots
    ]
    if not stratum_sizes or sum(stratum_sizes) != len(new_rows):
        raise AssertionError("foldable residual shuffle strata differ")
    return output, tuple(stratum_sizes)


def build_residual_assignment(
    token_bytes: Sequence[bytes],
    exposure_counts: np.ndarray,
    *,
    kind: AssignmentKind,
) -> np.ndarray:
    pieces = tuple(token_bytes)
    exposure_counts_sha256(exposure_counts)
    if len(pieces) != TARGET_VOCABULARY_SIZE:
        raise ValueError("foldable residual vocabulary size differs")
    if kind == "generic_surface":
        output = build_token_code_assignment(pieces, kind="generic_unicode")[
            :, :RESIDUAL_SLOT_COUNT
        ].copy()
    elif kind in ("jamo", "shuffled_jamo"):
        output = build_token_code_assignment(pieces, kind="hangul")[
            :, :RESIDUAL_SLOT_COUNT
        ].copy()
        if kind == "shuffled_jamo":
            output, _ = _shuffle_true_jamo(output, pieces, exposure_counts)
    else:
        raise ValueError("foldable residual assignment kind differs")
    if (
        output.dtype != np.int64
        or output.shape != (TARGET_VOCABULARY_SIZE, RESIDUAL_SLOT_COUNT)
        or np.any(output < 0)
        or np.any(output >= CODEBOOK_SIZE)
    ):
        raise AssertionError("foldable residual assignment differs")
    return output


def audit_residual_assignment(
    token_bytes: Sequence[bytes],
    exposure_counts: np.ndarray,
    *,
    kind: AssignmentKind,
) -> ResidualAssignmentAudit:
    pieces = tuple(token_bytes)
    assignment = build_residual_assignment(pieces, exposure_counts, kind=kind)
    true = build_residual_assignment(pieces, exposure_counts, kind="jamo")
    new = slice(BASE_VOCABULARY_SIZE, TARGET_VOCABULARY_SIZE)
    changed = np.any(assignment[new] != true[new], axis=1)
    new_exposures = np.asarray(exposure_counts, dtype=np.int64)[new]
    total_exposure = int(new_exposures.sum())
    unique_counts: list[int] = []
    entropies: list[float] = []
    for slot in range(RESIDUAL_SLOT_COUNT):
        counts = np.bincount(assignment[new, slot], minlength=CODEBOOK_SIZE)
        probabilities = counts[counts > 0].astype(np.float64) / int(counts.sum())
        unique_counts.append(int(np.count_nonzero(counts)))
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
    strata: tuple[int, ...] | None = None
    if kind == "shuffled_jamo":
        _, strata = _shuffle_true_jamo(true, pieces, exposure_counts)
    return ResidualAssignmentAudit(
        assignment_kind=kind,
        assignment_sha256=array_sha256(assignment),
        vocabulary_size=TARGET_VOCABULARY_SIZE,
        new_row_count=TARGET_VOCABULARY_SIZE - BASE_VOCABULARY_SIZE,
        slot_count=RESIDUAL_SLOT_COUNT,
        codebook_size=CODEBOOK_SIZE,
        residual_scale=RESIDUAL_SCALE,
        exposure_counts_sha256=exposure_counts_sha256(exposure_counts),
        exposure_stratum_definition=(
            "exact(raw_token_byte_length,actual_scheduled_token_count); "
            "random-order cyclic derangement for every nonsingleton stratum"
        ),
        shuffle_seed=RESIDUAL_SHUFFLE_SEED if strata is not None else None,
        stratum_count=None if strata is None else len(strata),
        singleton_stratum_count=None if strata is None else sum(size == 1 for size in strata),
        median_stratum_size=None if strata is None else float(np.median(strata)),
        maximum_stratum_size=None if strata is None else max(strata),
        changed_new_row_count_vs_true_jamo=int(changed.sum()),
        changed_new_row_fraction_vs_true_jamo=float(changed.mean()),
        changed_new_token_exposure_fraction_vs_true_jamo=(
            0.0
            if total_exposure == 0
            else float(new_exposures[changed].sum() / total_exposure)
        ),
        slot_unique_counts=tuple(unique_counts),
        slot_entropy_bits=tuple(entropies),
    )


class FoldableVocabularyResidual(nn.Module):
    def __init__(
        self,
        input_weight: torch.Tensor,
        output_weight: torch.Tensor,
        assignment: np.ndarray,
        *,
        tied: bool,
    ) -> None:
        super().__init__()
        if (
            input_weight.dtype != torch.float32
            or output_weight.dtype != torch.float32
            or input_weight.shape != output_weight.shape
            or input_weight.shape[0] != TARGET_VOCABULARY_SIZE
            or assignment.shape
            != (TARGET_VOCABULARY_SIZE, RESIDUAL_SLOT_COUNT)
            or assignment.dtype != np.int64
        ):
            raise ValueError("foldable residual tensors differ")
        self.tied = bool(tied)
        self.base_input_weight = nn.Parameter(input_weight.detach().clone())
        self.input_residual = nn.Parameter(
            torch.zeros(
                RESIDUAL_SLOT_COUNT,
                CODEBOOK_SIZE,
                input_weight.shape[1],
                dtype=input_weight.dtype,
            )
        )
        if self.tied:
            if not torch.equal(input_weight, output_weight):
                raise ValueError("foldable tied base weights differ")
            self.register_parameter("base_output_weight", None)
            self.register_parameter("output_residual", None)
        else:
            self.base_output_weight = nn.Parameter(output_weight.detach().clone())
            self.output_residual = nn.Parameter(
                torch.zeros_like(self.input_residual.detach())
            )
        offsets = (
            torch.arange(RESIDUAL_SLOT_COUNT, dtype=torch.long) * CODEBOOK_SIZE
        )
        codes = torch.from_numpy(assignment.copy()).long() + offsets.unsqueeze(0)
        self.register_buffer("code_indices", codes, persistent=True)
        mask = torch.zeros(TARGET_VOCABULARY_SIZE, 1, dtype=input_weight.dtype)
        mask[BASE_VOCABULARY_SIZE:] = 1.0
        self.register_buffer("new_row_mask", mask, persistent=True)

    @property
    def vocabulary_size(self) -> int:
        return int(self.base_input_weight.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.base_input_weight.shape[1])

    def _dense_residual(self, table: torch.Tensor) -> torch.Tensor:
        selected = F.embedding(
            self.code_indices.reshape(-1),
            table.reshape(RESIDUAL_SLOT_COUNT * CODEBOOK_SIZE, self.hidden_size),
        ).reshape(self.vocabulary_size, RESIDUAL_SLOT_COUNT, self.hidden_size)
        return selected.sum(dim=1) * RESIDUAL_SCALE * self.new_row_mask

    def effective_input_weight(self) -> torch.Tensor:
        return self.base_input_weight + self._dense_residual(self.input_residual)

    def effective_output_weight(self) -> torch.Tensor:
        if self.tied:
            return self.effective_input_weight()
        if self.base_output_weight is None or self.output_residual is None:
            raise AssertionError("foldable untied output parameters are absent")
        return self.base_output_weight + self._dense_residual(self.output_residual)

    def residual_parameter_count(self) -> int:
        return self.input_residual.numel() + (
            0 if self.output_residual is None else self.output_residual.numel()
        )

    def residuals_are_exact_zero(self) -> bool:
        return bool(
            torch.count_nonzero(self.input_residual.detach()).item() == 0
            and (
                self.output_residual is None
                or torch.count_nonzero(self.output_residual.detach()).item() == 0
            )
        )


class _ResidualAdapter(nn.Module):
    def __init__(self, target: FoldableVocabularyResidual, *, output: bool) -> None:
        super().__init__()
        object.__setattr__(self, "_target_reference", weakref.ref(target))
        self.output = bool(output)

    @property
    def target(self) -> FoldableVocabularyResidual:
        value = self._target_reference()
        if value is None:
            raise RuntimeError("foldable residual target no longer exists")
        return value

    @property
    def weight(self) -> torch.Tensor:
        return (
            self.target.effective_output_weight()
            if self.output
            else self.target.effective_input_weight()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.output:
            return F.linear(values, self.target.effective_output_weight())
        if values.dtype != torch.long:
            raise ValueError("foldable residual input IDs must be int64")
        return F.embedding(values, self.target.effective_input_weight())


def install_foldable_residual(
    model: Any,
    assignment: np.ndarray,
    *,
    tied: bool,
) -> Any:
    if (
        not hasattr(model, "model")
        or not hasattr(model.model, "embed_tokens")
        or not hasattr(model, "lm_head")
    ):
        raise ValueError("foldable residual target model differs")
    input_weight = model.model.embed_tokens.weight.detach().cpu().contiguous()
    output_weight = model.lm_head.weight.detach().cpu().contiguous()
    actual_tied = (
        model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    )
    if actual_tied is not tied:
        raise ValueError("foldable residual architecture tying differs")
    residual = FoldableVocabularyResidual(
        input_weight, output_weight, assignment, tied=tied
    )
    model.model.embed_tokens = _ResidualAdapter(residual, output=False)
    model.lm_head = _ResidualAdapter(residual, output=True)
    model.foldable_residual = residual
    model.config.tie_word_embeddings = False
    model.config.vocab_size = TARGET_VOCABULARY_SIZE
    model.vocab_size = TARGET_VOCABULARY_SIZE
    return model


def build_foldable_model(
    role: str,
    *,
    base_state: Mapping[str, torch.Tensor],
    base_tokenizer: Any,
    base_pieces: Sequence[bytes],
    target_pieces: Sequence[bytes],
    decompositions: Sequence[Sequence[int]],
    exposure_counts: np.ndarray,
) -> tuple[Any, Any, ResidualAssignmentAudit]:
    definition = role_definition(role)
    base_role = definition["base_initializer_role"]
    dense, initialization_audit, _ = build_transferred_model(
        base_role,
        base_state=base_state,
        base_tokenizer=base_tokenizer,
        base_pieces=base_pieces,
        target_pieces=target_pieces,
        decompositions=decompositions,
    )
    assignment = build_residual_assignment(
        target_pieces,
        exposure_counts,
        kind=definition["assignment_kind"],
    )
    model = install_foldable_residual(
        dense, assignment, tied=definition["architecture"] == "tied"
    )
    audit = audit_residual_assignment(
        target_pieces,
        exposure_counts,
        kind=definition["assignment_kind"],
    )
    return model, initialization_audit, audit


def expected_parameter_counts(role: str) -> dict[str, int]:
    definition = role_definition(role)
    model = build_target_graph(definition["base_initializer_role"])
    deployed = sum(parameter.numel() for parameter in model.parameters())
    hidden_size = int(model.config.hidden_size)
    multiplier = 1 if definition["architecture"] == "tied" else 2
    residual = multiplier * RESIDUAL_SLOT_COUNT * CODEBOOK_SIZE * hidden_size
    return {
        "deployed": deployed,
        "training_only_residual": residual,
        "training_total": deployed + residual,
    }


def folded_dense_state(model: Any, role: str) -> dict[str, torch.Tensor]:
    definition = role_definition(role)
    residual = getattr(model, "foldable_residual", None)
    if not isinstance(residual, FoldableVocabularyResidual):
        raise ValueError("foldable residual module is absent")
    target = build_target_graph(definition["base_initializer_role"])
    target_state = target.state_dict()
    source_state = model.state_dict()
    output: dict[str, torch.Tensor] = {}
    for name, target_value in target_state.items():
        if name == "model.embed_tokens.weight":
            value = residual.effective_input_weight()
        elif name == "lm_head.weight":
            value = residual.effective_output_weight()
        else:
            if name not in source_state:
                raise ValueError("foldable residual body state differs")
            value = source_state[name]
        value = value.detach().cpu().contiguous()
        if value.shape != target_value.shape or value.dtype != target_value.dtype:
            raise ValueError("foldable residual folded tensor differs")
        output[name] = value
    if set(output) != set(target_state):
        raise AssertionError("foldable residual deployed state keys differ")
    return output


def build_folded_dense_model(model: Any, role: str) -> Any:
    definition = role_definition(role)
    output = build_target_graph(definition["base_initializer_role"])
    output.load_state_dict(folded_dense_state(model, role), strict=True)
    return output


def fold_audit(model: Any, role: str) -> dict[str, Any]:
    counts = expected_parameter_counts(role)
    state = folded_dense_state(model, role)
    residual = model.foldable_residual
    old_input = residual.effective_input_weight()[:BASE_VOCABULARY_SIZE].detach().cpu()
    old_base = residual.base_input_weight[:BASE_VOCABULARY_SIZE].detach().cpu()
    old_output = residual.effective_output_weight()[:BASE_VOCABULARY_SIZE].detach().cpu()
    output_base = (
        residual.base_input_weight
        if residual.tied
        else residual.base_output_weight
    )
    if output_base is None:
        raise AssertionError("foldable residual output base differs")
    return {
        "deployed_parameter_count": counts["deployed"],
        "folded_state_sha256": state_mapping_sha256(state),
        "old_input_rows_unchanged_by_residual": torch.equal(old_input, old_base),
        "old_output_rows_unchanged_by_residual": torch.equal(
            old_output, output_base[:BASE_VOCABULARY_SIZE].detach().cpu()
        ),
        "residual_parameter_count": counts["training_only_residual"],
        "training_parameter_count": counts["training_total"],
    }


def residual_decision(
    contiguous_bpb: Mapping[str, float],
    document_nll: Mapping[str, np.ndarray],
    document_raw_bytes: np.ndarray,
    *,
    anchor_bpb: float,
) -> dict[str, Any]:
    expected_roles = {
        *RESIDUAL_ROLES,
        "untied_base",
        "tied_base",
    }
    if (
        set(contiguous_bpb) != expected_roles
        or set(document_nll) != expected_roles
        or document_raw_bytes.ndim != 1
        or len(document_raw_bytes) < 2
        or np.any(document_raw_bytes <= 0)
        or not math.isfinite(anchor_bpb)
        or anchor_bpb <= 0
    ):
        raise ValueError("foldable residual decision inputs differ")
    diagnostics: dict[str, Any] = {}
    qualified: list[str] = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        candidate = f"{architecture}_jamo"
        controls = (
            f"{architecture}_base",
            f"{architecture}_generic_surface",
            f"{architecture}_shuffled_jamo",
        )
        contrasts: dict[str, Any] = {}
        for control_index, control in enumerate(controls):
            point, lower, upper = document_bootstrap_upper(
                document_nll[candidate],
                document_nll[control],
                document_raw_bytes,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=BOOTSTRAP_SEED + architecture_index * 10 + control_index,
            )
            contiguous_difference = float(contiguous_bpb[candidate]) - float(
                contiguous_bpb[control]
            )
            minimum = 0.0 if control.endswith("_base") else MINIMUM_JAMO_ADVANTAGE_BPB
            point_pass = point < 0.0 if minimum == 0.0 else point <= -minimum
            contiguous_pass = (
                contiguous_difference < 0.0
                if minimum == 0.0
                else contiguous_difference <= -minimum
            )
            contrast_pass = bool(contiguous_pass and point_pass and upper <= 0.0)
            contrasts[control] = {
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
                "contiguous_bpb_difference": contiguous_difference,
                "document_bpb_difference": point,
                "minimum_required_advantage_bpb": minimum,
                "pass": contrast_pass,
            }
        anchor_gap = float(contiguous_bpb[candidate]) - anchor_bpb
        passes = bool(
            anchor_gap <= MAXIMUM_ANCHOR_GAP_BPB
            and all(row["pass"] for row in contrasts.values())
        )
        diagnostics[architecture] = {
            "anchor_gap_bpb": anchor_gap,
            "anchor_recovery_pass": anchor_gap <= MAXIMUM_ANCHOR_GAP_BPB,
            "candidate": candidate,
            "contrasts": contrasts,
            "joint_pass": passes,
        }
        if passes:
            qualified.append(candidate)
    return {
        "status": "foldable_jamo_residual_pass" if qualified else "foldable_jamo_residual_stopped",
        "qualified_jamo_roles": qualified,
        "fresh_equal_history_stage_authorized": bool(qualified),
        "minimum_jamo_advantage_bpb": MINIMUM_JAMO_ADVANTAGE_BPB,
        "maximum_anchor_gap_bpb": MAXIMUM_ANCHOR_GAP_BPB,
        "architecture_diagnostics": diagnostics,
        "threshold_or_role_fallback": None,
    }
