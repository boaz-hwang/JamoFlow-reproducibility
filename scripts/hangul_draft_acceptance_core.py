"""Pure contracts and draft heads for the Hangul block acceptance preflight.

This module deliberately lives under ``scripts/``.  The publication model and
quality locks seal the historical ``src/jamoflow`` package closure; exploratory
draft heads must not silently mutate that trust root.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PROTOCOL_ID = "jamoflow-hangul-draft-acceptance-v1"
ARCHITECTURES = (
    "generic_independent_utf8",
    "generic_joint_utf8",
    "hangul_parallel_components",
    "hangul_conditional_components",
)
PRIMARY_GENERIC_CONTROL = "generic_joint_utf8"
PRIMARY_HANGUL_DRAFT = "hangul_conditional_components"
HEAD_TRAINING_SEEDS = (20260813, 20260817, 20260819)
HIDDEN_WIDTH = 192
LEAD_COUNT = 4
CONTINUATION_CARDINALITY = 64
PAIR_CARDINALITY = CONTINUATION_CARDINALITY**2
ONSET_COUNT = 19
VOWEL_COUNT = 21
CODA_COUNT = 28
HANGUL_COUNT = ONSET_COUNT * VOWEL_COUNT * CODA_COUNT
CONDITIONAL_BEAM_WIDTH = 4


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def hangul_components(codepoint: int) -> tuple[int, int, int]:
    if not 0xAC00 <= codepoint <= 0xD7A3:
        raise ValueError("codepoint is not a precomposed Hangul syllable")
    offset = codepoint - 0xAC00
    return offset // 588, (offset % 588) // 28, offset % 28


def hangul_codepoint(onset: int, vowel: int, coda: int) -> int:
    if not (
        0 <= onset < ONSET_COUNT
        and 0 <= vowel < VOWEL_COUNT
        and 0 <= coda < CODA_COUNT
    ):
        raise ValueError("Hangul component index differs")
    return 0xAC00 + (onset * VOWEL_COUNT + vowel) * CODA_COUNT + coda


def pair_index(second: int, third: int) -> int:
    if not (0x80 <= second <= 0xBF and 0x80 <= third <= 0xBF):
        raise ValueError("UTF-8 continuation byte differs")
    return (second - 0x80) * CONTINUATION_CARDINALITY + (third - 0x80)


def pair_bytes(index: int) -> tuple[int, int]:
    if not 0 <= index < PAIR_CARDINALITY:
        raise ValueError("continuation-pair index differs")
    return 0x80 + index // CONTINUATION_CARDINALITY, 0x80 + index % CONTINUATION_CARDINALITY


@dataclass(frozen=True, slots=True)
class HangulTables:
    lead: np.ndarray
    onset: np.ndarray
    vowel: np.ndarray
    coda: np.ndarray
    pair: np.ndarray

    @classmethod
    def build(cls) -> "HangulTables":
        lead = np.empty(HANGUL_COUNT, dtype=np.int64)
        onset = np.empty(HANGUL_COUNT, dtype=np.int64)
        vowel = np.empty(HANGUL_COUNT, dtype=np.int64)
        coda = np.empty(HANGUL_COUNT, dtype=np.int64)
        pair = np.empty(HANGUL_COUNT, dtype=np.int64)
        for index, codepoint in enumerate(range(0xAC00, 0xD7A4)):
            raw = chr(codepoint).encode("utf-8")
            lead[index] = raw[0] - 0xEA
            onset[index], vowel[index], coda[index] = hangul_components(codepoint)
            pair[index] = pair_index(raw[1], raw[2])
        if set(lead.tolist()) != set(range(LEAD_COUNT)):
            raise AssertionError("Hangul lead table differs")
        return cls(lead=lead, onset=onset, vowel=vowel, coda=coda, pair=pair)


HANGUL_TABLES = HangulTables.build()


def _features(hidden: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_WIDTH:
        raise ValueError("draft hidden shape differs")
    if lead.shape != (len(hidden),) or lead.dtype != torch.long:
        raise ValueError("draft lead shape/dtype differs")
    if bool(torch.any((lead < 0) | (lead >= LEAD_COUNT))):
        raise ValueError("draft lead value differs")
    one_hot = F.one_hot(lead, num_classes=LEAD_COUNT).to(dtype=hidden.dtype)
    return torch.cat((hidden, one_hot), dim=-1)


class GenericIndependentUtf8(nn.Module):
    architecture = "generic_independent_utf8"

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(HIDDEN_WIDTH + LEAD_COUNT, 128)
        self.second = nn.Linear(128, CONTINUATION_CARDINALITY)
        self.third = nn.Linear(128, CONTINUATION_CARDINALITY)

    def forward(self, hidden: torch.Tensor, lead: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = F.gelu(self.trunk(_features(hidden, lead)))
        return self.second(state), self.third(state)


class GenericJointUtf8(nn.Module):
    architecture = "generic_joint_utf8"

    def __init__(self) -> None:
        super().__init__()
        # Rank nine makes this strong joint-pair control parameter-matched to
        # the independent and Hangul heads instead of hiding a 4096-way cost.
        self.trunk = nn.Linear(HIDDEN_WIDTH + LEAD_COUNT, 9)
        self.pair = nn.Linear(9, PAIR_CARDINALITY)

    def forward(self, hidden: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        return self.pair(F.gelu(self.trunk(_features(hidden, lead))))


class HangulParallelComponents(nn.Module):
    architecture = "hangul_parallel_components"

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(HIDDEN_WIDTH + LEAD_COUNT, 160)
        self.onset = nn.Linear(160, ONSET_COUNT)
        self.vowel = nn.Linear(160, VOWEL_COUNT)
        self.coda = nn.Linear(160, CODA_COUNT)

    def forward(
        self,
        hidden: torch.Tensor,
        lead: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = F.gelu(self.trunk(_features(hidden, lead)))
        return self.onset(state), self.vowel(state), self.coda(state)


class HangulConditionalComponents(nn.Module):
    architecture = "hangul_conditional_components"

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(HIDDEN_WIDTH + LEAD_COUNT, 128)
        self.onset = nn.Linear(128, ONSET_COUNT)
        self.onset_embedding = nn.Embedding(ONSET_COUNT, 48)
        self.vowel = nn.Linear(128 + 48, VOWEL_COUNT)
        self.vowel_embedding = nn.Embedding(VOWEL_COUNT, 48)
        self.coda = nn.Linear(128 + 96, CODA_COUNT)

    def base(self, hidden: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.trunk(_features(hidden, lead)))

    def forward(
        self,
        hidden: torch.Tensor,
        lead: torch.Tensor,
        onset_target: torch.Tensor,
        vowel_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.base(hidden, lead)
        onset_logits = self.onset(state)
        vowel_input = torch.cat((state, self.onset_embedding(onset_target)), dim=-1)
        vowel_logits = self.vowel(vowel_input)
        coda_input = torch.cat(
            (
                state,
                self.onset_embedding(onset_target),
                self.vowel_embedding(vowel_target),
            ),
            dim=-1,
        )
        return onset_logits, vowel_logits, self.coda(coda_input)


def build_head(architecture: str) -> nn.Module:
    builders = {
        "generic_independent_utf8": GenericIndependentUtf8,
        "generic_joint_utf8": GenericJointUtf8,
        "hangul_parallel_components": HangulParallelComponents,
        "hangul_conditional_components": HangulConditionalComponents,
    }
    try:
        return builders[architecture]()
    except KeyError as error:
        raise ValueError("draft architecture differs") from error


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def training_loss(
    model: nn.Module,
    hidden: torch.Tensor,
    lead: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
    onset: torch.Tensor,
    vowel: torch.Tensor,
    coda: torch.Tensor,
) -> torch.Tensor:
    architecture = getattr(model, "architecture", None)
    if architecture == "generic_independent_utf8":
        second_logits, third_logits = model(hidden, lead)
        return F.cross_entropy(second_logits, second) + F.cross_entropy(third_logits, third)
    if architecture == "generic_joint_utf8":
        return F.cross_entropy(model(hidden, lead), second * CONTINUATION_CARDINALITY + third)
    if architecture == "hangul_parallel_components":
        onset_logits, vowel_logits, coda_logits = model(hidden, lead)
        return (
            F.cross_entropy(onset_logits, onset)
            + F.cross_entropy(vowel_logits, vowel)
            + F.cross_entropy(coda_logits, coda)
        )
    if architecture == "hangul_conditional_components":
        onset_logits, vowel_logits, coda_logits = model(hidden, lead, onset, vowel)
        return (
            F.cross_entropy(onset_logits, onset)
            + F.cross_entropy(vowel_logits, vowel)
            + F.cross_entropy(coda_logits, coda)
        )
    raise ValueError("draft training architecture differs")


@dataclass(slots=True)
class DeviceHangulTables:
    lead: torch.Tensor
    onset: torch.Tensor
    vowel: torch.Tensor
    coda: torch.Tensor
    pair: torch.Tensor
    onset_allowed: torch.Tensor
    vowel_allowed: torch.Tensor
    coda_allowed: torch.Tensor

    @classmethod
    def build(cls, device: torch.device | str) -> "DeviceHangulTables":
        lead = torch.from_numpy(HANGUL_TABLES.lead).to(device)
        onset = torch.from_numpy(HANGUL_TABLES.onset).to(device)
        vowel = torch.from_numpy(HANGUL_TABLES.vowel).to(device)
        coda = torch.from_numpy(HANGUL_TABLES.coda).to(device)
        pair = torch.from_numpy(HANGUL_TABLES.pair).to(device)
        onset_allowed = torch.zeros((LEAD_COUNT, ONSET_COUNT), dtype=torch.bool, device=device)
        vowel_allowed = torch.zeros(
            (LEAD_COUNT, ONSET_COUNT, VOWEL_COUNT), dtype=torch.bool, device=device
        )
        coda_allowed = torch.zeros(
            (LEAD_COUNT, ONSET_COUNT, VOWEL_COUNT, CODA_COUNT),
            dtype=torch.bool,
            device=device,
        )
        onset_allowed[lead, onset] = True
        vowel_allowed[lead, onset, vowel] = True
        coda_allowed[lead, onset, vowel, coda] = True
        return cls(
            lead=lead,
            onset=onset,
            vowel=vowel,
            coda=coda,
            pair=pair,
            onset_allowed=onset_allowed,
            vowel_allowed=vowel_allowed,
            coda_allowed=coda_allowed,
        )


def _candidate_argmax(
    scores: torch.Tensor,
    lead: torch.Tensor,
    tables: DeviceHangulTables,
) -> torch.Tensor:
    output = torch.empty(len(scores), dtype=torch.long, device=scores.device)
    for current_lead in range(LEAD_COUNT):
        rows = torch.nonzero(lead == current_lead, as_tuple=False).flatten()
        if not len(rows):
            continue
        candidates = torch.nonzero(tables.lead == current_lead, as_tuple=False).flatten()
        local = scores[rows][:, candidates]
        chosen = candidates[local.argmax(dim=-1)]
        output[rows] = tables.pair[chosen]
    return output


def propose_pairs(
    model: nn.Module,
    hidden: torch.Tensor,
    lead: torch.Tensor,
    tables: DeviceHangulTables | None = None,
) -> torch.Tensor:
    if tables is None:
        tables = DeviceHangulTables.build(hidden.device)
    architecture = getattr(model, "architecture", None)
    if architecture == "generic_independent_utf8":
        second, third = model(hidden, lead)
        scores = second[:, tables.pair // CONTINUATION_CARDINALITY] + third[
            :, tables.pair % CONTINUATION_CARDINALITY
        ]
        return _candidate_argmax(scores, lead, tables)
    if architecture == "generic_joint_utf8":
        pair_logits = model(hidden, lead)
        return _candidate_argmax(pair_logits[:, tables.pair], lead, tables)
    if architecture == "hangul_parallel_components":
        onset, vowel, coda = model(hidden, lead)
        scores = onset[:, tables.onset] + vowel[:, tables.vowel] + coda[:, tables.coda]
        return _candidate_argmax(scores, lead, tables)
    if architecture != "hangul_conditional_components":
        raise ValueError("draft proposal architecture differs")

    conditional = model
    state = conditional.base(hidden, lead)
    onset_logits = conditional.onset(state).masked_fill(
        ~tables.onset_allowed[lead], -torch.inf
    )
    onset_scores, onset_indices = torch.topk(
        onset_logits,
        k=CONDITIONAL_BEAM_WIDTH,
        dim=-1,
    )
    batch = len(hidden)
    state_onset = state[:, None, :].expand(-1, CONDITIONAL_BEAM_WIDTH, -1)
    onset_embed = conditional.onset_embedding(onset_indices)
    vowel_logits = conditional.vowel(torch.cat((state_onset, onset_embed), dim=-1))
    vowel_mask = tables.vowel_allowed[
        lead[:, None].expand_as(onset_indices), onset_indices
    ]
    vowel_scores = vowel_logits.masked_fill(~vowel_mask, -torch.inf) + onset_scores[..., None]
    flat_vowel_scores = vowel_scores.reshape(batch, -1)
    beam_scores, flat_vowel = torch.topk(
        flat_vowel_scores,
        k=CONDITIONAL_BEAM_WIDTH,
        dim=-1,
    )
    parent = flat_vowel // VOWEL_COUNT
    vowel_indices = flat_vowel % VOWEL_COUNT
    selected_onset = onset_indices.gather(1, parent)
    state_beam = state[:, None, :].expand(-1, CONDITIONAL_BEAM_WIDTH, -1)
    coda_input = torch.cat(
        (
            state_beam,
            conditional.onset_embedding(selected_onset),
            conditional.vowel_embedding(vowel_indices),
        ),
        dim=-1,
    )
    coda_logits = conditional.coda(coda_input)
    coda_mask = tables.coda_allowed[
        lead[:, None].expand_as(selected_onset),
        selected_onset,
        vowel_indices,
    ]
    coda_scores = coda_logits.masked_fill(~coda_mask, -torch.inf) + beam_scores[..., None]
    flat_coda = coda_scores.reshape(batch, -1).argmax(dim=-1)
    final_parent = flat_coda // CODA_COUNT
    final_coda = flat_coda % CODA_COUNT
    final_onset = selected_onset.gather(1, final_parent[:, None]).squeeze(1)
    final_vowel = vowel_indices.gather(1, final_parent[:, None]).squeeze(1)
    codepoint = 0xAC00 + (final_onset * VOWEL_COUNT + final_vowel) * CODA_COUNT + final_coda
    first = 0xE0 | (codepoint >> 12)
    if not bool(torch.all(first - 0xEA == lead)):
        raise AssertionError("conditional Hangul proposal violates the target lead")
    second = 0x80 | ((codepoint >> 6) & 0x3F)
    third = 0x80 | (codepoint & 0x3F)
    return (second - 0x80) * CONTINUATION_CARDINALITY + (third - 0x80)


def proposal_metrics(
    prediction: np.ndarray,
    target_second: np.ndarray,
    target_third: np.ndarray,
    target_is_hangul: np.ndarray,
) -> dict[str, float | int]:
    predicted = np.asarray(prediction)
    second = np.asarray(target_second)
    third = np.asarray(target_third)
    hangul = np.asarray(target_is_hangul)
    shape = (len(predicted),)
    if (
        predicted.dtype != np.int64
        or second.dtype != np.uint8
        or third.dtype != np.uint8
        or hangul.dtype != np.bool_
        or predicted.shape != shape
        or second.shape != shape
        or third.shape != shape
        or hangul.shape != shape
        or not len(predicted)
        or np.any(predicted < 0)
        or np.any(predicted >= PAIR_CARDINALITY)
        or np.any(second >= CONTINUATION_CARDINALITY)
        or np.any(third >= CONTINUATION_CARDINALITY)
    ):
        raise ValueError("draft proposal metric arrays differ")
    predicted_second = predicted // CONTINUATION_CARDINALITY
    predicted_third = predicted % CONTINUATION_CARDINALITY
    first_match = predicted_second == second
    complete = first_match & (predicted_third == third)
    suffix = first_match.astype(np.int64) + complete.astype(np.int64)
    hangul_count = int(hangul.sum())
    return {
        "attempt_count": int(len(predicted)),
        "target_hangul_count": hangul_count,
        "target_hangul_rate": float(hangul.mean()),
        "first_continuation_acceptance": float(first_match.mean()),
        "complete_pair_acceptance": float(complete.mean()),
        "mean_accepted_suffix_bytes": float(suffix.mean()),
        "complete_pair_acceptance_when_target_hangul": (
            float(complete[hangul].mean()) if hangul_count else 0.0
        ),
    }


def paired_prompt_bootstrap(
    left_exact: np.ndarray,
    right_exact: np.ndarray,
    prompt_index: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    left = np.asarray(left_exact)
    right = np.asarray(right_exact)
    prompts = np.asarray(prompt_index)
    if (
        left.dtype != np.bool_
        or right.dtype != np.bool_
        or prompts.dtype != np.int64
        or left.shape != right.shape
        or left.shape != prompts.shape
        or not len(left)
        or repetitions <= 0
    ):
        raise ValueError("paired prompt bootstrap input differs")
    unique = np.unique(prompts)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
        raise ValueError("prompt indices must be dense")
    left_by_prompt = np.asarray([left[prompts == value].mean() for value in unique])
    right_by_prompt = np.asarray([right[prompts == value].mean() for value in unique])
    differences = left_by_prompt - right_by_prompt
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    samples = differences[draws].mean(axis=1)
    return {
        "prompt_count": int(len(unique)),
        "point_difference": float(differences.mean()),
        "ci_lower": float(np.quantile(samples, 0.025)),
        "ci_upper": float(np.quantile(samples, 0.975)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def evaluate_gates(
    architecture_summary: Mapping[str, Mapping[str, Any]],
    specificity: Mapping[str, Any],
    *,
    minimum_attempts: int,
    minimum_complete_pair_acceptance: float,
    minimum_mean_accepted_suffix_bytes: float,
    minimum_per_seed_complete_pair_acceptance: float,
    maximum_median_head_latency_ms: float,
    minimum_specificity_acceptance_gain: float,
) -> dict[str, Any]:
    if set(architecture_summary) != set(ARCHITECTURES):
        raise ValueError("draft gate architecture set differs")
    systems: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        row = architecture_summary[architecture]
        per_seed = row["per_seed_free_complete_pair_acceptance"]
        passed = (
            row["free_attempt_count"] >= minimum_attempts
            and row["median_free_complete_pair_acceptance"]
            >= minimum_complete_pair_acceptance
            and row["median_free_mean_accepted_suffix_bytes"]
            >= minimum_mean_accepted_suffix_bytes
            and min(per_seed.values()) >= minimum_per_seed_complete_pair_acceptance
            and row["median_head_latency_ms"] <= maximum_median_head_latency_ms
        )
        systems[architecture] = {
            "pass": bool(passed),
            "attempt_count": int(row["free_attempt_count"]),
            "median_complete_pair_acceptance": float(
                row["median_free_complete_pair_acceptance"]
            ),
            "median_mean_accepted_suffix_bytes": float(
                row["median_free_mean_accepted_suffix_bytes"]
            ),
            "minimum_seed_complete_pair_acceptance": float(min(per_seed.values())),
            "median_head_latency_ms": float(row["median_head_latency_ms"]),
        }
    primary_specificity = (
        specificity["point_difference"] >= minimum_specificity_acceptance_gain
        and specificity["ci_lower"] > 0.0
    )
    generic_authorized = systems[PRIMARY_GENERIC_CONTROL]["pass"]
    overall = systems[PRIMARY_HANGUL_DRAFT]["pass"] and primary_specificity
    if overall:
        recommended = "hangul_exact_verifier_prototype"
    elif generic_authorized:
        recommended = "generic_joint_exact_verifier_diagnostic_only"
    else:
        recommended = "stop_multi_byte_draft_branch"
    return {
        "systems_feasibility": systems,
        "primary_korean_specificity": {
            "hangul_architecture": PRIMARY_HANGUL_DRAFT,
            "generic_control": PRIMARY_GENERIC_CONTROL,
            "minimum_acceptance_gain": minimum_specificity_acceptance_gain,
            "pass": bool(primary_specificity),
            **specificity,
        },
        "generic_control_prototype_authorized": bool(generic_authorized),
        "overall_hangul_prototype_authorized": bool(overall),
        "recommended_next_stage": recommended,
    }
