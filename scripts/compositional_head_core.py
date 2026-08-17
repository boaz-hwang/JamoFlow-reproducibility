"""Model roles for the constant-budget compositional-head systems gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Sequence

import numpy as np
import torch

from compositional_token_head import (
    CODEBOOK_COUNT,
    CODEBOOK_ROWS,
    CODEBOOK_SIZE,
    CompositionalVocabulary,
    LowRankVocabulary,
    AssignmentKind,
    audit_token_code_assignment,
    build_token_code_assignment,
    install_factorized_vocabulary,
    low_rank_for_budget,
)
from scalar_runtime_core import build_bpe_model, model_parameter_count
from token_frontier_core import FRONTIER_SPECS


BASELINE_VOCABULARY_SIZE = 2_048
VOCABULARY_SIZES = (8_192, 16_000, 32_000)
MODEL_SEED = 20_260_822
BOOTSTRAP_SEED = 20_260_823
BOOTSTRAP_REPETITIONS = 10_000
MINIMUM_END_TO_END_REDUCTION = 0.10
MINIMUM_STEP_REDUCTION = 0.10
MINIMUM_POSITIVE_PROMPTS = 24
BASE_ROLE = "dense_v2048"
HEAD_KINDS = ("dense", "low_rank", "generic_code", "hangul_code")
ROLE_ORDER = (BASE_ROLE,) + tuple(
    f"{kind}_v{vocabulary_size}"
    for vocabulary_size in VOCABULARY_SIZES
    for kind in HEAD_KINDS
)
BODY_SPEC = FRONTIER_SPECS["byte_bpe_v2048_d8"]
HEAD_PARAMETER_BUDGET = BASELINE_VOCABULARY_SIZE * BODY_SPEC.hidden_size
BODY_PARAMETER_COUNT = BODY_SPEC.expected_parameters - HEAD_PARAMETER_BUDGET


@dataclass(frozen=True, slots=True)
class CompositionalHeadSpec:
    role: str
    head_kind: str
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    key_value_heads: int
    maximum_positions: int
    rank: int | None
    codebook_count: int | None
    codebook_size: int | None
    head_parameters: int
    expected_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_role(role: str) -> tuple[str, int]:
    if role == BASE_ROLE:
        return "dense", BASELINE_VOCABULARY_SIZE
    if "_v" not in role:
        raise ValueError("compositional-head role is malformed")
    kind, raw_vocabulary = role.rsplit("_v", maxsplit=1)
    vocabulary_size = int(raw_vocabulary)
    if kind not in HEAD_KINDS or vocabulary_size not in VOCABULARY_SIZES:
        raise ValueError("compositional-head role coordinates differ")
    if f"{kind}_v{vocabulary_size}" != role:
        raise ValueError("compositional-head role is not canonical")
    return kind, vocabulary_size


def role_spec(role: str) -> CompositionalHeadSpec:
    kind, vocabulary_size = parse_role(role)
    rank = None
    codebook_count = None
    codebook_size = None
    if kind == "dense":
        head_parameters = vocabulary_size * BODY_SPEC.hidden_size
    elif kind == "low_rank":
        rank = low_rank_for_budget(
            vocabulary_size, BODY_SPEC.hidden_size, HEAD_PARAMETER_BUDGET
        )
        head_parameters = rank * (vocabulary_size + BODY_SPEC.hidden_size)
    else:
        codebook_count = CODEBOOK_COUNT
        codebook_size = CODEBOOK_SIZE
        head_parameters = CODEBOOK_ROWS * BODY_SPEC.hidden_size
    return CompositionalHeadSpec(
        role=role,
        head_kind=kind,
        vocabulary_size=vocabulary_size,
        hidden_size=BODY_SPEC.hidden_size,
        intermediate_size=BODY_SPEC.intermediate_size,
        layers=BODY_SPEC.layers,
        attention_heads=BODY_SPEC.attention_heads,
        key_value_heads=BODY_SPEC.key_value_heads,
        maximum_positions=BODY_SPEC.maximum_positions,
        rank=rank,
        codebook_count=codebook_count,
        codebook_size=codebook_size,
        head_parameters=head_parameters,
        expected_parameters=BODY_PARAMETER_COUNT + head_parameters,
    )


ROLE_SPECS = {role: role_spec(role) for role in ROLE_ORDER}


def _new_generator(seed: int, domain: int) -> torch.Generator:
    if seed < 0 or domain < 0:
        raise ValueError("compositional-head initialization seed differs")
    return torch.Generator(device="cpu").manual_seed(seed + domain)


def _install_dense_vocabulary(model: Any, vocabulary_size: int, *, seed: int) -> Any:
    import torch.nn as nn

    hidden_size = int(model.config.hidden_size)
    generator = _new_generator(seed, vocabulary_size)
    weight = torch.empty(vocabulary_size, hidden_size, dtype=torch.float32)
    weight.normal_(mean=0.0, std=float(model.config.initializer_range), generator=generator)
    embedding = nn.Embedding(vocabulary_size, hidden_size)
    embedding.weight = nn.Parameter(weight)
    output = nn.Linear(hidden_size, vocabulary_size, bias=False)
    output.weight = embedding.weight
    model.model.embed_tokens = embedding
    model.lm_head = output
    model.config.vocab_size = vocabulary_size
    model.config.tie_word_embeddings = True
    model.vocab_size = vocabulary_size
    return model


def _install_low_rank_vocabulary(
    model: Any,
    vocabulary_size: int,
    *,
    seed: int,
) -> Any:
    hidden_size = int(model.config.hidden_size)
    rank = low_rank_for_budget(vocabulary_size, hidden_size, HEAD_PARAMETER_BUDGET)
    generator = _new_generator(seed, 100_000 + vocabulary_size)
    factors = torch.empty(vocabulary_size, rank, dtype=torch.float32)
    factors.normal_(
        mean=0.0,
        std=float(model.config.initializer_range),
        generator=generator,
    )
    projection = torch.empty(rank, hidden_size, dtype=torch.float32)
    projection.normal_(mean=0.0, std=rank**-0.5, generator=generator)
    return install_factorized_vocabulary(
        model, LowRankVocabulary.build(factors, projection)
    )


def _assignment_kind(head_kind: str) -> AssignmentKind:
    if head_kind == "generic_code":
        return "generic_unicode"
    if head_kind == "hangul_code":
        return "hangul"
    raise ValueError("head kind does not define a compositional assignment")


def build_model(
    role: str,
    *,
    token_bytes: Sequence[bytes] | None = None,
    seed: int = MODEL_SEED,
) -> Any:
    """Build one role while keeping every role's Transformer body identical."""

    kind, vocabulary_size = parse_role(role)
    model = build_bpe_model(BODY_SPEC.to_dict(), seed=seed)
    if kind == "dense" and vocabulary_size == BASELINE_VOCABULARY_SIZE:
        output = model
    elif kind == "dense":
        output = _install_dense_vocabulary(model, vocabulary_size, seed=seed)
    elif kind == "low_rank":
        output = _install_low_rank_vocabulary(model, vocabulary_size, seed=seed)
    else:
        if token_bytes is None or len(token_bytes) != vocabulary_size:
            raise ValueError("compositional role requires its exact token byte table")
        assignment = build_token_code_assignment(
            token_bytes, kind=_assignment_kind(kind)
        )
        initial = model.get_input_embeddings().weight.detach().reshape(
            CODEBOOK_COUNT, CODEBOOK_SIZE, BODY_SPEC.hidden_size
        )
        vocabulary = CompositionalVocabulary.build(
            initial, torch.tensor(assignment, dtype=torch.long)
        )
        output = install_factorized_vocabulary(model, vocabulary)
    actual = model_parameter_count(output)
    if actual != ROLE_SPECS[role].expected_parameters:
        raise RuntimeError(
            f"compositional-head parameter count differs: {role}: "
            f"{actual} != {ROLE_SPECS[role].expected_parameters}"
        )
    return output


def assignment_audit_for_role(
    role: str,
    token_bytes: Sequence[bytes],
) -> dict[str, Any] | None:
    kind, vocabulary_size = parse_role(role)
    if len(token_bytes) != vocabulary_size:
        raise ValueError("compositional-head token byte table differs")
    if kind not in ("generic_code", "hangul_code"):
        return None
    assignment_kind = _assignment_kind(kind)
    assignment = build_token_code_assignment(token_bytes, kind=assignment_kind)
    return audit_token_code_assignment(
        token_bytes, assignment, kind=assignment_kind
    ).to_dict()


def parameter_fraction_from_baseline(role: str) -> float:
    spec = ROLE_SPECS[role]
    return spec.expected_parameters / ROLE_SPECS[BASE_ROLE].expected_parameters - 1.0


def analytical_head_multiply_adds_per_position(role: str) -> int:
    spec = ROLE_SPECS[role]
    if spec.head_kind == "dense":
        return spec.vocabulary_size * spec.hidden_size
    if spec.head_kind == "low_rank":
        if spec.rank is None:
            raise AssertionError("low-rank role has no rank")
        return spec.rank * (spec.vocabulary_size + spec.hidden_size)
    if spec.codebook_count is None:
        raise AssertionError("codebook role has no codebook count")
    return CODEBOOK_ROWS * spec.hidden_size + spec.codebook_count * spec.vocabulary_size


def paired_latency_comparison(
    candidate_ms: np.ndarray,
    baseline_ms: np.ndarray,
    candidate_steps: np.ndarray,
    baseline_steps: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    """Collapse repetitions within prompt, then bootstrap paired prompts."""

    arrays = (candidate_ms, baseline_ms, candidate_steps, baseline_steps)
    if (
        any(value.ndim != 2 for value in arrays)
        or len({value.shape for value in arrays}) != 1
        or candidate_ms.shape[0] < 2
        or candidate_ms.shape[1] < 1
        or bootstrap_seed < 0
        or bootstrap_repetitions <= 0
        or not all(np.all(np.isfinite(value)) for value in arrays)
        or np.any(candidate_ms <= 0)
        or np.any(baseline_ms <= 0)
        or np.any(candidate_steps <= 0)
        or np.any(baseline_steps <= 0)
    ):
        raise ValueError("compositional-head paired timing arrays differ")
    candidate = np.median(candidate_ms, axis=1)
    baseline = np.median(baseline_ms, axis=1)
    candidate_step_values = np.median(candidate_steps, axis=1)
    baseline_step_values = np.median(baseline_steps, axis=1)
    point = 1.0 - float(np.median(candidate)) / float(np.median(baseline))
    step_reduction = 1.0 - float(candidate_step_values.sum()) / float(
        baseline_step_values.sum()
    )
    rng = np.random.default_rng(bootstrap_seed)
    estimates = np.empty(bootstrap_repetitions, dtype=np.float64)
    for index in range(bootstrap_repetitions):
        rows = rng.integers(0, len(candidate), size=len(candidate))
        estimates[index] = 1.0 - float(np.median(candidate[rows])) / float(
            np.median(baseline[rows])
        )
    return {
        "baseline_median_ms": float(np.median(baseline)),
        "candidate_median_ms": float(np.median(candidate)),
        "end_to_end_reduction": point,
        "bootstrap_95_lower": float(np.quantile(estimates, 0.025)),
        "bootstrap_95_upper": float(np.quantile(estimates, 0.975)),
        "positive_prompt_count": int(np.count_nonzero(candidate < baseline)),
        "prompt_count": len(candidate),
        "continuation_step_reduction": step_reduction,
        "candidate_continuation_steps": float(candidate_step_values.sum()),
        "baseline_continuation_steps": float(baseline_step_values.sum()),
    }


def preflight_decision(
    comparisons: dict[str, dict[str, Any]],
    correctness: dict[str, bool],
) -> dict[str, Any]:
    expected = set(ROLE_ORDER) - {BASE_ROLE}
    if set(comparisons) != expected or set(correctness) != set(ROLE_ORDER):
        raise ValueError("compositional-head decision role set differs")
    gates = {}
    for role in ROLE_ORDER:
        if role == BASE_ROLE:
            continue
        row = comparisons[role]
        required = {
            "bootstrap_95_lower",
            "continuation_step_reduction",
            "end_to_end_reduction",
            "positive_prompt_count",
            "prompt_count",
        }
        if not required.issubset(row):
            raise ValueError("compositional-head comparison schema differs")
        passed = bool(
            correctness[BASE_ROLE]
            and correctness[role]
            and float(row["end_to_end_reduction"])
            >= MINIMUM_END_TO_END_REDUCTION
            and float(row["bootstrap_95_lower"]) > 0.0
            and float(row["continuation_step_reduction"])
            >= MINIMUM_STEP_REDUCTION
            and int(row["positive_prompt_count"]) >= MINIMUM_POSITIVE_PROMPTS
            and int(row["prompt_count"]) >= MINIMUM_POSITIVE_PROMPTS
        )
        gates[role] = {
            "correctness_pass": bool(correctness[BASE_ROLE] and correctness[role]),
            "minimum_end_to_end_reduction": MINIMUM_END_TO_END_REDUCTION,
            "minimum_step_reduction": MINIMUM_STEP_REDUCTION,
            "minimum_positive_prompts": MINIMUM_POSITIVE_PROMPTS,
            "overall_pass": passed,
        }
    selected = None
    for vocabulary_size in VOCABULARY_SIZES:
        if all(
            gates[f"{kind}_v{vocabulary_size}"]["overall_pass"]
            for kind in ("generic_code", "hangul_code")
        ):
            selected = vocabulary_size
            break
    return {
        "status": (
            "compositional_head_systems_opportunity_pass"
            if selected is not None
            else "compositional_head_systems_branch_stopped"
        ),
        "selected_vocabulary_size": selected,
        "selected_candidate_role": (
            f"hangul_code_v{selected}" if selected is not None else None
        ),
        "selected_generic_control_role": (
            f"generic_code_v{selected}" if selected is not None else None
        ),
        "selected_low_rank_control_role": (
            f"low_rank_v{selected}" if selected is not None else None
        ),
        "selection_rule": (
            "smallest vocabulary where both generic and Hangul codebook roles "
            "pass the fixed systems gate"
        ),
        "gates": gates,
    }


def balanced_role_schedule(case_index: int, repetition: int) -> tuple[str, ...]:
    if case_index < 0 or repetition < 0:
        raise ValueError("compositional-head schedule coordinates differ")
    count = len(ROLE_ORDER)
    trial = case_index * 3 + repetition
    shift = trial % count
    rotated = ROLE_ORDER[shift:] + ROLE_ORDER[:shift]
    # Keep the Latin-cycle first position intact while alternating the remaining
    # traversal direction after each complete cycle.  Reversing ROLE_ORDER before
    # rotating would repeat first-position roles at the cycle boundary.
    if (trial // count) % 2:
        return (rotated[0],) + tuple(reversed(rotated[1:]))
    return rotated


def validate_specs() -> None:
    if set(ROLE_SPECS) != set(ROLE_ORDER) or len(ROLE_SPECS) != len(ROLE_ORDER):
        raise ValueError("compositional-head role grid differs")
    if HEAD_PARAMETER_BUDGET != CODEBOOK_ROWS * BODY_SPEC.hidden_size:
        raise ValueError("compositional codebook does not match the 2K head budget")
    for role in ROLE_ORDER:
        spec = ROLE_SPECS[role]
        if spec.expected_parameters <= 0 or not math.isfinite(
            parameter_fraction_from_baseline(role)
        ):
            raise ValueError("compositional-head model spec differs")


validate_specs()
