"""One-seed quality contracts for the selected 8K compositional vocabulary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from compositional_head_core import (
    BASE_ROLE,
    BODY_SPEC,
    MODEL_SEED as SYSTEMS_MODEL_SEED,
    ROLE_SPECS,
    build_model,
)
from compositional_token_head import (
    CompositionalVocabulary,
    audit_token_code_assignment,
    build_token_code_assignment,
    install_factorized_vocabulary,
)


SELECTED_VOCABULARY_SIZE = 8_192
QUALITY_ROLES = (
    BASE_ROLE,
    "dense_v8192",
    "low_rank_v8192",
    "generic_code_v8192",
    "shuffled_hangul_code_v8192",
    "hangul_code_v8192",
)
PRIMARY_CANDIDATE_ROLE = "hangul_code_v8192"
GENERIC_CONTROL_ROLE = "generic_code_v8192"
SHUFFLED_CONTROL_ROLE = "shuffled_hangul_code_v8192"
LOW_RANK_CONTROL_ROLE = "low_rank_v8192"
DENSE_CEILING_ROLE = "dense_v8192"
MODEL_SEED = 20_260_824
ORDER_SEED = 20_260_825
BOOTSTRAP_SEED = 20_260_826
BOOTSTRAP_REPETITIONS = 10_000
TRAIN_BYTES = 128_000_000
CALIBRATION_BYTES = 8_000_000
SEQUENCE_LENGTH = 512
EFFECTIVE_BATCH_SIZE = 32
TRAIN_MICROBATCH = {
    BASE_ROLE: 32,
    DENSE_CEILING_ROLE: 8,
    LOW_RANK_CONTROL_ROLE: 8,
    GENERIC_CONTROL_ROLE: 8,
    SHUFFLED_CONTROL_ROLE: 8,
    PRIMARY_CANDIDATE_ROLE: 8,
}
EVALUATION_BATCH = {
    BASE_ROLE: 64,
    DENSE_CEILING_ROLE: 16,
    LOW_RANK_CONTROL_ROLE: 16,
    GENERIC_CONTROL_ROLE: 16,
    SHUFFLED_CONTROL_ROLE: 16,
    PRIMARY_CANDIDATE_ROLE: 16,
}
LEARNING_RATE = 3e-4
MINIMUM_LEARNING_RATE = 3e-5
WARMUP_FRACTION = 0.05
BETA1 = 0.9
BETA2 = 0.95
EPSILON = 1e-8
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
BASELINE_NONINFERIORITY_BPB = 0.010
CONTROL_NONINFERIORITY_BPB = 0.002
MINIMUM_KOREAN_ADVANTAGE_BPB = 0.002
RESOURCE_WARMUP_STEPS = 1
RESOURCE_MEASURED_STEPS = 3
RESOURCE_WARMUP_EVALUATION_BATCHES = 1
RESOURCE_MEASURED_EVALUATION_BATCHES = 3
RESOURCE_CAMPAIGN_HOUR_LIMIT = 12.0
RESOURCE_SAFETY_FACTOR = 1.25
RESOURCE_MEMORY_FRACTION_LIMIT = 0.75


@dataclass(frozen=True, slots=True)
class QualityRoleSpec:
    role: str
    head_kind: str
    vocabulary_size: int
    expected_parameters: int
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    key_value_heads: int
    maximum_positions: int
    train_microbatch: int
    evaluation_batch: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_quality_role(role: str) -> tuple[str, int]:
    if role == BASE_ROLE:
        return "dense", 2_048
    if role == SHUFFLED_CONTROL_ROLE:
        return "shuffled_hangul_code", SELECTED_VOCABULARY_SIZE
    if role not in QUALITY_ROLES:
        raise ValueError("compositional quality role differs")
    kind, raw_size = role.rsplit("_v", maxsplit=1)
    size = int(raw_size)
    if size != SELECTED_VOCABULARY_SIZE:
        raise ValueError("compositional quality vocabulary differs")
    return kind, size


def _expected_parameters(role: str) -> int:
    if role == SHUFFLED_CONTROL_ROLE:
        return ROLE_SPECS[PRIMARY_CANDIDATE_ROLE].expected_parameters
    return ROLE_SPECS[role].expected_parameters


QUALITY_SPECS = {
    role: QualityRoleSpec(
        role=role,
        head_kind=parse_quality_role(role)[0],
        vocabulary_size=parse_quality_role(role)[1],
        expected_parameters=_expected_parameters(role),
        hidden_size=BODY_SPEC.hidden_size,
        intermediate_size=BODY_SPEC.intermediate_size,
        layers=BODY_SPEC.layers,
        attention_heads=BODY_SPEC.attention_heads,
        key_value_heads=BODY_SPEC.key_value_heads,
        maximum_positions=BODY_SPEC.maximum_positions,
        train_microbatch=TRAIN_MICROBATCH[role],
        evaluation_batch=EVALUATION_BATCH[role],
    )
    for role in QUALITY_ROLES
}


def build_quality_model(
    role: str,
    *,
    token_bytes: Sequence[bytes] | None,
    seed: int = MODEL_SEED,
):
    if role not in QUALITY_ROLES:
        raise ValueError("compositional quality build role differs")
    if role != SHUFFLED_CONTROL_ROLE:
        return build_model(role, token_bytes=token_bytes, seed=seed)
    if token_bytes is None or len(token_bytes) != SELECTED_VOCABULARY_SIZE:
        raise ValueError("shuffled compositional role requires exact 8K pieces")
    model = build_model(PRIMARY_CANDIDATE_ROLE, token_bytes=token_bytes, seed=seed)
    initial = model.factorized_vocabulary.weight.detach().clone()
    assignment = build_token_code_assignment(token_bytes, kind="shuffled_hangul")
    vocabulary = CompositionalVocabulary.build(
        initial,
        __import__("torch").tensor(assignment, dtype=__import__("torch").long),
    )
    return install_factorized_vocabulary(model, vocabulary)


def assignment_audit(role: str, token_bytes: Sequence[bytes]) -> dict[str, Any] | None:
    if role not in (GENERIC_CONTROL_ROLE, SHUFFLED_CONTROL_ROLE, PRIMARY_CANDIDATE_ROLE):
        return None
    kind = {
        GENERIC_CONTROL_ROLE: "generic_unicode",
        SHUFFLED_CONTROL_ROLE: "shuffled_hangul",
        PRIMARY_CANDIDATE_ROLE: "hangul",
    }[role]
    values = build_token_code_assignment(token_bytes, kind=kind)
    return audit_token_code_assignment(token_bytes, values, kind=kind).to_dict()


def state_subset_sha256(model: Any, *, transformer_body_only: bool) -> str:
    digest = hashlib.sha256(b"JamoFlow/compositional-quality-state/v1\0")
    selected = 0
    excluded = (
        "model.embed_tokens.",
        "lm_head.",
        "factorized_vocabulary.",
    )
    for name, tensor in sorted(model.state_dict().items()):
        if transformer_body_only and name.startswith(excluded):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes(order="C"))
        selected += 1
    if selected <= 0:
        raise ValueError("compositional quality state subset is empty")
    digest.update(selected.to_bytes(8, "big"))
    return digest.hexdigest()


def deterministic_order(sequence_count: int) -> np.ndarray:
    if sequence_count <= 0:
        raise ValueError("compositional quality order requires sequences")
    return np.random.default_rng(ORDER_SEED).permutation(sequence_count).astype(
        np.int64, copy=False
    )


def total_optimizer_steps(sequence_count: int) -> int:
    if sequence_count <= 0:
        raise ValueError("compositional quality sequence count differs")
    return math.ceil(sequence_count / EFFECTIVE_BATCH_SIZE)


def warmup_steps(sequence_count: int) -> int:
    return max(1, math.ceil(total_optimizer_steps(sequence_count) * WARMUP_FRACTION))


def cosine_learning_rate(step: int, total_steps: int, warmup: int) -> float:
    if not 0 <= step < total_steps or not 1 <= warmup <= total_steps:
        raise ValueError("compositional quality learning-rate coordinates differ")
    if step < warmup:
        return LEARNING_RATE * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MINIMUM_LEARNING_RATE + (LEARNING_RATE - MINIMUM_LEARNING_RATE) * cosine


def training_contract(role: str, sequence_count: int) -> dict[str, Any]:
    if role not in QUALITY_ROLES:
        raise ValueError("compositional quality training role differs")
    microbatch = TRAIN_MICROBATCH[role]
    if EFFECTIVE_BATCH_SIZE % microbatch:
        raise ValueError("compositional quality microbatch does not divide batch")
    return {
        "beta1": BETA1,
        "beta2": BETA2,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "epsilon": EPSILON,
        "evaluation_batch_size": EVALUATION_BATCH[role],
        "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // microbatch,
        "gradient_clip": GRADIENT_CLIP,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "model_seed": MODEL_SEED,
        "optimizer": "AdamW",
        "order_seed": ORDER_SEED,
        "sequence_count": sequence_count,
        "total_optimizer_steps": total_optimizer_steps(sequence_count),
        "train_microbatch_size": microbatch,
        "warmup_fraction": WARMUP_FRACTION,
        "warmup_steps": warmup_steps(sequence_count),
        "weight_decay_for_matrix_parameters": WEIGHT_DECAY,
        "weight_decay_for_vector_parameters": 0.0,
    }


def _contrast_pass(
    row: Mapping[str, Any],
    *,
    point_limit: float,
    upper_limit: float,
) -> bool:
    required = {
        "bootstrap_95_upper",
        "contiguous_bpb_difference",
        "document_bpb_difference",
    }
    if set(row) != required or any(
        not math.isfinite(float(row[key])) for key in required
    ):
        raise ValueError("compositional quality contrast differs")
    return bool(
        float(row["contiguous_bpb_difference"]) <= point_limit
        and float(row["document_bpb_difference"]) <= point_limit
        and float(row["bootstrap_95_upper"]) <= upper_limit
    )


def quality_decision(contrasts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        "hangul_vs_dense_2k",
        "hangul_vs_generic",
        "hangul_vs_shuffled",
        "hangul_vs_low_rank",
        "generic_vs_dense_2k",
    }
    if set(contrasts) != expected:
        raise ValueError("compositional quality decision contrast set differs")
    baseline = _contrast_pass(
        contrasts["hangul_vs_dense_2k"],
        point_limit=BASELINE_NONINFERIORITY_BPB,
        upper_limit=BASELINE_NONINFERIORITY_BPB,
    )
    generic = _contrast_pass(
        contrasts["hangul_vs_generic"],
        point_limit=-MINIMUM_KOREAN_ADVANTAGE_BPB,
        upper_limit=0.0,
    )
    shuffled = _contrast_pass(
        contrasts["hangul_vs_shuffled"],
        point_limit=-MINIMUM_KOREAN_ADVANTAGE_BPB,
        upper_limit=0.0,
    )
    low_rank = _contrast_pass(
        contrasts["hangul_vs_low_rank"],
        point_limit=CONTROL_NONINFERIORITY_BPB,
        upper_limit=CONTROL_NONINFERIORITY_BPB,
    )
    generic_baseline = _contrast_pass(
        contrasts["generic_vs_dense_2k"],
        point_limit=BASELINE_NONINFERIORITY_BPB,
        upper_limit=BASELINE_NONINFERIORITY_BPB,
    )
    overall = baseline and generic and shuffled and low_rank
    if overall:
        status = "korean_compositional_quality_opportunity_pass"
    elif generic_baseline:
        status = "generic_factorization_only_requires_novelty_reassessment"
    else:
        status = "compositional_quality_branch_stopped"
    return {
        "status": status,
        "primary_candidate": PRIMARY_CANDIDATE_ROLE,
        "primary_candidate_quality_noninferior": baseline,
        "hangul_beats_generic": generic,
        "hangul_beats_shuffled": shuffled,
        "hangul_noninferior_to_low_rank": low_rank,
        "generic_quality_noninferior": generic_baseline,
        "overall_pass": overall,
        "trained_actual_inference_authorized": overall,
        "no_result_dependent_candidate_fallback": True,
    }


def validate_quality_specs() -> None:
    if set(QUALITY_SPECS) != set(QUALITY_ROLES) or len(QUALITY_ROLES) != 6:
        raise ValueError("compositional quality role grid differs")
    if MODEL_SEED == SYSTEMS_MODEL_SEED:
        raise ValueError("quality and systems model seeds must be distinct")
    for role, spec in QUALITY_SPECS.items():
        if spec.expected_parameters <= 0 or EFFECTIVE_BATCH_SIZE % spec.train_microbatch:
            raise ValueError(f"compositional quality spec differs: {role}")


validate_quality_specs()
