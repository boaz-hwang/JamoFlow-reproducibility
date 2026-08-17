"""Deterministic parameter-matched token Transformer frontier helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from scalar_runtime_core import build_bpe_model


PARAMETER_TARGET = 19_596_096
PARAMETER_RELATIVE_TOLERANCE = 0.005
VOCABULARY_SIZES = (2_048, 4_096, 8_192, 16_000, 32_000, 64_000)
DEPTHS = (8, 12, 16)
MAXIMUM_POSITIONS = 512


@dataclass(frozen=True, slots=True)
class FrontierModelSpec:
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    key_value_heads: int
    maximum_positions: int
    expected_parameters: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def analytical_parameters(
    vocabulary_size: int,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
) -> int:
    if min(vocabulary_size, hidden_size, intermediate_size, layers) <= 0:
        raise ValueError("frontier dimensions must be positive")
    embedding = vocabulary_size * hidden_size
    attention = 4 * hidden_size * hidden_size
    feed_forward = 3 * hidden_size * intermediate_size
    layer_norms = 2 * hidden_size
    final_norm = hidden_size
    return embedding + layers * (attention + feed_forward + layer_norms) + final_norm


def derive_frontier_spec(vocabulary_size: int, layers: int) -> FrontierModelSpec:
    """Choose geometry by a result-blind exact integer grid.

    Parameter proximity is primary.  Exact ties prefer an FFN ratio near 3.5,
    a head dimension near 64, then a wider residual stream.  Runtime and model
    quality are deliberately absent from this search.
    """

    if vocabulary_size not in VOCABULARY_SIZES or layers not in DEPTHS:
        raise ValueError("unknown frontier vocabulary/depth")
    candidates: list[tuple[tuple[float | int, ...], FrontierModelSpec]] = []
    for hidden_size in range(128, 641, 16):
        for intermediate_size in range(32, int(4.5 * hidden_size) + 1, 32):
            ratio = intermediate_size / hidden_size
            if not 2.5 <= ratio <= 4.5:
                continue
            parameters = analytical_parameters(
                vocabulary_size,
                hidden_size,
                intermediate_size,
                layers,
            )
            for attention_heads in range(2, 21):
                if hidden_size % attention_heads:
                    continue
                head_dimension = hidden_size // attention_heads
                if not 24 <= head_dimension <= 80:
                    continue
                spec = FrontierModelSpec(
                    vocabulary_size=vocabulary_size,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    layers=layers,
                    attention_heads=attention_heads,
                    key_value_heads=attention_heads,
                    maximum_positions=MAXIMUM_POSITIONS,
                    expected_parameters=parameters,
                )
                rank = (
                    abs(parameters - PARAMETER_TARGET),
                    abs(ratio - 3.5),
                    abs(head_dimension - 64),
                    -hidden_size,
                    intermediate_size,
                    attention_heads,
                )
                candidates.append((rank, spec))
    if not candidates:
        raise RuntimeError("frontier parameter grid is empty")
    chosen = min(candidates, key=lambda item: item[0])[1]
    if abs(chosen.expected_parameters / PARAMETER_TARGET - 1.0) > PARAMETER_RELATIVE_TOLERANCE:
        raise RuntimeError("frontier parameter grid cannot match the target")
    return chosen


def role_name(vocabulary_size: int, layers: int) -> str:
    if vocabulary_size not in VOCABULARY_SIZES or layers not in DEPTHS:
        raise ValueError("unknown frontier role")
    return f"byte_bpe_v{vocabulary_size}_d{layers}"


def parse_role(role: str) -> tuple[int, int]:
    prefix = "byte_bpe_v"
    if not role.startswith(prefix) or "_d" not in role:
        raise ValueError("malformed frontier role")
    vocabulary, depth = role[len(prefix) :].split("_d", maxsplit=1)
    values = int(vocabulary), int(depth)
    if values[0] not in VOCABULARY_SIZES or values[1] not in DEPTHS:
        raise ValueError("frontier role values differ")
    if role_name(*values) != role:
        raise ValueError("frontier role is not canonical")
    return values


FRONTIER_SPECS = {
    role_name(vocabulary_size, layers): derive_frontier_spec(vocabulary_size, layers)
    for vocabulary_size in VOCABULARY_SIZES
    for layers in DEPTHS
}
RUNTIME_ROLES = tuple(FRONTIER_SPECS)


def build_frontier_model(role: str, *, seed: int) -> Any:
    spec = FRONTIER_SPECS[role]
    return build_bpe_model(spec.to_dict(), seed=seed)


def balanced_role_schedule(case_index: int, repetition: int) -> tuple[str, ...]:
    if case_index < 0 or repetition < 0:
        raise ValueError("schedule indices must be nonnegative")
    count = len(RUNTIME_ROLES)
    trial = case_index * 3 + repetition
    shift = trial % count
    values = RUNTIME_ROLES if (trial // count) % 2 == 0 else tuple(reversed(RUNTIME_ROLES))
    return values[shift:] + values[:shift]
