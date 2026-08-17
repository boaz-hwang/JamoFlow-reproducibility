"""Research implementation of fixed-budget compositional token vocabularies.

The module keeps a Transformer body's token-indexed trainable budget fixed
while allowing a larger external vocabulary.  Every token embedding is an
additive composition of one row from each independent codebook.  The output
head first projects a hidden state onto all codebook rows and then gathers the
same token-specific components, so it is exactly equivalent to a dense tied
embedding matrix constructed from the codes.

The assignments in this module are deterministic controls for Korean byte-BPE
vocabularies.  They are experimental research objects, not a general-purpose
tokenizer API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Literal, Sequence
import weakref

import numpy as np


AssignmentKind = Literal["generic_unicode", "hangul", "shuffled_hangul"]
CODEBOOK_COUNT = 16
CODEBOOK_SIZE = 128
CODEBOOK_ROWS = CODEBOOK_COUNT * CODEBOOK_SIZE
SURFACE_RADIX = CODEBOOK_SIZE - 1
SURFACE_AUXILIARY_SLOTS = (6, 7, 8, 9, 10, 11)
TOKEN_IDENTITY_SLOTS = (13, 14, 15)
MAXIMUM_SUPPORTED_VOCABULARY = CODEBOOK_SIZE ** len(TOKEN_IDENTITY_SLOTS)
PSEUDO_BYTE_BASE = 0x110000
VECTORIZED_OUTPUT_MAXIMUM_POSITIONS = 4
_SHUFFLE_SEED = 20_260_821


@dataclass(frozen=True, slots=True)
class AssignmentAudit:
    kind: str
    vocabulary_size: int
    codebook_count: int
    codebook_size: int
    unique_code_tuples: int
    strict_utf8_token_count: int
    hangul_surface_token_count: int
    incomplete_or_invalid_utf8_token_count: int
    maximum_piece_bytes: int
    assignment_sha256: str
    slot_unique_counts: tuple[int, ...]
    slot_entropy_bits: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # The plan is validated both before and after a JSON round trip.  Lists
        # are the canonical JSON representation; retaining tuples here makes a
        # freshly loaded plan compare unequal to its deterministic rebuild.
        value["slot_unique_counts"] = list(self.slot_unique_counts)
        value["slot_entropy_bits"] = list(self.slot_entropy_bits)
        return value


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(b"JamoFlow/compositional-token-codes/v1\0")
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes(order="C"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _surface_units(piece: bytes) -> tuple[int, ...]:
    if not isinstance(piece, bytes) or not piece:
        raise ValueError("compositional token pieces must be nonempty bytes")
    try:
        text = piece.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return tuple(PSEUDO_BYTE_BASE + value for value in piece)
    if not text:
        raise ValueError("compositional token text must be nonempty")
    return tuple(ord(character) for character in text)


def _base_digits(value: int, *, radix: int, width: int) -> tuple[int, ...]:
    if not 0 <= value < radix**width:
        raise ValueError("compositional code value exceeds its fixed radix")
    output = []
    current = value
    for _ in range(width):
        output.append(current % radix)
        current //= radix
    return tuple(output)


def _auxiliary_hashes(unit: int, piece: bytes, *, side: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(
        b"JamoFlow/generic-surface-aux/v1\0"
        + side.encode("ascii")
        + unit.to_bytes(4, "big")
        + len(piece).to_bytes(4, "big")
        + piece
    ).digest()
    return tuple(int.from_bytes(digest[index * 2 : index * 2 + 2], "big") % CODEBOOK_SIZE for index in range(3))


def _hangul_components(unit: int) -> tuple[int, int, int] | None:
    if not 0xAC00 <= unit <= 0xD7A3:
        return None
    offset = unit - 0xAC00
    return offset // 588, (offset % 588) // 28, offset % 28


def _base_assignment(token_bytes: Sequence[bytes], kind: AssignmentKind) -> np.ndarray:
    if kind not in ("generic_unicode", "hangul"):
        raise ValueError("base compositional assignment kind differs")
    vocabulary_size = len(token_bytes)
    if not 1 <= vocabulary_size <= MAXIMUM_SUPPORTED_VOCABULARY:
        raise ValueError("compositional vocabulary size differs")
    output = np.empty((vocabulary_size, CODEBOOK_COUNT), dtype=np.int64)
    for token_id, piece in enumerate(token_bytes):
        units = _surface_units(piece)
        first, last = units[0], units[-1]
        output[token_id, 0:3] = _base_digits(first, radix=SURFACE_RADIX, width=3)
        output[token_id, 3:6] = _base_digits(last, radix=SURFACE_RADIX, width=3)
        if kind == "hangul" and (components := _hangul_components(first)) is not None:
            output[token_id, 6:9] = components
        else:
            output[token_id, 6:9] = _auxiliary_hashes(first, piece, side="first")
        if kind == "hangul" and (components := _hangul_components(last)) is not None:
            output[token_id, 9:12] = components
        else:
            output[token_id, 9:12] = _auxiliary_hashes(last, piece, side="last")
        output[token_id, 12] = min(len(piece), CODEBOOK_SIZE - 1)
        output[token_id, 13:16] = _base_digits(
            token_id, radix=CODEBOOK_SIZE, width=3
        )
    return output


def _shuffle_hangul_surface(
    assignment: np.ndarray,
    token_bytes: Sequence[bytes],
) -> np.ndarray:
    output = assignment.copy()
    rng = np.random.default_rng(_SHUFFLE_SEED)
    lengths = np.asarray([len(piece) for piece in token_bytes], dtype=np.int64)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        if len(rows) <= 1:
            continue
        permuted = rows[rng.permutation(len(rows))]
        output[rows[:, None], np.asarray(SURFACE_AUXILIARY_SLOTS)] = assignment[
            permuted[:, None], np.asarray(SURFACE_AUXILIARY_SLOTS)
        ]
    return output


def build_token_code_assignment(
    token_bytes: Sequence[bytes],
    *,
    kind: AssignmentKind,
) -> np.ndarray:
    """Build a deterministic, collision-free token-to-code assignment."""

    pieces = tuple(token_bytes)
    if kind == "shuffled_hangul":
        output = _shuffle_hangul_surface(
            _base_assignment(pieces, "hangul"), pieces
        )
    else:
        output = _base_assignment(pieces, kind)
    if (
        output.dtype != np.int64
        or output.shape != (len(pieces), CODEBOOK_COUNT)
        or np.any(output < 0)
        or np.any(output >= CODEBOOK_SIZE)
        or len(np.unique(output, axis=0)) != len(pieces)
    ):
        raise AssertionError("compositional token code assignment is invalid")
    return output


def audit_token_code_assignment(
    token_bytes: Sequence[bytes],
    assignment: np.ndarray,
    *,
    kind: AssignmentKind,
) -> AssignmentAudit:
    pieces = tuple(token_bytes)
    expected = build_token_code_assignment(pieces, kind=kind)
    if not np.array_equal(assignment, expected):
        raise ValueError("compositional token assignment differs from its definition")
    strict = 0
    hangul = 0
    for piece in pieces:
        try:
            text = piece.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        strict += 1
        hangul += int(any("\uac00" <= character <= "\ud7a3" for character in text))
    unique_counts = []
    entropies = []
    for slot in range(CODEBOOK_COUNT):
        counts = np.bincount(assignment[:, slot], minlength=CODEBOOK_SIZE)
        probabilities = counts[counts > 0].astype(np.float64) / len(assignment)
        unique_counts.append(int(np.count_nonzero(counts)))
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))
    return AssignmentAudit(
        kind=kind,
        vocabulary_size=len(pieces),
        codebook_count=CODEBOOK_COUNT,
        codebook_size=CODEBOOK_SIZE,
        unique_code_tuples=len(np.unique(assignment, axis=0)),
        strict_utf8_token_count=strict,
        hangul_surface_token_count=hangul,
        incomplete_or_invalid_utf8_token_count=len(pieces) - strict,
        maximum_piece_bytes=max(len(piece) for piece in pieces),
        assignment_sha256=array_sha256(assignment),
        slot_unique_counts=tuple(unique_counts),
        slot_entropy_bits=tuple(entropies),
    )


def low_rank_for_budget(vocabulary_size: int, hidden_size: int, budget: int) -> int:
    if min(vocabulary_size, hidden_size, budget) <= 0:
        raise ValueError("low-rank budget coordinates differ")
    exact = budget / (vocabulary_size + hidden_size)
    candidates = {max(1, math.floor(exact)), max(1, math.ceil(exact))}
    return min(
        candidates,
        key=lambda rank: (
            abs(rank * (vocabulary_size + hidden_size) - budget),
            -rank,
        ),
    )


def _require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    return torch, nn, functional


class CompositionalVocabulary:
    """Factory namespace; the returned object is a regular ``nn.Module``."""

    @staticmethod
    def build(initial_weight: Any, assignment: Any):
        torch, nn, functional = _require_torch()

        class _Module(nn.Module):
            def __init__(self, weight, codes) -> None:
                super().__init__()
                if (
                    weight.ndim != 3
                    or tuple(weight.shape[:2]) != (CODEBOOK_COUNT, CODEBOOK_SIZE)
                    or codes.ndim != 2
                    or codes.shape[1] != CODEBOOK_COUNT
                    or codes.dtype != torch.long
                ):
                    raise ValueError("compositional vocabulary tensors differ")
                self.weight = nn.Parameter(weight.detach().clone())
                offsets = torch.arange(CODEBOOK_COUNT, dtype=torch.long) * CODEBOOK_SIZE
                code_indices = codes.detach().clone() + offsets.unsqueeze(0)
                self.register_buffer("code_indices", code_indices, persistent=True)
                self.scale = CODEBOOK_COUNT**-0.5

            @property
            def flat_gather_indices(self):
                # Derive the view after every device transfer.  Registering the
                # flattened alias as a second buffer makes ``Module.to`` copy the
                # same V×M index table twice on MPS.
                return self.code_indices.reshape(-1)

            @property
            def vocabulary_size(self) -> int:
                return int(self.code_indices.shape[0])

            @property
            def hidden_size(self) -> int:
                return int(self.weight.shape[2])

            def dense_weight(self):
                rows = functional.embedding(
                    self.flat_gather_indices,
                    self.weight.reshape(CODEBOOK_ROWS, self.hidden_size),
                ).reshape(self.vocabulary_size, CODEBOOK_COUNT, self.hidden_size)
                return rows.sum(dim=1) * self.scale

            def embed(self, token_ids):
                if token_ids.dtype != torch.long:
                    raise ValueError("compositional input IDs must be int64")
                indices = self.code_indices[token_ids]
                rows = functional.embedding(
                    indices,
                    self.weight.reshape(CODEBOOK_ROWS, self.hidden_size),
                )
                return rows.sum(dim=-2) * self.scale

            def project(self, hidden):
                if hidden.shape[-1] != self.hidden_size:
                    raise ValueError("compositional hidden dimension differs")
                basis = functional.linear(
                    hidden, self.weight.reshape(CODEBOOK_ROWS, self.hidden_size)
                )
                positions = hidden.numel() // self.hidden_size
                if positions <= VECTORIZED_OUTPUT_MAXIMUM_POSITIONS:
                    selected = basis.index_select(-1, self.flat_gather_indices)
                    return selected.reshape(
                        *hidden.shape[:-1], self.vocabulary_size, CODEBOOK_COUNT
                    ).sum(dim=-1) * self.scale
                output = None
                for slot in range(CODEBOOK_COUNT):
                    values = basis.index_select(-1, self.code_indices[:, slot])
                    output = values if output is None else output + values
                if output is None:
                    raise AssertionError("compositional output has no codebooks")
                return output * self.scale

        return _Module(initial_weight, assignment)


class LowRankVocabulary:
    """Factory namespace for an exact tied low-rank embedding/output control."""

    @staticmethod
    def build(token_factors: Any, projection: Any):
        torch, nn, functional = _require_torch()

        class _Module(nn.Module):
            def __init__(self, factors, project) -> None:
                super().__init__()
                if (
                    factors.ndim != 2
                    or project.ndim != 2
                    or factors.shape[1] != project.shape[0]
                ):
                    raise ValueError("low-rank vocabulary tensors differ")
                self.token_factors = nn.Parameter(factors.detach().clone())
                self.projection = nn.Parameter(project.detach().clone())

            @property
            def vocabulary_size(self) -> int:
                return int(self.token_factors.shape[0])

            @property
            def hidden_size(self) -> int:
                return int(self.projection.shape[1])

            def dense_weight(self):
                return self.token_factors @ self.projection

            def embed(self, token_ids):
                if token_ids.dtype != torch.long:
                    raise ValueError("low-rank input IDs must be int64")
                return functional.embedding(token_ids, self.token_factors) @ self.projection

            def project(self, hidden):
                return (hidden @ self.projection.transpose(0, 1)) @ self.token_factors.transpose(0, 1)

        return _Module(token_factors, projection)


def _adapter(module: Any, method_name: str):
    _, nn, _ = _require_torch()

    class _Adapter(nn.Module):
        def __init__(self, target) -> None:
            super().__init__()
            object.__setattr__(self, "_target_reference", weakref.ref(target))

        @property
        def target(self):
            value = self._target_reference()
            if value is None:
                raise RuntimeError("factorized vocabulary target no longer exists")
            return value

        @property
        def weight(self):
            return self.target.dense_weight()

        def forward(self, values):
            return getattr(self.target, method_name)(values)

    return _Adapter(module)


def install_factorized_vocabulary(model: Any, vocabulary: Any) -> Any:
    """Replace a Llama causal LM's tied dense vocabulary with ``vocabulary``."""

    torch, _, _ = _require_torch()
    if (
        not hasattr(model, "model")
        or not hasattr(model, "lm_head")
        or vocabulary.hidden_size != int(model.config.hidden_size)
        or vocabulary.vocabulary_size <= 0
    ):
        raise ValueError("factorized Llama installation coordinates differ")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vocabulary.to(device=device, dtype=dtype)
    model.model.embed_tokens = _adapter(vocabulary, "embed")
    model.lm_head = _adapter(vocabulary, "project")
    model.factorized_vocabulary = vocabulary
    model.config.vocab_size = vocabulary.vocabulary_size
    model.config.tie_word_embeddings = False
    model.vocab_size = vocabulary.vocabulary_size
    if any(parameter.device != device for parameter in vocabulary.parameters()):
        raise AssertionError("factorized vocabulary device installation failed")
    if not all(parameter.dtype == dtype for parameter in vocabulary.parameters()):
        raise AssertionError("factorized vocabulary dtype installation failed")
    if model.factorized_vocabulary is not vocabulary:
        raise AssertionError("factorized vocabulary installation changed identity")
    return model
