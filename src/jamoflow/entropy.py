"""A small, reproducible byte n-gram entropy proxy."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PositionScore:
    entropy_bits: float
    surprisal_bits: float
    probability: float
    context_order: int


class ByteNGramModel:
    """Additively smoothed byte n-gram model with suffix backoff.

    This model is a Phase 0 measurement proxy. It is intentionally small and
    deterministic; it is not presented as a replacement for BLT's learned
    entropy predictor.
    """

    vocabulary_size = 256

    def __init__(self, order: int = 4, alpha: float = 0.1) -> None:
        if order < 0:
            raise ValueError("order must be non-negative")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self._counts: list[dict[tuple[int, ...], Counter[int]]] = [
            defaultdict(Counter) for _ in range(order + 1)
        ]
        self.training_bytes = 0

    def fit_bytes(self, data: bytes) -> None:
        for index, value in enumerate(data):
            maximum = min(self.order, index)
            for context_order in range(maximum + 1):
                context = tuple(data[index - context_order : index])
                self._counts[context_order][context][value] += 1
            self.training_bytes += 1

    def fit(self, sequences: Iterable[bytes]) -> "ByteNGramModel":
        for data in sequences:
            self.fit_bytes(data)
        if self.training_bytes == 0:
            raise ValueError("cannot fit an entropy model without bytes")
        return self

    def _distribution_counts(self, prefix: bytes) -> tuple[Counter[int], int]:
        maximum = min(self.order, len(prefix))
        for context_order in range(maximum, -1, -1):
            context = tuple(prefix[-context_order:]) if context_order else ()
            counts = self._counts[context_order].get(context)
            if counts:
                return counts, context_order
        raise RuntimeError("entropy model has not been fitted")

    def probability(self, value: int, prefix: bytes) -> tuple[float, int]:
        counts, context_order = self._distribution_counts(prefix)
        total = sum(counts.values())
        denominator = total + self.alpha * self.vocabulary_size
        probability = (counts.get(value, 0) + self.alpha) / denominator
        return probability, context_order

    def predictive_entropy(self, prefix: bytes) -> tuple[float, int]:
        counts, context_order = self._distribution_counts(prefix)
        total = sum(counts.values())
        denominator = total + self.alpha * self.vocabulary_size

        entropy = 0.0
        for count in counts.values():
            probability = (count + self.alpha) / denominator
            entropy -= probability * math.log2(probability)
        unseen = self.vocabulary_size - len(counts)
        if unseen:
            probability = self.alpha / denominator
            entropy -= unseen * probability * math.log2(probability)
        return entropy, context_order

    def score(self, data: bytes) -> list[PositionScore]:
        scores: list[PositionScore] = []
        for index, value in enumerate(data):
            prefix = data[max(0, index - self.order) : index]
            entropy, entropy_order = self.predictive_entropy(prefix)
            probability, probability_order = self.probability(value, prefix)
            scores.append(
                PositionScore(
                    entropy_bits=entropy,
                    surprisal_bits=-math.log2(probability),
                    probability=probability,
                    context_order=min(entropy_order, probability_order),
                )
            )
        return scores

