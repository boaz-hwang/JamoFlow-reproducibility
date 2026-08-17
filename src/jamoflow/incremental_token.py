"""Verified batch-one incremental runtime for the publication BPE control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TokenRuntimeDiagnostics:
    observed_tokens: int
    cached_tokens: int
    maximum_positions: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class IncrementalTokenDecoder:
    """Stateful cached decoder whose logits align with full-prefix inference."""

    def __init__(self, model) -> None:
        import torch

        self.model = model
        self.model.eval()
        try:
            self.device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("incremental token model has no parameters") from error
        self.maximum_positions = int(model.config.max_position_embeddings)
        if self.maximum_positions <= 0:
            raise ValueError("incremental token model has invalid context capacity")
        self._torch = torch
        self._cache = None
        self._observed_tokens = 0

    def reset(self) -> None:
        self._cache = None
        self._observed_tokens = 0

    @property
    def diagnostics(self) -> TokenRuntimeDiagnostics:
        cached = (
            int(self._cache.get_seq_length())
            if self._cache is not None and hasattr(self._cache, "get_seq_length")
            else 0
        )
        return TokenRuntimeDiagnostics(
            observed_tokens=self._observed_tokens,
            cached_tokens=cached,
            maximum_positions=self.maximum_positions,
        )

    def _ids_tensor(self, token_ids: Sequence[int]):
        if not token_ids:
            raise ValueError("incremental token input cannot be empty")
        vocabulary_size = int(self.model.config.vocab_size)
        values = tuple(int(value) for value in token_ids)
        if any(value < 0 or value >= vocabulary_size for value in values):
            raise ValueError("incremental token id is outside the vocabulary")
        return self._torch.tensor(
            [values],
            dtype=self._torch.long,
            device=self.device,
        )

    def prefill_parallel(self, token_ids: Sequence[int]):
        """Reset state and prefill all prepared token ids in one cached forward."""

        values = tuple(token_ids)
        if len(values) > self.maximum_positions:
            raise ValueError("token prefill exceeds model context")
        inputs = self._ids_tensor(values)
        self.reset()
        with self._torch.inference_mode():
            outputs = self.model(input_ids=inputs, use_cache=True)
        self._cache = outputs.past_key_values
        self._observed_tokens = len(values)
        self._validate_cache_length()
        return outputs.logits[:, -1, :]

    def consume(self, token_id: int):
        """Append one prepared token and return logits for its next token."""

        if self._cache is None or self._observed_tokens <= 0:
            raise RuntimeError("token runtime must be prefilled before consume")
        if self._observed_tokens >= self.maximum_positions:
            raise ValueError("token consume exceeds model context")
        inputs = self._ids_tensor((token_id,))
        with self._torch.inference_mode():
            outputs = self.model(
                input_ids=inputs,
                past_key_values=self._cache,
                use_cache=True,
            )
        self._cache = outputs.past_key_values
        self._observed_tokens += 1
        self._validate_cache_length()
        return outputs.logits[:, -1, :]

    def _validate_cache_length(self) -> None:
        diagnostics = self.diagnostics
        if diagnostics.cached_tokens != diagnostics.observed_tokens:
            raise RuntimeError("token KV cache length differs from observed tokens")


def verify_token_incremental_equivalence(
    model,
    token_ids: Sequence[int],
    *,
    relative_tolerance: float = 2e-5,
    absolute_tolerance: float = 2e-5,
) -> dict[str, int | float | bool]:
    """Compare sequential cache and parallel prefill to every full prefix."""

    import torch

    values = tuple(int(value) for value in token_ids)
    if not values or relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("equivalence input or tolerance is invalid")
    model.eval()
    device = next(model.parameters()).device
    runtime = IncrementalTokenDecoder(model)
    maximum_absolute_error = 0.0
    all_argmax_equal = True
    sequential_logits = None
    for index, token_id in enumerate(values):
        if index == 0:
            sequential_logits = runtime.prefill_parallel((token_id,))
        else:
            sequential_logits = runtime.consume(token_id)
        prefix = torch.tensor(
            [values[: index + 1]],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            full_logits = model(input_ids=prefix, use_cache=False).logits[:, -1, :]
        difference = float(torch.max(torch.abs(sequential_logits - full_logits)).item())
        maximum_absolute_error = max(maximum_absolute_error, difference)
        if not torch.allclose(
            sequential_logits,
            full_logits,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        ):
            raise RuntimeError("cached token logits differ from full-prefix logits")
        all_argmax_equal = bool(
            all_argmax_equal
            and torch.equal(
                torch.argmax(sequential_logits, dim=-1),
                torch.argmax(full_logits, dim=-1),
            )
        )
    parallel = IncrementalTokenDecoder(model)
    parallel_logits = parallel.prefill_parallel(values)
    assert sequential_logits is not None
    if not torch.allclose(
        parallel_logits,
        sequential_logits,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        raise RuntimeError("parallel token prefill differs from sequential cache")
    return {
        "prefixes_checked": len(values),
        "observed_tokens": runtime.diagnostics.observed_tokens,
        "cached_tokens": runtime.diagnostics.cached_tokens,
        "parallel_cached_tokens": parallel.diagnostics.cached_tokens,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "maximum_absolute_error": maximum_absolute_error,
        "all_argmax_equal": all_argmax_equal,
        "overall_pass": True,
    }
