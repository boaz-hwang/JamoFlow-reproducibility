"""Small, explicit training loops for the Phase 1 BLT experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable

import numpy as np

from .neural_patching import entropy_from_logits


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    batch_size: int = 32
    router_batch_size: int = 64
    evaluation_batch_size: int = 64
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip: float = 1.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_OPTIMIZATION_SPEC = OptimizationSpec()


@dataclass(frozen=True, slots=True)
class TrainSummary:
    steps: int
    examples: int
    predicted_bytes: int
    mean_loss_nats: float
    final_loss_nats: float
    elapsed_seconds: float
    bytes_per_second: float
    final_learning_rate: float
    history: tuple[dict[str, float | int], ...]

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    examples: int
    predicted_bytes: int
    nll_nats: float
    bpb: float
    elapsed_seconds: float
    bytes_per_second: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def cosine_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    maximum: float,
    minimum: float,
) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return maximum * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return minimum + 0.5 * (maximum - minimum) * (1 + math.cos(math.pi * progress))


def shuffled_indices(example_count: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(example_count)


def _batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _input_batch(values: np.ndarray, indices: np.ndarray, device: str):
    import torch

    return torch.from_numpy(values[indices].astype(np.int64, copy=False)).to(device)


def _patch_batch(values: np.ndarray, indices: np.ndarray, device: str):
    import torch

    selected = values[indices]
    if selected.ndim != 2:
        raise ValueError("patch lengths must be a two-dimensional matrix")
    used_columns = np.flatnonzero(np.any(selected != 0, axis=0))
    if not used_columns.size:
        raise ValueError("patch batch contains no positive lengths")
    # Phase 2 variable-rate matrices are right-zero-padded globally.  Trimming
    # here makes the implemented global width depend on the current batch max,
    # rather than silently charging every batch for the corpus-wide maximum.
    selected = selected[:, : int(used_columns[-1]) + 1]
    return torch.from_numpy(selected.astype(np.int64, copy=False)).to(device)


def _optimizer(model: Any, spec: OptimizationSpec):
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
    )


def train_router(
    model: Any,
    inputs: np.ndarray,
    order: np.ndarray,
    device: str,
    spec: OptimizationSpec = DEFAULT_OPTIMIZATION_SPEC,
    log_every: int = 100,
) -> TrainSummary:
    import torch
    import torch.nn.functional as F

    model.to(device)
    model.train()
    optimizer = _optimizer(model, spec)
    total_steps = math.ceil(len(order) / spec.router_batch_size)
    loss_sum = 0.0
    target_count = 0
    final_loss = math.nan
    final_lr = 0.0
    history: list[dict[str, float | int]] = []
    synchronize(device)
    started = time.perf_counter()

    for step, batch_indices in enumerate(
        _batches(order, spec.router_batch_size)
    ):
        learning_rate = cosine_learning_rate(
            step,
            total_steps,
            spec.warmup_steps,
            spec.learning_rate,
            spec.minimum_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        batch = _input_batch(inputs, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        _, _, logits = model(batch, patch_size=None, use_cache=False)
        shifted_logits = logits[:, :-1, :].contiguous()
        targets = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            targets.view(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
        optimizer.step()

        count = targets.numel()
        final_loss = float(loss.detach().cpu())
        loss_sum += final_loss * count
        target_count += count
        final_lr = learning_rate
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == total_steps:
            history.append(
                {
                    "step": step + 1,
                    "loss_nats": final_loss,
                    "learning_rate": learning_rate,
                }
            )

    synchronize(device)
    elapsed = time.perf_counter() - started
    return TrainSummary(
        steps=total_steps,
        examples=len(order),
        predicted_bytes=target_count,
        mean_loss_nats=loss_sum / target_count,
        final_loss_nats=final_loss,
        elapsed_seconds=elapsed,
        bytes_per_second=target_count / elapsed,
        final_learning_rate=final_lr,
        history=tuple(history),
    )


def train_main_model(
    model: Any,
    inputs: np.ndarray,
    patch_lengths: np.ndarray,
    order: np.ndarray,
    device: str,
    spec: OptimizationSpec = DEFAULT_OPTIMIZATION_SPEC,
    log_every: int = 100,
) -> TrainSummary:
    import torch

    if len(inputs) != len(patch_lengths):
        raise ValueError("inputs and patch lengths must have equal examples")
    model.to(device)
    model.train()
    optimizer = _optimizer(model, spec)
    total_steps = math.ceil(len(order) / spec.batch_size)
    loss_sum = 0.0
    target_count = 0
    final_loss = math.nan
    final_lr = 0.0
    history: list[dict[str, float | int]] = []
    synchronize(device)
    started = time.perf_counter()

    for step, batch_indices in enumerate(_batches(order, spec.batch_size)):
        learning_rate = cosine_learning_rate(
            step,
            total_steps,
            spec.warmup_steps,
            spec.learning_rate,
            spec.minimum_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        batch = _input_batch(inputs, batch_indices, device)
        patches = _patch_batch(patch_lengths, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch,
            patch_lengths=patches,
            labels=batch,
            use_cache=False,
        )
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
        optimizer.step()

        count = batch.shape[0] * (batch.shape[1] - 1)
        final_loss = float(loss.detach().cpu())
        loss_sum += final_loss * count
        target_count += count
        final_lr = learning_rate
        if step == 0 or (step + 1) % log_every == 0 or step + 1 == total_steps:
            history.append(
                {
                    "step": step + 1,
                    "loss_nats": final_loss,
                    "learning_rate": learning_rate,
                }
            )

    synchronize(device)
    elapsed = time.perf_counter() - started
    return TrainSummary(
        steps=total_steps,
        examples=len(order),
        predicted_bytes=target_count,
        mean_loss_nats=loss_sum / target_count,
        final_loss_nats=final_loss,
        elapsed_seconds=elapsed,
        bytes_per_second=target_count / elapsed,
        final_learning_rate=final_lr,
        history=tuple(history),
    )


def evaluate_main_model(
    model: Any,
    inputs: np.ndarray,
    patch_lengths: np.ndarray,
    device: str,
    batch_size: int = 64,
    return_sequence_nll: bool = False,
) -> tuple[EvaluationSummary, np.ndarray | None]:
    import torch
    import torch.nn.functional as F

    if len(inputs) != len(patch_lengths):
        raise ValueError("inputs and patch lengths must have equal examples")
    model.to(device)
    model.eval()
    order = np.arange(len(inputs))
    total_nll = 0.0
    total_targets = 0
    sequence_nll: list[np.ndarray] = []
    synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_indices in _batches(order, batch_size):
            batch = _input_batch(inputs, batch_indices, device)
            patches = _patch_batch(patch_lengths, batch_indices, device)
            logits = model(
                input_ids=batch,
                patch_lengths=patches,
                use_cache=False,
            ).logits
            token_losses = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                batch[:, 1:],
                reduction="none",
            )
            local_nll = token_losses.sum(dim=1)
            total_nll += float(local_nll.sum().cpu())
            total_targets += token_losses.numel()
            if return_sequence_nll:
                sequence_nll.append(local_nll.float().cpu().numpy())

    synchronize(device)
    elapsed = time.perf_counter() - started
    summary = EvaluationSummary(
        examples=len(inputs),
        predicted_bytes=total_targets,
        nll_nats=total_nll / total_targets,
        bpb=total_nll / total_targets / math.log(2),
        elapsed_seconds=elapsed,
        bytes_per_second=total_targets / elapsed,
    )
    per_sequence = np.concatenate(sequence_nll) if return_sequence_nll else None
    return summary, per_sequence


def evaluate_main_model_masked(
    model: Any,
    inputs: np.ndarray,
    patch_lengths: np.ndarray,
    target_mask: np.ndarray,
    device: str,
    batch_size: int = 64,
    return_sequence_nll: bool = False,
) -> tuple[EvaluationSummary, np.ndarray | None]:
    """Evaluate only selected next-byte targets in otherwise complete rows."""

    import torch
    import torch.nn.functional as F

    if len(inputs) != len(patch_lengths):
        raise ValueError("inputs and patch lengths must have equal examples")
    expected_mask_shape = (len(inputs), inputs.shape[1] - 1)
    if target_mask.shape != expected_mask_shape:
        raise ValueError(
            f"target mask must have shape {expected_mask_shape}, "
            f"not {target_mask.shape}"
        )
    if not np.issubdtype(target_mask.dtype, np.bool_) and not np.all(
        (target_mask == 0) | (target_mask == 1)
    ):
        raise ValueError("target mask must be boolean or binary")
    expected_targets = int(target_mask.sum())
    if expected_targets <= 0:
        raise ValueError("target mask must select at least one byte")

    model.to(device)
    model.eval()
    order = np.arange(len(inputs))
    total_nll = 0.0
    total_targets = 0
    sequence_nll: list[np.ndarray] = []
    synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_indices in _batches(order, batch_size):
            batch = _input_batch(inputs, batch_indices, device)
            patches = _patch_batch(patch_lengths, batch_indices, device)
            selected = torch.from_numpy(
                target_mask[batch_indices].astype(np.bool_, copy=False)
            ).to(device)
            logits = model(
                input_ids=batch,
                patch_lengths=patches,
                use_cache=False,
            ).logits
            token_losses = F.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                batch[:, 1:],
                reduction="none",
            )
            local_nll = (token_losses * selected).sum(dim=1)
            total_nll += float(local_nll.sum().cpu())
            total_targets += int(selected.sum().cpu())
            if return_sequence_nll:
                sequence_nll.append(local_nll.float().cpu().numpy())

    synchronize(device)
    elapsed = time.perf_counter() - started
    if total_targets != expected_targets:
        raise AssertionError("masked target accounting changed during evaluation")
    summary = EvaluationSummary(
        examples=len(inputs),
        predicted_bytes=total_targets,
        nll_nats=total_nll / total_targets,
        bpb=total_nll / total_targets / math.log(2),
        elapsed_seconds=elapsed,
        bytes_per_second=total_targets / elapsed,
    )
    per_sequence = np.concatenate(sequence_nll) if return_sequence_nll else None
    return summary, per_sequence


def evaluate_router(
    model: Any,
    inputs: np.ndarray,
    device: str,
    batch_size: int = 128,
) -> EvaluationSummary:
    import torch
    import torch.nn.functional as F

    model.to(device)
    model.eval()
    order = np.arange(len(inputs))
    total_nll = 0.0
    total_targets = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_indices in _batches(order, batch_size):
            batch = _input_batch(inputs, batch_indices, device)
            _, _, logits = model(batch, patch_size=None, use_cache=False)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss.cpu())
            total_targets += batch.shape[0] * (batch.shape[1] - 1)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return EvaluationSummary(
        examples=len(inputs),
        predicted_bytes=total_targets,
        nll_nats=total_nll / total_targets,
        bpb=total_nll / total_targets / math.log(2),
        elapsed_seconds=elapsed,
        bytes_per_second=total_targets / elapsed,
    )


def router_entropy_scores(
    model: Any,
    inputs: np.ndarray,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Return H(X_t | x_<t) aligned to byte position t."""

    import torch

    model.to(device)
    model.eval()
    scores = np.zeros(inputs.shape, dtype=np.float32)
    order = np.arange(len(inputs))
    with torch.inference_mode():
        for batch_indices in _batches(order, batch_size):
            batch = _input_batch(inputs, batch_indices, device)
            _, _, logits = model(batch, patch_size=None, use_cache=False)
            entropies = entropy_from_logits(logits)
            aligned = torch.zeros_like(entropies)
            aligned[:, 1:] = entropies[:, :-1]
            scores[batch_indices] = aligned.float().cpu().numpy()
    return scores
