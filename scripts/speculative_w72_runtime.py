"""Exact greedy W72 generation with a two-byte Hangul continuation draft."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_runtime_v5 import state_dict_sha256
from jamoflow.utf8 import (
    STRICT_UTF8_INITIAL_STATE,
    StrictUtf8State,
    advance_strict_utf8,
)
from scripts.hangul_draft_acceptance_core import (
    CONTINUATION_CARDINALITY,
    DeviceHangulTables,
    HANGUL_TABLES,
    build_head,
    pair_bytes,
    propose_pairs,
    trainable_parameter_count,
)
from scripts.incremental_block_kernel import IncrementalBlockBltDecoder


@dataclass(slots=True)
class SpeculativeCounters:
    emitted_bytes: int = 0
    sequential_target_calls: int = 0
    target_block_calls: int = 0
    draft_head_calls: int = 0
    first_draft_accepts: int = 0
    complete_pair_accepts: int = 0
    first_mismatches: int = 0
    second_mismatches: int = 0
    cropped_speculative_bytes: int = 0
    correction_bytes: int = 0
    bonus_bytes: int = 0
    retry_block_calls: int = 0
    retry_third_accepts: int = 0
    retry_third_mismatches: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    generated: bytes
    counters: Mapping[str, int]
    diagnostics: Any


class DecoderHiddenCapture:
    """Capture the normalized local-decoder row that produced current logits."""

    def __init__(self, model: Any) -> None:
        self.latest: torch.Tensor | None = None

        def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.latest = output[:, -1, :]

        self._handle = model.model.local_decoder.norm.register_forward_hook(capture)

    def require(self) -> torch.Tensor:
        if self.latest is None or self.latest.ndim != 2 or self.latest.shape[0] != 1:
            raise RuntimeError("speculative decoder hidden state is unavailable")
        return self.latest

    def close(self) -> None:
        self._handle.remove()


class IndependentProposalEngine:
    """Exact lower-work proposal path for the frozen independent head."""

    def __init__(self, head: Any, device: str) -> None:
        self.head = head
        self.allowed_pairs = tuple(
            torch.from_numpy(
                HANGUL_TABLES.pair[HANGUL_TABLES.lead == lead].copy()
            ).to(device)
            for lead in range(4)
        )
        if any(not len(values) for values in self.allowed_pairs):
            raise AssertionError("independent proposal table is empty")

    def propose_with_context(
        self,
        hidden: torch.Tensor,
        lead_value: int,
    ) -> tuple[int, torch.Tensor]:
        lead = torch.tensor([lead_value], dtype=torch.long, device=hidden.device)
        second, third = self.head(hidden.float(), lead)
        pairs = self.allowed_pairs[lead_value]
        scores = second[:, pairs // CONTINUATION_CARDINALITY] + third[
            :, pairs % CONTINUATION_CARDINALITY
        ]
        return int(pairs[scores.argmax(dim=-1)].item()), third

    def propose(self, hidden: torch.Tensor, lead_value: int) -> int:
        pair, _ = self.propose_with_context(hidden, lead_value)
        return pair

    def retry_third(
        self,
        third_logits: torch.Tensor,
        lead_value: int,
        corrected_second: int,
    ) -> int | None:
        pairs = self.allowed_pairs[lead_value]
        matching = pairs[
            pairs // CONTINUATION_CARDINALITY == corrected_second - 0x80
        ]
        if not len(matching):
            return None
        third_indices = matching % CONTINUATION_CARDINALITY
        selected = third_indices[third_logits[:, third_indices].argmax(dim=-1)]
        return 0x80 + int(selected.item())


def load_independent_head(
    path: Path,
    *,
    device: str,
    expected_artifact_sha256: str,
    expected_state_sha256: str,
    expected_seed: int,
    expected_plan_sha256: str,
) -> Any:
    if hash_file(path) != expected_artifact_sha256:
        raise ValueError("speculative head artifact differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(payload)
        != {
            "architecture",
            "head_seed",
            "parameter_count",
            "plan_artifact_sha256",
            "state_dict",
            "training_loss_by_epoch",
        }
        or payload["architecture"] != "generic_independent_utf8"
        or int(payload["head_seed"]) != expected_seed
        or payload["plan_artifact_sha256"] != expected_plan_sha256
    ):
        raise ValueError("speculative head payload differs")
    model = build_head(payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if (
        trainable_parameter_count(model) != int(payload["parameter_count"])
        or state_dict_sha256(model) != expected_state_sha256
    ):
        raise ValueError("speculative head state differs")
    return model.to(device).eval()


def _masked_greedy(
    logits: torch.Tensor,
    state: StrictUtf8State,
    masks: Mapping[StrictUtf8State, torch.Tensor],
) -> int:
    allowed = masks[state]
    return int(logits.masked_fill(~allowed, -torch.inf).argmax(dim=-1).item())


def _append(
    output: bytearray,
    state: StrictUtf8State,
    value: int,
    *,
    minimum_output_bytes: int,
    maximum_output_bytes: int,
) -> tuple[StrictUtf8State, bool]:
    output.append(value)
    advanced = advance_strict_utf8(state, value)
    if not advanced.valid:
        raise AssertionError("speculative verifier emitted invalid UTF-8")
    if len(output) > maximum_output_bytes:
        raise AssertionError("speculative output exceeded its UTF-8 overshoot bound")
    stop = len(output) >= minimum_output_bytes and advanced.at_codepoint_boundary
    if len(output) == maximum_output_bytes and not stop:
        raise AssertionError("speculative output did not close at its maximum bound")
    return advanced, stop


def generate_baseline(
    runtime: Any,
    prompt: bytes,
    masks: Mapping[StrictUtf8State, torch.Tensor],
    *,
    minimum_output_bytes: int,
    maximum_output_bytes: int,
) -> GenerationTrace:
    capture = DecoderHiddenCapture(runtime.model)
    counters = SpeculativeCounters()
    try:
        logits = runtime.prefill_parallel(prompt)
        capture.require()
        state = STRICT_UTF8_INITIAL_STATE
        generated = bytearray()
        while True:
            value = _masked_greedy(logits, state, masks)
            state, stop = _append(
                generated,
                state,
                value,
                minimum_output_bytes=minimum_output_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
            counters.emitted_bytes += 1
            if stop:
                break
            logits = runtime.consume(value)
            counters.sequential_target_calls += 1
            capture.require()
        return GenerationTrace(
            generated=bytes(generated),
            counters=counters.to_dict(),
            diagnostics=runtime.diagnostics,
        )
    finally:
        capture.close()


def generate_speculative(
    runtime: IncrementalBlockBltDecoder,
    head: Any,
    prompt: bytes,
    masks: Mapping[StrictUtf8State, torch.Tensor],
    tables: DeviceHangulTables,
    *,
    minimum_output_bytes: int,
    maximum_output_bytes: int,
    proposal_engine: IndependentProposalEngine | None = None,
    proposal: Callable[[Any, torch.Tensor, torch.Tensor, DeviceHangulTables], torch.Tensor]
    = propose_pairs,
) -> GenerationTrace:
    capture = DecoderHiddenCapture(runtime.model)
    counters = SpeculativeCounters()
    try:
        logits = runtime.prefill_parallel(prompt)
        current_hidden = capture.require()
        state = STRICT_UTF8_INITIAL_STATE
        generated = bytearray()
        pending_state = state
        pending = _masked_greedy(logits, pending_state, masks)
        state, stop = _append(
            generated,
            pending_state,
            pending,
            minimum_output_bytes=minimum_output_bytes,
            maximum_output_bytes=maximum_output_bytes,
        )
        counters.emitted_bytes += 1
        while not stop:
            trigger = pending_state.at_codepoint_boundary and 0xEA <= pending <= 0xED
            if not trigger:
                logits = runtime.consume(pending)
                counters.sequential_target_calls += 1
                current_hidden = capture.require()
                pending_state = state
                pending = _masked_greedy(logits, pending_state, masks)
                state, stop = _append(
                    generated,
                    pending_state,
                    pending,
                    minimum_output_bytes=minimum_output_bytes,
                    maximum_output_bytes=maximum_output_bytes,
                )
                counters.emitted_bytes += 1
                continue

            lead_value = pending - 0xEA
            proposal_context: torch.Tensor | None = None
            if proposal_engine is None:
                lead = torch.tensor(
                    [lead_value], dtype=torch.long, device=current_hidden.device
                )
                pair = int(
                    proposal(head, current_hidden.float(), lead, tables).item()
                )
            else:
                pair, proposal_context = proposal_engine.propose_with_context(
                    current_hidden, lead_value
                )
            second_draft, third_draft = pair_bytes(pair)
            counters.draft_head_calls += 1
            transaction = runtime.consume_block_transaction(
                bytes((pending, second_draft, third_draft))
            )
            counters.target_block_calls += 1

            second_target = _masked_greedy(transaction.logits[0:1], state, masks)
            if second_target != second_draft:
                transaction.finish(1)
                counters.first_mismatches += 1
                counters.cropped_speculative_bytes += 2
                counters.correction_bytes += 1
                logits = transaction.logits[0:1]
                current_hidden = transaction.hidden[0:1]
                pending_state = state
                pending = second_target
                state, stop = _append(
                    generated,
                    pending_state,
                    pending,
                    minimum_output_bytes=minimum_output_bytes,
                    maximum_output_bytes=maximum_output_bytes,
                )
                counters.emitted_bytes += 1
                if proposal_engine is not None and proposal_context is not None:
                    retry_third = proposal_engine.retry_third(
                        proposal_context,
                        lead_value,
                        second_target,
                    )
                    if retry_third is not None:
                        retry = runtime.consume_block_transaction(
                            bytes((second_target, retry_third))
                        )
                        counters.target_block_calls += 1
                        counters.retry_block_calls += 1
                        third_target = _masked_greedy(
                            retry.logits[0:1], state, masks
                        )
                        if third_target != retry_third:
                            retry.finish(1)
                            counters.retry_third_mismatches += 1
                            counters.cropped_speculative_bytes += 1
                            counters.correction_bytes += 1
                            logits = retry.logits[0:1]
                            current_hidden = retry.hidden[0:1]
                            pending_state = state
                            pending = third_target
                            state, stop = _append(
                                generated,
                                pending_state,
                                pending,
                                minimum_output_bytes=minimum_output_bytes,
                                maximum_output_bytes=maximum_output_bytes,
                            )
                            counters.emitted_bytes += 1
                            continue

                        counters.retry_third_accepts += 1
                        state, stop = _append(
                            generated,
                            state,
                            retry_third,
                            minimum_output_bytes=minimum_output_bytes,
                            maximum_output_bytes=maximum_output_bytes,
                        )
                        counters.emitted_bytes += 1
                        if stop:
                            retry.finish(1)
                            counters.cropped_speculative_bytes += 1
                            break
                        retry.finish(2)
                        logits = retry.logits[1:2]
                        current_hidden = retry.hidden[1:2]
                        pending_state = state
                        pending = _masked_greedy(logits, pending_state, masks)
                        state, stop = _append(
                            generated,
                            pending_state,
                            pending,
                            minimum_output_bytes=minimum_output_bytes,
                            maximum_output_bytes=maximum_output_bytes,
                        )
                        counters.emitted_bytes += 1
                        counters.bonus_bytes += 1
                continue

            counters.first_draft_accepts += 1
            state, stop = _append(
                generated,
                state,
                second_draft,
                minimum_output_bytes=minimum_output_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
            counters.emitted_bytes += 1
            if stop:  # A continuation byte cannot close this three-byte scalar.
                raise AssertionError("second Hangul byte unexpectedly stopped generation")
            third_target = _masked_greedy(transaction.logits[1:2], state, masks)
            if third_target != third_draft:
                transaction.finish(2)
                counters.second_mismatches += 1
                counters.cropped_speculative_bytes += 1
                counters.correction_bytes += 1
                logits = transaction.logits[1:2]
                current_hidden = transaction.hidden[1:2]
                pending_state = state
                pending = third_target
                state, stop = _append(
                    generated,
                    pending_state,
                    pending,
                    minimum_output_bytes=minimum_output_bytes,
                    maximum_output_bytes=maximum_output_bytes,
                )
                counters.emitted_bytes += 1
                continue

            counters.complete_pair_accepts += 1
            state, stop = _append(
                generated,
                state,
                third_draft,
                minimum_output_bytes=minimum_output_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
            counters.emitted_bytes += 1
            if stop:
                # Preserve the same one-byte output/cache lag as ordinary
                # greedy decoding: the final accepted byte is emitted but is
                # not needed as an input once generation stops.
                transaction.finish(2)
                counters.cropped_speculative_bytes += 1
                break
            transaction.finish(3)
            logits = transaction.logits[2:3]
            current_hidden = transaction.hidden[2:3]
            pending_state = state
            pending = _masked_greedy(logits, pending_state, masks)
            state, stop = _append(
                generated,
                pending_state,
                pending,
                minimum_output_bytes=minimum_output_bytes,
                maximum_output_bytes=maximum_output_bytes,
            )
            counters.emitted_bytes += 1
            counters.bonus_bytes += 1

        if bytes(generated).decode("utf-8", errors="strict").encode("utf-8") != bytes(
            generated
        ):
            raise AssertionError("speculative output UTF-8 round trip differs")
        expected_observed = len(prompt) + len(generated) - 1
        if runtime.diagnostics.observed_bytes != expected_observed:
            raise AssertionError("speculative runtime/output lag invariant differs")
        return GenerationTrace(
            generated=bytes(generated),
            counters=counters.to_dict(),
            diagnostics=runtime.diagnostics,
        )
    finally:
        capture.close()
