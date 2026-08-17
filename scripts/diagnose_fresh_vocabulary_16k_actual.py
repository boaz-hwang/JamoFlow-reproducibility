#!/usr/bin/env python3
"""Build a descriptive, non-authorizing mechanism audit for the 16K result."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from compositional_head_preflight_protocol import load_tokenizers
from fresh_vocabulary_16k_actual_core import MEASURED_CASES, MODES, ROLES
from fresh_vocabulary_16k_actual_protocol import (
    OUTPUT_PATH as SUMMARY_PATH,
)
from fresh_vocabulary_16k_actual_protocol import (
    ROOT,
    TIMING_PATH,
    VOCABULARY_BY_ROLE,
    array_sha256,
    canonical_sha256,
    encode_raw,
    hash_file,
    json_bytes,
    read_plan_json,
)

OUTPUT_PATH = SUMMARY_PATH.parent / "mechanism-diagnostic.json"


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values)
    if array.shape != (MEASURED_CASES,) or not np.isfinite(array).all():
        raise ValueError("fresh-16k mechanism distribution differs")
    return {
        "minimum": int(array.min()),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": int(array.max()),
    }


def _relation_summary(effect: np.ndarray, relation: np.ndarray) -> dict[str, object]:
    output: dict[str, object] = {"case_count": int(np.count_nonzero(relation))}
    if output["case_count"]:
        output["median_end_to_end_reduction"] = float(np.median(effect[relation]))
        output["faster_case_count"] = int(np.count_nonzero(effect[relation] > 0.0))
    else:
        output["median_end_to_end_reduction"] = None
        output["faster_case_count"] = 0
    return output


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    summary = read_plan_json(SUMMARY_PATH)
    unsigned = dict(summary)
    recorded = unsigned.pop("summary_sha256", None)
    if (
        summary.get("kind") != "fresh_vocabulary_16k_actual_one_seed_result_v1"
        or canonical_sha256(unsigned) != recorded
        or summary.get("status") != "fail_16k_trained_actual_e2e_preflight"
        or hash_file(TIMING_PATH) != summary["runtime"]["timing_artifact_sha256"]
    ):
        raise ValueError("fresh-16k mechanism source result differs")
    free_mode = MODES.index("free_running_utf8_greedy")
    with np.load(TIMING_PATH, allow_pickle=False) as archive:
        counts = np.ascontiguousarray(archive["output_token_count"])
        e2e = np.ascontiguousarray(archive["end_to_end_ms"])
        token_ids = np.ascontiguousarray(archive["free_token_ids"])
        output_bytes = np.ascontiguousarray(archive["free_output_bytes"])
        output_lengths = np.ascontiguousarray(archive["free_output_lengths"])
    loaded = load_tokenizers()
    role_diagnostics: dict[str, object] = {}
    for role_index, role in enumerate(ROLES):
        tokenizer, pieces = loaded[VOCABULARY_BY_ROLE[role]]
        emitted = counts[free_mode, :, 0, role_index].astype(np.int64)
        canonical = np.empty(MEASURED_CASES, dtype=np.int64)
        characters = np.empty(MEASURED_CASES, dtype=np.int64)
        hangul = np.empty(MEASURED_CASES, dtype=np.int64)
        exact = 0
        for case_index in range(MEASURED_CASES):
            raw_length = int(output_lengths[case_index, 0, role_index])
            raw = bytes(output_bytes[case_index, 0, role_index, :raw_length])
            text = raw.decode("utf-8", errors="strict")
            emitted_ids = tuple(
                int(value)
                for value in token_ids[
                    case_index, 0, role_index, : int(emitted[case_index])
                ]
            )
            canonical_ids = encode_raw(raw, tokenizer, pieces)
            canonical[case_index] = len(canonical_ids)
            characters[case_index] = len(text)
            hangul[case_index] = sum(
                0xAC00 <= ord(character) <= 0xD7A3 for character in text
            )
            exact += int(emitted_ids == canonical_ids)
        gap = emitted - canonical
        role_diagnostics[role] = {
            "emitted_token_count": _distribution(emitted),
            "canonical_reencoded_token_count": _distribution(canonical),
            "retokenization_gap": _distribution(gap),
            "noncanonical_trace_case_count": int(np.count_nonzero(gap)),
            "exact_canonical_trace_case_count": exact,
            "character_count": _distribution(characters),
            "precomposed_hangul_count": _distribution(hangul),
        }

    pair_diagnostics: dict[str, object] = {}
    candidate_index = ROLES.index("candidate_16k")
    for pair_name, reference_role in (
        ("candidate_vs_2k", "baseline_2k"),
        ("candidate_vs_8k", "frontier_8k"),
    ):
        reference_index = ROLES.index(reference_role)
        candidate_time = np.median(e2e[free_mode, :, :, candidate_index], axis=1)
        reference_time = np.median(e2e[free_mode, :, :, reference_index], axis=1)
        effect = 1.0 - candidate_time / reference_time
        candidate_tokens = counts[free_mode, :, 0, candidate_index].astype(np.float64)
        reference_tokens = counts[free_mode, :, 0, reference_index].astype(np.float64)
        token_effect = 1.0 - candidate_tokens / reference_tokens
        correlation = float(np.corrcoef(effect, token_effect)[0, 1])
        if not math.isfinite(correlation):
            raise ValueError("fresh-16k mechanism correlation differs")
        pair_diagnostics[pair_name] = {
            "end_to_end_vs_token_reduction_pearson": correlation,
            "candidate_fewer_tokens": _relation_summary(
                effect, candidate_tokens < reference_tokens
            ),
            "equal_token_count": _relation_summary(
                effect, candidate_tokens == reference_tokens
            ),
            "candidate_more_tokens": _relation_summary(
                effect, candidate_tokens > reference_tokens
            ),
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "fresh_vocabulary_16k_actual_mechanism_diagnostic_v1",
        "status": "post_hoc_descriptive_non_authorizing",
        "source": {
            "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
            "summary_artifact_sha256": hash_file(SUMMARY_PATH),
            "summary_sha256": summary["summary_sha256"],
            "timing_artifact_sha256": hash_file(TIMING_PATH),
            "output_token_count_array_sha256": array_sha256(counts),
            "end_to_end_array_sha256": array_sha256(e2e),
        },
        "roles": role_diagnostics,
        "pairs": pair_diagnostics,
        "claim_boundary": {
            "post_hoc": True,
            "changes_primary_gate": False,
            "authorizes_multiseed": False,
            "retokenization_is_causal_explanation": False,
            "describes_token_step_dominance": True,
        },
    }
    payload["diagnostic_sha256"] = canonical_sha256(payload)
    _publish(OUTPUT_PATH, json_bytes(payload))
    print("status=post_hoc_descriptive_non_authorizing")
    print(f"diagnostic_sha256={payload['diagnostic_sha256']}")


if __name__ == "__main__":
    main()
