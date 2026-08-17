#!/usr/bin/env python3
"""Run the exact calibration-only W72 speculative E2E preflight."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

from jamoflow.hplt3 import hash_file
from jamoflow.inference_actual_runtime_v5 import load_actual_model, release_actual_model
from jamoflow.inference_actual_v5 import array_sha256
from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive
from jamoflow.inference_final_authorization_v2 import (
    FINAL_AUTHORIZATION_PATH,
    SELECTION_LOCK_PATH,
    validate_final_evaluation_authorization_v2,
)
from jamoflow.inference_selection_v2 import validate_selection_lock_v2
from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import research_versions
from jamoflow.neural_training import synchronize
from jamoflow.utf8 import (
    StrictUtf8State,
    strict_utf8_allowed_ranges,
    strict_utf8_reachable_states,
)
from scripts.hangul_draft_acceptance_core import DeviceHangulTables
from scripts.incremental_block_kernel import IncrementalBlockBltDecoder
from scripts.run_hangul_draft_acceptance_preflight import _select_prompts
from scripts.speculative_w72_preflight_core import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    COUNTER_KEYS,
    MODES,
    PROMPT_COUNT,
    PROTOCOL_ID,
    REPETITIONS,
    summarize_speculative_preflight,
)
from scripts.speculative_w72_runtime import (
    IndependentProposalEngine,
    generate_baseline,
    generate_speculative,
    load_independent_head,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/manifests/speculative-w72-preflight-v1.json"
SOURCE_PATH = ROOT / "data/processed/hplt3-korean-phase3/ko.jsonl"
AUTHORIZATION_PATH = ROOT / FINAL_AUTHORIZATION_PATH
SELECTION_PATH = ROOT / SELECTION_LOCK_PATH
ACCEPTANCE_PATH = ROOT / "results/hangul-draft-acceptance-v1/summary.json"
BLOCK_RESULT_PATH = ROOT / "results/target-block-kernel-v2/summary.json"
FREE_CACHE_PATH = ROOT / "artifacts/hangul-draft-acceptance-v1/free-target.npz"
HEAD_PATH = ROOT / (
    "artifacts/hangul-draft-acceptance-v1/heads/"
    "generic_independent_utf8-seed-20260813.pt"
)
RAW_PATH = ROOT / "artifacts/speculative-w72-preflight-v1/raw.npz"
OUTPUT_PATH = ROOT / "results/speculative-w72-preflight-v1/summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _publish_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def _clean_commit() -> str:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("speculative preflight requires a clean worktree")
    commit = _command("git", "rev-parse", "HEAD")
    if len(commit) != 40:
        raise ValueError("speculative preflight requires a Git commit")
    return commit


def _require_ac_power() -> str:
    state = _command("pmset", "-g", "batt")
    if "Now drawing from 'AC Power'" not in state:
        raise RuntimeError("speculative preflight requires AC power")
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _validate_plan(plan: Mapping[str, Any], commit: str) -> None:
    if (
        set(plan)
        != {
            "cases",
            "decision_rule",
            "head",
            "implementation_sha256",
            "input",
            "kind",
            "model",
            "output",
            "protocol_id",
            "schema_version",
            "status",
            "threat_model",
            "timing",
        }
        or plan.get("schema_version") != 1
        or plan.get("kind") != "speculative_w72_preflight_plan_v1"
        or plan.get("protocol_id") != PROTOCOL_ID
        or int(plan["cases"]["prompt_count"]) != PROMPT_COUNT
        or int(plan["timing"]["repetitions"]) != REPETITIONS
        or tuple(plan["timing"]["modes"]) != MODES
        or int(plan["decision_rule"]["bootstrap_repetitions"])
        != BOOTSTRAP_REPETITIONS
        or int(plan["decision_rule"]["bootstrap_seed"]) != BOOTSTRAP_SEED
        or plan["timing"]["torch_inference_mode"] is not True
        or plan["timing"]["device"] != "mps"
    ):
        raise ValueError("speculative preflight plan schema differs")
    for relative, expected in plan["implementation_sha256"].items():
        if hash_file(ROOT / relative) != expected:
            raise ValueError(f"speculative implementation differs: {relative}")
    checks = (
        (SOURCE_PATH, plan["input"]["source_sha256"]),
        (AUTHORIZATION_PATH, plan["model"]["authorization_artifact_sha256"]),
        (SELECTION_PATH, plan["model"]["selection_artifact_sha256"]),
        (ACCEPTANCE_PATH, plan["head"]["acceptance_artifact_sha256"]),
        (BLOCK_RESULT_PATH, plan["head"]["block_result_artifact_sha256"]),
        (FREE_CACHE_PATH, plan["cases"]["source_artifact_sha256"]),
        (HEAD_PATH, plan["head"]["checkpoint_artifact_sha256"]),
    )
    if any(hash_file(path) != expected for path, expected in checks):
        raise ValueError("speculative upstream artifact differs")
    if any(path.exists() for path in (RAW_PATH, OUTPUT_PATH)):
        raise FileExistsError("speculative preflight output already exists")
    if _command("git", "rev-parse", "HEAD") != commit:
        raise RuntimeError("speculative plan verification changed HEAD")


def _utf8_masks(device: str) -> dict[StrictUtf8State, torch.Tensor]:
    masks: dict[StrictUtf8State, torch.Tensor] = {}
    for state in strict_utf8_reachable_states():
        mask = torch.zeros(256, dtype=torch.bool, device=device)
        for lower, upper in strict_utf8_allowed_ranges(state):
            mask[lower : upper + 1] = True
        masks[state] = mask
    synchronize(device)
    return masks


def _speculative_runtime(bundle: Any) -> IncrementalBlockBltDecoder:
    return IncrementalBlockBltDecoder(
        bundle.model,
        "causal_whitespace_grid",
        horizon=512,
        patch_count=72,
        fixed_stride=6,
    )


def _mode_order(case: int, repetition: int) -> tuple[int, int]:
    return (0, 1) if (case + repetition) % 2 == 0 else (1, 0)


def _output_root(outputs: list[bytes]) -> str:
    digest = hashlib.sha256(b"JamoFlow/speculative-w72-outputs/v1\0")
    for index, output in enumerate(outputs):
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(output).to_bytes(8, "big"))
        digest.update(output)
    return digest.hexdigest()


def run() -> None:
    commit = _clean_commit()
    plan = _read_json(PLAN_PATH)
    _validate_plan(plan, commit)
    selection = _read_json(SELECTION_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)
    validate_selection_lock_v2(selection)
    validate_final_evaluation_authorization_v2(
        authorization,
        selection_lock=selection,
    )
    candidate = next(
        model for model in authorization["models"] if model["artifact_role"] == "candidate"
    )
    if (
        candidate["identity_sha256"] != plan["model"]["candidate_identity_sha256"]
        or candidate["seeds"][str(plan["model"]["seed"])]["checkpoint"]
        != plan["model"]["checkpoint"]
    ):
        raise ValueError("speculative target identity differs")

    stream = build_neural_stream(
        SOURCE_PATH,
        "ko",
        "calibration",
        int(plan["input"]["byte_limit"]),
        512,
    )
    if (
        len(stream.data) != int(plan["input"]["stream_bytes"])
        or hashlib.sha256(stream.data).hexdigest() != plan["input"]["stream_sha256"]
    ):
        raise ValueError("speculative calibration stream differs")
    prompts, offsets = _select_prompts(
        stream.data,
        prompt_bytes=int(plan["cases"]["prompt_bytes"]),
        count=PROMPT_COUNT,
        minimum_hangul_share=float(plan["cases"]["minimum_prompt_hangul_share"]),
    )
    with np.load(FREE_CACHE_PATH, allow_pickle=False) as source:
        if set(source.files) != set(plan["cases"]["source_artifact_keys"]):
            raise ValueError("speculative prompt-source schema differs")
        if not np.array_equal(prompts, source["prompts"]) or not np.array_equal(
            offsets, source["prompt_offsets"]
        ):
            raise ValueError("speculative reconstructed prompts differ")
    if (
        array_sha256(prompts) != plan["cases"]["prompts_array_sha256"]
        or array_sha256(offsets) != plan["cases"]["prompt_offsets_array_sha256"]
    ):
        raise ValueError("speculative prompt identity differs")

    started = time.time()
    power_sha256 = _require_ac_power()
    timings = np.empty((PROMPT_COUNT, REPETITIONS, len(MODES)), dtype=np.float64)
    output_lengths = np.empty(PROMPT_COUNT, dtype=np.int64)
    output_hashes = np.empty((PROMPT_COUNT, 32), dtype=np.uint8)
    counter_array = np.empty((PROMPT_COUNT, len(COUNTER_KEYS)), dtype=np.int64)
    canonical_outputs: list[bytes] = []
    with publication_mps_exclusive(), torch.inference_mode():
        bundle = load_actual_model(
            role="candidate",
            identity=candidate,
            seed=int(plan["model"]["seed"]),
            device="mps",
        )
        head = load_independent_head(
            HEAD_PATH,
            device="mps",
            expected_artifact_sha256=plan["head"]["checkpoint_artifact_sha256"],
            expected_state_sha256=plan["head"]["checkpoint_state_sha256"],
            expected_seed=int(plan["head"]["seed"]),
            expected_plan_sha256=plan["head"]["training_plan_artifact_sha256"],
        )
        tables = DeviceHangulTables.build("mps")
        engine = IndependentProposalEngine(head, "mps")
        masks = _utf8_masks("mps")
        minimum = int(plan["cases"]["minimum_output_bytes"])
        maximum = int(plan["cases"]["maximum_output_bytes"])

        # Untimed full correctness pass over every fixed prompt.
        for case, prompt_values in enumerate(prompts):
            prompt = bytes(prompt_values)
            baseline = generate_baseline(
                bundle.runtime(),
                prompt,
                masks,
                minimum_output_bytes=minimum,
                maximum_output_bytes=maximum,
            )
            speculative = generate_speculative(
                _speculative_runtime(bundle),
                head,
                prompt,
                masks,
                tables,
                minimum_output_bytes=minimum,
                maximum_output_bytes=maximum,
                proposal_engine=engine,
            )
            if (
                baseline.generated != speculative.generated
                or baseline.diagnostics != speculative.diagnostics
            ):
                raise AssertionError("speculative greedy output/cache differs")
            canonical_outputs.append(baseline.generated)
            output_lengths[case] = len(baseline.generated)
            output_hashes[case] = np.frombuffer(
                hashlib.sha256(baseline.generated).digest(), dtype=np.uint8
            )
            counter_array[case] = np.asarray(
                [speculative.counters[key] for key in COUNTER_KEYS], dtype=np.int64
            )

        # Warm both paths without using their times.
        for case in range(int(plan["timing"]["warmup_prompts"])):
            prompt = bytes(prompts[case])
            generate_baseline(
                bundle.runtime(), prompt, masks,
                minimum_output_bytes=minimum, maximum_output_bytes=maximum,
            )
            generate_speculative(
                _speculative_runtime(bundle), head, prompt, masks, tables,
                minimum_output_bytes=minimum, maximum_output_bytes=maximum,
                proposal_engine=engine,
            )

        for case, prompt_values in enumerate(prompts):
            prompt = bytes(prompt_values)
            for repetition in range(REPETITIONS):
                for mode in _mode_order(case, repetition):
                    synchronize("mps")
                    trial_started = time.perf_counter_ns()
                    if mode == 0:
                        trace = generate_baseline(
                            bundle.runtime(), prompt, masks,
                            minimum_output_bytes=minimum,
                            maximum_output_bytes=maximum,
                        )
                    else:
                        trace = generate_speculative(
                            _speculative_runtime(bundle), head, prompt, masks, tables,
                            minimum_output_bytes=minimum,
                            maximum_output_bytes=maximum,
                            proposal_engine=engine,
                        )
                    synchronize("mps")
                    timings[case, repetition, mode] = (
                        time.perf_counter_ns() - trial_started
                    ) / 1_000_000
                    if trace.generated != canonical_outputs[case]:
                        raise AssertionError("timed speculative output differs")
        release_actual_model(bundle)

    correctness = {
        "all_outputs_exact": True,
        "cache_comparisons": PROMPT_COUNT,
        "output_comparisons": PROMPT_COUNT,
        "output_hash_root_sha256": _output_root(canonical_outputs),
    }
    aggregate = summarize_speculative_preflight(
        timings_ms=timings,
        output_lengths=output_lengths,
        speculative_counters=counter_array,
        correctness=correctness,
        minimum_point_reduction=float(
            plan["decision_rule"]["minimum_point_reduction"]
        ),
        minimum_lower_bound=float(
            plan["decision_rule"]["minimum_bootstrap_lower_bound"]
        ),
        minimum_positive_prompts=int(
            plan["decision_rule"]["minimum_positive_prompts"]
        ),
    )
    arrays = {
        "timings_ms": timings,
        "output_lengths": output_lengths,
        "output_sha256": output_hashes,
        "speculative_counters": counter_array,
    }
    raw_bytes = _npz_bytes(arrays)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "speculative_w72_preflight_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "status": (
            "multi_seed_generic_comparator_authorized"
            if aggregate["gates"]["multi_seed_generic_comparator_authorized"]
            else "multi_byte_branch_stopped"
        ),
        "provenance": {
            "git_commit": commit,
            "plan_artifact_sha256": hash_file(PLAN_PATH),
            "authorization_artifact_sha256": hash_file(AUTHORIZATION_PATH),
            "block_result_artifact_sha256": hash_file(BLOCK_RESULT_PATH),
            "head_checkpoint_artifact_sha256": hash_file(HEAD_PATH),
            "power_snapshot_sha256": power_sha256,
            "runtime": {
                **research_versions(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch_inference_mode": True,
            },
        },
        "raw_evidence": {
            "artifact_path": RAW_PATH.relative_to(ROOT).as_posix(),
            "artifact_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "array_sha256": {
                key: array_sha256(value) for key, value in arrays.items()
            },
            "counter_order": list(COUNTER_KEYS),
        },
        "aggregate": aggregate,
        "elapsed_seconds": float(time.time() - started),
        "claim_boundary": {
            "calibration_only": True,
            "exact_same_target_greedy_output": True,
            "single_target_seed": True,
            "generic_all_byte_comparator_run": False,
            "final_or_publication_efficiency_claimed": False,
            "pass_authorizes": "multi-seed target heads and same-cost generic all-byte comparator",
        },
    }
    summary["summary_sha256"] = hashlib.sha256(_json_bytes(summary)).hexdigest()
    if _command("git", "rev-parse", "HEAD") != commit or _command(
        "git", "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("speculative preflight changed tracked repository state")
    _publish_no_clobber(RAW_PATH, raw_bytes)
    _publish_no_clobber(OUTPUT_PATH, _json_bytes(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "reduction": aggregate["end_to_end"]["reduction"],
                "lower": aggregate["end_to_end"]["prompt_bootstrap_95_interval"][
                    "lower"
                ],
                "positive_prompts": aggregate["end_to_end"][
                    "positive_prompt_count"
                ],
                "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
