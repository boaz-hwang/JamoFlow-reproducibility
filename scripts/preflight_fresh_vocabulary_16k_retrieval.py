#!/usr/bin/env python3
"""Timing-silent real-checkpoint exactness preflight for 16K retrieval drafting."""

from __future__ import annotations

import gc
import subprocess

import torch
from benchmark_fresh_vocabulary_16k_block import load_target, prepare_payloads
from fresh_vocabulary_16k_retrieval_actual_core import MODES, ROLES
from fresh_vocabulary_16k_retrieval_protocol import (
    PLAN_PATH,
    ROOT,
    load_table,
    read_json,
    reconstruct_cases,
    validate_plan,
)
from fresh_vocabulary_16k_retrieval_runtime import run_retrieval_trial

from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("16K retrieval preflight requires a clean worktree")
    plan = read_json(PLAN_PATH)
    validate_plan(plan, verify_derived=True)
    prompts, continuations, cases = reconstruct_cases()
    if cases != plan["cases"]:
        raise RuntimeError("16K retrieval preflight cases differ")
    table = load_table()
    maximum = plan["tokenizer_runtime"]["strict_utf8_transitions"]["maximum_free_output_bytes"]
    with publication_mps_exclusive():
        bundle = load_target(plan)
        payload = prepare_payloads(bundle, prompts[:1], continuations[:1])[0]
        for mode in MODES:
            ids = payload["controlled_ids" if mode == "controlled_replay" else "free_ids"]
            raw = payload["controlled_raw" if mode == "controlled_replay" else "free_raw"]
            for role in ROLES:
                _, trace = run_retrieval_trial(
                    bundle,
                    payload["prompt_raw"],
                    payload["prompt_ids"],
                    ids,
                    raw,
                    table,
                    role=role,
                    mode=mode,
                    continuation_bytes=plan["experiment"]["continuation_bytes"],
                    maximum_output_bytes=maximum,
                )
                if trace.token_ids != tuple(ids) or trace.raw != raw:
                    raise AssertionError("16K retrieval preflight output differs")
        bundle.model.to("cpu")
        del bundle
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
    print("trained 16K retrieval preflight: PASS")


if __name__ == "__main__":
    main()
