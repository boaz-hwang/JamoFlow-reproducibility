#!/usr/bin/env python3
"""Loss-silent one-effective-batch MPS feasibility check for the 16K graph."""

from __future__ import annotations

import gc
import subprocess

import numpy as np
import torch

from fresh_vocabulary_16k_core import CANDIDATE_ROLE, EFFECTIVE_BATCH_SIZE
from fresh_vocabulary_16k_protocol import (
    PLAN_PATH,
    ROOT,
    build_plan,
)
from run_fresh_vocabulary_16k import (
    _all_parameter_optimizer,
    _build_initial_model,
    _cleanup_role_data,
    _effective_step,
    _load_role_data,
)
from scalar_runtime_core import model_parameter_count

from jamoflow.inference_calibration_replay_v2 import publication_mps_exclusive


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("fresh-16k preflight requires a clean worktree")
    if PLAN_PATH.exists():
        raise RuntimeError("fresh-16k preflight must precede plan sealing")
    commit = _git("rev-parse", "HEAD")
    plan = build_plan(commit)
    data = _load_role_data(CANDIDATE_ROLE, plan)
    try:
        with publication_mps_exclusive():
            model = _build_initial_model(CANDIDATE_ROLE, plan, data).to("mps")
            optimizer = _all_parameter_optimizer(model)
            batch = np.asarray(
                data["train_sequences"][:EFFECTIVE_BATCH_SIZE], dtype=np.int64
            )
            batch_raw = data["optimizer_batch_raw_target_bytes"]
            _effective_step(
                model,
                optimizer,
                batch,
                role=CANDIDATE_ROLE,
                cumulative_raw_target_bytes=int(batch_raw[0]),
                total_raw_target_bytes=int(batch_raw.sum()),
                stage_one_raw_target_bytes=None,
                stage_one=False,
                copied_input_rows=None,
                copied_output_rows=None,
            )
            parameters = model_parameter_count(model)
            model.to("cpu")
            del optimizer, model
            gc.collect()
            torch.mps.empty_cache()
        if _git("rev-parse", "HEAD") != commit or _git(
            "status", "--porcelain", "--untracked-files=all"
        ):
            raise RuntimeError("repository changed during fresh-16k preflight")
        print("status=finite_loss_silent_16k_effective_batch")
        print(f"role={CANDIDATE_ROLE}")
        print(f"parameters={parameters}")
        print("reported_loss_value=false")
    finally:
        _cleanup_role_data(data)


if __name__ == "__main__":
    main()
