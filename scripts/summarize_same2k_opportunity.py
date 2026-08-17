#!/usr/bin/env python3
"""Independently replay and summarize the sealed same-2K opportunity gate."""

from __future__ import annotations

import subprocess

from same2k_opportunity_protocol import (
    ACTIVE_PATH,
    PIECES_PATH,
    PLAN_PATH,
    RESULT_PATH,
    ROOT,
    SENTENCEPIECE_MODEL_PATH,
    TRAINED_TOKENIZER_PATH,
    WORKER_PATH,
    canonical_sha256,
    hash_file,
    json_bytes,
    load_bpe_tokenizer,
    load_learned_tokenizers,
    read_json,
    reconstruct_shared_inputs,
    validate_plan,
)
from same2k_opportunity import (
    BPE_ROLE,
    evaluate_tokenizer_opportunity,
    metrics_identity_without_timing,
    opportunity_decision,
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main() -> None:
    if _command("git", "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("same-2K summary requires a clean root")
    if RESULT_PATH.exists() or ACTIVE_PATH.exists():
        raise FileExistsError("same-2K result or active marker already exists")
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    worker = read_json(WORKER_PATH)
    if (
        worker.get("kind") != "same2k_generic_opportunity_worker_v6"
        or worker.get("plan_artifact_sha256") != hash_file(PLAN_PATH)
        or worker.get("plan_sha256") != plan["plan_sha256"]
        or worker.get("run_git_commit") != _command("git", "rev-parse", "HEAD")
        or worker.get("complete") is not True
    ):
        raise ValueError("same-2K worker identity differs")
    expected_artifacts = {
        str(path.relative_to(ROOT)): hash_file(path)
        for path in (TRAINED_TOKENIZER_PATH, SENTENCEPIECE_MODEL_PATH, PIECES_PATH)
    }
    if worker.get("artifacts") != expected_artifacts:
        raise ValueError("same-2K worker artifact identity differs")

    learned, _, _ = load_learned_tokenizers()
    tokenizers = {BPE_ROLE: load_bpe_tokenizer(), **learned}
    calibration, prompts, continuations, _ = reconstruct_shared_inputs()
    replay = {
        role: evaluate_tokenizer_opportunity(
            role=role,
            tokenizer=tokenizer,
            calibration_raw=calibration,
            prompts=prompts,
            continuations=continuations,
        ).to_dict()
        for role, tokenizer in tokenizers.items()
    }
    worker_metrics = worker.get("metrics_by_role")
    if not isinstance(worker_metrics, dict) or set(worker_metrics) != set(replay):
        raise ValueError("same-2K worker metric role set differs")
    for role in replay:
        if metrics_identity_without_timing(replay[role]) != metrics_identity_without_timing(
            worker_metrics[role]
        ):
            raise ValueError("same-2K independent replay differs")
    decision = opportunity_decision(replay)
    payload = {
        "schema_version": 1,
        "kind": "same2k_generic_opportunity_result_v6",
        "protocol_id": plan["protocol_id"],
        "plan_artifact_sha256": hash_file(PLAN_PATH),
        "plan_sha256": plan["plan_sha256"],
        "worker_artifact_sha256": hash_file(WORKER_PATH),
        "tokenizer_artifacts": expected_artifacts,
        "training_metadata": worker["training_metadata"],
        "metrics_by_role": replay,
        "encode_throughput_replications": {
            role: {
                "worker": {
                    "seconds": worker_metrics[role]["encode_seconds"],
                    "median_megabytes_per_second": worker_metrics[role][
                        "encode_median_megabytes_per_second"
                    ],
                },
                "independent_replay": {
                    "seconds": replay[role]["encode_seconds"],
                    "median_megabytes_per_second": replay[role][
                        "encode_median_megabytes_per_second"
                    ],
                },
            }
            for role in replay
        },
        "decision": decision,
        "claim_boundary": plan["claim_boundary"],
        "independent_full_replay": True,
        "complete": True,
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("xb") as output:
        output.write(json_bytes(payload))
    print(f"wrote {RESULT_PATH.relative_to(ROOT)} decision={decision['next_action']}")


if __name__ == "__main__":
    main()
