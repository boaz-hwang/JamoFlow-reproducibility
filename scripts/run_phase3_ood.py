#!/usr/bin/env python3
"""Evaluate Phase 3 checkpoints on the full Leipzig Korean test partition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import torch

from jamoflow.neural_data import build_neural_stream
from jamoflow.neural_model import (
    build_main_model,
    parameter_count,
    research_versions,
)
from jamoflow.neural_training import evaluate_main_model, resolve_device
from jamoflow.phase1 import stream_arrays
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    merge_phase3_ood_manifest,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.phase3_confirmation import (
    CONFIRMATION_ONLY_SEEDS,
    load_confirmation_authorization,
    validate_confirmation_invocations,
    validate_confirmation_request,
)


POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
DEFAULT_SEEDS = (1729, 2718, 31415)
ALL_PREREGISTERED_SEEDS = (*DEFAULT_SEEDS, 57721, 65537)
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_status() -> str | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def _clean_git_commit() -> str:
    commit = _git_commit()
    status = _git_status()
    if not commit or status is None or status.strip():
        raise ValueError("Phase 3 OOD evidence requires a clean committed worktree")
    return commit


def _require_unchanged_clean_git(expected_commit: str) -> None:
    status = _git_status()
    if _git_commit() != expected_commit or status is None or status.strip():
        raise RuntimeError(
            "git HEAD/worktree changed during Phase 3 OOD evidence execution"
        )


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _release_model(model: Any, device: str) -> None:
    model.to("cpu")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()


def _validate_completed_result(
    report_path: Path,
    loss_path: Path,
    *,
    seed: int,
    policy: str,
    expected_examples: int,
    stream_sha256: str,
    source_sha256: str,
    patch_matrix_sha256: str,
    training_report_sha256: str,
    checkpoint_file_sha256: str,
    trained_state_sha256: str,
    git_commit: str,
    authorization: dict[str, Any] | None,
) -> None:
    """Reject stale OOD artifacts instead of silently treating them as done."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "seed": seed,
        "policy": policy,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "stream_selected_sha256": stream_sha256,
        "source_file_sha256": source_sha256,
        "patch_matrix_sha256": patch_matrix_sha256,
        "training_report_artifact_sha256": training_report_sha256,
        "checkpoint_artifact_sha256": checkpoint_file_sha256,
        "checkpoint_state_sha256": trained_state_sha256,
        "training_report_state_sha256": trained_state_sha256,
        "git_commit": git_commit,
        "git_worktree_clean_at_start": True,
        "authorization": authorization,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"stale OOD result for seed {seed}/{policy}: {key}; "
                "resume requires the exact original identity"
            )
    if report.get("parameters") != 19_596_096:
        raise ValueError(
            f"stale OOD result for seed {seed}/{policy}: parameters; "
            "resume requires the exact original identity"
        )
    evaluation = report.get("evaluation", {})
    expected_targets = expected_examples * (PHASE3_MODEL_SPEC.sequence_length - 1)
    if (
        evaluation.get("examples") != expected_examples
        or evaluation.get("predicted_bytes") != expected_targets
    ):
        raise ValueError(
            f"stale OOD result for seed {seed}/{policy}: evaluation counts; "
            "resume requires the exact original identity"
        )
    with np.load(loss_path) as archive:
        if archive.files != ["sequence_nll_nats"]:
            raise ValueError(f"unexpected loss keys in {loss_path}")
        values = archive["sequence_nll_nats"].astype(np.float64)
    if (
        values.shape != (expected_examples,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(f"invalid cached OOD losses in {loss_path}")
    reconstructed_bpb = float(values.sum()) / (expected_targets * math.log(2))
    if not math.isclose(
        reconstructed_bpb,
        float(evaluation["bpb"]),
        abs_tol=1e-7,
    ):
        raise ValueError(f"cached OOD loss/report mismatch in {report_path}")


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    policies = tuple(args.policies)
    unknown = set(policies) - set(POLICIES)
    if unknown or not policies or len(set(policies)) != len(policies):
        raise ValueError(f"unsupported OOD policies: {sorted(unknown)}")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or not set(seeds) <= set(ALL_PREREGISTERED_SEEDS)
    ):
        raise ValueError("OOD seeds must be unique preregistered Phase 3 seeds")
    run_git_commit = _clean_git_commit()
    device = resolve_device(args.device)
    training_run_root = Path(args.training_run_root)
    checkpoint_root = Path(args.checkpoint_root)
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    run_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    authorization = None
    if set(seeds) & set(CONFIRMATION_ONLY_SEEDS):
        validate_confirmation_request(seeds, policies)
        if args.force:
            raise ValueError("confirmation OOD evidence forbids --force overwrite")
        if args.authorization_summary is None:
            raise ValueError(
                "confirmation OOD requires --authorization-summary from corrected Gate I"
            )
        training_manifest_path = training_run_root / "manifest.json"
        if not training_manifest_path.is_file():
            raise ValueError("confirmation OOD lacks the primary training manifest")
        authorization = load_confirmation_authorization(
            Path(args.authorization_summary)
        )
        validate_confirmation_invocations(
            json.loads(training_manifest_path.read_text(encoding="utf-8")),
            authorization,
        )
    elif args.authorization_summary is not None:
        raise ValueError(
            "authorization summary is only valid for confirmation OOD seeds"
        )

    source_path = Path(args.data_root) / "ko.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_sha256 = _file_sha256(source_path)
    stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=args.byte_limit,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    whitespace = compact_whitespace_mask(stream.data).reshape(
        -1, stream.sequence_length
    )
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(
        -1, stream.sequence_length
    )
    matrices = structural_patch_matrices(
        boundaries,
        whitespace,
        spacelike,
    )
    for policy in policies:
        validate_padded_patch_matrix(
            matrices[policy], PHASE3_MODEL_SPEC.sequence_length
        )

    stream_sha256 = hashlib.sha256(stream.data).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": run_git_commit,
        "git_worktree_clean_at_start": True,
        "device": device,
        "platform": platform.platform(),
        "versions": research_versions(),
        "seeds": list(seeds),
        "policies": list(policies),
        "force": bool(args.force),
        "authorization": authorization,
        "requested_byte_limit": args.byte_limit,
        "source": {
            "filename": "ko.jsonl",
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "stream": {
            **stream.metadata(),
            "selected_stream_sha256": stream_sha256,
            "codepoint_boundaries_sha256": hashlib.sha256(
                stream.codepoint_boundaries
            ).hexdigest(),
        },
        "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
        "model_spec": PHASE3_MODEL_SPEC.to_dict(),
    }
    manifest_path = run_root / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    manifest = merge_phase3_ood_manifest(existing_manifest, manifest)
    _write_json(manifest_path, manifest)

    for seed in seeds:
        seed_run = run_root / f"seed-{seed}"
        seed_artifact = artifact_root / f"seed-{seed}"
        seed_run.mkdir(parents=True, exist_ok=True)
        seed_artifact.mkdir(parents=True, exist_ok=True)
        for policy in policies:
            output_path = seed_run / f"{policy}.json"
            loss_path = seed_artifact / f"{policy}-nll.npz"
            training_report_path = (
                training_run_root / f"seed-{seed}" / f"{policy}.json"
            )
            checkpoint_path = (
                checkpoint_root / f"seed-{seed}" / f"{policy}.pt"
            )
            if not training_report_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"missing training report/checkpoint for seed {seed}/{policy}"
                )
            training_report = json.loads(
                training_report_path.read_text(encoding="utf-8")
            )
            if (
                training_report.get("seed") != seed
                or training_report.get("policy") != policy
                or training_report.get("model_spec")
                != PHASE3_MODEL_SPEC.to_dict()
                or training_report.get("optimization_spec")
                != PHASE3_OPTIMIZATION_SPEC.to_dict()
                or training_report.get("parameters") != 19_596_096
            ):
                raise ValueError(
                    f"training report identity/design mismatch for seed "
                    f"{seed}/{policy}"
                )
            training_report_sha256 = _file_sha256(training_report_path)
            checkpoint_file_sha256 = _file_sha256(checkpoint_path)
            patch_matrix_sha256 = _array_sha256(matrices[policy])
            trained_state_sha256 = training_report.get("trained_state_sha256")
            if not isinstance(trained_state_sha256, str):
                raise ValueError(
                    f"training report state hash is missing for seed {seed}/{policy}"
                )
            if output_path.exists() != loss_path.exists():
                raise ValueError(
                    f"partial OOD result exists for seed {seed}/{policy}; "
                    "preserve it for forensic recovery and do not overwrite"
                )
            if output_path.exists() and loss_path.exists() and not args.force:
                _validate_completed_result(
                    output_path,
                    loss_path,
                    seed=seed,
                    policy=policy,
                    expected_examples=len(inputs),
                    stream_sha256=stream_sha256,
                    source_sha256=source_sha256,
                    patch_matrix_sha256=patch_matrix_sha256,
                    training_report_sha256=training_report_sha256,
                    checkpoint_file_sha256=checkpoint_file_sha256,
                    trained_state_sha256=trained_state_sha256,
                    git_commit=run_git_commit,
                    authorization=authorization,
                )
                print(f"seed {seed}/{policy}: OOD already complete", flush=True)
                continue
            model = build_main_model(
                PHASE3_MODEL_SPEC,
                seed=seed,
                global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
            )
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
            checkpoint_state_sha256 = _state_dict_sha256(model)
            if checkpoint_state_sha256 != training_report["trained_state_sha256"]:
                raise ValueError(
                    f"checkpoint state hash mismatch for seed {seed}/{policy}"
                )
            print(f"seed {seed}/{policy}: evaluating Leipzig OOD", flush=True)
            evaluation, sequence_nll = evaluate_main_model(
                model,
                inputs,
                matrices[policy],
                device,
                batch_size=64,
                return_sequence_nll=True,
            )
            if sequence_nll is None:
                raise AssertionError("OOD sequence losses were not produced")
            _save_npz(loss_path, sequence_nll_nats=sequence_nll)
            _write_json(
                output_path,
                {
                    "schema_version": 2,
                    "seed": seed,
                    "policy": policy,
                    "parameters": parameter_count(model),
                    "model_spec": PHASE3_MODEL_SPEC.to_dict(),
                    "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
                    "stream_selected_sha256": stream_sha256,
                    "source_file_sha256": source_sha256,
                    "training_report_artifact_sha256": training_report_sha256,
                    "checkpoint_artifact_sha256": checkpoint_file_sha256,
                    "checkpoint_state_sha256": checkpoint_state_sha256,
                    "training_report_state_sha256": training_report[
                        "trained_state_sha256"
                    ],
                    "git_commit": run_git_commit,
                    "git_worktree_clean_at_start": True,
                    "authorization": authorization,
                    "patch_matrix_sha256": patch_matrix_sha256,
                    "patch_diagnostics": variable_patch_diagnostics(
                        matrices[policy], boundaries
                    ).to_dict(),
                    "evaluation": evaluation.to_dict(),
                },
            )
            _release_model(model, device)
    _require_unchanged_clean_git(run_git_commit)
    print(f"completed Phase 3 OOD runs under {run_root}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument("--training-run-root", default="runs/phase3")
    parser.add_argument("--checkpoint-root", default="artifacts/phase3")
    parser.add_argument("--run-root", default="runs/phase3-ood")
    parser.add_argument("--artifact-root", default="artifacts/phase3-ood")
    parser.add_argument("--byte-limit", type=int, default=100_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--authorization-summary", type=Path)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
