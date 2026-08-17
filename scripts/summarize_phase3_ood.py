#!/usr/bin/env python3
"""Aggregate the preregistered Leipzig Korean OOD guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from jamoflow.neural_data import build_neural_stream
from jamoflow.phase1 import stream_arrays
from jamoflow.phase1_analysis import numeric_summary, paired_t_interval
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
)
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.phase3_analysis import hierarchical_paired_bootstrap_estimates
from jamoflow.phase3_confirmation import (
    CONFIRMATION_ONLY_SEEDS,
    load_confirmation_authorization,
    validate_confirmation_invocations,
)


F = "fixed_byte_6"
C = "causal_codepoint_grid"
W = "causal_whitespace_grid"
POLICIES = (F, C, W)
CONTRASTS = {
    "whitespace_minus_codepoint": (W, C),
    "whitespace_minus_fixed": (W, F),
}
TARGETS_PER_SEQUENCE = 511
EXPECTED_PARAMETERS = 19_596_096
GLOBAL_POSITION_LIMIT = PHASE3_MODEL_SPEC.sequence_length * 2 + 8
INITIAL_SEEDS = (1729, 2718, 31415)
CONFIRMATION_SEEDS = (*INITIAL_SEEDS, 57721, 65537)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"unexpected checkpoint entry in {path}")
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def ood_gate(contrast_means: dict[str, float]) -> dict[str, Any]:
    checks = {
        name: {
            "mean_bpb": contrast_means[name],
            "maximum_regression_bpb": 0.020,
            "pass": contrast_means[name] <= 0.020,
        }
        for name in CONTRASTS
    }
    return {
        "pass": all(value["pass"] for value in checks.values()),
        "checks": checks,
    }


def _validate_manifest_design(
    manifest: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("OOD manifest schema mismatch")
    if len(manifest.get("seeds", [])) != len(set(manifest.get("seeds", []))):
        raise ValueError("OOD manifest contains duplicate seeds")
    if len(manifest.get("policies", [])) != len(
        set(manifest.get("policies", []))
    ):
        raise ValueError("OOD manifest contains duplicate policies")
    if not set(seeds) <= set(manifest.get("seeds", [])):
        raise ValueError("requested OOD seeds are absent from the manifest")
    if set(manifest.get("policies", [])) != set(POLICIES):
        raise ValueError("OOD manifest policies differ from F/C/W")
    if manifest.get("model_spec") != PHASE3_MODEL_SPEC.to_dict():
        raise ValueError("OOD manifest model spec mismatch")
    if manifest.get("global_max_position_embeddings") != GLOBAL_POSITION_LIMIT:
        raise ValueError("OOD manifest global position limit mismatch")
    if not isinstance(manifest.get("requested_byte_limit"), int) or manifest[
        "requested_byte_limit"
    ] <= 0:
        raise ValueError("OOD manifest byte limit is invalid")
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("OOD manifest lacks invocation provenance")
    for seed in seeds:
        for policy in POLICIES:
            if not any(
                seed in invocation.get("seeds", [])
                and policy in invocation.get("policies", [])
                for invocation in invocations
                if isinstance(invocation, dict)
            ):
                raise ValueError(
                    f"OOD manifest has no invocation for seed {seed}/{policy}"
                )


def _validate_confirmation_report_binding(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    seed: int,
    policy: str,
    authorization: dict[str, Any] | None,
) -> None:
    if seed not in CONFIRMATION_ONLY_SEEDS:
        return
    if authorization is None:
        raise ValueError("confirmation OOD report lacks its authorization context")
    commit = report.get("git_commit")
    if (
        report.get("schema_version") != 2
        or report.get("git_worktree_clean_at_start") is not True
        or report.get("authorization") != authorization
        or not isinstance(commit, str)
        or len(commit) != 40
    ):
        raise ValueError(
            f"confirmation OOD report binding mismatch: {seed}/{policy}"
        )
    if not any(
        isinstance(invocation, dict)
        and seed in invocation.get("seeds", [])
        and policy in invocation.get("policies", [])
        and invocation.get("git_commit") == commit
        and invocation.get("git_worktree_clean_at_start") is True
        and invocation.get("authorization") == authorization
        for invocation in manifest.get("invocations", [])
    ):
        raise ValueError(
            f"confirmation OOD report has no exact clean invocation: {seed}/{policy}"
        )


def _reconstruct_ood_inputs(
    manifest: dict[str, Any],
    data_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_path = data_root / "ko.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = manifest.get("source", {})
    if (
        source.get("filename") != "ko.jsonl"
        or source.get("bytes") != source_path.stat().st_size
        or source.get("sha256") != _sha256(source_path)
    ):
        raise ValueError("OOD source artifact differs from the run manifest")

    stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=manifest["requested_byte_limit"],
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    recorded_stream = manifest.get("stream", {})
    for key, value in stream.metadata().items():
        if recorded_stream.get(key) != value:
            raise ValueError(f"OOD stream metadata mismatch: {key}")
    stream_sha256 = hashlib.sha256(stream.data).hexdigest()
    boundaries_sha256 = hashlib.sha256(stream.codepoint_boundaries).hexdigest()
    if recorded_stream.get("selected_stream_sha256") != stream_sha256:
        raise ValueError("OOD selected stream hash mismatch")
    if recorded_stream.get("codepoint_boundaries_sha256") != boundaries_sha256:
        raise ValueError("OOD boundary stream hash mismatch")

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
    all_matrices = structural_patch_matrices(
        boundaries,
        whitespace,
        spacelike,
    )
    matrices = {policy: all_matrices[policy] for policy in POLICIES}
    for matrix in matrices.values():
        validate_padded_patch_matrix(matrix, PHASE3_MODEL_SPEC.sequence_length)
    provenance = {
        "source_file_sha256": source["sha256"],
        "selected_stream_sha256": stream_sha256,
        "codepoint_boundaries_sha256": boundaries_sha256,
        "patch_matrix_sha256": {
            policy: _array_sha256(matrix) for policy, matrix in matrices.items()
        },
        "sequence_count": stream.sequence_count,
    }
    return inputs, boundaries, provenance


def run(args: argparse.Namespace) -> int:
    seeds = tuple(args.seeds)
    if seeds not in (INITIAL_SEEDS, CONFIRMATION_SEEDS):
        raise ValueError("OOD seeds must be the preregistered initial three or final five")
    run_root = Path(args.run_root)
    artifact_root = Path(args.artifact_root)
    training_run_root = Path(args.training_run_root)
    checkpoint_root = Path(args.checkpoint_root)
    manifest = _read_json(run_root / "manifest.json")
    _validate_manifest_design(manifest, seeds)
    confirmation_authorization = None
    if set(seeds) & set(CONFIRMATION_ONLY_SEEDS):
        if args.confirmation_authorization_summary is None:
            raise ValueError(
                "five-seed OOD summary requires "
                "--confirmation-authorization-summary"
            )
        confirmation_authorization = load_confirmation_authorization(
            Path(args.confirmation_authorization_summary)
        )
        validate_confirmation_invocations(manifest, confirmation_authorization)
        training_manifest_path = training_run_root / "manifest.json"
        if not training_manifest_path.is_file():
            raise ValueError("five-seed OOD summary lacks training manifest")
        validate_confirmation_invocations(
            _read_json(training_manifest_path), confirmation_authorization
        )
    elif args.confirmation_authorization_summary is not None:
        raise ValueError(
            "confirmation authorization is only valid for five-seed OOD evidence"
        )
    inputs, boundaries, source_provenance = _reconstruct_ood_inputs(
        manifest,
        Path(args.data_root),
    )
    expected_examples = len(inputs)
    if boundaries.shape != inputs.shape:
        raise ValueError("OOD input and boundary matrices differ in shape")
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    losses: dict[int, dict[str, np.ndarray]] = {}
    loss_hashes: dict[str, dict[str, str]] = {}
    training_report_hashes: dict[str, dict[str, str]] = {}
    checkpoint_artifact_hashes: dict[str, dict[str, str]] = {}
    checkpoint_state_hashes: dict[str, dict[str, str]] = {}
    for seed in seeds:
        reports[seed] = {}
        losses[seed] = {}
        loss_hashes[str(seed)] = {}
        training_report_hashes[str(seed)] = {}
        checkpoint_artifact_hashes[str(seed)] = {}
        checkpoint_state_hashes[str(seed)] = {}
        for policy in POLICIES:
            report_path = run_root / f"seed-{seed}" / f"{policy}.json"
            loss_path = artifact_root / f"seed-{seed}" / f"{policy}-nll.npz"
            training_report_path = (
                training_run_root / f"seed-{seed}" / f"{policy}.json"
            )
            checkpoint_path = checkpoint_root / f"seed-{seed}" / f"{policy}.pt"
            if not report_path.exists() or not loss_path.exists():
                raise FileNotFoundError(f"missing OOD result for seed {seed}/{policy}")
            if not training_report_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"missing primary training evidence for seed {seed}/{policy}"
                )
            report = _read_json(report_path)
            if report["seed"] != seed or report["policy"] != policy:
                raise ValueError(f"OOD identity mismatch in {report_path}")
            _validate_confirmation_report_binding(
                report,
                manifest,
                seed=seed,
                policy=policy,
                authorization=confirmation_authorization,
            )
            training_report = _read_json(training_report_path)
            if (
                training_report.get("seed") != seed
                or training_report.get("policy") != policy
                or training_report.get("parameters") != EXPECTED_PARAMETERS
                or training_report.get("model_spec")
                != PHASE3_MODEL_SPEC.to_dict()
                or training_report.get("optimization_spec")
                != PHASE3_OPTIMIZATION_SPEC.to_dict()
            ):
                raise ValueError(
                    f"primary training report mismatch for seed {seed}/{policy}"
                )
            training_report_hash = _sha256(training_report_path)
            checkpoint_artifact_hash = _sha256(checkpoint_path)
            checkpoint_state_hash = _checkpoint_state_sha256(checkpoint_path)
            if checkpoint_state_hash != training_report.get("trained_state_sha256"):
                raise ValueError(
                    f"primary checkpoint state mismatch for seed {seed}/{policy}"
                )
            expected_report_fields = {
                "parameters": EXPECTED_PARAMETERS,
                "model_spec": PHASE3_MODEL_SPEC.to_dict(),
                "global_max_position_embeddings": GLOBAL_POSITION_LIMIT,
                "stream_selected_sha256": source_provenance[
                    "selected_stream_sha256"
                ],
                "source_file_sha256": source_provenance["source_file_sha256"],
                "patch_matrix_sha256": source_provenance[
                    "patch_matrix_sha256"
                ][policy],
                "training_report_artifact_sha256": training_report_hash,
                "checkpoint_artifact_sha256": checkpoint_artifact_hash,
                "checkpoint_state_sha256": checkpoint_state_hash,
                "training_report_state_sha256": checkpoint_state_hash,
            }
            for key, expected in expected_report_fields.items():
                if report.get(key) != expected:
                    raise ValueError(
                        f"OOD provenance mismatch ({key}) in {report_path}"
                    )
            if (
                report["checkpoint_state_sha256"]
                != report["training_report_state_sha256"]
            ):
                raise ValueError(f"checkpoint integrity failed in {report_path}")
            if report["patch_diagnostics"]["mean_data_patches"] != 86.0:
                raise ValueError(f"OOD exact-rate failed in {report_path}")
            if (
                report["patch_diagnostics"].get("examples") != expected_examples
                or report["patch_diagnostics"]["minimum_data_patches"] != 86
                or report["patch_diagnostics"]["maximum_data_patches"] != 86
                or report["patch_diagnostics"]["padding_slots"] != 0
            ):
                raise ValueError(f"OOD row-level exact-rate failed in {report_path}")
            with np.load(loss_path) as archive:
                if archive.files != ["sequence_nll_nats"]:
                    raise ValueError(f"unexpected loss keys in {loss_path}")
                values = archive["sequence_nll_nats"].astype(np.float64)
            if (
                values.shape != (expected_examples,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError(f"invalid OOD sequence losses in {loss_path}")
            predicted_bytes = int(report["evaluation"]["predicted_bytes"])
            if (
                report["evaluation"].get("examples") != expected_examples
                or predicted_bytes != expected_examples * TARGETS_PER_SEQUENCE
            ):
                raise ValueError(f"OOD predicted-byte mismatch in {report_path}")
            reconstructed_bpb = float(values.sum()) / (
                predicted_bytes * math.log(2)
            )
            if not math.isclose(
                reconstructed_bpb,
                float(report["evaluation"]["bpb"]),
                abs_tol=1e-7,
            ):
                raise ValueError(f"OOD absolute loss/report mismatch in {report_path}")
            reports[seed][policy] = report
            losses[seed][policy] = values
            loss_hashes[str(seed)][policy] = _sha256(loss_path)
            training_report_hashes[str(seed)][policy] = training_report_hash
            checkpoint_artifact_hashes[str(seed)][policy] = (
                checkpoint_artifact_hash
            )
            checkpoint_state_hashes[str(seed)][policy] = checkpoint_state_hash

    quality = {
        policy: numeric_summary(
            [reports[seed][policy]["evaluation"]["bpb"] for seed in seeds]
        )
        for policy in POLICIES
    }
    contrasts: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        effects = [
            reports[seed][left]["evaluation"]["bpb"]
            - reports[seed][right]["evaluation"]["bpb"]
            for seed in seeds
        ]
        differences = [losses[seed][left] - losses[seed][right] for seed in seeds]
        for seed, expected, difference in zip(
            seeds, effects, differences, strict=True
        ):
            reconstructed_effect = float(difference.mean()) / (
                TARGETS_PER_SEQUENCE * math.log(2)
            )
            if not math.isclose(expected, reconstructed_effect, abs_tol=2e-5):
                raise ValueError(
                    f"OOD report/loss mismatch for {name}/seed-{seed}"
                )
        estimates = hierarchical_paired_bootstrap_estimates(
            differences,
            targets_per_sequence=TARGETS_PER_SEQUENCE,
            repetitions=args.bootstrap_repetitions,
            seed=20_260_824 + index,
        )
        lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
        contrasts[name] = {
            "left_policy": left,
            "right_policy": right,
            "difference_direction": "left_minus_right; negative favors left",
            "seed_order": list(seeds),
            "paired_differences_bpb": effects,
            "negative_seed_count": int(sum(value < 0 for value in effects)),
            "paired_t_95_interval": paired_t_interval(effects).to_dict(),
            "hierarchical_bootstrap_95_interval": {
                "repetitions": args.bootstrap_repetitions,
                "seed": 20_260_824 + index,
                "resampling_design": "crossed seeds x shared test sequences",
                "mean": float(estimates.mean()),
                "median": float(median),
                "lower": float(lower),
                "upper": float(upper),
            },
        }

    reported_patch_hashes = {
        policy: sorted(
            {reports[seed][policy]["patch_matrix_sha256"] for seed in seeds}
        )
        for policy in POLICIES
    }
    if any(len(values) != 1 for values in reported_patch_hashes.values()):
        raise ValueError("OOD structural matrices are not seed-independent")
    if any(
        values != [source_provenance["patch_matrix_sha256"][policy]]
        for policy, values in reported_patch_hashes.items()
    ):
        raise ValueError("OOD structural matrices differ from reconstruction")
    gate = ood_gate(
        {
            name: value["paired_t_95_interval"]["mean"]
            for name, value in contrasts.items()
        }
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "run_manifest": manifest,
        "confirmation_authorization": confirmation_authorization,
        "seeds": list(seeds),
        "policies": list(POLICIES),
        "quality": quality,
        "contrasts": contrasts,
        "gate_i_ood_guard": gate,
        "integrity": {
            "all_integrity_checks_pass": True,
            "source_artifact_matches_manifest": True,
            "selected_stream_matches_manifest": True,
            "boundary_stream_matches_manifest": True,
            "all_training_report_artifact_hashes_match": True,
            "all_checkpoint_artifact_hashes_match": True,
            "all_checkpoint_state_hashes_match": True,
            "all_patch_matrices_match_reconstruction": True,
            "all_policies_exactly_86_patches": True,
            "structural_matrices_seed_independent": True,
            "source_file_sha256": source_provenance["source_file_sha256"],
            "selected_stream_sha256": source_provenance[
                "selected_stream_sha256"
            ],
            "codepoint_boundaries_sha256": source_provenance[
                "codepoint_boundaries_sha256"
            ],
            "patch_matrix_sha256": source_provenance["patch_matrix_sha256"],
            "training_report_artifact_sha256": training_report_hashes,
            "checkpoint_artifact_sha256": checkpoint_artifact_hashes,
            "checkpoint_state_sha256": checkpoint_state_hashes,
            "loss_artifact_sha256": loss_hashes,
            "raw_text_promoted": False,
        },
        "interpretation_guardrail": (
            "Leipzig is a public domain-transfer guard, not a guaranteed "
            "contamination-free benchmark or superiority test."
        ),
    }
    _write_json(args.output, summary)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/phase3-ood")
    parser.add_argument("--artifact-root", default="artifacts/phase3-ood")
    parser.add_argument("--training-run-root", default="runs/phase3")
    parser.add_argument("--checkpoint-root", default="artifacts/phase3")
    parser.add_argument(
        "--data-root",
        default="data/processed/leipzig-wikipedia-100k-controls",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase3-ood/summary.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--confirmation-authorization-summary", type=Path)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
