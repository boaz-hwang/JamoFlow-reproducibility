#!/usr/bin/env python3
"""Verify five-seed quality noninferiority before actual inference timing."""

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

from jamoflow.compute_conversion import conversion_policy
from jamoflow.document_inference import reconstruct_document_window_map
from jamoflow.inference_quality import inference_quality_noninferiority
from jamoflow.neural_data import build_neural_stream
from jamoflow.phase3 import PHASE3_MODEL_SPEC


SEEDS = (1729, 2718, 31415, 57721, 65537)
INITIAL_SEEDS = SEEDS[:3]
TARGETS_PER_SEQUENCE = PHASE3_MODEL_SPEC.sequence_length - 1
PRIMARY_PHASE3_POLICIES = (
    "fixed_byte_6",
    "causal_codepoint_grid",
    "causal_whitespace_grid",
)
ARTIFACT_EVIDENCE_FIELDS = (
    "training_report_artifact_sha256",
    "loss_artifact_sha256",
    "checkpoint_artifact_sha256",
    "checkpoint_state_sha256",
)


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


def _checkpoint_state_sha256(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint is not a non-empty state dict: {path}")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"unexpected checkpoint entry: {path}")
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


def _validate_summaries(
    selection: dict[str, Any],
    phase3: dict[str, Any],
    conversion: dict[str, Any],
) -> tuple[str, str, str]:
    if (
        selection.get("selection_uses_latency") is not False
        or selection.get("status")
        != "locked_before_latency_pending_five_seed_quality"
        or tuple(selection.get("seed_order", [])) != INITIAL_SEEDS
    ):
        raise ValueError("inference comparator was not locked before timing")
    candidate = selection.get("candidate", {})
    reference = selection.get("reference", {})
    candidate_policy = candidate.get("policy")
    reference_policy = reference.get("policy")
    reference_family = reference.get("model_family")
    rate = candidate.get("patch_count")
    if (
        not isinstance(candidate_policy, str)
        or not isinstance(reference_policy, str)
        or reference_family not in {"phase3", "compute_conversion"}
        or not isinstance(rate, int)
        or candidate_policy != conversion_policy("whitespace", rate)
    ):
        raise ValueError("locked inference policy identities are malformed")
    if (
        tuple(phase3.get("seeds", [])) != SEEDS
        or phase3.get("integrity", {}).get("all_integrity_checks_pass") is not True
        or phase3.get("gate_j", {}).get("overall_pass") is not True
        or reference_family == "phase3"
        and reference_policy not in phase3.get("policies", [])
    ):
        raise ValueError("five-seed Phase 3 Gate J evidence is incomplete")
    selected_rate = conversion.get("calibration_rate_selection", {}).get(
        "selected_rate"
    )
    if (
        conversion.get("stage") != "confirmation"
        or tuple(conversion.get("seeds", [])) != SEEDS
        or conversion.get("integrity", {}).get("all_integrity_checks_pass")
        is not True
        or conversion.get("initial_conversion_gate", {}).get("overall_pass")
        is not True
        or conversion.get("confirmation_same_rate_gate", {}).get("overall_pass")
        is not True
        or selected_rate != rate
        or candidate_policy not in conversion.get("policies", [])
        or reference_family == "compute_conversion"
        and reference_policy != conversion_policy("codepoint", rate)
    ):
        raise ValueError("five-seed conversion confirmation is incomplete")
    return candidate_policy, reference_policy, reference_family


def _validate_selection_lineage(
    selection: dict[str, Any],
    phase3: dict[str, Any],
    conversion: dict[str, Any],
) -> None:
    initial_phase3 = selection["phase3_initial_summary"]
    initial_conversion = selection["conversion_initial_summary"]
    locked_summaries: list[dict[str, Any]] = []
    for item in (initial_phase3, initial_conversion):
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise ValueError("locked initial summary artifact changed")
        locked_summaries.append(_read_json(path))
    _validate_locked_initial_evidence(
        selection,
        locked_summaries[0],
        locked_summaries[1],
        phase3,
        conversion,
    )
    confirmation_selection = conversion.get("selection_summary", {})
    if confirmation_selection.get("sha256") != initial_conversion["sha256"]:
        raise ValueError("conversion confirmation used a different initial selection")
    phase3_source = phase3["run_manifest"]
    conversion_source = conversion["integrity"]["source_context"]
    if (
        phase3_source.get("source_artifact")
        != conversion_source.get("source_artifact")
        or phase3_source.get("source_integrity_artifact")
        != conversion_source.get("source_integrity_artifact")
    ):
        raise ValueError("quality families use different source artifacts")
    for split in ("train", "calibration", "test"):
        left = phase3_source["streams"][split]
        right = conversion_source["streams"][split]
        if (
            left.get("selected_stream_sha256")
            != right.get("selected_stream_sha256")
            or left.get("sequence_count") != right.get("sequence_count")
        ):
            raise ValueError(f"quality families use different {split} streams")


def _validate_locked_initial_evidence(
    selection: dict[str, Any],
    locked_phase3: dict[str, Any],
    locked_conversion: dict[str, Any],
    phase3: dict[str, Any],
    conversion: dict[str, Any],
) -> None:
    """Bind final summaries to the exact initial artifacts used for selection."""

    for key in (
        "source_artifact",
        "source_integrity_artifact",
        "model_spec",
        "optimization_spec",
        "limits",
        "streams",
    ):
        if locked_phase3.get("run_manifest", {}).get(key) != phase3.get(
            "run_manifest", {}
        ).get(key):
            raise ValueError(f"locked and final Phase 3 context differs: {key}")
    if locked_conversion.get("integrity", {}).get(
        "source_context"
    ) != conversion.get("integrity", {}).get("source_context"):
        raise ValueError("locked and final conversion source context differs")

    reference = selection["reference"]
    phase3_policies = set(PRIMARY_PHASE3_POLICIES)
    if reference.get("model_family") == "phase3":
        phase3_policies.add(str(reference["policy"]))
    for seed in INITIAL_SEEDS:
        locked_seed = locked_phase3.get("integrity", {}).get("by_seed", {}).get(
            str(seed), {}
        )
        final_seed = phase3.get("integrity", {}).get("by_seed", {}).get(
            str(seed), {}
        )
        for field in ARTIFACT_EVIDENCE_FIELDS:
            for policy in phase3_policies:
                if locked_seed.get(field, {}).get(policy) != final_seed.get(
                    field, {}
                ).get(policy):
                    raise ValueError(
                        "locked Phase 3 evidence changed: "
                        f"{seed}/{policy}/{field}"
                    )
        if reference.get("policy") in {
            "entropy_threshold_full",
            "entropy_threshold_codepoint",
        } and locked_seed.get("router_and_threshold_cache") != final_seed.get(
            "router_and_threshold_cache"
        ):
            raise ValueError(f"locked entropy-router evidence changed: {seed}")

    rate = int(selection["candidate"]["patch_count"])
    conversion_policies = {
        str(selection["candidate"]["policy"]),
        conversion_policy("codepoint", rate),
    }
    for seed in INITIAL_SEEDS:
        locked_artifacts = (
            locked_conversion.get("integrity", {})
            .get("by_seed", {})
            .get(str(seed), {})
            .get("conversion_artifacts", {})
        )
        final_artifacts = (
            conversion.get("integrity", {})
            .get("by_seed", {})
            .get(str(seed), {})
            .get("conversion_artifacts", {})
        )
        for policy in conversion_policies:
            if locked_artifacts.get(policy) != final_artifacts.get(policy):
                raise ValueError(
                    f"locked conversion evidence changed: {seed}/{policy}"
                )


def _recorded_evidence(
    summary: dict[str, Any],
    family: str,
    seed: int,
    policy: str,
) -> dict[str, str]:
    by_seed = summary["integrity"]["by_seed"][str(seed)]
    if family == "compute_conversion":
        evidence = by_seed["conversion_artifacts"][policy]
        return {
            "training_report_artifact_sha256": evidence[
                "training_report_artifact_sha256"
            ],
            "loss_artifact_sha256": evidence["loss_artifact_sha256"],
            "checkpoint_artifact_sha256": evidence[
                "checkpoint_artifact_sha256"
            ],
            "checkpoint_state_sha256": evidence["checkpoint_state_sha256"],
        }
    return {
        "training_report_artifact_sha256": by_seed[
            "training_report_artifact_sha256"
        ][policy],
        "loss_artifact_sha256": by_seed["loss_artifact_sha256"][policy],
        "checkpoint_artifact_sha256": by_seed[
            "checkpoint_artifact_sha256"
        ][policy],
        "checkpoint_state_sha256": by_seed["checkpoint_state_sha256"][policy],
    }


def _load_policy_losses(
    summary: dict[str, Any],
    family: str,
    policy: str,
    run_root: Path,
    artifact_root: Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    losses: dict[int, np.ndarray] = {}
    evidence_output: dict[str, Any] = {}
    reported_bpb: list[float] = []
    for seed in SEEDS:
        report_path = run_root / f"seed-{seed}" / f"{policy}.json"
        loss_path = artifact_root / f"seed-{seed}" / f"{policy}-test-nll.npz"
        checkpoint_path = artifact_root / f"seed-{seed}" / f"{policy}.pt"
        recorded = _recorded_evidence(summary, family, seed, policy)
        actual = {
            "training_report_artifact_sha256": _sha256(report_path),
            "loss_artifact_sha256": _sha256(loss_path),
            "checkpoint_artifact_sha256": _sha256(checkpoint_path),
            "checkpoint_state_sha256": _checkpoint_state_sha256(checkpoint_path),
        }
        if actual != recorded:
            raise ValueError(f"quality evidence mismatch: {family}/{seed}/{policy}")
        report = _read_json(report_path)
        if report.get("seed") != seed or report.get("policy") != policy:
            raise ValueError(f"quality report identity mismatch: {seed}/{policy}")
        with np.load(loss_path, allow_pickle=False) as archive:
            if archive.files != ["sequence_nll_nats"]:
                raise ValueError("quality loss keys mismatch")
            stored = archive["sequence_nll_nats"]
            if stored.dtype != np.float32:
                raise ValueError("quality loss dtype mismatch")
            values = stored.astype(np.float64)
        if (
            values.ndim != 1
            or not len(values)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("quality loss vector is invalid")
        bpb = float(values.mean()) / (TARGETS_PER_SEQUENCE * math.log(2))
        if not math.isclose(
            bpb,
            float(report["evaluation"]["test"]["bpb"]),
            abs_tol=1e-7,
        ):
            raise ValueError(f"quality BPB does not reconstruct: {seed}/{policy}")
        losses[seed] = values
        reported_bpb.append(bpb)
        evidence_output[str(seed)] = actual
    quality = summary["quality"][policy]
    summary_mean = (
        quality["test_bpb"]["mean"]
        if family == "compute_conversion"
        else quality["mean"]
    )
    if not math.isclose(
        float(np.mean(reported_bpb)),
        float(summary_mean),
        abs_tol=1e-12,
    ):
        raise ValueError(f"summary quality mean mismatch: {family}/{policy}")
    return losses, evidence_output


def run(args: argparse.Namespace) -> int:
    selection_path = Path(args.selection)
    phase3_path = Path(args.phase3_confirmation_summary)
    conversion_path = Path(args.conversion_confirmation_summary)
    selection = _read_json(selection_path)
    phase3 = _read_json(phase3_path)
    conversion = _read_json(conversion_path)
    candidate, reference, reference_family = _validate_summaries(
        selection,
        phase3,
        conversion,
    )
    _validate_selection_lineage(selection, phase3, conversion)
    source_path = Path(args.data_root) / "ko.jsonl"
    source_manifest = phase3["run_manifest"]
    expected_source = source_manifest["source_artifact"]
    actual_source = {
        "filename": "ko.jsonl",
        "bytes": source_path.stat().st_size,
        "sha256": _sha256(source_path),
    }
    if actual_source != expected_source:
        raise ValueError("quality document map source differs from Phase 3")
    test_limit = int(source_manifest["limits"]["test"])
    test_stream = build_neural_stream(
        source_path,
        language="ko",
        split="test",
        byte_limit=test_limit,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    recorded_test_stream = source_manifest["streams"]["test"]
    if (
        hashlib.sha256(test_stream.data).hexdigest()
        != recorded_test_stream["selected_stream_sha256"]
        or test_stream.sequence_count != recorded_test_stream["sequence_count"]
    ):
        raise ValueError("quality document map stream differs from Phase 3")
    document_window_map = reconstruct_document_window_map(
        source_path,
        split="test",
        byte_limit=test_limit,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
        expected_stream=test_stream.data,
    )
    candidate_losses, candidate_evidence = _load_policy_losses(
        conversion,
        "compute_conversion",
        candidate,
        Path(args.conversion_run_root),
        Path(args.conversion_artifact_root),
    )
    if reference_family == "compute_conversion":
        reference_summary = conversion
        reference_run_root = Path(args.conversion_run_root)
        reference_artifact_root = Path(args.conversion_artifact_root)
    else:
        reference_summary = phase3
        reference_run_root = Path(args.phase3_run_root)
        reference_artifact_root = Path(args.phase3_artifact_root)
    reference_losses, reference_evidence = _load_policy_losses(
        reference_summary,
        reference_family,
        reference,
        reference_run_root,
        reference_artifact_root,
    )
    gate = inference_quality_noninferiority(
        candidate_losses,
        reference_losses,
        seed_order=SEEDS,
        candidate_policy=candidate,
        reference_policy=reference,
        targets_per_sequence=TARGETS_PER_SEQUENCE,
        document_window_map=document_window_map,
        bootstrap_repetitions=args.bootstrap_repetitions,
    ).to_dict()
    output_path = Path(args.output)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_git_commit": _git_commit(),
        "selection": {"path": str(selection_path), "sha256": _sha256(selection_path)},
        "phase3_confirmation_summary": {
            "path": str(phase3_path),
            "sha256": _sha256(phase3_path),
        },
        "conversion_confirmation_summary": {
            "path": str(conversion_path),
            "sha256": _sha256(conversion_path),
        },
        "candidate": selection["candidate"],
        "reference": selection["reference"],
        "quality_noninferiority": gate,
        "integrity": {
            "all_integrity_checks_pass": True,
            "same_public_test_windows_across_families": True,
            "source_artifact": actual_source,
            "test_stream_sha256": recorded_test_stream[
                "selected_stream_sha256"
            ],
            "document_window_map": document_window_map.metadata(),
            "all_loss_metrics_and_checkpoint_hashes_reconstructed": True,
            "candidate_evidence": candidate_evidence,
            "reference_evidence": reference_evidence,
        },
        "interpretation_guardrail": (
            "Passing this gate permits actual timing but is not itself evidence "
            "of inference speed or downstream quality."
        ),
    }
    _write_json(output_path, payload)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        default="results/phase3-inference-selection/selection.json",
    )
    parser.add_argument(
        "--phase3-confirmation-summary",
        default="results/phase3-all-confirmation/summary.json",
    )
    parser.add_argument(
        "--conversion-confirmation-summary",
        default="results/phase3-compute-conversion/confirmation-summary.json",
    )
    parser.add_argument("--phase3-run-root", default="runs/phase3")
    parser.add_argument("--phase3-artifact-root", default="artifacts/phase3")
    parser.add_argument(
        "--conversion-run-root",
        default="runs/phase3-compute-conversion",
    )
    parser.add_argument(
        "--conversion-artifact-root",
        default="artifacts/phase3-compute-conversion",
    )
    parser.add_argument(
        "--output",
        default="results/phase3-inference-quality/summary.json",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed/hplt3-korean-phase3",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
