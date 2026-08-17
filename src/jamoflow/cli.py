"""Command-line interface for reproducible JamoFlow Phase 0 audits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import platform
from pathlib import Path
import sys
import time
from typing import Sequence

from .corpus import expand_input_paths, load_records, partition_records
from .entropy import ByteNGramModel
from .metrics import (
    ScoredRecord,
    average_patch_bytes,
    build_evaluation_context,
    evaluate_policy,
    make_bootstrap_weights,
)
from .patching import (
    BoundaryPolicy,
    CandidateEntropyPolicy,
    CJKIdeographPolicy,
    CodepointAlignedStridePolicy,
    EntropyPolicy,
    EojeolCappedPolicy,
    FixedStridePolicy,
    HangulSyllablePolicy,
    OrthographicCandidateEntropyPolicy,
    SpaceBytePolicy,
    assert_prefix_causal,
    calibrate_entropy_threshold,
)
from .report import write_reports
from .unicode_audit import audit_records


def _score_records(
    records: Sequence,
    model: ByteNGramModel,
) -> tuple[list[ScoredRecord], int]:
    started = time.perf_counter_ns()
    scored = [
        ScoredRecord(record=record, scores=tuple(model.score(record.raw)))
        for record in records
    ]
    return scored, time.perf_counter_ns() - started


def _causality_check(
    policies: Sequence[BoundaryPolicy],
    records: Sequence[ScoredRecord],
    max_records: int = 20,
    max_bytes: int = 256,
) -> int:
    checks = 0
    for policy in policies:
        for scored in records[:max_records]:
            data = scored.record.raw[:max_bytes]
            scores = scored.scores[: len(data)]
            assert_prefix_causal(policy, data, scores)
            checks += 1
    return checks


def run_audit(args: argparse.Namespace) -> int:
    resolved_paths = expand_input_paths(
        args.paths,
        include_suffixes=args.include_suffix,
    )
    records = [
        record
        for record in load_records(
            resolved_paths,
            corpus_format=args.format,
            text_field=args.text_field,
            plain_record_unit=args.plain_record_unit,
            deduplicate=args.deduplicate,
        )
        if record.raw
    ]
    if not records:
        raise SystemExit("no non-empty records were loaded")

    partitions = partition_records(records)
    for split_name in ("train", "calibration", "test"):
        if not partitions[split_name]:
            raise SystemExit(
                f"deterministic {split_name} split is empty; use more records"
            )

    model = ByteNGramModel(order=args.order, alpha=args.alpha)
    fit_started = time.perf_counter_ns()
    model.fit(record.raw for record in partitions["train"])
    fit_elapsed = time.perf_counter_ns() - fit_started

    calibration_records, calibration_score_elapsed = _score_records(
        partitions["calibration"], model
    )
    test_records, test_score_elapsed = _score_records(partitions["test"], model)
    scored_bytes = sum(
        len(scored.record.raw)
        for scored in [*calibration_records, *test_records]
    )

    rules: list[BoundaryPolicy] = [
        FixedStridePolicy(args.stride),
        CodepointAlignedStridePolicy(args.stride),
        SpaceBytePolicy(),
        EojeolCappedPolicy(args.eojeol_cap),
    ]
    if args.script == "hangul":
        rules.insert(3, HangulSyllablePolicy())
    elif args.script == "cjk":
        rules.insert(3, CJKIdeographPolicy())
    scored_calibration_pairs = [
        (scored.record.raw, scored.scores) for scored in calibration_records
    ]
    evaluation_context = build_evaluation_context(test_records)
    bootstrap_weights = make_bootstrap_weights(
        len(test_records),
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    metrics = []
    calibrations: list[dict[str, object]] = []
    causality_checks = 0
    for rule in rules:
        target_average = average_patch_bytes(calibration_records, rule)
        entropy_threshold = calibrate_entropy_threshold(
            scored_calibration_pairs,
            target_average_patch_bytes=target_average,
            candidate_only=False,
        )
        candidate_threshold = calibrate_entropy_threshold(
            scored_calibration_pairs,
            target_average_patch_bytes=target_average,
            candidate_only=True,
        )
        orthographic_policy = None
        orthographic_threshold = None
        if args.script != "none":
            orthographic_calibration_policy = OrthographicCandidateEntropyPolicy(
                threshold=float("inf"),
                script=args.script,
                max_patch_bytes=args.orthographic_cap,
            )
            orthographic_threshold = calibrate_entropy_threshold(
                scored_calibration_pairs,
                target_average_patch_bytes=target_average,
                candidate_policy=orthographic_calibration_policy,
            )
            orthographic_policy = OrthographicCandidateEntropyPolicy(
                threshold=orthographic_threshold,
                script=args.script,
                max_patch_bytes=args.orthographic_cap,
                label=f"orthographic_candidate_entropy_matched_{rule.name}",
            )
        entropy_policy = EntropyPolicy(
            threshold=entropy_threshold,
            min_patch_bytes=args.min_patch_bytes,
            label=f"entropy_matched_{rule.name}",
        )
        candidate_policy = CandidateEntropyPolicy(
            threshold=candidate_threshold,
            min_patch_bytes=args.min_patch_bytes,
            max_patch_bytes=args.hybrid_cap,
            label=f"candidate_entropy_matched_{rule.name}",
        )
        entropy_calibration_average = average_patch_bytes(
            calibration_records, entropy_policy
        )
        candidate_calibration_average = average_patch_bytes(
            calibration_records, candidate_policy
        )

        comparison_policies = [rule, entropy_policy, candidate_policy]
        roles = ["rule", "entropy_matched", "candidate_entropy_matched"]
        if orthographic_policy is not None:
            comparison_policies.append(orthographic_policy)
            roles.append("orthographic_candidate_entropy_matched")
        causality_checks += _causality_check(
            comparison_policies,
            calibration_records,
        )
        for role, policy in zip(roles, comparison_policies, strict=True):
            metrics.append(
                evaluate_policy(
                    test_records,
                    policy,
                    comparison_group=rule.name,
                    role=role,
                    runtime_repeats=args.runtime_repeats,
                    context=evaluation_context,
                    bootstrap_weights=bootstrap_weights,
                )
            )
        calibrations.append(
            {
                "group": rule.name,
                "target_average_patch_bytes": target_average,
                "entropy_threshold": entropy_threshold,
                "candidate_entropy_threshold": candidate_threshold,
                "entropy_calibration_average_patch_bytes": entropy_calibration_average,
                "candidate_calibration_average_patch_bytes": candidate_calibration_average,
                "orthographic_candidate_entropy_threshold": orthographic_threshold,
                "orthographic_candidate_calibration_average_patch_bytes": (
                    average_patch_bytes(calibration_records, orthographic_policy)
                    if orthographic_policy is not None
                    else None
                ),
            }
        )

    unicode_audit = audit_records(records)
    scoring_elapsed = calibration_score_elapsed + test_score_elapsed
    metadata: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_label": args.corpus_label,
        "interpretation_note": args.interpretation_note,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "input_files": ", ".join(str(Path(path)) for path in args.paths),
        "included_suffixes": (
            ", ".join(args.include_suffix)
            if args.include_suffix
            else "default text suffixes"
        ),
        "resolved_file_count": len(resolved_paths),
        "plain_record_unit": args.plain_record_unit,
        "records_total": len(records),
        "train_records": len(partitions["train"]),
        "calibration_records": len(partitions["calibration"]),
        "test_records": len(partitions["test"]),
        "training_bytes": model.training_bytes,
        "ngram_order": model.order,
        "alpha": model.alpha,
        "fit_seconds": fit_elapsed / 1_000_000_000,
        "scoring_seconds": scoring_elapsed / 1_000_000_000,
        "scoring_ns_per_byte": scoring_elapsed / scored_bytes
        if scored_bytes
        else 0.0,
        "causality_checks": causality_checks,
        "deduplicated": args.deduplicate,
        "reference_only": True,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
    }

    json_path, markdown_path = write_reports(
        args.output_dir,
        metadata,
        unicode_audit,
        metrics,
        calibrations,
    )
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jamoflow",
        description="Phase 0 audits for orthography-aware byte patching",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="run a corpus and boundary audit")
    audit.add_argument("paths", nargs="+", help="plain-text or JSONL corpus files")
    audit.add_argument(
        "--include-suffix",
        action="append",
        default=None,
        help=(
            "suffix to include while recursively expanding directories; "
            "repeat for multiple suffixes (for example: --include-suffix .md)"
        ),
    )
    audit.add_argument(
        "--format",
        choices=("auto", "plain", "jsonl"),
        default="auto",
    )
    audit.add_argument("--text-field", default="text")
    audit.add_argument(
        "--plain-record-unit",
        choices=("line", "file"),
        default="line",
        help="treat each plain-text line or each complete file as one split record",
    )
    audit.add_argument("--order", type=int, default=4)
    audit.add_argument("--alpha", type=float, default=0.1)
    audit.add_argument("--stride", type=int, default=6)
    audit.add_argument("--eojeol-cap", type=int, default=24)
    audit.add_argument(
        "--script",
        choices=("hangul", "cjk", "none"),
        default="hangul",
        help="script-specific rule and orthographic candidate family",
    )
    audit.add_argument(
        "--orthographic-cap",
        type=int,
        default=24,
        help="causal maximum patch size for script/delimiter candidate entropy",
    )
    audit.add_argument(
        "--hybrid-cap",
        type=int,
        default=None,
        help="optional causal maximum patch budget for candidate entropy",
    )
    audit.add_argument("--min-patch-bytes", type=int, default=1)
    audit.add_argument("--runtime-repeats", type=int, default=7)
    audit.add_argument("--bootstrap-repeats", type=int, default=0)
    audit.add_argument("--bootstrap-seed", type=int, default=1729)
    audit.add_argument(
        "--deduplicate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    audit.add_argument("--output-dir", default="results/phase0")
    audit.add_argument(
        "--corpus-label",
        default="unspecified convenience sample",
        help="human-readable corpus label written to the report",
    )
    audit.add_argument(
        "--interpretation-note",
        default=(
            "This convenience sample is not assumed to represent natural Korean."
        ),
        help="corpus-specific limitation written to the report",
    )
    audit.set_defaults(handler=run_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
