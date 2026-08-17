#!/usr/bin/env python3
"""Apply the sealed v5r3 summary-only seed-axis counter correction.

The plan-bound producer and validator both operate on one seed at a time with
counter shape ``(prompt, repetition)``.  The plan-bound final summarizer passed
the whole ``(seed, prompt, repetition)`` array to that validator.  This wrapper
leaves every plan-bound file and every statistic unchanged and applies the
original validator independently to each sealed seed slice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

import numpy as np

from jamoflow.inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_MEASURED_CASES,
    ACTUAL_INFERENCE_V5_REPETITIONS,
    ACTUAL_INFERENCE_V5_SUMMARY_PATH,
    RUNTIME_COUNTER_NAMES,
    canonical_sha256,
    is_sha256,
    validate_runtime_counter_arrays,
)
from jamoflow.inference_final_authorization_v2 import FINAL_SEEDS


ORIGINAL_SUMMARY_PATH = Path("scripts/summarize_inference_actual_v5.py")
CORRECTION_PATH = Path(
    "data/manifests/phase3-inference-actual-v5r3-summary-counter-fix.json"
)
OUTPUT_PATH = Path(ACTUAL_INFERENCE_V5_SUMMARY_PATH)
CORRECTION_KIND = "phase3_inference_actual_v5r3_summary_counter_fix_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_head_identity(path: Path) -> dict[str, str]:
    blob = subprocess.run(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        check=False,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "rev-list", "-1", "HEAD", "--", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        blob.returncode != 0
        or len(commit) != 40
        or not path.is_file()
        or path.is_symlink()
        or path.read_bytes() != blob.stdout
    ):
        raise ValueError(f"summary counter correction is not an exact HEAD blob: {path}")
    return {
        "git_commit": commit,
        "path": path.as_posix(),
        "sha256": hashlib.sha256(blob.stdout).hexdigest(),
    }


def validate_correction_manifest(payload: Mapping[str, Any]) -> None:
    expected = {
        "adapter_contract",
        "correction_files_sha256",
        "correction_sha256",
        "failure",
        "kind",
        "original_plan_bound_files_sha256",
        "plan_artifact_sha256",
        "plan_sha256",
        "result_inputs",
        "schema_version",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload.get("kind") != CORRECTION_KIND
        or payload.get("schema_version") != 1
        or not is_sha256(payload.get("plan_artifact_sha256"))
        or not is_sha256(payload.get("plan_sha256"))
        or payload.get("failure")
        != {
            "exception": "ValueError: emitted-output counter shape differs",
            "stage": "session_runtime_counter_validation_before_latency_summary",
            "summary_published": False,
        }
        or payload.get("adapter_contract")
        != {
            "input_shape": [5, 64, 5],
            "operation": "apply_original_validator_independently_to_each_seed_slice",
            "seed_order": list(FINAL_SEEDS),
            "slice_shape": [64, 5],
            "statistics_changed": False,
            "timing_artifacts_changed": False,
        }
        or payload.get("result_inputs")
        != {
            "aggregate_latency_inspected": False,
            "gate_result_inspected": False,
            "individual_latency_values_inspected": False,
            "only_array_keys_shapes_dtypes_and_exception_used": True,
        }
        or not isinstance(payload.get("original_plan_bound_files_sha256"), Mapping)
        or not isinstance(payload.get("correction_files_sha256"), Mapping)
        or payload.get("correction_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "correction_sha256"
            }
        )
    ):
        raise ValueError("summary counter correction manifest differs")
    expected_original = {
        "scripts/summarize_inference_actual_v5.py",
        "src/jamoflow/inference_actual_v5.py",
    }
    expected_correction = {
        "docs/90-v5r3-summary-counter-shape-debug.md",
        "scripts/summarize_inference_actual_v5r3_counter_fix.py",
        "tests/test_summarize_inference_actual_v5r3_counter_fix.py",
    }
    if (
        set(payload["original_plan_bound_files_sha256"]) != expected_original
        or set(payload["correction_files_sha256"]) != expected_correction
    ):
        raise ValueError("summary counter correction file set differs")
    for group in (
        payload["original_plan_bound_files_sha256"],
        payload["correction_files_sha256"],
    ):
        for path, expected_sha256 in group.items():
            if not is_sha256(expected_sha256) or _sha256(Path(path)) != expected_sha256:
                raise ValueError(f"summary counter correction file differs: {path}")


def load_correction_identity() -> dict[str, Any]:
    identity = _tracked_head_identity(CORRECTION_PATH)
    payload = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    validate_correction_manifest(payload)
    return {
        "artifact": identity,
        "correction_sha256": payload["correction_sha256"],
        "kind": CORRECTION_KIND,
    }


def validate_session_runtime_counter_arrays(
    counters: Mapping[str, np.ndarray],
    *,
    requires_entropy_router: bool,
    mode: str,
    emitted_output_bytes: np.ndarray,
    base_validator: Callable[..., None] = validate_runtime_counter_arrays,
) -> None:
    """Validate a full session by preserving the original per-seed contract."""

    emitted = np.asarray(emitted_output_bytes)
    expected_shape = (
        len(FINAL_SEEDS),
        ACTUAL_INFERENCE_V5_MEASURED_CASES,
        ACTUAL_INFERENCE_V5_REPETITIONS,
    )
    if (
        emitted.shape != expected_shape
        or set(counters) != set(RUNTIME_COUNTER_NAMES)
        or any(np.asarray(value).shape != expected_shape for value in counters.values())
    ):
        raise ValueError("corrected session runtime counter shape differs")
    for seed_index, _ in enumerate(FINAL_SEEDS):
        base_validator(
            {
                name: np.asarray(counters[name])[seed_index]
                for name in RUNTIME_COUNTER_NAMES
            },
            requires_entropy_router=requires_entropy_router,
            mode=mode,
            emitted_output_bytes=emitted[seed_index],
        )


def _load_original_summary_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "jamoflow_plan_bound_actual_summary_v5r3",
        ORIGINAL_SUMMARY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the plan-bound actual summary")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run() -> int:
    correction_identity = load_correction_identity()
    summary = _load_original_summary_module()
    original_validator = summary.validate_runtime_counter_arrays
    original_publish = summary.publish_no_clobber

    def corrected_validator(
        counters: Mapping[str, np.ndarray],
        *,
        requires_entropy_router: bool,
        mode: str,
        emitted_output_bytes: np.ndarray,
    ) -> None:
        validate_session_runtime_counter_arrays(
            counters,
            requires_entropy_router=requires_entropy_router,
            mode=mode,
            emitted_output_bytes=emitted_output_bytes,
            base_validator=original_validator,
        )

    def corrected_publish(path: Path, content: bytes) -> None:
        if Path(path) == summary.OUTPUT_PATH:
            payload = json.loads(content.decode("utf-8"))
            payload["summary_counter_shape_correction"] = correction_identity
            payload.pop("summary_sha256", None)
            payload["summary_sha256"] = canonical_sha256(payload)
            content = summary._json_bytes(payload)
        original_publish(path, content)

    summary.validate_runtime_counter_arrays = corrected_validator
    summary.publish_no_clobber = corrected_publish
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("summary_counter_shape_correction") != correction_identity:
            raise ValueError("existing actual summary lacks the sealed counter correction")
    return int(summary.run())


if __name__ == "__main__":
    raise SystemExit(run())
