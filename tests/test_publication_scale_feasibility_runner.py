from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from jamoflow.publication_scale import (
    PUBLICATION_EXPECTED_PARAMETERS,
    PUBLICATION_PROJECTED_TRAIN_STEPS,
    publication_model_spec,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "benchmark_publication_scale_feasibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_publication_scale_feasibility",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PublicationScaleFeasibilityRunnerTests(unittest.TestCase):
    target = 50
    rate = 64
    actual_sha256 = "a" * 64
    selection_sha256 = "b" * 64
    git_commit = "c" * 40
    data_context = {"selected_stream_sha256": "d" * 64}

    def _report(self) -> dict:
        seconds = [1.0, 2.0, 3.0]
        projected = 2.0 * PUBLICATION_PROJECTED_TRAIN_STEPS / 3600
        snapshots = [
            {
                "current_allocated_bytes": 10 + index,
                "driver_allocated_bytes": 20 + index,
                "recommended_max_memory_bytes": 100,
            }
            for index in range(7)
        ]
        return {
            "schema_version": 1,
            "git_commit": self.git_commit,
            "source_tree_clean": True,
            "authorization_summary_sha256": self.actual_sha256,
            "selection_sha256": self.selection_sha256,
            "target_millions": self.target,
            "model_spec": publication_model_spec(
                self.target,
                self.rate,
            ).to_dict(),
            "parameter_count": PUBLICATION_EXPECTED_PARAMETERS[self.target],
            "completed": True,
            "finite_steps": True,
            "training": {
                "batch_size": MODULE.PUBLICATION_BATCH_SIZE,
                "finite": True,
                "warmup_steps": MODULE.TRAIN_WARMUP_STEPS,
                "measurement_steps": MODULE.TRAIN_MEASUREMENT_STEPS,
                "measurement_seconds": seconds,
                "median_step_seconds": 2.0,
                "projected_steps_for_256m_bytes": (
                    PUBLICATION_PROJECTED_TRAIN_STEPS
                ),
                "projected_hours_per_model": projected,
            },
            "evaluation": {
                "batch_size": MODULE.PUBLICATION_EVALUATION_BATCH_SIZE,
                "elapsed_seconds": 1.0,
                "finite": True,
            },
            "incremental": {
                "prompt_bytes": 128,
                "decode_bytes": 1,
                "elapsed_seconds": 1.0,
                "finite": True,
            },
            "memory_snapshots": snapshots,
            "maximum_driver_allocated_bytes": 26,
            "recommended_max_memory_bytes": 100,
            "data_context": self.data_context,
            "quality_used_for_selection": False,
            "environment": {"device": "mps"},
        }

    def _validate(self, report: dict) -> None:
        MODULE._validate_worker_report(
            report,
            target=self.target,
            actual_sha256=self.actual_sha256,
            selection_sha256=self.selection_sha256,
            rate=self.rate,
            data_context=self.data_context,
            git_commit=self.git_commit,
        )

    def test_worker_report_reconstructs_projection_and_memory(self) -> None:
        self._validate(self._report())
        altered = deepcopy(self._report())
        altered["training"]["projected_hours_per_model"] += 0.01
        with self.assertRaisesRegex(ValueError, "projection"):
            self._validate(altered)

    def test_stage_finite_flags_are_independent_and_reconciled(self) -> None:
        report = self._report()
        report["evaluation"]["finite"] = False
        report["finite_steps"] = False
        self._validate(report)
        report["finite_steps"] = True
        with self.assertRaisesRegex(ValueError, "runtime"):
            self._validate(report)

    def test_failure_report_is_strict_json_and_safe_to_summarize(self) -> None:
        report = MODULE._failed_worker_report(
            target=self.target,
            actual_sha256=self.actual_sha256,
            selection_sha256=self.selection_sha256,
            rate=self.rate,
            data_context=self.data_context,
            git_commit=self.git_commit,
            returncode=1,
            stdout_tail="",
            stderr_tail="out of memory",
            validation_error=None,
        )
        self._validate(report)
        result = MODULE._result_from_report(self.target, report)
        serialized = json.dumps(result.to_dict(), allow_nan=False)
        self.assertIn('"projected_hours_per_model": null', serialized)
        self.assertIn('"memory_fraction": null', serialized)

    def test_json_helpers_reject_nonfinite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(ValueError):
                MODULE._write_json(path, {"value": float("inf")})
            path.write_text('{"value": Infinity}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE._read_json(path)

    def test_compact_authorization_requires_valid_output_summary_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            quality_path = root / "quality.json"
            actual_path = root / "actual.json"
            candidate = {
                "policy": MODULE.conversion_policy("whitespace", self.rate),
                "patch_count": self.rate,
            }
            MODULE._write_json(
                selection_path,
                {"candidate": candidate, "reference": {"policy": "fixed"}},
            )
            selection_hash = MODULE._sha256(selection_path)
            MODULE._write_json(
                quality_path,
                {
                    "selection": {"sha256": selection_hash},
                    "candidate": candidate,
                    "quality_noninferiority": {"overall_pass": True},
                    "integrity": {"all_integrity_checks_pass": True},
                },
            )
            MODULE._write_json(
                actual_path,
                {
                    "schema_version": 2,
                    "integrity": {"all_integrity_checks_pass": True},
                    "compact_actual_inference_gate": {"overall_pass": True},
                    "selection": {"sha256": selection_hash},
                    "candidate": candidate,
                    "quality_summary": {
                        "path": str(quality_path),
                        "sha256": MODULE._sha256(quality_path),
                    },
                },
            )
            _, _, rate = MODULE._validate_authorization(
                actual_path,
                selection_path,
            )
            self.assertEqual(rate, self.rate)

            stale = MODULE._read_json(actual_path)
            stale["schema_version"] = 1
            MODULE._write_json(actual_path, stale)
            with self.assertRaisesRegex(ValueError, "compact gate"):
                MODULE._validate_authorization(actual_path, selection_path)


if __name__ == "__main__":
    unittest.main()
