from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jamoflow.neural_model import build_main_model, build_router, parameter_count
from jamoflow.phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    merge_phase3_manifest,
    merge_phase3_ood_manifest,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from jamoflow.phase2_patching import (
    compact_whitespace_mask,
    validate_padded_patch_matrix,
)
from jamoflow.utf8 import prefix_boundary_mask


RUN_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_phase3.py"
RUN_SPEC = importlib.util.spec_from_file_location("run_phase3", RUN_SCRIPT)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
RUN_MODULE = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN_MODULE)


class Phase3Tests(unittest.TestCase):
    @staticmethod
    def _manifest(*, seed: int, policy: str, created_at: str) -> dict:
        return {
            "created_at": created_at,
            "quick_smoke_only": False,
            "git_commit": f"commit-{seed}",
            "device": "mps",
            "platform": "test-platform",
            "versions": {"torch": "test"},
            "language": "ko",
            "seeds": [seed],
            "policies": [policy],
            "limits": {"train": 128},
            "source_artifact": {
                "filename": "ko.jsonl",
                "bytes": 100,
                "sha256": "source",
            },
            "source_integrity_artifact": {
                "filename": "integrity.json",
                "bytes": 50,
                "sha256": "integrity",
            },
            "global_max_position_embeddings": 1032,
            "model_spec": {"sequence_length": 512},
            "optimization_spec": {"batch_size": 32},
            "force": False,
            "save_checkpoints": True,
            "streams": {"train": {"selected_stream_sha256": "abc"}},
        }

    def test_manifest_merge_preserves_invocations_and_unions(self) -> None:
        first = self._manifest(
            seed=1729,
            policy="fixed_byte_6",
            created_at="2026-01-01T00:00:00+00:00",
        )
        second = self._manifest(
            seed=2718,
            policy="spacebyte_spacelike",
            created_at="2026-01-02T00:00:00+00:00",
        )
        initialized = merge_phase3_manifest(None, first)
        merged = merge_phase3_manifest(initialized, second)
        self.assertEqual(merged["seeds"], [1729, 2718])
        self.assertEqual(
            merged["policies"], ["fixed_byte_6", "spacebyte_spacelike"]
        )
        self.assertEqual(len(merged["invocations"]), 2)
        self.assertEqual(merged["git_commit"], "commit-1729")
        self.assertEqual(merged["updated_at"], second["created_at"])

    def test_manifest_merge_preserves_confirmation_authorization(self) -> None:
        first = self._manifest(
            seed=1729,
            policy="fixed_byte_6",
            created_at="2026-01-01T00:00:00+00:00",
        )
        confirmation = self._manifest(
            seed=57721,
            policy="causal_whitespace_grid",
            created_at="2026-01-02T00:00:00+00:00",
        )
        confirmation["authorization"] = {
            "authorization_kind": "phase3_corrected_gate_i_confirmation_v1",
            "summary_artifact_sha256": "a" * 64,
        }
        merged = merge_phase3_manifest(
            merge_phase3_manifest(None, first), confirmation
        )
        self.assertEqual(
            merged["invocations"][-1]["authorization"],
            confirmation["authorization"],
        )

    def test_manifest_merge_preserves_clean_start_attestation(self) -> None:
        first = self._manifest(
            seed=1729,
            policy="fixed_byte_6",
            created_at="2026-01-01T00:00:00+00:00",
        )
        first["git_worktree_clean_at_start"] = True
        merged = merge_phase3_manifest(None, first)
        self.assertIs(
            merged["invocations"][0]["git_worktree_clean_at_start"], True
        )

    def test_manifest_merge_upgrades_legacy_and_rejects_data_change(self) -> None:
        first = self._manifest(
            seed=1729,
            policy="fixed_byte_6",
            created_at="2026-01-01T00:00:00+00:00",
        )
        legacy = deepcopy(first)
        del legacy["source_artifact"]
        del legacy["source_integrity_artifact"]
        second = deepcopy(first)
        second["seeds"] = [2718]
        second["created_at"] = "2026-01-02T00:00:00+00:00"
        upgraded = merge_phase3_manifest(legacy, second)
        self.assertEqual(len(upgraded["invocations"]), 2)
        self.assertEqual(upgraded["source_artifact"], second["source_artifact"])

        changed = deepcopy(second)
        changed["streams"]["train"]["selected_stream_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "invariant changed: streams"):
            merge_phase3_manifest(legacy, changed)

    def test_ood_manifest_merge_preserves_initial_and_confirmation_runs(self) -> None:
        first = {
            "schema_version": 1,
            "created_at": "first",
            "git_commit": "a",
            "device": "mps",
            "platform": "test",
            "versions": {},
            "seeds": [1729, 2718, 31415],
            "policies": [
                "fixed_byte_6",
                "causal_codepoint_grid",
                "causal_whitespace_grid",
            ],
            "force": False,
            "requested_byte_limit": 100_000_000,
            "source": {"sha256": "source"},
            "stream": {"selected_stream_sha256": "stream"},
            "global_max_position_embeddings": 1032,
            "model_spec": {"sequence_length": 512},
        }
        initialized = merge_phase3_ood_manifest(None, first)
        confirmation = deepcopy(first)
        confirmation.update(
            {
                "created_at": "later",
                "git_commit": "b",
                "seeds": [57721, 65537],
                "authorization": {
                    "authorization_kind": (
                        "phase3_corrected_gate_i_confirmation_v1"
                    ),
                    "summary_artifact_sha256": "a" * 64,
                },
            }
        )
        merged = merge_phase3_ood_manifest(initialized, confirmation)
        self.assertEqual(
            merged["seeds"],
            [1729, 2718, 31415, 57721, 65537],
        )
        self.assertEqual(len(merged["invocations"]), 2)
        self.assertEqual(
            merged["invocations"][-1]["authorization"],
            confirmation["authorization"],
        )
        self.assertEqual(merged["updated_at"], "later")

        changed = deepcopy(confirmation)
        changed["stream"] = {"selected_stream_sha256": "other"}
        with self.assertRaisesRegex(ValueError, "invariant changed: stream"):
            merge_phase3_ood_manifest(initialized, changed)

    def test_preregistered_parameter_counts(self) -> None:
        main = build_main_model(
            PHASE3_MODEL_SPEC,
            seed=1,
            global_max_position_embeddings=1032,
        )
        router = build_router(PHASE3_MODEL_SPEC, seed=1)
        self.assertEqual(parameter_count(main), 19_596_096)
        self.assertEqual(parameter_count(router), 2_016_960)
        self.assertEqual(PHASE3_OPTIMIZATION_SPEC.warmup_steps, 500)

    def test_spacebyte_mask_matches_causal_suppression(self) -> None:
        data = b"  A" + "한글".encode("utf-8")
        mask = spacebyte_causal_prefix_mask(data)
        # The first of two consecutive spaces triggers after byte 0; the
        # second is suppressed. Each Hangul lead byte triggers after itself.
        self.assertEqual(tuple(np.flatnonzero(mask)), (1, 4, 7))

    def test_spacebyte_state_crosses_window_boundaries(self) -> None:
        # If a window happens to begin on the second consecutive spacelike
        # byte, scanning rows independently would add a false event at t=1.
        continuous = spacebyte_causal_prefix_mask(b"A  B").reshape(2, 2)
        independent_second_row = spacebyte_causal_prefix_mask(b" B")
        self.assertEqual(tuple(np.flatnonzero(continuous[1])), (0,))
        self.assertEqual(tuple(np.flatnonzero(independent_second_row)), (1,))

    def test_structural_policies_cover_rows_and_exact_rates(self) -> None:
        text = ("한글 연구 test 문장. " * 80).encode("utf-8")
        data = text[: PHASE3_MODEL_SPEC.sequence_length]
        if len(data) < PHASE3_MODEL_SPEC.sequence_length:
            data += b" " * (PHASE3_MODEL_SPEC.sequence_length - len(data))
        boundaries = np.frombuffer(
            bytes(prefix_boundary_mask(data)[:-1]), dtype=np.uint8
        ).reshape(1, -1)
        whitespace = compact_whitespace_mask(data).reshape(1, -1)
        spacebyte = spacebyte_causal_prefix_mask(data).reshape(1, -1)
        matrices = structural_patch_matrices(
            boundaries,
            whitespace,
            spacebyte,
        )
        self.assertEqual(matrices["fixed_byte_6"].shape, (1, 87))
        self.assertEqual(matrices["causal_codepoint_grid"].shape, (1, 87))
        self.assertEqual(matrices["causal_whitespace_grid"].shape, (1, 87))
        self.assertGreater(
            matrices["spacebyte_spacelike"].shape[1],
            matrices["fixed_byte_6"].shape[1],
        )
        for matrix in matrices.values():
            validate_padded_patch_matrix(matrix, 512)

    def test_structural_cache_provenance_detects_source_change(self) -> None:
        arrays = {
            split: np.arange(12, dtype=np.uint8).reshape(3, 4)
            for split in RUN_MODULE.SPLITS
        }
        whitespace = {split: value.copy() for split, value in arrays.items()}
        spacelike = {split: value.copy() for split, value in arrays.items()}
        first = RUN_MODULE._structural_cache_provenance(
            arrays,
            whitespace,
            spacelike,
        )
        whitespace["test"][1, 2] += 1
        second = RUN_MODULE._structural_cache_provenance(
            arrays,
            whitespace,
            spacelike,
        )
        self.assertNotEqual(first, second)

    def test_cache_provenance_requires_exact_metadata(self) -> None:
        expected = {"schema_version": 1, "source": "abc"}
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = Path(directory) / "diagnostics.json"
            diagnostics.write_text(
                '{"_provenance":{"schema_version":1,"source":"abc"}}',
                encoding="utf-8",
            )
            self.assertTrue(
                RUN_MODULE._cache_provenance_matches(diagnostics, expected)
            )
            self.assertFalse(
                RUN_MODULE._cache_provenance_matches(
                    diagnostics,
                    {"schema_version": 1, "source": "changed"},
                )
            )

    def test_source_artifact_metadata_binds_processed_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ko.jsonl"
            source.write_bytes(b'{"text":"test"}\n')
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            integrity = root / "integrity.json"
            integrity.write_text(
                json.dumps(
                    {
                        "dataset_id": "hplt3-korean-phase3",
                        "output": {
                            "output_bytes": source.stat().st_size,
                            "output_sha256": source_hash,
                        },
                    }
                ),
                encoding="utf-8",
            )
            metadata = RUN_MODULE._source_artifact_metadata(root)
            self.assertEqual(
                metadata["source_artifact"]["sha256"],
                source_hash,
            )
            source.write_bytes(b'{"text":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "integrity"):
                RUN_MODULE._source_artifact_metadata(root)


if __name__ == "__main__":
    unittest.main()
