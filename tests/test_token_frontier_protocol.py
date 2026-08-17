from __future__ import annotations

import unittest

import numpy as np

from token_frontier_core import RUNTIME_ROLES
from token_frontier_protocol import (
    MEASURED_CASES,
    PROMPT_BYTES,
    WARMUP_CASES,
    array_sha256,
    canonical_sha256,
    reconstruct_cases,
)


class TokenFrontierProtocolTest(unittest.TestCase):
    def test_array_hash_binds_dtype_shape_and_values(self) -> None:
        values = np.arange(
            (WARMUP_CASES + MEASURED_CASES) * PROMPT_BYTES, dtype=np.uint8
        ).reshape(WARMUP_CASES + MEASURED_CASES, PROMPT_BYTES)
        self.assertEqual(array_sha256(values), array_sha256(values.copy()))
        changed = values.copy()
        changed[-1, -1] ^= 1
        self.assertNotEqual(array_sha256(values), array_sha256(changed))

    def test_canonical_hash_is_key_order_independent(self) -> None:
        self.assertEqual(
            canonical_sha256({"roles": list(RUNTIME_ROLES), "x": 1}),
            canonical_sha256({"x": 1, "roles": list(RUNTIME_ROLES)}),
        )

    def test_reconstructed_cases_are_private_safe_and_document_distinct(self) -> None:
        prompts, continuations, metadata = reconstruct_cases()
        self.assertEqual(prompts.shape, (WARMUP_CASES + MEASURED_CASES, PROMPT_BYTES))
        self.assertEqual(continuations.shape, prompts.shape)
        self.assertEqual(metadata["selected_cases"], WARMUP_CASES + MEASURED_CASES)
        self.assertEqual(
            metadata["selected_unique_clusters"], WARMUP_CASES + MEASURED_CASES
        )
        self.assertNotIn("selected_cluster_ids", metadata)
