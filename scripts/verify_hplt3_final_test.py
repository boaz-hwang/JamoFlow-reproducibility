#!/usr/bin/env python3
"""Independently reconstruct and verify the sealed HPLT3 final test."""

from __future__ import annotations

import json
from pathlib import Path

from jamoflow.hplt3_final_test import (
    reconstruct_final_test,
    serialize_seal_envelope,
    validate_seal_envelope,
)

from prepare_hplt3_final_test import (
    ARCHIVE,
    MANIFEST,
    OUTPUT,
    PREDECESSOR_INTEGRITY,
    PREDECESSOR_MANIFEST,
    PREDECESSOR_OUTPUT,
    PREDECESSOR_SUMMARY,
    SEAL,
)


def main() -> int:
    if not OUTPUT.is_file() or not SEAL.is_file():
        raise FileNotFoundError("final-test output or seal is missing")
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    validate_seal_envelope(sealed)
    if SEAL.read_bytes() != serialize_seal_envelope(sealed):
        raise ValueError("final-test seal is not canonically serialized")
    preparation_commit = sealed["payload"].get("preparation_git_commit")
    if not isinstance(preparation_commit, str):
        raise ValueError("final-test seal lacks its preparation commit")
    reconstructed_output, reconstructed_seal = reconstruct_final_test(
        manifest_path=MANIFEST,
        archive_path=ARCHIVE,
        predecessor_manifest_path=PREDECESSOR_MANIFEST,
        predecessor_summary_path=PREDECESSOR_SUMMARY,
        predecessor_integrity_path=PREDECESSOR_INTEGRITY,
        predecessor_output_path=PREDECESSOR_OUTPUT,
        preparation_git_commit=preparation_commit,
    )
    if OUTPUT.read_bytes() != reconstructed_output:
        raise ValueError("final-test output differs from full reconstruction")
    if sealed != reconstructed_seal:
        raise ValueError("final-test seal differs from full reconstruction")
    print(
        json.dumps(
            {
                "dataset_id": sealed["payload"]["dataset_id"],
                "payload_sha256": sealed["payload_sha256"],
                "status": "verified",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
