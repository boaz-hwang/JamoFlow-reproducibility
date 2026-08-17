#!/usr/bin/env python3
"""Independently reconstruct the fresh vocabulary-adaptation streams."""

from __future__ import annotations

import json

from hplt3_fresh_adaptation_protocol import (
    reconstruct,
    serialize_seal,
    validate_seal_envelope,
)

from prepare_hplt3_fresh_adaptation import (
    ARCHIVE,
    FINAL_MANIFEST,
    FINAL_OUTPUT,
    FINAL_SEAL,
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
        raise FileNotFoundError("fresh-data output or seal is missing")
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    validate_seal_envelope(sealed)
    if SEAL.read_bytes() != serialize_seal(sealed):
        raise ValueError("fresh-data seal serialization differs")
    output, envelope = reconstruct(
        manifest_path=MANIFEST,
        archive_path=ARCHIVE,
        final_manifest_path=FINAL_MANIFEST,
        final_seal_path=FINAL_SEAL,
        final_output_path=FINAL_OUTPUT,
        predecessor_manifest_path=PREDECESSOR_MANIFEST,
        predecessor_summary_path=PREDECESSOR_SUMMARY,
        predecessor_integrity_path=PREDECESSOR_INTEGRITY,
        predecessor_output_path=PREDECESSOR_OUTPUT,
        preparation_git_commit=sealed["payload"]["preparation_git_commit"],
    )
    if OUTPUT.read_bytes() != output or sealed != envelope:
        raise ValueError("fresh-data independent reconstruction differs")
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
