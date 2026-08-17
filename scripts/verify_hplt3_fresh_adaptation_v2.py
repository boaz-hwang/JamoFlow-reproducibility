#!/usr/bin/env python3
"""Independently reconstruct and verify the second fresh Korean corpus."""

from __future__ import annotations

import json

from hplt3_fresh_adaptation_v2_protocol import (
    serialize_seal,
    validate_seal_envelope,
)
from prepare_hplt3_fresh_adaptation_v2 import OUTPUT, SEAL, _reconstruct


def main() -> int:
    if not OUTPUT.is_file() or not SEAL.is_file():
        raise FileNotFoundError("fresh-v2 output or seal is missing")
    sealed = json.loads(SEAL.read_text(encoding="utf-8"))
    validate_seal_envelope(sealed)
    if SEAL.read_bytes() != serialize_seal(sealed):
        raise ValueError("fresh-v2 seal serialization differs")
    output, envelope = _reconstruct(sealed["payload"]["preparation_git_commit"])
    if OUTPUT.read_bytes() != output or sealed != envelope:
        raise ValueError("fresh-v2 independent reconstruction differs")
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
