# JamoFlow reproducibility archive

This named public-release archive was generated from regular Git-tracked files
at commit `56cdf45d0e76d5590a2513116227a23aa403079c`. File hashes are recorded in `ARTIFACT_MANIFEST.json`.

It includes source, tests, paper materials, protocol documents, manifests,
seals, and tracked aggregate results. It excludes raw/processed corpora,
checkpoints, machine-specific run artifacts, private vault content, raw model
outputs, and per-sequence losses. Licensing is defined by `LICENSE`.

This is not an anonymous ARR software attachment: the project/package identity
is searchable. Do not attach it to anonymous review without a separate
de-identification and license review.

Canonical regression command:

    PYTHONPATH=src .venv/bin/pytest -q tests
