# Release and named-preprint runbook

The anonymous ARR PDF, a named public code release, and an arXiv source upload
have different identity and licensing requirements. They must not be produced
as if they were interchangeable.

## 1. Anonymous ARR review

Use `build/arr/main.pdf`. It contains no author block or repository URL. The
current public project/package name is searchable, so the named reproducibility
archive below is **not** automatically safe to attach to anonymous review. An
anonymous software attachment would require a separate de-identification audit
and an author-approved license.

Create the ignored, mode-0600 private template from the exact current PDF,
checklist, and public-metadata hashes:

```bash
.venv/bin/python scripts/audit_arr_submission_readiness.py \
  --write-private-template
```

The command refuses to overwrite an existing file and refuses any destination
that is outside the repository or not covered by `.gitignore`. Fill
`paper/private/arr-submission-decisions.json` using
`arr-private-decisions.schema.json`. Every `TODO`, `false`, and `null` value
must be reviewed; the template is deliberately invalid and does not choose a
venue, preprint, release, consent, authorship, or license policy for the authors.
Empty funding, conflict, preprint, and acknowledgment fields must also be
affirmatively reviewed. Then run:

```bash
.venv/bin/python scripts/audit_arr_submission_readiness.py
```

The audit validates author order and profiles, reviewer readiness, checklist and
paper approval, venue/preprint choices, consents, licensing implications, the
public metadata, and the anonymous PDF. Approval fields are bound to the exact
SHA-256 of the PDF, checklist, and public metadata; a later edit requires fresh
author approval. Only a clean committed tree and fully passing decision file can
produce a private local handoff:

```bash
.venv/bin/python scripts/audit_arr_submission_readiness.py --write-handoff
```

`dist/arr-submission-handoff-v1.zip` contains the PDF, a copyable OpenReview
form, checklist, and hashes. It contains private author identity and is not a
single supplementary file to upload.

## 2. Named reproducibility release

Audit the exact tracked HEAD without creating an archive:

```bash
.venv/bin/python scripts/build_reproducibility_archive.py --audit
```

The audit selects regular Git-tracked files under explicit source, test, paper,
protocol, manifest, seal, and aggregate-result roots. It excludes raw/processed
corpora, checkpoints, run artifacts, private vault content, raw model outputs,
and per-sequence losses. It also scans for local user paths, known private
identifiers, and common credential prefixes.

The public build is intentionally unavailable until the authors add a tracked
`LICENSE`, `LICENSE.md`, or `COPYING` file. After the license and named-release
decision are approved:

```bash
.venv/bin/python scripts/build_reproducibility_archive.py \
  --named-public-release \
  --output dist/jamoflow-reproducibility-v1.tar.gz
```

The tarball is built only from `HEAD`, receives fixed metadata and ordering,
contains an aggregate file/hash manifest, and must remain below ARR's 200MB
software-archive ceiling. The explicit flag acknowledges that it is a named
public release, not anonymous supplementary material.

## 3. Named arXiv preprint

The same ignored `arr-submission-decisions.json` supplies the real ordered
authors, emails, optional ORCIDs, affiliations, addresses, and acknowledgments.
`preprint-author-metadata.schema.json` documents the smaller author projection
used by the preprint builder. Empty or placeholder identities are rejected.

Validate without producing a preprint:

```bash
.venv/bin/python scripts/build_arxiv_preprint.py --validate-metadata
```

The approved publication strategy is a named arXiv preprint and named public
code release before the October 2026 ARR cycle. The later anonymous ARR form
must disclose the existing preprint and must not claim the optional
no-named-preprint incentive. After the author list is fixed:

```bash
.venv/bin/python scripts/build_arxiv_preprint.py --verify-reproducible
```

This creates a final-mode named PDF at `build/arxiv/main.pdf` and an ignored
source archive at `dist/jamoflow-arxiv-source.tar.gz`. The archive includes the
generated TeX, author and acknowledgment inputs, official pinned ACL style,
BibTeX source and generated `.bbl`, and only referenced figures. It does not use
ACL review mode. The resulting arXiv PDF must still be inspected in arXiv's
submission preview before publication.

The builders do not upload to OpenReview, arXiv, GitHub Releases, or Hugging
Face. Authenticated upload steps consume the approved decisions separately.
The selected code license is Apache-2.0, the public repository is
`https://github.com/boaz-hwang/JamoFlow-reproducibility`, and raw corpora,
checkpoints, raw outputs, and per-sequence losses remain excluded.

Official arXiv source guidance checked on 2026-08-17:
<https://info.arxiv.org/help/submit_tex.html>.
