# JamoFlow paper package

This directory keeps two different manuscripts on purpose.

- `draft.md`: the full diagnostic/audit manuscript. It preserves historical
  screens, failed branches, and detailed provenance discussion.
- `arr-submission.md`: the anonymous, self-contained long-paper source. It
  concentrates the submission on the causal boundary comparison, matched-
  quality actual inference, and the negative scale-amplification result.

The two tracked figures are generated only from tracked aggregate evidence.

```bash
.venv/bin/python scripts/generate_paper_figures.py --verify
```

## Anonymous ACL/ARR PDF

The canonical builder downloads only `acl.sty` and `acl_natbib.bst` from the
official `acl-org/acl-style-files` repository at commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`, verifies their SHA-256 digests,
converts the Markdown source with Pandoc, and compiles it with Tectonic.

```bash
.venv/bin/python scripts/build_arr_paper.py
```

For a byte-for-byte reproducibility check under the same local toolchain:

```bash
.venv/bin/python scripts/build_arr_paper.py --verify-reproducible
```

The generated source, log, and anonymous review PDF are written to
`build/arr/`. The builder fails if the abstract exceeds 200 words, the PDF is
not A4, fonts are not embedded, or the conclusion does not begin within the
eight-page main-content allowance. It also rejects missing raster figures,
transparency masks, non-8-bit raster output, unresolved citations, and
horizontal overflow. A fixed `SOURCE_DATE_EPOCH` removes build-time PDF
metadata variation; `--verify-reproducible` requires two consecutive builds to
have the same SHA-256 under the same Pandoc/Tectonic toolchain. Build outputs
are rebuildable derivatives, not experimental evidence, and are intentionally
ignored by Git.

The Markdown source is the editable authority. `acl-template.tex` is an
anonymous review wrapper, and `filters/acl-tables.lua` turns captioned Markdown
tables into ACL floats. The tracked PNG fallbacks are deliberately opaque
8-bit RGB because a 16-bit alpha mask was found to become fully transparent in
the Tectonic PDF path; the paper-package tests prevent that regression.

As checked on 2026-08-17, the next listed ARR deadline is 2026-10-12, which is
the final ARR cycle for NAACL 2027 and COLING 2027. ARR long review versions
allow up to eight pages of main content, followed by required Limitations,
optional Ethical Considerations, and unlimited references. The review remains
anonymous. The natural primary area is `Efficient Methods for NLP`; defensible
contribution labels are `NLP engineering experiment`, `Model analysis &
interpretability`, and a disclosed negative/non-generalization result. The
paper is not submitted as a positive 10% speedup method.

The approved sole-author identity is Gyeongchan Hwang (Priming Water), ORCID
0009-0007-5840-3274. The approved strategy is a named arXiv preprint plus an
Apache-2.0 public reproducibility release before anonymous ARR submission. ARR
must therefore disclose the existing preprint. OpenReview reviewer
registration remains required, and the author list cannot be changed after the
initial submission without the resubmission rules applying.

`arr-submission-metadata.json` is the machine-readable public projection of the
approved OpenReview and release choices. Private contact and approval evidence
remain in the ignored decision file.
`arr-responsible-checklist-draft.md` maps the current paper to the ARR checklist
and deliberately leaves incomplete total-compute accounting, content-level PII
review, and unselected software licensing as negative or conditional answers.
Those answers must be approved by every author before submission.

Named release and preprint steps are separated in
`release-and-preprint-runbook.md`. The reproducibility archive refuses to build
without a tracked author-approved license, and the arXiv builder refuses to
build without real private author metadata. Neither output is silently treated
as an anonymous ARR attachment.
`audit_arr_submission_readiness.py` also prevents a private OpenReview handoff
from being created until the author order, profiles, reviewer readiness,
checklist approval, venue/preprint choices, and required consents are explicit.

For a Korean-language, step-by-step path from independent/company-affiliated
author identity through ARR review and venue commitment, see
`independent-author-publication-plan.ko.md`. It explains why no university
affiliation is required, how to represent a Korean sole proprietorship without
misstating it as a corporation, and which private identity and policy decisions
must still be supplied by the author.

Official references:

- <https://aclrollingreview.org/dates>
- <https://aclrollingreview.org/cfp>
- <https://acl-org.github.io/ACLPUB/formatting.html>
- <https://github.com/acl-org/acl-style-files>
- <https://aclrollingreview.org/submissionform>
- <https://aclrollingreview.org/responsibleNLPresearch/>
