# ARR Responsible NLP checklist draft

This is the author-approved working copy for the OpenReview form, not a paper
appendix. It is bound to the submission-readiness hash and must be re-approved
if edited. The official checklist accepts a binary answer plus a section or
justification; the notes below retain nuance so uncertainty is not silently
converted into `Yes`.

## A. Every submission

| Item | Draft answer | Section or justification |
|---|---|---|
| A1 limitations | **Yes** | `Limitations` and Sections 5--6 report the one-seed scale extension, undertraining, one-device scope, historical-test exposure, failed 10% and amplification hypotheses, and claim boundaries. |
| A2 potential risks | **Yes** | `Ethical Considerations` and `Limitations` discuss overgeneralization, web-corpus rights/privacy/content risks, incomplete compute accounting, intended research-only use, and the decision not to release a purportedly efficient model. |

## B. Scientific artifacts

| Item | Draft answer | Section or justification |
|---|---|---|
| B1 creators cited | **Yes** | Sections 2 and 4 cite BLT, SpaceByte, H-Net, AU-Net, related tokenization work, HPLT 3.0, and the HPLT catalogue. |
| B2 licenses or terms | **No, not fully in the anonymous paper** | `Limitations` reports HPLT's CC0 package label, Common Crawl/Internet Archive provenance, legal-compliance and takedown terms, and the fact that no corpus text is redistributed. The named public code release is licensed under Apache-2.0, but its identifying URL is intentionally omitted from the anonymous ARR PDF and no anonymous software archive is attached. |
| B3 intended use | **Yes** | `Ethical Considerations` limits the work to research and reproducibility. It is not a production model or a commercial corpus derivative, and no positive efficient-model release is claimed. |
| B4 PII/offensive-content checks | **No** | No content-level PII or offensive-content audit was performed on the HPLT web text. `Limitations` states this explicitly. Raw text is not redistributed; tracked artifacts use aggregate commitments, but hashing is not claimed to anonymize the source corpus. |
| B5 artifact documentation | **Yes** | Section 4 and `Limitations` identify the Korean Hangul-script HPLT web shard, source limitations, script/domain scope, and the absence of author-demographic claims. |
| B6 data statistics | **Yes** | Section 4 reports 128M training bytes, 8M calibration bytes, the exposed historical 16M screen, and the separately sealed 32M final stream with 1,482 documents and 62,500 windows. |

## C. Computational experiments

| Item | Draft answer | Section or justification |
|---|---|---|
| C1 parameters, total compute, infrastructure | **No** | Sections 4--5 report 19,596,096 and 188,639,808 trained parameters, random graphs through 1.618B, Apple M4 Pro/MPS infrastructure, byte budgets, and final session counts. `Limitations` discloses that a complete project-wide accelerator-hour total cannot be reconstructed without false precision because heterogeneous exploratory and abandoned diagnostics lack one authoritative elapsed-time schema. Stage-specific receipts remain available. |
| C2 setup and hyperparameters | **Yes** | Sections 3--4 specify policies, fixed calibration-only selection, seeds, data, model widths/depths, AdamW, batch size, warmup, learning-rate schedule, one allowed W80 rescue, and all stopping/gating rules. Section 5 reports failed and excluded branches. |
| C3 descriptive statistics | **Yes** | Section 5 reports five-seed effects, document and paired-seed bootstrap intervals, five fresh timing sessions, crossed session--seed--prompt intervals, and clearly labels the one-seed scale bridge. |
| C4 packages and implementations | **No in the anonymous package** | The paper gives the custom model/evaluator semantics and parameter settings, while the Apache-2.0 named public repository pins package versions and executable protocols. The anonymous PDF does not name that repository and no de-identified ARR software archive is attached. |

## D. Human participants or annotators

**No.** The study used no recruited annotators, crowdworkers, or human
participants. D1--D5 are therefore not applicable. Web documents in HPLT are
existing scientific artifacts, not data newly collected from participants by
this study; their privacy and rights limitations are addressed under B2--B4.

## E. AI assistants

| Item | Draft answer | Section or justification |
|---|---|---|
| E1 AI-assistant use | **Yes** | `Ethical Considerations`: AI assistants supported code drafting, adversarial protocol review, repository navigation, and language editing. They are not authors. Empirical claims were checked against tracked evidence or executable tests, and human authors retain responsibility. |

## Approved submission decisions

- sole author: Gyeongchan Hwang;
- preferred venue: NAACL 2027 through the October 2026 ARR cycle;
- named arXiv preprint and Apache-2.0 public code release before ARR;
- no de-identified anonymous software or raw-data attachment;
- no funding or conflicts of interest; reviewer consent granted;
- B2 and C4 remain negative for the anonymous package for the reasons above.

Official references checked on 2026-08-17:

- <https://aclrollingreview.org/responsibleNLPresearch/>
- <https://aclrollingreview.org/submissionform>
