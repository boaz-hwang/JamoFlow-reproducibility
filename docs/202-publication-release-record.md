# Publication release record

> Recorded: 2026-08-17
>
> Scope: named preprint and reproducibility materials, not peer-reviewed acceptance

## Public identifiers

- Paper title: *Causal Whitespace Patching for Korean Byte-Latent Language Models:
  Quality-Preserving Latency and the Limits of Scale Amplification*
- Sole author: Gyeongchan Hwang, Priming Water
- ORCID: [`0009-0007-5840-3274`](https://orcid.org/0009-0007-5840-3274)
- Version DOI: [`10.5281/zenodo.21973009`](https://doi.org/10.5281/zenodo.21973009)
- Concept DOI: [`10.5281/zenodo.21973008`](https://doi.org/10.5281/zenodo.21973008)
- Public repository: <https://github.com/boaz-hwang/JamoFlow-reproducibility>
- Immutable GitHub release: <https://github.com/boaz-hwang/JamoFlow-reproducibility/releases/tag/v0.1.0>

The Zenodo resource type is `Publication / Preprint`, version `v0.1.0`, with publication
date `2026-08-17`. The record is public under CC BY 4.0. Released code is Apache-2.0.

## Artifact integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `JamoFlow-paper.pdf` | 107,436 | `393e25d4d204e3e805cba3505267a68508eaf55b2b40ba24795a637cb28f8b3c` |
| `jamoflow-arxiv-source.tar.gz` | 66,919 | `ee94268cfc6f28853a7aca33249cede5b20622e565c2e875113d20a72d1eb5ba` |
| `jamoflow-reproducibility-v1.tar.gz` | 2,612,983 | `0f48cb350593843958754fa1cddf37b5dd15f642a8e5afdc7f7300681b5bb93b` |

The three files were downloaded again from the published Zenodo API endpoint and their
SHA-256 values matched the local release inputs exactly. The deposit also contains
`CITATION.cff` and `SHA256SUMS`.

## Claim boundary

The public result is a bounded positive systems finding and a negative scale-amplification
finding:

- 19.6M W72: 2.628% controlled and 2.531% strict-valid free-running latency reduction;
- 188.6M W80: 2.887% controlled and 2.475% strict-valid free-running reduction;
- the prespecified 10% target failed, and model size did not amplify the percentage gain.

This release does not claim a production-ready Korean LLM, a 10% trained speedup, a
general-hardware result, a memory reduction, or peer-reviewed acceptance.

## Included and excluded materials

Included are the named PDF, self-contained arXiv source, curated reproducibility archive,
citation metadata, checksums, source, tests, protocols, corrections, and aggregate evidence.
Excluded are raw corpora, model checkpoints, record identifiers, raw generated outputs, and
per-sequence loss arrays. The absence of checkpoints is deliberate: neither trained model
passed the positive 10% efficiency gate, so publishing it as an “efficient Korean model”
would contradict the result.

## Remaining publication workflow

- arXiv account, email verification, and ORCID linking are complete; submission `7958327`
  is blocked only by the first-time `cs.CL` endorsement requirement;
- the Hugging Face research-artifact mirror is in account-activation/upload progress and will
  not contain model weights;
- OpenReview profile `~Gyeongchan_Hwang1` is awaiting moderation;
- the target peer-review route is the 2026-10-12 ARR cycle, with the public preprint disclosed.

These pending services must not be described as completed submissions or acceptances.
