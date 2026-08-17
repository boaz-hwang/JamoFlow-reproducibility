# ARR 제출 패키지와 공개 결정

> 작성일: 2026-08-17
>
> 상태: **named preprint·공개 재현 패키지 게시 완료, arXiv endorsement와 ARR 제출 대기**
>
> 제출 원고: [`paper/arr-submission.md`](../paper/arr-submission.md)
>
> 상세 감사 원고: [`paper/draft.md`](../paper/draft.md)
>
> 연구 방향: [규모 확장 결과를 반영한 연구 방향 수정](./200-revised-scale-research-direction.md)

## 1. 현재 결론

현 단계에서 더 큰 모델을 추가로 학습해 양의 결과를 찾는 것보다, 이미 확정된 결과를
과장 없이 출판하는 편이 과학적으로 타당하다. 논문의 핵심은 다음 세 문장으로 고정한다.

1. 같은 72-patch rate에서 causal whitespace relocation은 codepoint placement보다
   Korean byte-model 품질이 좋았다.
2. C86과 품질을 맞춘 실제 cached generation은 19.6M과 188.6M 두 trained scale에서
   약 2.5--2.9% 빨랐다.
3. random graph의 시스템 headroom은 1.618B에서 10%를 넘었지만, trained model은
   품질 때문에 더 많은 patch를 유지해야 했으므로 speedup이 증폭되지 않았다.

이는 positive 10% speedup 논문이 아니다. **boundary-placement 결과, actual-inference
검증, 그리고 random systems headroom과 trained quality-constrained speedup의 분리**가
주요 기여다.

## 2. 제출 형식

2026-08-17 현재 공식 ARR 일정에서 다음 cycle deadline은 2026-10-12이며, NAACL 2027과
COLING 2027의 final ARR submission cycle로 표시되어 있다. 현재 원고는 ARR long-paper
review 형식에 맞춘다.

| 항목 | 현재 상태 |
|---|---|
| 형식 | official ACL review style, A4, 11pt, two-column, line numbers |
| 본문 | Conclusion까지 7쪽; 8쪽 한도 안 |
| Abstract | 181 words; 200-word 한도 안 |
| 필수 후면 섹션 | Limitations와 Ethical Considerations 포함 |
| 익명화 | PDF 저자 표시는 `Anonymous ACL submission`; repository URL 없음 |
| 인용 | Markdown citation key가 모두 `references.bib`에 존재 |
| 글꼴 | PDF의 모든 글꼴 embedded |
| 그림 | tracked aggregate evidence에서 재생성; 8-bit opaque RGB로 고정 |
| 빌드 | fixed ACL-style commit/SHA와 `SOURCE_DATE_EPOCH`; 연속 두 build SHA 일치 검사 |

2026-08-17 검증 환경은 Pandoc 3.10.1, Tectonic 0.17.0, ImageMagick 7.1.2-3,
Poppler `pdfinfo` 26.01.0이다. 이 환경에서 연속 두 build가 동일한 PDF SHA-256
`d9aef7a80cd041f0a645578ba2c54971d58a1a61ef5b84340e221fd36a8fdb42`를 만들었다.
PDF는 총 9쪽이다. Conclusion/Limitations 시작은 7쪽이어서 main-content 8쪽 한도 안이고,
추가된 9쪽은 참고문헌이다. Embedded font 11개와 visible 8-bit raster 1개를 확인했다.
이 해시는 source 또는 toolchain이 바뀌면 새 검증 기록으로 교체한다.
같은 closure 이후 ARR handoff audit까지 포함한 최신 full suite에서
`PYTHONPATH=src .venv/bin/pytest -q`는
`1036 passed, 92 subtests passed`로 끝났다.

권장 primary area는 `Efficient Methods for NLP`다. 부기할 수 있는 contribution type은
`NLP engineering experiment`, `Model analysis & interpretability`, 그리고 실패한
scale-amplification을 명시한 negative/non-generalization evidence다.

## 3. 제출 파일 구조

- `paper/arr-submission.md`: 익명 review 원고의 editable authority
- `paper/acl-template.tex`: official ACL review wrapper
- `paper/filters/acl-tables.lua`: captioned Markdown table을 ACL float로 변환
- `paper/references.bib`: 원고 인용
- `paper/figures/*`: tracked aggregate evidence에서 만든 SVG/PNG
- `scripts/generate_paper_figures.py`: figure 재생성 및 byte comparison
- `scripts/build_arr_paper.py`: style download/hash verification, Pandoc/Tectonic build,
  PDF 형식·인용·그림·재현성 검사
- `tests/test_paper_package.py`: 익명화, abstract, section order, citation, PNG format,
  style pin, figure-evidence 회귀 검사
- `paper/arr-submission-metadata.json`: OpenReview title/TLDR/abstract와 아직 결정하지 않은
  사람·법적 필드를 분리한 machine-readable 초안
- `paper/arr-responsible-checklist-draft.md`: ARR A1--E1 응답 초안과 근거 위치
- `paper/release-and-preprint-runbook.md`: 익명 review, named code release, arXiv source를
  분리한 실행 절차
- `scripts/build_reproducibility_archive.py`: HEAD allowlist/identity·credential scan,
  deterministic archive; license 없으면 public build 거부
- `scripts/build_arxiv_preprint.py`: ignored private author metadata를 요구하는 final-mode
  PDF와 arXiv source archive builder
- `paper/arr-private-decisions.schema.json`: 저자 순서·OpenReview profile·동의·venue·preprint·
  공개 결정을 한 파일로 받는 private schema
- `scripts/audit_arr_submission_readiness.py`: 모든 private 결정과 익명 PDF가 통과할 때만
  local OpenReview handoff ZIP을 만드는 fail-closed audit

Readiness audit에서 공개 부분은 모두 통과했다. 익명 PDF SHA-256은
`d9aef7a80cd041f0a645578ba2c54971d58a1a61ef5b84340e221fd36a8fdb42`, checklist SHA-256은
`f8e33df1b3bd96f7e27eb81647ed998c93e456acf3f7c4d51ca38cb36d6596fc`, public metadata
SHA-256은 `862ef115835733cb9d9cb64e874e9c04572de4df67a8d5fbf66e35a13eaf9621`이다. 현재
Clean clone에서 `ready=false`의 유일한 원인은 실제 private author/legal decision file이
없기 때문이다. `--write-private-template`은 위 세 해시를 자동으로 넣은 Git-ignored,
mode-0600 template을 no-clobber로 만든다. Template의 TODO·미동의 값은 실제 저자가
검토하기 전까지 계속 fail closed한다. Synthetic private fixture로 successful handoff build
경로를 테스트한 뒤 fixture와 ZIP은 모두 제거했다.

재현 명령은 다음과 같다.

```bash
.venv/bin/python scripts/generate_paper_figures.py --verify
.venv/bin/pytest -q tests/test_paper_package.py
.venv/bin/python scripts/build_arr_paper.py --verify-reproducible
```

산출 PDF는 `build/arr/main.pdf`이며 실험 evidence가 아니라 source에서 다시 만들 수 있는
ignored derivative다.

## 4. 주장 경계

### 허용하는 표현

- `small but reproducible matched-quality latency reduction`
- `same-rate causal boundary-placement effect`
- `replication across two trained scales without scale amplification`
- `post-result random-weight systems headroom`
- `quality-feasible patch density and a shared byte-local path limit the gain`

### 사용하지 않는 표현

- `10% trained speedup`, `fast Korean LLM`, `scaling law`
- `whitespace is Korean morphology` 또는 `Korean-specific optimal segmentation`
- `best raw-byte model`, `SpaceByte replacement`, `learned routing replacement`
- `production efficiency`, `CUDA/general-hardware result`, `memory reduction`
- random-weight 1.618B 결과를 trained language-model speedup으로 소개하는 문장

논문 제목과 초록은 이 경계를 반영한다. 상세한 역사와 실패 branch는
`paper/draft.md` 및 protocol/result 문서에 남기되, review 원고는 독립적으로 이해할 수
있도록 핵심 방법·수치·한계를 본문에 포함한다.

## 5. 코드·모델 공개 결정

GitHub에는 source, tests, protocol/errata/result 문서, canonical aggregate evidence,
그림 생성기와 논문 source를 공개할 가치가 있다. Raw HPLT text, record ID, private vault
문서, per-sequence loss와 raw model outputs는 공개하지 않는다.

현 checkpoint를 `efficient Korean model`로 Hugging Face에 올리지는 않는다. Compact
checkpoint는 diagnostic model이고 188.6M W80은 one-seed·0.6785 byte/parameter의
severely undertrained screen이다. 둘 다 positive 10% gate나 scale-amplification gate를
통과하지 못했다. 재현 목적의 checkpoint 보존과 유용한 pretrained model 배포는 다른
결정이다.

## 6. 저자 결정과 현재 외부 계정 상태

다음 항목은 repository나 Git author에서 추론하지 않고 저자가 직접 확정했다.

- Sole author: Gyeongchan Hwang; affiliation: Priming Water; ORCID:
  `0009-0007-5840-3274`; contact: `support@boaz.page`.
- 저자는 연구 설계, 구현, 실행, 분석, 검증, 원고 작성과 공개 결정을 모두 담당한다.
- External funding과 별도 conflict of interest는 없으며, 별도 acknowledgments도 없다.
- Code는 Apache-2.0, paper는 CC BY 4.0으로 공개한다. Checkpoint와 raw corpus는 배포하지 않는다.
- Named preprint와 public code release를 ARR 전에 공개한다. 이후 ARR form에서 existing
  non-anonymous preprint를 정확히 disclose한다.
- OpenReview profile `~Gyeongchan_Hwang1`은 생성됐고 moderation 대기 중이다.
- arXiv account와 ORCID linking은 완료됐으며 submission `7958327`은 첫 `cs.CL`
  endorsement 대기 중이다.

초기 checklist 초안은 정확한 한계를 보존하기 위해 세 항목을 자동으로 `Yes` 처리하지
않았다. 모든 탐색 실험을 합친 총 accelerator-hour는 단일 authoritative schema로 남아
있지 않고, HPLT 원문의 content-level PII/offensive-content audit를 수행하지 않았다.
Repository license는 이후 Apache-2.0으로 확정했다. 논문은 남은 두 사실과 raw text
비재배포·aggregate-only 추적·연구용 사용 범위를 명시한다.

ARR는 첫 제출 뒤 author list 변경을 제한한다. 따라서 이름과 순서를 임시값으로 넣어
먼저 올리지 않는다. 또한 ARR submission form에서 `no non-anonymous preprint` 선택을
하면 meta-review 공개 전 named preprint를 올릴 수 없으므로 preprint 결정도 제출 전에
고정한다.

## 7. 남은 실행 순서

1. GitHub public repository와 `v0.1.0` release는 게시 완료했다.
2. Named preprint와 재현성 bundle은 Zenodo DOI
   [`10.5281/zenodo.21973009`](https://doi.org/10.5281/zenodo.21973009)로 게시 완료했다.
3. Hugging Face에는 model weights가 아니라 동일한 공개
   [연구 artifact mirror](https://huggingface.co/datasets/boaz-hwang/JamoFlow-reproducibility)를
   게시했고, public re-download SHA-256 검증까지 완료했다.
4. arXiv `cs.CL` endorsement가 승인되면 submission `7958327`에 이미 검증한 source archive를
   올리고 generated PDF를 다시 확인한 뒤 제출한다.
5. OpenReview moderation이 끝나면 reviewer registration과 profile completeness를 확인한다.
6. 2026-10-12 ARR cycle에서 anonymous long paper를 제출하고 existing named preprint를
   disclose한다.

기계적으로도 이 순서를 강제했다. Pre-license audit 시점에는 named reproducibility archive
build가 의도적으로 실패했고, Apache-2.0 선택 후에만 public package를 만들었다. 저자 JSON은
`paper/private/` 아래에서만 읽으며 Git에서 무시된다. ArXiv builder는 placeholder/Anonymous
저자, ACL review mode, 누락된 `.bbl`/style/figure를 거부하고, 어떤 외부 서비스에도 자동
업로드하지 않는다.

Release-builder commit `0d8d0fa50d7dedf025b1cfdb397c0e2d2c7a4022`에 대한 clean-HEAD
audit은 regular tracked file 902개, 12,689,885 bytes를 선택했고 금지된 local path,
private identifier, credential-prefix finding은 0개였다. 해당 historical audit은
`license_path=null`이어서 `public_release_ready=false`였고, 공개 project/package identity가 검색 가능하므로
`anonymous_arr_attachment_ready=false`다. 이는 실패가 아니라 임의 license 선택과
익명성 누출을 막는 의도한 상태다.

현재 과학 실험은 출판을 위해 충분히 닫혔다. 남은 blocker는 추가 positive result가 아니라
사람이 제공해야 하는 authorship·법적·계정 정보다.
