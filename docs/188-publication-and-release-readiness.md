# Publication and release readiness

> 갱신일: 2026-08-17
>
> 상태: **trained extension complete; anonymous ARR package validated, author/legal metadata pending**
>
> 최신 연구 판정: [Balanced 200M W80 quality and actual-inference result](./199-balanced-200m-w80-quality-and-actual-result.md)
>
> ARR 제출본: [`paper/arr-submission.md`](../paper/arr-submission.md)
>
> 상세 감사본: [`paper/draft.md`](../paper/draft.md)
>
> 제출 결정: [ARR 제출 패키지와 공개 결정](./201-arr-submission-package-and-release-decision.md)

## 1. 사용자 성공 기준에 대한 답

사용자가 정한 기준은 “실제 모델 추론에서 효율이 개선됐는가”였다. 답은 둘로 나뉜다.

- **관측된 실제 개선:** quality-matched trained 19.6M W72는 C86보다 controlled
  2.628%, strict-valid free-running 2.531% 빨랐고 5/5 sessions와 5/5 seeds가 모두
  같은 방향이었다.
- **더 큰 trained screen:** 188.6M W72는 품질을 실패했지만, 단일 W80 rescue는
  C86 대비 `+0.00406 BPB`로 품질을 보존했다. Five-session actual inference는
  controlled 2.887%와 free-running 2.475% 개선을 보였고 16/16 prompts와 5/5
  sessions가 같은 방향이었다.
- **scale-amplification 기준:** 실패했다. Controlled point는 compact보다 0.259
  percentage points 컸지만 bootstrap lower가 compact point를 넘지 못했고,
  free-running point는 compact보다 0.056 points 작았다. Public EXAONE 7.8B generic
  retrieval도 14.938% 느렸다.

따라서 “실제 개선은 전혀 없다”도 틀리고 “가치 있는 고효율 기법을 완성했다”도 틀리다.
정확한 판정은 **두 trained scale에서 작고 재현 가능한 실제 개선은 있으나, 모델 크기가
개선율을 증폭한다는 가설과 사전 정의한 큰 positive-efficiency 기준은 실패했다**이다.

## 2. 출판 가치

현재 evidence는 top-tier positive speedup paper를 지지하지 않는다. 다음처럼 좁힌
empirical/diagnostic paper는 출판 가치가 있다.

1. Exact-rate W72/C72 비교로 Korean whitespace-informed boundary placement의
   modeling-quality 효과를 다섯 seed와 sealed final stream에서 식별한다.
2. Exact physical W72/C86 bundle의 controlled/free 실제 runtime을 다섯 sessions에서
   측정해 analytical savings와 wall time을 분리한다.
3. 49.8M--1.62B same-weight random graph에서 schedule headroom이 10.217%까지 증가하는
   systems curve를 제시하되, trained 188.6M 결과와 분리해 quality constraint가 그 headroom을
   어떻게 제한하는지 보여 준다.
4. 188.6M에서 W72 quality failure와 single-candidate W80 quality rescue를 함께 공개하고,
   five-session actual result로 random-weight headroom과 trained matched-quality 효과를 분리한다.
5. Learned router cost, strict UTF-8, cache equivalence, source-document bootstrap,
   result-blind correction과 실패 artifact를 포함한 재현성 방법을 공개한다.
6. Generic retrieval의 7.8B scale inversion을 supplementary total-cost caution으로
   제시한다.

적합한 framing은 systems/efficiency workshop, negative-results track, reproducibility
track 또는 empirical findings다. 제목·초록·결론에서 `fast Korean LLM`, `10% speedup`,
`best byte model`, `publication-scale trained result`를 사용하면 안 된다.

## 3. 논문 초안 상태

논문 source는 역할에 따라 둘로 분리했다.

- `paper/draft.md`는 전체 탐색·실패·provenance를 보존하는 extended audit manuscript다.
- `paper/arr-submission.md`는 181-word abstract와 8쪽 review layout으로 압축한 익명
  long-paper source다. 같은-rate boundary placement, compact actual inference,
  random systems headroom과 trained 188.6M negative amplification에 집중한다.

상세 원고는 다음을 포함한다.

- related work와 novelty boundary
- causal patch policy 정의
- exposed development split와 sealed-final chronology
- five-seed final quality
- actual-inference v5r3
- component mechanism
- 49.8M--1.62B random-weight schedule-scale sensitivity와 resource feasibility
- trained 188.6M W72 quality failure, W80 rescue와 five-session actual inference
- supplementary EXAONE retrieval stress test
- limitations, terminal conclusion과 claim boundary

모든 citation key가 `paper/references.bib`에 존재한다. Phase 3 total-cost Pareto가
authoritative artifact로 완성되지 않았으므로 해당 positive claim과 제목의 `Total Cost`를
제거했고, 확인 가능한 parameter/analytical facts와 미완료 범위를 분리했다.

Official ACL style의 A4/two-column PDF, tracked-evidence figure rendering, captioned table,
embedded font, citation, abstract/page-limit, byte-reproducible build 검증은 완료했다.
Author/affiliation, funding/conflict, license, preprint choice, OpenReview account와 외부 reviewer
feedback은 남아 있다. 이는 과학적 결과를 바꾸는 추가 실험이 아니라 사람의 메타데이터·
법적·제출 작업이다.

## 4. 공개할 artifact

### GitHub

공개 가치가 있고 현재 repository에서 재현 가능한 대상은 다음이다.

- source와 tests
- protocol/errata/result documents
- canonical plan과 aggregate result JSON
- paper draft와 references
- raw text를 포함하지 않는 provenance/hash

최종 scale preflight 직전 전체 회귀는 `956 passed, 92 subtests passed`였고, 결과 commit
뒤 read-only full-checkpoint verifier가 통과했다.

### Hugging Face

새 model upload는 하지 않는다.

- Trained 188.6M W80 checkpoint가 존재하지만 one-seed, 0.6785 byte/parameter의 severe-
  undertraining screen이다. Scale-amplification gate도 실패했으므로 이를 useful Korean model
  또는 efficient LLM로 배포하지 않는다.
- Compact W72 checkpoint도 small diagnostic model이며 strongest raw reference를 대체하지
  못하고 10% actual gate를 실패했다.
- EXAONE checkpoint는 제3자 model이고 JamoFlow가 재배포할 대상이 아니다.
- HPLT-derived raw cases와 private vault text는 repository/HF dataset으로 배포하지 않는다.

효율이 입증되지 않은 checkpoint를 “JamoFlow efficient Korean model”로 올리면 연구 결론과
artifact card가 모순된다. 향후 독립된 새 mechanism이 matched-quality actual gate를 통과할
때만 별도 model card와 weights release를 만든다.

## 5. 공개 전 비실험 체크리스트

- [ ] Repository license 선택 및 추가 — 저자가 명시적으로 결정해야 하는 법적 범위
- [ ] Author names, affiliations, acknowledgements와 funding disclosure
- [x] ARR long-paper 8-page main-content 형식으로 축약하고 긴 chronology를 audit manuscript로 분리
- [x] Main figure: random systems headroom과 two-trained-scale quality-qualified result
- [x] Tracked aggregate만으로 figure를 재생성·byte-compare하는 script
- [x] Official ACL style hash, A4, font, citation, visible raster와 reproducible-PDF build 검사
- [ ] Anonymous/public artifact URL과 exact release tag
- [ ] 최소 한 명의 외부 reviewer가 claim matrix와 statistical unit을 독립 검토
- [ ] 필요 시 archival DOI/Zenodo; local Git chronology를 public preregistration으로 표현하지 않음

License, 저자정보, venue와 외부 검토는 현재 repository evidence로 자동 결정할 수 없다.
그 외 새로운 expensive training이나 결과 기반 threshold 변경은 필요하지 않으며 허가되지
않는다.

## 6. 최종 one-sentence claim

> In a controlled Korean byte-latent model, relocating equal-rate causal patch
> boundaries toward observed whitespace improves modeling quality and yields a
> small reproducible latency reduction; a quality-rescued 188.6M replication
> reproduces roughly the same 2.5--2.9% effect rather than supporting automatic
> scale amplification.
