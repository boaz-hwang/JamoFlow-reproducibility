# Fresh-v2 16K trained actual result and target-block pivot

> 작성일: 2026-08-15
>
> implementation commits: `7b30766`, `c443055`
>
> plan commit: `347a62b`
>
> result commit: `7c7cee3`
>
> plan payload SHA-256: `b42cab2ca057bbd3384d55c10cfcf14dde2194edd21f0d3f9b1d8a46e5870c99`
>
> result payload SHA-256: `56192ee9c1986106705e002b46a91cee23a6e0f7322ef25e071ba42e9f965577`
>
> result file SHA-256: `dde3e1e6004df805bcad8499b542fd29e07739fd8df5fcdf9cf1c43fbbb04231`
>
> 판정: 16K primary joint gate 실패; dense-vocabulary multi-seed 승격 중단

## 결론

Fresh-v2에서 품질 gate를 통과한 trained 16K update-geometry model은 trained 2K anchor보다 동일
continuation의 controlled E2E를 **24.925%** 줄였다. Strict-valid free-running의 aggregate point도
**10.312%**, paired-prompt bootstrap 95% 구간도 **[1.065%, 24.519%]**로 양수였다.

그러나 free mode에서 candidate가 더 빨랐던 문서는 `43/64`뿐이었다. 결과 전에 고정한 primary는
각 mode마다 point `>=10%`, bootstrap lower `>0`, faster prompts `>=48/64`, 모든 correctness를
동시에 요구했다. 따라서 16K-vs-2K joint gate는 실패했고
`multiseed_confirmation_authorized=false`다. 43을 보고 48 기준을 낮추거나, controlled만 primary로
바꾸거나, aggregate 두 조건만 골라 현재 run을 성공으로 재분류하지 않는다.

Mandatory 16K-vs-8K diagnostic도 free mode에서 **-8.292%**로 역전됐다. 따라서 vocabulary size가
커질수록 free-generation latency가 단조 개선된다는 주장도 불가하다.

## 사전 고정 gate 결과

| pair | mode | candidate E2E | reference E2E | reduction | paired-prompt 95% interval | faster prompts | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| 16K vs 2K | controlled | 64.242 ms | 85.571 ms | **24.925%** | [22.007%, 28.865%] | 64/64 | pass |
| 16K vs 2K | free | 81.541 ms | 90.916 ms | **10.312%** | [1.065%, 24.519%] | 43/64 | **fail** |
| 16K vs 8K | controlled | 64.242 ms | 68.040 ms | **5.583%** | [2.602%, 10.387%] | 52/64 | diagnostic pass |
| 16K vs 8K | free | 81.541 ms | 75.297 ms | **-8.292%** | [-18.931%, 10.527%] | 30/64 | diagnostic fail |

16K free point가 우연히 10% 바로 위에 있다는 사실은 progression을 바꾸지 않는다. Primary 전체가
통과해야 한다는 계약은 `docs/161`과 plan에 고정돼 있다.

## Correctness와 실행 무결성

- 세 model은 같은 process에 동시에 resident했다.
- 64 documents × 5 repetitions × 2 modes × 3 roles의 1,920 trial을 6개 role permutation으로
  균형화했다.
- 모든 free output `960/960`개는 strict UTF-8과 scalar-boundary stop을 통과했다.
- 동일 prompt/role의 5회 free token trace와 raw bytes는 bitwise deterministic했다.
- Summary는 세 physical checkpoint를 다시 load해 measured 64 cases 전부의 free masked-greedy trace를
  재생성했다.
- Controlled와 free trace의 모든 position에서 parallel/incremental cache logits를 full
  `use_cache=False` logits와 대조했고 tolerance 및 argmax exact equality가 모두 통과했다.

그러므로 실패를 invalid output, cache drift, stale checkpoint 또는 timing-only self-attestation으로
설명할 근거가 없다.

## 무엇이 latency를 지배했는가

결과 뒤의 descriptive mechanism audit은 별도 non-authorizing artifact로 고정했다.

- diagnostic payload SHA-256:
  `7c110567c6507933fa684d39362470b796bfe73c44031b7b1a67e7fcc3f5a572`
- diagnostic file SHA-256:
  `b39a86817b03e26c9cf819d93c0cb2dce9586ac1eb2890f0f2154207024bfaa6`

### 1. 고정 내용에서는 step 감소가 실제 속도가 됐다

| comparison | controlled token median | controlled E2E reduction |
|---|---:|---:|
| 16K vs 2K | 25 vs 36 | 24.925% |
| 16K vs 8K | 25 vs 27.5 | 5.583% |

16K의 tokenizer encode와 TTFT는 오히려 2K보다 각각 약 6.9%, 4.1% 느렸다. 이득은 prompt 처리나
큰 head의 token당 비용이 아니라 줄어든 autoregressive decode step에서 왔다.

### 2. Free path에서는 모델마다 다른 token 수가 결과를 거의 완전히 결정했다

Prompt별 E2E reduction과 token-count reduction의 Pearson correlation은 다음과 같다.

- 16K vs 2K: `0.999843`
- 16K vs 8K: `0.999867`

16K-vs-2K에서 candidate가 더 적은 token을 생성한 경우는 `46/64`였다. 그 안에서는 `43/46`이
실제로 빨랐고 median reduction은 29.61%였다. Token 수가 같았던 4개와 더 많았던 14개에서는
빠른 case가 하나도 없었고 median effect는 각각 -4.70%, -28.99%였다. Faster-prompt gate `48/64`는
timing noise보다 생성 trajectory의 step 분포 때문에 구조적으로 실패했다.

16K-vs-8K도 같다. Candidate가 더 적은 token을 낸 case는 `33/64`, 실제 faster case는 `30/64`였다.
더 많은 token을 낸 26개에서 median effect는 -31.08%였다.

### 3. 비정규 token sequence가 주원인은 아니다

생성 byte 문자열을 같은 tokenizer로 다시 canonical encode했다.

| role | emitted token mean | canonical mean | noncanonical cases | median gap |
|---|---:|---:|---:|---:|
| 16K | 36.406 | 35.703 | 10/64 | 0 |
| 8K | 39.250 | 39.172 | 4/64 | 0 |
| 2K | 48.375 | 48.375 | 0/64 | 0 |

16K의 한 outlier에는 19-token retokenization gap이 있었지만 중앙값은 0이고 54/64 trace가 이미
canonical했다. 따라서 canonical-token constraint만 붙여 current failure를 해결한다는 가설은
우선순위가 낮다. 주된 문제는 같은 byte string의 비정규 분절이 아니라, 각 model이 생성한 **서로
다른 byte string의 tokenizability**다.

## 품질과 시스템 비용

16K candidate의 fresh-v2 document BPB는 1.393474로 2K의 1.408331과 8K의 1.397882보다 좋았다.
즉 이 latency 실패는 quality-qualified model 비교에서 발생했다.

반면 물리 비용은 더 크다.

- parameters: 2K 대비 +58.48%, 8K 대비 +23.82%
- checkpoint bytes: 2K 대비 +52.37%, 8K 대비 +23.82%
- 세 model 동시-resident timing이므로 role별 memory 개선은 측정하지도 주장하지도 않는다.

관측된 aggregate free speedup만 인용하면서 이 추가 비용과 prompt heterogeneity를 숨기면 안 된다.

## Fable 5 검토에 대한 최신 판정

이번 결과도 `fable5-연구-중간-검토.md`의 핵심 systems 경고를 다시 확인했다.

1. proxy/token/analytical 감소는 actual E2E 성공과 다르다.
2. 줄어든 sequential work와 늘어난 per-step/head 비용을 함께 계산해야 한다.
3. aggregate point뿐 아니라 uncertainty와 workload heterogeneity를 공개해야 한다.
4. compact Apple-MPS 한 seed를 일반 한국어 LLM이나 CUDA로 확대하지 않는다.

다만 speed gate 실패를 곧바로 quality/방법론 paper의 완성으로 간주하자는 제안은 여전히 사용자의
성공 기준과 맞지 않는다. 현재 positive paper를 선언하지 않는다.

## 계획 수정 여부

현재 16K gate 자체는 수정하지 않는다. Dense vocabulary expansion은 8K와 16K에서 연속으로
free-path stability를 실패했고, 32K는 기존 same-body systems frontier에서 16K보다 더 큰 head에도
불구하고 controlled E2E가 더 나빴다. 따라서 32K 추가 sweep, threshold 완화, prompt 재선택,
현재 recipe의 multi-seed 확대는 정보 대비 가치가 낮아 중단한다.

결과가 요구하는 수정은 **vocabulary를 더 키우는 것**이 아니라 **한 target model의 exact greedy
출력을 보존하면서 sequential target invocation을 block으로 줄이는 것**이다. 이것은 `docs/161`이
실패 시 열도록 명시한 architecture-level 방향이며, 현재 결과의 0.9998 correlation이 직접
동기를 제공한다.

## 다음 fail-fast: trained 16K target의 perfect-draft block upper bound

Standard speculative decoding 자체를 novelty로 주장하지 않는다. 먼저 현재 trained 16K target에서
draft quality를 완전히 제거한 target-side kernel opportunity만 새 result-blind protocol로 검증한다.

1. baseline은 exact 16K target의 ordinary cached AR이다.
2. candidate는 같은 target에 known-correct greedy token blocks를 넣는 perfect-draft target verifier다.
3. block size `2/4/8`은 결과를 보고 선택하지 않고 모두 측정하며, 후속 draft가 감당할 headroom을
   사전에 요구한다.
4. controlled와 free 모두 baseline target output/cache/logits를 byte-for-byte 보존한다.
5. target block kernel만으로 두 mode E2E가 충분히 크게 줄지 않으면 learned draft를 학습하지 않는다.
6. 통과하더라도 이는 upper bound일 뿐이며, 실제 draft compute·acceptance·rollback을 포함한 exact
   runtime이 별도 gate를 통과해야 효율 claim이 열린다.

기존 W72 exact speculation은 byte-local 실행구조에서 9.983%에 머물렀다. BPE-16K는 128 raw bytes당
중앙 25--32.5 target token만 사용하므로 block verification의 kernel geometry가 다르다. 그렇다고
과거 실패를 무시하지 않고, perfect-draft upper bound부터 다시 fail-fast한다.

## 현재 주장 경계

- 말할 수 있음: one seed에서 trained 16K는 동일 128-byte continuation E2E를 24.93% 줄였다.
- 말할 수 있음: free aggregate point는 10.31%이고 bootstrap lower도 양수였다.
- 반드시 함께 말해야 함: faster prompts는 43/64로 사전 stability gate를 실패했다.
- 말할 수 있음: prompt별 speed는 token-step 차이가 거의 완전히 지배했다.
- 말할 수 없음: dense 16K가 publication-grade inference method다, per-prompt로 안정적이다,
  memory-efficient하다, multi-seed/hardware에 일반화된다.
