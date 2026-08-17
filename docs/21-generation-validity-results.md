# Phase 2d results: low absolute UTF-8 validity, no whitespace-specific stop

> 작성일: 2026-08-10
>
> 사전 고정: [generation validity addendum](./20-generation-validity-addendum.md)
>
> 기계 판독 결과: [`results/phase2-generation/summary.json`](../results/phase2-generation/summary.json)
>
> 상태: **Gate H validity component 통과; absolute validity는 낮음**

## 1. 결론

Whitespace-aware W는 fixed-byte C0 대비 UTF-8 validity를 사전 고정 중단선인 1 percentage point 이상 악화시키지 않았다.

- greedy W − C0: **−0.625%p**
- sampled W − C0: **+1.953%p**
- Gate threshold: 각 mode −1%p 이상
- Gate H validity component: **통과**

하지만 모든 unconstrained policy의 절대 validity가 낮았다. Greedy는 평균 43.7–44.9%, sampled는 28.4–30.9%의 128-byte continuation만 strict UTF-8로 decode됐다. 따라서 이 결과는 생성 성공이 아니라 **policy-specific harm stop condition이 발동하지 않았다**는 결과다.

## 2. 고정된 실험

- held-out Korean Wikipedia prompts: 256
- prompt: 128 bytes, strict UTF-8, Hangul-heavy
- continuation: 128 bytes
- candidate prompts: 640, exact duplicates 0
- seeds: 5
- policies: fixed-byte / causal codepoint / causal whitespace
- decoding: greedy / temperature 0.8, top-p 0.95
- unconstrained continuations: 7,680
- hard-mask control continuations: 1,536

Prompt·source row·generated sample은 저장하지 않았다. Policy는 매 byte step에서 관측된 full prefix로 재구성했고 `use_cache=False`로 평가했다. 따라서 이 run의 소요 시간은 incremental decoding speed 지표가 아니다.

## 3. Unconstrained validity

5-seed 평균:

| Policy | Greedy UTF-8 | Sampled UTF-8 | Greedy bytes/valid CP | Sampled bytes/valid CP |
|---|---:|---:|---:|---:|
| C0 fixed-byte | 44.30% | 28.98% | 2.253 | 2.432 |
| C1 codepoint | **44.92%** | 28.44% | 2.333 | 2.438 |
| W whitespace | 43.67% | **30.94%** | 2.311 | 2.440 |

Policy 간 순위는 decoding mode에 따라 바뀐다. Greedy에서 C1이 가장 높았고 sampled에서 W가 가장 높았지만 큰 seed variance에 비해 차이가 작다.

W − C0 UTF-8-valid-rate의 seed별 difference:

| Mode | Seed differences (%p) | 평균 | paired-t 95% CI |
|---|---|---:|---:|
| greedy | −2.34, −0.78, +6.25, +0.78, −7.03 | **−0.625%p** | [−6.621, +5.371] |
| sampled | −1.17, +5.08, +4.69, +0.78, +0.39 | **+1.953%p** | [−1.494, +5.400] |

두 interval 모두 0을 포함한다. W의 validity 우위나 열위를 주장하지 않는다. Gate는 기존에 고정한 mean harm margin만 검사한다.

## 4. UTF-8 failure가 지배적이었다

Unconstrained condition에서 다음 세 rate는 모든 policy·mode·seed에서 같았다.

- strict UTF-8 valid
- UTF-8 valid 그리고 U+FFFD 없음
- UTF-8 valid 그리고 conjoining-Jamo transition valid

즉 strict UTF-8로 decode된 continuation에서는 U+FFFD 직접 생성이나 malformed Jamo sequence가 관측되지 않았다. 절대 구조 실패의 주요 원인은 invalid/incomplete UTF-8 byte sequence였다.

Bytes/valid-codepoint는 sampled에서 약 2.43–2.44, greedy에서 2.25–2.33이었다. 이 수치는 output script mixture를 보여 줄 뿐 semantic quality 지표가 아니다.

## 5. UTF-8 hard-mask control

대표 seed 1,729의 3 policies × 2 decoding modes에 UTF-8 DFA를 적용했다.

- strict UTF-8 valid: **6/6 조건에서 256/256, 100%**
- U+FFFD-free: 6/6에서 100%
- Jamo transition: 5/6에서 100%
- W sampled Jamo transition: 255/256, 99.609%

마지막 결과가 중요하다. UTF-8 DFA는 encoding validity를 보장하지만 Hangul Jamo grammar를 보장하지 않는다. 두 constraint는 별도 층으로 다뤄야 한다.

Hard mask가 100%를 달성한 것은 당연한 implementation invariant이며 model quality 개선 증거가 아니다. Constraint가 선택한 byte probability mass, semantic quality, production kernel overhead는 이 실험이 측정하지 않았다.

## 6. 절대 validity가 낮은 이유와 해석 한계

이 모델은 1.25M parameters, 약 11M Korean training bytes, one pass로 학습된 mechanism pilot이다. 128 bytes 전체가 유효해야 하는 sequence-level metric은 한 번의 byte 오류만 있어도 실패하므로 길이에 따라 빠르게 낮아진다.

그럼에도 30–45%는 생성 model로서 높은 수치가 아니다. 이 pilot으로 자연어 품질, long-form coherence, 실제 chatbot 유용성을 평가하지 않는다.

또한 full-prefix recomputation은 현재 policy의 causal semantics을 보존하기 위한 reference path다. HF BLT의 현 cache API로 open patch를 동일하게 갱신하는 production incremental decoder를 구현한 것이 아니다.

## 7. Gate H 판정과 다음 단계

Phase 2의 scale-up stop conditions을 종합하면:

- Gate D causal replication: 통과
- Gate E whitespace mechanism/ecology: 통과
- Gate F cost/Pareto: 통과
- duplicate noise: stop 아님
- aligned packing: reversal 없음
- Gate G NFD opportunity: 실패; method에 포함하지 않음
- generation validity stop: 발동하지 않음

따라서 **Gate H를 열고 Phase 3 scale-up으로 진행**한다. 단, scale-up의 필수 요소는 다음이다.

1. 더 큰 모델·더 많은 Korean training bytes에서 절대 validity 재측정
2. SpaceByte-compatible actual training baseline
3. incremental generation과 CUDA latency를 teacher-forced MPS cost와 분리
4. UTF-8 DFA control을 품질 method와 섞지 않음
5. Hangul-Jamo grammar constraint는 본 patching paper의 후속 축으로 분리
