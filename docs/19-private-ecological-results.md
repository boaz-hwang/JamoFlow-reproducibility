# Phase 2c results: no large ecological regression, but codepoint gains reverse

> 작성일: 2026-08-10
>
> 사전 고정: [read-only ecological protocol](./18-private-ecological-protocol.md)
>
> 기계 판독 결과: [`results/phase2-ecological/summary.json`](../results/phase2-ecological/summary.json)
>
> 상태: **Gate E ecological component 통과**

## 1. 결론

Korean Wikipedia에서 학습한 compact byte-BLT를 사용자 작성 Markdown test partition에 그대로 적용했다. Whitespace-aware policy는 codepoint grid 대비 큰 회귀가 없었고, 사전 고정한 +0.02 BPB margin을 모든 seed에서 만족했다.

- W − C1 평균: **−0.01511 BPB**
- seed 범위: −0.03760 ~ +0.00452 BPB
- paired-t 95% CI: [−0.03528, +0.00507]
- hierarchical bootstrap 95% CI: [−0.02773, −0.00299]
- 5개 seed 모두 +0.02 BPB 이하: 통과
- exact 43 patches: 통과

그러나 이를 “외부 문서에서도 안정적으로 우월하다”로 해석하지 않는다. Seed-level t interval은 0을 포함했고, 이 표본은 한 사용자의 영어·code·Markdown 혼합 convenience sample이다. 입증된 것은 protocol의 **large-regression 배제**다.

## 2. Privacy와 corpus aggregate

Vault는 읽기 전용으로 사용했다. 추적 결과에는 경로, 파일명, 원문, content hash, 문서/sequence별 metric이 없는지 별도 promotion script로 검사했다.

| 항목 | 수치 |
|---|---:|
| 발견한 Markdown 파일 | 1,265 |
| nonempty records | 1,235 |
| exact-byte 중복 | 8 |
| unique records | 1,227 |
| content-hash test records | 119 |
| strict UTF-8 valid test records | 119 |
| 평가 bytes | 590,592 |
| 256-byte windows | 2,307 |

이는 Phase 0에서 사용한 기존 hash-test partition이다. 추가 학습이나 threshold calibration에 사용하지 않았다.

표본 구성은 한국어 자연 산문과 많이 다르다.

- Latin-mixed windows: 2,082 / 2,307, **90.25%**
- digit-mixed: 1,382, 59.90%
- Hangul-heavy: 707, 30.65%
- compatibility-jamo: 6
- modern-jamo: 0

따라서 이 결과는 한국어 대표 benchmark가 아니라 mixed-document stress check다.

## 3. 전체 quality

| 정책 | 5-seed 평균 BPB | seed SD |
|---|---:|---:|
| fixed byte C0 | 4.56658 | 0.01307 |
| codepoint grid C1 | 4.57841 | 0.01326 |
| whitespace grid W | **4.56330** | 0.01996 |

### 3.1 Primary ecological contrast

Seed별 W − C1:

```text
−0.01542, −0.03760, +0.00452, −0.02247, −0.00458
```

4/5 seed에서 W가 낮았다. Bootstrap interval은 0 아래였지만 seed-level t interval은 0을 포함했다. 초기화 5개를 모집단으로 보는 confirmatory 추론에서는 유의한 우위를 주장하지 않는다.

### 3.2 Codepoint diagnostic의 reversal

C1 − C0는 5/5 seed에서 positive였다.

- 평균: **+0.01183 BPB**, codepoint가 나쁨
- paired-t 95% CI: [+0.00326, +0.02041]
- hierarchical bootstrap 95% CI: [+0.00588, +0.01711]

Public Korean Wikipedia에서 C1은 C0보다 −0.00654 BPB 좋았지만 mixed Markdown에서는 방향이 반전됐다. Encoding-safe boundary가 모든 domain에서 quality-optimal이라는 주장은 불가능하다.

Hangul-heavy 707 windows에서 C1 − C0는 +0.00006 BPB, 95% CI [−0.00277, +0.00289]로 거의 같았다. Latin-mixed에서는 +0.01338 [+0.00406, +0.02269]이었다. 이 후자 차이가 전체 reversal에 기여했을 가능성이 크지만 strata가 겹치므로 인과 분해로 보지 않는다.

### 3.3 결과 확인 후 contextual contrast

W − C0는 첫 ecological 결과를 확인한 뒤 해석을 완성하기 위해 추가했다. Confirmatory contrast가 아니다.

- 평균: −0.00328 BPB
- seed별 negative: 3/5
- paired-t 95% CI: [−0.01575, +0.00920]
- hierarchical bootstrap 95% CI: [−0.01153, +0.00434]

즉 W는 가장 싼 fixed baseline보다 평균은 좋았지만, initialization을 넘어 안정적인 우월성은 확인되지 않았다.

## 4. 탐색적 strata

W − C1은 대부분의 50-window 이상 stratum에서 평균 negative였다.

| Stratum | Windows | 평균 W − C1 | paired-t 95% CI |
|---|---:|---:|---:|
| Hangul-heavy | 707 | −0.01086 | [−0.02353, +0.00181] |
| Latin-mixed | 2,082 | −0.01560 | [−0.03689, +0.00569] |
| whitespace Q1 | 577 | −0.00566 | [−0.02174, +0.01042] |
| whitespace Q4 | 576 | **−0.03210** | [−0.06169, −0.00251] |

Whitespace density가 높은 quartile에서 차이가 크다는 것은 method와 정합적이지만, quartile들은 multiplicity-adjusted confirmatory endpoint가 아니므로 mechanism 입증으로 쓰지 않는다.

## 5. Gate E 최종 판정

Whitespace-focused Gate E의 구성요소는 모두 닫혔다.

1. Public Korean primary C2 − C1: 통과
2. Delayed-grid mechanism control: 통과
3. Causal hash-placebo control: 통과
4. Whitespace-only가 C2를 거의 재현: 확인
5. Exact patch rate: 통과
6. Private ecological regression margin: 통과

따라서 **Gate E는 통과**한다. 단, 이 gate는 scale-up 후보를 남기는 조건이지 논문 결론이 아니다.

## 6. 연구 방향에 주는 제약

Phase 3에서는 `causal_codepoint_grid`를 단독 주 method로 확대하지 않는다. 대신 다음 세 정책을 동일 rate와 다중 domain에서 직접 비교해야 한다.

1. fixed-byte
2. SpaceByte-compatible whitespace boundary
3. exact-rate local whitespace-preference grid

Public Korean에서는 whitespace grid의 quality 신호가 강했지만 private mixed text에서 fixed대비 우위는 불확실했다. 따라서 scale-up은 한국어 자연 산문·영어/Markdown 혼합·정규화 stress를 분리해 보고해야 하며, 한 최종 평균으로 합치지 않는다.
