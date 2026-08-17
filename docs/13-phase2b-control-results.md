# Phase 2b results: whitespace signal survives mechanism controls

> 작성일: 2026-08-10  
> 사전 고정 addendum: [Phase 2b protocol](./12-phase2b-mechanism-control-protocol.md)  
> 기계 판독 결과: [`results/phase2-controls/summary.json`](../results/phase2-controls/summary.json)  
> 상태: **artifact/mechanism controls 완료**

## 1. 결론

Phase 2 primary의 delimiter-aware 이득은 단순 target shift나 같은 빈도의 causal random event로 설명되지 않았다. 하지만 punctuation을 포함해야 할 이유도 없었다.

- C2 eojeol − delayed `+2` grid: **−0.00740 BPB**
- C2 eojeol − rate-matched causal hash placebo: **−0.01562 BPB**
- C2 eojeol − whitespace-only: **−0.00014 BPB**, 95% CI가 0을 포함
- whitespace-only − delayed `+2` grid: **−0.00726 BPB**

따라서 현재 데이터가 지지하는 가장 좁고 정확한 설명은 다음이다.

> Compact Korean byte-BLT에서 target 주변의 실제 whitespace boundary를 이용하는 prefix-causal equal-rate grid가 generic codepoint grid, fixed phase shift, 그리고 event-rate-matched causal placebo보다 더 낮은 BPB를 보였다.

“Eojeol+punctuation”이나 “형태론”을 입증한 것은 아니다. Scale-up method는 `whitespace-aware causal grid`로 좁힌다.

## 2. Artifact controls

### 2.1 Exact duplicate

seed 1,729 C1을 같은 initialization, order, inputs, patch matrix로 다시 학습했다.

| 지표 | 결과 |
|---|---:|
| primary BPB | 2.391849590 |
| duplicate BPB | 2.391851683 |
| BPB 차이 | +0.000002093 |
| primary C1−C0 효과 대비 비율 | 0.0320% |
| 최대 parameter 절대차 | 0.001241 |
| 평균 per-sequence NLL 절대차 | 0.003385 nats |
| 최대 per-sequence NLL 절대차 | 0.016968 nats |

MPS training은 최종 parameter에서 bitwise deterministic하지 않았다. 따라서 checkpoint 동일성을 주장하지 않는다. 그러나 BPB noise는 protocol의 0.001 threshold보다 약 478배 작고 primary effect의 0.032%이므로 stop condition에 해당하지 않는다.

### 2.2 Aligned packing

UTF-8 codepoint boundary에서만 row가 시작·끝나도록 다시 pack하고 newline을 약 0.34% 삽입했다. 이 control은 primary와 합치지 않는다.

| Seed | aligned C0 BPB | aligned C1 BPB | C1−C0 |
|---:|---:|---:|---:|
| 1,729 | 2.387566 | 2.382067 | −0.005499 |
| 2,718 | 2.372738 | 2.366521 | −0.006217 |
| 31,415 | 2.392898 | 2.386161 | −0.006736 |
| 평균 |  |  | **−0.006151** |

3-seed paired-t 95% CI는 [−0.00769, −0.00461]이다. Arbitrary chunk start가 C1 이득의 유일한 원인이라는 설명은 기각된다.

## 3. Mechanism controls

모든 새 모델은 primary C1/C2와 seed별 initialization hash, training-order hash가 같았다. 모든 split과 policy가 정확히 43 data patches를 사용했다.

### 3.1 평균 quality

| 정책 | 평균 test BPB |
|---|---:|
| C1 causal codepoint | 2.374214 |
| C2 delimiter-aware | **2.367135** |
| early `−2` grid | 2.375227 |
| delayed `+2` grid | 2.374532 |
| causal hash placebo | 2.382752 |
| whitespace-only | 2.367276 |

Early와 delayed control은 C1에서 각각 +0.00101, +0.00032 BPB 차이였다. ±2-byte phase만으로 C2의 약 −0.007 BPB 개선을 만들지 못했다.

### 3.2 사전 고정 mechanism contrast

차이는 left−right이며 negative가 left 우위다.

| Contrast | seed별 차이 | 평균 | paired-t 95% CI | crossed bootstrap 95% CI |
|---|---|---:|---:|---:|
| C2 − delayed | −.00785, −.00865, −.00455, −.00805, −.00789 | **−.00740** | [−.00942, −.00538] | [−.00850, −.00591] |
| C2 − placebo | −.01688, −.01490, −.01268, −.01738, −.01625 | **−.01562** | [−.01796, −.01328] | [−.01704, −.01396] |

Bootstrap 수치는 동일한 test sequence가 seed 사이에 공유된다는 점을 반영해 교차 재표집한 값이다. [교정 기록](./34-crossed-bootstrap-correction.md)에 구현과 영향 범위를 남겼으며, seed-level paired-t gate와 결론은 변하지 않았다.

두 contrast 모두 addendum gate의 세 조건을 만족했다.

- mean ≤ −0.003
- 5/5 negative
- paired-t upper < 0

### 3.3 Whitespace decomposition

Calibration의 C2 early trigger 55,834개 중:

- whitespace: 48,113, **86.17%**
- punctuation: 7,721, 13.83%

Punctuation share가 50%를 넘지 않아 protocol대로 punctuation-only 모델은 학습하지 않았다.

Whitespace-only와 C2의 차이는 다음과 같다.

- seed effects: −.00042, −.00063, +.00043, −.00034, +.00025
- mean C2−whitespace: −0.000141 BPB
- paired-t 95% CI: [−0.000708, +0.000426]

Punctuation을 추가한 C2의 안정적인 추가 이득은 관측되지 않았다. 반면 whitespace-only − delayed는 평균 −0.00726 BPB, 95% CI [−0.00887, −0.00564]였다.

## 4. Placebo matching은 무엇을 통제했는가

Calibration early-trigger fraction은 다음처럼 맞았다.

- C2 target: 0.348644
- hash placebo: 0.348626
- absolute mismatch: 0.000019

Test에서:

| 정책 | event fraction | mean target displacement | p05/p95 displacement | patch p95/max |
|---|---:|---:|---:|---:|
| C2 | 0.3467 | +2.063 B | 0 / +4 B | 9 / 12 B |
| placebo | 0.3477 | +2.183 B | −1 / +4 B | 10 / 12 B |
| whitespace-only | 0.3228 | +2.122 B | 0 / +4 B | 9 / 12 B |
| delayed | 0 | +2.837 B | +2 / +4 B | 8 / 10 B |

Placebo는 event 빈도와 대략적인 length tail을 통제하지만 전체 boundary distribution을 정확히 같게 만들지는 않는다. 따라서 “whitespace semantics만이 유일한 원인”이라는 강한 인과 문장까지는 쓰지 않는다. 다만 delayed와 placebo 양쪽에 큰 일관된 차이가 있고 whitespace-only가 C2를 거의 재현하므로, generic phase/variance 대안보다 실제 whitespace association이 더 설득력 있다.

## 5. Gate 업데이트

### Gate D — causal replication: **통과**

- primary C1−C0: 통과
- aligned mean direction: negative, 통과
- duplicate noise: 허용 범위, 통과

### 강화된 Gate E mechanism component: **통과**

- C2 vs delayed: 통과
- C2 vs placebo: 통과
- exact patch count: 통과

전체 Gate E는 read-only ecological/external regression check 전까지 pending이다.

### Gate F/G/H

- Gate F: cost benchmark pending
- Gate G: NFC/NFD/Hangul-unit pending
- Gate H: validity와 cost stop condition pending

## 6. 연구 방향 수정

### 유지할 것

1. `causal_codepoint_grid`를 generic encoding-safe baseline/방법으로 유지
2. `causal_whitespace_grid`를 Korean-focused scale-up 후보로 유지
3. learned threshold에는 router와 batch-padding을 모두 비용에 포함
4. arbitrary/aligned packing을 별도 보고

### 폐기하거나 축소할 것

1. `entropy_threshold_codepoint`: primary에서 일관되게 나빠 주 method에서 폐기
2. punctuation 포함 C2: whitespace-only보다 유의하게 낫지 않아 축소
3. morphology FST와 multi-Jamo generation: 현재 evidence보다 앞서므로 이번 핵심 method에서 제외

### 신규성 경계

Whitespace를 쓰는 것 자체는 SpaceByte 때문에 신규가 아니다. 잠정 기여는 다음 조합으로 한정한다.

- Korean raw-byte BLT에서 exact-rate prefix-causal comparison
- Unicode-safe grid와 local whitespace preference의 분리
- phase·causal placebo·aligned packing·duplicate controls
- learned entropy의 encoding-interior concentration 및 variable-batch cost
- Korean normalization/Jamo robustness까지 이어지는 평가

Scale-up에서는 반드시 SpaceByte-compatible baseline을 실제 학습 비교에 포함해야 한다. 그렇지 않으면 whitespace-aware method의 선행연구 대비 위치를 방어할 수 없다.
