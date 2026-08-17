# Exploratory component profile 결과와 architecture decision

> 작성일: 2026-08-13
>
> 상태: **v5r3 primary-negative 이후 원인 진단 완료; 다음 candidate 선택**
>
> authoritative aggregate:
> `results/exploratory-component-profile-v1/summary.json`

## 1. 결론

W72의 작은 실제 속도 이득은 우연한 checkpoint 차이가 아니라 patch schedule이 줄인
boundary update에서 나왔다. 그러나 현재 incremental BLT의 대부분 시간은 두 정책이
똑같이 127번 실행하는 local byte path에 쓰인다. 따라서 W72 cadence만 더 공격적으로
만들거나 그대로 scale-up하는 것은 10% matched-quality 목표에 대한 합리적인 다음
투자가 아니다.

다음 주기법은 **한글 표기구조에 맞춘 multi-byte draft를 하나의 target forward로
검증하여 순차 local byte invocation 자체를 줄이는 방법**이어야 한다. 단, generic
multi-token prediction, Medusa-style heads, byte-level joint MTP, BLT self-speculation은
이미 선행연구가 있으므로 그 자체를 신규성으로 주장하지 않는다. 잠정 연구 질문은
다음과 같다.

> 완성형 한글의 유효한 초성·중성·종성 조합과 UTF-8 scalar 경계를 draft 분포와
> block schedule에 넣으면, 같은 target BLT와 exact greedy output을 유지하면서 generic
> byte-MTP보다 높은 accepted-bytes-per-verification과 실제 한국어 E2E speedup을 얻을
> 수 있는가?

이 질문의 답이 `아니오`이면 한국어 규칙을 붙인 speculative decoding은 논문의 주기법이
될 수 없다. 규칙의 언어학적 그럴듯함이 아니라 generic draft 대비 추가 wall-time
이득이 기여의 필요조건이다.

## 2. 2×2 결과

각 다섯 seed에서 candidate와 reference checkpoint를 W72와 C86 schedule로 교차 실행했다.
따라서 native W72--C86 비교에 섞인 weight 차이와 schedule 차이를 분리할 수 있다.

### Whole-trial decode

| checkpoint weights | C86 schedule | W72 schedule | 같은 checkpoint 감소 | 양수 seed |
|---|---:|---:|---:|---:|
| candidate | 355.450 ms | 345.314 ms | **2.852%** | 5/5 |
| reference | 356.060 ms | 345.942 ms | **2.842%** | 5/5 |

Native candidate-W72 대 reference-C86은 3.018%였다. 16-case exploratory subset과 더 작은
측정 protocol이라 v5r3의 2.628%를 대체하지는 않지만, 효과 방향과 크기는 일치한다.
무엇보다 같은 weight 안에서도 약 2.85%가 재현되므로 schedule의 인과적 systems effect가
weight 운으로 설명되지 않는다.

### Whole-trial end to end

| checkpoint weights | C86 schedule | W72 schedule | 같은 checkpoint 감소 | 양수 seed |
|---|---:|---:|---:|---:|
| candidate | 361.290 ms | 351.385 ms | **2.742%** | 5/5 |
| reference | 361.904 ms | 352.080 ms | **2.715%** | 5/5 |

Native pair 감소는 2.907%였다. TTFT는 candidate weights에서 +0.788%, reference weights에서
-3.216%로 불안정했다. v5r3와 마찬가지로 해석 가능한 차이는 prefill이 아니라 decode에
있다.

## 3. Patch event가 실제 차이를 거의 전부 설명한다

두 schedule은 127개 controlled continuation byte를 모두 순차 consume한다. 달라지는 것은
decode 중 새 patch 수뿐이다.

| schedule | prompt patch median | final patch median | decode-new patch median |
|---|---:|---:|---:|
| C86 | 21 | 43 | **22** |
| W72 | 18 | 36 | **18** |

Synchronized step diagnostic의 네 cell에서 non-boundary step은 2.353--2.360ms, boundary
increment는 2.514--2.562ms였다. 중앙값을 약 2.54ms로 두면 W72가 제거한 네 번의 decode
boundary event는 약 `4 × 2.54 = 10.16ms`를 설명한다. 실제 same-checkpoint whole-decode
차이는 candidate 10.136ms, reference 10.118ms다.

즉 관찰된 차이는 다음 단순식과 거의 맞는다.

```text
decode time ≈ 127 × common local-byte step + decode boundary count × boundary increment

C86 ≈ 127 × 2.36ms + 22 × 2.54ms ≈ 355.6ms
W72 ≈ 127 × 2.36ms + 18 × 2.54ms ≈ 345.4ms
```

이는 정밀한 production latency decomposition이 아니다. 각 step 뒤 synchronize한 diagnostic은
kernel overlap을 바꾸며 중앙값 합의 항등식도 아니다. 그럼에도 다음 세 증거가 같은 결론을
지지한다.

1. native whole-trial과 2×2 counterfactual whole-trial의 방향·효과 크기가 일치한다.
2. 네 cell의 공통 local step과 boundary increment가 checkpoint/schedule에 걸쳐 안정적이다.
3. 제거된 boundary 수에 increment를 곱한 값이 실제 whole-decode 차이와 사실상 같다.

따라서 synchronized component 값은 정확한 share가 아니라 **intervention choice를 위한
mechanism diagnostic**으로만 쓴다.

## 4. Component diagnostic

| component | synchronized per-call median | 호출 구조 |
|---|---:|---|
| local encoder + hash embedding | 1.376--1.394 ms | 모든 byte, 2,540회/cell family |
| local decoder + global cross-attention | 1.244--1.266 ms | 모든 byte, 2,540회/cell family |
| byte LM head | 0.137--0.139 ms | 모든 byte, 2,540회/cell family |
| patch finalize + encoder cross + global update | 2.655--2.692 ms | W 355회, C 430회 |

Selector 자체는 W72 604ns/byte, C86 561ns/byte였고 idle synchronize 중앙값은
0.000125ms였다. Python selector가 현재 gap의 원인이라는 가설은 기각된다.

공통 local-byte base를 위 식으로 보면 reference decode의 약 84%가 patch schedule로
없앨 수 없는 경로다. W72 이후 남은 18 boundary increment를 overhead 없이 모두 지워도
절대 절감 가능량은 약 45.7ms뿐이다. C86 대비 10% 목표에 도달하려면 W72에서 추가로 약
25.5ms를 줄여야 하므로, global-only 방법은 남은 boundary 비용의 절반 이상을 zero-cost로
제거해야 한다. Verification, cache 관리, draft 실패 비용까지 고려하면 여유가 너무 작다.

## 5. Fable 5 검토에 대한 최종 판정

외부 검토의 가장 중요한 진단은 맞았다. Global patch 수를 줄이는 것만으로는 local
byte-sequential 경로가 남기 때문에 큰 E2E 개선으로 직결되지 않는다. 2×2 profile은 이
주장을 추측이 아니라 실제 runtime 경로로 좁혔다.

다만 다음 표현은 여전히 채택하지 않는다.

- `16.3% × global share`는 latency의 이론적 상한이 아니다.
- 효과는 0이 아니라 native v5r3에서 2.5--2.6%, exploratory same-weight 비교에서
  2.7--2.9%로 안정적이다.
- synchronized component 합을 production share로 부를 수 없다.
- 같은 parameter 수만으로 runtime memory가 같다고 미리 결론낼 수 없다.

올바른 결론은 **W72는 원래 의도한 positive efficiency technique에는 실패했지만,
boundary placement가 품질과 소폭 실제 latency에 미치는 효과를 잘 식별한 diagnostic
baseline**이라는 것이다.

## 6. 선행연구가 막는 신규성 주장

다음은 이미 선점됐다.

- [Fast BLT](https://arxiv.org/abs/2605.08044): local decoder가 patch 경계를 넘어 byte를
  draft하고 full model이 한 번에 검증하는 BLT-S, 병렬 block diffusion인 BLT-D, 그리고
  diffusion+verification을 제안한다. 따라서 `BLT에 self-speculation을 붙였다`는 기여가
  아니다. 보고된 주요 비용 수치도 estimated memory-bandwidth이므로 JamoFlow의 Apple
  wall time을 대신하지 않는다.
- [Multi-token Prediction](https://arxiv.org/abs/2404.19737): shared trunk 위 여러 future
  head와 self-speculative decoding, byte-level multi-byte training을 이미 보였다.
- [Medusa](https://arxiv.org/abs/2401.10774): 여러 head와 tree verification으로 lossless
  greedy acceleration을 수행한다. 단순 offset별 head는 신규성이 아니다.
- [MtPC](https://arxiv.org/abs/2511.11346): fully factorized MTP의 future-byte 독립 가정을
  비판하고 probabilistic circuit으로 joint future-byte 분포를 모델링하며, EvaByte 6.5B와
  byteified Llama 3.2 3B에서 speculative verification을 평가한다. Generic joint byte-MTP나
  dependence-aware head도 신규성이 아니다. 다만 이 연구는 English generation만 평가했고
  UTF-8 writing-system structure를 draft factorization으로 사용하지 않았다.
- Korean three-hot, Jamo BPE, SCRIPT는 Hangul factorization의 표현 효용을 선점했다. Jamo
  factorization 자체도 신규성이 아니다.

그러므로 가능한 신규 기여는 이 선행요소의 단순 합이 아니라 다음 empirical statement여야
한다.

> Generic byte-MTP와 동일한 target·data·parameter/cost envelope에서, Hangul composition과
> scalar alignment를 이용한 draft가 Korean에서 accepted bytes per verification과 실제
> matched-output wall time을 추가로 개선한다.

이 contrast가 없으면 `Korean-aware`라는 이름만 붙인 재구현이다.

## 7. 선택한 구조: orthography-aligned verified multi-byte decoding

명칭은 결과 전 잠정적이며, method contract는 다음과 같다.

1. **Target 보존**: quality-qualified W72 BLT의 next-byte distribution과 standard greedy
   output을 authority로 유지한다.
2. **Generic control**: 동일 hidden state·parameter budget에서 offset별 byte heads 또는
   fully-factorized byte-MTP를 학습한다.
3. **Hangul draft**: 다음 scalar가 precomposed Hangul일 때 초성 19, 중성 21, 종성 28의
   factorized/joint head로 완전한 유효 조합을 제안하고 NFC scalar와 세 UTF-8 bytes로
   합성한다. 비한글은 generic byte fallback을 쓴다.
4. **Scalar-aligned window**: draft block은 UTF-8 scalar 중간에서 끝나지 않는다. 길이는
   현재 prefix state와 제안 scalar 유형에서 결정하며 hidden final-test metric을 보지 않는다.
5. **Exact verification**: target BLT가 proposed bytes를 cached block forward로 검증한다.
   첫 mismatch 전까지만 수락하고 mismatch byte는 target argmax를 사용한다. Greedy output은
   byte-for-byte baseline과 같아야 한다.
6. **Cost accounting**: draft head, block append, rejected suffix, cache copy/rollback, DFA,
   host synchronization을 모두 timed scope에 넣는다.

초성·중성·종성 head를 독립으로 두는 것이 최종안이라고 미리 가정하지 않는다. MtPC가
지적하듯 future-byte independence는 잘못된 조합을 만들 수 있다. Hangul에서는 모든
19×21×28 조합이 유효 scalar라는 장점이 있지만 문맥 확률의 상관은 남는다. 따라서
fully-factorized Jamo head, 작은 joint/low-rank head, generic byte-MTP를 acceptance와
latency 양쪽에서 비교한다.

## 8. 다음 실험의 순서와 kill rule

### A. Calibration-only opportunity/acceptance preflight

새 final test를 만들거나 열기 전에 train/calibration에서만 다음을 측정한다.

- output byte 중 ASCII/2-byte/3-byte/4-byte scalar와 precomposed Hangul 비율
- perfect scalar oracle이 줄일 수 있는 sequential target invocation의 상한
- generic byte-MTP, UTF-8 scalar-aligned generic MTP, Hangul-factorized draft의 top-1 및
  exact-prefix acceptance
- block당 accepted bytes, verifier calls per emitted byte, rejection 위치
- draft parameter·forward time과 cached block-verification time

Oracle은 가능성만 보여주며 speed claim이 아니다. Learned draft가 generic control을 넘지
못하면 Korean-specific branch를 중단한다.

### B. 저비용 frozen-target prototype

기존 다섯 W72 checkpoint를 freeze하고 작은 draft head만 train split에서 학습한다. 먼저
head-only retrofit을 쓰는 이유는 candidate 전체를 다시 pretrain하기 전에 acceptance와
systems feasibility를 싸게 판별하기 위해서다. 한 seed에서 hyperparameter를 고른 뒤 다른
seed와 calibration documents로 재현한다.

Go 조건은 다음 세 축의 교집합이다.

1. Hangul-aware draft가 같은-cost generic draft보다 acceptance/throughput proxy에서 낫다.
2. block verifier를 포함한 measured calibration-only E2E가 W72 AR보다 충분히 빨라 10%
   final 목표에 현실적인 여유가 있다.
3. exact greedy equivalence, strict UTF-8, first-boundary stop, cache equivalence가 모두
   bitwise 또는 사전 고정 tolerance contract를 통과한다.

### C. 새 matched-quality actual timing

Preflight가 통과할 때만 새 disjoint final-quality/timing protocol을 봉인한다. Comparator는
W72 standard AR, generic byte-MTP+verification, Hangul-aware candidate의 세 축이다.
`candidate vs W72`가 총효율을, `candidate vs generic MTP`가 한국어 구조의 추가 가치를
식별한다. 최종 주장에는 두 contrast가 모두 필요하다.

### 중단 조건

- perfect scalar oracle조차 target-call reduction 여유가 작으면 종료
- learned Hangul draft가 generic control을 안정적으로 넘지 못하면 Korean-specific claim
  종료
- acceptance는 높지만 block verification/cache overhead 때문에 actual E2E가 10% 미만이면
  scale-up 금지
- quality를 바꾸는 approximate acceptance만 빠르면 사용자 기준의 연구 성공으로 보지 않음

## 9. 논문 구성에 미치는 영향

현 boundary-patching 결과만으로는 사용자가 정한 `실제 추론 효율 개선` 기준의 가치 있는
positive paper가 완성되지 않았다. 이 결과는 폐기하지 않고 다음 논문의 동기와 인과 ablation,
그리고 강한 negative baseline으로 사용한다.

새 구조가 성공하면 논문의 중심은 다음 세 단계가 된다.

1. 같은-rate Korean boundary placement가 quality를 개선한다.
2. 실제 profiling은 global cadence만으로 2.5--2.9%밖에 얻지 못함을 보인다.
3. Hangul-aligned verified multi-byte decoding이 generic byte-MTP보다 acceptance와 실제
   matched-output latency를 더 개선한다.

세 번째가 실패하면 첫 두 단계는 정직한 systems study로 남지만, positive efficiency
기법이라는 최종 목표는 달성되지 않은 것으로 판정한다.

## 10. Opportunity preflight 개봉 결과

Calibration-only data oracle은 다음 neural 단계의 최소 opportunity gate를 통과했다.
Complete byte의 86.389%가 precomposed Hangul에 속했고, perfect Hangul-only scalar block은
target calls를 57.593% 줄일 수 있었다. 이는 전체 scalar grouping savings의 98.681%다.

동시에 규칙 결정론 가설은 기각됐다. 첫 Hangul UTF-8 byte를 조건으로 해도 continuation
pair entropy는 6.305 bits이고 두 continuation byte의 conditional mutual information은
2.409 bits였다. 독립 byte mode의 context-free exact-pair rate는 6.952%에 불과했다.
따라서 규칙 engine이 나머지를 그대로 출력하는 구조가 아니라, target context를 이용한
cheap joint/conditional draft를 학습해야 한다. 결과와 claim 경계는
`docs/95-hangul-block-opportunity-result.md`에 기록한다.
