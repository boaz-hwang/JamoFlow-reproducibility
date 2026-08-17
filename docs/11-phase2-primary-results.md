# Phase 2 primary results: causal Korean-aware patching

> 작성일: 2026-08-10  
> 사전등록: [Phase 2 protocol](./10-phase2-korean-causal-protocol.md)  
> 기계 판독 결과: [`results/phase2-korean-primary/summary.json`](../results/phase2-korean-primary/summary.json)  
> 상태: **primary run 완료, 인과·비용·강건성 control 진행 중**

## 1. 먼저 결론

compact BLT의 한국어-only paired experiment에서 두 현상은 매우 안정적으로 재현됐다.

1. 미래 후보를 보지 않는 `causal_codepoint_grid`는 같은 43-patch `fixed_byte_6`보다 평균 **−0.00654 BPB** 좋았다.
2. 같은 43 patches에서 whitespace·punctuation 뒤를 우선하는 `causal_eojeol_grid`는 codepoint grid보다 다시 **−0.00708 BPB** 좋았다.

두 효과 모두 다섯 seed 전부에서 음수였고 paired-t 및 seed × 공통 sequence 교차 bootstrap의 95% 구간도 0 아래였다. 따라서 Phase 1 결과는 단순 offline nearest-boundary lookahead만의 산물이 아니었다.

하지만 아직 다음 명제를 주장할 수는 없다.

> “한국어 어절이라는 언어학적 단위 자체가 개선을 일으켰다.”

`causal_eojeol_grid`는 delimiter 정렬 외에도 target 대비 경계 위상과 patch-length tail을 바꾼다. 일반적인 지연 경계나 같은 빈도의 causal placebo event도 같은 효과를 낼 수 있다. 이 대안을 직접 제거하기 전에는 method paper의 중심 주장을 “Korean eojeol-aware”로 고정하지 않는다.

## 2. 고정된 실험 조건

- 언어: 한국어만 사용
- 학습: 10,999,808 bytes, 42,968 sequences
- calibration/test: 각각 999,936 bytes, 3,906 sequences
- 모델: 1,251,136-parameter compact HF BLT
- seed: 1,729 / 2,718 / 31,415 / 57,721 / 65,537
- 모든 정책이 seed 안에서 같은 초기화 hash와 같은 train-order hash 사용
- C0/C1/C2는 모든 split에서 sequence당 정확히 43 data patches
- threshold는 seed별 calibration split에서만 결정
- 결과 확인 전 protocol commit: `90346f8`

다섯 정책의 초기화 hash는 각 seed 안에서 완전히 같았고, 세 구조 정책의 patch-matrix hash는 seed와 무관하게 같았다.

## 3. 품질 결과

### 3.1 평균 test BPB

| 정책 | 평균 BPB | seed SD |
|---|---:|---:|
| C0 `fixed_byte_6` | 2.380752 | 0.016159 |
| C1 `causal_codepoint_grid` | 2.374214 | 0.016547 |
| C2 `causal_eojeol_grid` | **2.367135** | 0.017483 |
| C3 `entropy_threshold_full` | 2.376937 | 0.013871 |
| C4 `entropy_threshold_codepoint` | 2.388774 | 0.014212 |

### 3.2 사전등록 contrast

차이는 `left − right`이며 음수일수록 left가 좋다.

| Contrast | seed별 BPB 차이 | 평균 | paired-t 95% CI | crossed bootstrap 95% CI |
|---|---|---:|---:|---:|
| C1 − C0 | −.00650, −.00667, −.00570, −.00700, −.00681 | **−.00654** | [−.00716, −.00591] | [−.00709, −.00595] |
| C2 − C1 | −.00787, −.00805, −.00394, −.00878, −.00675 | **−.00708** | [−.00944, −.00472] | [−.00848, −.00534] |
| C4 − C3 | +.01248, +.01410, +.01334, +.00428, +.01498 | **+.01184** | [+.00647, +.01721] | [+.00787, +.01454] |
| C1 − C3 | +.00013, −.00028, +.00127, −.01291, −.00183 | −.00272 | [−.00993, +.00448] | [−.00799, +.00069] |
| C2 − C3 | −.00775, −.00833, −.00267, −.02169, −.00857 | **−.00980** | [−.01858, −.00102] | [−.01619, −.00516] |

Bootstrap 수치는 동일한 test sequence가 seed 사이에 공유된다는 점을 반영해 교차 재표집한 값이다. 변경 이유와 영향은 [교정 기록](./34-crossed-bootstrap-correction.md)에 정리했다. 사전등록 판정의 주 근거인 seed-level paired-t 결과는 바뀌지 않았다.

### 3.3 무엇이 반증됐는가

`entropy_threshold_codepoint`는 `entropy_threshold_full`보다 다섯 seed 모두 나빴다. Phase 1의 offline top-k에서는 codepoint 제한의 손상이 작았지만, causal threshold에 같은 제한을 단순 적용하면 평균 +0.01184 BPB가 됐다.

따라서 다음 두 문장은 서로 다르다.

- UTF-8 내부 경계를 피하는 **고정률 causal grid**가 fixed-byte보다 낫다.
- learned entropy trigger를 codepoint 후보에서만 허용하면 낫다.

첫 문장은 지지됐고 둘째 문장은 현재 설정에서 반증됐다. codepoint restriction이 cap과 threshold trigger의 timing을 함께 바꾸기 때문이다. 이후 연구는 C4를 주 method로 확장하지 않는다.

## 4. 평균 patch rate가 감춘 비용

Calibration에서는 C3/C4가 목표 43.0 patches에 ±0.1 이내로 맞았다. test 평균도 C3 42.991, C4 43.200으로 가깝다. 그러나 분산이 매우 컸다.

| 정책 | test 평균 patches | corpus-wide padded width 평균 | zero-padding 비율 평균 | train 시간 평균 |
|---|---:|---:|---:|---:|
| C0/C1/C2 | 43.000 | 43.0 | 0% | 58.9 s |
| C3 | 42.991 | 111.6 | 61.46% | 78.3 s |
| C4 | 43.200 | 126.6 | 65.84% | 82.9 s |

train split에서는 calibration threshold를 그대로 적용했을 때 C3 46.166, C4 48.355 patches까지 drift했다. 이는 data leakage가 아니라 calibration-only threshold를 고정한 결과이지만, “평균 43 patches”를 모든 split의 동등 compute로 해석할 수 없다는 뜻이다.

학습 시간은 inference latency가 아니므로 Gate F 판정에 직접 쓰지 않는다. 다만 variable patch count의 batch-max padding을 실제 cost model과 benchmark에서 반드시 포함해야 한다는 근거다.

## 5. 한국어 strata에서 보이는 것과 보이지 않는 것

Primary test의 3,906 windows 중 3,892개가 Hangul-heavy였다. C1−C0는 다음 두 chunk-start stratum에서 모두 비슷했다.

- codepoint boundary에서 시작: −0.00638 BPB
- codepoint 내부에서 시작: −0.00664 BPB

따라서 arbitrary chunk start만으로 전체 효과를 설명하기는 어렵다. C2−C1 역시 whitespace-density 네 quartile 모두 음수였고, 낮은 세 quartile에서도 개선됐다. 단순히 공백이 많은 window만 이득을 만든 것은 아니다.

그러나 희소 stratum은 결론에 쓰지 않는다.

- Latin-mixed: 151
- Hanja-mixed: 31
- compatibility-jamo-present: 8
- modern-jamo-present: 0

특히 자연 발생 modern Jamo가 0개이므로 normalization/Jamo claim은 별도 결정적 변환 실험 없이는 불가능하다.

## 6. 현재 gate 판정

### Gate D — causal replication

Arbitrary-packing primary component는 통과했다.

- mean C1−C0 ≤ −0.003: 통과
- 4/5 이상 negative: 5/5, 통과
- paired-t upper < 0: −0.00591, 통과

전체 Gate D는 aligned-packing 3-seed 방향 확인 전까지 `pending`이다.

### Gate E — Korean eojeol value

Primary component와 exact-rate component는 통과했다.

- mean C2−C1 ≤ −0.003: 통과
- 4/5 이상 negative: 5/5, 통과
- exact patch count: 통과

하지만 private ecological/external diagnostic과 causal placebo controls 전까지 `pending`이다. 원 protocol보다 더 엄격하게, 아래 7절의 alternative mechanism을 제거해야 “eojeol value”라고 부른다.

### Gate F — parameter-free Pareto

품질 component는 통과했다. C1과 C2 모두 C3의 0.015-BPB harm margin 안이고 평균으로는 더 좋았다. analytical FLOPs, batch-1 direct latency, padding-aware cost가 남았다.

### Gate G/H

Normalization robustness, duplicate noise, aligned packing, generation validity, cost가 아직 없으므로 판정하지 않는다.

## 7. primary 결과가 새로 요구한 인과 control

이 control은 primary 결과를 본 뒤 추가됐으므로 primary confirmatory endpoint와 섞지 않고 **Phase 2b mechanism control**로 표시한다.

1. **Delayed grid:** delimiter를 전혀 쓰지 않고 각 target `+2` 뒤 첫 codepoint에서 경계를 낸다. C2 개선이 단순 right shift 또는 patch-lag 완화인지 검사한다.
2. **Early grid:** 각 target `−2` 뒤 첫 codepoint를 사용한다. 절대 grid phase에 민감한지 검사한다.
3. **Causal placebo-event grid:** 관측된 prefix의 rolling hash event를 calibration에서 delimiter-trigger 빈도에 맞춘 뒤 C2와 같은 window/deadline 논리를 쓴다. delimiter 의미 없이 비슷한 경계 변동성을 준다.
4. **Whitespace-only vs punctuation-only decomposition:** 실제 C2가 어느 delimiter class에 의해 움직였는지 먼저 집계하고, 충분한 event 수가 있을 때만 분해한다.
5. **Boundary-distribution accounting:** C1/C2/control의 target displacement, patch-length histogram, delimiter-hit rate를 함께 보고한다.

C2가 delayed/early/placebo보다 계속 좋을 때만 한국어 어절 prior를 중심 기여로 유지한다. 그렇지 않으면 논문의 중심은 “UTF-8-safe causal grid와 patch-lag/variance 분석”으로 축소한다.

## 8. 현 시점의 올바른 연구 방향

현재 가장 강한 방법 후보는 복잡한 자소 FSM이나 morphology FST가 아니다. 데이터가 지지하는 최소 주장은 다음이다.

> 한국어 raw-byte BLT에서 parameter-free prefix-causal codepoint grid는 fixed-byte와 learned entropy threshold에 비해 품질–비용의 강한 후보이며, 관측된 delimiter-aware 추가 이득은 인과 placebo control을 통과해야 한국어 어절 효과로 해석할 수 있다.

따라서 다음 순서는 다음과 같다.

1. duplicate와 aligned packing으로 artifact 제거
2. Phase 2b control로 eojeol 의미와 generic boundary phase 분리
3. padding-aware cost와 direct latency
4. NFC/NFD/Hangul-unit 및 생성 validity
5. 위 결과로 scale-up 정책 하나만 선택

자소 단위 multi-symbol generation과 형태론 FST는 여전히 장기 방향이지만, 현재 실험이 입증한 범위를 넘어선다. 먼저 더 단순한 경계 가설을 강하게 확립하는 편이 출판 가능한 논문에 가깝다.
