# Exploratory incremental component profile v1

> 작성일: 2026-08-13
>
> 상태: **v5r3 primary-negative 결과 개봉 후 고정한 원인 진단**

## 목적

W72의 8.332% counted dense-matmul 감소가 2.5–2.6% E2E 감소로만 나타난 원인을
분해한다. 이 분석은 confirmatory 효율 검정이 아니라 다음 구조를 선택하기 위한
outcome-aware exploratory profiling이다. 기존 v5r3 artifact, 통계, gate에는 손대지
않는다.

## 2×2 교차 설계

서로 다른 checkpoint의 native runtime만 비교하면 schedule과 weight가 섞인다. 따라서
각 다섯 seed에서 candidate와 reference checkpoint를 각각 두 schedule로 실행한다.

| checkpoint | W72 schedule | C86 schedule |
|---|---:|---:|
| candidate weights | native | counterfactual runtime |
| reference weights | counterfactual runtime | native |

Counterfactual cell은 quality evidence가 아니며 생성 내용을 평가하지 않는다. 같은
가중치와 동일 controlled bytes에서 patch schedule만 바꾼 systems diagnostic이다.

## 고정 workload

- exact v5r3 case artifact
- 다섯 physical model seed
- 128-byte prompt + controlled continuation의 앞 127 observed bytes
- 첫 4 case는 warmup
- 다음 16 case × 3 repetition은 whole-trial measurement
- 다음 4 case는 synchronized step/component diagnostic
- W72: causal whitespace grid, 72 patches / 512-byte horizon
- C86: causal codepoint grid, 86 patches / 512-byte horizon

Whole-trial measurement는 v5r3 controlled path와 같은 prefill/decode synchronization
경계를 사용한다. Schedule order는 checkpoint×seed×case×repetition에서 교차시킨다.
Repetition은 독립 표본으로 세지 않는다.

## 계측

1. whole TTFT, decode, E2E
2. prompt/final/decode-new patch counts
3. 매 byte 뒤 synchronize한 boundary/non-boundary step time
4. 다음 method를 각각 pre/post synchronize한 per-call diagnostic
   - local encoder와 hash embedding
   - patch reduction + encoder cross-attention + global update
   - local decoder와 global cross-attention
   - byte LM head
5. 빈 MPS synchronize와 pure CPU selector의 별도 overhead

MPS Event는 이 환경의 PyTorch 2.13 probe에서 종료되지 않아 사용하지 않는다. 명시적
synchronize는 kernel dispatch overlap을 바꾸고 barrier overhead를 추가한다. 따라서
component 시간의 합을 production latency share로 해석하거나 v5r3 E2E 수치를 대체하지
않는다. Whole-trial 결과가 native cell에서 v5r3 방향과 대략 일치하는지 먼저 확인하고,
step/component 값은 병목 순위와 다음 intervention 선택에만 사용한다.

## 사전 해석 규칙

- 같은 checkpoint에서 W72 schedule이 일관되게 빠르고 그 차이가 boundary-step 감소로
  설명되면 2.5% 효과는 weight 우연이 아니라 schedule 경로 효과로 본다.
- non-boundary 127-step 공통 경로가 대부분의 decode 시간을 차지하면 W cadence만
  scale-up하지 않고 multi-byte/block 또는 local self-speculation을 우선한다.
- patch-finalize 비용이 예상보다 지배적이면 larger global/local ratio의 inference-only
  preflight를 추가할 수 있으나, quality 없는 shape extrapolation은 publication evidence가
  아니다.
- 결과가 불안정하면 profiler 계측을 결론으로 사용하지 않고 v5r3 raw E2E만 유지한다.

## 산출물과 claim 경계

Raw profile arrays는 ignored `artifacts/exploratory-component-profile-v1/`에 두고,
aggregate와 provenance만 `results/exploratory-component-profile-v1/summary.json`에
추적한다. 이 분석은 final-test-blind, preregistered 또는 hardware-general claim을 하지
않는다.
