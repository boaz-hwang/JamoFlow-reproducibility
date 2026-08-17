# Conditional local compute frozen-W72 sensitivity protocol

> 작성일: 2026-08-14
>
> 상태: **frozen calibration evaluation 전 봉인 예정**
>
> protocol: `jamoflow-conditional-local-frozen-sensitivity-v1`

## 1. 목적

정적 `thin160 E1/D1`은 실제 generation을 22.8--24.3% 줄였지만 calibration BPB를
0.0956 악화시켰다. 이 결과는 local path가 latency 병목인 동시에 품질에 필요한 capacity임을
보였다. 다음 비싼 학습 전에 기존 W72 seed-1729 checkpoint를 고정하고, position-conditional
local update가 정적 thinning보다 충분히 작은 품질 위험을 보이는지 검사한다.

이 단계는 frozen-checkpoint sensitivity screen이다. Conditional architecture로 다시 학습한
품질이나 실제 latency를 측정하지 않으며 publication efficiency claim을 만들지 않는다.

## 2. Causal route

두 route는 현재 위치의 byte까지 소비한 prefix만 사용한다.

- `utf8_incomplete`: strict UTF-8 scalar가 아직 닫히지 않은 위치. 모든 script에 적용되는
  generic orthographic control이다.
- `hangul_prefix`: 현재 prefix가 precomposed Hangul U+AC00--U+D7A3로 완성될 수 있는
  EA--ED lead/second-byte 범위. Future continuation byte를 읽지 않는다.

Hangul route는 generic UTF-8 route의 causal subset이어야 한다. Route는 model logit,
target byte, learned router 또는 test/final metric을 사용하지 않는다.

모델 출력을 계산하기 전에 같은 8,000,000-byte calibration input에서 route geometry만
감사했다. `utf8_incomplete`는 4,664,439 positions(58.3054875%), `hangul_prefix`는
4,602,889 positions(57.5361125%)이며 Hangul positions는 generic positions의
98.6804415%다. 따라서 두 route의 계산 노출은 이 Korean stream에서 거의 같지만 처치
집합도 거의 겹친다. 이 screen은 두 route의 안전성을 함께 확인할 수 있을 뿐,
Hangul-specific effect를 식별하는 실험은 아니다.

## 3. 고정 2×2×2 factorial

모든 candidate는 original W72의 19,596,096 parameters와 checkpoint를 그대로 사용한다.
새 parameter를 더하지 않는다.

| 축 | 수준 |
|---|---|
| route | `utf8_incomplete`, `hangul_prefix` |
| component | decoder second stage only, encoder+decoder second stage |
| operator | second-layer MLP residual만 생략, second layer 전체를 생략하되 K/V semantics 보존 |

Full-sequence frozen evaluation은 dense kernel로 전체 값을 계산한 뒤 easy position residual을
mask한다. `second_layer_kv`의 easy output은 layer input identity지만 hard future position이
참조할 K/V는 해당 input에서 유지되는 모델 정의다. 이후 실제 incremental prototype은 같은
정의를 K/V-only update로 구현해야 하며 full conditional forward와 logit/cache가 일치해야
한다.

예상 절감이 큰 pair부터 고정 순서는 다음과 같다.

1. encoder+decoder `second_layer_kv`
2. decoder `second_layer_kv`
3. encoder+decoder `second_mlp`
4. decoder `second_mlp`

각 pair 안에서는 generic UTF-8와 Hangul route를 모두 평가한다. 결과를 보고 route, operator,
component 또는 순서를 추가하지 않는다.

## 4. 데이터와 통계

- checkpoint: 기존 W72 seed 1729 exact artifact/state
- patch schedule: 기존 W72 matrix
- data: 8,000,000-byte Korean calibration stream, 15,625×512 windows
- comparator: 기존 exact per-sequence W72 calibration NLL
- estimand: conditional frozen NLL - original W72 NLL, BPB
- document bootstrap: whole document 10,000회, seed 20,261,101
- coverage: 기존 386 documents, 15,240/15,625 eligible windows

Frozen sensitivity pass는 다음을 모두 요구한다.

1. full-stream mean difference <= 0.020 BPB
2. document bootstrap one-sided 95% upper <= 0.020 BPB
3. easy-position rate >= 30%
4. eligible-window coverage >= 95%

0.020은 최종 matched-quality margin 0.010의 두 배인 **학습 위험 screen**이다. 이를 최종
noninferiority로 재사용하지 않는다. Static candidate의 +0.0956과 같은 큰 손상만 싼 단계에서
제거하고, 선택 candidate는 새 학습 후 다시 0.010 margin을 통과해야 한다.

## 5. 선택과 stop rule

한 operator/component pair의 UTF-8와 Hangul route가 **둘 다** frozen gate를 통과해야 한다.
고정 pair order의 첫 통과 pair만 두-route actual-runtime prototype을 허가한다. 이는 Hangul
variant만 결과를 보고 고르거나 generic comparator를 누락하는 것을 막는다.

어떤 pair도 통과하지 못하면 현재 후보 집합을 actual-runtime 단계로 진행하지 않는다.
Margin을 늘리거나 route rate를 줄이거나 candidate를 추가하지 않는다. 이는 dense W72
가중치의 frozen perturbation screen에 따른 자원배분 stop이며, 처음부터 conditional graph로
학습한 모든 모델의 가능성을 과학적으로 기각하는 결과는 아니다. 통과하더라도 다음에
허가되는 것은 random/frozen actual-runtime correctness와 latency preflight뿐이며, 한-seed
training은 그 preflight가 별도 gate를 통과한 뒤에만 연다.

## 6. Claim 경계

이 screen은 calibration-only이며 historical test, sealed final, downstream, 기존 latency
result를 selection input으로 읽지 않는다. Static failure를 pass로 바꾸지 않으며 frozen
checkpoint의 conditional sensitivity를 trained conditional quality라고 부르지 않는다.

일반 conditional depth는 Mixture-of-Depths 등 선행연구가 있다. 후속 기여가 성립하려면
prefix-only orthographic route, BLT local path 적용, generic UTF-8와 Hangul의 same-operator
대조, matched-quality multi-seed actual generation을 모두 보여야 한다.

이 screen에 쓴 calibration stream은 architecture selection에 노출되므로 후속 trained
candidate의 confirmatory quality stream으로 재사용하지 않는다. 한-seed training 전에
기존 train/calibration/final 문서와 분리된 새 Korean validation stream을 봉인한다. 또한
Korean-specific claim은 이 화면의 거의 겹치는 두 mask만으로 열지 않는다. 최소한 계산량을
결속한 generic comparator와 Hangul/non-Hangul coverage interaction 또는 별도 script-stratified
evidence가 필요하다. 그렇지 않으면 기여 범위는 Korean data에서 검증한 generic
UTF-8-structural conditional computation으로 제한한다.
