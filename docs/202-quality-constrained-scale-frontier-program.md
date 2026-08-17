# 품질 제약 규모 frontier 후속 연구 프로그램

> 작성일: 2026-08-17  
> 상태: **현재 ARR 원고를 변경하지 않는 출판 후 후속 연구 방향**  
> 선행 결론: [규모 확장 결과를 반영한 연구 방향 수정](./200-revised-scale-research-direction.md)

## 1. 왜 단순한 대형 모델 추가 실험이 아닌가

현재 증거는 서로 다른 두 질문에 답한다.

- 같은 random weights에서 W72와 C86을 비교하면 model scale이 커질수록 schedule의
  systems headroom이 커졌고, 1.618B에서 controlled reduction이 10.217%였다.
- 실제로 학습한 188.6M에서는 W72가 품질을 잃었다. 품질을 회복한 W80은 실제
  controlled/free time을 2.887%/2.475% 줄였지만 compact 결과보다 커지지 않았다.

따라서 `parameter count`만 늘리는 실험은 세 원인을 섞는다.

1. global event 한 번의 비용이 규모와 함께 변하는 효과
2. 학습량 부족 때문에 candidate와 reference의 품질 차이가 변하는 효과
3. 품질을 맞추기 위해 W72에서 W80처럼 patch density를 바꾸는 효과

후속 연구의 목표는 양의 결과를 더 찾는 것이 아니라 이 세 축을 분리하는 것이다.
현재 188.6M screen은 0.6785 raw byte/parameter만 학습했으므로 large-model scaling law의
한 점으로 사용하지 않는다.

## 2. 분리해서 보고할 세 estimand

### 2.1 고정 schedule의 순수 규모 효과

scale $s$와 고정 whitespace policy $r=72$에 대해

$$
A_{\mathrm{fixed}}(s)=1-\frac{T(W72,s)}{T(C86,s)}
$$

를 정의한다. 단, W72가 같은 scale의 C86에 대해 사전 고정한 quality noninferiority를
통과할 때만 actual timing estimand가 존재한다. 품질을 실패한 scale을 0%로 바꾸거나 W80로
대체하지 않는다. 이 estimand만이 `동일 policy에서 scale이 효과를 증폭했는가`에 답한다.

### 2.2 품질 frontier의 배포 가능 효과

각 scale에서 사전 봉인한 grid $R=\{72,76,80,84\}$ 중 calibration-only 규칙으로 가장
공격적인 quality-feasible whitespace rate $r^*(s)$를 하나 고정하고

$$
A_{\mathrm{frontier}}(s)=1-\frac{T(W_{r^*(s)},s)}{T(C86,s)}
$$

를 보고한다. 이는 `그 scale에서 품질을 지키며 얻을 수 있는 실제 효율`에 답하지만,
순수 scale effect가 아니다. 선택 뒤 다른 rate로 fallback하지 않고, 아무 rate도
통과하지 않으면 그 scale의 frontier는 실패로 기록한다.

### 2.3 같은 rate의 boundary-placement 효과

선택된 $r^*(s)$에 대해 W와 C를 같은 patch count로 맞춘

$$
Q_{\mathrm{placement}}(s)=\mathrm{BPB}(W_{r^*},s)-
\mathrm{BPB}(C_{r^*},s)
$$

를 별도로 측정한다. 이는 whitespace 위치가 codepoint 위치보다 유용한지를 묻는 품질
mechanism contrast다. C86 대비 latency와 같은-rate W--C 품질을 한 숫자로 합치지 않는다.

## 3. 실험 축

후속 campaign은 다음 세 축을 동시에 고정한다.

| 축 | 역할 | 원칙 |
|---|---|---|
| model scale | global-event cost 변화 | Mac-feasible 약 50M과 100M을 먼저 사용하고, 200M 이상은 gate 뒤에만 연다 |
| training adequacy | 저학습 confound 제거 | 동일한 raw bytes/parameter와 learning-curve checkpoint를 scale 사이에 맞춘다 |
| patch density | quality--latency frontier | W72/76/80/84를 결과 전에 봉인하고 calibration-only로 한 rate를 고정한다 |

Training adequacy의 후보 budget은 resource 결과를 보기 전에
`1, 2, 4, 8 raw bytes/parameter`로 고정한다. 품질을 보고 budget을 고르지 않는다.
Model-free memory/time preflight가 모든 required role과 seed에 공통으로 실행 가능한 가장 큰
budget을 정하고, 모든 scale에 같은 비율을 적용한다. 이를 compute-optimal training이라고
부르지 않고 **matched learning-curve study**라고 부른다.

## 4. 단계별 실행안

### Stage 0 — 현재 논문 동결

현재 19.6M five-seed 결과, 188.6M one-seed replication, random systems curve와 실패
branch를 먼저 ARR 원고로 제출한다. 후속 결과를 얻기 위해 현 원고의 hypothesis, threshold,
또는 해석을 바꾸지 않는다.

### Stage 1 — 새 데이터와 resource seal

1. 현재 final/test/calibration에 사용한 모든 문서를 exact UTF-8, NFKC/case/whitespace,
   그리고 사전 고정한 long-shingle near-duplicate audit로 제외한다.
2. 새 calibration과 final stream을 model output 없이 먼저 봉인한다.
3. 50M/100M의 모든 required C86, W-grid와 confirmation role에 대해 memory와 projected
   training time을 실제 step으로 잰다.
4. 공통으로 통과하는 가장 큰 bytes/parameter budget을 result-blind rule로 고정한다.

새 untouched source를 만들 수 없으면 confirmatory scale claim을 시작하지 않고, 기존
stream을 쓴 결과는 exploratory replication으로만 분류한다.

### Stage 2 — training adequacy와 calibration selection

- 각 scale에서 세 initial model seed를 calibration selection에 사용하고, lock 뒤 두
  confirmation seed를 추가해 final quality와 actual inference는 정확히 다섯 seed로 닫는다.
- C86과 모든 W-grid model은 seed 안에서 initialization, document order, optimizer와
  predicted-byte denominator를 공유한다.
- Learning-curve checkpoint를 사전 budget마다 저장하되, final stream은 보지 않는다.
- W72 fixed-policy 판정과 $r^*(s)$ frontier 선택을 서로 다른 typed decision으로 남긴다.
- $r^*(s)$가 고정된 뒤에만 same-rate $C_{r^*}$ confirmation role을 실행한다.

Scale별 budget을 품질 결과에 맞춰 다르게 늘리거나, W82/W86처럼 grid 밖 후보를 추가하거나,
실패 seed를 교체하지 않는다.

### Stage 3 — 새 final quality와 actual inference

새 final stream을 한 번 열어 physical checkpoint별 per-sequence NLL을 독립 replay한다.
Quality를 통과한 exact bundle만 같은 Apple-MPS process block에서 timing한다.

- controlled replay와 strict-valid free running을 co-primary로 유지
- 정확히 다섯 fresh sessions, 다섯 model seeds, 64 distinct-document prompts
- candidate/reference뿐 아니라 두 scale도 같은 session block 안에서 counterbalance
- selector, local/global path, cache, argmax, DFA와 synchronization을 timer 안에 포함
- patch count, local/global calls와 component time은 mechanism evidence로 함께 기록

효과 존재와 증폭을 구분한다.

1. **actual benefit:** 두 mode 모두 reduction point가 양수이고 crossed 95% lower bound가
   0보다 크며 5/5 session과 최소 4/5 model seed가 양수다.
2. **scale amplification:** 큰 scale와 작은 scale의 reduction 차이에 대한 paired/hierarchical
   95% lower bound가 0보다 크다. Session과 prompt는 scale 사이에 같은 index로 resample하고,
   model seed는 scale 안에서 resample해 서로 다른 폭의 weight를 직접 paired weight로
   가장하지 않는다.
3. **materiality:** 5%와 10% threshold는 별도 secondary label로 보고한다.

현재 논문의 10% gate를 소급 완화하지 않는다. 새 연구에서는 `효과가 존재하는가`,
`규모와 함께 커졌는가`, `실용적으로 큰가`를 서로 다른 판정으로 사전 등록한다.

### Stage 4 — 외부 replication

100M까지 amplification이 확인된 경우에만 188--200M의 충분한 학습량, CUDA kernel/profile,
chat-template workload와 다른 언어 control을 연다. 확인되지 않으면 Mac 규모 campaign을
음성 결과로 닫고 더 큰 model을 추가로 탐색하지 않는다.

## 5. 병목 가설을 직접 검증하는 방법

각 scale의 latency를 다음 회계로만 해석한다.

$$
T = T_{\mathrm{byte\ local}} + N_{\mathrm{global}}C_{\mathrm{global}}(s)
  + T_{\mathrm{selector}} + T_{\mathrm{head/sync/other}}.
$$

Random-weight curve는 주로 $C_{\mathrm{global}}(s)$와 systems headroom을 알려 준다.
Trained frontier는 quality가 허용한 $\Delta N_{\mathrm{global}}$을 알려 준다. 실제 gain은
두 값의 곱만으로 정하지 않고, 공통 byte-local/head/sync 비용을 포함한 measured E2E로
검증한다.

사전 component model의 예측치와 actual reduction을 비교해 다음을 구분한다.

- event cost는 커졌지만 quality 때문에 event를 거의 제거하지 못한 경우
- event는 충분히 제거했지만 byte-local path가 Amdahl ceiling이 된 경우
- selector/synchronization overhead가 절감분을 상쇄한 경우
- 학습량이 늘면서 W72 quality gap이 실제로 줄어드는 경우

이 분석은 사후 latency를 잘 맞추는 회귀식이 아니라, timing 전 계수 정의와 오차 허용치를
고정한 예측 검증으로 수행한다.

## 6. 중단 기준

- 공통 training budget이 memory/time cap을 넘으면 scale을 낮춰 결과를 찾지 않고
  `external compute required`로 종료한다.
- 어느 W-grid도 final quality를 통과하지 못하면 actual timing을 열지 않는다.
- 50M/100M에서 actual benefit이 재현되지 않으면 200M/1B training으로 가지 않는다.
- actual benefit은 있으나 amplification contrast가 실패하면 `small stable effect without
  amplification`으로 종료한다.
- amplification이 통과해도 CUDA와 chat workload replication 전에는 production/general
  hardware claim을 하지 않는다.

## 7. 현재 논문과의 경계

이 프로그램은 현재 결과를 다시 선택하거나 188.6M W80을 성공으로 재분류하지 않는다.
현재 paper의 최종 문장은 계속 다음과 같다.

> 품질을 보존한 whitespace-aware schedule은 두 trained scale에서 약 2.5--2.9%의 실제
> 개선을 재현했지만, 큰 trained screen에서 scale amplification은 관찰되지 않았다.

후속 campaign이 완료되기 전에는 `larger models increase the speedup`, `scaling law`,
`10% trained speedup`을 사용하지 않는다. 후속 model도 실제 matched-quality inference gate와
충분한 학습·사용성 평가를 통과하기 전에는 Hugging Face의 positive efficient model로
공개하지 않는다.
