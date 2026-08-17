# Parameter-matched global-heavy W72 systems bridge protocol

> 작성일: 2026-08-16
>
> 상태: **첫 global-heavy timing 전에 고정할 single-architecture protocol**
>
> 선행 판단: [1.6B resource result and pivot](./192-large-scale-training-feasibility-result-and-architecture-pivot.md)

> 실행 전 정정: 최초 v1 plan은 entrypoint import 오류로 model build나 timing 전에 종료됐다.
> 이미 commit된 v1 plan을 소급 수정하지 않고, 과학적 geometry/cases/gate는 그대로 둔 채
> `jamoflow-global-heavy-schedule-bridge-v2` namespace에서 import wiring만 교정해 다시 봉인한다.

## 1. 가설

Balanced 49,823,488-parameter BLT에서 W72의 C86 대비 controlled E2E 개선은 3.572%였다.
Balanced family를 1,617,558,528 parameters까지 키우면 같은 patch-event 감소에서 개선이
10.217%로 커졌다. Amdahl-style proxy는 saved global patch events가 전체 시간에서 차지하는
비중이 model scale과 함께 커졌다는 설명과 일치한다.

이번 실험은 parameter count를 키우지 않고 이 설명을 더 직접 검증한다.

> 약 50M total parameters 안에서 local byte path를 작게, global transformer를 크게 배분하면
> W72가 줄이는 global patch event가 전체 E2E에서 차지하는 비중이 커져 10% headroom을 재현할
> 수 있다.

## 2. 단 하나의 고정 geometry

Timing 전에 analytic parameter allocation만으로 다음 architecture를 선택한다. 여러 geometry를
timing한 뒤 가장 좋은 것을 고르지 않는다.

| field | value |
|---|---:|
| exact total parameters | 46,644,640 |
| local / global width | 160 / 640 |
| local / global heads | 5 / 10 |
| local encoder / decoder layers | 1 / 1 |
| global layers | 8 |
| local / global FFN | 480 / 1,920 |
| global-transformer parameters | 42,813,440 |
| global parameter share | 91.786% |

Balanced 50M control은 49,823,488 parameters이므로 새 model은 오히려 6.38% 작다. 따라서
10%를 통과하면 “parameter가 더 많아서”가 아니라 parameter placement/global compute share가
유력한 설명이 된다. Parameter share 자체는 runtime component share가 아니므로 measured timing을
대체하지 않는다.

## 3. 고정 workload

- model seed: `20260816`
- dtype/device: float32 / Apple MPS
- global position capacity: 1,032; patching horizon: 512
- reference/candidate: C86 causal codepoint grid / W72 causal whitespace grid
- target×session에서 exact 동일 random-weight model object 공유
- 기존 scale experiment와 동일한 4 warmup + 16 measured document-independent cases
- prompt 128 bytes + controlled continuation 127 consume calls
- inner repetitions 3; cell median으로 접고 독립 표본으로 세지 않음
- fresh subprocess sessions 3
- timer 안: fresh runtime, parallel prefill, 127 cached consumes, final MPS synchronize
- correctness: 각 schedule/session에서 512 sequential-vs-parallel logit/argmax 및 모든 prefix
  boundary oracle exact
- AC, thermal, implementation, environment, state SHA, parameter count, memory 75% gate

## 4. Primary gate

Session×prompt cell을 crossed bootstrap 10,000회(`seed=20260831`) 재표집한다. 다음을 모두
요구한다.

1. correctness/identity/environment/memory evidence pass
2. median E2E reduction `>=10%`
3. crossed 95% lower bound `>=8%`
4. positive prompts `>=15/16`
5. 3/3 sessions positive
6. sessions 중 `>=2/3`이 각각 `>=10%`

실패하면 geometry를 미세 조정하거나 더 큰 후보로 fallback하지 않는다. 새 architecture를
시도하려면 별도 result-aware protocol이 필요하다. 통과해도 허용되는 것은 이 exact geometry의
training-resource measurement와 trained-quality protocol 작성뿐이다.

## 5. 기존 결과와의 비교

동일한 C86/W72 schedule과 유사한 50M parameter budget에서 다음 두 점을 descriptive하게
비교한다.

- balanced 49.823M: 3.572%, global width ratio 1:2, global parameter share lower
- global-heavy 46.645M: 이번 실험의 prospective result

이 비교는 동일 model state나 exact geometry를 공유하지 않으므로 paired statistical test가 아니다.
하지만 total parameter 규모를 거의 고정한 architecture-level mechanism diagnostic이다.

## 6. Claim boundary

통과 시:

> At approximately matched 50M parameter count, reallocating capacity toward the
> global patch transformer reproduced the 10% W72 controlled-runtime headroom
> seen only at 1.6B in the balanced family.

아직 주장하지 않는 것:

- trained quality가 보존된다.
- global parameter share가 인과적 runtime component share와 동일하다.
- free-running generation, CUDA, serving에서 같은 값이 나온다.
- whitespace가 Korean morphology를 직접 모델링해서 빨라졌다.
