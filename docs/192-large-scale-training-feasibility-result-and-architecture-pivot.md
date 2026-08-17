# 1.6B training resource result and architecture pivot

> 작성일: 2026-08-16
>
> 상태: **1.6B optimizer-step resource pass; meaningful-data bridge deferred**
>
> Protocol: [Large-scale training feasibility](./191-large-scale-training-feasibility-protocol.md)
>
> Canonical summary: `results/large-scale-training-feasibility-v1/summary.json`

## 1. 결론

1,617,558,528-parameter balanced BLT는 이 Mac에서 float32 AdamW 학습 step을 실제로 완료했다.
Standard C86/W72 pair의 synchronized driver allocation은 각각 MPS recommended maximum의
70.73%/70.64%였고, 고정 75% cap을 통과했다. 64M source-byte pilot의 pair projection은
77.32 hours로, 고정 240-hour pair budget도 통과했다.

따라서 1.6B random graph의 10.217% inference headroom이 단지 “학습 state를 절대 담을 수
없는 graph”에서 나온 결과는 아니다. Model, gradients, float32 AdamW states, forward/backward,
gradient clip과 optimizer step이 실제 device에서 함께 실행됐다.

## 2. 측정 결과

모든 값은 effective batch 4 sequences, 2,048 source bytes/update의 두 measured update median이다.

| target | role | median update | 64M projection | 256M projection | max driver / recommended |
|---:|---|---:|---:|---:|---:|
| 200M | C86 | 0.608 s | 5.28 h | 21.13 h | 11.49% |
| 200M | W72 | 0.592 s | 5.14 h | 20.57 h | 11.32% |
| 400M | C86 | 1.125 s | 9.76 h | 39.05 h | 19.73% |
| 400M | W72 | 1.108 s | 9.62 h | 38.48 h | 19.65% |
| 800M | C86 | 2.248 s | 19.51 h | 78.05 h | 36.92% |
| 800M | W72 | 2.205 s | 19.14 h | 76.55 h | 36.71% |
| 1600M | C86 | 4.508 s | 39.13 h | 156.52 h | 70.73% |
| 1600M | W72 | 4.399 s | 38.19 h | 152.75 h | 70.64% |
| 1600M checkpointed | C86 | 4.743 s | 41.17 h | 164.68 h | 67.31% |
| 1600M checkpointed | W72 | 4.679 s | 40.62 h | 162.47 h | 67.26% |

Standard 1.6B가 이미 통과했으므로 fixed rule에 따라 standard가 선택됐다. Gradient
checkpointing은 memory fraction을 약 3.5 percentage points 낮췄지만 pair 시간을 5.8% 정도
늘렸다. 현재 workload에서는 필요하지 않다.

## 3. 새로 드러난 제한

Resource gate 통과와 publication-worthy trained model은 같은 말이 아니다. 64M bytes를 1.6B
parameters에 쓰면 약 `0.04 source bytes/parameter`뿐이다. 기존 19.6M trained experiment의
128M bytes는 약 `6.53 bytes/parameter`였다. 1.6B에 같은 ratio를 맞추려면 약 10.56B source
bytes가 필요하고, 현재 linear projection으로 pair wall time은 수만 시간 규모다.

즉 64M 1.6B pair는 다음에는 답할 수 있다.

- 몇 optimizer update 뒤에도 random-weight systems headroom이 유지되는가
- C86/W72의 초기 learning curve가 즉시 갈라지는가

하지만 다음 강한 주장에는 부족하다.

- 충분히 학습된 1.6B Korean byte LM에서 quality가 보존된다
- 19.6M 결과와 동등한 data-per-parameter 조건의 scaling evidence다

77시간짜리 severely-undertrained pair를 최종 quality bridge로 실행하는 것은 wall time은 들지만
논문 증거 가치는 제한적이다. 결과를 확인한 뒤 이 점 때문에 연구 계획을 수정한다.

## 4. 수정된 연구 방향

1.6B에서 10%를 만든 직접 원인은 parameter count 자체보다 saved global patch event의 상대
비용 증가로 보인다. 따라서 다음 단계는 total parameters만 키우는 대신, 현재 Mac에서 128M
bytes 이상 학습 가능한 크기 안에서 global compute share를 의도적으로 높인 BLT family를
사전에 고정해 탐색하는 것이다.

요구사항은 다음과 같다.

1. 100M--200M 안팎의 trainable budget을 우선 사용한다.
2. local byte path의 width/layers를 줄이고 global width/layers에 compute를 재배분한다.
3. W72와 C86는 exact 동일 weight/model object를 공유한다.
4. 첫 timing 전에 geometry 후보와 단일 선택 규칙을 봉인한다.
5. Random-weight 10% systems gate를 통과한 단 하나의 고정 architecture만 trained-quality
   protocol 후보가 된다.
6. Quality가 나빠지면 다른 geometry로 사후 switch하지 않는다.

이 방향은 “큰 model이면 무조건 빨라진다”가 아니라 다음 더 구체적인 가설을 검증한다.

> W72의 E2E 이득은 saved global patch events가 차지하는 compute share의 함수이며, global-heavy
> geometry는 total parameter 수를 1.6B까지 늘리지 않고도 10% headroom을 재현할 수 있다.

## 5. Claim boundary

현재 추가로 허용되는 주장:

> The 1.618B graph that crossed 10% inference headroom also completed real
> float32 AdamW updates within the fixed Apple-MPS resource budget.

아직 허용되지 않는 주장:

- 1.6B trained quality가 matched/noninferior다.
- 64M-byte 1.6B pilot가 충분히 학습된 language model이다.
- Gradient checkpointing이 quality 또는 inference speed를 개선한다.
- Global-heavy architecture가 아직 10%를 통과했다.

Evidence identities:

- resource plan SHA-256:
  `2b49d798410591ac4d5ca436557a3cbebc49951554a6904c09a40996a767ee7f`
- resource summary SHA-256:
  `4a9667226924e5b7179f8421a8cf2ec9d93369d8a2e2cb478b3d9fd7e543a1b7`
- read-only reconstruction: pass
