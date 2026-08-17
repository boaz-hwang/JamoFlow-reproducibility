# Large-scale trained-quality bridge resource protocol

> 작성일: 2026-08-16
>
> 상태: **첫 200M--1.6B optimizer step 전에 고정할 outcome-aware resource protocol**
>
> 선행 결과: [1.6B schedule headroom crossing](./190-scale-schedule-extrapolation-result-and-research-pivot.md)

## 1. 왜 연구 방향을 바꾸는가

19.6M trained model에서는 W72의 실제 추론 개선이 약 2.5%였지만, 동일 weight를 공유한
random-weight graph에서는 model이 커질수록 saved global patch event의 상대 비용이 커졌다.
1,617,558,528 parameters에서 controlled E2E 개선은 10.217%였고 사전 고정한 systems gate를
통과했다. 따라서 “100M에서 작았으므로 더 큰 model도 가치가 없다”는 결론은 폐기한다.

다음 과학적 질문은 latency가 아니라 다음 두 조건을 동시에 만족하는지다.

1. 이 규모의 C86과 W72를 현재 48GB Apple-silicon Mac에서 실제로 학습할 수 있는가?
2. 학습 후 quality를 맞춘 상태에서도 1.6B random-weight systems headroom이 유지되는가?

이번 단계는 첫 질문만 답한다. 실제 학습 quality나 최종 latency를 미리 주장하지 않는다.

## 2. 고정 model과 비교 역할

선행 scale plan의 exact model spec, parameter count, CPU initialization state SHA-256를 그대로
상속한다. 결과를 본 뒤 width/layer를 줄이지 않는다.

| label | exact parameters | standard resource workers |
|---:|---:|---:|
| 200M | 188,639,808 | C86, W72 |
| 400M | 378,058,176 | C86, W72 |
| 800M | 790,449,408 | C86, W72 |
| 1600M | 1,617,558,528 | C86, W72 |

1.6B에는 standard 외에 activation gradient checkpointing C86/W72 worker를 하나씩 더 둔다.
Checkpointing은 memory rescue일 뿐 model geometry, dtype, optimizer 또는 quality target을 바꾸지
않는다. Standard가 통과하면 standard를 우선한다.

## 3. 실제 optimizer-step workload

- device/dtype: Apple MPS / float32
- source: canonical HPLT Korean train stream의 첫 32개 완전한 512-byte sequence
- microbatch: 1 sequence
- gradient accumulation: 4
- effective batch: 4 sequences = 2,048 source bytes/update
- optimizer: AdamW, LR `1.5e-4`, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `0.1`
- gradient clipping: `1.0`
- warmup update: 1 (optimizer states를 실제 할당)
- measured updates: 2
- 각 update timer: 네 input/patch MPS transfers, forward, scaled loss backward, gradient clip,
  AdamW step, final MPS synchronize

C86 matrix는 exact 86-patch causal codepoint grid이고 W72 matrix는 exact 72-patch causal
whitespace grid다. 두 역할 모두 동일한 model graph와 parameter count를 사용한다.

## 4. Memory와 wall-time gate

각 worker는 fresh subprocess에서 `torch.mps.set_per_process_memory_fraction(0.75)`를 먼저
적용한다. 따라서 successful optimizer update는 MPS 권장 maximum의 75% cap 아래서 model,
gradients, AdamW states와 activation workload가 실제 실행됐다는 뜻이다. 동기화 stage snapshot도
함께 남기지만 resettable high-water API가 아니므로 snapshot만을 peak라고 과장하지 않는다.

Measured update median으로 다음을 고정 계산한다.

- 64M source bytes: `ceil(64,000,000 / 2,048)` updates
- 256M source bytes: `ceil(256,000,000 / 2,048)` updates

64M pilot resource pass는 각 model이 `<=120 h`, C86+W72 pair가 `<=240 h`, 두 역할이 모두
finite complete이고 75% cap을 통과해야 한다. 256M은 descriptive projection이며 이번 gate가
아니다. 이 한계는 research compute budget이지 보편적인 학습 가능성 경계가 아니다.

## 5. 고정 선택 규칙

1. Standard 200→400→800→1600을 모두 측정한다.
2. 1.6B checkpointed pair도 결과와 무관하게 측정한다.
3. 1.6B standard가 통과하면 standard를 선택한다.
4. Standard가 실패하고 checkpointed 1.6B가 통과하면 checkpointed를 선택한다.
5. 둘 다 실패하면 **800M이나 더 작은 target으로 systems success를 대체하지 않는다.**

1.6B만 선행 10% systems endpoint이기 때문이다. 더 작은 model의 resource pass는 후속
architecture 설계를 위한 진단일 뿐이다. 1.6B가 resource-infeasible이면 결과를 공개하고,
trainable parameter budget 안에서 global compute share를 늘리는 별도 global-heavy protocol로
전환한다. 그 protocol은 새 random-weight timing과 trained quality를 다시 검증해야 한다.

어떤 resource pass도 곧바로 학습을 실행하지 않는다. 허용되는 것은 별도의 trained-quality
bridge protocol 작성뿐이다.

## 6. Evidence state machine

1. 이 문서, core, sealer, runner, verifier와 tests를 clean commit한다.
2. 선행 scale plan/summary, model states, train arrays, implementation, environment를 plan에 봉인한다.
3. Plan을 단독 commit한다.
4. Plan HEAD의 고정 worker order로 10개 fresh subprocess를 실행한다.
5. OOM/timeout/exception도 favorable worker 재선택 없이 typed failure receipt로 보존한다.
6. 모든 receipt 뒤 tracked summary를 한 번만 쓰고 별도 commit한다.
7. Read-only verifier가 worker identity, raw timing projection, selection을 재구성한다.

## 7. Claim boundary

통과 시에도 허용되는 문장은 좁다.

> The exact 1.618B C86/W72 pair completed real float32 AdamW updates under the
> fixed Apple-MPS memory and 64M-byte wall-time budget, permitting a separate
> trained-quality pilot protocol.

실패 시:

> The 1.618B random-weight graph showed 10% inference headroom but was not
> resource-feasible for the fixed float32 training pilot on this Mac; a smaller
> result cannot replace that endpoint without a new architecture protocol.

다음을 주장하지 않는다.

- resource projection만으로 trained quality가 유지된다.
- 다른 batch, mixed precision, CUDA 또는 cluster에서도 같은 가능/불가능 경계다.
- checkpointing이 추론 개선을 바꾼다.
- 64M byte pilot이 최종 language-model quality에 충분하다.
