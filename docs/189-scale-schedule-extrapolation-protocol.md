# Post-100M W72/C86 schedule extrapolation protocol

> 작성일: 2026-08-16
>
> 상태: **200M/400M/800M/1600M timing을 보기 전에 고정할 exploratory protocol**
>
> 선행 결과: [50M--100M schedule sensitivity](./187-scale-schedule-preflight-result-and-terminal-research-decision.md)

## 1. 수정된 연구 질문

19.6M trained model의 controlled W72--C86 E2E 감소는 2.628%였다. 이후 동일 weight를
공유한 random graph에서 감소율은 49.8M 3.572%, 76.5M 3.758%, 98.4M 4.460%로
증가했다. 98.4M은 사전에 고정한 10% training gate를 실패했지만, 이 결과만으로 더 큰
model에서도 10%에 도달하지 않는다고 결론내릴 수는 없다.

따라서 이전의 “unchanged scale-up 종료” 결론을 다음처럼 좁힌다.

> W72는 tested 100M 이하에서 10% actual-inference 기준을 충족하지 못했다. 증가 추세가
> 더 큰 balanced BLT graph에서도 이어지는지는 미해결이다.

이번 실험은 그 미해결 질문을 model output이나 새 timing을 보기 전에 고정한 네 크기에서
검증한다. Random weight graph이므로 quality 또는 trained-model scaling evidence는 아니다.

## 2. 고정 model family

모든 model은 기존 publication family와 같은 byte vocabulary, 512-byte policy horizon,
local:global width 1:2, local/global head dimension 32/64, FFN width 3×, local
encoder/decoder 각 2 layers를 유지한다. Target 선택은 실제 latency가 아니라 parameter
규모와 head divisibility로만 했다.

| label | exact parameters | local/global width | heads | global layers |
|---:|---:|---:|---:|---:|
| 200M | 188,639,808 | 448 / 896 | 14 / 14 | 16 |
| 400M | 378,058,176 | 576 / 1,152 | 18 / 18 | 20 |
| 800M | 790,449,408 | 768 / 1,536 | 24 / 24 | 24 |
| 1600M | 1,617,558,528 | 1,024 / 2,048 | 32 / 32 | 28 |

- target order: 200M → 400M → 800M → 1600M
- model seed: `20260816`
- global position capacity: 1,032; patching horizon: 512
- target×session마다 하나의 model object를 C86과 W72가 exact 공유
- model state SHA-256와 parameter count를 timing 전 plan에 봉인
- device/dtype: repository-pinned float32 / Apple MPS

1600M은 inference curve의 상한을 보기 위한 target이다. 이 experiment가 통과해도 1600M
training을 직접 허가하지 않는다.

## 3. schedule, cases, timer

비교 pair는 이전 실험과 동일하다.

| role | causal policy | nominal patch count |
|---|---|---:|
| reference | codepoint grid C86 | 86 |
| candidate | whitespace grid W72 | 72 |

이전 50M--100M 결과 뒤 case나 threshold를 다시 고르지 않는다. 동일한 20개 document-
independent controlled windows를 그대로 사용한다.

- warmup 4, measured 16
- 각 255-byte observed window는 서로 다른 source document 안에 완전히 포함
- prompt 128 bytes + controlled continuation 앞 127 consume bytes
- inner repetitions 3; repetition은 독립 표본으로 세지 않고 cell median으로 접음
- fresh subprocess sessions 3개
- role order: `(target_index + session_index + prompt_index + repetition) mod 2`
- timer 안: fresh runtime, parallel prefill, 127 cached consume, final MPS synchronize
- timer 밖: model build, source load, correctness replay

이 case pool은 compact W72 결과 뒤 만들어졌고 50M--100M timing도 이미 알려져 있으므로,
이번 실험을 untouched confirmatory 또는 public preregistration이라 부르지 않는다. 다만
200M--1600M의 model output과 timing은 plan 뒤에만 연다.

## 4. correctness, environment, memory

각 target×session×schedule에서 첫 4 measured cases의 128 positions를 sequential runtime과
parallel-prefill runtime으로 비교한다.

- 512 logit/argmax comparisons per schedule
- argmax exact, normalized error `<=1` under `atol=2e-5`, `rtol=1e-4`
- online boundary trace와 모든 prefix의 independent offline oracle exact
- cache diagnostics, observed byte count, global patch count exact
- parameter count와 CPU state SHA-256 exact
- AC power, no thermal/performance warning, shared publication MPS lock
- start/end hardware/software environment byte-identical
- synchronized post-trial driver allocation positive and recommended maximum의 75% 이하

한 target이라도 위 조건을 실패하면 performance threshold와 관계없이 전체 evidence는 실패다.

## 5. 통계와 고정 판정

Session×prompt×role의 세 repetition median을 만든 뒤 3 sessions와 16 prompts를 같은 paired
index로 교차 재표집한다. Target별 10,000회 percentile bootstrap seed는
`20260817 + target_millions`다. 세 session뿐이므로 interval은 small-cluster sensitivity이며
일반 hardware CI로 해석하지 않는다.

200M, 400M, 800M은 curve diagnostics이고 **1600M만 primary**다. 다음을 모두 요구한다.

1. 네 target의 correctness, identity, environment, memory evidence pass
2. 1600M median E2E reduction `>=10%`
3. 1600M crossed bootstrap lower bound `>=8%`
4. faster prompts `>=15/16`
5. 3/3 sessions positive
6. session 중 `>=2/3`이 각각 `>=10%`

크기별 point estimate가 엄격히 증가하는지는 descriptive로 보고하지만 gate는 아니다. 1600M이
실패했는데 더 작은 target이 우연히 통과해도 target fallback이나 favorable-size selection을 하지
않는다. 1600M이 통과해도 허가되는 것은 별도 trained-model feasibility 설계뿐이며, training
자체나 quality claim은 허가되지 않는다.

## 6. mechanism interpretation

고정 case에서 W72는 C86보다 patch event를 16.279% 줄인다. Target별로 다음 descriptive ratio를
함께 보고한다.

`affected-time-share proxy = measured E2E reduction / 0.16279`

이는 단순 Amdahl-style 진단이다. Event별 비용이 일정하다고 가정한 measured component share가
아니며 scaling law나 out-of-range extrapolation에 사용하지 않는다. Larger graph에서 ratio가
증가하면 saved global events의 상대 비용이 커진다는 가설과 일치하고, 포화되면 local byte path가
계속 지배한다는 가설과 일치한다.

## 7. 증거 state machine

1. 이 protocol, core, sealer, runner, verifier와 tests를 clean commit한다.
2. 네 CPU model의 exact state와 implementation/environment/case hashes를 no-clobber plan에 쓴다.
3. Plan을 별도 commit한다.
4. Plan HEAD에서 `.active`를 만들고 4 targets × 3 sessions를 고정 순서로 실행한다.
5. 각 worker report/NPZ는 ignored namespace에 no-clobber publish한다.
6. 12 workers가 모두 valid일 때만 tracked summary를 만들고 별도 commit한다.
7. Read-only verifier가 raw arrays/statistics와 네 full checkpoint correctness를 독립 재생한다.

Timing을 본 뒤 model geometry, target, case, repetition, threshold를 바꾸려면 새 protocol이 필요하며
이 결과를 덮어쓰지 않는다.

## 8. Claim boundary

통과 시 허용되는 주장:

> On the tested Apple-MPS random-weight BLT family, the W72 schedule crosses the
> fixed 10% controlled-runtime headroom threshold by 1.6B parameters.

실패 시 허용되는 주장:

> The W72 effect increased through 100M but did not establish 10% headroom at the
> fixed 1.6B endpoint under this balanced scaling family.

어느 경우에도 다음은 주장하지 않는다.

- 200M--1600M trained W72가 C86 quality를 보존한다.
- larger trained model이나 free-running generation이 10% 빠르다.
- 이 네 점으로 scaling law 또는 10% crossing size를 외삽했다.
- CUDA, serving stack 또는 다른 model family에 일반화된다.
- whitespace가 Korean morphology 자체의 causal mechanism이다.
