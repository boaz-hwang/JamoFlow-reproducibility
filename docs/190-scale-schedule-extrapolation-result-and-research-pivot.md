# Post-100M schedule extrapolation result and research pivot

> 작성일: 2026-08-16
>
> 상태: **1.618B random-weight systems headroom pass; trained-quality bridge required**
>
> Protocol: [Post-100M extrapolation](./189-scale-schedule-extrapolation-protocol.md)
>
> Canonical summary: `results/scale-schedule-extrapolation-v1/summary.json`

## 1. 결론

사용자가 제기한 “model이 커지면 W72의 실제 개선율도 올라갈 수 있다”는 가설은 tested
balanced random-weight BLT family에서 지지됐다. 1,617,558,528-parameter graph에서 W72는
C86보다 controlled cached incremental E2E가 **10.217%** 빨랐다. Crossed session×prompt
bootstrap 95% interval은 **[9.104%, 10.987%]**였고, 16/16 prompts와 3/3 fresh sessions가
W72를 지지했다. 세 session 모두 개별 감소율이 10%를 넘었다.

고정 primary gate의 모든 절을 통과했다.

- point `>=10%`: pass
- lower bound `>=8%`: pass
- positive prompts `>=15/16`: pass (`16/16`)
- positive sessions `3/3`: pass
- sessions `>=10%`: pass (`3/3`)
- all-target correctness/identity/environment/memory: pass

이 결과는 이전 문서의 “unchanged W72 scale-up 경로 종료”를 **100M 이하에서의 당시
결정**으로 제한한다. 더 큰 graph에서 효과가 증가할 가능성은 실제로 남아 있었고, 이번
실험에서 1.6B endpoint가 고정 10% systems threshold를 넘었다.

## 2. 전체 curve

| label | exact parameters | C86 median | W72 median | E2E reduction | crossed 95% interval | positive prompts |
|---:|---:|---:|---:|---:|---:|---:|
| 200M | 188,639,808 | 504.085 ms | 467.703 ms | 7.218% | [3.868%, 8.934%] | 15/16 |
| 400M | 378,058,176 | 612.780 ms | 569.520 ms | 7.060% | [6.788%, 7.500%] | 16/16 |
| 800M | 790,449,408 | 830.625 ms | 758.241 ms | 8.714% | [8.284%, 8.948%] | 16/16 |
| 1600M | 1,617,558,528 | 1,355.525 ms | 1,217.025 ms | 10.217% | [9.104%, 10.987%] | 16/16 |

크기별 point는 200M에서 400M으로 0.158 percentage point 내려갔으므로 엄격 단조는 아니다.
따라서 네 점으로 scaling law를 fitting하거나 10% crossing size를 역산하지 않는다. 하지만
800M과 1.6B에서 효과가 다시 증가했고, primary endpoint의 모든 session이 10%를 넘었다는
사실은 단순한 한 prompt 또는 한 session의 이상치로 설명되지 않는다.

1.6B session별 감소율은 다음과 같다.

- session-0: 10.757%
- session-1: 10.445%
- session-2: 10.750%

## 3. mechanism interpretation

고정 cases에서 W72는 C86보다 patch event를 모든 크기에서 16.279% 줄였다. E2E 감소를 이
고정 event 감소로 나눈 descriptive Amdahl ratio는 다음처럼 변했다.

| target | affected-time-share proxy |
|---:|---:|
| 200M | 44.34% |
| 400M | 43.37% |
| 800M | 53.53% |
| 1600M | 62.76% |

이는 measured component share가 아니라 `E2E reduction / patch-event reduction`이다. 그럼에도
larger graph에서 saved global events가 전체 runtime에서 차지하는 비중이 커진다는 설명과
일치한다. 1.6B에서는 local byte-sequential path가 여전히 크지만, 더 이상 10% 전체 개선을
막을 만큼 압도적이지 않았다.

## 4. 증거 무결성

- exact plan SHA-256:
  `4d74456b8f01666b661ecfa76deb1aa86f47dfd8a5116b71e4000ea10e6c5bb5`
- exact summary SHA-256:
  `9ef1ff5b126727d42ed40962f4e410e7504ff5b04d4e9d226353bf39ddb60e70`
- implementation commit: `432db41207421f09487f4a974c5fd50e9e36b05f`
- plan commit: `91ebfda`
- summary commit: `463bb61`
- workers: 4 targets × 3 fresh subprocess sessions = 12/12 complete
- all 12 reports: correctness, model state, boundary oracle, cache, start/end environment pass
- maximum synchronized driver/recommended memory fraction:
  - 200M 6.39%
  - 400M 7.87%
  - 800M 12.29%
  - 1600M 21.05%
- read-only verifier: raw timing/statistic reconstruction plus four deterministic full-checkpoint
  correctness replays pass

## 5. 정확한 claim boundary

이 결과가 증명하는 것:

> On this Apple-MPS balanced random-weight BLT family, W72's controlled schedule
> headroom grows enough to pass the fixed 10% systems threshold at 1.618B
> parameters.

아직 증명하지 않는 것:

- 1.6B trained W72가 C86 quality를 보존한다.
- trained free-running generation도 10% 빠르다.
- 200M--1.6B가 엄격한 scaling law를 따른다.
- CUDA 또는 production serving에서도 같은 crossing이 발생한다.
- 1.6B training이 현재 Mac에서 시간·memory상 실현 가능하다.

따라서 논문의 systems conclusion은 강화되지만, 사용자 성공 기준인 “실제 학습 model의
quality-matched large-scale inference 개선”은 아직 완결되지 않았다.

## 6. 수정된 다음 연구 방향

다음 단계는 더 큰 random graph를 추가하는 것이 아니다. 먼저 200M/400M/800M/1600M의 실제
training step memory와 throughput을 같은 Mac에서 측정해, quality bridge를 만들 수 있는
최대 feasible target과 data budget을 정한다.

1. batch-1 forward/backward/optimizer-step을 fresh subprocess에서 측정
2. gradient accumulation을 포함한 실제 bytes/step과 optimizer-state memory 측정
3. 64M 및 256M source-byte budget의 wall-time projection
4. 75% recommended MPS memory와 사전 wall-time 한계 적용
5. 결과를 보기 전에 feasible target selection rule 봉인

1.6B가 feasible이면 그 target의 one-seed W72/C86 matched-quality training을 별도 protocol로
진행한다. Infeasible이면 800M으로 조용히 fallback하지 않는다. 800M은 systems point가 10%를
통과하지 않았기 때문이다. 대신 결과를 공개하고, 작은 trainable model에서 global share를
늘리는 새 geometry를 시험하려면 별도 result-aware architecture protocol을 먼저 작성해야 한다.

## 7. 논문 방향 수정

현재 draft의 “100M에서도 실패했으므로 unchanged scale-up을 종료한다”는 문장은 더 이상 최종
결론이 아니다. 다음처럼 교체해야 한다.

> The effect was small below 100M, rose to 8.7% at 790M, and crossed the fixed
> 10% controlled-runtime threshold at 1.618B random-weight scale. This establishes
> scale-dependent systems headroom, while trained quality retention remains open.

논문 게시 전에는 resource feasibility와, 가능하면 최소 한 trained larger checkpoint의 quality와
actual timing을 확인한다. 불가능할 경우에는 random-weight 1.6B result를 systems sensitivity로
정직하게 제한하고 trained 19.6M의 2.5%를 유일한 quality-matched actual result로 유지한다.
