# Global-heavy fixed-geometry result and trained-scale pivot

> 작성일: 2026-08-16
>
> 상태: **fixed global-heavy geometry failed; balanced 200M trained screen next**
>
> Protocol: [Global-heavy bridge](./193-global-heavy-schedule-bridge-protocol.md)
>
> Canonical summary: `results/global-heavy-schedule-bridge-v2/summary.json`

## 1. 결론

46,644,640-parameter global-heavy model은 고정 10% gate를 실패했다. W72의 C86 대비 controlled
E2E 감소는 **3.923%**, crossed session×prompt bootstrap 95% interval은
**[3.247%, 4.310%]**였다. 16/16 prompts와 3/3 sessions 방향은 모두 W72를 지지했지만,
어느 session도 10%에 도달하지 않았다.

| session | E2E reduction |
|---|---:|
| session-0 | 3.841% |
| session-1 | 3.808% |
| session-2 | 4.027% |

Correctness, state, environment와 memory evidence는 모두 통과했다. 따라서 실패는 증거 오류가
아니라 effect size 부족이다.

## 2. 가설에서 틀린 부분

새 architecture는 total parameters의 91.786%를 global transformer에 두었다. 그럼에도 기존
balanced 49.823M model의 3.572%보다 개선된 값은 **0.351 percentage point**뿐이었다.

- balanced 49.823M: 3.572%
- global-heavy 46.645M: 3.923%
- balanced 1.618B: 10.217%

따라서 `global parameter share`는 `global patch events가 실제 E2E에서 차지하는 time share`의
대리변수로 충분하지 않다. 새 model의 global transformer는 parameter 비중은 높지만 absolute
global width/layer compute가 1.6B보다 훨씬 작다. Local encode/decode와 kernel/dispatch 고정비도
계속 지배한다. Amdahl proxy도 24.10%로, 1.6B의 62.76%와 크게 달랐다.

이 결과 때문에 더 깊고 좁은 geometry를 사후에 추가해 baseline을 일부러 느리게 만드는 방식은
채택하지 않는다. 그런 구조는 10% threshold를 맞출 수 있어도 연구 질문을 architecture gaming으로
바꿀 위험이 크다.

## 3. 다음으로 가치 있는 실험

이제 남은 핵심 질문은 **실제 trained balanced model에서 크기가 커질 때 2.5%가 증가하는가**다.
Random-weight evidence는 다음과 같다.

- trained 19.6M: 약 2.5%
- random 49.8M: 3.572%
- random 98.4M: 4.460%
- random 188.6M: 7.218%
- random 790.4M: 8.714%
- random 1.618B: 10.217%

현재 canonical Korean train stream은 정확히 128M bytes다. 188,639,808-parameter balanced
200M model의 measured 64M pair projection은 10.423 hours였으므로, 128M C86/W72 pair는 약
20.85 hours다. 이는 1.6B 64M pair의 77.32 hours보다 짧고, data/parameter 비율도 약
`0.68 bytes/parameter`로 개선된다. 충분히 학습된 frontier model은 아니지만 10배 larger trained
mechanism replication으로는 1.6B 64M보다 해석 가치가 높다.

따라서 다음 protocol은 다음처럼 고정한다.

1. Balanced 188.6M exact geometry
2. 기존 canonical 128M Korean train stream one pass
3. 동일 initialization seed의 C86/W72 두 model
4. Calibration BPB noninferiority와 learning direction을 먼저 평가
5. Quality screen을 통과한 뒤에만 exact trained checkpoint actual inference를 실행
6. 성공하면 새 disjoint Korean train data를 model outcome과 무관하게 구성해 512M/1.024B
   continuation을 별도 protocol로 검토

## 4. 정확한 claim boundary

이번 실패가 허용하는 문장:

> Reallocating 91.8% of a 46.6M model's parameters to the global transformer
> raised W72 headroom only from 3.57% to 3.92%; absolute scale, not parameter
> share alone, remained necessary in this implementation.

아직 허용되지 않는 문장:

- 200M trained model의 실제 개선이 7%다.
- 128M bytes가 200M model을 충분히 학습한다.
- global-heavy 구조가 quality상 열등하다.
- 더 깊고 좁은 global network가 실패한다.

Evidence identities:

- v2 plan SHA-256:
  `7904105f4942b430c68ef981062387f7343f4715a7d15e311bcde2fbc7bd64c0`
- summary SHA-256:
  `968c4a16933af7b680fdc655eef07fa46f8c3ed925916bb61656ec8534b0971a`
- full independent correctness/statistic reconstruction: pass
- v1 plan: entrypoint import failure before model build/timing; no v1 result artifact
