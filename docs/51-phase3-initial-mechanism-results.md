# Phase 3 initial mechanism result: Gate M

> **Historical analysis warning (2026-08-11):** 이 문서는 당시 seed×window bootstrap으로 내린 initial attribution 기록이다. [문서 군집 교정](./52-document-cluster-inference-integrity-addendum.md) 뒤 새 경로에서 Gate M을 다시 판정하며, 아래 결과만으로 mechanism confirmation 또는 inference efficiency를 주장하지 않는다.

> 작성일: 2026-08-11  
> 상태: initial 3-seed D/P mechanism controls 완료  
> 사전등록: [Phase 3 mechanism addendum](./29-phase3-mechanism-addendum.md)  
> 선행 authorization: [Phase 3 Gate I result](./50-phase3-initial-results.md)  
> 기계 판정: [`results/phase3-mechanism/summary.json`](../results/phase3-mechanism/summary.json)

## 결론

Initial 3-seed Gate M은 통과했다. Exact 86-patch, 동일 19,596,096-parameter graph와 동일 seed initialization/training order에서 `causal_whitespace_grid`(W)는 whitespace를 보지 않는 delayed-grid control(D)과 calibration에서 event frequency를 맞춘 causal rolling-hash placebo(P)보다 세 seed 모두 낮은 HPLT3 test BPB를 보였다.

허용되는 결론은 다음으로 제한한다.

> 이 graph, data, scale에서 관측된 whitespace에 연관된 W의 boundary relocation 신호는 사전 지정한 delayed-phase 및 rate-matched causal-event 대안 설명을 넘었다.

이는 Korean morphology, morpheme boundary, optimal segmentation, learned router 일반에 대한 우월성, 또는 실제 inference speedup을 뜻하지 않는다.

## 결과

| Policy | Mean test BPB | Seed별 BPB |
|---|---:|---|
| W: causal whitespace grid | **1.636415** | **1.639217 / 1.635167 / 1.634859** |
| D: delayed grid | 1.646722 | 1.648469 / 1.646494 / 1.645204 |
| P: causal hash placebo | 1.657114 | 1.658834 / 1.656460 / 1.656049 |

Paired contrast는 `W − control`이며 negative가 W에 유리하다.

| Contrast | Seed-paired mean BPB | 방향 | Crossed bootstrap 95% interval | Paired-seed t 95% interval | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| W − D | **−0.010308** | 3/3 negative | [−0.011278, −0.009311] | [−0.012887, −0.007728] | 0.001683 |
| W − P | **−0.020700** | 3/3 negative | [−0.021435, −0.019697] | [−0.023034, −0.018366] | 0.000686 |

Initial Gate M은 각 contrast에 mean `<= −0.002 BPB`, 최소 2/3 negative, 무결성 통과를 요구했다. 두 contrast는 모든 조건을 통과했다. Bootstrap과 Holm 결과는 initial gate의 필수조건보다 강한 진단으로 함께 보고하며, final 5-seed Gate M을 대체하지 않는다.

## 무엇을 구분했는가

### Delayed grid D

D는 W에서 early whitespace event가 한 번도 발동하지 않을 때의 `target + 2` codepoint-boundary schedule이다. W가 D보다 나았으므로 W의 initial 효과를 단순히 deadline phase를 늦춘 결과로만 설명하기 어렵다.

### Rolling-hash placebo P

P는 whitespace 대신 prefix-causal FNV-1a event를 사용하고, calibration W의 nonfinal early-event trigger fraction에 가장 가까운 threshold를 결과 전에 고정했다. W가 P보다 나았으므로 “같은 빈도의 임의 causal event를 넣으면 된다”는 설명은 이 setting에서 지지되지 않았다.

## 남은 식별 한계

1. P는 event frequency를 맞췄지만 target displacement와 patch-length distribution 전체를 exact-match하지 않는다.
2. W는 whitespace 자체뿐 아니라 그로 인해 바뀐 local byte grouping과 optimization geometry를 함께 intervention한다.
3. D와 P는 두 개의 사전 지정 alternative일 뿐 가능한 모든 placebo family를 소진하지 않는다.
4. 비한국어 matched experiment가 없으므로 한국어 고유 기전이라고 부를 수 없다.
5. W는 morphological analyzer가 아니며 eojeol 내부 morpheme을 식별하지 않는다.
6. 세 seed 결과는 independent confirmation 두 seed가 추가된 final Gate M보다 약한 증거다.

## 무결성

- W/D/P는 split마다 exact 86 patches를 사용했다.
- seed별 initialization과 shuffled training order가 일치했다.
- 31,250개 test sequence의 loss vector에서 BPB와 paired contrast를 재구성했다.
- checkpoint state/artifact, report, loss, matrix hash를 독립 검증했다.
- D/P matrices와 diagnostics를 current code 및 source stream에서 다시 구성해 일치시켰다.
- Gate I authorization과 primary W evidence lineage를 summarization 시점에 다시 확인했다.

## 다음 단계

Gate M은 attribution gate이며 Gate J/K를 열지 않는다. 사전등록대로 다음을 진행한다.

1. Initial S/E/EC를 학습해 authentic spacelike cadence와 learned entropy router를 포함한 quality-cost 위치를 측정한다.
2. F/C/W independent confirmation seeds와 public OOD를 완료해 Gate J를 판정한다.
3. Gate J 통과 시 D/P confirmation seeds를 실행해 final five-seed Gate M을 계산한다.
4. Gate J/K와 reduced-rate/actual-inference gates가 모두 통과하기 전에는 효율 기여를 주장하지 않는다.
