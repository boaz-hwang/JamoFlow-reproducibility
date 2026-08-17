# Phase 3 initial result: Gate I

> **Historical analysis warning (2026-08-11):** 이 문서는 당시 seed×window bootstrap으로 내린 authorization 기록이다. Packed windows의 source-document dependence를 반영한 [사후 무결성 교정](./52-document-cluster-inference-integrity-addendum.md)이 추가되었으며, 이후 진행 여부와 논문 주장은 새 `phase3-primary-clustered` 판정만 따른다. 아래 수치를 preregistered document-cluster evidence로 인용하지 않는다.

> 작성일: 2026-08-11  
> 상태: initial 3-seed primary 및 public OOD 평가 완료  
> 사전등록: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md)  
> 기계 판정: [`results/phase3-primary/summary.json`](../results/phase3-primary/summary.json), [`results/phase3-ood/summary.json`](../results/phase3-ood/summary.json)

## 결론

Phase 3 Gate I는 통과했다. 19,596,096-parameter HF BLT graph를 HPLT 3.0 Korean 128M train bytes에 one pass로 학습했을 때, exact 86-patch `causal_whitespace_grid`(W)는 같은 rate의 `causal_codepoint_grid`(C)와 `fixed_byte_6`(F)보다 세 seed 모두 낮은 held-out BPB를 보였다. Leipzig Korean Wikipedia OOD에서도 효과 방향이 유지되어 사전등록한 regression guard를 통과했다.

이 결과가 허용하는 해석은 좁다. 이미 관측한 whitespace 근처로 고정 grid boundary를 제한적으로 이동하는 정책이 이 graph와 scale에서 재현 가능한 **same-rate modeling-quality 신호**를 보였다는 뜻이다. Global patch 수가 같으므로 실제 compute 감소나 autoregressive generation speedup을 증명하지 않는다.

## Primary HPLT3 결과

| Policy | Mean test BPB | Seed별 BPB |
|---|---:|---|
| F: fixed byte | 1.650904 | 1.652938 / 1.650537 / 1.649237 |
| C: causal codepoint grid | 1.646297 | 1.648100 / 1.646581 / 1.644210 |
| W: causal whitespace grid | **1.636415** | **1.639217 / 1.635167 / 1.634859** |

Paired contrast는 `left − right`이며 negative가 W에 유리하다.

| Contrast | Seed-paired mean BPB | 3/3 방향 | Crossed bootstrap 95% interval | Paired-seed t 95% interval |
|---|---:|---:|---:|---:|
| W − C | **−0.009882** | 3/3 negative | [−0.011333, −0.008835] | [−0.013228, −0.006537] |
| W − F | **−0.014489** | 3/3 negative | [−0.015321, −0.013711] | [−0.016554, −0.012425] |
| C − F | −0.004607 | 3/3 negative | [−0.005125, −0.004001] | [−0.006026, −0.003188] |

Gate I의 핵심 기준은 mean `W − C <= −0.002`, 최소 2/3 negative, public OOD guard 통과였다. 관측값은 −0.009882, 3/3 negative였고 모든 provenance/integrity 검사가 통과했다.

## Public OOD 결과

Leipzig Korean Wikipedia held-out stream의 mean BPB는 F 1.868574, C 1.865761, W **1.851867**이었다.

| Contrast | Seed-paired mean BPB | 3/3 방향 | Crossed bootstrap 95% interval |
|---|---:|---:|---:|
| W − C | **−0.013894** | 3/3 negative | [−0.015244, −0.012498] |
| W − F | **−0.016707** | 3/3 negative | [−0.018374, −0.014993] |

두 contrast 모두 허용 가능한 최대 regression `+0.020 BPB`보다 낮아 OOD guard를 통과했다. Leipzig 자료가 HPLT web crawl과 의미상 중복되지 않는다고 보장할 수 없으므로 이를 contamination-free 외부 검증이나 한국어 일반 우월성으로 부르지 않는다.

## 무결성

- 세 policy는 seed별 initialization, training order, train/calibration/test byte stream, model graph와 exact 86-patch rate를 공유했다.
- 31,250개 test sequence의 per-sequence loss에서 모든 BPB와 paired contrast를 재구성했다.
- checkpoint state/artifact, report, loss artifact hash가 모두 일치했다.
- patch matrix와 diagnostics를 현재 코드와 source stream에서 독립 재구성해 일치시켰다.
- OOD summary가 같은 primary checkpoint lineage를 사용했음을 재검증했다.

## 아직 결론 내릴 수 없는 것

1. **실제 효율:** W86은 C86/F86과 global patch 수가 같다. BPB 개선은 latency나 FLOPs 감소가 아니다.
2. **원인 식별:** W−C는 observed whitespace association뿐 아니라 boundary phase, displacement, patch-length distribution도 함께 바꾼다. D/P mechanism controls가 필요하다.
3. **확정성:** Gate J는 독립 confirmation seed 57,721/65,537를 포함한 5-seed 결과가 있어야 평가된다.
4. **Pareto 우위:** authentic SpaceByte-compatible cadence S와 learned entropy policies E/EC의 quality·router-inclusive cost가 아직 없다.
5. **생성 품질과 속도:** teacher-forced BPB는 free-running UTF-8 validity나 incremental batch-1 latency를 보장하지 않는다.
6. **출판 규모와 표준 baseline:** 19.6M/128M-byte 결과는 mechanism scale이다. 50–100M/최소 256M bytes, standard byte-BPE, downstream Korean tasks가 남아 있다.
7. **한국어 고유성:** 비한국어 matched control이 없으므로 현재 효과를 한국어 특유의 현상으로 부를 수 없다.

## 사전 고정된 다음 단계

결과를 이용해 W를 수정하지 않는다. 다음 순서로 진행한다.

1. Initial D/P mechanism controls로 whitespace association이 delayed phase와 matched-frequency placebo를 넘는지 검사한다.
2. S/E/EC initial policies를 학습하고 W의 learned-router-inclusive Pareto 위치를 계산한다.
3. F/C/W confirmation seeds와 OOD를 완료해 Gate J를 판정한다.
4. Gate J가 통과할 때 D/P confirmation과 Gate M을 계산한다.
5. Reduced-rate C/W 64/72-patch conversion을 calibration-only rule로 선택·확인한다.
6. Five-seed BPB noninferiority를 통과한 경우에만 실제 incremental controlled/free-running latency를 측정한다.
7. Compact Final Value Gate 통과 뒤에만 50–100M publication-scale feasibility와 standard byte-BPE/downstream comparison으로 확장한다.

논문의 `efficient inference` 또는 `faster generation` 주장은 compact와 publication-scale Final Value Gate를 모두 통과하기 전에는 사용하지 않는다.
