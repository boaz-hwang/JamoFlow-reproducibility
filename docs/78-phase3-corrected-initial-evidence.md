# Phase 3 corrected initial evidence

> 작성일: 2026-08-12  
> 상태: **initial 3-seed 결과 봉인; confirmation·compute-conversion·actual-inference 전**  
> 상위 protocol: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md), [document-cluster correction](./52-document-cluster-inference-integrity-addendum.md), [mechanism authorization correction](./58-mechanism-reanalysis-authorization-correction.md)  
> 해석 범위: 19.6M-parameter, 128M Korean training-byte compact mechanism study

## 1. 결론

교정된 initial Gate I와 initial Gate M은 모두 통과했다. 동일한 86-patch
graph에서 whitespace-associated boundary relocation인 W는 C보다 낮은 test BPB를
보였고, 이 차이는 delayed-phase D와 rate-matched causal-event placebo P에 대해서도
유지됐다.

그러나 이 결과는 아직 **추론 효율 개선의 증거가 아니다**. 가장 강한
calibration-quality raw baseline은 `spacebyte_spacelike`(S)였고, S는 W보다 훨씬
많은 global patches를 사용해 훨씬 낮은 BPB를 얻었다. 따라서 현재 단계의 올바른
판정은 다음과 같다.

> Korean whitespace와 연관된 causal boundary relocation에는 재현 확인 가치가 있는
> same-rate quality 신호가 있다. 그 신호가 실제 compute 감소와 강한-baseline
> quality noninferiority로 변환되는지는 아직 전혀 확인되지 않았다.

## 2. 봉인된 evidence

| Artifact | SHA-256 |
|---|---|
| `results/phase3-primary-clustered/summary.json` | `b015434d146567fce1790c441eb55bf601caec8cdaa07176be8c026e6a0a5706` |
| `results/phase3-all-initial/summary.json` | `f3173b8220a17dea0eb5a43c1b4b33cae893dc769c6e5b4dcd7b05bc8ba496f6` |
| `results/phase3-mechanism-clustered/summary.json` | `5ad5fab30024da0138b2806f71cd4b5e80589bf928a5139e53665923a2057e91` |

세 summary는 commit `a5ef939c0a1913f640af17937c0bc1793ea394f3`의 코드로 생성됐다.
Raw checkpoint, per-window NLL과 patch matrix는 tracked result에 복제하지 않고,
summary가 그 artifact와 state hash를 결속한다.

공통 무결성 조건은 모두 통과했다.

- seed 1,729 / 2,718 / 31,415
- 동일 initialization과 training order의 paired comparison
- F/C/W 모두 split별 정확히 86 data patches
- 31,250 test windows, 734 source documents
- document-cluster inference에 사용 가능한 windows 30,517개(97.6544%)
- checkpoint state/artifact, report, NLL, stream, patch-matrix hash 재구성 일치
- public Leipzig Korean OOD guard 통과

## 3. Gate I: same-rate W−C

| 항목 | 결과 |
|---|---:|
| mean test `W − C` | −0.009882 BPB |
| seed별 차이 | −0.008883 / −0.011414 / −0.009350 BPB |
| negative seeds | 3/3 |
| document-cluster 95% interval | [−0.011428, −0.008663] BPB |
| paired-seed one-sided p-value | 0.003067 |
| Gate I | **pass** |

이는 effect threshold `−0.002 BPB`, sign threshold 2/3, document-cluster upper
bound `< 0`, OOD guard를 모두 만족한다. Initial screen이므로 이 수치를 final
five-seed method evidence로 표현하지 않는다.

## 4. Gate M: 사전 고정 mechanism controls

| Contrast | Mean | Seed signs | Document-cluster 95% upper | Holm-adjusted p | 판정 |
|---|---:|---:|---:|---:|---|
| `W − D` | −0.010308 BPB | 3/3 negative | −0.009202 | 0.001683 | pass |
| `W − P` | −0.020700 BPB | 3/3 negative | −0.019571 | 0.000686 | pass |

따라서 관측된 W 효과를 패치 수가 같은 단순 delayed phase나 calibration에서
event rate를 맞춘 causal placebo만으로 설명하기는 어렵다. 허용되는 attribution은
“observed whitespace association survives the two specified controls”까지다.
Korean morphology, morpheme segmentation, optimal patching 또는 일반적인 learned
routing 우위를 식별한 것은 아니다.

## 5. Six-policy quality 결과와 가장 강한 반증

### 5.1 Initial 3-seed mean BPB

| Policy | Calibration BPB | Test BPB |
|---|---:|---:|
| S: SpaceByte-compatible spacelike cadence | **1.530750** | **1.548823** |
| W: causal whitespace grid, 86 patches | 1.621408 | 1.636415 |
| C: causal codepoint grid, 86 patches | 1.631042 | 1.646297 |
| F: fixed-byte grid, 86 patches | 1.636231 | 1.650904 |
| E: full entropy threshold | 1.638470 | 1.654581 |
| EC: codepoint-constrained entropy threshold | 1.643627 | 1.660590 |

Comparator 선택은 test가 아니라 initial 3-seed mean calibration BPB만 사용한다.
Selected same-rate C는 compute-conversion 뒤 후보에 추가되므로 아직 최종 descriptor를
만들 수 없지만, 현재 여섯 후보 중에는 S가 명확히 최저다.

### 5.2 Quality와 global-compute를 분리해서 읽어야 한다

S의 test mean data-patch count는 window당 153.313이고 W는 정확히 86이다.
즉 S는 W보다 약 78.3% 많은 global tokens를 사용한다. `S − W`의 mean test BPB는
−0.087592이고 document-cluster 95% upper도 −0.084372로, quality 차이는 매우 크다.
이는 S가 좋은 효율 방법임을 자동으로 뜻하지 않지만, W의 compute 감소가 quality를
얼마나 희생하는지 보여주는 강한 Pareto 기준이다.

E와 EC는 대략 W와 같은 mean patch rate를 맞췄지만 W보다 각각 0.018166,
0.024175 BPB 나빴다. EC도 E보다 0.006010 BPB 나빴다. 이 compact geometry에서는
별도 learned entropy router가 structural W를 이기지 못했으며, router train,
full-stream score와 online runtime까지 포함하면 비용 면에서도 불리하다. 이 결과를
learned routing 일반의 실패로 확대하지 않는다.

## 6. 연구 결정

Gate I와 initial Gate M이 모두 통과했으므로 사전 고정된 confirmation 진행 조건은
충족됐다. 다음 실행 순서는 결과에 따라 바꾸지 않는다.

1. F/C/W의 독립 confirmation seeds 57,721 / 65,537 학습
2. 같은 checkpoint의 public Leipzig Korean OOD 평가와 corrected five-seed Gate J
3. Gate J 통과 시 D/P confirmation 및 final Gate M
4. C/W 64/72 initial compute-conversion의 calibration-only rate 선택
5. 선택 rate의 confirmation과, initial calibration만으로 고정한 strongest comparator의
   필요한 confirmation checkpoint 학습
6. five-seed quality noninferiority가 통과한 경우에만 router/selector/cache를 포함한
   실제 incremental latency 측정

S와의 `+0.010 BPB` noninferiority를 통과하지 못하거나 실제 batch-1 latency가 10%
이상 개선되지 않으면 이 branch를 positive inference-efficiency 연구로 판정하지 않는다.
그 경우 결과를 숨기거나 W를 사후 수정하지 않고, 별도 protocol의 multi-byte
proposal-and-verification 연구로 전환한다.

