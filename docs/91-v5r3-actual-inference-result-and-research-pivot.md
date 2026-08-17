# v5r3 실제 추론 결과와 연구 방향 전환

> 작성일: 2026-08-13
>
> 상태: **다섯 세션·열 개 격리 memory unit·immutable summary 완료**
>
> authoritative artifact:
> `results/phase3-inference-actual-v5r3/summary.json`

## 1. 결론

W72는 품질이 맞는 C86보다 실제 batch-1 incremental generation에서 일관되게
빨랐지만, 사전 고정한 10% 효율 기준에는 크게 못 미쳤다.

- controlled replay end-to-end: **2.628% 감소**, crossed-bootstrap 95% CI
  **[2.026%, 3.526%]**
- strict-valid free-running end-to-end: **2.531% 감소**, 95% CI
  **[1.687%, 3.127%]**
- 두 mode 모두 5/5 timing session과 5/5 model seed에서 방향은 양수였다.
- 두 mode 모두 10% 이상인 session은 0/5였다.
- 따라서 사전 고정한 primary gate는 실패했고 status는
  `fail_matched_quality_actual_efficiency_v5r3`다.

이는 `효과 없음`도, `positive efficiency paper`도 아니다. 가장 정확한 판정은
**재현 가능한 소폭 실제 개선을 동반한 primary-negative systems result**다. 10% 기준을
사후에 낮추거나 analytical FLOPs를 실제 speedup으로 대체하지 않는다.

## 2. Co-primary actual-inference 결과

| mode | W72 median | C86 median | 감소율 | crossed 95% CI | 양수 session | 10% session | 양수 seed | median seed | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| controlled replay E2E | 361.070 ms | 370.816 ms | **2.628%** | [2.026%, 3.526%] | 5/5 | 0/5 | 5/5 | 2.795% | fail |
| strict-valid free-running E2E | 377.970 ms | 387.783 ms | **2.531%** | [1.687%, 3.127%] | 5/5 | 0/5 | 5/5 | 2.376% | fail |

Repetition은 독립 표본으로 세지 않았다. 각 seed×prompt cell의 다섯 repetition을
median으로 접은 뒤 timing session, model seed, shared prompt를 crossed resampling했다.
두 mode의 CI가 모두 0을 넘으므로 작은 양의 효과 자체는 안정적이다. 그러나 gate는
aggregate 10%, 3/5 session 10%, median seed 10%를 요구했고 모두 충족하지 못했다.

### 세션별 E2E 감소율

| session | controlled | free running |
|---|---:|---:|
| 1 | 2.521% | 2.256% |
| 2 | 2.418% | 2.548% |
| 3 | 2.736% | 2.078% |
| 4 | 2.806% | 2.708% |
| 5 | 2.675% | 2.191% |

순서 민감도에서도 방향 반전은 없었다. Controlled E2E는 candidate-first cell에서
2.445%, reference-first cell에서 2.718%였고, free running은 각각 2.232%와
2.425%였다.

## 3. 보조 latency 결과

| endpoint | W72 | C86 | 감소율 | crossed 95% CI | 해석 |
|---|---:|---:|---:|---:|---|
| controlled decode | 348.556 ms | 358.556 ms | 2.789% | [2.141%, 3.615%] | 작은 양의 decode 효과 |
| controlled TTFT | 13.175 ms | 13.196 ms | 0.157% | [-1.584%, 2.121%] | 효과 없음 |
| free decode | 365.910 ms | 375.447 ms | 2.540% | [1.732%, 3.197%] | 작은 양의 decode 효과 |
| free TTFT | 13.210 ms | 13.198 ms | -0.090% | [-1.760%, 1.675%] | 효과 없음 |

속도 차이는 prefill/TTFT가 아니라 decode 구간에 있다. 이는 patch cadence가
autoregressive consume 중 global update 횟수를 바꾸지만 prompt parallel prefill에는
큰 차이를 만들지 않는 구현 구조와 일치한다. 이 표는 mechanism diagnostic이며
co-primary E2E gate를 대체하지 않는다.

## 4. 정확성·출력 유효성

다섯 세션의 모든 seed×role correctness check가 통과했다. CPU original semantic
oracle, MPS safety/TV envelope, full-causal 대 cached runtime, parallel prefill, boundary
trace, timed masked-greedy byte가 모두 재구성됐다.

- 검증된 free-running output: **16,000개**
- strict UTF-8: 16,000/16,000
- 최초 허용 boundary에서 정지: 16,000/16,000
- session·repetition 간 deterministic content: pass
- replacement character 없음: 16,000/16,000
- Jamo transition valid: 16,000/16,000

따라서 2.5% 차이를 output corruption, 다른 stopping rule 또는 approximate decoding으로
설명할 수 없다.

## 5. Descriptive memory

Memory는 사전 고정대로 publication gate가 아니라 role-isolated descriptive endpoint다.

| 항목 | W72 | C86 | 차이 |
|---|---:|---:|---:|
| parameter bytes | 78,384,384 | 78,384,384 | 0 |
| max MPS current increment | 78,387,968 | 78,387,968 | 0 |
| max MPS driver increment | 206,520,320 | 206,520,320 | 0 |
| max process RSS high-water increment | 555,778,048 | 553,484,288 | +2,293,760 |

Seed별 RSS 차이는 양·음 방향이 섞였고 native resettable MPS peak도 지원되지 않았다.
따라서 memory improvement claim은 하지 않는다. 같은 parameter graph이고 router가 없는
두 정책이라는 점과 관측값은 일치한다.

## 6. Analytical workload와 실제 latency의 관계

W72는 C86보다 512-byte sequence당 data patch를 86에서 72로 줄인다.

- data patches: **16.279% 감소**
- dummy 포함 global positions: **16.092% 감소**
- 현재 dense-matmul 회계: **8.332% 감소**
- 실제 E2E latency: **2.53–2.63% 감소**

8.332%와 2.5%는 동일한 분모가 아니므로 `31% 실현율` 같은 값을 이론적 효율로
해석하지 않는다. 다만 gap의 방향은 명확하다. Incremental runtime에서는 두 정책
모두 output byte마다 다음을 그대로 수행한다.

1. local encoder byte update
2. local decoder byte update
3. byte LM head와 greedy 선택
4. UTF-8/Jamo state update와 host-side dispatch

W72가 제거하는 것은 boundary에서만 실행되는 patch reduce, cross/global update 중
7회(43→36)뿐이다. Sequential byte step 수는 controlled에서 양쪽 모두 127회다.
따라서 global patch cadence만 바꾸는 현재 방법은 품질에는 유용하지만 실제 latency의
주요 local-byte 경로를 제거하지 못한다.

## 7. Fable 5 검토에 대한 결과 기반 판정

`fable5-연구-중간-검토.md`가 제기한 핵심 우려, 즉 global compute 감소가 local
byte-sequential 비용 때문에 큰 end-to-end speedup으로 이어지지 않을 수 있다는 판단은
결과로 지지됐다. 반면 다음 강한 표현은 여전히 수용하지 않는다.

- `16.3% × global share`를 이론적 latency 상한으로 부를 수 없다.
- 실제 결과는 0%가 아니라 두 mode 모두 약 2.5%의 재현 가능한 개선이다.
- 같은 parameter bytes가 곧 모든 runtime memory가 같다는 사전 결론은 정당하지
  않았지만, 이번 descriptive 측정에서는 유의미한 memory 이득이 관찰되지 않았다.
- quality/geometry만으로 효율 논문의 성공을 선언할 수 없다.

따라서 외부 검토의 위험 진단은 수용하되, 실제 측정이 제공한 효과 크기와 claim
경계를 최종 판단으로 사용한다.

## 8. 연구 방향 수정

### 중단하거나 보류할 것

1. W72를 그대로 50M/75M/100M으로 확장하는 대규모 학습은 보류한다.
2. patch count 또는 FLOPs만으로 더 큰 규모에서 10%가 될 것이라 외삽하지 않는다.
3. v5r3 threshold를 낮추거나 TTFT/decode-only를 primary로 바꾸지 않는다.
4. S rate×placement 분해는 중요한 mechanism 연구지만 실제 효율 구조를 고치기 전에는
   우선순위를 낮춘다.

### 다음 주가설

> 한국어 UTF-8의 구조를 이용해 expensive global patch event뿐 아니라 순차적인 local
> byte decoding step도 줄여야 matched-quality end-to-end 개선이 10%를 넘을 수 있다.

가장 유망한 방향은 W patching에 **multi-byte block generation 또는 local
self-speculation**을 결합하는 것이다. 특히 precomposed Hangul syllable이 보통 세 UTF-8
byte라는 점을 이용하되, 결과 분포와 strict UTF-8 validity를 바꾸지 않는 검증 경로가
필요하다.

### 단계별 실행

1. **Exploratory component profiler**: exact W72/C86 checkpoint와 calibration-only case를
   사용해 local encoder, patch finalize/global, local decoder, LM head, selector/host
   비용을 분해한다. Component별 synchronize가 runtime을 왜곡하므로 이 결과는
   diagnostic으로만 사용하고 whole-trial wall time과 함께 보고한다.
2. **Acceptance ceiling**: final test를 쓰지 않고 train/calibration에서 2–3 byte block의
   oracle 및 cheap-draft acceptance를 측정한다.
3. **저비용 prototype**: shallow local draft 또는 multi-token head가 UTF-8 DFA 아래
   여러 byte를 제안하고 main local decoder가 한 번에 검증하도록 구현한다. Greedy
   exactness 또는 표준 speculative acceptance를 보존한다.
4. **Compact matched-quality actual timing**: 기존 W72와 C86이 아니라 새 candidate와
   적절한 quality-matched baseline을 새 protocol로 비교한다. 실제 E2E gate를 통과하지
   못하면 다시 구조를 수정한다.
5. **그 이후에만 scale/BPE/CUDA**: compact actual 효율이 충분할 때 가장 큰 Mac-feasible
   scale, BPE16K/32K, Korean downstream과 별도 CUDA replication을 연다.

이 수정은 결과를 보고 목표를 바꾼 것이 아니라, 원래 목표인 실제 추론 효율을
달성하지 못한 원인을 반영해 intervention 대상을 global cadence에서 sequential byte
generation으로 옮긴 것이다.

## 9. Summary correction과 공개 범위

첫 summary 실행은 저장된 counter array가 `(seed,prompt,repetition)`인데 원 validator가
단일-seed `(prompt,repetition)`만 받는 shape mismatch로 결과 출력 전에 중단됐다. 결과를
읽지 않은 상태에서 seed 축을 고정 순서로 순회하며 원 validator를 그대로 적용하는
summary-only adapter를 별도 manifest와 tests로 봉인했다. Timing artifacts, bootstrap,
gate, correctness threshold는 바꾸지 않았다. 전체 587 tests 통과 뒤 summary를 생성·commit한
후 처음 결과를 열었다. 자세한 기록은 `docs/90-v5r3-summary-counter-shape-debug.md`에 있다.

이 결과는 Apple M4 Pro 한 대, 19.6M model, 다섯 checkpoint/role, 64개의 Hangul-heavy
128-byte prompt, 128–131 output bytes 범위다. General hardware, production serving,
larger LLM 또는 memory improvement claim으로 일반화하지 않는다.

## 10. 후속 component profile 결과

사후 고정한 2×2 checkpoint×schedule profile에서 candidate와 reference weight 모두
W72 schedule이 C86 schedule보다 decode를 각각 2.852%, 2.842% 줄였다. C86의 22개와
W72의 18개 decode-new patch 차이에 synchronized boundary increment 약 2.54ms를 곱한
10.16ms가 실제 same-checkpoint decode gap 10.12--10.14ms를 거의 전부 설명했다.
반대로 두 schedule은 약 2.36ms의 local-byte base를 127번 똑같이 지불했다.

따라서 본 문서의 pivot은 유지하되 더 구체화한다. 단순 BLT self-speculation이나 generic
multi-token head는 Fast BLT, Medusa, multi-token prediction, MtPC가 이미 선점했다.
다음 주기법 후보는 exact target verification을 유지하면서 Hangul 조합과 UTF-8 scalar
경계를 draft 분포·길이에 이용하고, 같은-cost generic byte-MTP보다 실제로 나은지를
검증하는 orthography-aligned multi-byte decoding이다. 자세한 수치와 kill rule은
`docs/93-exploratory-component-profile-result-and-architecture-decision.md`에 기록한다.

## 11. Learned-draft preflight 이후 수정

Frozen W72에서 약 40K parameter로 맞춘 generic independent/joint UTF-8와 Hangul
parallel/conditional head를 세 초기화로 비교했다. 네 head 모두 사전 gate를 실패했다.
Generic independent가 complete pair 24.379%, first continuation 42.373%로 가장 높았고,
Hangul conditional은 17.702%에 그쳤다. 조합구조 head를 positive candidate로 발전시키는
분기는 종료한다.

단, 사전 acceptance gate는 speculative mismatch correction과 verifier bonus byte를
target-call cost에 포함하지 않았다. Independent head의 관측 acceptance는 verifier당
`2.667522` bytes를 확정할 기회를 뜻하지만, 실제 block target cost는 아직 모른다.
따라서 head를 재튜닝하지 않고 perfect-draft target block kernel의 exactness와 latency
upper-bound만 먼저 측정한다. 자세한 판정과 claim 경계는
`docs/97-hangul-draft-acceptance-result-and-cost-model-correction.md`를 따른다.
