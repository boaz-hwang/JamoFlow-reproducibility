# Publication-scale Mac feasibility addendum

> 작성일: 2026-08-11
> 상태: **compact actual-inference 및 scale-feasibility 결과 전 고정**
> 실행 조건: compact actual-inference gate 통과
> 상위 protocol: [Actual-inference and compute-conversion protocol](./44-actual-inference-and-compute-conversion-protocol.md)
> BPE campaign 교정: [Dual-BPE sealed-test correction](./61-dual-bpe-sealed-test-correction.md)
> final scale-lock 교정: [family-aware feasibility](./64-family-aware-scale-feasibility-correction.md)

## 1. 목적

사용 가능한 장비는 Apple M4 Pro와 unified memory 48 GB다. “가능한 큰 모델”을 결과나 선호에 따라 정하지 않도록 50M/75M/100M 근방의 세 graph와 선택 기준을 미리 고정한다. Feasibility는 quality 비교가 아니며 candidate/reference loss를 열어 scale을 선택하지 않는다.

## 2. 고정 graph

세 graph 모두 512-byte context, encoder 2층, decoder 2층, local/global FFN 3배, hash vocabulary 16,384를 쓴다. Global position capacity는 compact와 같은 1,032로 통일한다. Selected rate는 graph parameter를 바꾸지 않는다.

| Target | Exact main-graph params | Local width/heads | Global width/heads/layers |
|---:|---:|---:|---:|
| 50M | 49,823,488 | 256 / 8 | 512 / 8 / 12 |
| 75M | 76,492,480 | 320 / 8 | 640 / 10 / 12 |
| 100M | 98,403,360 | 352 / 11 | 704 / 11 / 13 |

세부 spec은 `src/jamoflow/publication_scale.py`가 단일 source of truth다. 이 표는
candidate와 raw reference의 main BLT graph만 센다. E/EC raw가 선택되면 후속
family-aware lock에서 별도 entropy-router parameter, train/score/runtime 시간과 memory를
더하며 [auxiliary audit](./77-publication-auxiliary-router-and-execution-audit.md)가 우선한다.

## 3. Blind candidate-only preflight workload

Compact gate 통과 뒤 같은 public HPLT3 train stream의 고정된 첫 batch와 selected W-rate matrix로 다음을 측정한다.

- float32 AdamW, batch 32, 512 bytes의 warmup 1회와 steady train step 3회
- batch 64 teacher-forced evaluation step
- batch-1 128-byte parallel prefill과 1-byte incremental decode
- parameter count, step wall time, MPS current/driver allocation, recommended maximum memory

100M 결과부터 골라 실행하지 않고 세 후보를 모두 독립 subprocess에서 측정한다. Worker failure와 OOM도 결과로 남긴다. Quality metric은 계산하거나 scale 선택에 사용하지 않는다. 이 workload는 candidate graph의 provisional preflight이며 final scale lock은 네 runtime family의 후속 preflight를 요구한다.

실행은 clean Git tree에서만 허용한다. 각 worker는 compact actual-inference summary, 사전 고정 comparator selection, HPLT3 원본과 integrity artifact, 선택된 train stream과 patch matrix의 hash를 기록한다. Parent는 worker가 보고한 model spec, parameter 수, 세 step의 median과 256M-byte 투영식, stage별 finite flag, 모든 memory snapshot을 독립적으로 다시 계산한다. Worker는 UUID가 붙은 임시 경로에만 쓰고 검증을 통과한 완성본만 target 경로로 원자적으로 교체한다. 따라서 강제 재실행 실패가 과거 성공 산출물을 새 결과처럼 재사용할 수 없다. 실패 시간은 JSON `null`로 기록하며 `Infinity`/`NaN`은 허용하지 않는다.

## 4. 안전 기준과 선택

256M train bytes, batch 32, one pass의 model당 시간을 steady train-step median으로 투영한다. 다음을 모두 만족한 후보만 feasible이다.

1. train/eval/incremental workload가 exception 없이 끝나고 모든 loss와 시간이 finite
2. exact parameter count 일치
3. 관측된 MPS driver high-water가 `recommended_max_memory`의 75% 이하
4. projected model time에 1.20 safety factor를 곱한 값이 12시간 이하
5. provisional screen에서 candidate model time × 12가 120시간 이하

5는 final campaign time의 증거가 아니라 candidate-only early rejection이다. Final selection은 candidate, raw reference, body-matched 16K BPE, parameter-matched 32K BPE의 family별 실제 batch/time/memory를 모두 측정하고 `3 × 1.20 × Σ family hours <= 120`을 통과한 가장 큰 target으로 다시 고정한다. 100M이 실패하면 75M, 그다음 50M으로 내려간다. 어느 것도 통과하지 못하면 작은 model 결과로 publication-scale claim을 대신하지 않고 외부 compute 확보가 필요하다고 판정한다.

MPS에는 reset 가능한 peak allocator API가 없으므로 `driver_allocated_memory`의 worker-session high-water를 사용하며 이를 hardware peak의 완전한 측정이라고 부르지 않는다. 분모에는 worker 중 관측한 `recommended_max_memory`의 최솟값을 써 보수적으로 판정한다. 각 후보를 별도 process에서 실행해 allocator 잔류가 다음 후보에 섞이지 않게 한다. 세 후보를 모두 측정하므로 OOM·속도 실패도 선택 과정의 관측값이며, 실패 후보를 더 작은 batch나 다른 precision으로 사후 구조하지 않는다.

## 5. Scale 결과의 범위

Feasibility 통과는 학습을 허용할 뿐 method 성공이 아니다. 선택된 scale에서 최소 256M Korean bytes, 세 seeds, candidate/raw reference/16K BPE/32K BPE, 실제 incremental latency, Korean downstream noninferiority를 끝내야 Final Value Gate를 판정한다.
