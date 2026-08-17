# Phase 2 cost results: mean patch rate is not implemented cost

> 작성일: 2026-08-10  
> Protocol: [Phase 2](./10-phase2-korean-causal-protocol.md) §6, §11  
> 기계 판독 결과: [`results/phase2-cost/summary.json`](../results/phase2-cost/summary.json)  
> 상태: **Gate F 통과; MPS teacher-forced 범위**

## 1. 결론

Parameter-free causal grid 세 정책은 learned entropy-full보다 quality harm margin 안에 있었고, router와 batch padding을 포함하면 analytical FLOPs와 MPS latency 모두 10% 이상 낮았다. Gate F는 통과했다.

Scale-up 후보인 whitespace-aware grid 기준:

- ideal unpadded dense-matmul FLOPs 절감: **27.39%**
- batch-64 padding-aware FLOPs 절감: **34.52%**
- batch-1 direct teacher-forced latency 절감: **30.52%**
- batch-1 median: 5.306 ms vs entropy-full 7.637 ms

이 결과는 256-byte teacher-forced forward의 selector 비용 비교다. Autoregressive generation latency나 CUDA serving 성능을 입증하지 않는다.

## 2. Benchmark 조건

- 장치: Apple M4 Pro, MPS, float32
- seed/checkpoint: 1,729
- Korean test windows: primary held-out split
- batch sizes: 1 / 8 / 64
- warm-up: 조건별 10회
- 측정: 조건별 100회
- 조건 순서: 매 repetition 무작위 interleave
- 각 측정 전후 device synchronize
- inputs는 device에 미리 적재

Direct pipeline에 포함한 것:

- structural boundary selector
- entropy router forward
- entropy MPS→CPU transfer
- Python/NumPy threshold selector
- patch-length device upload
- main BLT forward

Parser candidate mask는 streaming decoder가 이미 유지하는 UTF-8 state로 간주했다. Mask를 corpus에서 읽는 I/O는 timing에 넣지 않았다.

## 3. Direct pipeline latency

Median milliseconds:

| 정책 | B=1 | B=8 | B=64 |
|---|---:|---:|---:|
| fixed-byte | 4.949 | 5.320 | 28.243 |
| causal codepoint | 5.295 | 5.790 | 29.617 |
| delimiter-aware | 5.275 | 5.878 | 31.132 |
| whitespace-aware | 5.306 | 5.887 | 31.102 |
| entropy-full | 7.637 | 8.605 | 47.125 |

Batch 1에서 whitespace selector가 fixed-byte보다 약 0.36 ms 추가되지만 entropy router pipeline보다 약 2.33 ms 짧았다.

## 4. Analytical FLOPs

Dense matmul만 세며 multiply-add를 2 FLOPs로 계산했다. Fixed 43-patch main BLT는 sequence당 257,261,568 FLOPs다. Entropy-full은 seed 1,729 test에서 평균 43.191 patches였으며 router 포함 ideal mean은 354,291,882 FLOPs였다.

### Ideal unpadded

| 정책 | mean FLOPs/sequence | entropy-full 대비 절감 |
|---|---:|---:|
| structural 43-patch | 257,261,568 | **27.39%** |
| entropy-full + router | 354,291,882 | — |

### Implemented batch-max

Entropy policy는 row마다 patch 수가 달라 batch 최대 폭으로 계산된다.

| Batch | entropy mean batch-max patches | patch-slot padding | entropy mean FLOPs | structural 대비 절감 |
|---:|---:|---:|---:|---:|
| 1 | 43.191 | 0% | 354,291,882 | 27.39% |
| 8 | 51.278 | 15.75% | 374,314,071 | 31.27% |
| 64 | 58.581 | 26.28% | 392,868,690 | 34.52% |

Corpus-wide padded width 112 같은 최악값을 모든 batch에 부과하지 않고, 실제 순서의 batch별 maximum을 사용했다. 반대로 평균 43만 넣어 padding을 무시하지도 않았다.

## 5. Gate F

Whitespace-aware candidate:

- whitespace quality difference vs entropy-full: −0.00966 BPB, margin 통과
- ideal analytical reduction ≥10%: 통과
- batch-1 direct latency reduction ≥10%: 통과
- batch-64 padding-aware reduction ≥10%: 통과

Codepoint와 delimiter-aware 후보도 같은 cost gate를 통과했다. Phase 2b에서 whitespace-only와 C2가 사실상 동률이므로 scale-up은 더 좁은 whitespace-aware policy를 선택한다.

## 6. 해석 한계

1. MPS 결과는 CUDA kernel·serving stack으로 일반화되지 않는다.
2. Teacher forcing은 autoregressive sequential step 수를 측정하지 않는다.
3. Python selector는 연구 구현이다. 최적화된 fused selector에서는 structural·entropy 양쪽 latency가 달라질 수 있다.
4. FLOPs는 embedding lookup, normalization, RoPE, activation, softmax, hashing, memory movement를 제외한다.
5. 별도 router 방식의 비용을 측정했다. Main model에 integrated된 future learned router와 동일하지 않다.

따라서 scale-up에서는 CUDA incremental generation benchmark가 필수다. 현재 결과는 “compact setting에서 비용 우위가 Python overhead를 포함해도 사라지지 않는다”까지만 지지한다.
