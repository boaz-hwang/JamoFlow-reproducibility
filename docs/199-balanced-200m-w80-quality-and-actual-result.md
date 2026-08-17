# Balanced 200M W80 quality and actual-inference result

> 작성일: 2026-08-17
>
> 상태: **training, independent checkpoint replay, five-session actual inference complete**
>
> 선행 결정: [W72 quality failure and W80 pivot](./197-balanced-200m-quality-failure-result-and-w80-pivot.md)
>
> 사전 계약: [W80 quality-rescue protocol](./198-balanced-200m-w80-rescue-protocol.md)

## 1. 한 문장 결론

188.6M-parameter W80은 C86 대비 calibration 품질을 보존했고 실제 controlled/free 추론을
각각 `2.887%`/`2.475%` 줄였지만, compact 19.6M의 `2.628%`/`2.531%`보다 개선율이
커졌다는 통계적 근거는 얻지 못했다.

따라서 이번 확장은 **품질을 보존한 작은 실제 효율 효과의 두 trained scale 재현**에는
성공했지만, **모델이 커질수록 개선율이 증가한다는 scale-amplification 가설**에는
실패했다.

## 2. 품질 결과

Candidate와 reference는 동일한 188,639,808-parameter graph, initial state, 127,991,808
training bytes, sequence order, AdamW 설정과 7,812 updates를 사용했다. 달라진 것은 causal
patch schedule뿐이다.

- reference: codepoint-grid C86
- candidate: whitespace-grid W80
- C86 calibration BPB: `1.4411260692502428`
- W80 calibration BPB: `1.4451835724492714`
- W80 minus C86: **`+0.0040575031990286 BPB`**
- 사전 고정 noninferiority margin: `+0.010 BPB`
- contiguous 64-sequence block bootstrap 95% interval:
  `[+0.0030696088, +0.0051141943] BPB`
- 판정: **quality pass**

별도 Git commit의 verifier가 W80 checkpoint를 다시 로드해 15,625-sequence calibration
forward 전체를 재실행했다. 재계산한 float32 NLL 배열은 학습 worker가 저장한 배열과 bitwise
동일했다. 이 replay가 통과한 뒤에만 actual timing을 열었다.

W72의 같은-scale 차이는 `+0.0242004779 BPB`로 실패했다. W80은 C86 대비 제거하는
schedule positions를 14개에서 6개로 줄여, 이 품질 손실을 허용선 안으로 회복했다.

## 3. 실제 추론 결과

다섯 fresh-process Apple-MPS sessions에서 4 warmup + 16 measured Korean cases를 사용했다.
각 prompt/role/mode는 3회 반복 후 cell median으로 접었고, 세션과 프롬프트를 crossed
bootstrap의 독립 축으로 사용했다. Timer는 runtime construction, structural selector,
parallel prefill, cached incremental decode, argmax/strict UTF-8 DFA/stop과 final MPS
synchronization을 포함했다.

| Mode | C86 median | W80 median | Reduction | Crossed 95% interval | Positive prompts | Positive sessions |
|---|---:|---:|---:|---:|---:|---:|
| controlled replay | 533.636 ms | 518.231 ms | **2.887%** | [2.119%, 3.209%] | 16 / 16 | 5 / 5 |
| strict-valid free running | 561.023 ms | 547.140 ms | **2.475%** | [1.948%, 3.052%] | 16 / 16 | 5 / 5 |

모든 session에서 incremental/parallel/full logits, argmax, structural boundary, cache
diagnostics와 strict-valid free output 검증을 통과했다. 255-byte observed path에서 C86은
43 patches, W80은 40 patches를 만들었다. 즉 약 `6.98%`의 patch-event 감소가 실제 E2E에서
약 `2.5--2.9%`로 전환됐다.

## 4. 왜 사전 primary는 실패했는가

사전 계약은 각 mode의 point estimate가 해당 compact matched-quality 값을 넘어야 했다.

| Mode | Compact reduction | 188.6M W80 reduction | Difference |
|---|---:|---:|---:|
| controlled | 2.628% | 2.887% | +0.259 percentage points |
| free running | 2.531% | 2.475% | -0.056 percentage points |

Controlled point는 조금 커졌지만 bootstrap lower `2.119%`가 compact point `2.628%`를
넘지 못했다. Free-running point는 compact point보다 작았다. 두 mode co-primary 계약에서
free-running point clause가 실패했으므로 overall status는 `w80_actual_primary_fail`이다.
어느 mode도 lower bound가 compact point를 넘지 않아 strong scale-amplification support도
false다.

이는 “W80가 빠르지 않다”는 뜻이 아니다. 두 mode 모두 CI가 0보다 높고 16/16 prompts,
5/5 sessions가 같은 방향이다. 실패한 주장은 **실제 개선의 존재**가 아니라 **compact보다
개선율이 커졌다는 주장**이다.

## 5. 수정된 연구 결론

기존 질문은 “2.5% 개선이 모델 크기와 함께 커질 수 있는가”였다. 현재 증거가 허용하는
답은 다음과 같다.

1. Random-weight graph에서는 W72 schedule gap이 크기와 함께 커져 1.6B에서 약 10.2%까지
   도달할 수 있다. 이는 systems headroom이지 trained matched-quality evidence가 아니다.
2. Trained 188.6M에서 그대로 쓴 W72는 품질을 크게 잃어 비교 자격을 상실했다.
3. 더 보수적인 W80은 품질을 회복하고 약 2.5--2.9% 실제 개선을 재현했다.
4. 그러나 그 개선은 compact 19.6M의 약 2.5--2.6%보다 통계적으로 커지지 않았다.

따라서 **파라미터 수 자체가 whitespace boundary 효과를 자동 증폭한다**는 해석은
지지되지 않는다. 현재 가장 타당한 해석은 다음이다.

> 품질이 허용하는 patch-density 감소량이 scale에 따라 달라질 수 있고, 이 품질 제약이
> random-weight systems headroom의 상당 부분을 상쇄한다. 품질을 보존한 범위에서는 두
> trained scale 모두 작지만 일관된 약 2.5% 실제 E2E 효과를 보였다.

W72와 W80을 서로 다른 scale에서 사용했으므로 이 실험은 순수 scale causal contrast도
아니다. 반대로 그 차이를 숨기고 하나의 scaling law를 적합해서도 안 된다.

## 6. 연구 방향 결정

- W82/W84 후속 탐색은 하지 않는다. 결과를 본 뒤 patch count를 계속 조정하면 선택 편향만
  늘고, 남은 theoretical headroom도 작다.
- random-weight 10% 결과를 trained speedup으로 승격하지 않는다.
- 현 논문의 중심은 `19.6M five-seed quality + actual`, `188.6M one-seed quality-rescue +
  five-session actual`, 그리고 `scale amplification negative`의 삼각 증거다.
- 188.6M은 0.6785 raw byte/parameter만 학습한 severe-undertraining screen이다. 충분히 학습된
  large LLM, production Korean, CUDA/general-hardware claim은 하지 않는다.
- 순수 scale 효과를 다시 묻는 후속 연구는 동일 policy density, 다중 model seeds, 더 높은
  byte/parameter와 별도 hardware를 사전에 고정해야 한다. 현재 논문 제출을 위해 그 대규모
  캠페인을 결과 기반으로 추가하지 않는다.

## 7. 가장 강하게 허용되는 문장

> Across a five-seed 19.6M study and a one-seed, severely undertrained 188.6M
> replication, quality-qualified whitespace-aware causal patch schedules
> produced consistent 2.5--2.9% end-to-end reductions on Apple MPS. The larger
> trained screen did not support scale amplification: its free-running point
> estimate did not exceed the compact result, and neither mode's confidence
> lower bound exceeded the compact point estimate.

`larger models increase the speedup`, `10% trained speedup`, `publication-scale efficient LLM`,
또는 `pure scale effect`는 이 결과로 허용되지 않는다.
