# Fresh Korean vocabulary-adaptation one-seed result

> 작성일: 2026-08-15
>
> 증거 커밋: `5629bcc`
>
> 결과 SHA-256: `8817b5ff57b5edd9d3e300cae2718604efe0eaf753734bb277c1b8a4469fe625`
>
> 판정: optimizer-geometry와 dense-8K deployment opportunity 모두 다음 fail-fast 단계 승인

## 결론부터

새 한국어 train/calibration stream에서 고정 update geometry는 단순 dense-8K와 two-stage
tokenizer-expansion 대조군을 모두 이겼다. 또한 dense-2K continuation보다 calibration BPB가 낮았고,
같은 128MB raw train stream에서 optimizer wall time도 줄었다. 따라서 이 branch는 더 이상
historical B1의 post-hoc 관측만이 아니다.

그러나 이 결과만으로 논문 성공이나 추론 효율 개선을 주장할 수는 없다. 아직 모델 seed가 하나이고,
선택과 평가가 fresh calibration에 머물며, 8K graph는 2K보다 parameter가 27.99% 많다. 다음 질문은
오직 하나다.

> 품질을 회복한 실제 trained dense-8K checkpoint가 batch-1 controlled replay와 free-running
> generation의 전체 경로를 dense-2K보다 각각 10% 이상 줄이는가?

이 실제 측정을 통과하기 전에는 token-step 감소나 training-time 감소를 inference success로 부르지
않는다.

## 사전 고정 실험과 독립 검증

계획은 코드 커밋 `615b7c0`, plan-only 커밋 `0bd4647` 순서로 결과 전에 봉인했다. 네 역할은 동일한
ordered 128MB raw stream을 한 번씩 소비했다.

| role | vocab | optimizer steps | document BPB | optimizer seconds |
|---|---:|---:|---:|---:|
| `dense2k_joint` | 2,048 | 2,213 | 1.394225 | 2,309.73 |
| `dense8k_standard_joint` | 8,192 | 1,677 | 1.402361 | 1,836.42 |
| `dense8k_inplace_two_stage` | 8,192 | 1,677 | 1.424880 | 1,641.59 |
| `dense8k_update_geometry` | 8,192 | 1,677 | **1.384009** | **1,493.12** |

Summary는 worker가 저장한 metric을 그대로 신뢰하지 않았다. 네 final checkpoint를 다시 load해
contiguous NLL과 383개 full-document NLL을 모두 재계산했고, 네 역할의 모든 float array가 worker
artifact와 bitwise identical임을 확인한 뒤 bootstrap과 선택을 수행했다.

## Quality 결과

### dense-2K 대비 noninferiority

차이는 `8K - 2K`이며 낮을수록 좋다.

| 8K role | point BPB | 95% interval | fixed +0.010 gate |
|---|---:|---:|---|
| standard joint | +0.008137 | [+0.006957, +0.009383] | pass |
| in-place two-stage | +0.030656 | [+0.028956, +0.032446] | fail |
| update geometry | **-0.010216** | **[-0.011340, -0.009018]** | pass |

Ordinary 8K도 margin 안에 들어왔다. 그러므로 larger vocabulary 자체의 deployment opportunity는
geometry novelty와 분리해도 남는다. Geometry는 더 강하다. 이 seed에서는 2K와 같아진 정도가 아니라
약 0.0102 BPB 더 낮았다.

### optimizer-method gate

차이는 `geometry - control`이며, 사전 계약은 point `<= -0.002`와 upper bound `<= 0`을 둘 다
요구했다.

| control | point BPB | 95% interval | method gate |
|---|---:|---:|---|
| standard joint | **-0.018352** | [-0.018974, -0.017768] | pass |
| in-place two-stage | **-0.040871** | [-0.042260, -0.039564] | pass |

따라서 audit-fixed geometry는 fresh one-seed method screen을 통과했다. 효과 크기가 최소 기준보다
약 9배 크고 두 대조군 모두에 같은 방향이라는 점은 다음 seed 확인을 정당화한다. 다만 이 단계는
recipe를 새 model seed에 재현하지 않았으므로 optimizer 방법의 publication claim은 승인하지 않는다.

## Systems accounting

| 8K role | step 감소 | optimizer-time 감소 | parameter 증가 |
|---|---:|---:|---:|
| standard joint | 24.22% | 20.49% | 27.99% |
| in-place two-stage | 24.22% | 28.93% | 27.99% |
| update geometry | 24.22% | **35.36%** | 27.99% |

Geometry의 optimizer-time 이득이 standard보다 큰 이유를 곧바로 알고리즘 효과로 해석하면 안 된다.
세 역할은 같은 step 수이지만 단일 순차 session에서 실행되었고 thermal/order 효과가 완전히 분리된
반복 측정이 아니다. 이 수치는 실제 기록된 training cost이며 유망한 보조 증거지만 primary novelty
근거는 quality 대조와 이후 inference다.

Two-stage는 학습시간을 줄였지만 quality가 크게 나빠졌다. 이 compact 조건에서는 body를 60% raw
progress 동안 동결하는 대가가 너무 컸다. 대형 논문의 600B+400B recipe가 틀렸다는 결론은 아니다.
우리 128MB/19–25M graph에서 그 축소 analogue가 강한 대조군이 아니었다는 결과다.

## Fable 5 검토와의 관계

수용할 비판은 결과에도 그대로 적용한다.

- 24.22% fewer token steps는 실제 generation latency가 아니다.
- 35.36% shorter optimizer time도 inference 효율의 대체 지표가 아니다.
- 8K는 parameter/checkpoint가 더 크므로 memory 개선으로 부를 수 없다.
- 새-row update geometry가 quality를 고친 사실만으로 한국어 특화 architecture novelty가 되지 않는다.
- controlled replay와 free-running을 둘 다 재야 tokenizer 경계와 실제 autoregressive feedback 비용을
  분리할 수 있다.

반대로 이 결과 때문에 받아들이지 않아야 할 결론도 분명하다. 이전 negative architecture 결과만으로
연구를 작은 분석 논문으로 종료할 필요는 없다. Fresh-data, strong-control, presealed experiment에서
ordinary dense vocabulary adaptation과 geometry가 실제 positive를 냈기 때문이다. 이제 더 큰 연구로
갈지 결정하는 병목은 실제 trained inference 한 가지다.

## 연구 방향 결정

계획을 불필요하게 넓히지 않는다.

1. `dense8k_update_geometry`와 `dense2k_joint`의 exact trained checkpoint를 고정한다.
2. Fresh calibration에서 outcome-independent하게 고른 서로 다른 64문서의 128-byte prompt와
   128-byte continuation을 사용한다.
3. prompt tokenization, model prefill, 매 autoregressive step의 argmax/readback, KV cache, output-byte
   reconstruction, strict UTF-8 stop까지 포함한 batch-1 E2E를 잰다.
4. controlled raw-continuation replay와 free-running strict-UTF8 greedy를 공동 primary mode로 둔다.
5. 두 mode의 paired median point reduction이 각각 10% 이상이고 prompt bootstrap lower가 양수일
   때만 multi-seed quality+timing으로 확장한다.
6. parameter/checkpoint 증가는 그대로 공개하고 memory 개선은 주장하지 않는다.

Actual preflight가 실패하면 update geometry의 quality method 가능성은 별도로 남더라도, 사용자가 정한
핵심 가치 기준인 inference-efficiency paper branch는 중단하거나 architecture를 다시 설계한다. 통과하면
recipe를 새 model seeds에 고정해 quality를 재현하고, 독립 timing sessions와 새 sealed final quality로
확인한다.

## 현재 주장 경계

- 말할 수 있음: fresh one-seed에서 geometry가 두 강한 8K control보다 낮은 BPB를 냈고, 2K보다도
  낮았으며, 같은 raw train stream의 optimizer wall time이 줄었다.
- 말할 수 없음: 실제 inference가 빨라졌다, multi-seed로 재현된다, 새로운 optimizer 원리다,
  memory-efficient하다, 다른 언어·hardware·모델 크기에 일반화된다.

다음 actual preflight의 양성 결과가 나와야 첫 번째 inference-efficiency evidence가 생긴다.
