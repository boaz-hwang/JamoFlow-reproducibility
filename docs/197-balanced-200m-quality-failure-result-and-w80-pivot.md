# Balanced 200M quality failure result and W80 pivot

> 작성일: 2026-08-16
>
> 상태: W72 결과와 독립 checkpoint replay가 확정된 뒤, W80 학습 전에 작성
>
> 후속 결과: [W80 quality and actual-inference result](./199-balanced-200m-w80-quality-and-actual-result.md)

## 1. 확정된 결과

188,639,808-parameter byte-latent model을 동일 initial state, 동일 127,991,808
training bytes, 동일 example order와 optimizer로 학습했다. C86과 W72 사이에서 달랐던 것은
patch matrix뿐이다.

- C86 calibration BPB: `1.4411260692502428`
- W72 calibration BPB: `1.4653265471956850`
- W72 minus C86: **`+0.0242004779454421 BPB`**
- 사전 고정 noninferiority margin: `+0.010 BPB`
- 판정: **quality fail**

두 checkpoint의 전체 15,625-sequence calibration forward를 별도 verifier가 다시 수행해 저장
NLL과 bitwise float32 일치를 확인했다. 따라서 이 차이는 summary 계산이나 artifact stitching의
오류가 아니다. 사전 계약에 따라 actual incremental inference timing은 실행하지 않았다.

W72는 C86보다 training source-byte throughput이 약 `7.45%`, full calibration evaluator
throughput이 약 `10.26%` 높았다. 그러나 이는 실제 cached autoregressive inference 결과가
아니며, 품질 gate 실패를 상쇄하지 않는다.

## 2. 무엇을 알게 되었는가

Random-weight graph에서 W72--C86 controlled latency 차이는 model size와 함께 커졌다. 하지만
실제 188.6M weights를 0.6785 raw byte/parameter만큼 학습한 결과 W72는 품질을 보존하지
못했다. 따라서 다음 명제는 기각한다.

> 동일 W72 policy를 더 큰 graph에 적용하면 compact model의 약 2.5% actual gain이 자동으로
> 더 큰 matched-quality gain으로 바뀐다.

Post-outcome structural analysis에서 W72 손실은 소수의 긴 patch나 극단 sequence에 집중되지
않았다. W72가 더 나쁜 sequence는 전체의 약 `75.4%`였고, whitespace가 적은 구간에서 더
나빴지만 whitespace가 많은 구간도 `+0.010 BPB`를 넘었다. 이는 rare outlier보다 전반적인
global patch-token density 감소와 severe undertraining을 우선 의심하게 한다. 다만 현재 한
seed 자료로 두 원인을 인과적으로 분리할 수는 없다.

## 3. 수정된 연구 질문

이제 모델 크기 하나를 독립변수로 주장하지 않는다. 다음 질문으로 좁힌다.

> 188.6M graph에서 whitespace-aware policy의 patch count를 72에서 80으로 완화하면, C86
> 대비 calibration quality를 보존하면서 compact matched-quality 결과보다 큰 actual
> controlled/free-running 효율을 얻을 수 있는가?

W80은 W72가 C86 대비 제거한 14 patches 중 6개만 제거한다. 관측 W72 quality delta를
removed-patch count에 단순 선형 보간하면 W80은 `+0.01037 BPB`로 정확히 gate 경계에
놓인다. 이 값은 실험 설계 heuristic이지 quality 예측치가 아니다. Random-weight 188.6M
sensitivity를 같은 비율로 축소한 headroom은 약 `3.1%`로, 기존 compact controlled
`2.628%`와 free-running `2.531%`보다 약간 높다. 따라서 W80은 품질과 실제 효율 사이의
경계를 가장 적은 추가 학습으로 판별하는 단일 후보이다.

## 4. 왜 W82/W84를 동시에 탐색하지 않는가

여러 후보를 학습하고 calibration 결과가 좋은 후보만 timing하면 선택 편향이 생긴다. W80만
먼저 고정한다. W80이 quality gate를 실패하면 W82/W84를 자동으로 이어서 학습하지 않는다.
W82/W84는 품질 회복 가능성은 더 높지만, patch reduction이 각각 4/86과 2/86에 불과해 기존
compact 2.5%를 실질적으로 넘어설 headroom이 작다. W80 실패 시 이 density-rescue 경로는
종료하고, 더 많은 clean training data를 양쪽 baseline에 동일하게 적용하는 별도 연구로
전환한다.

## 5. 해석 한계

- W80 성공은 model size의 순수 인과 효과가 아니다. model size와 policy density가 함께 달라진다.
- W80도 one-seed, 0.68 byte/parameter screen이다. 충분히 학습된 large LLM claim이 아니다.
- compact 2.5%를 point estimate로 넘는 것과 통계적으로 넘는 것은 다르다. 후자는 W80의
  actual bootstrap lower bound가 compact point estimate를 넘을 때만 인정한다.
- quality를 통과하지 않으면 training/evaluator throughput을 actual inference 개선으로 바꾸어
  표현하지 않는다.
