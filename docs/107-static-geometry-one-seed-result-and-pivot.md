# Trained static-geometry one-seed result and pivot

> 작성일: 2026-08-14
>
> protocol commit: `673fc4bfd96d75ed955679ca494de8021a661ccf`
>
> protocol: `jamoflow-static-geometry-one-seed-v1`
>
> result: `results/static-geometry-one-seed-v1/summary.json`

## 1. 판정

`thin160_e1_d1_g384x9`는 실제 Apple MPS 추론 속도 gate를 넉넉하게 통과했지만 Korean
calibration quality noninferiority를 명확하게 실패했다. 따라서 사전 규칙대로 이 정적
geometry의 나머지 네 seed 학습은 수행하지 않는다.

| 항목 | 관측값 | 사전 gate | 판정 |
|---|---:|---:|---|
| calibration BPB 차이, candidate - W72 | **+0.095601** | <= +0.010 | fail |
| document bootstrap one-sided 95% upper | **+0.096740** | <= +0.010 | fail |
| eligible calibration windows | 97.536% | >= 95% | pass |
| controlled E2E 감소 | **24.307%** | point >= 15%, lower >= 10% | pass |
| controlled prompt bootstrap 95% | **[23.770%, 24.630%]** | lower >= 10% | pass |
| strict free-running E2E 감소 | **22.841%** | point >= 15%, lower >= 10% | pass |
| free-running prompt bootstrap 95% | **[22.370%, 23.164%]** | lower >= 10% | pass |

Candidate BPB는 1.733536, 기존 W72 seed-1729 baseline은 1.637935였다. 두 timing mode
모두 64/64 prompt에서 candidate가 빨랐다. Sequential/parallel incremental 비교의 1,024개
argmax가 두 역할 모두 exact했고 boundary/cache trace, strict UTF-8 output gate도 전부
통과했다.

이 결과는 경계 사례가 아니다. 품질 평균 차이는 margin의 약 9.56배이고 document
bootstrap 분포의 중앙 90%도 [0.094149, 0.096740] BPB에 있다. 한 seed screen이라는
한계는 유지하지만, 이 exact candidate에 네 seed를 더 쓰지 않는 kill decision에는 충분히
강한 결과다.

## 2. 결과가 식별한 것과 식별하지 못한 것

첫째, 반복 local path를 직접 줄이면 실제 latency가 크게 내려간다는 systems 가설은
학습된 checkpoint에서도 재현됐다. Random-weight preflight의 24.417%와 학습 후 controlled
24.307%가 매우 가깝고, free-running에서도 22.841%가 유지됐다. 기존 W72 boundary
schedule의 2.5--2.6%와 exact speculation의 9.983%보다 큰 효과다.

둘째, local encoder/decoder capacity를 정적으로 제거하고 parameter를 global layer로
옮겨도 품질은 복구되지 않았다. Parameter 수는 거의 같지만 local width 192→160,
encoder/decoder depth 2/2→1/1로 줄인 손실을 global depth 8→9가 보상하지 못했다. 이는
compact Korean byte LM에서 per-byte local representation이 단순한 systems overhead만은
아니라는 직접 증거다.

셋째, 이 한 실험만으로 width, encoder depth, decoder depth 중 어느 축이 품질 손실의
원인인지 분리할 수는 없다. 세 축과 global FFN/depth가 함께 바뀌었다. 또한 모든 정적
local/global 재배치가 실패했다고 일반화할 수도 없다. 판정 범위는 사전 선택된
`thin160_e1_d1_g384x9`, 이 데이터 예산, 이 seed다.

넷째, 속도와 품질을 서로 다른 결과에서 이어 붙이지 않는다. 이 candidate는 빨랐지만
matched quality가 아니므로 사용자 기준의 유효한 효율 개선이 아니다. 논문의 positive
efficiency 근거로 사용할 수 없고, speed--quality trade-off를 보여 주는 negative control로만
남긴다.

## 3. 연구 방향 수정

정적 geometry branch는 종료한다. 다중-seed static replication, 이 geometry를 공통
backbone으로 삼는 확장, 사후 margin 완화는 하지 않는다.

다만 연구 전체를 종료할 근거는 아니다. 이번 결과는 다음 두 사실을 동시에 보였다.

1. local path를 줄일 때 얻을 수 있는 실제 E2E 이득은 20%를 넘는다.
2. 모든 위치에서 local capacity를 줄이면 Korean BPB가 크게 무너진다.

따라서 다음 가설은 **local capacity 제거 여부를 위치별로 달리해야 한다**는 것이다.
쉬운 orthographic continuation에서는 싼 path를 쓰되, 정보가 필요한 위치에서는 original
W72 local width/depth를 유지한다. 이는 정적 candidate의 pass를 근거로 자동 승인된 확장이
아니다. 실제 result의 `conditional_local_compute_research_authorized=false`를 그대로
보존하고, `docs/106` §5가 남긴 별도 가설에 대해 새 prospective protocol을 만들어야 한다.

다음 단계는 곧바로 비싼 다중-seed 학습이 아니라 아래 순서로 제한한다.

1. incremental BLT에서 per-position local-depth gating이 cache semantics와 실제 MPS
   kernel work를 줄일 수 있는지 구현·correctness preflight로 확인한다.
2. original W72 width/depth를 보존한 상태에서 deterministic generic UTF-8 state와
   Hangul-specific state가 각각 어느 위치를 easy로 분류하는지 calibration-only로 고정한다.
3. 같은 route rate와 같은 parameter/cost를 갖는 generic control과 Hangul route를 함께
   설계한다. Hangul route만 유리한 예산이나 별도 learned router를 주지 않는다.
4. random/frozen feasibility가 실제 latency potential을 보일 때에만 한 seed를 처음부터
   conditional training한다.
5. 한 seed에서 BPB noninferiority와 controlled/free actual latency가 모두 통과한 뒤에만
   multi-seed, held-out final, downstream, scale 및 CUDA replication을 연다.

## 4. Claim 경계

안전한 결론은 다음과 같다.

> Parameter-matched static local-to-global reallocation은 Korean calibration BPB를 0.0956
> 악화시켜 matched-quality 효율에 실패했지만, 학습된 checkpoint의 actual end-to-end
> generation을 22.8--24.3% 줄였다. 따라서 compact Korean BLT의 local path는 큰 latency
> 병목인 동시에 품질에 필요한 capacity이며, 후속 연구는 전역 thinning이 아니라
> position-conditional local computation을 검증해야 한다.

이 문장은 conditional method의 성공, 한국어 고유 이득, 일반 hardware 속도 또는 정적
geometry novelty를 주장하지 않는다. 다음 positive claim은 generic UTF-8 control을 같은
cost에서 이기고 matched quality의 실제 generation 개선이 여러 seed에서 재현될 때만
가능하다.
