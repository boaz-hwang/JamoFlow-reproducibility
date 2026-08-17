# Static local/global geometry feasibility control

> 작성일: 2026-08-13
>
> 상태: **actual timing 전 봉인 예정**
>
> protocol: `jamoflow-static-geometry-preflight-v1`

## 1. 질문

W72와 exact speculative runtime은 실제 latency를 각각 약 2.5--2.6%, 9.983% 줄였지만
사전 gate를 실패했다. Component profile은 모든 output byte에서 반복되는 local path가 주된
병목임을 보였다. 다음 비싼 학습 전에 아래의 더 싼 falsification을 수행한다.

> 현재 19.596M W72의 local encoder/decoder를 얇게 하고 거의 같은 parameter budget을
> global trunk로 옮기면, 품질과 무관한 geometry 자체에 최소 20%의 실제 E2E latency
> potential이 존재하는가?

이 실험은 random-weight systems preflight다. BPB, generation quality 또는 최종 효율을
측정하지 않는다.

## 2. Novelty 경계

[BLT](https://arxiv.org/abs/2412.09871)는 longer patches에서 절약한 compute를 더 큰 global
model로 옮기는 scaling axis를 이미 제안했다. [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)는
token 위치별 conditional depth allocation을 이미 제안했다. 따라서 이 단계의 정적
local-to-global 재배치는 다음 후보를 평가하기 위한 **generic control**이지 새 방법이 아니다.

이 control이 통과한 뒤에도 최종 novelty가 성립하려면 다음을 모두 보여야 한다.

1. learned router가 아니라 prefix에서 결정되는 UTF-8/Hangul state를 사용한다.
2. BLT의 반복 local-byte compute를 조건부로 줄인다.
3. parameter/cost-matched generic UTF-8 state control보다 Hangul-specific route가 낫다.
4. matched quality에서 actual end-to-end generation이 개선된다.

## 3. 고정 geometry

모든 모델은 W72, sequence horizon 512, global position capacity 1,032, float32, seed
20,260,813을 사용한다. Global width는 384로 유지하고 layer/FFN을 조정해 parameter count를
baseline의 0.25% 안에 둔다.

| 역할 | local width | encoder/decoder layers | global layers/FFN | params | counted dense FLOPs | baseline 대비 |
|---|---:|---:|---:|---:|---:|---:|
| baseline W72 | 192 | 2 / 2 | 8 / 1152 | 19,596,096 | 5,640,155,136 | 0% |
| thin128 E1/D2 | 128 | 1 / 2 | 9 / 1168 | 19,605,888 | 3,984,926,208 | -29.347% |
| thin160 E1/D1 | 160 | 1 / 1 | 9 / 1128 | 19,571,872 | 3,889,040,896 | -31.047% |
| thin128 E1/D1 | 128 | 1 / 1 | 9 / 1192 | 19,575,680 | 3,587,538,432 | -36.393% |

Candidate order는 `thin128 E1/D2 -> thin160 E1/D1 -> thin128 E1/D1`로 고정한다. 이는
관측 속도가 가장 큰 후보를 사후 선택하지 않고, BLT의 decoder depth를 더 많이 보존하는
후보부터 학습하기 위한 quality-conservative order다.

## 4. Calibration-only workload

- source: 기존 HPLT Korean calibration stream 999,936 bytes
- prompts: model-free bottom-hash Hangul-heavy 128 prompts 중 고정된 첫 32개
- prompt: 128 bytes
- feedback: 같은 calibration source에서 prompt 직후의 고정 127 bytes
- repetitions: prompt/geometry마다 3회
- role order: prompt x repetition 안에서 4개 geometry의 위치를 cyclic balance
- timing: fresh runtime construction, parallel prefill, 127 incremental consumes, final MPS sync
- warmup: geometry마다 첫 4 prompts
- AC power 및 shared publication MPS lock 필수

Random model의 greedy output은 geometry별로 달라질 수 있으므로 timing에 쓰지 않는다. 모든
geometry가 정확히 같은 observed bytes와 W72 boundary schedule을 처리하는 controlled replay만
비교한다.

각 geometry의 첫 case에서는 sequential prefill/consume과 parallel-prefill runtime의 128개
logit 위치, argmax, boundary trace와 cache diagnostics를 독립 비교한다. Reference-side
`atol + rtol * abs(reference)` (`2e-5`, `1e-4`)로 정규화한 최대 오차가 1 이하여야 한다.

## 5. 사전 gate와 선택

반복은 표본으로 세지 않고 prompt 안에서 median으로 접는다. Prompt를 10,000회 bootstrap하며
seed는 20,260,901로 고정한다. Candidate는 다음을 모두 만족해야 한다.

1. parameter-count relative difference <= 0.25%
2. counted dense-matmul reduction >= 20%
3. sequential/parallel correctness 전부 통과
4. actual E2E point reduction >= 20%
5. prompt-bootstrap 95% lower bound >= 15%
6. positive prompts >= 24/32

여러 후보가 통과하면 고정 order의 첫 후보만 선택한다. 통과는 그 geometry의 Korean
train/calibration **한 seed**만 허가한다. 어떤 후보도 통과하지 못하면 threshold나 geometry를
결과 후 조정하지 않고 static branch를 종료한다.

## 6. 다음 단계

선택된 정적 control은 동일 Korean byte budget에서 baseline W72와 한 seed로 학습한다.
Calibration BPB noninferiority와 actual latency를 함께 통과해야 다음 orthographic-state
candidate의 필수 대조군이 된다. 정적 control만 성공한 결과는 BLT scaling-axis의 compact
재현으로 보고하며 JamoFlow의 새 기법으로 주장하지 않는다.

최종 method 후보는 original W72, 이 정적 control, generic UTF-8 state-conditioned control,
Hangul-specific state-conditioned candidate의 네 축을 분리한다. 한국어 고유 주장은 마지막
후보가 같은-cost generic control을 이겼을 때만 허용한다.
