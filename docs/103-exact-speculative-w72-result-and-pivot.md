# Exact speculative W72 결과와 비-speculative architecture 전환

> 작성일: 2026-08-13
>
> 상태: **actual improvement 관측, 사전 gate 실패, multi-byte branch 종료**
>
> authoritative result: `results/speculative-w72-preflight-v1/summary.json`

## 1. 결론

Frozen W72 seed 1729와 frozen generic independent head seed 20260813을 실제로 결합한
exact speculative runtime은 baseline W72 greedy output을 128/128 prompts에서 byte-for-byte
재현하면서 E2E wall time을 줄였다.

- baseline AR median: 364.942 ms
- speculative median: 328.510 ms
- reduction: **9.983%**
- prompt-bootstrap 95% interval: **[7.579%, 11.695%]**
- positive prompts: **110/128**

그러나 사전 gate는 point ≥20%, lower ≥10%, positive ≥96/128이었다. Correctness와
prompt direction만 통과했고 point/lower gate는 실패했다. 따라서 summary status는
`multi_byte_branch_stopped`이며, 이 결과를 보고 head, retry, activation 또는 threshold를
더 조정하지 않는다.

## 2. Exactness와 실제 포함 범위

Untimed oracle에서 128개 output과 final cache diagnostics가 모두 exact였다. Timed 3회
반복도 같은 canonical output을 재현했다. Timer에는 다음이 모두 들어간다.

- fresh runtime와 128-byte parallel prefill
- frozen 41,728-parameter head forward
- same-lead valid Hangul proposal restriction
- 3-byte target block
- first mismatch 뒤 2-byte third retry
- encoder/decoder/global cache crop
- open-patch state와 W72 selector rollback
- target correction, verifier bonus, strict UTF-8 masks와 stop logic
- Python device-to-host scalar readback 및 final MPS synchronization

따라서 앞선 perfect-draft/isolated-head projection과 달리 실제 generation cost다. 다만
calibration-only single model/head seed이므로 final publication efficiency evidence는 아니다.

## 3. Mechanism

128 prompts에서 총 16,497 bytes를 생성했다.

| diagnostic | count/rate |
|---|---:|
| baseline target calls | 16,369 |
| speculative sequential calls | 4,990 |
| target block calls | 7,576 |
| total target invocations | 12,566 |
| invocation reduction | 23.233% |
| primary head calls | 4,861 |
| first continuation acceptance | 44.147% |
| complete pair acceptance | 25.098% |
| retry blocks | 2,715 |
| retry third acceptance | 16.943% |
| correction bytes | 5,896 |
| bonus bytes | 1,657 |

Target invocation은 23.2% 줄었지만 wall time은 10.0%만 줄었다. 남은 차이는 head/argmax
readback, block/crop 제어, retry의 낮은 acceptance, 그리고 block이 ordinary call보다
비싼 점이다.

## 4. 앞선 cost model의 최종 교정

Perfect-draft v2의 fixed-head projection 42.730%는 **모든 output bytes가 proposal cycle의
amortization을 받는 것처럼 해석하면 안 된다**. 실제로는 Hangul lead에서만 primary
proposal을 시작하고, correction/continuation/non-Hangul pending bytes 상당수는 sequential
path에 남는다. Rollback과 retry도 target work를 추가한다.

따라서 systems evidence의 권위 순서는 다음과 같다.

1. perfect block result: target kernel이 기술적으로 빠르고 exact하다는 upper bound
2. acceptance result: proposal이 맞는 비율
3. **현재 exact E2E result: 두 요소와 모든 overhead를 합친 실제 효과**

논문이나 README에서 42.7%를 예상 actual speedup으로 쓰지 않는다. 허가되는 수치는
single-seed calibration에서의 9.983%뿐이다.

## 5. 연구 방향 수정

이 결과는 두 가지를 동시에 말한다.

1. Byte-local block execution은 실제로 큰 여지가 있다.
2. 그 여지를 learned future-byte acceptance로 회수하는 방식은 현재 gate를 넘지 못했다.

그러므로 multi-byte/draft 분기는 종료한다. 다음에는 future byte를 맞힐 필요가 없는
**non-speculative local/global compute reallocation**을 검토한다. 기존 profiler에서 반복되는
per-byte local path가 주된 병목으로 관측됐으므로, local encoder/decoder depth·width를
줄이고 같은 parameter budget을 global trunk로 옮기는 후보를 먼저 random-weight actual
runtime에서 평가한다.

단, BLT 원 논문 자체가 longer patch의 절약분을 global model capacity로 재배치하는 scaling
axis를 이미 제안한다. 따라서 정적 local-to-global reallocation은 새 논문의 novelty가 아니라
현재 compact geometry가 병목을 과도하게 만든 것인지 확인하는 **generic feasibility control**로
취급한다.

다음 단계의 순서는 다음과 같다.

1. 기존 19.596M W72와 parameter-matched geometry 후보를 열거한다.
2. Frozen random-weight geometry로 per-byte/patch-boundary/E2E actual latency를 측정한다.
3. 최소 20% latency potential이 있는 후보만 Korean train/calibration에서 한 seed 학습한다.
4. Calibration BPB noninferiority와 exact actual inference가 둘 다 통과하면 정적 geometry를
   mandatory generic control로 고정한다.
5. 최종 method 후보는 prefix-causal UTF-8/Hangul state에 따라 쉬운 continuation 위치의
   local compute를 조건부로 줄이고, 정적 control보다 추가 E2E 이득을 내야 한다.
6. 이 candidate가 generic UTF-8 state control까지 이길 때만 seed 확장과 새 disjoint
   held-out evaluation을 연다.

이 후보는 speculative branch의 threshold를 우회하는 수정이 아니다. Future-byte head와
rollback을 전혀 쓰지 않고, profiler가 독립적으로 확인한 model geometry 병목을 직접
다루는 별도 가설이다.

## 6. 논문 가치의 현재 판정

현재까지 얻은 9.983% actual improvement는 사용자 기준의 중요한 positive signal이지만,
single-seed calibration exploratory result라 그 자체로 논문 최종 claim은 아니다. 또한
Hangul-specific heads가 generic head보다 낮았으므로 한국어 조합규칙의 우월성도 지지하지
않는다.

반면 다음 결과들은 연구적으로 단단하다.

- matched-quality W72의 실제 2.5--2.6% 소폭 개선과 10% primary failure
- local byte path의 84% 병목과 patch scheduling ceiling
- Hangul-specific draft의 generic control 대비 실패
- exact block kernel의 61% target-call reduction
- 모든 overhead를 포함한 exact speculative generation의 9.983% 개선/20% gate 실패

최종 paper는 새 conditional-local-compute candidate가 matched quality에서 정적
local-to-global control과 generic UTF-8 state control을 넘어 실제 개선을 재현하면 positive
architecture paper로 확장한다. 정적 geometry 조정만 양성이면 BLT scaling-axis의 재현으로
한정하고 독자적 novelty로 주장하지 않는다. 모두 실패하면 위 결과를 과장하지 않고
orthography-guided byte-LM efficiency의 한계와 measurement methodology를 다룬 negative
systems study로 정리한다.
