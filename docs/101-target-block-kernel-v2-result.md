# Target block-kernel v2 결과

> 작성일: 2026-08-13
>
> 상태: **exact speculative rollback prototype 허가**
>
> authoritative result: `results/target-block-kernel-v2/summary.json`

## 1. 결론

Inference-mode correction 뒤에도 perfect-draft target block kernel은 모든 사전 gate를 큰
폭으로 통과했다.

| gate | point reduction | 95% case-bootstrap interval | threshold point/lower |
|---|---:|---:|---:|
| empirical-boundary-weighted 3-byte target call | **61.014%** | [58.857%, 62.094%] | 30% / 20% |
| perfect-Hangul whole path | **44.044%** | [37.261%, 50.050%] | 20% / 10% |
| fixed independent head projection | **42.730%** | [40.298%, 43.766%] | 20% / 10% |

따라서 target-side local encoder/decoder를 세 byte에 대해 block으로 실행하는 것이 실제
MPS workload에서도 충분한 systems margin을 만든다는 가설은 지지됐다. 다음 단계는 이미
고정된 generic independent head를 실제로 실행하고 mismatch rollback/correction까지 넣는
calibration-only exact speculative runtime이다.

## 2. Patch boundary별 결과

| stratum | sequential 3 calls | block call | reduction |
|---|---:|---:|---:|
| block 안 새 patch 없음 | 6.731 ms | 2.079 ms | 69.114% |
| block 첫 byte에 새 patch 하나 | 11.493 ms | 5.474 ms | 52.374% |
| empirical mixture | 8.419 ms | 3.282 ms | 61.014% |

Whole cases의 Hangul block 1,185개 중 420개, 즉 35.443%가 새 W72 boundary를 만들었다.
Boundary block은 global update를 순차적으로 수행해야 해 이득이 작지만 여전히 절반 이상
줄었다. 이는 block kernel의 이득이 단순히 global patch call을 숨긴 결과가 아니라 공통
local byte path의 launch/call amortization에서 나온다는 profiler 해석과 일치한다.

## 3. Whole-path와 correctness

128-byte prompt prefill부터 약 255-byte continuation까지 포함한 perfect-Hangul oracle은

- sequential median: 717.014 ms
- block median: 401.214 ms
- reduction: 44.044%

였다. Draft head와 rollback이 빠진 이상적 상한선이므로 이 숫자를 실제 generation
speedup으로 쓰지 않는다.

Correctness oracle은 다음을 통과했다.

- logit/argmax comparisons: 5,409 positions
- cache/boundary diagnostic comparisons: 704
- maximum absolute logit error: `9.536743e-6`
- maximum normalized tolerance ratio: 0.0611, required ≤1
- every argmax and final cache trace exact

특히 whole continuation의 마지막 값만 본 것이 아니라 모든 block 내부 위치를 별도 untimed
oracle로 비교했다.

## 4. v1 correction 검증

v1과 v2 case artifacts는 모두 SHA-256
`e1e90398c344bab91933b714e6903a9b15abde60ac94553160c0d71d2f168b79`로
byte-for-byte 동일했다. 바뀐 것은 `torch.inference_mode()`뿐이다.

그 결과 sequential target cost는 v1의 6.697 ms/byte에서 v2의 2.806 ms/byte로 정상화됐다.
v1 수치를 폐기한 판단이 옳았음을 보여 준다. v1은 forensic artifact로만 보존하며 positive
claim에 사용하지 않는다.

## 5. Fixed-head projection

기존 independent head seed 20260813의 값은 변경하지 않았다.

- first continuation acceptance: 42.373%
- complete pair acceptance: 24.379%
- expected committed bytes: 2.667522
- isolated head latency: 1.004959 ms

Measured target block과 이 head cost를 단순 결합하면 committed byte당 1.607 ms이고,
sequential target 2.806 ms/byte보다 42.730% 낮다. 하지만 아직 다음 비용이 빠져 있다.

- speculative cache crop/rollback
- first/second mismatch branch
- target correction 및 bonus byte state 관리
- strict UTF-8 mask, stop condition, output bookkeeping
- proposal hidden capture와 실제 head integration

그러므로 이 projection은 구현을 허가하는 상한선이지 논문 결과가 아니다.

## 6. 다음 단계와 stop rule

다음 prototype은 target greedy output과 byte-for-byte 같아야 한다.

1. 현재 target logit에서 `b1`을 선택한다.
2. `b1`이 precomposed Hangul lead일 때 frozen independent head가 `d2,d3`를 제안한다.
3. target이 `[b1,d2,d3]`를 한 block으로 검증한다.
4. 첫 mismatch 위치까지 cache를 정확히 crop하고 target correction을 확정한다.
5. 둘 다 맞으면 verifier bonus byte까지 확정한다.
6. 실제 head, rollback, masks, stop logic을 모두 timer 안에 둔다.

Calibration preflight에서 exact output과 최소 20% point / 10% lower-bound E2E reduction을
통과할 때만 multi-seed 및 generic all-byte comparator 단계로 진행한다. 실패하면 block
kernel 자체의 성공을 systems diagnostic으로만 남기고 multi-byte branch를 종료한다.

통과하더라도 현재 evidence는 한국어-specific head 우월성을 보여 주지 않는다. Generic
independent head가 모든 Jamo/구성 head보다 강했으므로 최종 novelty는 Hangul-heavy
scalar-aligned activation이 same-cost generic all-byte MTP보다 추가 이득을 내는지에 달려
있다.
