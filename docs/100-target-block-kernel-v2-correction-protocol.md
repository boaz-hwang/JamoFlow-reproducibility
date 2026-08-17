# Target block-kernel v2 inference-mode correction protocol

> 봉인일: 2026-08-13
>
> 상태: **v2 timing 전 correction seal**

## 1. 수정 범위

v1은 model execution을 `torch.inference_mode()`로 감싸지 않아 autograd graph가 생성됐다.
그 수치는 `results/target-block-kernel-v1/invalidation.json`으로 무효화했다. v2는 이 한
가지 workload 오류만 고친다.

- `_measure_micro`와 `_measure_whole` 진입 시 inference mode를 assert한다.
- 실제 두 측정과 whole-path correctness oracle 전부를 하나의
  `torch.inference_mode()` scope 안에서 실행한다.
- summary가 `runtime.torch_inference_mode=true`를 기록한다.
- v1 artifact는 수정·삭제하지 않고 v2 namespace에 새 결과를 쓴다.

## 2. 바꾸지 않은 항목

다음은 v1 seal과 동일하다.

- W72 seed 1729 checkpoint와 512-byte policy horizon
- 999,936-byte calibration stream
- micro strata별 32 cases, 5 repetitions
- perfect-Hangul whole 16 cases, 3 repetitions
- exact parity-balanced mode order
- full-logit/argmax/cache correctness contract
- 10,000 case bootstrap, seed 20260829
- point/lower-bound gate: micro 30%/20%, perfect whole 20%/10%, fixed-head
  projection 20%/10%
- independent head seed 20260813의 기존 acceptance와 latency
- 하나라도 실패하면 head를 retune하지 않고 multi-byte branch 종료

Case selection도 같은 함수와 domain-separated rank를 사용하므로 v1/v2 case artifact는
byte-for-byte 동일해야 한다. 이 equality는 결과 검증에서 확인한다.

## 3. Claim 경계

v2도 perfect-draft exploratory upper bound다. 통과는 calibration-only rollback prototype만
허가하며 actual speculative speedup, quality, 한국어-specific novelty를 뜻하지 않는다.
실제 head, mismatch correction, cache rollback, UTF-8 mask와 stop logic이 포함된 W72 E2E가
기준을 통과하기 전에는 positive efficiency claim을 만들지 않는다.
