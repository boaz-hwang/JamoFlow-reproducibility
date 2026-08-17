# EXAONE baseline-only resource calibration protocol

> 작성일: 2026-08-15
>
> 상태: **ordinary baseline timing 전에 고정할 resource-only protocol**

## 목적

7.8B EXAONE actual comparison을 Mac 한 대에서 재현성 있게 끝낼 수 있도록 session 수와 inner
repetition 수를 정한다. 이 단계는 ordinary greedy만 실행한다. Retrieval table을 load하지 않고,
candidate proposal, acceptance, target-call reduction, candidate latency를 계산하거나 출력하지 않는다.

## 고정 workload

- data dependency: independently reconstructed EXAONE retrieval data/case verification
- cases: sealed rank order의 8 warmup + 64 measured cases 전부
- prompt: 각 case의 128 EXAONE tokens
- generation: fixed 128 greedy tokens
- repetition: case당 1회
- model/file/case load: timer 밖
- prompt text의 tokenizer encode: timer 안
- fresh KV cache, parallel prompt prefill, 128 cached greedy decode steps: timer 안
- full prompt+output decode 및 final MLX synchronize: timer 안
- decode 후 re-encode exactness: timer 밖 correctness gate

Raw text나 output token IDs는 tracked result에 넣지 않는다. Ignored NPZ에는 72개 output IDs, token/text
hash, elapsed nanoseconds, exact target-call counters를 보존한다.

## 자원 gate

- 보수적 observed working set ≤ `max_recommended_working_set_size`의 75%
  - model load 직후와 전체 generation 뒤의 `active + cache`
  - reset 이후 MLX peak active memory
  - 프로세스 peak RSS
  - 위 네 값의 최댓값을 gate에 사용
- 모든 prompt token count = 128
- 모든 output token count = 128
- 모든 baseline generation target calls = 128, prefill calls = 1
- 모든 prompt/output tokenizer round trip exact
- finite positive latency 72개

## actual campaign schedule의 기계적 선택

Actual comparison의 candidate가 baseline보다 최대 2배 느릴 수 있다고 보수적으로 가정한다. Fresh
session마다 model load 실측값과 120초 고정 overhead를 포함한다. 다음 순서에서 projected total이 8시간
이하인 첫 조합을 고른다.

1. 5 sessions × 3 inner repetitions
2. 5 sessions × 2 inner repetitions
3. 5 sessions × 1 inner repetition
4. 3 sessions × 1 inner repetition

어느 조합도 통과하지 못하면 8B actual branch는 resource-infeasible로 중단한다. Baseline latency를 보고
table 크기, proposal cap, case set, primary metric, 효율 threshold는 바꾸지 않는다.

## claim boundary

이 결과는 7.8B ordinary greedy의 단일-hardware resource feasibility와 actual campaign 크기만 정한다.
Retrieval candidate가 빠르다거나, acceptance가 충분하다거나, Korean-specific 효율이 있다는 근거가
아니다. Actual candidate comparison은 이 결과와 schedule을 commit한 뒤 별도 plan으로 봉인한다.
