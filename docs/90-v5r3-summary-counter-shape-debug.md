# Actual-inference v5r3 summary counter-shape 디버그 기록

> 최초 관측: 2026-08-13
>
> 상태: 조사 중
>
> 결과 노출: summarizer가 session artifact schema 검증 중 중단했으며 summary는
> publish되지 않았다. Aggregate latency와 gate 결과는 아직 계산·열람하지 않았다.

## 문제 정의

- 기대: 다섯 v5r3 session과 열 개 memory receipt를 재검증해 immutable summary를
  생성한다.
- 실제: 첫 session의 runtime-counter 검증에서
  `ValueError: emitted-output counter shape differs`로 fail-closed했다.
- 심각도: publication summary blocker. 기존 timing/output receipt의 correctness나
  latency 값을 바꾸는 근거는 아니다.

## 재현

```bash
PYTHONPATH=src .venv/bin/python scripts/summarize_inference_actual_v5.py
```

## 가설

- [x] H1 기각: producer의 timing counter는 의도한 repetition 축을 포함하며 모든
  session에서 `(5 seeds, 64 prompts, 5 repetitions)`이다.
- [x] H2 확인: summarizer가 올바른 counter key를 읽었지만, per-seed 2차원 계약인
  원 validator에 전체 3차원 session array를 전달했다.
- [x] H3 기각: producer는 각 seed가 끝날 때 `(64, 5)` slice를 같은 validator로
  이미 검사한다. protocol revision이나 artifact schema drift가 아니다.

## 조사 로그

| 시각 | 작업 | 결과 | 다음 단계 |
|---|---|---|---|
| 최초 실행 | 최종 summarizer 실행 | `emitted-output counter shape differs`; summary 미생성 | producer/consumer shape를 값 비노출로 대조 |
| shape 관측 | 다섯 timing NPZ의 key·shape·dtype만 확인 | 모든 counter와 timing array가 일관된 `(5,64,5)`; 값/aggregate 미열람 | producer의 validator 호출 위치 확인 |
| call-site 대조 | runner와 summarizer의 validator 호출 비교 | runner는 `array[seed_index]`, summarizer는 전체 array 전달 | summary-only adapter 설계 |

## Root cause

`validate_runtime_counter_arrays()`의 의도된 입력은 `(prompt, repetition)` 2차원이다.
runner는 seed 하나를 끝낼 때 정확히 `timing_array[seed_index]`를 넘겨 모든 session
artifact를 생성 전에 검증했다. 최종 summarizer만 seed 축을 자르지 않고
`(seed, prompt, repetition)` 전체를 넘겼다. 따라서 counter 값이나 timing evidence의
오류가 아니라 consumer call-site의 seed-axis 누락이다.

## Solution

plan-bound 원본 producer, validator, summarizer와 모든 session artifact를 변경하지
않는다. 별도 summary-only correction은 정확한 `(5,64,5)`만 받아 고정 seed 순서로
원 validator를 다섯 번 호출한다. 통계, bootstrap, gate, latency array 및 memory
receipt는 바꾸지 않는다. Correction manifest는 원본 파일 hash, correction 구현/test
hash, failure와 `latency values not inspected` 선언을 결과 계산 전에 봉인한다.

## Prevention

- full `(5,64,5)` 입력이 정확히 다섯 `(64,5)` 호출로 분해되는 regression test
- seed count 또는 repetition shape가 바뀌면 base validator 전에 거부하는 test
- 한 counter만 shape가 회전해도 거부하는 test
- 최종 summary에 tracked correction artifact identity를 포함하고 canonical hash로
  함께 봉인
