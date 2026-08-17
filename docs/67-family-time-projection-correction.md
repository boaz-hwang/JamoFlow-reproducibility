# Family time-projection correction

> 작성일: 2026-08-12
> 상태: **family-aware preflight 구현·실행 전 결과맹 교정**
> 상위 protocol: [family-aware scale feasibility](./64-family-aware-scale-feasibility-correction.md)

## 1. 발견한 두 번째 self-attestation

Parameter identity를 봉인한 뒤에도 `FamilyScaleFeasibility`는 worker가 보고한
`projected_hours_per_model`을 그대로 gate에 사용했다. 이 값이 실제 steady-step
시간이나 해당 step이 처리한 raw source byte 수에서 재구성되는지는 타입이
검증하지 않았다. Worker report의 projected time을 임의로 작게 쓰면 family당
12시간과 campaign 120시간 gate를 모두 우회할 수 있었다.

Family-aware worker와 report는 아직 존재하지 않으며 어떤 runtime time도 측정되지
않았다. 따라서 이 변경 역시 관측 결과와 무관하다.

## 2. 파생값으로만 허용

Final family 결과는 각 실행 component마다 다음 관측량을 받는다.

- 세 steady train step의 median wall seconds
- 그 step의 원래 clean publication source에 포함된 raw UTF-8 byte 수

Model당 projected time은 코드가 component별 total source bytes에서 다음처럼 계산한다.

```text
component_steps = ceil(component_total_raw_bytes / observed_raw_source_bytes_per_step)
component_hours = median_step_seconds * component_steps / 3600

T_family = T_main_train
         + I[entropy reference] * (T_router_train + T_router_score)
```

Worker가 별도의 projected hour를 정답처럼 전달할 수 없다. BPE token 수, padding과
output-head 비용은 median step time에 들어가고, 서로 다른 tokenization을
`tokens/step`으로 잘못 환산하지 않도록 campaign budget의 공통 분모는 source
UTF-8 bytes로 유지한다.

Main과 router train은 256M clean train bytes를 각각 한 번 처리한다. Router score는
offline one-pass patch-cache contract로 실행하며 최소한 같은 256M train stream 전체를
덮어야 한다. Calibration/test scoring과 threshold/cache 구축은 공통 stream manifest가
봉인한 정확한 추가 byte 수를 별도 기록한다. Core 120시간 표가 train-stream scoring만
포함한다면 이 고정 overhead는 별도 운영 예산으로 공개하며 누락을 전체 campaign
시간으로 오표기하지 않는다. Main train step 안에 router scoring을 online으로 넣는
다른 구현은 별도 score 시간을 다시 더할 수 없고, execution contract 자체를 새로
사전등록해야 한다.

## 3. Gate와 artifact 요구

- raw bytes/step은 양수여야 한다.
- median step time은 finite positive여야 한다.
- report는 raw bytes/step, projected train bytes, 파생 step 수와 시간을 모두
  기록한다.
- E/EC report는 router train/score 각각의 완료·finite 3-step median, exact total
  source bytes, config/workload hash와 stage별 memory high-water를 기록한다.
- 실제 cached incremental preflight에서 router observed/scored/cached bytes가 기대값과
  같고 router forward count가 양수여야 한다.
- 기존 1.20 safety factor, family당 12시간, 세 seed와 전체 120시간 조건은
  파생 시간에만 적용한다.
- 향후 runner summarizer는 source batch lineage와 실제 selected raw byte count를
  다시 만들어 worker의 raw-byte 분모도 검증해야 한다.

같은 step time에서 raw bytes/step을 절반으로 바꾸면 projected time은 정확히
두 배가 된다. Zero-byte 또는 nonfinite/zero-time family는 통과하지 못한다.
