# Actual-inference protocol integrity audit

> 작성일: 2026-08-11
> 상태: **actual-inference artifact 생성 전 교정**
> 목적: 실행기·요약기 drift와 source-document 의존성을 실측 전에 제거
> 후속 교정: protocol v3의 fixed raw-byte free running은 [valid-output actual-inference correction](./65-valid-output-actual-inference-correction.md)의 v4가 대체함. Document clustering, thermal, sentinel과 controlled off-by-one 교정은 그대로 유지함.

## 1. 감사 시점과 봉인 상태

감사 시점에 `runs/phase3-actual-inference`, `artifacts/phase3-actual-inference`, `results/phase3-actual-inference`는 모두 존재하지 않았다. S/E/EC initial family는 실행 중이었고 개별 quality 값은 열지 않았다. 따라서 이 교정은 관측된 latency나 quality에 맞춘 사후 선택이 아니라, future evidence의 estimand와 재구성 가능성을 바로잡는 변경이다. Training/model/data/policy module은 건드리지 않았다.

## 2. 발견한 문제

### 2.1 실행기와 요약기의 protocol drift

실행기는 prompt당 5 repetition과 document-contained selector를 사용하도록 바뀌었지만 요약기는 3 repetition과 이전 selector를 재구성했다. 이 상태에서는 정상 timing artifact도 요약 단계에서 shape 또는 case-context 불일치로 실패하고, 더 나쁘게 느슨한 검증을 추가하면 다른 표본을 같은 실험으로 오인할 수 있었다.

### 2.2 time-to-output 진단의 off-by-one

128-byte prompt의 prefill logit이 output byte 1을 이미 예측하므로 128-byte output에는 feedback forward가 127회 필요하다. Runtime이 실제 관측하는 byte 수는 `128 + 127 = 255`인데, 요약기의 bytes/global-patch 진단은 256으로 나눴다. Latency 원자료에는 영향을 주지 않지만 patch 효율 진단을 일관되게 과대평가하는 오류였다.

### 2.3 전원·열 상태가 기록에만 머문 문제

이전 문서는 power/thermal 상태 기록을 요구했지만 실행 eligibility로 강제하지 않았다. 노트북 MPS timing은 전원 모드와 thermal throttling의 영향을 받을 수 있으므로 기록만으로는 evidence quality가 부족하다.

### 2.4 prompt와 source document의 불일치

원문을 출력하지 않는 content-free audit에서 기존 document-contained 규칙의 72개 case는 63개 source document에서 왔다. 8개 문서가 여러 prompt를 제공했고 한 문서의 최대 기여는 3개였다. Prompt를 독립적으로 bootstrap하면 이 군집 의존성을 반영하지 못한다.

UTF-8 집계의 이중 loop 가능성도 별도로 확인했으나 이는 겹쳐 출력된 source 구간을 잘못 읽은 것이며 실제 코드는 seed당 한 번만 집계했다. 존재하지 않는 오류는 수정하지 않았다.

## 3. 교정

1. `src/jamoflow/actual_inference_protocol.py`를 실행기와 요약기의 단일 versioned source로 둔다.
2. Protocol v3는 seeds, modes, roles, 5 repetitions, 정확한 output horizon과 `255` runtime-observed bytes를 공유한다.
3. Selector는 512-byte window가 한 문서에 들어가는 조건에 더해 source document당 case를 하나만 허용한다. 재구성 결과는 633개 candidate document 중 서로 다른 72개 문서를 선택한다.
4. 전체 session 시작과 각 seed timing 직전·직후에 AC power, AC default power mode `0`, thermal/performance warning 부재를 요구한다. 조건을 읽지 못하거나 만족하지 않은 seed는 timing artifact로 저장·재사용하지 않는다.
5. 요약기는 manifest schedule, warmup schedule, seed order, case context, checkpoint lineage, seed별 시작·종료 환경, timing-array shape와 patch-count upper bound를 독립 재구성한다.
6. Seed 실행 전에 in-progress sentinel을 원자적으로 기록한다. 강제 재실행이 실패하거나 process가 중단되어 예전 report/artifact가 남아도 sentinel이 제거되지 않으므로 요약기는 이를 evidence로 읽지 않는다.

## 4. 통계적 의미

Measured 64개 prompt가 이제 64개 서로 다른 source document를 대표하므로 crossed seed-by-prompt bootstrap은 crossed seed-by-source-document bootstrap과 같은 sampling unit을 갖는다. Repetition 5개는 각 seed×document×policy cell에서 median으로 먼저 축약하며 독립 표본으로 세지 않는다. Warmup 8개도 measured 문서와 겹치지 않는다.

이 변경은 speedup을 유리하게 만드는 threshold 변경이 아니다. 오히려 표본 다양성을 강제하고 timing eligibility를 엄격하게 만들어 positive result의 문턱을 높인다.

## 5. 승인 조건

Actual timing은 다음을 모두 만족한 뒤에만 시작한다.

- 현재 교정 focused test 27개와 전체 test suite 290개 통과
- S/E/EC family 완결 뒤 sealed summary 생성
- calibration-only comparator 선택과 five-seed quality noninferiority 통과
- 장시간 학습·router scoring·다른 MPS process가 없는 단독 timing session

실측 뒤 protocol을 변경해야 하면 기존 artifact와 섞지 않고 version과 output root를 새로 고정해 전체 seed를 다시 실행한다.
