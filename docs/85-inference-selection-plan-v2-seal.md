# Inference selection plan v2 seal

> 작성일: 2026-08-12
> 상태: **calibration evidence 생성 전, 선택 수학과 새 final-test outcome에 대해 outcome-blind plan 봉인 완료**

새 Korean final-test seal을 선행 입력으로 삼아 inference comparator 선택 계획을
고정했다. 이 계획은 rate와 reference를 initial 세 seed의 checkpoint에서 다시 계산한
calibration NLL로만 선택한다. Historical screening test, 새 final test 및 latency는
선택 입력이 아니다. Historical screening 결과는 plan 생성 전에 이미 알려져 있었지만
selection criterion의 입력은 아니었다. 따라서 모든 과거 결과에 blind였다고 주장하지
않는다.

## 봉인 identity

- path: `data/manifests/phase3-inference-selection-plan-v2.json`
- file SHA-256:
  `d16980d7a86c9d3a8873062c049238664354878b58fccf80392896a50e5c3bca`
- canonical plan SHA-256:
  `3f315167cc394c6bb1d108573cca653b188b9703892a71f850cf02126ca9fddd`
- plan base commit:
  `8820f947d4b839e3cf72cd7dedfc26e607c3e95f`
- final-test seal file SHA-256:
  `ce42e8a0b2d8161cc59e0b30d5d121b547e22d28709fe48284aa777df4a2290b`
- final-test seal payload SHA-256:
  `97cf90d1e6e7191e7f8336647f278ae6c0e82d70540bf0f5c43f9cb426e75dc8`

Plan 생성 시 새 final test의 `evaluated_at_plan`은 `false`였으며, conversion artifact
root와 selection output은 존재하지 않았다.

## 고정한 선택 규칙

1. Initial seeds는 `1729, 2718, 31415`로 고정한다.
2. Compute conversion rate는 `64, 72` 순서로 검사하고, C86 대비 mean calibration
   BPB 차이가 `+0.010` 이하이며 최소 두 seed가 margin 안인 첫 rate만 선택한다.
3. Reference는 F/C/W/S/E/EC와 selected-rate C 가운데 initial 세 seed mean
   calibration BPB가 가장 낮은 policy다. Exact tie만 고정 순서로 해소한다.
4. Candidate는 selected-rate whitespace policy다.
5. Candidate와 strongest reference의 broad calibration gap이 mean `+0.010` 이하이고
   최소 두 seed가 margin 안일 때만 broad claim confirmation을 허용한다.
6. Rate가 없거나 broad futility 조건이 실패하면 comparator를 약한 후보로 바꾸지
   않는다. Narrow compute-conversion claim과 broad strongest-reference claim은 별도로
   판정한다.

Calibration evaluator는 MPS, batch 64, 512-byte sequence 15,625개를 사용하며,
각 checkpoint에서 per-sequence float32 NLL을 다시 생성한다. Canonical BPB는
float64 `fsum / (count * 511 * ln(2))`로 계산한다. 기존 report의 scalar BPB는 선택
값이 아니라 독립 재구성 결과와 비교하는 integrity check에만 사용한다.

현행 hardening은 동일 sealed Apple MPS 환경에서 exact 30-unit causal replay를 두 번
요구한다. 첫 evaluator가 evidence receipt를 만들고, selection-lock builder가 같은
checkpoint·router·matrix에서 별도의 두 번째 forward를 실행해 float32 NLL hash와
BPB를 다시 확인한다. Decision은 두 번째 replay BPB로만 생성한다. Plan 시점의
rate/reference 함수와 핵심 의존 파일도 AST/file hash로 현재 코드와 비교한다.
`identity artifact ≤ first evaluator < evidence artifact ≤ second verifier < selection-lock
artifact` 순서 및 두 replay commit의 exact implementation blob은 downstream에서도
다시 검증한다.

## 검증

Plan schema, outcome-use 경계, hash/path/rule tamper, calibration receipt rotation,
confirmation authorization 결속을 다루는 14개 targeted unit test가 plan 봉인 시점에
통과했다. 이는 이후 추가된 hardening 전체의 현재 test count를 뜻하지 않는다.

봉인 당시 다음 단계는 이 plan commit을 고정 조상으로 둔 C64/W64/C72/W72 initial
학습이었다. Plan 자체의 hash와 내용은 이후 소급 변경하지 않았다.

현행 hardening DAG는 다음과 같다.

1. initial run을 원래 clean run commit에서 완료한다.
2. selection/confirmation implementation과 검증 계약을 commit한다.
3. 3×10 physical model/source/router/run identity를 봉인하고 단독 commit한다.
4. 첫 30-unit calibration causal replay evidence를 만들고 단독 commit한다.
5. 두 번째 30-unit replay로 selection lock을 만들고 단독 commit한다.
6. Post-selection selected-rate C/W와, broad futility를 통과할 때만 selected S/E/EC를
   sealed implementation에서 학습하고 모든 required training-completion receipt를
   evaluator보다 앞선 tracked commit 하나 이상에 고정한다.
7. Confirmation calibration replay evidence와 post-confirmation authorization을 각각
   단독 commit한 뒤에만 새 final quality evaluation을 연다.

Hardening code 자체는 initial training 뒤에 추가됐으므로 “모든 model outcome 전에
implementation을 preregister했다”고 주장하지 않는다. 정확한 범위는 decision rule과
final-test identity가 plan에서 선행 고정됐고, selection metric replay/lock과
post-selection C/W·조건부 S/E/EC confirmation 실행 전에는
implementation·environment가 봉인됐다는 것이다. Initial
conversion checkpoint identity seal은 재현 가능한 post-run provenance이지 외부
append-only ledger나 독립 재학습 증명은 아니다. 최종 효율 주장은 그 뒤 prospective
confirmation seed와 sealed final test, actual-inference replication을 추가로 요구한다.
C86 및 다른 F/C/W의 confirmation seeds는 이 prospective chain 이전에 생성된
historical five-seed evidence이며, plan에 봉인된 summary의 checkpoint/report/state
hash로 별도 검증한다.
Completion 전 ignored active marker와 학습 산출물을 전부 삭제해 버린 로컬 attempt가
없었다는 사실은 Git만으로 증명하지 못한다. 따라서 prospective confirmation은
cryptographic single-attempt가 아니라, tracked completion으로 선택된 공식 run을 두 번의
calibration causal replay로 재검증하는 절차로 기술한다.
