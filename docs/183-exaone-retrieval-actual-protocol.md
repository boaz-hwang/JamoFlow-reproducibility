# EXAONE 7.8B retrieval actual-inference protocol

> 작성일: 2026-08-15
>
> 상태: **첫 EXAONE retrieval candidate 실행 전에 고정할 실제 추론 프로토콜**

## 연구 질문

작은 16K-vocabulary 모델에서 자유 생성 end-to-end latency를 줄였던 exact retrieval
speculative decoding이, 같은 한국어 중심 workload의 7.8B EXAONE 4-bit 모델에서도 실제 추론 시간을
줄이는가를 검증한다. 이번 단계에서 바꾸는 것은 decoding path뿐이다. checkpoint, tokenizer, prompt,
greedy decision, output token 수는 같다.

이 실험은 generic retrieval speculative decoding을 새 방법으로 주장하지 않는다. 확인하려는 것은
Mac에서 실행 가능한 큰 한국어 모델로의 scale transfer와, 이후 한국어 고유 기제를 generic retrieval과
동일 비용으로 비교할 가치가 있는지다.

## 고정 비교

- 모델: `mlx-community/EXAONE-3.5-7.8B-Instruct-4bit`, pinned revision
  `6f8fba5756a6e2987aecacd8d7e8bb9410ef1a53`
- baseline: ordinary cached greedy autoregressive decoding
- candidate: train-only 200,000-entry Korean token n-gram table을 우선 사용하고, miss면 prompt/self-output
  lookup을 사용하는 exact forced speculative decoding
- maximum draft: 3 tokens
- prompt: 128 EXAONE tokens
- output: free-running greedy 128 tokens
- cases: 8 warmup + 64 measured, 서로 다른 Korean documents
- sessions: fresh process 5개
- inner repetitions: case·role별 3회

Latency estimand을 case마다 동일하게 만들기 위해 EOS가 중간에 나오더라도 멈추지 않고 정확히 128 tokens를
생성한다. 이는 고정-horizon raw-completion throughput/latency 실험이며, 실제 chat serving의 stop-aware
latency는 후속 별도 workload가 필요하다.

Resource calibration V3는 candidate를 실행하지 않고 baseline만 측정해 5 sessions × 3 repetitions를
첫 feasible schedule로 선택했다. 따라서 candidate latency와 acceptance를 본 뒤 반복 수를 줄이거나 늘리지
않는다. 이전 compatibility 단계에서는 synthetic forced-proposal 경로의 exact greedy output을 이미
검증했으므로 모델 output 자체에 완전히 blind한 단계도 아니다. 다만 이 plan 전에 실제 Korean retrieval
table의 acceptance나 latency는 실행·관찰하지 않았다.

후속 provenance 감사에서 case rank key가 compatibility result artifact와 그 summary hash를 seed에
포함하고, compatibility summary가 synthetic deterministic/forced model-output hashes를 포함한다는 사실을
확인했다. 그러므로 upstream data plan의 `historical_model_output=false`를 literal claim으로 사용하지 않는다.
72 cases는 compatibility model output의 함수이고, compatibility pass가 먼저 공개된 뒤 case protocol이
작성되었다. 반면 이 72 cases에서 retrieval candidate의 output, acceptance, latency는 전혀 관찰되지 않았다.
정확한 교정과 case를 유지한 이유는 [provenance correction](184-exaone-case-selection-provenance-correction.md)에
기록한다.

Actual plan은 반복 수만 가져오지 않는다. Resource result의 최종 상태가
`pass_baseline_resource_feasibility`, schedule 상태가 `feasible`, baseline high-water memory가 MPS 권장
working set의 75% 이내인 `safety_pass=true`일 때만 봉인된다. Actual session에서도 model과 retrieval
table을 함께 올린 관측 high-water가 같은 75% 안전선을 넘으면 latency가 좋아도 해당 session과 primary
gate를 실패시킨다.

## 공정한 timer 경계

두 role 모두 다음을 end-to-end timer 안에 둔다.

1. 이미 고정된 prompt text를 `add_special_tokens=False`로 tokenization
2. fresh KV cache를 이용한 prefill과 128-token generation
3. greedy argmax와 모든 device-host scalar readback
4. candidate의 corpus lookup, prompt lookup, target verification, rejection rollback
5. prompt와 output 전체 detokenization
6. 마지막 MLX synchronize

모델·tokenizer·retrieval table load와 case artifact load는 timer 밖이다. 각 role은 같은 process에서 같은
모델을 공유하지만 매 trial fresh KV cache를 만든다. 역할 순서는
`session + floor(canonical_case_index / 2) + repetition` parity로 정한다. 각 session 안에서는 정확히
반반이고, 고정 canonical case×repetition과 고정 temporal position×repetition 모두 다섯 session에 걸쳐
3:2 또는 2:3으로 교대한다. 각 session의 case order는 stride 13 cyclic rotation이다.

전체 `elapsed = tokenization + generation + detokenization`을 정수 nanoseconds로 저장한다. 구성요소
시간은 원인 진단용이고 primary gate는 end-to-end elapsed다. Memory는 각 trial resettable MLX peak와
session high-water를 보존하지만 개선 gate에는 넣지 않는다.

## 정확성 불변식

모든 paired trial에서 baseline과 candidate의 128 output token IDs, token-sequence SHA-256, strict UTF-8
decoded-sequence SHA-256이 exact해야 한다. 다르면 해당 session 전체를 실패시킨다. Candidate는 다음
counter identity도 만족해야 한다.

- target calls = proposal attempts + no-proposal calls
- proposal attempts = full accept + immediate reject + partial accept cycles
- correction tokens = immediate reject + partial accept cycles
- output 128 = no proposal + accepted draft + correction + bonus
- final cache offset = 128 prompt + 128 output - 1 = 255

세션 요약 전에는 저장 counter의 대수 관계만 보는 데 그치지 않는다. 다섯 receipt가 모두 commit된 뒤
동일 checkpoint/table/cases로 warmup과 measured 64 cases를 다시 생성하고, 저장된 모든 role·repetition의
output과 counter를 독립 forward 결과에 exact 대조한다.

Corpus와 prompt/self-output source별 proposal calls, proposed tokens, accepted draft tokens도 따로
보존한다. 따라서 primary가 통과하면 어느 source가 coverage와 acceptance를 만들었는지는 기술할 수 있다.
다만 corpus-only·prompt-only의 별도 latency와 controlled replay는 이 resource-calibrated two-role primary
campaign에 사후로 끼워 넣지 않는다. 필요하면 primary 결과 뒤 별도 prospective secondary protocol로
실행하며, 이 secondary 결과로 primary gate를 바꾸지 않는다.

## 측정 환경과 증거 순서

각 session은 별도 Python process에서 시작한다. 시작, warmup 뒤, measured case 16개마다, 종료 시점에
다음을 timer 밖에서 확인한다.

- AC power
- macOS가 보고하는 thermal/performance warning 없음
- 현재 PID가 parse되는 process inventory
- 알려진 JamoFlow neural/MPS entrypoint와 동시 실행 없음

Plan에 봉인된 exact hardware(`device_name`, architecture, physical/recommended memory), macOS/Python 및
MLX·MLX-LM·Transformers 등 package identity는 각 fresh process 시작과 receipt publish 직전 다시
수집해 exact 일치시킨다. 최종 독립 replay도 시작과 summary publish 직전에 같은 대조를 반복한다.

Machine-global flock도 잡으므로 같은 계약을 따르는 다른 worktree의 publication runner와 중첩되지 않는다.
다만 임의의 이름으로 실행한 외부 MPS 프로그램까지 완전 탐지한다는 주장은 하지 않는다.

실행 DAG는 다음과 같다.

1. 본 문서, runner, 통계, tests를 commit
2. clean tree에서 actual plan을 exclusive-create하고 별도 commit
3. session 0 실행: ignored NPZ와 metric-free tracked receipt 생성
4. receipt를 commit한 뒤에만 다음 fresh-process session 허용
5. session 1–4도 같은 순서로 각각 실행·commit
6. 정확히 다섯 receipt가 HEAD에 있고 각 path의 Git publication history가 한 번뿐일 때 독립 replay
7. replay가 통과한 뒤 처음으로 timing arrays를 통계화하고 summary를 exclusive-create
8. summary를 commit한 뒤 별도 read-only verifier가 checkpoint/table forward, raw arrays, 통계,
   receipt/plan Git lineage를 다시 구성해 published summary와 exact 대조

Verifier는 단순히 각 artifact가 HEAD의 조상인지 따로 보는 데 그치지 않는다. Implementation base commit
`<` plan publication commit `≤` session-0 run commit `<` receipt-0 publication commit `≤` session-1 run
commit `<` receipt-1 publication commit ... 순서의 모든 인접 edge를 직접 검사하고, 각 receipt publication
commit을 최종 lineage에 남긴다. 자기 run→receipt edge는 반드시 strict하며, 앞 receipt를 포함한 HEAD에서
다음 run을 시작하는 edge만 equality를 허용한다.

각 receipt에는 raw timing artifact hash와 array hashes가 있지만 성능 aggregate는 없다. Partial artifact,
남은 active sentinel, deleted/reissued receipt, non-prefix session, plan/implementation drift는 자동 재시작하지
않고 forensic failure로 처리한다.

이 순서는 공식 분석의 result-dependent 변경을 막기 위한 운영 계약이다. 로컬 파일 삭제 권한을 가진 사람이
ignored timing NPZ를 별도 도구로 열거나 첫 commit 전 통째로 삭제했다는 가능성까지 암호학적으로 막지는
못한다. 그러므로 이를 public preregistration이나 cryptographic one-shot이라고 부르지 않는다.

## 사전 고정 통계와 gate

각 session×prompt×role에서 세 repetition median을 하나의 cell로 만든다. Repetition을 독립 표본으로
세지 않는다. 같은 session·prompt index로 paired candidate/baseline을 계산하고, 5 sessions와 64 prompts를
독립 축으로 함께 resample하는 crossed bootstrap 10,000회, seed `20260815`를 사용한다.

Primary pass는 아래를 모두 요구한다.

- 모든 output/counter와 독립 replay correctness 통과
- aggregate median end-to-end reduction ≥ 10%
- crossed session×prompt bootstrap 95% lower bound > 0
- 64 prompts 중 적어도 48개에서 session-median candidate가 빠름
- 다섯 session 각각의 aggregate reduction > 0

Baseline-first와 candidate-first paired trials의 reduction은 별도 order diagnostic으로 공개하지만 gate를
사후 변경하는 입력으로 쓰지 않는다.

이 gate는 “점추정 10% 이상이며 95% interval이 0을 제외한다”는 계약이지, 95% lower bound가 10%를
넘는다는 계약은 아니다. 결과가 실패하면 comparator를 교체하거나 같은 evaluation pool에서 threshold를
재선택하지 않는다.

## 해석 범위

72 cases는 candidate latency/acceptance/output을 보지 않고 고정했지만, compatibility model-output hash를
rank seed에 포함하고 과거 품질 평가에 이미 사용한 sealed evaluation document pool에서 왔다. 모델은
Instruct checkpoint지만 chat template 없이 raw continuation으로 측정한다.
따라서 이 결과는 final-blind confirmatory quality, chat deployment 일반성, 다른 hardware, memory 개선을
증명하지 않는다.

Pass가 뜻하는 것은 다음뿐이다.

> On one Apple M4 Pro environment and a fixed Korean-heavy raw-completion workload, exact generic retrieval
> speculative decoding transferred to the 7.8B EXAONE checkpoint with prospectively Git-sealed,
> output-identical actual
> end-to-end latency improvement.

이는 actual timing plan 이후의 candidate 결과에 대해서만 prospective한 exploratory scale-transfer evidence다.
Case selection 자체를 model-output-blind, final-blind 또는 public preregistration이라고 주장하지 않는다.

Pass하면 다음 단계에서 generic retrieval을 강한 baseline으로 유지한 채, 동일 proposal budget과 같은
target verification 비용 아래 한국어 형태·띄어쓰기·자모 구조가 추가 이득을 주는지를 새 미사용 workload로
검증한다. 그 후속 workload의 case-rank key는 compatibility/model output 이후 hash를 입력으로 받지 않는
pre-output identity에서만 유일하게 도출해야 한다. Fail하면 “generic retrieval의 8B 실제 효율 전이”부터
성립하지 않으므로 morphology 확장을 정당화하지 않는다.
