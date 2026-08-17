# Publication runtime evidence correction

> 작성일: 2026-08-12
> 상태: **publication actual-inference 실행 전 결과맹 증거 계약 구현**
> 주의: 이 문서는 측정 결과가 아니라 false-pass를 막는 protocol correction이다.

## 1. 재감사에서 발견한 문제

기존 `PublicationComparatorInferenceGate`는 runtime equivalence, timing integrity,
valid-output contract를 boolean으로 받고, 네 latency 배열과 네 encoding-rate mapping을
별도로 받았다. 통계식이 맞더라도 서로 다른 checkpoint, tokenizer, case set 또는
timing run의 일부를 이어 붙여 pass를 만들 수 있었다.

추가로 다음 결함이 있었다.

1. Full/incremental 비교 builder는 `allclose`와 argmax를 모두 확인했지만 dataclass
   validator는 argmax 100%만 역검증했다. 모든 logit에 큰 상수를 더해 argmax만 같은
   경우 stale/tampered object가 통과할 여지가 있었다.
2. 8개 warm-up case와 seed 실행 순서가 timing evidence에 봉인되지 않았다.
3. 동일한 128 raw-byte prompt/continuation이라도 BPE와 byte model의 실제 model-unit
   열은 다르지만, 그 변환 결과가 pair lineage에 명시적으로 들어 있지 않았다.
4. Raw-byte output의 0--3 byte overshoot와 BPE token output의 overshoot를 같은
   상한으로 다루면 BPE를 부당하게 탈락시킨다. 반대로 BPE output을 unit 수와 token
   byte 길이에 결속하지 않으면 허위 completion도 가능하다.
5. Raw/16K/32K 세 comparator gate가 같은 candidate family key만 쓰고 실제 candidate
   checkpoint/case hash는 달라도 Final Value Gate가 구분하지 못했다.

## 2. 봉인한 runtime lineage

`src/jamoflow/publication_runtime.py`의 `PublicationRuntimeLineage`는 한 candidate–
comparator pair에 대해 다음을 SHA-256 identity 하나로 묶는다.

- 고정된 세 seed 순서와 양쪽 checkpoint state hash
- seed별 model-config hash
- Raw reference의 compact selection descriptor hash와 concrete policy. E/EC이면 descriptor에서 `entropy_router`를 파생하고 각 seed의 structured router bundle을 결속
- Router bundle의 checkpoint artifact/state, report, 공통 architecture config, train/calibration/test stream, scalar threshold, maximum patch length, E/EC candidate-position 규칙, threshold cache/diagnostics와 split별 patch-matrix hash
- candidate/comparator tokenizer와 strict UTF-8 transition-table hash
- 실제 runtime source와 timing-scope audit artifact hash
- case manifest, 공통 raw prompt, 공통 raw replay continuation hash
- candidate/comparator의 prompt-unit 및 replay-unit 배열 hash
- 양쪽 unit 열이 공통 raw bytes를 정확히 복원한 unitization audit hash
- protocol version과 정확한 timing-scope contract

따라서 공통 raw text가 같다는 사실과 각 architecture가 실제로 실행한 unit 열이
같은 실험에서 나왔다는 사실을 동시에 보존한다. Text 자체는 evidence object에 넣지
않아 private corpus가 aggregate artifact로 새지 않는다.

Structural raw policy이면 descriptor의 `auxiliary_kind=none`에서 모든 router field가
비어야 한다. E/EC이면 반대로 세 seed bundle이 모두 필요하다. Worker가 auxiliary
종류를 직접 선언하지 않으며 selection descriptor의 policy에서만 파생한다. Runtime
protocol v3와 model-lock protocol v3가 이 descriptor와 bundle을 canonical identity에
포함한다.

## 3. 등가성 증거

모든 `3 seeds × 2 roles × 2 paths = 12` 비교를 정확히 요구한다. 두 path는 다음과
같다.

- full-prefix 대 token/byte incremental consume
- parallel prefill final logit 대 incremental continuation 시작 logit

각 비교는 최소 16개 logit vector를 포함하고 `rtol=2e-5`, `atol=2e-5`의 전체
`allclose`와 argmax 100%를 동시에 만족해야 한다. Array dtype, shape와 양쪽 content
hash를 manifest hash에 포함한다. `allclose_pass` 자체도 identity에 들어가며 최종
pass는 `allclose_pass AND argmax_match_rate == 1`로만 재구성된다.

## 4. Timing 증거

Publication timing design은 다음 값으로 고정했다.

| 항목 | 값 |
|---|---:|
| model seeds | 1729, 2718, 31415 |
| warm-up cases | 8 |
| measured prompts | 64 |
| repetitions/prompt | 5 |
| prompt raw bytes | 128 |
| minimum valid output bytes | 128 |
| modes | controlled replay, free-running strict UTF-8 greedy |

Builder는 12개의 `mode × component × role` positive `float64` 배열을 정확히
`3×64×5` shape로 요구한다. TTFT와 decode 합이 end-to-end와 일치하는지 원시
배열에서 확인하고, primary 두 estimand를 다시 계산한다.

- controlled replay: batch-1 decode latency
- free running: batch-1 end-to-end latency

Candidate/reference 실행 순서는 seed×mode×prompt×repeat마다 사전 고정 seed로
균형 무작위화한다. Warm-up 순서와 모든 warm-up role의 완료 trace, seed 자체의
실행 순서도 별도 고정 seed로 검사한다. 각 seed timing 시작과 끝에서 AC power,
default power mode, thermal/performance warning 부재를 확인한다. Schedule, environment,
timing array와 두 crossed-bootstrap summary는 같은 lineage identity에 묶인다. Timing
배열과 output diagnostic은 동일한 private trial artifact SHA-256을 반드시 공유하므로
서로 다른 실행에서 가져온 두 배열을 결합할 수 없다.

MPS에서 다른 학습 또는 timing job을 동시에 실행하지 않는다. 코드·문서·CPU 합성
검사는 독립적으로 할 수 있지만 publication latency가 시작되면 시스템 부하를 만들
수 있는 작업도 모두 중지한다.

## 5. Valid-output 증거

두 mode, 두 role의 모든 seed×prompt×repeat에 대해 다음 정수 배열을 요구한다.

- prompt model units, emitted raw bytes와 emitted model units
- decode forward steps와 runtime-observed model units
- router observed/cached/scored model units와 router forward calls
- overshoot, valid stop, final UTF-8 accept, transition-trace validity
- replacement-character absence와 output Unicode-scalar count

다음 항등식을 원시 배열에서 재구성한다.

```text
decode forwards = emitted model units - 1
runtime observed units = prompt model units + decode forwards
overshoot = emitted raw bytes - 128              # free running
```

Controlled replay는 양쪽 모두 정확히 128 raw bytes를 생성한다. Candidate와 raw-byte
reference는 unit=byte이며 free-running output이 처음 가능한 UTF-8 boundary에서
멈추므로 128--131 bytes만 허용한다. BPE reference는 token 하나가 여러 byte를 담을
수 있어 0--3 상한을 적용하지 않는다. 대신 모든 emitted token이 non-empty이고,
`emitted units <= 512`, `bytes <= units × tokenizer의 실제 최대 token-byte 길이`를
요구하며 그 tokenizer/transition table hash를 lineage에 묶는다.

Greedy diagnostics는 다섯 repetition에서 완전히 같아야 한다. Private raw-output/
model-unit trace artifact hash와, 그 trace에서 UTF-8 state·replacement·codepoint·stop
진단을 재구성한 audit hash도 함께 요구한다. Completion 및 replacement-free seed
rate는 진단 배열에서 다시 계산하며 report가 선언한 rate를 입력으로 받지 않는다.

E/EC reference의 모든 measured trial에서 router observed/cached/scored units는 해당
reference의 runtime-observed units와 같고, router forward calls는
`decode_forward_steps + 1`이어야 한다. Candidate, BPE와 structural raw reference의
router counter는 모두 0이어야 한다. 따라서 router bundle만 lineage에 넣고 실제 timed
path에서는 router를 건너뛰는 실행은 integrity gate를 통과하지 못한다.

## 6. Gate 연결 교정

Comparator gate의 입력에서 다음을 제거했다.

- 세 runtime pass boolean
- 네 latency 배열
- 네 output-rate mapping
- 호출자가 따로 지정하는 candidate/comparator/family/seed

대신 검증된 `PublicationRuntimeEvidence` 하나만 받는다. Gate는 nested timing
summary와 output rate를 사용하며 BPB 및 downstream identity와 일치하지 않으면
계산 전에 실패한다.

Final Value Gate는 raw, 16K BPE, 32K BPE runtime lineage 사이에서 candidate
checkpoint/config/tokenizer/transition, case manifest, raw cases, candidate unitization,
runtime source와 scope contract가 모두 같은지 추가 확인한다. Comparator별 결과를
낼 때 candidate나 prompt set을 바꾸는 우회를 허용하지 않는다.

## 7. 음성 검증

Protocol v3의 reference/model-lock/runtime/scale/downstream/data-adequacy/inference
집중 test 66개와 전체 회귀 test 371개가 통과했다. Runtime 음성 검증은 다음
false-pass를 직접 재현하고 차단한다.

- argmax는 같지만 logit이 tolerance 밖인 incremental path
- 비교 path 하나의 verification vector 부족
- 16K equivalence와 32K timing/output 결합
- comparator별로 서로 다른 candidate checkpoint/case lineage 사용
- measured schedule, warm-up completion, seed order 변조
- timing component 항등식 파괴와 부적격 전원/thermal 환경
- raw-byte 3-byte 초과 overshoot
- repetition마다 달라지는 greedy diagnostic
- forward-step/runtime-observed-unit 항등식 파괴
- nested evidence를 identity hash 갱신 없이 교체
- timing과 output diagnostic의 trial artifact hash 불일치
- entropy policy를 structural auxiliary로 위장하거나 router checkpoint/config/calibration bundle을 교체
- E/EC timed trial의 router observed/cached/scored/forward counter 누락

BPE reference의 3-byte 초과 overshoot가 token-byte bound 안에서는 정상적으로
허용되는 positive test도 포함했다. 전체 suite는 단일 MPS training family가 완전히
끝난 뒤 실행했다.

## 8. 아직 publication evidence가 아닌 것

현재 test는 content-free 합성 배열로 계약을 검증한다. 실제 효율 향상을 입증하지
않으며 논문 표에 넣지 않는다. 다음 단계의 actual runner가 실제 checkpoint와 private
numeric artifact에서 이 객체를 재구성해야 한다.

BPB loss, downstream prediction, learning-curve artifact와 runtime을 concrete
checkpoint hash로 잇는 공통 model-lock graph는 후속
`76-publication-model-lock-graph.md`에서 구현했다. 다만 두 계약 모두 아직 합성 배열로
검증된 상태다. 실제 runner가 실제 checkpoint와 private numeric artifact에서 runtime
evidence와 model-lock graph를 재구성하기 전에는 Final Value Gate의 결과를 논문
결론으로 사용하지 않는다.
