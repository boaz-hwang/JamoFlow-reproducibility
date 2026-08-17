# Multi-seed actual-inference addendum

> 작성일: 2026-08-11
> 상태: **compute-conversion 및 actual-latency 결과 생성 전 고정**
> 상위 protocol: [Actual-inference and compute-conversion protocol](./44-actual-inference-and-compute-conversion-protocol.md)
> 목적: 실제 추론 판정에 사용할 checkpoint seed, 통계 단위, 실행 순서, 열·메모리 통제를 결과 전에 명시함

## 1. 추가 고정이 필요한 이유

상위 protocol은 prompt를 paired unit으로 정했지만 어느 model seed의 checkpoint를 timing할지는 명시하지 않았다. Structural policy의 controlled replay 연산량은 weight와 거의 무관하더라도 실제 kernel 실행, free-running output 경로, emitted patch 수는 model seed와 생성 결과에 따라 달라질 수 있다. 한 seed만 골라 측정하면 품질 실험은 five-seed인데 시스템 결론은 single-checkpoint가 되는 불일치가 생긴다.

이 addendum을 고정할 때 다음 결과는 존재하지 않았다.

- 3-seed primary F/C/W의 완결 summary와 OOD Gate I
- 64/72-patch compute-conversion 결과
- incremental actual-latency 결과
- S/E/EC 및 selected-rate의 inference-comparator 선택 결과

따라서 아래 선택은 latency나 완결된 primary effect를 보고 정한 것이 아니다.

## 2. Timing checkpoint와 crossed design

Compact Final Value Gate는 quality confirmation에 사용한 다섯 seed `1729, 2718, 31415, 57721, 65537`를 모두 측정한다.

- candidate는 calibration-only rule로 선택되고 five-seed quality gate를 통과한 W64 또는 W72다.
- reference는 initial 세 seed의 mean calibration BPB만으로 latency 전에 고정한 primary inference comparator다. Test BPB는 선택 뒤 quality 평가에만 사용한다.
- reference가 S/E/EC 또는 selected same-rate C여서 confirmation checkpoint가 없으면 두 confirmation seed를 먼저 학습한다.
- 같은 64개 measured prompt와 8개 disjoint warmup prompt를 모든 seed와 두 policy가 공유하며, 72개 case는 72개 서로 다른 원문 문서에서 하나씩 온다.
- 각 seed × prompt × policy에서 다섯 번 독립 runtime을 만들고 측정한다.

Model loading, checkpoint hashing, source reconstruction, prompt selection은 timing 밖이다. Candidate와 reference model은 함께 device에 올려 model-swap과 disk I/O가 비교에 섞이지 않게 한다. Runtime cache는 매 repetition 새로 만든다.

## 3. Correctness prerequisite의 실행 가능한 구체화

작은 무작위 graph의 모든 prefix 비교는 unit test에 유지한다. 실제 19.6M checkpoint에서는 다음의 두 층 검사를 각 seed와 policy에 적용한다.

1. 8개 disjoint correctness/warmup prompt에서 시작·끝·모든 boundary 전후를 포함해 stable-hash로 고른 최소 16개 prefix position의 full-prefix logit과 incremental logit을 비교한다.
2. 64개 measured prompt 모두에서 parallel prefill과 sequential incremental prefill의 final logit·cache length를 비교하고, 이어지는 controlled 16 bytes의 모든 logit과 cache state를 비교한다.

모든 비교는 `rtol=2e-5`, `atol=2e-5`와 argmax 100% 일치를 요구한다. 이 설계는 실제 graph에서 boundary transition과 모든 measured prompt를 덮으면서, 512-byte full graph를 prefix마다 수만 번 재실행하는 비생산적 검사를 피한다. 어느 하나라도 실패하면 timing은 증거로 승격하지 않는다.

## 4. 실행 순서와 측정 정의

각 seed 안에서 prompt와 repetition마다 candidate/reference 순서를 고정 seed로 무작위화한다. Seed 순서도 고정 난수 순열로 정한다. Warmup 8개는 측정 배열과 bootstrap에 들어가지 않는다.

한 trial은 다음 host-observed 구간을 기록한다.

1. `TTFT`: fresh runtime 생성 직전부터 `prefill_parallel()` 완료 후 device synchronize까지
2. `decode`: TTFT synchronize 직후부터 controlled replay의 정확히 128 source bytes 또는 shared-DFA free running의 최소 128 valid bytes가 첫 UTF-8 accept state에서 완성될 때까지. Controlled는 127회, free running은 overshoot에 따라 127--130회 feedback forward를 쓰며 마지막 output 뒤 쓰지 않을 logit은 계산하지 않음
3. `end_to_end`: 위 두 구간의 합. Stage 경계 synchronize를 포함하는 host-synchronized latency로 명시한다.

Controlled replay에서는 같은 정답 continuation byte를 두 model에 넣으며 logit을 host로 복사하지 않는다. Free-running에서는 공통 strict UTF-8 mask, device argmax, 다음 Python byte로의 전달, DFA state와 stop 검사를 timing에 포함한다. Static DFA mask compilation만 trial 밖이다. Selector, learned router, tensor 생성, cache update, device synchronization도 모두 포함한다.

## 5. Multi-seed primary statistic

Seed `s`, prompt `p`, policy `A`의 다섯 repetition median을 `m_{A,s,p}`라 한다. Compact primary point estimate는 다음이다.

```text
1 - median_{s,p}(m_candidate,s,p) / median_{s,p}(m_reference,s,p)
```

10,000회 crossed bootstrap에서 seed 다섯 개와 서로 다른 source document를 대표하는 prompt 64개를 각각 복원추출하고 두 추출의 Cartesian crossing에 같은 ratio statistic을 적용한다. 같은 resampled seed와 prompt/document index를 candidate/reference가 공유한다. Repetition 5개를 독립 표본처럼 bootstrap하지 않는다.

각 seed별 기존 paired-prompt ratio와 95% interval도 함께 보고한다. Compact latency component는 다음을 모두 만족해야 통과한다.

- crossed point reduction `>= 10%`
- crossed percentile 95% lower bound `> 0`
- 다섯 seed별 point reduction의 median `>= 10%`
- 최소 4/5 seed에서 point reduction `> 0`

이 규칙을 controlled replay decode와 free-running valid-output end-to-end에 각각 적용한다. Free-running emitted bytes와 0--3 byte overshoot도 함께 보고한다. 상위 protocol의 single paired-prompt 조건은 이 multi-seed crossed 조건으로 구체화한다.

## 6. 노트북 열·메모리 통제

Actual timing은 장시간 학습 process와 동시에 실행하지 않는다. 전체 session 시작과 각 seed의 측정 직전·직후에 `pmset`으로 AC 전원, AC 기본 power mode(`0`), thermal warning 부재, performance warning 부재를 확인한다. 상태를 읽지 못하거나 하나라도 만족하지 못하면 해당 seed timing은 evidence artifact로 저장하지 않고 동일 protocol로 seed 전체를 다시 실행한다. Hardware/OS/Python/PyTorch/Transformers 버전과 device도 manifest에 기록한다.

- policy 실행 순서 무작위화로 완만한 thermal drift를 pair 안에서 상쇄한다.
- MPS timing 전후에 synchronize한다.
- MPS current/driver allocated memory와 reset 가능한 peak allocator 값을 policy별로 기록한다.
- process RSS는 단조 증가할 수 있으므로 비교 gate가 아니라 session diagnostic으로만 둔다.
- cache 비우기나 cooling interval을 한 policy에만 적용하지 않는다.

실행기와 요약기는 반복 수, seed·mode·role, 정확한 time-to-output, runtime-observed byte 수와 환경 eligibility를 하나의 versioned protocol module에서 import한다. 요약기는 manifest뿐 아니라 각 seed의 시작·종료 상태와 timing-array shape를 다시 확인한다.

메모리 감소는 보조 결과이며 latency 실패를 구제하지 않는다.

## 7. Final Value Gate에 미치는 영향

상위 protocol의 품질, UTF-8, downstream, publication-scale 조건은 그대로 유지한다. Compact actual-inference 성공에는 이 문서의 five-seed crossed latency 조건이 추가로 필요하다. 한 seed에서만 빠르거나 pooled point estimate만 10%를 넘고 crossed lower bound 또는 4/5 방향 재현성이 실패하면 actual-inference success로 부르지 않는다.

Publication-scale model은 비용 때문에 최소 세 seed가 허용되지만, compact five-seed 결과와 같은 방향이어야 한다. Mac 측정은 Apple MPS 결과로만 표기하고 다른 accelerator로 일반화하지 않는다.
