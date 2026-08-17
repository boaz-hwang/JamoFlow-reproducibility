# Actual-inference quality gate addendum

> 작성일: 2026-08-11
> 상태: **compute-conversion, comparator selection, actual-latency 결과 전 고정**
> 상위 protocol: [Actual-inference and compute-conversion protocol](./44-actual-inference-and-compute-conversion-protocol.md)
> selection 후속 교정: [Selection and time-to-output correction](./53-selection-and-time-to-output-correction.md)

## 1. 고정할 모호성

상위 protocol의 Final Value Gate는 selected W-rate가 고정 comparator보다 `+0.010 BPB` 이내라는 five-seed paired 95% upper bound를 요구하지만 interval 종류를 명시하지 않았다. 작은 seed 수에서 결과를 본 뒤 더 유리한 interval을 선택하지 않도록 다음으로 고정한다.

- primary noninferiority interval: seed별 paired BPB 차이에 대한 two-sided Student-$t$ 95% interval
- 차이 방향: `candidate W-rate − locked reference`; 낮을수록 candidate에 유리
- 통과: paired-$t$ interval의 upper bound가 엄격하게 `< +0.010 BPB`
- 동일 test window를 seed와 policy가 공유해야 함
- crossed seed × sequence hierarchical bootstrap 95% interval도 보고하지만 compact quality gate를 대체하지 않음

Candidate와 reference의 report scalar만 사용하지 않는다. 각 seed의 float32 per-sequence NLL, training report, checkpoint state, source stream lineage를 독립 재구성한다.

## 2. 실행 순서

1. Initial 세 seed의 **mean calibration BPB**로 comparator를 latency 전에 고정한다. Test BPB는 comparator가 잠긴 뒤 quality/noninferiority 평가에만 사용한다.
2. Candidate와 comparator의 다섯 checkpoint를 모두 확보한다.
3. Same-rate conversion confirmation과 primary Gate J를 먼저 통과한다.
4. 이 문서의 noninferiority gate를 계산한다.
5. 통과한 경우에만 actual-inference timing을 evidence run으로 허용한다.

Quality gate 실패 뒤 comparator를 더 약한 모델로 바꾸거나 margin을 넓혀 speed result를 구제하지 않는다.

## 3. 범위 제한

BPB noninferiority는 downstream Korean task noninferiority가 아니다. Compact timing을 열기 위한 최소 quality guard이며, publication-scale Final Value Gate의 downstream 조건은 별도로 남는다.
