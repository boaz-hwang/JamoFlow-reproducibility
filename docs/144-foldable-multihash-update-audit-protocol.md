# Foldable multi-hash AdamW first-update audit protocol

> 작성일: 2026-08-15
>
> 상태: historical v1/v2는 update 관측 전에 중단; v3는 결과 봉인 전 alignment 정의 오류로 중단;
> corrected v4는 `docs/147`이 지배

## 목적

Foldable B1에서 `untied_generic_surface`가 no-residual base보다 0.01556 BPB 좋았다. 그러나
13-way zero-initialized residual은 AdamW에서 ordinary dense row와 다른 effective update를 만든다.
이 audit은 quality를 다시 비교하지 않고 동일한 step-0 graph와 고정 first batch에서 다음만 측정한다.

1. ordinary dense input/output 새 row update
2. multi-hash effective input/output 새 row update
3. 두 update의 projection multiplier, norm ratio, cosine와 orthogonal fraction
4. global gradient-clipping norm 차이
5. hash bucket aggregate gradient와 token-row gradient의 alignment

다섯 번째 metric의 cosine은 input/output별 direct lexical gradient가 정확히 nonzero인 신규 행에서만
정의한다. 첫 배치에 등장하지 않은 input row의 zero gradient는 정상이며, 전체 행 수, nonzero 행 수,
제외된 zero 행 수와 슬롯별 zero selected-bucket 수를 함께 기록한다.

결과는 다음 mechanism screen의 단일 update-matched control을 고정하는 데만 사용한다. BPB, test,
latency 또는 기존 final result를 control multiplier 선택에 사용하지 않는다.

## 고정 입력

- architecture/role: `untied_generic_surface`
- initialization: foldable B1 step-0 physical checkpoint
- dense comparator: 같은 checkpoint를 독립적으로 exact fold한 ordinary untied dense graph
- batch: B1 order seed가 정한 첫 32개 512-token sequence
- optimizer: 기존 AdamW `(beta1,beta2,eps)=(0.9,0.95,1e-8)`
- body/head learning-rate schedule, weight decay, microbatch 8×4와 global clip 1.0: B1과 동일
- device: Apple MPS, shared publication lock 아래 단독 실행

두 graph는 update 전 effective input/output weights, body state와 training loss가 같아야 한다. Audit
runner는 B1 plan, summary, worker receipt와 step-0 checkpoint의 file/state hash를 모두 재검증한다.

## metric

새 6,144 row의 ordinary dense update를 `d`, multi-hash effective update를 `m`이라 한다.

- projection multiplier: `<m,d> / ||d||²`
- norm ratio: `||m|| / ||d||`
- cosine: `<m,d> / (||m|| ||d||)`
- orthogonal fraction: `||m - projection*d|| / ||m||`
- per-row norm-ratio/cosine quantile
- scheduled exposure quartile별 median

Input과 output을 분리한다. Control multiplier는 각각의 projection multiplier 하나로 고정하며,
finite이고 `(1,16)` 안이어야 한다. 이 범위 실패 시 row-scaled dense control을 만들지 않고 audit을
중단한다.

## 해석 경계

- projection이 커도 multi-hash가 quality를 개선했다는 새 증거가 아니다.
- orthogonal fraction은 collision-coupled direction의 크기를 보여 주지만 그것이 유용함을 뜻하지
  않는다.
- first-step multiplier가 later Adam moments를 완전히 재현하지 않는다. 다음 screen의 강한 simple
  optimizer control일 뿐 exact optimizer equivalence라고 과장하지 않는다.
- audit batch는 이미 알려진 development train stream이다. publication quality evidence가 아니다.

## 다음 단계

Audit 성공 뒤 별도 plan에서 untied `update_matched_dense`와 same-budget
`balanced_random_multihash`만 새로 학습한다. Current multi-hash가 update-matched dense보다 사전
고정 minimum과 document uncertainty gate를 통과해 좋아야 fresh-data stage를 연다. Random control이
current surface assignment와 같거나 더 좋으면 surface/Unicode semantic claim을 제거하고 stronger
generic hash recipe만 후보로 남긴다.
