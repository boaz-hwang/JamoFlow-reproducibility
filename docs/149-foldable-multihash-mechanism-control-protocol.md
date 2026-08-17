# Foldable multi-hash mechanism-control protocol

> 작성일: 2026-08-15
>
> 상태: sealed development protocol; 실행 완료, primary 실패

> **결과:** `untied_generic_surface`가 `update_matched_dense`보다 contiguous `+0.007990`,
> document `+0.007953 BPB` 나빠 primary gate를 실패했다. Surface assignment와 random shared-hash
> opportunity도 실패했으며 fresh stage는 열리지 않았다. 전체 판정은 `docs/150`을 따른다.

## 질문

기존 `untied_generic_surface`가 ordinary dense 8K보다 좋았던 이유를 세 층으로 분리한다.

1. AdamW가 만든 새 row update의 정렬된 scale 증가
2. shared bucket을 통한 cross-token gradient diffusion
3. byte/Unicode surface assignment가 주는 token-specific inductive bias

첫-update audit은 1만으로 전체 update를 설명할 수 없음을 보였다. 이 screen은 같은 B1 train 및
calibration stream, model seed, order, 512 optimizer step을 유지한 채 다음 세 역할만 새로 학습한다.

| role | training graph | 변경점 | deployed graph |
|---|---|---|---|
| `update_matched_dense` | ordinary dense 8K | 매 AdamW step 뒤 신규 input/output row update에 각각 고정 배수 적용 | ordinary dense 8K |
| `stratified_generic_shuffle` | 13×128 residual | `(raw byte length, scheduled exposure)`가 같은 행끼리 generic 13-code vector 순환 치환 | exact-folded dense 8K |
| `balanced_random_multihash` | 13×128 residual | 각 slot에서 신규 6,144행을 128 bucket에 정확히 48개씩 독립 배정 | exact-folded dense 8K |

Historical `untied_base`와 `untied_generic_surface`는 새로 학습하지 않고 physical checkpoint와 NLL을
독립 재검증한다.

## Update-matched dense

V4 audit에서 quality metric 없이 고정된 배수만 사용한다.

- input: `1.485414522979104`
- output: `2.170601418278963`

각 optimizer step 직전 신규 row를 복사하고 ordinary AdamW+global clipping을 실행한 뒤,
`before + multiplier * (after - before)`로 그 step의 신규 row 변화만 치환한다. Old vocabulary,
Transformer body, optimizer moment는 ordinary dense와 같다. 이는 first-step dense-aligned projection을
재현하는 단순 control이며 multi-hash와 optimizer-equivalent하다고 주장하지 않는다.

## Assignment controls

### Stratified generic shuffle

각 exact `(UTF-8 byte length, 512-step scheduled token exposure)` stratum에서 seed가 정한 순서로
cyclic derangement한다. Singleton은 불가피하게 유지한다. 이 방식은 stratum별 generic code-vector
multiset, bucket occupancy와 exposure distribution을 보존하면서 token↔code 의미 연결을 끊는다.

### Balanced random

각 slot의 신규 row occupancy가 정확히 48이 되도록 0..127 label multiset을 독립 permutation한다.
Parameter, lookup, residual scale은 generic과 동일하다. 이는 엄밀한 semantic matched control이 아니라
더 균등한 generic shared-hash recipe diagnostic이다.

두 assignment seed는 plan 전에 고정하고 결과 기반 seed 재시도는 금지한다.

## Primary gate

Fresh-data stage를 여는 유일한 후보는 historical `untied_generic_surface`다. Candidate에서
`update_matched_dense`를 뺀 차이에 대해 다음을 모두 요구한다.

1. contiguous calibration BPB difference `<= -0.002`
2. document BPB difference `<= -0.002`
3. paired document bootstrap 95% upper bound `<= 0`
4. historical B1 anchor gap `<= 0.05 BPB`
5. historical ordinary dense base보다도 document/contiguous 모두 좋고 bootstrap upper `<= 0`

하나라도 실패하면 positive branch를 중단한다. Random role이 좋아도 이 plan 안에서 fallback하지 않고,
별도 새 가설·프로토콜 없이는 fresh stage를 열지 않는다.

## Secondary interpretation

- `generic_surface`가 matched shuffle와 balanced random 각각보다 `0.002 BPB` 이상 좋고 bootstrap
  upper `<=0`이면 `surface_assignment_supported`로 분류한다.
- 그렇지 않으면 surface/Jamo semantic claim을 제거한다.
- 각 random role 대 update-matched dense contrast는 shared-hash mechanism diagnostic으로만 보고한다.
- Recovery curves와 training time/memory는 descriptive이며 selection input이 아니다.

## Claim boundary

이것은 이미 알려진 development corpus와 한 model seed의 mechanism screen이다. 통과해도 논문 성공,
일반화, actual inference speedup을 뜻하지 않는다. 다음 단계는 disjoint Korean data, multi-seed,
strong vocabulary-adaptation optimizer control, sealed quality와 batch-1 actual E2E timing을 새로 요구한다.
