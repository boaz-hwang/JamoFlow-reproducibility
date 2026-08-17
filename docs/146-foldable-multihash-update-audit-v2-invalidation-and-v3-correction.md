# Foldable multi-hash update audit v2 무효화와 v3 shape 교정

> 작성일: 2026-08-15
>
> 상태: v2는 model forward·gradient·update 관측 전에 중단; v3에서 sequence view 교정

## 실패 내용

V2는 historical parent blob 검증과 128MB train stream encoding을 통과했지만, 첫 batch shape
검사에서 다음 오류로 중단됐다.

`RuntimeError: update-audit first batch differs`

`encode_stream_to_memmap`의 반환은 flattened one-dimensional token memmap이다. 기존 B1 worker는
`[:full_sequence_count*512].reshape(full_sequence_count,512)`를 먼저 적용한다. V2 runner는
permutation sequence index를 flattened array에 직접 적용해 `(32,)`를 만들었고, 봉인된 기대 shape
`(32,512)`에서 fail-closed 됐다.

V2에서는 checkpoint file/state hash를 읽었지만 model construction, state load, forward, loss,
backward, gradient, optimizer step과 update 관측은 일어나지 않았다. Result artifact도 없다.

## V3 교정

V3는 기존 B1과 동일하게 full sequence prefix를 `(sequence_count,512)`로 reshape한 뒤 first 32
sequence를 선택한다. Scheduled exposure count도 같은 2D sequence view에서 재구성한다.

다음은 바꾸지 않는다.

- first-batch index hash
- token budget와 optimizer
- role/checkpoint
- update geometry metric
- projection multiplier safety range
- result/claim boundary

Protocol ID, plan/result path와 schema만 v3로 올리고 v1/v2 plan은 삭제하지 않는다.
