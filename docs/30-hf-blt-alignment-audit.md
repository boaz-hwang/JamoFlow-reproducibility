# Phase 3 implementation audit: Hugging Face BLT patch alignment

> 작성일: 2026-08-10
> 상태: seed 1,729 F scalar 확인 후, C/W 비교 결과 확인 전 기록
> 영향: algorithm·gate 변경 없음; causal interpretation과 claim boundary 강화
> 관련 구현: `src/jamoflow/neural_patching.py`, `transformers.models.blt.modeling_blt`, `tests/test_neural_causality.py`

## 1. 감사 이유

Phase 3가 측정하는 것은 추상적인 모든 byte-latent architecture의 경계 효과가 아니라 Hugging Face BLT의 local encoder/global trunk/local decoder graph에서 외부 `patch_lengths`를 바꾼 효과다. 이 API는 decoder lag를 맞추기 위해 첫 길이가 1인 dummy patch를 요구한다. 따라서 논문에서 말하는 boundary position과 encoder가 실제로 묶는 byte position이 같은지, suffix 정보가 prefix logit에 새지 않는지를 명시적으로 확인해야 한다.

## 2. 확인된 정렬 규칙

Data patch starts를 `B = (0, b1, b2, ...)`라 하면 runner는 다음 길이를 전달한다.

```text
(1, lengths(B))
```

HF 구현의 두 patch-ID 경로는 다르게 정렬된다.

- decoder association은 dummy를 제거한 길이를 사용하므로 ID가 `b_j`에서 바뀐다.
- local encoder grouping은 dummy를 포함하므로 해당 ID 전환이 `b_j + 1`에서 일어난다.
- 결과적으로 decoder patch `j`가 참조하는 global patch `j`는 앞 경계 다음 byte부터 현재 decoder boundary byte까지의 causal local state를 집약한다.

예를 들어 data boundaries가 `(0, 6, 12, 18)`이면 첫 8 positions의 ID는 다음과 같다.

```text
encoder: 0 1 1 1 1 1 1 2
decoder: 0 0 0 0 0 0 1 1
```

이는 현재 코드의 임의 선택이 아니라 HF `BltModel.forward`가 encoder에는 전체 lengths를, decoder에는 `patch_lengths[:, 1:]`를 주는 방식과 일치한다.

## 3. Causality test

`tests/test_neural_causality.py`는 다음 두 검사를 고정한다.

1. 위 dummy/encoder/decoder ID 전환을 exact vector로 확인
2. 160-byte prefix가 같고 suffix byte 및 suffix whitespace가 다른 두 input에 W matrix를 각각 구성한 뒤, prefix의 모든 logits가 허용오차 안에서 같은지 확인

두 검사는 통과했다. 특히 position 159 logit은 서로 다른 position 160 byte를 예측하지만 값이 같으므로, suffix-dependent patch boundaries나 local patch reduction이 미래 byte를 노출하지 않는다.

## 4. 연구 해석에 미치는 영향

### 유지되는 식별

F/C/W는 같은 dummy shift, graph, initialization, order, byte stream, global-position count를 공유한다. 따라서 W−C는 여전히 **이 HF graph에 공급한 decoder patch schedule을 observed whitespace로 relocation한 효과**를 식별한다. D/P mechanism controls도 같은 shift를 공유한다.

### 금지되는 과장

W boundary `b`가 whitespace 직후라 해도 global patch가 logit `b−1`에서 다음 byte `x_b`를 예측하기 전에 즉시 갱신되는 것은 아니다. Global representation은 position `b`를 관측한 뒤 decoder association이 바뀐 구간에서 쓰인다. 이것이 BLT patch lag의 한 형태다.

따라서 다음 문장은 금지한다.

- “whitespace 직후 즉시 global Transformer를 실행한다”
- “일반적인 BLT architecture에서 같은 효과가 난다”
- “이 결과가 scratchpad로 lag를 제거한 architecture에도 유지된다”

허용되는 문장은 더 좁다.

> In the Hugging Face BLT encoder/global/decoder shift, a prefix-causal schedule conditioned on observed whitespace changes the quality of equal-rate global-state assignment.

## 5. 결정

학습을 중단하거나 matrix를 바꾸지 않는다. 현재 lengths는 HF API가 요구하는 공식 dummy convention과 일치하고 미래 누출도 없다. 대신 이 one-byte alignment를 methods와 limitations에 공개하고, positive 결과가 나와도 architecture-independent boundary claim을 하지 않는다.
