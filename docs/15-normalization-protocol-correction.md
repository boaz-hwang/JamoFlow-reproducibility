# Protocol correction: NFD Hangul-unit boundaries are not prefix-causal

> 작성일: 2026-08-10  
> 상태: **normalization 결과 확인 전 고정**  
> 수정 대상: [Phase 2 protocol](./10-phase2-korean-causal-protocol.md) §8, Gate G

## 0. 발견한 설계 결함

원 protocol은 canonical NFD에서 `L + V + optional T`를 한 Hangul unit으로 보고 unit 완료 뒤에만 boundary를 허용하는 `hangul_unit_grid`를 제안했다. 그러나 이 boundary mask는 일반적인 next-byte generation에서 prefix-causal하지 않다.

Prefix가 `L+V`까지 왔을 때 가능한 해석은 둘이다.

```text
L V       → open syllable complete
L V T     → final consonant still follows
```

`V` 직후에는 다음 codepoint가 `T`인지 알 수 없다. 다음 symbol을 본 뒤 `V` 직후 위치로 boundary를 소급해서 삽입하는 것은 불가능하다. 따라서 다음 세 방식 중 하나가 필요하다.

1. `V` 뒤에 항상 boundary: `LVT`의 T를 분리하므로 full Hangul unit이 아님
2. 다음 codepoint까지 기다린 뒤 boundary: 다음 unit 일부를 이전 patch에 포함하므로 exact unit boundary가 아님
3. one-codepoint lookahead로 `V` 뒤 boundary 결정: full unit이지만 prefix-causal하지 않음

원 protocol은 이 ambiguity를 명시하지 않았다. 이를 수정하지 않고 결과를 내면 causal claim과 모순된다.

## 1. 수정된 지위

`hangul_unit_grid`를 다음 이름과 지위로 바꾼다.

> `oracle_hangul_unit_grid`: complete transformed sequence를 보고 canonical `L+V+optional T` group을 만든 뒤 candidate mask를 구성하는 **non-causal lookahead oracle**

이 oracle은 다음 질문만 답한다.

> NFD에서 Hangul unit을 보존할 수 있는 이상적 boundary 정보가 있다면, NFC-trained compact model의 quality가 codepoint-only grid보다 좋아지는가?

다음을 주장하지 않는다.

- deployable streaming patcher
- causal inference speedup
- Gate H scale-up method
- grammar engine이 공짜로 해결하는 문제

Oracle이 좋아도 실제 method는 delayed patch assignment, scratchpad, bidirectional local encoder, 또는 explicit syllable emission 같은 별도 architecture가 필요하다.

## 2. 고정 evaluation data

Primary Korean test stream의 strict-decodable prefix를 source text로 사용한다. 같은 source text에 다음 결정적 변환을 적용한다.

1. `original`
2. Unicode NFC
3. canonical NFD
4. precomposed modern Hangul syllable를 compatibility-jamo L/V/(T) sequence로 바꾼 stress condition

각 transformed stream은 complete UTF-8 codepoint에서 row가 시작·끝나도록 Phase 2b aligned packer로 pack한다.

- 최대 raw bytes per row: 256
- newline padding: 일반 row당 0–3 bytes
- 마지막 253 bytes 미만 tail 폐기
- raw/inserted/dropped bytes 기록

이 선택은 normalization 간 byte length가 달라도 같은 semantic source prefix를 최대한 유지하고, partial UTF-8 edge를 metric에서 제거하기 위한 것이다. Primary arbitrary-packing BPB와 직접 합치지 않는다.

## 3. Policies와 checkpoints

다섯 primary seeds의 checkpoint를 고정해 추가 학습 없이 평가한다.

1. C0 `fixed_byte_6`
2. C1 `causal_codepoint_grid`
3. Phase 2b `causal_whitespace_grid`
4. `oracle_hangul_unit_grid` using the C1 checkpoint

Oracle은 별도 학습 모델이 아니다. C1 checkpoint의 inference patch matrix만 바꾼다.

NFC에서 `oracle_hangul_unit_grid`와 C1 matrix는 byte-for-byte 같아야 한다. 이 invariant가 깨지면 결과를 폐기한다.

## 4. Oracle unit definition

- precomposed U+AC00–U+D7A3: 한 codepoint가 한 unit
- canonical leading Jamo L: 뒤의 V와 optional T를 함께 한 unit
- 고립된 V/T 또는 malformed order: 각 codepoint가 한 unit
- non-Hangul: 각 codepoint가 한 unit
- newline padding: 각 codepoint가 한 unit

Oracle candidate mask는 complete sequence lookahead로 unit 끝에만 1을 둔다. Absolute 43-patch causal grid target 뒤 첫 oracle candidate를 선택하지만 candidate mask 자체가 non-causal하므로 전체 policy는 oracle이다.

## 5. Metrics

- BPB on transformed bytes
- total bits per represented original source codepoint
- total bits per represented original precomposed Hangul syllable
- boundary-inside-codepoint rate
- boundary-inside-oracle-Hangul-unit rate
- aligned-pack insertion/drop rate

Inserted newline NLL도 total에 포함하며 삽입 비율을 함께 보고한다. Original-codepoint denominator는 transformed raw prefix에 완전히 포함된 source codepoint 수를 cumulative transform lengths로 계산한다.

## 6. 수정된 Gate G

원 Gate G의 1% 조건은 **opportunity diagnostic**으로만 유지한다.

- NFD에서 oracle이 C1보다 bits/original-codepoint를 1% 이상 개선
- NFC matrix identity invariant 통과

두 조건이 맞으면:

> “causal Hangul-unit policy를 scale-up에 포함”이 아니라 “NFD unit preservation을 위한 별도 causal architecture 연구를 열 가치가 있음”으로 판정한다.

두 조건이 맞지 않으면 normalization은 evaluation item으로만 남긴다.

## 7. Compatibility Jamo 해석

Compatibility-jamo 변환은 canonical equivalence가 아니고 canonical L/V/T block 정보도 직접 제공하지 않는다. Oracle은 이를 각 codepoint unit으로 처리한다. 이 condition은 quality equality나 Gate G에 사용하지 않고 representation stress test로만 보고한다.

## 8. Claim correction

이 수정은 negative result를 피하기 위한 조건 변경이 아니라, 실험 전 causal implementability audit에서 발견한 논리적 오류 수정이다. 원 protocol의 해당 문장은 최종 논문에서 그대로 인용하지 않고, 이 correction과 함께 투명하게 설명한다.
