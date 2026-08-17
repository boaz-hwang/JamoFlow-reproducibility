# Phase 3 addendum: paired NFC/NFD stress without an oracle

> 상태: **Phase 3 normalization 평가 전에 고정**  
> 고정 시점: 2026-08-10  
> 선행 문서: [Phase 3 confirmatory protocol](./22-phase3-confirmatory-protocol.md), [Phase 3 data audit](./23-phase3-data-audit.md)  
> 주의: 이 addendum을 작성할 때 seed 1,729의 F 학습만 끝났고 C/W 비교와 Phase 3 normalization 평가는 아직 없었다.
> 결과 전 provenance 보강: [normalization provenance addendum](./40-phase3-normalization-provenance-addendum.md)

## 1. 왜 세부 규칙을 추가로 고정하는가

Phase 3 protocol은 HPLT3 test를 NFC와 NFD로 변환해 paired stress test를 한다고 정했지만, 다음 구현 세부사항은 고정하지 않았다.

1. 표현별 byte 길이가 다를 때 같은 원문 범위를 보장하는 방법
2. 512-byte window의 마지막 incomplete row 처리
3. representation-invariant denominator
4. Phase 2의 non-causal Hangul-unit oracle 포함 여부

이 선택들은 NFD 결과를 크게 바꿀 수 있다. 따라서 normalization 결과를 계산하기 전에 아래와 같이 고정한다. 이 문서는 Gate I/J/K의 가설이나 threshold를 바꾸지 않는다.

## 2. 연구 질문의 범위

이 실험이 묻는 질문은 하나다.

> 사실상 NFC인 Korean web text로 학습한 동일 BLT checkpoint와 동일한 deployable causal patch policy가, 의미상 같은 source를 canonical NFD로 표현했을 때 얼마나 민감한가?

다음을 묻지 않는다.

- NFD가 자연 한국어 분포에서 NFC보다 우월한가
- Jamo-level modeling이 일반적으로 유리한가
- non-causal Hangul-unit boundary가 deployable한가
- normalization stress가 main natural-text 평균에 포함돼야 하는가

Phase 2에서 NFD `L+V+(T)` unit boundary는 prefix-causal하지 않고, exact 43-patch rate에서 full-unit preservation과 rate가 구조적으로 충돌함을 이미 확인했다. 따라서 Phase 3 stress에는 oracle과 compatibility-jamo condition을 넣지 않는다.

## 3. Source와 변환

1. Phase 3 primary와 같은 `data/processed/hplt3-korean-phase3/ko.jsonl` test split을 사용한다.
2. `build_neural_stream(..., byte_limit=16_000_000, sequence_length=512)`로 primary test source를 재구성한다.
3. stream 끝의 incomplete UTF-8 codepoint가 있으면 strict-decodable prefix만 source text로 삼는다. 버린 byte 수를 보고한다.
4. **동일한 전체 source text**에 Python `unicodedata.normalize`의 `NFC`와 `NFD`를 각각 한 번 적용한다.
5. 변환된 두 byte stream은 자르지 않는다. 마지막 incomplete 512-byte row만 LF byte로 채운다.
6. 마지막 row의 artificial padding을 target loss에서 제외한다. Padding byte는 actual source 뒤에만 있으므로 actual prefix의 causal boundary decision에는 관여하지 않는다.

원문, 변환문, record 식별자는 tracked artifact에 저장하지 않는다. Source/condition stream SHA-256과 aggregate length만 기록한다.

## 4. Window와 target convention

학습·primary 평가와 동일하게 각 512-byte row의 position 0은 context이고 positions 1–511만 next-byte target이다. 따라서 표현이 길어져 row 수가 늘면 row-leading unscored byte 수도 늘어난다. 이를 숨기지 않고 condition별로 다음을 보고한다.

- actual transformed bytes
- padded bytes와 terminal padding bytes
- rows
- scored actual target bytes
- `scored_actual_bytes / actual_transformed_bytes`

NFC와 NFD의 target coverage 차이가 있으므로 `bits/source-codepoint`를 완전한 arithmetic codelength라고 부르지 않고 **scored bits per source codepoint**라고 부른다.

## 5. Policy와 checkpoint

대상은 같은-rate deployable structural policy 세 개뿐이다.

- F: `fixed_byte_6`
- C: `causal_codepoint_grid`
- W: `causal_whitespace_grid`

각 row는 Phase 3와 같은 exact 86 data patches를 가져야 한다. 학습된 NFC checkpoint를 그대로 사용하며 fine-tuning, threshold calibration, normalization-specific routing을 하지 않는다.

초기 분석은 seed 1,729 / 2,718 / 31,415로 수행한다. Gate I를 통과해 confirmation checkpoint가 실제로 생성되면 동일 script로 seed 57,721 / 65,537을 추가할 수 있다. 첫 세 seed 결과만 보고 추가 seed 여부를 normalization 성능에 따라 정하지 않는다.

## 6. 고정 metric

Condition·policy·seed마다 다음을 보고한다.

1. BPB: scored actual target byte당 NLL bits
2. scored bits per source UTF-8 byte
3. scored bits per source Unicode codepoint
4. scored bits per source precomposed Hangul syllable
5. elapsed evaluation throughput은 diagnostic으로만 기록

`source UTF-8 byte`, `source codepoint`, `source precomposed Hangul syllable` denominator는 NFC/NFD에 공통인 변환 전 source에서 센다.

Seed-paired aggregate는 다음을 보고한다.

- policy별 `NFD − NFC` absolute BPB
- policy별 `(NFD / NFC) − 1` scored-bits/source-codepoint 증가율
- NFC와 NFD 각각의 `W − C`, `W − F`
- mean, standard deviation, paired t 95% interval, seed별 값

세 seed의 stress test에 별도 pass/fail gate를 만들지 않는다. 특히 NFD degradation을 Gate I OOD guard에 넣지 않는다.

## 7. 해석 guardrail

1. NFD는 training distribution에서 크게 벗어난 synthetic canonical-equivalence stress다.
2. NFC/NFD의 BPB는 서로 다른 byte alphabet sequence length를 조건으로 하므로 BPB만으로 총 representation cost를 비교하지 않는다.
3. scored bits/source-codepoint도 window-leading target omission을 포함한 모델 평가량이지 lossless compressor의 완전 codelength가 아니다.
4. W가 NFD에서 C보다 나쁘더라도 whitespace 자체의 causal validity가 깨지는 것은 아니다. Codepoint geometry와 학습 분포가 함께 변한 결과다.
5. W가 NFD에서 좋더라도 이를 Jamo-aware architecture evidence로 부르지 않는다. W는 Jamo composition state를 사용하지 않는다.
6. Phase 3 method claim과 publication-scale 진입 여부는 사전등록된 natural-text Gate I/J/K만 결정한다.

## 8. 재현 산출물

Tracked aggregate:

- 실행 manifest와 source/condition stream hash
- condition geometry와 patch diagnostics
- seed/policy별 scalar report
- aggregate summary

Full summarizer는 processed source부터 condition stream, target mask, F/C/W matrix, primary checkpoint와 모든 보고 metric을 독립 재구성한다. Initial 3 또는 final 5의 완전한 seed set만 aggregate로 승격한다.

Ignored local artifact:

- seed/policy/condition별 per-sequence NLL과 target count
- model checkpoint

Raw source 및 normalized text는 어느 산출물에도 복제하지 않는다.
