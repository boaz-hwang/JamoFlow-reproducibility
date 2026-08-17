# BPE body-match correction

> 작성일: 2026-08-11
> 상태: **publication-scale tokenizer/model 학습 전 고정**
> 교정 대상: [dual-BPE sealed-test correction](./61-dual-bpe-sealed-test-correction.md) §4
> 영향 범위: 16K BPE stress graph만 변경; 32K primary BPE와 compact Phase 3는 불변

## 1. 총 파라미터 재매칭이 분리하지 못한 것

16K stress control의 목적은 32K BPE에서 vocabulary와 tied input/output matrix를 줄였을 때 생기는 실제 latency 이점을 직접 통제하는 것이다. 그런데 기존안처럼 줄어든 embedding 파라미터를 더 큰 hidden/FFN body로 다시 채우면 두 변화가 동시에 일어난다.

1. vocabulary/output projection은 작아진다.
2. attention/FFN body는 커지고 head geometry도 바뀐다.

이때 16K가 빠르거나 느린 이유를 output head와 sequence length trade-off로 분리할 수 없다. “작은 vocabulary만으로 더 빠른 BPE를 만들 수 있는가”라는 stress 질문에는 total-parameter matching보다 body matching이 맞다.

## 2. 두 BPE의 서로 다른 역할

- **32K primary BPE:** Candidate와 total trainable parameters를 1% 이내로 맞춘 deployment comparator다.
- **16K stress BPE:** 각 target의 32K BPE와 hidden size, FFN size, layer 수, attention head 수, context와 모든 비-embedding 설정을 동일하게 둔다. Vocabulary와 tied embedding/output rows만 16K로 줄인다.

따라서 16K는 의도적으로 parameter 수가 더 적다. 이는 불공정하게 약한 baseline을 만들기 위한 것이 아니라, 더 작은 head가 실제 inference에서 유리할 수 있다는 반론에 유리한 control이다. Candidate가 quality를 유지하면서 이 더 작은 model보다도 빨라야 vocabulary-size confound를 제거한 broad claim이 가능하다.

## 3. 고정 graph

| Vocabulary | Target | width / heads / layers / FFN | exact params | Candidate보다 작은 비율 |
|---:|---:|---:|---:|---:|
| 16K body-matched | 50M | 448 / 7 / 12 / 1,600 | 42,617,792 | 14.462% |
| 16K body-matched | 75M | 608 / 8 / 12 / 1,792 | 66,710,368 | 12.788% |
| 16K body-matched | 100M | 704 / 11 / 12 / 2,048 | 86,975,680 | 11.613% |
| 32K parameter-matched | 50M | 448 / 7 / 12 / 1,600 | 49,785,792 | 0.076% mismatch |
| 32K parameter-matched | 75M | 608 / 8 / 12 / 1,792 | 76,438,368 | 0.071% mismatch |
| 32K parameter-matched | 100M | 704 / 11 / 12 / 2,048 | 98,239,680 | 0.166% mismatch |

각 target에서 두 BPE의 parameter 차이는 정확히 `(32,000 − 16,000) × hidden_size`여야 한다. `src/jamoflow/publication_bpe.py`는 이 식, analytical Llama parameter count, 32K parameter-grid 선택과 16K body identity를 모두 검사한다.

## 4. 학습·비교 공정성

두 BPE는 다음을 공유한다.

- 같은 clean Korean train document 순서와 paired seeds
- 같은 16K/32K tokenizer 학습 source, byte alphabet, normalization 부재와 pretokenizer family
- 같은 optimizer family와 checkpoint byte budgets
- data-matched와 architecture별 analytical train-FLOP-matched checkpoint
- 같은 raw UTF-8 test documents, downstream prompts와 actual-inference cases

총 parameter가 다르므로 표에는 embedding parameter, non-embedding parameter, train FLOPs, 실제 train wall time, peak memory와 inference latency를 분리해 공개한다. 16K가 quality floor를 못 넘으면 빠르다는 사실만으로 강한 comparator가 되지 않는다. 반대로 quality-qualified 16K를 candidate가 actual gate에서 이기지 못하면 32K만 이긴 결과를 broad claim으로 올리지 않는다.

## 5. 왜 세 번째 16K total-matched model을 core에 넣지 않는가

16K total-matched graph는 equal-parameter sensitivity로는 의미가 있지만 output-head causal control과 architecture가 겹친다. Core estimand는 이미 32K equal-parameter 비교와 16K same-body 비교로 분리된다.

- Equal total parameters: candidate 대 32K BPE
- Vocabulary/head intervention: 32K body 대 같은 body의 16K BPE
- Strong deployment gate: candidate가 raw, 16K, 32K를 모두 통과

16K total-matched variant를 결과에 따라 추가하면 baseline family를 사후 확장하게 된다. 따라서 core 12-run campaign에는 넣지 않는다. 외부 compute로 별도 sensitivity를 실행하더라도 broad gate를 대체하지 않고 부록에만 둔다.

## 6. Claim rule

이 교정은 candidate에게 더 쉬운 비교가 아니다. 16K control이 11.6–14.5% 더 작아질 수 있음을 명시적으로 허용하고도 actual latency에서 이겨야 한다. 그 결과가 없으면 다음처럼 제한한다.

- 32K만 이김: equal-parameter standard-BPE 결과
- 16K만 이김: vocabulary-specific 결과
- 둘 다 이김: output-head 크기 선택에 견디는 deployment-level 결과 후보

Publication-scale 결과는 아직 없으므로 이 변경은 outcome-adaptive correction이 아니다.
