# Same-2K v3 WordPiece runtime invalidation

> 상태: failed pre-result protocol; superseded by v4

## 결론

v3는 metric, tokenizer, vocabulary artifact를 공개하기 전에 중단했다. 실패는
연구 가설이나 10% opportunity gate 때문이 아니라 left-most-longest 대조군을
구현한 Hugging Face WordPiece runtime의 병적인 장문 비용 때문이다.

따라서 데이터, 2,048 vocabulary budget, learned pieces, 네 역할, 8MB calibration
stream, 42 continuation cases, 3회 encode repetition, 10% 이중 gate는 바꾸지 않는다.
v4는 left-most-longest의 의미만 그대로 유지하면서 bounded byte trie로 실행한다.

## 관찰된 실패 경계

- v3 plan commit: `cbe1ba8be84ad19149d985d3d6408e0132e1dac9`
- v3 plan SHA-256: `0e826392def8e40c00f2ede12658ca2c0bbe5e1a9abd2b5c4983feec2549f94e`
- active marker SHA-256: `6fc90fcef10ed1dd68170e4aa6aab623bc093e430647edaa28ade1da33088164`
- 실행 시간: 약 68분
- 공개된 model/tokenizer/metric artifact: 없음
- 중단 위치: `evaluate_tokenizer_opportunity()`의 반복 encode 구간
- 두 차례 독립 process sample: Rust `tokenizers` WordPiece의 substring 탐색

v3의 입력은 `use_regex=False`이므로 최대 262,144-byte 문서 하나가 WordPiece에
하나의 unsplit unit으로 들어간다. WordPiece는 bounded trie를 한 번 걷는 방식이
아니라 남은 긴 substring에서 끝점을 계속 줄이며 vocabulary lookup을 수행한다.
최대 learned piece가 48 bytes뿐인 이 조건에서는 논문식 maximum-munch와 결과는
같더라도 계산 경로가 사실상 quadratic에 가까워진다.

## v4 교정

v4 adapter는 raw UTF-8 bytes에서 직접 동작한다.

1. IDs 0..255의 full byte fallback을 포함한 ordered vocabulary로 trie를 만든다.
2. 현재 byte 위치에서 최대 48 bytes까지만 trie를 전진한다.
3. 마지막으로 관찰한 terminal, 즉 가장 긴 piece를 방출한다.
4. full byte fallback 때문에 모든 byte stream에서 반드시 전진한다.
5. vocabulary serialization과 decode는 기존 canonical HF tokenizer에 위임한다.

복잡도는 emitted boundary마다 최대 piece length로 제한된다. 이는 Length-MAX
논문의 frozen-vocabulary left-most-longest semantics와 맞고, WordPiece wrapper의
장문 구현 병목은 포함하지 않는다.

함께 발견된 verifier transport 결함도 결과 공개 전에 교정했다. Worker의 Python
tuple은 JSON에서 list가 되므로, 독립 replay equality는 wall-clock field를 제거한
뒤 canonical JSON representation에서 비교한다. 숫자, 배열 순서, hash, token count,
vocabulary structure 중 어느 것도 완화하지 않는다.

## 연구 해석

v3는 결과가 아니므로 token-step 우월성이나 열등성의 근거로 사용하지 않는다.
다만 tokenizer 알고리즘의 이론적 복잡도와 실제 library 호출 경로를 구분해야
한다는 systems finding으로 보존한다. 최종 논문에서 left-most-longest를 사용한다면
반드시 bounded trie 구현의 tokenizer wall time을 end-to-end latency에 포함해야 한다.
