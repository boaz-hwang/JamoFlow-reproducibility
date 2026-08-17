# Same-2K v5 deployable runtime invalidation

> 상태: completed worker, failed pre-metric independent verification; superseded by v6

## 결론

v5도 worker는 완료했지만 independent summarizer가 metric을 읽기 전 tokenizer
reconstruction gate에서 중단됐다. v4 진단은 저장 JSON payload와 fresh constructor를
비교해 두 객체가 의미상 같음을 확인했지만, 실제 summarizer는 저장 JSON을
`Tokenizer.from_file()`로 로드한 뒤 다시 직렬화했다. Rust loader는 2,048 score 중
356개를 약 1 ULP 다른 decimal로 표현한다.

Piece 문자열과 ID 순서에는 차이가 없었다. 그러나 실제 공개·배포될 artifact는
from-file runtime이므로, 이 차이를 tolerance로 숨기거나 fresh runtime만 평가하는 대신
worker와 verifier 모두 artifact round-trip runtime을 평가해야 한다.

Worker 내부 metric은 계속 열람하지 않았고, v5 decision/result artifact도 없다.

## v6 교정

1. SentencePiece와 score projection으로 canonical tokenizer JSON bytes를 만든다.
2. artifact를 publish하기 전에 그 JSON bytes를 `Tokenizer.from_str()`로 다시 로드한다.
3. worker의 scored-Unigram 역할은 이 deployable runtime을 사용한다.
4. verifier는 raw artifact JSON payload와 pieces/scores로 만든 fresh JSON의 canonical
   semantic hash를 비교해 artifact construction을 검증한다.
5. verifier의 실제 token counts는 `Tokenizer.from_file()` runtime으로 재계산한다.
6. Worker와 verifier token-stream hashes/counts는 exact equality를 유지한다. 따라서
   1 ULP loader 차이가 segmentation에 영향을 주면 숨겨지지 않는다.

데이터, vocabulary, structural segmentation, calibration/case coordinates, repetition,
10% gate는 v3-v5와 동일하다. 이 수정은 배포 artifact의 재현성을 강화할 뿐 후보 선택
기준을 바꾸지 않는다.
