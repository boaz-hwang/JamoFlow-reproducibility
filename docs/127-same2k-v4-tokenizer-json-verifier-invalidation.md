# Same-2K v4 tokenizer JSON verifier invalidation

> 상태: completed worker, failed pre-metric independent verification; superseded by v5

## 결론

v4 worker는 정상 완료했지만 독립 summarizer가 metric을 읽거나 decision을 만들기 전
learned tokenizer reconstruction 단계에서 중단됐다. 저장 tokenizer와 piece artifact로
재구성한 tokenizer는 2,048개 ordered piece와 모든 score가 같았지만, Hugging Face
`Tokenizer.from_file()`과 fresh constructor의 JSON key serialization order가 달랐다.
v4 verifier는 raw JSON string equality를 요구했으므로 의미상 같은 tokenizer를 잘못
거부했다.

v4 worker artifact는 독립 summary가 없는 비증거 산출물로 보존한다. Worker JSON의
metric payload는 열람하지 않았고 selection decision도 계산하지 않았다.

## 고정된 관찰

- v4 plan commit: `9a7632f`
- v4 plan SHA-256: `2896141f099c5d5e92f9b92b39460067bdfec5aceb964e1b4c6fdbc5c310598b`
- worker SHA-256: `ffd39b9c6da758478de2bf48a27558d1272740eb6de18865ade83b356e8f853c`
- tokenizer SHA-256: `73a4cc7be6b539c6b89d993e3f9bff76af357608253b107f8add92c7574116ee`
- piece artifact SHA-256: `461a1a6722aba837f3f40bb2610a3edd036185157e725344ccfc91feb74c1b13`
- SentencePiece model SHA-256: `1d1d228552c913e13bc3443b79278c95c6e5436b893ef1a98df8eb9d47fbbfc8`
- worker wall time: 약 22분
- independent replay metric 접근: 없음

진단은 tokenizer JSON의 top-level/model 구조, ordered piece, score equality만 확인했다.
모든 2,048개 piece string과 score가 일치했고 차이는 serializer의 object key order뿐이었다.

## v5 교정

v5는 raw `to_str()` bytes를 비교하지 않는다. 두 tokenizer JSON을 parse하고, key-sort와
compact separators를 적용한 canonical semantic object SHA-256을 비교한다. Vocabulary
ID 순서와 model vocabulary array 순서는 JSON array이므로 그대로 보존되며, object key
순서만 identity에서 제거된다.

그 밖의 데이터, model artifact, learned scores, 네 tokenizer 역할, calibration/case
inputs, 3회 encode timing, 10% opportunity gate는 모두 v4와 동일하다. v5도 결과를 보기
전에 새 implementation hash와 plan을 commit한 뒤 worker부터 다시 실행한다.
