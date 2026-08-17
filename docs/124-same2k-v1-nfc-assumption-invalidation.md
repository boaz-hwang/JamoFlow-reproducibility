# Same-2K v1 NFC 가정 무효화와 v2 교정

> 작성일: 2026-08-14
>
> 상태: failed pre-result protocol; superseded by v2

## 무엇이 실패했는가

`same2k-generic-opportunity-v1` plan을 commit한 뒤 deterministic SentencePiece worker를 실행했다.
약 3분 후 train split의 한 문서가 NFC가 아니라는 이유로 trainer-side preflight가 예외를 냈다.
worker는 이 시점에 model proto, tokenizer JSON, pieces, metrics, result를 하나도 publish하지
않았고 `.active` marker만 남겼다.

- v1 plan SHA-256:
  `a338420012f86d687aef9d2a1efbfb9afa9eaadba701fee466b40c466311ab00`
- failed active marker SHA-256:
  `9fc2c5905317990b90d8a70e910fdd874a1ddf38c83780843499eacf40712913`
- 실패 종류: source identity mismatch가 아니라 **새 trainer의 잘못된 NFC-only assumption**

## 왜 normalization하지 않는가

이 연구의 공통 데이터 계약은 pinned UTF-8 bytes를 그대로 모델링하고 raw-byte BPB를 계산하는
것이다. 입력을 NFC로 바꾸면 roundtrip 대상과 byte denominator가 달라져 BPE comparator와
동일한 corpus가 아니게 된다. 올바른 교정은 source를 정규화하는 것이 아니라 arbitrary valid
UTF-8 normalization form을 그대로 byte alphabet에 매핑하는 것이다.

앞선 BPE systems frontier도 normalization을 하지 않는 `train_exact_byte_bpe`를 사용했다.
따라서 byte-Unigram 역시 non-empty valid text만 요구하고 exact `text.encode('utf-8')`를 변환한다.

## protocol 처리

v1 plan은 이미 Git에 봉인됐으므로 수정하거나 삭제하지 않는다. 구현 변경 뒤 v1을 다시
실행하면 pre-result adaptation을 같은 identity로 숨기게 된다. 다음처럼 처리한다.

1. v1은 `failed_before_tokenizer_or_metric_artifact`로 보존한다.
2. NFC guard를 제거하고 arbitrary normalization-form roundtrip test를 추가한다.
3. protocol/plan/artifact/result namespace를 v2로 올린다.
4. 수정된 구현 전체를 먼저 commit한 뒤 새 v2 plan을 봉인한다.
5. v2 worker는 v1의 실패 marker나 산출물을 입력으로 사용하지 않는다.

이 수정은 결과를 보고 gate를 바꾼 것이 아니다. token count나 vocabulary가 만들어지기 전에
발견된 source compatibility bug를 고친 것이며, roles, 2K budget, 10% gate, cases, downstream
decision은 그대로 유지한다.
