# Same-2K v2 byte-fallback 가정 무효화와 v3 교정

> 작성일: 2026-08-14
>
> 상태: failed pre-result protocol; superseded by v3

## 실패 내용

v2 worker는 exact source bytes를 약 23분 동안 정상 학습했지만, SentencePiece model proto를
runtime vocabulary로 투영하기 전 structural audit에서 중단됐다. 256개의 single-byte synthetic
row를 넣었음에도 SentencePiece Unigram pruning이 일부 희귀 byte characters를 vocabulary에서
제거했기 때문이다.

- v2 plan SHA-256:
  `fcc3c490d4a5256560bb7947696a542040fa50eda4725ff9908f64f00b9374fd`
- failed active marker SHA-256:
  `1d9bd4df30ab0fb1bbc3337640a2f3aff09e617ab2dc29cb45a85097f2eb5be4`
- publish된 model/tokenizer/pieces/metrics/result: 없음

token count나 learned pieces가 artifact로 공개되기 전에 `full byte alphabet` invariant가 실패한
것이므로 2K/10% gate와 역할을 바꾸지 않는다.

## 교정

Synthetic occurrence는 single-byte fallback 보장의 충분조건이 아니다. v3는 SentencePiece가
학습한 2,048 non-unk rows를 다음 deterministic projection에 통과시킨다.

1. learned single-byte pieces와 scores를 읽는다.
2. 빠진 byte values를 전부 삽입하며 score는 learned minimum score보다 10 낮게 둔다. 이 score는
   해당 byte가 다른 learned piece로 덮이지 않을 때만 fallback으로 선택되도록 한다.
3. multi-byte learned pieces를 score 내림차순으로 정렬하되 exact tie는 SentencePiece 원래
   순서로 깬다.
4. 상위 1,792개만 유지해 total vocabulary를 exact 2,048로 만든다.
5. mandatory byte pieces를 IDs 0--255로 고정한다.

이 projection은 결과에 따른 후보 선택이 아니라 모든 future corpus에 적용되는 byte-fallback
정의다. 누락 byte 수, 제거한 learned piece 수, fallback score를 metadata에 공개한다.

## protocol 처리

v2 plan과 실패 marker는 보존한다. 구현을 고친 뒤 v2 plan을 재사용하지 않고 v3 plan을 새로
봉인한다. v3는 같은 train/calibration source, 역할, cases, vocabulary size, 10% gate를 유지한다.
