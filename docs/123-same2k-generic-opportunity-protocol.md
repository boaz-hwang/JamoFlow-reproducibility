# Same-2K generic tokenizer opportunity protocol

> 작성일: 2026-08-14
>
> 상태: v6 independently replayed development result 완료
>
> 결과: [docs/129](./129-same2k-generic-opportunity-result-and-length-gain-pivot.md)

## 목적

같은 2,048 vocabulary와 동일한 향후 2K×8L model graph에서, vocabulary objective와 frozen
segmentation만 바꾸어 BPE 대비 최소 10%의 decode-step headroom이 존재하는지 먼저 판정한다.
이 실험은 model을 학습하지 않으며 Korean-aware 방법도 평가하지 않는다. Generic 역할조차
10% 기회를 만들지 못하면, quality training 전에 length-gain vocabulary를 별도로 설계하거나
same-2K branch를 중단한다.

## 고정 역할

1. `byte_bpe_2k`: 앞선 BPE frontier의 exact tokenizer artifact.
2. `byte_unigram_2k_scored`: deterministic SentencePiece가 학습한 pieces와 likelihood scores를
   HF Unigram Viterbi runtime에 설치한 generic control.
3. `byte_unigram_vocab_2k_leftmost_longest`: 2와 같은 ordered pieces를 논문 Length-MAX의
   frozen application에 맞춘 maximum-munch trie로 적용한다.
4. `byte_unigram_vocab_2k_minimum_token_dp`: 2와 같은 ordered pieces에 모두 같은 score를 주어
   global minimum-token DP로 적용한다.

2와 3의 차이는 learned score에 따른 segmentation 효과다. 3과 4의 차이는 vocabulary를 완전히
고정한 segmentation-only 효과다. 3은 Length-MAX vocabulary reproduction이 아니며, 오직
left-most-longest application control이다.

## deterministic Byte-Unigram training

- source: pinned HPLT Korean train split 5,791 documents, document 순서 고정
- Unicode normalization은 하지 않는다. NFC/NFD 여부를 포함한 exact UTF-8 source bytes를 보존한다.
- raw UTF-8 bytes를 GPT-2 byte-to-Unicode alphabet으로 일대일 변환
- 256개의 one-character synthetic rows로 byte fallback 강제
- SentencePiece 0.2.1, Unigram, identity normalization
- whitespace/script/number split 없음, shuffle 없음, one CPU thread
- target SentencePiece pieces 2,049: `<unk>` 1개를 제거해 model vocabulary는 exact 2,048
- maximum piece 48 bytes, initial seed size 1,000,000, shrinking 0.75, two subiterations
- SentencePiece가 synthetic row에도 불구하고 rare single byte를 prune할 수 있으므로, 누락된
  bytes를 결정적으로 삽입하고 같은 수의 최저-score learned multi-byte pieces를 제거한다.
- mandatory bytes는 IDs 0--255로 재배열하고, 나머지 1,792 slots는 learned score 상위
  multi-byte pieces로 채운다. score tie는 SentencePiece 원래 순서를 따른다.
- trained model proto, ordered pieces/scores, runtime tokenizer JSON을 모두 hash 봉인

Hugging Face UnigramTrainer는 같은 입력에서도 vocabulary JSON이 재현되지 않았기 때문에 sealed
worker에서는 사용하지 않는다. 그 결과는 docs/122의 disclosed exploration일 뿐이다.

초기 v1 plan은 source가 모두 NFC라는 잘못된 사전조건 때문에 worker가 training 도중
fail-closed로 멈췄다. 결과를 만들지 않았고, 구현을 바꾼 뒤 같은 plan을 소급 재사용하지 않고
v2 namespace를 새로 봉인했다. v2도 SentencePiece가 rare byte pieces를 유지한다는 가정이
틀려 structural audit에서 결과 생성 전 중단됐다. 위 explicit fallback projection을 추가한 뒤
v3 namespace를 새로 봉인했다. v3는 WordPiece의 unsplit 장문 substring 병목 때문에 metric
공개 전에 중단했고 bounded trie로 v4를 만들었다. v4와 v5 worker는 완료했지만 독립 verifier의
JSON key-order 및 loader float round-trip identity 결함을 각각 metric 접근 전에 발견했다.
deployable from-file runtime을 worker와 verifier 양쪽에 고정한 v6만 최종 evidence로 사용한다.
자세한 내용은 docs/124--128에 기록한다.

## 공통 입력과 측정

- contiguous calibration: exact 8,000,000 raw bytes. UTF-8 마지막 incomplete suffix는 encoding
  밖에 두고 그 길이를 보고한다.
- inference cases: 기존 BPE systems plan의 outcome-independent, one-document-per-case selector를
  그대로 재구성한 6 warmup + 36 measured cases.
- 각 case는 128-byte prompt와 별도 128-byte continuation이다.
- calibration token count, vocabulary utilization, piece byte length, strict UTF-8/Hangul/cross-eojeol
  vocabulary diagnostics를 보고한다.
- prompt, separately encoded continuation, joint prompt+continuation token count를 모두 보고한다.
- encode throughput은 세 회 wall-clock median이며 selection criterion이 아닌 diagnostic이다.
- worker 뒤 summarizer가 모든 token IDs/counts/structure를 독립 재생한다. 시간은 정확히 같을 수
  없으므로 worker와 verifier throughput을 둘 다 보존한다.

## Gate

각 generic 역할은 exact BPE 대비 다음을 **둘 다** 만족해야 one-seed quality training 자격을
얻는다.

1. calibration token count reduction ≥ 10%
2. 36 measured continuation의 총 separately encoded token count reduction ≥ 10%

Roundtrip 또는 exact 2,048 vocabulary가 깨지면 결과 생성 자체를 거부한다. Token-only gate는
quality noninferiority나 actual model latency를 대신하지 않는다. 통과 역할도 동일 128M raw-byte
학습과 trained-model MPS E2E에서 다시 검증해야 한다.

## 결과별 다음 행동

- scored Unigram 통과: likelihood control을 one-seed quality로 보낸다.
- 같은 vocabulary의 longest-match/DP만 통과: segmentation을 독립 mechanism으로 평가하되
  Length-MAX vocabulary claim은 하지 않는다.
- 세 generic 역할 모두 실패: model training을 시작하지 않고 length-weighted vocabulary가
  실제 10% headroom을 만드는지 별도 sealed gate를 설계한다.
- generic 통과 뒤 Korean constraint는 token 수가 아니라 generic 대비 quality–latency Pareto를
  개선할 때만 주 방법으로 승격한다.

## Claim boundary

이 결과는 calibration-development token-only upper bound다. Publication comparator, model
quality, actual model latency, Korean-aware novelty, free-running generation을 주장하지 않는다.
