# Korean BPE systems frontier preflight protocol

> 작성일: 2026-08-14
>
> 상태: 이미 알려진 16K/32K engineering anchor를 공개한 뒤, 새 2K/4K/8K/64K
> token count와 18개 graph latency를 보기 전 고정

## 목적

Scalar branch는 parameter-matched BPE보다 느려 중단됐다. 그러나 기존 BPE16K와 BPE32K는
depth·width·FFN이 달라 어느 vocabulary/geometry가 이 Mac의 batch-1 frontier인지 알 수
없다. 새 한국어 tokenizer를 약한 기준선과 비교하지 않도록, 먼저 ordinary reversible byte
BPE의 vocabulary–depth systems surface를 측정한다.

이 실험은 새 tokenizer 방법이나 matched-quality 결과가 아니다. Random weights로 graph
실행 비용만 측정한다.

## 고정 grid

- vocabulary: `2,048 / 4,096 / 8,192 / 16,000 / 32,000 / 64,000`
- Transformer depth: `8 / 12 / 16`
- 총 18개 graph
- target parameters: `19,596,096`
- 허용 오차: ±0.5%
- tied input/output embeddings, bias-free Llama, float32 Apple MPS, batch 1
- maximum positions: 512

각 `(vocabulary, depth)`의 hidden/FFN/head geometry는 latency와 quality를 입력받지 않는
정수 grid로 고른다. 순서는 parameter 차이, FFN ratio 3.5와의 거리, head dimension 64와의
거리, 더 넓은 hidden 순이다. Grid와 최종 18개 spec은 plan에 봉인한다.

## Tokenizer

모든 tokenizer는 동일한 clean Korean train split 5,791 documents에서 학습한다.

- Hugging Face `tokenizers==0.22.2`
- ByteLevel initial alphabet 256개 전체
- normalizer 없음
- `add_prefix_space=False`, `use_regex=True`
- minimum frequency 2
- special token과 unknown token 없음
- vocabulary별 두 번 학습해 compact JSON byte identity 확인
- calibration 8,000,000 raw bytes를 decode 및 raw token-byte concatenation으로 왕복 확인
- 같은 calibration text의 CPU encode를 5회 재서 tokenizer 비용을 별도 diagnostic으로 보존

2K/4K/8K/64K count는 plan 작성 전에 보지 않는다. 16K/32K count와 이전 runtime은 이미
알려진 engineering anchor로 plan에 공개한다.

## Cases와 timing

새 tokenizer와 무관한 기존 deterministic document selector로 calibration stream에서
42개 서로 다른 문서를 고른다.

- warmup 6 documents
- measured 36 documents
- prompt 128 raw bytes
- controlled continuation 128 raw bytes
- repetition 3회
- 18 role 순서는 108 measured trial block에서 각 temporal position을 정확히 6회씩 점유하는
  cyclic/reversed schedule

Prompt와 continuation은 각각 독립적으로 reversible encode한다. 이는 실제 generation에서
prompt가 token boundary에서 끝나고 이후 token을 생성하는 조건에 해당한다. Tokenization은
model timer 밖이며, runtime construction·parallel prefill·full vocabulary argmax·cached
teacher-forced decode·MPS synchronization은 timer 안이다.

## Correctness

각 graph는 6개 warmup case에서 다음을 통과해야 한다.

1. full no-cache logits
2. one-token-prefix sequential cache logits
3. parallel prompt prefill + incremental continuation logits
4. PyTorch `atol=1e-4`, `rtol=2e-5`와 같은 reference-side denominator의 normalized worst
   ratio ≤ 1
5. every compared position의 argmax exact equality
6. cache token count exact equality

하나라도 실패하면 timing 결과를 사용하지 않는다.

## 통계와 판정

Repetition은 표본 수로 세지 않고 document 안 median으로 축약한다. Role별 E2E/TTFT/decode
중앙값과 token step을 보고한다. Pairwise 비교는 동일 36개 document의 paired median ratio와
10,000회 prompt bootstrap 95% interval을 사용한다.

가장 낮은 aggregate E2E median role을 `systems-only fastest BPE`로 표시하되 random weights로
quality comparator를 확정하지 않는다. 같은 vocabulary에서 가장 빠른 depth와
token-count/latency Pareto surface를 보존한다.

성공은 다음 단계의 BPE quality-frontier protocol을 작성할 근거일 뿐이다. 새 Korean method의
효율 개선으로 해석하지 않는다.

## Claim 경계

- calibration-development only
- random weights only
- fixed teacher-forced route only
- single Apple MPS session
- model inference timer에서 tokenizer encode 제외
- matched quality, free-running validity, memory, training efficiency 증거 없음
- 결과를 보고 threshold·grid·case를 바꾸지 않음
