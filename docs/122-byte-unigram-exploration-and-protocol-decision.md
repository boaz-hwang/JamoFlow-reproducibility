# Byte-Unigram 탐색 결과와 재현 protocol 결정

> 작성일: 2026-08-14
>
> 상태: unsealed exploratory result; selection 또는 publication evidence 아님

> 후속 결과: deterministic full-corpus gate의 최종 결과는
> [docs/129](./129-same2k-generic-opportunity-result-and-length-gain-pivot.md)에 기록했다.
> 탐색의 HF Unigram과 sealed SentencePiece control은 서로 다른 vocabulary learner이며,
> 최종 selection evidence는 v6만 사용한다.

## 결론

같은 2,048 vocabulary에서 generic likelihood tokenizer가 BPE의 한국어 token 수를 얼마나 더
줄일 수 있는지 빠르게 확인했다. Hugging Face `tokenizers==0.22.2`의 byte-Unigram을 full
5,791-document train split에 학습하고, 기존 contiguous calibration의 7,999,999 predicted raw
bytes를 평가한 결과는 다음과 같다.

| exploratory tokenizer | pretokenizer | calibration tokens | BPE 대비 | bytes/token | vocab used | train time |
|---|---|---:|---:|---:|---:|---:|
| sealed BPE-2K | ByteLevel regex | 2,263,476 | 기준 | 3.534 | 미측정 | 기존 artifact |
| HF Unigram-2K | no regex | 2,173,590 | **-3.97%** | 3.681 | 1,972/2,048 | 585.5 s |
| HF Unigram-2K | regex | 2,328,984 | **+2.89%** | 3.435 | 1,972/2,048 | 67.2 s |

두 Unigram 모두 exact text roundtrip을 통과했다. no-regex 역할은 cross-boundary 후보를 허용해
BPE보다 짧아졌지만, 논문 진입 기준인 10% step headroom에는 크게 못 미쳤다. regex 역할은
BPE보다 오히려 길었다. 이 결과는 다음 두 사실을 지지한다.

1. BPE의 merge objective를 likelihood로 바꾸는 것만으로는 현재 2K Korean baseline을 10%
   이상 줄일 가능성이 낮다.
2. pretokenization 경계가 작은 vocabulary에서 큰 영향을 준다. 다만 경계를 없애 얻은 이득도
   약 4%뿐이므로, cross-space token만 추가하는 방식으로는 목표에 부족하다.

## 왜 이 결과를 그대로 봉인하지 않는가

이 실행은 문헌 보정 중 수행한 opportunity exploration이며 plan을 먼저 봉인하지 않았다.
더 중요하게, `tokenizers`의 `UnigramTrainer`는 seed를 노출하지 않고 같은 tiny corpus와 single
Rayon thread에서도 tokenizer JSON hash가 반복 실행마다 달라졌다. 즉 현재 API로 학습한
vocabulary를 deterministic reference artifact라고 부를 수 없다.

탐색 artifact를 저장하지 않았고 위 JSON hashes는 실행 로그 진단일 뿐이다.

- no-regex tokenizer JSON SHA-256:
  `831de931e4485adf0e704bf6c7a9158f38ec376bdb41fb84d8c881985294235f`
- regex tokenizer JSON SHA-256:
  `97c0c6cabb12ef19220834f20b23d30cb576a5fae2475e740b00a094635f7d15`

이 수치를 후보 선택이나 quality claim에 사용하지 않는다. 오직 다음 sealed protocol의 역할과
중단 규칙을 정하는 사전 탐색으로 공개한다.

## deterministic control로 바꾸는 방법

새 control은 `sentencepiece==0.2.1`을 vocabulary/score estimator로 사용한다.

1. raw UTF-8 bytes를 GPT-2의 whitespace-free 256-character alphabet으로 일대일 변환한다.
2. 각 byte character를 하나의 synthetic one-character row로 넣어, 인공적인 multi-byte
   n-gram 없이 full byte fallback을 강제한다.
3. identity normalization, no whitespace/script/number split, no shuffle, one CPU thread,
   maximum piece 48 bytes로 Unigram을 학습한다.
4. SentencePiece의 mandatory `<unk>` 한 class를 제외하기 위해 2,049 pieces를 학습한 뒤,
   256 byte pieces를 IDs 0--255로 재배열하여 exact 2,048 classes를 만든다.
5. 그 pieces와 scores를 pinned Hugging Face Unigram Viterbi runtime에 설치한다.

작은 corpus에서 세 번 반복한 SentencePiece model proto hash가 일치했고, repository unit
test에서는 별도 두 번의 학습 결과가 pieces, scores, model proto, metadata 수준에서 모두
같음을 검증한다. 이 deterministic 역할을 sealed opportunity gate에서 다시 full corpus로
학습·저장한다.

## 다음 단계에 미치는 영향

계획을 전면 변경할 필요는 없다. 다만 generic likelihood control은 탐색상 10% gate를 통과하지
못했으므로, full model training으로 바로 보내지 않는다. 다음 token-only gate는 다음 질문을
순서대로 답해야 한다.

1. deterministic Byte-Unigram 결과가 탐색의 약 4% 감소를 재현하는가?
2. 같은 learned vocabulary를 minimum-token DP로 바꿨을 때 segmentation만으로 추가 headroom이
   얼마나 생기는가?
3. length-gain vocabulary가 BPE 대비 10%를 넘는가?
4. Korean completeness constraint가 그 headroom을 유지하면서 generic vocabulary의 구조적
   위험을 줄이는가?

1--3 어디에서도 BPE 대비 10% step headroom이 나오지 않으면 same-2K tokenizer branch를
full training 전에 중단한다. 반대로 DP만 10%를 넘고 paper-style longest-match는 못 넘으면,
주장은 Length-MAX reproduction이 아니라 segmentation algorithm의 효과로 제한한다.
