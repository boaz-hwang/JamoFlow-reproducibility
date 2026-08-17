# Raw-context-matched rolling BPB correction

> 작성일: 2026-08-12
> 상태: **publication tokenizer·model 학습 및 held-out BPB 결과 전 고정**
> 교정 대상: [publication comparator protocol](./48-publication-comparator-and-downstream-protocol.md) §8
> 우선순위: 평가 공정성·재현성·증거 lineage > 평가 실행 시간

## 1. 발견한 평가 문맥의 빈틈

기존 문서는 두 문장을 동시에 두었다.

1. Candidate는 512 raw-byte 문맥, BPE는 최대 512 token 문맥을 가진다.
2. Main BPB는 document-contiguous streaming으로 각 document의 첫 unit만 제외한다.

두 번째 문장은 유한 문맥 model에서 document를 어떻게 자르고, BPE token이 window 경계를 가로지를 때 무엇을 점수화하며, 양쪽이 실제로 몇 raw bytes의 과거를 보는지 정의하지 않았다. 단순한 model-native 512-unit sliding evaluation은 BPE에 512 token, 즉 대개 512보다 훨씬 많은 raw bytes를 허용한다. 반대로 같은 non-overlap 512-byte window를 독립 encode하면 BPE와 byte model이 매 window에서 서로 다른 길이의 첫 unit을 잃는다. 어느 방식도 “같은 문맥에서의 tokenizer-independent BPB”라고 자동으로 볼 수 없다.

[BLT §4.3](https://arxiv.org/html/2412.09871#S4.SS3)은 큰 patch model에 긴 문맥 이점을 주지 않도록 batch의 **기대 raw bytes**를 일정하게 유지했다고 명시하고, [§4.6](https://arxiv.org/html/2412.09871#S4.SS6)은 NLL을 raw bytes로 나누는 BPB를 정의한다. 그러나 공개 BLT 구현 commit [`9774ed4`](https://github.com/facebookresearch/blt/tree/9774ed4fcc78313f9f218295f3d7e4decdadf2ae)의 [PPL evaluator](https://github.com/facebookresearch/blt/blob/9774ed4fcc78313f9f218295f3d7e4decdadf2ae/bytelatent/eval.py#L203-L213)는 `bytes`와 `blt`만 처리하고 tokenized baseline은 `NotImplementedError`다. 따라서 공개 구현이 JamoFlow의 cross-tokenization 문맥 규약을 대신 검증해 주지 않는다.

## 2. Main estimand: pairwise natural-unit raw-capped rolling BPB

Core comparison 세 개를 따로 만든다.

1. `candidate` 대 `raw_byte_reference`
2. `candidate` 대 `byte_bpe_16000_body_matched`
3. `candidate` 대 `byte_bpe_32000`

각 pair의 held-out document마다 다음 순서를 고정한다.

1. Strict UTF-8·NFC 원문을 comparator의 자연 unit으로 분해한다.
   - raw comparator: 한 byte가 한 unit
   - BPE comparator: 해당 tokenizer로 **full document를 joint encode**한 token이 unit
2. Natural-unit bytes를 이어 붙였을 때 원문과 byte-for-byte 같지 않으면 중단한다.
3. Natural unit을 순서대로 합쳐 **UTF-8 scalar 경계에서 끝나는 최소 evaluation group**을 만든다. Raw pair에서는 Unicode scalar 하나가 group이고, BPE pair에서는 scalar 중간에서 끝난 token 뒤의 token들을 scalar boundary가 나올 때까지 같은 group에 넣는다.
4. 첫 UTF-8-complete group이 덮는 raw prefix는 양쪽 model에서 모두 제외한다. 이 선택은 첫 natural unit 하나보다 조금 더 제외할 수 있지만 모든 scored source가 valid UTF-8 boundary에서 시작하게 한다.
5. 나머지 group을 앞에서부터 연속으로 묶되 target block은 최대 256 raw bytes다. 단일 UTF-8-complete group도 256 bytes를 넘을 수 없다.
6. 각 target block 끝에서 뒤로 가며 완전한 group만 추가해, source span이 512 raw bytes 이하인 가장 긴 left context를 만든다.
7. Candidate와 comparator가 같은 `[context_start, target_end)` raw source를 받고 같은 `[target_start, target_end)` raw target에 대한 NLL을 낸다. BPE는 full-document token id subsequence를 그대로 사용하고 substring을 다시 tokenize하지 않는다.
8. Target block은 겹치지 않고 첫 group 뒤의 모든 raw byte를 정확히 한 번 덮는다. Window 사이 overlap은 left context에만 존재한다.

256-byte target cap을 택한 이유는 결과가 아니라 geometry다. 양쪽 model의 최대 source span 512 bytes 안에서 이후 block에 최소 하나의 완전한 predecessor group을 보장하고, 정상적인 group 길이에서는 대략 절반 이상의 source를 left context로 남긴다. Held-out natural tokenization에서 UTF-8-complete group이 256 bytes를 넘으면 tokenizer/evaluation artifact가 실패하며 결과를 열지 않는다.

## 3. 왜 BPE boundary에 맞추는가

BPE token의 확률은 그 token 전체에 대한 값이다. 한 token이 raw target 경계를 가로지를 때 그 NLL의 일부를 임의로 앞·뒤 byte에 나누는 원칙은 없다. 더구나 GPT-2-style ByteLevel BPE token은 한글의 3-byte UTF-8 scalar 내부에서 끝날 수 있다. 그 token boundary를 그대로 model-input 시작점으로 쓰면 candidate가 continuation byte에서 시작하는 비정상 source를 받는다. Fixed raw boundary에서 token을 강제로 쪼개면 원래 BPE model이 학습한 vocabulary event가 아니고, prompt와 continuation을 별도 encode하면 full-document BPB가 아니라 API-boundary likelihood가 된다.

따라서 main LM-quality estimand에서는 full-document natural tokenization을 보존하되, **token boundary와 UTF-8 scalar boundary를 동시에 만족하는 최소 group boundary**만 source/target 경계로 사용한다. 그 대신 candidate도 정확히 같은 raw prefix를 제외하고 같은 target span만 점수화한다. 이는 candidate에 BPE보다 많은 target coverage를 주지 않는다.

## 4. Pair별 denominator와 보고 방식

16K와 32K tokenizer의 첫 UTF-8-complete group 길이와 이후 boundary가 다를 수 있다. 세 pair는 다음 identity를 각각 가진다.

- tokenizer JSON SHA-256 또는 raw-byte fixed identity
- ordered held-out document stream SHA-256
- document별 natural-token id·length, UTF-8-complete group length와 group당 token-count SHA-256
- 모든 rolling window offset의 plan SHA-256
- scored document 순서 SHA-256
- document별 scored-byte count SHA-256
- context contract와 evidence 전체 identity SHA-256

따라서 하나의 candidate 절대 BPB를 세 reference와 공유하지 않는다. 표기는 다음처럼 pair-specific하게 한다.

| Pair | Candidate 열 | Comparator 열 | 공통 denominator |
|---|---|---|---|
| raw | `candidate@raw` | `raw_byte_reference` | first Unicode scalar 뒤 shared bytes |
| BPE-16K | `candidate@bpe16k` | `byte_bpe_16000_body_matched` | 16K first UTF-8-complete group 뒤 shared bytes |
| BPE-32K | `candidate@bpe32k` | `byte_bpe_32000` | 32K first UTF-8-complete group 뒤 shared bytes |

Pair 안의 차이와 noninferiority interval만 confirmatory 비교다. Denominator와 context plan이 다른 pair 사이의 절대 BPB 순위는 descriptive하지도 않으며 쓰지 않는다.

## 5. 통계 단위

각 rolling window의 NLL을 원문 document 단위로 다시 합친다. Seed별 pair 차이는

\[
\frac{\sum_d (NLL_{candidate,d} - NLL_{reference,d})}
     {\ln 2 \sum_d bytes_d}
\]

로 계산한다. Crossed bootstrap은 model seed와 shared document를 재표집한다. 같은 document에서 나온 여러 overlapping-context window를 독립 표본으로 취급하지 않는다. 첫 UTF-8-complete group 뒤에 점수화할 byte가 없는 document는 양쪽에서 함께 제외하고 input/scored/unscored document 수를 공개한다.

`src/jamoflow/publication_inference.py`는 이제 raw scored-byte 배열만 받지 않는다. Candidate/comparator key, tokenizer, document order, scored-byte 배열과 window plan이 연결된 `PublicationBPBContextEvidence`를 요구한다. 다른 tokenizer나 다른 document plan의 loss를 끼워 넣으면 final gate 전에 실패한다.

## 6. Training-reset sensitivity

Main rolling 결과가 overlapping context 선택에만 의존하는지 보기 위해, 학습에 사용한 동일한 512-byte source window를 각각 독립 document로 간주한 sensitivity를 함께 계산한다. 이때도 첫 UTF-8-complete comparator group의 raw bytes를 양쪽에서 함께 제외하고 token을 쪼개지 않는다.

이 sensitivity는 다음만 진단한다.

- 학습 시 512-byte reset과 main rolling 평가 사이의 차이
- BPE first-group exclusion이 window마다 반복될 때의 coverage 변화
- context 길이 분포 변화에 대한 pairwise BPB 안정성

Sensitivity는 final noninferiority gate를 대체하지 않고, main과 방향이 다르면 context-dependent result로 제한해 보고한다.

## 7. 구현 검증

`src/jamoflow/publication_bpb.py`와 tests는 다음을 기계적으로 검사한다.

- strict UTF-8·NFC source
- raw 또는 sealed 16K/32K comparator role
- BPE natural token의 exact byte reconstruction
- exact natural token-id sequence와 scoring subsequence의 identity
- natural token을 최소 UTF-8-complete group으로 묶는 boundary와 token-index mapping
- 모든 source/target 시작·끝의 UTF-8 scalar boundary
- UTF-8-complete group 최대 256 bytes
- source span 최대 512 bytes, target 최대 256 bytes, positive left context
- target의 완전하고 정확히 한 번인 coverage
- 모든 boundary의 comparator-group 정렬
- seeded random unit partitions의 invariant
- 실제 Tokenizers 0.22.2 ByteLevel BPE의 token-byte 복원과 rolling plan
- document order·tokenizer·scored bytes·plan을 바꾼 evidence rejection

이 교정은 아직 publication result가 없는 상태에서 이루어졌다. 진행 중 compact Phase 3의 학습 graph, policy, seed와 결과에는 영향을 주지 않는다.
