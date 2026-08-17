# Publication comparator and Korean downstream protocol

> 작성일: 2026-08-11
> 상태: **Phase 3 primary·publication-scale·downstream·BPE 결과 확인 전 고정**
> valid-output 후속 교정: [valid-output actual-inference correction](./65-valid-output-actual-inference-correction.md)
> evidence-lineage 후속 교정: [publication evidence-identity correction](./68-publication-evidence-identity-correction.md)
> comparator-role 후속 고정: [publication comparator-role lock](./69-publication-comparator-role-lock.md)
> downstream label-boundary 후속 교정: [downstream label-boundary correction](./70-downstream-label-boundary-correction.md)
> BPE valid-output 후속 교정: [BPE token UTF-8 transition correction](./72-bpe-token-utf8-transition-correction.md)
> BPB raw-context 후속 교정: [raw-context-matched rolling BPB correction](./73-raw-context-matched-bpb-correction.md)
> 상위 protocol: [Actual-inference and compute-conversion protocol](./44-actual-inference-and-compute-conversion-protocol.md), [Mac feasibility addendum](./47-publication-scale-feasibility-addendum.md)
> BPE runtime 교정: [prompt-boundary addendum](./54-bpe-prompt-boundary-runtime-addendum.md)
> BPE sealed-test 교정: [dual-BPE correction](./61-dual-bpe-sealed-test-correction.md)
> learning-curve 교정: [last-two-budget noninferiority](./62-learning-curve-noninferiority-correction.md)
> 16K graph 교정: [BPE body match](./63-bpe-body-match-correction.md)
> 실행 조건: compact Final Value Gate 통과
> 목적: 실제 추론 개선을 품질 저하, 작은 모델의 task floor, 약한 BPE와 혼동하지 않도록 publication-scale 비교를 사전등록함

## 1. 먼저 고정하는 부정적 결론

50–100M parameter model을 256M raw bytes만 학습한 결과는 **Mac에서 가능한 가장 큰 mechanism-scale replication**이지, 충분히 학습된 한국어 LLM의 frontier 결과가 아니다. BLT는 BPE 비교에서 같은 pretraining distribution, byte-normalized loss, expected bytes/context와 component FLOPs를 맞췄고, 작은 training budget에서는 BPE가 앞서다가 scale과 data가 커질 때 crossover가 나타남을 보고했다. 특히 BLT의 약 470M fixed-inference class 비교는 수십 billion training bytes 영역을 사용한다([BLT §4–5](https://arxiv.org/html/2412.09871)).

따라서 다음 세 문장을 금지한다.

- 256M-byte result만으로 `Korean LLM efficiency frontier`라고 부르기
- 큰 parameter count를 충분한 학습의 대용으로 쓰기
- raw-byte family 안의 speedup을 standard BPE보다 빠르다는 주장으로 바꾸기

Fast BLT도 generation 효율을 NFE와 추정 memory-bandwidth cost로 평가하며 sequential byte generation 자체를 병목으로 다룬다([Fast BLT §4](https://arxiv.org/html/2605.08044)). JamoFlow의 차별적 검증 기준은 이 추정치를 실제 Apple MPS incremental wall time으로 대체하는 데 있다. 그러나 wall time이 실제로 줄지 않으면 이 차이는 기여가 아니다.

## 2. 질문과 두 단계 주장

Publication scale에서 답할 질문은 둘이다.

1. **Within-family:** 동일 raw-byte latent graph에서 selected W-rate가 calibration-only로 고정되고 sealed quality gate를 통과한 raw-byte reference보다 실제 incremental inference가 빠른가?
2. **Deployment-level:** total-parameter-matched 32K와 더 작은 same-body 16K standard Korean byte-BPE Transformer 모두보다 Korean quality를 유지하면서 실제 end-to-end inference가 빠른가?

1만 통과하면 `byte-latent family 내부 개선`만 허용한다. 사용자 기준의 넓은 `한국어 LLM 동작 효율 개선`과 논문 제목의 `efficient Korean inference`는 2까지 통과해야 한다.

## 3. 고정 데이터셋과 라이선스

Dataset row를 저장소에 복사하지 않는다. loader code, pinned revision, aggregate count·metric·provenance만 공개한다. 실행 전에 revision에서 실제 split, schema, row count와 file SHA-256을 다시 검증하며 불일치하면 결과를 열지 않고 중단한다.

| Dataset | 고정 revision | 확인된 조건 | 사용 |
|---|---|---|---|
| [KLUE](https://huggingface.co/datasets/klue/klue) | `349481ec73fff722f88e0453ca05c77a447d967c` | CC BY-SA 4.0; public train/validation, official test hidden | YNAT·NLI primary |
| [KoBEST v1](https://huggingface.co/datasets/skt/kobest_v1) | `a5ea15e3ac77ed694b79f6204eb31889a2ba989f` | CC BY-SA 4.0 | BoolQ·COPA·WiC·SentiNeg primary |
| [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU) | `d61b3f19e552c576bf5960dd24289763edc36a88` | CC BY-ND 4.0; knowledge-heavy multiple choice | floor-gated secondary only; transformed data 공개 금지 |
| [HAE-RAE Bench](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH_1.0) | `d5082e9b46bdd7012471d60ee1851e734606af72` | repository/card에서 dataset license를 확인하지 못함 | license 명확화 전 제외 |

[KLUE paper](https://arxiv.org/abs/2105.09680)는 8개 과제와 YNAT macro-F1, NLI accuracy를 정의한다. [KoBEST paper](https://aclanthology.org/2022.coling-1.325/)는 한국어 언어 지식을 겨냥한 다섯 과제와 F1 평가를 제시한다. 현재 KoBEST HF card의 HellaSwag split 표기는 논문 Table 1과 일치하지 않으므로 HellaSwag는 file-level split 재구성과 truncation audit가 끝나기 전 primary에서 제외한다. SentiNeg는 negation/antonym 변환에 대한 dev–test robustness를 보므로 primary robustness task로 유지한다.

KMMLU는 45개 과목·35,030 test examples의 전문 지식 평가이며 작은 from-scratch model이 chance floor에 머물 가능성이 높다([KMMLU paper](https://arxiv.org/abs/2402.11548)). HAE-RAE Bench도 1,538개 test-only 문항이어서 pretraining capability보다 prompt/floor를 재기 쉽다([official repository](https://github.com/HAE-RAE/HAE-RAE-BENCH)). 이 둘을 primary로 올려 `모두 낮으니 비열등`이라는 결론을 만들지 않는다.

## 4. Primary Korean suite

| Family | Task | fit / selection / sealed evaluation | primary metric |
|---|---|---|---|
| KoBEST | BoolQ | train / validation / test | macro-F1 |
| KoBEST | COPA | train / validation / test | macro-F1 |
| KoBEST | WiC | train / validation / test | macro-F1 |
| KoBEST | SentiNeg | train / validation / test | macro-F1 |
| KLUE | YNAT | official train의 90% / 내부 10% / public validation | macro-F1 |
| KLUE | NLI | official train의 90% / 내부 10% / public validation | accuracy |

KLUE public test label은 숨겨져 있으므로 official train 안에서 label별 canonical row-id SHA-256 순서의 10%를 internal selection split으로 고정하고 public validation을 한 번만 여는 sealed evaluation으로 사용한다. 비율 반올림, 동률 처리, row identity와 split hash는 loader 구현 전에 별도 addendum로 고정한다. KoBEST는 official validation으로 checkpoint와 reference를 고르고 test를 한 번만 연다.

KoBEST 원 논문은 모든 discrete-label task에 F1을 쓴다고만 적고 averaging convention은 명시하지 않는다. 따라서 재현 가능한 operational definition을 고정 label set 전체의 macro-F1, absent class와 zero division의 F1은 0으로 명시한다. Accuracy도 항상 함께 보고하되 primary gate를 바꾸지 않는다.

## 5. 같은 문제를 풀게 하는 fine-tuning interface

Architecture마다 별도 classification head를 붙이지 않는다. 각 과제를 고정 Korean prompt와 한 개 ASCII digit label `0`…`6`의 conditional likelihood 문제로 바꾼다.

- option·label 의미의 mapping은 dataset label id 순서를 그대로 사용한다.
- prompt token/byte loss는 mask하고 정답 digit 한 byte/token의 loss만 학습한다.
- evaluation은 허용된 digit 각각의 next-unit log probability를 계산해 argmax한다. unconstrained generation을 metric으로 쓰지 않는다.
- byte-BPE가 각 digit을 정확히 한 token으로 encode하는지 tokenizer artifact gate에서 검사한다.
- Prompt와 각 digit은 반드시 별도 encode한다. `encode(prompt + digit)`은 경계 merge diagnostic에만 사용하며 primary fine-tuning·prediction sequence를 만들지 않는다.
- template, field order, separators, Unicode NFC와 truncation은 모든 model에 동일한 raw UTF-8 string을 사용한다.
- template 후보를 test에서 고르지 않는다. 한 개 template를 pilot-free로 고정하고 부록의 paraphrase sensitivity는 primary 뒤에만 실행한다.
- task별 독립 fine-tuning을 사용한다. KoBEST가 보고한 multi-task degradation 가능성을 primary 결과와 섞지 않는다.

Fine-tuning의 prompt와 정답 digit을 합친 전체 model sequence는 512 source bytes 이하여야 하므로 prompt는 최대 511 bytes다. 이를 넘는 input은 answer와 모든 option을 보존하고 context field의 **오른쪽 끝**부터 UTF-8 scalar 경계에서 잘라낸다. Task마다 truncation rate를 공개한다. 어느 primary task든 sealed split의 10%를 넘게 자르면 그 task는 primary에서 탈락하고, 결과를 보지 않은 새 long-context protocol 없이는 대체하지 않는다.

Pretraining seed `1729, 2718, 31415`를 model·data order·fine-tuning randomness에 paired하게 사용한다. Epoch, learning rate, warmup, early stopping patience는 publication checkpoint를 열기 전에 작은 synthetic smoke test와 model-independent memory measurement만으로 고정한다. Test metric으로 hyperparameter나 epoch를 선택하지 않는다.

## 6. Benchmark contamination 차단

Public benchmark가 HPLT web pretraining에 포함됐을 가능성을 무시하지 않는다. Publication train stream을 만들기 **전**, label을 읽지 않는 다음 detector를 모든 primary split에 적용한다.

1. input fields만 NFC, line-ending canonicalization, Unicode whitespace collapse한다. 정답 label과 prompt instruction은 넣지 않는다.
2. canonical input이 20 Unicode scalars 이상이고 alphanumeric/Hangul 같은 정보성 scalar가 8개 이상이면 normalized exact local containment를 검사한다. 이 완화는 짧은 YNAT headline을 누락하지 않기 위한 것이다.
3. 5-scalar shingles 중 benchmark shingle coverage가 `>= 0.80`이고 shared shingle이 10개 이상이면 near duplicate로 표시한다. Coverage와 보조 Jaccard는 긴 web document 전체가 아니라 benchmark 길이의 0.80–1.25배인 candidate local span에 대해 계산한다. 13-scalar Jaccard는 40자 내외 Korean input의 한두 글자 변형에도 과도하게 민감하므로 사용하지 않는다.
4. 어느 primary train/selection/evaluation example과 match한 HPLT document도 candidate, raw reference, BPE의 pretraining stream에서 동일하게 제거한다.
5. detector version, input dataset revision, 제거 document/example 수, collision audit와 output stream hash만 저장한다. benchmark text나 HPLT text는 tracked artifact에 저장하지 않는다.

Evaluation label을 사용하지 않는 중복 제거는 허용하지만 detector threshold를 evaluation score 뒤에 조정하지 않는다. Exact/near-match audit가 완료되지 않으면 downstream 결과를 논문 증거로 사용하지 않는다.

`src/jamoflow/contamination.py`는 원문을 결과에 넣지 않는 correctness reference다. 같은 모듈의 reference-complete inverted index는 candidate retrieval만 가속하고 최종 match를 반드시 이 reference predicate로 재검증한다. 구현·완전성 근거는 [contamination indexed-retrieval correction](./71-contamination-index-correction.md)에 고정했다. Full-corpus runner가 synthetic exact/near/non-match fixture에서 reference와 100% 일치하지 않으면 publication stream을 만들 수 없다.

## 7. Standard BPE comparator

[Korean tokenization 비교](https://aclanthology.org/2020.aacl-main.17/)는 BPE가 강한 기본선이며 morphology-then-BPE가 여러 과제에서 더 좋지만 모든 과제에서 일관되지는 않음을 보였다. Required comparators는 재현 가능한 16K와 32K standard byte-level BPE로 고정한다. 32K는 ordinary baseline이고 16K는 작은 output embedding/head confound를 통제한다.

- Hugging Face Tokenizers `0.22.2`
- vocabulary 16,000과 32,000, 각각 full 256-byte initial alphabet, minimum frequency 2
- GPT-2-style byte-level pretokenization, `add_prefix_space=False`
- Unicode normalizer 없음; HPLT NFC source bytes를 그대로 보존
- added/special token은 쓰지 않는다. 단독 NUL byte의 base-token id를 masked batching PAD로 재사용하며 attention/loss에서 제외한다. 실제 NUL byte는 attention mask로 구분한다.
- merge 학습은 contamination 제거 후 publication pretraining train split만 사용
- downstream·held-out·private Markdown로 vocabulary를 학습하지 않음
- tokenizer JSON, trainer arguments, source stream hash와 round-trip identity audit 공개
- controlled replay는 prompt와 continuation을 별도 encode해 이미 고정된 prompt KV cache를 보존하고, joint encoding은 경계-merge sensitivity로만 보고

두 vocabulary 중 하나를 calibration 또는 test 결과로 탈락시키지 않는다. 32K는 한국어 tokenization 선행의 강한 plain-BPE 영역이고, 16K는 Mac-scale model에서 output projection 비중이 커지는 가능성을 직접 검사한다. 둘 다 모든 Korean/ASCII/code-mixed audit string에서 `decode(encode(x)) == x` byte identity를 통과해야 한다.

Token Transformer는 Llama-style decoder-only causal graph, full multi-head attention, tied input/output embedding, cached incremental decoding을 사용한다. 32K는 candidate total trainable parameters의 ±1%에 맞도록 width/head/layer grid를 **quality와 timing 전에** parameter count만으로 선택했다. 16K는 같은 target의 32K body를 고정하고 vocabulary rows만 줄인다. Candidate와 같은 optimizer family, precision, raw source order와 paired seed를 쓴다.

| Vocabulary | Target | Candidate params | BPE params | BPE width / heads / layers / FFN | candidate relation |
|---:|---:|---:|---:|---:|---:|
| 16K body-matched | 50M | 49,823,488 | 42,617,792 | 448 / 7 / 12 / 1,600 | 14.462% smaller |
| 16K body-matched | 75M | 76,492,480 | 66,710,368 | 608 / 8 / 12 / 1,792 | 12.788% smaller |
| 16K body-matched | 100M | 98,403,360 | 86,975,680 | 704 / 11 / 12 / 2,048 | 11.613% smaller |
| 32K | 50M | 49,823,488 | 49,785,792 | 448 / 7 / 12 / 1,600 | 0.076% |
| 32K | 75M | 76,492,480 | 76,438,368 | 608 / 8 / 12 / 1,792 | 0.071% |
| 32K | 100M | 98,403,360 | 98,239,680 | 704 / 11 / 12 / 2,048 | 0.166% |

### 7.1 Data-matched와 compute-matched checkpoint

각 BPE run에서 두 checkpoint를 만든다.

1. **data-matched:** candidate와 같은 256M clean raw bytes를 본 시점
2. **compute-matched:** candidate의 누적 analytical training FLOPs에 도달할 때까지 같은 clean stream을 계속 본 시점

Backward=`2 × forward`라는 BLT 관례와 embedding/local/global/cross-attention/router component를 모두 포함한 식을 사용하고 실제 train wall time도 별도로 보고한다. Compute-matched BPE가 더 많은 raw bytes를 보게 되는 것은 숨기지 않는다. Final deployment-quality reference 후보에는 더 강한 compute-matched checkpoint를 넣고, data-matched 결과는 data 효과 분리용으로 남긴다.

Compute-matched 연장은 clean HPLT document를 반복하지 않고 아직 보지 않은 document만 사용한다. 필요한 누적 FLOPs 전에 고정 corpus가 소진되면 마지막 available-data checkpoint를 `compute_match_unavailable`로 보고하고 broad comparison을 통과시키지 않는다. 1B extension에서도 candidate budget에 대응하는 data-matched와 compute-matched checkpoint를 다시 만든다.

## 8. Raw-byte span, BPB와 context fairness

모든 architecture는 같은 ordered UTF-8 documents를 학습하고 document split을 공유한다. Candidate는 512 raw-byte context를 사용한다. BPE batch는 같은 512-byte source windows를 독립 tokenize하고 padding을 mask해 **expected raw bytes/batch**를 맞춘다. Window는 UTF-8 scalar 경계에서 끝나며 잘린 byte를 다음 window에서 버리지 않는다.

BPE의 512 token capacity를 그대로 문맥으로 쓰면 512-byte candidate보다 훨씬 긴 원문을 볼 수 있으므로 허용하지 않는다. Main BPB는 [후속 교정](./73-raw-context-matched-bpb-correction.md)의 **pairwise natural-unit raw-capped rolling** 규약으로 계산한다.

- 각 held-out document를 comparator의 자연 unit으로 한 번만 나눈다. Raw reference의 unit은 한 byte이고, BPE의 unit은 full-document natural tokenization 결과다.
- 자연 unit을 순서대로 합쳐 UTF-8 scalar 경계에서 끝나는 최소 evaluation group을 만든다. 따라서 source와 target 경계는 BPE token을 쪼개지 않으면서 모두 valid UTF-8 scalar boundary다.
- 첫 UTF-8-complete group이 덮는 raw bytes는 candidate와 comparator 양쪽에서 함께 제외한다.
- 이후 group을 최대 256 target bytes의 연속 block으로 묶고, 완전한 group만 사용해 최대 512 raw bytes 안에서 가능한 가장 긴 왼쪽 문맥을 붙인다.
- Candidate와 comparator는 각 window에서 **동일한 raw context span과 동일한 raw target span**을 점수화한다. BPE token을 쪼개거나 token loss를 byte 일부에 배분하지 않는다.
- 모든 document에서 제외된 첫 group 뒤의 raw bytes는 target으로 정확히 한 번만 등장한다. Overlap은 context에만 허용한다.
- `total NLL / (ln 2 × exactly shared scored raw bytes)`를 쓰고, seed×document crossed bootstrap의 cluster는 window가 아니라 원문 document다.

첫 UTF-8-complete group 길이와 token boundary가 vocabulary마다 다르므로 candidate–raw, candidate–16K, candidate–32K는 각각 독립된 shared-span estimand다. 따라서 candidate BPB도 `candidate@raw`, `candidate@bpe16k`, `candidate@bpe32k`처럼 pair별로 보고하며 서로 다른 denominator의 절대값을 한 열에서 직접 순위화하지 않는다. Training-reset sensitivity는 같은 512-byte source windows를 각각 독립 문서처럼 처리한 pairwise plan으로만 보고하고 final gate에는 쓰지 않는다. Token 수나 character 수로 BPB 분모를 바꾸지 않는다.

Parameter-matched, raw-data-matched, train-FLOP-matched는 서로 다른 estimand이므로 한 열로 합치지 않는다. 다음을 모두 공개한다.

- exact parameters와 embedding 비중
- raw train bytes와 optimizer updates
- analytical train FLOPs와 actual MPS train wall time
- context raw-byte 분포와 BPE token 분포
- held-out BPB, downstream score, UTF-8 validity
- peak/high-water memory의 측정 한계

## 9. Downstream floor와 noninferiority gate

작은 모델끼리 chance에서 같아 보이는 것을 품질 유지로 인정하지 않는다.

### 9.1 Reference 선택

각 task의 sealed split을 열기 전에 세 seed의 selection score 평균으로 다음 중 reference 하나를 고정한다.

- publication-scale strongest raw-byte comparator
- compute-matched 16K standard BPE
- compute-matched 32K standard BPE

최고 평균과 32K BPE의 차이가 `0.5 percentage points` 이하면 deployment default인 32K BPE를 선택한다. 그렇지 않으면 가장 높은 raw/16K/32K reference를 고른다. Candidate 자신은 reference 후보가 아니다. 선택 JSON을 commit한 뒤 sealed split을 연다.

### 9.2 Informativeness floor

Accuracy 기준 uninformed baseline은 train-majority classifier의 sealed accuracy와 `1 / number_of_labels` 중 큰 값이다. Chosen reference의 paired seed×example bootstrap 95% lower bound가 이 baseline보다 **5 percentage points 초과**해야 task가 informative하다.

- KLUE 두 task가 모두 informative해야 한다.
- KoBEST 네 task 중 최소 세 task가 informative해야 한다.
- 미달 task를 `비열등 성공`으로 세지 않고 전체 publication downstream gate를 block한다.

### 9.3 Quality noninferiority

Informative task에서 `candidate − chosen reference`를 percentage-point 단위로 계산한다. KoBEST와 KLUE family별 task macro-average를 paired seed×example hierarchical bootstrap한다.

- 두 family 각각 one-sided `97.5%` lower bound가 `−2.0 pp`보다 커야 한다. 두 family에 대한 Bonferroni familywise 5% 조건이다.
- 어느 individual task의 paired-seed mean도 `−5.0 pp`보다 작으면 안 된다.
- 각 family에서 seed 3개 중 최소 2개의 family mean이 `−2.0 pp` 이내여야 한다.
- 위에서 고정한 primary metric으로 gate를 계산하고 accuracy·per-class score·confusion matrix를 함께 공개한다.

이 gate는 superiority를 주장하지 않는다. `−2 pp` 안의 품질 보존을 입증할 뿐이다. Candidate가 유의하게 더 좋다는 별도 결론은 two-sided interval과 multiplicity correction 없이는 쓰지 않는다.

## 10. Cross-architecture actual inference

실제 efficiency는 analytical FLOPs나 teacher-forced throughput으로 대체하지 않는다. [기존 incremental protocol](./44-actual-inference-and-compute-conversion-protocol.md)의 equivalence, synchronization, randomized order와 prompt-paired bootstrap을 그대로 사용하되 BPE에는 다음을 추가한다.

- 같은 raw UTF-8 prompt와 held-out continuation을 사용한다.
- controlled replay에서 candidate는 byte step, BPE는 **prompt와 별도 encode한 continuation token step**을 실제 cached forward로 실행한다. Prefill이 첫 unit을 예측하므로 N units에 N−1 feedback forwards만 수행하고 unused next-logit은 계산하지 않는다.
- `encode(prompt + continuation)`은 prompt 경계를 가로지르는 merge 비율과 token-count sensitivity에만 사용하며 primary timing에 쓰지 않는다.
- free-running은 후보와 baseline 모두에 같은 strict RFC 3629 transition mask를 적용하고, UTF-8 DFA가 accept 상태에서 최소 `128 valid UTF-8 source bytes`에 처음 도달한 token/byte에서 멈춘다. Byte model은 0--3 byte, BPE는 emitted token 길이에 따른 overshoot를 기록한다. Mask로 absolute validity를 보장하되 intrinsic unconstrained validity와 혼동하지 않는다.
- BPE mask는 현재 DFA state×token raw-byte transition table로 컴파일한다. Singleton token을 Unicode decode해 bytes를 추정하지 않는다. Token table compilation만 timing 밖이며 state-row mask 적용과 transition/stop 검사는 timing 안이다.
- total completion latency, TTFT, decode latency, ms/scored-byte, Unicode scalars/s, peak memory와 sequential steps를 함께 보고한다.
- BPE tokenizer, sampling/argmax, detokenization, stop 검사와 device synchronization을 timing 안에 넣는다.
- Candidate의 UTF-8 encode/DFA/byte 선택도 같은 범위 안에 넣는다. Prepared-unit model-only replay는 별도 diagnostic이며 end-to-end 결과를 대신하지 않는다.
- 서로 다른 output을 만드는 free-running 비교는 controlled replay보다 아래의 secondary estimand다.

BPE cache도 `src/jamoflow/incremental_token.py`의 runtime으로 모든 prefix의 full-forward logits, argmax, parallel-prefill final logits와 먼저 대조한다. Cache length가 observed token count와 다르거나 `rtol=2e-5, atol=2e-5`를 넘으면 해당 checkpoint의 timing은 금지한다.

Publication-scale actual-inference gate는 candidate가 **chosen raw comparator, compute-matched 16K BPE, compute-matched 32K BPE 각각에 대해** 다음을 모두 만족해야 한다.

1. downstream noninferiority 통과
2. held-out document의 seed×document crossed bootstrap에서 `candidate − comparator` BPB one-sided 97.5% upper bound `< +0.010 BPB`, 그리고 세 seed 중 최소 두 seed의 차이가 `<= +0.010 BPB`
3. controlled-replay batch-1 decode median reduction `>= 10%`
4. paired seed×prompt crossed-bootstrap 95% lower bound `> 0`
5. 세 model seed 중 최소 두 seed에서 reduction `>= 10%`
6. free-running end-to-end median reduction `>= 10%`
7. free-running crossed-bootstrap 95% lower bound `> 0`
8. 공통 valid-output contract가 재구성되고 모든 seed의 completion rate가 100%이며, replacement-free rate가 comparator보다 2 pp 넘게 낮지 않음

8의 replacement guard는 평균뿐 아니라 세 model seed 중 최소 두 seed에서도 각각 2 pp margin 안이어야 한다. `src/jamoflow/publication_inference.py`가 runtime equivalence, timing integrity, document-byte-weighted BPB, crossed latency, data adequacy와 raw/BPE 최종 claim level의 단일 판정 구현이다. 하위 gate의 pass boolean을 직접 주입하지 않고 candidate/comparator identity가 연결된 검증 객체만 전달한다.

두 BPE 중 하나라도 실패하면 broad Korean inference-efficiency claim은 실패다. 한 vocabulary만 이기면 vocabulary-specific result로 축소하고, raw comparator만 이기면 within-family claim으로 축소한다. BPE보다 품질이 좋지만 느린 경우도 사용자 기준의 성공이 아니다.

## 11. Data adequacy와 다음 scale gate

256M-byte checkpoint에서 모든 positive gate가 통과해도 바로 최종 논문 주장을 쓰지 않는다. Candidate, raw comparator, **data-matched 16K와 32K BPE**의 64M, 128M, 256M held-out BPB learning curve를 결과와 함께 공개하고 다음으로 분기한다. Compute-matched BPE는 본 raw bytes가 다르므로 이 matched-data adequacy 계산에 넣지 않는다.

- Candidate–raw, candidate–16K BPE와 candidate–32K BPE 각 pair는 128M과 256M 모두에서 paired-seed one-sided 97.5% upper bound가 `+0.010 BPB`보다 작고 최소 2/3 seed가 margin 안이어야 한다. 두 model의 mean과 최소 2/3 seed BPB도 마지막 doubling에서 악화되면 안 된다. Margin 안의 near-tie와 부호 반전은 허용한다.
- reference가 downstream informativeness floor를 못 넘으면 `capability-undertrained`이다.
- 어느 하나면 256M evidence는 mechanism-scale로만 남긴다. Local extension을 실행한다면 512M과 1.024B matched raw-byte checkpoint를 모두 만든 뒤 512M–1.024B pair에 같은 gate를 적용한다.
- 1.024B에서도 last-two-budget noninferiority와 downstream floor가 안정되지 않으면 broad claim을 중단한다. 더 큰 data를 결과에 맞춰 임의로 추가하며 유리한 checkpoint를 고르지 않는다.

1.024B는 충분함을 보장하는 마법의 수치가 아니라 Mac-only 연구에서 결과 전에 고정한 최대 local extension이다. BLT 수준의 broad scaling claim은 훨씬 큰 data/model과 외부 accelerator replication이 필요하다는 limitation을 그대로 쓴다.

`src/jamoflow/data_adequacy.py`가 last-two-budget paired-seed noninferiority, 양쪽 model의 계속된 학습과 검증된 downstream gate의 informativeness를 계산한다. Final Value Gate에는 naked boolean이 아니라 candidate/raw/BPE identity를 포함한 `PublicationDataAdequacy` 객체 전체를 전달한다. 현재-budget document-clustered BPB gate는 별도로 유지한다.

## 12. Morphology-aware baseline의 위치

[KoRTok](https://github.com/kakaobrain/kortok)은 Apache-2.0 구현으로 morphology+BPE 재현 후보지만 오래된 dependency와 analyzer version을 먼저 고정해야 한다. [Morpheme Matters](https://aclanthology.org/2026.eacl-short.22/)의 [MoB repository](https://github.com/Dohy-Lee/mob)는 직접 관련된 강한 선행이나 확인 시점에 명시적 code license를 찾지 못했다.

따라서 MoB code를 복사하거나 자체 근사를 `MoB reproduction`이라고 부르지 않는다. Core gate가 통과한 뒤 KoRTok dependency가 clean environment에서 재현되면 secondary morphology+BPE model을 추가한다. 실행하지 못하면 `standard BPE보다 빠름`까지만 주장하고 morphology-aware tokenizer보다 우월하다는 문장은 금지한다.

## 13. 공개와 claim matrix

| Evidence | 허용되는 결론 |
|---|---|
| 256M same-graph BPB만 positive | Korean byte-boundary mechanism observation |
| raw comparator actual gate 통과, 두 BPE 모두 실패 | byte-latent family 내부 actual speedup |
| 16K/32K 중 한 BPE만 통과 | vocabulary-specific comparison; broad claim 실패 |
| 두 BPE actual gate 통과, downstream floor 실패 | speed/quality trade-off; positive efficiency claim 실패 |
| downstream noninferiority, raw+16K+32K actual gate, data stability 통과 | Mac-scale Korean inference-efficiency paper 후보 |
| 1B extension 또는 외부-scale replication까지 유지 | 더 넓은 Korean LM efficiency claim 검토 가능 |
| morphology baseline 미실행 | morphology-aware method에 대한 우월 주장 금지 |

Hugging Face에는 model weights, tokenizer JSON, config, inference runtime, evaluation code, immutable source revision manifest, aggregate results와 model card를 공개한다. Dataset row, private vault content, HPLT raw text, HAE-RAE derived artifact, KMMLU 변형본은 공개하지 않는다. Model card는 MPS hardware, scale/data 한계, 실패 gate와 intended-use limitation을 positive result와 같은 가시성으로 적는다.

Machine-checkable constants는 `src/jamoflow/publication_protocol.py`를 단일 source of truth로 사용한다. 이후 runner가 문서의 revision, task suite, label alphabet, BPE vocabulary, margin 또는 seed를 재정의하면 provenance validation이 실패해야 한다.
