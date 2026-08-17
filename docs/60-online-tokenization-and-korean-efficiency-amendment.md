# Online tokenization and Korean inference-efficiency literature amendment

> 확인일: 2026-08-11  
> 방법: 사용자 지정 `aside-browser`로 논문 원문을 읽고, 공개 저장소는 고정 commit에서 별도로 감사  
> 시점: initial F/C/W·D/P 결과 관측 후, S/E/EC family 완성 결과와 모든 actual-inference 결과 전  
> 영향: novelty 축소, cross-tokenizer 측정 단위 강화, publication comparator 보강. 진행 중인 Phase 3 policy·seed·gate는 불변
> 후속 무결성 교정: §7.2의 post-test conditional 16K 선택은 [dual-BPE sealed-test correction](./61-dual-bpe-sealed-test-correction.md)으로 대체됨
> graph 교정: 16K는 [same-body output-head stress](./63-bpe-body-match-correction.md)로 고정됨

## 1. 결론부터

추가 선행연구는 JamoFlow의 현재 Phase 3 질문을 선점하지 않지만, 더 넓은 주장을 상당히 좁힌다.

1. **동적 tokenization 자체는 신규가 아니다.** ACL 2025의 dynamic-tokenization retrofit과 NeurIPS 2025의 `zip2zip`이 이미 존재한다.
2. **한 autoregressive step에서 여러 기존 token을 내보내는 축도 신규가 아니다.** `zip2zip`은 online LZW hypertoken으로 이를 실제 4B/14B 모델에서 수행한다.
3. **한국어 어휘 축소·확장도 신규가 아니다.** EEVE는 한국어 vocabulary expansion을 모델 수준에서 수행했고, 2026년 Korean token-pruning preprint는 실제 latency까지 보고했다.
4. **token count 또는 cross-tokenizer tokens/s는 실제 사용자 효율의 충분한 증거가 아니다.** 출력 단위가 달라지면 같은 `256 tokens`가 같은 text/byte 양을 뜻하지 않는다.
5. **현재 남는 중심 질문은 더 좁고 명확하다.** 동일 HF BLT graph와 동일 global-position rate에서, 이미 관측된 한국어 whitespace 쪽으로 prefix-causal boundary를 옮기는 것이 quality를 보존하고, 그 결과가 동일 출력 byte 수까지의 실제 cached inference에서 살아남는가이다.

이 보정 뒤에도 현재 Phase 3는 계속할 가치가 있다. 다만 최종 논문 가치는 teacher-forced BPB나 analytical FLOPs가 아니라, quality-qualified actual inference gate의 통과 여부로만 결정한다.

## 2. `zip2zip`: 가장 중요한 신규 누락

[Geng et al., *zip2zip: Inference-Time Adaptive Tokenization via Online Compression*](https://arxiv.org/abs/2506.01084)은 base BPE sequence 위에서 LZW codebook을 online으로 만들고, 새로 생긴 hypertoken의 embedding과 output score를 hyper-encoder/unencoder로 계산한다. Hypertoken은 최대 세 base tokens를 대표하며 autoregressive generation의 한 step에서 여러 base tokens를 출력할 수 있다.

이는 다음 broad claim을 선점한다.

- inference-time에 동적으로 새 token unit을 만든 최초의 decoder LM
- 한 neural forward에서 여러 기존 symbol/token을 출력한 최초의 adaptive tokenizer
- online compression을 generation step 감소로 연결한 최초의 연구

### 2.1 보고된 실제 속도

논문 Table 5의 Phi-3.5 4B, Apple M1 16GB decode throughput 개선은 다음과 같다.

| prompt + generation setting | reported decode gain |
|---:|---:|
| 256 + 256 | +7.5% |
| 512 + 256 | +34.8% |
| 1024 + 256 | +3.9% |
| 2048 + 256 | +7.5% |

같은 표에서 M1 prefill은 두 setting에서 각각 -11.8%, -6.6%로 느려진다. H100 decode gain은 4B에서 +9.3%~+46.6%, 14B에서 +9.6%~+48.1%로 보고된다. 따라서 다중-unit generation이 실제 wall-clock을 개선할 수 있다는 선행 증거는 분명하지만, hardware와 context에 따라 효과가 크게 달라진다.

### 2.2 품질은 비열등으로 확립되지 않았다

4B의 byte-PPL은 base 대비 WikiText 1.58→1.69, Pile 1.79→1.95, mC4 1.88→2.00, dC4 1.74→1.82로 나빠졌다. GSM8K two-shot accuracy는 0.82→0.15, 14B는 0.84→0.25로 크게 하락했다. 번역에서도 언어쌍에 따라 작지 않은 저하가 있다. 저자들도 lossless representation의 이론적 가능성과 실제 optimization 결과가 다르다고 limitation에서 명시한다.

따라서 `zip2zip`은 **실제 step/latency baseline으로는 강하지만 quality-preserving baseline으로 자동 합격하지 않는다.** JamoFlow가 더 작은 품질 저하를 보인다면 그 자체는 유효한 차별점이지만, 속도까지 이겨야 한다는 뜻은 아니다.

### 2.3 Table 5는 JamoFlow의 primary metric과 동등하지 않다

논문은 Table 5를 `tokens/sec`로 정의하고 두 방법 모두 `256-token generation length`라고 기술한다. 그러나 base token과 hypertoken은 한 token이 대표하는 UTF-8 byte 수가 다르다. 원문과 arXiv source에는 다음이 명확히 고정되어 있지 않다.

- 256이 decompressed base-token 수인지 model step/hypertoken 수인지
- 두 방법이 동일한 text 또는 동일한 UTF-8 byte 수까지 실행됐는지
- free-running output 차이를 어떻게 통제했는지
- prompt×repeat 분포, warmup, synchronization, variance

공개 저장소 [commit `4717b7711e8fdc5a0fcaced8509c49cd33974771`](https://github.com/epfl-dlab/zip2zip/tree/4717b7711e8fdc5a0fcaced8509c49cd33974771)에서도 Table 5 재현 runner, MLX implementation, 논문이 언급한 Triton benchmark path를 찾지 못했다. 이는 논문 수치가 거짓이라는 뜻이 아니라, 공개 artifact만으로 JamoFlow의 동일-output estimand에 재계산할 수 없다는 뜻이다.

따라서 서로 다른 output unit을 비교할 때 primary metric은 계속 다음으로 고정한다.

- 동일 held-out continuation의 **time-to-N source UTF-8 bytes**
- free-running의 **time-to-N valid UTF-8 bytes**와 overshoot/failure
- ms/source-byte와 Unicode scalars/s
- TTFT와 decode를 분리한 end-to-end latency

`tokens/s`는 각 model 내부 diagnostic으로만 남기며 cross-tokenizer superiority 근거로 쓰지 않는다.

## 3. ACL 2025 dynamic-tokenization retrofit

[Feher et al., *Retrofitting Large Language Models with Dynamic Tokenization*](https://aclanthology.org/2025.acl-long.1444/)은 batch 안에서 BPE-style merge를 만들고 pretrained hypernetwork로 새 embedding을 계산한다. XLM-R 계열에서는 14개 언어에 걸쳐 평균 20% 이상의 sequence reduction을 보고하며, Mistral-7B에서는 full sequence를 미리 아는 scoring/prefill에 적용해 최대 약 17% 감소를 보고한다.

이 연구는 다음 이유로 현재 Phase 3와 다르다.

- 14-language encoder 평가는 한국어를 포함하지 않는다.
- decoder의 true dynamic method는 full sequence를 미리 아는 scoring/prefill에 한정된다.
- autoregressive generation은 unbounded dynamic vocabulary가 아니라 별도로 만든 static 1M vocabulary와 ANN retrieval을 사용한다.
- 그 generation control은 MMLU English에서 61.8→55.9, MT-Bench에서 7.54→6.64의 품질 저하를 보인다.
- throughput 결론은 주로 analytical FLOPs이고, 동일 output byte까지의 cached generation latency가 아니다.

그러나 이 논문은 `dynamic tokenization`, `hypernetwork embedding`, `prefill compression`의 선행성을 확실히 차지한다. JamoFlow는 이 범주 전체의 최초성을 주장할 수 없다.

## 4. `SemToken`: 서지적 선행과 재현 가능한 증거를 분리한다

[Liu and Yu, *SemToken*](https://aclanthology.org/2026.starsem-conference.1/)은 *SEM 2026 oral 논문으로, semantic density에 따라 input spans를 병합해 2.4× token reduction과 1.9× latency speedup을 보고한다. 따라서 관련 연구 목록에서는 누락하면 안 된다.

그러나 현재 공개 evidence를 직접 감사하면 numerical systems baseline으로 사용할 수 없다. 공개 저장소 commit [`d8d09096ee661db31e2fa8587fa6b58419845551`](https://github.com/FastLM/SemToken/tree/d8d09096ee661db31e2fa8587fa6b58419845551)에서 확인한 사항은 다음과 같다.

1. [`evaluation/benchmark.py`](https://github.com/FastLM/SemToken/blob/d8d09096ee661db31e2fa8587fa6b58419845551/evaluation/benchmark.py#L238-L264)는 실제 WikiText/LongBench/BookSum이 아니라 반복된 세 개 mock string을 사용한다.
2. `model_name`은 LLaMA/GPT-J/GPT-NeoX로 순회하지만 실제 LM을 load하거나 forward/generate하지 않는다.
3. [quality metric](https://github.com/FastLM/SemToken/blob/d8d09096ee661db31e2fa8587fa6b58419845551/evaluation/benchmark.py#L171-L190)은 코드 주석 그대로 `would need actual model evaluation`이며 모두 `None`이다.
4. 같은 runner가 측정한 `latency`는 `semtoken.tokenize(text)` compression 시간이고 LM inference가 아니다.
5. [61.2/48.4/47.9ms 같은 paper baseline 값](https://github.com/FastLM/SemToken/blob/d8d09096ee661db31e2fa8587fa6b58419845551/evaluation/benchmark.py#L461-L480)은 비교 dictionary에 상수로 들어 있다.
6. [`TokenMerger`](https://github.com/FastLM/SemToken/blob/d8d09096ee661db31e2fa8587fa6b58419845551/semtoken/utils.py#L147-L172)는 semantic allocator가 반환한 `selected_spans`를 실제 span 구성에 사용하지 않고 모든 연속 token을 최대 3개씩 묶는다.
7. 새 ID는 base vocabulary 뒤에서 증가하지만 이를 기존 LM이 소비할 embedding/output 경로가 공개 구현에 없다.

또한 논문 Algorithm 1은 low-entropy region을 coarsen한다는 설명과 달리 high-entropy top-B cluster를 `merge`한 집합만 반환하도록 적혀 있어, 보존/병합/탈락의 의미가 불명확하다. 논문에는 hardware, context/generation length, sample count, seed, repetition, warmup과 variance도 없다.

따라서 취할 태도는 다음과 같다.

- *SEM 2026 논문이라는 서지 사실과 reported claim은 정확히 인용한다.
- 독립적인 end-to-end reproduction 전에는 그 latency·quality 수치를 baseline 표의 검증된 값으로 합치지 않는다.
- 공개 code의 현재 상태가 논문 실험을 재현하지 못한다는 사실만 기술하고, 논문 결과가 거짓이라고 단정하지 않는다.

## 5. 한국어 효율 연구가 주는 직접 교훈

### 5.1 Korean token pruning

[Kim and Kim, *Optimizing Korean-Centric LLMs via Token Pruning*](https://arxiv.org/abs/2604.16235)은 Qwen3, Gemma-3, Llama-3, Aya 등의 vocabulary에서 비대상 script token을 제거한다. Korean benchmark와 번역 품질을 폭넓게 비교하고, Seed-X-PPO-7B에서 vocabulary를 65,269→41,704로 36.1% 줄였을 때 latency가 1126→1116ms, 즉 0.89% 개선됐다고 보고한다.

이는 최초 대화의 중요한 비판을 직접 지지한다.

> Output head 후보 수를 줄이는 것만으로는 attention, FFN, KV-cache movement와 sequential decoding bottleneck을 없애지 못한다.

다만 이 arXiv v1의 latency section은 hardware 종류, prompt/output 길이, timing scope, sample 수, warmup, repetition과 variance를 공개하지 않고 한 model의 aggregate만 제시한다. 따라서 0.89%를 정밀한 effect size로 가져오지는 않는다. 방향성 증거와 output-head confound를 요구하는 근거로만 사용한다.

### 5.2 EEVE

[Kim et al., *Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models*](https://arxiv.org/abs/2402.14714)은 SOLAR-10.7B와 Phi-2에 한국어 token 8,960개를 추가하고 단계적으로 embedding과 model을 학습한다. 한국어 token consumption이 거의 3배 개선됐고 같은 data를 훨씬 적은 token으로 학습할 수 있다고 보고하며, 실제 한국어/영어 downstream model 결과도 제공한다.

이는 강한 Korean model-level adaptation 선행이지만, 논문의 4× training-efficiency 표현은 token-count 기반 해석이며 actual generation latency를 측정한 결과가 아니다. JamoFlow의 실제 inference 주장을 대신하지도, JamoFlow가 한국어 token efficiency를 처음 연구했다는 주장을 허용하지도 않는다.

### 5.3 Writing-system-level BPE surgery

[Didenko, *Writing-System-Level Tokenizer Adaptation for Byte-Level BPE*](https://arxiv.org/abs/2608.00582)은 fixed vocabulary 안에서 byte-BPE merge graph와 기존 ID를 최대한 보존하며 Ukrainian token count를 33.5%/36.6% 줄인다. 저자는 이를 construction-time compatibility로 한정하고 downstream 또는 training savings에는 model-level 실험이 필요하다고 명시한다.

이 연구의 script-aware removal manifest에는 Hangul row도 포함되지만, 이는 Ukrainian slot 재할당의 제거 내역일 뿐 Korean 보존 평가가 아니다. 따라서 한국어 결과로 인용하면 안 된다. 대신 다음 교훈을 준다.

- vocabulary row 수뿐 아니라 merge reachability와 same-ID semantics를 감사해야 한다.
- tokenizer-only compression을 model quality나 latency로 승격하면 안 된다.
- publication BPE artifact는 ordinary tokenizer runtime에서 전 token의 reachability와 round trip을 검증해야 한다.

## 6. 수정된 novelty와 claim boundary

### 금지할 주장

- 최초의 linguistically informed/dynamic/adaptive tokenizer
- 최초의 inference-time vocabulary adaptation
- 최초의 multi-token 또는 multi-byte neural generation
- 최초의 한국어 token-efficiency 또는 vocabulary adaptation
- token count, FLOPs 또는 tokens/s만으로 실제 inference가 빨라졌다는 주장
- S/E/EC와의 same-graph 비교만으로 최신 adaptive tokenization system 전반을 이겼다는 주장

### 아직 가능한 주장

- fixed global-position budget에서 **observed Korean whitespace 쪽으로 prefix-causal patch boundary를 relocation**한 통제 실험
- F/C/W/S/E/EC를 동일 HF BLT graph에서 비교하고 learned-router total cost를 포함한 Korean boundary study
- source-document clustered quality와 exact time-to-same-valid-bytes를 함께 요구한 end-to-end 검증
- actual gate가 통과할 경우에만, parameter-free Korean whitespace relocation의 좁은 within-byte-latent inference-efficiency 결과

이 novelty는 화려하지 않지만 causal identification이 명확하다. 반대로 실제 latency가 개선되지 않으면 사용자가 정한 가치 기준상 positive method paper가 아니다.

## 7. Publication protocol 보강

### 7.1 진행 중 Phase 3는 바꾸지 않는다

현재 S/E/EC family의 policy, seed, training order, checkpoint와 Gate I/M/J는 유지한다. 새 문헌을 본 뒤 결과에 맞춰 policy를 추가하면 initial comparison의 해석이 오염된다.

### 7.2 BPE output-head confound control

이 절의 원래 conditional-selection 순서는 publication test를 먼저 사용해야 한다는 모순이 확인되어 후속 교정되었다. 현재 적용 규칙은 32K ordinary baseline과 **16K byte-BPE stress control을 publication test 전에 함께 고정하고, candidate가 둘 다 이기도록 요구**하는 것이다.

- 16K와 32K 모두 full 256-byte alphabet, 동일 train stream, paired seeds, data/compute-matched checkpoints를 사용한다.
- 32K는 candidate와 total-parameter match하고, 16K는 32K body를 고정한 채 vocabulary/output rows만 줄인다. 두 graph 모두 quality나 timing 전에 고정한다.
- calibration latency로 한 vocabulary를 탈락시키지 않는다.
- lock artifact는 두 tokenizer/model/checkpoint와 source·calibration evidence를 모두 잠근다.
- candidate는 16K와 32K 각각에 대해 sealed BPB, downstream과 actual-inference gate를 통과해야 한다.

이 control은 어휘 크기 축소만으로 큰 속도가 나지 않는다는 선행 결과를 맹신하지 않고, 작은 Mac-scale model에서 output projection 비중이 더 클 수 있음을 직접 통제한다.

### 7.3 `zip2zip`의 위치

현재 W boundary paper의 primary gate에 3.8B `zip2zip` checkpoint를 억지로 parameter-matched baseline으로 넣지 않는다. scale, backbone, training data와 objective가 모두 다르기 때문이다. 대신 현재 candidate가 raw와 BPE gate를 통과할 경우 다음 external systems audit를 시도한다.

1. 공개 3.8B base Phi-3.5와 paired `zip2zip` checkpoint를 같은 Mac에서 실행한다.
2. Korean held-out prompts에서 strict UTF-8, Korean quality floor와 time-to-same-valid-bytes를 측정한다.
3. 공개 implementation이 MPS/MLX에서 paper path를 재현하지 못하면 `not reproducible on target hardware`로 남기고 수치를 자체 구현으로 보충하지 않는다.
4. 이 표는 cross-scale contextual reference이며 JamoFlow superiority gate가 아니다.

반면 향후 multi-unit output method로 pivot하면 `zip2zip` 또는 정당한 재현은 직접 baseline이 된다. Fast BLT만 인용하고 online hypertoken baseline을 생략할 수 없다.

### 7.4 결과가 실패할 때의 pivot

- W가 corrected quality gate부터 실패하면 boundary method를 더 복잡하게 살리지 않는다.
- W가 quality는 통과하지만 raw actual gate를 실패하면 teacher-forced boundary observation으로만 정리한다.
- W가 raw를 이기지만 16K/32K BPE 중 하나라도 이기지 못하면 vocabulary-specific 또는 within-byte-latent 결과로 축소한다.
- BPE 대비 sequential-step bottleneck이 결정적이면 다음 독립 연구는 **Korean-aware online multi-unit proposal/verification**으로 정의한다. 이때 linguistic rule은 proposal prior일 뿐이며, `zip2zip`, Fast BLT와 plain speculative decoding을 이겨야 한다.

## 8. 최종 판단

새 문헌은 원래 아이디어의 넓은 novelty를 더 줄였지만, 현재 연구를 중단시킬 이유는 아니다. 오히려 논문 가치의 판정 기준을 선명하게 만든다.

> Korean-specific boundary placement가 실제 cached inference에서 동일한 valid UTF-8 output을 더 빨리 만들지 못하면 positive efficiency claim은 실패다. 성공한다면 기여는 “새 tokenizer”가 아니라 “cheap prefix-causal Korean boundary relocation의 quality-qualified systems effect”다.

이 기준은 `zip2zip`의 강한 실제 속도 증거, SemToken 공개 artifact의 재현성 문제, 한국어 vocabulary pruning의 1% 미만 latency 효과를 모두 반영한다.
