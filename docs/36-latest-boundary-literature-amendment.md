# Latest boundary-model literature amendment

> 확인일: 2026-08-10  
> 방법: 사용자 지정 `aside-browser`로 arXiv·ACL Anthology 원문을 직접 열고, Bolmo는 PDF 본문까지 대조  
> 시점: seed 1,729의 F/C 결과만 존재하고 W는 학습 중; W contrast·OOD·mechanism 결과 미관측  
> 영향: Phase 3 policy와 gate 수치 불변, 신규성·related work·publication-scale baseline 보강
> 후속 보강: [ByteFlow causal/systems audit](./55-byteflow-causal-literature-amendment.md)

## 1. 결론

기존 감사는 Korean tokenizer와 BLT/SpaceByte/H-Net 계열을 상당히 넓게 다뤘지만, **pretrained model byteification**과 **boundary distribution 자체의 이전·분리**라는 최신 축을 충분히 반영하지 못했다. 특히 Bolmo는 `01-verification-report.md`에서 이름만 언급됐으나 이후 연구 방향과 paper draft에서 빠졌다.

이 누락은 현재 Phase 3의 same-rate W−C 질문을 무효화하지 않는다. 대신 다음 주장을 금지한다.

- scratch pretraining만이 고성능 byte model을 만드는 유일한 경로
- language-modeling capability와 patch boundary를 분리해 생각한 최초 연구
- learned latent tokenizer 전반에 대한 W의 우월성
- 19.6M from-scratch BLT의 절대 성능을 최신 byte model 수준으로 일반화

## 2. Bolmo

[Minixhofer et al., *Bolmo: Byteifying the Next Generation of Language Models*](https://arxiv.org/abs/2512.15586)는 OLMo 계열 subword model을 1B·7B byte-level latent-tokenizer model로 변환한다. 원문에서 확인한 핵심은 다음과 같다.

1. 전형적인 pretraining budget의 1% 미만인 39.3B tokens로 byteification했다고 보고한다.
2. prefill boundary predictor는 한 byte의 future context를 사용한다.
3. autoregressive decoding에서는 LM head가 다음 byte와 다음 patch boundary를 함께 예측한다.
4. source subword embedding과 global backbone을 적극 재사용한다.
5. compression factor를 높여 추가 속도–품질 trade-off를 만들 수 있다고 보고한다.

JamoFlow와의 차이는 명확하다.

| 축 | Bolmo | JamoFlow Phase 3 |
|---|---|---|
| 출발점 | pretrained subword OLMo | from-scratch compact BLT |
| primary 질문 | capability-preserving byteification | Korean boundary placement의 same-rate 효과 |
| prefill boundary | one-byte lookahead | strict prefix-causal |
| decode boundary | byte와 joint prediction | 외부 고정/entropy patch schedule |
| 규모 | 1B/7B | 19.6M mechanism model |
| 언어 초점 | 범용·benchmark 중심 | Korean UTF-8/spacing |

따라서 Bolmo는 Phase 3a의 같은-graph ablation에 직접 넣을 수 없지만 publication-scale systems context에서는 필수다. 공개 1B checkpoint를 같은 Korean evaluation stream에 적용할 수 있다면 별도 pretrained-system 표에 보고한다. 다만 pretraining data와 backbone이 다르므로 절대 BPB를 W의 인과 효과와 한 ranking으로 합치지 않는다.

## 3. Boundary 분리 가설의 직접 선행

[Haltiuk, *Disentangling Language Modeling and Boundaries*](https://arxiv.org/abs/2608.03599)는 2026-08-04 공개된 position paper다. Byte model이 만드는 next-byte distribution과 boundary distribution을 분리해, 하나를 바꾸는 동안 다른 하나를 self-distillation으로 보존하는 두 실험을 제안한다.

원문이 명확히 밝히듯 두 핵심 실험은 아직 계획이며, preliminary measurement는 boundary divergence와 embedding reset 가능성에 한정된다. 따라서 이 논문은 JamoFlow의 실험 결과를 선점하지 않는다. 그러나 “boundary와 capability를 분리해 연구한다”는 문제 설정은 더 이상 신규가 아니다.

두 연구의 관계는 상보적이다.

- Haltiuk: 이미 학습된 byte model에서 boundary distribution을 **바꿔도 capability를 유지할 수 있는가**
- JamoFlow: 같은 초기화·데이터·global rate로 처음부터 학습할 때 boundary schedule이 **held-out BPB를 바꾸는가**

JamoFlow가 positive이면 boundary가 무관한 nuisance가 아니라 학습 결과에 영향을 준다는 증거가 된다. Negative이면 post-hoc boundary transfer의 가능성과 더 잘 양립한다. 어느 경우에도 post-hoc self-distillation을 직접 실행하지 않았으므로 disentanglement 자체를 입증했다고 쓰지 않는다.

## 4. H-Net++와 ATDC

[H-Net++](https://arxiv.org/abs/2508.05628)은 Persian morphologically rich language를 대상으로 bidirectional GRU router, context mixer, latent prior, morphology-related objective를 사용하고 morphological-boundary alignment를 보고한 2025 preprint다. 본문·부록의 구성은 “explicit supervision 없이 형태를 학습했다”는 강한 요약을 그대로 일반화하기 어렵게 하므로, 보고된 결과를 독립 재현 전 확정 사실로 사용하지 않는다. 그러나 **morphologically rich language용 learned byte chunking**의 선행 주장으로는 반드시 인용한다.

[Adaptive Targeted Dynamic Chunking](https://arxiv.org/abs/2605.30080)은 H-Net의 target compression factor를 training curriculum으로 변화시키고 350M–1.3B 실험을 보고한 2026 preprint다. 이는 어떤 언어 경계가 좋은가보다 **compression target을 언제 얼마나 강하게 적용할 것인가**를 다룬다.

두 연구가 주는 경계는 다음과 같다.

- W가 이겨도 learned morphology-aware chunking 일반을 이겼다고 할 수 없다.
- Exact-rate Phase 3는 ATDC의 adaptive-rate optimization과 다른 식별 질문이다.
- Gate L에서 H-Net 계열을 빼면 cross-architecture learned routing에 대한 결론은 limitation으로 남겨야 한다.

## 5. 신규성 문장의 최종 형태

현재 허용 가능한 result-independent 문장은 다음이다.

> We provide a rate-controlled Korean study of prefix-causal boundary relocation inside one BLT graph, separating Unicode geometry and observed-whitespace association from global-position rate and detector-inclusive cost.

이 문장은 “first”를 쓰지 않으며 Bolmo, H-Net++, ATDC, boundary-transfer 가설과 충돌하지 않는다. 결과가 positive여도 다음처럼 범위를 유지한다.

> Under the tested from-scratch Korean BLT setting, observed-whitespace relocation changed the quality–cost point relative to generic codepoint and one learned entropy router.

ByteFlow는 coding-rate importance와 full-sequence Top-K로 static-graph adaptive byte compression을 구현하고 강한 quality를 보고했다. 따라서 위 문장의 `one learned entropy router` 범위를 넘어 learned dynamic patching 일반에 대한 우월성을 주장하지 않는다. ByteFlow의 selector는 논문에 기술된 형태로는 suffix-independent prefix boundary를 보장하지 않고 실제 cached autoregressive latency도 보고하지 않았으므로, JamoFlow의 prefix-causal actual-inference 질문과는 구분한다.

## 6. 실행 우선순위 변화

Phase 3a의 F/C/W/S/E/EC와 gate는 바꾸지 않는다. 결과 전 policy를 추가하면 실험 family와 시간만 늘어난다. 대신 Gate J/K 통과 뒤 우선순위를 다음처럼 고정한다.

1. 50–100M same-graph replication
2. standard Korean BPE Transformer
3. 공개 Bolmo 1B Korean evaluation 또는 명시적 비호환성 기록
4. ByteFlow quality/training-throughput comparison과 incremental compatibility audit
5. 재현 가능한 H-Net/ATDC 계열 baseline
6. morphology+BPE/Hierarchical-BPE 계열 low-runtime-cost baseline

Bolmo와 H-Net 계열이 빠진 상태에서는 “state-of-the-art Korean byte LM”이나 “learned routing보다 우월”을 주장하지 않는다. 현재 연구의 가치는 강한 systems leaderboard가 아니라 boundary, rate, encoding geometry, detector cost를 분해한 측정 설계에서 나온다.
