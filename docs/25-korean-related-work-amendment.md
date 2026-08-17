# Phase 3 amendment: Korean tokenization prior art and claim boundary

> 작성일: 2026-08-10  
> 상태: **Phase 3 primary 결과 생성 전 문헌 재감사**  
> 상위 문서: [연구 방향 재검토](./02-critical-research-direction-review.md), [Phase 3 protocol](./22-phase3-confirmatory-protocol.md)  
> 목적: 기존 검토에서 빠진 한국어 직접 선행을 반영하고, 결과와 무관하게 주장 범위와 publication-scale baseline을 고정한다.

> 2026-08-10 후속 최신 문헌 감사에서 Bolmo와 boundary-disentanglement 선행을 추가 확인했다. 상세 판정은 [36 amendment](./36-latest-boundary-literature-amendment.md)를 따른다.

## 1. 결론

JamoFlow는 다음 중 어느 것도 신규성으로 주장할 수 없다.

- 한국어 모델에 자소 분해를 처음 사용한다.
- 한국어 tokenization에 형태소 또는 어절 경계를 처음 사용한다.
- 언어학적 경계로 sequence를 줄이는 방법을 처음 제안한다.
- rule-based boundary가 learned segmentation보다 일반적으로 우월하다.

남는 질문은 더 좁다.

> **Raw UTF-8 byte latent model의 같은 graph와 같은 global-position rate 안에서, 이미 관측된 한국어 whitespace를 이용한 prefix-causal boundary relocation이 generic codepoint cadence보다 재현 가능한 BPB 이득을 주는가? 별도 entropy router의 품질과 총비용을 함께 놓았을 때 이 policy가 Pareto frontier에 남는가?**

이 질문은 vocabulary tokenization이나 형태소 분석기의 우열을 묻지 않는다. Phase 3의 W는 형태소 경계를 계산하지 않으며, 관측된 Unicode whitespace 주변에서 고정 grid boundary를 제한적으로 옮길 뿐이다.

## 2. 추가로 확인한 직접 선행

### 2.1 Korean morphology-aware tokenization

[Park et al. (AACL 2020)](https://aclanthology.org/2020.aacl-main.17/)은 한국어 NLP 과제에서 여러 tokenization strategy를 비교했다. 형태소 분석 뒤 BPE를 적용한 hybrid가 번역과 다수 NLU 과제에서 가장 좋았지만 KorQuAD에서는 plain BPE가 가장 좋았다. 따라서 “언어학적 segmentation은 항상 우월하다”는 결론도 선행 결과와 맞지 않는다.

[Lee and Shin (W-NUT 2021)](https://aclanthology.org/2021.wnut-1.45/)은 인터넷 은어·고유명사·신조어를 겨냥한 Korean Morphologically Tight-Fitting Tokenizer를 제안했다. Formal text의 morphology rule이 noisy user text로 그대로 일반화되지 않는다는 점은 private Markdown diagnostic과 code-mixing stratum을 유지해야 할 근거다.

[Moon et al. (LREC 2022)](https://aclanthology.org/2022.lrec-1.531/)은 한국어 공백이 영어식 word boundary보다 큰 단위이고, 기존 형태소 분석 결과가 spacing과 subcharacter normalization 정보를 잃어 생성 시 원문 복원을 어렵게 할 수 있음을 지적했다. 따라서 JamoFlow에서 whitespace를 곧바로 “word boundary”라고 부르지 않고 **eojeol/spacing signal**로 한정한다.

[Lee et al. (EACL 2026)](https://aclanthology.org/2026.eacl-short.22/)의 *Morpheme Matters*는 어절 간·어절 내 형태 구조를 반영한 subword selection으로 한국어 모델을 pretrain했고, 기존 접근보다 일반적으로 좋은 task 성능과 더 적은 input tokens를 보고했다. 이는 한국어 morphology-aware efficiency의 가장 직접적인 선행이다.

### 2.2 Korean subcharacter representation

[Cognetta et al. (EACL 2023)](https://aclanthology.org/2023.eacl-main.172/)은 한 음절 timestep을 유지한 three-hot 자소 factorization으로 syllable embedding parameter를 99.6% 줄이면서 번역 품질 저하가 없음을 보였다.

[Lee et al. (LoResMT 2025)](https://aclanthology.org/2025.loresmt-1.8/)은 jamo 기반 subword가 low-resource·restricted-vocabulary 한국어 번역에서 syllable·byte baseline보다 일관되게 좋다고 보고했다.

[Kim et al. (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.104/)의 SCRIPT는 기존 subword PLM embedding에 자소 compositional knowledge를 주입한다.

이 연구들은 jamo representation의 가치를 보이지만 Phase 3의 intervention과는 축이 다르다. Phase 3는 source NFC UTF-8 bytes를 바꾸지 않고 patch start만 바꾼다. 따라서 positive 결과가 나와도 자소 representation의 우월성으로 해석하지 않는다.

### 2.3 Dynamic and byte-level segmentation

다음 선행은 기존 [인용 검증](./03-citation-verification.md)의 판정을 유지한다.

- [Dynamic Token Pooling](https://aclanthology.org/2023.acl-long.353/): learned, entropy-supervised, tokenizer-supervised, linguistically motivated boundary를 비교했다.
- [Learn Your Tokens](https://aclanthology.org/2023.findings-emnlp.662/): word boundary에서 byte/character를 pooling하고 내부 symbol을 병렬 복원했다.
- [SpaceByte](https://arxiv.org/abs/2404.14408): parameter-free spacelike rule로 large block 위치를 정했다.
- [BLT](https://aclanthology.org/2025.acl-long.453/): next-byte entropy로 dynamic byte patches를 만들고 8B/4T-byte까지 scaling했다.
- [H-Net](https://arxiv.org/abs/2507.07955): content- and context-dependent chunking을 end-to-end로 학습했다.
- [Scratchpad Patching](https://arxiv.org/abs/2605.09630): patch lag와 compute allocation을 boundary rule에서 분리했다.
- [FLEXITOKENS](https://aclanthology.org/2026.findings-acl.848/): distribution adaptation을 위한 flexible learned boundary objective를 제안했다.

그러므로 신규성은 rule의 존재가 아니라 **Korean UTF-8 geometry에서 same-rate causal boundary와 router-inclusive total cost를 분리하는 통제 실험**에만 둘 수 있다.

## 3. Phase 3 해석에 미치는 영향

### 3.1 변하지 않는 항목

문헌 누락은 구현 bug나 hypothesis 변경 사유가 아니다. 다음은 바꾸지 않는다.

- F/C/W/S/E/EC 정의
- HPLT3 split과 byte budget
- primary contrast와 effect threshold
- Gate I/J/K
- 결과를 본 뒤 morphology feature를 W에 추가하지 않는 원칙

### 3.2 강화되는 claim guardrail

Positive 결과가 나와도 다음처럼만 쓴다.

- 허용: “observed whitespace-conditioned boundary relocation improved BPB over generic codepoint relocation in this controlled Korean byte-latent setting”
- 금지: “Korean morphology explains the gain”
- 금지: “our method tokenizes Korean more efficiently than morphology-aware BPE”
- 금지: “whitespace is the optimal Korean boundary”
- 금지: “linguistic rules beat learned tokenization”

W는 형태소 label, POS, future context를 사용하지 않는다. 따라서 W의 효과가 형태론적 일치 때문인지 punctuation·formatting·document structure 때문인지는 현재 experiment alone으로 식별할 수 없다.

## 4. Publication-scale baseline 결정

Gate J와 K가 모두 통과해 Gate L로 갈 때에는 같은-graph 세 policy 확장만으로 “한국어 LM 효율”을 일반 주장하지 않는다. 다음 cross-architecture control을 별도 표로 추가한다.

1. raw-byte Phase 3 backbone의 가장 강한 policy
2. standard BPE token Transformer
3. 재현 가능한 경우 morphology-then-BPE Korean Transformer
4. 공개 artifact가 평가 가능한 경우 Bolmo 1B의 Korean byte-interface 결과를 별도 pretrained-system 표에 제시

비교는 최소한 parameter count, training raw bytes, training FLOPs, context bytes, test BPB 또는 byte-normalized NLL, teacher-forced throughput을 함께 보고한다. Token Transformer와 BLT의 graph가 다르므로 이 표는 W boundary의 인과 ablation이 아니라 external systems comparison이다.

Bolmo는 다른 원자료와 subword source model에서 byteification됐으므로 Phase 3의 같은-data ranking에 섞지 않는다. 공개 checkpoint를 동일 Korean corpus에 평가하더라도 domain·pretraining 차이를 명시하고, BPB 절대값으로 W의 인과 우월성을 주장하지 않는다.

Morpheme Matters 구현을 사용한다면 해당 논문의 tokenizer 이름을 그대로 쓰고 재현 범위를 명시한다. 자체 근사 morphology tokenizer를 원 논문의 baseline인 것처럼 부르지 않는다.

Gate J/K가 실패하면 이 cross-architecture scale-up은 method rescue 용도로 실행하지 않는다. 대신 기존 Korean tokenization 연구와 Phase 3 failure geometry를 연결해, byte patch boundary의 효과가 vocabulary compression 결과와 왜 다를 수 있는지를 논의한다.

## 5. 논문 related-work 구조

최종 초안은 서로 다른 세 축을 섞지 않는다.

1. **Korean representation:** syllable, jamo, three-hot, SCRIPT
2. **Korean vocabulary tokenization:** morphology+BPE, noisy-text tokenizer, Morpheme Matters
3. **Tokenizer-free latent computation:** Dynamic Token Pooling, SpaceByte, BLT, H-Net, Scratchpad, FLEXITOKENS

JamoFlow의 실험은 세 번째 축에 속하며 첫 번째와 두 번째 축은 동기·한계·external baseline이다. 이 구분이 없으면 “자소 연구인가, tokenizer 연구인가, BLT patcher 연구인가”라는 기존 scope ambiguity가 다시 생긴다.
