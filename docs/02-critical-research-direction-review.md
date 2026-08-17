# JamoFlow 연구 방향 재검토

> 작성일: 2026-08-08  
> 상태: **기존 주제 확정 철회 및 후보 가설 재정의 권고**  
> 검토 대상: ChatGPT 공유 대화 「자소 분리와 효율성」 전체 8턴, [01-verification-report.md](./01-verification-report.md), [00-topic-selection.md](./00-topic-selection.md)  
> 관점: principal/top-tier LLM systems engineer 수준의 신규성·기술 타당성·end-to-end 효율·출판 가능성 검토

---

## 0. 최종 결론

JamoFlow가 다루려는 큰 문제, 즉 **문자 수준 언어모델에서 언어·문자체계의 구조를 이용해 불필요한 neural compute를 줄일 수 있는가**는 연구 가치가 있다. 그러나 현재 `00-topic-selection.md`에서 확정한 다음 주제에는 동의하기 어렵다.

> 한글 음절·어절 경계 기반 rule patcher가 BLT의 학습된 entropy patcher를 무비용으로 대체하고, 동일 BPB에서 동등 이상의 patch 길이를 달성한다.

이 문장은 아직 결론이 아니라 여러 개의 독립적인 미검증 가설을 한꺼번에 참으로 가정한 것이다.

1. 한글 경계를 저렴하게 탐지할 수 있다.
2. 그 경계가 정보량 관점에서도 좋은 patch boundary다.
3. rule boundary가 entropy boundary와 같은 품질을 낸다.
4. entropy patcher 제거가 실제 wall-clock latency로 이어진다.
5. 이 차이가 SpaceByte나 UTF-8 codepoint-aligned fixed stride와 구별되는 신규 기여다.

1번이 참이어도 2~5번은 자동으로 따라오지 않는다. 특히 **경계 탐지 비용이 싸다는 사실과 그 경계의 모델링 가치가 높다는 사실을 동일시한 것**이 현재 최종 주제의 가장 큰 오류다.

따라서 권장하는 중심 질문은 다음과 같다.

> **학습형 patcher의 품질 이득은 patcher 자체의 계산비용을 정당화하는가? 결정적 orthographic structure를 learned routing의 대체가 아닌 저비용 prior로 사용하면 rule-only와 learned-only보다 나은 BPB–latency–KV-cache Pareto frontier를 만들 수 있는가?**

권장 가제:

> **Do Learned Patchers Pay for Themselves? Orthography-Aware Cost-Constrained Patching for Byte-Level Language Models**

한국어 중심으로 좁힐 경우:

> **Orthography-Aware Hybrid Patching for Korean and Code-Mixed Byte-Latent Language Models**

---

## 1. 전체 연구 결정 과정에 대한 평가

| 단계 | 잘된 점 | 결정적 문제 | 판정 |
|---|---|---|---|
| 최초 ChatGPT 대화 | 순수 자소 AR의 sequence 증가, output masking의 낮은 ROI, latent patching·conditional compute·multi-byte generation 필요성을 찾아냄 | 문자 규칙의 legality와 실제 다음 심볼의 predictability를 혼동. 과도한 기술 결합과 conjunction novelty | 좋은 연구 브레인스토밍이지만 논문 가설로는 과대 확장 |
| `01` 검증 보고서 | 비용 계산, KV 증가, code-mixing, 3-hot 중복, 작은 모델부터 검증해야 한다는 비판이 유효 | 일부 축을 근거 없이 FATAL로 폐기하고, 가장 불확실한 `zero-cost patcher`를 유일한 free lunch로 확정 | 유용한 adversarial review지만 최종 판정기로 사용하면 안 됨 |
| `00` 최종 문서 | 범위를 줄이고 반증 가능한 주장과 중단 조건을 만들려 한 점은 좋음 | `01`의 미검증 survivor를 재검증 없이 확정 주제로 승격. 알고리즘·causality·baseline·Phase 0 불일치 | 현재 형태는 top-tier 주제로 부족하며 재프레이밍 필요 |

전체 흐름은 다음과 같이 요약된다.

> **과대 확장 → 과도한 가지치기 → 미검증 survivor의 조기 확정**

---

## 2. 최초 ChatGPT 공유 대화에 대한 비판적 검토

### 2.1 올바르게 발전한 부분

공유 대화는 다음 핵심 사항을 비교적 정확히 찾아갔다.

1. **순수 자소 autoregressive modeling은 추론을 자동으로 효율화하지 않는다.**  
   음절 하나가 여러 자소 timestep으로 늘어나므로 attention context, local decode step, KV cache가 증가할 수 있다.

2. **후보 자소 masking만으로는 큰 속도 향상을 만들기 어렵다.**  
   실제 LLM 비용은 작은 output head보다 Transformer의 attention, FFN/MoE, KV-cache access, weight movement에서 발생한다.

3. **속도 향상에는 expensive forward 횟수 감소가 필요하다.**  
   latent patching, hierarchical compression, speculative/block generation을 함께 검토한 방향은 맞다.

4. **기존 token-level 모델에 LoRA/SFT만 적용해서는 계산 단위가 바뀌지 않는다.**  
   tokenizer, local/global hierarchy, output parameterization을 바꾸려면 continued pretraining이나 architecture-level training이 필요하다는 판단은 타당하다.

5. **SpaceByte와 BLT를 핵심 선행축으로 찾은 것**도 적절했다.

### 2.2 가장 큰 개념적 오류: 규칙성은 예측 가능성이 아니다

한글 FSM은 현재 위치가 초성·중성·종성 중 무엇이어야 하는지 제한할 수 있다. 하지만 어떤 자소가 나와야 하는지는 대개 의미와 문맥에 달려 있다.

- 초성 다음이 중성이라는 사실: 구조적으로 결정 가능
- 어떤 중성이 나오는지: 여러 후보 중 semantic prediction 필요
- 한글 음절 조합이 가능한지: hard legality
- 어떤 단어·조사·어미를 쓸지: 확률적·문맥적 결정

따라서 FSM이 직접 neural prediction을 대체할 수 있는 구간은 매우 제한적이다. 한국어 형태론과 띄어쓰기 역시 분석 중의성, 표기 변이, 구어·웹 문체 때문에 hard-deterministic rule로 보기 어렵다.

### 2.3 신규성을 AND 조건의 공백으로 주장했다

공유 대화는 다음 여섯 조건을 모두 만족하는 연구가 없다는 점을 연구 공백으로 제시했다.

- 자소 atomic representation
- Hangul FSM decoding constraint
- 형태론·공백 state
- predictable transition의 expensive compute skip
- multi-jamo generation
- end-to-end pretraining 및 latency/FLOPs 평가

그러나 여러 알려진 구성요소를 AND로 묶어 빈 교집합을 만드는 것 자체는 강한 novelty가 아니다. 논문 기여가 되려면 다음이 추가로 입증되어야 한다.

1. 결합이 기술적으로 비자명한가?
2. 각 구성요소가 상호 보완적인가?
3. 결합이 기존 learned routing보다 실제로 나은가?
4. end-to-end 병목을 줄이는가?

### 2.4 놓친 직접 선행연구

공유 대화와 `01` 모두 다음 연구를 충분히 반영하지 못했다.

- [Learn Your Tokens](https://aclanthology.org/2023.findings-emnlp.662/): word boundary로 byte/character를 pooling한 뒤 단어 내부 문자를 병렬 복원한다. 경계 규칙과 multi-symbol decoding이 공존할 수 있음을 보여준다.
- [Efficient Transformers with Dynamic Token Pooling](https://aclanthology.org/2023.acl-long.353/): entropy, tokenizer, linguistically motivated boundary를 직접 비교한다.
- [AU-Net](https://arxiv.org/html/2506.14761): regex word boundary와 2단어·4단어 계층 pooling을 사용한다. 모든 관련 선행이 learned statistical signal만 쓴다는 주장의 반례다.
- [FLEXITOKENS](https://aclanthology.org/2026.findings-acl.848/): multilingual/domain adaptation에서 learnable boundary predictor를 다룬다.

따라서 남는 공백은 “문자 규칙을 사용한 연구가 없다”가 아니다. 더 정확한 공백은 다음이다.

> **결정적 orthographic structure와 learned uncertainty를 총비용 관점에서 명시적으로 결합하고, boundary predictor 자체의 비용까지 포함한 end-to-end Pareto frontier를 통제 실험한 연구가 부족하다.**

---

## 3. `01-verification-report.md`에 대한 비판적 검토

### 3.1 유지해야 할 비판

`01`의 다음 결론은 유지할 가치가 있다.

- output candidate masking은 Transformer 본체 비용을 줄이지 못한다.
- raw jamo AR은 음절/BPE 대비 sequence와 KV cache를 늘릴 위험이 크다.
- 한글 FSM은 타입을 제한할 뿐 정확한 다음 자소를 결정하지 못한다.
- code-mixing, compatibility jamo, NFC/NFD, 옛한글을 무시하면 실제 시스템에서 이득이 사라질 수 있다.
- one-person project가 1B부터 시작하는 것은 비효율적이며 작은 controlled pilot이 먼저다.
- 3-hot은 한 음절 timestep에서 자소 factorization을 사용하는 축을 이미 강하게 선점했다. [EACL 2023](https://aclanthology.org/2023.eacl-main.172/)

### 3.2 “FSM 제약과 multi-jamo generation은 자기모순”은 잘못된 판정

FSM을 매 자소 AR step에만 적용한다고 가정하면 block generation과 동시에 쓸 수 없을 수 있다. 그러나 이것은 특정 구현의 충돌이지 논리적 자기모순이 아니다.

가능한 결합은 다음과 같다.

- block diffusion의 joint output support를 FSM으로 제한
- speculative draft 전체를 DFA/FST로 검증
- multi-head joint prediction에 legality mask 적용
- invalid UTF-8/Hangul block을 verification 단계에서 reject

[Fast BLT](https://arxiv.org/abs/2605.08044)는 여러 byte의 병렬 생성과 AR verification을 함께 사용한다. 이 논문이 한글 FSM을 구현한 것은 아니지만, 두 개념을 결합할 수 있는 구조적 기반을 보여준다.

정확한 판정은 다음과 같아야 한다.

> **순수 rule-only multi-jamo drafter는 의미를 결정하지 못해 약하다. 그러나 learned block proposer + hard orthographic constraint/verifier는 여전히 유효한 별도 연구 방향이다.**

### 3.3 Three-hot의 선점 범위를 과도하게 확장했다

Three-hot은 한 syllable timestep에서 자소 factorization을 사용하고 자소식 sequence 증가를 피한다. 따라서 “한 hidden state로 한 음절의 자소를 표현·복원한다”는 축은 선점되어 있다.

하지만 다음까지 선점한 것은 아니다.

- 여러 음절의 variable-length block generation
- speculative multi-byte draft
- grammar-constrained verification
- patch boundary를 넘는 learned local generation

따라서 단일 음절 factorization과 variable-length block decoding을 같은 문제로 취급해서는 안 된다.

### 3.4 Scratchpad Patching의 해석이 부정확하다

`01`은 Scratchpad Patching을 “한 forward에서 가변 길이 byte 생성”으로 요약했지만, 실제 핵심은 patch 내부에 transient trunk state를 넣어 patch lag를 줄이는 것이다. scratchpad state는 최종 KV cache에 남지 않지만 Fast BLT식 block generation과는 다른 기술이다.

이 논문의 더 중요한 결과는 다음이다.

- Scratchpad를 추가하면 fixed patch와 SpaceByte 같은 단순 방식이 entropy/H-Net 계열과 가까워진다.
- 정확한 boundary rule보다 compute allocation이 더 중요할 수 있다.
- entropy boundary predictor를 별도 100M LM이 아니라 encoder 위의 두 Transformer layer와 auxiliary head로 구현한다.

이는 “100M patcher 제거”를 장기적인 유일 핵심 기여로 삼은 현재 최종 주제를 약화한다. [Scratchpad Patching](https://arxiv.org/html/2605.09630)

### 3.5 “유일한 free lunch”는 입증되지 않았다

`01`은 다음 두 명제를 하나로 취급했다.

1. 한글/Unicode 경계는 저렴하게 탐지할 수 있다.
2. 그 경계가 entropy boundary만큼 좋은 patch boundary다.

1번이 참이어도 2번은 따라오지 않는다.

또한 `S = 0xAC00 + (L×21+V)×28+T`는 완성형 한글 codepoint와 L/V/T를 변환하는 공식이다. UTF-8 byte boundary를 찾는 공식이 아니다. UTF-8 codepoint 경계는 generic leading/continuation bit로 이미 알 수 있다.

[SpaceByte](https://arxiv.org/html/2404.14408)는 multibyte UTF-8 leading byte를 규칙적으로 처리해 중국어와 같은 다중 byte 문자에도 global computation cadence를 제공한다. 따라서 NFC 한글의 “공짜 문자 경계” 자체는 이미 매우 가까운 선행이 존재한다.

한글 음절마다 patch를 만들면 일반적인 NFC 완성형 한글은 약 3 bytes/patch가 된다. BLT는 평균 4.5, 6, 8 bytes/patch를 실험한다. 따라서 every-syllable patching은 global step을 더 늘릴 수 있다.

여러 음절을 묶는다면 다음과 같이 기존 방식과 가까워진다.

- k개 음절마다 자르기 → codepoint-aligned fixed stride
- 어절마다 자르기 → SpaceByte/word-boundary pooling
- 문맥에 따라 묶기 → learned router 필요

결국 존재하는 것은 “싼 경계”이지, 아직 “싼 동시에 좋은 경계”가 아니다.

### 3.6 31%는 결과가 아니라 민감한 추정치

[BLT](https://arxiv.org/html/2412.09871)의 기본 entropy model이 100M이고 본체 FLOP 식에 이 모델이 포함되지 않은 점은 확인된다. 이 accounting gap은 실제로 연구 가치가 있다.

그러나 BLT는 동시에 다음도 보고한다.

- 1M~100M entropy model 실험
- 50M 이상에서 diminishing return
- 짧은 receptive field라면 lookup table 구현 가능

따라서 31%는 다음 가정에 의존하는 theoretical estimate다.

- 정확히 100M patcher 사용
- byte마다 독립적인 online forward
- FLOPs가 wall-clock에 선형 반영
- memory bandwidth, kernel launch, overlap, batching 무시
- integrated lightweight router를 사용하지 않음

문서 자체 추정으로도 8B에서는 5.3%로 감소한다. 모델이 커질수록 significance가 약해지는 기여를 top-tier scale story로 내세우려면 실제 runtime에서 이론 이상으로 patcher가 병목임을 보여야 한다.

따라서 “31% 절감”은 핵심 결과가 아니라 다음처럼 표기해야 한다.

> **가정이 명시된 theoretical upper-bound estimate이며 실제 wall-clock과 modern router 구현으로 검증 필요**

### 3.7 Amdahl·데이터·비용 분석은 sensitivity analysis로 내려야 한다

`01`의 code-mixing Amdahl 계산은 중요한 경고지만 비한글 slowdown `D=4`와 한글 비율을 가정한 예시다. 이것을 fatal conclusion으로 사용하면 안 된다. 실제 구현의 fallback 속도와 corpus mix를 측정한 뒤 sensitivity curve로 보고해야 한다.

한국어 데이터 상한과 GPU 비용 추정도 프로젝트 계획에는 유용하지만 연구 주제의 생사를 결정하는 이론적 근거는 아니다. 데이터 deduplication, 라이선스, hardware utilization에 따라 크게 달라지므로 별도 실행 계획으로 분리해야 한다.

---

## 4. `00-topic-selection.md`에 대한 비판적 검토

### 4.1 잘된 부분

- 원래의 과도한 범위를 한 개의 측정 가능한 축으로 줄였다.
- BPB, patch length, wall-clock, code-mixing curve를 언급했다.
- 실패 가능성과 kill criteria를 명시하려 했다.
- 1B full pretraining 전에 저비용 검증을 두려 했다.

이 구조는 유지할 가치가 있다.

### 4.2 주제를 확정하기에는 핵심 알고리즘이 정의되지 않았다

“한글 음절·어절 경계 기반 rule patcher”가 다음 중 무엇인지 명확하지 않다.

- every syllable patch
- k-syllable grouped patch
- eojeol patch
- 최대 byte budget 안에서 가장 가까운 음절 경계 선택
- syllable/space/punctuation hierarchy
- rule candidate 중 learned score로 선택

각 방식은 평균 patch length, patch lag, causality, novelty가 모두 다르다. 알고리즘이 정의되지 않은 상태에서 “동등 이상의 patch 길이”를 핵심 주장으로 확정할 수 없다.

### 4.3 `zero-cost` 표현은 부정확하다

Unicode parsing이 parameter-free인 것은 맞지만 실제 비용이 0은 아니다.

- byte classification
- state transition
- branch 또는 lookup
- CPU/GPU synchronization
- malformed sequence fallback
- code-mixed dispatch

특히 branch-heavy CPU router가 GPU decode loop와 synchronization을 만들면 적은 FLOPs와 낮은 latency가 일치하지 않을 수 있다. 논문 표현은 `zero-cost`보다 `parameter-free`, `constant-time`, `negligible measured overhead` 중 실측에 맞는 표현을 써야 한다.

### 4.4 prefix-causality가 정의되지 않았다

BLT 생성 시 patch boundary는 아직 생성되지 않은 미래 byte에 의존할 수 없다. 학습 데이터 전체를 본 뒤 얻은 형태소·어절 segmentation을 generation rule로 그대로 사용하면 leakage다.

필요한 조건은 다음과 같다.

> 모든 boundary policy `f`에 대해 prefix에서 계산한 결정이 전체 sequence에서 계산한 동일 prefix의 결정과 같아야 한다.

음절 종료, 이미 출력된 공백, UTF-8 parser state는 causal하게 만들 수 있다. 반면 완성된 단어에 대한 형태소 분석이나 오른쪽 문맥을 쓰는 segmentation은 별도 예측기가 필요하다.

### 4.5 핵심 가설을 결론처럼 서술했다

현재 문서는 다음을 이미 사실로 전제한다.

- rule boundary ≥ entropy boundary
- 동일 BPB 달성
- 동등 이상의 patch length
- 31% 실제 절감

이는 모두 실험 대상이다. 논문 주제 문장은 결과를 선취하지 않는 질문형 또는 조건부 가설이어야 한다.

### 4.6 Phase 0이 최종 주제와 맞지 않는다

현재 Phase 0의 다음 지표는 폐기한 deterministic jamo skip 가설을 검사한다.

- `|A(s)|=1`
- 자소 n-gram skip 상한 `f`
- `Δ=f−f₀`

이 수치는 rule patcher의 boundary quality, BPB, patch lag, end-to-end latency를 직접 예측하지 못한다. 원래 가설이 왜 폐기됐는지를 기록하는 appendix 분석으로는 유용하지만 새 주제의 Go/No-Go gate로는 부적절하다.

5%p·15%p 기준도 runtime 또는 BPB와 연결되지 않은 임의 임계값이다.

### 4.7 baseline 분류가 섞여 있다

Three-hot과 BPE+MTP는 patch boundary policy가 아니므로 direct patcher baseline과 같은 표에서 비교하면 해석이 흐려진다.

- direct patching baseline: fixed stride, codepoint stride, SpaceByte, entropy BLT, H-Net, Scratchpad variants
- representation ablation: NFC UTF-8, NFD/jamo, syllable + three-hot
- output generation ablation: AR byte, MTP, speculative, Fast BLT-style block generation

세 축을 분리해야 어느 부분이 실제 이득을 만들었는지 알 수 있다.

### 4.8 `JamoFlow`라는 이름과 실제 최종 방법이 불일치한다

현재 최종 주제는 raw UTF-8 byte와 음절·어절 경계를 사용하며 자소 representation이나 자소 FSM을 핵심 방법에서 제거했다. 저장소 codename으로 JamoFlow를 유지할 수는 있지만 논문 제목에서 Jamo가 기술적 기여처럼 보이면 리뷰어가 문제 삼을 수 있다.

자소가 paper title에 남으려면 최소한 다음 중 하나가 주요 실험축이어야 한다.

- jamo-factorized representation
- Hangul L/V/T-aware candidate feature
- jamo legality-constrained block decoding
- NFC/NFD/jamo encoding ablation

---

## 5. 권장 1차 연구 방향

### 5.1 연구 질문

> **동일한 average bytes/patch에서 deterministic orthographic boundary, learned entropy boundary, orthography-constrained learned boundary의 품질을 비교한다. 이후 boundary detector 비용까지 포함한 동일 end-to-end inference budget에서 hybrid policy가 BPB–latency–KV-cache Pareto frontier를 개선하는지 검증한다.**

### 5.2 권장 아키텍처

#### A. Deterministic causal boundary automaton

다음 상태를 parameter-free하게 추적한다.

- UTF-8 leading/continuation state
- Unicode codepoint/grapheme boundary
- Hangul syllable 및 L/V/T class
- whitespace·punctuation
- malformed input fallback
- script transition

#### B. Candidate-restricted learned scorer

learned router를 매 byte마다 실행하지 않고 automaton이 생성한 causal candidate position에서만 실행한다.

- rule은 다음 내용을 생성하지 않는다.
- rule은 경계 후보와 structural feature만 제공한다.
- learned scorer가 context-sensitive information density를 판단한다.

#### C. Cost/budget controller

- target average patch size
- maximum patch length
- learned router evaluation budget
- script별 fallback policy

를 명시적으로 제어한다.

총비용은 개념적으로 다음과 같이 분해한다.

```text
C_total / byte
  = C_local
  + C_global / E[bytes per patch]
  + C_router / E[bytes per router evaluation]
```

핵심 기여는 `C_router`를 무조건 0으로 만든다는 주장이 아니라, boundary quality를 유지하면서 router evaluation 빈도와 용량을 줄이는 것이다.

### 5.3 사전 가설

결과를 선취하지 않는 형태로 다음을 둔다.

- **H1:** rule-only policy는 router 비용을 줄이지만 context-sensitive boundary quality가 낮아질 수 있다.
- **H2:** hybrid policy는 learned-only의 boundary quality 대부분을 유지하면서 router 비용을 줄일 수 있다.
- **H3:** 이득의 일부는 한글 고유 구조가 아니라 generic UTF-8/codepoint alignment에서 나올 수 있다.
- **H4:** hard orthographic constraint는 생성 품질보다 validity·robustness에서 먼저 이득을 보일 수 있다.
- **H5:** patcher FLOPs 절감이 실제 latency 향상으로 이어지는지는 batch size와 memory-bandwidth regime에 따라 달라진다.

---

## 6. 수정된 실험 계획

### 6.1 Phase 0 — corpus 및 routing audit

GPU 학습 전에 다음을 측정한다.

1. **표현 분포**
   - NFC/NFD 비율
   - compatibility jamo 및 단독 자모
   - 옛한글·emoji·숫자·라틴·코드·URL
   - 문서/domain별 code-mixing 비율

2. **causality 검증**
   - 각 boundary rule에 prefix invariance test 적용
   - 미래 문자를 사용한 offline segmentation 제거

3. **matched patch-rate 비교**
   - 목표 평균 3/4.5/6/8 bytes per patch
   - fixed byte, codepoint-aligned fixed, SpaceByte, entropy, rule variants 비교

4. **boundary quality proxy**
   - oracle next-byte surprisal peak coverage
   - high-surprisal position과 직전 global update 사이 거리
   - patch 내 surprisal 분포
   - script별 patch-length tail 및 maximum lag
   - learned router가 필요해지는 candidate 비율

5. **실제 patcher cost**
   - theoretical FLOPs
   - parameter memory
   - batch 1/8/64 latency
   - CPU/GPU implementation 및 synchronization 비용

`|A(s)|=1`, 자소 conditional entropy, `Δ=f−f₀`는 원래 deterministic skip 가설의 폐기 근거를 기록하는 appendix 분석으로 이동한다.

### 6.2 Phase 1 — 100~300M controlled training

동일 backbone에서 다음을 비교한다.

#### Direct patcher baselines

1. fixed byte stride
2. UTF-8 codepoint-aligned fixed stride
3. SpaceByte
4. BLT entropy patching: 가능한 여러 patcher size
5. H-Net learned routing
6. Scratchpad + fixed
7. Scratchpad + SpaceByte
8. Scratchpad + entropy
9. orthography rule-only
10. orthography + learned hybrid

#### Representation ablation

1. raw NFC UTF-8
2. NFD/jamo UTF-8
3. syllable timestep + three-hot factorization
4. 필요하면 alternative compact character encoding

UTF-8 자체가 efficiency와 multilingual fairness에 최적이 아닐 수 있으므로 representation과 patching 효과를 분리해야 한다. [From Bytes to Subwords](https://aclanthology.org/2026.findings-acl.530/)

#### Fairness controls

- 동일 raw training bytes
- 동일 effective context in bytes
- 동일 global trunk parameter scale
- parameter-matched 비교와 total-FLOPs-matched 비교를 별도 보고
- 동일 average bytes/patch 비교
- 최소 3 seeds 또는 pilot variance에 근거한 반복 수

### 6.3 평가 언어와 domain

- 한국어 natural text
- 한국어 중심 code-mixed text
- 중국어 control
- 영어 control
- code/URL-heavy text
- 가능하면 일본어 또는 combining-mark가 많은 추가 script

중국어는 한국어와 마찬가지로 3-byte UTF-8 문자가 많지만 Hangul L/V/T 조합구조는 없다. 중국어에서도 동일한 이득이 나오면 효과는 한글 고유 규칙보다 generic codepoint alignment일 가능성이 높다.

[H-Net](https://arxiv.org/html/2507.07955)은 중국어 환경에서 learned chunking이 space heuristic보다 강한 결과를 보고한다. 이것이 한국어에서 rule이 질 것이라는 증명은 아니지만, `rule ≥ learned`를 사전 결론으로 둘 수 없게 만드는 직접 반례다.

### 6.4 필수 지표

#### 모델링 품질

- BPB
- downstream 한국어 이해·생성
- code-mixed/domain별 BPB
- long-tail 및 noisy input robustness

#### 실제 시스템 효율

- batch 1/8/64 bytes/sec
- normalized Unicode characters/sec
- TTFT
- inter-character latency
- p50/p95 decode latency
- KV bytes/generated character
- resident parameter memory
- patcher 포함 total FLOPs와 wall-clock

#### 구조 정확성

- invalid UTF-8 rate
- invalid Hangul/jamo sequence rate
- normalization instability
- malformed·compatibility jamo fallback rate

### 6.5 수정된 Go/No-Go 기준

현재의 `Δ < 5%p` 같은 기준은 제거한다. pilot variance와 실제 runtime noise를 먼저 측정한 뒤 최종 실험 전에 다음 형태로 사전 고정한다.

1. rule-only 또는 hybrid가 trivial codepoint-aligned fixed stride와 SpaceByte를 이겨야 한다.
2. 같은 BPB confidence interval에서 유의미한 end-to-end latency 또는 KV 개선이 있어야 한다.
3. 또는 같은 measured latency에서 유의미한 BPB/downstream 개선이 있어야 한다.
4. 이득이 code-mixed 조건에서 즉시 사라지지 않아야 한다.
5. 100~300M에서 scale trend가 없으면 1B 실험으로 넘어가지 않는다.

10% latency 같은 절대 임계값을 사용하려면 먼저 hardware variance와 measurement error를 측정하고, 최종 결과를 보기 전에 고정해야 한다.

---

## 7. 원래 아이디어를 보존하는 별도 2차 연구

원래 대화의 `rule + multi-jamo generation`은 완전히 폐기할 필요가 없다. 다만 patching 논문과 분리해야 한다.

권장 주제:

> **Orthography-Constrained Block Decoding for Byte-Level Language Models**

구성:

1. Fast BLT-style learned block proposer
2. UTF-8/Hangul DFA가 block 전체 legality 제한
3. speculative 또는 diffusion verification
4. invalid block rejection/fallback
5. accepted bytes per expensive forward 측정

규칙은 semantic byte를 직접 생성하지 않는다. learned proposal이 구조적으로 불가능한 byte sequence를 생성하지 못하게 하고 verification을 보조한다.

[Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models](https://arxiv.org/abs/2606.14122)는 perplexity보다 UTF-8 structural validity가 늦게 안정될 수 있음을 보고한다. 따라서 hard constraint는 speed가 아니라 correctness·robustness에서 먼저 기여할 가능성이 있다.

이 축의 핵심 평가:

- accepted bytes/full-model forward
- verification rejection rate
- invalid UTF-8/Hangul rate
- actual latency와 memory-bandwidth
- rare character 및 noisy prompt robustness

1차 patching 연구와 2차 block decoding 연구를 한 논문에 다시 결합하면 원래의 scope explosion이 반복되므로 분리하는 것이 원칙이다.

---

## 8. 출판 가치와 예상 심사 판단

### 8.1 현재 `00` 주제 그대로 제출할 경우

리뷰어의 핵심 공격은 다음과 같을 가능성이 높다.

1. “SpaceByte의 UTF-8 규칙과 무엇이 다른가?”
2. “왜 syllable boundary가 information boundary인가?”
3. “100M entropy model만 가정해 이미 낡은 구현을 공격한 것 아닌가?”
4. “3-byte every-syllable patch가 어떻게 4.5~8-byte entropy patch보다 긴가?”
5. “FLOPs 추정이 아니라 실제 batch-1 latency 개선은 어디 있는가?”
6. “한글 특성인지 generic UTF-8 alignment인지 분리했는가?”
7. “generation boundary가 prefix-causal한가?”

현재 형태는 긍정 결과가 나오더라도 workshop 또는 좁은 Findings 수준일 가능성이 높다.

### 8.2 top-tier 가능성이 생기는 조건

다음 조건을 충족하면 충분히 강한 연구 프로그램이 된다.

- patcher cost와 boundary quality를 독립적으로 분해
- rule-only, learned-only, hybrid를 동일 backbone에서 비교
- codepoint stride, SpaceByte, H-Net, Scratchpad를 공정하게 통제
- multilingual/script control과 code-mixing 포함
- 실제 decode latency·KV-memory Pareto 개선
- 100~300M을 필터로 사용하고 살아남은 설정을 최소 1B에서 검증
- 한국어 전용 트릭이 아닌 일반적인 cost-aware boundary mechanism을 제시

### 8.3 negative result의 조건

한국어 corpus에서 rule과 entropy boundary의 alignment만 측정한 결과는 top-tier negative paper로 부족하다. 음수 결과가 논문이 되려면 다음이 필요하다.

- 여러 문자체계와 domain을 포함한 benchmark
- oracle boundary upper bound
- boundary quality와 detector cost의 체계적 분해
- 공개 가능한 evaluation harness
- 향후 patch-based LM 연구에 재사용 가능한 설계 원칙

---

## 9. 기존 문서에 반영해야 할 수정 사항

### `00-topic-selection.md`

1. `주제 확정`을 `후보 주제: Phase 0 검증 중`으로 변경
2. `zero-cost`를 `parameter-free` 또는 실측 후 `negligible overhead`로 변경
3. 31%를 확정 수치가 아닌 assumption-dependent upper bound로 변경
4. rule patcher의 causal algorithm을 구체적으로 정의
5. `rule ≥ entropy`를 열린 연구 질문으로 변경
6. Phase 0을 matched patch-rate 및 boundary-cost audit로 교체
7. Three-hot/MTP를 direct patcher baseline에서 분리
8. SpaceByte와 UTF-8 codepoint stride를 최우선 baseline으로 승격
9. H-Net 및 Scratchpad integrated router를 현행 baseline으로 추가
10. 중국어 control과 실제 wall-clock evaluation을 필수화

### `01-verification-report.md`

1. `FSM + multi-jamo = A∧¬A` 판정을 특정 naive AR 구현의 충돌로 완화
2. multi-jamo 전체 폐기 대신 learned proposal + hard constraint 방향을 별도 연구로 보존
3. Scratchpad Patching 설명 수정
4. AU-Net이 rule-based regex hierarchy를 사용한다는 사실 반영
5. Learn Your Tokens, Dynamic Token Pooling, FLEXITOKENS 추가
6. `유일한 free lunch`를 `검증할 가치가 있는 cost-accounting hypothesis`로 완화
7. Unicode Hangul decomposition 공식과 UTF-8 boundary detection을 구분
8. 31% 수치에 가정·현대 구현·hardware caveat 추가
9. 기존 `Δ=f−f₀`를 원안 폐기 appendix로 이동

---

## 10. 최종 권고

JamoFlow를 중단할 이유는 없다. 다만 현재 확정 주제를 그대로 구현하는 것은 권하지 않는다.

가장 올바른 다음 행동은 다음 순서다.

1. 현재 주제를 후보 가설로 되돌린다.
2. rule patcher의 정확한 prefix-causal algorithm을 먼저 정의한다.
3. generic UTF-8/codepoint rule, SpaceByte, entropy, learned, hybrid를 동일 patch rate에서 비교한다.
4. 100M entropy patcher를 제거한다는 전제가 아니라 실제 patcher cost distribution을 측정한다.
5. corpus proxy가 살아남으면 100~300M controlled pilot을 진행한다.
6. hybrid가 실제 Pareto frontier를 개선할 때만 1B scale로 간다.
7. multi-jamo/FSM은 orthography-constrained block decoding이라는 별도 후속 연구로 분리한다.

최종적으로 남겨야 할 중심 명제는 다음이다.

> **한글 규칙이 neural computation을 공짜로 대신한다는 주장이 아니라, 결정적 문자구조가 learned boundary routing의 비용을 얼마나 줄이면서 그 품질을 보존할 수 있는지를 측정하고 설계한다.**

이 프레이밍은 최초 대화의 직관을 보존하면서도, `01`의 정당한 비판을 수용하고, `00`의 조기 확정을 피한다. 또한 결과가 positive이든 negative이든 무엇을 새로 알게 되었는지가 명확한 연구 질문을 제공한다.
