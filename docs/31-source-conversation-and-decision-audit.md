# 최초 대화와 연구 의사결정 전 과정 감사

> 작성일: 2026-08-10
> 상태: **Phase 3 C/W 결과 확인 전 고정**
> 검토 대상: ChatGPT 공유 대화 「자소 분리와 효율성」 전체 8턴, [01 검증 보고서](./01-verification-report.md), [00 주제 선정 문서](./00-topic-selection.md), 이후의 [02 재검토](./02-critical-research-direction-review.md), [28 신규성·식별 감사](./28-novelty-and-identification-audit.md), [30 HF BLT 정렬 감사](./30-hf-blt-alignment-audit.md)
> 원 대화: [ChatGPT 공유 링크](https://chatgpt.com/share/6a76f7e2-6a40-83ee-8b8a-a5038d9138ea)
> 관점: principal/top-tier LLM systems engineer 수준의 개념 타당성, 인과 식별, 총비용, 신규성, 출판 가능성

## 0. 최종 판정

최초 대화는 **좋은 아이디어 생성 과정**이지만 연구 결론으로 사용하면 안 된다. 대화는 순수 자소 autoregressive 모델의 sequence 증가와 output masking의 낮은 ROI를 정확히 발견했다. 그러나 이후 다음 세 개의 서로 다른 명제를 반복해서 섞었다.

1. 한글 표기가 **구조적으로 합법**인지 알 수 있다.
2. 다음 내용이 **확률적으로 쉽게 예측**되는지 알 수 있다.
3. 그 정보를 이용해 실제 장치에서 **비싼 neural step을 제거**할 수 있다.

1은 대체로 참이지만 2와 3은 자동으로 따라오지 않는다. 이 구분이 흐려지면서 자소 표현, 형태론, 띄어쓰기, semantic patching, adaptive depth, multi-byte generation이 하나의 거대한 구조로 합쳐졌고, 구성요소의 빈 교집합이 신규성처럼 제시됐다.

`01`은 이 과장을 강하게 교정했지만 반대 방향의 과잉 판정을 만들었다. 특히 특정 구현의 충돌을 논리적 자기모순이라고 하고, 민감한 비용 추정을 확정값처럼 쓰고, 마지막으로 살아남은 cheap boundary 가설을 충분한 검증 없이 “유일한 free lunch”로 승격했다. 문서 상단 errata가 이를 상당 부분 바로잡았으나 본문은 역사적 기록이라 서로 모순되는 문장이 함께 남아 있다.

`00`의 개정판은 주제를 질문형으로 되돌리고 matched-rate·total-cost 실험으로 바꾼 점이 옳다. 다만 현재 실제 연구는 다시 더 좁아졌다. JamoFlow가 지금 식별하는 것은 “한글 규칙이 compute를 대신한다”가 아니다.

> **동일한 Hugging Face BLT graph와 동일한 global-position 수에서, generic UTF-8 codepoint-safe cadence를 이미 관측된 whitespace에 맞춰 prefix-causal하게 재배치했을 때 Korean byte modeling quality가 달라지는가? 그 차이가 rate, router, padding, memory를 포함한 총비용 비교에서도 남는가?**

이 질문은 연구할 가치가 있다. 그러나 현재 19.6M 모델 한 규모의 결과만으로는 top-tier 효율 논문이 되지 않는다. 긍정 결과라면 더 큰 공개 학습, tokenized 외부 baseline, 장치 실측, 한국어 semantic 평가가 필요하다. 음수 결과라면 여러 문자체계에서 boundary semantics와 detector cost를 분해한 benchmark로 확장해야 한다.

## 1. 감사 방법과 증거의 한계

2026-08-10에 사용자가 열어 둔 공유 대화를 `aside-browser`로 직접 읽고, 8개의 사용자 발화와 각 답변이 만든 주장 변화를 문서와 대조했다. 원 대화의 사적인 주변 탭·기록은 연구 근거로 수집하거나 저장하지 않았다. 이 문서는 대화 원문 전체를 복제하지 않고, 연구 결정에 영향을 준 주장만 분석한다.

다음 세 층을 구분했다.

- **사실 층:** 논문·모델이 존재하고 인용 수치가 원문과 맞는가
- **추론 층:** 그 선행에서 JamoFlow 가설이 실제로 따라오는가
- **시스템 층:** 제안이 FLOPs, memory movement, KV, sequential steps, wall-clock 중 무엇을 줄이는가

논문이 실존한다는 사실은 그 논문들이 한 결론을 지지한다는 뜻이 아니다. 여러 agent가 같은 검색 결과를 반복 확인하는 것도 독립 재현이나 실험 증거를 대신하지 않는다.

## 2. 최초 공유 대화의 턴별 감사

### Turn 1 — 순수 자소 모델에 대한 최초 답변

유지할 판단:

- 자소마다 global autoregressive step을 쓰면 sequence가 늘고 KV와 decode step이 악화될 수 있다는 지적은 맞다.
- 자소를 compositional feature로 쓰되 main Transformer 단위는 음절·latent patch로 유지하자는 구분은 타당하다.
- vocabulary 크기 감소와 attention 효율을 같은 것으로 보지 않은 점도 좋다.

부족한 점:

- factorized representation의 parameter 효율과 end-to-end inference 효율을 연결할 증거가 없었다.
- local encoder가 자소를 압축한다고 해서 BPE보다 더 좋은 rate–quality trade-off가 생기는 것은 아니다.
- “한국어라서 특히 가능성 있다”는 문장은 비교 언어·동일 byte budget 없이 나온 직관이다.

### Turn 2 — 최신 자소·byte 연구 검색

유지할 판단:

- Three-hot 계열이 sequence 증가 없이 자소 구조를 표현한다는 사실을 핵심 선행으로 둔 것은 적절하다.
- 자소 기반 BPE와 BLT를 찾아 representation과 latent patching을 분리한 것은 유용하다.

부족한 점:

- encoder형 NLU, 저자원 MT, embedding factorization, decoder-only byte LM의 결과를 하나의 증거선처럼 합쳤다. 이들은 목적함수·규모·평가가 다르다.
- “연구가 sequence 증가를 피하는 쪽으로 수렴한다”는 방향성은 합리적이지만 체계적 문헌 검토로 입증하지 않았다.
- BLT의 robustness·inference 장점을 한국어 자소 구조의 효율 근거로 읽은 것은 외삽이다.
- 한국어 표현학습에서 자소가 유용하다는 결과는 runtime compute 감소를 증명하지 않는다.

### Turn 3 — Kimi K3 fine-tuning 제안

유지할 판단:

- LoRA/SFT만으로 tokenizer와 autoregressive 계산 단위가 바뀌지 않는다는 답은 맞다.
- 새로운 local/global graph라면 continued pretraining 또는 별도 architecture training이 필요하다는 구분도 맞다.

부족한 점:

- 초대형 MoE를 출발점으로 제안하는 순간 mechanism identification보다 시스템 통합과 자원 문제가 지배한다.
- teacher–student distillation은 원 아이디어의 효과가 아니라 teacher 품질·distillation recipe의 효과를 섞는다.
- 작은 synthetic/mechanism experiment 전에 7B 이상을 논한 것은 연구 순서가 거꾸로다.

### Turn 4 — 한글 규칙, 실사용 음절, 형태론, 띄어쓰기의 계층화

유지할 판단:

- 띄어쓰기를 hard boundary가 아니라 override 가능한 prior로 보자는 경고는 중요하다.
- 문자·음절·형태·어절의 서로 다른 granularity를 local/global hierarchy와 연결해 보자는 것은 탐색할 만하다.

결정적 문제:

- `30 token → 12 latent token`, `50,000 → 20,000` 같은 수치는 측정도 rate definition도 없는 예시였다. 이후 KV·context·reasoning 이득은 이 가상 압축률에 의존한다.
- morphology encoder와 semantic patcher가 무엇을 보존하고 어떤 loss로 복원되는지 정의되지 않았다.
- 표면 중복을 줄이면 reasoning capacity가 늘어난다는 주장은 plausible story일 뿐 인과 증거가 없다.
- “공짜 segmentation signal”과 “variation 때문에 learned override 필요”를 동시에 말하면서, override 비용과 실패율은 계산하지 않았다.
- 단계가 많아질수록 각 층의 parameter, latency, synchronization, training instability가 추가되지만 이 비용은 빠졌다.

이 턴에서 아이디어는 커졌지만 가설은 선명해지지 않았다. **architecture diagram이 생겼다고 mechanism이 정의된 것은 아니다.**

### Turn 5 — 사용자가 요구한 ‘자소 기반 제한적 추론’ 재정식화

이 답변은 가장 중요한 시스템 한계를 스스로 인정했다.

- 후보 50개를 10개로 줄이는 것은 attention·FFN 비용을 거의 줄이지 않는다.
- 속도 향상에는 expensive forward 횟수 자체를 줄여야 한다.
- hard pruning은 이름·신조어·외래어·오타를 훼손할 수 있다.

그러나 핵심 결론에는 순환논증이 있다.

> 모델이 먼저 “한” 또는 “한국어”를 결정하면 규칙 decoder가 자소를 무료로 전개할 수 있다.

“한”이나 “한국어”를 고르는 것이 바로 내용 예측이다. 그 결정을 한 step에 했다면 비교 대상은 raw-jamo LM이 아니라 음절·subword·block LM이다. 이후의 자소 전개는 detokenization이며, 한글 규칙이 semantic neural decision을 제거한 것이 아니다. 이 baseline을 생략해 rule decoder의 공로를 과대평가했다.

추가 문제:

- `ㄱ`을 곧바로 초성 state로 해석했지만 compatibility jamo, conjoining jamo, NFC/NFD는 서로 다른 serialization이다. alphabet을 먼저 고정해야 FSM이 정의된다.
- orthographic parser state, corpus frequency prior, lexical statistics, spacing probability, semantics를 모두 “규칙”이라는 한 층으로 묶었다. 뒤의 세 항목은 결정적 문법이 아니다.
- `50→10~20→5~10→3~5`는 corpus와 state definition이 없는 추정이다.
- 공백 logit을 아예 만들지 않는 small head와 full hidden state 계산 절감은 다른 문제다.
- “언어가 아는 것을 neural network가 다시 배울 필요가 없다”는 철학은 legality에는 맞지만, 어떤 합법 문자열을 쓸지는 여전히 모델이 결정해야 한다.

### Turn 6 — SpaceByte·BLT·제약 디코딩을 합친 딥리서치

유지할 판단:

- SpaceByte, BLT, block/speculative generation, grammar-constrained decoding, 자소 표현 연구를 서로 다른 축으로 찾은 것은 좋은 literature map이다.
- output head masking이 병목이 아니며 multi-symbol generation 없이는 decode latency를 크게 줄이기 어렵다고 명시한 점은 정확하다.
- baseline과 ablation이 필요하다고 말한 것도 맞다.

과장 또는 누락:

- SpaceByte는 단순히 `SPACE` 뒤에서만 큰 compute를 실행하는 방식이 아니다. 공식 `spacelike` predicate는 punctuation·ASCII 범주와 UTF-8 lead byte도 포함한다. Korean NFC에서는 문자 cadence 자체가 생긴다.
- “자연어 writing system을 grammar로 다룬 사례가 없다”는 표현은 너무 강하다. linguistic/word boundary pooling과 morphology-aware encoding의 직접 선행이 존재한다.
- 여섯 조건을 모두 만족하는 논문이 없다는 것은 method novelty를 증명하지 않는다. 좁은 conjunction은 언제든 만들 수 있다.
- Fast BLT가 statistical이고 제안은 rule이라는 차이는 곧 우월성이 아니다. learned proposer는 legality뿐 아니라 문맥까지 모델링한다.
- SpaceByte + BLT + grammar + morphology FST + adaptive depth + block generation의 결합은 한 논문의 기여가 아니라 여러 독립 연구 프로그램이다.
- `BLT 1B부터`는 mechanism pilot의 출발점으로 지나치게 크며, 실패 원인 식별을 어렵게 한다.
- “한국어가 결정적 구조를 가지므로 상당 부분을 학습할 필요 없다”는 문장은 구조의 합법성과 실제 다음 내용의 불확실성을 다시 혼동한다.

### Turn 7 — 논문과 Hugging Face 공개 순서

논문·코드·checkpoint를 함께 공개하고, 모델 크기보다 controlled ablation과 실제 비용을 우선하라는 조언은 타당하다. 다만 `100M~1B`는 한 단계가 아니다. 적은 규모에서 mechanism을 통과시킨 뒤 scale을 올려야 하며, latency 수치는 구현과 hardware가 고정되기 전에 예시로도 목표값처럼 제시하지 않는 편이 낫다.

### Turn 8 — 프로젝트 이름

`JamoFlow`는 저장소 codename으로 사용할 수 있다. 그러나 현재 Phase 3는 NFC raw UTF-8을 유지하고 Jamo representation을 구현하지 않는다. 논문 제목·abstract에서 Jamo-aware method처럼 읽히게 사용하면 실제 intervention과 불일치한다.

## 3. `01-verification-report.md`의 과정과 결론 감사

### 3.1 잘한 점

`01`은 원 대화가 피한 정량 질문을 제기했다.

- output head와 trunk FLOPs를 분리했다.
- raw-jamo step/KV 증가를 baseline fertility와 비교했다.
- code-mixing, compatibility jamo, normalization, long tail을 실제 실패 조건으로 올렸다.
- 1B부터 시작하지 말고 작은 통제 실험을 하도록 순서를 바꿨다.
- conjunction novelty와 reasoning 향상 주장을 공격했다.
- 결과가 음수여도 측정 질문을 남기려 했다.

이 비판의 방향은 대체로 옳고, 이후 프로젝트를 실험 가능한 크기로 줄이는 데 결정적이었다.

### 3.2 검토 절차의 한계

`7팀`, `686k 토큰`, `186 도구 호출`은 노력의 양이지 검증 품질의 척도가 아니다. 저장소 안에서 다음을 재현할 수 없다.

- 검색 query와 검색 종료 기준
- 문헌 inclusion/exclusion 기준
- claim별 원문 위치와 반대 증거
- agent 간 중복·의존성
- 의견 충돌의 adjudication 절차
- 비용 추정에 사용한 계산 sheet와 민감도 범위

후속 [03 인용 검증](./03-citation-verification.md)이 핵심 인용을 1차 출처로 다시 확인해 사실 층은 강화했다. 그래도 “논문이 존재하고 문장이 맞다”와 “JamoFlow에 대한 판정이 따라온다”는 구분이 필요하다.

### 3.3 과도했던 실질 판정

상단 errata가 이미 바로잡은 항목은 다음과 같다.

- FSM과 block generation은 논리적 자기모순이 아니다. naive per-jamo AR에만 충돌한다.
- Three-hot은 한 음절 factorization을 선점하지만 variable-length block decoding 전체를 선점하지 않는다.
- Scratchpad Patching은 multi-byte generation이 아니라 transient compute로 patch lag를 줄인다.
- 31%는 100M 별도 router와 단순 FLOP 합산에 의존한 상한이지 wall-clock 결과가 아니다.
- Unicode Hangul 조합식과 UTF-8 codepoint boundary 검출은 다른 문제다.
- cheap boundary가 좋은 information boundary라는 보장은 없다.
- rule/linguistic boundary 선행이 이미 존재한다.

추가로 본문의 다음 숫자는 방향성 경고로만 읽어야 한다.

- output-head `0.042%`: 특정 width/layer/vocab 가정의 산술이며 모든 architecture의 측정값이 아니다.
- 자소 KV `2.1~5배`: 비교 tokenizer와 corpus에 따라 달라진다.
- 한국어 자연 데이터 `40~60B` 상한: dedup·품질·라이선스 정의에 민감하다.
- Amdahl 예시: 비한글 slowdown을 실제 구현에서 측정한 결과가 아니다.
- GPU 비용: 계획 수립용 snapshot이지 연구 명제의 근거가 아니다.

핵심 방향은 견고하지만 exact number에 `FATAL`이라는 수사를 결합한 것은 과했다.

### 3.4 가장 큰 과정상 오류: survivor의 조기 승격

원안의 대부분을 기각한 뒤 남은 “entropy patcher를 공짜 경계로 대체”를 충분한 재검증 없이 유일한 기여로 확정했다. 이는 survivor bias다.

- detector가 싸다 ≠ boundary quality가 좋다
- codepoint boundary가 결정적이다 ≠ Korean-specific novelty다
- auxiliary router FLOPs가 있다 ≠ 실제 latency 병목이다
- every-syllable cadence가 가능하다 ≠ target bytes/patch를 만족한다

이 오류를 [02 재검토](./02-critical-research-direction-review.md)가 다시 잡았고, 현재 실험은 rule superiority가 아니라 rate-controlled boundary relocation을 검사한다.

### 3.5 문서 사용 원칙

`01`은 삭제할 문서가 아니라 연구가 어떻게 과잉 아이디어를 걸러냈는지 보여주는 역사적 기록이다. 다만 단독으로 읽으면 errata와 본문이 충돌한다. 현재 판정에는 이 문서와 [02](./02-critical-research-direction-review.md), [28](./28-novelty-and-identification-audit.md)을 함께 사용해야 한다.

## 4. `00-topic-selection.md` 결론 감사

### 4.1 옳게 고친 부분

- “확정 주제”를 후보 가설로 되돌렸다.
- `zero-cost`를 `parameter-free`로 낮췄다.
- rule-only, learned-only, hybrid를 열린 비교로 만들었다.
- matched patch rate, causality, code-mixing, wall-clock을 요구했다.
- representation, patching, output-generation 축을 분리했다.
- multi-Jamo/block decoding을 후속 연구로 분리했다.

이는 최초 대화와 `01`보다 훨씬 올바른 연구 결정이다.

### 4.2 아직 부족하거나 현재와 달라진 부분

1. **중심 가설과 현재 primary가 다르다.** `00`은 candidate-restricted learned scorer를 중심 hybrid로 두지만 현재 Phase 3 primary는 F/C/W의 same-rate 효과다. EC는 codepoint-restricted entropy baseline이지 whitespace candidate scorer와 동일하지 않다.
2. **외부 baseline은 아직 계획이다.** H-Net, Scratchpad, standard BPE, Hierarchical BPE-style control을 결과표에 실제로 넣기 전에는 넓은 Pareto 우월성을 말할 수 없다.
3. **데이터 버전이 바뀌었다.** 현재 Phase 3는 HPLT v2가 아니라 pinned HPLT 3.0 shard를 사용한다.
4. **‘한국어 규칙’의 범위가 더 좁아졌다.** 현 W가 사용하는 것은 형태소 FSM이나 자소 FSM이 아니라 이미 관측된 Unicode whitespace다.
5. **architecture dependence가 확인됐다.** HF BLT의 dummy/encoder/decoder shift 때문에 W는 “공백 직후 즉시 global compute”가 아니다. [30](./30-hf-blt-alignment-audit.md)
6. **19.6M은 mechanism scale이다.** 이 규모에서의 효과를 실용적 한국어 LLM 효율로 일반화할 수 없다.

따라서 `00`은 올바른 research program 문서지만, 현재 논문의 exact claim을 정의하는 최신 protocol로 쓰면 안 된다.

## 5. 현재 Phase 3 방향에 대한 독립 판정

현재의 F/C/W primary는 원 아이디어를 많이 포기했지만 과학적으로 더 강하다.

- raw bytes, model graph, initialization, training order, byte budget을 고정한다.
- F/C/W의 data patch count를 window당 86으로 고정한다.
- C로 generic UTF-8 safety를 분리하고 W로 observed whitespace의 추가 효과를 본다.
- 결과를 보기 전에 Gate I/J/K와 OOD guard를 고정했다.
- delayed grid와 rate-matched hash placebo를 conditional mechanism control로 둔다.
- SpaceByte-compatible cadence와 learned entropy router는 same-rate primary와 total-cost 비교를 분리한다.
- NFD, generation validity, private Markdown ecology를 primary 평균에 섞지 않는다.

이는 “왜 좋아졌는지”를 묻는 causal measurement study로서 가치가 있다. 다만 다음 빈틈은 남는다.

1. **W−C는 whitespace 의미만 순수하게 식별하지 않는다.** phase, patch-length variance, early-event 분포도 함께 바뀐다. D/P control이 필요하지만 그것도 모든 대안 설명을 제거하지는 않는다.
2. **HF BLT one-byte shift에 종속된다.** scratchpad나 다른 local/global association에서 효과가 유지된다는 보장이 없다.
3. **teacher-forced throughput은 AR latency가 아니다.** decode step, cache update, dynamic batching을 구현하지 않으면 generation speed claim은 금지해야 한다.
4. **한 언어·한 shard·한 모델 규모다.** Korean-specificity와 scale persistence가 아직 식별되지 않는다.
5. **tokenized competitor가 없다.** 같은 raw-byte graph 안의 좋은 intervention과 실제 Korean LM 시스템 우월성은 다른 주장이다.
6. **semantic quality가 없다.** BPB 개선이 한국어 태스크나 생성 유용성으로 이어지는지 모른다.
7. **fixed patch count가 total local cost까지 완전히 같게 하지는 않는다.** patch length distribution, padding, kernel shape, memory access를 별도 측정해야 한다.

따라서 Phase 3는 진행할 가치가 있지만 **논문 결론이 아니라 scale-up 여부를 정하는 confirmatory mechanism gate**다.

## 6. 권장하는 올바른 연구 방향

### 6.1 현재 논문 축: Korean boundary placement and total cost

연구 질문을 다음 네 개로 고정한다.

1. **Same-rate effect:** W가 C보다 seed와 sequence를 넘어 재현 가능하게 낮은 BPB를 보이는가?
2. **Mechanism:** 그 차이가 단순 phase·irregularity가 아니라 observed whitespace association과 관련 있는가?
3. **Total cost:** authentic SpaceByte cadence와 learned E/EC의 router·padding·memory 비용까지 포함해 W가 Pareto frontier에 남는가?
4. **Scope:** 효과가 public Korean OOD, mixed text, normalization stress와 더 큰 모델에서 어디까지 유지되는가?

현재 paper title과 abstract는 결과 중립적으로 유지한다. `Jamo`, `morphology`, `adaptive compute`, `faster generation`, `first`를 기여 단어로 쓰지 않는다.

### 6.2 긍정 결과일 때 필요한 출판 단계

Gate I/J와 mechanism control이 통과해도 다음이 없으면 top-tier 효율 주장을 하지 않는다.

1. 50–100M main model, 최소 256M 공개 Korean training bytes
2. 최소 5-seed confirmation 또는 pilot variance에 근거한 power justification
3. standard Korean byte-BPE Transformer
4. 구현 가능한 morphology/BPE 또는 Hierarchical BPE-style cheap-boundary control
5. authentic SpaceByte와 learned router의 rate·total-cost frontier
6. CUDA에서 teacher-forced throughput, memory, TTFT와 구현 시 verified incremental decode
7. 한국어 natural downstream 또는 controlled semantic generation 평가 최소 하나
8. Chinese/Japanese/English 중 적어도 두 control로 Korean-specificity 분리

이 단계에서 반복된다면 method paper가 된다. 반복되지 않으면 compact-model optimization effect를 발견한 것으로 결론을 낮춘다.

### 6.3 음수 결과일 때 필요한 출판 단계

Gate I가 실패하면 W를 다른 heuristic으로 구제하지 않는다. 대신 질문을 다음처럼 바꾼다.

> 왜 linguistically plausible boundary가 작은 Korean byte model에서는 이득을 보이지만 scale·domain에서 사라지는가?

단일 Korean negative result는 약하다. 다음을 묶어 재사용 가능한 benchmark·measurement paper로 만든다.

- 여러 script의 same-rate boundary matrix
- boundary semantics, geometry, rate, variance의 분해
- oracle surprisal와 causal heuristic의 gap
- selector/router/padding 비용 표준 accounting
- public evaluation harness와 failure taxonomy

### 6.4 별도 후속 축: orthography-constrained block decoding

원 대화의 multi-Jamo 직관은 현재 patching 논문에 다시 넣지 않는다. 후속 연구는 다음으로 제한한다.

- learned multi-byte proposer
- UTF-8/Hangul DFA support 또는 verifier
- unconstrained proposer와 동일 model/acceptance budget 비교
- structural validity, rejection, accepted bytes/full forward, 실제 latency 측정

규칙이 semantic content를 생성한다고 주장하지 않는다. 1차 기대 기여는 speed보다 rare/noisy text의 structural robustness다. 이 축은 patch placement와 독립적으로 preregister하고 학습해야 한다.

## 7. 논문 claim gate

| 주장 | 허용 조건 |
|---|---|
| W가 C보다 좋은 same-rate boundary schedule이다 | Gate J의 사전 고정 효과·반복·interval 통과 |
| whitespace association이 원인이다 | D/P mechanism contrast까지 통과; 그 graph와 데이터 범위로 한정 |
| learned routing보다 효율적이다 | E/EC 품질 비열등 + router-inclusive analytical/direct cost 통과 |
| Korean-specific effect다 | 다른 spaced language와 CJK control에서 interaction 확인 |
| 실용적 inference speedup이다 | incremental AR CUDA benchmark와 품질 동등성 확인 |
| Jamo-aware architecture다 | 실제 Jamo representation 또는 L/V/T intervention을 주요 ablation으로 구현 |
| morphology-aware method다 | 형태 분석/경계 intervention과 non-morph control을 직접 비교 |
| top-tier 효율 기여다 | publication scale, tokenized baseline, semantic quality, hardware evidence 충족 |

## 8. 문서 권위 순서와 최종 권고

연구 의사결정에는 다음 순서를 사용한다.

1. 현재 exact experiment: [22 Phase 3 protocol](./22-phase3-confirmatory-protocol.md)과 결과 전 addenda
2. 현재 novelty/claim boundary: [28 신규성·식별 감사](./28-novelty-and-identification-audit.md)
3. 구현 해석: [30 HF BLT 정렬 감사](./30-hf-blt-alignment-audit.md)
4. 전체 방향 감사: 본 문서와 [02](./02-critical-research-direction-review.md)
5. 역사적 결정 기록: [00](./00-topic-selection.md), [01](./01-verification-report.md), 최초 공유 대화

최종 권고는 명확하다.

> **현재 Phase 3를 계획대로 끝내되, 이를 Jamo-based adaptive computation의 증명으로 해석하지 않는다. 먼저 Korean raw-byte BLT에서 rate-controlled boundary placement 효과와 그 총비용을 식별한다. 긍정 결과가 반복·scale·systems·semantic gate를 모두 통과할 때만 경량 method paper로 승격하고, multi-Jamo/FSM은 별도 block-decoding 연구로 남긴다.**

이 방향은 최초 아이디어를 가장 적게 과장하면서도, 성공과 실패 어느 쪽에서도 새 지식을 남길 수 있는 경로다.
