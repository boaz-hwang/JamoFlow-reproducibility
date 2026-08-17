# JamoFlow 아이디어 검증 보고서 (Claude 검토 결과)

> 작성일: 2026-08-08
> 검토 대상: ChatGPT 대화 "자소 분리와 효율성" 1~6번 턴 (딥리서치 응답 포함)
> 검토 방식: 독립 에이전트 7팀 병렬 — 사실검증 3 (바이트 아키텍처 / 한국어 자소 / 최신 모델 스펙) · 선행연구 사냥 1 · 기술비판 1 · 실현가능성 1 · 출판심사 1
> 웹 검증: arXiv · ACL Anthology · Hugging Face · 공식 GitHub 1차 출처 (총 686k 토큰, 186 도구 호출)
> HTML 원본: https://claude.ai/code/artifact/ffdc3176-fa40-4893-af4c-29d15519f4dc
> 현재 해석: 본문은 역사적 기록이며 상단 errata, [02 재검토](./02-critical-research-direction-review.md), [31 전 과정 감사](./31-source-conversation-and-decision-audit.md)가 판정에 우선한다.

---

## ⚠️ 정정 (Errata, 2026-08-08)

본 보고서의 일부 판정이 [02 재검토](./02-critical-research-direction-review.md)에서 반박되었고, 02의 인용 9건·세부주장 10건은 [03 인용 검증](./03-citation-verification.md)에서 전수 확인됐다. 아래 정정이 본문에 우선한다.

1. **"FSM + multi-jamo = A∧¬A 자기모순" (§3 FATAL 1) → 과잉 판정.** 자소별 AR 디코딩 구현에 한정된 충돌이며 논리적 모순이 아니다. FSM은 block proposal의 joint support 제한·speculative draft의 DFA 검증기로 결합 가능. multi-jamo 축은 전면 폐기가 아니라 **"Orthography-Constrained Block Decoding" 별도 2차 연구로 보존** (02 §7).
2. **"유일한 free lunch (patcher 대체)" → "검증할 가치가 있는 cost-accounting 가설"로 격하.** 경계 탐지가 싸다는 것(참)과 그 경계가 좋은 patch boundary라는 것(미검증)을 혼동했다. every-syllable patch는 ~3 bytes로 BLT의 4.5~8 bytes보다 짧아질 수 있다.
3. **"31% 절감" → 가정 의존 이론 상한.** 100M patcher·byte별 독립 forward·FLOPs=latency 가정에 의존. BLT 원문 자체가 1M~100M entropy 모델 실험, **50M 이상 수확체감**, 짧은 receptive field의 **lookup table 구현 가능성**을 명시(03에서 원문 확인). Scratchpad Patching(arXiv:2605.09630)은 patcher를 encoder 위 **2-layer aux head로 통합**했고, SP 적용 시 SpaceByte(NLU 56.2)가 H-Net(55.5)·entropy(55.3)를 역전 — "별도 100M 모델 제거"는 낡은 구현 공격이 될 수 있다.
4. **"선행연구는 전부 학습된 통계 신호만 쓴다" (§2 novelty 서술) → 오류.** AU-Net은 regex 단어 경계("splits on spaces using different regular expressions", 03 확인), SpaceByte 자체가 규칙 patcher다. **누락된 직계 선행 3건 추가**: Learn Your Tokens(EMNLP 2023 Findings — word boundary pooling + 단어 내부 병렬 복원), **Dynamic Token Pooling(ACL 2023 — entropy/tokenizer/언어학적 경계를 이미 직접 비교)**, FLEXITOKENS(Findings of ACL 2026 — learnable boundary predictor). 남는 공백은 "규칙 사용 여부"가 아니라 **"boundary detector 비용까지 포함한 총비용 Pareto 통제 실험의 부재"**다.
5. **`S = 0xAC00 + (L×21+V)×28 + T`는 codepoint 조합식이지 UTF-8 경계 탐지 공식이 아니다.** codepoint 경계는 leading/continuation bit로 generic하게 공짜이며, SpaceByte가 이미 leading byte를 spacelike로 처리해 CJK에 문자당 1회 global cadence를 제공한다(단 SpaceByte 자체 예비실험에서 중국어 성능은 subword 대비 열위 — 규칙 cadence의 CJK 유효성은 여전히 열린 질문).
6. **§4 MVP의 `Δ = f − f₀`·`|A(s)|=1` 게이트는 폐기된 compute-skip 가설용.** 원안 폐기 근거를 기록하는 appendix 분석으로 이동하며, 새 Go/No-Go 게이트는 [00(개정)](./00-topic-selection.md) §5의 matched patch-rate audit 기준을 따른다.
7. **Scratchpad Patching 요약 정정**: "한 forward에서 가변 길이 byte 생성"이 아니라 patch 내부 transient scratchpad(persistent KV에 미포함)로 patch lag를 줄이는 기법. 핵심 시사점은 "boundary 위치보다 compute allocation이 더 중요할 수 있다"(원문 §4.3).

이하 본문은 원본 그대로 보존한다 (역사적 기록 + 위 정정 이외의 판정은 유효).

---

## 최종 결론 한눈에

| 질문 | 판정 |
|---|---|
| ChatGPT 인용의 사실성 | **신뢰 가능** — 17건 검증 전수 실존, 날조 0건. 단 2건 뉘앙스 부정확 (§1) |
| 최초 시도인가 | **좁은 의미로만** — 6조건 AND를 만족하는 단일 논문은 없음. 그러나 조건 (2)와 (5)는 상호 모순이라 "없는" 이유가 공백이 아니라 자기모순. 실질 메커니즘 대부분은 선행 존재 |
| 기존 대비 발전인가 | **재프레이밍하면 Yes** — "규칙으로 neural compute 대체" 원안은 산술적으로 무너짐. "학습된 경계 vs 손수 만든 문자체계 문법"이라는 과학적 질문으로 바꾸면 진짜 기여가 됨 |
| 기술적 검토 가능성 | **가능, $0으로** — 핵심 가설의 생사는 GPU 없이 주말 이틀의 정보이론 측정으로 판정 가능 (§4 MVP) |
| 논문 가치 | **현 프레이밍 REJECT → 측정 논문으로 축소 시 CONDITIONAL ACCEPT** — 현실 타깃: ARR 2026-10-12 → NAACL/COLING 2027 Findings·워크숍 |
| 살아남는 핵심 기여 | **단 하나** — 한글은 Unicode에서 음절 경계가 공짜로 결정적이므로, BLT류 모델의 유료 entropy patcher(1B급 decode compute의 약 31%)를 0-cost 규칙 patcher로 대체할 수 있다는 것. 진짜이고, 아무도 측정한 적 없음 |

---

## §1 사실검증 — ChatGPT 딥리서치는 얼마나 정확했나

인용 17건을 1차 출처(arXiv·ACL Anthology·공식 GitHub·HF)에서 전수 재검증했다.

**결론: 날조 0건.** 지식 컷오프 이후 발표라 존재 자체가 의심스러웠던 4건(Fast BLT, SCRIPT, Kimi K3, K-EXAONE 2.0)까지 전부 실존이 확인됐다. LLM 리서치 대화 기준으로 이례적으로 높은 정확도다.

| 인용 | 판정 | 확인 내용 |
|---|---|---|
| SpaceByte (2024) | ✅ 확인 | arXiv:2404.14408, Kevin Slagle (Rice), NeurIPS 2024. 공백 경계에서만 큰 Transformer 블록 실행. 고정 compute budget에서 byte-level baseline 능가, tokenized Transformer와 유사 성능 — 서술 정확 |
| BLT (Meta) | ✅ 확인 | arXiv:2412.09871 "Patches Scale Better Than Tokens". entropy 기반 dynamic patching, 8B/4T bytes FLOP-controlled scaling. facebook/blt-1b·blt-7b 실존, transformers v4.57.0(2025-10-03)에 공식 포함 |
| Fast BLT (2026) | ✅ 확인 | arXiv:2605.08044, 2026-05-08, Julie Kallini(Stanford) + Meta(Pagnoni·Zettlemoyer·Iyer 등), ICML 2026 poster. BLT-Diffusion / BLT Self-Speculation / Diffusion+Verification 3방식, "memory-bandwidth cost 50%+ 감소" abstract 원문 일치 |
| Mixture-of-Depths (2024) | ✅ 확인 | arXiv:2404.02258, Google DeepMind. top-k routing으로 token별 layer skip, 샘플링 step 최대 50%+ 가속 일치 |
| Automata-based Constraints (2024) | ✅ 확인 | arXiv:2407.08103, Koo·Liu·He (DeepMind), COLM 2024. constraint compilation ~7,000배 가속 일치 |
| Flexible & Efficient GCD (ICML 2025) | ✅ 확인 | arXiv:2502.05111, Park·Zhou·D'Antoni, PMLR v267. offline preprocessing 17.71배 가속 일치 |
| Korean 3-hot (EACL 2023) | ✅ 확인 | Cognetta·Moon·Wolf-Sonkin·Okazaki, 2023.eacl-main.172. 3-hot factorized jamo embedding으로 embedding 파라미터 99.6% 감소(11.4M→36K), 번역 품질 무손실, 음절 수준 sequence length 유지 — 전부 일치 |
| KOMBO (ACL 2024 Findings) | ✅ 확인 | Kim·Park·Kim·Lee, 2024.findings-acl.302. 훈민정음 결합 원리를 표현에 반영, 5개 NLU task 평균 +2.11% 일치 |
| Jamo-level BPE (2025) | ✅ 확인 | Lee·Cognetta·Moon·Okazaki, LoResMT 2025 워크숍(2025.loresmt-1.8). 자소 최대 68개 위 BPE, low-resource에서 syllable·byte baseline 일관 우위 |
| SCRIPT (2026) | ⚠️ 부분 | 실존 (arXiv:2604.12377, 2026-04-14). 단 정확한 게재처는 **ACL 2026 Findings** — ChatGPT는 메인처럼 읽히게 서술. 저자: Kim·Park·Atalay·Lee (KOMBO 후속 라인). plug-and-play 주입, 문법 규칙성 포착 분석은 실존 |
| Thunder-Tok (2025) | ⚠️ 부분 | 실존 (arXiv:2506.15138). 정식 제목은 **"Less Is More: Reducing Token Counts Without Compromising Performance"** (Thunder-Tok은 제안 토크나이저명). 한국어 fertility 감소 ~9%, 추론시간 감소 ~8% (≈10% 아님). "속도 개선"의 실체는 토큰당 속도가 아니라 생성 토큰 수 감소. peer-review 게재 미확인 preprint |
| Klex (2004) | ✅ 확인 | LDC2004L01, Na-Rae Han, XFST 기반 한국어 lexical transducer |
| Constrained FS Morphotactics (2005) | ✅ 확인 | Ju·Park·Lee·Lee, PACLIC 19 (Y05-1025) |
| Kimi K3 | ✅ 확인 | MoonshotAI 공식 GitHub: 2.8T total / ~104B activated / 896 experts (16 active, Stable LatentMoE) / 160K vocab / 93 layers = 69 KDA + 24 Gated MLA. 2026-07-16 공개, arXiv:2607.24653. K2(1T/32B/384 experts)와 혼동 아님 |
| K-EXAONE | ✅ 확인 | LGAI-EXAONE/K-EXAONE-236B-A23B: 236B/23B active, 128 experts, 256K ctx, 2025-12 공개, arXiv:2601.01739 |
| K-EXAONE 2.0 | ✅ 확인 | K-EXAONE-2.0-750B-A37B: 750B/37B active, 256 experts 중 8 선택, Apache 2.0, 2026-07-31 공개. 정부 소버린 AI 2단계 산출물. EXAONE 4.0(32B dense)과 별개 프로젝트 |
| HF jamo 체크포인트 | ✅ 확인 (단서) | devngho/llama-ablation-large-korean-corpus-jamo(2B) 등 실존하나 전부 0.2B~2B급 개인 ablation — "대규모 자소 LLM은 없다"는 서술과 상충하지 않음 |

---

## §2 신규성 — "연구 공백" 주장은 어디까지 유효한가

**판정: 부분 붕괴.** 좁은 의미의 6조건 AND는 아직 비어 있으나, 실질 공백은 ChatGPT 서술보다 훨씬 작고 빠르게 닫히는 중이다.

### ChatGPT 검색이 놓친 결정적 선행연구 3건

| 논문 | 시점 | 겹치는 부분 |
|---|---|---|
| **H-Net** — Dynamic Chunking (Hwang·Wang·Gu) | 2025-07 (arXiv:2507.07955) | 학습된 동적 청킹으로 byte 모델이 BPE Transformer를 동일 FLOPs에서 능가. **hand-crafted 공백 경계 변형(H-Net(space))보다 learned 경계가 낫다는 것을 실험으로 이미 보임** — 이 아이디어의 전제와 정면 충돌. 단 boundary는 100% learned이고 한국어 언급 전무 |
| **AU-Net** (Meta) | 2025-06 (arXiv:2506.14761) | byte→word→word-pair 계층 pooling으로 compute 절감 — 언어 비의존 계층 압축 |
| **Scratchpad Patching** (Google) | 2026-05 (arXiv:2605.09630) | 가변 compute 배분 + 한 forward에서 가변길이 byte 생성 — 조건 4+5 조합을 정확히 겨냥 |

추가 발견: Multi-token Prediction(Gloeckle et al., ICML 2024, arXiv:2404.19737 — 조건 5의 가장 잘 알려진 선행), Constrained Decoding with Speculative Lookaheads(NAACL 2025, arXiv:2412.10418 — 문법 제약+speculative 결합 2.2~12배 가속), MYTE(ACL 2024, arXiv:2403.10691 — 형태소 경계를 byte encoding에 반영), Grammar-Faithful Speculative Decoding(2026-05, arXiv:2605.07698 — 이 라인의 최신 논문조차 자연어 문자체계는 다루지 않음을 명시), 한국어 ASR의 jamo 최소단위 사용(Interspeech 2020).

→ **"통계적 adaptive compute + 다중 심볼 생성"은 2026년 5~7월에 동시다발로 수렴 중인 매우 붐비는 클러스터**다.

### 그래도 남는 진짜 novelty

이 모든 선행연구는 하나같이 **학습된 통계 신호**(entropy·gating·router)를 쓴다. 사용자 아이디어의 고유 지점은:

1. **완전히 결정론적인 문자체계 문법을 신호로 써서 해당 구간의 neural compute를 "더 적게"가 아니라 "0으로"** 만들겠다는 것
2. 문법 제약 기법군이 아직 formal grammar(JSON·코드)에만 적용되고 **자연어 표기 체계(한글)에는 적용된 적 없다**는 것
3. 이를 한국어 decoder-only LLM의 pretraining 단계에서 FLOPs/latency ablation까지 측정한 사례가 없다는 것

단, 이는 "새 패러다임"이 아니라 **기존 아키텍처의 라우팅 신호를 교체한 좁은 기여**이며, 인접 분야의 속도(수개월 단위 후속 논문)를 고려하면 이 틈도 오래 유지되지 않을 가능성이 높다. 부가 주장이었던 "reasoning 자체도 좋아질 것"은 선행연구로도 이번 검색으로도 전혀 뒷받침되지 않음.

---

## §3 기술 비판 — 가차없는 검토

기술 코어의 약 90%가 무너지고 하나가 살아남는다.

### FATAL — 6조건 스펙은 자기모순이다

조건 (5) "1 forward로 여러 자소 생성"이 성립하면 조건 (2) "FSM이 자소별 디코딩을 제약"할 스텝 자체가 사라진다. 반대로 자소를 하나씩 뽑으면 스텝이 음절 대비 약 2.4배 늘어 조건 (4)의 목적(지연 감소)과 정반대. **"이 6조건을 만족하는 연구가 없다"는 참이지만, 이유는 공백이 아니라 정의가 A∧¬A이기 때문.**

### FATAL — "음절 결정 → rule decoder 전개"는 EACL 2023이 이미 했다

그 구조는 (i) 시퀀스 길이 = 음절 수, (ii) hidden state 1개로 음절 결정, (iii) 자소 방출은 파라미터 없는 결정적 함수 — 정확히 **음절 tokenizer + 결정적 detokenization**의 정의다. Cognetta et al.(EACL 2023)은 3-hot **decoder**까지 이미 구현했고 코드도 공개돼 있다(`mcognetta/ThreeHotKoreanModeling`). 유일한 차이(음절 내부 의존성 보존: 11,172-way head + legality mask)는 오히려 FLOPs를 0.014%→2.3%로 **늘리는** 방향이다.

### FATAL — 후보 자소 마스킹의 절감 이론 상한 = 0.04%

SwiGLU+GQA 기준 forward FLOPs = 2×사용 파라미터로 계산:

| 구성 | output head 비중 |
|---|---|
| 1B급 jamo LM (d=2048, L=16, V≈200) | **0.042%** |
| 7B급 jamo LM (V≈200) | **0.013%** |
| 1B급 syllable LM (V=11,172) | 2.297% |
| 1B급 3-hot head (19+21+28=68 출력) | 0.014% |

게다가 실무 constrained decoding은 logit을 전부 계산한 뒤 −∞를 더하므로 **실측 절감은 정확히 0**. Llama-3.2-1B에서 head가 21.3%인 것은 128K vocab 때문인데, 그건 마스킹이 아니라 vocab 축소 자체가 없애는 것이고 그 공로는 2023년 논문 소유다.

### FATAL — 한글 FSM에는 결정적(branching factor 1) transition이 하나도 없다

초성 다음에 올 수 있는 심볼 21개, 중성 다음 약 48개(종성 27 + 새 초성 19 + 공백·부호). hard rule은 후보 집합(support)만 줄일 뿐 조건부 엔트로피를 줄이지 않으며, **Transformer forward의 FLOPs는 출력 분포의 엔트로피와 무관**하다.

더 근본적으로 — 한글은 "규칙적이라 예측 가능한" 문자가 아니라, 자소마다 거의 독립적인 정보를 싣는 **효율적인** 문자 체계다(음절 13.4bits가 UTF-8 3byte에 2/6/6으로 거의 균등 분산). 착취할 철자 잉여는 라틴 문자(qu, -tion)보다 오히려 적다. **규칙성(regularity)과 예측가능성(predictability)의 혼동이 이 아이디어의 근원적 오류다.**

### MAJOR — KV cache 절감 주장은 거짓, 오히려 2.1~5배 증가

KV cache는 vocab과 무관하고 스텝 수에만 비례한다(1B급 GQA 기준 32,768 B/token). 실제 baseline인 한국어 특화 BPE는 이미 어절당 1.35~1.50 토큰(Thunder-Tok Table 4: BPE 1.50, Thunder-Tok 1.37, SuperBPE 1.35 — 토큰당 2.1~2.4음절)이므로, 음절 단위 디코딩은 2.1~2.4배, 자소 단위는 5~6배 느리고 KV도 그만큼 커진다. "30토큰→12 latent"급 압축은 1스텝에 ~5.8음절(약 14자소)을 방출해야 하는데, 조합 규칙이 줄 수 있는 최대치는 2.4자소→1음절이 전부다.

### MAJOR — multi-jamo 생성은 speculative decoding의 열등한 특수사례

rule decoder = "acceptance rate 1로 가정된 draft"인데, 엔트로피 0 구간이 없으므로 그 가정이 깨진다. 검증을 붙이면 그냥 speculative decoding이고, Fast BLT의 BLT-S(학습된 draft가 patch 경계 너머까지 draft 후 1회 검증)가 이미 규칙 draft를 지배한다 — 규칙이 아는 것(조합 legality)은 학습 draft도 알고, 학습 draft는 문맥까지 안다. prompt-lookup/n-gram drafting은 이미 vLLM·transformers에 들어있는 0-cost 규칙형 drafter다.

### MAJOR — Amdahl 법칙: 비한글 15~20%면 이득 전체가 증발

실사용 텍스트의 영어·숫자·코드·URL 구간은 byte/자소 fallback에서 BPE 대비 ~4배 느리다. 총 시간 = f/S_k + (1−f)·D (f=한글 비중, S_k=한글 구간 가속, D=비한글 감속):

- f=0.9, S_k=2, D=4 → 0.85 (겨우 1.18배)
- f=0.8, S_k=2, D=4 → **1.20 (0.83배 — 17% 더 느려짐)**
- f=0.7, S_k=2, D=4 → 1.55 (0.65배)

FSM 붕괴는 더 근본적: ㅋㅋ·ㅠㅠ·ㅎㅎ는 Hangul Compatibility Jamo(U+3131~)로 초·중·종 FSM을 원천 위반하며 한국어 웹 코퍼스에서 극도로 빈번. 조합형/호환형 이중 체계, 겹모음·겹받침 분해 여부(NFD는 안 쪼갬 — spec 미정의), 옛한글, Tier 라우터가 매 스텝 실행되는 CPU branch라는 오버헤드까지.

### MAJOR — "BLT는 통계, 나는 규칙" 차이의 marginal gain 분해

- (a) **보장성**: malformed 한글 출력 0% — 실제 이득이나 latency가 아닌 correctness, 그리고 충분히 학습된 모델은 어차피 거의 안 틀림
- (b) **patcher가 못 찾는 저엔트로피 구간을 규칙이 발견**: 거짓일 가능성 높음 — 한글 음절 내부 byte는 애초에 저엔트로피가 아님. patcher가 "발견 못 하는" 게 아니라 발견할 게 없음
- (c) **worst-case 안정성**: O(1) lookup — 실제 이득이나 마케팅할 크기 아님
- 직접 반례: **H-Net은 공백조차 없는 중국어에서 learned chunking이 heuristic과 tokenized Transformer를 둘 다 이겼다.** "한국어에서는 손 규칙이 이긴다"를 주장하려면 이 결과를 정면으로 뒤집어야 하는데 근거가 없음

### MAJOR — 한국어 전용 pretraining은 데이터 천장에 걸린다

공개 코퍼스 간 중복 제거 후 고유 고품질 자연 한국어는 약 40~60B 토큰이 상한. KORMo(10.8B 모델, arXiv:2510.09426)는 1.09T origin에서 필터 후 48B(생존률 4.4%)만 건졌고 한국어 데이터의 68.74%를 합성으로 채웠다. 단, **1B 이하 controlled ablation(20B 토큰)에는 문제가 안 된다** — "쓸 만한 한국어 LLM"과 "논문용 통제 실험"을 섞어 말하면 안 된다.

### MINOR — "실사용 음절이 적다" 활용은 ROI 최악

11,172→2,500자로 잘라도 1B급에서 절감 1.8%p, 대가는 인명·신조어·음차 커버리지(하필 neural compute가 정말 필요한 영역). 3-hot factorized head를 쓰면 애초에 자를 것도 없다(0.014%).

### MINOR — 현재 프레이밍은 과대 포장

"Rule-Guided Adaptive Computation"이라는 프레이밍은 실제 기여 대비 과대하다. 리뷰어가 Section 1에서 "1 forward → 여러 자소"를 읽는 순간 EACL 2023을 떠올리고 desk-reject 위험 구간에 들어간다. 실제 살아있는 기여를 정직하게 workshop~short paper 크기로 파는 게 통과 확률이 높다.

### ✅ 그럼에도 살아남는 것 — 진짜 기여 1개 + 보조 가설 1개 【⚠️ Errata 2·3 적용: "확정 기여"가 아니라 "cost-accounting 가설"로 읽을 것】

**① BLT의 유료 patcher를 공짜로 만들 수 있다.**
BLT의 entropy patcher는 별도 100M 파라미터 byte LM이고 autoregressive 생성 시 byte마다 온라인 forward가 필요해, patch당 0.90 GFLOPs — BLT-1B global(patch당 2.0 GFLOPs) 대비 **decode compute의 약 31%**를 차지한다(8B급 5.3%). BLT 논문은 이 비용을 FLOP 수식(Sec 4.5)에서 제외한 채 "50% fewer FLOPs"를 주장한다. 한글은 Unicode 산술(`S = 0xAC00 + (L×21+V)×28 + T`)만으로 음절 경계가 결정적이므로 이 patcher를 0-cost 규칙으로 대체 가능 — **순수 이득(free lunch)이 나오는 유일한 지점.**

**② 한글 UTF-8은 byte-entropy patching과 구조적으로 궁합이 나쁠 개연성.**
BLT의 영어 평균 patch 4.5byte는 한국어에서 겨우 1.5음절 — Llama-3 토큰(3.7~4.4byte)이나 한국어 BPE보다 나쁠 수 있다. **모델 학습 없이 며칠이면 측정 가능하고, 아무도 재지 않은 숫자다.**

---

## §4 실현 가능성 — 1인 사이드프로젝트 기준

### 비용 (2026-08 실측 GPU 시세: H100 on-demand ~$2/h 기준)

| 범위 | 비용 | 기간 | 산출물 |
|---|---|---|---|
| 정보이론 MVP | **$0** | 2주말 | go/no-go 판정 + arXiv 노트·워크숍 페이퍼 감 |
| 100M 파일럿 6 config | $40~65 | 2~4주 | 1차 실증 |
| 100M+350M 전체 ablation + BPB 평가 | $500~1,500 | 3~6개월 | ACL/EMNLP short 또는 강한 워크숍 |
| ChatGPT가 묘사한 그 논문 (1B+ over-train, 3시드, 실측) | **$66k~106k** | 6~12개월 풀타임 | **funded lab 프로젝트 — 1인 범위 밖** |

단일 1B Chinchilla 런은 $674~1,079 (BLT MFU 페널티 반영: 이상적 84 H100h → 현실 337 H100h). 진짜 제약은 돈보다 wall-clock — 1B ablation 하나가 8×H100 노드 10~12일 무중단.

### ChatGPT 조언 중 1인 기준 틀린 것 2가지

1. **"BLT 1B부터 시작"** → 틀림. 100M부터 시작 (1B은 config당 $674~1,079)
2. **"HF에 들어왔으니 맨바닥 불필요"** → 절반만. `facebookresearch/blt`는 CC BY-NC 4.0 + 9개월 정체 + open issues 51개(재현 실패 다수: #125 tokenization, #143 entropy dim 불일치 등), facebook/blt 가중치는 FAIR Noncommercial. **transformers v4.57.0의 clean-room 구현만 사용할 것**

### 데이터

- **HPLT v2 Korean cleaned**: 19.69B words / 89.27B chars / 201.89 GB, **CC0** — 척추로 사용
- FineWeb-2 `kor_Hang`: 60.9M rows, 보조
- **AI-Hub: 배제** (비상업 연구 한정 + 제3자 제공 금지 — 가중치 공개 시 회색지대)
- 모두의말뭉치: KOGL 1유형, 개별 신청 필요, ≤1B 실험엔 불필요
- 코퍼스 간 **70% 중복** (KORMo 실측) — 더해서 세면 안 됨
- 단위 함정: 89.27B 한글 문자 ≈ 180~220B 자소 심볼 ≈ 40~60B BPE 토큰. **매칭 축은 반드시 UTF-8 바이트**

### 핵심 리스크 (요약)

1. **[개념]** 한글 FSM은 branching factor 1 transition을 0개 생성 — "결정적 스킵"은 hard rule만으로 0번 발동
2. **[평가]** FLOPs 절감 ≠ latency 절감 — batch-1 decode는 memory-bandwidth-bound. Fast BLT조차 실측 아닌 "estimated" 보고
3. **[스케일]** 100M~1B에서 BPE vs byte 격차가 시드 노이즈에 묻힐 위험 (BLT 주장은 8B/4T에서 확립)
4. **[평가]** 토크나이저가 다르면 per-token PPL 무의미 — BPB 필수. 소규모에서 KMMLU류는 chance floor 근처라 "품질 동일" 증명 불가
5. **[일정]** 스코프 크립 확정적 — 형태론 FSM 하나만도 수개월. Kiwi는 decode 루프에 못 넣음(~3,667 analyses/s) → v1에서 형태론 제외 권장

### MVP — GPU 0시간, 이 두 숫자가 생사를 결정

로컬(M4 Pro)에서 HPLT 샤드 2~5GB로:

1. **`|A(s)|=1` 비율** (예측값 정확히 0% — 확인 시 "deterministic skip" 서사 즉시 재작성)
2. **Δ = f − f₀**: 자소 n-gram 조건부 엔트로피의 skip 상한 f에서, 한글 지식 없는 BLT entropy patcher가 이미 잡는 f₀를 뺀 값

**Kill criteria (사전 고정):**

| Δ | 판정 |
|---|---|
| < 5%p | **중단** — 한국어 특화 기여가 노이즈, BLT 재구현일 뿐 |
| 5~15%p | 100M 파일럿($40~65)까지만, 워크숍 사이즈 |
| > 15%p | 실제 천장 존재 — 350M 진행($500~800) |

부가 산출물: 음절 빈도 90/99/99.9% 커버리지 수치 — 공개된 수치가 없어(NOT_FOUND) 만들면 그 자체로 새 데이터.

---

## §5 논문 가치 — 리뷰어 시뮬레이션

**판정: 현 프레이밍 그대로 REJECT → 재프레이밍 + 범위 축소 시 CONDITIONAL ACCEPT**

### 리뷰어가 반드시 공격할 지점

1. **결정성 착시**: "FSM은 타입만 결정하지 정체를 결정하지 않는다. 당신 절감 메커니즘은 EACL 2023과 동일하고 나머지는 장식이다"
2. **자기반박 인용**: 마스킹 무용론은 대화록에 ChatGPT 스스로 적어놨다 — 논문이 자기 반박을 이미 인정한 형태
3. **H-Net 반례**: learned 경계가 hand-crafted 경계를 이미 이겼다(공백 없는 중국어에서도). 이 비교를 의도적으로 넣고 지는 경우까지 프레이밍해야 함
4. **선점**: 조건 5는 Fast BLT가, 조건 4의 실질은 Cognetta 2023이 가져감
5. **전제 붕괴**: "scratch pretraining 필수" 전제도 공격당함 — Ai2 Bolmo가 기존 모델의 저비용 byteification 대안을 이미 제시
6. **헤드룸 반박**: Thunder-Tok이 한국어에서 짜낸 게 ~9% — 30~50% 절감 기대의 근거 요구
7. **측정 공정성**: FLOPs·wall-clock(batch 1/8/64)·memory-bandwidth 3종 동시 보고 필수. PPL은 bits-per-character로 정규화
8. **강건성**: ㅋㅋ/ㅠㅠ·코드믹싱 스트레스 테스트와 fallback rate 요구
9. **필수 baseline 누락 시 즉시 reject**: SpaceByte식 공백 patching, Cognetta 3-hot, BPE+MTP — 이길 자신 없어도 표에 포함
10. **스케일 외삽**: IsoFLOP 매칭 + 최소 2사이즈 스케일 곡선 없으면 "작은 모델에서만 나는 이득" 반론에 무방비
11. **일반화**: "왜 한국어만?" — 다른 조합형 문자(데바나가리·태국어 등) 전이 1개 요구. 역으로 "한글은 조합 규칙이 완전 형식화된 유일한 주요 문자 = 이상적 통제 조건"으로 방어 가능
12. **conjunction novelty 감점**: "6개를 처음 합쳤다"는 AC가 engineering combination으로 분류 → main→Findings 강등

### 현실적 출판 경로

| 단계 | 내용 | 타깃 |
|---|---|---|
| **MPU-A** 측정 논문 (권장 1순위) | "How Much of Korean Is Free?" — FSM 결정성 상한, entropy patcher 대비 marginal gain, 강건성 곡선, 100~300M 최소 대조군. GPU 1~2장, 4~8주. **음수 결과도 논문이 됨** | **ARR 2026-10-12** → NAACL/COLING 2027 · HCLT 2026 병행 · TokShop 2027 |
| **MPU-B** 소규모 아키텍처 논문 | MPU-A 긍정 시만. 100~300M IsoFLOP 6종 + 스케일 곡선. 3~5개월 | ACL 2027 (ARR 2027-01) · COLM 2027 |
| **MPU-C** 6조건 풀 구현 1B+ | **1인 범위에서 제외 권장** — MPU-A/B 통과 후 공동연구·컴퓨트 유치 근거로 사용 | — |

타이밍: TokShop@COLM 2026(6/23)·COLM 2026 main(3/31)·EMNLP 2026 main(5/25)·ARR 8월 cycle(8/3) 모두 종료. **다음 문은 ARR 2026-10-12 하나.**

---

## §6 종합 결론

1. **ChatGPT 딥리서치의 사실은 믿어도 되지만, 해석은 믿으면 안 된다.** 인용 17건 전수 실존(날조 0)으로 사실 층위는 이례적으로 정확했다. 그러나 "연구 공백" 프레이밍은 (a) 자기모순인 6조건 스펙 위에 서 있고, (b) H-Net·AU-Net·Scratchpad Patching이라는 결정적 반례군을 놓쳤으며, (c) 마스킹 무용론을 스스로 인정하고도 결론에서 뒤집었다.

2. **원안의 서사 — "언어 규칙으로 neural compute를 대체한다" — 는 산술적으로 성립하지 않는다.** 한글 FSM에 결정적 transition이 0개이고, 마스킹 절감 상한이 0.04%이고, KV cache는 오히려 늘어난다. 규칙성≠예측가능성이라는 혼동이 뿌리다.

3. **그러나 프로젝트를 접을 이유는 없다.** "한국어에서 byte-latent patcher를 공짜로 만든다"는 좁지만 진짜인 기여가 남아 있고, 그 위에 결과가 어느 쪽이든 출판되는 과학적 질문을 세울 수 있다. 한글이 이 질문의 유일한 이상적 testbed라는 점이 프로젝트의 정당성이다.

4. **다음 행동은 코드가 아니라 측정이다.** HPLT 샤드 하나로 `|A(s)|=1` 비율과 Δ = f − f₀ 두 숫자를 뽑는다. Δ < 5%p면 접고, 살아 있으면 kill criteria를 문서로 고정한 뒤 $40짜리 100M 파일럿으로 간다. ChatGPT가 그린 1B 프리트레이닝 그림($66k+, funded lab 규모)은 MPU-A/B가 통과한 뒤 컴퓨트를 유치할 근거로만 쓴다.

---

## 부록 — 근거 출처

arXiv:2404.14408 (SpaceByte) · 2412.09871 (BLT) · 2605.08044 (Fast BLT) · 2404.02258 (MoD) · 2407.08103 (Automata Constraints) · 2502.05111 (GCD, ICML 2025) · 2507.07955 (H-Net) · 2506.14761 (AU-Net) · 2605.09630 (Scratchpad Patching) · 2404.19737 (MTP) · 2412.10418 (Spec. Lookaheads) · 2403.10691 (MYTE) · 2506.15138 (Thunder-Tok/Less Is More) · 2604.12377 (SCRIPT) · 2510.09426 (KORMo) · 2601.01739 (K-EXAONE)
ACL Anthology: 2023.eacl-main.172 (3-hot) · 2024.findings-acl.302 (KOMBO) · 2025.loresmt-1.8 (Jamo BPE) · Y05-1025 (Morphotactics)
기타: LDC2004L01 (Klex) · MoonshotAI/Kimi-K3 · LGAI-EXAONE · HPLT v2 · huggingface.co/docs/transformers/model_doc/blt

> 수치 추정(FLOPs 비율·비용·Δ 임계값)은 에이전트 계산 기반이며 Phase 0 실측으로 확정 필요.
