# JamoFlow — 논문 주제 선정 배경·과정·결론

> 작성일: 2026-08-08 · **개정: 2026-08-08 (확정 철회)**
> 상태: **역사적 주제 선정 기록** ([02 재검토](./02-critical-research-direction-review.md) 수용, 인용은 [03](./03-citation-verification.md)에서 전수 확인; 현재 판정은 [31 전 과정 감사](./31-source-conversation-and-decision-audit.md) 참조)
> 관련 문서: [검증 보고서](https://claude.ai/code/artifact/ffdc3176-fa40-4893-af4c-29d15519f4dc) · [연구 로드맵](https://claude.ai/code/artifact/7a93ddab-8f46-4295-99b2-8e635686065d)

---

## 0. 현재 상태 — 후보 가설 (구 "확정 주제"의 격하)

> ⚠️ **2026-08-08 개정.** [02 재검토](./02-critical-research-direction-review.md)가 아래 구 확정 주제의 결함
> (싼 경계 ≠ 좋은 경계 혼동, 알고리즘 미정의, 31% 추정의 가정 의존성, every-syllable = 3 bytes 문제)을 지적했고,
> 02의 인용 9건·세부주장 10건은 [03 인용 검증](./03-citation-verification.md)에서 전수 확인됐다.
> 따라서 "주제 확정"을 철회하고 아래와 같이 재정의한다.

**중심 연구 질문 (개정):**

> **학습형 patcher의 품질 이득은 patcher 자체의 계산비용을 정당화하는가?
> 결정적 orthographic structure를 learned routing의 "대체"가 아닌 "저비용 prior/후보 생성기"로 쓰면,
> rule-only와 learned-only보다 나은 BPB–latency–KV-cache Pareto frontier를 만들 수 있는가?**

가제(개정): **_Do Learned Patchers Pay for Themselves? Orthography-Aware Cost-Constrained Patching for Byte-Level Language Models_**
(한국어 중심 축소판: _Orthography-Aware Hybrid Patching for Korean and Code-Mixed Byte-Latent Language Models_)

**후보 가설 (구 "핵심 주장"의 개정 — 결과를 선취하지 않는 형태):**

| 가설 | 상태 |
|---|---|
| H1 — rule-only policy는 router 비용을 줄이지만 context-sensitive boundary 품질이 낮아질 수 있다 | 열린 질문 |
| H2 — hybrid(automaton이 causal candidate 위치 제공 + learned scorer가 그 위치에서만 실행)는 learned-only 품질 대부분을 유지하며 router 비용을 줄인다 | 열린 질문 (핵심 기여 후보) |
| H3 — 이득의 일부는 한글 고유 구조가 아니라 generic UTF-8/codepoint alignment에서 나온다 | 중국어 control로 분리 |
| H4 — hard orthographic constraint는 speed보다 validity·robustness에서 먼저 기여한다 | 별도 2차 연구 (02 §7) |
| H5 — patcher FLOPs 절감이 실제 latency로 이어지는지는 batch·memory-bandwidth regime에 따라 다르다 | 실측 필수 |

**구 주제 대비 핵심 정정 3가지:**
1. `zero-cost` → **`parameter-free`** (byte 분류·상태 전이·CPU/GPU sync 비용은 0이 아님 — 실측 후 "negligible measured overhead" 여부 판정)
2. "1B급 decode compute ~31% 절감" → **가정 의존 이론 상한** (100M patcher·byte별 독립 forward·FLOPs=latency 가정. BLT 원문은 50M 이상 수확체감·lookup table 가능성 명시, Scratchpad Patching은 patcher를 encoder 위 2-layer aux head로 통합)
3. "동등 이상의 patch 길이 달성" → **rule patcher의 causal 알고리즘(음절/k음절/어절/byte-budget) 정의가 선행 과제.** every-syllable은 ~3 bytes/patch로 BLT의 4.5~8보다 짧아질 수 있음. 모든 boundary 결정은 prefix-causal이어야 함(미래 문자 의존 segmentation은 leakage)

---

## 1. 배경 — 최초 아이디어 (ChatGPT 대화, 2026-08-07~08)

ChatGPT와의 대화("자소 분리와 효율성", 8턴)에서 출발한 가설:

> "한글을 처음부터 자소 분리해서 프리트레이닝하면 추론 런타임의 어텐션이 더 효율적이지 않을까?"

대화를 거치며 아이디어가 다음으로 진화했다:

1. 자소 단위 프리트레이닝 → 추론 효율 개선 (최초 가설)
2. 한글 조합 규칙(초·중·종성 FSM)·실사용 음절 분포·형태론·띄어쓰기를 **deterministic constraint**로 사용
3. 예측 가능한 transition에서는 Transformer forward를 생략, rule decoder가 multi-jamo 생성
4. ChatGPT 딥리서치 결론: 6조건(자소 atomic + FSM 디코딩 제약 + 형태론/공백 state +
   결정 구간 compute 생략 + multi-jamo 생성 + end-to-end pretraining/FLOPs 평가)을
   모두 만족하는 연구는 없다 → **"연구 공백"** 주장
5. ChatGPT가 스스로 "연구적으로 가장 정확한 표현"이라 못박은 최종 정식화:
   > "entropy-based patching을 **linguistically informed entropy-based patching**으로 바꾸는 것"

## 2. 검증 과정 — Claude 에이전트 7팀 재검토 (2026-08-08)

ChatGPT 대화 1~6번 턴 전체를 독립 에이전트 7팀으로 적대적 검증했다.

| 팀 | 역할 | 방법 |
|---|---|---|
| 사실검증 ×3 | 바이트 아키텍처 / 한국어 자소 연구 / 최신 모델 스펙 인용 검증 | arXiv·ACL Anthology·공식 GitHub·HF 1차 출처 대조 |
| 신규성 사냥 ×1 | "연구 공백" 주장을 무너뜨리는 선행연구 탐색 | 2023–2026 문헌 적대적 검색 |
| 기술 비판 ×1 | 논증의 정량적 반박 (FLOPs·KV cache·Amdahl 계산) | 산술 검증 |
| 실현가능성 ×1 | 1인 개발자 기준 비용·데이터·엔지니어링 리스크 | 2026-08 GPU 시세·코퍼스 실측 |
| 출판 심사 ×1 | 리뷰어 시뮬레이션·venue 분석 | ARR/학회 일정 실측 |

## 3. 검증 결과 요약

### 3-1. 사실검증: 인용 17건 전수 실존, 날조 0건

- 컷오프 이후 인용 4건까지 실존 확인: Fast BLT(arXiv:2605.08044, ICML 2026),
  SCRIPT(arXiv:2604.12377, ACL 2026 **Findings**), Kimi K3(2.8T/104B/896 experts),
  K-EXAONE 2.0(750B-A37B).
- 부정확 2건: SCRIPT venue(Findings 누락), Thunder-Tok(정식 제목 "Less Is More",
  한국어 수치 9%/8%, preprint 상태).
- **결론: ChatGPT의 사실은 신뢰 가능. 그러나 해석·프레이밍은 별개.**

### 3-2. 신규성: "연구 공백" 부분 붕괴

- ChatGPT가 놓친 결정적 선행연구: **H-Net**(2025-07, learned 청킹이 hand-crafted 경계를
  실험으로 이김), **AU-Net**(Meta 2025-06), **Scratchpad Patching**(Google 2026-05).
  "adaptive compute + 다중 심볼 생성"은 2026년 5~7월 동시 수렴 중인 붐비는 클러스터.
- 6조건 AND는 좁은 의미로 비어 있으나, 조건 (2) FSM 디코딩 제약과 (5) multi-jamo 생성이
  **논리적으로 상호 모순**(자소별 스텝이 없으면 제약할 대상도 없음) — 공백이 아니라 자기모순.

### 3-3. 기술 비판: 코어의 ~90% 기각

| 주장 | 판정 | 근거 |
|---|---|---|
| FSM 후보 마스킹으로 compute 절감 | **기각** | output head는 자소 LM forward FLOPs의 0.042%(1B)/0.013%(7B). 실측 절감 0 |
| "음절 결정 → rule decoder 전개" | **기각(선점)** | 음절 tokenizer + 결정적 detokenization과 계산 동치 = EACL 2023 Cognetta 3-hot decoder (코드 공개) |
| 한글은 규칙적 → 예측 가능 구간 많음 | **기각** | 결정적(branching factor 1) transition **0개** (초성 뒤 21개, 중성 뒤 ~48개). 규칙성 ≠ 예측가능성 — 한글은 자소마다 독립 정보를 싣는 효율적 문자라 철자 잉여가 라틴 문자보다 적음 |
| 자소 분리 → KV cache/스텝 감소 | **기각(역방향)** | 한국어 BPE(어절당 1.35~1.5토큰) 대비 스텝 5~6배·KV 2.1~5배 **증가** |
| multi-jamo rule decoder | **기각** | speculative decoding의 열등한 특수사례. Fast BLT의 학습 draft가 지배 |
| 결정론 구간 학습 제외 (loss masking) | **기각** | 제외할 결정론 구간이 0%. 표면 규칙은 학습 초반 수천 스텝에 흡수되어 절약 풀 자체가 미미. 학습 절약 ≠ 추론 절약 |
| 한국어 전용 대형 pretraining | **기각** | 고유 자연 한국어 상한 ~40-60B 토큰(KORMo는 68.74% 합성으로 충당). 풀 실험 $66k~106k = funded lab 규모 |
| **한글 음절 경계 = 공짜 → entropy patcher 대체** | **생존** | BLT patcher는 100M 모델·byte당 온라인 forward 필요 = 1B급 decode compute ~31%. 한글은 Unicode 산술로 경계 결정적. **유일한 free lunch, 미측정** |

### 3-4. 출판 심사

- 원래 프레이밍("Rule-Guided Adaptive Computation") 그대로는 **REJECT** —
  리뷰어 1순위 공격: "절감 메커니즘은 EACL 2023과 동일, 나머지는 장식."
- 살아남은 주제로 축소·재프레이밍 시 **CONDITIONAL ACCEPT**.
- 현실 타깃: **ARR 2026-10-12 → NAACL/COLING 2027** (TokShop@COLM 2026은 6/23 마감 종료).

## 4. 주제 선정 논리 — 왜 이 주제인가

**선정 기준: (a) ChatGPT 대화의 최종 결론에서 검토 가치가 있다고 지목된 영역 ∩ (b) Claude 재검증에서 실제로 살아남은 영역.**

- (a) = ChatGPT의 최종 정식화 "linguistically informed entropy-based patching"
- (b) = "한글 음절 경계는 공짜로 결정적 → BLT의 유료 entropy patcher를 0-cost 규칙으로 대체"
- **교집합 = 정확히 하나. 그것이 §0의 주제다.**

폐기된 것: Rule-Guided Adaptive Computation 계열 전부 (compute router, FSM 스킵,
multi-jamo rule decoder, 자소 attention, 결정론 구간 학습 제외, 한국어 전용 대형 pretraining).

규칙의 역할 재정의: 모델의 예측을 **대체**하는 자(불가능) → 모델에게 **공짜 정보를 주는** 자
① 경계 신호(patcher 대체 — 본 논문) ② legality guard(정확성) ③ 학습형 경계의 prior(H-Net 접목 변형).

"hand-crafted vs learned" 질문은 논문 주제가 아니라 **H-Net 반례에 대한 관련연구 방어 문구**로만 사용:
"우리는 학습기를 정확도로 이기려는 게 아니라, 학습기 없이 같은 경계를 공짜로 얻는다" —
정확도 싸움이 아닌 비용 싸움으로 포지셔닝.

## 5. 다음 단계 — Phase 0: corpus·routing audit ($0, ~2주) 【02 §6.1 반영 개정】

GPU 학습 없이 로컬(M4 Pro)에서 HPLT v2 Korean cleaned(CC0) 샤드로:

1. **표현 분포 audit** — NFC/NFD 비율, compatibility jamo(ㅋㅋ·ㅠㅠ)·단독 자모, 옛한글·emoji·숫자·라틴·코드·URL, 도메인별 code-mixing 비율. (+ 음절 빈도 90/99/99.9% 커버리지 — 공개 수치 부재, 그 자체로 새 데이터)
2. **causality 검증** — 각 boundary rule에 prefix invariance test: prefix에서 계산한 경계 결정 = 전체 sequence에서 계산한 동일 prefix의 결정. 미래 문자를 쓰는 offline segmentation 제거
3. **matched patch-rate 비교** — 목표 평균 3 / 4.5 / 6 / 8 bytes per patch에서 fixed byte stride · UTF-8 codepoint-aligned stride · SpaceByte · entropy · rule variants(음절/k음절/어절/budget) 비교
4. **boundary quality proxy** — oracle next-byte surprisal peak coverage, high-surprisal 위치와 직전 global update 사이 거리(patch lag), patch 내 surprisal 분포, script별 patch-length tail, learned router가 필요해지는 candidate 비율
5. **실측 patcher cost** — theoretical FLOPs, parameter memory, batch 1/8/64 latency, CPU/GPU sync 비용

> 구 지표 `|A(s)|=1`·자소 조건부 엔트로피·`Δ = f − f₀`는 **폐기된 compute-skip 가설의 폐기 근거를 기록하는 appendix 분석**으로 이동 — 새 주제의 Go/No-Go 게이트로 쓰지 않는다.

**Go/No-Go 기준 (02 §6.5 반영 — 절대 임계값은 pilot variance 측정 후 최종 실험 전에 사전 고정):**

1. rule-only 또는 hybrid가 **trivial codepoint-aligned fixed stride와 SpaceByte를 이겨야** 진행
2. 같은 BPB 신뢰구간에서 유의미한 end-to-end latency 또는 KV 개선 — 또는 같은 실측 latency에서 유의미한 BPB/downstream 개선
3. 이득이 code-mixed 조건에서 즉시 사라지지 않아야 함
4. 100~300M에서 scale trend가 없으면 1B로 넘어가지 않음

**전 과정 수칙:** 비교 축은 UTF-8 바이트(BPB) · baseline 3축 분리(patching policy / representation / output generation — 3-hot·MTP는 patcher baseline과 같은 표에 두지 않음) · SpaceByte·codepoint stride를 최우선 baseline으로, H-Net·Scratchpad(통합 router)를 현행 baseline으로 · 중국어 control과 wall-clock 실측 필수 · negative result도 발표 가능하게 설계 · HPLT(CC0) 척추, HF transformers v4.57+ clean-room BLT 구현만 사용, `facebookresearch/blt`(CC BY-NC)·AI-Hub 배제.

**별도 2차 연구로 분리 (02 §7):** 원안의 rule + multi-jamo 축은 _Orthography-Constrained Block Decoding_ (learned block proposer + Hangul/UTF-8 DFA 검증)으로 보존 — patching 논문과 결합 금지(scope explosion 방지). 1차 기대 기여는 speed가 아니라 validity·robustness (근거 힌트: UTF-8 validity가 perplexity보다 ~2배 늦게 수렴, arXiv:2606.14122).

---

## 부록 — 핵심 근거 문헌

| 문헌 | 역할 |
|---|---|
| BLT (arXiv:2412.09871) | 대체 대상 아키텍처. entropy patcher 비용의 출처 |
| Fast BLT (arXiv:2605.08044, ICML 2026) | multi-symbol 생성 선점 — 해당 축 폐기 근거 |
| SpaceByte (arXiv:2404.14408, NeurIPS 2024) | 가장 싼 hand-crafted 경쟁 baseline (공백 patching) |
| H-Net (arXiv:2507.07955) | learned 경계 > hand-crafted 반례 — 방어 대상이자 prior 주입 접목처 |
| Cognetta et al. (EACL 2023, 2023.eacl-main.172) | "음절 결정→자소 전개" 선점 — 필수 baseline |
| KOMBO (2024.findings-acl.302) · SCRIPT (arXiv:2604.12377) | 자소 표현의 품질 효용 (속도 아님) 계보 |
| Thunder-Tok "Less Is More" (arXiv:2506.15138) | 한국어 토큰화 헤드룸 ~9% — 기대치 보정 근거 |
| KORMo (arXiv:2510.09426) | 한국어 데이터 천장 실증 (68.74% 합성) |
| HPLT v2 Korean | 학습 코퍼스 척추 (CC0, 89.27B chars) |
