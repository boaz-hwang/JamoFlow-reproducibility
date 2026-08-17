# Retrieval novelty closure and large-model replication direction

> 작성일: 2026-08-15
>
> 상태: **최신 선행연구와 mechanism failure를 반영한 연구 방향 교정**
>
> 적용 범위: `docs/165`--`170` 이후 retrieval/speculative-decoding branch

## 결론

현재 증거는 한국어 경계 라우터나 단순 어절 사전을 새 방법으로 확대할 근거가 없다.

- trained fresh-v2 16K target에서 generic train-only corpus + prompt/self-output hybrid는 exact
  target-greedy free generation을 ordinary AR보다 26.244% 빠르게 했지만, controlled same-output은
  5.310%에 그쳐 사전 고정 joint gate를 실패했다.
- 후속 mechanism screen에서 `within Hangul eojeol` prompt proposal은 공백 직후 proposal보다
  acceptance가 높지 않았다. paired contrast는 예상과 반대였고 paired-case coverage도 부족했다.
- 2026년 선행연구는 linguistic-prior adaptive drafting, 비라틴 언어 dictionary speculation,
  prompt/corpus retrieval, tokenizer 불일치 정렬을 이미 각각 직접 다룬다.

따라서 다음 primary는 새 한국어-aware method가 아니다.

> **공개 7.8B 한국어 중심 모델을 실제 Apple Silicon에서 실행해, exact generic retrieval
> speculative decoding이 ordinary autoregressive decoding보다 실제 end-to-end generation을
> 빠르게 하는지 재현한다.**

이 단계는 소형 자체 모델에서 관측한 positive free-running effect의 외적 타당성을 판정한다.
통과하더라도 generic retrieval method novelty를 주장하지 않는다. 실패하면 현재 retrieval branch를
종료한다. 형태론 후속은 대형 모델에서도 강한 generic baseline이 재현되고, 별도 사전등록된
equal-cost screen이 추가 이득을 예측할 때만 다시 연다.

## 1. 이번 교정이 필요한 이유

### 1.1 기존 actual 결과가 말하는 것과 말하지 않는 것

현재 16K 결과의 가장 중요한 positive evidence는 proxy가 아니라 실제 target execution이다.

- target verification, cache crop/rollback, correction/bonus, retrieval lookup, strict UTF-8 stop이
  timed region에 포함되었다.
- free-running output token IDs와 bytes는 ordinary strict greedy AR과 exact했다.
- full-forward/cache equivalence와 counter reconstruction이 모두 통과했다.

그러나 target은 작은 one-seed development model이다. 이 결과만으로 production-size LLM의
speedup, 다른 tokenizer/model에서의 재현성, 한국어 구조의 인과 기여를 주장할 수 없다.
Controlled gate 실패도 숨기지 않는다. 그것은 retrieval의 효과가 고정 continuation replay보다
모델이 실제 생성하는 trajectory에서 더 크다는 **estimand 차이**를 보여 주지만, 그 차이 자체가
한국어-specific mechanism이라는 증거는 아니다.

### 1.2 사전 고정한 Hangul-boundary 가설은 닫혔다

`docs/168`--`170`의 가설은 공백 직후보다 한글 어절 내부에서 prompt copy가 더 길게 맞는다는
것이었다. 실제 contrast는 반대 방향이었고 paired-case 최소 수도 충족하지 못했다. 따라서 다음을
하지 않는다.

- 508개의 descriptive `Hangul-inside/no-proposal` cycle을 사후 primary로 승격
- threshold, stratum, pair 정의를 바꾸어 같은 case를 다시 검정
- 공백/어절 boundary router를 새 held-out set에서 즉시 재시도
- 실패한 가설을 형태소 경계나 linguistic uncertainty의 간접 증거로 해석

이는 negative result를 버리는 것이 아니라, boundary-router branch를 올바르게 종료하는 것이다.

## 2. 2024--2026 선행연구가 닫은 novelty

### 2.1 언어학적 prior로 draft depth/verification을 조절하는 발상

[LinguaSpec](https://aclanthology.org/2026.findings-acl.1065/)은 static linguistic category,
category-normalized surprisal, syntactically guided elastic expansion, POS-adaptive deferred
verification을 결합한다. 한국어 형태론을 평가하지 않았고 exact greedy equivalence와도 estimand가
다르지만, 다음과 같은 넓은 주장은 이미 불가능하다.

- `linguistic uncertainty -> adaptive draft/compute`가 처음이다.
- predictable syntactic region에서 draft depth를 늘리는 것이 처음이다.
- POS/문법 category를 verification policy에 넣는 것이 그 자체로 기여다.

한국어 후속이 가능하려면 LinguaSpec의 단순 언어 교체가 아니라, exact decoding 아래에서 generic
retrieval보다 실제 latency를 더 줄이는 별도 mechanism을 보여야 한다.

### 2.2 비라틴 언어 corpus dictionary와 prompt hybrid

[DictSpec](https://aclanthology.org/2026.unlp-1.15/)은 Ukrainian과 Crimean Tatar를 대상으로
unlabeled corpus의 static n-gram table을 사용한다. trainable parameter나 GPU draft model 없이
5 MB 미만의 table을 만들고, prompt-local lookup과 결합한 live vLLM 결과에서 최대 1.76배를
보고한다. 따라서 다음은 Korean replication으로서 의미는 있어도 method novelty는 아니다.

- 한국어 corpus에서 어절/문자/token continuation dictionary를 만드는 것
- non-Latin/high-fertility tokenizer에서 lookup draft가 유리하다는 것
- corpus-first + prompt/self-output fallback
- compact static table을 on-device inference에 붙이는 것

### 2.3 prompt/self-output retrieval와 hardware-aware adaptation

[SSSD](https://aclanthology.org/2026.acl-long.1530/)는 lightweight n-gram matching과
hardware-aware speculation을 결합하고 최대 2.9배 latency reduction을 보고한다.
[LogitSpec](https://aclanthology.org/2026.findings-acl.1655/)은 target의 last logit을 이용해
retrieval query 범위를 넓히며 Llama-3.1-8B와 Qwen3-8B에서 actual speedup을 보고한다.
그러므로 prompt suffix, generated-history cache, 장치별 draft length, target-logit-guided lookup 중
하나를 단독 novelty로 삼지 않는다.

### 2.4 tokenizer 불일치와 language-specific vocabulary

[TokenTiming](https://aclanthology.org/2026.acl-long.1983/)은 draft string을 target tokenizer로
다시 encode하고 probability distribution을 동적으로 정렬해 vocabulary가 다른 model pair를
지원한다. [OmniDraft](https://proceedings.neurips.cc/paper_files/paper/2025/file/3c2fe1417eed1c6ff9acf169617981ea-Paper-Conference.pdf)는
cross-vocabulary online n-gram cache와 adaptive drafting을 on-device setting에서 다룬다.
[Hong et al.](https://aclanthology.org/2024.findings-acl.660/)은 excessively tokenized language에
target-language vocabulary head를 붙여 generation speed를 개선한다. 따라서 character/Jamo draft를
target tokenizer로 재직렬화하거나 Korean-specific head를 두는 것만으로는 새 기여가 아니다.

### 2.5 한국어 형태 토큰화

[Morpheme Matters](https://aclanthology.org/2026.eacl-short.22/)는 inter-/intra-eojeol
morpheme-based subword selection으로 한국어 모델을 pretrain하고 더 적은 token과 downstream
개선을 보고한다. 이는 speculative decoding 논문은 아니지만, `한국어 형태를 이용해 sequence를
효율화한다`는 넓은 framing을 선점한다.

## 3. novelty 판정표

| 후보 주장 | 판정 | 이유 |
|---|---|---|
| 한글/공백 boundary로 draft 길이 조절 | 종료 | 사전등록 mechanism 실패 + LinguaSpec |
| 한국어 어절 dictionary speculative decoding | novelty 아님 | DictSpec/SSSD |
| Jamo/문자를 target token으로 재직렬화 | novelty 아님 | TokenTiming/OmniDraft |
| 한국어 tokenizer fertility 때문에 retrieval이 빠름 | replication question | DictSpec이 비라틴 언어에서 직접 연구 |
| generic exact retrieval의 Korean 8B Apple 실측 | 가치 있는 재현/시스템 질문 | 현재 직접 증거 없음, 실제 latency가 사용자 성공 기준 |
| productive morpheme-normalized retriever | 조건부 연구 가설 | surface n-gram과 기능적으로 달라야 하고 equal-cost actual gain 필요 |

마지막 후보의 핵심은 형태 분석기를 호출한다는 사실이 아니다. 여러 활용형을 하나의 productive
pattern으로 묶어 **unseen inflection에서 proposal coverage를 늘리고**, 동일 table bytes와 동일
target verification budget에서 generic Unicode/token retriever보다 빨라야 한다. 이 조건을 충족하지
못하면 단순 사전 변형이다.

## 4. Fable 5 검토에서 수용한 부분

`fable5-연구-중간-검토.md`의 핵심 경고는 타당했다.

1. 분석적 compute 절감은 wall-clock 개선이 아니다.
2. 작은 model과 단일 Apple 장치 결과는 top-tier scale/general-hardware evidence가 아니다.
3. local/global cost 분해상 이론적 개선이 있어도 dispatch, cache, readback 때문에 E2E gate를
   실패할 수 있다.
4. 결과가 음성이면 새 feature를 사후 추가하지 말고 claim을 줄여야 한다.

현재 연구는 이 네 항목을 실제로 반영했다. 기존 boundary architecture의 v5r3 actual 결과와
retrieval 16K actual 결과를 proxy로 부르지 않았고, free/controlled estimand를 분리했으며,
boundary-router 실패 뒤 morphology feature를 붙이지 않았다.

다만 Fable 5의 `S 우위 rate-matched 분해` 권고는 당시 boundary paper를 강화하는 좋은 제안이지만,
사용자의 최종 기준인 실제 LLM inference efficiency를 가장 직접적으로 높이는 다음 투자로는
채택하지 않는다. W/S compact 분석을 더 확장하는 것보다 이미 actual positive signal이 나온
retrieval을 8B public target에서 검증하는 편이 현재 의사결정 가치가 높다. 기존 W/S 결과는
별도 empirical paper evidence로 보존한다.

## 5. 다음 단계의 정확한 연구 질문

### Primary question

> 고정된 공개 한국어 중심 7.8B 4-bit model과 고정 Apple Silicon 환경에서, train-only compact
> token n-gram + prompt/self-output fallback이 ordinary greedy AR과 token-for-token exact한 출력을
> 유지하면서 free-running end-to-end generation latency를 줄이는가?

### Secondary questions

- controlled same-output replay에서도 target-call 절감이 wall time으로 이어지는가?
- corpus-only, prompt-only, hybrid의 차이는 proposal coverage와 accepted tokens/call 중 어디서
  발생하는가?
- tokenizer fertility, repetition, prompt/output overlap이 per-case speedup을 설명하는가?
- lookup/table memory와 process peak를 포함해도 on-device deployment cost가 작은가?

Secondary 분석은 primary 실패 뒤 winner를 고르거나 설정을 바꾸는 근거가 아니다.

## 6. model과 runtime 선택 규칙

### 6.1 고정 우선순위

1. **Primary:** `mlx-community/EXAONE-3.5-7.8B-Instruct-4bit`, model revision
   `6f8fba5756a6e2987aecacd8d7e8bb9410ef1a53`
2. **Technical fallback only:** `mlx-community/Qwen3-8B-4bit`, model revision
   `545dc4251c05440727734bcd94334791f6ab0192`

EXAONE 3.5는 Korean/English bilingual model이며 7.8B, 32 layers, 102,400 vocabulary다. 현재 MLX
artifact는 약 4.40 GB이고 MLX-LM main에는 built-in `exaone` implementation이 존재한다. Mac의
48 GB unified memory에서 사용자가 요구한 “가능한 큰 model”과 한국어 중심성을 동시에 만족한다.

Qwen3-8B는 성능이나 speed 결과를 보고 바꾸는 대안이 아니다. 다음 **비성능 호환성 조건** 중
하나라도 EXAONE에서 실패할 때만 사용한다.

- pinned MLX-LM에서 built-in model loader가 checkpoint를 읽지 못함
- tokenizer/chat template round trip 실패
- full-prefix와 cached incremental의 finite logits 또는 greedy argmax가 일치하지 않음
- 반복 deterministic greedy output 불일치
- baseline 128-token generation이 memory safety limit를 넘거나 runtime crash

Acceptance, target-call reduction, tokens/s, Korean output quality를 보고 fallback할 수 없다. EXAONE이
preflight를 통과하면 primary는 EXAONE으로 고정한다. 두 model 중 빠른 쪽을 고르는 model shopping도
금지한다.

### 6.2 runtime

- MLX-LM `0.31.3`, MLX `0.31.2`를 별도 optional environment로 exact pin한다.
- stock speculative loop의 correctness를 가정하지 않는다. Model/cache primitives만 사용하고
  JamoFlow transaction을 독립 구현한다.
- ordinary AR와 retrieval path 모두 같은 quantized weights, tokenizer, prompt template, stop rule,
  cache class, synchronization rule을 사용한다.
- model load와 one-time table construction은 latency 밖이지만 resident table memory는 보고한다.
  Per-request lookup, prompt index construction, verification, rollback, correction/bonus, detokenization은
  명시한 end-to-end scope에 포함한다.

## 7. 결과를 보기 전의 단계별 gate

### Stage A: timing-silent compatibility preflight

- model/config/tokenizer artifact와 revision hash 봉인
- tokenizer/chat-template deterministic round trip
- actual parallel prefill/cached decode vs full-prefix finite-logit 및 greedy-decision equivalence
- cache crop/rollback correctness on synthetic tokens
- ordinary greedy repeated-output identity
- fixed tiny speculative transaction의 target-token exactness
- memory high-water safety

이 단계에서는 candidate-vs-baseline latency ratio를 계산하거나 출력하지 않는다. 호환성만 판정한다.

### Stage B: data/table integrity

- public Korean train split만으로 table 구성
- evaluation documents와 exact digest disjointness
- fixed byte budget과 deterministic collision/tie rule
- table artifact bytes, entry count, tokenizer revision, build-source hash 봉인
- prompt/self-output fallback implementation과 exact lookup trace test

`vault/`의 사용자 Markdown은 오류·code-mixing·긴 문맥을 찾는 read-only secondary diagnostic에만
사용한다. Public primary case 선택, table 생성, threshold 결정에는 사용하지 않는다.

### Stage C: prospectively sealed actual protocol

Compatibility가 통과한 뒤, timing을 열기 전에 다음을 고정한다.

- new public Korean one-document-per-case prompt set
- 64 primary prompts, prompt/output horizon, stop semantics
- hybrid table budget와 lookup order
- proposal length primary `3` tokens, 즉 target block `4`
- ordinary AR vs exact hybrid free-running primary
- controlled replay, prompt-only, corpus-only는 secondary diagnostic
- independent fresh-process session 수와 balanced role order
- bootstrap unit은 prompt와 session이며 inner repetitions는 median으로만 collapse

Primary publication gate의 방향은 다음이다.

- free-running median end-to-end reduction `>= 10%`
- paired prompt/session bootstrap 95% lower bound `> 0`
- 적어도 `48/64` prompts faster
- 모든 session에서 aggregate reduction positive
- output token IDs와 decoded bytes exact
- full/cache/rollback/counter/equivalence gate 전부 pass

정확한 session 수와 inner repetition 수는 **candidate 비교를 하지 않는 baseline-only resource
calibration**으로 정한 뒤 plan에 봉인한다. Controlled가 10% 미만이라는 이유만으로 free primary를
실패 처리하지 않는다. 이전 joint gate는 small-model mechanism screen의 보수적 조건이었고,
사용자의 실제 free inference 기준과 현재 관측된 estimand split을 반영해 controlled를 secondary로
내린다. 이 변경은 새 model/case/timing을 보기 전에 기록되므로 사후 gate 변경이 아니다.

## 8. 성공과 실패가 허용하는 논문

### 8.1 EXAONE primary가 통과할 때

한 model/한 hardware 통과만으로 새 speculative-decoding method paper를 주장하지 않는다. 허용되는
결론은 다음과 같다.

> Compact exact retrieval speculative decoding can transfer from a controlled small Korean target to a
> public 7.8B Korean-centric model and improve free-running on-device generation on one Apple Silicon
> configuration.

그 뒤 publication-strength evidence를 위해 다음 순서로 확장한다.

1. 같은 protocol을 Qwen3-8B에 복제해 model-family generality를 확인
2. 적어도 하나의 CUDA/vLLM 또는 llama.cpp-style external runtime과 대조
3. DictSpec/SSSD에 맞춘 table-size와 prompt-only control 공개
4. energy/peak memory는 측정 capability가 검증될 때만 secondary로 보고
5. Hugging Face에 table builder, exact decoder, sealed cases, receipts 공개

이 경로의 논문 정체성은 Korean/on-device exact systems replication과 careful target-trajectory
analysis다. top-tier method novelty를 자동으로 충족하지는 않는다.

### 8.2 EXAONE primary가 실패할 때

- block length, table size, prompt set을 바꾸어 같은 model에서 재도전하지 않는다.
- Qwen fallback은 EXAONE **호환성 실패**일 때만 허용되므로 latency 실패 뒤 대체 model로 사용할 수
  없다.
- 16K positive와 7.8B negative를 함께 보고 scale-transfer failure로 정리한다.
- current retrieval/Korean-draft branch를 종료하고 paper는 기존 byte-boundary quality/cost 결과와
  negative actual scaling evidence로 축소한다.

### 8.3 morphology를 다시 열 수 있는 조건

다음을 모두 만족할 때만 새 문서와 새 disjoint case로 연다.

1. EXAONE generic primary 통과
2. public/reproducible Korean analyzer 또는 analyzer-free normalization 정의
3. surface token n-gram과 기능적으로 다른 productive generalization 예시
4. equal serialized table bytes, equal proposal cap, identical target/runtime
5. latency를 보지 않는 coverage/acceptance screen이 사전 minimum effect 통과
6. generic hybrid 대비 actual free E2E 추가 개선을 독립 gate로 봉인

형태론 분석기 latency, normalization, ambiguity, OOV fallback을 timer 밖에 숨길 수 없다. 이 여섯
조건 중 하나라도 준비되지 않으면 형태론은 future work다.

## 9. 실행 순서

1. 본 novelty closure와 direction을 commit한다.
2. MLX-LM dependency와 exact revisions를 pin한다.
3. EXAONE metadata-only validation 후 4-bit checkpoint를 한 번 내려받는다.
4. timing-silent compatibility/correctness preflight를 구현하고 실행한다.
5. pass하면 public Korean table/case protocol을 구현하고 plan을 commit한다.
6. actual sessions를 실행하고 immutable receipt를 session별 commit한다.
7. 결과를 한 번 요약해 primary gate를 판정한다.
8. pass일 때만 Qwen3 replication과 publication package로 확장한다.

계획은 새 증거가 현재 가정을 실제로 반박할 때만 수정한다. 호환성 실패는 predeclared fallback을
발동하고, latency/acceptance 결과는 설정 변경이 아니라 pass/stop 판정만 만든다.
