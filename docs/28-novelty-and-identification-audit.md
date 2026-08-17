# Phase 3 novelty and causal-identification audit

> 작성일: 2026-08-10
> 상태: **Phase 3 C/W 비교 결과 확인 전 고정**
> 범위: 최초 대화, `01` 검증, 최종 주제 선정, Phase 2 결과, Phase 3 protocol과 2026-08-10 현재 1차 문헌의 재감사
> 목적: “무엇이 새롭나”와 “무엇이 실험으로 식별되나”를 분리하고, 논문이 살아남기 위한 추가 증거를 결과 전에 정한다.

> 최신 보완: 2026-08-04까지 공개된 Bolmo·boundary-disentanglement·H-Net++·ATDC의 영향은 [36 amendment](./36-latest-boundary-literature-amendment.md)에 추가했다.

## 1. 최종 판정

현재 JamoFlow는 **새로운 한국어 tokenizer**, **최초의 linguistic patcher**, **최초의 cheap non-neural boundary**, **Jamo-aware architecture**로 제출하면 안 된다. 이 네 framing은 모두 직접 선행과 현재 구현에 의해 기각된다.

가장 방어 가능한 질문은 다음이다.

> 동일한 raw-byte latent graph와 동일한 global-position budget에서, Korean UTF-8의 codepoint geometry와 이미 관측된 whitespace를 이용한 prefix-causal boundary relocation이 재현 가능한 품질 차이를 만드는가? Authentic SpaceByte cadence와 learned entropy router까지 실제 rate·padding·router 비용을 포함하면 어떤 quality–cost frontier가 생기는가?

여기서 잠재 기여는 하나의 기발한 규칙이 아니라 다음 **식별 설계와 측정 묶음**이다.

1. boundary placement와 patch count를 분리한 same-rate 실험
2. Unicode-safe generic C와 whitespace-conditioned W의 분리
3. SpaceByte predicate가 Korean UTF-8 lead byte에서 만드는 실제 cadence 측정
4. entropy router, batch padding, selector, memory를 포함한 total cost
5. NFC natural text, NFD synthetic stress, public OOD, mixed Markdown ecology의 분리
6. positive/negative 결과에 같은 사전등록 gate를 적용하는 failure-resistant protocol

이 묶음은 논문 가치가 있지만, 19.6M mechanism study 하나만으로 top-tier “효율적인 한국어 LLM” 주장을 지탱하지는 못한다.

## 2. 이번 재감사에서 추가로 발견한 직접 선행

### 2.1 Hierarchical BPE는 가장 가까운 누락 baseline이다

[Dolga et al. (Findings EMNLP 2025)](https://aclanthology.org/2025.findings-emnlp.595/)은 기존 BPE token 구조를 character patch로 바꾸고 second-level BPE로 patch granularity를 조절한다. 논문은 별도 auxiliary model 없이 dynamic entropy- 및 whitespace-based patching과 맞먹거나 능가한다고 보고한다.

이 선행이 주는 수정은 크다.

- “W는 E보다 싸다”만으로 cheap boundary의 state of the art를 이겼다고 할 수 없다.
- standard BPE 하나만으로도 publication-scale external control이 충분하지 않다.
- vocabulary-free라는 장점과 vocabulary-derived boundary의 품질을 같은 축으로 놓아야 한다.
- 공식 artifact가 확인되지 않으므로 자체 근사를 원 논문의 재현이라고 부르면 안 된다. 구현하지 못하면 수치 baseline 누락을 limitation으로 적는다.

차이는 여전히 있다. Hierarchical BPE는 학습된 vocabulary, explicit patch marker, encoding pipeline을 사용한다. JamoFlow W는 raw UTF-8와 byte loss를 그대로 두고 online patch start만 바꾼다. 따라서 같은 intervention은 아니지만, **low-runtime-cost boundary competitor**로는 직접 관련된다.

### 2.2 Morphology-driven byte encoding은 이미 존재한다

[MYTE (ACL 2024)](https://aclanthology.org/2024.acl-long.804/)는 morpheme inventory를 이용한 byte encoding을 제안하고 99개 언어에서 더 짧은 encoding과 multilingual LM 개선을 보고했다. 따라서 “morphology를 byte modeling 아래층에 처음 넣는다”는 framing도 불가능하다.

MYTE는 원문 UTF-8을 다른 encoding으로 바꾸고 JamoFlow는 encoding을 보존한다. 차이는 분명하지만, future morphology-FST extension을 독립적인 새 아이디어처럼 제안해서는 안 된다.

### 2.3 Korean token length와 morphology를 다루는 경쟁선은 더 넓다

[LeVoC (Findings ACL 2024)](https://aclanthology.org/2024.findings-acl.135/)는 Korean BPE의 over-segmentation을 줄이기 위해 긴 단어를 vocabulary에 통합했고, MT와 morphology preservation 개선을 보고했다. [Morpheme Matters (EACL 2026)](https://aclanthology.org/2026.eacl-short.22/)는 inter-/intra-eojeol morpheme subword를 사용해 일반적으로 더 좋은 task 성능과 더 적은 token 수를 보고했다. [MorphBPE (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.2068/)는 Korean을 포함하지 않지만 300M/1B decoder-only LM에서 morpheme-respecting merge가 더 짧은 token 때문이 아닌 cross-entropy 개선을 만들 수 있음을 보였다.

따라서 W가 이기더라도 “짧은 sequence가 원인”, “형태 경계가 원인”, “Korean morphology-aware efficiency가 최초”라는 세 문장은 모두 데이터보다 강하다.

### 2.4 UTF-8 validity metric도 신규가 아니다

[Moon et al. (ICML 2026)](https://arxiv.org/abs/2606.14122)은 355M model·80B tokens·420 checkpoints에서 UTF-8 validity가 perplexity보다 늦게 안정화됨을 보이고, DFA partial credit과 semantic completion을 분리했다. JamoFlow의 generation validity는 이 연구보다 작고 목적도 다르다.

남는 역할은 patch policy별 structural harm 비교다. Failure taxonomy, UTF-8 hard mask, prefix metric을 방법 기여로 세지 않는다.

## 3. 선행 대비 claim matrix

| 가능한 문장 | 판정 | 이유 |
|---|---|---|
| linguistic boundary를 처음 patching에 사용 | 금지 | Dynamic Token Pooling, Learn Your Tokens, SpaceByte |
| auxiliary model 없는 dynamic grouping을 처음 제안 | 금지 | SpaceByte, Hierarchical BPE |
| Korean morphology-aware efficiency를 처음 연구 | 금지 | morphology+BPE, LeVoC, Morpheme Matters |
| morphology를 byte encoding에 처음 사용 | 금지 | MYTE |
| Jamo representation이 우월함을 보임 | 금지 | 현 Phase 3는 NFC UTF-8를 유지; 3-hot/Jamo BPE/SCRIPT가 직접 선행 |
| learned routing 일반보다 우월 | 금지 | E/EC는 한 router 구현일 뿐; H-Net·Scratchpad·Hierarchical BPE 미포괄 |
| autoregressive generation이 빨라짐 | 금지 | output step을 줄이지 않았고 full-prefix path만 검증 |
| same-rate Korean BLT에서 W가 C보다 낮은 BPB | Gate J 통과 시 허용 | 정확히 측정한 intervention |
| W가 router-inclusive Pareto frontier에 남음 | Gate K 통과 시 허용 | E/EC/S와 total cost 범위 안에서만 |
| Korean UTF-8에서 SpaceByte predicate가 syllable-like cadence를 만듦 | geometry 결과로 허용 | full SpaceByte architecture 성능 주장과 분리 |
| cheap boundary의 일반적 우월성 | publication-scale external controls 전 금지 | BPE/Hierarchical BPE와 다른 언어가 빠짐 |
| 경계 분포와 언어모델 분포의 최초 분리 | 금지 | Bolmo가 boundary predictor를 내부화했고 Haltiuk (2026)이 명시적 disentanglement 가설을 선점 |
| pretrained LLM을 처음 byte model로 전환 | 금지 | Bolmo가 1B/7B byteification을 보고 |

## 4. 현재 Phase 3의 causal-identification 빈틈

Primary F/C/W만으로 W가 C보다 좋아져도 **whitespace association 자체**가 원인이라고 완전히 식별되지 않는다. W는 동시에 다음을 바꾼다.

1. scheduled target 주변의 phase
2. patch-length variance와 tail
3. early-event frequency
4. 실제 whitespace와의 일치

Phase 2는 delayed `+2` grid와 rate-matched causal rolling-hash placebo를 실행해 이 대안을 상당 부분 줄였다. 하지만 compact 결과가 scale에서 재현되는지는 Phase 3 primary만으로 알 수 없다.

따라서 다음을 **Gate I 통과 시에만** Phase 3 mechanism replication으로 실행한다.

- D: `causal_grid_delayed2`, exact 86 patches
- P: calibration W의 nonfinal early-trigger fraction에 맞춘 prefix-causal rolling-hash event, exact 86 patches
- initial 3 seeds에서 W−D와 W−P를 보고
- Gate J까지 통과하면 confirmation 2 seeds도 같은 정의로 추가

이 조건은 W를 사후 구조 변경해 살리는 실험이 아니다. W는 그대로 두고 alternative explanation을 검사한다. Gate I가 실패하면 D/P를 method rescue로 실행하지 않는다.

Mechanism attribution의 권장 판정은 initial 3 seeds에서 각 contrast가 다음을 모두 만족하는 것이다.

- mean `W − control <= −0.002 BPB`
- 3 seeds 중 최소 2개 negative
- exact-rate/data/hash integrity 통과

Final 5 seeds를 실행했다면 다음으로 강화한다.

- mean `<= −0.003 BPB`
- 최소 4/5 negative
- crossed bootstrap 95% upper `< 0`
- 두 mechanism contrast의 단측 paired-seed Student-$t$ Holm-adjusted p-value `<= 0.05`

실패하면 W를 “whitespace-semantic method”가 아니라 “해당 geometry에서 잘 작동한 deterministic relocation heuristic”으로 축소한다. Gate J/K 자체의 수치는 바꾸지 않는다.

## 5. 논문 가치가 생기는 결과별 방향

### A. Gate J/K와 mechanism replication 모두 성공

Method paper의 씨앗이다. 그래도 제목과 abstract의 효율 주장은 Gate L 뒤에만 확정한다.

필수 다음 단계:

1. 50–100M main model, 최소 256M Korean train bytes
2. strongest raw-byte policies 3개
3. standard Korean byte-BPE token Transformer
4. 가능한 범위의 morphology+BPE와 Hierarchical BPE-style external control
5. CUDA teacher-forced와, 구현 가능할 때만 verified incremental latency
6. natural Korean downstream 또는 zero-shot evaluation 하나 이상

### B. Gate J 성공, K 또는 mechanism 실패

Efficiency/morphology method가 아니다. **Boundary placement observation**과 Korean UTF-8 geometry paper로 축소한다. 정확한 negative mechanism이 기여가 된다.

### C. Gate I 실패

Phase 2 small-model/domain artifact를 밝히는 결과다. 가장 가치 있는 framing은 다음이다.

> Linguistically plausible byte boundaries can produce stable compact-model gains that disappear under larger Korean web pretraining; rate control, UTF-8 geometry, and router-inclusive cost reveal why.

이 경우 BPE scale-up으로 W를 구제하지 않는다. 대신 F/C/W/S/E/EC geometry, domain transfer, NFD failure, private ecology, generation validity를 하나의 controlled failure study로 묶는다. Top-tier를 노리려면 Chinese/Japanese 같은 no-space 또는 다른 CJK control을 추가해 “Korean 특수성”과 “UTF-8 CJK 공통 효과”를 분리하는 편이 강하다.

## 6. Top-tier 최소 기준과 현재 상태

| 항목 | 현재 | top-tier 제출 전 요구 |
|---|---|---|
| reproducible public data | HPLT3 pinned sample | 충족, license 한계 명시 |
| same-graph causal ablation | F/C/W, S/E/EC | Phase 3 결과 필요 |
| mechanism identification | Phase 2만 완료 | positive Phase 3이면 D/P replication |
| model/data scale | 19.6M / 128M bytes | positive efficiency claim은 Gate L 필요 |
| tokenized external baseline | 없음 | standard BPE 필수, stronger cheap grouping 권장 |
| direct hardware evidence | MPS teacher-forced 예정 | CUDA 필요; AR latency와 분리 |
| semantic quality | BPB 중심 | positive method paper면 Korean task 최소 1개 |
| normalization/code-mix | protocol·runner 존재 | 결과 필요, primary 평균과 분리 |
| generation | structural validity only | semantic generation claim 금지 |
| absence/first claim | 검색 기반일 뿐 | “first” 회피, intervention을 정확히 기술 |

## 7. 최종 연구 방향

연구 중심을 다시 Jamo/FSM으로 넓히지 않는다. 현 논문의 가장 올바른 축은 다음이다.

> **Korean byte-latent models에서 boundary semantics, UTF-8 geometry, compression rate, learned-router cost를 분리하는 causal measurement study; positive evidence가 충분할 때만 whitespace-conditioned relocation을 경량 method로 승격한다.**

후속 논문 축은 별도로 둔다.

- NFD/Jamo unit을 보존하는 prefix-causal architecture
- morphology-aware prior와 learned chunking의 hybrid
- Fast-BLT류 multi-byte generation

이 세 축을 현재 W 결과에 붙이면 contribution이 많아지는 것이 아니라 식별이 무너진다.
