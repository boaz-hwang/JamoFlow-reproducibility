# 논문 주장–증거 매트릭스

> 갱신일: 2026-08-17
>
> 문서 번호: 88 (`docs/87-*` 중복을 해소하기 위해 파일명만 변경)
>
> 상태: **compact와 trained 188.6M actual complete; scale-amplification gate failed**
>
> 원칙: JamoFlow의 positive 연구 가치는 품질이 맞는 물리 모델 pair에서 실제
> batch-1 incremental generation이 빨라졌을 때만 성립한다.

## 1. 주장 우선순위

논문이 답해야 할 질문의 순서는 다음과 같다.

1. 새 sealed final test에서 candidate가 C86 대비 품질을 유지하는가?
2. 그 exact checkpoint pair가 실제 controlled와 free-running 생성에서 모두
   유의미하게 빠른가?
3. 차이가 whitespace-informed boundary relocation에 귀속되는가?
4. strongest raw-byte reference와 비교해도 품질–비용 frontier에 남는가?
5. 더 큰 규모와 BPE16K/32K, Korean downstream에서도 같은 결론이 재현되는가?

2번이 실패하면 3–5번의 일부 분석이 양성이어도 **추론 효율 기법의 positive
paper**라고 부르지 않는다. Analytical FLOPs, patch count, teacher-forced latency,
parameter 수 또는 tokenizer 길이는 2번의 대체 증거가 아니다.

## 2. 허가 매트릭스

| 주장 | 필요한 authoritative evidence | 최소 pass 조건 | 허용되는 문장 | 실패 시 의무 문장 |
|---|---|---|---|---|
| calibration-only policy selection | `selection-lock.json`, initial identity lock, 두 calibration replay | 고정 64→72 rule, exact 3×10 replay, final/test/latency input 없음 | “rate와 reference는 initial calibration NLL로 고정했다” | rate 없음이면 progression 종료; 다른 rate/margin으로 교체하지 않음 |
| C86 matched quality | final authorization, final evidence, independently replayed final-quality lock | paired-seed upper `< +0.010 BPB`, document-bootstrap upper `< +0.010`, 4/5 seed within margin, coverage `>=95%` | “candidate는 C86 대비 sealed-final noninferior였다” | primary timing 미실행; matched-quality claim 폐기 |
| primary actual efficiency | exact candidate–C86 timing authorization, five v5r3 session receipts, raw output/counter artifacts, v5r3 summary | CPU original semantic oracle + MPS safety/TV/greedy gate; controlled와 free E2E 각각 aggregate `>=10%`, CI lower `>0`, 5/5 session positive, 3/5 session `>=10%`, 4/5 seed positive, median seed `>=10%`, strict-output pass | “해당 Apple MPS workload에서 matched-quality end-to-end generation이 개선됐다” | analytical/teacher-forced 절감을 speedup으로 부르지 않음 |
| whitespace mechanism | final candidate–same-rate-C gate | mean `<=-0.002 BPB`, paired-seed upper `<0`, document upper `<0`, 4/5 negative | “이 graph에서 whitespace-informed relocation이 same-rate C보다 품질을 개선했다” | efficiency가 양성이어도 whitespace 원인 귀속 금지 |
| strongest raw replacement | calibration futility authorization, conditional broad confirmation, broad final NI, 별도 timing authorization | broad final upper bounds `<+0.010`, 4/5 within margin; speed를 주장하면 해당 broad pair timing도 별도 pass | “strongest locked raw reference와도 quality–cost frontier에 남았다” | `best raw-byte model replacement` 주장 금지; C86 좁은 estimand만 유지 |
| entropy-router total cost | router checkpoint/config/calibration identity, router-inclusive runtime counters, main/aux/total params와 memory | learned auxiliary train/score/runtime가 모두 분모·분자에 포함 | “detector-inclusive cost로 비교했다” | main-only 비용표로 learned/structural 우열 주장 금지 |
| random-weight schedule-scale sensitivity | sealed scale plan, 9 worker report/NPZ, canonical summary, full checkpoint correctness replay | 100M point `>=10%`, lower `>=8%`, 15/16 prompts, 3/3 positive sessions, 2/3 sessions `>=10%`, all-target evidence pass | “same-weight larger graph에서도 schedule 방향이 재현됐다” | trained quality/scaling claim 금지; 100M training 중단 |
| post-100M random-weight scale extension | separately sealed 188.6M--1.618B plan, 12 worker receipts/NPZ, canonical summary, deterministic correctness replay | 1.618B point `>=10%`, lower `>=8%`, 15/16 prompts, 3/3 sessions positive, 2/3 sessions `>=10%` | “Apple-MPS random graph에서 1.618B W72 schedule headroom이 10%를 넘었다” | trained quality, free-running 또는 scaling-law claim 금지 |
| trained 188.6M density-adjusted bridge | W72/C86 training summary+replay, presealed single-W80 plan, W80 training summary+bitwise replay, five session receipts와 actual summary | W80 mean 및 block-bootstrap upper `<=+0.010 BPB`; controlled/free actual CI lower `>0`, 15/16 prompts, 5/5 sessions; amplification은 compact point/lower 추가 gate | “severely undertrained one-seed 188.6M W80가 품질을 보존하고 2.5--2.9% actual 감소를 재현했다” | pure scale, multiseed large-model, 10% trained speedup 또는 statistically larger claim 금지 |
| publication-scale Korean efficiency | family-aware feasibility, sealed 50M/75M/100M scale, 3-seed candidate/raw/BPE16K/BPE32K, learning curves, downstream floor, scale actual timing | compact positive 이후에만 실행; matched quality와 actual timing을 scale에서도 통과 | “larger Korean LM setting과 tokenized controls에 재현됐다” | compact-only result로 명시; 규모 일반화 금지 |
| hardware generality | 다른 장치에서 동일 sealed workload replication | 장치별 독립 quality-authorized actual timing | 해당 장치 범위만 기술 | Apple MPS 한 대 결과를 CUDA/serving/general hardware로 일반화 금지 |

## 3. 논문 표와 artifact 연결

### Main Table 1 — sealed-final quality

- source: final-quality lock의 canonical gate와 다섯 seed NLL receipt
- rows: candidate−C86, candidate−selected-rate-C, 조건부 candidate−broad
- 필수 열: mean BPB difference, paired-seed interval, document interval,
  seed-count criterion, coverage, pass/fail
- 금지: historical 16MB screening NLL을 결합한 five-seed interval

### Main Table 2 — actual inference v5r3

- source: exact five tracked session receipts와 summary가 재검증한 heavy artifacts
- rows: controlled replay E2E, strict-valid free-running E2E
- 필수 열: aggregate reduction, crossed CI, positive/10%-session counts,
  positive-seed count, median-seed reduction, pass/fail
- 필수 보조: session별 효과, seed별 효과, role-order sensitivity, MAD/IQR
- 금지: repetition을 독립 표본으로 세기, TTFT/decode-only를 primary로 교체하기

### Main Table 3 — total cost와 broader comparators

- compact 단계: F/C/W/S/E/EC geometry, main/aux/total parameters,
  detector-inclusive FLOPs와 teacher-forced diagnostic
- publication scale을 열었을 때: candidate, locked raw reference, BPE16K, BPE32K의
  품질·실제 latency·descriptive memory·training/scoring cost
- 금지: patch 수 또는 token 수만으로 wall-clock 개선을 추론하기

## 4. 해석 분기

### A. 품질과 actual v5r3가 모두 통과

정확한 19.6M BLT, 128M training bytes, sealed Korean final stream, Apple hardware와
고정 workload 범위의 **within-family positive efficiency evidence**다. 이때만
publication-scale 및 BPE/Korean-downstream 확장을 연다. Compact 결과만으로 top-tier
scaling 또는 production-serving claim을 하지 않는다.

### B. 품질 통과, actual v5r3 실패

효율 기법으로는 음성이다. Boundary geometry 또는 품질 효과는 별도 분석 결과로
남길 수 있지만 제목·초록에서 inference efficiency를 제거한다. Profiler로 byte-step,
global trunk, selector, synchronization 병목을 나눈 뒤 multi-byte/speculative generation은
새 protocol의 독립 가설로만 연다.

### C. sealed-final 품질 실패

Primary timing을 열지 않는다. Rate, comparator, margin, split을 바꿔 같은 final test에
재도전하지 않는다. Historical compact signal이 새 문서/seed에서 재현되지 않았다는
negative result로 보고한다.

### D. compact positive, scale/BPE 실패

Compact within-family observation으로 범위를 축소한다. 더 큰 모델에서 좋아질 것이라는
외삽이나 한국어 일반 효율 claim을 하지 않는다.

## 5. 위협 모델과 재현성 문구

허용되는 가장 강한 provenance 표현은 다음이다.

> one prospectively Git-sealed analytic evaluation plus a deterministic
> checkpoint-forward verification replay

이는 public registry, trusted execution environment, cryptographic one-shot 또는 저자가
commit 전 ignored run을 삭제하지 않았다는 증명이 아니다. Historical 16MB test와
F/C/W confirmation seed는 이미 알려진 development evidence이며, post-selection
selected-rate compute-conversion C/W와 조건부 S/E/EC, 새 final quality, actual v5r3만
각각 명시한 prospective chain을 따른다.

## 6. 현재 허가된 결론

현재 calibration-only selection은 W72를 고정했다. 새 sealed final에서 W72−C86은
`+0.003682 BPB`였고 paired-seed/document upper가 각각 `+0.004780`, `+0.004612`라
고정 `+0.010` noninferiority를 5/5 seed에서 통과했다. W72−C72는
`-0.010781 BPB`, paired/document upper가 `-0.009868`, `-0.010010`이며 5/5 seed가
음수여서 same-rate boundary-placement 효과도 통과했다. 반면 calibration 최강인
SpaceByte 대비 W72는 `+0.103950 BPB`, margin 내 0/3으로 broad evaluation이
futility 종료됐다.

따라서 현재 허가되는 positive 결론은 **C86 matched quality**, **same-rate W72의
품질 개선**, 그리고 **W72의 재현 가능한 소폭 latency 감소**다. 다만 소폭 감소는
사전 고정한 primary efficiency threshold를 통과하지 못했으므로 positive efficiency
technique으로 승격되지 않는다. 최초 v5는 latency 공개 전 near-tie correctness failure로,
v5r1 첫 시도는 모델 실행 전 배터리 전환으로 제외됐고 두 번째 시도는 timing 전 한 MPS
저확률 logit의 shape-dependent tolerance 초과로 폐쇄됐다. CPU 원래 계약 10/10 및 사전
고정한 MPS safety/TV 감사 10/10 통과 후 v5r2 contract를 고정했다. 한 case dry run에서
`mps`/`mps:0` guard 오류를 발견해 latency 전 v5r2를 폐쇄했고 같은 contract의 v5r3를
봉인했다. 다섯 eligible v5r3 session에서 controlled E2E는 2.628%
([2.026%, 3.526%]), free-running E2E는 2.531% ([1.687%, 3.127%]) 감소했다.
두 mode 모두 5/5 session과 5/5 seed가 양수였지만 10% session은 0/5였으므로
`fail_matched_quality_actual_efficiency_v5r3`로 종료됐다. Total-efficiency positive,
publication-scale 또는 broad raw replacement 결론은 허가되지 않는다.

별도로 W72는 C86보다 data patches가 16.279% 적고, dummy를 포함한 Hugging Face
global positions가 16.092% 적다. 현재 구현의 dense-matmul 회계에서는
`6,152,810,496 → 5,640,155,136 FLOPs/512 bytes`, 즉 8.332% 감소다. 이 수치는
analytical workload claim이다. 실제 latency 감소는 별도 v5r3 측정값 2.5–2.6%로만
보고한다. 두 모델의 weight parameter 수와 MPS memory increment는 같았고, process RSS는
작고 seed별 방향이 섞여 memory-improvement claim을 허가하지 않는다.

이후 same-weight 50M/75M/100M random-graph sensitivity는 controlled E2E를 각각
3.572%, 3.758%, 4.460% 줄였고 모든 16 prompts와 3/3 sessions가 양수였다. 그러나
100M lower bound는 3.846%, 10% session은 0/3이어서 별도 고정 gate를 실패했다.
이 결과는 larger graph에서도 schedule 방향이 유지된다는 system evidence이지만 trained
quality 또는 publication-scale efficiency가 아니다. 100M training authorization은 false다.

별도 post-result plan의 188.6M/378.1M/790.4M/1.618B random graph extension은 controlled
감소 7.218%/7.060%/8.714%/10.217%를 측정했다. 1.618B interval은
[9.104%, 10.987%]이고 16/16 prompts, 3/3 sessions가 양수여서 그 systems-only gate는
통과했다. 이는 random-weight Apple-MPS schedule headroom이며 trained quality나 free-running
10%의 증거가 아니다.

Trained 188.6M W72는 C86보다 `+0.024200 BPB` 나빠 고정 `+0.010` margin을 실패했고
actual timing을 열지 않았다. 사전 고정한 단일 W80 rescue는 `+0.004058 BPB`, block-bootstrap
upper `+0.005114`로 통과했고 full checkpoint replay가 bitwise 동일했다. Five-session actual은
controlled 2.887% ([2.119%, 3.209%]), free-running 2.475% ([1.948%, 3.052%])였으며
각각 16/16 prompts와 5/5 sessions가 양수였다. 그러나 free point가 compact보다 작고 어느
lower도 compact point를 넘지 않아 scale amplification은 실패했다.

## 7. 역사적 후속 탐색

아래는 v5r3 이후 단계별로 허가됐던 역사적 탐색 경로다. 각 단계의 고정 gate에 따라
실행됐으며, 현재 terminal decision을 새로 열어 주는 authorization으로 읽지 않는다.

역사적으로 당시 W72를 그대로 scale-up하는 것은 허가되지 않았다. 먼저 exact checkpoint와
calibration-only case를 이용한 exploratory component profiler로 local byte path,
patch/global update, LM head, selector/host 비용을 분해한다. 병목이 예상대로 매-byte
local path라면 Korean UTF-8 구조를 이용한 multi-byte/block generation 또는 local
self-speculation을 새 candidate로 만든다. 새 구조가 compact matched-quality actual
inference에서 의미 있는 개선을 보인 뒤에만 50M/75M/100M, BPE16K/32K, downstream,
CUDA replication을 연다. v5r3 threshold를 사후 완화하지 않는다.

Profiler는 이 조건을 충족했다. 같은 checkpoint에서 W72 schedule은 C86보다 decode를
candidate weights에서 2.852%, reference weights에서 2.842% 줄였고 모든 seed가 양수였다.
W72가 제거한 네 decode boundary와 약 2.54ms boundary increment의 곱은 whole-decode
차이 약 10.1ms와 일치했다. 공통 약 2.36ms local-byte step은 양쪽 모두 127회 남았다.
이 결과는 exploratory mechanism evidence이며 v5r3 confirmatory 수치를 대체하지 않는다.

다음 candidate의 필수 comparator는 standard W72 AR과 same-cost generic byte-MTP다.
Hangul/scalar-aligned draft가 generic draft보다 accepted bytes와 실제 E2E를 추가 개선해야만
한국어 구조의 기여를 주장할 수 있다. Fast BLT, Medusa, generic MTP 및 MtPC 때문에
multi-byte/speculative decoding 자체는 novelty claim에서 제외한다.

Calibration-only perfect oracle에서는 complete byte의 86.389%가 precomposed Hangul에
속했고 Hangul-only grouping이 target-call count를 57.593% 줄일 수 있었다. 전체 scalar
grouping savings의 98.681%가 Hangul에서 왔다. 이는 learned-draft preflight의 opportunity
gate만 통과시킨다. 첫 Hangul lead byte 뒤 joint continuation entropy 6.305 bits와
conditional mutual information 2.409 bits는 규칙만으로 suffix가 결정되지 않음을 보인다.
따라서 이 수치들을 acceptance 또는 speed evidence로 사용하지 않는다.

후속 frozen-W72 acceptance preflight도 음성이었다. 약 40K parameter로 맞춘 네 head 중
generic independent UTF-8가 complete pair 24.379%, first continuation 42.373%로 가장
높았고, Hangul conditional은 17.702%였다. 모든 head가 사전 acceptance/cost gate를
실패했으며 Hangul head는 세 initialization 모두 strongest generic head보다 낮았다.
따라서 Jamo/composition draft claim은 금지한다.

다만 exact speculative verifier는 mismatch에서 correction byte를 함께 확정하므로
independent head의 진단적 expected committed bytes는
`2 + 0.423728 + 0.243794 = 2.667522`다. 이 값도 speedup이 아니다. Perfect-draft target
block kernel의 cache/logit exactness와 wall time을 먼저 측정하고, measured upper bound가
실패하면 multi-byte branch를 종료한다. 결과를 보고 head threshold/architecture를 다시
튜닝하지 않는다.

Static local thinning은 실제 E2E를 22.8--24.3% 줄였지만 calibration BPB를 +0.095601
악화시켜 matched-quality gate를 실패했다. 이어진 frozen conditional-local 2×2×2 screen도
모두 실패했고, 가장 덜 손상된 decoder-MLP/Hangul intervention이 +0.198832 BPB였다.
따라서 이 candidate 집합에는 actual-runtime 또는 efficiency claim이 허가되지 않는다.
Frozen failure는 trained conditional architecture 일반의 기각이 아니지만, 같은 calibration에서
skip rate나 margin을 사후 조정하지 않는다. 다음 representation audit은 UTF-8 byte steps를
reversible scalar decision으로 묶는 방향이 generic Unicode-scalar 및 BPE control보다 실제
cost frontier를 가질 수 있는지 판단하는 새 exploratory 단계다.

## 8. 최신 terminal decision

그 뒤의 vocabulary-transfer, 8K/16K trained actual, perfect-block upper bound,
generic retrieval과 EXAONE 7.8B transfer는 각각 해당 문서의 고정 gate로 수행됐다. 일부
controlled 또는 free estimand는 컸지만, matched-quality·같은-output·prompt stability·strong
baseline을 동시에 만족하는 새 Korean-specific inference method는 남지 않았다. 특히 public
EXAONE 3.5 7.8B의 generic retrieval은 ordinary AR보다 14.938% 느렸다. 이후 별도 계획의
random-weight W72 extension은 1.618B에서 10.217% systems headroom을 보였지만, trained
188.6M W72는 품질을 잃었다. 단일 W80 rescue는 품질과 작은 actual effect를 회복했으나
compact 대비 증폭을 통과하지 못했다.

따라서 현재 campaign에서 허가되는 최종 paper scope는 다음뿐이다.

- sealed-final five-seed W72/C72 boundary-placement quality effect
- W72/C86 matched-quality 19.6M Apple-MPS의 재현 가능한 2.5--2.6% 소폭 E2E 감소
- 동일-weight random 49.8M--1.618B의 3.6--10.2% systems headroom curve
- one-seed severely-undertrained 188.6M W80/C86의 quality-rescued 2.5--2.9% actual replication
- compact 대비 scale-amplification 실패
- patch/event·analytical cost와 actual wall time 사이의 gap 및 byte-local bottleneck
- 모든 fixed threshold 실패와 post-result correction의 공개

Positive 10% inference-efficiency technique, trained publication-scale scaling, strongest
raw/BPE replacement, Korean morphology superiority, CUDA/general-hardware 또는 production
claim은 허가되지 않는다. Multi-seed large-scale family와 Hugging Face efficient-model
upload도 실행하지 않는다. 현 시점 공개 대상은 code, aggregate evidence, protocol/audit
trail과 diagnostic paper draft다.
