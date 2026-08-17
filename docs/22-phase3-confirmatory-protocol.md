# Phase 3 사전등록: 한국어 byte patching의 중간 규모 확인 실험

> 작성일: 2026-08-10  
> 상태: **결과 확인 전 고정**  
> 선행 결과: [Phase 2 primary](./11-phase2-primary-results.md), [mechanism controls](./13-phase2b-control-results.md), [cost](./14-phase2-cost-results.md), [normalization](./17-normalization-results.md), [ecological check](./19-private-ecological-results.md), [generation validity](./21-generation-validity-results.md)  
> 결과 전 addenda: [OOD](./24-phase3-ood-addendum.md), [Unicode stress](./26-phase3-unicode-stress-addendum.md), [generation](./27-phase3-generation-addendum.md), [novelty audit](./28-novelty-and-identification-audit.md), [conditional mechanism replication](./29-phase3-mechanism-addendum.md), [HF BLT alignment audit](./30-hf-blt-alignment-audit.md), [cost input sampling](./32-phase3-cost-sampling-addendum.md), [statistical integrity](./33-phase3-inference-integrity-addendum.md), [OOD provenance](./38-phase3-ood-provenance-addendum.md), [mechanism provenance](./39-phase3-mechanism-provenance-addendum.md), [normalization provenance](./40-phase3-normalization-provenance-addendum.md), [generation provenance](./41-phase3-generation-provenance-addendum.md), [primary provenance](./42-phase3-primary-provenance-addendum.md), [cost provenance/stability](./43-phase3-cost-provenance-and-stability-addendum.md), [actual inference/compute conversion](./44-actual-inference-and-compute-conversion-protocol.md)
> 사후 무결성 교정: [document clustering](./52-document-cluster-inference-integrity-addendum.md), [selection/time-to-output](./53-selection-and-time-to-output-correction.md), [BPE prompt boundary](./54-bpe-prompt-boundary-runtime-addendum.md). 이 셋은 initial F/C/W·D/P 결과 뒤 추가되었으므로 결과 전 addendum으로 소급 표현하지 않는다.
> 목적: 1.25M mechanism pilot에서 나온 신호를 19.6M 모델과 독립 공개 한국어 학습 코퍼스에서 확인하고, 더 큰 학습으로 넘어갈 근거가 있는지 판정한다.

## 1. 먼저 고정하는 결론의 범위

Phase 2는 출판 결론이 아니다. 현재까지 지지된 것은 다음의 좁은 관찰뿐이다.

1. 256-byte 문맥과 정확히 43개 patch를 쓰는 compact BLT에서, 이미 관측된 whitespace를 기준으로 고정 grid를 최대 2 bytes 이동한 정책은 fixed-byte 및 generic causal codepoint grid보다 Korean Wikipedia BPB가 낮았다.
2. punctuation을 추가해도 whitespace-only보다 좋아지지 않았다. 따라서 이를 형태론 또는 광범위한 `eojeol` 이해의 효과라고 부를 수 없다.
3. learned entropy threshold는 같은 compact setting에서 더 좋은 품질을 주지 못했고, variable patch count의 batch padding과 auxiliary router 비용을 포함하면 더 비쌌다.
4. NFD/Jamo robustness 가설은 실패했다. Jamo-aware architecture는 현 방법의 기여가 아니다.
5. private Markdown 표본에서는 whitespace policy의 fixed-byte 대비 비열등성만 확인됐고, codepoint policy는 public Korean 결과와 반대로 악화됐다.
6. unconstrained byte generation의 절대 UTF-8 validity는 낮았다. 정책 특이적 악화 중단선만 통과했을 뿐 생성 모델로서 충분하다는 증거가 아니다.

따라서 Phase 3가 검증할 명제는 다음과 같다.

> **Korean-dominant UTF-8 text에서 고정된 global-compute cadence를 유지하면서, 이미 관측된 whitespace 직후로 patch boundary를 제한적으로 이동하면 generic byte/codepoint cadence보다 재현 가능한 modeling-quality 이득을 얻을 수 있는가? 이 parameter-free 이득은 별도 learned router의 총비용을 지불할 만큼 강한 learned entropy baseline과 비교해도 Pareto frontier에 남는가?**

다음 주장은 이번 단계에서 하지 않는다.

- 한글 조합 FSM이 semantic neural prediction을 대체한다.
- whitespace가 한국어 형태소 경계를 정확히 분석한다.
- `JamoFlow`가 Jamo-level representation 또는 multi-Jamo decoding을 구현한다.
- teacher-forced MPS 처리량이 production autoregressive latency를 대표한다.
- 19.6M 모델의 결과가 1B 이상으로 자동 확장된다.
- 같은 BLT graph에 SpaceByte boundary rule을 넣은 조건이 완전한 SpaceByte architecture 재현이다.

## 2. Phase 2 증거에 대한 비판적 종합

### 2.1 살아남은 신호

Phase 2의 가장 일관된 효과는 `causal_whitespace_grid`였다.

- delimiter-aware C2와 whitespace-only의 차이는 사실상 0이었다.
- whitespace-only는 delayed-grid보다 좋아 boundary 위치 이상의 local event signal을 보였다.
- hash placebo보다 좋았으므로 단순 irregularity만으로 설명되지는 않았다.
- exact patch count가 같고, aligned packing에서도 codepoint 효과 방향이 뒤집히지 않았다.

이는 scale-up할 가치가 있는 신호다. 다만 “한국어 형태론 prior”가 아니라 **rate-controlled whitespace boundary preference**라는 더 좁은 기전으로 해석한다.

### 2.2 아직 제거되지 않은 대안 설명

Phase 3는 다음 설명을 구분해야 한다.

1. **일반 whitespace 효과:** 한국어 고유 현상이 아니라 대부분의 spaced language에 적용되는 신호일 수 있다.
2. **UTF-8 geometry 효과:** fixed byte가 codepoint 내부를 자르는 불이익을 whitespace가 우연히 줄였을 수 있다.
3. **small-model optimization 효과:** 1.25M, one-pass 조건에서만 학습이 쉬워졌을 수 있다.
4. **Wikipedia/domain 효과:** 정제 문장과 사설 Markdown에서 방향과 크기가 달랐다.
5. **packing 효과:** 256-byte arbitrary chunks와 absolute grid의 위상 관계가 효과를 만들 수 있다.
6. **architecture-specific 효과:** Hugging Face BLT의 encoder/global/decoder shift에서만 나타날 수 있다.

따라서 fixed-byte와 whitespace만 비교해서는 충분하지 않다. generic codepoint, authentic spacelike cadence, learned entropy를 함께 둔다.

### 2.3 SpaceByte를 특히 조심해야 하는 이유

[SpaceByte 논문](https://arxiv.org/abs/2404.14408)과 [공식 구현](https://github.com/kjslag/spacebyte/tree/321111315c92bce0bc2f9f1630cb0bc82b897c57)은 다음 byte를 `spacelike`로 분류한다.

```text
b < '0'
or '9' < b < 'A'
or 'Z' < b < 'a'
or 'z' < b < 0x80
or b >= 0xC0
```

연속한 spacelike position은 첫 position만 남긴다. NFC 한글 음절의 UTF-8 첫 byte는 매번 `>= 0xC0`이므로 Korean-dominant text에서는 거의 매 음절에 global position이 생겨 약 3 bytes/global-event가 된다. 이는 오류가 아니라 SpaceByte가 multibyte 문자를 처리하는 원래 설계지만, 약 6 bytes/patch인 이번 fixed/whitespace 조건보다 global compute가 훨씬 많다.

그러므로 두 비교를 분리한다.

- 같은 rate에서의 boundary-quality 비교
- 각 방법이 실제로 만든 rate와 총비용을 포함한 Pareto 비교

SpaceByte 규칙을 약 6 bytes/patch로 강제로 희석한 조건을 “SpaceByte”라고 부르지 않는다.

## 3. 연구 질문과 사전 가설

### RQ1 — same-rate quality

동일하게 sequence당 86개 data patch를 쓰는 조건에서 whitespace grid가 fixed byte와 generic causal codepoint grid보다 낮은 test BPB를 내는가?

- **H1a:** mean `W − C <= −0.003 BPB`
- **H1b:** mean `W − F <= −0.003 BPB`

여기서 `F`는 fixed byte, `C`는 causal codepoint grid, `W`는 causal whitespace grid다. H1a가 한국어 whitespace의 추가 정보를 검사하는 더 중요한 비교다.

### RQ2 — learned router가 값을 하는가

별도 causal entropy router를 포함한 learned 정책과 비교할 때 W가 품질 비열등성과 총비용 우위를 동시에 만족하는가?

- **H2-quality:** `W − min(E, EC) <= +0.010 BPB`
- **H2-cost:** router를 포함한 analytical FLOPs와 직접 측정 teacher-forced latency가 각각 10% 이상 낮다.

`E`는 all-byte entropy threshold, `EC`는 codepoint-restricted entropy threshold다. 이 가설은 W가 learned routing보다 반드시 더 정확하다고 주장하지 않는다.

### RQ3 — authentic spacelike cadence의 Korean geometry

공식 SpaceByte의 spacelike event를 causal BLT patch boundary로 옮겼을 때 realized bytes/patch, UTF-8 내부 boundary, 품질, padding-aware cost가 어떤 Pareto point를 만드는가?

이 조건은 patch rate가 다르므로 BPB 단독 우승자를 정하는 primary test에 넣지 않는다.

### RQ4 — domain과 Unicode robustness

W의 신호가 HPLT3 held-out에만 존재하는지, Korean Wikipedia와 mixed Markdown에서 심각한 regression 없이 유지되는지 검사한다. NFD는 별도 stress test이며 natural-corpus 평균과 합치지 않는다.

## 4. 공개 학습 데이터

### 4.1 source pin

새 학습 데이터는 [HPLT 3.0](https://huggingface.co/datasets/HPLT/HPLT3.0)의 `sorted/kor_Hang` release를 사용한다.

- language map: `https://data.hplt-project.org/three/sorted/kor_Hang.map`
- selected source shard: `10_1.jsonl.zst`
- source URL: `https://data.hplt-project.org/three/sorted/kor_Hang/10_1.jsonl.zst`
- 확인된 compressed content length: `1,862,302,013 bytes`
- 확인일: 2026-08-10
- source record의 본문 field: `text`

HPLT3 dataset card는 배포 패키지를 CC0로 표시하지만 원 웹문서의 권리가 모두 소멸했다고 보장하지 않는다. 원문과 파생 corpus는 저장소에 커밋하거나 재배포하지 않는다. URL, source metadata, 다운로드 후 SHA-256, 필터 통계, 처리 결과 hash만 기록한다.

### 4.2 deterministic selection

전체 compressed shard를 끝까지 순차 scan한다. source 순서의 앞부분만 잘라 쓰지 않는다.

1. JSONL과 UTF-8을 strict하게 파싱한다.
2. 비어 있는 `text`와 UTF-8 byte 길이가 256 미만 또는 262,144 초과인 문서는 제외한다.
3. exact text SHA-256으로 deduplicate한다.
4. 기존 `stable_record_id(text_bytes)`와 `split_for_record`를 사용해 80/10/10 train/calibration/test를 정한다.
5. split마다 `SHA-256("JamoFlow-Phase3-v1\0" || text_hash)`가 작은 문서를 우선하는 deterministic bottom-hash sample을 만든다.
6. 최종 문서는 selection hash 순서로 저장한다. 각 split에 필요한 byte quota와 record separator 여유가 확보돼야 한다.

이 절차는 source shard 전체에서 재현 가능한 표본을 만들고, source ordering과 split 간 누출을 줄인다. HPLT의 upstream dedup 표시는 자체 exact-dedup 검사를 대체하지 않는다.

### 4.3 byte budgets

Phase 3a의 `build_neural_stream` limit은 다음과 같다.

| Split | Bytes |
|---|---:|
| train | 128,000,000 |
| calibration | 8,000,000 |
| test | 16,000,000 |

동일 raw UTF-8 byte stream, record split, sequence packing을 모든 policy가 공유한다. NFC normalization을 강제로 적용하지 않고 source form을 보존한다. normalization 분포는 별도로 보고한다.

### 4.4 public OOD와 private diagnostic

- public domain-transfer: 기존 pinned Leipzig Korean Wikipedia 2021 corpus의 held-out split
- Unicode stress: HPLT3 test를 NFC와 NFD로 각각 변환한 paired evaluation
- private ecology: `../assist-creator/vault` Markdown hash-test split, aggregate only

Leipzig 문장이 HPLT web crawl과 의미상 겹치지 않는다고 보장할 수 없으므로 “contamination-free external benchmark”라고 부르지 않는다. private 결과는 convenience-sample diagnostic이며 원문, path, record hash를 추적하지 않는다.

## 5. 모델과 최적화

### 5.1 main BLT

동일한 Hugging Face `BltForCausalLM` graph를 모든 policy에 사용한다.

| Item | Value |
|---|---:|
| raw-byte vocabulary | 256 |
| sequence length | 512 bytes |
| target fixed-rate patches | 86 |
| nominal bytes/patch | 5.953 |
| local width / layers | 192 / encoder 2, decoder 2 |
| global width / layers | 384 / 8 |
| local/global heads | 6 / 8 |
| local/global FFN | 576 / 1,152 |
| hash group / vocabulary | 3 / 8,192 |
| cross-attention k | 2 |
| parameters | **19,596,096** |

이 모델은 BLT 논문의 scale 재현이 아니라 Phase 2보다 15.7배 큰 controlled mechanism model이다.

### 5.2 entropy router

`E`와 `EC`는 다음 causal `BltPatcher`를 seed별로 따로 학습한다.

| Item | Value |
|---|---:|
| width / layers / heads | 192 / 4 / 6 |
| FFN | 576 |
| parameters | **2,016,960** |

Router는 main model과 같은 train bytes를 one pass로 학습한다. Calibration split에서 scalar entropy threshold만 맞추며 test label이나 full future sequence를 boundary decision에 사용하지 않는다. Router의 parameter memory, FLOPs, 전체-byte scoring latency를 총비용에 포함한다.

### 5.3 optimization fairness

- seeds: 1,729 / 2,718 / 31,415
- conditional confirmation seeds: 57,721 / 65,537
- batch size: 32 main, 64 router
- optimizer: AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`
- learning rate: cosine `3e-4` to `3e-5`
- warmup: 500 steps
- gradient clip: 1.0
- precision: float32
- passes: exactly one over selected train sequences
- dropout: 0

같은 seed 안에서 main model initial state hash와 shuffled training order hash는 모든 policy가 같아야 한다. Policy마다 data bytes, optimizer steps, batch order를 바꾸지 않는다. Variable-rate policy가 느리다는 이유로 더 적게 학습하지 않는다.

## 6. 비교 정책

### F — `fixed_byte_6`

512-byte window의 `0, 6, 12, ..., 510`에서 시작하는 86개 patch다. UTF-8 codepoint 내부 boundary를 허용한다.

### C — `causal_codepoint_grid`

86개 absolute target 각각에서, target 이후 처음 완전히 관측된 UTF-8 codepoint boundary를 선택한다. Exact rate이고 prefix-invariant다.

### W — `causal_whitespace_grid`

Phase 2의 사전 고정 알고리즘을 그대로 사용한다.

- 86개 absolute target
- 이미 관측된 Unicode whitespace 직후가 target의 2-byte 이전부터 나타나면 early event
- 없으면 target 2 bytes 이후 첫 codepoint boundary에서 deadline
- 마지막 target은 codepoint boundary에서 마감
- minimum patch length 2

Whitespace/punctuation 범위를 결과를 보고 바꾸지 않는다. Punctuation은 포함하지 않는다.

### S — `spacebyte_spacelike`

공식 SpaceByte commit `321111315c92bce0bc2f9f1630cb0bc82b897c57`의 byte predicate와 consecutive suppression을 그대로 적용한다. SpaceByte global position `t`에서 현재 byte를 본 뒤 다음 local prediction에 정보를 주는 causal 의미를 BLT에 맞추기 위해, BLT boundary는 flagged byte 직후 prefix position `t+1`에 둔다. Boundary 0은 항상 포함한다.

이는 **SpaceByte-compatible patchifier를 동일 BLT graph에 넣은 정책**이다. SpaceByte의 initial/global/final block architecture 자체를 재현했다고 표기하지 않는다.

### E — `entropy_threshold_full`

별도 causal router의 next-byte entropy가 calibration threshold 이상이거나 patch가 24 bytes에 도달하면 boundary를 낸다. 모든 byte position이 후보이고 calibration mean은 86 patches/window, 허용 오차 0.1이다.

### EC — `entropy_threshold_codepoint`

E와 같지만 완전히 관측된 UTF-8 codepoint boundary에서만 threshold와 24-byte cap을 적용한다. Calibration mean과 허용 오차는 같다.

## 7. 무결성 검증

학습 전에 자동 test와 smoke run이 다음을 모두 통과해야 한다.

1. F/C/W는 모든 512-byte row에 정확히 86개 positive data patch를 가진다.
2. 모든 policy의 positive patch lengths 합은 512다.
3. C/W/EC의 noninitial boundary는 UTF-8 codepoint 내부에 오지 않는다.
4. W는 full sequence와 모든 prefix에서 이미 내린 boundary가 동일하다.
5. S predicate는 공식 SpaceByte 구현의 ASCII 경계값, UTF-8 lead/continuation, consecutive suppression 예제와 일치한다.
6. E/EC threshold는 calibration 외 split에서 재조정하지 않는다.
7. 같은 seed의 F/C/W/S/E/EC initialization hash와 training-order hash가 같다.
8. 데이터 split 간 text hash overlap은 0이다.
9. raw text, document ID, URL, prompt, generation sample은 tracked result에 들어가지 않는다.

## 8. 평가와 통계

### 8.1 primary quality endpoint

Primary endpoint는 HPLT3 test의 bits per byte다. Seed별 paired contrast와 다음 두 interval을 보고한다.

- seed-level paired t 95% interval
- seed와 공통 test sequence를 교차 resample하는 paired bootstrap 95% interval

H1a와 H1b를 primary family로 둔다. 다중비교의 정식 판정은 단측 paired-seed Student-$t$ p-value에 Holm correction을 적용한다. Crossed bootstrap 95% interval은 공통 test sample 민감도 조건으로 별도 사용한다. [추론 교정 addendum](./35-phase3-primary-family-inference-correction.md)에 변경 이유와 결과 열람 시점을 기록했다. Threshold effect, seed sign, paired-t interval, crossed-bootstrap interval을 모두 공개하고 어느 하나만 골라 결론내리지 않는다.

### 8.2 strata

다음 test sequence strata의 BPB를 사전 보고한다.

- Hangul byte fraction: `<25%`, `25–75%`, `>75%`
- ASCII/Latin mixed 여부
- whitespace rate tercile
- newline 포함 여부
- sequence start가 UTF-8 codepoint 내부인지

Stratum은 설명용이며 새로운 primary hypothesis를 만들지 않는다.

### 8.3 cost

다음을 별도로 보고한다.

1. realized patches/window와 bytes/patch
2. batch별 maximum width 및 padding waste
3. local/global/cross-attention dense-matmul analytical FLOPs
4. router analytical FLOPs와 parameters
5. selector-only CPU time
6. end-to-end teacher-forced bytes/s: batch 1/8/32/64
7. resident model/router parameter bytes와 MPS allocated memory

Direct timing은 warmup 뒤 최소 30회, 중앙값과 p95를 보고한다. MPS synchronization을 timing 안에 둔다. Input sampling은 [32 addendum](./32-phase3-cost-sampling-addendum.md)에 따라 8개의 disjoint seeded timing batch를 모든 policy가 공통·균형 순서로 사용한다.

### 8.4 generation

Phase 2와 같은 strict protocol을 256-byte continuation으로 반복한다.

- greedy
- temperature 0.8 / top-p 0.95
- strict UTF-8 sequence validity
- U+FFFD-free rate
- conjoining-Jamo transition validity
- bytes/valid-codepoint

Prompt는 HPLT3 test에서 hash로 고르고 저장하지 않는다. UTF-8 DFA hard mask는 architecture control로만 보고 quality method와 합치지 않는다.

HF BLT cache에서 open patch를 올바르게 갱신하는 incremental decoder가 구현·검증되기 전에는 generation wall-clock을 속도 결과로 보고하지 않는다.

### 8.5 OOD guard

각 OOD set에서 `W − C`와 `W − F`를 보고한다. 어느 주요 Korean natural-text set에서든 mean regression이 `+0.020 BPB`를 넘으면 general Korean superiority 주장을 중단하고 domain-conditional result로 축소한다.

NFD는 이 guard에 넣지 않는다. Phase 2에서 현재 architecture가 NFD full-unit preservation을 할 수 없음을 이미 확인했기 때문이다.

## 9. 단계별 gate

### Gate I — 3-seed confirmation

먼저 seeds 1,729 / 2,718 / 31,415를 완료한다. 다음을 모두 만족하면 두 confirmation seed를 추가한다.

- mean `W − C <= −0.002 BPB`
- 3 seeds 중 최소 2개가 negative
- HPLT3 이외 Korean natural-text set에서 regression `> +0.020 BPB` 없음
- initialization/order/rate integrity 모두 통과

실패하면 W method scale-up을 중단한다. Threshold를 바꾸거나 punctuation/morphology feature를 사후 추가하지 않고, 결과를 Korean UTF-8 patching의 empirical failure analysis로 전환한다.

### Gate J — 5-seed method evidence

Confirmation seeds를 포함해 다음을 만족해야 positive method evidence로 판정한다.

- H1a mean `W − C <= −0.003 BPB`
- H1b mean `W − F <= −0.003 BPB`
- 각 contrast에서 5 seeds 중 최소 4개 negative
- crossed bootstrap 95% upper bound `< 0`
- 두 contrast의 단측 paired-seed Student-$t$ Holm-adjusted p-value `<= 0.05`
- OOD guard 통과

### Gate K — Pareto evidence

Gate J와 함께 다음을 만족해야 더 큰 model scale로 간다.

- H2-quality 통과
- router 포함 W의 analytical FLOPs 10% 이상 절감
- batch-1 또는 batch-8 teacher-forced latency 10% 이상 절감
- S를 포함한 quality-cost plot에서 W가 dominated되지 않음

Gate J는 통과하지만 K가 실패하면 boundary-quality observation만 남기고 efficiency method 주장을 하지 않는다.

### Gate L — publication scale

Gate J/K가 통과하면 동일 protocol을 최소 다음 한 단계로 확장한다.

- 50–100M main parameters
- 최소 256M Korean training bytes
- 가장 강한 3 policies만 선택
- 가능하면 CUDA incremental latency

이 확장을 완료하기 전에는 top-tier scale claim을 쓰지 않는다. 현재 하드웨어에 CUDA가 없다는 사실은 결과 누락을 정당화하지 않으며, MPS 결과와 CUDA 결과를 섞어서도 안 된다.

## 10. baseline 포함·제외 판단

### H-Net

[H-Net](https://arxiv.org/abs/2507.07955)과 [공식 MIT 구현](https://github.com/goombalab/hnet)은 중요한 learned dynamic chunking baseline이다. 그러나 architecture가 BLT patch policy와 동일하지 않아 Phase 3a same-graph causal contrast에 직접 섞으면 boundary와 backbone 효과가 함께 바뀐다.

Phase 3a에서는 E/EC를 same-graph learned baseline으로 사용한다. H-Net은 publication-scale cross-architecture baseline으로 별도 실행하거나, 실행하지 못하면 명시적 제한으로 남긴다. “learned routing 일반을 이겼다”고 확대 해석하지 않는다.

### Scratchpad Patching

[Scratchpad Patching](https://arxiv.org/abs/2605.09630)은 boundary lag와 integrated auxiliary head를 다루므로 매우 가깝지만, 2026-08-10 현재 확인된 공식 공개 구현이 없다. Transient trunk state까지 재현하지 않은 자체 구현을 논문 이름의 baseline으로 부르지 않는다. 결과 해석에서는 fixed/SpaceByte boundary가 scratchpad와 결합될 때 격차가 줄 수 있음을 반드시 논의한다.

### full SpaceByte

S는 공식 predicate를 이식한 boundary-policy control이지 full SpaceByte다. Publication-scale에서 full architecture를 실행할 경우 같은 parameters뿐 아니라 같은 training FLOPs와 context bytes를 맞춘 별도 cross-architecture table로 보고한다.

## 11. 예상 가능한 결과별 올바른 논문 방향

### A. W가 same-rate와 Pareto를 모두 통과

방법 논문 후보가 된다. 기여는 “한국어 문법 엔진”이 아니라 다음으로 제한한다.

> Korean UTF-8 geometry에서 authentic spacelike cadence의 compute inflation을 분석하고, 고정 global budget 안에서 observed whitespace로 경계를 causal하게 스냅하는 저비용 policy가 learned router와 경쟁하는 품질-비용 점을 만든다.

### B. same-rate 품질만 통과하고 Pareto 실패

Whitespace boundary의 representation/optimization 효과에 대한 empirical paper로 축소한다. 효율 향상을 제목이나 초록에 쓰지 않는다.

### C. HPLT3에서 재현 실패

Phase 2의 small-model/domain artifact를 밝힌 negative result다. 이 경우 새 feature를 붙여 method를 살리지 않는다. SpaceByte/UTF-8/whitespace/entropy의 Korean geometry, normalization, code mixing, router total cost를 묶은 benchmark·failure taxonomy가 논문의 중심이 된다.

### D. in-domain만 통과하고 mixed/OOD에서 실패

`Korean natural prose under NFC and high Hangul density`라는 조건부 결과로 제한한다. 범용 Korean 또는 multilingual method라고 부르지 않는다.

## 12. 재현 산출물

추적한다.

- 이 사전등록 문서와 이후 amendment
- HPLT source manifest, URL, size, ETag/Last-Modified, local SHA-256
- selection/filter/split aggregate와 processed-file SHA-256
- model/router config와 parameter count
- seed별 scalar logs, initialization/order/patch hash
- aggregate per-stratum loss와 interval
- cost/timing raw scalar samples와 summarization script
- figure/table 생성 script
- 결과에 맞춰 범위를 제한한 paper draft

추적하지 않는다.

- HPLT/Leipzig/private 원문
- processed JSONL
- checkpoints와 per-sequence text
- source URL/ID의 per-record 목록
- prompt와 generated sample
- private vault의 path, content, record hash

## 13. 사전등록 이후 변경 원칙

결과를 보기 전에 발견된 implementation bug, source 변경, OOM 또는 명백한 infeasibility는 amendment로 문서화할 수 있다. Amendment에는 날짜, 원인, 영향을 받는 hypothesis, 변경 전후 값을 남긴다.

Primary seed 결과를 본 뒤에는 다음을 바꾸지 않는다.

- W의 window와 delimiter 정의
- model width/depth
- train/calibration/test byte budgets
- primary contrasts와 effect thresholds
- Gate I/J/K 기준
- 유리한 domain만 남기는 평가 범위

이 문서의 역할은 Phase 2에서 나온 작은 positive signal을 다시 결론으로 선취하지 못하게 하는 것이다.
