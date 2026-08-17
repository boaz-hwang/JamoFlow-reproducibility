# Phase 2 preregistration: causal Korean-aware patching

> 작성일: 2026-08-10  
> 상태: **결과 확인 전 고정**  
> 선행 결과: [Phase 1 compact BLT results](./09-phase1-neural-results.md)  
> 주 언어: **한국어**  
> 대조 언어: Phase 1의 중국어·영어 결과를 재사용하고 구조 통계만 보조 갱신

## 0. 연구 위치와 신규성 경계

Phase 1은 compact BLT에서 다음을 보였다.

- `fixed_codepoint − fixed_byte`: Korean −0.00667 BPB, paired-t 95% CI [−0.01115, −0.00219]
- 이 효과는 영어보다 −0.00623 BPB, 중국어보다 −0.00398 BPB 더 컸다.
- learned entropy router는 한국어 경계의 97.02%를 UTF-8 codepoint 내부에 놓았다.
- 그 내부 경계를 금지해도 Korean `entropy_codepoint − entropy_full`은 +0.00440 BPB였고 명백한 손상은 관측되지 않았다.
- 별도 dense router는 compact end-to-end analytical FLOPs의 27.27%였다.

그러나 **character-preserving patch 자체는 신규 아이디어로 주장할 수 없다**. 2026년 공개된 [Parameter-efficient Adaptation of Tokenizer-free BLT](https://kanavalau.com/projects/multilingual_blt/) 프로젝트는 multi-byte UTF-8 문자를 쪼개지 않는 것이 여러 언어에서 자주 도움이 됐다고 이미 기술한다. 정식 논문·재현 가능한 상세 결과는 아니지만 선취 사례로 취급한다.

더 강한 위협은 [Scratchpad Patching](https://arxiv.org/abs/2605.09630)이다. 이 연구는 patch lag를 내부 scratchpad compute로 완화하면 fixed·SpaceByte·entropy·H-Net 계열 간 격차가 좁아진다고 보고한다. 따라서 단순 fixed boundary의 소규모 BPB 이득만으로 architecture contribution을 주장할 수 없다.

[Beyond Perplexity: UTF-8 Validity in Byte-aware Language Models](https://arxiv.org/abs/2606.14122)는 355M 모델과 80B tokens에서 UTF-8 validity가 perplexity보다 늦게 수렴하며 별도 평가 대상임을 보였다. Phase 2는 validity를 새로 발견했다고 주장하지 않고, patch boundary와 validity/normalization robustness의 관계를 측정한다.

Phase 2의 잠정적 신규성은 다음 조합에 한정한다.

1. entropy boundary를 **encoding state별로 분해**해 한국어 UTF-8 continuation uncertainty가 patch allocation을 지배하는 정도를 정량화
2. offline top-k가 아닌 **prefix-causal** 정책을 동일 realized patch rate에서 비교
3. generic codepoint prior와 Korean syllable/eojeol prior를 분리하는 ablation
4. router와 Python/device transfer를 포함한 end-to-end cost
5. NFC/NFD, 현대 자모, 호환 자모, 한영 혼용에서 Korean-specific robustness 분석

## 1. Research questions

> **RQ3 — Causal replication:** 미래 후보를 보는 Phase 1 `fixed_codepoint`의 Korean 이득이 prefix-causal codepoint grid에서도 재현되는가?

> **RQ4 — Entropy necessity:** 한국어 calibration에서 같은 평균 patch rate로 맞춘 causal entropy threshold가 parameter-free causal codepoint grid보다 유의미하게 나은가?

> **RQ5 — Korean structure beyond Unicode:** codepoint 정렬 위에 eojeol/whitespace 우선순위를 추가하면 동일 patch 수에서 추가 이득이 있는가?

> **RQ6 — Encoding robustness:** NFC에서 학습한 모델이 NFD Hangul Jamo, compatibility jamo, emoji, 숫자·Latin 혼용을 만날 때 patch policy별 quality와 UTF-8 validity가 어떻게 달라지는가?

> **RQ7 — Artifact and noise:** Phase 1 효과가 chunk packing, MPS nondeterminism, 또는 작은 router의 undertraining만으로 설명되는가?

## 2. Scope

Phase 2는 여전히 mechanism study다. 다음을 달성해야 더 큰 confirmatory model로 이동한다.

- causal policy에서 Phase 1 방향 재현
- 한국어 특화 prior가 generic Unicode prior를 넘어서는지 판정
- streaming boundary construction과 teacher-forced total-cost Pareto 확인
- scale-up할 정책을 하나로 축소

이 단계에서도 1B scaling, production CUDA latency, multi-byte parallel generation, downstream instruction following은 주장하지 않는다.

## 3. Data

### 3.1 Primary Korean corpus

Phase 1과 동일한 Leipzig Korean Wikipedia 2021 corpus와 content-hash split을 사용한다. 결과를 본 뒤 test record를 바꾸지 않는다.

| Split | Byte cap | Fixed-length bytes after truncation |
|---|---:|---:|
| train | 11,000,000 | 10,999,808 |
| calibration | 1,000,000 | 999,936 |
| test | 1,000,000 | 999,936 |

train cap은 현재 hash-train partition의 11,474,000 available bytes보다 작다. 모든 정책은 동일 record prefix와 동일 bytes를 본다.

### 3.2 Informal Korean external set

NSMC의 official 50K test split을 out-of-domain diagnostic으로 사용한다. [공개 repository](https://github.com/e9t/nsmc)는 데이터 설명과 split을 제공하지만 명시적 license 문구가 비어 있다. 따라서 다음 원칙을 적용한다.

- 원문·ID를 Git 또는 논문 부록에 재배포하지 않는다.
- source URL, download hash, parsing code, aggregate만 기록한다.
- 명시적 재배포 허가를 확인하지 못하면 confirmatory primary endpoint나 training data로 사용하지 않는다.
- 접근 또는 법적 상태가 불명확하면 이 항목은 `unavailable`로 보고하고 gate 판정에서 제외한다.

사용자의 `../assist-creator/vault` Markdown 문서는 read-only private ecological audit에만 사용한다. 내용·경로·문서별 수치는 공개 결과에 넣지 않으며 논문의 primary evidence로 사용하지 않는다.

### 3.3 Korean test strata

원 test sequence를 결과 확인 전에 결정적인 Unicode 규칙으로 분류한다.

- Hangul-heavy: letter codepoint 중 precomposed Hangul 비율 ≥80%
- Latin-mixed: ASCII Latin letter가 하나 이상
- digit-mixed: ASCII digit가 하나 이상
- Hanja-mixed: CJK ideograph가 하나 이상
- compatibility-jamo-present
- modern-jamo-present
- whitespace-density quartile
- sequence starts at/inside a UTF-8 codepoint

한 sequence는 여러 stratum에 속할 수 있다. stratum 결과는 전체 primary test를 대체하지 않는다.

## 4. Model and paired training

Phase 1의 compact HF BLT graph를 유지한다.

- main parameters: 1,251,136
- sequence: 256 bytes
- local encoder: 1×64
- global: 4×128
- local decoder: 2×64
- cross-attention `k=2`
- float32 MPS
- seeds: 1,729 / 2,718 / 31,415 / 57,721 / 65,537

단 global position limit은 variable threshold policy의 padded patch dimension을 수용하도록 늘릴 수 있다. 이는 learned parameter 수를 바꾸지 않는다.

정책별로 다음을 공유한다.

- 동일 seed의 initial state
- 동일 Korean train sequence order
- 동일 optimizer와 one-pass schedule
- 동일 train/calibration/test bytes
- entropy 두 정책의 동일 router checkpoint와 entropy scores

Korean-only train examples는 42,968개이고 batch size 32를 사용한다. optimizer는 Phase 1과 동일하다. 정책별 early stopping과 test-based selection은 없다.

## 5. Causal boundary policies

모든 정책은 boundary를 prefix byte와 현재 UTF-8 parser state만으로 결정한다. test 전체의 score rank나 미래 문자를 볼 수 없다.

### C0 — `fixed_byte_6`

6-byte stride. 256 bytes에 43 data patches. Phase 2 Korean-only training으로 다시 학습하며 Phase 1 multilingual checkpoint를 재사용하지 않는다.

### C1 — `causal_codepoint_grid`

목표 grid는 Phase 1과 같은 43 patches를 갖도록 `τ_j = ceil(j × 256 / 43)`, `j=1..42`로 둔다. 다음 목표가 열린 뒤 처음 관측되는 complete UTF-8 prefix에서 boundary를 낸다.

온라인 표현은 다음과 같다.

```text
next_target = ceil(j × 256 / 43)
for each consumed byte position t:
    if t >= next_target and utf8_state(t) == COMPLETE:
        emit boundary at t
        j += 1
```

이는 목표 위치 이후의 후보를 미리 검색하지 않는다. UTF-8 codepoint가 최대 4 bytes이므로 정상 UTF-8 window에서 목표 간격을 건너뛰지 않고 42개 boundary를 낼 수 있다. 구현은 prefix invariance test를 통과해야 한다.

### C2 — `causal_eojeol_grid`

C1과 patch count를 정확히 맞추되, 각 target 주위 ±2-byte causal window에서 이미 소비한 whitespace 또는 punctuation 뒤를 우선한다.

- `t >= τ_j − 2`에서 delimiter가 완료되면 즉시 boundary
- 그렇지 않으면 `t >= τ_j + 2` 이후 첫 complete codepoint boundary
- 마지막 target은 end-of-window에서 boundary를 잃지 않도록 C1과 같이 `t >= τ_j`의 첫 complete codepoint에서 즉시 boundary
- 동일 위치에 boundary를 중복 생성하지 않음
- 최소 patch length가 2 bytes보다 작아지면 delimiter trigger를 무시

이 정책은 future delimiter 존재를 알지 못한 채 deadline까지 기다린다. punctuation 포함/제외 ablation은 primary 결과 뒤에 하지 않고, calibration 구조 통계에서 punctuation이 전체 delimiter의 50%를 넘으면 protocol addendum을 먼저 commit한 경우에만 별도 조건으로 허용한다.

### C3 — `entropy_threshold_full`

Korean-only 2-layer router가 예측한 causal next-byte entropy에 threshold를 적용한다. 모든 byte position이 후보이다.

- threshold는 seed별 calibration split에서만 정함
- 평균 data patches를 43.0 ± 0.1로 맞춤
- minimum patch length 1 byte
- starvation을 막기 위해 maximum patch length 24 bytes
- cap boundary는 byte 위치에 놓이므로 codepoint 내부일 수 있음
- threshold와 realized train/calibration/test rate를 모두 기록

### C4 — `entropy_threshold_codepoint`

C3와 동일 router·score·target average rate를 사용하되 complete UTF-8 prefix에서만 entropy trigger와 24-byte cap을 평가한다. threshold는 candidate 수가 다르므로 별도로 calibration한다.

### Structural-only control — `spacebyte_compatible`

BLT가 정의한 space-like byte heuristic을 causal 구조 통계와 quality-compute 예상치에 포함한다. Korean calibration에서 예상 rate가 target 5.953 bytes/patch의 ±10% 밖이면 primary neural training 대상에서 제외한다. Phase 2 protocol 작성 전 feasibility audit에서 약 3.33 bytes/patch로 관측됐으므로 현재는 **rate-unmatched structural control**이며 gate에 쓰지 않는다.

## 6. Variable patch matrices and compute accounting

C3/C4는 sequence별 patch 수가 다르다. matrix는 오른쪽을 zero-pad하고 매 batch에서 all-zero trailing columns를 잘라 HF model에 전달한다. 다음 invariant를 검사한다.

- 각 row의 positive data lengths 합 = 256
- dummy patch = 1
- zero는 positive patch 뒤에만 존재
- decoder logits finite
- batch padding을 제외한 realized patches와 실제 global tensor width를 모두 기록

비용은 두 축으로 보고한다.

1. **ideal unpadded:** sequence별 실제 patch 수를 넣은 analytical FLOPs
2. **implemented batched:** batch maximum patch 수와 실제 MPS latency

평균 rate만 같고 tail이 긴 entropy policy가 batching에서 더 비쌀 수 있으므로 둘을 합치지 않는다.

## 7. Noise-floor and packing controls

### 7.1 Exact duplicate

seed 1,729의 C1을 동일 initial state, data order, patch matrix로 두 번 학습한다. test BPB 차이, 최대 parameter absolute difference, per-sequence NLL difference를 기록한다.

- bitwise identical이면 MPS path를 이 설정에서 deterministic으로 간주
- Korean BPB 차이 >0.001이면 policy effect의 noise floor로 보고 모든 0.001 이하 차이를 해석하지 않음

### 7.2 Chunk-start alignment

Primary stream은 Phase 1과 같은 arbitrary 256-byte packing이다. 별도 control은 codepoint boundary에서 시작·끝나도록 최대 256 bytes를 채우고 남는 0–3 bytes를 newline으로 채운다. 삽입 byte 비율을 보고한다.

계산 예산을 통제하기 위해 C0/C1만 seeds 1,729 / 2,718 / 31,415에서 aligned packing으로 재학습한다. 이 결과는 primary five-seed comparison과 합치지 않는다. C1−C0의 방향이 arbitrary packing과 반대가 되면 scale-up gate를 중단한다.

## 8. Normalization and Hangul unit robustness

Primary 모델은 corpus 원문을 변경하지 않은 NFC-dominant stream에서 학습한다. test record에 다음 결정적 변환을 적용한다.

1. Unicode NFC
2. canonical NFD
3. precomposed modern Hangul syllable를 compatibility jamo sequence로 변환한 stress condition
4. original text에서 자연 발생한 mixed-script strata

NFD에서는 codepoint boundary가 Hangul syllable boundary와 같지 않다. 따라서 C1 checkpoint를 그대로 두고 inference patch만 바꾸는 `hangul_unit_grid`를 추가한다.

- precomposed Hangul syllable: 한 codepoint가 한 unit
- canonical modern Jamo: `L + V + optional T`를 한 unit
- non-Hangul: 한 codepoint가 한 unit
- boundary는 unit 완료 후에만 허용

NFC에서 C1과 `hangul_unit_grid` patch matrix가 정확히 같아야 한다. NFD에서만 차이가 난다. 이는 별도 학습된 우월 모델이 아니라 **representation robustness ablation**이다.

지표:

- BPB
- bits per original NFC codepoint
- bits per Hangul syllable unit
- invalid UTF-8 prefix/output rate
- patch boundary inside UTF-8 codepoint / Hangul unit

Compatibility jamo 변환은 canonical equivalence가 아니므로 quality equality를 기대하지 않으며 stress test로만 표시한다.

## 9. Generation validity

Phase 2 compact model의 자연어 품질을 과장하지 않기 위해 generation은 구조 지표만 사용한다.

- prompts: held-out Korean prefix 256개, prefix 자체는 공개하지 않음
- continuation: 128 bytes
- decoding: greedy와 temperature 0.8/top-p 0.95, seed 고정
- 정책별 생성 수 동일
- metrics: valid UTF-8 completion, replacement-character-free rate, valid Hangul syllable/Jamo transition, bytes per generated codepoint

UTF-8 hard mask를 켠 결과는 별도 oracle/control로 보고 unconstrained model 성능과 섞지 않는다. [Beyond Perplexity](https://arxiv.org/abs/2606.14122)의 평가 목적과 중복되는 부분은 reproduction/extension으로 명시한다.

## 10. Statistics

Primary contrasts:

1. C1 − C0: causal codepoint value
2. C2 − C1: eojeol value beyond Unicode
3. C4 − C3: codepoint restriction under causal entropy
4. C1 − C3: parameter-free vs learned entropy
5. C2 − C3: Korean hybrid vs learned entropy

각 contrast에 대해 다음을 보고한다.

- five paired seed effects
- mean and sample SD
- paired-t 95% interval
- seed→sequence hierarchical paired bootstrap 10,000회
- Korean strata effects

정책 5개 전체의 사후 순위 검정은 하지 않는다. primary contrast 외 pair는 exploratory로 표시한다. p-value 별표 대신 effect와 interval을 보고한다.

## 11. Decision gates

### Gate D — causal replication

다음을 모두 만족하면 Phase 1 fixed-alignment 결과가 causal setting에서 재현된 것으로 본다.

- mean `C1 − C0 <= −0.003 BPB`
- 5 seeds 중 최소 4개가 negative
- paired-t upper bound < 0
- aligned-packing 3-seed mean의 방향도 negative

### Gate E — Korean eojeol value

다음을 모두 만족할 때만 **Korean-aware method**를 중심 기여 후보로 유지한다.

- mean `C2 − C1 <= −0.003 BPB`
- 5 seeds 중 최소 4개가 negative
- C2와 C1의 exact data patch count 동일
- Korean informal/external set이 사용 가능하면 regression >0.02 BPB 없음

실패하면 eojeol prior를 폐기하고 논문을 Unicode/encoding-entropy 분석으로 축소한다.

### Gate F — parameter-free Pareto

다음을 모두 만족하면 C1 또는 C2를 scale-up한다.

- C1/C2 중 적어도 하나가 C3의 Korean BPB에서 0.015 이내
- router 포함 analytical FLOPs 10% 이상 절감
- batch-1 direct teacher-forced latency 10% 이상 절감
- threshold policy의 padding waste를 포함해도 비용 우위

### Gate G — robustness

NFD에서 `hangul_unit_grid`가 C1보다 bits/original-codepoint를 1% 이상 개선하고 NFC에서는 patch identity invariant를 만족하면 Hangul unit handling을 scale-up method에 포함한다. 그렇지 않으면 normalization은 평가 항목으로만 남긴다.

### Gate H — scale-up

Gate D 또는 F를 통과하고 다음 stop condition이 없을 때 Phase 3로 간다.

- duplicate noise > primary effect의 50%
- aligned packing에서 effect reversal
- UTF-8 validity가 fixed-byte보다 1 percentage point 이상 악화
- cost advantage가 Python selector overhead 제거 후 사라짐

## 12. Negative-result value

모든 method gate가 실패해도 다음 결과는 논문 가치가 있다.

1. raw-byte entropy가 CJK encoding uncertainty를 linguistic uncertainty로 오인하는 정량 증거
2. offline matched-rate 이득이 causal threshold 또는 packing control에서 사라지는지에 대한 반증
3. Korean NFC/NFD·Jamo·mixed-script의 patching failure taxonomy
4. auxiliary router와 batch-padding을 포함한 honest total-cost accounting

이 경우 제목과 초록은 method paper가 아니라 empirical analysis paper로 바꾼다.

## 13. Reproducibility outputs

추적할 것:

- protocol, configs, source revisions
- public source manifests and hashes
- seed별 scalar logs
- thresholds and realized rates
- aggregate per-stratum results
- timing samples and analytical cost formula
- generation validity counts
- paper figures/tables 생성 script

추적하지 않을 것:

- raw public/private text
- NSMC record IDs or reviews
- vault path/content
- checkpoints and per-example text
- generated samples containing memorized source passages

## 14. Phase 3가 열릴 경우의 최소 강한 baseline

Phase 2 gate 통과는 출판 결론이 아니다. Phase 3에서는 최소한 다음이 필요하다.

- 모델 10–30M parameters 이상, Korean training bytes 한 단계 확대
- fixed, SpaceByte, official-like entropy, character-preserving entropy
- 가능하면 H-Net 또는 integrated learned router
- Scratchpad Patching의 공개 구현이 없으면 method-level reproduction 또는 명확한 비포함 사유
- Korean downstream/targeted evaluation
- CUDA에서 incremental generation benchmark

Phase 2 결과가 나오기 전에는 10–30M configuration과 최종 claim을 선택하지 않는다.
