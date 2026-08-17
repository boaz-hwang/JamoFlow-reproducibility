# Phase 1 neural results: Unicode-aligned patching in a compact BLT

> 완료일: 2026-08-10  
> 사전등록: [Phase 1 protocol](./08-phase1-neural-protocol.md)  
> 기계 판독 결과: [`results/phase1-neural/summary.json`](../results/phase1-neural/summary.json)  
> 관측치 표: [`results/phase1-neural/observations.csv`](../results/phase1-neural/observations.csv)

## 1. 결론부터

Phase 1의 세 pilot gate는 사전 정의한 운영 기준으로 모두 통과했다. 그러나 결과가 지지하는 연구 방향은 처음 대화에서 제안된 “자소 규칙 기반 LLM”보다 훨씬 좁고 구체적이다.

> **compact BLT에서는 learned entropy router가 선택한 UTF-8 codepoint 내부 경계가 한국어 품질에 필요하다는 증거가 없었다. 반대로 parameter-free codepoint-aligned fixed segmentation은 fixed-byte segmentation보다 한국어에서 작지만 반복 가능한 BPB 개선을 보였고, 별도 router를 포함한 비용을 피했다.**

이 결과는 한글 자소 LM, 형태소 FST, multi-jamo generation의 효과를 입증하지 않는다. 지금 정당화된 다음 질문은 **한국어의 결정적 Unicode·음절·어절 구조를 이용한 causal streaming patcher가 learned entropy patcher의 비용을 줄이면서 품질을 유지할 수 있는가**이다.

## 2. 실행 및 무결성

- 모델: Hugging Face `BltForCausalLM`, main model 1,251,136 parameters
- auxiliary entropy router: `BltPatcher`, 139,584 parameters
- 언어: 한국어·중국어·영어 Wikipedia sentence corpus
- 학습량: 언어별 5,999,872 bytes, 합계 17,999,616 bytes per model
- test: 언어별 999,936 bytes, 3,906 sequences
- context: 256 bytes
- rate: 모든 정책에서 정확히 43 data patches, 5.953 bytes/patch
- seed: 1,729 / 2,718 / 31,415 / 57,721 / 65,537
- 완료 산출물: router 5개, main model 20개, policy report 20개, per-sequence paired loss 60개 language arrays
- 통계: seed-level paired t 95% interval과 seed × 공통 test sequence 교차 paired bootstrap 10,000회
- 비용 timing: Apple M4 Pro/MPS, warm-up 10회 후 무작위 interleaving 100회
- 테스트: 39개 unit/integration tests 통과

모든 neural checkpoint, patch cache, 원문 corpus는 Git에서 제외했다. Git에는 aggregate scalar, seed별 BPB, timing measurement만 기록한다.

## 3. Held-out quality

표의 값은 5개 seed의 test BPB 평균 ± sample standard deviation이다. 낮을수록 좋다.

| Policy | Korean | Chinese | English |
|---|---:|---:|---:|
| `fixed_byte` | 2.53759 ± 0.01273 | 3.72900 ± 0.02647 | 2.92013 ± 0.01704 |
| `fixed_codepoint` | **2.53092 ± 0.01124** | **3.72631 ± 0.02582** | 2.91969 ± 0.01783 |
| `entropy_full` | 2.54115 ± 0.01957 | 3.73494 ± 0.01945 | 2.90421 ± 0.01740 |
| `entropy_codepoint` | 2.54555 ± 0.01201 | 3.73244 ± 0.02396 | **2.89868 ± 0.01703** |

이 표의 행들은 각기 별도 학습된 모델이다. 단일 seed에서 test policy만 교체한 결과가 아니며, bold는 단순 최솟값이지 다중비교 후 우월성 판정이 아니다.

### 3.1 사전등록 primary contrasts

차이는 `left − right`이므로 음수이면 왼쪽 정책이 낫다.

| Contrast | Language | Mean Δ BPB | paired-t 95% CI | crossed bootstrap 95% CI |
|---|---|---:|---:|---:|
| `entropy_codepoint − entropy_full` | Korean | +0.00440 | [−0.00646, +0.01525] | [−0.00266, +0.01120] |
|  | Chinese | −0.00250 | [−0.01152, +0.00652] | [−0.00795, +0.00339] |
|  | English | **−0.00553** | [−0.00797, −0.00309] | [−0.00680, −0.00377] |
| `fixed_codepoint − fixed_byte` | Korean | **−0.00667** | [−0.01115, −0.00219] | [−0.00966, −0.00399] |
|  | Chinese | **−0.00269** | [−0.00446, −0.00093] | [−0.00387, −0.00156] |
|  | English | −0.00044 | [−0.00157, +0.00068] | [−0.00127, +0.00021] |
| `fixed_codepoint − entropy_full` | Korean | −0.01023 | [−0.02461, +0.00414] | [−0.01980, −0.00180] |
|  | Chinese | −0.00863 | [−0.02283, +0.00557] | [−0.01790, +0.00057] |
|  | English | +0.01548 | [+0.00930, +0.02165] | [+0.01217, +0.01986] |

해석에서 paired-t interval을 우선한다. seed가 5개뿐인 상황에서 공통 sequence 축까지 교차 재표집한 percentile bootstrap은 보조 진단이며 더 좁은 구간을 근거로 확정적 유의성을 주장하지 않는다. 재표집 설계 교정의 상세 내용은 [교정 기록](./34-crossed-bootstrap-correction.md)에 남겼다.

### 3.2 한국어에서 더 큰가

사전등록한 language × policy interaction을 seed별 difference-of-differences로 계산했다.

| Interaction for `fixed_codepoint − fixed_byte` | Mean ΔΔ BPB | paired-t 95% CI |
|---|---:|---:|
| Korean minus English | **−0.00623** | [−0.01101, −0.00144] |
| Korean minus Chinese | **−0.00398** | [−0.00739, −0.00056] |
| Chinese minus English | −0.00225 | [−0.00476, +0.00026] |

따라서 generic Unicode 효과만 관측된 것은 아니다. 이 compact setting에서는 codepoint alignment가 fixed-byte 대비 한국어에서 영어뿐 아니라 중국어보다도 더 큰 개선을 보였다. 다만 효과 크기는 baseline Korean BPB의 약 0.26%이고 seed가 5개이므로, “한글 고유의 우월성”이 아니라 **확대 검증할 가치가 있는 interaction**으로 취급한다.

## 4. Router가 실제로 선택한 것은 무엇인가

| Language | `fixed_byte` codepoint-internal rate | `entropy_full` codepoint-internal rate | full/constrained boundary overlap |
|---|---:|---:|---:|
| Korean | 58.95% | **97.02%** | 2.98% |
| Chinese | 63.27% | **95.99%** | 4.01% |
| English | 0.14% | 0.53% | 99.47% |

한국어 `entropy_full` 내부 경계의 평균 96.62%p는 완성형 한글 음절 내부였다. 즉 작은 entropy router는 한국어에서 거의 모든 expensive global boundary를 “새 글자의 의미적 시작”보다 UTF-8 continuation-byte 예측 불확실성에 배정했다. `entropy_codepoint`는 한국어에서 이 경계의 약 97%를 다른 위치로 옮겼지만 평균 손상은 +0.00440 BPB였고 방향도 seed마다 달랐다.

이는 중요한 기계적 단서다.

1. raw-byte entropy는 semantic difficulty와 Unicode encoding difficulty를 구분하지 않는다.
2. CJK의 높은 continuation-byte entropy가 patch budget을 지배할 수 있다.
3. 동일 patch 수라면 codepoint state라는 값싼 prior가 router의 encoding-level distraction을 제거할 가능성이 있다.

다만 이것만으로 entropy routing 일반이 잘못됐다고 말할 수 없다. router가 139K parameters로 매우 작고 one-pass만 학습됐기 때문에, 더 강한 router가 다른 경계를 학습할 수 있다.

영어에서는 두 entropy 정책의 경계가 99.47% 겹치는데도 `entropy_codepoint`가 모든 seed에서 더 좋았다. 작은 segmentation 차이가 누적된 결과일 수 있지만, MPS 재현성 또는 학습 궤적 민감도도 배제해야 한다. Phase 2에는 **동일 segmentation·동일 seed 재학습 duplicate**를 넣어 noise floor를 측정한다.

## 5. Chunk-start misalignment 교란 검토

256-byte window가 codepoint 중간에서 시작한 비율은 한국어 59.75%, 중국어 63.85%, 영어 0.13%였다. 이 때문에 `fixed_codepoint`의 이점이 단순히 시작 위상을 복구한 artifact일 가능성을 사후 검토했다.

| Language | Start stratum | Sequences | `fixed_codepoint − fixed_byte` mean BPB | paired-t 95% CI | fixed-byte internal rate |
|---|---|---:|---:|---:|---:|
| Korean | aligned | 1,572 | −0.00695 | [−0.01148, −0.00243] | 59.28% |
| Korean | internal | 2,334 | −0.00648 | [−0.01093, −0.00202] | 58.72% |
| Chinese | aligned | 1,412 | −0.00260 | [−0.00464, −0.00056] | 58.64% |
| Chinese | internal | 2,494 | −0.00274 | [−0.00435, −0.00113] | 65.89% |

한국어 효과는 두 stratum에서 비슷했다. 시작점이 정렬돼도 ASCII·공백·문장부호가 3-byte cadence의 위상을 계속 바꾸므로 fixed 6-byte boundary의 내부 분할률도 여전히 높았다. 따라서 chunk start만으로 전체 효과를 설명할 수는 없다.

하지만 이 분석은 결과 확인 후 수행한 confound check이고, 두 모델 모두 misaligned training mixture에서 학습됐다. Phase 2에서는 **codepoint-aligned packing 여부를 학습 단계에서 무작위화한 2×2 control**로 재검증해야 한다.

## 6. Total cost

### 6.1 Analytical dense-matmul FLOPs

구현된 HF forward path의 QKVO, SwiGLU, self/cross-attention matmul을 세고 multiply-add를 2 FLOPs로 계산했다. normalization, RoPE, activation, softmax, hashing, memory movement는 명시적으로 제외했다.

| Component | FLOPs / 256-byte sequence | FLOPs / byte |
|---|---:|---:|
| compact BLT main | 257,261,568 | 1,004,928 |
| entropy router | 96,468,992 | 376,832 |
| entropy end-to-end | 353,730,560 | 1,381,760 |

- router/main overhead: 37.50%
- router share of entropy end-to-end: 27.27%
- router parameter share: 10.04% of the entropy system, 139,584 / 1,390,720

이 수치는 BLT 논문의 full-scale FLOPs를 재현한 값이 아니라 **이번 compact HF graph의 투명한 dense-matmul count**다.

### 6.2 M4 Pro direct pipeline latency

입력은 MPS에 미리 올렸다. entropy pipeline에는 router, MPS→CPU score transfer, NumPy top-k/candidate selection, patch upload, main BLT를 모두 포함했다. fixed-codepoint pipeline에도 Python boundary construction과 patch upload를 포함했다.

| Batch | `fixed_codepoint` median | `entropy_full` median | Fixed reduction |
|---:|---:|---:|---:|
| 1 | 5.254 ms | 7.691 ms | **31.69%** |
| 8 | 7.279 ms | 9.076 ms | **19.79%** |
| 64 | 39.891 ms | 50.153 ms | **20.46%** |

100회 측정의 batch-1 p10–p90은 fixed-codepoint 4.159–6.762 ms, entropy-full 6.047–8.671 ms였다. 이는 256-byte teacher-forced window latency이지 incremental generation latency가 아니며, Apple MPS 결과를 CUDA serving에 일반화할 수 없다.

## 7. Gate 판정과 보수적 해석

### Gate A — pass, formal equivalence는 미확립

세 언어 모두 `entropy_codepoint − entropy_full` 평균이 +0.015 BPB 이하였고 모든 seed-language cell이 +0.03 이하라 pilot gate는 통과했다. 그러나 한국어 paired-t upper bound는 +0.01525로 margin +0.015를 0.00025 넘는다. 따라서 “동등함을 입증했다”가 아니라 **다음 단계 확대를 허용한 pilot pass**다.

### Gate B — pass, 영어 trade-off가 핵심

`fixed_codepoint`의 mean BPB는 세 언어에서 `entropy_full`의 0.02 이내였고 분석·실측 비용 감소가 10%를 넘었다. 하지만 영어의 평균 손실은 +0.01548이고 paired-t upper bound는 +0.02165다. multilingual universal replacement로 주장하면 안 된다. 연구 가치는 오히려 **한국어·중국어와 영어의 상반된 interaction**에 있다.

### Gate C — pass for this router

별도 router는 분석 FLOPs의 27.27%이고 batch-1 component latency에서도 10%를 넘었다. 다만 이는 2-layer dense router에 대한 결과다. integrated router, lookup table, n-gram entropy, H-Net류 end-to-end routing까지 “비싸다”고 일반화하지 않는다.

## 8. 지금 가능한 주장과 불가능한 주장

### 이 실험이 지지하는 주장

1. 1.25M compact BLT와 같은 patch count에서 fixed boundary를 UTF-8 codepoint에 맞추면 Korean BPB가 fixed bytes보다 개선됐다.
2. 그 개선은 영어보다 크고, 이 실험에서는 중국어보다도 컸다.
3. 작은 entropy router는 CJK에서 codepoint 내부 entropy에 patch budget을 집중했다.
4. 그 내부 경계를 금지해도 한국어·중국어 quality가 명백하게 무너지지 않았다.
5. 이번 별도 dense router는 무시할 수 없는 end-to-end cost였다.

### 아직 불가능한 주장

- 대형 또는 1B급 BLT에서도 같은 효과가 난다.
- online generation에서 20–32%가 빨라진다.
- 한글 자소 decomposition이 유리하다.
- 형태소/띄어쓰기 규칙이 neural entropy를 대체한다.
- fixed codepoint가 learned routing보다 보편적으로 낫다.
- 한국어 생성 품질 또는 downstream 성능이 개선된다.
- Unicode alignment 자체가 새로운 발명이다.

## 9. 연구 방향 수정

Phase 1 결과를 반영한 논문 중심축은 다음이 적절하다.

> **Encoding entropy is not linguistic uncertainty: can causal Korean-aware byte patching avoid Unicode-internal routing and auxiliary patcher cost without sacrificing language modeling quality?**

한국어 제목으로는 다음이 정확하다.

> **한국어 byte language model에서 Unicode 내부 entropy와 언어적 불확실성의 분리: 저비용 causal patching 연구**

핵심 기여 후보는 “한글 FSM으로 LLM compute를 없앤다”가 아니다.

1. **진단적 기여:** raw-byte entropy patching이 한국어에서 완성형 한글의 UTF-8 내부 경계를 과도하게 선택한다는 정량 분석
2. **방법 기여:** generic UTF-8 state와 Korean syllable/eojeol state를 결합한 causal rate controller
3. **효율 기여:** router를 포함한 total cost accounting과 streaming latency
4. **언어학적 기여:** generic CJK 효과와 Korean-specific interaction, NFC/NFD·호환 자모·한영 혼용 robustness 분리

## 10. Phase 2에서 반드시 해결할 것

1. **Causality:** window top-k와 미래 후보를 보는 fixed control을 버리고 calibration threshold 또는 causal rate controller 사용
2. **Matched realized rate:** 정책별 실제 bytes/patch 분포와 quality-cost Pareto curve 비교
3. **Packing control:** codepoint-aligned vs arbitrary byte chunk training을 교차해 window artifact 제거
4. **Noise floor:** 동일 segmentation·초기화·순서 duplicate training으로 nondeterministic drift 측정
5. **Korean specificity:** generic UTF-8, Hangul syllable, whitespace/eojeol, hybrid를 순차 ablation
6. **Representation robustness:** NFC, canonical NFD, compatibility jamo(`ㅋㅋ`, `ㅠㅠ`), emoji, 한영·숫자 혼용
7. **Router controls:** learned entropy 외 n-gram/lookup router와 더 강한 router 포함
8. **Scale:** pilot을 재현한 뒤 모델/data budget을 최소 한 단계 확대
9. **Generation:** teacher-forced throughput과 incremental cached decoding을 분리
10. **Quality:** BPB 외 UTF-8 validity, Korean spacing, generation sample, downstream 또는 targeted minimal-pair 평가

자소 atomic representation, 형태소 FST, multi-jamo diffusion은 이 열 가지를 통과하기 전에는 결합하지 않는다. 동시에 여러 아이디어를 넣으면 어떤 요소가 이득을 만들었는지 다시 알 수 없게 된다.

## 11. 재현 명령

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase1.py
PYTHONPATH=src .venv/bin/python scripts/benchmark_phase1.py \
  --warmup-rounds 10 --repetitions 100
PYTHONPATH=src .venv/bin/python scripts/summarize_phase1.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`run_phase1.py`는 완료된 checkpoint/report를 감지해 재시작할 수 있다. aggregate 결과의 source manifest hash와 run-start Git commit은 `summary.json`에 기록된다.
